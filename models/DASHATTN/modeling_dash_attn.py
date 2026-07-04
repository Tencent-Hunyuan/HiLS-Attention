from turtle import position
from typing import Callable, Optional, Tuple, Union

import torch
from typing import Literal
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    replace_return_docstrings,
)

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.utils import logging
from veomni.utils.import_utils import (
    is_liger_kernel_available,
    is_torch_npu_available,
    is_transformers_version_greater_or_equal_to,
)
from veomni.models.module_utils import GradientCheckpointingLayer

from functools import lru_cache


_DASH_KERNEL_IMPORT_ERROR: Optional[BaseException] = None
try:
    from infllmv2_entmax import topk_sparse_attention
    from infllmv2_entmax.stage1 import compressed_attention
    from infllmv2_entmax.transform_score import transform_score
    from adasplash import triton_entmax
    _DASH_TRAIN_KERNELS_AVAILABLE = True
except Exception as _dash_kernel_import_err:  # noqa: BLE001
    _DASH_TRAIN_KERNELS_AVAILABLE = False
    _DASH_KERNEL_IMPORT_ERROR = _dash_kernel_import_err


if is_torch_flex_attn_available():
    pass


if is_liger_kernel_available():
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


rms_norm_fn = None

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
    USE_FLASH_ATTN_RMSNORM = True
except ImportError:
    USE_FLASH_ATTN_RMSNORM = False


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "Qwen3Config"


# ---------------------------------------------------------------------------
# Inlined helpers from dash-attention/dash_attention.py.
# Copied verbatim (modulo formatting) so we do not need to import that module,
# which would pull in megatron.core.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _dash_calc_chunks_with_stride(cu_seqlen, moba_chunk_size, kernel_stride):
    """Build per-chunk gather indices used by `CompressK`.

    Mirrors `calc_chunks_with_stride` in dash_attention/dash_attention.py.
    """
    from itertools import accumulate  # noqa: F401  (parity with original)

    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1]
    max_seq_len = torch.max(batch_sizes)
    max_num_chunks_per_seq = (max_seq_len - moba_chunk_size) // kernel_stride + 1
    chunk_start_offsets = torch.arange(
        0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device
    )
    seq_starts = cu_seqlen[:-1]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]
    chunk_end_in_seq = chunk_start_in_seq + moba_chunk_size
    valid_chunk_mask = chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]
    del chunk_start_in_seq

    chunk_indices = torch.arange(0, moba_chunk_size, device=cu_seqlen.device)[None, :]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices
    filtered_indices = filtered_indices.view(-1)

    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    )
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0)
    return filtered_indices, cu_seqlens_compressed


class _DashCompressK(nn.Module):
    """Compress K along the sequence dim into chunk summaries.

    Inlined from `CompressK` in dash_attention/dash_attention.py.
    Uses the learned `indexer_q` as the chunk-local query.
    """

    def __init__(self, head_num_k, head_dim, kernel_size, kernel_stride=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.head_num_k = head_num_k
        self.head_dim = head_dim
        self.kernel_stride = kernel_stride

    def forward(self, indexer_q: torch.Tensor, k: torch.Tensor, cu_seqlens: torch.Tensor):
        filtered_k_indices, cu_seqlens_compressed = _dash_calc_chunks_with_stride(
            cu_seqlens, self.kernel_size, self.kernel_stride
        )

        filtered_k = k.index_select(0, filtered_k_indices.view(-1))
        filtered_q = indexer_q.index_select(0, filtered_k_indices.view(-1))

        # Reshape into chunks of `kernel_size`.
        filtered_k = filtered_k.view(
            filtered_k.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim
        )
        filtered_q = filtered_q.view(
            filtered_q.shape[0] // self.kernel_size, self.kernel_size, self.head_num_k, self.head_dim
        )

        # Mean query within each chunk, then a local softmax-attention to summarize K.
        filtered_q = filtered_q.mean(dim=1)
        comp_score = torch.einsum('nhd,nmhd->nmh', filtered_q, filtered_k)
        comp_prob = torch.nn.functional.softmax(
            comp_score * (filtered_k.shape[-1] ** -0.5), dim=1
        )
        compress_k = torch.einsum('nmh,nmhd->nhd', comp_prob, filtered_k)
        return compress_k, cu_seqlens_compressed


def _dash_compressed_attention(
    q: torch.Tensor,
    compressed_k: torch.Tensor,
    kernel_size: int,
    kernel_stride: int,
    block_size: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
):
    """Stage 1: compute topk block indices and pooled per-head scores.

    Inlined from `compressed_attention` in dash_attention/dash_attention.py
    (the unused `self`, `v` and `cache_lens` arguments are dropped).
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    # Per-query block index relative to its own sequence's max length.
    q_idx = torch.cat(
        [
            (
                torch.arange(cu_seqlens_q[i + 1] - cu_seqlens_q[i], device=q.device)
                + max_seqlen_q
                - (cu_seqlens_q[i + 1] - cu_seqlens_q[i])
            )
            // block_size
            for i in range(batch_size)
        ],
        dim=0,
    )

    score = compressed_attention(
        q,
        compressed_k,
        kernel_size,
        kernel_stride,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q.shape[-1] ** (-0.5),
    )


    score = triton_entmax(score.float(), alpha=1.5, n_iter=3)
    score_ = score[:, : q_idx.shape[0], :]
    score = score.reshape(compressed_k.shape[1], -1, score.shape[-2], score.shape[-1])
    score = score.mean(dim=1)
    score = score[:, : q_idx.shape[0], :]

    pooled_score = score_
    # Pad one column to mimic the layout `transform_score` expects.
    pooled_score = torch.nn.functional.pad(pooled_score, (0, 1), value=0)

    block_score = transform_score(
        score.contiguous(),
        kernel_size,
        kernel_stride,
        block_size,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )

    bmask = block_score > 0
    max_k = bmask.sum(dim=-1).max().item()
    topk_idx = block_score.topk(k=max_k, dim=-1).indices
    bmask_indexed = torch.gather(bmask, dim=-1, index=topk_idx)
    topk_idx[~bmask_indexed] = -1
    # Causal: a query can only attend to blocks at or before its own block.
    topk_idx[topk_idx > q_idx[None, :, None]] = -1
    topk_idx = topk_idx.to(torch.int32)

    return topk_idx, max_k, pooled_score



def rms_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return weight * hidden_states.to(input_dtype)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen3FlashAttnRMSNorm(Qwen3RMSNorm):
    def forward(self, hidden_states):
        global rms_norm_fn
        if rms_norm_fn is None:
            try:
                from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_attn_rms_norm_fn
            except ImportError as exc:
                raise ImportError(
                    "flash_attn.ops.triton.layer_norm.rms_norm_fn is required for Qwen3FlashAttnRMSNorm."
                ) from exc
            rms_norm_fn = flash_attn_rms_norm_fn

        hidden_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_shape[-1])
        hidden_states = rms_norm_fn(
            hidden_states,
            self.weight,
            None,
            eps=self.variance_epsilon,
        )
        return hidden_states.reshape(hidden_shape)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int, mode: Literal['swa', 'full-attn']):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.enable_scaling = getattr(config, 'enable_scaling', False)
        if mode == 'swa':
            self.apply_rope = True
            self.sliding_window = config.sliding_window
        else:
            self.apply_rope = getattr(config, 'rope_full_attn', False)
            self.sliding_window = None
        
        print(f'init Qwen3Attention, layer_idx: {layer_idx}, mode: {mode}, apply_rope: {self.apply_rope}, sliding_window: {self.sliding_window}')

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) #(B, h, L, d)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.apply_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # assert not torch.allclose(q_rot, query_states, atol=1e-5)
            # assert not torch.allclose(k_rot, key_states, atol=1e-5)
            # query_states = q_rot
            # key_states = k_rot

        if self.sliding_window is None and not self.training and self.enable_scaling:
            a = 362
            scaling_factor = torch.log(a + position_ids) / torch.log(torch.tensor(a, dtype=hidden_states.dtype, device=hidden_states.device))
            # (B, L)
            query_states = query_states * scaling_factor.unsqueeze(1).unsqueeze(-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # if self.sliding_window is not None:
            #     key_states = key_states[:, :, -self.sliding_window :, :]
            #     value_states = value_states[:, :, -self.sliding_window :, :]

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # for debug
        # print(f'layer idx: {self.layer_idx}, apply rope: {self.apply_rope}, sliding window: {self.sliding_window}')
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DashAttentionLayer(nn.Module):
    """Dash Attention training layer.

    Wraps the open-sourced training kernels (`compressed_attention`,
    `transform_score`, `topk_sparse_attention`) in an HF-compatible
    `[B, L, hidden]` interface. Internally we flatten to varlen
    `[total_len, num_heads, head_dim]` and run the two-stage Dash pipeline.

    `cu_seqlens` follows a 3-way fallback:
      1. external packed kwargs (`cu_seq_lens_q` / `max_length_q`),
      2. derive boundaries from `position_ids` resets to 0,
      3. assume each row of the batch is one packed segment of length L.
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.apply_rope = getattr(config, 'rope_full_attn', False)
        self.sliding_window = None

        # Dash training hyper-parameters. Defaults follow the official
        # dash-attention/dash_attention.py reference setting.
        self.dash_kernel_size = getattr(config, 'dash_kernel_size', 64)
        self.dash_kernel_stride = getattr(config, 'dash_kernel_stride', 64)
        self.dash_block_size = getattr(config, 'dash_block_size', 64)
        self.dash_topk = getattr(config, 'dash_topk', 32)
        self.dash_window_size = getattr(config, 'dash_window_size', 0)
        self.dash_init_blocks = getattr(config, 'dash_init_blocks', 0)
        self.dash_sigma = getattr(config, 'dash_sigma', 1.0e6)
        # `local_blocks` is derived from window size in tokens.
        self.dash_local_blocks = self.dash_window_size // self.dash_block_size

        # Learnable indexer query (same role as `indexer_q` in attention.py),
        # used only by the compressor stage. Dtype follows the model's dtype
        # via standard parameter casting.
        self.indexer_q = nn.Parameter(
            torch.zeros(self.num_key_value_heads * self.head_dim)
        )

        # K compressor (chunk-level summary).
        self.compress_k = _DashCompressK(
            head_num_k=self.num_key_value_heads,
            head_dim=self.head_dim,
            kernel_size=self.dash_kernel_size,
            kernel_stride=self.dash_kernel_stride,
        )

    def _build_cu_seqlens(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        position_ids: Optional[torch.LongTensor],
        cu_seq_lens_q: Optional[torch.Tensor],
        max_length_q: Optional[int],
    ) -> Tuple[torch.Tensor, int]:
        """3-way fallback for cu_seqlens; see class docstring."""
        # Path 1: packed kwargs from the dataloader (rmpad_with_pos_ids path).
        if cu_seq_lens_q is not None:
            cu = cu_seq_lens_q.to(device=device, dtype=torch.int32)
            if max_length_q is not None:
                m = int(max_length_q)
            else:
                m = int((cu[1:] - cu[:-1]).max().item())
            return cu, m

        # Path 2: derive boundaries from position_ids resetting to 0
        # at each document start (matches veomni's flash_attn helper).
        if position_ids is not None and position_ids.numel() > 1:
            flat_pos = position_ids.reshape(-1)
            # A new segment begins where position id is 0.
            boundary_mask = flat_pos == 0
            if boundary_mask.any() and boundary_mask.sum().item() > 1:
                starts = torch.nonzero(boundary_mask, as_tuple=False).flatten()
                cu = torch.cat(
                    [starts, torch.tensor([flat_pos.numel()], device=flat_pos.device)]
                ).to(dtype=torch.int32)
                m = int((cu[1:] - cu[:-1]).max().item())
                return cu, m

        # Path 3: default - each row is one packed segment of length L.
        cu = torch.arange(
            0, (batch_size + 1) * seq_len, seq_len, device=device, dtype=torch.int32
        )
        return cu, seq_len

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not _DASH_TRAIN_KERNELS_AVAILABLE:
            raise ImportError(
                "Dash Attention training kernels are not importable. "
                "Add the path containing `infllmv2_entmax/` (e.g. dash-attention repo root) "
                "to PYTHONPATH and install `adasplash`. "
                f"Underlying error: {type(_DASH_KERNEL_IMPORT_ERROR).__name__}: {_DASH_KERNEL_IMPORT_ERROR}"
            ) from _DASH_KERNEL_IMPORT_ERROR

        # Inference / KV-cache path is intentionally not supported here;
        # decoding should fall back to dash_attn/decoding kernels.
        if past_key_value is not None:
            raise NotImplementedError(
                "DashAttentionLayer training path does not support KV cache; "
                "use the inference-time dash_attn kernels for generation."
            )

        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        # 1) Standard q/k/v projection + qk-norm. Keep [B, H, L, D] layout for RoPE.
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 2) Build the indexer query from the learnable parameter, then RoPE.
        cos, sin = position_embeddings
        indexer_q = (
            self.indexer_q.to(hidden_states.dtype)
            .view(1, 1, self.num_key_value_heads, self.head_dim)
            .expand(bsz, seq_len, -1, -1)
            .transpose(1, 2)
            .contiguous()
        )
        if self.apply_rope:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            # Same RoPE for indexer_q (only the q-side output is needed).
            indexer_q, _ = apply_rotary_pos_emb(indexer_q, indexer_q, cos, sin)

        # 3) Flatten to varlen layout [total_len, H, D] for the kernels.
        # [B, H, L, D] -> [B, L, H, D] -> [B*L, H, D]
        q_var = query_states.transpose(1, 2).reshape(bsz * seq_len, self.num_heads, self.head_dim).contiguous()
        k_var = key_states.transpose(1, 2).reshape(bsz * seq_len, self.num_key_value_heads, self.head_dim).contiguous()
        v_var = value_states.transpose(1, 2).reshape(bsz * seq_len, self.num_key_value_heads, self.head_dim).contiguous()
        iq_var = indexer_q.transpose(1, 2).reshape(bsz * seq_len, self.num_key_value_heads, self.head_dim).contiguous()

        # 4) Resolve cu_seqlens: prefer external packed kwargs, then position_ids, then default.
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        max_length_q = kwargs.get("max_length_q", None)
        cu_seqlens, max_seqlen = self._build_cu_seqlens(
            batch_size=bsz,
            seq_len=seq_len,
            device=q_var.device,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            max_length_q=max_length_q,
        )

        # 5) Stage 1: compress K and select topk blocks per query.
        compressed_k, compressed_cu = self.compress_k(iq_var, k_var, cu_seqlens)
        compressed_seqlens = compressed_cu[1:] - compressed_cu[:-1]
        topk_idx, _max_k, pooled_score = _dash_compressed_attention(
            q_var,
            compressed_k,
            self.dash_kernel_size,
            self.dash_kernel_stride,
            self.dash_block_size,
            cu_seqlens,
            compressed_cu,
            max_seqlen,
            int(compressed_seqlens.max().item()),
            init_blocks=self.dash_init_blocks,
            local_blocks=self.dash_local_blocks,
        )

        # 6) Map entmax pooled scores back to logit space.
        # Same recipe as official dash_attention.py:DashAttnDotProductAttention.
        score_mask = pooled_score > 0
        mask_cnt = score_mask.sum(dim=-1, keepdim=True)
        ps_valid = torch.zeros_like(pooled_score)
        ps_valid[score_mask] = torch.log(pooled_score[score_mask] / self.dash_block_size)
        ps_mean = ps_valid.sum(dim=-1, keepdim=True) / mask_cnt.clamp(min=1)
        pooled_score = (ps_valid - ps_mean) / self.dash_sigma

        # 7) Stage 2: topk sparse attention with full fwd+bwd kernel.
        attn_var = topk_sparse_attention(
            q_var,
            k_var,
            v_var,
            topk_idx,
            pooled_score,
            self.dash_block_size,
            cu_seqlens,
        )  # [total_len, num_heads, head_dim]

        # 8) Reshape back to [B, L, H*D] and project.
        attn_output = attn_var.view(bsz, seq_len, self.num_heads * self.head_dim).to(hidden_states.dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int, mode: Literal['swa', 'full-attn', 'dash'] = 'swa'):
        super().__init__()
        self.hidden_size = config.hidden_size
        if mode == 'dash':
            self.self_attn = DashAttentionLayer(config=config, layer_idx=layer_idx)
        else:
            self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx, mode=mode)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        # print(f'self.inv_freq: {self.inv_freq}, {self.inv_freq[None, :, None]}')
        self.reinit = False


    @torch.no_grad()
    def forward(self, x, position_ids):
        if not self.reinit:
            self.reinit = True
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, x.device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        # print(f'self.inv_freq: {self.inv_freq[None, :, None]}')
        # print(f'fwd: pos_ids: {position_ids}')
        # print(f'inv_freq: {inv_freq_expanded}')

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


QWEN3_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3RMSNorm):
            module.weight.data.fill_(1.0)


QWEN3_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen3 Model outputting raw hidden-states without any specific head on top.",
    QWEN3_START_DOCSTRING,
)
class DashAttnModel(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.full_attn_interleave = config.full_attn_interleave
        def layer_type(layer_idx: int) -> Literal['swa', 'dash']:
            if self.full_attn_interleave > 0 and (layer_idx % self.full_attn_interleave == self.full_attn_interleave - 1):
                return 'dash'
            else:
                return 'swa'
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx, mode=layer_type(layer_idx)) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        # if not isinstance(causal_mask_mapping := attention_mask, dict):
        #     # Prepare mask arguments
        #     mask_kwargs = {
        #         "config": self.config,
        #         "input_embeds": inputs_embeds,
        #         "attention_mask": attention_mask,
        #         "cache_position": cache_position,
        #         "past_key_values": past_key_values,
        #         "position_ids": position_ids,
        #     }
        #     # Create the masks
        #     causal_mask_mapping = {
        #         "full_attention": create_causal_mask(**mask_kwargs),
        #     }
        #     # The sliding window alternating layers are not always activated depending on the config
        #     if self.has_sliding_layers:
        #        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # --- slice position embedding if using sp ---
        sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
        position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
        # --- slice position embedding if using sp ---

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class KwargsForCausalLM(FlashAttentionKwargs): ...


class DashAttnForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = DashAttnModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @add_start_docstrings_to_model_forward(QWEN3_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        loss = None
        logits = None
        if labels is not None:
            loss, logits = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if is_liger_kernel_available():
    apply_rotary_pos_emb = liger_rotary_pos_emb
    Qwen3MLP = LigerSwiGLUMLP
    logger.info_rank0("Apply liger kernel to Qwen3.")
if USE_FLASH_ATTN_RMSNORM:
    Qwen3RMSNorm = Qwen3FlashAttnRMSNorm
    logger.info_rank0("Apply flash-attn RMSNorm kernel to Qwen3.")

if is_torch_npu_available() and is_transformers_version_greater_or_equal_to("4.50.4"):
    from .npu_patch import apply_qwen3_npu_patch

    apply_qwen3_npu_patch()

__all__ = ["DashAttnForCausalLM", "DashAttnModel", "Qwen3PreTrainedModel"]
