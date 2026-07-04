from typing import Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import math
from .HoPE import HoPERotaryEmbedding
from torch import nn
from .hils_attention import HiLSAttention as LandmarkHSA_base
# from .lhsa_layer_pope_fused import LandmarkHSA as LandmarkHSA_pope
from .lhsa_layer_pope_naive import LandmarkHSA as LandmarkHSA_naive
from .configuration_hils import HSAConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin

from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    replace_return_docstrings,
)

from functools import partial
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.utils.env import get_env
from veomni.utils import logging
from veomni.utils.import_utils import (
    is_liger_kernel_available,
    is_torch_npu_available,
    is_transformers_version_greater_or_equal_to,
)

from veomni.models.module_utils import GradientCheckpointingLayer
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
from .hils_forward import hils_model_forward, hils_causal_lm_forward
from .pope import PoPERotaryEmbWrapper
from ops.flex_attn_tilelang import flex_attn_tl

if is_torch_flex_attn_available():
    pass


logger = logging.get_logger(__name__)


def get_env_with_fallback(name: str, default: str) -> str:
    try:
        return get_env(name)
    except KeyError:
        return default


USE_LIGER_KERNEL = is_liger_kernel_available() and get_env_with_fallback("USE_LIGER_KERNEL", "1") == "1"
USE_LIGER_ROPE = USE_LIGER_KERNEL and get_env_with_fallback("USE_LIGER_ROPE", "1") == "1"
USE_LIGER_SWIGLU = USE_LIGER_KERNEL and get_env_with_fallback("USE_LIGER_SWIGLU", "1") == "1"

rms_norm_fn = None

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
    USE_FLASH_ATTN_RMSNORM = True
except ImportError:
    USE_FLASH_ATTN_RMSNORM = False

if USE_LIGER_ROPE or USE_LIGER_SWIGLU:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP


def resolve_olmo_head_dim(config: HSAConfig) -> int:
    head_dim = config.hidden_size // config.num_attention_heads
    config_head_dim = getattr(config, "head_dim", None)
    if config_head_dim != head_dim:
        logger.warning_once(
            f"`olmo_hils` ignores config.head_dim={config_head_dim} and uses "
            f"hidden_size // num_attention_heads = {head_dim}."
        )
        config.head_dim = head_dim
    return head_dim



@dataclass
class GenerateState:
    active: bool = False
    decode_token_count: int = 0
    next_pos: int = 0
    lmk_positions_in_input: Optional[list] = None

    def reset(self):
        self.active = False
        self.decode_token_count = 0
        self.next_pos = 0
        self.lmk_positions_in_input = None


def rms_norm(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)

    return weight * hidden_states.to(input_dtype)


@dataclass
class GenerateState:
    """State container for generation mode, including all decode-time state variables."""
    active: bool = False                          # Whether generation mode is active
    decode_token_count: int = 0                    # Number of decoded tokens in the current chunk
    next_pos: int = 0                              # Position id of the next token
    cache_seq_len: int = 0                         # KV cache length including LMK tokens, used to compute cache_position
    lmk_positions_in_input: Optional[list] = None  # Positions of LMK tokens in the current input

    def reset(self):
        """Reset all state after generation finishes."""
        self.active = False
        self.decode_token_count = 0
        self.next_pos = 0
        self.cache_seq_len = 0
        self.lmk_positions_in_input = None


def next_of_y(x, y):
    return (x + y - 1) // y * y


def get_model_vocab_size(config) -> int:
    enable_external_lmk_embed = getattr(config, "enable_external_lmk_embed", False)
    insert_landmarks = bool(
        getattr(config, "insert_landmarks", False)
        or getattr(config, "adjust_lmk_pos", False)
    )
    if enable_external_lmk_embed or not insert_landmarks:
        return config.vocab_size
    return next_of_y(config.vocab_size + 1, 32)


def ensure_qwen3_rope_parameters(config: HSAConfig) -> None:
    if getattr(config, "rope_parameters", None) is not None:
        return

    rope_scaling = getattr(config, "rope_scaling", None) or {}
    rope_parameters = dict(rope_scaling)
    rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
    rope_parameters.setdefault("rope_theta", getattr(config, "rope_theta", 10000.0))
    config.rope_parameters = rope_parameters


class Olmo3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Olmo3FlashAttnRMSNorm(Olmo3RMSNorm):
    def forward(self, hidden_states):
        global rms_norm_fn
        if rms_norm_fn is None:
            try:
                from flash_attn.ops.triton.layer_norm import rms_norm_fn as flash_attn_rms_norm_fn
            except ImportError as exc:
                raise ImportError(
                    "flash_attn.ops.triton.layer_norm.rms_norm_fn is required for Olmo3FlashAttnRMSNorm."
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


Olmo3TorchRMSNorm = Olmo3RMSNorm


class Olmo3MLP(nn.Module):
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
    # Match HF Olmo3: q/k keep their original dtype, while the rotary module
    # computes angles in fp32 and casts cos/sin back before returning them.
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


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


class Olmo3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HSAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = resolve_olmo_head_dim(config)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.mask_lmk_token = getattr(config, "mask_lmk_token", False)
        if self.mask_lmk_token:
            self.chunk_size = config.chunk_size

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
        self.q_norm = Olmo3RMSNorm(config.num_attention_heads * self.head_dim, config.rms_norm_eps)
        self.k_norm = Olmo3RMSNorm(config.num_key_value_heads * self.head_dim, config.rms_norm_eps)
        
        self.apply_rope = True
        layer_types = getattr(config, "layer_types", None)
        self.attention_type = layer_types[layer_idx] if layer_types is not None else "full_attention"
        self.sliding_window = config.sliding_window if self.attention_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if self.sliding_window is not None and not self.mask_lmk_token:
                # When mask_lmk_token is enabled, skip KV cache truncation to avoid
                # misalignment between tensor indices and landmark positions.
                if self.layer_idx < len(past_key_values.layers):
                    kv_item = past_key_values.layers[self.layer_idx]
                    if kv_item.keys is not None and kv_item.keys.shape[-2] > self.sliding_window:
                        kv_item.keys = kv_item.keys[:, :, -self.sliding_window :, :]
                        kv_item.values = kv_item.values[:, :, -self.sliding_window :, :]



        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        kwargs.pop("position_ids", None)
        kwargs.pop("pope_pos_embeddings", None)
        if not self.mask_lmk_token:
            attn_output, attn_weights = attention_interface(
                self,
                query_states, # (B, h, L, d)
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )
            # attn_output: (B, L, h, d) — attention_interface already transposes
        else:
            attn_weights = None
            attn_output, _ = flex_attn_tl(
                query_states,
                key_states,
                value_states,
                window_size = self.sliding_window,
                chunk_size = self.chunk_size,
                training = past_key_values is None,
                mask_lmk= True,
                expand_to_chunk = False
            )
            # flex_attn_tl returns (B, h, L, d), transpose to (B, L, h, d)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Olmo3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: HSAConfig, layer_idx: int, attn_cls):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attn_cls(config=config, layer_idx=layer_idx)
        self.use_hsa_rotary_embedding = hasattr(self.self_attn, "hsa_func")

        self.mlp = Olmo3MLP(config)
        self.post_attention_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        layer_types = getattr(config, "layer_types", None)
        self.attention_type = layer_types[layer_idx] if layer_types is not None else "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class Olmo3RotaryEmbedding(nn.Module):
    def __init__(self, config: HSAConfig, device=None):
        super().__init__()
        resolve_olmo_head_dim(config)
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


    def __init__(self, config: HSAConfig, device=None):
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

        # ----------------------------------------------------------------
        # Partial-RoPE / HoPE: wavelength-based masking (opt-in).
        #   ``enable_inrange_rope`` keeps only freq-pairs whose period
        #   ``lambda_j = 2*pi / inv_freq[j]`` fits inside the training
        #   context window ``L`` -- so every kept freq completes at least
        #   ``m`` full cycles within ``L`` tokens and the model never sees
        #   rotary angles it was not trained on.  Cutoff:
        #       ``inv_freq[j] >= 2*pi*m / L``
        #   where
        #       ``L = rope_context_length``   (default max_position_embeddings)
        #       ``m = rope_period_multiplier`` (default 1.0; 2.0 = "at least
        #                                       two full cycles in L").
        #   The HoPE paper corresponds to ``m = 1.0``.
        #
        # Because HF's rotary pairs dim ``j`` with dim ``j + head_dim/2`` and
        # both share ``inv_freq[j]``, zeroing ``inv_freq[j]`` makes BOTH of
        # those head-dim components position-invariant.
        # ----------------------------------------------------------------
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self._rope_head_dim_half = head_dim // 2
        self.enable_inrange_rope = bool(getattr(config, "enable_inrange_rope", False))
        self.rope_context_length = int(
            getattr(config, "rope_context_length", config.max_position_embeddings)
        )
        self.rope_period_multiplier = float(
            getattr(config, "rope_period_multiplier", 1.0)
        )

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        if self.enable_inrange_rope:
            inv_freq = self._mask_long_period_(inv_freq)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        # print(f'self.inv_freq: {self.inv_freq}, {self.inv_freq[None, :, None]}')
        self.reinit = False

    def _mask_long_period_(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """Zero out freq-pairs whose period exceeds the training window.

        Keeps ``inv_freq[j]`` iff ``2*pi / inv_freq[j] <= L / m``, i.e.
        ``inv_freq[j] >= 2*pi*m / L`` where ``L = self.rope_context_length``
        and ``m = self.rope_period_multiplier``.  HoPE is the ``m=1`` case.
        At least one freq-pair (the very highest) is always kept so the
        rotary path does not silently degenerate to full NoPE.  Caller is
        responsible for gating this on ``self.enable_inrange_rope``.
        """
        L = self.rope_context_length
        m = self.rope_period_multiplier
        # omega >= 2*pi*m / L  <=>  period <= L/m
        threshold = (2.0 * math.pi * m) / max(L, 1)
        keep = inv_freq >= threshold                                     # (head_dim//2,)
        keep[0] = True  # guarantee at least the top freq survives
        n_keep = int(keep.sum().item())
        if not hasattr(self, "_inrange_rope_logged"):
            self._inrange_rope_logged = True
            longest_kept_period = (
                2 * math.pi / inv_freq[keep][-1].item() if n_keep > 0 else 0.0
            )
            logger.warning_once(
                f"[inrange-RoPE] head_dim_half={self._rope_head_dim_half}, "
                f"L={L}, m={m}, threshold(inv_freq)={threshold:.4e} rad/token "
                f"-> keep {n_keep} / {self._rope_head_dim_half} freq-pairs; "
                f"longest kept period = {longest_kept_period:.1f} tokens."
            )
        inv_freq[~keep] = 0.0
        return inv_freq


    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        if not self.reinit:
            self.reinit = True
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, x.device)
            if self.enable_inrange_rope:
                inv_freq = self._mask_long_period_(inv_freq)
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


class Olmo3PreTrainedModel(PreTrainedModel):
    config_class = HSAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Olmo3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = False
    _supports_static_cache = False
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
        elif isinstance(module, Olmo3RMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, HiLSModel):
            # Standalone landmark embedding (when ``enable_external_lmk_embed``
            # is on).  It plays the role of one extra row of ``embed_tokens``,
            # so we initialize it with the same ``Normal(0, std)`` schedule
            # as every other embedding row.
            if getattr(module, "lmk_embed", None) is not None:
                module.lmk_embed.data.normal_(mean=0.0, std=std)


class HiLSModel(Olmo3PreTrainedModel):

    def __init__(self, config: HSAConfig):
        super().__init__(config)
        resolve_olmo_head_dim(config)
        ensure_qwen3_rope_parameters(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = get_model_vocab_size(config)

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.full_attn_interleave = config.full_attn_interleave
        self.num_swa_layers = getattr(config, "num_swa_layers", 0)
        self.replace_full_attention_with_lhsa = getattr(config, "replace_full_attention_with_lhsa", True)
        self.chunk_size = getattr(config, 'chunk_size', 64)
        self.use_hsa_alibi = getattr(config, "use_hsa_alibi", False)
        self.enable_intra_chunk_pos = getattr(config, "enable_intra_chunk_pos", False)
        # LandmarkHSA = LandmarkHSA_pope if self.use_pope else LandmarkHSA_base
        lmk_cls = None
        if self.use_hsa_alibi:
            from .lhsa_layer_alibi import LandmarkHSA as LandmarkHSA_alibi
            lmk_cls = LandmarkHSA_alibi
        else:
            from .hils_attention import HiLSAttention as LandmarkHSA_base
            lmk_cls = LandmarkHSA_base

        def layer_type(layer_idx: int):
            if (
                self.replace_full_attention_with_lhsa
                and self.full_attn_interleave > 0
                and layer_idx >= self.num_swa_layers
                and ((layer_idx - self.num_swa_layers) % self.full_attn_interleave == self.full_attn_interleave - 1)
            ):
                return partial(lmk_cls, norm_cls=Olmo3RMSNorm)
            else:
                return Olmo3Attention
        self.layers = nn.ModuleList(
            [Olmo3DecoderLayer(config, layer_idx, attn_cls=layer_type(layer_idx)) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Olmo3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if getattr(config, "use_olmo3_rotary_embedding", False):
            # self.rotary_emb = Olmo3RotaryEmbedding(config)
            from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding as Olmo3RotaryEmbedding_olmo3
            self.rotary_emb = Olmo3RotaryEmbedding_olmo3(config=config)
        else:
            self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.hsa_rotary_emb = (
            HoPERotaryEmbedding(config=config)
            if getattr(config, 'use_hope', False)
            else None
        )
        
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        insert_landmarks = getattr(config, "insert_landmarks", True)
        enable_external_lmk_embed = getattr(config, "enable_external_lmk_embed", False)
        self.lmk_id = config.vocab_size if insert_landmarks else None
        if insert_landmarks and enable_external_lmk_embed:
            self.lmk_embed = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.lmk_embed = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        return hils_model_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            use_cache,
            output_attentions,
            output_hidden_states,
            cache_position,
            **flash_attn_kwargs,
        )


class KwargsForCausalLM(FlashAttentionKwargs): ...


class HiLSForCausalLM(Olmo3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        auto_insert_lmk = kwargs.pop('auto_insert_lmk', None)
        super().__init__(config)
        self.model = HiLSModel(config)
        self.vocab_size = get_model_vocab_size(config)
        self.chunk_size = config.chunk_size
        self.insert_landmarks = getattr(config, "insert_landmarks", True)
        self.adjust_lmk_pos = getattr(config, "adjust_lmk_pos", self.insert_landmarks)
        self.lmk_id = config.vocab_size if self.insert_landmarks else None
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.auto_insert_lmk = auto_insert_lmk if auto_insert_lmk is not None else getattr(config, 'auto_insert_lmk', False)

        self._gen_state = GenerateState()

        _auto_lmk = kwargs.pop("auto_insert_lmk", None) or getattr(config, "auto_insert_lmk", False)
        self.auto_insert_lmk = str(_auto_lmk).lower() in ('true', '1', 'yes') if isinstance(_auto_lmk, str) else bool(_auto_lmk)
        if not self.insert_landmarks and self.auto_insert_lmk:
            logger.warning_once(
                "auto_insert_lmk=True is ignored because insert_landmarks=False."
            )
            self.auto_insert_lmk = False

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

    def _filter_lmk_hidden_states(
        self,
        hidden_states: torch.Tensor,
        non_lmk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gs = self._gen_state
        if gs.lmk_positions_in_input is not None:
            mask = torch.ones(hidden_states.shape[1], dtype=torch.bool, device=hidden_states.device)
            for pos in gs.lmk_positions_in_input:
                mask[pos] = False
            hidden_states = hidden_states[:, mask, :]
            gs.lmk_positions_in_input = None
        elif non_lmk_mask is not None:
            hidden_states = hidden_states[:, non_lmk_mask, :]
        return hidden_states

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None,
        cache_position=None, position_ids=None, use_cache=True,
        logits_to_keep=0, num_logits_to_keep=None, **kwargs,
    ):
        gs = self._gen_state
        if num_logits_to_keep is not None:
            logits_to_keep = num_logits_to_keep
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            # Generation only needs the next-token logits. Keeping this at 1
            # also matches the manual cache path and avoids shape-dependent
            # bf16 lm_head differences from computing logits for the full
            # prefill sequence.
            logits_to_keep = 1

        # When landmark insertion is disabled, fall back to the standard
        # HuggingFace generation path (no special token insertion, no
        # landmark-aware position id / cache bookkeeping).
        if not self.insert_landmarks:
            gs.active = False
            gs.lmk_positions_in_input = None

            past_length = 0
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    past_length = past_key_values.get_seq_length()
                else:
                    past_length = past_key_values[0][0].shape[2] if past_key_values else 0

            if past_length > 0:
                model_input_ids = input_ids[:, -1:]
            else:
                model_input_ids = input_ids

            seq_len = model_input_ids.shape[1]
            if cache_position is None:
                cache_position = torch.arange(
                    past_length, past_length + seq_len, device=input_ids.device
                )

            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    if past_length > 0:
                        position_ids = position_ids[:, -seq_len:]
                else:
                    position_ids = cache_position.unsqueeze(0).expand(
                        model_input_ids.shape[0], -1
                    )

            return {
                "input_ids": model_input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "logits_to_keep": logits_to_keep,
            }

        if past_key_values is None or (
            isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0
        ):
            # HF GenerationMixin may pass an empty DynamicCache on the first
            # call. Treat it the same as None so prefill follows the same path
            # as manual cache decoding and full forward.
            past_key_values = None
            orig_len = input_ids.shape[1]
            position_ids = create_position_ids_with_landmarks(
                None, orig_len, self.chunk_size, input_ids.device
            )
            input_ids = insert_special_tokens(input_ids, self.lmk_id, self.chunk_size)
            seq_len_with_lmk = input_ids.shape[1]
            cache_position = torch.arange(0, seq_len_with_lmk, device=input_ids.device)

            gs.decode_token_count = orig_len % (self.chunk_size - 1)
            gs.next_pos = orig_len
            gs.cache_seq_len = seq_len_with_lmk
            gs.active = True
            gs.lmk_positions_in_input = None

            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": None,
                "cache_position": cache_position,
                "logits_to_keep": logits_to_keep,
            }

        last_token = input_ids[:, -1:]
        # DynamicCache.get_seq_length() reports the first layer's cache length.
        # In LHSA models that layer may be sliding-window attention and can be
        # truncated, while generation bookkeeping needs the full LMK-inserted
        # cache length. Track that length explicitly in GenerateState.
        if gs.cache_seq_len > 0:
            past_length = gs.cache_seq_len
        elif isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
        else:
            past_length = 0

        if cache_position is not None and cache_position.numel() > 0:
            incoming_pos = int(cache_position[-1].item())
            # HF generate may keep cache_position in raw-token coordinates, while
            # callers that follow our model cache may pass LMK-inserted positions.
            # Use past_length to distinguish the two cases.
            if incoming_pos < past_length:
                last_real_pos = incoming_pos
            else:
                last_real_pos = incoming_pos - incoming_pos // self.chunk_size
        else:
            last_real_pos = input_ids.shape[1] - 1

        chunk_offset = last_real_pos % (self.chunk_size - 1)

        if chunk_offset < self.chunk_size - 2:
            model_input_ids = last_token
            pos_ids = torch.tensor([[last_real_pos]], device=input_ids.device)
            cache_position = torch.tensor([past_length], device=input_ids.device)
            gs.decode_token_count = (chunk_offset + 1) % (self.chunk_size - 1)
            gs.next_pos = last_real_pos + 1
            gs.cache_seq_len = past_length + 1
            gs.lmk_positions_in_input = None
        else:
            lmk_token = torch.full_like(last_token, self.lmk_id)
            model_input_ids = torch.cat([last_token, lmk_token], dim=1)
            pos_ids = torch.tensor(
                [[last_real_pos, last_real_pos + 1]], device=input_ids.device
            )
            cache_position = torch.tensor([past_length, past_length + 1], device=input_ids.device)
            gs.lmk_positions_in_input = [1]
            gs.decode_token_count = 0
            gs.next_pos = last_real_pos + 1
            gs.cache_seq_len = past_length + 2

        return {
            "input_ids": model_input_ids,
            "position_ids": pos_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": None,
            "cache_position": cache_position,
            "logits_to_keep": logits_to_keep,
        }

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
        return hils_causal_lm_forward(
            self,
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            cache_position,
            logits_to_keep,
            **kwargs,
        )


# if USE_LIGER_ROPE:
#     apply_rotary_pos_emb = liger_rotary_pos_emb
#     logger.info_rank0("Apply liger RoPE kernel to Olmo3.")
if USE_FLASH_ATTN_RMSNORM:
    Olmo3RMSNorm = Olmo3FlashAttnRMSNorm
    logger.info_rank0("Apply flash-attn RMSNorm kernel to Olmo3.")
if USE_LIGER_SWIGLU:
    Olmo3MLP = LigerSwiGLUMLP
    logger.info_rank0("Apply liger SwiGLU kernel to Olmo3.")

logger.info_rank0(f"Liger: KERNEL={USE_LIGER_KERNEL}, ROPE={USE_LIGER_ROPE}, SWIGLU={USE_LIGER_SWIGLU}, RMSNORM={USE_FLASH_ATTN_RMSNORM}")

__all__ = [
    "HiLSForCausalLM",
    "HiLSModel",
    "Olmo3PreTrainedModel",
    "Olmo3RMSNorm",
    "Olmo3TorchRMSNorm",
    "Olmo3FlashAttnRMSNorm",
]
