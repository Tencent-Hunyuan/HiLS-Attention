from typing import Any, Callable, Optional, Tuple, Union
from dataclasses import dataclass, field

from .HoPE import HoPERotaryEmbedding
import torch
import math
from torch import nn
from .hils_attention import HiLSAttention
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
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

from functools import partial
from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.utils import logging
from veomni.utils.import_utils import (
    is_liger_kernel_available,
    is_torch_npu_available,
    is_transformers_version_greater_or_equal_to,
)
from veomni.models.module_utils import GradientCheckpointingLayer
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
from .hils_forward import hils_model_forward, hils_causal_lm_forward
from ops.flex_attn_tilelang import flex_attn_tl


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
    active: bool = False
    decode_token_count: int = 0
    next_pos: int = 0
    cache_seq_len: int = 0
    lmk_positions_in_input: Optional[list] = None

    def reset(self):
        self.active = False
        self.decode_token_count = 0
        self.next_pos = 0
        self.cache_seq_len = 0
        self.lmk_positions_in_input = None


def next_of_y(x, y):
    return (x + y - 1) // y * y


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
    
class FlashAttnRMSNorm(Qwen3RMSNorm):
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


class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
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
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape

        self.apply_rope = True
        self.sliding_window = config.sliding_window

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

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (B, h, L, d)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # (B, h, L, d)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # assert not torch.allclose(q_rot, query_states, atol=1e-5)
        # assert not torch.allclose(k_rot, key_states, atol=1e-5)
        # query_states = q_rot
        # key_states = k_rot
        

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            if self.sliding_window is not None and not self.mask_lmk_token:
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
            q_blhd = query_states.transpose(1, 2).contiguous()
            k_blhd = key_states.transpose(1, 2).contiguous()
            v_blhd = value_states.transpose(1, 2).contiguous()
            kv_start = 0
            if past_key_values is not None:
                kv_end = None
                if cache_position is not None and cache_position.numel() > 0:
                    kv_end = int(cache_position[-1].item()) + 1
                elif self.layer_idx < len(past_key_values.layers):
                    kv_layer = past_key_values.layers[self.layer_idx]
                    if hasattr(kv_layer, "get_seq_length"):
                        kv_end = int(kv_layer.get_seq_length())
                if kv_end is not None:
                    kv_start = max(kv_end - k_blhd.shape[1], 0)
            attn_output, _ = flex_attn_tl(
                q_blhd,
                k_blhd,
                v_blhd,
                window_size = self.sliding_window,
                chunk_size = self.chunk_size,
                training = self.training,
                mask_lmk= True,
                expand_to_chunk = False,
                kv_start=kv_start,
            )
            # flex_attn_tl returns (B, L, h, d) directly

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int, attn_cls):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = attn_cls(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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
            past_key_values=past_key_values,
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

        return cos, sin


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
        elif isinstance(module, HiLSModel):
            # Standalone landmark embedding (when ``enable_external_lmk_embed``
            # is on).  It plays the role of one extra row of ``embed_tokens``,
            # so we initialize it with the same ``Normal(0, std)`` schedule
            # as every other embedding row.
            if getattr(module, "lmk_embed", None) is not None:
                module.lmk_embed.data.normal_(mean=0.0, std=std)


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

            If `past_key_valuess` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_valuess` key value states are returned and can be used to speed up decoding (see
            `past_key_valuess`).
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
class HiLSModel(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: Qwen3Config
    """

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = next_of_y(config.vocab_size + 1, 32)

        self.embed_tokens = nn.Embedding(self.vocab_size, config.hidden_size, self.padding_idx)
        self.full_attn_interleave = config.full_attn_interleave
        self.num_swa_layers = getattr(config, "num_swa_layers", 0)
        if self.num_swa_layers > 0 and self.num_swa_layers != config.num_hidden_layers // 2:
            logger.warning_once("Recomment num_swa_layers to be half of num_hidden_layers")

        from .hils_attention import HiLSAttention
        lmk_cls = HiLSAttention

        def layer_type(layer_idx: int):
            if layer_idx < self.num_swa_layers:
                return Qwen3Attention
            if self.full_attn_interleave > 0 and ((layer_idx - self.num_swa_layers) % self.full_attn_interleave == self.full_attn_interleave - 1):
                return partial(lmk_cls, norm_cls=Qwen3RMSNorm)
            else:
                return Qwen3Attention
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx, attn_cls=layer_type(layer_idx)) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not getattr(config, 'use_hope', False):
            pos_cls = Qwen3RotaryEmbedding
        else:
            pos_cls = HoPERotaryEmbedding
        self.rotary_emb = pos_cls(config=config)
        
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
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        **flash_attn_kwargs,
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


def get_model_vocab_size(config) -> int:
    enable_external_lmk_embed = getattr(config, "enable_external_lmk_embed", False)
    if not enable_external_lmk_embed:
        return next_of_y(config.vocab_size + 1, 32)
    return config.vocab_size

class HiLSForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, **kwargs):
        auto_insert_lmk = kwargs.pop('auto_insert_lmk', None)
        super().__init__(config)
        self.model = HiLSModel(config)
        self.vocab_size = get_model_vocab_size(config)
        self.chunk_size = config.chunk_size
        self.lmk_id = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.insert_landmarks = getattr(config, 'insert_landmarks', True)
        self.adjust_lmk_pos = getattr(config, "adjust_lmk_pos", self.insert_landmarks)
        self.auto_insert_lmk = auto_insert_lmk if auto_insert_lmk is not None else getattr(config, 'auto_insert_lmk', False)

        self._gen_state = GenerateState()

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

    @staticmethod
    def _is_empty_generation_cache(past_key_values) -> bool:
        if past_key_values is None:
            return True
        if not isinstance(past_key_values, Cache):
            return False
        layers = getattr(past_key_values, "layers", None)
        if layers is None:
            return past_key_values.get_seq_length() == 0
        return all(layer.get_seq_length() == 0 for layer in layers)

    def generate(self, *args, **kwargs):
        self._gen_state.reset()
        try:
            return GenerationMixin.generate(self, *args, **kwargs)
        finally:
            self._gen_state.reset()

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
            logits_to_keep = 1

        if not self.insert_landmarks:
            gs.active = False
            gs.lmk_positions_in_input = None

            past_length = 0
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    past_length = past_key_values.get_seq_length()
                else:
                    past_length = past_key_values[0][0].shape[2] if past_key_values else 0

            model_input_ids = input_ids[:, -1:] if past_length > 0 else input_ids
            seq_len = model_input_ids.shape[1]
            if cache_position is None:
                cache_position = torch.arange(past_length, past_length + seq_len, device=input_ids.device)

            if position_ids is None:
                if attention_mask is not None:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    if past_length > 0:
                        position_ids = position_ids[:, -seq_len:]
                else:
                    position_ids = cache_position.unsqueeze(0).expand(model_input_ids.shape[0], -1)

            return {
                "input_ids": model_input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "logits_to_keep": logits_to_keep,
            }

        if self._is_empty_generation_cache(past_key_values):
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
        if gs.cache_seq_len > 0:
            past_length = gs.cache_seq_len
        elif isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
        else:
            past_length = 0

        if cache_position is not None and cache_position.numel() > 0:
            incoming_pos = int(cache_position[-1].item())
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
            pos_ids = torch.tensor([[last_real_pos, last_real_pos + 1]], device=input_ids.device)
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
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
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

__all__ = ["HiLSForCausalLM", "HiLSModel", "Qwen3PreTrainedModel"]
