import torch
import torch.nn as nn

from typing import Optional, Tuple, Union, Dict, Callable
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb


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

class SlidingWindowAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

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
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        assert self.sliding_window is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Dict] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # print(f'layer: {self.layer_idx} self.p_proj dtype: {self.q_proj.weight.dtype}, hidden states dtype: {hidden_states.dtype}')

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        assert cos.shape[1] == query_states.shape[-2]
        # print(f'before apply rope: {query_states.shape}, {cos.shape}, {sin.shape}')
        # if self.training:
        query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin)
        # else:
        #     # issue: liger_rotary_pos_emb nan when eval

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            # cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            past_key_cache, past_value_cache = past_key_value.get(self.layer_idx, (None, None))
            if past_key_cache is None and past_value_cache is None:
                past_key_cache = key_states
                past_value_cache = value_states
            else:
                key_states = torch.cat([past_key_cache, key_states], dim=-2)
                value_states = torch.cat([past_value_cache, value_states], dim=-2)

            # TODO: implement with torch.roll
            past_key_value[self.layer_idx] = (key_states[:, :, -self.sliding_window:, :].contiguous(), value_states[:, :, -self.sliding_window:, :].contiguous())

        # if query_states.dtype not in (torch.float16, torch.bfloat16):
        #     query_states = query_states.to(torch.bfloat16)
        #     key_states = key_states.to(torch.bfloat16)
        #     value_states = value_states.to(torch.bfloat16)

        # assert not torch.any(torch.isnan(query_states))
        # assert not torch.any(torch.isnan(key_states))
        # assert not torch.any(torch.isnan(value_states))
        # TODO: replace with SP attention ops
        # print(f'qkv shape: {query_states.transpose(1, 2).shape}, {query_states.transpose(1, 2).is_contiguous()}, {key_states.transpose(1, 2).shape}')
        # attn_output = flash_attn_func(
        #     query_states,
        #     key_states,
        #     value_states,
        #     softmax_scale=None,
        #     causal=True,
        #     window_size=(-self.sliding_window, 0),  # diff with Llama
        # )
        # assert not torch.any(torch.isnan(attn_output))

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
            None,
            dropout=0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs
        )


        # print(f'attn output dtype: {attn_output.dtype} o_proj.dtype: {self.o_proj.weight.dtype}')
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output