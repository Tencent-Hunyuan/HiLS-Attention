"""Naive Block Sparse Attention (BSA) layer.

Drop-in attention module implementing the paper's naive BSA
(exact chunk-mass top-K + factorized sparse attention) via
``naive_bsa_kernel`` (or ``naive_bsa_attention`` when
``use_naive_bsa_torch_ref`` is set).

Constraints: GQA-friendly, ``chunk_size >= 32``, ``hsa_topk`` preferably
a power of 2, training-only (no KV cache), plain causal mask.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from veomni.utils import logging

from .naive_bsa_kernel import naive_bsa_kernel
from .naive_bsa_ref import naive_bsa_attention


logger = logging.get_logger(__name__)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(q_dtype), k_embed.to(k_dtype)


class NaiveBSA(nn.Module):
    """Naive BSA attention layer."""

    def __init__(self, config, layer_idx, norm_cls=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.d_model = config.hidden_size
        self.h_q = config.num_attention_heads
        self.h_kv = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            self.d_model // self.h_q
        )

        assert self.h_q % self.h_kv == 0, (
            f"NaiveBSA requires num_attention_heads ({self.h_q}) to be a "
            f"multiple of num_key_value_heads ({self.h_kv}) for GQA grouping."
        )
        assert norm_cls is not None, "NaiveBSA expects a norm_cls (e.g. RMSNorm)"

        self.q_proj = nn.Linear(
            self.d_model, self.h_q * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.d_model, self.h_kv * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.d_model, self.h_kv * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.h_q * self.head_dim, self.d_model, bias=False
        )

        self.q_norm = norm_cls(self.head_dim)
        self.k_norm = norm_cls(self.head_dim)

        self.chunk_size = config.chunk_size
        self.topk = config.hsa_topk
        self.hsa_sliding_window = getattr(
            config, "hsa_sliding_window", config.sliding_window
        )
        self.apply_hsa_rope = getattr(config, "apply_hsa_rope", True)
        self.scaling = self.head_dim ** -0.5
        self.is_causal = True

        if self.chunk_size < 32:
            raise ValueError(
                f"NaiveBSA: chunk_size must be >= 32 (HiLS_block_M_head "
                f"TensorCore constraint); got {self.chunk_size}"
            )

        self.use_torch_ref = bool(
            getattr(config, "use_naive_bsa_torch_ref", False)
        )
        if self.use_torch_ref:
            logger.info_once(
                "NaiveBSA: using torch ref (naive_bsa_attention), not kernel."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if use_cache or past_key_values is not None:
            raise NotImplementedError(
                "NaiveBSA does not support KV caching yet (training-only)."
            )
        if attention_mask is not None:
            logger.warning_once(
                "NaiveBSA ignores attention_mask; assumes plain causal attention."
            )
        if position_embeddings is None:
            raise ValueError(
                "NaiveBSA needs (cos, sin) via position_embeddings."
            )

        B, L, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = rearrange(
            self.q_proj(hidden_states), "B L (h d) -> B L h d", d=self.head_dim
        )
        k = rearrange(
            self.k_proj(hidden_states), "B L (h d) -> B L h d", d=self.head_dim
        )
        v = rearrange(
            self.v_proj(hidden_states), "B L (h d) -> B L h d", d=self.head_dim
        )

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.apply_hsa_rope:
            q_bhld = q.transpose(1, 2)
            k_bhld = k.transpose(1, 2)
            q_bhld, k_bhld = apply_rotary_pos_emb(q_bhld, k_bhld, cos, sin)
            q = q_bhld.transpose(1, 2).contiguous()
            k = k_bhld.transpose(1, 2).contiguous()

        v = v.contiguous()

        bsa_fn = naive_bsa_attention if self.use_torch_ref else naive_bsa_kernel
        o = bsa_fn(
            q, k, v,
            chunk_size=self.chunk_size,
            window_size=self.hsa_sliding_window,
            topk=self.topk,
            sm_scale=self.scaling,
        )

        o = rearrange(o, "B L h d -> B L (h d)")
        return self.o_proj(o), None
