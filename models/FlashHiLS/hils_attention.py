from typing import Optional, Tuple
import math
import torch.nn as nn
from einops import rearrange
from veomni.utils import logging
from ops.rope_tilelang_fp32 import single_tensor_rope_autograd
import torch
import torch.nn.functional as F
from ops.flex_attn_tilelang import flex_attn_tl
from ops.chunk_attn_pool_tilelang import chunk_attn_pool_tilelang
from veomni.utils.import_utils import (
    is_liger_kernel_available,
)
if is_liger_kernel_available():
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

logger = logging.get_logger(__name__)


def _chunk_attn_pool_impl(
    mu_q: torch.Tensor,
    k_chunked: torch.Tensor,
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    in_dtype = k_chunked.dtype
    mu_f32 = mu_q.float()
    k_f32 = k_chunked.float()
    h_q = mu_f32.shape[2]
    h_kv = k_f32.shape[3]
    G = h_q // h_kv

    if G != 1:
        k_q = k_f32.repeat_interleave(G, dim=3)
    else:
        k_q = k_f32

    logits = torch.einsum("bnhd,bnshd->bnsh", mu_f32, k_q) * sm_scale

    S_chunk = logits.shape[2]
    last_mask = torch.zeros(S_chunk, dtype=torch.bool, device=logits.device)
    last_mask[-1] = True
    logits = logits.masked_fill(last_mask.view(1, 1, S_chunk, 1), float('-inf'))

    p = F.softmax(logits, dim=2)
    lmk_k = torch.einsum("bnsh,bnshd->bnhd", p, k_q).to(in_dtype)

    log_p = F.log_softmax(logits, dim=2)
    log_p_safe = torch.where(
        torch.isfinite(log_p), log_p, log_p.new_zeros(())
    )
    lmk_b = -(p * log_p_safe).sum(dim=2)

    return lmk_k, lmk_b


try:
    _chunk_attn_pool_compiled = torch.compile(
        _chunk_attn_pool_impl,
        fullgraph=True,
        dynamic=True,            # B / N change with sequence length, so dynamic.
    )
except Exception:                  # noqa: BLE001
    _chunk_attn_pool_compiled = None


def chunk_attn_pool(
    mu_q: torch.Tensor,
    k_chunked: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parameter-free Taylor-expansion chunk pooling.

    For each chunk ``c`` we use the lmk_q at the chunk's LAST token as the
    "prior" query ``mu_q[c]`` and compute attention over the K tokens INSIDE
    that chunk:

        p_j = softmax_j(s * <mu_q[c], K_chunked[c, j]>)
        k'  = sum_j p_j * K_chunked[c, j]      # surrogate KEY  (same shape as a single K)
        b'  = -sum_j p_j * log p_j             # surrogate BIAS (entropy of the prior attn)

    By a first-order Taylor expansion of ``g(q) = log sum_j exp(s * q . K_j)``
    around ``q = mu_q``, we have ``g(q_i) ~= s * q_i . k' + b' + const`` --
    so ``(k', b')`` is a strictly more informative chunk summary than just
    "last-token-as-lmk", at the cost of one extra in-chunk attention.

    Args:
        mu_q:      (B, num_chunks, h_q, d) -- prior query per (batch, chunk, q-head).
                   Caller is responsible for picking the right slice from
                   lmk_q_norm (typically: ``lmk_q_norm[:, S-1::S, :, :]``,
                   i.e. the last lmk_q in each chunk).
        k_chunked: (B, num_chunks, S, h_kv, d) -- K reshaped per-chunk, at
                   the model's NATIVE KV-head count (no caller-side
                   ``repeat_interleave``).  GQA group size ``G = h_q // h_kv``
                   is inferred internally; ``h_q == h_kv`` (no GQA) is fine.
        sm_scale:  attention scale.  Defaults to ``1 / sqrt(d)``.

    Returns:
        lmk_k: (B, num_chunks, h_q, d), in K's dtype.
        lmk_b: (B, num_chunks, h_q), always in fp32.

    Notes:
        * Pure function, no parameters; the only "learnable" thing influencing
          ``mu_q`` is whatever produced ``lmk_q_norm`` upstream (lmk_q_proj +
          its norm), which the caller already trains.
        * All in-chunk math is fp32 for stable softmax/log; outputs cast back.
        * ``mu_q`` and ``k_chunked`` MUST agree on the head count ``H`` and
          the per-head dim ``d``.  When the model uses GQA-style retrieval
          (``h_q != h_kv``), the caller must broadcast/
          repeat-interleave ``mu_q`` (or ``k_chunked``) BEFORE calling this.
        * Implementation dispatches to the fused TileLang kernel for speed;
          falls back to torch.compile/eager for shapes the kernel doesn't cover.
    """
    if mu_q.dim() != 4 or k_chunked.dim() != 5:
        raise ValueError(
            f"chunk_attn_pool: expected mu_q (B, N, h_q, d) and k_chunked "
            f"(B, N, S, h_kv, d); got {tuple(mu_q.shape)} and {tuple(k_chunked.shape)}"
        )
    B, N, h_q, D = mu_q.shape
    Bk, Nk, S, h_kv, Dk = k_chunked.shape
    if (B, N, D) != (Bk, Nk, Dk):
        raise ValueError(
            f"chunk_attn_pool: mu_q vs k_chunked shape mismatch on "
            f"(B, N, d) -- got mu_q={tuple(mu_q.shape)}, "
            f"k_chunked={tuple(k_chunked.shape)}"
        )
    if h_q % h_kv != 0:
        raise ValueError(
            f"chunk_attn_pool: mu_q h_q ({h_q}) must be a multiple of "
            f"k_chunked h_kv ({h_kv}) for GQA grouping"
        )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    return chunk_attn_pool_tilelang(mu_q, k_chunked, sm_scale)


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

class HiLSAttention(nn.Module):
    def __init__(self, config, layer_idx, norm_cls=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.head_dim = self.d_model // config.num_attention_heads
        self.d_kv = self.head_dim * config.num_key_value_heads
        self.h_kv = config.num_key_value_heads
        self.h_q = config.num_attention_heads

        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.h_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.h_kv * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.enable_lmk_q_proj = getattr(config, "enable_lmk_q_proj", False)
        self.apply_hsa_rope = getattr(config, "apply_hsa_rope", False)

        if self.enable_lmk_q_proj:
            self.lmk_q_lora_dim = getattr(config, 'lmk_q_lora_dim', -1)
            if self.lmk_q_lora_dim <= 0:
                self.lmk_q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
            else:
                self.lmk_q_proj = nn.Sequential(
                    nn.Linear(self.d_model, self.lmk_q_lora_dim, bias=False),
                    nn.Linear(self.lmk_q_lora_dim, self.d_model, bias=False)
                )
            self.layerwise_lmkq_norm = getattr(config, "layerwise_lmkq_norm", False)
            if self.layerwise_lmkq_norm:
                self.lmk_q_norm = norm_cls(self.d_model)
            else:
                self.lmk_q_norm = norm_cls(self.head_dim)

        if not self.enable_lmk_q_proj:
            logger.warning_once("Recommend to set enable_lmk_q_proj=True for better performance")

        self.topk = config.hsa_topk

        self.chunk_size = config.chunk_size
        self.layerwise_qk_norm = getattr(config, "layerwise_qk_norm", False)
        if not self.layerwise_qk_norm:
            self.q_norm = norm_cls(self.head_dim)
            self.k_norm = norm_cls(self.head_dim)
        else:
            self.q_norm = norm_cls(self.d_model)
            self.k_norm = norm_cls(self.h_kv * self.head_dim)

        if self.enable_lmk_q_proj and not self.layerwise_lmkq_norm:
            self.lmk_q_norm = norm_cls(self.head_dim)

        self.scaling = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window
        self.hsa_sliding_window = getattr(config, "hsa_sliding_window", self.sliding_window)
        
        self.is_causal = True
        
        from ops.hils_fwd_bwd_head import HSA_block_M_head as HSA
        from ops.topk_head_softmax import online_softmax_topk_head as topk_func
        self.topk_func = topk_func
        self.hsa_func = HSA

        self.enable_prior_query = getattr(config, "enable_prior_query", True)
        self.enable_chunk_pooling = getattr(config, "enable_chunk_pooling", False)
        self.shared_q_c = getattr(config, "shared_q_c", False)
        if self.shared_q_c:
            assert self.enable_prior_query, "shared_q_c requires enable_prior_query=True"
            std = getattr(config, "initializer_range", 0.02)
            self.q_c = nn.Parameter(torch.empty(self.h_q, self.head_dim))
            nn.init.normal_(self.q_c, mean=0.0, std=std)

        self.enable_softmax1 = config.enable_softmax1
        

    def forward(self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        B, L, _ = hidden_states.shape
        cos, sin = position_embeddings

        hsa_q = self.q_proj(hidden_states)
        if self.enable_lmk_q_proj:
            lmk_q = self.lmk_q_proj(hidden_states)
            if self.lmk_q_lora_dim > 0:
                lmk_q = lmk_q + hsa_q
            if self.layerwise_lmkq_norm:
                lmk_q_norm = self.lmk_q_norm(lmk_q)
                lmk_q_norm = rearrange(lmk_q_norm, 'B L (h d)->B L h d', h=self.h_q)
            else:
                lmk_q = rearrange(lmk_q, 'B L (h d)->B L h d', h=self.h_q)
                lmk_q_norm = self.lmk_q_norm(lmk_q)  # (B, L, h_q, d)

        if self.layerwise_qk_norm:
            hsa_q = self.q_norm(hsa_q)
        hsa_q = rearrange(hsa_q, 'B L (h d)->B L h d', d=self.head_dim)

        if not self.layerwise_qk_norm:
            hsa_q_norm_nope = self.q_norm(hsa_q)  # (B, L, h, d)
        else:
            hsa_q_norm_nope = hsa_q

        if not self.enable_lmk_q_proj:
            lmk_q_norm = hsa_q_norm_nope

        hsa_k = self.k_proj(hidden_states)
        if self.layerwise_qk_norm:
            hsa_k = self.k_norm(hsa_k)
        hsa_k = rearrange(hsa_k, 'B L (h d)->B L h d', d=self.head_dim)

        if not self.layerwise_qk_norm:
            hsa_k_norm_nope = self.k_norm(hsa_k)
        else:
            hsa_k_norm_nope = hsa_k

        hsa_v = self.v_proj(hidden_states)
        hsa_v = rearrange(hsa_v, 'B L (h d)->B L h d', d=self.head_dim)

        if self.apply_hsa_rope:
            hsa_q_norm_rope, hsa_k_norm_rope = apply_rotary_pos_emb(hsa_q_norm_nope.transpose(1, 2), hsa_k_norm_nope.transpose(1, 2), cos, sin)
            hsa_q_norm_rope = hsa_q_norm_rope.transpose(1, 2).contiguous()
            hsa_k_norm_rope = hsa_k_norm_rope.transpose(1, 2).contiguous()
            if self.enable_lmk_q_proj:
                lmk_q_norm = single_tensor_rope_autograd(lmk_q_norm, cos, sin)
            else:
                lmk_q_norm = hsa_q_norm_rope
        else:
            hsa_q_norm_rope = hsa_q_norm_nope
            hsa_k_norm_rope = hsa_k_norm_nope

        # Inference/chunk-prefill: HSA KV cache.  Keep the cached K in the
        # same representation that retrieval and HSA attention consume.
        hsa_k_for_cache = hsa_k_norm_rope if self.apply_hsa_rope else hsa_k_norm_nope
        if use_cache and past_key_values is not None:
            hsa_cache_idx = self.layer_idx + self.config.num_hidden_layers
            cache_kwargs = {"cache_position": cache_position}
            hsa_k_for_cache, hsa_v = past_key_values.update(
                hsa_k_for_cache, hsa_v, hsa_cache_idx, cache_kwargs
            )
            if self.apply_hsa_rope:
                hsa_k_norm_rope = hsa_k_for_cache
            else:
                hsa_k_norm_nope = hsa_k_for_cache
                hsa_k_norm_rope = hsa_k_norm_nope

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        doc_ids = kwargs.get("doc_ids", None)
        if cu_seq_lens_q is not None and cu_seq_lens_q.shape[0] > 2:
            assert hidden_states.shape[0] == 1, f'cu_seq_lens_q is only supported for batch size 1, but got {hidden_states.shape[0]}'
            assert torch.all((cu_seq_lens_q[:-1] % self.chunk_size) ==0), f'cu_seq_lens_q must be divisible by chunk_size, cu_seq_lens_q: {cu_seq_lens_q}'
        else:
            cu_seq_lens_q = None

        if hidden_states.shape[0] > 1:
            assert doc_ids is None, f'doc_ids is not supported for batch size > 1, but got doc ids {doc_ids}'

        swa_o, lse_sum = flex_attn_tl(
            hsa_q_norm_rope.transpose(1, 2),
            hsa_k_norm_rope.transpose(1, 2),
            hsa_v.transpose(1, 2),
            window_size=self.hsa_sliding_window,
            chunk_size=self.chunk_size,
            training=(past_key_values is None),
            mask_lmk=True,
            expand_to_chunk=True
        )
        lse_sum = lse_sum.contiguous()
        lse_sum = lse_sum.to(hidden_states.dtype)

        full_seq_len = hsa_k_for_cache.shape[1]
        if full_seq_len >= self.chunk_size:
            if self.apply_hsa_rope:
                lmk_k_source = hsa_k_norm_rope
            else:
                lmk_k_source = hsa_k_norm_nope

            prior_b = None  # (B, num_chunks, h_q) or None
            full_chunks = full_seq_len // self.chunk_size
            if self.enable_prior_query:
                h_q_lmk = self.h_q if self.shared_q_c else lmk_q_norm.shape[2]

                is_continuation = (use_cache and past_key_values is not None and L < full_seq_len)

                if is_continuation:
                    if not hasattr(past_key_values, '_hsa_prior_cache'):
                        past_key_values._hsa_prior_cache = {}
                    cache_key = self.layer_idx
                    cached = past_key_values._hsa_prior_cache.get(cache_key, None)

                    prev_seq_len = full_seq_len - L
                    prev_full_chunks = prev_seq_len // self.chunk_size
                    new_chunks = full_chunks - prev_full_chunks

                    if new_chunks > 0:
                        new_k = lmk_k_source[:, prev_full_chunks * self.chunk_size : full_chunks * self.chunk_size, :, :]
                        new_k = rearrange(new_k, "b (n s) h d -> b n s h d", s=self.chunk_size)
                        new_k_native = new_k

                        boundary_local = [
                            self.chunk_size * (prev_full_chunks + k) - 1 - prev_seq_len
                            for k in range(1, new_chunks + 1)
                        ]
                        if self.shared_q_c:
                            new_mu_q = self.q_c.unsqueeze(0).unsqueeze(0).expand(B, new_chunks, -1, -1)
                            if self.apply_hsa_rope:
                                q_c_cos = cos[:, boundary_local, :]
                                q_c_sin = sin[:, boundary_local, :]
                                new_mu_q = single_tensor_rope_autograd(new_mu_q.contiguous(), q_c_cos, q_c_sin)
                        else:
                            new_mu_q = lmk_q_norm[:, boundary_local, :, :]  # (B, new_chunks, h_q_lmk, D)

                        if h_q_lmk != self.h_kv:
                            assert h_q_lmk % self.h_kv == 0, (
                                f"lmk_q head count ({h_q_lmk}) must be a multiple of "
                                f"h_kv ({self.h_kv}) for GQA-style chunk pool"
                            )
                            g = h_q_lmk // self.h_kv
                            new_k = new_k.repeat_interleave(g, dim=3)

                        new_lmk_k, new_prior_b = chunk_attn_pool(
                            new_mu_q,
                            new_k,
                        )
                        if self.enable_chunk_pooling:
                            new_lmk_k = new_k_native.mean(dim=2)

                        if cached is not None:
                            lmk_k = torch.cat([cached['lmk_k'], new_lmk_k], dim=1)
                            prior_b = torch.cat([cached['prior_b'], new_prior_b], dim=1)
                        else:
                            lmk_k = new_lmk_k
                            prior_b = new_prior_b

                        past_key_values._hsa_prior_cache[cache_key] = {
                            'lmk_k': lmk_k,
                            'prior_b': prior_b,
                        }
                    else:
                        if cached is not None:
                            lmk_k = cached['lmk_k']
                            prior_b = cached['prior_b']
                        else:
                            prior_b = None
                            lmk_k = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, :, :]
                else:
                    k_chunked = lmk_k_source[:, : full_chunks * self.chunk_size]
                    k_chunked = rearrange(
                        k_chunked, "b (n s) h d -> b n s h d", s=self.chunk_size
                    )
                    k_chunked_native = k_chunked

                    if self.shared_q_c:
                        mu_q = self.q_c.unsqueeze(0).unsqueeze(0).expand(B, full_chunks, -1, -1)
                        if self.apply_hsa_rope:
                            q_c_cos = cos[:, self.chunk_size - 1::self.chunk_size, :][:, :full_chunks]
                            q_c_sin = sin[:, self.chunk_size - 1::self.chunk_size, :][:, :full_chunks]
                            mu_q = single_tensor_rope_autograd(mu_q.contiguous(), q_c_cos, q_c_sin)
                    else:
                        mu_q = lmk_q_norm[:, self.chunk_size - 1::self.chunk_size, :, :][:, :full_chunks]
                    if mu_q.shape[1] == 0:
                        prior_b = None
                        if self.enable_chunk_pooling and full_chunks > 0:
                            lmk_k = k_chunked_native.mean(dim=2)
                        else:
                            lmk_k = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, :, :]
                    else:
                        if h_q_lmk != self.h_kv:
                            assert h_q_lmk % self.h_kv == 0, (
                                f"lmk_q head count ({h_q_lmk}) must be a multiple of "
                                f"h_kv ({self.h_kv}) for GQA-style chunk pool"
                            )
                            g = h_q_lmk // self.h_kv
                            k_chunked = k_chunked.repeat_interleave(g, dim=3)
                        pooled_lmk_k, prior_b = chunk_attn_pool(
                            mu_q,
                            k_chunked,
                        )
                        if self.enable_chunk_pooling:
                            lmk_k = k_chunked_native.mean(dim=2)
                        else:
                            lmk_k = pooled_lmk_k

                    if use_cache and past_key_values is not None and prior_b is not None:
                        if not hasattr(past_key_values, '_hsa_prior_cache'):
                            past_key_values._hsa_prior_cache = {}
                        past_key_values._hsa_prior_cache[self.layer_idx] = {
                            'lmk_k': lmk_k,
                            'prior_b': prior_b,
                        }
            else:
                if self.enable_chunk_pooling:
                    lmk_k = lmk_k_source[:, : full_chunks * self.chunk_size, :, :]
                    lmk_k = rearrange(lmk_k, "b (n s) h d -> b n s h d", s=self.chunk_size).mean(dim=2)
                else:
                    lmk_k = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, :, :]  # (B, L // S, hsa_kv, d)

            B, S,  H, D = lmk_k.shape
            lmk_k = lmk_k.reshape(B, S, H, D)
            # q_offset = int(cache_position[0].item()) if (use_cache and cache_position is not None) else 0
            q_offset = full_seq_len - L if (use_cache and past_key_values is not None) else 0

            assert cu_seq_lens_q is None, "cu_seq_lens_q is not supported for headwise topk"
            
            indices, scores = self.topk_func(
                lmk_q_norm,
                lmk_k,
                lse_sum,
                self.topk,
                block_size=self.chunk_size,
                window_size=self.hsa_sliding_window,
                is_training=self.training,
                is_causal=True,
                q_offset=q_offset,
                G=lmk_k.shape[2] // self.h_kv,
                bias=prior_b
            )
            # indices: [N, L, h_kv, K]
            # scores: [N, L, h_q, K]


            if self.enable_prior_query and prior_b is not None:
                K_topk = indices.shape[-1]
                h_kv_idx = indices.shape[2]
                h_q_b = prior_b.shape[2]
                if h_q_b != h_kv_idx:
                    assert h_q_b % h_kv_idx == 0, (
                        f"prior_b head ({h_q_b}) must be a multiple of "
                        f"indices head ({h_kv_idx}) for GQA broadcast"
                    )
                    g_kernel = h_q_b // h_kv_idx
                    indices_hq = indices.repeat_interleave(g_kernel, dim=2) # (B, L, h_q, K)
                else:
                    indices_hq = indices
                src = prior_b.unsqueeze(-1).expand(-1, -1, -1, K_topk)       # (B, N, h_q, K), view-only
                idx = indices_hq.clamp_min(0).long()                         # (B, L, h_q, K)
                gathered = torch.gather(src, dim=1, index=idx)               # (B, L, h_q, K)

                scores = scores + gathered.to(scores.dtype)

            if not self.enable_softmax1:
                cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1)], dim=-1)  # (B, L, h_kv, K + 1)
                swa_weight_idx = -1
            else:
                cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1), torch.zeros(B, L, scores.shape[2], 1, device=hidden_states.device)], dim=-1)
                swa_weight_idx = -2
            chunk_weights = F.softmax(cat_scores, dim=-1).to(hidden_states.dtype) # (B, L, h_kv, K)

            if self.apply_hsa_rope:
                hsa_q_norm = hsa_q_norm_rope
                hsa_k_norm = hsa_k_norm_rope
            else:
                hsa_q_norm = hsa_q_norm_nope
                hsa_k_norm = hsa_k_norm_nope

            hsa_o = self.hsa_func(hsa_q_norm, hsa_k_norm, hsa_v, weights=chunk_weights, indices=indices, block_size=self.chunk_size, mask_last_token=True, is_training=self.training)
            swa_o_weight = chunk_weights[:, :, :, swa_weight_idx]  # (B, L, h_kv)

            swa_o_weight_expanded = swa_o_weight
            o_lower = torch.addcmul(hsa_o, swa_o, swa_o_weight_expanded.unsqueeze(-1))
        else:
            if self.enable_softmax1:
                swa_o_weight = torch.sigmoid(lse_sum).to(swa_o.dtype)
                o_lower = swa_o * swa_o_weight.unsqueeze(-1)
            else:
                o_lower = swa_o
        
        o = rearrange(o_lower, 'B L h d->B L (h d)')
        
        return self.o_proj(o), None

if is_liger_kernel_available():
    apply_rotary_pos_emb = liger_rotary_pos_emb
    logger.info_rank0("Apply liger kernel to LHSA.")
