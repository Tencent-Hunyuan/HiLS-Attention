from typing import Any, Callable, Optional, Tuple, Union
import math
import torch.nn as nn
from einops import rearrange
from veomni.utils import logging
from ops.rope_tilelang_fp32 import single_tensor_rope_autograd, hsa_intra_chunk_rope
import torch
import os
import torch.nn.functional as F
from ops.flex_attn_tilelang import flex_attn_tl
from ops.chunk_attn_pool_tilelang import chunk_attn_pool_tilelang
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from veomni.utils.import_utils import (
    is_liger_kernel_available,
)
if is_liger_kernel_available():
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
from ops.flex_attn_tilelang import flex_attn_tl

logger = logging.get_logger(__name__)


def _chunk_attn_pool_impl(
    mu_q: torch.Tensor,
    k_chunked: torch.Tensor,
    sm_scale: float,
    compact_lmk_k: bool,
    compressed_lmk_k: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inner math of ``chunk_attn_pool`` (compileable).

    Kept separate from the public wrapper so ``torch.compile`` only sees
    plain tensor ops -- no shape-check ``raise``, no ``Optional`` defaulting
    -- which is what the inductor backend handles best.

    Inputs are validated by the caller; this fn assumes::

        mu_q:      (B, N,   h_q, D), any dtype
        k_chunked: (B, N, S, h_kv, D), any dtype  -- NOT pre-expanded along
                   the head axis; we infer the GQA group size as
                   ``G = h_q // h_kv`` from the two head dims.
        sm_scale:  positive float

    GQA strategy (set by ``compressed_lmk_k``):
      * False (default, "MHA-style"): every q-head computes its OWN prior
        attention + its OWN aggregated KEY  ->  lmk_k shape (B, N, h_q, D),
        prior_b shape (B, N, h_q).  We ``repeat_interleave`` K to h_q heads
        internally (cheap on this small tensor) so each q-head dots its own
        copy of K.
      * True ("compressed"): mu_q is averaged within each q-group (mean over
        the G q-heads sharing one KV head) BEFORE the QK dot for ``lmk_k``,
        so lmk_k comes out at h_kv head granularity (B, N, h_kv, D).  The
        entropy bias ``prior_b`` is STILL computed in the per-q-head MHA
        layout (B, N, h_q) -- otherwise downstream ``topk_func`` couldn't
        accept it as a per-q-head additive bias.
    """
    in_dtype = k_chunked.dtype
    mu_f32 = mu_q.float()                                                  # (B, N, h_q, D)
    k_f32 = k_chunked.float()                                              # (B, N, S, h_kv, D)
    h_q = mu_f32.shape[2]
    h_kv = k_f32.shape[3]
    G = h_q // h_kv

    # --- Per-q-head prior attention (always computed; used for prior_b
    # always, and for non-compressed lmk_k).  We expand K along the head
    # axis from h_kv to h_q so each q-head dots its own copy.  When
    # ``h_q == h_kv`` (no GQA) this is a no-op view-friendly path.
    if G != 1:
        k_q = k_f32.repeat_interleave(G, dim=3)                            # (B, N, S, h_q, D)
    else:
        k_q = k_f32
    # logits[b, n, j, h] = sm_scale * <mu_q[b, n, h, :], K[b, n, j, h, :]>
    logits = torch.einsum("bnhd,bnshd->bnsh", mu_f32, k_q) * sm_scale       # (B, N, S, h_q)
    # Mask out the LAST K token of each chunk so the prior attention does
    # not select the chunk's own boundary token (which already serves as
    # the lmk anchor and would otherwise leak next-token info).  Use
    # ``masked_fill`` (out-of-place) instead of in-place index assignment
    # so this stays autograd- and ``torch.compile``-friendly.
    S_chunk = logits.shape[2]
    last_mask = torch.zeros(S_chunk, dtype=torch.bool, device=logits.device)
    last_mask[-1] = True
    logits = logits.masked_fill(last_mask.view(1, 1, S_chunk, 1), float('-inf'))

    p = F.softmax(logits, dim=2)
    lmk_k = None
    if not compact_lmk_k:                                      # (B, N, S, H)
        lmk_k = torch.einsum("bnsh,bnshd->bnhd", p, k_f32).to(in_dtype)                     # (B, N, H, D)

    # --- lmk_k (chunk-level surrogate KEY) ---
    lmk_k = None
    if not compact_lmk_k:
        if not compressed_lmk_k:
            # MHA-style: per-q-head k' = sum_j p_j * K_j (with the expanded
            # ``k_q`` from above).  Output: (B, N, h_q, D).
            lmk_k = torch.einsum("bnsh,bnshd->bnhd", p, k_q).to(in_dtype)
        else:
            # Compressed (GQA): average mu_q across each q-group and dot
            # with the original h_kv-head K (no expansion needed -- the
            # caller passes K with its native h_kv head count).  This gives
            # one lmk_k per KV head -- output shape (B, N, h_kv, D).
            B, N, _, _, D = k_f32.shape
            mu_g = mu_f32.view(B, N, h_kv, G, D).mean(dim=3)                # (B, N, h_kv, D)
            logits_kv = torch.einsum("bnhd,bnshd->bnsh", mu_g, k_f32) * sm_scale  # (B, N, S, h_kv)
            logits_kv = logits_kv.masked_fill(
                last_mask.view(1, 1, S_chunk, 1), float('-inf')
            )
            p_kv = F.softmax(logits_kv, dim=2)                              # (B, N, S, h_kv)
            lmk_k = torch.einsum("bnsh,bnshd->bnhd", p_kv, k_f32).to(in_dtype)  # (B, N, h_kv, D)

    # --- Entropy bias (always per-q-head, from the MHA-style ``p``) ---
    # When some logit positions are -inf (e.g. masked-out chunk-last token),
    # ``log_softmax`` puts -inf there and softmax puts 0 there, so the
    # product ``p * log_p`` evaluates to ``0 * (-inf) = NaN`` even though
    # the mathematical convention is ``0 * log 0 = 0``.  We compute the
    # entropy as ``-sum p * (log_p where finite else 0)`` to silence the
    # NaN without changing the value on valid positions.
    log_p = F.log_softmax(logits, dim=2)                                   # (B, N, S, h_q)
    log_p_safe = torch.where(
        torch.isfinite(log_p), log_p, log_p.new_zeros(())
    )
    lmk_b = -(p * log_p_safe).sum(dim=2)                                   # (B, N, h_q)

    return lmk_k, lmk_b


# Compile the inner math into one fused kernel.  Falls back to eager if the
# environment doesn't support compile (older torch, exotic backends, etc.).
# The wrapper below picks compiled vs eager at call time.
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
    compact_lmk_k: Optional[bool] = False,
    compressed_lmk_k: Optional[bool] = False,
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
        compact_lmk_k: if True, skip the GEMM that produces ``lmk_k``
                   (caller will substitute its own).  Returns lmk_k=None.
        compressed_lmk_k: if True, average mu_q within each q-group before
                   the QK dot for ``lmk_k`` -> output at h_kv granularity
                   (B, num_chunks, h_kv, d).  ``prior_b`` stays per-q-head
                   in either mode.  Mutually exclusive with compact_lmk_k.

    Returns:
        lmk_k: shape depends on ``compressed_lmk_k``:
                 - False: (B, num_chunks, h_q, d), in K's dtype.
                 - True:  (B, num_chunks, h_kv, d), in K's dtype.
                 None when ``compact_lmk_k`` is True.
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

    if compact_lmk_k and compressed_lmk_k:
        raise ValueError(
            "compact_lmk_k and compressed_lmk_k are mutually exclusive"
        )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    return chunk_attn_pool_tilelang(mu_q, k_chunked, sm_scale, compact_lmk_k=compact_lmk_k)


def create_chunk_dropout_mask(
    B: int,
    L: int,
    chunk_size: int,
    hsa_sliding_window: int,
    p: float,
    device: torch.device
) -> torch.Tensor:
    num_chunks = L // chunk_size

    # ----------------------------------------------------------------
    # Step 1: generate random scores; set invisible positions to -inf
    # ----------------------------------------------------------------
    rand = torch.rand(B, L, num_chunks, device=device)

    positions   = torch.arange(L, device=device)
    chunk_ids   = torch.arange(num_chunks, device=device)
    chunk_offset = (positions - hsa_sliding_window + 1) // chunk_size                       # (L,)
    available   = chunk_ids[None, :] < chunk_offset[:, None]   # (L, num_chunks)

    rand = rand.masked_fill(~available.unsqueeze(0), float('-inf'))

    # ----------------------------------------------------------------
    # Step 2: sort in descending order
    # ----------------------------------------------------------------
    indices = rand.argsort(dim=-1, descending=True)            # (B, L, num_chunks)

    # ----------------------------------------------------------------
    # Step 3: sample cnt from a binomial distribution
    # cnt ~ Binomial(num_available, p), then clamp(min=1)
    # if no visible chunks, keep cnt as 0
    # ----------------------------------------------------------------
    num_available = available.sum(dim=-1).float()              # (L,)

    # torch.binomial requires count and prob to have the same shape
    prob = torch.full_like(num_available, p)                   # (L,)
    cnt_base = torch.binomial(num_available, prob).long()      # (L,)  ~ Binomial(n_avail, p)

    # mask at least 1 (only for positions with available chunks)
    cnt = torch.where(
        num_available.long() > 0,
        cnt_base.clamp(min=1),
        cnt_base                    # keep 0 when no visible chunks
    )                                                          # (L,)

    cnt = cnt.unsqueeze(0).expand(B, -1)                       # (B, L)

    # ----------------------------------------------------------------
    # Step 4: rank_mask: top cnt ranks are True
    # ----------------------------------------------------------------
    rank     = torch.arange(num_chunks, device=device)         # (num_chunks,)
    rank_mask = rank[None, None, :] < cnt.unsqueeze(-1)        # (B, L, num_chunks)

    # ----------------------------------------------------------------
    # Step 5: scatter back to the chunk dimension
    # ----------------------------------------------------------------
    dropout_mask = torch.zeros(B, L, num_chunks, dtype=torch.bool, device=device)
    dropout_mask.scatter_(dim=2, index=indices, src=rank_mask)

    return dropout_mask.to(torch.int32)

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
        self.nope_retrieval = getattr(config, 'nope_retrieval', False)
        self.shared_lmk_q_norm = getattr(config, "shared_lmk_q_norm", False)

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

        if self.enable_lmk_q_proj:
            if self.shared_lmk_q_norm:
                assert not self.layerwise_qk_norm, (
                    "shared_lmk_q_norm is incompatible with layerwise_qk_norm: "
                    "q_norm has shape (d_model,), but lmk_q_norm needs (head_dim,)."
                )
                self.lmk_q_norm = self.q_norm
            elif not self.layerwise_lmkq_norm:
                self.lmk_q_norm = norm_cls(self.head_dim)

        self.hsa_visible_window = getattr(config, "hsa_visible_window", -1)
        if getattr(config, 'full_upper_hsa', False) and self.layer_idx >= config.num_hidden_layers // 2:
            self.hsa_visible_window = -1
        if self.hsa_visible_window != -1:
            self.topk = min(self.hsa_visible_window // self.chunk_size, self.topk)
            self.hsa_visible_window += config.sliding_window
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window
        self.hsa_sliding_window = getattr(config, "hsa_sliding_window", self.sliding_window)
        self.intra_chunk_rope = getattr(config, "intra_chunk_rope", False)
        self.enable_hsa_swa = getattr(config, "enable_hsa_swa", True)
        self.compact_lmk_k = getattr(config, "compact_lmk_k", False)
        self.compressed_lmk_k = getattr(config, "compressed_lmk_k", False)
        assert not (self.compact_lmk_k and self.compressed_lmk_k), 'compact_lmk_k and compressed_lmk_k cannot be True at the same time'
        
        self.is_causal = True
        
        hsa_mode = config.hsa_mode

        from ops.hsa_fwd_bwd_head import HSA_block_M_head as HSA
        from ops.topk_head_softmax import online_softmax_topk_head as topk_func
        self.topk_func = topk_func
        self.hsa_func = HSA
        self.hsa_intra_chunk_rope = hsa_intra_chunk_rope
        self.hsa_mode = hsa_mode

        self.enable_prior_query = getattr(config, "enable_prior_query", True)

        self.enable_softmax1 = config.enable_softmax1
        self.hsa_dropout_prob = getattr(config, "hsa_dropout_prob", 0.0)
        self.hsa_disturb_prob = getattr(config, "hsa_disturb_prob", 0.0)
        self.enable_gumbel_noise = getattr(config, "enable_gumbel_noise", False)
        

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
                # HERE, APPLY ROPE to lmk_q
                # lmk_q_norm: (B, L, r_head_num, d)
                if not self.nope_retrieval:
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
            if self.apply_hsa_rope and self.intra_chunk_rope:
                raise NotImplementedError("HSA intra-chunk RoPE does not support KV cache yet.")
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

        # apply rope for hsa part
        if self.enable_hsa_swa:
            swa_o, lse_sum = flex_attn_tl(
                hsa_q_norm_rope.transpose(1, 2), # (B, H, L, d)
                hsa_k_norm_rope.transpose(1, 2), 
                hsa_v.transpose(1, 2), 
                window_size=self.hsa_sliding_window,
                chunk_size=self.chunk_size,
                training=(past_key_values is None),
                mask_lmk=True,
                expand_to_chunk=True
            )
            # lse_sum = lse_sum.transpose(1, 2)  # (B, L, h_q)
            lse_sum = lse_sum.contiguous()  # (B, L, hG)
            lse_sum = lse_sum.to(hidden_states.dtype)  # (B, L, h_kv)

        full_seq_len = hsa_k_for_cache.shape[1]
        if full_seq_len >= self.chunk_size:
            if self.apply_hsa_rope and not self.nope_retrieval:
                lmk_k_source = hsa_k_norm_rope
            else:
                lmk_k_source = hsa_k_norm_nope

            # ----------------------------------------------------------
            # Build the chunk-level landmark key ``lmk_k``.  Two modes:
            #   (a) default: take the LAST token of each chunk.
            #   (b) prior_chunk_pool: replace it with the Taylor-expansion
            #       surrogate ``k' = sum_j p_j K_j`` plus the entropy bias
            #       ``b' = H(p)``.  ``b'`` is gathered along ``indices``
            #       AFTER topk and added to ``scores`` below.  Here the
            #       prior query ``mu_q`` is the lmk_q at the chunk's last
            #       token (parameter-free; trained indirectly through the
            #       upstream lmk_q_proj).
            # ----------------------------------------------------------
            prior_b = None  # (B, num_chunks, h_q) or None
            if self.enable_prior_query:
                # Reshape K stream into per-chunk blocks.  Discard any tail
                # tokens that don't fill a full chunk (matches the old
                # behavior of ``[chunk_size-1::chunk_size]`` which also
                # only counts a chunk once its last token has arrived).
                full_chunks = full_seq_len // self.chunk_size
                h_q_lmk = lmk_q_norm.shape[2]

                # ----------------------------------------------------------
                # Decode / inference caching for prior_query:
                # chunk_attn_pool results are immutable per chunk once the
                # chunk is complete, so we cache (lmk_k, prior_b) and only
                # recompute when NEW chunk boundaries are crossed.
                # In compact_lmk_k mode, chunk_attn_pool only computes the
                # entropy bias; lmk_k reuses the chunk-boundary K token.
                # Covers both single-token decode (L=1) and chunk-prefill
                # continuation (L > 1, but L < full_seq_len).
                # ----------------------------------------------------------
                is_continuation = (use_cache and past_key_values is not None and L < full_seq_len)

                if is_continuation:
                    # Initialize per-layer cache dict on the cache object
                    if not hasattr(past_key_values, '_hsa_prior_cache'):
                        past_key_values._hsa_prior_cache = {}
                    cache_key = self.layer_idx
                    cached = past_key_values._hsa_prior_cache.get(cache_key, None)

                    # Determine how many NEW complete chunks appeared since last call.
                    prev_seq_len = full_seq_len - L
                    prev_full_chunks = prev_seq_len // self.chunk_size
                    new_chunks = full_chunks - prev_full_chunks

                    if new_chunks > 0:
                        # Extract K for all newly completed chunks from the KV cache.
                        new_k = lmk_k_source[:, prev_full_chunks * self.chunk_size : full_chunks * self.chunk_size, :, :]
                        new_k = rearrange(new_k, "b (n s) h d -> b n s h d", s=self.chunk_size)
                        # (B, new_chunks, S, h_kv, D)

                        # mu_q: the last token of each new chunk. Their absolute
                        # positions are chunk_size*(prev_full_chunks+k)-1 for
                        # k=1..new_chunks. Local indices in lmk_q_norm (which
                        # has L tokens starting at prev_seq_len):
                        boundary_local = [
                            self.chunk_size * (prev_full_chunks + k) - 1 - prev_seq_len
                            for k in range(1, new_chunks + 1)
                        ]
                        new_mu_q = lmk_q_norm[:, boundary_local, :, :]  # (B, new_chunks, h_q_lmk, D)

                        # GQA expansion
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
                            compact_lmk_k=self.compact_lmk_k,
                        )
                        if self.compact_lmk_k:
                            new_lmk_k = lmk_k_source[
                                :, self.chunk_size - 1::self.chunk_size, :, :
                            ][:, prev_full_chunks:full_chunks]

                        if cached is not None:
                            lmk_k = torch.cat([cached['lmk_k'], new_lmk_k], dim=1)
                            prior_b = torch.cat([cached['prior_b'], new_prior_b], dim=1)
                        else:
                            lmk_k = new_lmk_k
                            prior_b = new_prior_b

                        # Update cache
                        past_key_values._hsa_prior_cache[cache_key] = {
                            'lmk_k': lmk_k,
                            'prior_b': prior_b,
                        }
                    else:
                        # No new chunk boundary -- reuse cached results
                        if cached is not None:
                            lmk_k = cached['lmk_k']
                            prior_b = cached['prior_b']
                        else:
                            # Before any chunk completes (first chunk_size-1 tokens)
                            prior_b = None
                            lmk_k = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, :, :]
                else:
                    # ----------------------------------------------------------
                    # Prefill path (training or first forward with cache)
                    # ----------------------------------------------------------
                    k_chunked = lmk_k_source[:, : full_chunks * self.chunk_size]
                    k_chunked = rearrange(
                        k_chunked, "b (n s) h d -> b n s h d", s=self.chunk_size
                    )                                                              # (B, N, S, h_kv, d)

                    mu_q = lmk_q_norm[:, self.chunk_size - 1::self.chunk_size, :, :][:, :full_chunks]  # (B, N, h_q_lmk, d)
                    # Guard: if Q is shorter than chunk_size (e.g. during chunk-prefill
                    # with a short query), mu_q has 0 chunks → skip weighted lmk and
                    # fall back to last-token-as-lmk for all available chunks.
                    if mu_q.shape[1] == 0:
                        prior_b = None
                        lmk_k = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, :, :]
                    else:
                        # Expand K from h_kv to h_q_lmk along the head axis so
                        # each q-head gets its OWN prior attention over the chunk's
                        # K (no group-mean shortcut -- mean would corrupt b').  The
                        # repeat is along the KV-head axis, matching standard GQA:
                        #   K[:, :, :, kv, :] is reused by all q-heads in that group.
                        if h_q_lmk != self.h_kv:
                            assert h_q_lmk % self.h_kv == 0, (
                                f"lmk_q head count ({h_q_lmk}) must be a multiple of "
                                f"h_kv ({self.h_kv}) for GQA-style chunk pool"
                            )
                            g = h_q_lmk // self.h_kv
                            k_chunked = k_chunked.repeat_interleave(g, dim=3)         # (B, N, S, h_q_lmk, d)
                        pooled_lmk_k, prior_b = chunk_attn_pool(
                            mu_q,
                            k_chunked,
                            compact_lmk_k=self.compact_lmk_k,
                        )         # k': (B, N, h_q_lmk, d), b': (B, N, h_q_lmk)
                        if self.compact_lmk_k:
                            lmk_k = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, :, :][:, :full_chunks]
                        else:
                            lmk_k = pooled_lmk_k

                    # Cache results for subsequent decode steps
                    if use_cache and past_key_values is not None and prior_b is not None:
                        if not hasattr(past_key_values, '_hsa_prior_cache'):
                            past_key_values._hsa_prior_cache = {}
                        past_key_values._hsa_prior_cache[self.layer_idx] = {
                            'lmk_k': lmk_k,
                            'prior_b': prior_b,
                        }
            else:
                lmk_k: Any = lmk_k_source[:, self.chunk_size - 1::self.chunk_size, : ,:]  # (B, L // S, hsa_kv, d)

            hsa_visible_window = self.hsa_visible_window if self.training else -1
            B, S,  H, D = lmk_k.shape
            lmk_k = lmk_k.reshape(B, S, H, D)
            # q_offset = int(cache_position[0].item()) if (use_cache and cache_position is not None) else 0
            q_offset = full_seq_len - L if (use_cache and past_key_values is not None) else 0

            drop_mask = None
            if self.training and self.hsa_dropout_prob > 0.0:
                drop_mask = create_chunk_dropout_mask(
                    B, L, self.chunk_size, self.hsa_sliding_window, self.hsa_dropout_prob, device=hidden_states.device
                )
                if self.hsa_disturb_prob > 0.0:
                    gate = (torch.rand(B, L, 1, device=lmk_q_norm.device) < self.hsa_disturb_prob).to(torch.int32)
                    drop_mask = drop_mask * gate

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
                drop_mask=drop_mask,
                q_offset=q_offset,
                use_gumbel=self.enable_gumbel_noise and self.training,
                G=None if self.compact_lmk_k else lmk_k.shape[2] // self.h_kv,
                bias=prior_b
            )
            # indices: [N, L, h_kv, K]
            # scores: [N, L, h_q, K]


            if self.enable_prior_query and prior_b is not None:
                # Gather b' along the chunk-index axis.  Shapes:
                #   prior_b : (B, N, h_q)         -- per-q-head entropy bias
                #   indices : (B, L, h_kv, K)     -- topk kernel returns at
                #                                    KV-head granularity
                #   scores  : (B, L, h_q, K)
                #
                # Step 1: GQA broadcast indices on the head axis  h_kv -> h_q
                #         so it lines up with prior_b's h_q dim.  (q-heads in
                #         the same group share their KV head's chunk
                #         selection, so we just repeat each KV-head row
                #         G_kernel times.)
                # Step 2: gather along the chunk axis straight in the native
                #         (B, L, h_q, K) layout -- no permutes, no contig
                #         copies.
                #     src   = prior_b.unsqueeze(-1).expand(B, N, h_q, K)   # view
                #     idx   = indices_hq.long()                             # (B, L, h_q, K)
                #     gather along dim=1:
                #         out[b, l, h, k] = src[b, idx[b,l,h,k], h, k]
                #                         = prior_b[b, idx[b,l,h,k], h]
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

            if self.enable_hsa_swa:
                if not self.enable_softmax1:
                    cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1)], dim=-1)  # (B, L, h_kv, K + 1)
                    swa_weight_idx = -1
                else:
                    # softmax off by one
                    cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1), torch.zeros(B, L, scores.shape[2], 1, device=hidden_states.device)], dim=-1)
                    swa_weight_idx = -2
                chunk_weights = F.softmax(cat_scores, dim=-1).to(hidden_states.dtype) # (B, L, h_kv, K)
            else:
                valid_row = torch.isfinite(scores).any(dim=-1, keepdim=True)  # (B, L, h, 1)
                safe_scores = scores.masked_fill(~valid_row, 0.0)
                chunk_weights = F.softmax(safe_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)  # (B, L, h_kv, K)
                chunk_weights = chunk_weights * valid_row.to(chunk_weights.dtype)

            if self.apply_hsa_rope:
                hsa_q_norm = hsa_q_norm_rope
                hsa_k_norm = hsa_k_norm_rope
            else:
                hsa_q_norm = hsa_q_norm_nope
                hsa_k_norm = hsa_k_norm_nope

            hsa_o = self.hsa_func(hsa_q_norm, hsa_k_norm, hsa_v, weights=chunk_weights, indices=indices, block_size=self.chunk_size, mask_last_token=True, is_training=self.training)
            if self.enable_hsa_swa:
                swa_o_weight = chunk_weights[:, :, :, swa_weight_idx]  # (B, L, h_kv)

                swa_o_weight_expanded = swa_o_weight
                o_lower = torch.addcmul(hsa_o, swa_o, swa_o_weight_expanded.unsqueeze(-1))
            else:
                o_lower = hsa_o
        else:
            if self.enable_hsa_swa and self.enable_softmax1:
                # Match the no-visible-chunk full path: softmax([lse_swa, 0]).
                swa_o_weight = torch.sigmoid(lse_sum).to(swa_o.dtype)
                o_lower = swa_o * swa_o_weight.unsqueeze(-1)
            else:
                o_lower = swa_o
        
        if self.hsa_denom > 1:
            o = torch.cat([o_upper, o_lower], dim=2)
        else:
            o = o_lower
        o = rearrange(o, 'B L h d->B L (h d)')
        
        return self.o_proj(o), None

if is_liger_kernel_available():
    apply_rotary_pos_emb = liger_rotary_pos_emb
    logger.info_rank0("Apply liger kernel to LHSA.")
