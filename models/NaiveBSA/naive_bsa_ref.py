"""Naive Block Sparse Attention (BSA) — pure PyTorch reference.

Implements the paper's naive BSA formula:

    s_{i,j}   = q_i^T k_j / sqrt(d),  j <= i
    l(i)      = floor((i - W + 1) / S) * S
    Z_{i,swa} = sum_{j = l(i)}^{i} exp(s_{i,j})
    Z_{i,c}   = sum_{j in chunk c} exp(s_{i,j})
    I_i       = top-K remote chunks by Z_{i,c}
    w_{i,j}   = exp(s_{i,j}) / (sum_{c in I_i} Z_{i,c} + Z_{i,swa})
                if c(j) in I_i or l(i) <= j <= i, else 0
    o_i       = sum_j w_{i,j} v_j

Only remote chunks fully before the local window participate in top-K.
Uses plain torch ops only (O(B * h_q * L^2) memory). Intended for
small-L ablations and equivalence checks against the kernel path.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def _expand_kv_to_q_heads(x: Tensor, h_q: int) -> Tensor:
    """GQA broadcast: (B, L, h_kv, d) -> (B, L, h_q, d)."""
    h_kv = x.shape[2]
    if h_kv == h_q:
        return x
    if h_q % h_kv != 0:
        raise ValueError(
            f"h_q ({h_q}) must be a multiple of h_kv ({h_kv}) for GQA"
        )
    return x.repeat_interleave(h_q // h_kv, dim=2)


def _pad_to_chunks(x: Tensor, chunk_size: int, fill: float) -> Tuple[Tensor, int, int]:
    """Right-pad last dim to a multiple of ``chunk_size``."""
    L = x.shape[-1]
    n_chunks = (L + chunk_size - 1) // chunk_size
    padded_L = n_chunks * chunk_size
    if padded_L == L:
        return x, n_chunks, padded_L
    return F.pad(x, (0, padded_L - L), value=fill), n_chunks, padded_L


def naive_bsa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    chunk_size: int,
    window_size: int,
    topk: int,
    sm_scale: Optional[float] = None,
    mask_last_token: bool = False,
    return_aux: bool = False,
) -> Tensor | Tuple[Tensor, dict]:
    """Naive Block Sparse Attention forward.

    Args:
        q: (B, L, h_q, d)
        k: (B, L, h_kv, d)  — h_kv divides h_q (GQA OK)
        v: (B, L, h_kv, d)
        chunk_size: S
        window_size: W; ``l(i) = floor((i - W + 1) / S) * S``
        topk: K remote chunks per query (group-shared under GQA)
        sm_scale: defaults to ``1 / sqrt(d)``
        mask_last_token: mask landmark keys at chunk ends ``(j+1) % S == 0``
        return_aux: also return debug intermediates

    Returns:
        o: (B, L, h_q, d) in q's dtype
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "expected q/k/v shaped (B, L, h, d); got "
            f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(v.shape)}"
        )
    B, L, h_q, d = q.shape
    Bk, Lk, h_kv, dk = k.shape
    Bv, Lv, h_kv_v, dv = v.shape
    if (Bk, Lk, dk) != (B, L, d) or (Bv, Lv, h_kv_v, dv) != (B, L, h_kv, d):
        raise ValueError(
            f"shape mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}, "
            f"v={tuple(v.shape)}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if topk < 0:
        raise ValueError(f"topk must be >= 0, got {topk}")

    S = chunk_size
    W = window_size
    K = topk
    out_dtype = q.dtype
    device = q.device

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    q_f = q.float()
    k_f = _expand_kv_to_q_heads(k, h_q).float()
    v_f = _expand_kv_to_q_heads(v, h_q).float()

    s = torch.einsum("bihd,bjhd->bhij", q_f, k_f) * sm_scale

    pos = torch.arange(L, device=device)
    causal = pos[None, :] > pos[:, None]
    s = s.masked_fill(causal.view(1, 1, L, L), float("-inf"))

    if mask_last_token:
        is_lmk_key = ((pos + 1) % S == 0)
        s = s.masked_fill(is_lmk_key.view(1, 1, 1, L), float("-inf"))

    s_pad, N_chunk, _ = _pad_to_chunks(s, S, fill=float("-inf"))
    s_chunked = s_pad.view(B, h_q, L, N_chunk, S)
    log_Z_chunk = torch.logsumexp(s_chunked, dim=-1)

    l_left = torch.div(pos - W + 1, S, rounding_mode="floor") * S
    l_left = l_left.clamp_min(0)
    chunk_starts = torch.arange(N_chunk, device=device) * S
    is_remote = (chunk_starts[None, :] + S) <= l_left[:, None]
    log_Z_chunk = log_Z_chunk.masked_fill(
        ~is_remote.view(1, 1, L, N_chunk), float("-inf")
    )

    # Group-shared top-K after softmax-normalizing chunk mass vs SWA mass.
    G = h_q // h_kv
    j_idx_pre = pos.view(1, L)
    local_mask = (j_idx_pre >= l_left.view(L, 1)) & (j_idx_pre <= pos.view(L, 1))
    s_local = s.masked_fill(~local_mask.view(1, 1, L, L), float("-inf"))
    lse_swa = torch.logsumexp(s_local, dim=-1)

    lse_hils = torch.logsumexp(log_Z_chunk, dim=-1)
    lse_total = torch.logaddexp(lse_swa, lse_hils)
    log_probs = log_Z_chunk - lse_total.unsqueeze(-1)

    if G == 1:
        score_pool = log_probs
    else:
        score_pool = log_probs.view(B, h_kv, G, L, N_chunk).max(dim=2).values

    K_eff = min(K, N_chunk)
    if K_eff > 0:
        _, chunk_idx_kv = torch.topk(score_pool, K_eff, dim=-1)
        chunk_idx = (
            chunk_idx_kv.repeat_interleave(G, dim=1) if G > 1 else chunk_idx_kv
        )
    else:
        chunk_idx = torch.empty(B, h_q, L, 0, dtype=torch.long, device=device)

    chunk_in_topk = torch.zeros(B, h_q, L, N_chunk, dtype=torch.bool, device=device)
    if K_eff > 0:
        valid_row = torch.isfinite(
            log_Z_chunk.gather(-1, chunk_idx[..., :1])
        ).expand_as(chunk_idx)
        safe_idx = chunk_idx.masked_fill(~valid_row, 0)
        chunk_in_topk.scatter_(-1, safe_idx, valid_row)
    token_in_topk = chunk_in_topk.repeat_interleave(S, dim=-1)[..., :L]

    kept = token_in_topk | local_mask.view(1, 1, L, L)
    s_kept = s.masked_fill(~kept, float("-inf"))

    all_neg_inf = torch.isneginf(s_kept).all(dim=-1, keepdim=True)
    w = F.softmax(s_kept, dim=-1)
    w = torch.where(all_neg_inf, torch.zeros_like(w), w)

    o = torch.einsum("bhij,bjhd->bihd", w, v_f).to(out_dtype)

    if not return_aux:
        return o

    return o, {
        "log_Z_chunk": log_Z_chunk,
        "chunk_indices": chunk_idx,
        "log_Z_swa": lse_swa,
        "log_probs": log_probs,
        "score_pool": score_pool,
        "l_left": l_left,
        "kept_mask": kept,
    }
