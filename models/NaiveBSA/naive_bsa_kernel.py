"""Kernel-backed naive Block Sparse Attention (BSA).

Equivalent to ``naive_bsa_ref.naive_bsa_attention`` up to bf16 rounding.
Uses three TileLang kernels:

    * ``ops.flex_attn_tilelang.flex_attn_tl``           — SWA + ``lse_swa``
    * ``ops.hils_fwd_bwd_head.HiLS_block_M_head``     — sparse chunk attention
    * ``ops.exact_chunk_log_z_tilelang.exact_chunk_log_z_tl`` — exact ``log Z``

Factorized form: with ``D_i = sum_{c in I_i} Z_{i,c} + Z_{i,swa}``::

    o_i = sum_{c in I_i} alpha_{i,c} * o_intra_c + alpha_{i,swa} * o_swa
    alpha = softmax([log Z_{i,c_1}, ..., log Z_{i,c_K}, log Z_{i,swa}])

Assumptions: GQA-friendly (``h_q % h_kv == 0``), CUDA bf16, ``chunk_size >= 32``,
``topk`` preferably a power of 2. For CPU / small-L checks use ``naive_bsa_ref``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


def _chunk_log_z_step(
    q_slice: torch.Tensor,
    k_chunk: torch.Tensor,
    causal_mask: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    s_c = torch.einsum("bihd,bjhd->bihj", q_slice, k_chunk) * sm_scale
    s_c = s_c.masked_fill(causal_mask, float("-inf"))
    return torch.logsumexp(s_c, dim=-1)


def exact_chunk_log_z(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    chunk_size: int,
    window_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Pure-PyTorch ``log Z_{i,c}`` (CPU / testing reference; O(L^2) memory).

    log_Z[b, i, h, c] = log sum_{j in chunk c} exp(sm_scale * <q_i, k_j>)
    with causal and remote-only masks. Returns (B, L, h_q, N_chunk) fp32.
    """
    if q.dim() != 4 or k.dim() != 4:
        raise ValueError(
            "expected q, k in (B, L, H, D); got "
            f"{tuple(q.shape)}, {tuple(k.shape)}"
        )
    B, L, h_q, D = q.shape
    Bk, Lk, h_kv, Dk = k.shape
    if (Bk, Lk, Dk) != (B, L, D):
        raise ValueError(
            f"q/k batch/seq/dim mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    if h_q % h_kv != 0:
        raise ValueError(
            f"exact_chunk_log_z: h_q ({h_q}) must be a multiple of "
            f"h_kv ({h_kv}) for GQA broadcast"
        )
    device = q.device
    S = chunk_size
    W = window_size
    G = h_q // h_kv

    pad = (-L) % S
    N_chunk = (L + pad) // S

    q_f = q.float()
    k_f = k.repeat_interleave(G, dim=2).float() if G > 1 else k.float()
    use_ckpt = (q.requires_grad or k.requires_grad) and torch.is_grad_enabled()

    log_z_chunks: list[torch.Tensor] = []
    neg_inf_full: Optional[torch.Tensor] = None

    for c in range(N_chunk):
        c_start = c * S
        c_end_global = c_start + S
        c_end = min(c_end_global, L)
        S_c = c_end - c_start

        i_min = c_end_global + W - 1
        if i_min >= L:
            if neg_inf_full is None:
                neg_inf_full = torch.full(
                    (B, L, h_q), float("-inf"),
                    device=device, dtype=torch.float32,
                )
            log_z_chunks.append(neg_inf_full)
            continue

        q_slice = q_f[:, i_min:, :, :]
        k_chunk = k_f[:, c_start:c_end, :, :]
        L_q = L - i_min

        i_global = i_min + torch.arange(L_q, device=device)
        j_global = c_start + torch.arange(S_c, device=device)
        causal_mask = (j_global[None, :] > i_global[:, None]).view(1, L_q, 1, S_c)

        if use_ckpt:
            log_z_slice = _ckpt(
                _chunk_log_z_step, q_slice, k_chunk, causal_mask, sm_scale,
                use_reentrant=False,
            )
        else:
            log_z_slice = _chunk_log_z_step(q_slice, k_chunk, causal_mask, sm_scale)

        log_z_c = F.pad(log_z_slice, (0, 0, i_min, 0), value=float("-inf"))
        log_z_chunks.append(log_z_c)

    return torch.stack(log_z_chunks, dim=-1).contiguous()


def naive_bsa_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    chunk_size: int,
    window_size: int,
    topk: int,
    sm_scale: Optional[float] = None,
    mask_last_token: bool = False,
    return_aux: bool = False,
) -> torch.Tensor:
    """Kernel-backed naive BSA forward.

    Args:
        q: (B, L, h_q, D)
        k: (B, L, h_kv, D)
        v: (B, L, h_kv, D)
        chunk_size, window_size, topk: paper parameters S, W, K
        sm_scale: defaults to ``1 / sqrt(D)``
        mask_last_token: mask landmark keys at chunk ends
        return_aux: also return intermediates for debugging

    Returns:
        o: (B, L, h_q, D) in q's dtype
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("q, k, v must be 4D: (B, L, H, D)")
    B, L, h_q, D = q.shape
    if k.shape[:2] != (B, L) or v.shape[:2] != (B, L) or k.shape[-1] != D:
        raise ValueError(
            f"q/k/v batch/seq/dim mismatch: q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    h_kv = k.shape[2]
    if v.shape[2] != h_kv:
        raise ValueError(
            f"k and v must share head count; got k.h_kv={h_kv}, "
            f"v.h_kv={v.shape[2]}"
        )
    if h_q % h_kv != 0:
        raise ValueError(
            f"h_q ({h_q}) must be a multiple of h_kv ({h_kv}) for GQA"
        )
    G = h_q // h_kv
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    from ops.flex_attn_tilelang import flex_attn_tl
    from ops.hils_fwd_bwd_head import HiLS_block_M_head
    from ops.exact_chunk_log_z_tilelang import exact_chunk_log_z_tl

    S = chunk_size
    W = window_size

    swa_o, lse_swa = flex_attn_tl(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        window_size=W,
        chunk_size=S,
        training=True,
        mask_lmk=mask_last_token,
        expand_to_chunk=True,
        sm_scale=sm_scale,
    )

    K = topk
    if L < S or K <= 0:
        out = swa_o.to(q.dtype)
        if return_aux:
            return out, {
                "swa_o": swa_o, "lse_swa": lse_swa,
                "log_Z": None, "indices": None, "chunk_weights": None,
            }
        return out

    log_Z = exact_chunk_log_z_tl(
        q, k, chunk_size=S, window_size=W, sm_scale=sm_scale,
        mask_lmk=mask_last_token,
    )
    N_chunk = log_Z.shape[-1]
    K_eff = min(K, N_chunk)

    # Group-shared top-K on softmax-normalized log-probs (max-pool over G).
    lse_hils = torch.logsumexp(log_Z, dim=-1)
    lse_total = torch.logaddexp(lse_swa.to(log_Z.dtype), lse_hils)
    log_probs = log_Z - lse_total.unsqueeze(-1)
    if G == 1:
        score_pool = log_probs
    else:
        score_pool = log_probs.view(B, L, h_kv, G, N_chunk).max(dim=3).values
    _, indices_kv = torch.topk(score_pool, K_eff, dim=-1)

    if G == 1:
        indices_hq = indices_kv
    else:
        indices_hq = indices_kv.repeat_interleave(G, dim=2)
    scores_hq = log_Z.gather(dim=-1, index=indices_hq.long())

    cat_scores = torch.cat(
        [scores_hq, lse_swa.to(scores_hq.dtype).unsqueeze(-1)], dim=-1
    )
    chunk_weights_f32 = F.softmax(cat_scores, dim=-1)
    sparse_w = chunk_weights_f32[..., :K_eff].to(q.dtype).contiguous()
    swa_w = chunk_weights_f32[..., -1:].to(q.dtype)

    # Pad to fixed kernel topk so selected_blocks is compile-time constant.
    if K_eff < K:
        pad = K - K_eff
        sparse_w = F.pad(sparse_w, (0, pad), value=0.0).contiguous()
        pad_idx = torch.zeros(
            *indices_kv.shape[:-1], pad,
            dtype=indices_kv.dtype, device=indices_kv.device,
        )
        indices_kv = torch.cat([indices_kv, pad_idx], dim=-1).contiguous()
    indices_kv_i32 = indices_kv.to(torch.int32).contiguous()

    is_training = q.requires_grad or k.requires_grad or v.requires_grad
    hils_o = HiLS_block_M_head(
        q.contiguous(), k.contiguous(), v.contiguous(),
        sparse_w, indices_kv_i32,
        block_size=S,
        sm_scale=sm_scale,
        block_M=None,
        mask_last_token=mask_last_token,
        is_training=is_training,
    )

    o = torch.addcmul(hils_o.to(q.dtype), swa_o.to(q.dtype), swa_w)

    if return_aux:
        return o, {
            "swa_o": swa_o,
            "lse_swa": lse_swa,
            "log_Z": log_Z,
            "log_probs": log_probs,
            "score_pool": score_pool,
            "indices_kv": indices_kv,
            "chunk_weights": chunk_weights_f32,
        }
    return o
