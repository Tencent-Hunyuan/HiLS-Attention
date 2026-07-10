"""TileLang kernels for exact per-chunk ``log Z`` in naive BSA.

Computes ``log Z_{i,c} = log sum_{j in chunk c} exp(scale * <q_i, k_j>)``
without materializing a dense ``(B, H_q, L, L)`` score matrix.

Masks (match ``naive_bsa_kernel.exact_chunk_log_z``):
* causal:      ``j_global > i_global`` -> -inf
* remote-only: ``i_global < (c+1)*S + W - 1`` -> -inf
* OOB:         ``i_global >= L`` or ``j_global >= L`` -> -inf
"""
from __future__ import annotations

from typing import Optional

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def exact_chunk_log_z_fwd(
    batch, heads, head_kv, seq_len, dim,
    chunk_size, window_size,
    block_M=64, sm_scale=None,
    num_threads=128,
    mask_lmk=False,
):
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    sm_scale_f = float(sm_scale)

    G = heads // head_kv
    n_chunk = (seq_len + chunk_size - 1) // chunk_size
    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, head_kv, dim]
    log_z_shape = [batch, seq_len, heads, n_chunk]

    @T.prim_func
    def fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        LogZ: T.Tensor(log_z_shape, accum_dtype),  # type: ignore
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_M), heads, batch,
            threads=num_threads,
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([chunk_size, dim], dtype)
            acc_s = T.alloc_fragment([block_M, chunk_size], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            log_z_val = T.alloc_fragment([block_M], accum_dtype)
            log_z_smem = T.alloc_shared([block_M], accum_dtype)

            tile_offset = bx * block_M

            for i, j in T.Parallel(block_M, dim):
                tq = tile_offset + i
                if tq < seq_len:
                    Q_shared[i, j] = Q[bz, tq, by, j]
                else:
                    Q_shared[i, j] = T.Cast(dtype, 0)

            c_remote_end_raw = T.floordiv(
                tile_offset + block_M - window_size, chunk_size,
            )
            c_remote_end = T.max(T.min(c_remote_end_raw, n_chunk), 0)

            for c in T.Pipelined(0, c_remote_end, num_stages=2):
                for i, j in T.Parallel(chunk_size, dim):
                    ts = c * chunk_size + i
                    if ts < seq_len:
                        K_shared[i, j] = K[bz, ts, by // G, j]
                    else:
                        K_shared[i, j] = T.Cast(dtype, 0)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, chunk_size):
                    i_global = tile_offset + i
                    j_global = c * chunk_size + j
                    # mask_lmk is compile-time; do not re-bind ``masked``.
                    if mask_lmk:
                        masked = (
                            (j_global > i_global)
                            | (i_global < (c + 1) * chunk_size + window_size - 1)
                            | (i_global >= seq_len)
                            | (j_global >= seq_len)
                            | (T.floormod(j_global + 1, chunk_size) == 0)
                        )
                    else:
                        masked = (
                            (j_global > i_global)
                            | (i_global < (c + 1) * chunk_size + window_size - 1)
                            | (i_global >= seq_len)
                            | (j_global >= seq_len)
                        )
                    acc_s[i, j] = T.if_then_else(
                        masked,
                        -T.infinity(accum_dtype),
                        acc_s[i, j] * sm_scale_f,
                    )

                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i, j in T.Parallel(block_M, chunk_size):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp(acc_s[i, j] - scores_max[i]),
                        0.0,
                    )
                T.fill(scores_sum, 0.0)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    log_z_val[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.log(scores_sum[i]) + scores_max[i],
                        -T.infinity(accum_dtype),
                    )

                T.copy(log_z_val, log_z_smem)
                for i in T.Parallel(block_M):
                    tq = tile_offset + i
                    if tq < seq_len:
                        LogZ[bz, tq, by, c] = log_z_smem[i]

    return fwd


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def exact_chunk_log_z_bwd_dq(
    batch, heads, head_kv, seq_len, dim,
    chunk_size, window_size,
    block_M=64, sm_scale=None,
    num_threads=128,
):
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    sm_scale_f = float(sm_scale)

    G = heads // head_kv
    n_chunk = (seq_len + chunk_size - 1) // chunk_size
    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, head_kv, dim]
    log_z_shape = [batch, seq_len, heads, n_chunk]
    dq_shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def bwd_dq(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        LogZ: T.Tensor(log_z_shape, accum_dtype),  # type: ignore
        dLogZ: T.Tensor(log_z_shape, accum_dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, dtype),  # type: ignore
    ):
        with T.Kernel(
            heads, T.ceildiv(seq_len, block_M), batch,
            threads=num_threads,
        ) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([chunk_size, dim], dtype)

            acc_s = T.alloc_fragment([block_M, chunk_size], accum_dtype)
            ds = T.alloc_fragment([block_M, chunk_size], accum_dtype)
            ds_shared = T.alloc_shared([block_M, chunk_size], dtype)

            log_z_local = T.alloc_fragment([block_M], accum_dtype)
            dlog_z_local = T.alloc_fragment([block_M], accum_dtype)

            dq = T.alloc_fragment([block_M, dim], accum_dtype)
            dq_shared = T.alloc_shared([block_M, dim], dtype)

            tile_offset = by * block_M

            for i, j in T.Parallel(block_M, dim):
                tq = tile_offset + i
                if tq < seq_len:
                    Q_shared[i, j] = Q[bz, tq, bx, j]
                else:
                    Q_shared[i, j] = T.Cast(dtype, 0)

            T.clear(dq)

            c_remote_end_raw = T.floordiv(
                tile_offset + block_M - window_size, chunk_size,
            )
            c_remote_end = T.max(T.min(c_remote_end_raw, n_chunk), 0)

            for c in T.Pipelined(0, c_remote_end, num_stages=2):
                for i, j in T.Parallel(chunk_size, dim):
                    ts = c * chunk_size + i
                    if ts < seq_len:
                        K_shared[i, j] = K[bz, ts, bx // G, j]
                    else:
                        K_shared[i, j] = T.Cast(dtype, 0)
                for i in T.Parallel(block_M):
                    tq = tile_offset + i
                    if tq < seq_len:
                        log_z_local[i] = LogZ[bz, tq, bx, c]
                        dlog_z_local[i] = dLogZ[bz, tq, bx, c]
                    else:
                        log_z_local[i] = -T.infinity(accum_dtype)
                        dlog_z_local[i] = 0.0

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, chunk_size):
                    i_global = tile_offset + i
                    j_global = c * chunk_size + j
                    masked = (
                        (j_global > i_global)
                        | (i_global < (c + 1) * chunk_size + window_size - 1)
                        | (i_global >= seq_len)
                        | (j_global >= seq_len)
                    )
                    acc_s[i, j] = T.if_then_else(
                        masked,
                        -T.infinity(accum_dtype),
                        acc_s[i, j] * sm_scale_f,
                    )

                for i, j in T.Parallel(block_M, chunk_size):
                    ds[i, j] = T.if_then_else(
                        log_z_local[i] > -1e30,
                        dlog_z_local[i]
                        * T.exp(acc_s[i, j] - log_z_local[i])
                        * sm_scale_f,
                        0.0,
                    )

                T.copy(ds, ds_shared)
                T.gemm(ds_shared, K_shared, dq, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dq, dq_shared)
            for i, j in T.Parallel(block_M, dim):
                tq = tile_offset + i
                if tq < seq_len:
                    dQ[bz, tq, bx, j] = dq_shared[i, j]

    return bwd_dq


@tilelang.jit(
    out_idx=[4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def exact_chunk_log_z_bwd_dk(
    batch, heads, head_kv, seq_len, dim,
    chunk_size, window_size,
    block_N=64, sm_scale=None,
    num_threads=128,
):
    """Produces ``dK_per_hq`` of shape ``(B, L, H_q, D)``; caller reduces over GQA."""
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5
    sm_scale_f = float(sm_scale)

    G = heads // head_kv
    n_chunk = (seq_len + chunk_size - 1) // chunk_size
    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim]
    k_shape = [batch, seq_len, head_kv, dim]
    log_z_shape = [batch, seq_len, heads, n_chunk]
    dk_per_hq_shape = [batch, seq_len, heads, dim]

    @T.prim_func
    def bwd_dk(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        LogZ: T.Tensor(log_z_shape, accum_dtype),  # type: ignore
        dLogZ: T.Tensor(log_z_shape, accum_dtype),  # type: ignore
        dK_per_hq: T.Tensor(dk_per_hq_shape, dtype),  # type: ignore
    ):
        with T.Kernel(
            heads, n_chunk, batch,
            threads=num_threads,
        ) as (bx, by, bz):
            K_shared = T.alloc_shared([chunk_size, dim], dtype)
            Q_shared = T.alloc_shared([block_N, dim], dtype)

            sT = T.alloc_fragment([chunk_size, block_N], accum_dtype)
            dsT = T.alloc_fragment([chunk_size, block_N], accum_dtype)
            dsT_shared = T.alloc_shared([chunk_size, block_N], dtype)

            log_z_local = T.alloc_fragment([block_N], accum_dtype)
            dlog_z_local = T.alloc_fragment([block_N], accum_dtype)

            dk = T.alloc_fragment([chunk_size, dim], accum_dtype)
            dk_shared = T.alloc_shared([chunk_size, dim], dtype)

            c = by

            for i, j in T.Parallel(chunk_size, dim):
                ts = c * chunk_size + i
                if ts < seq_len:
                    K_shared[i, j] = K[bz, ts, bx // G, j]
                else:
                    K_shared[i, j] = T.Cast(dtype, 0)

            T.clear(dk)

            # Static 0-start pipelined loop; fold start into index offset.
            q_tile_st = T.floordiv(
                (c + 1) * chunk_size + window_size - 1, block_N,
            )
            q_tile_ed = T.ceildiv(seq_len, block_N)
            n_q_tiles = T.max(q_tile_ed - q_tile_st, 0)

            for t in T.Pipelined(0, n_q_tiles, num_stages=2):
                q_tile = q_tile_st + t
                for i, j in T.Parallel(block_N, dim):
                    tq = q_tile * block_N + i
                    if tq < seq_len:
                        Q_shared[i, j] = Q[bz, tq, bx, j]
                    else:
                        Q_shared[i, j] = T.Cast(dtype, 0)
                for i in T.Parallel(block_N):
                    tq = q_tile * block_N + i
                    if tq < seq_len:
                        log_z_local[i] = LogZ[bz, tq, bx, c]
                        dlog_z_local[i] = dLogZ[bz, tq, bx, c]
                    else:
                        log_z_local[i] = -T.infinity(accum_dtype)
                        dlog_z_local[i] = 0.0

                T.clear(sT)
                T.gemm(K_shared, Q_shared, sT, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullRow)

                for ik, jq in T.Parallel(chunk_size, block_N):
                    i_q = q_tile * block_N + jq
                    j_k = c * chunk_size + ik
                    masked = (
                        (j_k > i_q)
                        | (i_q < (c + 1) * chunk_size + window_size - 1)
                        | (i_q >= seq_len)
                        | (j_k >= seq_len)
                    )
                    sT[ik, jq] = T.if_then_else(
                        masked,
                        -T.infinity(accum_dtype),
                        sT[ik, jq] * sm_scale_f,
                    )

                for ik, jq in T.Parallel(chunk_size, block_N):
                    dsT[ik, jq] = T.if_then_else(
                        log_z_local[jq] > -1e30,
                        dlog_z_local[jq]
                        * T.exp(sT[ik, jq] - log_z_local[jq])
                        * sm_scale_f,
                        0.0,
                    )

                T.copy(dsT, dsT_shared)
                T.gemm(dsT_shared, Q_shared, dk, policy=T.GemmWarpPolicy.FullRow)

            T.copy(dk, dk_shared)
            for i, j in T.Parallel(chunk_size, dim):
                ts = c * chunk_size + i
                if ts < seq_len:
                    dK_per_hq[bz, ts, bx, j] = dk_shared[i, j]

    return bwd_dk


class _ExactChunkLogZ(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        chunk_size: int,
        window_size: int,
        sm_scale: float,
        mask_lmk: bool = False,
    ) -> torch.Tensor:
        assert q.dim() == 4 and k.dim() == 4
        B, L, H_q, D = q.shape
        Bk, Lk, H_kv, Dk = k.shape
        assert (Bk, Lk, Dk) == (B, L, D), (
            f"q/k batch/seq/dim mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
        assert H_q % H_kv == 0, (
            f"H_q ({H_q}) must be a multiple of H_kv ({H_kv}) for GQA"
        )

        q_c = q.contiguous()
        k_c = k.contiguous()

        N_chunk = (L + chunk_size - 1) // chunk_size
        log_Z = torch.full(
            (B, L, H_q, N_chunk),
            float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        fwd_kernel = exact_chunk_log_z_fwd(
            batch=B, heads=H_q, head_kv=H_kv, seq_len=L, dim=D,
            chunk_size=chunk_size, window_size=window_size,
            sm_scale=sm_scale, mask_lmk=bool(mask_lmk),
        )
        fwd_kernel(q_c, k_c, log_Z)

        ctx.save_for_backward(q_c, k_c, log_Z)
        ctx.chunk_size = chunk_size
        ctx.window_size = window_size
        ctx.sm_scale = float(sm_scale)
        ctx.H_kv = H_kv
        ctx.mask_lmk = bool(mask_lmk)
        return log_Z

    @staticmethod
    def backward(ctx, d_log_Z: torch.Tensor):
        q, k, log_Z = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        window_size = ctx.window_size
        sm_scale = ctx.sm_scale
        H_kv = ctx.H_kv
        B, L, H_q, D = q.shape

        if getattr(ctx, "mask_lmk", False):
            raise NotImplementedError(
                "exact_chunk_log_z_tl(mask_lmk=True) is forward/eval-only; "
                "backward is not implemented for the landmark-masked path."
            )

        d_log_Z = d_log_Z.contiguous()

        bwd_dq_kernel = exact_chunk_log_z_bwd_dq(
            batch=B, heads=H_q, head_kv=H_kv, seq_len=L, dim=D,
            chunk_size=chunk_size, window_size=window_size,
            sm_scale=sm_scale,
        )
        dq = bwd_dq_kernel(q, k, log_Z, d_log_Z)

        bwd_dk_kernel = exact_chunk_log_z_bwd_dk(
            batch=B, heads=H_q, head_kv=H_kv, seq_len=L, dim=D,
            chunk_size=chunk_size, window_size=window_size,
            sm_scale=sm_scale,
        )
        dk_per_hq = bwd_dk_kernel(q, k, log_Z, d_log_Z)

        G = H_q // H_kv
        if G > 1:
            dk = dk_per_hq.view(B, L, H_kv, G, D).float().sum(dim=3).to(k.dtype)
        else:
            dk = dk_per_hq

        return dq, dk, None, None, None, None


def exact_chunk_log_z_tl(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    chunk_size: int,
    window_size: int,
    sm_scale: Optional[float] = None,
    mask_lmk: bool = False,
) -> torch.Tensor:
    """Kernel-backed ``log Z_{i,c}``. Returns (B, L, H_q, N_chunk) fp32.

    ``mask_lmk=True`` is forward/eval-only (backward raises).
    """
    if sm_scale is None:
        sm_scale = (1.0 / q.shape[-1]) ** 0.5
    return _ExactChunkLogZ.apply(
        q, k, int(chunk_size), int(window_size), float(sm_scale), bool(mask_lmk),
    )
