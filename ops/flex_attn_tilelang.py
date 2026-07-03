"""
Tilelang implementation of flex_attn used in HSA layer.

Replaces utils/flex_attn.py's flex_attention call. Two boolean flags control
the mask semantics directly inside the kernel (no Python-side mask tensor):

- mask_lmk:        if True, mask out landmark token columns where (kv_idx + 1) % chunk_size == 0
- expand_to_chunk: if True, sliding window left edge is expanded to the start of the chunk
                   that contains (q_idx - window_size + 1)

Layout convention used by the public API:
    q: (B, H_q,  Lq, D_qk)
    k: (B, H_kv, Lk, D_qk)
    v: (B, H_kv, Lk, D_v)
    -> o:   (B, Lq, H_q, D_v)
       lse: (B, Lq, H_q)        (natural log)

Internally the kernels use (B, L, H, D) layout (matching example_gqa_bwd.py),
so outputs are returned directly in the internal sequence-major layout.

Forward kernel fuses training (Lq == Lk) and inference (Lq <= Lk) by accepting
dynamic q_len / kv_len via T.dynamic when use_cache=True.

Backward kernel is atomic-add only (no split variant), training only
(use_cache must be False during backward, which matches inference where no
gradient is required).
"""

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


@tilelang.jit(
    out_idx=[3, 4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_fwd(
    batch,
    heads,
    q_len,
    kv_len,
    dim_qk,
    dim_v,
    window_size,
    chunk_size,
    mask_lmk,
    expand_to_chunk,
    block_M,
    block_N,
    groups=1,
    use_cache=False,
    sm_scale=None,
    use_alibi=False,
):
    """Forward kernel.

    When use_cache=False, q_len/kv_len are compile-time constants (must equal).
    When use_cache=True, q_len/kv_len are runtime dynamic (q_len <= kv_len).

    Tail handling: every global load / store is guarded by `tq < q_len_var`
    or `ts < kv_len_var`, so non-multiple-of-block seq lengths are handled
    directly by the kernel without requiring the caller to pad. OOB slots
    in Q/K/V are filled with zeros in shared memory and then masked to
    -inf before the softmax, matching the tail strategy used by
    `ops/topk_head_softmax.py`.

    `sm_scale` is the natural-log-domain softmax scale applied to the raw
    Q @ K^T scores (i.e. `score = sm_scale * <q, k>`). Default (None) is
    `1 / sqrt(dim_qk)`, matching the canonical attention scale. Callers can
    override it to decouple the attention scale from the last-dim size (for
    example, when q/k are PoPE-expanded from D to D+R and the logically
    correct scale is still `1/sqrt(D)`).
    """
    _base = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale = _base * 1.44269504  # log2(e)
    head_kv = heads // groups
    dtype = "bfloat16"
    accum_dtype = "float"

    if use_cache:
        q_len_var = T.dynamic("q_len")
        kv_len_var = T.dynamic("kv_len")
    else:
        q_len_var = q_len
        kv_len_var = kv_len

    q_shape = [batch, q_len_var, heads, dim_qk]
    k_shape = [batch, kv_len_var, head_kv, dim_qk]
    v_shape = [batch, kv_len_var, head_kv, dim_v]
    o_shape = [batch, q_len_var, heads, dim_v]
    lse_shape = [batch, q_len_var, heads]

    @T.prim_func
    def fwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
        Lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        LseNope: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        M_h: T.Tensor([heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(q_len_var, block_M), heads, batch, threads=256) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)
            O_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_out_shared = T.alloc_shared([block_M], accum_dtype)
            lse_nope_out_shared = T.alloc_shared([block_M], accum_dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_raw = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim_v], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)
            scores_max_nope = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev_nope = T.alloc_fragment([block_M], accum_dtype)
            scores_scale_nope = T.alloc_fragment([block_M], accum_dtype)
            scores_sum_nope = T.alloc_fragment([block_M], accum_dtype)
            logsum_nope = T.alloc_fragment([block_M], accum_dtype)

            for i, j in T.Parallel(block_M, dim_qk):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Q_shared[i, j] = Q[bz, tq, by, j]
                else:
                    Q_shared[i, j] = T.Cast(dtype, 0)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.fill(logsum_nope, 0)
            T.fill(scores_max_nope, -T.infinity(accum_dtype))

            q_offset = kv_len_var - q_len_var

            q_block_start = bx * block_M + q_offset
            q_block_end = (bx + 1) * block_M + q_offset

            if expand_to_chunk:
                tmp_left = q_block_start - window_size + 1
                left_kv_raw = T.floordiv(tmp_left, chunk_size) * chunk_size
            else:
                left_kv_raw = q_block_start - window_size + 1
            left_kv = T.max(left_kv_raw, 0)
            right_kv = T.min(q_block_end, kv_len_var)

            loop_st = T.floordiv(left_kv, block_N)
            loop_ed = T.ceildiv(right_kv, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=1):
                for i, j in T.Parallel(block_N, dim_qk):
                    ts = k * block_N + i
                    if ts < kv_len_var:
                        K_shared[i, j] = K[bz, ts, by // groups, j]
                    else:
                        K_shared[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(block_N, dim_v):
                    ts = k * block_N + i
                    if ts < kv_len_var:
                        V_shared[i, j] = V[bz, ts, by // groups, j]
                    else:
                        V_shared[i, j] = T.Cast(dtype, 0)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if expand_to_chunk and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real_idx = bx * block_M + i + q_offset
                        kv_real_idx = k * block_N + j
                        chunk_start = T.floordiv(q_real_idx - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real_idx >= chunk_start)
                            & (kv_real_idx <= q_real_idx)
                            & (kv_real_idx < kv_len_var)
                            & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real_idx + 1, chunk_size) != 0),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                elif expand_to_chunk and (not mask_lmk):
                    for i, j in T.Parallel(block_M, block_N):
                        q_real_idx = bx * block_M + i + q_offset
                        kv_real_idx = k * block_N + j
                        chunk_start = T.floordiv(q_real_idx - window_size + 1, chunk_size) * chunk_size
                        acc_s[i, j] = T.if_then_else(
                            (kv_real_idx >= chunk_start)
                            & (kv_real_idx <= q_real_idx)
                            & (kv_real_idx < kv_len_var)
                            & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                elif (not expand_to_chunk) and mask_lmk:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real_idx = bx * block_M + i + q_offset
                        kv_real_idx = k * block_N + j
                        chunk_start = q_real_idx - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real_idx >= chunk_start)
                            & (kv_real_idx <= q_real_idx)
                            & (kv_real_idx < kv_len_var)
                            & ((bx * block_M + i) < q_len_var)
                            & (T.floormod(kv_real_idx + 1, chunk_size) != 0),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real_idx = bx * block_M + i + q_offset
                        kv_real_idx = k * block_N + j
                        chunk_start = q_real_idx - window_size + 1
                        acc_s[i, j] = T.if_then_else(
                            (kv_real_idx >= chunk_start)
                            & (kv_real_idx <= q_real_idx)
                            & (kv_real_idx < kv_len_var)
                            & ((bx * block_M + i) < q_len_var),
                            acc_s[i, j],
                            -T.infinity(accum_dtype),
                        )

                for i, j in T.Parallel(block_M, block_N):
                    acc_s_raw[i, j] = acc_s[i, j]

                if use_alibi:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real_idx = bx * block_M + i + q_offset
                        kv_real_idx = k * block_N + j
                        acc_s[i, j] += M_h[by] * T.Cast(accum_dtype, kv_real_idx - q_real_idx)


                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                for i in T.Parallel(block_M):
                    scores_scale[i] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale),
                        1.0,
                    )
                for i, j in T.Parallel(block_M, dim_v):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.if_then_else(
                        scores_max[i] > -1e30,
                        T.exp2(acc_s[i, j] * scale - scores_max[i] * scale),
                        0.0,
                    )
                T.copy(acc_s, acc_s_cast)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(scores_max_nope, scores_max_prev_nope)
                T.reduce_max(acc_s_raw, scores_max_nope, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max_nope[i] = T.max(scores_max_nope[i], scores_max_prev_nope[i])
                for i in T.Parallel(block_M):
                    scores_scale_nope[i] = T.if_then_else(
                        scores_max_nope[i] > -1e30,
                        T.exp2(scores_max_prev_nope[i] * scale - scores_max_nope[i] * scale),
                        1.0,
                    )
                for i, j in T.Parallel(block_M, block_N):
                    acc_s_raw[i, j] = T.if_then_else(
                        scores_max_nope[i] > -1e30,
                        T.exp2(acc_s_raw[i, j] * scale - scores_max_nope[i] * scale),
                        0.0,
                    )
                T.reduce_sum(acc_s_raw, scores_sum_nope, dim=1)
                for i in T.Parallel(block_M):
                    logsum_nope[i] = logsum_nope[i] * scores_scale_nope[i] + scores_sum_nope[i]

            for i, j in T.Parallel(block_M, dim_v):
                acc_o[i, j] = T.if_then_else(logsum[i] > 0, acc_o[i, j] / logsum[i], 0.0)
            for i in T.Parallel(block_M):
                logsum[i] = T.if_then_else(
                    logsum[i] > 0,
                    T.log2(logsum[i]) + scores_max[i] * scale,
                    -T.infinity(accum_dtype),
                )
            for i in T.Parallel(block_M):
                logsum_nope[i] = T.if_then_else(
                    logsum_nope[i] > 0,
                    T.log2(logsum_nope[i]) + scores_max_nope[i] * scale,
                    -T.infinity(accum_dtype),
                )
            T.copy(acc_o, O_shared)
            T.copy(logsum, lse_out_shared)
            T.copy(logsum_nope, lse_nope_out_shared)
            for i, j in T.Parallel(block_M, dim_v):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Output[bz, tq, by, j] = O_shared[i, j]
            for i in T.Parallel(block_M):
                tq = bx * block_M + i
                if tq < q_len_var:
                    Lse[bz, tq, by] = lse_out_shared[i]
                    LseNope[bz, tq, by] = lse_nope_out_shared[i]

    return fwd


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flex_attn_bwd_preprocess(batch, heads, seq_len, dim_v):
    """Compute Delta = sum(O * dO, dim=-1).

    Tail handling: per-element guarded load / store -- when seq_len is not a
    multiple of `blk`, OOB rows skip both the read (so we never read garbage
    bytes) and the write (so we never corrupt neighbouring tensors).
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_v]
    blk = 32

    @T.prim_func
    def prep(
        O: T.Tensor(shape, dtype),  # type: ignore
        dO: T.Tensor(shape, dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            delta_shared = T.alloc_shared([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim_v, blk)):
                for i, j in T.Parallel(blk, blk):
                    tq = by * blk + i
                    if tq < seq_len:
                        o[i, j] = O[bz, tq, bx, k * blk + j]
                        do[i, j] = dO[bz, tq, bx, k * blk + j]
                    else:
                        o[i, j] = T.Cast(dtype, 0)
                        do[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, delta_shared)
            for i in T.Parallel(blk):
                tq = by * blk + i
                if tq < seq_len:
                    Delta[bz, bx, tq] = delta_shared[i]

    return prep


def make_dq_layout(dQ):
    return T.Layout(
        dQ.shape,
        lambda b, l, h, d: [b, l // 8, h, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@tilelang.jit(
    out_idx=[1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flex_attn_bwd_postprocess(batch, heads, seq_len, dim_qk):
    """Cast accumulator-precision dQ to bf16 with guarded tail copy."""
    dtype = "bfloat16"
    accum_dtype = "float"
    shape = [batch, seq_len, heads, dim_qk]
    blk = 64

    @T.prim_func
    def post(
        dQ: T.Tensor(shape, accum_dtype),  # type: ignore
        dQ_out: T.Tensor(shape, dtype),  # type: ignore
    ):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            for i, d in T.Parallel(blk, dim_qk):
                tq = bx * blk + i
                if tq < seq_len:
                    dQ_out[bz, tq, by, d] = T.Cast(dtype, dQ[bz, tq, by, d])

    return post


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def flex_attn_bwd(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    window_size,
    chunk_size,
    mask_lmk,
    expand_to_chunk,
    block_M,
    block_N,
    threads=256,
    num_stages=2,
    groups=1,
    sm_scale=None,
    use_alibi=False,
):
    """Backward kernel.

    Tail handling: every global load / store is guarded by `tq < seq_len` or
    `ts < seq_len`, mirroring the strategy used by `ops/topk_head_softmax.py`.
    OOB rows in K / V / Q / dO / lse / Delta are filled with zero in shared
    memory; the same row/col bound terms in the mask zero out their
    contributions to qkT before any GEMM that produces dq / dk / dv.
    Atomic adds to dQ / dK / dV are guarded so we never write past the
    allocated buffer.

    `sm_scale` must match the forward kernel's `sm_scale` (default
    `1 / sqrt(dim_qk)` when None).
    """
    sm_scale = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale = sm_scale * 1.44269504
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    dtype = "bfloat16"
    accum_dtype = "float"

    @T.prim_func
    def bwd(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor([batch, seq_len, heads, dim_v], dtype),  # type: ignore
        lse: T.Tensor([batch, seq_len, heads], accum_dtype),  # type: ignore
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),  # type: ignore
        dQ: T.Tensor(q_shape, accum_dtype),  # type: ignore
        dK: T.Tensor(k_shape, accum_dtype),  # type: ignore
        dV: T.Tensor(v_shape, accum_dtype),  # type: ignore
        M_h: T.Tensor([heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)
            q = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)
            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
            dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta = T.alloc_shared([block_N], accum_dtype)
            do = T.alloc_shared([block_N, dim_v], dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dq = T.alloc_fragment([block_N, dim_qk], accum_dtype)

            for i, j in T.Parallel(block_M, dim_qk):
                ts = by * block_M + i
                if ts < seq_len:
                    K_shared[i, j] = K[bz, ts, bx // groups, j]
                else:
                    K_shared[i, j] = T.Cast(dtype, 0)
            for i, j in T.Parallel(block_M, dim_v):
                ts = by * block_M + i
                if ts < seq_len:
                    V_shared[i, j] = V[bz, ts, bx // groups, j]
                else:
                    V_shared[i, j] = T.Cast(dtype, 0)

            T.clear(dv)
            T.clear(dk)

            kv_lo = by * block_M
            kv_hi = (by + 1) * block_M
            q_lo = kv_lo
            q_hi = kv_hi + window_size + chunk_size
            if q_hi > seq_len:
                q_hi = seq_len

            loop_st = T.floordiv(q_lo, block_N)
            loop_ed = T.ceildiv(q_hi, block_N)

            for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):

                for i, j in T.Parallel(block_N, dim_qk):
                    tq = k * block_N + i
                    if tq < seq_len:
                        q[i, j] = Q[bz, tq, bx, j]
                    else:
                        q[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(block_N, dim_v):
                    tq = k * block_N + i
                    if tq < seq_len:
                        do[i, j] = dO[bz, tq, bx, j]
                    else:
                        do[i, j] = T.Cast(dtype, 0)
                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        lse_shared[i] = lse[bz, tq, bx]
                    else:
                        lse_shared[i] = T.Cast(accum_dtype, 0)
                for i in T.Parallel(block_N):
                    tq = k * block_N + i
                    if tq < seq_len:
                        delta[i] = Delta[bz, bx, tq]
                    else:
                        delta[i] = T.Cast(accum_dtype, 0)

                T.clear(qkT)
                T.gemm(K_shared, q, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                if use_alibi:
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k * block_N + j
                        qkT[i, j] += M_h[bx] * T.Cast(accum_dtype, kv_real - q_real)
                for i, j in T.Parallel(block_M, block_N):
                    qkT[i, j] = T.if_then_else(
                        lse_shared[j] > -1e30,
                        T.exp2(qkT[i, j] * scale - lse_shared[j]),
                        0.0,
                    )

                for i, j in T.Parallel(block_M, block_N):
                    kv_real = by * block_M + i
                    q_real = k * block_N + j
                    if expand_to_chunk:
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                    else:
                        chunk_start = q_real - window_size + 1
                    if mask_lmk:
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= chunk_start)
                            & (kv_real <= q_real)
                            & (kv_real < seq_len)
                            & (q_real < seq_len)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            qkT[i, j],
                            0.0,
                        )
                    else:
                        qkT[i, j] = T.if_then_else(
                            (kv_real >= chunk_start)
                            & (kv_real <= q_real)
                            & (kv_real < seq_len)
                            & (q_real < seq_len),
                            qkT[i, j],
                            0.0,
                        )

                T.clear(dsT)
                T.gemm(V_shared, do, dsT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(qkT, qkT_cast)
                T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                for i, j in T.Parallel(block_M, block_N):
                    dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale

                T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                T.copy(dsT_cast, dsT_shared)
                T.clear(dq)
                T.gemm(dsT_shared, K_shared, dq, transpose_A=True)

                for i, j in T.Parallel(block_N, dim_qk):
                    if (k * block_N + i) < seq_len:
                        T.atomic_add(dQ[bz, k * block_N + i, bx, j], dq[i, j])

            for i, d in T.Parallel(block_M, dim_v):
                if (by * block_M + i) < seq_len:
                    T.atomic_add(dV[bz, by * block_M + i, bx // groups, d], dv[i, d])
            for i, d in T.Parallel(block_M, dim_qk):
                if (by * block_M + i) < seq_len:
                    T.atomic_add(dK[bz, by * block_M + i, bx // groups, d], dk[i, d])

    return bwd


class _FlexAttnTL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_blhd, k_blhd, v_blhd, window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache, sm_scale=None, m_h=None):
        """Inputs are already in (B, L, H, D) layout (transposed by wrapper).

        The kernels themselves handle non-multiple-of-block seq lengths via
        per-element guarded load / store (see `flex_attn_fwd` / `flex_attn_bwd`),
        so no caller-side padding is needed.

        `m_h`: optional (H_q,) fp32 tensor of ALiBi slopes. When provided, each
        logit `qk[i,j]` is offset by `m_h[h] * (kv_idx - q_idx)` before softmax
        (bias is per-q-head, unscaled). When None, no bias is applied.

        Implementation note: kernels compute softmax as `exp2((acc_s) * scale)`
        where `scale = sm_scale_nat * log2(e)`. That means any value added to
        `acc_s` is multiplied by `sm_scale_nat` inside the exp. Standard ALiBi
        adds `bias = m_h * (kv - q)` AFTER the sm_scale multiplication, so we
        must pre-divide the slopes by `sm_scale_nat` on the host; the kernel
        then multiplies by `sm_scale_nat` inside the exp, recovering the
        unscaled bias. The backward path reconstructs P from the same saved
        lse / scale / pre-divided slopes, so fwd-bwd remain self-consistent.
        """
        BATCH, Q_LEN, H, D_QK = q_blhd.shape
        KV_LEN = k_blhd.shape[1]
        H_KV = k_blhd.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV

        block_M = 128
        block_N = 64

        use_alibi = m_h is not None
        if use_alibi:
            assert m_h.shape == (H,), f"m_h shape {tuple(m_h.shape)} != ({H},)"
            sm_scale_nat = (1.0 / D_QK) ** 0.5 if sm_scale is None else float(sm_scale)
            m_h_arg = (m_h.to(device=q_blhd.device, dtype=torch.float32)
                          / sm_scale_nat).contiguous()
        else:
            m_h_arg = torch.zeros(H, dtype=torch.float32, device=q_blhd.device)

        mod = flex_attn_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            window_size, chunk_size,
            mask_lmk, expand_to_chunk,
            block_M, block_N, groups,
            use_cache=use_cache,
            sm_scale=sm_scale,
            use_alibi=use_alibi,
        )
        o, lse, lse_nope = mod(q_blhd, k_blhd, v_blhd, m_h_arg)

        ctx.save_for_backward(q_blhd, k_blhd, v_blhd, o, lse, m_h_arg)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.sm_scale = sm_scale
        ctx.use_alibi = use_alibi
        return o, lse, lse_nope

    @staticmethod
    def backward(ctx, do, dlse, dlse_nope):
        del dlse_nope
        q, k, v, o, lse, m_h_arg = ctx.saved_tensors
        if ctx.use_cache:
            raise RuntimeError("flex_attn_tl backward is not supported in inference (use_cache=True) mode")

        BATCH, N_CTX, H, D_QK = q.shape
        H_KV = v.shape[-2]
        D_V = v.shape[-1]
        groups = H // H_KV

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        block_M = 128
        block_N = 32

        mod_prep = flex_attn_bwd_preprocess(BATCH, H, N_CTX, D_V)
        mod_post = flex_attn_bwd_postprocess(BATCH, H, N_CTX, D_QK)
        delta = mod_prep(o, do)

        if dlse is not None:
            ln2 = 0.6931471805599453
            dlse_bhl = dlse.permute(0, 2, 1).contiguous().to(delta.dtype)
            delta = delta - dlse_bhl / ln2

        kernel = flex_attn_bwd(
            BATCH, H, N_CTX, D_QK, D_V,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_M, block_N,
            threads=256, num_stages=2, groups=groups,
            sm_scale=ctx.sm_scale,
            use_alibi=ctx.use_alibi,
        )
        dq = torch.zeros([BATCH, N_CTX, H, D_QK], dtype=torch.float32, device=q.device)
        dk = torch.zeros([BATCH, N_CTX, H_KV, D_QK], dtype=torch.float32, device=q.device)
        dv = torch.zeros([BATCH, N_CTX, H_KV, D_V], dtype=torch.float32, device=q.device)
        kernel(q, k, v, do, lse, delta, dq, dk, dv, m_h_arg)
        dq = mod_post(dq)
        dk = dk.to(torch.bfloat16)
        dv = dv.to(torch.bfloat16)
        return dq, dk, dv, None, None, None, None, None, None, None


@tilelang.jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_bwd_dq(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    window_size,
    chunk_size,
    mask_lmk,
    expand_to_chunk,
    block_M,
    block_N,
    threads=128,
    num_stages=2,
    groups=1,
    sm_scale=None,
    use_alibi=False,
):
    """dQ-only backward kernel (atomic-free).

    Grid: (heads, ceildiv(seq_len, block_M), batch).
        - block_M: Q rows owned by this kernel block (the "output" tile).
        - block_N: KV columns iterated inside the kernel.

    For each Q tile we walk the KV columns that fall in
        [max(0, q_lo - W + 1 [or chunk_start of it]), q_hi)
    accumulate dq in a fragment, then write directly to bf16 dQ.

    Tail handling mirrors the atomic kernel: every load / store is guarded
    by `tq < seq_len` / `ts < seq_len`, OOB Q/K/V/dO entries are zero in
    shared memory, OOB rows / cols are zeroed in the qkT mask.

    `sm_scale` must match the forward kernel's `sm_scale` (default
    `1 / sqrt(dim_qk)` when None).
    """
    sm_scale = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale_log2 = sm_scale * 1.44269504  # log2(e)
    head_kv = heads // groups
    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    do_shape = [batch, seq_len, heads, dim_v]
    lse_shape = [batch, seq_len, heads]
    delta_shape = [batch, heads, seq_len]
    dq_shape = [batch, seq_len, heads, dim_qk]

    @T.prim_func
    def bwd_dq(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        dQ: T.Tensor(dq_shape, dtype),  # type: ignore
        M_h: T.Tensor([heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dO_shared = T.alloc_shared([block_M, dim_v], dtype)
            lse_shared = T.alloc_shared([block_M], accum_dtype)
            delta_shared = T.alloc_shared([block_M], accum_dtype)

            K_shared = T.alloc_shared([block_N, dim_qk], dtype)
            V_shared = T.alloc_shared([block_N, dim_v], dtype)

            qk = T.alloc_fragment([block_M, block_N], accum_dtype)
            p = T.alloc_fragment([block_M, block_N], accum_dtype)
            dp = T.alloc_fragment([block_M, block_N], accum_dtype)
            ds = T.alloc_fragment([block_M, block_N], accum_dtype)
            # ds is staged to shared (no intermediate bf16 fragment): the
            # `T.copy(ds, ds_shared)` performs the fp32->bf16 cast implicitly,
            # and using shared as GEMM-A keeps layout inference unambiguous
            # and matches the ldmatrix path used by mma. Pattern borrowed
            # from `hsa_fwd_bwd_single_tilelang.py::bwd_dq_dw`.
            ds_shared = T.alloc_shared([block_M, block_N], dtype)

            dq = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            # dq is staged to shared on the way out: shared store pipelines
            # better with vectorized global stores than fragment store.
            dq_shared = T.alloc_shared([block_M, dim_qk], dtype)

            # --- Load Q / dO / lse / Delta for this Q tile (guarded). ---
            for i, j in T.Parallel(block_M, dim_qk):
                tq = by * block_M + i
                if tq < seq_len:
                    Q_shared[i, j] = Q[bz, tq, bx, j]
                else:
                    Q_shared[i, j] = T.Cast(dtype, 0)
            for i, j in T.Parallel(block_M, dim_v):
                tq = by * block_M + i
                if tq < seq_len:
                    dO_shared[i, j] = dO[bz, tq, bx, j]
                else:
                    dO_shared[i, j] = T.Cast(dtype, 0)
            for i in T.Parallel(block_M):
                tq = by * block_M + i
                if tq < seq_len:
                    lse_shared[i] = lse[bz, tq, bx]
                    delta_shared[i] = Delta[bz, bx, tq]
                else:
                    lse_shared[i] = T.Cast(accum_dtype, 0)
                    delta_shared[i] = T.Cast(accum_dtype, 0)

            T.clear(dq)

            # --- KV iteration range: only blocks that can mask-intersect. ---
            q_lo = by * block_M
            q_hi = (by + 1) * block_M
            # Lower bound on kv that any q in this tile can attend to.
            if expand_to_chunk:
                kv_lo = T.floordiv(q_lo - window_size + 1, chunk_size) * chunk_size
            else:
                kv_lo = q_lo - window_size + 1
            if kv_lo < 0:
                kv_lo = 0
            # Upper bound: kv_real <= q_real <= q_hi - 1.
            kv_hi = q_hi
            if kv_hi > seq_len:
                kv_hi = seq_len

            loop_st = T.floordiv(kv_lo, block_N)
            loop_ed = T.ceildiv(kv_hi, block_N)

            for k_iter in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                # Guarded load K / V for this KV tile.
                for i, j in T.Parallel(block_N, dim_qk):
                    ts = k_iter * block_N + i
                    if ts < seq_len:
                        K_shared[i, j] = K[bz, ts, bx // groups, j]
                    else:
                        K_shared[i, j] = T.Cast(dtype, 0)
                for i, j in T.Parallel(block_N, dim_v):
                    ts = k_iter * block_N + i
                    if ts < seq_len:
                        V_shared[i, j] = V[bz, ts, bx // groups, j]
                    else:
                        V_shared[i, j] = T.Cast(dtype, 0)

                # qk = Q @ K^T  ->  P = exp2(qk * scale_log2 - lse)
                T.clear(qk)
                T.gemm(Q_shared, K_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # ALiBi bias: add slope_h * (kv_real - q_real) to qk before
                # exp2(qk*scale - lse) reconstruction (must mirror fwd).
                if use_alibi:
                    for i, j in T.Parallel(block_M, block_N):
                        q_real = by * block_M + i
                        kv_real = k_iter * block_N + j
                        qk[i, j] += M_h[bx] * T.Cast(accum_dtype, kv_real - q_real)
                for i, j in T.Parallel(block_M, block_N):
                    p[i, j] = T.exp2(qk[i, j] * scale_log2 - lse_shared[i])

                # Mask: window + (optional) lmk + causal + bounds.
                for i, j in T.Parallel(block_M, block_N):
                    q_real = by * block_M + i
                    kv_real = k_iter * block_N + j
                    if expand_to_chunk:
                        chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                    else:
                        chunk_start = q_real - window_size + 1
                    if mask_lmk:
                        p[i, j] = T.if_then_else(
                            (kv_real >= chunk_start)
                            & (kv_real <= q_real)
                            & (kv_real < seq_len)
                            & (q_real < seq_len)
                            & (T.floormod(kv_real + 1, chunk_size) != 0),
                            p[i, j],
                            0.0,
                        )
                    else:
                        p[i, j] = T.if_then_else(
                            (kv_real >= chunk_start)
                            & (kv_real <= q_real)
                            & (kv_real < seq_len)
                            & (q_real < seq_len),
                            p[i, j],
                            0.0,
                        )

                # dp = dO @ V^T
                T.clear(dp)
                T.gemm(dO_shared, V_shared, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # ds = P * (dp - delta) * sm_scale
                for i, j in T.Parallel(block_M, block_N):
                    ds[i, j] = p[i, j] * (dp[i, j] - delta_shared[i]) * sm_scale

                # dq += ds @ K   (route ds through shared directly, skipping
                # the intermediate bf16 fragment; the copy fp32_frag -> bf16_shared
                # casts implicitly. Mirrors `bwd_dq_dw` from
                # `hsa_fwd_bwd_single_tilelang.py`.)
                T.copy(ds, ds_shared)
                T.gemm(ds_shared, K_shared, dq, policy=T.GemmWarpPolicy.FullRow)

            # --- Stage dQ through shared, then guarded global store. ---
            T.copy(dq, dq_shared)
            for i, j in T.Parallel(block_M, dim_qk):
                tq = by * block_M + i
                if tq < seq_len:
                    dQ[bz, tq, bx, j] = dq_shared[i, j]

    return bwd_dq


@tilelang.jit(
    out_idx=[6, 7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def flex_attn_bwd_dkdv(
    batch,
    heads,
    seq_len,
    dim_qk,
    dim_v,
    window_size,
    chunk_size,
    mask_lmk,
    expand_to_chunk,
    block_M,
    block_N,
    threads=256,
    num_stages=2,
    groups=1,
    sm_scale=None,
    use_alibi=False,
):
    """dK/dV-only backward kernel (atomic-free).

    Grid: (head_kv, ceildiv(seq_len, block_M), batch).
        - block_M: KV rows owned by this kernel block (the "output" tile).
        - block_N: Q rows iterated inside the kernel.
        - For GQA the kernel walks all `groups` Q-heads sharing this KV head
          and accumulates their dk/dv contributions into the same fragment.

    Iteration range (q-direction):
        q_lo = by * block_M
        q_hi = (by+1)*block_M + W [+ C if expand_to_chunk]

    `sm_scale` must match the forward kernel's `sm_scale` (default
    `1 / sqrt(dim_qk)` when None).
    """
    sm_scale = (1.0 / dim_qk) ** 0.5 if sm_scale is None else float(sm_scale)
    scale_log2 = sm_scale * 1.44269504
    head_kv = heads // groups
    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, seq_len, heads, dim_qk]
    k_shape = [batch, seq_len, head_kv, dim_qk]
    v_shape = [batch, seq_len, head_kv, dim_v]
    do_shape = [batch, seq_len, heads, dim_v]
    lse_shape = [batch, seq_len, heads]
    delta_shape = [batch, heads, seq_len]
    dk_shape = [batch, seq_len, head_kv, dim_qk]
    dv_shape = [batch, seq_len, head_kv, dim_v]

    @T.prim_func
    def bwd_dkdv(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        K: T.Tensor(k_shape, dtype),  # type: ignore
        V: T.Tensor(v_shape, dtype),  # type: ignore
        dO: T.Tensor(do_shape, dtype),  # type: ignore
        lse: T.Tensor(lse_shape, accum_dtype),  # type: ignore
        Delta: T.Tensor(delta_shape, accum_dtype),  # type: ignore
        dK: T.Tensor(dk_shape, dtype),  # type: ignore
        dV: T.Tensor(dv_shape, dtype),  # type: ignore
        M_h: T.Tensor([heads], accum_dtype),  # type: ignore
    ):
        with T.Kernel(head_kv, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bx, by, bz):
            K_shared = T.alloc_shared([block_M, dim_qk], dtype)
            V_shared = T.alloc_shared([block_M, dim_v], dtype)

            Q_shared = T.alloc_shared([block_N, dim_qk], dtype)
            dO_shared = T.alloc_shared([block_N, dim_v], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_shared = T.alloc_shared([block_N], accum_dtype)

            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
            # Stage qkT and dsT to shared (fp32_frag -> bf16_shared) instead
            # of going through bf16 fragments. Both shared buffers are then
            # used as GEMM-A. Pattern from `bwd_dkdv` in
            # `hsa_fwd_bwd_single_tilelang.py`.
            qkT_shared = T.alloc_shared([block_M, block_N], dtype)
            dsT_shared = T.alloc_shared([block_M, block_N], dtype)

            dk = T.alloc_fragment([block_M, dim_qk], accum_dtype)
            dv = T.alloc_fragment([block_M, dim_v], accum_dtype)
            # Stage dK / dV through shared on the way out (vectorized stores).
            dk_shared = T.alloc_shared([block_M, dim_qk], dtype)
            dv_shared = T.alloc_shared([block_M, dim_v], dtype)

            # --- Load K / V for this KV tile (guarded). ---
            for i, j in T.Parallel(block_M, dim_qk):
                ts = by * block_M + i
                if ts < seq_len:
                    K_shared[i, j] = K[bz, ts, bx, j]
                else:
                    K_shared[i, j] = T.Cast(dtype, 0)
            for i, j in T.Parallel(block_M, dim_v):
                ts = by * block_M + i
                if ts < seq_len:
                    V_shared[i, j] = V[bz, ts, bx, j]
                else:
                    V_shared[i, j] = T.Cast(dtype, 0)

            T.clear(dk)
            T.clear(dv)

            # --- Q iteration range: same logic as the atomic kernel. ---
            kv_lo = by * block_M
            kv_hi = (by + 1) * block_M
            q_lo = kv_lo
            if expand_to_chunk:
                q_hi = kv_hi + window_size + chunk_size
            else:
                q_hi = kv_hi + window_size
            if q_hi > seq_len:
                q_hi = seq_len

            loop_st = T.floordiv(q_lo, block_N)
            loop_ed = T.ceildiv(q_hi, block_N)

            # --- For each Q-head sharing this KV head, accumulate dk/dv. ---
            for g in T.serial(groups):
                hq = bx * groups + g

                for k_iter in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    # Guarded load Q / dO / lse / Delta for this Q tile.
                    for i, j in T.Parallel(block_N, dim_qk):
                        tq = k_iter * block_N + i
                        if tq < seq_len:
                            Q_shared[i, j] = Q[bz, tq, hq, j]
                        else:
                            Q_shared[i, j] = T.Cast(dtype, 0)
                    for i, j in T.Parallel(block_N, dim_v):
                        tq = k_iter * block_N + i
                        if tq < seq_len:
                            dO_shared[i, j] = dO[bz, tq, hq, j]
                        else:
                            dO_shared[i, j] = T.Cast(dtype, 0)
                    for i in T.Parallel(block_N):
                        tq = k_iter * block_N + i
                        if tq < seq_len:
                            lse_shared[i] = lse[bz, tq, hq]
                            delta_shared[i] = Delta[bz, hq, tq]
                        else:
                            lse_shared[i] = T.Cast(accum_dtype, 0)
                            delta_shared[i] = T.Cast(accum_dtype, 0)

                    # qkT = K @ Q^T  ->  P^T = exp2(qkT * scale_log2 - lse[j])
                    T.clear(qkT)
                    T.gemm(K_shared, Q_shared, qkT, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    # ALiBi bias: add slope_hq * (kv_real - q_real) to qkT
                    # before exp2(qkT*scale - lse) reconstruction. Slopes are
                    # per-q-head; use hq (not bx which is the kv-head index).
                    if use_alibi:
                        for i, j in T.Parallel(block_M, block_N):
                            kv_real = by * block_M + i
                            q_real = k_iter * block_N + j
                            qkT[i, j] += M_h[hq] * T.Cast(accum_dtype, kv_real - q_real)
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * scale_log2 - lse_shared[j])

                    # Mask (same semantics as forward).
                    for i, j in T.Parallel(block_M, block_N):
                        kv_real = by * block_M + i
                        q_real = k_iter * block_N + j
                        if expand_to_chunk:
                            chunk_start = T.floordiv(q_real - window_size + 1, chunk_size) * chunk_size
                        else:
                            chunk_start = q_real - window_size + 1
                        if mask_lmk:
                            qkT[i, j] = T.if_then_else(
                                (kv_real >= chunk_start)
                                & (kv_real <= q_real)
                                & (kv_real < seq_len)
                                & (q_real < seq_len)
                                & (T.floormod(kv_real + 1, chunk_size) != 0),
                                qkT[i, j],
                                0.0,
                            )
                        else:
                            qkT[i, j] = T.if_then_else(
                                (kv_real >= chunk_start)
                                & (kv_real <= q_real)
                                & (kv_real < seq_len)
                                & (q_real < seq_len),
                                qkT[i, j],
                                0.0,
                            )

                    # dsT = V @ dO^T (i.e. (dP)^T before mask scaling)
                    T.clear(dsT)
                    T.gemm(V_shared, dO_shared, dsT, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    # dv += P^T @ dO  (route qkT through shared)
                    T.copy(qkT, qkT_shared)
                    T.gemm(qkT_shared, dO_shared, dv, policy=T.GemmWarpPolicy.FullRow)

                    # dsT = qkT * (dsT - delta[j]) * sm_scale
                    for i, j in T.Parallel(block_M, block_N):
                        dsT[i, j] = qkT[i, j] * (dsT[i, j] - delta_shared[j]) * sm_scale

                    # dk += dsT @ Q   (route dsT through shared)
                    T.copy(dsT, dsT_shared)
                    T.gemm(dsT_shared, Q_shared, dk, policy=T.GemmWarpPolicy.FullRow)

            # --- Stage dK / dV through shared, then guarded global store. ---
            T.copy(dk, dk_shared)
            T.copy(dv, dv_shared)
            for i, j in T.Parallel(block_M, dim_qk):
                ts = by * block_M + i
                if ts < seq_len:
                    dK[bz, ts, bx, j] = dk_shared[i, j]
            for i, j in T.Parallel(block_M, dim_v):
                ts = by * block_M + i
                if ts < seq_len:
                    dV[bz, ts, bx, j] = dv_shared[i, j]

    return bwd_dkdv


# ---------------------------------------------------------------------------
# Two-phase autograd Function (atomic-free backward).
# ---------------------------------------------------------------------------
class _FlexAttnTLTwoPhase(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_blhd, k_blhd, v_blhd, window_size, chunk_size, mask_lmk, expand_to_chunk, use_cache, sm_scale=None, m_h=None):
        """Same forward as `_FlexAttnTL`; only the backward path differs.

        See `_FlexAttnTL.forward` for the semantics of `m_h` and the scaling
        convention (slopes are pre-divided by `sm_scale_nat` so that the
        kernel's post-multiplication by `sm_scale_nat` inside the softmax
        exp2 recovers the unscaled ALiBi bias).
        """
        BATCH, Q_LEN, H, D_QK = q_blhd.shape
        KV_LEN = k_blhd.shape[1]
        H_KV = k_blhd.shape[2]
        D_V = v_blhd.shape[-1]
        groups = H // H_KV

        block_M = 128
        block_N = 64

        use_alibi = m_h is not None
        if use_alibi:
            assert m_h.shape == (H,), f"m_h shape {tuple(m_h.shape)} != ({H},)"
            sm_scale_nat = (1.0 / D_QK) ** 0.5 if sm_scale is None else float(sm_scale)
            m_h_arg = (m_h.to(device=q_blhd.device, dtype=torch.float32)
                          / sm_scale_nat).contiguous()
        else:
            m_h_arg = torch.zeros(H, dtype=torch.float32, device=q_blhd.device)

        mod = flex_attn_fwd(
            BATCH, H, Q_LEN, KV_LEN, D_QK, D_V,
            window_size, chunk_size,
            mask_lmk, expand_to_chunk,
            block_M, block_N, groups,
            use_cache=use_cache,
            sm_scale=sm_scale,
            use_alibi=use_alibi,
        )
        o, lse, lse_nope = mod(q_blhd, k_blhd, v_blhd, m_h_arg)

        # `lse_nope` is a non-differentiable side output (see `_FlexAttnTL`).
        ctx.save_for_backward(q_blhd, k_blhd, v_blhd, o, lse, m_h_arg)
        ctx.window_size = window_size
        ctx.chunk_size = chunk_size
        ctx.mask_lmk = mask_lmk
        ctx.expand_to_chunk = expand_to_chunk
        ctx.use_cache = use_cache
        ctx.sm_scale = sm_scale
        ctx.use_alibi = use_alibi
        return o, lse, lse_nope

    @staticmethod
    def backward(ctx, do, dlse, dlse_nope):
        del dlse_nope  # ignored; LseNope has no gradient path
        q, k, v, o, lse, m_h_arg = ctx.saved_tensors
        if ctx.use_cache:
            raise RuntimeError(
                "flex_attn_tl_two_phase backward is not supported in inference (use_cache=True) mode"
            )

        BATCH, N_CTX, H, D_QK = q.shape
        H_KV = v.shape[-2]
        D_V = v.shape[-1]
        groups = H // H_KV

        do = do.contiguous()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()

        # --- Preprocess: Delta = sum(O * dO, dim=-1). ---
        mod_prep = flex_attn_bwd_preprocess(BATCH, H, N_CTX, D_V)
        delta = mod_prep(o, do)

        # Fold dlse into Delta (see `_FlexAttnTL.backward` for the derivation).
        if dlse is not None:
            ln2 = 0.6931471805599453
            dlse_bhl = dlse.permute(0, 2, 1).contiguous().to(delta.dtype)
            delta = delta - dlse_bhl / ln2

        # --- Phase 1: dQ kernel. ---
        block_Mq = 128
        block_Nk = 128
        kernel_dq = flex_attn_bwd_dq(
            BATCH, H, N_CTX, D_QK, D_V,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_Mq, block_Nk,
            threads=256, num_stages=1, groups=groups,
            sm_scale=ctx.sm_scale,
            use_alibi=ctx.use_alibi,
        )
        dq = kernel_dq(q, k, v, do, lse, delta, m_h_arg)

        # --- Phase 2: dK / dV kernel. ---
        block_Mk = 128
        block_Nq = 64
        kernel_dkdv = flex_attn_bwd_dkdv(
            BATCH, H, N_CTX, D_QK, D_V,
            ctx.window_size, ctx.chunk_size,
            ctx.mask_lmk, ctx.expand_to_chunk,
            block_Mk, block_Nq,
            threads=256, num_stages=0, groups=groups,
            sm_scale=ctx.sm_scale,
            use_alibi=ctx.use_alibi,
        )
        dk, dv = kernel_dkdv(q, k, v, do, lse, delta, m_h_arg)

        return dq, dk, dv, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def flex_attn_tl(
    q,
    k,
    v,
    window_size: int,
    chunk_size: int,
    training: bool = True,
    mask_lmk: bool = True,
    expand_to_chunk: bool = True,
    sm_scale: float = None,
    m_h=None,
    return_nope_lse: bool = False,
):
    """Tilelang flex attention.

    Args:
        q: (B, H_q,  Lq, D_qk)
        k: (B, H_kv, Lk, D_qk)
        v: (B, H_kv, Lk, D_v)
        window_size:    sliding window size (in tokens)
        chunk_size:     chunk size for landmark / window-expansion semantics
        training:       True for training (Lq == Lk); False for inference (Lq <= Lk).
                        Backward only supported when training=True.
        mask_lmk:       if True, exclude landmark token columns where (kv_idx + 1) % chunk_size == 0
        expand_to_chunk:if True, sliding window left edge expands to start of containing chunk
        sm_scale:       optional softmax scale (natural-log domain), applied as
                        `score = sm_scale * <q, k>` before the softmax. When
                        None (default), uses `1 / sqrt(D_qk)` to match the
                        canonical attention scale. Callers can override this to
                        decouple the scale from the last-dim size (e.g. when
                        q/k are PoPE-expanded from D to D+R but the logically
                        correct scale is still `1 / sqrt(D)`).
        m_h:            optional (H_q,) fp32 tensor of ALiBi slopes. When
                        provided, each logit `qk[i,j]` is offset by
                        `m_h[h] * (kv_idx - q_idx)` before softmax (per-q-head
                        bias, NOT scaled by sm_scale, matching the standard
                        ALiBi definition). None disables ALiBi.
        return_nope_lse: if True, additionally return `lse_nope`, the LSE
                        computed WITHOUT the ALiBi bias (regardless of
                        whether `m_h` is provided). This is a non-
                        differentiable side output intended for external
                        topk selection. When `m_h is None`, `lse_nope`
                        numerically equals `lse`.

    Returns:
        if return_nope_lse is False (default):
            o:   (B, Lq, H_q, D_v)
            lse: (B, Lq, H_q)         natural log, includes ALiBi bias if any
        if return_nope_lse is True:
            o:        (B, Lq, H_q, D_v)
            lse:      (B, Lq, H_q)    natural log, with ALiBi bias if any
            lse_nope: (B, Lq, H_q)    natural log, WITHOUT ALiBi bias
    """
    # public inputs use (B, H, L, D) layout; internal kernel uses (B, L, H, D).
    # Use transpose + reshape to a flat 1D view and back, which forces PyTorch
    # to normalize strides to packed (B, L, H, D) layout. A plain .contiguous()
    # can be a no-op when a singleton dim hides non-contiguous strides (e.g.
    # Lq=1 during single-token decode).
    def _to_blhd(x):
        y = x.transpose(1, 2).contiguous()
        B_, L_, H_, D_ = y.shape
        return y.reshape(-1).reshape(B_, L_, H_, D_)

    q_blhd = _to_blhd(q)
    k_blhd = _to_blhd(k)
    v_blhd = _to_blhd(v)

    use_cache = not training
    o_blhd, lse_log2, lse_nope_log2 = _FlexAttnTL.apply(
        q_blhd, k_blhd, v_blhd,
        window_size, chunk_size,
        mask_lmk, expand_to_chunk, use_cache,
        sm_scale,
        m_h,
    )
    # convert log2-domain lse to natural log
    ln2 = 0.6931471805599453  # ln(2)
    lse = lse_log2 * ln2
    if return_nope_lse:
        lse_nope = lse_nope_log2 * ln2
        return o_blhd, lse, lse_nope
    return o_blhd, lse


def flex_attn_tl_two_phase(
    q,
    k,
    v,
    window_size: int,
    chunk_size: int,
    training: bool = True,
    mask_lmk: bool = True,
    expand_to_chunk: bool = True,
    sm_scale: float = None,
    m_h=None,
    return_nope_lse: bool = False,
):
    """Same API as `flex_attn_tl` but uses the atomic-free two-phase backward.

    Forward is identical to `flex_attn_tl`. Backward launches two independent
    kernels back-to-back: a dQ kernel (grid sliced over Q-blocks) and a dKdV
    kernel (grid sliced over KV-blocks). Both write directly to bf16 outputs,
    eliminating fp32 staging buffers, atomic_add traffic, and the post-cast
    kernel that the original `flex_attn_tl` backward needed.

    Inference (training=False) is currently unsupported (matches the original).

    See `flex_attn_tl` for the semantics of `sm_scale`, `m_h` and
    `return_nope_lse`.
    """
    def _to_blhd(x):
        y = x.transpose(1, 2).contiguous()
        B_, L_, H_, D_ = y.shape
        return y.reshape(-1).reshape(B_, L_, H_, D_)

    q_blhd = _to_blhd(q)
    k_blhd = _to_blhd(k)
    v_blhd = _to_blhd(v)

    use_cache = not training
    o_blhd, lse_log2, lse_nope_log2 = _FlexAttnTLTwoPhase.apply(
        q_blhd, k_blhd, v_blhd,
        window_size, chunk_size,
        mask_lmk, expand_to_chunk, use_cache,
        sm_scale,
        m_h,
    )
    ln2 = 0.6931471805599453  # ln(2)
    lse = lse_log2 * ln2
    if return_nope_lse:
        lse_nope = lse_nope_log2 * ln2
        return o_blhd, lse, lse_nope
    return o_blhd, lse


# ---------------------------------------------------------------------------
# Reference (torch) implementation for testing
# ---------------------------------------------------------------------------
def _torch_ref_flex_attn(q, k, v, window_size, chunk_size, mask_lmk, expand_to_chunk, m_h=None):
    """Reference using torch.nn.attention.flex_attention with the same mask.

    When `m_h` is provided (shape `(H_q,)`, fp32), applies the ALiBi bias
    `m_h[h] * (kv_idx - q_idx)` to the logits via `score_mod`.
    """
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    L_q = q.shape[-2]
    L_k = k.shape[-2]
    q_offset = L_k - L_q

    def mask_mod(b, h, q_idx, kv_idx):
        real_q = q_idx + q_offset
        if expand_to_chunk:
            start = real_q - window_size + 1
            chunk_start = (start // chunk_size) * chunk_size
        else:
            chunk_start = real_q - window_size + 1
        keep = (kv_idx >= chunk_start) & (kv_idx <= real_q)
        if mask_lmk:
            keep = keep & ((kv_idx + 1) % chunk_size != 0)
        return keep

    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=L_q, KV_LEN=L_k)

    score_mod = None
    if m_h is not None:
        # m_h must live on the same device as q/k/v and in fp32.
        m_h_ref = m_h.to(device=q.device, dtype=torch.float32)

        def score_mod(score, b, h, q_idx, kv_idx):
            real_q = q_idx + q_offset
            # bias = m_h[h] * (kv_idx - real_q); cast to score dtype to keep
            # flex_attention's type-checker happy.
            bias = m_h_ref[h] * (kv_idx - real_q).to(torch.float32)
            return score + bias.to(score.dtype)

    o, lse = flex_attention(
        q, k, v,
        block_mask=block_mask,
        score_mod=score_mod,
        enable_gqa=True,
        return_lse=True,
    )
    return o, lse



# ---------------------------------------------------------------------------
# Consistency test
# ---------------------------------------------------------------------------
import pytest


# Build a comprehensive list of (name, cfg_dict) test cases covering:
# - flag combinations (mask_lmk x expand_to_chunk)
# - GQA group counts (1, 2, 4, 8)
# - short / long sequences
# - window vs chunk size relations (W == S, W > S, W < S, W not multiple of S)
# - different head_dim
# - inference: q_len < kv_len, single-token decode, prefill boundary cases
# - edge: seq_len exactly fits block_M / not; window_size bigger than seq_len
_TEST_CASES = [
    # -------------------- training cases --------------------
    # flag combinations (baseline config)
    ("train_lmk_expand",     dict(training=True,  mask_lmk=True,  expand_to_chunk=True,  seq_len=512, window_size=128, chunk_size=64)),
    ("train_no_flags",       dict(training=True,  mask_lmk=False, expand_to_chunk=False, seq_len=512, window_size=128, chunk_size=64)),
    ("train_lmk_only",       dict(training=True,  mask_lmk=True,  expand_to_chunk=False, seq_len=512, window_size=128, chunk_size=64)),
    ("train_expand_only",    dict(training=True,  mask_lmk=False, expand_to_chunk=True,  seq_len=512, window_size=128, chunk_size=64)),

    # # GQA group variations
    # ("train_gqa_g1",         dict(training=True,  heads_q=4, heads_kv=4, seq_len=512)),
    # ("train_gqa_g2",         dict(training=True,  heads_q=8, heads_kv=4, seq_len=512)),
    # ("train_gqa_g4",         dict(training=True,  heads_q=8, heads_kv=2, seq_len=512)),
    # ("train_gqa_g8",         dict(training=True,  heads_q=8, heads_kv=1, seq_len=512)),

    # # batch variations
    # ("train_batch2",         dict(training=True,  batch=2, seq_len=512)),
    # ("train_batch4",         dict(training=True,  batch=4, seq_len=256)),

    # # short sequence
    # ("train_short_seq",      dict(training=True,  seq_len=256, window_size=64,  chunk_size=32)),
    # ("train_seq128",         dict(training=True,  seq_len=128, window_size=64,  chunk_size=32)),

    # # long sequence
    # ("train_long_seq",       dict(training=True,  seq_len=1024, window_size=256, chunk_size=64)),
    # ("train_long_seq_2048",  dict(training=True,  seq_len=2048, window_size=512, chunk_size=128)),

    # # window == chunk_size
    # ("train_w_eq_cs",        dict(training=True,  seq_len=512, window_size=64,  chunk_size=64)),
    # # window > chunk_size (multi-chunk window)
    # ("train_w_gt_cs",        dict(training=True,  seq_len=512, window_size=256, chunk_size=64)),
    # # window < chunk_size (window inside a chunk, only meaningful with expand_to_chunk)
    # ("train_w_lt_cs_expand", dict(training=True,  seq_len=512, window_size=32,  chunk_size=64, expand_to_chunk=True,  mask_lmk=True)),
    # ("train_w_lt_cs_noexp",  dict(training=True,  seq_len=512, window_size=32,  chunk_size=64, expand_to_chunk=False, mask_lmk=True)),
    # # window not a multiple of chunk_size
    # ("train_w_not_mult",     dict(training=True,  seq_len=512, window_size=100, chunk_size=64)),
    # # window >= seq_len (degenerate: full causal)
    # ("train_w_ge_seq",       dict(training=True,  seq_len=256, window_size=512, chunk_size=64)),

    # # head_dim variations
    # ("train_hd64",           dict(training=True,  seq_len=512, dim_qk=64,  dim_v=64)),
    # ("train_hd128",          dict(training=True,  seq_len=512, dim_qk=128, dim_v=128)),
    # # dim_qk != dim_v
    # ("train_hd_qk_ne_v",     dict(training=True,  seq_len=512, dim_qk=192, dim_v=128)),

    # # seq_len not a multiple of block_M (block_M=128)
    # ("train_seq384",         dict(training=True,  seq_len=384, window_size=128, chunk_size=64)),
    # ("train_seq640",         dict(training=True,  seq_len=640, window_size=128, chunk_size=64)),

    # -------------------- inference cases --------------------
    # q_len < kv_len, various prefill chunk sizes
    ("infer_prefill_half",   dict(training=False, seq_len=512,  q_len_inf=256,  window_size=128, chunk_size=64)),
    ("infer_prefill_small",  dict(training=False, seq_len=512,  q_len_inf=64,   window_size=128, chunk_size=64)),
    ("infer_prefill_big",    dict(training=False, seq_len=1024, q_len_inf=512,  window_size=256, chunk_size=64)),
    # single-token decode
    ("infer_decode_q1",      dict(training=False, seq_len=512,  q_len_inf=1,    window_size=128, chunk_size=64)),
    ("infer_decode_q1_long", dict(training=False, seq_len=2048, q_len_inf=1,    window_size=512, chunk_size=128)),
    # q_len == kv_len via inference path
    ("infer_full",           dict(training=False, seq_len=256,  q_len_inf=256,  window_size=128, chunk_size=64)),
    # inference with GQA
    ("infer_gqa_g4_q1",      dict(training=False, seq_len=512,  q_len_inf=1,    heads_q=8, heads_kv=2)),
    ("infer_gqa_g8_q1",      dict(training=False, seq_len=512,  q_len_inf=1,    heads_q=8, heads_kv=1)),
    # inference flag combinations
    ("infer_no_flags",       dict(training=False, seq_len=512,  q_len_inf=128,  mask_lmk=False, expand_to_chunk=False)),
    ("infer_lmk_only",       dict(training=False, seq_len=512,  q_len_inf=128,  mask_lmk=True,  expand_to_chunk=False)),
    ("infer_expand_only",    dict(training=False, seq_len=512,  q_len_inf=128,  mask_lmk=False, expand_to_chunk=True)),
    # inference q_len on block boundary / not
    ("infer_q_block_edge",   dict(training=False, seq_len=1024, q_len_inf=128,  window_size=256, chunk_size=64)),
    ("infer_q_non_aligned",  dict(training=False, seq_len=1024, q_len_inf=100,  window_size=256, chunk_size=64)),

    # -------------------- ALiBi cases --------------------
    # training: baseline ALiBi
    ("train_alibi_baseline",      dict(training=True,  seq_len=512,  window_size=128, chunk_size=64, use_alibi=True)),
    ("train_alibi_no_flags",      dict(training=True,  seq_len=512,  window_size=128, chunk_size=64, use_alibi=True,
                                        mask_lmk=False, expand_to_chunk=False)),
    ("train_alibi_lmk_only",      dict(training=True,  seq_len=512,  window_size=128, chunk_size=64, use_alibi=True,
                                        mask_lmk=True,  expand_to_chunk=False)),
    # training: ALiBi + GQA
    ("train_alibi_gqa_g4",        dict(training=True,  seq_len=512,  heads_q=8, heads_kv=2, use_alibi=True)),
    ("train_alibi_gqa_g8",        dict(training=True,  seq_len=512,  heads_q=8, heads_kv=1, use_alibi=True)),
    # training: ALiBi + misaligned seq_len
    ("train_alibi_seq333",        dict(training=True,  seq_len=333,  window_size=128, chunk_size=64, use_alibi=True)),
    ("train_alibi_seq1000",       dict(training=True,  seq_len=1000, window_size=256, chunk_size=64, use_alibi=True)),
    # training: ALiBi + long seq
    ("train_alibi_long",          dict(training=True,  seq_len=2048, window_size=512, chunk_size=128, use_alibi=True)),
    # training: ALiBi + hd64
    ("train_alibi_hd64",          dict(training=True,  seq_len=512,  dim_qk=64, dim_v=64, use_alibi=True)),
    # inference: ALiBi + prefill / decode
    ("infer_alibi_prefill_half",  dict(training=False, seq_len=512,  q_len_inf=256, window_size=128, chunk_size=64, use_alibi=True)),
    ("infer_alibi_decode_q1",     dict(training=False, seq_len=512,  q_len_inf=1,   window_size=128, chunk_size=64, use_alibi=True)),
    ("infer_alibi_decode_q1_gqa", dict(training=False, seq_len=512,  q_len_inf=1,   heads_q=8, heads_kv=2, use_alibi=True)),
]


# ---------------------------------------------------------------------------
# Regression cases: seq_len / q_len not aligned to block_N (=32 for bwd, =64 for fwd).
# The dynamic-batching production path produces non-multiple-of-block_N lengths;
# previously the bwd kernel read out-of-bounds Q/lse/dO/Delta on the tail block,
# which combined with IEEE 754 (0 * NaN = NaN) silently corrupted dq/dk/dv.
# The training-loop NaN was reproducible only when seq_len % block_N != 0.
# Keep these listed first so we test the most fragile cases up front.
_REGRESSION_CASES = [
    # ---- training: seq_len not a multiple of block_N (bwd block_N=32) ----
    # ("train_seq130_misaligned",   dict(training=True, seq_len=130, window_size=64,  chunk_size=32)),
    # ("train_seq200_misaligned",   dict(training=True, seq_len=200, window_size=64,  chunk_size=32)),
    # ("train_seq250_misaligned",   dict(training=True, seq_len=250, window_size=128, chunk_size=64)),
    # ("train_seq333_misaligned",   dict(training=True, seq_len=333, window_size=128, chunk_size=64)),
    # ("train_seq513_misaligned",   dict(training=True, seq_len=513, window_size=128, chunk_size=64)),
    # ("train_seq1000_misaligned",  dict(training=True, seq_len=1000, window_size=256, chunk_size=64)),
    # # repro the production setting (mask_lmk=True, expand_to_chunk=False) at
    # # an awkward length close to the failing run (max_seq_len=8192, dyn_bsz tail).
    # ("train_seq2049_lmk_no_exp",  dict(training=True, seq_len=2049, window_size=512, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),
    # ("train_seq4097_lmk_no_exp",  dict(training=True, seq_len=4097, window_size=512, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),
    # # GQA + misaligned (matches production: heads_q=16, heads_kv=4 -> groups=4)
    # ("train_misaligned_gqa",      dict(training=True, seq_len=333, heads_q=16, heads_kv=4,
    #                                     window_size=128, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),
    # # head_dim=64 (matches production config_hsa_testnan.json)
    # ("train_misaligned_hd64",     dict(training=True, seq_len=333, dim_qk=64, dim_v=64,
    #                                     window_size=128, chunk_size=64,
    #                                     mask_lmk=True, expand_to_chunk=False)),

    # ---- inference: q_len / kv_len not aligned to block_N (fwd block_N=64) ----
    ("infer_q_misaligned_70",     dict(training=False, seq_len=512, q_len_inf=70,
                                        window_size=128, chunk_size=64)),
    ("infer_q_misaligned_130",    dict(training=False, seq_len=512, q_len_inf=130,
                                        window_size=128, chunk_size=64)),
    ("infer_q_misaligned_333",    dict(training=False, seq_len=1024, q_len_inf=333,
                                        window_size=256, chunk_size=64)),
    ("infer_kv_misaligned",       dict(training=False, seq_len=1023, q_len_inf=100,
                                        window_size=256, chunk_size=64)),
    ("infer_both_misaligned",     dict(training=False, seq_len=777, q_len_inf=55,
                                        window_size=128, chunk_size=64,
                                        mask_lmk=True, expand_to_chunk=False)),
    # decode (q_len=1) at a non-aligned kv_len
    ("infer_decode_q1_kv1023",    dict(training=False, seq_len=1023, q_len_inf=1,
                                        window_size=256, chunk_size=64)),
    ("infer_decode_q1_kv333",     dict(training=False, seq_len=333, q_len_inf=1,
                                        window_size=128, chunk_size=64)),

    # ---- regression: ALiBi + misaligned seq_len ----
    ("train_alibi_seq333_misaligned",  dict(training=True, seq_len=333, window_size=128, chunk_size=64,
                                             use_alibi=True, mask_lmk=True, expand_to_chunk=False)),
    ("train_alibi_seq513_misaligned",  dict(training=True, seq_len=513, window_size=128, chunk_size=64,
                                             use_alibi=True)),
    ("infer_alibi_q_misaligned_130",   dict(training=False, seq_len=512, q_len_inf=130,
                                             window_size=128, chunk_size=64, use_alibi=True)),
]


# Cases for the `return_nope_lse=True` side-output path. Runs a dedicated
# runner (`_run_case_nope_lse`) that validates all three of (o, lse, lse_nope)
# against the torch reference. Not wired into pytest / `_TEST_CASES` because
# it exercises a distinct API surface.
_NOPE_LSE_CASES = [
    ("nope_lse_alibi_basic",            dict(training=True, seq_len=512,  window_size=128, chunk_size=64,
                                              mask_lmk=True,  expand_to_chunk=False)),
    ("nope_lse_alibi_seq333_misaligned", dict(training=True, seq_len=333,  window_size=128, chunk_size=64,
                                              mask_lmk=True,  expand_to_chunk=False)),
    ("nope_lse_infer_alibi",            dict(training=False, seq_len=512, q_len_inf=130,
                                              window_size=128, chunk_size=64,
                                              mask_lmk=True,  expand_to_chunk=False)),
]

# Run regression cases first so any newly-introduced bug here surfaces early.
_TEST_CASES = _REGRESSION_CASES + _TEST_CASES


@pytest.mark.parametrize("test_name, cfg", _TEST_CASES, ids=[c[0] for c in _TEST_CASES])
def test_flex_attn_tl_correctness(test_name, cfg):
    """pytest entry. Runs a single (name, cfg) combo."""
    _run_case(test_name, **cfg)


def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """Standard ALiBi slopes (Press et al. 2021).

    For n_heads a power of 2: slopes[i] = 2 ** (-8 * (i+1) / n_heads).
    For non-power-of-2, fall back to the closest-power-of-2 base plus
    interpolated extras (matches the reference implementation used in
    HuggingFace / Meta's BLOOM).
    """
    def _pow2_slopes(n):
        start = 2 ** (-(2 ** -(torch.log2(torch.tensor(n)).item() - 3)))
        return torch.tensor([start * (start ** i) for i in range(n)], dtype=torch.float32)

    import math
    if math.log2(n_heads).is_integer():
        return _pow2_slopes(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    base = _pow2_slopes(closest)
    extra = _pow2_slopes(2 * closest)[0::2][: n_heads - closest]
    return torch.cat([base, extra])


def _run_case(
    test_name="default",
    batch=1,
    heads_q=8,
    heads_kv=2,
    seq_len=512,
    dim_qk=128,
    dim_v=128,
    window_size=128,
    chunk_size=64,
    mask_lmk=True,
    expand_to_chunk=True,
    training=True,
    q_len_inf=None,
    ratio=6e-3,
    seed=0,
    use_alibi=False,
):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    L_k = seq_len
    L_q = seq_len if training else (q_len_inf if q_len_inf is not None else (seq_len // 2))

    print(f"\n{'=' * 70}")
    print(f"Test: {test_name}")
    print(f"Config: B={batch}, Hq={heads_q}, Hkv={heads_kv}, Lq={L_q}, Lk={L_k}, "
          f"D_qk={dim_qk}, D_v={dim_v}")
    print(f"        window={window_size}, chunk={chunk_size}, "
          f"mask_lmk={mask_lmk}, expand_to_chunk={expand_to_chunk}, training={training}, "
          f"alibi={use_alibi}")
    print(f"{'=' * 70}")

    q = torch.randn(batch, heads_q, L_q, dim_qk, device=device, dtype=dtype) * 0.5
    k = torch.randn(batch, heads_kv, L_k, dim_qk, device=device, dtype=dtype) * 0.5
    v = torch.randn(batch, heads_kv, L_k, dim_v, device=device, dtype=dtype) * 0.5

    m_h = _get_alibi_slopes(heads_q).to(device) if use_alibi else None

    if training:
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

    o_tl, lse_tl = flex_attn_tl(
        q, k, v,
        window_size=window_size,
        chunk_size=chunk_size,
        training=training,
        mask_lmk=mask_lmk,
        expand_to_chunk=expand_to_chunk,
        m_h=m_h,
    )

    # ref (compute up-front so we can reuse `invalid_lse_blh` to mask the
    # shared do / dlse used by both the tilelang and reference backwards).
    q_ref = q.detach().clone().requires_grad_(training)
    k_ref = k.detach().clone().requires_grad_(training)
    v_ref = v.detach().clone().requires_grad_(training)
    o_ref, lse_ref = _torch_ref_flex_attn(
        q_ref, k_ref, v_ref,
        window_size=window_size, chunk_size=chunk_size,
        mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=m_h,
    )

    o_ref_blhd = o_ref.transpose(1, 2).contiguous()
    lse_ref_blh = lse_ref.transpose(1, 2).contiguous()

    if training:
        # Build shared do / dlse, then mask out fully-masked rows on BOTH
        # paths to avoid NaN propagation. dlse is non-zero so the bwd kernel
        # actually exercises the dlse-fold-into-Delta path (lhsa_layer feeds
        # lse into a downstream softmax in real training).
        do = torch.randn_like(o_tl)
        dlse_full = torch.randn_like(lse_tl)
        with torch.no_grad():
            invalid_lse_blh = ~torch.isfinite(lse_ref_blh)  # (B, Lq, Hq)
            do_masked = do.clone()
            do_masked[invalid_lse_blh] = 0.0
            dlse_masked = dlse_full.clone()
            dlse_masked[invalid_lse_blh] = 0.0

        # ----- tilelang backward -----
        torch.autograd.backward(
            tensors=[o_tl, lse_tl],
            grad_tensors=[do_masked, dlse_masked],
            retain_graph=False,
        )
        dq_tl, q.grad = q.grad.clone(), None
        dk_tl, k.grad = k.grad.clone(), None
        dv_tl, v.grad = v.grad.clone(), None

        # ----- ref backward -----
        do_ref = do_masked.transpose(1, 2).contiguous()
        dlse_ref = dlse_masked.transpose(1, 2).contiguous()  # (B, Hq, Lq)
        torch.autograd.backward(
            tensors=[o_ref, lse_ref],
            grad_tensors=[do_ref, dlse_ref],
            retain_graph=False,
        )
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    # error helpers
    def _get_abs_err(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        return (x[m] - y[m]).abs().max().item()

    def _get_err_ratio(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        err = (x[m] - y[m]).square().mean().sqrt().item()
        base = (x[m]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def _assert_close(prefix, ref, tri, ratio_thr=ratio):
        ref_f = ref.detach().float()
        tri_f = tri.detach().float()
        # sanitize NaN introduced by -inf * 0 grads
        ref_f = torch.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)
        tri_f = torch.nan_to_num(tri_f, nan=0.0, posinf=0.0, neginf=0.0)
        abs_err = _get_abs_err(ref_f, tri_f)
        rel_ratio = _get_err_ratio(ref_f, tri_f)
        msg = f"  {prefix:<4s} abs_diff={abs_err:.6f}  rel_ratio={rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio_thr, f"[{test_name}] {prefix} failed: {msg}"

    valid_rows = torch.isfinite(lse_ref_blh).sum().item()
    print(f"  valid_rows={valid_rows}/{lse_ref_blh.numel()}")
    _assert_close("o",   o_ref_blhd, o_tl)
    _assert_close("lse", lse_ref_blh, lse_tl)

    if training:
        _assert_close("dv", dv_ref, dv_tl)
        _assert_close("dk", dk_ref, dk_tl)
        _assert_close("dq", dq_ref, dq_tl)

    print(f"[PASS] {test_name}")


# ---------------------------------------------------------------------------
# Nope-LSE side-output consistency test.
#
# Validates `flex_attn_tl(..., m_h=m_h, return_nope_lse=True)` returns:
#   - o         : attention output WITH ALiBi bias (matches m_h=m_h ref)
#   - lse       : LSE WITH ALiBi bias              (matches m_h=m_h ref)
#   - lse_nope  : LSE WITHOUT ALiBi bias           (matches m_h=None ref)
# Also checks that with `m_h=None`, `lse == lse_nope` bit-wise (since the
# kernel runs both online softmax branches on identical logits).
# ---------------------------------------------------------------------------
def _run_case_nope_lse(
    test_name="default",
    batch=1,
    heads_q=8,
    heads_kv=2,
    seq_len=512,
    dim_qk=128,
    dim_v=128,
    window_size=128,
    chunk_size=64,
    mask_lmk=True,
    expand_to_chunk=True,
    training=True,
    q_len_inf=None,
    ratio=6e-3,
    seed=0,
):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    L_k = seq_len
    L_q = seq_len if training else (q_len_inf if q_len_inf is not None else (seq_len // 2))

    print(f"\n{'=' * 70}")
    print(f"Test: {test_name}  [nope_lse runner]")
    print(f"Config: B={batch}, Hq={heads_q}, Hkv={heads_kv}, Lq={L_q}, Lk={L_k}, "
          f"D_qk={dim_qk}, D_v={dim_v}")
    print(f"        window={window_size}, chunk={chunk_size}, "
          f"mask_lmk={mask_lmk}, expand_to_chunk={expand_to_chunk}, training={training}")
    print(f"{'=' * 70}")

    q = torch.randn(batch, heads_q, L_q, dim_qk, device=device, dtype=dtype) * 0.5
    k = torch.randn(batch, heads_kv, L_k, dim_qk, device=device, dtype=dtype) * 0.5
    v = torch.randn(batch, heads_kv, L_k, dim_v, device=device, dtype=dtype) * 0.5

    m_h = _get_alibi_slopes(heads_q).to(device)

    # error helpers (same as _run_case)
    def _get_abs_err(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        return (x[m] - y[m]).abs().max().item()

    def _get_err_ratio(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        err = (x[m] - y[m]).square().mean().sqrt().item()
        base = (x[m]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def _assert_close(prefix, ref, tri, ratio_thr=ratio):
        ref_f = ref.detach().float()
        tri_f = tri.detach().float()
        ref_f = torch.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)
        tri_f = torch.nan_to_num(tri_f, nan=0.0, posinf=0.0, neginf=0.0)
        abs_err = _get_abs_err(ref_f, tri_f)
        rel_ratio = _get_err_ratio(ref_f, tri_f)
        msg = f"  {prefix:<10s} abs_diff={abs_err:.6f}  rel_ratio={rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio_thr, f"[{test_name}] {prefix} failed: {msg}"

    # ---- Part 1: use_alibi=True, assert o / lse / lse_nope all correct.
    # When training=True, also validate that the modified autograd path
    # (`backward(ctx, do, dlse, dlse_nope)` with `dlse_nope` ignored) still
    # produces correct dq / dk / dv - i.e. adding the lse_nope side output
    # did not break gradient flow through `o`.
    if training:
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)

    o_tl, lse_tl, lse_nope_tl = flex_attn_tl(
        q, k, v,
        window_size=window_size, chunk_size=chunk_size,
        training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=m_h, return_nope_lse=True,
    )

    if training:
        do = torch.randn_like(o_tl)
        # Only flow grad through `o`; `lse` and `lse_nope` are not
        # differentiated (matches the real usage pattern where lse_nope
        # feeds an external topk with no gradient path).
        o_tl.backward(do, retain_graph=False)
        dq_tl, q.grad = q.grad.clone(), None
        dk_tl, k.grad = k.grad.clone(), None
        dv_tl, v.grad = v.grad.clone(), None

    # ref WITH alibi (for o / lse / grads)
    q_ref = q.detach().clone().requires_grad_(training)
    k_ref = k.detach().clone().requires_grad_(training)
    v_ref = v.detach().clone().requires_grad_(training)
    o_ref_al, lse_ref_al = _torch_ref_flex_attn(
        q_ref, k_ref, v_ref,
        window_size=window_size, chunk_size=chunk_size,
        mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=m_h,
    )
    # ref WITHOUT alibi (for lse_nope; no backward needed on this path)
    q_ref_np = q.detach().clone()
    k_ref_np = k.detach().clone()
    v_ref_np = v.detach().clone()
    _, lse_ref_nope = _torch_ref_flex_attn(
        q_ref_np, k_ref_np, v_ref_np,
        window_size=window_size, chunk_size=chunk_size,
        mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=None,
    )

    o_ref_al_blhd   = o_ref_al.transpose(1, 2).contiguous()
    lse_ref_al_blh  = lse_ref_al.transpose(1, 2).contiguous()
    lse_ref_np_blh  = lse_ref_nope.transpose(1, 2).contiguous()

    if training:
        # Mirror _run_case: mask fully-masked rows in grad_output to avoid
        # NaN propagation through -inf lse rows in the ref path.
        with torch.no_grad():
            invalid_lse_blh = ~torch.isfinite(lse_ref_al_blh)  # (B, Lq, Hq)
            do_masked = do.clone()
            do_masked[invalid_lse_blh] = 0.0
        do_ref = do_masked.transpose(1, 2).contiguous()
        o_ref_al.backward(do_ref, retain_graph=False)
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    valid_rows = torch.isfinite(lse_ref_al_blh).sum().item()
    print(f"  valid_rows={valid_rows}/{lse_ref_al_blh.numel()}")
    _assert_close("o",         o_ref_al_blhd, o_tl)
    _assert_close("lse",       lse_ref_al_blh, lse_tl)
    _assert_close("lse_nope",  lse_ref_np_blh, lse_nope_tl)

    if training:
        _assert_close("dv", dv_ref, dv_tl)
        _assert_close("dk", dk_ref, dk_tl)
        _assert_close("dq", dq_ref, dq_tl)

    # ---- Part 2: use_alibi=False, lse must equal lse_nope bitwise ----
    # Use detached tensors: we only need forward equality here.
    q_np = q.detach()
    k_np = k.detach()
    v_np = v.detach()
    o2_tl, lse2_tl, lse2_nope_tl = flex_attn_tl(
        q_np, k_np, v_np,
        window_size=window_size, chunk_size=chunk_size,
        training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=None, return_nope_lse=True,
    )
    # Compare only over finite entries: fully-masked rows are -inf on both
    # sides (e.g. landmark tokens with mask_lmk=True), so equality check
    # over the whole tensor is fine with nan_to_num.
    if not torch.equal(lse2_tl, lse2_nope_tl):
        diff = (lse2_tl - lse2_nope_tl).abs()
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        max_diff = diff.max().item()
        msg = (f"[{test_name}] m_h=None: lse != lse_nope bitwise, "
               f"max_abs_diff={max_diff}")
        print(msg)
        assert max_diff == 0.0, msg
    print("  m_h=None: lse == lse_nope  (bitwise OK)")

    print(f"[PASS] {test_name}")


# ---------------------------------------------------------------------------
# Two-phase backward consistency test (training only).
#
# Strategy: run both `flex_attn_tl` (atomic bwd) and `flex_attn_tl_two_phase`
# (two-phase bwd) on identical inputs, then compare:
#   1. fwd outputs (o, lse): MUST be bitwise identical (same fwd kernel).
#   2. dq / dk / dv vs the torch reference: same tolerance as the atomic path.
#   3. dq / dk / dv vs the atomic path: tighter tolerance (both bf16, same fwd
#      lse / Delta inputs, only the bwd kernel differs).
# ---------------------------------------------------------------------------

# Only training cases are valid for two-phase (bwd not supported in inference).
_TEST_CASES_TWO_PHASE = [(name, cfg) for (name, cfg) in _TEST_CASES if cfg.get("training", True)]


@pytest.mark.parametrize(
    "test_name, cfg", _TEST_CASES_TWO_PHASE, ids=[c[0] for c in _TEST_CASES_TWO_PHASE]
)
def test_flex_attn_tl_two_phase_correctness(test_name, cfg):
    """pytest entry for the two-phase backward path."""
    _run_case_two_phase(test_name, **cfg)


def _run_case_two_phase(
    test_name="default",
    batch=1,
    heads_q=8,
    heads_kv=2,
    seq_len=512,
    dim_qk=128,
    dim_v=128,
    window_size=128,
    chunk_size=64,
    mask_lmk=True,
    expand_to_chunk=True,
    training=True,  # ignored: two-phase bwd is training-only
    q_len_inf=None,  # ignored
    ratio=6e-3,
    ratio_vs_atomic=1e-3,
    seed=0,
    use_alibi=False,
):
    assert training, "two-phase bwd is training-only"
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    L_k = seq_len
    L_q = seq_len

    print(f"\n{'=' * 70}")
    print(f"Test (two-phase): {test_name}")
    print(f"Config: B={batch}, Hq={heads_q}, Hkv={heads_kv}, Lq={L_q}, Lk={L_k}, "
          f"D_qk={dim_qk}, D_v={dim_v}")
    print(f"        window={window_size}, chunk={chunk_size}, "
          f"mask_lmk={mask_lmk}, expand_to_chunk={expand_to_chunk}, alibi={use_alibi}")
    print(f"{'=' * 70}")

    # Shared base inputs (no grad). Each path gets its own clone with grad.
    q_base = torch.randn(batch, heads_q, L_q, dim_qk, device=device, dtype=dtype) * 0.5
    k_base = torch.randn(batch, heads_kv, L_k, dim_qk, device=device, dtype=dtype) * 0.5
    v_base = torch.randn(batch, heads_kv, L_k, dim_v, device=device, dtype=dtype) * 0.5

    m_h = _get_alibi_slopes(heads_q).to(device) if use_alibi else None

    def _clone_with_grad():
        return (
            q_base.detach().clone().requires_grad_(True),
            k_base.detach().clone().requires_grad_(True),
            v_base.detach().clone().requires_grad_(True),
        )

    # ----- atomic path -----
    q_a, k_a, v_a = _clone_with_grad()
    o_a, lse_a = flex_attn_tl(
        q_a, k_a, v_a,
        window_size=window_size, chunk_size=chunk_size,
        training=True, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=m_h,
    )

    # ----- two-phase path -----
    q_p, k_p, v_p = _clone_with_grad()
    o_p, lse_p = flex_attn_tl_two_phase(
        q_p, k_p, v_p,
        window_size=window_size, chunk_size=chunk_size,
        training=True, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=m_h,
    )

    # Forward outputs use the same fwd kernel -> must be bitwise identical.
    assert torch.equal(o_a, o_p), \
        f"[{test_name}] forward o mismatch between atomic and two-phase paths"
    assert torch.equal(lse_a, lse_p), \
        f"[{test_name}] forward lse mismatch between atomic and two-phase paths"

    # ----- shared do / dlse (mask out fully-masked rows to avoid NaN propagation) -----
    do = torch.randn_like(o_a)
    dlse_full = torch.randn_like(lse_a)

    # Build the same lse-based row validity mask used by `_run_case`.
    q_ref = q_base.detach().clone().requires_grad_(True)
    k_ref = k_base.detach().clone().requires_grad_(True)
    v_ref = v_base.detach().clone().requires_grad_(True)
    o_ref, lse_ref = _torch_ref_flex_attn(
        q_ref, k_ref, v_ref,
        window_size=window_size, chunk_size=chunk_size,
        mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        m_h=m_h,
    )
    o_ref_blhd = o_ref.transpose(1, 2).contiguous()
    lse_ref_blh = lse_ref.transpose(1, 2).contiguous()
    with torch.no_grad():
        invalid_lse_blh = ~torch.isfinite(lse_ref_blh)  # (B, Lq, Hq)
        do_masked = do.clone()
        do_masked[invalid_lse_blh] = 0.0
        dlse_masked = dlse_full.clone()
        dlse_masked[invalid_lse_blh] = 0.0

    # Backward both tilelang paths with the same masked do / dlse. dlse is
    # non-zero so the dlse-fold-into-Delta path is exercised on both kernels.
    torch.autograd.backward(
        tensors=[o_a, lse_a],
        grad_tensors=[do_masked, dlse_masked],
        retain_graph=False,
    )
    dq_a, dk_a, dv_a = q_a.grad.clone(), k_a.grad.clone(), v_a.grad.clone()

    torch.autograd.backward(
        tensors=[o_p, lse_p],
        grad_tensors=[do_masked, dlse_masked],
        retain_graph=False,
    )
    dq_p, dk_p, dv_p = q_p.grad.clone(), k_p.grad.clone(), v_p.grad.clone()

    # Reference grads (torch flex_attention) for absolute-correctness checks.
    do_ref = do_masked.transpose(1, 2).contiguous()
    dlse_ref = dlse_masked.transpose(1, 2).contiguous()  # (B, Hq, Lq)
    torch.autograd.backward(
        tensors=[o_ref, lse_ref],
        grad_tensors=[do_ref, dlse_ref],
        retain_graph=False,
    )
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    # ----- error helpers (same as `_run_case`) -----
    def _get_abs_err(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        return (x[m] - y[m]).abs().max().item()

    def _get_err_ratio(x, y):
        m = (x > -1e5) & (y > -1e5)
        if m.sum() == 0:
            return 0.0
        err = (x[m] - y[m]).square().mean().sqrt().item()
        base = (x[m]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def _assert_close(prefix, ref, tri, ratio_thr):
        ref_f = ref.detach().float()
        tri_f = tri.detach().float()
        ref_f = torch.nan_to_num(ref_f, nan=0.0, posinf=0.0, neginf=0.0)
        tri_f = torch.nan_to_num(tri_f, nan=0.0, posinf=0.0, neginf=0.0)
        abs_err = _get_abs_err(ref_f, tri_f)
        rel_ratio = _get_err_ratio(ref_f, tri_f)
        msg = f"  {prefix:<24s} abs_diff={abs_err:.6f}  rel_ratio={rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio_thr, f"[{test_name}] {prefix} failed: {msg}"

    valid_rows = torch.isfinite(lse_ref_blh).sum().item()
    print(f"  valid_rows={valid_rows}/{lse_ref_blh.numel()}")

    # 1) two-phase grads vs torch reference (same tolerance as atomic path).
    _assert_close("dv (2P vs ref)",     dv_ref, dv_p, ratio)
    _assert_close("dk (2P vs ref)",     dk_ref, dk_p, ratio)
    _assert_close("dq (2P vs ref)",     dq_ref, dq_p, ratio)

    # 2) two-phase grads vs atomic-path grads (tighter tolerance: only bwd
    #    kernel differs, fwd lse / Delta are identical).
    _assert_close("dv (2P vs atomic)",  dv_a, dv_p, ratio_vs_atomic)
    _assert_close("dk (2P vs atomic)",  dk_a, dk_p, ratio_vs_atomic)
    _assert_close("dq (2P vs atomic)",  dq_a, dq_p, ratio_vs_atomic)

    print(f"[PASS] {test_name}")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(
    batch=1,
    heads_q=8,
    heads_kv=2,
    seq_len=4096,
    dim_qk=128,
    dim_v=128,
    window_size=512,
    chunk_size=64,
    mask_lmk=True,
    expand_to_chunk=True,
    training=True,
    q_len_inf=None,
    warmup=10,
    iters=50,
    skip_ref=False,
    seed=0,
):
    """Benchmark fwd / bwd / total of flex_attn_tl vs torch flex_attention reference.

    Times are reported in milliseconds. Torch ref is invoked the same way as in
    the consistency test (uses torch.nn.attention.flex_attention with the same
    mask_mod). Set skip_ref=True to benchmark the tilelang kernel only.
    """
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    L_k = seq_len
    L_q = seq_len if training else (q_len_inf if q_len_inf is not None else (seq_len // 2))

    print(f"\n{'=' * 70}")
    print(f"Benchmark: B={batch}, Hq={heads_q}, Hkv={heads_kv}, Lq={L_q}, Lk={L_k}, "
          f"D_qk={dim_qk}, D_v={dim_v}, window={window_size}, chunk={chunk_size}")
    print(f"           mask_lmk={mask_lmk}, expand_to_chunk={expand_to_chunk}, "
          f"training={training}, warmup={warmup}, iters={iters}")
    print(f"{'=' * 70}")

    def _make_inputs(req_grad):
        q = torch.randn(batch, heads_q, L_q, dim_qk, device=device, dtype=dtype) * 0.5
        k = torch.randn(batch, heads_kv, L_k, dim_qk, device=device, dtype=dtype) * 0.5
        v = torch.randn(batch, heads_kv, L_k, dim_v, device=device, dtype=dtype) * 0.5
        if req_grad and training:
            q.requires_grad_(True)
            k.requires_grad_(True)
            v.requires_grad_(True)
        return q, k, v

    def _bench(fn):
        # warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters  # ms

    # ----- tilelang kernel -----
    q_tl, k_tl, v_tl = _make_inputs(req_grad=True)

    def _tl_fwd():
        o, lse = flex_attn_tl(
            q_tl, k_tl, v_tl,
            window_size=window_size, chunk_size=chunk_size,
            training=training, mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
        )
        return o, lse

    tl_fwd_ms = _bench(lambda: _tl_fwd())

    if training:
        o_tl, _ = _tl_fwd()
        do_tl = torch.randn_like(o_tl)

        def _tl_bwd_only():
            # rerun fwd inside autograd-tracking, then bwd; we measure both halves
            # separately by also timing fwd alone above.
            o2, _ = _tl_fwd()
            for p in (q_tl, k_tl, v_tl):
                if p.grad is not None:
                    p.grad = None
            o2.backward(do_tl, retain_graph=False)

        tl_total_ms = _bench(_tl_bwd_only)
        tl_bwd_ms = tl_total_ms - tl_fwd_ms
    else:
        tl_total_ms = tl_fwd_ms
        tl_bwd_ms = 0.0

    # ----- torch reference -----
    if not skip_ref:
        q_ref, k_ref, v_ref = _make_inputs(req_grad=True)

        def _ref_fwd():
            o, lse = _torch_ref_flex_attn(
                q_ref, k_ref, v_ref,
                window_size=window_size, chunk_size=chunk_size,
                mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk,
            )
            return o, lse

        # one warm call to trigger torch.compile inside flex_attention
        _ref_fwd()
        torch.cuda.synchronize()

        ref_fwd_ms = _bench(lambda: _ref_fwd())

        if training:
            o_ref, _ = _ref_fwd()
            do_ref = torch.randn_like(o_ref)

            def _ref_bwd_only():
                o2, _ = _ref_fwd()
                for p in (q_ref, k_ref, v_ref):
                    if p.grad is not None:
                        p.grad = None
                o2.backward(do_ref, retain_graph=False)

            ref_total_ms = _bench(_ref_bwd_only)
            ref_bwd_ms = ref_total_ms - ref_fwd_ms
        else:
            ref_total_ms = ref_fwd_ms
            ref_bwd_ms = 0.0
    else:
        ref_fwd_ms = ref_bwd_ms = ref_total_ms = float("nan")

    # ----- print -----
    def _fmt(x):
        return f"{x:8.3f}" if x == x else "    n/a "  # NaN check

    def _spd(ref, tl):
        if ref != ref or tl != tl or tl == 0:
            return "    n/a"
        return f"{ref / tl:6.2f}x"

    print(f"{'phase':<8s} | {'torch_ref (ms)':>14s} | {'tilelang (ms)':>14s} | {'speedup':>8s}")
    print("-" * 60)
    print(f"{'fwd':<8s} | {_fmt(ref_fwd_ms):>14s} | {_fmt(tl_fwd_ms):>14s} | {_spd(ref_fwd_ms, tl_fwd_ms):>8s}")
    if training:
        print(f"{'bwd':<8s} | {_fmt(ref_bwd_ms):>14s} | {_fmt(tl_bwd_ms):>14s} | {_spd(ref_bwd_ms, tl_bwd_ms):>8s}")
        print(f"{'total':<8s} | {_fmt(ref_total_ms):>14s} | {_fmt(tl_total_ms):>14s} | {_spd(ref_total_ms, tl_total_ms):>8s}")


_BENCH_CASES = [
    # ---- training: typical training shapes ----
    ("train_2k_w512",   dict(training=True,  seq_len=2048, window_size=512,  chunk_size=64)),
    ("train_4k_w512",   dict(training=True,  seq_len=4096, window_size=512,  chunk_size=64)),
    ("train_8k_w1024",  dict(training=True,  seq_len=8192, window_size=1024, chunk_size=128)),
    # ---- inference: prefill ----
    ("infer_prefill_4k_q1k", dict(training=False, seq_len=4096, q_len_inf=1024, window_size=512, chunk_size=64)),
    # ---- inference: single token decode ----
    ("infer_decode_4k_q1",   dict(training=False, seq_len=4096, q_len_inf=1,    window_size=512, chunk_size=64)),
    ("infer_decode_8k_q1",   dict(training=False, seq_len=8192, q_len_inf=1,    window_size=1024, chunk_size=128)),
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        for name, cfg in _BENCH_CASES:
            print(f"\n>>> {name}")
            try:
                benchmark(**cfg)
            except Exception as e:
                print(f"[ERROR] {name}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        sys.exit(0)

    # Test selection:
    #   (no arg) / "regression"  -> only newly added misalignment regression cases (default).
    #   "all"                    -> regression cases first, then the rest.
    #   "nope_lse"               -> only the nope-LSE side-output cases.
    mode = sys.argv[1] if len(sys.argv) > 1 else "regression"
    if mode == "all":
        base_cases = _TEST_CASES
    elif mode == "regression":
        base_cases = _REGRESSION_CASES
    elif mode == "nope_lse":
        base_cases = []
    else:
        print(f"Unknown mode: {mode}. Use 'regression' (default), 'all', 'nope_lse', or 'bench'.")
        sys.exit(2)

    # Nope-LSE cases always run first (on every mode except "bench") so any
    # regression in the side-output path surfaces before we enter the heavy
    # correctness sweep.
    nope_lse_cases = _NOPE_LSE_CASES
    total = len(nope_lse_cases) + len(base_cases)
    print(f"Running {total} test cases (mode={mode}): "
          f"{len(nope_lse_cases)} nope_lse + {len(base_cases)} {mode}")

    failures = []
    for name, cfg in nope_lse_cases:
        try:
            _run_case_nope_lse(name, **cfg)
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}: {e}\n")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name}: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()

    for name, cfg in base_cases:
        try:
            _run_case(name, **cfg)
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}: {e}\n")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name}: {type(e).__name__}: {e}\n")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Summary: {total - len(failures)}/{total} passed")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        raise SystemExit(1)
    print("All tests passed")

