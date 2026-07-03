"""
TileLang fused kernel for chunk_attn_pool (forward + backward).

Design v8 - "H as M, zero waste":
  - Grid: (BN, 1) - one thread block per batch*chunk, processes ALL heads
  - GEMM1: Q[H, D] @ K[S, D]^T -> [H, S]  (M=H=32, zero waste, each row = 1 head)
  - Softmax: reduce_max/sum(dim=1) over S per head
  - GEMM2: P[H, S] @ K[S, D] -> [H, D]   (M=H=32, zero waste)
  - K[S, D] loaded once, reused by all heads via shared memory
  - Requires H >= 16 (true for MHA with H=32)
"""

import math
import torch
import tilelang
from tilelang import language as T


# =====================================================================
# Forward Kernel
# =====================================================================

@tilelang.jit(
    out_idx=[2, 3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def chunk_attn_pool_fwd_kernel(BN, chunk_size, heads, head_dim, sm_scale_val, threads=128, compact_lmk_k=False):
    """
    Forward kernel. H heads as M dimension - zero GEMM waste.

    GEMM1: Q[H, D] @ K[S, D]^T -> logits[H, S]
    Softmax over dim=1 (S) per head
    GEMM2: P[H, S] @ K[S, D] -> lmk_k[H, D]
    """
    dtype = "bfloat16"
    accum_dtype = "float"

    S = chunk_size   # 64
    D = head_dim     # 128
    H = heads        # 32 (MHA)
    LN2 = 0.6931471805599453

    q_shape = [BN, heads, head_dim]
    k_shape = [BN, chunk_size, heads, head_dim]
    out_k_shape = [BN, heads, head_dim]
    out_b_shape = [BN, heads]
    out_p_shape = [BN, chunk_size, heads]

    @T.prim_func
    def main(
        MuQ: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LmkK: T.Tensor(out_k_shape, dtype),
        LmkB: T.Tensor(out_b_shape, accum_dtype),
        P_out: T.Tensor(out_p_shape, accum_dtype),
    ):
        with T.Kernel(BN, threads=threads) as (i_bn,):
            # ---- Shared memory ----
            Q_shared = T.alloc_shared([H, D], dtype)        # each row = one head's Q
            K_shared = T.alloc_shared([S, D], dtype)        # K shared across all heads
            P_shared = T.alloc_shared([H, S], dtype)        # P for GEMM2 input
            O_shared = T.alloc_shared([H, D], dtype)        # output buffer
            P_f32_shared = T.alloc_shared([H, S], accum_dtype)  # P in fp32 for save & entropy

            # ---- Fragments ----
            acc_s = T.alloc_fragment([H, S], accum_dtype)   # logits -> P
            acc_o = T.alloc_fragment([H, D], accum_dtype)   # output
            scores_max = T.alloc_fragment([H], accum_dtype)
            scores_sum = T.alloc_fragment([H], accum_dtype)
            ent_frag = T.alloc_fragment([H, S], accum_dtype)
            ent_sum = T.alloc_fragment([H], accum_dtype)

            # ---- Load Q[H, D]: each row h = mu_q[bn, h, :] ----
            for h, d in T.Parallel(H, D):
                Q_shared[h, d] = MuQ[i_bn, h, d]

            # ---- Load K[S, D]: K is the SAME for all heads (shared KV) ----
            # k_shape is [BN, S, H, D] but all heads share same K
            # Use head 0 as representative (caller ensures K is identical across heads)
            for s, d in T.Parallel(S, D):
                K_shared[s, d] = K[i_bn, s, 0, d]

            # ==== GEMM1: logits = Q @ K^T -> [H, S] ====
            T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow)

            # Scale + mask last token
            for i, j in T.Parallel(H, S):
                acc_s[i, j] = T.if_then_else(
                    j == S - 1,
                    -T.infinity(accum_dtype),
                    acc_s[i, j] * sm_scale_val,
                )

            # ==== Softmax over dim=1 (S) per head ====
            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=True)

            for i, j in T.Parallel(H, S):
                acc_s[i, j] = T.exp2(acc_s[i, j] * 1.44269504 - scores_max[i] * 1.44269504)

            T.fill(scores_sum, 0.0)
            T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

            for i, j in T.Parallel(H, S):
                acc_s[i, j] = acc_s[i, j] / scores_sum[i]

            # ==== Entropy = -sum(p * log(p)) per head, reduce dim=1 ====
            for i, j in T.Parallel(H, S):
                ent_frag[i, j] = T.if_then_else(
                    acc_s[i, j] > 0.0,
                    acc_s[i, j] * T.log2(acc_s[i, j]) * LN2,
                    0.0,
                )
            T.fill(ent_sum, 0.0)
            T.reduce_sum(ent_frag, ent_sum, dim=1, clear=True)

            # ==== Save P and entropy ====
            # Copy acc_s to shared for save + GEMM2
            T.copy(acc_s, P_shared)
            T.sync_threads()

            # Save P_out[bn, s, h] and LmkB[bn, h]
            for h, s in T.Parallel(H, S):
                P_out[i_bn, s, h] = T.cast(P_shared[h, s], accum_dtype)

            for h in T.Parallel(H):
                LmkB[i_bn, h] = -ent_sum[h]

            if not compact_lmk_k:
                # ==== GEMM2: lmk_k = P @ K -> [H, D] ====
                T.clear(acc_o)
                T.gemm(P_shared, K_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                # Store all heads
                T.copy(acc_o, O_shared)
                for h, d in T.Parallel(H, D):
                    LmkK[i_bn, h, d] = O_shared[h, d]
            else:
                # Compact mode only needs p*log(p) entropy; caller reuses the
                # chunk-boundary K token as lmk_k.
                for h, d in T.Parallel(H, D):
                    LmkK[i_bn, h, d] = 0.0

    return main


# =====================================================================
# Backward Kernel
# =====================================================================

@tilelang.jit(
    out_idx=[6, 7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def chunk_attn_pool_bwd_kernel(BN, chunk_size, heads, head_dim, sm_scale_val, threads=128, compact_lmk_k=False):
    """
    Backward kernel. H heads as M dimension.

    GEMM1: GradK[H, D] @ K[S, D]^T -> dP[H, S]
    d_logits computed per head via fragment ops
    GEMM2: DLogits[H, S] @ K[S, D] -> d_mu_q[H, D]
    d_K: accumulate from all heads
    """
    dtype = "bfloat16"
    accum_dtype = "float"

    S = chunk_size
    D = head_dim
    H = heads
    LN2 = 0.6931471805599453

    q_shape = [BN, heads, head_dim]
    k_shape = [BN, chunk_size, heads, head_dim]
    p_shape = [BN, chunk_size, heads]
    b_shape = [BN, heads]
    grad_k_shape = [BN, heads, head_dim]
    grad_b_shape = [BN, heads]
    d_mu_q_shape = [BN, heads, head_dim]
    d_k_shape = [BN, chunk_size, heads, head_dim]

    @T.prim_func
    def main(
        MuQ: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        P_saved: T.Tensor(p_shape, accum_dtype),
        LmkB_saved: T.Tensor(b_shape, accum_dtype),
        GradLmkK: T.Tensor(grad_k_shape, dtype),
        GradLmkB: T.Tensor(grad_b_shape, accum_dtype),
        DMuQ: T.Tensor(d_mu_q_shape, accum_dtype),
        DK: T.Tensor(d_k_shape, accum_dtype),
    ):
        with T.Kernel(BN, threads=threads) as (i_bn,):
            # ---- Shared memory ----
            K_shared = T.alloc_shared([S, D], dtype)
            GradK_shared = T.alloc_shared([H, D], dtype)
            MuQ_shared = T.alloc_shared([H, D], dtype)
            DLogits_shared = T.alloc_shared([H, S], dtype)
            O_shared = T.alloc_shared([H, D], dtype)
            # P and dP in shared for d_logits computation
            P_shared = T.alloc_shared([H, S], accum_dtype)
            dP_shared = T.alloc_shared([H, S], accum_dtype)

            # ---- Fragments ----
            dP_frag = T.alloc_fragment([H, S], accum_dtype)
            p_frag = T.alloc_fragment([H, S], accum_dtype)
            d_logits_frag = T.alloc_fragment([H, S], accum_dtype)
            delta_partial = T.alloc_fragment([H, S], accum_dtype)
            delta = T.alloc_fragment([H], accum_dtype)
            d_mu_q_frag = T.alloc_fragment([H, D], accum_dtype)

            # ---- Load K (head 0, same for all heads) ----
            for s, d in T.Parallel(S, D):
                K_shared[s, d] = K[i_bn, s, 0, d]

            # ---- Load grad_lmk_k: each row h = grad[bn, h, :] ----
            if not compact_lmk_k:
                for h, d in T.Parallel(H, D):
                    GradK_shared[h, d] = GradLmkK[i_bn, h, d]
            else:
                for h, d in T.Parallel(H, D):
                    GradK_shared[h, d] = 0.0

            # ---- Load mu_q: each row h = mu_q[bn, h, :] ----
            for h, d in T.Parallel(H, D):
                MuQ_shared[h, d] = MuQ[i_bn, h, d]

            # ---- Load p into fragment [H, S] ----
            for h, s in T.Parallel(H, S):
                p_frag[h, s] = P_saved[i_bn, s, h]

            # ==== GEMM1: dP = GradK @ K^T -> [H, S] ====
            T.clear(dP_frag)
            if not compact_lmk_k:
                T.gemm(GradK_shared, K_shared, dP_frag, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

            # ==== delta = sum_s p_s * dP_s per head (reduce dim=1) ====
            for i, j in T.Parallel(H, S):
                delta_partial[i, j] = p_frag[i, j] * dP_frag[i, j]
            T.fill(delta, 0.0)
            T.reduce_sum(delta_partial, delta, dim=1, clear=True)

            # ==== d_logits = p * [(dP - delta) - grad_b * (log_p + entropy)] ====
            for i, j in T.Parallel(H, S):
                log_p_val = T.if_then_else(
                    p_frag[i, j] > 0.0,
                    T.log2(p_frag[i, j]) * LN2,
                    0.0,
                )
                entropy_h = LmkB_saved[i_bn, i]
                grad_b_h = GradLmkB[i_bn, i]
                d_logits_frag[i, j] = T.if_then_else(
                    p_frag[i, j] > 0.0,
                    p_frag[i, j] * (
                        (dP_frag[i, j] - delta[i])
                        - grad_b_h * (log_p_val + entropy_h)
                    ),
                    0.0,
                )

            # ==== GEMM2: d_mu_q = sm_scale * DLogits @ K -> [H, D] ====
            T.copy(d_logits_frag, DLogits_shared)
            T.clear(d_mu_q_frag)
            T.gemm(DLogits_shared, K_shared, d_mu_q_frag,
                    policy=T.GemmWarpPolicy.FullRow)

            # Store d_mu_q with sm_scale
            T.copy(d_mu_q_frag, O_shared)
            for h, d in T.Parallel(H, D):
                DMuQ[i_bn, h, d] = T.cast(O_shared[h, d], accum_dtype) * sm_scale_val

            # ==== d_K[s, h, d] = sm_scale * d_logits[h, s] * mu_q[h, d] + p[h, s] * grad_k[h, d] ====
            # Need d_logits and p in shared for access
            T.copy(d_logits_frag, dP_shared)  # reuse dP_shared buffer for d_logits
            T.copy(p_frag, P_shared)
            T.sync_threads()

            # Shared-K: aggregate per-head grads into head 0, zero remaining heads.
            if not compact_lmk_k:
                for s, d in T.Parallel(S, D):
                    dk_acc = T.alloc_local([1], accum_dtype)
                    dk_acc[0] = 0.0
                    for h in T.serial(H):
                        dk_acc[0] = dk_acc[0] + (
                            sm_scale_val * T.cast(dP_shared[h, s], accum_dtype) * T.cast(MuQ_shared[h, d], accum_dtype)
                            + P_shared[h, s] * T.cast(GradK_shared[h, d], accum_dtype)
                        )
                    DK[i_bn, s, 0, d] = dk_acc[0]
            else:
                for s, d in T.Parallel(S, D):
                    dk_acc = T.alloc_local([1], accum_dtype)
                    dk_acc[0] = 0.0
                    for h in T.serial(H):
                        dk_acc[0] = dk_acc[0] + (
                            sm_scale_val * T.cast(dP_shared[h, s], accum_dtype) * T.cast(MuQ_shared[h, d], accum_dtype)
                        )
                    DK[i_bn, s, 0, d] = dk_acc[0]

            # Zero out heads 1..H-1 (shared-K: aggregated grad lives in head 0)
            for h_idx in T.serial(H - 1):
                for s, d in T.Parallel(S, D):
                    DK[i_bn, s, h_idx + 1, d] = 0.0

    return main


# =====================================================================
# Fallback Forward Kernel (for H < 16: pad M=16, grid per head)
# =====================================================================

@tilelang.jit(
    out_idx=[2, 3, 4],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def chunk_attn_pool_fwd_kernel_fallback(BN, chunk_size, heads, head_dim, sm_scale_val, threads=128, compact_lmk_k=False):
    """Fallback forward kernel for H < 16. Pads M to 16, grid = (BN, heads)."""
    dtype = "bfloat16"
    accum_dtype = "float"
    S = chunk_size
    D = head_dim
    M = 16
    LN2 = 0.6931471805599453

    q_shape = [BN, heads, head_dim]
    k_shape = [BN, chunk_size, heads, head_dim]
    out_k_shape = [BN, heads, head_dim]
    out_b_shape = [BN, heads]
    out_p_shape = [BN, chunk_size, heads]

    @T.prim_func
    def main(
        MuQ: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LmkK: T.Tensor(out_k_shape, dtype),
        LmkB: T.Tensor(out_b_shape, accum_dtype),
        P_out: T.Tensor(out_p_shape, accum_dtype),
    ):
        with T.Kernel(BN, heads, threads=threads) as (i_bn, i_h):
            Q_shared = T.alloc_shared([M, D], dtype)
            K_shared = T.alloc_shared([S, D], dtype)
            P_shared = T.alloc_shared([M, S], dtype)
            O_shared = T.alloc_shared([M, D], dtype)
            P_row_shared = T.alloc_shared([S], accum_dtype)

            acc_s = T.alloc_fragment([M, S], accum_dtype)
            acc_o = T.alloc_fragment([M, D], accum_dtype)
            scores_max = T.alloc_fragment([M], accum_dtype)
            scores_sum = T.alloc_fragment([M], accum_dtype)

            for s, d in T.Parallel(S, D):
                K_shared[s, d] = K[i_bn, s, i_h, d]
            for m, d in T.Parallel(M, D):
                Q_shared[m, d] = MuQ[i_bn, i_h, d]

            T.clear(acc_s)
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(M, S):
                acc_s[i, j] = T.if_then_else(j == S - 1, -T.infinity(accum_dtype), acc_s[i, j] * sm_scale_val)

            T.fill(scores_max, -T.infinity(accum_dtype))
            T.reduce_max(acc_s, scores_max, dim=1, clear=True)
            for i, j in T.Parallel(M, S):
                acc_s[i, j] = T.exp2(acc_s[i, j] * 1.44269504 - scores_max[i] * 1.44269504)
            T.fill(scores_sum, 0.0)
            T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)
            for i, j in T.Parallel(M, S):
                acc_s[i, j] = acc_s[i, j] / scores_sum[i]

            T.copy(acc_s, P_shared)
            T.sync_threads()
            for j in T.Parallel(S):
                P_out[i_bn, j, i_h] = T.cast(P_shared[0, j], accum_dtype)
            for j in T.Parallel(S):
                P_row_shared[j] = T.cast(P_shared[0, j], accum_dtype)

            entropy_local = T.alloc_local([1], accum_dtype)
            entropy_local[0] = 0.0
            for s in T.serial(S):
                p_val = P_row_shared[s]
                if p_val > 0.0:
                    entropy_local[0] = entropy_local[0] + p_val * T.log2(p_val) * LN2
            LmkB[i_bn, i_h] = -entropy_local[0]

            if not compact_lmk_k:
                T.copy(acc_s, P_shared)
                T.clear(acc_o)
                T.gemm(P_shared, K_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                T.copy(acc_o, O_shared)
                for d in T.Parallel(D):
                    LmkK[i_bn, i_h, d] = O_shared[0, d]
            else:
                for d in T.Parallel(D):
                    LmkK[i_bn, i_h, d] = 0.0

    return main


# =====================================================================
# Fallback Backward Kernel (for H < 16)
# =====================================================================

@tilelang.jit(
    out_idx=[6, 7],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def chunk_attn_pool_bwd_kernel_fallback(BN, chunk_size, heads, head_dim, sm_scale_val, threads=128, compact_lmk_k=False):
    """Fallback backward kernel for H < 16. Pads M to 16, grid = (BN, heads)."""
    dtype = "bfloat16"
    accum_dtype = "float"
    S = chunk_size
    D = head_dim
    M = 16
    LN2 = 0.6931471805599453

    q_shape = [BN, heads, head_dim]
    k_shape = [BN, chunk_size, heads, head_dim]
    p_shape = [BN, chunk_size, heads]
    b_shape = [BN, heads]
    grad_k_shape = [BN, heads, head_dim]
    grad_b_shape = [BN, heads]
    d_mu_q_shape = [BN, heads, head_dim]
    d_k_shape = [BN, chunk_size, heads, head_dim]

    @T.prim_func
    def main(
        MuQ: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        P_saved: T.Tensor(p_shape, accum_dtype),
        LmkB_saved: T.Tensor(b_shape, accum_dtype),
        GradLmkK: T.Tensor(grad_k_shape, dtype),
        GradLmkB: T.Tensor(grad_b_shape, accum_dtype),
        DMuQ: T.Tensor(d_mu_q_shape, accum_dtype),
        DK: T.Tensor(d_k_shape, accum_dtype),
    ):
        with T.Kernel(BN, heads, threads=threads) as (i_bn, i_h):
            K_shared = T.alloc_shared([S, D], dtype)
            GradK_shared = T.alloc_shared([M, D], dtype)
            MuQ_shared = T.alloc_shared([M, D], dtype)
            DLogits_shared = T.alloc_shared([M, S], dtype)
            O_shared = T.alloc_shared([M, D], dtype)
            P_row_shared = T.alloc_shared([S], accum_dtype)
            dP_row_shared = T.alloc_shared([S], accum_dtype)
            dLogits_row_shared = T.alloc_shared([S], accum_dtype)

            dP_frag = T.alloc_fragment([M, S], accum_dtype)
            d_mu_q_frag = T.alloc_fragment([M, D], accum_dtype)

            for s, d in T.Parallel(S, D):
                K_shared[s, d] = K[i_bn, s, i_h, d]
            if not compact_lmk_k:
                for m, d in T.Parallel(M, D):
                    GradK_shared[m, d] = GradLmkK[i_bn, i_h, d]
            else:
                for m, d in T.Parallel(M, D):
                    GradK_shared[m, d] = 0.0
            for m, d in T.Parallel(M, D):
                MuQ_shared[m, d] = MuQ[i_bn, i_h, d]

            T.clear(dP_frag)
            if not compact_lmk_k:
                T.gemm(GradK_shared, K_shared, dP_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

            dP_shared_buf = T.alloc_shared([M, S], accum_dtype)
            T.copy(dP_frag, dP_shared_buf)
            T.sync_threads()

            for j in T.Parallel(S):
                P_row_shared[j] = P_saved[i_bn, j, i_h]
                dP_row_shared[j] = dP_shared_buf[0, j]
            T.sync_threads()

            entropy_val_local = T.alloc_local([1], accum_dtype)
            grad_b_local = T.alloc_local([1], accum_dtype)
            delta_local = T.alloc_local([1], accum_dtype)
            entropy_val_local[0] = LmkB_saved[i_bn, i_h]
            grad_b_local[0] = GradLmkB[i_bn, i_h]

            delta_local[0] = 0.0
            for s in T.serial(S):
                delta_local[0] = delta_local[0] + P_row_shared[s] * dP_row_shared[s]

            for s in T.serial(S):
                if P_row_shared[s] > 0.0:
                    log_p_s = T.log2(P_row_shared[s]) * LN2
                    dLogits_row_shared[s] = P_row_shared[s] * (
                        (dP_row_shared[s] - delta_local[0]) - grad_b_local[0] * (log_p_s + entropy_val_local[0])
                    )
                else:
                    dLogits_row_shared[s] = 0.0
            T.sync_threads()

            for m, s in T.Parallel(M, S):
                DLogits_shared[m, s] = T.cast(dLogits_row_shared[s], dtype)

            T.clear(d_mu_q_frag)
            T.gemm(DLogits_shared, K_shared, d_mu_q_frag, policy=T.GemmWarpPolicy.FullRow)

            T.copy(d_mu_q_frag, O_shared)
            for d in T.Parallel(D):
                DMuQ[i_bn, i_h, d] = T.cast(O_shared[0, d], accum_dtype) * sm_scale_val

            if not compact_lmk_k:
                for s, d in T.Parallel(S, D):
                    DK[i_bn, s, i_h, d] = (
                        sm_scale_val * T.cast(dLogits_row_shared[s], accum_dtype) * T.cast(MuQ_shared[0, d], accum_dtype)
                        + P_row_shared[s] * T.cast(GradK_shared[0, d], accum_dtype)
                    )
            else:
                for s, d in T.Parallel(S, D):
                    DK[i_bn, s, i_h, d] = (
                        sm_scale_val * T.cast(dLogits_row_shared[s], accum_dtype) * T.cast(MuQ_shared[0, d], accum_dtype)
                    )

    return main


# =====================================================================
# Autograd Wrapper (dual path dispatch)
# =====================================================================

class ChunkAttnPoolFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu_q, k_chunked, sm_scale, compact_lmk_k=False):
        B, N, H, D = mu_q.shape
        S = k_chunked.shape[2]
        BN = B * N

        mu_q_flat = mu_q.reshape(BN, H, D).contiguous()
        k_flat = k_chunked.reshape(BN, S, H, D).contiguous()
        compact_lmk_k = bool(compact_lmk_k)

        # Dispatch: use fast v8 kernel when H >= 16 and H % 16 == 0
        # AND k_chunked has shared K across heads (all heads same K).
        # For safety, always use fallback which works for any K layout.
        # v8 (multi-head) can be enabled explicitly via chunk_attn_pool_tilelang_shared_k().
        use_multi_head = False

        if use_multi_head:
            fwd = chunk_attn_pool_fwd_kernel(BN, S, H, D, sm_scale, compact_lmk_k=compact_lmk_k)
        else:
            fwd = chunk_attn_pool_fwd_kernel_fallback(BN, S, H, D, sm_scale, compact_lmk_k=compact_lmk_k)

        lmk_k, lmk_b, p_saved = fwd(mu_q_flat, k_flat)

        ctx.save_for_backward(mu_q_flat, k_flat, p_saved, lmk_b)
        ctx.sm_scale = sm_scale
        ctx.orig_shape = (B, N, H, D, S)
        ctx.use_multi_head = use_multi_head
        ctx.compact_lmk_k = compact_lmk_k

        return lmk_k.reshape(B, N, H, D), lmk_b.reshape(B, N, H)

    @staticmethod
    def backward(ctx, grad_lmk_k, grad_lmk_b):
        mu_q_flat, k_flat, p_saved, lmk_b = ctx.saved_tensors
        B, N, H, D, S = ctx.orig_shape
        BN = B * N

        if grad_lmk_k is None or ctx.compact_lmk_k:
            grad_lmk_k = torch.zeros((BN, H, D), dtype=mu_q_flat.dtype, device=mu_q_flat.device)
        else:
            grad_lmk_k = grad_lmk_k.reshape(BN, H, D).contiguous()
        if grad_lmk_b is None:
            grad_lmk_b = torch.zeros((BN, H), dtype=lmk_b.dtype, device=lmk_b.device)
        else:
            grad_lmk_b = grad_lmk_b.reshape(BN, H).contiguous()

        if ctx.use_multi_head:
            bwd = chunk_attn_pool_bwd_kernel(BN, S, H, D, ctx.sm_scale, compact_lmk_k=ctx.compact_lmk_k)
        else:
            bwd = chunk_attn_pool_bwd_kernel_fallback(BN, S, H, D, ctx.sm_scale, compact_lmk_k=ctx.compact_lmk_k)

        d_mu_q, d_k = bwd(
            mu_q_flat, k_flat, p_saved, lmk_b,
            grad_lmk_k, grad_lmk_b
        )

        return (
            d_mu_q.to(mu_q_flat.dtype).reshape(B, N, H, D),
            d_k.to(k_flat.dtype).reshape(B, N, S, H, D),
            None,
            None,
        )


def chunk_attn_pool_tilelang(
    mu_q: torch.Tensor,
    k_chunked: torch.Tensor,
    sm_scale: float = None,
    compact_lmk_k: bool = False,
):
    """
    Drop-in replacement for chunk_attn_pool using TileLang kernel.

    Args:
        mu_q:      (B, N, H, D) bfloat16
        k_chunked: (B, N, S, H, D) bfloat16
        sm_scale:  float (default 1/sqrt(D))
        compact_lmk_k: if True, compute only entropy bias and skip P @ K.

    Returns:
        lmk_k: (B, N, H, D) bfloat16
        lmk_b: (B, N, H) float32
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(mu_q.shape[-1])
    return ChunkAttnPoolFunction.apply(mu_q, k_chunked, sm_scale, compact_lmk_k)


def chunk_attn_pool_tilelang_shared_k(
    mu_q: torch.Tensor,
    k_chunked: torch.Tensor,
    sm_scale: float = None,
):
    """
    Optimized version for shared-K scenario (all H heads have identical K).
    Uses v8 kernel with H as GEMM M-dimension for zero-waste tensor core utilization.
    Requires: H >= 16 and H % 16 == 0.

    Args:
        mu_q:      (B, N, H, D) bfloat16
        k_chunked: (B, N, S, H, D) bfloat16 - K MUST be identical across H dimension
        sm_scale:  float (default 1/sqrt(D))
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(mu_q.shape[-1])
    H = mu_q.shape[2]
    assert H >= 16 and H % 16 == 0, f"shared_k variant requires H>=16 and H%16==0, got H={H}"
    return ChunkAttnPoolSharedKFunction.apply(mu_q, k_chunked, sm_scale)


class ChunkAttnPoolSharedKFunction(torch.autograd.Function):
    """Autograd for shared-K v8 path."""
    @staticmethod
    def forward(ctx, mu_q, k_chunked, sm_scale):
        B, N, H, D = mu_q.shape
        S = k_chunked.shape[2]
        BN = B * N
        mu_q_flat = mu_q.reshape(BN, H, D).contiguous()
        k_flat = k_chunked.reshape(BN, S, H, D).contiguous()

        fwd = chunk_attn_pool_fwd_kernel(BN, S, H, D, sm_scale)
        lmk_k, lmk_b, p_saved = fwd(mu_q_flat, k_flat)

        ctx.save_for_backward(mu_q_flat, k_flat, p_saved, lmk_b)
        ctx.sm_scale = sm_scale
        ctx.orig_shape = (B, N, H, D, S)
        return lmk_k.reshape(B, N, H, D), lmk_b.reshape(B, N, H)

    @staticmethod
    def backward(ctx, grad_lmk_k, grad_lmk_b):
        mu_q_flat, k_flat, p_saved, lmk_b = ctx.saved_tensors
        B, N, H, D, S = ctx.orig_shape
        BN = B * N
        grad_lmk_k = grad_lmk_k.reshape(BN, H, D).contiguous()
        grad_lmk_b = grad_lmk_b.reshape(BN, H).contiguous()

        bwd = chunk_attn_pool_bwd_kernel(BN, S, H, D, ctx.sm_scale)
        d_mu_q, d_k = bwd(
            mu_q_flat, k_flat, p_saved, lmk_b,
            grad_lmk_k, grad_lmk_b
        )
        return (
            d_mu_q.to(mu_q_flat.dtype).reshape(B, N, H, D),
            d_k.to(k_flat.dtype).reshape(B, N, S, H, D),
            None,
        )


# =====================================================================
# PyTorch Reference (for testing)
# =====================================================================

def chunk_attn_pool_ref(mu_q, k_chunked, sm_scale=None):
    """Pure PyTorch reference (from lhsa_layer.py)."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(mu_q.shape[-1])

    import torch.nn.functional as F

    in_dtype = k_chunked.dtype
    mu_f32 = mu_q.float()
    k_f32 = k_chunked.float()

    logits = torch.einsum("bnhd,bnshd->bnsh", mu_f32, k_f32) * sm_scale
    S_chunk = logits.shape[2]
    last_mask = torch.zeros(S_chunk, dtype=torch.bool, device=logits.device)
    last_mask[-1] = True
    logits = logits.masked_fill(last_mask.view(1, 1, S_chunk, 1), float('-inf'))

    p = F.softmax(logits, dim=2)
    lmk_k = torch.einsum("bnsh,bnshd->bnhd", p, k_f32)

    log_p = F.log_softmax(logits, dim=2)
    log_p_safe = torch.where(torch.isfinite(log_p), log_p, log_p.new_zeros(()))
    lmk_b = -(p * log_p_safe).sum(dim=2)

    return lmk_k.to(in_dtype), lmk_b
