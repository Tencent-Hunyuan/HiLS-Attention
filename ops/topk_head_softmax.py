"""
Top-K Chunk Selection
==============================
TileLang implementation of the HiLS-Attention chunk selection kernel.

Author: Xinyu Wei
"""

import torch
import tilelang
import tilelang.language as T
from typing import Optional
import math

def ref_softmax_topk_max_pooling(q, k_lmks, lse_swa, topk, block_size, window_size, is_causal=False, q_offset=0, drop_mask=None, bias=None, gumbel_noise=None):
    """
    Reference implementation for Softmax-then-Max Top-K strategy.

    Args:
        q: [B, L, h_kv, G, D]
        k_lmks: [B, S, h_kv, D]  (per-KV-head, GQA-shared lmks)
                or [B, S, h_kv * G, D]  (per-q-head lmks; topk still done on h_kv)
        lse_swa: [B, L, h_q] or [B, L, h_kv, G]
        topk: int
        block_size: int
        window_size: int
        is_causal: bool
        q_offset: int (for causal masking when q does not start from 0)
        drop_mask: [B, L, S] int32 tensor, 1  drop  chunk
        bias: optional per-chunk additive bias. Supported shapes are
              [B, S, h_q] or [B, S, h_kv, G]. When provided, selection logits
              and HiLS LSE add bias[b, chunk, head]. Returned scores stay raw
              scaled qk and do not include bias or Gumbel noise.
        gumbel_noise: optional [1, 1, h_q, S] tensor. When provided, it is
              added to per-q-head selection logits and HiLS LSE logits. Returned
              scores do not include Gumbel noise.

    Returns:
        indices_sorted: [B, L, h_kv, topk]
        scores_sorted: [B, L, h_kv, G, topk] (raw scaled qk, without bias or Gumbel noise)
    """
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    lmks_h = k_lmks.shape[2]
    if lmks_h == h_kv:
        per_qhead_lmks = False
    elif lmks_h == h_kv * G:
        per_qhead_lmks = True
    else:
        raise AssertionError(
            f"k_lmks h dim ({lmks_h}) must be either h_kv ({h_kv}) or h_kv*G ({h_kv * G})"
        )

    if per_qhead_lmks:
        # k_lmks: [B, S, h_kv, G, D]
        k_lmks_v = k_lmks.view(B, S, h_kv, G, D)
        logits_hils = torch.einsum("blhgd,bshgd->blhgs", q.float(), k_lmks_v.float())
    else:
        logits_hils = torch.einsum("blhgd,bshd->blhgs", q.float(), k_lmks.float())

    sm_scale = 1.0 / math.sqrt(D)
    logits_hils_scaled = logits_hils * sm_scale

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1)
        i_idx_global = i_idx + q_offset
        j_idx = torch.arange(S, device=q.device).unsqueeze(0)

        if window_size > 0:
            threshold_idx = (i_idx_global - window_size + 1).div(block_size, rounding_mode='floor')
        else:
            threshold_idx = i_idx_global.div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= threshold_idx

        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        logits_hils_scaled = logits_hils_scaled.masked_fill(causal_mask_expanded, float('-inf'))

    if bias is not None:
        bias = bias.to(device=q.device, dtype=torch.float32)
        if bias.dim() == 3:
            assert bias.shape == (B, S, h_kv * G), (
                f"bias shape {tuple(bias.shape)} != ({B}, {S}, {h_kv * G})"
            )
            bias_view = bias.reshape(B, S, h_kv, G).permute(0, 2, 3, 1).unsqueeze(1)
        elif bias.dim() == 4:
            assert bias.shape == (B, S, h_kv, G), (
                f"bias shape {tuple(bias.shape)} != ({B}, {S}, {h_kv}, {G})"
            )
            bias_view = bias.permute(0, 2, 3, 1).unsqueeze(1)
        else:
            raise AssertionError(f"bias must be [B, S, h_q] or [B, S, h_kv, G], got {tuple(bias.shape)}")
        logits_hils_for_select = logits_hils_scaled + bias_view
    else:
        bias_view = None
        logits_hils_for_select = logits_hils_scaled

    if gumbel_noise is not None:
        gumbel_noise = gumbel_noise.detach().to(device=q.device, dtype=torch.float32)
        assert gumbel_noise.shape == (1, 1, h_kv * G, S), (
            f"gumbel_noise shape {tuple(gumbel_noise.shape)} != ({1}, {1}, {h_kv * G}, {S})"
        )
        gumbel_view = gumbel_noise.view(1, 1, h_kv, G, S)
        logits_hils_for_select = logits_hils_for_select + gumbel_view

    lse_hils = torch.logsumexp(logits_hils_for_select, dim=-1)

    if lse_swa.dim() == 3:
        lse_swa_view = lse_swa.view(B, L, h_kv, G)
    else:
        lse_swa_view = lse_swa

    lse_total = torch.logaddexp(lse_swa_view, lse_hils)

    log_probs = logits_hils_for_select - lse_total.unsqueeze(-1)

    scores_max_pooling = log_probs.max(dim=3).values

    if drop_mask is not None:
        drop_bool = drop_mask.bool()  # [B, L, S]
        scores_max_pooling = scores_max_pooling.masked_fill(drop_bool.unsqueeze(2), float('-inf'))

    actual_topk = min(topk, S)

    topk_scores, topk_indices = torch.topk(scores_max_pooling, k=actual_topk, dim=-1, sorted=False)

    topk_indices[topk_scores == float('-inf')] = -1

    if actual_topk < topk:
        pad_size = topk - actual_topk
        pad_indices = torch.full(
            (B, L, h_kv, pad_size), -1,
            dtype=topk_indices.dtype, device=topk_indices.device
        )
        topk_indices = torch.cat([topk_indices, pad_indices], dim=-1)

    indices_sorted, order = torch.sort(topk_indices, dim=-1)

    sort_temp = topk_indices.clone()
    sort_temp[sort_temp < 0] = S + 1000
    indices_sorted, order = torch.sort(sort_temp, dim=-1)
    indices_sorted[indices_sorted >= S] = -1

    order_expanded = order.unsqueeze(3).expand(-1, -1, -1, G, -1)

    safe_indices_sorted = indices_sorted.clone()
    safe_indices_sorted[safe_indices_sorted < 0] = 0
    indices_expanded = safe_indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1)

    scores_sorted = torch.gather(logits_hils_scaled, -1, indices_expanded)

    invalid_mask = indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1) < 0
    scores_sorted = scores_sorted.masked_fill(invalid_mask, float('-inf'))

    return indices_sorted, scores_sorted




# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [2,4,8,16]
# BLOCK_S = [16,32,64]
# threads = [64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_S,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_S": c[1],
#         "threads": c[2],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hils_lse_kernel(
    batch, h_kv, groups, head_dim, block_size, window_size,
    is_causal, is_training=True, seq_len=None, s_len=None,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    sm_scale=None,
    use_bias=False,
    use_gumbel=False,
    per_qhead_lmks=False,
):
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
    else:
        seq_len_var = seq_len
        s_len_var = s_len

    dtype = "bfloat16"
    accum_dtype = "float"

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len_var, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len_var, h_kv, head_dim]

    lse_shape = [batch, seq_len_var, h_kv, groups]
    bias_shape = [batch, s_len_var, h_kv, groups] if use_bias else [1, 1, 1, 1]
    gumbel_shape = [1, 1, h_kv * groups, s_len_var] if use_gumbel else [1, 1, 1, 1]

    if BLOCK_L is None:
        BLOCK_L = 16 if per_qhead_lmks else (16 + groups - 1) // groups
    if BLOCK_S is None: BLOCK_S = 64
    if threads is None: threads = 128


    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_S
    GEMM_K = head_dim

    num_s_blocks = tilelang.cdiv(s_len_var, BLOCK_S)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LSE_Out: T.Tensor(lse_shape, accum_dtype),
        Q_Offset: T.Tensor([1], "int32"),
        bias: T.Tensor(bias_shape, accum_dtype),
        GumbelNoise: T.Tensor(gumbel_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
            i_b, i_h = bz, by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([GEMM_M, GEMM_K], dtype)
            K_shared = T.alloc_shared([GEMM_N, GEMM_K], dtype)

            score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)

            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)
            Q_g_shared = T.alloc_shared([BLOCK_L, GEMM_K], dtype)
            acc_s_g = T.alloc_fragment([BLOCK_L, GEMM_N], accum_dtype)

            m_curr = T.alloc_fragment([GEMM_M], accum_dtype)
            m_prev = T.alloc_fragment([GEMM_M], accum_dtype)
            l_prev = T.alloc_fragment([GEMM_M], accum_dtype)

            scores_max = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_sum = T.alloc_fragment([GEMM_M], accum_dtype)
            scores_scale = T.alloc_fragment([GEMM_M], accum_dtype)

            T.annotate_layout({Q_shared: tilelang.layout.make_swizzled_layout(Q_shared)})

            T.fill(m_prev, -T.infinity(accum_dtype))
            T.fill(l_prev, 0.0)

            for i, j in T.Parallel(GEMM_M, GEMM_K):
                tq = base_l + (i // groups)
                if tq < seq_len_var:
                    Q_shared[i, j] = Q[i_b, tq, i_h, i % groups, j]
                else:
                    Q_shared[i, j] = 0.0

            loop_limit_base = tilelang.cdiv(s_len_var, BLOCK_S)
            if is_causal:
                global_end = q_offset + base_l + BLOCK_L
                loop_limit = T.min(loop_limit_base, tilelang.cdiv(global_end, BLOCK_S))
            else:
                loop_limit = loop_limit_base

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                if per_qhead_lmks:
                    for g in T.serial(groups):
                        for s_idx, j in T.Parallel(GEMM_N, GEMM_K):
                            ts = base_s + s_idx
                            if ts < s_len_var:
                                K_shared[s_idx, j] = K[i_b, ts, i_h * groups + g, j]
                            else:
                                K_shared[s_idx, j] = 0.0
                        for l_idx, j in T.Parallel(BLOCK_L, GEMM_K):
                            Q_g_shared[l_idx, j] = Q_shared[l_idx * groups + g, j]
                        T.sync_threads()
                        T.clear(acc_s_g)
                        T.gemm(Q_g_shared, K_shared, acc_s_g, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        for l_idx, s_idx in T.Parallel(BLOCK_L, GEMM_N):
                            score_shared[l_idx * groups + g, s_idx] = acc_s_g[l_idx, s_idx]
                        T.sync_threads()
                else:
                    for i, j in T.Parallel(GEMM_N, GEMM_K):
                        ts = base_s + i
                        if ts < s_len_var:
                            K_shared[i, j] = K[i_b, ts, i_h, j]
                        else:
                            K_shared[i, j] = 0.0

                    T.sync_threads()

                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(acc_s, score_shared)

                for i, j in T.Parallel(GEMM_M, GEMM_N):
                    ts = base_s + j
                    if ts >= s_len_var:
                        score_shared[i, j] = -T.infinity(accum_dtype)
                    elif is_causal:
                        l_idx = i // groups
                        tq_local = base_l + l_idx
                        tq_global = q_offset + tq_local
                        if tq_local < seq_len_var:
                            if window_size > 0:
                                if ts >= (tq_global - window_size + 1) // block_size:
                                    score_shared[i, j] = -T.infinity(accum_dtype)
                            else:
                                if ts >= tq_global // block_size:
                                    score_shared[i, j] = -T.infinity(accum_dtype)

                T.sync_threads()
                T.copy(score_shared, acc_s)

                if use_bias or use_gumbel:
                    for i, j in T.Parallel(GEMM_M, GEMM_N):
                        tq = base_l + (i // groups)
                        ts = base_s + j
                        if (tq < seq_len_var) and (ts < s_len_var):
                            if acc_s[i, j] == -T.infinity(accum_dtype):
                                acc_s[i, j] = -T.infinity(accum_dtype)
                            else:
                                acc_s[i, j] = acc_s[i, j] * sm_scale
                                if use_bias:
                                    acc_s[i, j] += bias[i_b, ts, i_h, i % groups]
                                if use_gumbel:
                                    acc_s[i, j] += GumbelNoise[0, 0, i_h * groups + (i % groups), ts]
                        else:
                            acc_s[i, j] = -T.infinity(accum_dtype)

                T.copy(m_prev, m_curr)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)

                for i in T.Parallel(GEMM_M):
                    if use_bias or use_gumbel:
                        m_prev[i] = T.max(m_prev[i], scores_max[i])
                    else:
                        scores_max[i] = scores_max[i] * sm_scale
                        m_prev[i] = T.max(m_prev[i], scores_max[i])

                for i in T.Parallel(GEMM_M):
                    if m_prev[i] == -T.infinity(accum_dtype):
                        scores_scale[i] = 1.0
                    else:
                        scores_scale[i] = T.exp(m_curr[i] - m_prev[i])

                for i, j in T.Parallel(GEMM_M, GEMM_N):
                    ts = base_s + j
                    if ts < s_len_var:
                        if use_bias or use_gumbel:
                            val = acc_s[i, j]
                        else:
                            val = acc_s[i, j] * sm_scale
                        if val == -T.infinity(accum_dtype) and m_prev[i] == -T.infinity(accum_dtype):
                            acc_s[i, j] = 0.0
                        else:
                            acc_s[i, j] = T.exp(val - m_prev[i])
                    else:
                        acc_s[i, j] = 0.0

                T.reduce_sum(acc_s, scores_sum, dim=1)

                for i in T.Parallel(GEMM_M):
                    l_prev[i] = l_prev[i] * scores_scale[i] + scores_sum[i]

                T.sync_threads()

            for i in T.Parallel(GEMM_M):
                tq = base_l + (i // groups)
                if tq < seq_len_var:
                    if l_prev[i] == 0:
                         LSE_Out[i_b, tq, i_h, i % groups] = -T.infinity(accum_dtype)
                    else:
                         LSE_Out[i_b, tq, i_h, i % groups] = m_prev[i] + T.log(l_prev[i])

    return kernel




# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [2,4,8,16]
# BLOCK_S = [16,32,64]
# threads = [64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_S,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_S": c[1],
#         "threads": c[2],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def weighted_select_kernel(
    batch, h_kv, groups, head_dim, topk, block_size, window_size,
    is_causal, is_training=True, seq_len=None, s_len=None,
    use_drop_mask=False,
    BLOCK_L=None, BLOCK_S=None, threads=None,
    sm_scale=None,
    use_bias=False,
    use_gumbel=False,
    per_qhead_lmks=False,
):
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
    else:
        seq_len_var = seq_len
        s_len_var = s_len

    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len_var, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len_var, h_kv, head_dim]
    lse_shape = [batch, seq_len_var, h_kv, groups]
    bias_shape = [batch, s_len_var, h_kv, groups] if use_bias else [1, 1, 1, 1]
    gumbel_shape = [1, 1, h_kv * groups, s_len_var] if use_gumbel else [1, 1, 1, 1]

    out_indices_shape = [batch, seq_len_var, h_kv, topk]
    drop_mask_shape = [batch, seq_len_var, s_len_var] if use_drop_mask else [1, 1, 1]
    if BLOCK_L is None:
        BLOCK_L = 16 if per_qhead_lmks else (16 + groups - 1) // groups
    if BLOCK_S is None: BLOCK_S = 16
    if threads is None: threads = 64

    GEMM_M = BLOCK_L * groups
    num_s_blocks = tilelang.cdiv(s_len_var, BLOCK_S)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        LSE_Total: T.Tensor(lse_shape, accum_dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
        Q_Offset: T.Tensor([1], "int32"),
        DropMask: T.Tensor(drop_mask_shape, idx_dtype),
        bias: T.Tensor(bias_shape, accum_dtype),
        GumbelNoise: T.Tensor(gumbel_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            q_offset = T.if_then_else(is_training, 0, Q_Offset[0])
            i_b, i_h = bz, by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([GEMM_M, head_dim], dtype)
            K_shared = T.alloc_shared([BLOCK_S, head_dim], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)
            Q_g_shared = T.alloc_shared([BLOCK_L, head_dim], dtype)
            acc_s_g = T.alloc_fragment([BLOCK_L, BLOCK_S], accum_dtype)

            topk_max_scores = T.alloc_local([topk], accum_dtype)
            topk_indices = T.alloc_local([topk], idx_dtype)

            lse_local = T.alloc_local([groups], accum_dtype)

            T.fill(topk_max_scores, -T.infinity(accum_dtype))
            T.fill(topk_indices, -1)

            tx = T.get_thread_binding()
            if tx < BLOCK_L and (base_l + tx) < seq_len_var:
                for g in T.serial(groups):
                    lse_local[g] = LSE_Total[i_b, base_l + tx, i_h, g]

            for l_idx, g, d in T.Parallel(BLOCK_L, groups, head_dim):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len_var:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = 0

            loop_limit_base = num_s_blocks
            if is_causal:
                global_end = q_offset + base_l + BLOCK_L
                loop_limit = T.min(loop_limit_base, tilelang.cdiv(global_end, BLOCK_S))
            else:
                loop_limit = loop_limit_base
            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                if per_qhead_lmks:
                    for g in T.serial(groups):
                        for s_idx, d in T.Parallel(BLOCK_S, head_dim):
                            ts = base_s + s_idx
                            if ts < s_len_var:
                                K_shared[s_idx, d] = K[i_b, ts, i_h * groups + g, d]
                            else:
                                K_shared[s_idx, d] = 0
                        for l_idx, d in T.Parallel(BLOCK_L, head_dim):
                            Q_g_shared[l_idx, d] = Q_shared[l_idx * groups + g, d]
                        T.sync_threads()
                        T.clear(acc_s_g)
                        T.gemm(Q_g_shared, K_shared, acc_s_g, transpose_B=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        for l_idx, s_idx in T.Parallel(BLOCK_L, BLOCK_S):
                            score_shared[l_idx * groups + g, s_idx] = acc_s_g[l_idx, s_idx]
                        T.sync_threads()
                else:
                    for s_idx, d in T.Parallel(BLOCK_S, head_dim):
                        ts = base_s + s_idx
                        if ts < s_len_var:
                            K_shared[s_idx, d] = K[i_b, ts, i_h, d]
                        else:
                            K_shared[s_idx, d] = 0
                    T.sync_threads()

                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(acc_s, score_shared)
                if is_causal:
                    for i, j in T.Parallel(GEMM_M, BLOCK_S):
                        l_idx = i // groups
                        tq_local = base_l + l_idx
                        tq_global = q_offset + tq_local
                        ts = base_s + j
                        if window_size > 0:
                            if ts >= (tq_global - window_size + 1) // block_size:
                                score_shared[i, j] = -T.infinity(accum_dtype)
                        else:
                            if ts >= tq_global // block_size:
                                score_shared[i, j] = -T.infinity(accum_dtype)
                T.sync_threads()

                if tx < BLOCK_L and (base_l + tx) < seq_len_var:
                    my_l_idx = tx
                    tq = base_l + my_l_idx
                    tq_global = q_offset + tq
                    limit_chunk = T.alloc_var(idx_dtype)
                    if window_size > 0:
                        limit_chunk = (tq_global - window_size + 1) // block_size
                    else:
                        limit_chunk = tq_global // block_size
                    val = T.alloc_var(accum_dtype)
                    norm_score = T.alloc_var(accum_dtype)
                    cur_max_norm_score = T.alloc_var(accum_dtype)
                    is_valid = T.alloc_var("bool")
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        in_range = T.alloc_var("bool")
                        in_range = (ts < s_len_var)
                        if in_range:
                            is_valid = (not is_causal) or (ts < limit_chunk)
                            if use_drop_mask:
                                is_valid = is_valid and (DropMask[i_b, tq, ts] == 0)
                            if is_valid:
                                cur_max_norm_score = -T.infinity(accum_dtype)

                                for g in T.serial(groups):
                                    val = score_shared[my_l_idx * groups + g, s_idx] * sm_scale
                                    if use_bias:
                                        val += bias[i_b, ts, i_h, g]
                                    if use_gumbel:
                                        val += GumbelNoise[0, 0, i_h * groups + g, ts]
                                    if val == -T.infinity(accum_dtype):
                                        norm_score = -T.infinity(accum_dtype)
                                    else:
                                        norm_score = val - lse_local[g]
                                    cur_max_norm_score = T.max(cur_max_norm_score, norm_score)

                                if cur_max_norm_score > topk_max_scores[topk - 1]:
                                    moving = T.alloc_var("bool")
                                    moving = True
                                    for kk in T.serial(topk):
                                        k = topk - 1 - kk
                                        if moving:
                                            if (k > 0) and (cur_max_norm_score > topk_max_scores[k - 1]):
                                                topk_max_scores[k] = topk_max_scores[k - 1]
                                                topk_indices[k] = topk_indices[k - 1]
                                            else:
                                                topk_max_scores[k] = cur_max_norm_score
                                                topk_indices[k] = ts
                                                moving = False
                T.sync_threads()

            if tx < BLOCK_L and (base_l + tx) < seq_len_var:
                for k in T.serial(topk):
                    OutIndices[i_b, base_l + tx, i_h, k] = topk_indices[k]

    return kernel



# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [2,4,8]
# BLOCK_TK = [16,32,64]
# threads = [64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         BLOCK_TK,
#         threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "BLOCK_TK": c[1],
#         "threads": c[2],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
# ...existing code...
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def recompute_topk_max_pooling_scores_kernel(
    batch,  h_kv, groups, head_dim, topk, seq_len=None, s_len=None, is_training=True,
    BLOCK_L=None, BLOCK_TK=None, threads=None,
    sm_scale=None,
    per_qhead_lmks=False,
):
    """
    of topk indices  scores.
    Q:        [B, L, h_kv, G, D]
    K:        [B, S, h_kv, D]   (per-KV-head)  or  [B, S, h_kv*G, D] (per-q-head)
    Indices:  [B, L, h_kv, topk]
    OutScores:[B, L, h_kv, G, topk]
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    if not is_training:
        seq_len_var = T.dynamic("seq_len")
        s_len_var = T.dynamic("s_len")
    else:
        seq_len_var = seq_len
        s_len_var = s_len

    q_shape = [batch, seq_len_var, h_kv, groups, head_dim]
    if per_qhead_lmks:
        k_shape = [batch, s_len_var, h_kv * groups, head_dim]
    else:
        k_shape = [batch, s_len_var, h_kv, head_dim]
    indices_shape = [batch, seq_len_var, h_kv, topk]
    out_scores_shape = [batch, seq_len_var, h_kv, groups, topk]

    if BLOCK_L is None:
        BLOCK_L = 16 if per_qhead_lmks else (16 + groups - 1) // groups
    if BLOCK_TK is None:
        BLOCK_TK = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = 64

    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_L * BLOCK_TK
    tk_blocks = (topk + BLOCK_TK - 1) // BLOCK_TK

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    @T.prim_func
    def fwd_recompute(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(indices_shape, idx_dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L
            K_shared = T.alloc_shared([BLOCK_L * BLOCK_TK, BLOCK_D], dtype)

            if per_qhead_lmks:
                Q_g_shared = T.alloc_shared([BLOCK_L, BLOCK_D], dtype)
                acc_s_g = T.alloc_fragment([BLOCK_L, GEMM_N], accum_dtype)

                for tk_block in T.serial(tk_blocks):
                    tk_base = tk_block * BLOCK_TK
                    tk_size = T.min(BLOCK_TK, topk - tk_base)

                    for g in T.serial(groups):
                        for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                            tq = base_l + l_idx
                            off = l_idx * BLOCK_TK + tk_idx
                            if (tq < seq_len_var) and (tk_idx < tk_size):
                                k_id = tk_base + tk_idx
                                idx = Indices[i_b, tq, i_h, k_id]
                                if (idx >= 0) and (idx < s_len_var):
                                    K_shared[off, d] = K[i_b, idx, i_h * groups + g, d]
                                else:
                                    K_shared[off, d] = T.Cast(dtype, 0.0)
                            else:
                                if off < BLOCK_L * BLOCK_TK:
                                    K_shared[off, d] = T.Cast(dtype, 0.0)
                        for l_idx, d in T.Parallel(BLOCK_L, BLOCK_D):
                            tq = base_l + l_idx
                            if tq < seq_len_var:
                                Q_g_shared[l_idx, d] = Q[i_b, tq, i_h, g, d]
                            else:
                                Q_g_shared[l_idx, d] = T.Cast(dtype, 0.0)
                        T.sync_threads()
                        T.clear(acc_s_g)
                        T.gemm(
                            Q_g_shared,
                            K_shared,
                            acc_s_g,
                            transpose_B=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for l_idx, tk_idx in T.Parallel(BLOCK_L, BLOCK_TK):
                            tq = base_l + l_idx
                            if (tq < seq_len_var) and (tk_idx < tk_size):
                                k_id = tk_base + tk_idx
                                idx = Indices[i_b, tq, i_h, k_id]
                                if idx < 0:
                                    OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                                else:
                                    col = l_idx * BLOCK_TK + tk_idx
                                    OutScores[i_b, tq, i_h, g, k_id] = acc_s_g[l_idx, col] * sm_scale
                        T.sync_threads()
            else:
                Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
                score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
                acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)

                for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                    tq = base_l + l_idx
                    flat_m = l_idx * groups + g
                    if tq < seq_len_var:
                        Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                    else:
                        Q_shared[flat_m, d] = T.Cast(dtype, 0.0)

                for tk_block in T.serial(tk_blocks):
                    tk_base = tk_block * BLOCK_TK
                    tk_size = T.min(BLOCK_TK, topk - tk_base)

                    for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                        tq = base_l + l_idx
                        off = l_idx * BLOCK_TK + tk_idx
                        if (tq < seq_len_var) and (tk_idx < tk_size):
                            k_id = tk_base + tk_idx
                            idx = Indices[i_b, tq, i_h, k_id]
                            if (idx >= 0) and (idx < s_len_var):
                                K_shared[off, d] = K[i_b, idx, i_h, d]
                            else:
                                K_shared[off, d] = T.Cast(dtype, 0.0)
                        else:
                            if off < BLOCK_L * BLOCK_TK:
                                K_shared[off, d] = T.Cast(dtype, 0.0)
                    T.sync_threads()

                    T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow
                    )
                    T.copy(acc_s, score_shared)
                    T.sync_threads()

                    for l_idx, g, tk_idx in T.Parallel(BLOCK_L, groups, BLOCK_TK):
                        tq = base_l + l_idx
                        if (tq < seq_len_var) and (tk_idx < tk_size):
                            k_id = tk_base + tk_idx
                            idx = Indices[i_b, tq, i_h, k_id]
                            if idx < 0:
                                OutScores[i_b, tq, i_h, g, k_id] = -T.infinity(accum_dtype)
                            else:
                                row = l_idx * groups + g
                                col = l_idx * BLOCK_TK + tk_idx
                                val = score_shared[row, col]
                                OutScores[i_b, tq, i_h, g, k_id] = val * sm_scale

    return fwd_recompute







# from tilelang.autotuner import autotune
# import itertools
# BLOCK_L = [1,2,4,8,16,32]
# num_threads = [32,64,128,256]
# _configs = list(
#     itertools.product(
#         BLOCK_L,
#         num_threads,
#     ))

# configs = [
#     {
#         "BLOCK_L": c[0],
#         "num_threads": c[1],
#     } for c in _configs
# ]

# @autotune(
#     configs=configs,
#     warmup=5,
#     rep=10,
# )
@tilelang.jit(
    out_idx=[1],
    pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
},
)
def sort_topk_indices_kernel(
    batch: int,

    h_kv: int,
    topk: int,
    BLOCK_L: int = 16,
    num_threads: int = 64,
    seq_len=None,
    is_training: bool = True,
):
    """
     IndicesIn: [B, L, h_kv, topk]  bitonic ( chunk id).
    - with: 0 <= idx < s_len
    - without: idx < 0( -1), is key = s_len, in.
    """
    if not is_training:
        seq_len_var = T.dynamic("seq_len")
    else:
        seq_len_var = seq_len
    INVALID_KEY = 0x7FFFFFFF
    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)
    num_iters = int(round(math.log2(topk)))

    indices_shape = [batch, seq_len_var, h_kv, topk]

    @T.prim_func
    def sort_kernel(
        IndicesIn: T.Tensor(indices_shape, INT32),
        IndicesOut: T.Tensor(indices_shape, INT32),
    ):

        with T.Kernel(tilelang.cdiv(seq_len_var, BLOCK_L), h_kv, batch, threads=num_threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            idx_shared = T.alloc_shared([BLOCK_L, topk], dtype=INT32)

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len_var:
                    idx_shared[l_idx, k] = IndicesIn[i_b, lq, i_h, k]
                else:
                    idx_shared[l_idx, k] = -1
            T.sync_threads()

            k_step = T.alloc_var(INT32)
            j_step = T.alloc_var(INT32)
            val_i = T.alloc_var(INT32)
            val_j = T.alloc_var(INT32)
            key_i = T.alloc_var(INT32)
            key_j = T.alloc_var(INT32)

            k_step = 2
            for _ in T.serial(num_iters):
                j_step = k_step // 2
                while j_step > 0:
                    for l_idx, i in T.Parallel(BLOCK_L, topk):
                        lq = base_l + l_idx
                        if lq < seq_len_var:
                            i_idx = i
                            ixj = i_idx ^ j_step
                            if (ixj > i_idx) and (ixj < topk):
                                val_i = idx_shared[l_idx, i_idx]
                                val_j = idx_shared[l_idx, ixj]

                                if val_i >= 0:
                                    key_i = val_i
                                else:
                                    key_i = INVALID_KEY
                                if val_j >= 0:
                                    key_j = val_j
                                else:
                                    key_j = INVALID_KEY

                                up = (i_idx & k_step) == 0

                                do_swap = T.alloc_var("bool")
                                do_swap = False
                                if up:
                                    if key_i > key_j:
                                        do_swap = True
                                else:
                                    if key_i < key_j:
                                        do_swap = True

                                if do_swap:
                                    idx_shared[l_idx, i_idx] = val_j
                                    idx_shared[l_idx, ixj] = val_i
                    T.sync_threads()
                    j_step = j_step // 2
                k_step = k_step * 2

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len_var:
                    IndicesOut[i_b, lq, i_h, k] = idx_shared[l_idx, k]

    return sort_kernel



class SoftmaxTopKMaxPoolingFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, lmks, lse_swa,
                lse_kernel,
                select_kernel,
                sort_kernel,
                recompute_kernel,
                q_offset_tensor,
                drop_mask,
                sm_scale,
                bias,
                gumbel_noise,
                per_qhead_lmks,
                ):
        # lmks: [B, S, h_kv, D]      (per-KV-head)
        #    or [B, S, h_kv*G, D]    (per-q-head when per_qhead_lmks=True)
        # lse_swa: [B, L, h_q] or [B, L, h_kv, G]
        B, L, h_kv, G, D = q.shape
        B2, S, lmks_h, D2 = lmks.shape
        dtype = q.dtype

        assert B == B2 and D == D2
        if per_qhead_lmks:
            assert lmks_h == h_kv * G, (
                f"per_qhead_lmks expects lmks h dim = h_kv*G ({h_kv * G}), got {lmks_h}"
            )
        else:
            assert lmks_h == h_kv, (
                f"default mode expects lmks h dim = h_kv ({h_kv}), got {lmks_h}"
            )

        q_in = q.contiguous()
        k_in = lmks.contiguous()
        bias = bias.contiguous()
        gumbel_noise = gumbel_noise.contiguous()

        # lse_hils: [B, L, h_kv, G]
        lse_hils = lse_kernel(q_in, k_in, q_offset_tensor, bias, gumbel_noise)

        if lse_swa.dim() == 3: # [B, L, h_q]
            lse_swa_view = lse_swa.view(B, L, h_kv, G)
        else:
            lse_swa_view = lse_swa

        lse_total = torch.logaddexp(lse_swa_view, lse_hils)

        if drop_mask is None:
            drop_mask_in = torch.zeros(1, 1, 1, dtype=torch.int32, device=q.device)
        else:
            drop_mask_in = drop_mask
        indices_raw = select_kernel(q_in, k_in, lse_total, q_offset_tensor, drop_mask_in, bias, gumbel_noise)  # int32

        indices_sorted = sort_kernel(indices_raw)

        best_scores_buf = recompute_kernel(q_in, k_in, indices_sorted)  # float32

        ctx.save_for_backward(q_in, k_in, indices_sorted)
        ctx.h_kv = h_kv
        ctx.G = G
        ctx.shapes = (B, L, S, h_kv, D)
        ctx.sm_scale = sm_scale
        ctx.per_qhead_lmks = per_qhead_lmks

        return indices_sorted, best_scores_buf.to(dtype)

    @staticmethod
    def backward(ctx, grad_indices_unused, grad_scores_selected):
        q_in, k_in, indices = ctx.saved_tensors
        indices = indices.long()
        B, L, S, h_kv, D = ctx.shapes
        G = ctx.G
        per_qhead_lmks = ctx.per_qhead_lmks

        sm_scale = ctx.sm_scale if ctx.sm_scale is not None else 1.0 / math.sqrt(D)

        grad_scores_dense = torch.zeros(
            (B, h_kv, G, L, S),
            dtype=grad_scores_selected.dtype,
            device=grad_scores_selected.device,
        )

        indices_expanded = indices.permute(0, 2, 1, 3).unsqueeze(2).expand(-1, -1, G, -1, -1)
        valid_mask = (indices_expanded >= 0) & (indices_expanded < S)

        safe_indices = indices_expanded.clone()
        safe_indices[~valid_mask] = 0

        safe_grad = grad_scores_selected.permute(0, 2, 3, 1, 4).clone()
        safe_grad.mul_(sm_scale)
        safe_grad[~valid_mask] = 0

        grad_scores_dense.scatter_(4, safe_indices, safe_grad)

        bs_hg = B * h_kv * G

        dense_in = grad_scores_dense.reshape(bs_hg, L, S)

        q_flat = q_in.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, D)

        if per_qhead_lmks:
            k_view = k_in.view(B, S, h_kv, G, D)
            k_flat = k_view.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, D)
        else:
            k_expanded = k_in.unsqueeze(3).expand(-1, -1, -1, G, -1)
            k_flat = k_expanded.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, D)

        grad_q_flat = torch.bmm(dense_in, k_flat)
        grad_q = grad_q_flat.view(B, h_kv, G, L, D).permute(0, 3, 1, 2, 4)

        grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_flat)

        grad_k_grouped = grad_k_flat.view(B, h_kv, G, S, D)
        if per_qhead_lmks:
            grad_lmks = grad_k_grouped.permute(0, 3, 1, 2, 4).reshape(B, S, h_kv * G, D)
        else:
            grad_k_sum = grad_k_grouped.sum(dim=2)
            grad_lmks = grad_k_sum.permute(0, 2, 1, 3)  # [B, S, h_kv, D]
        return grad_q, grad_lmks, None, None, None, None, None, None, None, None, None, None, None


class SoftmaxTopKMaxPooling_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal, is_training=True, use_drop_mask=False, per_qhead_lmks=False):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self.is_training = is_training
        self.use_drop_mask = use_drop_mask
        self.per_qhead_lmks = per_qhead_lmks
        self._cached_lse_kernel = None
        self._cached_select_kernel = None
        self._cached_sort_kernel = None
        self._cached_recompute_kernel = None
        self._cached_shape = None

    def forward(self, q, lmks, lse_swa, q_offset, drop_mask=None, sm_scale=None, bias=None, use_gumbel=False, gumbel_noise=None):
        # q:    [B, L, h_kv, G, D]
        # lmks: [B, S, h_kv, D]   (per-KV-head)
        #    or [B, S, h_kv*G, D] (per-q-head when per_qhead_lmks=True)
        # lse_swa: [B, L, h_q]
        B, L, h_kv, G, D = q.shape
        _, S, _, _ = lmks.shape
        per_qhead_lmks = self.per_qhead_lmks
        topk = self.topk
        is_causal = self.is_causal
        block_size = self.block_size
        window_size = self.window_size
        is_training = self.is_training
        use_drop_mask = self.use_drop_mask
        use_bias = bias is not None
        if use_bias:
            bias_arg = bias.to(device=q.device, dtype=torch.float32)
            if bias_arg.dim() == 3:
                assert bias_arg.shape == (B, S, h_kv * G), (
                    f"bias shape {tuple(bias_arg.shape)} != ({B}, {S}, {h_kv * G})"
                )
                bias_arg = bias_arg.reshape(B, S, h_kv, G).contiguous()
            elif bias_arg.dim() == 4:
                assert bias_arg.shape == (B, S, h_kv, G), (
                    f"bias shape {tuple(bias_arg.shape)} != ({B}, {S}, {h_kv}, {G})"
                )
                bias_arg = bias_arg.contiguous()
            else:
                raise AssertionError(
                    f"bias must be [B, S, h_q] or [B, S, h_kv, G], got {tuple(bias_arg.shape)}"
                )
        else:
            bias_arg = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q.device)
        use_gumbel = bool(use_gumbel) or (gumbel_noise is not None)
        if use_gumbel:
            assert is_training, "gumbel_noise is only supported in training mode"
            if gumbel_noise is None:
                u = torch.rand(1, 1, h_kv * G, S, device=q.device, dtype=torch.float32)
                u.clamp_(min=1e-20, max=1 - 1e-7)
                gumbel_noise_arg = -torch.log(-torch.log(u))
            else:
                assert gumbel_noise.shape == (1, 1, h_kv * G, S), (
                    f"gumbel_noise shape {tuple(gumbel_noise.shape)} != ({1}, {1}, {h_kv * G}, {S})"
                )
                gumbel_noise_arg = gumbel_noise.detach().to(device=q.device, dtype=torch.float32).contiguous()
        else:
            gumbel_noise_arg = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device=q.device)

        if not is_training:
            shape_key = (B, h_kv, G, D, topk, block_size, window_size, is_causal, is_training, use_drop_mask, sm_scale, use_bias, use_gumbel, per_qhead_lmks)
        else:
            shape_key = (B, L, S, h_kv, G, D, topk, block_size, window_size, is_causal, is_training, use_drop_mask, sm_scale, use_bias, use_gumbel, per_qhead_lmks)

        if self._cached_shape != shape_key:
            seq_len_param = None if not is_training else L
            s_len_param = None if not is_training else S

            self._cached_lse_kernel = hils_lse_kernel(
                B, seq_len=seq_len_param, s_len=s_len_param, h_kv=h_kv, groups=G, head_dim=D,
                block_size=block_size, window_size=window_size, is_causal=is_causal,
                is_training=is_training, sm_scale=sm_scale, use_bias=use_bias, use_gumbel=use_gumbel,
                per_qhead_lmks=per_qhead_lmks,
            )

            self._cached_select_kernel = weighted_select_kernel(
                B, seq_len=seq_len_param, s_len=s_len_param, h_kv=h_kv, groups=G, head_dim=D,
                topk=topk, block_size=block_size, window_size=window_size, is_causal=is_causal,
                is_training=is_training, use_drop_mask=use_drop_mask, sm_scale=sm_scale, use_bias=use_bias, use_gumbel=use_gumbel,
                per_qhead_lmks=per_qhead_lmks,
            )

            self._cached_sort_kernel = sort_topk_indices_kernel(
                B, seq_len=seq_len_param, h_kv=h_kv, topk=topk,
                is_training=is_training
            )

            self._cached_recompute_kernel = recompute_topk_max_pooling_scores_kernel(
                B, seq_len=seq_len_param, s_len=s_len_param, h_kv=h_kv, groups=G, head_dim=D, topk=topk,
                is_training=is_training, sm_scale=sm_scale,
                per_qhead_lmks=per_qhead_lmks,
            )
            self._cached_shape = shape_key

        lse_kernel = self._cached_lse_kernel
        select_kernel = self._cached_select_kernel
        sort_kernel = self._cached_sort_kernel
        recompute_kernel = self._cached_recompute_kernel

        q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=q.device)

        indices, scores = SoftmaxTopKMaxPoolingFusedFn.apply(
            q, lmks, lse_swa, lse_kernel, select_kernel, sort_kernel, recompute_kernel,
            q_offset_tensor, drop_mask, sm_scale, bias_arg, gumbel_noise_arg,
            per_qhead_lmks,
        )
        scores = scores.view(B, L, h_kv * G, -1)
        return indices, scores

_SOFTMAX_MODULE_CACHE = {}


def online_softmax_topk_head(
    q: torch.Tensor,
    lmks: torch.Tensor,
    lse_swa: torch.Tensor,
    topk: int,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    q_offset: int = 0,
    is_training: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    bias: torch.Tensor = None,
    use_gumbel: bool = False,
    gumbel_noise: torch.Tensor = None,
    G: int = 1,
):
    """
    Functional API for SoftmaxTopKMaxPooling_Fused

    items query token, in  landmark key chunks in, use softmax-then-max-pooling
     top-k scores and returns chunk indices and scores.
    differentof q heads  chunks not chunk count.
    support causal mask, sliding window, DropMask, GQA/MQA .
    support autograd BWD.

    Args:
        q (torch.Tensor):
            Query tensor.shape = [B, L, h_q, D]
        lmks (torch.Tensor):
            Landmark key tensor.shape = [B, S, h_lmk, D]
            ``G``  lmks count KV-head of,  ``h_lmk == h_kv * G``:
            * ``G == 1``():``h_lmk == h_kv``, K in group ( GQA/MQA),
              kernel  ``G_kernel = h_q // h_kv``  query group.
            * ``G > 1``:``h_lmk == h_q``(per-q-head lmks), K not in group ,
               ``h_kv = h_q // G``, topk in ``h_kv`` (max over G).
            Note: if the D dimension of lmks is an integer multiple of q (D_lmk = D_q * ratio),
            reshape is [B, S, h_lmk * ratio, D_q] .
        lse_swa (torch.Tensor):
            SWA of LSE.shape = [B, L, h_q]  [B, L, h_kv, G]
        topk (int):
            Number of top-k chunks selected per query token.
        block_size (int):
            Chunk .
        window_size (int):
            Sliding window .
        is_causal (bool, default=True):
            Whether to enable causal masking.
        q_offset (int, default=0):
            query in KV cache in ,  inference  causal .
        is_training (bool, default=True):
            Whether this is training mode. If False, seq_len/s_len use dynamic shapes.
        drop_mask (torch.Tensor, optional):
            shape = [B, L, S], dtype = int32, 0/1 bitmap.
            1  chunk  drop, notand topk .
        bias (torch.Tensor, optional):
            Per-chunk per-query-head additive bias with shape [B, S, h_q]
            or [B, S, h_kv, G]. When provided, HiLS LSE and topk selection add
            bias[b, chunk, head], while returned scores stay raw scaled qk.
        use_gumbel (bool, optional):
            If True, enable Gumbel sampling in training mode. When
            `gumbel_noise` is None, noise is generated internally in Python.
        gumbel_noise (torch.Tensor, optional):
            Optional pre-generated noise tensor with shape [1, 1, h_q, S]. When
            provided, it overrides internal generation and is useful for
            deterministic ref/fused consistency tests. No gradient is
            propagated to gumbel_noise.
        G (int, optional, default=1):
            ``lmks``  KV-head replication factor,  ``h_lmk = h_kv * G``.
            ``G == 1``  K path( GQA/MQA);``G > 1``  per-q-head path,
             ``lmks.shape[2] == q.shape[2] == h_q``, ``h_kv = h_q // G``,
            topk in ``h_kv`` (max over G).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (indices, scores)
        - indices: [B, L, h_kv, topk], ( KV head )
        - scores: [B, L, h_q, topk], of raw scaled qk scores
    """

    if q.dim() == 4:
        B, L, h_q, D = q.shape
        lmks_h = lmks.shape[2]
        D_lmk = lmks.shape[3]

        if D != D_lmk:
            assert D_lmk % D == 0, f"lmks D dim ({D_lmk}) must be divisible by q D dim ({D})"
            d_ratio = D_lmk // D
            lmks = lmks.reshape(lmks.shape[0], lmks.shape[1], lmks_h * d_ratio, D)
            lmks_h = lmks_h * d_ratio

        if G == 1:
            assert h_q % lmks_h == 0, f"h_q ({h_q}) must be divisible by lmks_h ({lmks_h})"
            h_kv = lmks_h
            G_eff = h_q // h_kv
            per_qhead_lmks = False
        else:
            assert lmks_h == h_q, (
                f"when G>1, lmks must have h_q ({h_q}) heads, got {lmks_h}"
            )
            assert h_q % G == 0, f"h_q ({h_q}) must be divisible by G ({G})"
            h_kv = h_q // G
            G_eff = G
            per_qhead_lmks = True
        q = q.view(B, L, h_kv, G_eff, D)
    else:
        B, L, h_kv, G_in, D = q.shape
        h_q = h_kv * G_in
        D_lmk = lmks.shape[3]
        if D != D_lmk:
            assert D_lmk % D == 0, f"lmks D dim ({D_lmk}) must be divisible by q D dim ({D})"
            d_ratio = D_lmk // D
            lmks = lmks.reshape(lmks.shape[0], lmks.shape[1], lmks.shape[2] * d_ratio, D)
        lmks_h = lmks.shape[2]
        if G == 1:
            if lmks_h == h_kv:
                per_qhead_lmks = False
                G_eff = G_in
            elif lmks_h == h_q:
                per_qhead_lmks = True
                G_eff = G_in
            else:
                raise AssertionError(
                    f"lmks_h ({lmks_h}) must be h_kv ({h_kv}) or h_q ({h_q})"
                )
        else:
            assert lmks_h == h_q, (
                f"when G>1, lmks must have h_q ({h_q}) heads, got {lmks_h}"
            )
            assert h_q % G == 0, f"h_q ({h_q}) must be divisible by G ({G})"
            new_h_kv = h_q // G
            new_G = G
            if (new_h_kv != h_kv) or (new_G != G_in):
                q = q.reshape(B, L, new_h_kv, new_G, D)
                if lse_swa.dim() == 4:
                    lse_swa = lse_swa.reshape(B, L, new_h_kv, new_G)
                h_kv = new_h_kv
                G_eff = new_G
            else:
                G_eff = G_in
            per_qhead_lmks = True
        if lse_swa.dim() == 3:
            lse_swa = lse_swa.view(B, L, h_kv, G_eff)
        elif lse_swa.dim() == 4 and (lse_swa.shape[2] != h_kv or lse_swa.shape[3] != G_eff):
            lse_swa = lse_swa.reshape(B, L, h_kv, G_eff)
    use_drop_mask = drop_mask is not None
    use_bias = bias is not None
    use_gumbel = bool(use_gumbel) or (gumbel_noise is not None)
    cache_key = (topk, block_size, window_size, is_causal, is_training, use_drop_mask, use_bias, use_gumbel, per_qhead_lmks)

    if cache_key not in _SOFTMAX_MODULE_CACHE:
        _SOFTMAX_MODULE_CACHE[cache_key] = SoftmaxTopKMaxPooling_Fused(
            topk, block_size, window_size, is_causal, is_training, use_drop_mask,
            per_qhead_lmks=per_qhead_lmks,
        )

    return _SOFTMAX_MODULE_CACHE[cache_key](q, lmks, lse_swa, q_offset=q_offset, drop_mask=drop_mask, sm_scale=sm_scale, bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise)



def test_fused_topk_softmax_max_pooling_correctness():
    print("\n" + "=" * 70)
    print("=== Testing Fused Softmax TopK Max-Pooling Kernel Correctness ===")
    print("=" * 70)

    B, L, D = 64, 4096, 128
    h_kv = 2
    G = 8
    h_q = h_kv * G
    S = 64
    topk = 16
    is_causal = True
    block_size = 32
    window_size = 32

    dtype = torch.bfloat16
    device = "cuda"

    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}, window_size={window_size}")

    torch.manual_seed(4200)

    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)
    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")

    # Reference returns: indices [B, L, h_kv, topk], scores [B, L, h_q, topk]
    ref_indices, ref_scores = ref_softmax_topk_max_pooling(
        q.detach(), lmks.detach(), lse_swa.detach().float(), topk, block_size, window_size, is_causal
    )

    # Fused returns: indices [B, L, h_kv, topk], scores [B, L, h_q, topk]
    fused_indices, fused_scores = online_softmax_topk_head(
        q, lmks, lse_swa, topk, block_size, window_size, is_causal
    )
    # print(fused_indices[0,0,0,:])
    # print(fused_scores[0,0,0,:])
    # print(fused_indices[0,64,0,:])
    # print(fused_scores[0,64,0,:])
    # print(fused_indices[0,132,0,:])
    # print(fused_scores[0,132,0,:])
    # print(fused_indices[0,133,0,:])
    # print(fused_scores[0,133,0,:])

    # Reshape fused scores back to [B, L, h_kv, G, topk] for comparison
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)

    # Calculate ground truth scores for all candidates
    scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q.float(), lmks.float())
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D))

    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        # Update manual check to match window mechanism
        if window_size > 0:
            aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        else:
            aligned_threshold = i_idx.div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= aligned_threshold

        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    safe_indices = fused_indices.clone()
    safe_indices[safe_indices < 0] = 0

    # Gather scores using fused indices to verify values
    indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    # Compare valid scores (ignore masked/padding)
    valid_mask = (scores_gathered_ref > -1e9) & (fused_scores_reshaped.float() > -1e9)

    if valid_mask.sum() == 0:
        print("Warning: No valid scores to compare (all masked?)")
        max_score_diff = 0.0
        rel_l2_score = 0.0
    else:
        # Use reshaped fused scores for comparison
        score_diff = torch.abs(scores_gathered_ref[valid_mask] - fused_scores_reshaped.float()[valid_mask])
        max_score_diff = score_diff.max().item()
        rel_l2_score = score_diff.norm().item() / (scores_gathered_ref[valid_mask].norm().item() + 1e-6)

    print(f"Forward scores (valid only) - Max Diff: {max_score_diff:.6f}")
    print(f"Forward scores (valid only) - L2 RelErr: {rel_l2_score:.6f}")

    indices_match = (fused_indices.long() == ref_indices.long()) # [B, L, h_kv, topk]

    ref_scores_pooled = ref_scores.max(dim=3).values # [B, L, h_kv, topk]

    valid_indices_mask = (ref_scores_pooled > -1e9) # [B, L, h_kv, topk]

    match_rate = indices_match[valid_indices_mask].float().mean().item()


    print(f"Indices Match Rate (Valid Elements): {match_rate*100:.6f}%")

    if match_rate >= 0.99 and max_score_diff < 1 and rel_l2_score < 1e-2:
        print(" Fused Forward PASSED")
    else:
        print(" Fused Forward FAILED")

    # ============ Backward Correctness ============
    print("\n--- Backward Correctness ---")

    q_fused = q.detach().clone().requires_grad_(True)
    lmks_fused = lmks.detach().clone().requires_grad_(True)
    q_ref = q.detach().clone().requires_grad_(True)
    lmks_ref = lmks.detach().clone().requires_grad_(True)

    # grad_output shape matches fused output: [B, L, h_q, topk]
    grad_output = torch.randn(B, L, h_kv * G, topk, dtype=dtype, device=device)

    # Create a view for reference implementation: [B, L, h_kv, G, topk]
    grad_output_ref_view = grad_output.view(B, L, h_kv, G, topk)

    # Mask gradients for invalid positions based on reference forward pass
    with torch.no_grad():
        _, ref_scores_check = ref_softmax_topk_max_pooling(
            q_ref, lmks_ref, lse_swa, topk, block_size, window_size, is_causal
        )
        invalid_mask = (ref_scores_check < -1e9)
        grad_output_ref_view[invalid_mask] = 0.0
        # Update grad_output with masked values (in-place modification affects grad_output)

    # Fused Backward
    indices_fused_bwd, scores_fused_bwd = online_softmax_topk_head(
        q_fused, lmks_fused, lse_swa, topk, block_size, window_size, is_causal
    )
    # scores_fused_bwd is [B, L, h_q, topk], grad_output is [B, L, h_q, topk]
    loss_fused = (scores_fused_bwd * grad_output).sum()
    loss_fused.backward()
    grad_q_fused = q_fused.grad.clone()
    grad_lmks_fused = lmks_fused.grad.clone()

    # Ref Backward
    scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q_ref.float(), lmks_ref.float())
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D)) # Scale!

    if is_causal:
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    # Use fused indices for gather to match the graph
    safe_indices_bwd = indices_fused_bwd.clone()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1).long()

    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    # Use reshaped grad_output for reference: [B, L, h_kv, G, topk]
    loss_ref = (scores_gathered_ref * grad_output_ref_view.float()).sum()
    loss_ref.backward()
    grad_q_ref = q_ref.grad.clone()
    grad_lmks_ref = lmks_ref.grad.clone()

    # Handle NaNs in grads (from -inf scores)
    if torch.isnan(grad_q_ref).any():
        grad_q_ref = torch.nan_to_num(grad_q_ref, 0.0)
    if torch.isnan(grad_lmks_ref).any():
        grad_lmks_ref = torch.nan_to_num(grad_lmks_ref, 0.0)
    if torch.isnan(grad_q_fused).any():
        grad_q_fused = torch.nan_to_num(grad_q_fused, 0.0)
    if torch.isnan(grad_lmks_fused).any():
        grad_lmks_fused = torch.nan_to_num(grad_lmks_fused, 0.0)

    diff_grad_q = torch.abs(grad_q_fused - grad_q_ref)
    diff_grad_lmks = torch.abs(grad_lmks_fused - grad_lmks_ref)

    max_diff_q = diff_grad_q.max().item()
    max_diff_lmks = diff_grad_lmks.max().item()

    norm_q_ref = torch.norm(grad_q_ref).item()
    norm_lmks_ref = torch.norm(grad_lmks_ref).item()
    norm_diff_q = torch.norm(diff_grad_q).item()
    norm_diff_lmks = torch.norm(diff_grad_lmks).item()

    rel_err_q = norm_diff_q / (norm_q_ref + 1e-6)
    rel_err_lmks = norm_diff_lmks / (norm_lmks_ref + 1e-6)

    print(f"Fused vs Ref - Max grad_q Diff: {max_diff_q:.6f}")
    print(f"Fused vs Ref - Max grad_lmks Diff: {max_diff_lmks:.6f}")
    print(f"Fused vs Ref - L2 Relative Error grad_q: {rel_err_q:.6f} ({rel_err_q*100:.4f}%)")
    print(f"Fused vs Ref - L2 Relative Error grad_lmks: {rel_err_lmks:.6f} ({rel_err_lmks*100:.4f}%)")

    if rel_err_q < 0.01 and rel_err_lmks < 0.01:
        print(" Fused Backward PASSED")
    else:
        print(" Fused Backward FAILED")

def test_fused_softmax_topk_max_pooling_memory_and_speed(
    name: str = "default",
    B: int = 4, L: int = 8192, D: int = 64, h_kv: int = 32, G: int = 1,
    S: int = 128, topk: int = 32, block_size: int = 64, window_size: int = 512,
    is_causal: bool = True,
    use_bias: bool = False,
    use_gumbel: bool = False,
    per_qhead_lmks: bool = False,
    n_iters: int = 20, n_warmup: int = 5,
    skip_ref: bool = False,
    pass_G: bool = None,  # whether to pass G to online_softmax_topk_head; defaults to True iff per_qhead_lmks
):
    """Benchmark fused vs reference impl. Supports per-chunk bias / per-q-head lmks / Gumbel."""
    print("\n" + "=" * 70)
    print(f"=== Benchmark [{name}] Fused Softmax TopK Max-Pooling ===")
    print("=" * 70)

    h_q = h_kv * G

    dtype = torch.bfloat16
    device = "cuda"

    if pass_G is None:
        pass_G = per_qhead_lmks
    G_arg = G if pass_G else None

    print(
        f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, "
        f"topk={topk}, block_size={block_size}, window_size={window_size}, "
        f"use_bias={use_bias}, use_gumbel={use_gumbel}, per_qhead_lmks={per_qhead_lmks}, "
        f"pass_G={pass_G}"
    )

    torch.manual_seed(42)
    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    if per_qhead_lmks:
        lmks = torch.randn(B, S, h_kv * G, D, dtype=dtype, device=device)
    else:
        lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    lse_swa = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    grad_output = torch.randn(B, L, h_q, topk, dtype=dtype, device=device)
    grad_output_ref = grad_output.view(B, L, h_kv, G, topk)

    bias = None
    if use_bias:
        # bias shape [B, S, h_q] is also accepted by both fused and ref
        bias = torch.randn(B, S, h_q, dtype=torch.float32, device=device) * 0.5

    gumbel_noise = None
    if use_gumbel:
        # Match shape used elsewhere: [1, 1, h_q, S]
        u = torch.rand(1, 1, h_q, S, dtype=torch.float32, device=device).clamp_(1e-6, 1 - 1e-6)
        gumbel_noise = -torch.log(-torch.log(u))

    def _make_inputs():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)
        lse_swa_t = lse_swa.detach().clone()
        return q_t, lmks_t, lse_swa_t

    def run_fused():
        q_t, lmks_t, lse_swa_t = _make_inputs()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # JIT compile + first call
        _, scores = online_softmax_topk_head(
            q_t, lmks_t, lse_swa_t, topk, block_size, window_size, is_causal,
            bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise, G=G_arg,
        )
        loss = (scores * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(n_warmup):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_softmax_topk_head(
                q_t, lmks_t, lse_swa_t, topk, block_size, window_size, is_causal,
                bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise, G=G_arg,
            )
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        _, scores = online_softmax_topk_head(
            q_t, lmks_t, lse_swa_t, topk, block_size, window_size, is_causal,
            bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise, G=G_arg,
        )
        loss = (scores * grad_output).sum()
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)

        # Fwd only
        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _ = online_softmax_topk_head(
                q_t, lmks_t, lse_swa_t, topk, block_size, window_size, is_causal,
                bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise, G=G_arg,
            )
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters

        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_softmax_topk_head(
                q_t, lmks_t, lse_swa_t, topk, block_size, window_size, is_causal,
                bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise, G=G_arg,
            )
            loss = (scores * grad_output).sum()
            loss.backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters

        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms

    def run_ref():
        q_t, lmks_t, lse_swa_t = _make_inputs()
        lse_swa_ref = lse_swa_t.float()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def forward_only():
            _ = ref_softmax_topk_max_pooling(
                q_t, lmks_t, lse_swa_ref, topk, block_size, window_size, is_causal,
                bias=bias, gumbel_noise=gumbel_noise,
            )

        def forward_backward():
            _, scores = ref_softmax_topk_max_pooling(
                q_t, lmks_t, lse_swa_ref, topk, block_size, window_size, is_causal,
                bias=bias, gumbel_noise=gumbel_noise,
            )
            loss = (scores * grad_output_ref).sum()
            loss.backward()

        for _ in range(n_warmup):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
            forward_backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        forward_backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)

        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters

        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            forward_backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters

        avg_bwd_ms = avg_all_ms - avg_fwd_ms
        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms

    print("\nRunning benchmarks...")

    mem_fused, fwd_fused, all_fused, bwd_fused = run_fused()
    print(f"\n[Fused Softmax TopK Max-Pooling]")
    print(f"  Peak Memory: {mem_fused:.2f} MB")
    print(f"  Avg Fwd Latency: {fwd_fused:.2f} ms")
    print(f"  Avg Fwd+Bwd Latency: {all_fused:.2f} ms")
    print(f"  Derived Bwd Latency: {bwd_fused:.2f} ms")

    if skip_ref:
        return

    try:
        mem_ref, fwd_ref, all_ref, bwd_ref = run_ref()
        print(f"\n[Reference (PyTorch)]")
        print(f"  Peak Memory: {mem_ref:.2f} MB")
        print(f"  Avg Fwd Latency: {fwd_ref:.2f} ms")
        print(f"  Avg Fwd+Bwd Latency: {all_ref:.2f} ms")
        print(f"  Derived Bwd Latency: {bwd_ref:.2f} ms")

        print("\n" + "-" * 70)
        print("Comparison:")
        print("-" * 70)
        print(f"{'Method':<25} {'Memory (MB)':<15} {'Fwd (ms)':<12} {'Fwd+Bwd (ms)':<15} {'Bwd (ms)':<12}")
        print("-" * 70)
        print(f"{'Fused':<25} {mem_fused:<15.2f} {fwd_fused:<12.2f} {all_fused:<15.2f} {bwd_fused:<12.2f}")
        print(f"{'Reference':<25} {mem_ref:<15.2f} {fwd_ref:<12.2f} {all_ref:<15.2f} {bwd_ref:<12.2f}")
        print("-" * 70)
        print(f"Speedup (Fwd): {fwd_ref / max(fwd_fused, 1e-6):.2f}x")
        print(f"Speedup (Bwd): {bwd_ref / max(bwd_fused, 1e-6):.2f}x")
        print(f"Speedup (Fwd+Bwd): {all_ref / max(all_fused, 1e-6):.2f}x")
        print(f"Memory Saving: {mem_ref / max(mem_fused, 1e-6):.2f}x")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n[Reference (PyTorch)] OOM - Cannot run with this config")
            print("\n" + "-" * 70)
            print("Comparison (Reference OOM):")
            print("-" * 70)
            print(f"{'Method':<25} {'Memory (MB)':<15} {'Fwd (ms)':<12} {'Fwd+Bwd (ms)':<15} {'Bwd (ms)':<12}")
            print("-" * 70)
            print(f"{'Fused':<25} {mem_fused:<15.2f} {fwd_fused:<12.2f} {all_fused:<15.2f} {bwd_fused:<12.2f}")
            print("-" * 70)
        else:
            raise e

import pytest
@pytest.mark.parametrize("B, L, S, h_kv, G, D, topk, block_size, window_size", [
    (2, 4096, 64, 2, 8, 64, 16, 64, 64),
    (2, 4096, 64, 8, 8, 64, 16, 64, 64),
])
def test_topk_correctness_robust(B, L, S, h_kv, G, D, topk, block_size, window_size):
    device = "cuda"
    dtype = torch.bfloat16
    is_causal = True

    torch.manual_seed(42)

    print(f"\nTesting Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G}, D={D}, topk={topk}, BS={block_size}, WS={window_size}")

    # Q: [B, L, h_kv, G, D]
    q_raw = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    # Lmks: [B, S, h_kv, D]
    lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    # LSE SWA: [B, L, h_kv, G]
    lse_swa_raw = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)

    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)

    lse_swa = lse_swa_raw.clone().detach()

    # Fused Kernel
    # indices: [B, L, h_kv, topk]
    # scores:  [B, L, h_q, topk]
    indices_fused, scores_fused = online_softmax_topk_head(q_fused, lmks_fused, lse_swa, topk, block_size, window_size, is_causal)

    # Reference
    # scores: [B, L, h_kv, G, topk]
    indices_ref, scores_ref = ref_softmax_topk_max_pooling(q_ref, lmks_ref, lse_swa.float(), topk, block_size, window_size, is_causal)

    def get_abs_err(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        return (x[mask] - y[mask]).abs().max().item()

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio=0.01):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    # Reshape fused scores to match reference [B, L, h_kv, G, topk]
    assert_close("FWD Scores", scores_ref.float(), scores_fused.view(B, L, h_kv, G, topk).float())

    grad_output = torch.randn_like(scores_fused, dtype=dtype)

    with torch.no_grad():
        # scores_ref is [B, L, h_kv, G, topk]
        invalid_mask = scores_ref < -1e5
        # Reshape grad_output to match mask for masking
        grad_output_view = grad_output.view(B, L, h_kv, G, topk)
        grad_output_view[invalid_mask] = 0.0

    # Fused Backward
    scores_fused.backward(grad_output)
    dq_fused, dlmks_fused = q_fused.grad.clone(), lmks_fused.grad.clone()

    # Ref Backward
    # Reshape grad_output to match ref scores [B, L, h_kv, G, topk]
    scores_ref.backward(grad_output.view(B, L, h_kv, G, topk))
    dq_ref, dlmks_ref = q_ref.grad.clone(), lmks_ref.grad.clone()

    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)

    assert_close("DQ", dq_ref.float(), dq_fused.float(), ratio=0.05)
    assert_close("DLmks", dlmks_ref.float(), dlmks_fused.float(), ratio=0.05)

    print(f"Test Passed: B={B}, L={L}, S={S}, G={G}, topk={topk}")



import pytest
@pytest.mark.parametrize(
    "test_name, B, q_len, kv_len, h_kv, G, D, topk, block_size, window_size, is_training, q_offset, use_bias, use_gumbel, per_qhead_lmks",
    [
        ("train_basic", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, False, False, False),
        ("train_basic_bias", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, False, False),
        ("train_basic_bias_gumbel", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, True, False),
        ("train_perqhead", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, False, False, True),
        ("train_perqhead_bias", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, False, True),
    ]
)
def test_train_inference_correctness(test_name, B, q_len, kv_len, h_kv, G, D, topk, block_size, window_size, is_training, q_offset, use_bias, use_gumbel, per_qhead_lmks=False, is_causal=True):
    """
    test TopK in training and inference scenario correctness

    trainingscenario:
    - q_len == kv_len
    - is_training = True
    - q_offset = 0
    - support batch size

    scenario:
    - q_len <= kv_len
    - is_training = False
    - q_offset = kv_len - q_len (Q  KV in )
    - batch is 1
    """
    device = "cuda"
    dtype = torch.bfloat16
    h_q = h_kv * G

    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"Config: B={B}, q_len={q_len}, kv_len={kv_len}, h_kv={h_kv}, G={G}, D={D}")
    print(f"        topk={topk}, block_size={block_size}, window_size={window_size}")
    print(f"        is_training={is_training}, is_causal={is_causal}, q_offset={q_offset}, use_bias={use_bias}, use_gumbel={use_gumbel}, per_qhead_lmks={per_qhead_lmks}")
    print(f"{'='*70}")

    if not is_training:
        assert B == 1, f"Inference mode requires B=1, got B={B}"

    if not is_training and q_len < kv_len:
        expected_q_offset = kv_len - q_len
        assert q_offset == expected_q_offset, f"q_offset should be {expected_q_offset}, got {q_offset}"

    torch.manual_seed(42)

    S = kv_len // block_size

    # Q: [B, q_len, h_kv, G, D]
    q_raw = torch.randn(B, q_len, h_kv, G, D, dtype=dtype, device=device)
    if per_qhead_lmks:
        lmks_raw = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
    else:
        lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    # LSE SWA: [B, q_len, h_kv, G]
    lse_swa_raw = torch.randn(B, q_len, h_kv, G, dtype=dtype, device=device) * 5 + 10

    q_fused = q_raw.clone().detach().requires_grad_(is_training)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(is_training)

    q_ref = q_raw.clone().detach().requires_grad_(is_training)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(is_training)

    lse_swa = lse_swa_raw.clone().detach()
    if use_bias:
        # Deterministic per-chunk per-head additive bias. Values vary across
        # both chunk and q head so LSE, selection, and selected-score gather are tested.
        chunk_axis = torch.linspace(-1.0, 1.0, S, device=device, dtype=torch.float32).view(1, S, 1)
        head_axis = torch.linspace(-0.5, 0.5, h_kv * G, device=device, dtype=torch.float32).view(1, 1, h_kv * G)
        batch_axis = torch.arange(B, device=device, dtype=torch.float32).view(B, 1, 1) * 0.01
        bias = (chunk_axis + head_axis + batch_axis).contiguous()
    else:
        bias = None
    if use_gumbel:
        assert is_training, "gumbel_noise test case must run in training mode"
        torch.manual_seed(123)
        u = torch.rand(1, 1, h_kv * G, S, device=device, dtype=torch.float32).clamp_(min=1e-20, max=1 - 1e-7)
        gumbel_noise = -torch.log(-torch.log(u))
    else:
        gumbel_noise = None

    G_arg = G if per_qhead_lmks else None

    print("\n--- Forward Pass ---")

    indices_ref, scores_ref = ref_softmax_topk_max_pooling(
        q_ref, lmks_ref, lse_swa.float(), topk, block_size, window_size, is_causal,
        q_offset=q_offset, bias=bias, gumbel_noise=gumbel_noise
    )

    # Fused Kernel
    indices_fused, scores_fused = online_softmax_topk_head(
        q_fused, lmks_fused, lse_swa, topk, block_size, window_size, is_causal,
        q_offset=q_offset, is_training=is_training, bias=bias, use_gumbel=use_gumbel, gumbel_noise=gumbel_noise,
        G=G_arg,
    )

    def get_abs_err(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0:
            return 0.0
        return (x[mask] - y[mask]).abs().max().item()

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0:
            return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = x[mask].square().mean().sqrt().item()
        return err / (base + 1e-12)

    # Reshape fused scores to match reference [B, q_len, h_kv, G, topk]
    scores_fused_reshaped = scores_fused.view(B, q_len, h_kv, G, topk)

    fwd_abs_err = get_abs_err(scores_ref.float(), scores_fused_reshaped.float())
    fwd_rel_err = get_err_ratio(scores_ref.float(), scores_fused_reshaped.float())

    print(f"Forward Scores - Abs Error: {fwd_abs_err:.6f}")
    print(f"Forward Scores - Rel Error: {fwd_rel_err:.6f}")

    indices_match = (indices_fused.long() == indices_ref.long())
    scores_ref_pooled = scores_ref.max(dim=3).values
    valid_mask = scores_ref_pooled > -1e5
    if valid_mask.sum() > 0:
        match_rate = indices_match[valid_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}%")
    else:
        match_rate = 1.0
        print("Indices Match Rate: N/A (all masked)")

    assert fwd_rel_err < 0.02, f"Forward relative error too large: {fwd_rel_err}"
    print(" Forward PASSED")

    if is_training:
        print("\n--- Backward Pass ---")

        grad_output = torch.randn_like(scores_fused, dtype=dtype)

        with torch.no_grad():
            invalid_mask = scores_ref < -1e5
            grad_output_view = grad_output.view(B, q_len, h_kv, G, topk)
            grad_output_view[invalid_mask] = 0.0

        # Fused Backward
        scores_fused.backward(grad_output)
        dq_fused = q_fused.grad.clone()
        dlmks_fused = lmks_fused.grad.clone()

        # Ref Backward
        scores_ref.backward(grad_output.view(B, q_len, h_kv, G, topk))
        dq_ref = q_ref.grad.clone()
        dlmks_ref = lmks_ref.grad.clone()

        dq_fused = torch.nan_to_num(dq_fused, 0.0)
        dq_ref = torch.nan_to_num(dq_ref, 0.0)
        dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
        dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)

        dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
        dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())

        print(f"dQ Rel Error: {dq_rel_err:.6f}")
        print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")

        assert dq_rel_err < 0.05, f"dQ relative error too large: {dq_rel_err}"
        assert dlmks_rel_err < 0.05, f"dLmks relative error too large: {dlmks_rel_err}"
        print(" Backward PASSED")
    else:
        print("\n--- Backward Pass ---")
        print("Skipped:  Skipped (inference mode)")

    print(f"\n Test '{test_name}' PASSED")


def test_drop_mask_correctness(per_qhead_lmks: bool = False):
    """
    test DropMask functionality correctness.
    validate: chunk  drop , fused kernel  ref of.
    """
    print("\n" + "=" * 70)
    print(f"=== Testing DropMask Correctness (per_qhead_lmks={per_qhead_lmks}) ===")
    print("=" * 70)

    B, L, D = 2, 1024, 128
    h_kv = 2
    G = 8
    h_q = h_kv * G
    S = 64
    topk = 8
    is_causal = True
    block_size = 32
    window_size = 32

    dtype = torch.bfloat16
    device = "cuda"

    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}")

    torch.manual_seed(42)

    q_raw = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    if per_qhead_lmks:
        lmks_raw = torch.randn(B, S, h_q, D, dtype=dtype, device=device)
    else:
        lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)
    lse_swa_raw = torch.randn(B, L, h_kv, G, dtype=dtype, device=device) * 5 + 10

    drop_mask = (torch.rand(B, L, S, device=device) < 0.3).to(torch.int32)

    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)
    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)
    lse_swa = lse_swa_raw.clone().detach()

    G_arg = G if per_qhead_lmks else None

    # ============ Forward ============
    print("\n--- Forward Correctness ---")

    ref_indices, ref_scores = ref_softmax_topk_max_pooling(
        q_ref, lmks_ref, lse_swa.float(), topk, block_size, window_size, is_causal, drop_mask=drop_mask
    )

    fused_indices, fused_scores = online_softmax_topk_head(
        q_fused, lmks_fused, lse_swa, topk, block_size, window_size, is_causal,
        is_training=True, drop_mask=drop_mask, G=G_arg,
    )

    # Reshape fused scores for comparison
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    if per_qhead_lmks:
        lmks_raw_v = lmks_raw.view(B, S, h_kv, G, D)
        scores_all_ref = torch.einsum("blhgd,bshgd->blhgs", q_raw.float(), lmks_raw_v.float())
    else:
        scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q_raw.float(), lmks_raw.float())
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D))
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        if window_size > 0:
            aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        else:
            aligned_threshold = i_idx.div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= aligned_threshold
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    safe_indices = fused_indices.clone()
    safe_indices[safe_indices < 0] = 0
    # fused_indices: [B, L, h_kv, topk] -> expand to [B, L, h_kv, G, topk]
    indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    # scores_all_ref: [B, L, h_kv, G, S] -> gather on S dim
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    valid_mask = (scores_gathered_ref > -1e9) & (fused_scores_reshaped.float() > -1e9)
    if valid_mask.sum() == 0:
        max_score_diff = 0.0
        rel_l2_score = 0.0
    else:
        score_diff = torch.abs(scores_gathered_ref[valid_mask] - fused_scores_reshaped.float()[valid_mask])
        max_score_diff = score_diff.max().item()
        rel_l2_score = score_diff.norm().item() / (scores_gathered_ref[valid_mask].norm().item() + 1e-6)

    print(f"Forward scores (aligned by fused indices) - Max Diff: {max_score_diff:.6f}")
    print(f"Forward scores (aligned by fused indices) - L2 RelErr: {rel_l2_score:.6f}")

    indices_match = (fused_indices.long() == ref_indices.long())
    scores_ref_pooled = ref_scores.max(dim=3).values
    valid_indices_mask = scores_ref_pooled > -1e5
    if valid_indices_mask.sum() > 0:
        match_rate = indices_match[valid_indices_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}% (head  LSE  < 100%)")
    else:
        match_rate = 1.0

    # fused_indices: [B, L, h_kv, topk]
    # drop_mask: [B, L, S]
    dropped_selected = 0
    total_valid = 0
    for b in range(B):
        for l in range(L):
            for h in range(h_kv):
                for k in range(topk):
                    idx = fused_indices[b, l, h, k].item()
                    if idx >= 0 and idx < S:
                        total_valid += 1
                        if drop_mask[b, l, idx].item() == 1:
                            dropped_selected += 1

    if total_valid > 0:
        drop_violation_rate = dropped_selected / total_valid
        print(f"Drop Violation Rate: {drop_violation_rate*100:.4f}% ({dropped_selected}/{total_valid})")
    else:
        drop_violation_rate = 0.0
        print("No valid indices to check")

    if max_score_diff < 1.0 and rel_l2_score < 1e-2 and drop_violation_rate < 0.001:
        print(" DropMask Forward PASSED")
    else:
        print(" DropMask Forward FAILED")

    # ============ Backward ============
    print("\n--- Backward Correctness ---")

    grad_output = torch.randn_like(fused_scores, dtype=dtype)

    with torch.no_grad():
        fused_scores_view = fused_scores.view(B, L, h_kv, G, topk)
        invalid_mask_bwd = fused_scores_view.float() < -1e5
        grad_output_view = grad_output.view(B, L, h_kv, G, topk)
        grad_output_view[invalid_mask_bwd] = 0.0

    # Fused Backward
    loss_fused = (fused_scores * grad_output).sum()
    loss_fused.backward()
    dq_fused = q_fused.grad.clone()
    dlmks_fused = lmks_fused.grad.clone()

    if per_qhead_lmks:
        lmks_ref_v = lmks_ref.view(B, S, h_kv, G, D)
        scores_all_ref_bwd = torch.einsum("blhgd,bshgd->blhgs", q_ref.float(), lmks_ref_v.float())
    else:
        scores_all_ref_bwd = torch.einsum("blhgd,bshd->blhgs", q_ref.float(), lmks_ref.float())
    scores_all_ref_bwd = scores_all_ref_bwd * (1.0 / math.sqrt(D))
    if is_causal:
        scores_all_ref_bwd = scores_all_ref_bwd.masked_fill(causal_mask_expanded, float('-inf'))

    safe_indices_bwd = fused_indices.clone().detach()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded_bwd = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref_bwd = torch.gather(scores_all_ref_bwd, -1, indices_expanded_bwd)

    loss_ref = (scores_gathered_ref_bwd * grad_output.view(B, L, h_kv, G, topk).float()).sum()
    loss_ref.backward()
    dq_ref = q_ref.grad.clone()
    dlmks_ref = lmks_ref.grad.clone()

    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)

    dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
    dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())

    print(f"dQ Rel Error: {dq_rel_err:.6f}")
    print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")

    if dq_rel_err < 0.05 and dlmks_rel_err < 0.05:
        print(" DropMask Backward PASSED")
    else:
        print(" DropMask Backward FAILED")


def test_gqa_d_reshape_correctness():
    """
    test GQA  (D_lmk != D_q) correctness.
     lmks of D  q an integer multiple of,  reshape isof head.
    """
    print("\n" + "=" * 70)
    print("=== Testing GQA D-Reshape (D_lmk != D_q) Correctness ===")
    print("=" * 70)

    B, L = 2, 512
    D_q = 64
    D_lmk = 128
    h_kv_orig = 2
    G_orig = 4
    h_q = h_kv_orig * G_orig  # = 8
    S = 16
    topk = 8
    is_causal = True
    block_size = 32
    window_size = 32

    dtype = torch.bfloat16
    device = "cuda"

    d_ratio = D_lmk // D_q
    h_kv_new = h_kv_orig * d_ratio
    G_new = h_q // h_kv_new

    print(f"Config: B={B}, L={L}, S={S}")
    print(f"  D_q={D_q}, D_lmk={D_lmk}, d_ratio={d_ratio}")
    print(f"  h_kv_orig={h_kv_orig}, G_orig={G_orig}, h_q={h_q}")
    print(f"  After reshape: h_kv_new={h_kv_new}, G_new={G_new}")
    print(f"  topk={topk}, block_size={block_size}, window_size={window_size}")

    torch.manual_seed(42)

    q_raw = torch.randn(B, L, h_q, D_q, dtype=dtype, device=device)
    # lmks: [B, S, h_kv_orig, D_lmk]
    lmks_raw = torch.randn(B, S, h_kv_orig, D_lmk, dtype=dtype, device=device)
    # lse_swa: [B, L, h_q]
    lse_swa_raw = torch.randn(B, L, h_q, dtype=dtype, device=device) * 5 + 10

    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)

    q_ref = q_raw.clone().detach()
    lmks_ref = lmks_raw.clone().detach()
    lse_swa = lse_swa_raw.clone().detach()

    # ============ Forward ============
    print("\n--- Forward ---")

    fused_indices, fused_scores = online_softmax_topk_head(
        q_fused, lmks_fused, lse_swa, topk, block_size, window_size, is_causal,
        is_training=True
    )

    # lmks reshape: [B, S, h_kv_orig, D_lmk] -> [B, S, h_kv_new, D_q]
    lmks_reshaped = lmks_ref.reshape(B, S, h_kv_new, D_q)
    # q reshape: [B, L, h_q, D_q] -> [B, L, h_kv_new, G_new, D_q]
    q_reshaped = q_ref.view(B, L, h_kv_new, G_new, D_q)
    # lse_swa reshape: [B, L, h_q] -> [B, L, h_kv_new, G_new]
    lse_swa_reshaped = lse_swa.view(B, L, h_kv_new, G_new)

    ref_indices, ref_scores = ref_softmax_topk_max_pooling(
        q_reshaped, lmks_reshaped, lse_swa_reshaped.float(), topk, block_size, window_size, is_causal
    )

    # fused_scores: [B, L, h_q, topk] -> [B, L, h_kv_new, G_new, topk]
    fused_scores_reshaped = fused_scores.view(B, L, h_kv_new, G_new, topk)

    def get_err_ratio(x, y):
        mask = (x > -1e5) & (y > -1e5)
        if mask.sum() == 0: return 0.0
        err = (x[mask] - y[mask]).square().mean().sqrt().item()
        base = (x[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    fwd_rel_err = get_err_ratio(ref_scores.float(), fused_scores_reshaped.float())
    print(f"Forward Scores Rel Error: {fwd_rel_err:.6f}")

    indices_match = (fused_indices.long() == ref_indices.long())
    scores_ref_pooled = ref_scores.max(dim=3).values
    valid_mask = scores_ref_pooled > -1e5
    if valid_mask.sum() > 0:
        match_rate = indices_match[valid_mask].float().mean().item()
        print(f"Indices Match Rate: {match_rate*100:.2f}%")
    else:
        match_rate = 1.0

    if fwd_rel_err < 0.02 and match_rate >= 0.99:
        print(" GQA D-Reshape Forward PASSED")
    else:
        print(" GQA D-Reshape Forward FAILED")

    # ============ Backward ============
    print("\n--- Backward ---")

    grad_output = torch.randn_like(fused_scores, dtype=dtype)

    with torch.no_grad():
        fused_scores_view = fused_scores.view(B, L, h_kv_new, G_new, topk)
        invalid_mask_bwd = fused_scores_view.float() < -1e5
        grad_output_view = grad_output.view(B, L, h_kv_new, G_new, topk)
        grad_output_view[invalid_mask_bwd] = 0.0

    # Fused Backward
    loss_fused = (fused_scores * grad_output).sum()
    loss_fused.backward()
    dq_fused = q_fused.grad.clone()
    dlmks_fused = lmks_fused.grad.clone()

    assert dq_fused.shape == q_raw.shape, f"q grad shape mismatch: {dq_fused.shape} vs {q_raw.shape}"
    assert dlmks_fused.shape == lmks_raw.shape, f"lmks grad shape mismatch: {dlmks_fused.shape} vs {lmks_raw.shape}"
    print(f"q grad shape: {dq_fused.shape} (expected {q_raw.shape})")
    print(f"lmks grad shape: {dlmks_fused.shape} (expected {lmks_raw.shape})")

    q_ref_bwd = q_raw.clone().detach().requires_grad_(True)
    lmks_ref_bwd = lmks_raw.clone().detach().requires_grad_(True)

    # reshape: [B, L, h_q, D_q] -> [B, L, h_kv_new, G_new, D_q]
    q_reshaped_bwd = q_ref_bwd.view(B, L, h_kv_new, G_new, D_q)
    # reshape: [B, S, h_kv_orig, D_lmk] -> [B, S, h_kv_new, D_q]
    lmks_reshaped_bwd = lmks_ref_bwd.reshape(B, S, h_kv_new, D_q)

    scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q_reshaped_bwd.float(), lmks_reshaped_bwd.float())
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D_q))
    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        if window_size > 0:
            aligned_threshold = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        else:
            aligned_threshold = i_idx.div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= aligned_threshold
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    safe_indices_bwd = fused_indices.clone().detach()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded_bwd = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G_new, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded_bwd)

    loss_ref = (scores_gathered_ref * grad_output.view(B, L, h_kv_new, G_new, topk).float()).sum()
    loss_ref.backward()
    dq_ref = q_ref_bwd.grad.clone()
    dlmks_ref = lmks_ref_bwd.grad.clone()

    dq_fused = torch.nan_to_num(dq_fused, 0.0)
    dq_ref = torch.nan_to_num(dq_ref, 0.0)
    dlmks_fused = torch.nan_to_num(dlmks_fused, 0.0)
    dlmks_ref = torch.nan_to_num(dlmks_ref, 0.0)

    dq_rel_err = get_err_ratio(dq_ref.float(), dq_fused.float())
    dlmks_rel_err = get_err_ratio(dlmks_ref.float(), dlmks_fused.float())

    print(f"dQ Rel Error: {dq_rel_err:.6f}")
    print(f"dLmks Rel Error: {dlmks_rel_err:.6f}")

    if dq_rel_err < 0.05 and dlmks_rel_err < 0.05:
        print(" GQA D-Reshape Backward PASSED")
    else:
        print(" GQA D-Reshape Backward FAILED")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Running Per-Chunk Bias / Per-Q-Head Lmks Correctness Tests")
    print("=" * 70)

    # Per-chunk bias on default (per-KV-head) lmks
    test_train_inference_correctness(
        "train_basic_bias", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, False, False
    )
    test_train_inference_correctness(
        "train_basic_bias_gumbel", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, True, False
    )

    test_train_inference_correctness(
        "train_bias_nondiv_999", 2, 999, 999, 2, 8, 128, 16, 64, 64, True, 0, True, False, False
    )
    test_train_inference_correctness(
        "train_bias_nondiv_1023", 2, 1023, 1023, 2, 8, 128, 16, 64, 64, True, 0, True, False, False
    )
    test_train_inference_correctness(
        "train_bias_nondiv_1100", 1, 1100, 1100, 2, 8, 128, 16, 64, 64, True, 0, True, False, False
    )

    # Per-q-head lmks (G is given explicitly)
    test_train_inference_correctness(
        "train_perqhead", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, False, False, True
    )
    test_train_inference_correctness(
        "train_perqhead_bias", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, False, True
    )
    test_train_inference_correctness(
        "train_perqhead_bias_gumbel", 2, 1024, 1024, 2, 8, 128, 16, 64, 64, True, 0, True, True, True
    )
    test_train_inference_correctness(
        "train_perqhead_nondiv_999", 2, 999, 999, 2, 8, 128, 16, 64, 64, True, 0, True, False, True
    )

    test_train_inference_correctness(
        "infer_perqhead_prefill_bias", 1, 1024, 1024, 2, 8, 128, 16, 64, 64, False, 0, True, False, True
    )
    test_train_inference_correctness(
        "infer_perqhead_chunk2_bias", 1, 1024, 2048, 2, 8, 128, 16, 64, 64, False, 1024, True, False, True
    )
    test_train_inference_correctness(
        "infer_perqhead_decode_bias", 1, 1, 2049, 2, 8, 128, 16, 64, 64, False, 2048, True, False, True
    )
    test_train_inference_correctness(
        "train_perqhead_G1_bias", 2, 1024, 1024, 4, 1, 128, 16, 64, 64, True, 0, True, False, True
    )
    test_train_inference_correctness(
        "train_gumbel_nondiv", 2, 999, 999, 2, 8, 128, 16, 64, 64, True, 0, False, True, False
    )
    test_train_inference_correctness(
        "train_perqhead_gumbel_nondiv", 2, 999, 999, 2, 8, 128, 16, 64, 64, True, 0, True, True, True
    )

    # DropMask + per_qhead
    test_drop_mask_correctness(per_qhead_lmks=False)
    test_drop_mask_correctness(per_qhead_lmks=True)

    gqa_cfg = dict(B=4, L=8192, D=128, h_kv=4, G=8, S=128, topk=32,
                   block_size=64, window_size=512, n_iters=20, n_warmup=5,
                   skip_ref=True)
    test_fused_softmax_topk_max_pooling_memory_and_speed(
        name="gqa_shared_lmks", per_qhead_lmks=False, **gqa_cfg
    )
    test_fused_softmax_topk_max_pooling_memory_and_speed(
        name="gqa_perqhead_lmks", per_qhead_lmks=True, **gqa_cfg
    )

    # print("\n" + "=" * 70)
    # print("Running Train/Inference Correctness Tests")
    # print("=" * 70)

    # test_cases = [# test_name, B, q_len, kv_len, h_kv, G, D, topk, block_size, window_size, is_training, q_offset
    #             ("train_basic", 2, 1048, 1048, 2, 8, 128, 16, 64, 64, True, 0),
    #             ("train_large_batch", 4, 2048, 2048, 2, 8, 128, 16, 64, 64, True, 0),
    #             ("train_non_divisible", 2, 999, 999, 2, 8, 128, 16, 64, 64, True, 0),

    #             ("prefill_chunk1", 1, 1024, 1024, 2, 8, 128, 16, 64, 64, False, 0),
    #             ("prefill_chunk2", 1, 1024, 2048, 2, 8, 128, 16, 64, 64, False, 1024),
    #             ("prefill_chunk3", 1, 1024, 3072, 2, 8, 128, 16, 64, 64, False, 2048),
    #             ("prefill_non_divisible", 1, 512, 1536, 2, 8, 128, 16, 64, 64, False, 1024),
    #             ("decode_step1", 1, 1, 1025, 2, 8, 128, 16, 64, 64, False, 1024),
    #             ("decode_step2", 1, 1, 2049, 2, 8, 128, 16, 64, 64, False, 2048),
    #             ("edge_small_topk", 1, 512, 1024, 2, 8, 128, 4, 64, 64, False, 512),
    #             ("edge_large_window", 1, 1024, 2048, 2, 8, 128, 16, 64, 128, False, 1024),
    # ]

    # for params in test_cases:
    #     test_train_inference_correctness(*params)


    # params_list = [
    #     (2, 4096, 64, 2, 8, 128, 16, 64, 64),
    #     (1, 2048, 64, 1, 8, 128, 16, 32, 40),
    #     (3, 2048, 64, 1, 8, 128, 8, 32, 33),
    #     (2, 4096, 64, 1, 8, 128, 32, 64, 100),
    # ]
    # for p in params_list:
    #     test_topk_correctness_robust(*p)

    # test_drop_mask_correctness()

    # test_gqa_d_reshape_correctness()


# python ops/topk_head_softmax.py