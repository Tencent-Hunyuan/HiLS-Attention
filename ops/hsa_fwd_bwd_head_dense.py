import torch
import math
import logging
logging.getLogger("tilelang.jit.kernel").setLevel(logging.WARNING)
logging.getLogger("tilelang").setLevel(logging.WARNING)
import tilelang
from tilelang import language as T




from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int, mask_last_token: bool = False, window_size: int = -1):
    """
    (same formula as test_group_qa):
    ...
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]
    G = HQ // H
    q_blocks = L // block_q
    device = q.device

    if indices.shape[1] != q_blocks:
        idx_view = indices.view(B, q_blocks, block_q, H, -1)
        indices_q = idx_view[:, :, 0, :, :].contiguous()
    else:
        indices_q = indices

    valid_mask = (indices_q >= 0)  # (B, q_blocks, H, K)
    safe_indices = indices_q.clamp_min(0)

    N = L // chunk_size
    valid_L = N * chunk_size

    k_truncated = k[:, :valid_L, :, :]
    v_truncated = v[:, :valid_L, :, :]

    k_chunks = rearrange(k_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)

    idx_flat = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1)  # (B, BqK, 1, h, 1)
    idx_flat = idx_flat.expand(-1, -1, chunk_size, -1, D)                                   # (B, BqK, S, h, D)
    idx_flat = idx_flat.long()
    gather_k = k_chunks.gather(dim=1, index=idx_flat)  # (B, BqK, S, h, D)
    gather_v = v_chunks.gather(dim=1, index=idx_flat)

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)  # (B, Bq, S, K, HQ, D)

    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    # qk: (B, Bq, X, S, K, HQ)
    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())
    qk = qk * float(sm_scale)

    # [Modified] Causal Mask Logic for Reference
    # q_indices: (1, q_blocks, block_q, 1, 1, 1)
    q_indices = torch.arange(L, device=device).view(1, q_blocks, block_q, 1, 1, 1)
    q_real_blk_ids = q_indices // chunk_size

    # k_blk_ids: (B, q_blocks, 1, 1, K, 1)
    k_blk_ids = rearrange(indices_q, 'b q h k -> b q 1 1 k h')
    k_blk_ids = torch.repeat_interleave(k_blk_ids, repeats=G, dim=-1) # (B, q_blocks, 1, 1, K, HQ)


    # Mask: Past only (k < q)
    mask = k_blk_ids < q_real_blk_ids

    # Apply mask to qk
    qk = qk.masked_fill(~mask, float("-inf"))


    if window_size > 0:
        # Window Mask: Mask out chunk if it is within window_size relative to q
        # Valid condition: k_blk_ids < (q_indices - window_size + 1) // chunk_size
        threshold_blk_ids = (q_indices - window_size + 1).div(chunk_size, rounding_mode='floor')
        window_mask = k_blk_ids < threshold_blk_ids
        qk = qk.masked_fill(~window_mask, float("-inf"))

    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)
    p = torch.nan_to_num(p, nan=0.0)

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float() # (B, Bq, HQ, K)
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'b q x h d -> b (q x) h d')
    return o_ref.to(torch.float32)



def make_dq_layout_hsa(DQ):
    return T.Layout(DQ.shape,
        lambda b, l, h, d: [
            b,
            h,
            l // 8,
            d // 8,
            (d % 2),
            4 * (l % 8) + (d % 8) // 2
        ])

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(batch, q_len, heads, head_dim):
    shape = [batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64

    @T.prim_func
    def hsa_post(
            dQ_swizzled: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(q_len, blk), heads, batch, threads=32) as (bx, by, bz):
            i_b = bz

            T.annotate_layout({dQ_swizzled: make_dq_layout_hsa(dQ_swizzled)})

            T.copy(
                dQ_swizzled[i_b, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[i_b, bx * blk:(bx + 1) * blk, by, :],
            )
    return hsa_post




import torch
def build_block_indices_block_M(
    B: int,
    SEQ_LEN: int,
    H: int,
    S: int,
    block_size: int,
    overlap_ratio: float = 0.5,
    block_M: int = 2,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Construct the block_indices tensor:
    - Within each block_M token window, selected block sets for adjacent tokens satisfy the given overlap ratio.
    - Selected indices for each query are sorted ascending.
    - Missing entries up to S are padded with -1.

    Args:
        B: batch size
        SEQ_LEN: sequence length
        H: number of heads
        S: number of blocks selected per query
        block_size: block size
        overlap_ratio: overlap ratio between adjacent tokens [0, 1]
        block_M: number of tokens per window( pair=2,  block_M kernel is M)
        device: output device
    """
    import torch

    assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio must be in [0, 1]"
    assert block_M >= 1, "block_M must be >= 1"

    num_blocks = SEQ_LEN // block_size
    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)

    for b in range(B):
        for h in range(H):
            t = 0
            while t < SEQ_LEN:
                block_start = t
                block_end = min(t + block_M, SEQ_LEN)

                t0 = block_start
                max_blocks_t0 = min(t0 // block_size + 1, num_blocks)
                if max_blocks_t0 <= 0:
                    t = block_end
                    continue

                num_select = min(S, max_blocks_t0)
                idx_prev = torch.randperm(max_blocks_t0, device=device)[:num_select]
                idx_prev_sorted = torch.sort(idx_prev)[0]
                block_indices[b, t0, h, :len(idx_prev_sorted)] = idx_prev_sorted

                for tt in range(t0 + 1, block_end):
                    max_blocks_tt = min(tt // block_size + 1, num_blocks)
                    if max_blocks_tt <= 0:
                        continue

                    num_select_tt = min(S, max_blocks_tt)

                    num_overlap = int(overlap_ratio * num_select_tt)
                    num_overlap = min(num_overlap, len(idx_prev))

                    if num_overlap > 0:
                        perm_prev = torch.randperm(len(idx_prev), device=device)
                        overlap_blocks = idx_prev[perm_prev[:num_overlap]]
                    else:
                        overlap_blocks = idx_prev.new_empty((0,), dtype=idx_prev.dtype)

                    remaining_blocks_all = torch.arange(max_blocks_tt, device=device)
                    mask = torch.ones(max_blocks_tt, dtype=torch.bool, device=device)
                    if overlap_blocks.numel() > 0:
                        mask[overlap_blocks] = False
                    candidates = remaining_blocks_all[mask]

                    num_new = num_select_tt - num_overlap
                    if num_new > 0 and candidates.numel() > 0:
                        perm_cand = torch.randperm(candidates.numel(), device=device)
                        new_blocks = candidates[perm_cand[:num_new]]
                        idx_curr = torch.cat([overlap_blocks, new_blocks], dim=0)
                    else:
                        idx_curr = overlap_blocks.clone()

                    idx_curr_sorted = torch.sort(idx_curr)[0]
                    block_indices[b, tt, h, :len(idx_curr_sorted)] = idx_curr_sorted

                    idx_prev = idx_curr

                t = block_end

    return block_indices



@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_block_M(batch, heads, q_len, kv_len, head_dim,
                                          scale=None, block_size=64, groups=16,
                                          selected_blocks=16, num_weights=None,  block_M = 64, mask_last_token=True,
                                          window_size=-1, dtype = "bfloat16", accum_dtype = "float", num_threads = 128):

    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504
    else:
        scale = scale * 1.44269504

    # [Modified] Grid setup uses total query heads
    q_shape = [batch, q_len, heads, head_dim]

    # KV shape uses kv_heads
    head_kv = heads // groups
    kv_shape = [batch, kv_len, head_kv, head_dim]

    num_kv_blocks = kv_len // block_size

    if num_weights is None:
        num_weights = num_kv_blocks
    # weight_shape = [batch, q_len, heads, num_kv_blocks]
    weight_shape = [batch, q_len, heads, num_weights]
    scores_lse_shape = [batch, q_len, heads, num_weights]

    # dtype = "bfloat16"
    # accum_dtype = "float"

    # [Modified] block_M is now independent of groups
    if block_M is None or block_M <= 0:
        block_M = 64
    # block_M = 64
    print("Using block_M =", block_M, "for fwd_block_M kernel (Head-Parallel)")

    BS = block_size
    BK = BV = min(128, tilelang.math.next_power_of_2(head_dim))

    num_stages = 2
    # threads = 128

    @T.prim_func
    def hsa_block_M(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        # [Modified] Grid Z = batch * heads (One block per Query Head)
        with T.Kernel(tilelang.cdiv(q_len, block_M), batch * heads, threads=num_threads) as (bx, bz):
            # Shared Memory (No groups dimension)
            Q_shared = T.alloc_shared([block_M, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([block_M, BV], dtype)

            acc_s = T.alloc_fragment([block_M, BS], accum_dtype)
            acc_o = T.alloc_fragment([block_M, BV], accum_dtype)

            P_shared = T.alloc_shared([block_M, BS], dtype)
            W_curr_shared = T.alloc_fragment([block_M], dtype)

            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)

            i_t_base_idx, i_bh = bx, bz

            i_b = i_bh // heads
            i_h = i_bh % heads

            i_h_kv = i_h // groups

            base_t = i_t_base_idx * block_M
            q_blk_idx = base_t // BS

            pipeline_limit = T.alloc_var("int32")
            pipeline_limit = q_blk_idx
            if window_size > 0:
                max_t = base_t + block_M - 1
                pipeline_limit  = T.floordiv(max_t - window_size + 1, BS)
                pipeline_limit = T.max(pipeline_limit, 0)

            T.copy(Q[i_b, base_t:base_t + block_M, i_h, :], Q_shared)

            T.fill(acc_o, 0)
            T.sync_threads()

            for k_blk_idx in T.Pipelined(pipeline_limit, num_stages=num_stages):

                i_s = k_blk_idx * BS

                T.copy(K[i_b, i_s:i_s + BS, i_h_kv, :], K_shared)
                T.copy(V[i_b, i_s:i_s + BS, i_h_kv, :], V_shared)

                T.copy(W[i_b, base_t:base_t + block_M, i_h, k_blk_idx], W_curr_shared)

                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                for r, c in T.Parallel(block_M, BS):
                    acc_s[r, c] = T.if_then_else(c == BS - 1 and enable_last_token_mask, -T.infinity(accum_dtype), acc_s[r, c])

                if window_size > 0:
                    for r, c in T.Parallel(block_M, BS):
                        acc_s[r, c] = T.if_then_else(
                            k_blk_idx >= T.floordiv(base_t + r - window_size + 1, BS),
                            -T.infinity(accum_dtype),
                            acc_s[r, c]
                        )

                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                for r, c in T.Parallel(block_M, BS):
                    acc_s[r, c] = T.if_then_else(
                        scores_max[r] == -T.infinity(accum_dtype),
                        0.0,
                        T.exp2(acc_s[r, c] * scale - scores_max[r] * scale)
                    )
                T.fill(scores_sum, 0.0)
                T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)

                for r in T.Parallel(block_M):
                    tq = base_t + r
                    if tq < q_len:
                        ScoresLSE[i_b, tq, i_h, k_blk_idx] = T.if_then_else(
                            scores_sum[r] > 0,
                            scores_max[r] * scale + T.log(scores_sum[r]) * 1.44269504,
                            -T.infinity(accum_dtype),
                        )

                for r, c in T.Parallel(block_M, BS):
                    acc_s[r, c] = T.if_then_else(
                        scores_sum[r] > 0,
                        acc_s[r, c] * W_curr_shared[r] / scores_sum[r],
                        0.0
                    )
                T.copy(acc_s, P_shared)
                T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)

            T.copy(O_shared, Output[i_b, base_t : base_t + block_M, i_h, :])

    return hsa_block_M


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dqkv_block_M(
    batch, heads, q_len, kv_len, head_dim,
    scale=None, block_size=64, groups=16, selected_blocks=16, num_weights=None,
    block_M = 0, mask_last_token=True, window_size = -1, dtype="bfloat16", accum_dtype="float", num_threads = 256
):
    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale
    scale_log2 = sm_scale * 1.44269504

    B = batch
    BS = block_size
    G = groups
    Vdim = head_dim
    Kdim = head_dim
    BK = tilelang.next_power_of_2(Kdim)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    NS_kv = kv_len // BS

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    do_shape = [batch, q_len, heads, head_dim]

    dq_shape = [batch, q_len, heads, head_dim]

    dk_shape = [groups, batch, kv_len, heads_kv, head_dim]
    dv_shape = [groups, batch, kv_len, heads_kv, head_dim]

    if num_weights is None:
        num_weights = NS_kv

    weight_shape = [batch, q_len, heads, num_weights]
    dw_shape = [batch, q_len, heads, num_weights]
    scores_lse_shape = [batch, q_len, heads, num_weights]

    if block_M is None or block_M <= 0:
        block_M = 64
    print("Using block_M =", block_M, "for bwd_block_M kernel (Head-Parallel)")

    M = block_M
    NP = tilelang.cdiv(q_len, M)

    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv_block_M(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, dtype),
        DO: T.Tensor(do_shape, dtype),
        ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
        DQ: T.Tensor(dq_shape, accum_dtype),
        DK: T.Tensor(dk_shape, dtype),
        DV: T.Tensor(dv_shape, dtype),
        DW: T.Tensor(dw_shape, dtype),
    ):
        with T.Kernel(NS_kv, B * heads, threads=num_threads) as (i_s, i_bh):
            i_b = i_bh // heads
            i_h = i_bh % heads
            i_h_kv = i_h // groups
            g_idx = i_h % groups

            i_s_global = i_s * BS

            S_buf = T.alloc_shared([M, BS], dtype)
            dO_buf = T.alloc_shared([M, BV], dtype)

            P_shared = S_buf
            dS_shared = S_buf

            dO_shared =  dO_buf
            dO_weighted_shared = dO_buf

            Q_shared = T.alloc_shared([M, BK], dtype)

            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)

            dK_shared = T.alloc_shared([BS, BK], dtype)
            dV_shared = T.alloc_shared([BS, BV], dtype)

            T_raw_frag = T.alloc_fragment([M, BV], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M, BS], dtype)
            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dQ_local = T.alloc_fragment([M, BK], accum_dtype)
            delta_rows = T.alloc_fragment([M], accum_dtype)

            acc_s_tmp = T.alloc_fragment([M, BS], accum_dtype)
            saved_lse = T.alloc_fragment([M], accum_dtype)

            dw_row_sum_frag = T.alloc_fragment([M], accum_dtype)
            dw_row_sum_shared = T.alloc_shared([M], accum_dtype)

            W_local = T.alloc_shared([M], dtype)

            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h_kv, :], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h_kv,:], V_shared)
            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            T.annotate_layout({
                DQ: make_dq_layout_hsa(DQ),
            })
            ip_start = T.floordiv(i_s * BS, M) + 1
            for ip in T.Pipelined(ip_start, NP, num_stages=num_stages):
                base_t = ip * M
                q_blk_idx = base_t // BS

                if q_blk_idx > i_s:
                    T.copy(Q[i_b, base_t:base_t+M, i_h, :], Q_shared)
                    T.copy(DO[i_b, base_t:base_t+M, i_h, :], dO_shared)

                    T.clear(acc_s_tmp)
                    T.gemm(Q_shared, K_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for i, s in T.Parallel(M, BS):
                        tq = base_t + i
                        acc_s_tmp[i, s] = T.if_then_else(
                        (tq >= q_len) or (enable_last_token_mask & (s == BS - 1)) or (window_size > 0 and i_s >= T.floordiv(tq - window_size + 1, BS)),
                        -T.infinity(accum_dtype),
                        acc_s_tmp[i, s]
                    )

                    for i in T.Parallel(M):
                        tq = base_t + i
                        saved_lse[i] = T.if_then_else(
                            tq < q_len,
                            ScoresLSE[i_b, tq, i_h, i_s],
                            -T.infinity(accum_dtype),
                        )

                    for i, s in T.Parallel(M, BS):
                        tq = base_t + i
                        p_val = T.if_then_else(
                            (tq < q_len) & (saved_lse[i] != -T.infinity(accum_dtype)),
                            T.exp2(acc_s_tmp[i, s] * scale_log2 - saved_lse[i]),
                            0.0,
                        )
                        acc_s_tmp[i, s] = p_val
                        P_shared[i, s] = p_val


                    T.copy(W[i_b, base_t:base_t + M, i_h, i_s], W_local)

                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for r, s in T.Parallel(M, BS):
                        acc_s_tmp[r, s] = acc_s_tmp[r, s] * dV_PdO_frag[r, s]

                    T.reduce_sum(acc_s_tmp, dw_row_sum_frag, dim=1, clear=True)
                    T.copy(dw_row_sum_frag, dw_row_sum_shared)
                    T.copy(dw_row_sum_shared, DW[i_b, base_t:base_t + M, i_h, i_s])

                    for row_idx, v in T.Parallel(M, BV):
                        dO_weighted_shared[row_idx, v] = W_local[row_idx] * dO_shared[row_idx, v]

                    for g_row, s in T.Parallel(M, BS):
                        dV_PdO_frag[g_row, s] = dV_PdO_frag[g_row, s] * W_local[g_row]

                    T.gemm(P_shared, dO_weighted_shared, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    for g_row, s in T.Parallel(M, BS):
                        acc_s_tmp[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                    T.reduce_sum(acc_s_tmp, delta_rows, dim=1, clear=True)

                    for g_row, s in T.Parallel(M, BS):
                        dS_frag[g_row, s] = sm_scale * (acc_s_tmp[g_row, s] - P_shared[g_row, s] * delta_rows[g_row])

                    T.copy(dS_frag, dS_shared)

                    T.gemm(dS_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    T.clear(dQ_local)
                    T.gemm(dS_shared, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)

                    for i, k in T.Parallel(M, BK):
                        tq = base_t + i
                        if tq < q_len:
                            T.atomic_add(DQ[i_b, tq, i_h, k], dQ_local[i, k])

            T.copy(dK_accum, dK_shared)
            T.copy(dV_accum, dV_shared)

            T.copy(dK_shared, DK[g_idx, i_b, i_s_global:i_s_global + BS, i_h_kv, :])
            T.copy(dV_shared, DV[g_idx, i_b, i_s_global:i_s_global + BS, i_h_kv, :])

    return hsa_bwd_dqkv_block_M




class _hsa_block_M_attention_dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W, block_size, sm_scale, block_M, mask_last_token, window_size,
                dtype, accum_dtype, num_threads):
        # Q: [B, L, HQ, D]
        # K, V: [B, L, H, D]
        # W: [B, L, HQ, S]
        B, SEQ_LEN, HQ, D = Q.shape
        H = K.shape[2]
        groups = HQ // H
        num_weights = W.shape[-1]

        if block_M is None or block_M <= 0:
            block_M = 64

        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.mask_last_token = mask_last_token
        ctx.window_size = window_size
        ctx.groups = groups
        ctx.dtype = dtype
        ctx.accum_dtype = accum_dtype
        ctx.num_threads = num_threads
        ctx.num_weights = num_weights

        fwd_kernel = hierarchical_sparse_attention_block_M(
            batch=B, heads=HQ, q_len=SEQ_LEN, kv_len=SEQ_LEN, head_dim=D,
            scale=sm_scale, block_size=block_size, groups=groups,
            selected_blocks=0, num_weights=num_weights,
            block_M=block_M, mask_last_token=mask_last_token, window_size=window_size,
            dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
        )

        scores_lse = torch.full(
            (B, SEQ_LEN, HQ, num_weights),
            float("-inf"),
            dtype=torch.float32,
            device=Q.device,
        )
        O = fwd_kernel(Q, K, V, W, scores_lse)

        ctx.save_for_backward(Q, K, V, W, scores_lse)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, W, scores_lse = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        mask_last_token = ctx.mask_last_token
        window_size = ctx.window_size
        dtype = ctx.dtype
        accum_dtype = ctx.accum_dtype
        num_threads = ctx.num_threads
        groups = ctx.groups
        num_weights = ctx.num_weights

        B, SEQ_LEN, HQ, D = Q.shape
        H = K.shape[2]
        dq_shape = [B, SEQ_LEN, HQ, D]
        dk_shape = [groups, B, SEQ_LEN, H, D]
        dv_shape = [groups, B, SEQ_LEN, H, D]
        dw_shape = [B, SEQ_LEN, HQ, num_weights]

        DQ = torch.zeros(dq_shape, dtype=torch.float32, device=Q.device)
        DW = torch.zeros(dw_shape, dtype=W.dtype, device=Q.device)
        DK = torch.zeros(dk_shape, dtype=K.dtype, device=Q.device)
        DV = torch.zeros(dv_shape, dtype=V.dtype, device=Q.device)

        hierarchical_sparse_attention_bwd_dqkv_block_M(
            batch=B, heads=HQ, q_len=SEQ_LEN, kv_len=SEQ_LEN, head_dim=D,
            scale=sm_scale, block_size=block_size, groups=groups,
            selected_blocks=0, num_weights=num_weights,
            block_M=block_M, mask_last_token=mask_last_token, window_size=window_size,
            dtype=dtype, accum_dtype=accum_dtype, num_threads=num_threads,
        )(Q, K, V, W, grad_output, scores_lse, DQ, DK, DV, DW)

        post_kernel = hsa_bwd_postprocess(B, SEQ_LEN, HQ, D)
        DQ = post_kernel(DQ).to(Q.dtype)
        DK = DK.sum(dim=0).to(K.dtype)
        DV = DV.sum(dim=0).to(V.dtype)

        return DQ, DK, DV, DW, None, None, None, None, None, None, None, None

def HSA_block_M_head_dense(Q, K, V, W, block_size=64, sm_scale=None, block_M=64, mask_last_token=True,
                           window_size=-1,
                           dtype="bfloat16", accum_dtype="float", num_threads=128):
    return _hsa_block_M_attention_dense.apply(
        Q, K, V, W, block_size, sm_scale, block_M, mask_last_token, window_size,
        dtype, accum_dtype, num_threads,
    )




def _create_chunk_window_mask(
    L: int,
    num_chunks: int,
    chunk_size: int,
    window_size: int,
    device,
    q_offset: int = 0,
    is_causal: bool = True,
):
    if not is_causal:
        return torch.ones((L, num_chunks), dtype=torch.bool, device=device)
    q_idx = torch.arange(L, device=device) + q_offset
    chunk_idx = torch.arange(num_chunks, device=device)
    if window_size > 0:
        threshold = (q_idx - window_size + 1).div(chunk_size, rounding_mode="floor")
    else:
        threshold = q_idx.div(chunk_size, rounding_mode="floor")
    return chunk_idx[None, :] < threshold[:, None]


def _dense_head_layout(q: torch.Tensor, k: torch.Tensor, lmk: torch.Tensor, G: int = None):
    B, L, HQ, D = q.shape
    H = k.shape[2]
    lmks_h = lmk.shape[2]
    D_lmk = lmk.shape[3]
    if D_lmk != D:
        assert D_lmk % D == 0, f"lmk D dim ({D_lmk}) must be divisible by q D dim ({D})"
        d_ratio = D_lmk // D
        lmk = lmk.reshape(lmk.shape[0], lmk.shape[1], lmks_h * d_ratio, D)
        lmks_h *= d_ratio
    if G is None:
        assert HQ % H == 0, f"HQ ({HQ}) must be divisible by H ({H})"
        G_eff = HQ // H
        if lmks_h == H:
            per_qhead_lmks = False
        elif lmks_h == HQ:
            per_qhead_lmks = True
        else:
            raise AssertionError(f"lmk heads ({lmks_h}) must be H ({H}) or HQ ({HQ})")
    else:
        assert HQ % G == 0, f"HQ ({HQ}) must be divisible by G ({G})"
        H_from_G = HQ // G
        assert H_from_G == H, f"G ({G}) implies H ({H_from_G}), but K has H ({H})"
        assert lmks_h == HQ, f"when G is given, lmk must have HQ ({HQ}) heads, got {lmks_h}"
        G_eff = G
        per_qhead_lmks = True
    return lmk, H, G_eff, per_qhead_lmks


def _reshape_lse_swa_head(lse_swa: torch.Tensor, B: int, L: int, H: int, G: int):
    HQ = H * G
    if lse_swa.dim() == 3:
        assert lse_swa.shape == (B, L, HQ), f"lse_swa shape {tuple(lse_swa.shape)} != {(B, L, HQ)}"
        return lse_swa
    assert lse_swa.shape == (B, L, H, G), f"lse_swa shape {tuple(lse_swa.shape)} != {(B, L, H, G)}"
    return lse_swa.reshape(B, L, HQ)


def _reshape_bias_head(bias: torch.Tensor, B: int, S: int, H: int, G: int):
    if bias is None:
        return None
    HQ = H * G
    if bias.dim() == 3:
        assert bias.shape == (B, S, HQ), f"bias shape {tuple(bias.shape)} != {(B, S, HQ)}"
        return bias.reshape(B, S, H, G).permute(0, 2, 3, 1).reshape(B, 1, HQ, S)
    assert bias.dim() == 4, f"bias must be [B, S, HQ] or [B, S, H, G], got {tuple(bias.shape)}"
    assert bias.shape == (B, S, H, G), f"bias shape {tuple(bias.shape)} != {(B, S, H, G)}"
    return bias.permute(0, 2, 3, 1).reshape(B, 1, HQ, S)


def _compute_dense_chunk_scores_head(
    q: torch.Tensor,
    k: torch.Tensor,
    lmk: torch.Tensor,
    block_size: int,
    window_size: int,
    *,
    lmk_q: torch.Tensor = None,
    bias: torch.Tensor = None,
    q_offset: int = 0,
    is_causal: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    G: int = None,
):
    score_q = lmk_q if lmk_q is not None else q
    B, L, HQ, D = score_q.shape
    S = lmk.shape[1]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    lmk, H, G_eff, per_qhead_lmks = _dense_head_layout(score_q, k, lmk, G=G)
    q_view = score_q.reshape(B, L, H, G_eff, D)
    if per_qhead_lmks:
        lmk_view = lmk.reshape(B, S, H, G_eff, D)
        scores = torch.einsum("blhgd,bshgd->blhgs", q_view, lmk_view)
    else:
        scores = torch.einsum("blhgd,bshd->blhgs", q_view, lmk)
    scores = scores.float() * float(sm_scale)
    allow = _create_chunk_window_mask(L, S, block_size, window_size, q.device, q_offset=q_offset, is_causal=is_causal)
    scores = scores.masked_fill(~allow.view(1, L, 1, 1, S), float("-inf"))
    if drop_mask is not None:
        assert drop_mask.shape == (B, L, S), f"drop_mask shape {tuple(drop_mask.shape)} != {(B, L, S)}"
        scores = scores.masked_fill(drop_mask.bool().view(B, L, 1, 1, S), float("-inf"))
    scores = scores.reshape(B, L, HQ, S)
    bias_view = _reshape_bias_head(bias, B, S, H, G_eff)
    if bias_view is not None:
        scores = scores + bias_view.to(device=scores.device, dtype=scores.dtype)
    return scores.to(q.dtype), H, G_eff


def HSA_dense_interface(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lmk: torch.Tensor,
    lse_swa: torch.Tensor,
    block_size: int,
    window_size: int,
    enable_softmax1: bool = True,
    mask_last_token: bool = True,
    lmk_q: torch.Tensor = None,
    bias: torch.Tensor = None,
    q_offset: int = 0,
    is_causal: bool = True,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    G: int = None,
):
    B, L, HQ, D = q.shape
    scores, H, G_eff = _compute_dense_chunk_scores_head(
        q, k, lmk, block_size, window_size,
        lmk_q=lmk_q, bias=bias, q_offset=q_offset, is_causal=is_causal,
        drop_mask=drop_mask, sm_scale=sm_scale, G=G,
    )
    lse_last = _reshape_lse_swa_head(lse_swa, B, L, H, G_eff).to(scores.dtype).unsqueeze(-1)
    if not enable_softmax1:
        cat_scores = torch.cat([scores, lse_last], dim=-1)
        lse_idx = -1
    else:
        ones = torch.zeros((B, L, HQ, 1), device=q.device, dtype=scores.dtype)
        cat_scores = torch.cat([scores, lse_last, ones], dim=-1)
        lse_idx = -2
    chunk_weights_all = torch.softmax(cat_scores, dim=-1)

    out = HSA_block_M_head_dense(
        Q=q,
        K=k,
        V=v,
        W=chunk_weights_all.to(q.dtype).contiguous(),
        block_size=block_size,
        sm_scale=sm_scale,
        mask_last_token=mask_last_token,
        window_size=window_size,
    )
    return out, chunk_weights_all, lse_idx

import torch.nn.functional as F
from ops.topk_head_softmax import online_softmax_topk_head
from ops.hsa_fwd_bwd_head import HSA_block_M_head


def _add_gathered_bias_to_topk_scores(scores: torch.Tensor, indices: torch.Tensor, bias: torch.Tensor, H: int, G: int):
    if bias is None:
        return scores
    B, L, HQ, K_topk = scores.shape
    S = bias.shape[1]
    bias_hq = _reshape_bias_head(bias, B, S, H, G).reshape(B, HQ, S).permute(0, 2, 1)
    indices_hq = indices.repeat_interleave(G, dim=2) if indices.shape[2] != HQ else indices
    safe_idx = indices_hq.clamp_min(0).long()
    src = bias_hq.unsqueeze(-1).expand(-1, -1, -1, K_topk)
    gathered = torch.gather(src, dim=1, index=safe_idx)
    return scores + gathered.to(scores.dtype)


def HSA_dense_interface_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lmk: torch.Tensor,
    lse_swa: torch.Tensor,
    block_size: int,
    window_size: int,
    is_causal: bool = True,
    enable_softmax1: bool = True,
    mask_last_token: bool = True,
    lmk_q: torch.Tensor = None,
    bias: torch.Tensor = None,
    q_offset: int = 0,
    drop_mask: torch.Tensor = None,
    sm_scale: float = None,
    G: int = None,
    topk: int = None,
):
    B, L, HQ, D = q.shape
    S = lmk.shape[1]
    lmk, H, G_eff, per_qhead_lmks = _dense_head_layout(q, k, lmk, G=G)
    topk = S if topk is None else topk
    topk_power_of_2 = 1 << (topk - 1).bit_length() if topk > 0 else 1
    G_arg = G_eff if per_qhead_lmks else None
    score_q = lmk_q if lmk_q is not None else q

    indices, scores = online_softmax_topk_head(
        q=score_q,
        lmks=lmk,
        lse_swa=lse_swa,
        topk=topk_power_of_2,
        block_size=block_size,
        window_size=window_size,
        is_causal=is_causal,
        q_offset=q_offset,
        drop_mask=drop_mask,
        sm_scale=sm_scale,
        bias=bias,
        G=G_arg,
    )
    scores = _add_gathered_bias_to_topk_scores(scores, indices, bias, H, G_eff)

    lse_last = _reshape_lse_swa_head(lse_swa, B, L, H, G_eff).to(scores.dtype).unsqueeze(-1)
    if not enable_softmax1:
        cat_scores = torch.cat([scores, lse_last], dim=-1)
        lse_idx = -1
    else:
        ones = torch.zeros((B, L, HQ, 1), device=q.device, dtype=scores.dtype)
        cat_scores = torch.cat([scores, lse_last, ones], dim=-1)
        lse_idx = -2
    chunk_weights_all = F.softmax(cat_scores, dim=-1)

    out = HSA_block_M_head(
        q.contiguous(), k.contiguous(), v.contiguous(),
        weights=chunk_weights_all.to(q.dtype).contiguous(),
        indices=indices.contiguous(),
        block_size=block_size,
        sm_scale=sm_scale,
        mask_last_token=mask_last_token,
    )
    return out, chunk_weights_all, lse_idx



import pytest

@pytest.mark.parametrize("B,L,H,HQ,D,block_size,window_size,is_causal,enable_softmax1,use_topk_kernel,per_qhead_lmks,use_bias,bias_dim,use_drop_mask", [
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, False, 3, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, True,  True,  False, 3, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, True),
    (2, 1024,2, 16, 128,64, 256, True,  False, False, True,  True,  3, False),
    (1, 512, 1, 8,  64, 64, 64,  True,  True,  False, True,  True,  4, False),
    (1, 512, 1, 8,  64, 64, 128, True,  False, False, True,  True,  3, True),
])
def test_dense_interface_vs_ref(
    B, L, H, HQ, D, block_size, window_size, is_causal, enable_softmax1, use_topk_kernel,
    per_qhead_lmks, use_bias, bias_dim, use_drop_mask
):
    device = "cuda"
    torch.manual_seed(0)

    num_chunks = L // block_size
    assert HQ % H == 0
    print(
        f"\n[test_dense_interface_vs_ref] B={B} L={L} H={H} HQ={HQ} D={D} "
        f"block_size={block_size} window_size={window_size} is_causal={is_causal} "
        f"enable_softmax1={enable_softmax1} use_topk_kernel={use_topk_kernel} "
        f"per_qhead_lmks={per_qhead_lmks} use_bias={use_bias} bias_dim={bias_dim} "
        f"use_drop_mask={use_drop_mask} num_chunks={num_chunks}"
    )

    q = torch.randn((B, L, HQ, D), device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((B, L, H, D),  device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((B, L, H, D),  device=device, dtype=torch.bfloat16, requires_grad=True)

    lmk_q = (q.detach() + 0.1 * torch.randn_like(q)).requires_grad_(True)

    lmk_heads = HQ if per_qhead_lmks else H
    lmk = torch.randn((B, num_chunks, lmk_heads, D), device=device, dtype=torch.bfloat16)
    lse_swa = torch.randn((B, L, HQ), device=device, dtype=torch.bfloat16)
    if use_bias:
        if bias_dim == 4:
            bias = torch.randn((B, num_chunks, H, HQ // H), device=device, dtype=torch.float32)
        else:
            bias = torch.randn((B, num_chunks, HQ), device=device, dtype=torch.float32)
    else:
        bias = None
    G_arg = HQ // H if per_qhead_lmks else None
    if use_drop_mask:
        allow = _create_chunk_window_mask(L, num_chunks, block_size, window_size, q.device, is_causal=is_causal)
        drop_mask = (torch.rand((B, L, num_chunks), device=device) < 0.2) & allow.view(1, L, num_chunks)
        has_visible = allow.any(dim=-1).view(1, L, 1)
        visible_keep = allow.view(1, L, num_chunks) & (~drop_mask)
        bad = has_visible & (~visible_keep.any(dim=-1, keepdim=True))
        drop_mask = torch.where(bad, torch.zeros_like(drop_mask), drop_mask).to(torch.int32)
    else:
        drop_mask = None

    out_hsa, w_hsa, lse_idx_hsa = HSA_dense_interface(
        q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
        block_size=block_size, window_size=window_size,
        enable_softmax1=enable_softmax1,
        lmk_q=lmk_q,
        bias=bias,
        drop_mask=drop_mask,
        G=G_arg,
    )

    out_ref, w_ref, lse_idx_ref = HSA_dense_interface_ref(
        q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
        block_size=block_size, window_size=window_size, is_causal=is_causal,
        enable_softmax1=enable_softmax1,
        lmk_q=lmk_q,
        bias=bias,
        drop_mask=drop_mask,
        G=G_arg,
    )

    def rms(x):
        return x.float().flatten().square().mean().sqrt()

    def check(name, a, b, thr):
        diff = (a.float() - b.float())
        ratio = (diff.flatten().square().mean().sqrt() / (rms(b) + 1e-12)).item()
        mx = diff.abs().max().item()
        print(f"[{name}] ratio={ratio:.6f} max_abs={mx:.6e}")
        assert ratio < thr, f"{name} mismatch ratio={ratio}, max_abs={mx}"

    S = num_chunks
    if use_drop_mask:
        check("ChunkWeightMass(einsum vs topk)", w_hsa[:, :, :, :S].sum(dim=-1), w_ref[:, :, :, :lse_idx_ref].sum(dim=-1), thr=5e-3)
    else:
        check("Weights(einsum vs topk)", w_hsa[:, :, :, :S], w_ref[:, :, :, :S], thr=5e-3)

    diff = (out_hsa.float() - out_ref.float())
    ratio = (diff.flatten().square().mean().sqrt() / (rms(out_ref) + 1e-12)).item()
    max_abs = diff.abs().max().item()
    print(f"[FWD output] ratio={ratio:.6f} max_abs={max_abs:.6e}")
    assert ratio < 5e-2, f"FWD mismatch ratio={ratio}, max_abs={max_abs}"

    grad = torch.randn_like(out_hsa)
    (out_hsa * grad).sum().backward()
    dq_hsa, dk_hsa, dv_hsa = q.grad.clone(), k.grad.clone(), v.grad.clone()

    q.grad = k.grad = v.grad = None
    (out_ref * grad).sum().backward()
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()

    check("DQ", dq_hsa, dq_ref, thr=5e-3)
    check("DK", dk_hsa, dk_ref, thr=5e-3)
    check("DV", dv_hsa, dv_ref, thr=5e-3)









def main_block_M_correctness():
    """
     HSA_pair FWD/BWD numerical correctness
    and hsa_torch_ref compare (Dense Mode)
    """
    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange

    def print_max_err_compare(name, tensor_hsa, tensor_ref):
        if tensor_hsa is None or tensor_ref is None:
            print(f"\n[{name} Error Analysis]: Tensor is None")
            return

        hsa_f = tensor_hsa.float()
        ref_f = tensor_ref.float()
        diff_abs = (hsa_f - ref_f).abs()

        max_abs_err = diff_abs.max().item()
        indices = torch.where(diff_abs == max_abs_err)
        idx_abs = tuple(idx[0].item() for idx in indices)

        diff_rel = diff_abs / (ref_f.abs() + 1e-6)
        max_rel_err = diff_rel.max().item()
        indices_rel = torch.where(diff_rel == max_rel_err)
        idx_rel = tuple(idx[0].item() for idx in indices_rel)

        print(f"\n[{name} Error Analysis]:")
        print(f"  -> Max Absolute Error: {max_abs_err:.6e} at {idx_abs}")
        print(f"     HSA Val: {hsa_f[idx_abs].item():.10f} | Ref Val: {ref_f[idx_abs].item():.10f}")

        print(f"  -> Max Relative Error: {max_rel_err:.6e} at {idx_rel}")
        print(f"     HSA Val: {hsa_f[idx_rel].item():.10f} | Ref Val: {ref_f[idx_rel].item():.10f}")

    B, SEQ_LEN, H, HQ, D, block_size = 1, 512, 1, 8, 128, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M=64
    mask_last_token=True
    window_size = 128
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    num_kv_blocks = SEQ_LEN // block_size

    print(f"Correctness Config (Dense): Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, NumKVBlocks={num_kv_blocks}, BlockSize={block_size}, WindowSize={window_size}")

    torch.manual_seed(42)

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, SEQ_LEN, HQ, num_kv_blocks), dtype=dtype, device=device)

    # Causal Mask for Weights with Window Size Logic
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)

    # Logic: k_blk_idx < (q_idx - window_size + 1) // block_size
    if window_size > 0:
        threshold_blk = (q_indices - window_size + 1).div(block_size, rounding_mode='floor')
    else:
        threshold_blk = q_indices // block_size

    weight_mask = k_blk_indices < threshold_blk

    logits = logits.masked_fill(~weight_mask, -1e4)
    W = F.softmax(logits, dim=-1).requires_grad_(True)  # leaf tensor

    block_indices_dense = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    block_indices_dense = block_indices_dense.view(1, 1, 1, num_kv_blocks).expand(B, SEQ_LEN, H, num_kv_blocks)

    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)


    O_hsa = HSA_block_M_head_dense(Q, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)

    O_ref = hsa_torch_ref(
        Q.float().detach(),
        K.float().detach(),
        V.float().detach(),
        W.detach(),
        block_indices_dense,
        chunk_size=block_size,
        sm_scale=scale,
        block_q=1,
        mask_last_token=mask_last_token,
        window_size=window_size
    )

    print("[Tilelang HSA_block_M Dense] vs [Torch Reference]:")
    print_max_err_compare("Forward Output", O_hsa, O_ref)



    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None

    O_ref_bwd= hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices_dense,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
        window_size=window_size
    )
    O_ref_bwd.backward(grad_output.float())

    DQ_ref = Q.grad.clone()
    DK_ref = K.grad.clone()
    DV_ref = V.grad.clone()
    DW_ref = W.grad.clone()

    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None

    O_hsa_bwd = HSA_block_M_head_dense(Q, K, V, W,
                         block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
    O_hsa_bwd.backward(grad_output)

    DQ_hsa = Q.grad.clone()
    DK_hsa = K.grad.clone()
    DV_hsa = V.grad.clone()
    DW_hsa = W.grad.clone()

    print_max_err_compare("DQ", DQ_hsa, DQ_ref)
    print_max_err_compare("DK", DK_hsa, DK_ref)
    print_max_err_compare("DV", DV_hsa, DV_ref)
    print_max_err_compare("DW", DW_hsa, DW_ref)



def main_block_M_latency():
    """
    test tilelang HSA_block_M (Dense) of FWD, BWD,  (FWD+BWD) latency
    """
    import torch
    import torch.nn.functional as F
    import time
    import math
    from einops import rearrange

    B, SEQ_LEN, H, HQ, D, block_size = 16, 4096, 1, 8, 128, 64
    block_M=64
    mask_last_token=True
    window_size = 128
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    num_kv_blocks = SEQ_LEN // block_size

    print(f"Latency Config (Dense): B={B}, L={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, NumKVBlocks={num_kv_blocks}, block={block_size}, G={G}, block_M={block_M}, mask_last_token={mask_last_token}, window_size={window_size}")

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, SEQ_LEN, HQ, num_kv_blocks), dtype=torch.bfloat16, device=device)

    # Causal Mask for Weights
    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)

    if window_size > 0:
        threshold_blk = (q_indices - window_size + 1).div(block_size, rounding_mode='floor')
    else:
        threshold_blk = q_indices // block_size

    weight_mask = k_blk_indices < threshold_blk

    logits = logits.masked_fill(~weight_mask, -1e4)
    W = F.softmax(logits, dim=-1).requires_grad_(True)


    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    # ---------- TileLang ----------
    Q_tile = Q.detach().clone().requires_grad_(True)
    grad_output_tile = grad_output

    num_warmup = 20
    num_iters = 50

    # =========================================================
    # Helper function for timing
    # =========================================================
    def measure_time(func, *args, **kwargs):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_iters):
            func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / num_iters  # ms per iter


    # =========================================================
    # =========================================================
    for _ in range(num_warmup):
        O = HSA_block_M_head_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
        O.backward(grad_output_tile)

    def tile_fwd():
        with torch.no_grad():
            O = HSA_block_M_head_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
    tile_fwd_ms = measure_time(tile_fwd)

    def tile_fwd_bwd():
        Q_tile.grad = K.grad = V.grad = W.grad = None
        O = HSA_block_M_head_dense(Q_tile, K, V, W, block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token, window_size=window_size)
        O.backward(grad_output_tile)
    tile_total_ms = measure_time(tile_fwd_bwd)

    tile_bwd_ms = tile_total_ms - tile_fwd_ms

    # =========================================================
    # =========================================================
    print(f"[TileLang Dense] FWD: {tile_fwd_ms:.3f} ms | BWD: {tile_bwd_ms:.3f} ms | Total: {tile_total_ms:.3f} ms")
    print()



# ...existing code...

import pytest
import torch
import math
import torch.nn.functional as F

@pytest.mark.parametrize("B, SEQ_LEN, H, HQ, D, block_size, window_size", [
    (1, 512, 1, 8, 64, 64, 128),
    (2, 1024, 2, 16, 128, 64, 256),
    (1, 512, 1, 8, 64, 64, -1), # Disable window size
])
def test_correctness_fp32(B, SEQ_LEN, H, HQ, D, block_size, window_size):
    device = "cuda"
    dtype = torch.float32
    scale = 1.0 / math.sqrt(D)
    block_M = min(64, block_size)
    mask_last_token = True
    torch.manual_seed(42)

    num_kv_blocks = SEQ_LEN // block_size

    Q_raw = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)
    K_raw = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device)
    V_raw = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device)
    logits_raw = torch.randn((B, SEQ_LEN, HQ, num_kv_blocks), dtype=dtype, device=device)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    q_indices = torch.arange(SEQ_LEN, device=device).view(1, SEQ_LEN, 1, 1)
    k_blk_indices = torch.arange(num_kv_blocks, device=device).view(1, 1, 1, num_kv_blocks)


    # q_blk_indices = (q_indices // block_M * block_M) // block_size # Old logic

    if window_size > 0:
        threshold_blk = (q_indices - window_size + 1).div(block_size, rounding_mode='floor')
    else:
        threshold_blk = q_indices // block_size

    weight_mask = k_blk_indices < threshold_blk

    logits_raw.masked_fill_(~weight_mask, float('-inf'))
    W_raw = torch.softmax(logits_raw, dim=-1)
    W_raw = torch.nan_to_num(W_raw, 0.0)

    Q_hsa = Q_raw.clone().detach().requires_grad_(True)
    K_hsa = K_raw.clone().detach().requires_grad_(True)
    V_hsa = V_raw.clone().detach().requires_grad_(True)
    W_hsa = W_raw.clone().detach().requires_grad_(True)

    Q_ref = Q_raw.clone().detach().requires_grad_(True)
    K_ref = K_raw.clone().detach().requires_grad_(True)
    V_ref = V_raw.clone().detach().requires_grad_(True)
    W_ref = W_raw.clone().detach().requires_grad_(True)

    block_indices_dense = torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
    block_indices_dense = block_indices_dense.view(1, 1, 1, num_kv_blocks).expand(B, SEQ_LEN, H, num_kv_blocks)

    O_hsa = HSA_block_M_head_dense(
        Q_hsa, K_hsa, V_hsa, W_hsa,
        block_size=block_size, sm_scale=scale, block_M=block_M,
        mask_last_token=mask_last_token, dtype="float32", accum_dtype="float", window_size=window_size
    )

    O_ref = hsa_torch_ref(
        Q_ref, K_ref, V_ref, W_ref, block_indices_dense,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
        window_size=window_size
    )

    def get_abs_err(x, y):
        return (x - y).flatten().abs().max().item()
    def get_err_ratio(x, y):
        err = (x - y).flatten().square().mean().sqrt().item()
        base = (x).flatten().square().mean().sqrt().item()
        return err / (base + 1e-12)
    def assert_close(prefix, ref, tri, ratio=0.005):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    assert_close("FWD", O_ref, O_hsa)

    O_hsa.backward(grad_output)
    O_ref.backward(grad_output)

    assert_close("DQ", Q_ref.grad, Q_hsa.grad)
    assert_close("DK", K_ref.grad, K_hsa.grad)
    assert_close("DV", V_ref.grad, V_hsa.grad)
    assert_close("DW", W_ref.grad, W_hsa.grad)

    print(f"FP32 Dense Correctness Test Passed for B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, block_size={block_size}, window_size={window_size}")




import time
def benchmark_dense_interface_weight_methods(
    B: int = 2,
    L: int = 8192,
    H: int = 2,
    HQ: int = 16,
    D: int = 128,
    block_size: int = 64,
    window_size: int = 512,
    is_causal: bool = True,
    enable_softmax1: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    sparse_topk: int = 32,
    num_warmup: int = 20,
    num_iters: int = 50,
):

    device = "cuda"
    torch.manual_seed(0)

    num_chunks = L // block_size

    # inputs
    q = torch.randn((B, L, HQ, D), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, L, H, D), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, L, H, D), device=device, dtype=dtype, requires_grad=True)
    lmk = torch.randn((B, num_chunks, H, D), device=device, dtype=dtype)
    lse_swa = torch.randn((B, L, HQ), device=device, dtype=dtype)

    grad_out = torch.randn((B, L, HQ, D), device=device, dtype=dtype)

    def _warmup(fn):
        for _ in range(num_warmup):
            fn()
        torch.cuda.synchronize()

    def _time_ms(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3 / num_iters

    # ---------------------------
    # 1) dense score + dense HSA path
    # ---------------------------
    def fwd_a():
        _, _, _ = HSA_dense_interface(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            enable_softmax1=enable_softmax1,
        )

    def fwd_bwd_a():
        q.grad = k.grad = v.grad = None
        out, _, _ = HSA_dense_interface(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            enable_softmax1=enable_softmax1,
        )
        out.backward(grad_out)

    _warmup(fwd_a)
    _warmup(fwd_bwd_a)
    a_fwd = _time_ms(fwd_a)
    a_total = _time_ms(fwd_bwd_a)
    a_bwd = a_total - a_fwd

    # ---------------------------
    # 2) sparse topk + sparse HSA path
    # ---------------------------
    q.grad = k.grad = v.grad = None

    def fwd_b():
        _, _, _ = HSA_dense_interface_ref(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            is_causal=is_causal, enable_softmax1=enable_softmax1,
            topk=sparse_topk,
        )

    def fwd_bwd_b():
        q.grad = k.grad = v.grad = None
        out, _, _ = HSA_dense_interface_ref(
            q=q, k=k, v=v, lmk=lmk, lse_swa=lse_swa,
            block_size=block_size, window_size=window_size,
            is_causal=is_causal, enable_softmax1=enable_softmax1,
            topk=sparse_topk,
        )
        out.backward(grad_out)

    _warmup(fwd_b)
    _warmup(fwd_bwd_b)
    b_fwd = _time_ms(fwd_b)
    b_total = _time_ms(fwd_bwd_b)
    b_bwd = b_total - b_fwd

    print("==== Benchmark: dense vs sparse HSA paths ====")
    print(f"Config: B={B} L={L} H={H} HQ={HQ} D={D} block={block_size} chunks={num_chunks} sparse_topk={sparse_topk} window={window_size} causal={is_causal} softmax1={enable_softmax1} dtype={dtype}")
    print(f"[dense scores + dense HSA]       FWD {a_fwd:.3f} ms | BWD {a_bwd:.3f} ms | Total {a_total:.3f} ms")
    print(f"[topk kernel + sparse HSA]       FWD {b_fwd:.3f} ms | BWD {b_bwd:.3f} ms | Total {b_total:.3f} ms")



if __name__ == "__main__":
    # main_block_M_correctness()
    # main_block_M_latency()

    # params_list = [
    #     (1, 1000, 1, 8, 64, 32, 128),
    #     (2, 1024, 1, 8, 64, 32, 256),
    #     (3, 512, 1, 8, 64, 32, 128),
    #     (4, 512, 1, 8, 64, 64, -1),
    #     (5, 256, 1, 8, 64, 64, 64),
    # ]
    # for p in params_list:
    #     test_correctness_fp32(*p)

    params_list = [
        (1, 500, 1, 8,  64, 64, 128, True,  False, False, False, False, 3, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, True,  True,  False, 3, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, False, False, True,  3, True),
        (1, 512, 1, 8,  64, 64, 64,  True,  True,  False, True,  True,  4, False),
        (1, 512, 1, 8,  64, 64, 128, True,  False, False, True,  True,  3, True),
    ]
    for p in params_list:
        test_dense_interface_vs_ref(*p)

    benchmark_dense_interface_weight_methods()
