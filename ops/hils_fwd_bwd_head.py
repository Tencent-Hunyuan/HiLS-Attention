"""
HiLS Sparse Attention
======================================
TileLang implementation of the HiLS-Attention sparse attention kernel.

Author: Xinyu Wei
"""

import torch
import math
import logging

logging.getLogger("tilelang.jit.kernel").setLevel(logging.WARNING)
logging.getLogger("tilelang").setLevel(logging.WARNING)

import tilelang
from tilelang import language as T
from einops import rearrange


def hils_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int, mask_last_token: bool = True, slope=None):

    B, q_len, HQ, D = q.shape
    _, kv_len, H, Dk = k.shape
    Dv = v.shape[-1]
    assert Dk == D, f"Q head_dim ({D}) must equal K head_dim ({Dk})"
    G = HQ // H
    q_blocks = q_len // block_q
    device = q.device

    if indices.shape[1] != q_blocks:
        idx_view = indices.view(B, q_blocks, block_q, H, -1)
        indices_q = idx_view[:, :, 0, :, :].contiguous()
    else:
        indices_q = indices

    valid_mask = (indices_q >= 0)
    safe_indices = indices_q.clamp_min(0)


    N = kv_len // chunk_size
    valid_kv_len = N * chunk_size


    k_truncated = k[:, :valid_kv_len, :, :]
    v_truncated = v[:, :valid_kv_len, :, :]

    k_chunks = rearrange(k_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v_truncated, 'B (N S) h d -> B N S h d', S=chunk_size)


    idx_flat_base = rearrange(safe_indices, 'B Bq h K -> B (Bq K) h').unsqueeze(2).unsqueeze(-1).long()
    idx_flat_k = idx_flat_base.expand(-1, -1, chunk_size, -1, Dk)
    idx_flat_v = idx_flat_base.expand(-1, -1, chunk_size, -1, Dv)
    gather_k = k_chunks.gather(dim=1, index=idx_flat_k)
    gather_v = v_chunks.gather(dim=1, index=idx_flat_v)

    gather_k = rearrange(gather_k, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)
    gather_v = rearrange(gather_v, 'B (Bq K) S h d -> B Bq S K h d', Bq=q_blocks)

    k_ = torch.repeat_interleave(gather_k, dim=-2, repeats=G)
    v_ = torch.repeat_interleave(gather_v, dim=-2, repeats=G)

    q_chunked = rearrange(q, 'B (Bq X) hq d -> B Bq X hq d', X=block_q)

    qk = torch.einsum('b q x h d, b q s k h d -> b q x s k h', q_chunked.float(), k_.float())

    if slope is not None:
        # slope: (HQ,)  -> broadcast over (B, Bq, block_q, chunk_size, K, HQ).
        slope_f = slope.to(qk.dtype).to(qk.device)
        s_pos = torch.arange(chunk_size, device=qk.device, dtype=qk.dtype)
        # dist[s] = max(0, (chunk_size - 2) - s)  -> shape (chunk_size,)
        dist = ((chunk_size - 2) - s_pos).clamp(min=0)
        # bias shape: (1, 1, 1, chunk_size, 1, HQ)
        bias = -slope_f.view(1, 1, 1, 1, 1, HQ) * dist.view(1, 1, 1, chunk_size, 1, 1)
        qk = qk + bias

    qk = qk * float(sm_scale)

    if mask_last_token:
        qk[:, :, :, -1, :, :] = float("-inf")

    p = torch.softmax(qk, dim=3)

    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    valid_mask_expanded = torch.repeat_interleave(valid_mask, dim=-2, repeats=G)
    w_masked = w_masked.masked_fill(~valid_mask_expanded, 0)
    w_exp = w_masked.float()
    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'b q x h d -> b (q x) h d')
    return o_ref.to(torch.float32)


def build_block_indices_block_M(
    B: int,
    SEQ_LEN: int,
    H: int,
    S: int,
    block_size: int,
    overlap_ratio: float = 0.5,
    block_M: int = 2,
    device: str = "cuda",
    kv_len: int = None,
) -> torch.Tensor:

    import torch

    assert 0.0 <= overlap_ratio <= 1.0, "overlap_ratio must be in [0, 1]"
    assert block_M >= 1, "block_M must be >= 1"


    if kv_len is None:
        num_blocks = SEQ_LEN // block_size
    else:
        num_blocks = kv_len // block_size
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
def hierarchical_sparse_attention_block_M_reuseIndices(
        batch, heads, head_dim, q_len=None, kv_len=None,
        scale=None, block_size=64, groups=16,
        selected_blocks=16, num_weights=None, block_M=None,
        mask_last_token=True, dtype="bfloat16", accum_dtype="float",
        num_threads=None, is_training=True, v_head_dim=None,
        enable_alibi=False):

    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504
    else:
        scale = scale * 1.44269504

    if num_weights is None:
        num_weights = selected_blocks
    if v_head_dim is None:
        v_head_dim = head_dim
    head_kv = heads // groups
    if not is_training:
        q_len = T.dynamic("q_len")
        kv_len = T.dynamic("kv_len")
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, head_kv, head_dim]
    v_shape = [batch, kv_len, head_kv, v_head_dim]
    o_shape = [batch, q_len, heads, v_head_dim]
    weight_shape = [batch, q_len, heads, num_weights]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    block_S = block_size
    block_T_k = min(256, tilelang.math.next_power_of_2(head_dim))
    block_T_v = min(256, tilelang.math.next_power_of_2(v_head_dim))

    assert tilelang.cdiv(head_dim, block_T_k) == 1, "The key dimension can not be larger than 256"
    assert tilelang.cdiv(v_head_dim, block_T_v) == 1, "The value dimension can not be larger than 256"

    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, groups)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    print(f"Using M = {M} for fwd_block_M_reuseIndices kernel, "
          f"batch={batch}, heads={heads}, head_kv={head_kv}, "
          f"q_len={q_len}, kv_len={kv_len}, head_dim={head_dim}, v_head_dim={v_head_dim}")
    print("num_threads:", num_threads)
    M_G = M * groups

    S = selected_blocks
    BS = block_S
    BK = block_T_k
    BV = block_T_v
    num_stages = 1

    if num_threads is None:
        num_threads = 128

    NP = tilelang.cdiv(q_len, M)
    N_BH = batch * head_kv
    scores_lse_shape = [batch, q_len, heads, selected_blocks]
    fwd_merged_indices_shape = [N_BH, NP, S * M]
    fwd_block_ownership_shape = [N_BH, NP, S * M]
    fwd_chunk_weights_shape = [N_BH, NP, S * M, M_G]
    fwd_merged_s_indices_shape = [N_BH, NP, S * M, M]
    fwd_merged_len_shape = [N_BH, NP]
    slope_shape = [heads]

    @T.prim_func
    def hils_block_M_reuseIndices(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            V: T.Tensor(v_shape, dtype),
            W: T.Tensor[weight_shape, dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Slope: T.Tensor(slope_shape, accum_dtype),
            ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
            FwdMergedIndices: T.Tensor(fwd_merged_indices_shape, "int32"),
            FwdBlockOwnership: T.Tensor(fwd_block_ownership_shape, "int32"),
            FwdChunkWeights: T.Tensor(fwd_chunk_weights_shape, dtype),
            FwdMergedSIndices: T.Tensor(fwd_merged_s_indices_shape, "int32"),
            FwdMergedLen: T.Tensor(fwd_merged_len_shape, "int32"),
            Output: T.Tensor(o_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), batch * head_kv, threads=num_threads) as (bx, bz):
            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([M_G, BV], dtype)

            acc_s = T.alloc_fragment([M_G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([M_G, BS], dtype)
            acc_o = T.alloc_fragment([M_G, BV], accum_dtype)

            P_shared = T.alloc_shared([M_G, BS], dtype)

            acc_s_tmp = T.alloc_fragment([groups, BS], accum_dtype)
            scores_max = T.alloc_fragment([M_G], accum_dtype)
            scores_sum = T.alloc_fragment([M_G], accum_dtype)

            if is_training:
                lse_local = T.alloc_fragment([M_G], accum_dtype)

            merged_indices = T.alloc_shared([S * M], block_indices_dtype)
            block_ownership = T.alloc_shared([S * M], "int32")
            merged_len = T.alloc_shared([1], "int32")



            chunk_weights_row = T.alloc_shared([M_G], dtype)
            merged_s_indices = T.alloc_shared([S * M, M], "int32")

            i_t_base_idx, i_bh = bx, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv
            base_t = i_t_base_idx * M

            T.fill(Q_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    T.copy(Q[i_b, tq, i_h * groups:(i_h + 1) * groups, :],
                           Q_shared[q_idx * groups:(q_idx + 1) * groups, :])

            T.fill(acc_o, 0)
            T.fill(merged_indices, -1)
            T.fill(block_ownership, 0)
            T.fill(merged_s_indices, -1)
            T.fill(merged_len, 0)

            W_local_shared = T.alloc_shared([M_G, S], dtype)
            T.fill(W_local_shared, 0.0)
            for i_mg, s_idx in T.Parallel(M_G, S):
                q_idx = i_mg // groups
                g = i_mg % groups
                tq = base_t + q_idx
                if tq < q_len:
                    W_local_shared[i_mg, s_idx] = W[i_b, tq, i_h * groups + g, s_idx]

            if T.get_thread_binding() == 0:
                valid_lens = T.alloc_fragment([M], "int32")
                pointers = T.alloc_fragment([M], "int32")
                k = T.alloc_var("int32")
                cur_val = T.alloc_fragment([M], "int32")

                for q_idx in T.Parallel(M):
                    tq = base_t + q_idx
                    valid_lens[q_idx] = 0
                    pointers[q_idx] = 0
                    if tq < q_len:
                        for j in T.serial(S):
                            if BlockIndices[i_b, tq, i_h, j] >= 0:
                                valid_lens[q_idx] = valid_lens[q_idx] + 1
                            else:
                                T.loop_break()

                k = 0
                ownership_mask = T.alloc_var("int32")
                for _ in T.serial(S * M):
                    min_val = T.alloc_var("int32")
                    min_val = 2147483647
                    has_valid = T.alloc_var("int32")
                    has_valid = 0

                    for q_idx in T.serial(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                                has_valid = 1
                                val_q = BlockIndices[i_b, tq, i_h, pointers[q_idx]]
                                cur_val[q_idx] = val_q
                                if val_q < min_val:
                                    min_val = val_q
                        else:
                            cur_val[q_idx] = 2147483647

                    if has_valid == 0:
                        T.loop_break()

                    merged_indices[k] = min_val
                    ownership_mask = 0
                    for q_idx in T.unroll(M):
                        tq = base_t + q_idx
                        if tq < q_len and pointers[q_idx] < valid_lens[q_idx]:
                            s_idx = pointers[q_idx]
                            val_q = cur_val[q_idx]
                            if val_q == min_val:
                                ownership_mask = ownership_mask | (1 << q_idx)
                                merged_s_indices[k, q_idx] = s_idx
                                pointers[q_idx] = pointers[q_idx] + 1
                    block_ownership[k] = ownership_mask
                    k = k + 1

                merged_len[0] = k

            T.sync_threads()

            if is_training:
                T.copy(merged_indices, FwdMergedIndices[i_bh, i_t_base_idx, :])
                T.copy(block_ownership, FwdBlockOwnership[i_bh, i_t_base_idx, :])
                T.copy(merged_s_indices, FwdMergedSIndices[i_bh, i_t_base_idx, :, :])
                FwdMergedLen[i_bh, i_t_base_idx] = merged_len[0]

            merged_len_local = T.alloc_var("int32")
            merged_len_local = merged_len[0]
            h_start = T.alloc_var("int32")

            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s = blk_idx * BS
                ownership = block_ownership[i]

                if (blk_idx >= 0):


                    for r in T.Parallel(M_G):
                        q_idx_w = r // groups
                        s_idx_w = merged_s_indices[i, q_idx_w]
                        chunk_weights_row[r] = T.if_then_else(
                            s_idx_w >= 0,
                            W_local_shared[r, T.if_then_else(s_idx_w >= 0, s_idx_w, 0)],
                            T.cast(0, dtype),
                        )

                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    if enable_alibi:
                        for r, c in T.Parallel(M_G, BS):
                            acc_s[r, c] = acc_s[r, c] - Slope[i_h * groups + (r % groups)] * T.if_then_else(
                                c >= BS - 2,
                                T.cast(0, accum_dtype),
                                T.cast(BS - 2 - c, accum_dtype),
                            )

                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups

                        acc_s[r, c] = T.if_then_else(
                                                    ((ownership & (1 << q_idx)) == 0) or ((c == BS - 1) and enable_last_token_mask),
                                                    -T.infinity(accum_dtype),
                                                    acc_s[r, c]
                                                )

                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups

                        acc_s[r, c] = T.if_then_else(
                                                    (ownership & (1 << q_idx)) != 0,
                                                    T.exp2(acc_s[r, c] * scale - scores_max[r] * scale),
                                                    0.0
                                                )

                    T.fill(scores_sum, 0.0)
                    T.reduce_sum(acc_s, scores_sum, dim=1, clear=True)


                    if is_training:
                        for r_lse in T.Parallel(M_G):
                                lse_local[r_lse] = T.if_then_else(
                                    scores_sum[r_lse] > 0,
                                    scores_max[r_lse] * scale + T.log(scores_sum[r_lse]) * 1.44269504,
                                    -T.infinity(accum_dtype)
                                )
                        for q_idx_lse, g_lse in T.Parallel(M, groups):
                            tq_lse = base_t + q_idx_lse
                            s_idx_lse = merged_s_indices[i, q_idx_lse]
                            if tq_lse < q_len and s_idx_lse >= 0:
                                r_lse = q_idx_lse * groups + g_lse
                                ScoresLSE[i_b, tq_lse, i_h * groups + g_lse, s_idx_lse] = lse_local[r_lse]

                    for r, c in T.Parallel(M_G, BS):
                        q_idx = r // groups

                        acc_s[r, c] = T.if_then_else(
                                                    (ownership & (1 << q_idx)) != 0,
                                                    acc_s[r, c] * T.cast(chunk_weights_row[r], accum_dtype) / scores_sum[r],
                                                    0.0
                                                )

                    T.copy(acc_s, P_shared)

                    T.copy(V[i_b, i_s:i_s + BS, i_h, :], V_shared)
                    T.gemm(P_shared, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)

            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * groups
                    for g, v in T.Parallel(groups, BV):
                        Output[i_b, tq, i_h * groups + g, v] = O_shared[h_start + g, v]

    return hils_block_M_reuseIndices


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
    batch, heads, q_len, kv_len, head_dim,
    scale=None, block_size=64, groups=16, selected_blocks=16, num_weights=None,
    dtype="bfloat16", accum_dtype="float", block_M=0, mask_last_token=True,
    num_threads=None, v_head_dim=None, enable_alibi=False,
):

    enable_last_token_mask = False
    if mask_last_token:
        enable_last_token_mask = True

    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale
    scale_log2 = sm_scale * 1.44269504

    if v_head_dim is None:
        v_head_dim = head_dim

    B = batch
    BS = block_size
    G = groups
    Vdim = v_head_dim
    Kdim = head_dim
    BK = min(256, tilelang.next_power_of_2(Kdim))
    BV = min(256, tilelang.next_power_of_2(Vdim))
    assert tilelang.cdiv(Kdim, BK) == 1, "The key dimension can not be larger than 256"
    assert tilelang.cdiv(Vdim, BV) == 1, "The value dimension can not be larger than 256"
    S = selected_blocks

    if num_weights is None:
        num_weights = S

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, v_head_dim]
    do_shape = [batch, q_len, heads, v_head_dim]
    dq_shape = [batch, q_len, heads, head_dim]
    dk_shape = [batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, v_head_dim]
    weight_shape = [batch, q_len, heads, num_weights]
    dw_shape = [batch, q_len, heads, num_weights]
    scores_lse_shape = [batch, q_len, heads, selected_blocks]

    MIN_GEMM_ROWS = 16
    M_min = tilelang.cdiv(MIN_GEMM_ROWS, G)
    if block_M is None or block_M <= 0:
        M = M_min
    else:
        M = max(block_M, M_min)
    print("Using M =", M, "for bwd_block_M_inverse_reuseIndices kernel",
          "num_threads:", num_threads)
    M_G = M * G

    if num_threads is None:
        num_threads = 128
    num_stages = 0


    NP_bwd = tilelang.cdiv(q_len, M)
    N_BH = batch * heads_kv
    fwd_merged_indices_shape = [N_BH, NP_bwd, S * M]
    fwd_block_ownership_shape = [N_BH, NP_bwd, S * M]
    fwd_chunk_weights_shape = [N_BH, NP_bwd, S * M, M_G]
    fwd_merged_s_indices_shape = [N_BH, NP_bwd, S * M, M]
    fwd_merged_len_shape = [N_BH, NP_bwd]
    slope_shape = [heads]

    @T.prim_func
    def hils_bwd_dqkv_block_M_inverse_reuseIndices(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        V: T.Tensor(v_shape, dtype),
        W: T.Tensor(weight_shape, dtype),
        DO: T.Tensor(do_shape, dtype),
        Slope: T.Tensor(slope_shape, accum_dtype),
        FwdMergedIndices: T.Tensor(fwd_merged_indices_shape, "int32"),
        FwdBlockOwnership: T.Tensor(fwd_block_ownership_shape, "int32"),
        FwdChunkWeights: T.Tensor(fwd_chunk_weights_shape, dtype),
        FwdMergedSIndices: T.Tensor(fwd_merged_s_indices_shape, "int32"),
        FwdMergedLen: T.Tensor(fwd_merged_len_shape, "int32"),
        ScoresLSE: T.Tensor(scores_lse_shape, accum_dtype),
        DQ: T.Tensor(dq_shape, accum_dtype),
        DK: T.Tensor(dk_shape, accum_dtype),
        DV: T.Tensor(dv_shape, accum_dtype),
        DW: T.Tensor(dw_shape, dtype),
    ):
        with T.Kernel(tilelang.cdiv(q_len, M), B * heads_kv, threads=num_threads) as (bx, bz):
            i_t_base_idx, i_bh = bx, bz
            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            base_t = i_t_base_idx * M

            Q_shared = T.alloc_shared([M_G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            dO_shared = T.alloc_shared([M_G, BV], dtype)
            dO_weighted_shared = T.alloc_shared([M_G, BV], dtype)
            P_shared = T.alloc_shared([M_G, BS], dtype)
            dQ_shared = T.alloc_shared([M_G, BK], accum_dtype)
            dQ_accum = T.alloc_fragment([M_G, BK], accum_dtype)
            acc_s_tmp = T.alloc_fragment([M_G, BS], accum_dtype)
            dV_PdO_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            dS_frag = T.alloc_fragment([M_G, BS], accum_dtype)
            delta_rows = T.alloc_fragment([M_G], accum_dtype)
            saved_lse_shared = T.alloc_shared([M_G], accum_dtype)
            dV_accum_local = T.alloc_fragment([BS, BV], accum_dtype)
            dK_accum_local = T.alloc_fragment([BS, BK], accum_dtype)
            di_rows = T.alloc_fragment([M_G], accum_dtype)
            merged_indices = T.alloc_shared([S * M], "int32")
            block_ownership = T.alloc_shared([S * M], "int32")
            W_local_shared = T.alloc_shared([M_G, S], dtype)
            chunk_weights_row = T.alloc_shared([M_G], dtype)
            merged_s_indices = T.alloc_shared([S * M, M], "int32")

            T.fill(Q_shared, 0)
            T.fill(dO_shared, 0)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(Q[i_b, tq, i_h * G:(i_h + 1) * G, :], Q_shared[h_start:h_start + G, :])
                    T.copy(DO[i_b, tq, i_h * G:(i_h + 1) * G, :], dO_shared[h_start:h_start + G, :])

            T.fill(dQ_accum, 0)
            T.fill(saved_lse_shared, 0.0)
            T.copy(FwdMergedIndices[i_bh, i_t_base_idx, :], merged_indices)
            T.copy(FwdBlockOwnership[i_bh, i_t_base_idx, :], block_ownership)
            T.copy(FwdMergedSIndices[i_bh, i_t_base_idx, :, :], merged_s_indices)
            T.fill(W_local_shared, 0.0)
            for i_mg, s_idx in T.Parallel(M_G, S):
                q_idx_w = i_mg // G
                g_w = i_mg % G
                tq_w = base_t + q_idx_w
                if tq_w < q_len:
                    W_local_shared[i_mg, s_idx] = W[i_b, tq_w, i_h * G + g_w, s_idx]

            T.sync_threads()

            merged_len_local = T.alloc_var("int32")
            merged_len_local = FwdMergedLen[i_bh, i_t_base_idx]
            h_start = T.alloc_var("int32")
            s_idx_local = T.alloc_var("int32")

            for i in T.Pipelined(merged_len_local, num_stages=num_stages):
                blk_idx = merged_indices[i]
                i_s_global = blk_idx * BS
                ownership = block_ownership[i]

                if blk_idx >= 0:
                    for r in T.Parallel(M_G):
                        q_idx_w = r // G
                        s_idx_w = merged_s_indices[i, q_idx_w]
                        chunk_weights_row[r] = T.if_then_else(
                            s_idx_w >= 0,
                            W_local_shared[r, T.if_then_else(s_idx_w >= 0, s_idx_w, 0)],
                            T.cast(0, dtype),
                        )

                    T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :], K_shared)
                    T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, :], V_shared)

                    T.clear(acc_s_tmp)
                    T.gemm(Q_shared, K_shared, acc_s_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    if enable_alibi:
                        for i_mg, s in T.Parallel(M_G, BS):
                            acc_s_tmp[i_mg, s] = acc_s_tmp[i_mg, s] - Slope[i_h * G + (i_mg % G)] * T.if_then_else(
                                s >= BS - 2,
                                T.cast(0, accum_dtype),
                                T.cast(BS - 2 - s, accum_dtype),
                            )

                    for i_mg in T.Parallel(M_G):
                        qi_lse = i_mg // G
                        g_lse = i_mg % G
                        tq_lse = base_t + qi_lse
                        s_idx_lse = merged_s_indices[i, qi_lse]
                        saved_lse_shared[i_mg] = T.if_then_else(
                            (tq_lse < q_len)
                            and (((ownership >> qi_lse) & 1) == 1)
                            and (s_idx_lse >= 0),
                            ScoresLSE[
                                i_b,
                                T.if_then_else(tq_lse < q_len, tq_lse, 0),
                                i_h * G + g_lse,
                                T.if_then_else(s_idx_lse >= 0, s_idx_lse, 0),
                            ],
                            0.0,
                        )

                    for i_mg, s in T.Parallel(M_G, BS):
                        q_idx = i_mg // G
                        is_owned = (ownership >> q_idx) & 1
                        if enable_last_token_mask:
                            acc_s_tmp[i_mg, s] = T.if_then_else(
                                is_owned == 1 and s != BS - 1,
                                T.exp2(acc_s_tmp[i_mg, s] * scale_log2 - saved_lse_shared[i_mg]),
                                0.0
                            )
                        else:
                            acc_s_tmp[i_mg, s] = T.if_then_else(
                                is_owned == 1,
                                T.exp2(acc_s_tmp[i_mg, s] * scale_log2 - saved_lse_shared[i_mg]),
                                0.0
                            )
                    T.copy(acc_s_tmp, P_shared)

                    T.clear(dV_PdO_frag)
                    T.gemm(dO_shared, V_shared, dV_PdO_frag, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for g_row, s in T.Parallel(M_G, BS):
                        dS_frag[g_row, s] = P_shared[g_row, s] * dV_PdO_frag[g_row, s]
                    T.reduce_sum(dS_frag, di_rows, dim=1, clear=True)

                    for r in T.Parallel(M_G):
                        q_idx = r // G
                        g_w = r % G
                        tq = base_t + q_idx
                        s_idx_local_w = merged_s_indices[i, q_idx]
                        if (
                            (tq < q_len)
                            and (((ownership >> q_idx) & 1) == 1)
                            and (s_idx_local_w >= 0)
                        ):
                            DW[
                                i_b,
                                tq,
                                i_h * G + g_w,
                                T.if_then_else(s_idx_local_w >= 0, s_idx_local_w, 0),
                            ] = di_rows[r]

                    for g_row, v in T.Parallel(M_G, BV):
                        dO_weighted_shared[g_row, v] = T.cast(chunk_weights_row[g_row], accum_dtype) * dO_shared[g_row, v]

                    T.clear(dV_accum_local)
                    T.gemm(P_shared, dO_weighted_shared, dV_accum_local, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    for g_row, s in T.Parallel(M_G, BS):
                        dS_frag[g_row, s] = sm_scale * T.cast(chunk_weights_row[g_row], accum_dtype) * P_shared[g_row, s] * (dV_PdO_frag[g_row, s] - di_rows[g_row])
                    T.copy(dS_frag, P_shared)

                    T.clear(dK_accum_local)
                    T.gemm(P_shared, Q_shared, dK_accum_local, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    T.gemm(P_shared, K_shared, dQ_accum, policy=T.GemmWarpPolicy.FullRow)

                    for s, k_d in T.Parallel(BS, BK):
                        T.atomic_add(DK[i_b, i_s_global + s, i_h, k_d], dK_accum_local[s, k_d])
                    for s, v_d in T.Parallel(BS, BV):
                        T.atomic_add(DV[i_b, i_s_global + s, i_h, v_d], dV_accum_local[s, v_d])

            T.copy(dQ_accum, dQ_shared)
            for q_idx in T.serial(M):
                tq = base_t + q_idx
                if tq < q_len:
                    h_start = q_idx * G
                    T.copy(dQ_shared[h_start:h_start + G, :], DQ[i_b, tq, i_h * G:(i_h + 1) * G, :])

    return hils_bwd_dqkv_block_M_inverse_reuseIndices




class _hils_block_M_attention_inverse(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, weights, indices,
                block_size, sm_scale, block_M,
                mask_last_token, dtype, accum_dtype,
                num_threads_fwd, num_threads_bwd, is_training,
                slope=None):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        assert weights.is_contiguous() and indices.is_contiguous()

        B, L, HQ, D = q.shape
        L_kv = k.shape[1]
        H = k.shape[2]
        Dk = k.shape[-1]
        Dv = v.shape[-1]
        S = indices.shape[-1]
        G = HQ // H
        num_weights = weights.shape[-1]

        assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"
        assert Dk == D, f"Q head_dim ({D}) must equal K head_dim ({Dk})"

        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(D)

        enable_alibi = slope is not None
        if enable_alibi:
            assert slope.shape == (HQ,), (
                f"slope must be 1-D with shape ({HQ},), got {tuple(slope.shape)}"
            )
            slope_dev = slope.to(dtype=torch.float32, device=q.device).contiguous()
        else:
            slope_dev = torch.zeros(HQ, dtype=torch.float32, device=q.device)

        fwd_kernel = hierarchical_sparse_attention_block_M_reuseIndices(
            batch=B, heads=HQ, q_len=L, kv_len=L_kv, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            block_M=block_M, mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype,
            num_threads=num_threads_fwd, is_training=is_training,
            v_head_dim=Dv, enable_alibi=enable_alibi,
        )

        if not is_training:
            o = fwd_kernel(q, k, v, weights.to(q.dtype), indices, slope_dev,
                           None, None, None, None, None, None)
            ctx.is_training = False
            return o

        MIN_GEMM_ROWS = 16
        M_min = (MIN_GEMM_ROWS + G - 1) // G
        M_actual = block_M
        if M_actual is None or M_actual <= 0:
            M_actual = M_min
        else:
            M_actual = max(M_actual, M_min)
        NP = (L + M_actual - 1) // M_actual
        N_BH = B * H
        SM = S * M_actual
        fwd_merged_indices = torch.full((N_BH, NP, SM), -1, dtype=torch.int32, device=q.device)
        fwd_block_ownership = torch.zeros((N_BH, NP, SM), dtype=torch.int32, device=q.device)

        fwd_chunk_weights = None
        fwd_merged_s_indices = torch.full((N_BH, NP, SM, M_actual), -1, dtype=torch.int32, device=q.device)
        fwd_merged_len = torch.zeros((N_BH, NP), dtype=torch.int32, device=q.device)

        scores_lse = torch.full((B, L, HQ, S), float('-inf'), dtype=torch.float32, device=q.device)

        o = fwd_kernel(q, k, v, weights.to(q.dtype), indices, slope_dev, scores_lse,
                       fwd_merged_indices, fwd_block_ownership, fwd_chunk_weights,
                       fwd_merged_s_indices, fwd_merged_len)

        ctx.save_for_backward(q, k, v, weights, indices, scores_lse,
                              fwd_merged_indices, fwd_block_ownership,
                              fwd_merged_s_indices, fwd_merged_len, slope_dev)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.block_M = block_M
        ctx.mask_last_token = mask_last_token
        ctx.dtype = dtype
        ctx.accum_dtype = accum_dtype
        ctx.num_threads_bwd = num_threads_bwd
        ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G = B, L, HQ, H, D, S, G
        ctx.Dv = Dv
        ctx.num_weights = num_weights
        ctx.is_training = True
        ctx.enable_alibi = enable_alibi
        return o

    @staticmethod
    def backward(ctx, do):
        if not getattr(ctx, "is_training", True):
            raise RuntimeError(
                "[_hils_block_M_attention_inverse] forward  is_training=False , "
                "BWDof merged indices / ScoresLSE, without backward."
            )
        q, k, v, weights, indices, scores_lse, \
            fwd_merged_indices, fwd_block_ownership, \
            fwd_merged_s_indices, fwd_merged_len, slope_dev = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        block_M = ctx.block_M
        mask_last_token = ctx.mask_last_token
        dtype = ctx.dtype
        accum_dtype = ctx.accum_dtype
        num_threads_bwd = ctx.num_threads_bwd
        B, L, HQ, H, D, S, G = ctx.B, ctx.L, ctx.HQ, ctx.H, ctx.D, ctx.S, ctx.G
        Dv = ctx.Dv
        num_weights = ctx.num_weights
        L_kv = k.shape[1]
        enable_alibi = ctx.enable_alibi

        do = do.contiguous()

        bwd_kernel = hierarchical_sparse_attention_bwd_dqkv_block_M_inverse(
            batch=B, heads=HQ, q_len=L, kv_len=L_kv, head_dim=D,
            block_size=block_size, groups=G, selected_blocks=S,
            num_weights=num_weights, scale=sm_scale,
            block_M=block_M, mask_last_token=mask_last_token,
            dtype=dtype, accum_dtype=accum_dtype,
            num_threads=num_threads_bwd,
            v_head_dim=Dv, enable_alibi=enable_alibi,
        )

        DQ = torch.zeros((B, L, HQ, D), dtype=torch.float32, device=q.device)
        DK = torch.zeros((B, L_kv, H, D), dtype=torch.float32, device=k.device)
        DV = torch.zeros((B, L_kv, H, Dv), dtype=torch.float32, device=v.device)
        DW = torch.zeros((B, L, HQ, num_weights), dtype=weights.dtype, device=weights.device)

        bwd_kernel(q, k, v, weights.to(q.dtype), do, slope_dev,
                   fwd_merged_indices, fwd_block_ownership, None,
                   fwd_merged_s_indices, fwd_merged_len,
                   scores_lse, DQ, DK, DV, DW)

        DQ = DQ.to(q.dtype)
        DK = DK.to(k.dtype)
        DV = DV.to(v.dtype)
        DW = DW.to(weights.dtype)

        return (DQ, DK, DV, DW, None,
                None, None, None, None, None, None, None, None, None, None)


# Empirically tuned (block_M, num_threads_fwd, num_threads_bwd) per group size G
# for HiLS_block_M_head. Only used when the corresponding kwarg is not explicitly
# provided by the caller. Format: G -> (block_M, fwd_threads, bwd_threads).
_HILS_PREROTATE_TUNED_BY_G = {
    16: (4, 128, 128),
    8:  (8, 128, 128),
    4:  (8, 64, 128),
    2:  (16, 128, 128),
    1:  (16, 128, 128),
}


def HiLS_block_M_head(
    q, k, v, weights, indices,
    block_size=64, sm_scale=None, block_M=None,
    mask_last_token=True, dtype="bfloat16", accum_dtype="float",
    num_threads=None, num_threads_fwd=None, num_threads_bwd=None,
    is_training=True, slope=None,
):
    B, L, HQ, D = q.shape
    _, L_kv, H, Dk = k.shape
    assert D == Dk, f"Q head_dim ({D}) must equal K head_dim ({Dk})"
    assert HQ % H == 0, f"HQ={HQ} must be divisible by H={H}"
    G = HQ // H

    tuned = _HILS_PREROTATE_TUNED_BY_G.get(G)
    if tuned is not None:
        tuned_M, tuned_fwd_t, tuned_bwd_t = tuned
        applied = []
        if block_M is None:
            block_M = tuned_M
            applied.append(f"block_M={tuned_M}")
        if num_threads_fwd is None and num_threads is None:
            num_threads_fwd = tuned_fwd_t
            applied.append(f"num_threads_fwd={tuned_fwd_t}")
        if num_threads_bwd is None and num_threads is None:
            num_threads_bwd = tuned_bwd_t
            applied.append(f"num_threads_bwd={tuned_bwd_t}")

    if num_threads_fwd is None:
        num_threads_fwd = num_threads
    if num_threads_bwd is None:
        num_threads_bwd = num_threads
    return _hils_block_M_attention_inverse.apply(
        q, k, v, weights, indices,
        block_size, sm_scale, block_M,
        mask_last_token, dtype, accum_dtype,
        num_threads_fwd, num_threads_bwd, is_training, slope,
    )


def compute_grad_diff(grad_hils, grad_ref, name):
    if grad_hils is None or grad_ref is None:
        print(f"{name}: grad is None")
        return None
    diff = (grad_hils.float() - grad_ref.float()).abs()
    max_diff = diff.max().item()
    print(f"{name} max error: {max_diff:.6e}")
    return max_diff


def main_block_M_correctness():
    import torch.nn.functional as F

    B, q_len, kv_len, H, HQ, D, S, block_size = 1, 512, 612, 1, 8, 192, 4, 64
    dtype = torch.bfloat16
    device = "cuda"
    block_M = 4
    mask_last_token = True
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    print(f"Correctness Config: B={B}, q_len={q_len}, kv_len={kv_len}, H={H}, HQ={HQ}, D={D}, G={G}, S={S}, block_size={block_size}")
    torch.manual_seed(42)

    block_indices = build_block_indices_block_M(
        B=B, SEQ_LEN=q_len, H=H, S=S, block_size=block_size,
        overlap_ratio=0.8, block_M=block_M, device=device, kv_len=kv_len,
    )

    Q = torch.randn((B, q_len, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, kv_len, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, q_len, HQ, S), dtype=dtype, device=device)
    valid_mask_hq = torch.repeat_interleave(block_indices != -1, repeats=G, dim=2)
    logits = logits.masked_fill(~valid_mask_hq, float("-inf"))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)

    grad_output = torch.randn((B, q_len, HQ, D), dtype=dtype, device=device)

    O_ref = hils_torch_ref(
        Q.float().detach(), K.float().detach(), V.float().detach(), W.detach(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
    )

    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    O_ref_bwd = hils_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1, mask_last_token=mask_last_token,
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
    O_hils = HiLS_block_M_head(
        Q, K, V, W, block_indices,
        block_size=block_size, sm_scale=scale, block_M=block_M, mask_last_token=mask_last_token,
    )
    print("[Tilelang HiLS_block_M_head] vs [Torch Reference]:")
    print(f"forward max error: {(O_hils.float() - O_ref.float()).abs().max().item():.6e}")
    O_hils.backward(grad_output)

    compute_grad_diff(Q.grad.clone(), DQ_ref, "DQ (inverse)")
    compute_grad_diff(K.grad.clone(), DK_ref, "DK (inverse)")
    compute_grad_diff(V.grad.clone(), DV_ref, "DV (inverse)")
    compute_grad_diff(W.grad.clone(), DW_ref, "DW (inverse)")


def test_correctness_fp32(B=1, SEQ_LEN=1024, H=1, HQ=8, D=64, S=4, block_size=32):
    import torch.nn.functional as F

    device = "cuda"
    dtype = torch.float32
    scale = 1.0 / math.sqrt(D)
    block_M = 2
    mask_last_token = True
    torch.manual_seed(42)

    block_indices = torch.full((B, SEQ_LEN, H, S), -1, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for t in range(SEQ_LEN):
        max_blocks = min(t // block_size + 1, num_blocks)
        if max_blocks > 0:
            num_select = min(S, max_blocks)
            selected = torch.randperm(max_blocks, device=device)[:num_select]
            block_indices[:, t, :, :num_select] = selected.sort()[0]

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device)
    logits.masked_fill_(block_indices.repeat_interleave(HQ // H, dim=2) == -1, float("-inf"))
    W = F.softmax(logits, dim=-1)
    W = torch.nan_to_num(W, 0.0).detach().requires_grad_(True)
    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)

    O_hils = HiLS_block_M_head(
        Q, K, V, W, block_indices,
        block_size=block_size, sm_scale=scale, block_M=block_M,
        mask_last_token=mask_last_token, dtype="float", accum_dtype="float",
        num_threads=64, is_training=True,
    )
    O_ref = hils_torch_ref(
        Q.detach(), K.detach(), V.detach(), W.detach(), block_indices,
        chunk_size=block_size,
        sm_scale=scale,
        block_q=1,
        mask_last_token=mask_last_token,
    )

    torch.testing.assert_close(O_hils.float(), O_ref.float(), atol=0.005, rtol=0.005, msg="Forward mismatch")

    O_hils.backward(grad_output)
    DQ_hils = Q.grad.clone()
    DK_hils = K.grad.clone()
    DV_hils = V.grad.clone()
    DW_hils = W.grad.clone()

    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None
    O_ref_bwd = hils_torch_ref(
        Q, K, V, W, block_indices,
        chunk_size=block_size,
        sm_scale=scale,
        block_q=1,
        mask_last_token=mask_last_token,
    )
    O_ref_bwd.backward(grad_output)

    def get_abs_err(x, y):
        return (x - y).flatten().abs().max().item()

    def get_err_ratio(x, y):
        err = (x - y).flatten().square().mean().sqrt().item()
        base = x.flatten().square().mean().sqrt().item()
        return err / base

    def assert_close(prefix, ref, tri, ratio):
        msg = f"{prefix} diff: {get_abs_err(ref, tri):.6f} ratio: {get_err_ratio(ref, tri):.6f}"
        print(msg)
        assert get_err_ratio(ref, tri) < ratio, msg

    assert_close("DQ (inverse)", Q.grad, DQ_hils, 0.005)
    assert_close("DK (inverse)", K.grad, DK_hils, 0.005)
    assert_close("DV (inverse)", V.grad, DV_hils, 0.005)
    assert_close("DW (inverse)", W.grad, DW_hils, 0.005)
    print(f"FP32 Correctness Test Passed for B={B}, SEQ_LEN={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, S={S}, block_size={block_size}")


def _load_real_indices_for_breakdown(pt_path, B, layer_idx=None, device="cuda"):

    print(f"\n[real-indices] Loading: {pt_path}")
    saved = torch.load(pt_path, map_location="cpu", weights_only=False)
    config = saved["config"]
    samples = saved["samples"]
    num_samples = len(samples)
    print(f"  config: seq_len={config['seq_len']}, chunk_size={config['chunk_size']}, "
          f"hils_topk={config['hils_topk']}, H_kv={config['num_key_value_heads']}")
    print(f"  num_samples={num_samples}, required B={B}")
    assert num_samples >= B, f"num_samples {num_samples} is less than B={B}"

    all_layer_idxs = sorted({li for s in samples for li in s["layers"].keys()})
    if layer_idx is None:
        layer_idx = all_layer_idxs[0]
        print(f"  Automatically selected the first HiLS layer: layer_idx={layer_idx}")
    else:
        assert layer_idx in all_layer_idxs
    print(f"  Available HiLS layers: {all_layer_idxs}")

    S = config["hils_topk"]
    indices_list, weights_list = [], []
    for i in range(B):
        layer_data = samples[i]["layers"][layer_idx]
        idx = layer_data["indices"]
        cw  = layer_data["chunk_weights"][:, :, :, :S]
        indices_list.append(idx)
        weights_list.append(cw)
    block_indices = torch.cat(indices_list, dim=0).to(dtype=torch.int32, device=device)
    weights       = torch.cat(weights_list, dim=0).to(dtype=torch.bfloat16, device=device)
    actual_seq_len = block_indices.shape[1]
    print(f"  loaded: indices={list(block_indices.shape)}, weights={list(weights.shape)}, "
          f"actual_seq_len={actual_seq_len}")
    return block_indices, weights, actual_seq_len


def _adapt_indices_h_kv(block_indices, weights, target_H_kv):

    real_h_kv = block_indices.shape[2]
    if target_H_kv == real_h_kv:
        return block_indices, weights
    if target_H_kv > real_h_kv:
        assert target_H_kv % real_h_kv == 0, (
            f"target_H_kv={target_H_kv} is not real_h_kv={real_h_kv} an integer multiple of")
        rep = target_H_kv // real_h_kv
        print(f"  H_kv adjust: {real_h_kv} -> {target_H_kv} (repeat_interleave x{rep})")
        block_indices = block_indices.repeat_interleave(rep, dim=2).contiguous()
        if weights is not None:
            weights = weights.repeat_interleave(rep, dim=2).contiguous()
    else:
        print(f"  H_kv adjust: {real_h_kv} -> {target_H_kv} (slice)")
        block_indices = block_indices[:, :, :target_H_kv, :].contiguous()
        if weights is not None:
            weights = weights[:, :, :target_H_kv, :].contiguous()
    return block_indices, weights


def _expand_weights_h_kv_to_hq(weights, HQ):





    H_kv = weights.shape[2]
    if H_kv == HQ:
        return weights.contiguous()
    assert HQ % H_kv == 0, f"HQ={HQ} is not H_kv={H_kv} an integer multiple of"
    G = HQ // H_kv
    return weights.repeat_interleave(G, dim=2).contiguous()


def main_bwd_breakdown_latency(
    B=1, HQ=32, H=8, SEQ_LEN=8192, S=16, D=64, block_size=64,
    real_indices_path="real_indices/indices_8192.pt",
    layer_idx=None,
    M_list=(4, 32),
    nt_fwd_list=(64, 128),
    nt_bwd_list=(64, 128),
    paired_only=False,
    num_warmup=20, num_iters=50,
    mask_last_token=True,
):

    import os
    device = "cuda"
    torch.manual_seed(0)


    def _as_tuple(x):
        if x is None:
            return ()
        if isinstance(x, (list, tuple)):
            return tuple(x)
        return (x,)
    M_list       = _as_tuple(M_list)
    nt_fwd_list  = _as_tuple(nt_fwd_list)
    nt_bwd_list  = _as_tuple(nt_bwd_list)
    assert len(M_list)       > 0, "M_list must not be empty"
    assert len(nt_fwd_list)  > 0, "nt_fwd_list must not be empty"
    assert len(nt_bwd_list)  > 0, "nt_bwd_list must not be empty"

    print("=" * 78)
    print(f"[BWD Breakdown] B={B}, SEQ_LEN={SEQ_LEN}, HQ={HQ}, H={H}, D={D}, "
          f"S={S}, block_size={block_size}")
    print("=" * 78)


    block_indices = None
    real_weights  = None
    actual_L = SEQ_LEN
    if real_indices_path is not None and os.path.exists(real_indices_path):
        block_indices, real_weights, actual_L = _load_real_indices_for_breakdown(
            real_indices_path, B=B, layer_idx=layer_idx, device=device)
        block_indices, real_weights = _adapt_indices_h_kv(block_indices, real_weights, H)

        if actual_L != SEQ_LEN:
            print(f"  Note: real indices ofsequence length {actual_L} != requested {SEQ_LEN}, "
                  f"using actual_L for benchmarking")
        SEQ_LEN = actual_L
    else:
        print(f"  [WARN] not found real_indices_path={real_indices_path}, falling back to"
              f" build_block_indices_block_M(overlap=0.8)")
        block_indices = build_block_indices_block_M(
            B=B, SEQ_LEN=SEQ_LEN, H=H, S=S,
            block_size=block_size, overlap_ratio=0.8,
            block_M=max(M_list),
            device=device,
        )
        real_weights = None




    dtype = torch.bfloat16
    q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device).contiguous()
    k = torch.randn((B, SEQ_LEN, H,  D), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, SEQ_LEN, H,  D), dtype=dtype, device=device).contiguous()
    if real_weights is not None:
        weights = _expand_weights_h_kv_to_hq(real_weights, HQ)
    else:
        weights = torch.randn((B, SEQ_LEN, HQ, S), dtype=dtype, device=device).contiguous()

    print(f"  final shapes: q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}, "
          f"weights={list(weights.shape)}, indices={list(block_indices.shape)}")


    if paired_only:
        assert len(M_list) == len(nt_fwd_list) == len(nt_bwd_list), \
            "paired_only=True requires the three lists to have the same length"
        cfgs = list(zip(M_list, nt_fwd_list, nt_bwd_list))
    else:
        cfgs = [(m, nf, nb)
                for m in M_list
                for nf in nt_fwd_list for nb in nt_bwd_list]

    print(f"\nTotal {len(cfgs)} configurations to benchmark "
          f"(M in {list(M_list)}, nt_fwd in {list(nt_fwd_list)}, "
          f"nt_bwd in {list(nt_bwd_list)}, paired={paired_only})")


    print()
    print("=" * 78)
    print("[HiLS_block_M_head autograd]  using the full autograd.Function path (single-kernel backward)")
    print("=" * 78)
    header_inv = (f"{'M':>3} {'nt_f':>4} {'nt_b':>4} | "
                  f"{'FWD':>8} {'BWD':>8} {'TOTAL':>8} | {'BWD/FWD':>7}")
    print("-" * len(header_inv))
    print(header_inv)
    print("-" * len(header_inv))
    inv_results = []
    for (m, nf, nb) in cfgs:
        try:
            sm_scale_g = 1.0 / math.sqrt(D)
            g_q = q.detach().clone().contiguous().requires_grad_(True)
            g_k = k.detach().clone().contiguous().requires_grad_(True)
            g_v = v.detach().clone().contiguous().requires_grad_(True)
            g_weights = weights.detach().clone().contiguous().requires_grad_(True)
            g_indices = block_indices.detach().clone().contiguous()
            g_do = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device).contiguous()

            def _zero_grads():
                g_q.grad = None; g_k.grad = None; g_v.grad = None; g_weights.grad = None


            for _ in range(num_warmup):
                _ = HiLS_block_M_head(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                )
                g_o = HiLS_block_M_head(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                )
                g_o.backward(g_do)
                _zero_grads()
            torch.cuda.synchronize()

            e_start = torch.cuda.Event(enable_timing=True)
            e_end   = torch.cuda.Event(enable_timing=True)


            torch.cuda.synchronize()
            e_start.record()
            for _ in range(num_iters):
                _ = HiLS_block_M_head(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                )
            e_end.record()
            torch.cuda.synchronize()
            g_fwd_ms = e_start.elapsed_time(e_end) / num_iters


            torch.cuda.synchronize()
            e_start.record()
            for _ in range(num_iters):
                g_o = HiLS_block_M_head(
                    g_q, g_k, g_v, g_weights, g_indices,
                    block_size=block_size, sm_scale=sm_scale_g,
                    block_M=m,
                    num_threads_fwd=nf, num_threads_bwd=nb,
                    mask_last_token=mask_last_token,
                )
                g_o.backward(g_do)
                _zero_grads()
            e_end.record()
            torch.cuda.synchronize()
            g_fwd_bwd_ms = e_start.elapsed_time(e_end) / num_iters
            g_bwd_ms = g_fwd_bwd_ms - g_fwd_ms
            g_total_ms = g_fwd_ms + g_bwd_ms
            g_ratio = g_bwd_ms / max(g_fwd_ms, 1e-6)

            print(f"{m:>3} {nf:>4} {nb:>4} | "
                  f"{g_fwd_ms:>8.3f} {g_bwd_ms:>8.3f} {g_total_ms:>8.3f} | "
                  f"{g_ratio:>7.2f}")
            inv_results.append(dict(
                M=m, nt_fwd=nf, nt_bwd=nb,
                fwd_ms=g_fwd_ms, bwd_ms=g_bwd_ms,
                total_ms=g_total_ms, ratio=g_ratio,
            ))

            del g_q, g_k, g_v, g_weights, g_indices, g_do
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{m:>3} {nf:>4} {nb:>4} |  FAILED: {type(e).__name__}: {e}")
            import traceback as _tb
            _tb.print_exc()
            torch.cuda.empty_cache()
    print("-" * len(header_inv))
    print("=" * 78)


    if inv_results:
        best_total = min(inv_results, key=lambda r: r["total_ms"])
        best_fwd   = min(inv_results, key=lambda r: r["fwd_ms"])
        best_bwd   = min(inv_results, key=lambda r: r["bwd_ms"])

        print()
        print("=" * 78)
        print(" Best configuration(different, HiLS_block_M_head autograd path)")
        print("=" * 78)
        def _fmt(r, tag):
            return (f"  [{tag:<9}] M={r['M']:>2}, "
                    f"nt_f={r['nt_fwd']:>3}, nt_b={r['nt_bwd']:>3} | "
                    f"FWD={r['fwd_ms']:.3f}  BWD={r['bwd_ms']:.3f}  "
                    f"TOTAL={r['total_ms']:.3f}  (BWD/FWD={r['ratio']:.2f})")
        print(_fmt(best_total, "Fastest total"))
        print(_fmt(best_fwd,   "Fastest FWD"))
        print(_fmt(best_bwd,   "Fastest BWD"))
        print("=" * 78)
    return inv_results

