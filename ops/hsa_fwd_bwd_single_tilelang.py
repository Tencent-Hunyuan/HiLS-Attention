# ruff: noqa
import torch

import tilelang
from tilelang import language as T
import tilelang.testing

tilelang.testing.set_random_seed(0)


from einops import rearrange
def hsa_torch_ref(q, k, v, weights, indices, *, chunk_size: int, sm_scale: float, block_q: int):
    """
    (same formula as test_group_qa):
    - items query block in  K :
      p = softmax_s(q @ k^T * sm_scale)
      o_k = p @ v
    - output: o = sum_k (weights[:, :, :, k] * o_k)

    shape:
    - q: (B, L, HQ, D)
    - k, v: (B, L, H, D)
    - weights: (B, q_blocks, H, K)
    - indices: (B, q_blocks, H, K)  (B, L, H, K)  L == q_blocks * block_q
    - return: o_ref: (B, L, HQ, D) float32
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
    k_chunks = rearrange(k, 'B (N S) h d -> B N S h d', S=chunk_size)
    v_chunks = rearrange(v, 'B (N S) h d -> B N S h d', S=chunk_size)

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

    p = torch.softmax(qk, dim=3)

    # o_k: (B, Bq, X, K, HQ, D)
    o_k = torch.einsum('b q x s k h, b q s k h d -> b q x k h d', p, v_.float())

    w_masked = weights.clone()
    w_masked = w_masked.masked_fill(~valid_mask, 0)
    w_exp = torch.repeat_interleave(w_masked, dim=-2, repeats=G).float()  # (B, Bq, HQ, K)

    o_ref = torch.einsum('b q x k h d, b q h k -> b q x h d', o_k, w_exp)
    o_ref = rearrange(o_ref, 'B Bq X hq d -> B (Bq X) hq d')
    return o_ref.to(torch.float32)



def make_dq_layout_hsa(dQ):

    NV, B, L, HQ, D = dQ.shape
    return T.Layout(dQ.shape,
    lambda nv, b, l, h, d:   [nv,b,l, h//8, d//16, (d%16)//2, (h%8), (d%2)]
 )

@tilelang.jit(
    out_idx=[1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
)
def hsa_bwd_postprocess(nv, batch, q_len, heads, head_dim):
    shape = [nv, batch, q_len, heads, head_dim]
    accum_dtype = "float"
    dtype = "bfloat16"
    blk = 64

    @T.prim_func
    def hsa_post(
            dQ_swizzled: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
    ):
        with T.Kernel(T.ceildiv(q_len, blk), heads, batch * nv, threads=128) as (bx, by, bz):
            i_nv = bz // batch
            i_b = bz % batch

            T.annotate_layout({dQ_swizzled: make_dq_layout_hsa(dQ_swizzled)})

            T.copy(
                dQ_swizzled[i_nv, i_b, bx * blk:(bx + 1) * blk, by, :],
                dQ_out[i_nv, i_b, bx * blk:(bx + 1) * blk, by, :],
            )
    return hsa_post




@tilelang.jit(
    out_idx=[-1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def hierarchical_sparse_attention(batch,
                                  heads,
                                  q_len,
                                  kv_len,
                                  head_dim,
                                  scale=None,
                                  block_size=64,
                                  groups=16,
                                  selected_blocks=16):
    if scale is None:
        scale = (1.0 / head_dim)**0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504  # log2(e)

    head_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    kv_shape = [batch, kv_len, head_kv, head_dim]
    weight_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_shape = [batch, q_len, head_kv, selected_blocks]
    block_indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(head_dim))

    NK = tilelang.cdiv(head_dim, block_T)
    NV = tilelang.cdiv(head_dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T
    num_stages = 2
    threads = 32

    @T.prim_func
    def hsa(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(kv_shape, dtype),
            V: T.Tensor(kv_shape, dtype),
            W: T.Tensor[weight_shape, accum_dtype],
            BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
            Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(q_len, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)

            i_t, i_v, i_bh = bx, by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, i_t, i_h * G:(i_h + 1) * G, :], Q_shared)


            T.fill(acc_o, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i_1 in T.Pipelined(NS, num_stages=num_stages):
                blk_idx = BlockIndices[i_b, i_t, i_h, i_1]
                i_s = blk_idx * BS
                if i_s >= 0:
                    T.copy(K[i_b, i_s:i_s + BS, i_h, :], K_shared)
                    chunk_weight = W[i_b, i_t, i_h, i_1]

                    T.clear(acc_s)

                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow)

                    # Softmax
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for g, v in T.Parallel(G, BS):
                        acc_s[g, v] = T.exp2(acc_s[g, v] * scale - scores_max[g] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for g, v in T.Parallel(G, BS):
                        acc_s[g, v] = chunk_weight * acc_s[g, v] / scores_sum[g]
                    T.copy(acc_s, acc_s_cast)

                    # V * softmax(Q * K)
                    T.copy(V[i_b, i_s:i_s + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, i_t, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV])

    return hsa

@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
})
def hierarchical_sparse_attention_bwd_dqkv(
    batch,
    heads,
    q_len,
    kv_len,
    head_dim,
    scale=None,
    block_size=64,
    groups=16,
    selected_blocks=16,
    dtype="bfloat16",
    accum_dtype="float",
):
    if scale is None:
        sm_scale = (1.0 / head_dim)**0.5
    else:
        sm_scale = scale

    scale = sm_scale * 1.44269504  # log2(e)

    from tilelang import language as T

    B = batch
    BS = block_size
    G = groups
    V = head_dim
    K = head_dim
    BK = tilelang.next_power_of_2(K)
    BV = min(128, tilelang.next_power_of_2(head_dim))
    NS_kv = tilelang.cdiv(kv_len, BS)
    NV = tilelang.cdiv(V, BV)
    S = selected_blocks
    # NV=1

    heads_kv = heads // groups
    q_shape = [batch, q_len, heads, head_dim]
    k_shape = [batch, kv_len, heads_kv, head_dim]
    v_shape = [batch, kv_len, heads_kv, head_dim]
    o_shape = [batch, q_len, heads, head_dim]
    do_shape = [batch, q_len, heads, head_dim]
    dq_shape = [NV, batch, q_len, heads, head_dim]
    dk_shape = [NV, batch, kv_len, heads_kv, head_dim]
    dv_shape = [batch, kv_len, heads_kv, head_dim]

    weight_shape = [batch, q_len, heads_kv, selected_blocks]
    dw_shape = [batch, q_len, heads_kv, selected_blocks]
    block_mask_shape = [batch, q_len, heads_kv, NS_kv]


    num_threads = 128
    num_stages = 0

    @T.prim_func
    def hsa_bwd_dqkv(
            Q: T.Tensor(q_shape, dtype),
            K: T.Tensor(k_shape, dtype),
            V: T.Tensor(v_shape, dtype),
            W: T.Tensor(weight_shape, accum_dtype),
            DO: T.Tensor(do_shape, dtype),
            DQ: T.Tensor(dq_shape, accum_dtype),
            DK: T.Tensor(dk_shape, dtype),
            DV: T.Tensor(dv_shape, dtype),
            DW: T.Tensor(dw_shape, accum_dtype),
            BlockMask: T.Tensor(block_mask_shape, "int32"),

    ):
        with T.Kernel(NV, NS_kv, B * heads_kv, threads=num_threads) as (i_v, i_s, i_bh):

            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            Q_shared = T.alloc_shared([G, BK], dtype)
            dO_shared = T.alloc_shared([G, BV], dtype)

            qk = T.alloc_fragment([G, BS], accum_dtype)
            P_raw = T.alloc_fragment([G, BS], accum_dtype)
            P_raw_shared = T.alloc_shared([G, BS], dtype)
            T_raw = T.alloc_fragment([G, BV], accum_dtype)

            dV_PdO = T.alloc_fragment([G, BS], accum_dtype)
            dS = T.alloc_fragment([G, BS], accum_dtype)

            dK_accum = T.alloc_fragment([BS, BK], accum_dtype)
            dV_accum = T.alloc_fragment([BS, BV], accum_dtype)

            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)



            i_b, i_h = i_bh // heads_kv, i_bh % heads_kv
            i_s_global = i_s * BS

            T.copy(K[i_b, i_s_global:i_s_global + BS, i_h, :BK], K_shared)
            T.copy(V[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV], V_shared)

            T.fill(dK_accum, 0)
            T.fill(dV_accum, 0)

            T.annotate_layout({
                DQ: make_dq_layout_hsa(DQ),
                K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                V_shared: tilelang.layout.make_swizzled_layout(V_shared),
            })

            for i_q in T.Pipelined(q_len, num_stages=num_stages):
                found_pos = T.alloc_var("int32")
                found_pos = BlockMask[i_b, i_q, i_h, i_s]

                if found_pos != -1:
                    T.copy(Q[i_b, i_q, i_h * G:(i_h + 1) * G, :BK], Q_shared)
                    T.copy(DO[i_b, i_q, i_h * G:(i_h + 1) * G, i_v * BV:(i_v + 1) * BV], dO_shared)

                    chunk_weight = W[i_b, i_q, i_h, found_pos]

                    T.clear(qk)
                    T.gemm(Q_shared, K_shared, qk, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(qk, scores_max, dim=1, clear=True)
                    for g, i in T.Parallel(G, BS):
                        P_raw[g, i] = T.exp2((qk[g, i] * scale - scores_max[g] * scale))
                    T.fill(scores_sum, 0)
                    T.reduce_sum(P_raw, scores_sum, dim=1, clear=True)
                    for g, i in T.Parallel(G, BS):
                        P_raw[g, i] = P_raw[g, i] / scores_sum[g]

                    T.copy(P_raw, P_raw_shared)

                    T.clear(T_raw)
                    T.gemm(P_raw_shared, V_shared, T_raw, policy=T.GemmWarpPolicy.FullRow)


                    dw_local_partial = T.alloc_shared([G, BV], accum_dtype)
                    dw_local_g = T.alloc_shared([G], accum_dtype)
                    dw_sum=T.alloc_shared([1], accum_dtype)
                    for g, v in T.Parallel(G, BV):
                        dw_local_partial[g, v] = dO_shared[g, v] * T_raw[g, v]
                    T.reduce_sum(dw_local_partial, dw_local_g, dim=1, clear=True)
                    T.reduce_sum(dw_local_g, dw_sum, dim=0, clear=True)
                    DW[i_b, i_q, i_h, found_pos]=dw_sum[0]


                    dO_weighted = T.alloc_fragment([G, BV], accum_dtype)
                    for g, v in T.Parallel(G, BV):
                        dO_weighted[g, v] = chunk_weight * dO_shared[g, v]
                    dO_weighted_cast = T.alloc_shared([G, BV], dtype)
                    T.copy(dO_weighted, dO_weighted_cast)

                    T.gemm(P_raw_shared, dO_weighted_cast, dV_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    T.clear(dV_PdO)
                    T.gemm(dO_weighted_cast, V_shared, dV_PdO, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)


                    delta_temp = T.alloc_fragment([G, BS], accum_dtype)
                    delta = T.alloc_fragment([G], accum_dtype)
                    for g, i in T.Parallel(G, BS):
                        delta_temp[g, i] = P_raw[g, i] * dV_PdO[g, i]
                    T.reduce_sum(delta_temp, delta, dim=1, clear=True)

                    for g, i in T.Parallel(G, BS):
                        dS[g, i] =  P_raw[g, i] * (dV_PdO[g, i] - delta[g]) * sm_scale

                    # dS_cast = T.alloc_shared([G, BS], dtype) # Removed
                    # T.copy(dS, dS_cast)
                    T.copy(dS, P_raw_shared) # Reuse P_raw_shared

                    # T.gemm(dS_cast, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(P_raw_shared, Q_shared, dK_accum, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)

                    dQ_local = T.alloc_fragment([G, BK], accum_dtype)
                    T.clear(dQ_local)
                    # T.gemm(dS_cast, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(P_raw_shared, K_shared, dQ_local, policy=T.GemmWarpPolicy.FullRow)

                    for g, k in T.Parallel(G, BK):
                        T.atomic_add(DQ[i_v, i_b, i_q, i_h * G + g, k], dQ_local[g, k])

            # Reuse K_shared and V_shared for output
            T.copy(dK_accum, K_shared)
            T.copy(dV_accum, V_shared)
            T.copy(K_shared, DK[i_v, i_b, i_s_global:i_s_global + BS, i_h, :BK])
            T.copy(V_shared, DV[i_b, i_s_global:i_s_global + BS, i_h, i_v * BV:(i_v + 1) * BV])

    return hsa_bwd_dqkv




@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def hsa_kernel_block_mask(
    batch,
    heads,
    q_len,
    kv_len,
    selected_blocks,
    block_size,
    dtype="int32",
):
    """
    : (q_len, batch, heads * S) ,
    BlockMask[b, t, h, kv_blk] = s ( kv_blk in S itemsininof)
                               = -1 (in)
    """
    from tilelang import language as T

    S = selected_blocks
    NS_kv = tilelang.cdiv(kv_len, block_size)

    block_indices_shape = [batch, q_len, heads, selected_blocks]
    block_mask_shape = [batch, q_len, heads, NS_kv]

    @T.prim_func
    def build_block_mask(
        BlockIndices: T.Tensor(block_indices_shape, dtype),
        BlockMask: T.Tensor(block_mask_shape, dtype),
    ):
        with T.Kernel(q_len, batch, heads * S) as (i_t, i_b, i_hs):
            i_h = i_hs // S
            i_s = i_hs % S

            block_idx = BlockIndices[i_b, i_t, i_h, i_s]

            if block_idx >= 0 and block_idx < NS_kv:
                BlockMask[i_b, i_t, i_h, block_idx] = i_s

    return build_block_mask



class _HSA_single(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, w, block_indices,
                block_size: int, sm_scale: float | None):
        """
        q: (B, L, HQ, D)
        k,v: (B, L, H, D)
        w: (B, L, H, S)
        block_indices: (B, L, H, S), int32,  block_size isof KV block
        """
        assert q.is_cuda and k.is_cuda and v.is_cuda and w.is_cuda
        assert block_indices.is_cuda

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        G = HQ // H
        assert HQ % H == 0
        assert L == k.shape[1] == v.shape[1]

        if sm_scale is None:
            import math
            sm_scale = 1.0 / math.sqrt(D)

        fwd_kernel = hierarchical_sparse_attention(
            batch=B,
            heads=HQ,
            q_len=L,
            kv_len=L,
            head_dim=D,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
            scale=sm_scale,
        )

        O = fwd_kernel(q, k, v, w, block_indices)

        ctx.save_for_backward(q, k, v, w, block_indices)
        ctx.block_size = block_size
        ctx.sm_scale = sm_scale
        ctx.G = G

        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, w, block_indices = ctx.saved_tensors
        block_size = ctx.block_size
        sm_scale = ctx.sm_scale
        G = ctx.G

        B, L, HQ, D = q.shape
        H = k.shape[2]
        S = block_indices.shape[-1]
        device = q.device

        build_mask = hsa_kernel_block_mask(
            batch=B,
            heads=H,
            q_len=L,
            kv_len=L,
            selected_blocks=S,
            block_size=block_size,
        )
        NS_kv = L // block_size
        block_mask = torch.full(
            (B, L, H, NS_kv),
            -1,
            dtype=torch.int32,
            device=device,
        )
        build_mask(block_indices, block_mask)

        bwd_kernel = hierarchical_sparse_attention_bwd_dqkv(
            batch=B,
            heads=HQ,
            q_len=L,
            kv_len=L,
            head_dim=D,
            block_size=block_size,
            groups=G,
            selected_blocks=S,
            scale=sm_scale,
        )
        NV = tilelang.cdiv(D, min(128, tilelang.next_power_of_2(D)))

        DQ = torch.zeros(
            (NV, B, L, HQ, D),
            dtype=torch.float32,
            device=device,
        )
        DK = torch.zeros(
            (NV, B, L, H, D),
            dtype=torch.bfloat16,
            device=device,
        )
        DV = torch.zeros(
            (B, L, H, D),
            dtype=torch.bfloat16,
            device=device,
        )
        DW = torch.zeros(
            (B, L, H, S),
            dtype=torch.float32,
            device=device,
        )

        bwd_kernel(q, k, v, w, dO, DQ, DK, DV, DW, block_mask)

        post_kernel = hsa_bwd_postprocess(NV, B, L, HQ, D)
        DQ = post_kernel(DQ)
        DQ = DQ.sum(0)         # sum over NV

        DK = DK.sum(0)

        return DQ, DK, DV, DW, None, None, None


def HSA_single(q, k, v, w, block_indices,
               block_size: int = 64,
               sm_scale: float | None = None):
    return _HSA_single.apply(q, k, v, w, block_indices, block_size, sm_scale)




import math
import torch
import torch.nn.functional as F
from einops import rearrange

def main_block_M_correctness():

    import math
    import torch
    import torch.nn.functional as F
    from einops import rearrange
    B, SEQ_LEN, H, HQ, D, S, block_size = 1, 1024, 1, 8, 128, 8, 32
    dtype = torch.bfloat16
    device = "cuda"
    G = HQ // H
    scale = 1.0 / math.sqrt(D)

    print(f"Correctness Config: Batch={B}, SeqLen={SEQ_LEN}, H={H}, HQ={HQ}, D={D}, G={G}, S={S}, BlockSize={block_size}")

    torch.manual_seed(42)

    block_indices = torch.full((B, SEQ_LEN, H, S), SEQ_LEN, dtype=torch.int32, device=device)
    num_blocks = SEQ_LEN // block_size
    for b in range(B):
        for t in range(SEQ_LEN):
            for h in range(H):
                max_blocks = min(t // block_size + 1, num_blocks)
                if max_blocks > 0:
                    num_select = min(S, max_blocks)
                    selected = torch.randperm(max_blocks, device=device)[:num_select]
                    block_indices[b, t, h, :num_select] = selected

    block_indices = block_indices.sort(-1)[0]

    block_indices[block_indices == SEQ_LEN] = -1

    Q = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device, requires_grad=True)
    K = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)
    V = torch.randn((B, SEQ_LEN, H, D), dtype=dtype, device=device, requires_grad=True)

    logits = torch.randn((B, SEQ_LEN, H, S), dtype=torch.float32, device=device)
    valid_mask = (block_indices != -1)
    logits = logits.masked_fill(~valid_mask, -1e9)
    W = F.softmax(logits, dim=-1).requires_grad_(True)


    grad_output = torch.randn((B, SEQ_LEN, HQ, D), dtype=dtype, device=device)



    O_ref = hsa_torch_ref(
        Q.float().detach(),
        K.float().detach(),
        V.float().detach(),
        W.detach(),
        block_indices,
        chunk_size=block_size,
        sm_scale=scale,
        block_q=1
    )



    Q.grad = None
    K.grad = None
    V.grad = None
    W.grad = None

    O_ref_bwd = hsa_torch_ref(
        Q.float(), K.float(), V.float(), W.float(), block_indices,
        chunk_size=block_size, sm_scale=scale, block_q=1
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


    def compute_grad_diff(grad_hsa, grad_ref, name):
        if grad_hsa is None or grad_ref is None:
            print(f"{name}: is None")
            return None
        diff = (grad_hsa.float() - grad_ref.float()).abs()
        max_diff = diff.max().item()
        print(f"{name}max error: {max_diff:.6e}")
        return max_diff


    BLOCK_Q = 1
    q_blocks = SEQ_LEN // BLOCK_Q

    indices_blocks = block_indices.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()
    weights_blocks = W.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :].contiguous()

    indices_blocks_hsa = indices_blocks.clone()
    safe_block = max(0, (SEQ_LEN // block_size) - 1)
    invalid_mask_blocks = (indices_blocks_hsa < 0)
    indices_blocks_hsa[indices_blocks_hsa < 0] = safe_block

    weights_blocks_hsa = weights_blocks.detach().clone()
    weights_blocks_hsa.requires_grad_(True)
    indices_blocks_hsa = indices_blocks_hsa.contiguous()
    weights_blocks_hsa = weights_blocks_hsa.contiguous()

    min_G = 16
    pad_ratio = max(1, (min_G + G - 1) // G)
    HQ_padded = pad_ratio * G * H
    need_padding = (HQ_padded > HQ)

    if need_padding:
        pad_heads = HQ_padded - HQ
        Q_hsa = torch.cat([Q.detach(), torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2).clone().requires_grad_(True)
        grad_output_hsa = torch.cat([grad_output, torch.zeros(B, SEQ_LEN, pad_heads, D, dtype=dtype, device=device)], dim=2)
    else:
        Q_hsa = Q.detach().clone().requires_grad_(True)
        grad_output_hsa = grad_output

    K.grad = None
    V.grad = None

    O_triton_hsa_padded = HSA_single(Q_hsa, K, V, weights_blocks_hsa, indices_blocks_hsa, block_size=block_size, sm_scale=scale)
    O_triton_hsa = O_triton_hsa_padded[:, :, :HQ, :].float()
    print()
    print("[Triton HSA] vs [Torch Reference]:")
    fwd_err = (O_triton_hsa - O_ref.float()).abs().max().item()
    print(f"max forward error: {fwd_err:.6e}")

    # backward
    O_triton_hsa_padded.backward(grad_output_hsa)

    DQ_triton = Q_hsa.grad[:, :, :HQ, :].clone() if Q_hsa.grad is not None else None
    DK_triton = K.grad.clone() if K.grad is not None else None
    DV_triton = V.grad.clone() if V.grad is not None else None
    DW_triton_blocks = weights_blocks_hsa.grad.clone() if weights_blocks_hsa.grad is not None else None

    DW_ref_blocks = DW_ref.view(B, q_blocks, BLOCK_Q, H, S)[:, :, 0, :, :]

    compute_grad_diff(DQ_triton, DQ_ref, "DQ")
    compute_grad_diff(DK_triton, DK_ref, "DK")
    compute_grad_diff(DV_triton, DV_ref, "DV")
    DW_triton_blocks = weights_blocks_hsa.grad.clone()
    DW_triton_blocks[invalid_mask_blocks] = 0
    compute_grad_diff(DW_triton_blocks, DW_ref_blocks, "DW")


if __name__ == "__main__":
    main_block_M_correctness()
