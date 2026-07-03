import torch
import tilelang
import tilelang.language as T
from typing import Optional
import math

def ref_topk_max_pooling(q, k_lmks, topk, block_size, window_size, is_causal=False):
    B, L, h_kv, G, D = q.shape
    S = k_lmks.shape[1]
    sm_scale = 1.0 / math.sqrt(D)
    scores_all = torch.einsum("blhgd,bshd->blhgs", q.float(), k_lmks.float())

    if is_causal:
        i_idx = torch.arange(L, device=q.device).unsqueeze(1) # [L, 1]
        j_idx = torch.arange(S, device=q.device).unsqueeze(0) # [1, S]

        # Mask out chunks covered by sliding window
        threshold_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= threshold_idx

        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all = scores_all.masked_fill(causal_mask_expanded, float('-inf'))

    scores_max_pooling = scores_all.max(dim=3).values
    _, topk_indices = torch.topk(scores_max_pooling, k=topk, dim=-1, sorted=False)
    indices_sorted, order = torch.sort(topk_indices, dim=-1)
    order_expanded = order.unsqueeze(3).expand(-1, -1, -1, G, -1)
    indices_expanded = indices_sorted.unsqueeze(3).expand(-1, -1, -1, G, -1)
    scores_sorted = torch.gather(scores_all, -1, indices_expanded)
    scores_sorted = scores_sorted * sm_scale
    return indices_sorted, scores_sorted


@tilelang.jit(
    out_idx=[2],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fused_topk_max_pooling_kernel(batch, seq_len, s_len, h_kv, groups, head_dim, topk, block_size, window_size, is_causal,
                                  BLOCK_L=None, BLOCK_S=None, threads=None):
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    q_shape = [batch, seq_len, h_kv, groups, head_dim]
    k_shape = [batch, s_len, h_kv, head_dim]
    out_scores_shape = [batch, seq_len, h_kv, groups, topk]
    out_indices_shape = [batch, seq_len, h_kv, topk]

    if BLOCK_L is None:
        BLOCK_L = 4  # {'BLOCK_L': 4, 'BLOCK_S': 16, 'threads': 64}
    if BLOCK_S is None:
        BLOCK_S = 16
    BLOCK_D = head_dim
    if threads is None:
        threads = 64

    GEMM_M = BLOCK_L * groups
    num_s_blocks = tilelang.cdiv(s_len, BLOCK_S)

    if BLOCK_S >= BLOCK_L:
        BLOCK_TK = BLOCK_S // BLOCK_L
        if BLOCK_TK < 1:
            BLOCK_TK = 1
    else:
        BLOCK_TK = 1

    tk_blocks = (topk + BLOCK_TK - 1) // BLOCK_TK

    @T.prim_func
    def fwd_kernel_max_pooling(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        OutIndices: T.Tensor(out_indices_shape, idx_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_S, BLOCK_D], dtype)
            score_shared = T.alloc_shared([GEMM_M, BLOCK_S], accum_dtype)

            acc_s = T.alloc_fragment([GEMM_M, BLOCK_S], accum_dtype)

            topk_max_scores_local = T.alloc_local([topk], accum_dtype)
            topk_indices_local = T.alloc_local([topk], idx_dtype)

            topk_indices_shared = T.alloc_shared([BLOCK_L, topk], idx_dtype)

            T.fill(topk_max_scores_local, -T.infinity(accum_dtype))
            T.fill(topk_indices_local, -1)
            T.fill(topk_indices_shared, -1)

            for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = 0

            loop_limit = num_s_blocks
            if is_causal:
                limit_ts = tilelang.cdiv(base_l + BLOCK_L, block_size)
                loop_limit = T.min(loop_limit, tilelang.cdiv(limit_ts, BLOCK_S) + 1)

            for s_block in T.serial(loop_limit):
                base_s = s_block * BLOCK_S

                for s_idx, d in T.Parallel(BLOCK_S, BLOCK_D):
                    ts = base_s + s_idx
                    if ts < s_len:
                        K_shared[s_idx, d] = K[i_b, ts, i_h, d]
                    else:
                        K_shared[s_idx, d] = 0
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

                if is_causal:
                    for i, j in T.Parallel(GEMM_M, BLOCK_S):
                        l_idx = i // groups
                        tq = base_l + l_idx
                        ts = base_s + j
                        # if ts >= (tq // block_size):
                        if ts >= (tq - window_size + 1) // block_size:
                            score_shared[i, j] = -T.infinity(accum_dtype)

                T.sync_threads()

                tx = T.get_thread_binding()
                my_l_idx = tx
                my_tq = base_l + my_l_idx
                cur_max_val = T.alloc_var(accum_dtype)

                if (my_tq < seq_len) and (tx < BLOCK_L):
                    for s_idx in T.serial(BLOCK_S):
                        ts = base_s + s_idx
                        if ts < s_len:
                            cur_max_val = -T.infinity(accum_dtype)
                            for g in T.serial(groups):
                                val = score_shared[my_l_idx * groups + g, s_idx]
                                if val > cur_max_val:
                                    cur_max_val = val

                            if cur_max_val > topk_max_scores_local[topk - 1]:
                                moving = T.alloc_var("bool")
                                moving = True
                                for kk in T.serial(topk):
                                    k = topk - 1 - kk
                                    if moving:
                                        if (k > 0) and (cur_max_val > topk_max_scores_local[k - 1]):
                                            topk_max_scores_local[k] = topk_max_scores_local[k - 1]
                                            topk_indices_local[k] = topk_indices_local[k - 1]
                                        else:
                                            topk_max_scores_local[k] = cur_max_val
                                            topk_indices_local[k] = ts
                                            moving = False
                T.sync_threads()

            tx = T.get_thread_binding()
            my_l_idx = tx
            my_tq = base_l + my_l_idx
            if (my_tq < seq_len) and (tx < BLOCK_L):
                for k in T.serial(topk):
                    idx_val = topk_indices_local[k]
                    topk_indices_shared[my_l_idx, k] = idx_val
                    OutIndices[i_b, my_tq, i_h, k] = idx_val


    return fwd_kernel_max_pooling



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
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def recompute_topk_max_pooling_scores_kernel(
    batch, seq_len, s_len, h_kv, groups, head_dim, topk,
    BLOCK_L=None, BLOCK_TK=None, threads=None
):
    """
    of topk indices  scores.
    Q:        [B, L, h_kv, G, D]
    K:        [B, S, h_kv, D]
    Indices:  [B, L, h_kv, topk]
    OutScores:[B, L, h_kv, G, topk]
    """
    dtype = "bfloat16"
    accum_dtype = "float"
    idx_dtype = "int32"

    q_shape = [batch, seq_len, h_kv, groups, head_dim]
    k_shape = [batch, s_len, h_kv, head_dim]
    indices_shape = [batch, seq_len, h_kv, topk]
    out_scores_shape = [batch, seq_len, h_kv, groups, topk]

    if BLOCK_L is None:
        BLOCK_L = 2       # {'BLOCK_L': 2, 'BLOCK_TK': 32, 'threads': 64}
    if BLOCK_TK is None:
        BLOCK_TK = 32
    BLOCK_D = head_dim
    if threads is None:
        threads = 64

    GEMM_M = BLOCK_L * groups
    GEMM_N = BLOCK_L * BLOCK_TK
    tk_blocks = (topk + BLOCK_TK - 1) // BLOCK_TK

    sm_scale = 1.0 / math.sqrt(head_dim)

    @T.prim_func
    def fwd_recompute(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(k_shape, dtype),
        Indices: T.Tensor(indices_shape, idx_dtype),
        OutScores: T.Tensor(out_scores_shape, accum_dtype),
    ):
        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L
            Q_shared = T.alloc_shared([GEMM_M, BLOCK_D], dtype)
            K_shared = T.alloc_shared([BLOCK_L * BLOCK_TK, BLOCK_D], dtype)
            score_shared = T.alloc_shared([GEMM_M, GEMM_N], accum_dtype)
            acc_s = T.alloc_fragment([GEMM_M, GEMM_N], accum_dtype)

            for l_idx, g, d in T.Parallel(BLOCK_L, groups, BLOCK_D):
                tq = base_l + l_idx
                flat_m = l_idx * groups + g
                if tq < seq_len:
                    Q_shared[flat_m, d] = Q[i_b, tq, i_h, g, d]
                else:
                    Q_shared[flat_m, d] = T.Cast(dtype, 0.0)

            for tk_block in T.serial(tk_blocks):
                tk_base = tk_block * BLOCK_TK
                tk_size = T.min(BLOCK_TK, topk - tk_base)

                for l_idx, tk_idx, d in T.Parallel(BLOCK_L, BLOCK_TK, BLOCK_D):
                    tq = base_l + l_idx
                    off = l_idx * BLOCK_TK + tk_idx
                    if (tq < seq_len) and (tk_idx < tk_size):
                        k_id = tk_base + tk_idx
                        idx = Indices[i_b, tq, i_h, k_id]
                        if (idx >= 0) and (idx < s_len):
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
                    if (tq < seq_len) and (tk_idx < tk_size):
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
    seq_len: int,
    s_len: int,
    h_kv: int,
    topk: int,
    BLOCK_L: int = 16,
    num_threads: int = 64,
):
    """
     IndicesIn: [B, L, h_kv, topk]  bitonic ( chunk id).
    - with: 0 <= idx < s_len
    - without: idx < 0( -1), is key = s_len, in.
    """

    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"
    assert topk == tilelang.math.next_power_of_2(topk)
    num_iters = int(round(math.log2(topk)))

    indices_shape = [batch, seq_len, h_kv, topk]

    @T.prim_func
    def sort_kernel(
        IndicesIn: T.Tensor(indices_shape, INT32),
        IndicesOut: T.Tensor(indices_shape, INT32),
    ):

        with T.Kernel(tilelang.cdiv(seq_len, BLOCK_L), h_kv, batch, threads=num_threads) as (bx, by, bz):
            i_b = bz
            i_h = by
            base_l = bx * BLOCK_L

            idx_shared = T.alloc_shared([BLOCK_L, topk], dtype=INT32)

            for l_idx, k in T.Parallel(BLOCK_L, topk):
                lq = base_l + l_idx
                if lq < seq_len:
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
                        if lq < seq_len:
                            i_idx = i
                            ixj = i_idx ^ j_step
                            if (ixj > i_idx) and (ixj < topk):
                                val_i = idx_shared[l_idx, i_idx]
                                val_j = idx_shared[l_idx, ixj]

                                if val_i >= 0:
                                    key_i = val_i
                                else:
                                    key_i = s_len
                                if val_j >= 0:
                                    key_j = val_j
                                else:
                                    key_j = s_len

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
                if lq < seq_len:
                    IndicesOut[i_b, lq, i_h, k] = idx_shared[l_idx, k]

    return sort_kernel

class TopKMaxPoolingFusedFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, lmks, topk: int,
                select_kernel,
                sort_kernel,
                recompute_kernel
                ):
        # q:   [B, L, h_kv, G, D]
        # lmks:[B, S, h_kv, D]
        B, L, h_kv, G, D = q.shape
        B2, S, h_kv2, D2 = lmks.shape
        dtype = q.dtype

        assert B == B2 and h_kv == h_kv2 and D == D2

        q_in = q.contiguous()
        k_in = lmks.contiguous()

        indices_raw = select_kernel(q_in, k_in)  # int32

        indices_sorted = sort_kernel(indices_raw)

        # indices_sorted_64 = indices_sorted.to(torch.int64)

        indices_for_kernel = indices_sorted

        best_scores_buf = recompute_kernel(q_in, k_in, indices_for_kernel)  # float32

        ctx.save_for_backward(q_in, k_in, indices_sorted)
        ctx.h_kv = h_kv
        ctx.G = G
        ctx.topk = topk
        ctx.shapes = (B, L, S, h_kv, D)

        return indices_sorted, best_scores_buf.to(dtype)

    @staticmethod
    def backward(ctx, grad_indices_unused, grad_scores_selected):
        q_in, k_in, indices = ctx.saved_tensors
        indices = indices.long()
        B, L, S, h_kv, D = ctx.shapes
        G = ctx.G

        sm_scale = 1.0 / math.sqrt(D)

        grad_scores_dense = torch.zeros(
            (B, L, h_kv, G, S),
            dtype=grad_scores_selected.dtype,
            device=grad_scores_selected.device,
        )

        indices_expanded = indices.unsqueeze(3).expand(-1, -1, -1, G, -1)

        valid_mask = (indices_expanded >= 0) & (indices_expanded < S)
        safe_indices = indices_expanded.clone()
        safe_indices[~valid_mask] = 0
        safe_grad = grad_scores_selected.clone()
        safe_grad[~valid_mask] = 0

        grad_scores_dense.scatter_(4, safe_indices, safe_grad)

        bs_hg = B * h_kv * G

        dense_in = grad_scores_dense.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, S)

        q_flat = q_in.permute(0, 2, 3, 1, 4).reshape(bs_hg, L, D)

        k_expanded = k_in.unsqueeze(3).expand(-1, -1, -1, G, -1)
        k_flat = k_expanded.permute(0, 2, 3, 1, 4).reshape(bs_hg, S, D)

        grad_q_flat = torch.bmm(dense_in, k_flat)
        grad_q = grad_q_flat.view(B, h_kv, G, L, D).permute(0, 3, 1, 2, 4)

        grad_q = grad_q * sm_scale

        grad_k_flat = torch.bmm(dense_in.transpose(1, 2), q_flat)

        grad_k_grouped = grad_k_flat.view(B, h_kv, G, S, D)
        grad_k_sum = grad_k_grouped.sum(dim=2)
        grad_lmks = grad_k_sum.permute(0, 2, 1, 3)  # [B, S, h_kv, D]

        grad_lmks = grad_lmks * sm_scale

        return grad_q, grad_lmks, None, None, None, None


class TopKMaxPooling_Fused(torch.nn.Module):
    def __init__(self, topk, block_size, window_size, is_causal):
        super().__init__()
        self.topk = topk
        self.block_size = block_size
        self.window_size = window_size
        self.is_causal = is_causal
        self._cached_select_kernel = None
        self._cached_sort_kernel = None
        self._cached_recompute_kernel = None
        self._cached_shape = None

    def forward(self, q, lmks):
        # q:   [B, L, h_kv, G, D]
        # lmks:[B, S, h_kv, D]
        B, L, h_kv, G, D = q.shape
        _, S, _, _ = lmks.shape
        topk = self.topk
        block_size = self.block_size
        window_size = self.window_size
        is_causal = self.is_causal

        shape_key = (B, L, S, h_kv, G, D, topk, block_size, window_size, is_causal)
        if self._cached_shape != shape_key:

            self._cached_select_kernel = fused_topk_max_pooling_kernel(
                B, L, S, h_kv, G, D, topk, block_size, window_size, is_causal
            )

            self._cached_sort_kernel = sort_topk_indices_kernel(
                B, L, S, h_kv, topk
            )

            self._cached_recompute_kernel = recompute_topk_max_pooling_scores_kernel(
                B, L, S, h_kv, G, D, topk
            )
            self._cached_shape = shape_key

        select_kernel = self._cached_select_kernel
        sort_kernel = self._cached_sort_kernel
        recompute_kernel = self._cached_recompute_kernel

        indices, scores = TopKMaxPoolingFusedFn.apply(
            q, lmks, topk, select_kernel, sort_kernel, recompute_kernel
        )
        # Reshape scores: [B, L, h_kv, G, topk] -> [B, L, h_q, topk]
        scores = scores.view(B, L, h_kv * G, -1)
        return indices, scores

_MODULE_CACHE = {}
def online_topk_head(q: torch.Tensor, lmks: torch.Tensor, topk: int, block_size: int, window_size: int, is_causal: bool = False):
    """
    Functional API for TopKMaxPooling_Fused

    Args:
        q: [B, L, h_q, D]
        lmks: [B, S, h_kv, D]
        topk: int
        block_size: int
        window_size: int
        is_causal: bool

    Returns:
        indices: [B, L, h_kv, topk]
        scores: [B, L, h_q, topk]
    """
    if q.dim() == 4:
        B, L, h_q, D = q.shape
        h_kv = lmks.shape[2]
        assert h_q % h_kv == 0, f"h_q ({h_q}) must be divisible by h_kv ({h_kv})"
        G = h_q // h_kv
        q = q.view(B, L, h_kv, G, D)

    cache_key = (topk, block_size, window_size, is_causal)
    if cache_key not in _MODULE_CACHE:
        _MODULE_CACHE[cache_key] = TopKMaxPooling_Fused(topk, block_size, window_size, is_causal)

    return _MODULE_CACHE[cache_key](q, lmks)



# ...existing code...
def test_fused_topk_max_pooling_correctness():
    print("\n" + "=" * 70)
    print("=== Testing Fused TopK Max-Pooling Kernel Correctness ===")
    print("=" * 70)

    B, L, D = 64, 4096, 128
    h_kv = 2
    G = 8
    h_q = h_kv * G
    S = 64
    topk = 16
    is_causal = True
    block_size = 64
    window_size = 64

    dtype = torch.bfloat16
    device = "cuda"

    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}")

    torch.manual_seed(4200)

    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)

    # ============ Forward Correctness ============
    print("\n--- Forward Correctness ---")

    # Reference returns: indices [B, L, h_kv, topk], scores [B, L, h_kv, G, topk]
    ref_indices, ref_scores = ref_topk_max_pooling(q.detach(), lmks.detach(), topk, block_size, window_size, is_causal)

    # Fused returns: indices [B, L, h_kv, topk], scores [B, L, h_q, topk]
    fused_indices, fused_scores = online_topk_head(q, lmks, topk, block_size, window_size, is_causal)

    # Reshape fused scores back to [B, L, h_kv, G, topk] for comparison
    fused_scores_reshaped = fused_scores.view(B, L, h_kv, G, topk)

    scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q.float(), lmks.float())
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D))

    if is_causal:
        i_idx = torch.arange(L, device=device).unsqueeze(1)
        j_idx = torch.arange(S, device=device).unsqueeze(0)
        threshold_idx = (i_idx - window_size + 1).div(block_size, rounding_mode='floor')
        causal_mask = j_idx >= threshold_idx
        causal_mask_expanded = causal_mask.view(1, L, 1, 1, S)
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))


    safe_indices = fused_indices.clone()
    safe_indices[safe_indices < 0] = 0

    indices_expanded = safe_indices.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    # Use reshaped fused scores for comparison
    valid_mask = (scores_gathered_ref > -1e9) & (fused_scores_reshaped.float() > -1e9)

    if valid_mask.sum() == 0:
        max_score_diff = 0.0
        rel_l2_score = 0.0
    else:
        score_diff = torch.abs(scores_gathered_ref[valid_mask] - fused_scores_reshaped.float()[valid_mask])
        max_score_diff = score_diff.max().item()
        rel_l2_score = score_diff.norm().item() / (scores_gathered_ref[valid_mask].norm().item() + 1e-6)

    print(f"Forward scores (valid only) - Max Diff: {max_score_diff:.6f}")
    print(f"Forward scores (valid only) - L2 RelErr: {rel_l2_score:.6f}")

    indices_match = (fused_indices.long() == ref_indices.long())
    ref_scores_pooled = ref_scores.max(dim=3).values
    valid_indices_mask = (ref_scores_pooled > -1e9)

    if valid_indices_mask.sum() > 0:
        match_rate = indices_match[valid_indices_mask].float().mean().item()
    else:
        match_rate = 1.0

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

    # grad_output shape: [B, L, h_kv, G, topk]
    grad_output = torch.randn(B, L, h_kv, G, topk, dtype=dtype, device=device)

    # Mask grad_output for invalid positions
    with torch.no_grad():
        _, ref_scores_check = ref_topk_max_pooling(
            q_ref, lmks_ref, topk, block_size, window_size, is_causal
        )
        invalid_mask = (ref_scores_check < -1e9)
        grad_output[invalid_mask] = 0.0

    indices_fused_bwd, scores_fused_bwd = online_topk_head(q_fused, lmks_fused, topk, block_size, window_size, is_causal)

    # Reshape grad_output to [B, L, h_q, topk] for fused backward
    loss_fused = (scores_fused_bwd * grad_output.view(B, L, h_kv * G, topk)).sum()
    loss_fused.backward()
    grad_q_fused = q_fused.grad.clone()
    grad_lmks_fused = lmks_fused.grad.clone()

    scores_all_ref = torch.einsum("blhgd,bshd->blhgs", q_ref.float(), lmks_ref.float())
    scores_all_ref = scores_all_ref * (1.0 / math.sqrt(D)) # Scale!

    if is_causal:
        scores_all_ref = scores_all_ref.masked_fill(causal_mask_expanded, float('-inf'))

    safe_indices_bwd = indices_fused_bwd.clone()
    safe_indices_bwd[safe_indices_bwd < 0] = 0
    indices_expanded = safe_indices_bwd.unsqueeze(3).expand(-1, -1, -1, G, -1).long()
    scores_gathered_ref = torch.gather(scores_all_ref, -1, indices_expanded)

    # Use original grad_output [B, L, h_kv, G, topk] for ref backward
    loss_ref = (scores_gathered_ref * grad_output.float()).sum()
    loss_ref.backward()
    grad_q_ref = q_ref.grad.clone()
    grad_lmks_ref = lmks_ref.grad.clone()

    if torch.isnan(grad_q_ref).any(): grad_q_ref = torch.nan_to_num(grad_q_ref, 0.0)
    if torch.isnan(grad_lmks_ref).any(): grad_lmks_ref = torch.nan_to_num(grad_lmks_ref, 0.0)
    if torch.isnan(grad_q_fused).any(): grad_q_fused = torch.nan_to_num(grad_q_fused, 0.0)
    if torch.isnan(grad_lmks_fused).any(): grad_lmks_fused = torch.nan_to_num(grad_lmks_fused, 0.0)

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


def test_fused_topk_max_pooling_memory_and_speed():
    print("\n" + "=" * 70)
    print("=== Benchmark Fused TopK Max-Pooling Memory and Speed ===")
    print("=" * 70)

    B, L, D = 32, 4096, 128
    h_kv = 2
    G = 8
    h_q = h_kv * G
    S = 64
    topk = 16
    is_causal = True
    block_size = 64
    window_size = 64
    dtype = torch.bfloat16
    device = "cuda"

    print(f"Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G} (h_q={h_q}), D={D}, topk={topk}, is_causal={is_causal}, block_size={block_size}")

    torch.manual_seed(42)
    q = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device, requires_grad=True)
    lmks = torch.randn(B, S, h_kv, D, dtype=dtype, device=device, requires_grad=True)

    # grad_output shape: [B, L, h_q, topk]
    grad_output = torch.randn(B, L, h_kv * G, topk, dtype=dtype, device=device)

    n_iters = 20

    def run_fused():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        _, scores = online_topk_head(q_t, lmks_t, topk, block_size, window_size, is_causal)
        loss = (scores * grad_output).sum()
        loss.backward()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None

            _ = online_topk_head(q_t, lmks_t, topk, block_size, window_size, is_causal)

            _, scores = online_topk_head(q_t, lmks_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        _, scores = online_topk_head(q_t, lmks_t, topk, block_size, window_size, is_causal)
        loss = (scores * grad_output).sum()
        loss.backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)

        # Fwd only
        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _ = online_topk_head(q_t, lmks_t, topk, block_size, window_size, is_causal)
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters

        # Fwd + Bwd
        start_all.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            _, scores = online_topk_head(q_t, lmks_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output).sum()
            loss.backward()
        end_all.record()
        torch.cuda.synchronize()
        avg_all_ms = start_all.elapsed_time(end_all) / n_iters
        avg_bwd_ms = avg_all_ms - avg_fwd_ms

        return peak_mem, avg_fwd_ms, avg_all_ms, avg_bwd_ms

    def run_ref():
        q_t = q.detach().clone().requires_grad_(True)
        lmks_t = lmks.detach().clone().requires_grad_(True)

        # Reshape grad_output for reference: [B, L, h_kv, G, topk]
        grad_output_ref = grad_output.view(B, L, h_kv, G, topk)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        def forward_only():
            _ = ref_topk_max_pooling(q_t, lmks_t, topk, block_size, window_size, is_causal)

        def forward_backward():
            _, scores = ref_topk_max_pooling(q_t, lmks_t, topk, block_size, window_size, is_causal)
            loss = (scores * grad_output_ref).sum()
            loss.backward()

        for _ in range(5):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
            forward_backward()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        q_t.grad = None
        lmks_t.grad = None
        forward_backward()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_all = torch.cuda.Event(enable_timing=True)
        end_all = torch.cuda.Event(enable_timing=True)

        # Fwd only
        start_fwd.record()
        for _ in range(n_iters):
            q_t.grad = None
            lmks_t.grad = None
            forward_only()
        end_fwd.record()
        torch.cuda.synchronize()
        avg_fwd_ms = start_fwd.elapsed_time(end_fwd) / n_iters

        # Fwd + Bwd
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

    # Run benchmarks
    print("\nRunning benchmarks...")

    mem_fused, fwd_fused, all_fused, bwd_fused = run_fused()
    print(f"\n[Fused TopK Max-Pooling]")
    print(f"  Peak Memory: {mem_fused:.2f} MB")
    print(f"  Avg Fwd Latency: {fwd_fused:.2f} ms")
    print(f"  Avg Fwd+Bwd Latency: {all_fused:.2f} ms")
    print(f"  Derived Bwd Latency: {bwd_fused:.2f} ms")

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
        print(f"Speedup (Fwd): {fwd_ref / fwd_fused:.2f}x")
        print(f"Speedup (Bwd): {bwd_ref / bwd_fused:.2f}x")
        print(f"Speedup (Fwd+Bwd): {all_ref / all_fused:.2f}x")
        print(f"Memory Saving: {mem_ref / mem_fused:.2f}x")

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
])
def test_topk_correctness_robust(B, L, S, h_kv, G, D, topk, block_size, window_size):
    device = "cuda"
    dtype = torch.bfloat16
    is_causal = True
    # window_size = block_size # default window_size

    torch.manual_seed(42)

    print(f"\nTesting Config: B={B}, L={L}, S={S}, h_kv={h_kv}, G={G}, D={D}, topk={topk}, BS={block_size}, WS={window_size}")

    # Q: [B, L, h_kv, G, D]
    q_raw = torch.randn(B, L, h_kv, G, D, dtype=dtype, device=device)
    # Lmks: [B, S, h_kv, D]
    lmks_raw = torch.randn(B, S, h_kv, D, dtype=dtype, device=device)

    q_fused = q_raw.clone().detach().requires_grad_(True)
    lmks_fused = lmks_raw.clone().detach().requires_grad_(True)

    q_ref = q_raw.clone().detach().requires_grad_(True)
    lmks_ref = lmks_raw.clone().detach().requires_grad_(True)

    # Fused Kernel
    # indices: [B, L, h_kv, topk]
    # scores:  [B, L, h_q, topk]
    indices_fused, scores_fused = online_topk_head(q_fused, lmks_fused, topk, block_size, window_size, is_causal)

    # Reference
    # scores: [B, L, h_kv, G, topk]
    indices_ref, scores_ref = ref_topk_max_pooling(q_ref, lmks_ref, topk, block_size, window_size, is_causal)

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
# ...existing code...

if __name__ == "__main__":
    test_fused_topk_max_pooling_correctness()
    test_fused_topk_max_pooling_memory_and_speed()
    params_list = [
         (2, 4096, 64, 2, 8, 64, 16, 64, 64),
        (1, 2048, 64, 1, 8, 64, 16, 32, 32),
        (3, 2048, 64, 1, 8, 64, 8, 32, 32),
        (3, 2048, 64, 1, 8, 64, 8, 32, 40),
        (3, 2048, 64, 1, 8, 64, 8, 32, 64),
    ]
    for p in params_list:
        test_topk_correctness_robust(*p)
