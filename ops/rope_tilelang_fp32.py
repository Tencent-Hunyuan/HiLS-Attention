import torch
import tilelang
from tilelang import language as T

@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def rope_fwd_kernel(batch, seq_len, heads_q, heads_k, dim, threads=64):

    dtype = "bfloat16"
    compute_dtype = "float32"

    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        Q: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K: T.Buffer((batch, seq_len, heads_k, dim), dtype),
        Cos: T.Buffer((batch, seq_len, dim), dtype),
        Sin: T.Buffer((batch, seq_len, dim), dtype),
        Q_out: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K_out: T.Buffer((batch, seq_len, heads_k, dim), dtype),
    ):

        total_tokens = batch * seq_len

        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len
            cos_frag = T.alloc_fragment((dim,), compute_dtype)
            sin_frag = T.alloc_fragment((dim,), compute_dtype)

            for d in T.Parallel(dim):
                cos_frag[d] = T.cast(Cos[b_idx, l_idx, d], compute_dtype)
                sin_frag[d] = T.cast(Sin[b_idx, l_idx, d], compute_dtype)

            for h in T.unroll(heads_q):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half

                    q1 = T.cast(Q[b_idx, l_idx, h, idx_1], compute_dtype)
                    q2 = T.cast(Q[b_idx, l_idx, h, idx_2], compute_dtype)

                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]

                    o1 = q1 * c_val - q2 * s_val
                    o2 = q2 * c_val + q1 * s_val

                    Q_out[b_idx, l_idx, h, idx_1] = T.cast(o1, dtype)
                    Q_out[b_idx, l_idx, h, idx_2] = T.cast(o2, dtype)

            for h in T.unroll(heads_k):
                for d in T.Parallel(dim // 2):
                    idx_1 = d
                    idx_2 = d + dim_half

                    k1 = T.cast(K[b_idx, l_idx, h, idx_1], compute_dtype)
                    k2 = T.cast(K[b_idx, l_idx, h, idx_2], compute_dtype)

                    c_val = cos_frag[idx_1]
                    s_val = sin_frag[idx_1]

                    o1 = k1 * c_val - k2 * s_val
                    o2 = k2 * c_val + k1 * s_val

                    K_out[b_idx, l_idx, h, idx_1] = T.cast(o1, dtype)
                    K_out[b_idx, l_idx, h, idx_2] = T.cast(o2, dtype)

    return main



def rope_rotary_pos_emb(q, k, cos, sin):
    """
    Apply Rotary Positional Embedding using TileLang (Fused & FP32 Compute).

    Args:
        q: [batch, seq_len, heads_q, dim] (BF16)
        k: [batch, seq_len, heads_k, dim] (BF16)
        cos: [batch, seq_len, dim] (BF16)
        sin: [batch, seq_len, dim] (BF16)

    Returns:
        q_out, k_out: [batch, seq_len, heads, dim] (BF16)
    """
    B, L, HQ, D = q.shape
    H = k.shape[2]

    kernel_func = rope_fwd_kernel(B, L, HQ, H, D)

    return kernel_func(q, k, cos, sin)



@tilelang.jit(
    out_idx=[4, 5],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def hsa_intra_chunk_rope_kernel(batch, seq_len, heads_q, heads_k, dim, chunk_size_q, chunk_size_k, threads=64):
    dtype = "bfloat16"
    compute_dtype = "float32"

    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        Q: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K: T.Buffer((batch, seq_len, heads_k, dim), dtype),
        Q_cos: T.Buffer((chunk_size_q, dim), dtype),
        Q_sin: T.Buffer((chunk_size_q, dim), dtype),
        Q_out: T.Buffer((batch, seq_len, heads_q, dim), dtype),
        K_out: T.Buffer((batch, seq_len, heads_k, dim), dtype),
        K_cos: T.Buffer((chunk_size_k, dim), dtype),
        K_sin: T.Buffer((chunk_size_k, dim), dtype),
    ):
        total_tokens = batch * seq_len
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len

            q_cos_idx = l_idx % chunk_size_q
            q_cos_lo = T.alloc_fragment((dim_half,), compute_dtype)
            q_sin_lo = T.alloc_fragment((dim_half,), compute_dtype)
            q_cos_hi = T.alloc_fragment((dim_half,), compute_dtype)
            q_sin_hi = T.alloc_fragment((dim_half,), compute_dtype)

            for d in T.Parallel(dim_half):
                q_cos_lo[d] = T.cast(Q_cos[q_cos_idx, d], compute_dtype)
                q_sin_lo[d] = T.cast(Q_sin[q_cos_idx, d], compute_dtype)
                q_cos_hi[d] = T.cast(Q_cos[q_cos_idx, d + dim_half], compute_dtype)
                q_sin_hi[d] = T.cast(Q_sin[q_cos_idx, d + dim_half], compute_dtype)

            for h in T.unroll(heads_q):
                for d in T.Parallel(dim_half):
                    q1 = T.cast(Q[b_idx, l_idx, h, d], compute_dtype)
                    q2 = T.cast(Q[b_idx, l_idx, h, d + dim_half], compute_dtype)

                    o1 = q1 * q_cos_lo[d] - q2 * q_sin_lo[d]
                    o2 = q2 * q_cos_hi[d] + q1 * q_sin_hi[d]

                    Q_out[b_idx, l_idx, h, d] = T.cast(o1, dtype)
                    Q_out[b_idx, l_idx, h, d + dim_half] = T.cast(o2, dtype)

            k_cos_idx = l_idx % chunk_size_k
            k_cos_lo = T.alloc_fragment((dim_half,), compute_dtype)
            k_sin_lo = T.alloc_fragment((dim_half,), compute_dtype)
            k_cos_hi = T.alloc_fragment((dim_half,), compute_dtype)
            k_sin_hi = T.alloc_fragment((dim_half,), compute_dtype)

            for d in T.Parallel(dim_half):
                k_cos_lo[d] = T.cast(K_cos[k_cos_idx, d], compute_dtype)
                k_sin_lo[d] = T.cast(K_sin[k_cos_idx, d], compute_dtype)
                k_cos_hi[d] = T.cast(K_cos[k_cos_idx, d + dim_half], compute_dtype)
                k_sin_hi[d] = T.cast(K_sin[k_cos_idx, d + dim_half], compute_dtype)

            for h in T.unroll(heads_k):
                for d in T.Parallel(dim_half):
                    k1 = T.cast(K[b_idx, l_idx, h, d], compute_dtype)
                    k2 = T.cast(K[b_idx, l_idx, h, d + dim_half], compute_dtype)

                    o1 = k1 * k_cos_lo[d] - k2 * k_sin_lo[d]
                    o2 = k2 * k_cos_hi[d] + k1 * k_sin_hi[d]

                    K_out[b_idx, l_idx, h, d] = T.cast(o1, dtype)
                    K_out[b_idx, l_idx, h, d + dim_half] = T.cast(o2, dtype)

    return main


def hsa_intra_chunk_rope(q, k, q_cos, q_sin, k_cos, k_sin):
    """
    Python wrapper for HSA Intra-Chunk RoPE (training-only, merged version with L_q == L_k).

    The caller only needs to pass in the cos/sin chunk slices (zero-copy slice);
    the kernel indexes into them with a `%` operation internally.

    Args:
        q:     (B, L, H_q, D)  - BF16, Q tensor
        k:     (B, L, H_k, D)  - BF16, K tensor (L must match Q)
        q_cos: (B, chunk_size_q, D) - BF16, cos chunk slice for Q
        q_sin: (B, chunk_size_q, D) - BF16, sin chunk slice for Q
        k_cos: (B, chunk_size_k, D) - BF16, cos chunk slice for K
        k_sin: (B, chunk_size_k, D) - BF16, sin chunk slice for K

    Returns:
        q_out: (B, L, H_q, D) - BF16
        k_out: (B, L, H_k, D) - BF16
    """
    B, L, HQ, D = q.shape
    assert k.shape[1] == L, f"Training-only version requires L_q == L_k, but got L_q={L}, L_k={k.shape[1]}"
    HK = k.shape[2]

    if q_cos.ndim == 3:
        q_cos = q_cos[0]
        q_sin = q_sin[0]
    if k_cos.ndim == 3:
        k_cos = k_cos[0]
        k_sin = k_sin[0]

    q_cos = q_cos.contiguous()
    q_sin = q_sin.contiguous()
    k_cos = k_cos.contiguous()
    k_sin = k_sin.contiguous()

    CS_q = q_cos.shape[0]
    CS_k = k_cos.shape[0]

    kernel_func = hsa_intra_chunk_rope_kernel(B, L, HQ, HK, D, CS_q, CS_k)

    return kernel_func(q, k, q_cos, q_sin, k_cos, k_sin)


class HSAIntraChunkRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, q_cos, q_sin, k_cos, k_sin):
        q_out, k_out = hsa_intra_chunk_rope(q, k, q_cos, q_sin, k_cos, k_sin)
        ctx.save_for_backward(q_cos, q_sin, k_cos, k_sin)
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        q_cos, q_sin, k_cos, k_sin = ctx.saved_tensors
        dq_out, dk_out = hsa_intra_chunk_rope(
            dq.contiguous(), dk.contiguous(),
            q_cos, -q_sin, k_cos, -k_sin
        )
        return dq_out, dk_out, None, None, None, None


def hsa_intra_chunk_rope_autograd(q, k, q_cos, q_sin, k_cos, k_sin):
    return HSAIntraChunkRoPEFunction.apply(q, k, q_cos, q_sin, k_cos, k_sin)



@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def single_tensor_rope_kernel(batch, seq_len, heads, dim, cos_sin_dtype="bfloat16", threads=64):
    x_dtype = "bfloat16"
    compute_dtype = "float32"

    assert dim % 2 == 0
    dim_half = dim // 2

    @T.prim_func
    def main(
        X: T.Buffer((batch, seq_len, heads, dim), x_dtype),
        X_cos: T.Buffer((seq_len, dim), cos_sin_dtype),
        X_sin: T.Buffer((seq_len, dim), cos_sin_dtype),
        X_out: T.Buffer((batch, seq_len, heads, dim), cos_sin_dtype),
    ):
        total_tokens = batch * seq_len
        with T.Kernel(total_tokens, threads=threads) as bx:
            b_idx = bx // seq_len
            l_idx = bx % seq_len

            cos_lo = T.alloc_fragment((dim_half,), compute_dtype)
            sin_lo = T.alloc_fragment((dim_half,), compute_dtype)
            cos_hi = T.alloc_fragment((dim_half,), compute_dtype)
            sin_hi = T.alloc_fragment((dim_half,), compute_dtype)

            for d in T.Parallel(dim_half):
                cos_lo[d] = T.cast(X_cos[l_idx, d], compute_dtype)
                sin_lo[d] = T.cast(X_sin[l_idx, d], compute_dtype)
                cos_hi[d] = T.cast(X_cos[l_idx, d + dim_half], compute_dtype)
                sin_hi[d] = T.cast(X_sin[l_idx, d + dim_half], compute_dtype)

            for h in T.unroll(heads):
                for d in T.Parallel(dim_half):
                    x1 = T.cast(X[b_idx, l_idx, h, d], compute_dtype)
                    x2 = T.cast(X[b_idx, l_idx, h, d + dim_half], compute_dtype)

                    o1 = x1 * cos_lo[d] - x2 * sin_lo[d]
                    o2 = x2 * cos_hi[d] + x1 * sin_hi[d]

                    X_out[b_idx, l_idx, h, d] = T.cast(o1, cos_sin_dtype)
                    X_out[b_idx, l_idx, h, d + dim_half] = T.cast(o2, cos_sin_dtype)

    return main


def _single_tensor_rope(x, x_cos, x_sin):
    B, L, H, D = x.shape
    if x_cos.ndim == 3:
        assert x_cos.shape[0] == 1 and x_sin.shape[0] == 1, \
            f"Expected batch dim of x_cos/x_sin to be 1 when ndim==3 (shared across batch), but got x_cos.shape={tuple(x_cos.shape)}, x_sin.shape={tuple(x_sin.shape)}"
        x_cos = x_cos[0]
        x_sin = x_sin[0]
    x_cos = x_cos.contiguous()
    x_sin = x_sin.contiguous()
    if x_cos.dtype == torch.float32:
        cos_sin_dtype = "float32"
    else:
        cos_sin_dtype = "bfloat16"
    kernel_func = single_tensor_rope_kernel(B, L, H, D, cos_sin_dtype)
    return kernel_func(x, x_cos, x_sin)


class SingleTensorRoPEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_cos, x_sin):
        x_dtype = x.dtype
        x_out = _single_tensor_rope(x, x_cos, x_sin)
        ctx.save_for_backward(x_cos, x_sin)
        ctx.x_dtype = x_dtype
        return x_out.to(x_dtype)

    @staticmethod
    def backward(ctx, dx):
        x_cos, x_sin = ctx.saved_tensors
        dx_out = _single_tensor_rope(dx.contiguous(), x_cos, -x_sin)
        return dx_out.to(ctx.x_dtype), None, None


def single_tensor_rope_autograd(x, x_cos, x_sin):
    """
    Args:
        x:     (B, L, H, D)  - BF16
        x_cos: (B, L, D) - BF16 or FP32
        x_sin: (B, L, D) - BF16 or FP32
    Returns:
        x_out: (B, L, H, D) - same dtype as x (BF16)
    """
    return SingleTensorRoPEFunction.apply(x, x_cos, x_sin)


def rotate_half_torch(x):
    """Rotates half the hidden dims of the input. testof PyTorch ."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def test_hsa_rope_correctness():
    """
    validate hsa_intra_chunk_rope kernel(training, L_q == L_k)
    and PyTorch numerical consistency.
     lhsa_layer.py in apply_hsa_rope ofpath.
    """
    torch.manual_seed(42)
    device = "cuda"

    B = 2
    L = 512*16
    H_q = 16
    H_k = 4
    D = 64          # head dim
    chunk_size = 64
    hsa_sliding_window = 0
    max_seq_len = 2 * chunk_size + hsa_sliding_window + 256

    print(f"Testing HSA RoPE (merged kernel): B={B}, L={L}, H_q={H_q}, H_k={H_k}, D={D}")
    print(f"  chunk_size={chunk_size}, hsa_sliding_window={hsa_sliding_window}")

    hsa_q_norm = torch.randn((B, L, H_q, D), dtype=torch.bfloat16, device=device)
    hsa_k_norm = torch.randn((B, L, H_k, D), dtype=torch.bfloat16, device=device)

    freqs = torch.randn((1, max_seq_len, D // 2), dtype=torch.float32, device=device)
    emb = torch.cat((freqs, freqs), dim=-1)  # (1, max_seq_len, D)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)

    # ================================================================
    # ================================================================
    # q positions: periodic within chunk (chunk_size positions, tiled to L)
    q_cos_chunk_ref = cos[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :]  # (B, chunk_size, D)
    q_sin_chunk_ref = sin[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :]
    n_repeats = (L + chunk_size - 1) // chunk_size
    q_cos_ref = q_cos_chunk_ref.repeat(1, n_repeats, 1)[:, :L, :]  # (B, L, D)
    q_sin_ref = q_sin_chunk_ref.repeat(1, n_repeats, 1)[:, :L, :]

    # k positions: periodic with chunk_size
    k_positions = torch.arange(L, device=device) % chunk_size
    k_cos_ref = cos[:, k_positions, :]  # (B, L, D)
    k_sin_ref = sin[:, k_positions, :]

    # Apply separately since q and k use different cos/sin
    hsa_q_rot = hsa_q_norm.transpose(1, 2).float()  # (B, h, L, d)
    hsa_k_rot = hsa_k_norm.transpose(1, 2).float()  # (B, h, L, d)
    q_cos_u = q_cos_ref.float().unsqueeze(1)  # (B, 1, L, D)
    q_sin_u = q_sin_ref.float().unsqueeze(1)
    k_cos_u = k_cos_ref.float().unsqueeze(1)  # (B, 1, L, D)
    k_sin_u = k_sin_ref.float().unsqueeze(1)

    q_out_ref = ((hsa_q_rot * q_cos_u) + (rotate_half_torch(hsa_q_rot) * q_sin_u)).transpose(1, 2).contiguous().to(torch.bfloat16)
    k_out_ref = ((hsa_k_rot * k_cos_u) + (rotate_half_torch(hsa_k_rot) * k_sin_u)).transpose(1, 2).contiguous().to(torch.bfloat16)

    # ================================================================
    # ================================================================
    q_cos_chunk = cos[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :].contiguous()  # (B, chunk_size, D)
    q_sin_chunk = sin[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :].contiguous()
    k_cos_chunk = cos[:, :chunk_size, :].contiguous()  # (B, chunk_size, D)
    k_sin_chunk = sin[:, :chunk_size, :].contiguous()

    q_out_kernel, k_out_kernel = hsa_intra_chunk_rope(
        hsa_q_norm, hsa_k_norm,
        q_cos_chunk, q_sin_chunk,
        k_cos_chunk, k_sin_chunk,
    )

    # ================================================================
    # ================================================================
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

    def assert_close(prefix, ref, tri, ratio=0.005):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    # ================================================================
    # ================================================================
    assert_close("HSA RoPE Q", q_out_ref.float(), q_out_kernel.float())
    assert_close("HSA RoPE K", k_out_ref.float(), k_out_kernel.float())

    print(" HSA Intra-Chunk RoPE kernel correctness validation!")

    # ================================================================
    # ================================================================
    print("\n--- test Fixed Q Position mode (chunk_size_q=1) ---")
    q_cos_fixed = cos[:, chunk_size + hsa_sliding_window : chunk_size + hsa_sliding_window + 1, :].contiguous()  # (B, 1, D)
    q_sin_fixed = sin[:, chunk_size + hsa_sliding_window : chunk_size + hsa_sliding_window + 1, :].contiguous()

    q_cos_fixed_ref = cos[:, chunk_size + hsa_sliding_window, :].unsqueeze(1).expand(-1, L, -1).float()  # (B, L, D)
    q_sin_fixed_ref = sin[:, chunk_size + hsa_sliding_window, :].unsqueeze(1).expand(-1, L, -1).float()

    hsa_q_rot2 = hsa_q_norm.transpose(1, 2).float()
    q_cos_u2 = q_cos_fixed_ref.unsqueeze(1)
    q_sin_u2 = q_sin_fixed_ref.unsqueeze(1)
    q_out_fixed_ref = ((hsa_q_rot2 * q_cos_u2) + (rotate_half_torch(hsa_q_rot2) * q_sin_u2)).transpose(1, 2).contiguous().to(torch.bfloat16)

    q_out_fixed_kernel, _ = hsa_intra_chunk_rope(
        hsa_q_norm, hsa_k_norm,
        q_cos_fixed, q_sin_fixed,
        k_cos_chunk, k_sin_chunk,
    )

    assert_close("HSA RoPE Q (fixed pos)", q_out_fixed_ref.float(), q_out_fixed_kernel.float())
    print(" Fixed Q Position mode validation!")

    # ================================================================
    # ================================================================
    print("\n--- testBWDconsistency ---")

    hsa_q_grad = torch.randn((B, L, H_q, D), dtype=torch.float32, device=device, requires_grad=True)
    hsa_k_grad = torch.randn((B, L, H_k, D), dtype=torch.float32, device=device, requires_grad=True)

    grad_q_out = torch.randn((B, L, H_q, D), dtype=torch.float32, device=device)
    grad_k_out = torch.randn((B, L, H_k, D), dtype=torch.float32, device=device)

    q_rot_ref = hsa_q_grad.transpose(1, 2)  # (B, h, L, d)
    k_rot_ref = hsa_k_grad.transpose(1, 2)
    q_cos_u_ref = q_cos_ref.float().unsqueeze(1)  # (B, 1, L, D)
    q_sin_u_ref = q_sin_ref.float().unsqueeze(1)
    k_cos_u_ref = k_cos_ref.float().unsqueeze(1)
    k_sin_u_ref = k_sin_ref.float().unsqueeze(1)

    q_out_pt = ((q_rot_ref * q_cos_u_ref) + (rotate_half_torch(q_rot_ref) * q_sin_u_ref)).transpose(1, 2).contiguous()
    k_out_pt = ((k_rot_ref * k_cos_u_ref) + (rotate_half_torch(k_rot_ref) * k_sin_u_ref)).transpose(1, 2).contiguous()

    q_out_pt.backward(grad_q_out, retain_graph=True)
    k_out_pt.backward(grad_k_out)

    dq_ref = hsa_q_grad.grad.clone()
    dk_ref = hsa_k_grad.grad.clone()

    hsa_q_kernel = hsa_q_grad.detach().to(torch.bfloat16).requires_grad_(True)
    hsa_k_kernel = hsa_k_grad.detach().to(torch.bfloat16).requires_grad_(True)

    q_out_ag, k_out_ag = hsa_intra_chunk_rope_autograd(
        hsa_q_kernel, hsa_k_kernel,
        q_cos_chunk, q_sin_chunk,
        k_cos_chunk, k_sin_chunk,
    )

    q_out_ag.backward(grad_q_out.to(torch.bfloat16), retain_graph=True)
    k_out_ag.backward(grad_k_out.to(torch.bfloat16))

    dq_kernel = hsa_q_kernel.grad.clone()
    dk_kernel = hsa_k_kernel.grad.clone()

    assert_close("HSA RoPE BWD dQ", dq_ref.float(), dq_kernel.float(), ratio=0.01)
    assert_close("HSA RoPE BWD dK", dk_ref.float(), dk_kernel.float(), ratio=0.01)

    print(" HSA Intra-Chunk RoPE BWDvalidate!")


def test_single_tensor_rope_correctness():
    """
    Validate numerical consistency between single_tensor_rope kernel (generic version)
    and the PyTorch reference implementation.
    cos/sin shape is (B, L, D), matching x's sequence length exactly.
    Also tests backward gradient consistency.
    """
    torch.manual_seed(42)
    device = "cuda"

    # ---- Config ----
    B = 1
    L = 512 * 16       # sequence length
    H = 8              # number of heads (generic, not tied to a specific value)
    D = 64             # head dim

    print(f"\nTesting Single Tensor RoPE: B={B}, L={L}, H={H}, D={D}")

    # ---- Build inputs ----
    x = torch.randn((B, L, H, D), dtype=torch.bfloat16, device=device)

    # cos/sin: (1, L, D), shared across all batches (matching real Qwen3RotaryEmbedding behavior)
    freqs = torch.randn((1, L, D // 2), dtype=torch.float32, device=device)
    emb = torch.cat((freqs, freqs), dim=-1)  # (1, L, D)
    cos = emb.cos().to(torch.bfloat16)
    sin = emb.sin().to(torch.bfloat16)

    # ================================================================
    # ================================================================
    def get_abs_err(a, b):
        mask = (a > -1e5) & (b > -1e5)
        if mask.sum() == 0: return 0.0
        return (a[mask] - b[mask]).abs().max().item()

    def get_err_ratio(a, b):
        mask = (a > -1e5) & (b > -1e5)
        if mask.sum() == 0: return 0.0
        err = (a[mask] - b[mask]).square().mean().sqrt().item()
        base = (a[mask]).square().mean().sqrt().item()
        return err / (base + 1e-12)

    def assert_close(prefix, ref, tri, ratio=0.005):
        abs_err = get_abs_err(ref, tri)
        rel_ratio = get_err_ratio(ref, tri)
        msg = f"{prefix} diff: {abs_err:.6f} ratio: {rel_ratio:.6f}"
        print(msg)
        assert rel_ratio < ratio, msg

    # ================================================================
    # Test 1: standard RoPE forward
    # ================================================================
    print("\n--- Test: standard RoPE forward ---")

    # PyTorch reference: x * cos + rotate_half(x) * sin
    x_rot = x.transpose(1, 2).float()  # (B, H, L, D)
    cos_ref = cos.float().unsqueeze(1)  # (B, 1, L, D)
    sin_ref = sin.float().unsqueeze(1)
    x_out_ref = ((x_rot * cos_ref) + (rotate_half_torch(x_rot) * sin_ref)).transpose(1, 2).contiguous().to(torch.bfloat16)

    # Kernel implementation
    x_out_kernel = single_tensor_rope_autograd(x, cos, sin)

    assert_close("Single RoPE FWD", x_out_ref.float(), x_out_kernel.float())
    print(" Standard RoPE forward passed!")

    # ================================================================
    # Test 2: generality across head_dim (D=128, H=1)
    # ================================================================
    print("\n--- Test: different head_dim (D=128, H=1) ---")
    D2 = 128
    H2 = 1
    x2 = torch.randn((B, L, H2, D2), dtype=torch.bfloat16, device=device)
    freqs2 = torch.randn((1, L, D2 // 2), dtype=torch.float32, device=device)
    emb2 = torch.cat((freqs2, freqs2), dim=-1)  # (1, L, D2)
    cos2 = emb2.cos().to(torch.bfloat16)
    sin2 = emb2.sin().to(torch.bfloat16)

    x2_rot = x2.transpose(1, 2).float()
    x2_out_ref = ((x2_rot * cos2.float().unsqueeze(1)) + (rotate_half_torch(x2_rot) * sin2.float().unsqueeze(1))).transpose(1, 2).contiguous().to(torch.bfloat16)

    x2_out_kernel = single_tensor_rope_autograd(x2, cos2, sin2)

    assert_close("Single RoPE (D=128, H=1)", x2_out_ref.float(), x2_out_kernel.float())
    print(" Different head_dim generality passed!")

    # ================================================================
    # Test 3: generality across number of heads (H=32)
    # ================================================================
    print("\n--- Test: different number of heads (H=32, D=64) ---")
    H3 = 32
    x3 = torch.randn((B, L, H3, D), dtype=torch.bfloat16, device=device)

    x3_rot = x3.transpose(1, 2).float()
    x3_out_ref = ((x3_rot * cos.float().unsqueeze(1)) + (rotate_half_torch(x3_rot) * sin.float().unsqueeze(1))).transpose(1, 2).contiguous().to(torch.bfloat16)

    x3_out_kernel = single_tensor_rope_autograd(x3, cos, sin)

    assert_close("Single RoPE (H=32)", x3_out_ref.float(), x3_out_kernel.float())
    print(" Different number of heads generality passed!")

    # ================================================================
    # Test 4: backward gradient consistency
    # ================================================================
    print("\n--- Test: backward gradient consistency ---")

    x_grad = torch.randn((B, L, H, D), dtype=torch.float32, device=device, requires_grad=True)
    grad_out = torch.randn((B, L, H, D), dtype=torch.float32, device=device)

    # PyTorch reference
    x_rot_ref = x_grad.transpose(1, 2)
    x_out_pt = ((x_rot_ref * cos_ref) + (rotate_half_torch(x_rot_ref) * sin_ref)).transpose(1, 2).contiguous()
    x_out_pt.backward(grad_out)
    dx_ref = x_grad.grad.clone()

    # Kernel autograd implementation
    x_kernel = x_grad.detach().to(torch.bfloat16).requires_grad_(True)
    x_out_ag = single_tensor_rope_autograd(x_kernel, cos, sin)
    x_out_ag.backward(grad_out.to(torch.bfloat16))
    dx_kernel = x_kernel.grad.clone()

    assert_close("Single RoPE BWD dX", dx_ref.float(), dx_kernel.float(), ratio=0.01)
    print(" Single Tensor RoPE backward gradient passed!")

    print("\n All Single Tensor RoPE kernel correctness tests passed!")


def test_single_tensor_rope_performance():
    """
    Single Tensor RoPE test.
    compare PyTorch path vs TileLang Kernel latency.
    """
    print("\n" + "=" * 60)
    print(" Single Tensor RoPE Latency Benchmark")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    B = 8
    L = 4096
    H = 8
    D = 128

    print(f"Config: B={B}, L={L}, H={H}, D={D}")

    x = torch.randn((B, L, H, D), dtype=dtype, device=device)
    freqs = torch.randn((1, L, D // 2), dtype=torch.float32, device=device)
    emb = torch.cat((freqs, freqs), dim=-1)  # (1, L, D)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    _ = _single_tensor_rope(x, cos, sin)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def pytorch_baseline():
        x_rot = x.transpose(1, 2)
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        return ((x_rot * cos_u) + (rotate_half_torch(x_rot) * sin_u)).transpose(1, 2).contiguous()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def tilelang_kernel_only():
        return _single_tensor_rope(x, cos, sin)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def benchmark(func, name, num_iters=100, num_warmup=20):
        for _ in range(num_warmup):
            func()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / num_iters
        print(f"  [{name}] Average Latency: {avg_time:.3f} ms")
        return avg_time

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    print("-" * 50)
    t_baseline = benchmark(pytorch_baseline, "PyTorch path")
    t_kernel = benchmark(tilelang_kernel_only, "TileLang  kernel")

    print("-" * 50)
    print(f"  : {t_baseline / t_kernel:.2f}x")
    print("=" * 60)


import torch.nn.functional as F
from liger_kernel.transformers.rope import liger_rotary_pos_emb

def test_rope_correctness():
    torch.manual_seed(42)
    device = "cuda"

    # Config
    B, L, H, D = 2, 128, 4, 64
    test_inverse = True

    # Inputs (BF16)
    q = torch.randn((B, L, H, D), dtype=torch.bfloat16, device=device)
    k = torch.randn((B, L, H, D), dtype=torch.bfloat16, device=device)

    # Random Cos/Sin for testing
    cos = torch.randn((B, L, D), dtype=torch.bfloat16, device=device)
    sin = torch.randn((B, L, D), dtype=torch.bfloat16, device=device)

    print(f"Testing Shape: Q{q.shape}, Cos{cos.shape}, Test Inverse={test_inverse}")

    # -------------------------------------------------------
    # 1. Baseline: Liger Kernel (High Precision Reference)
    # -------------------------------------------------------
    q_ref_in = q.transpose(1, 2).contiguous().float()
    k_ref_in = k.transpose(1, 2).contiguous().float()
    cos_ref = cos.float()
    sin_ref = sin.float()

    sin_input_liger = -sin_ref if test_inverse else sin_ref

    q_out_liger, k_out_liger = liger_rotary_pos_emb(q_ref_in, k_ref_in, cos_ref, sin_input_liger)

    q_out_ref = q_out_liger.transpose(1, 2).contiguous().to(torch.bfloat16)
    k_out_ref = k_out_liger.transpose(1, 2).contiguous().to(torch.bfloat16)

    # -------------------------------------------------------
    # 2. TileLang Kernel (Using Wrapper)
    # -------------------------------------------------------
    sin_input_tl = -sin if test_inverse else sin

    q_out_tl, k_out_tl = rope_rotary_pos_emb(q, k, cos, sin_input_tl)

    # -------------------------------------------------------
    # 3. Validation
    # -------------------------------------------------------
    diff_q = (q_out_ref.float() - q_out_tl.float()).abs()
    max_diff_q = diff_q.max().item()

    diff_k = (k_out_ref.float() - k_out_tl.float()).abs()
    max_diff_k = diff_k.max().item()

    print(f"Max Diff Q: {max_diff_q:.6f}")
    print(f"Max Diff K: {max_diff_k:.6f}")

    if max_diff_q < 1e-2:
        print(" Test Passed! TileLang kernel matches Liger (FP32) behavior.")
    else:
        print(" Test Failed! Large discrepancy detected.")




def test_rope_performance():
    import time

    print("\n" + "=" * 60)
    print(" RoPE Kernel Raw Latency Benchmark (Kernel Only)")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"

    # B=Batch, L=SeqLen, H=Heads, D=HeadDim
    B, L, HQ, H, D = 128, 4096, 32, 4, 128
    dtype = torch.bfloat16

    print(f"Config: Batch={B}, SeqLen={L}, Heads={H}, Dim={D}")

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    q_tl = torch.randn((B, L, HQ, D), dtype=dtype, device=device)
    k_tl = torch.randn((B, L, H, D), dtype=dtype, device=device)
    cos_tl = torch.randn((B, L, D), dtype=dtype, device=device)
    sin_tl = torch.randn((B, L, D), dtype=dtype, device=device)

    q_liger = q_tl.transpose(1, 2).contiguous()
    k_liger = k_tl.transpose(1, 2).contiguous()
    cos_liger = cos_tl
    sin_liger = sin_tl

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    def baseline_kernel_only():
        liger_rotary_pos_emb(q_liger, k_liger, cos_liger, sin_liger)

    def tilelang_kernel_only():
        rope_rotary_pos_emb(q_tl, k_tl, cos_tl, sin_tl)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def benchmark(func, name, num_iters=100, num_warmup=20):
        # Warmup
        for _ in range(num_warmup):
            func()
        torch.cuda.synchronize()

        # Timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / num_iters
        print(f"[{name}] Average Latency: {avg_time:.3f} ms")
        return avg_time

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------

    print("-" * 30)
    time_liger = benchmark(baseline_kernel_only, "Liger Kernel")
    time_tilelang = benchmark(tilelang_kernel_only, "TileLang Kernel")


def test_hsa_rope_performance():
    """
    HSA Intra-Chunk RoPE test.
    Compare latency for:
      1. PyTorch path (cos/sin + apply)
      2. TileLang kernel path (slice + kernel)
    """
    print("\n" + "=" * 60)
    print(" HSA Intra-Chunk RoPE Latency Benchmark")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    B = 8
    L = 4096
    H_q = 32
    H_k = 4
    D = 128            # head dim
    chunk_size = 64
    hsa_sliding_window = 128
    max_seq_len = 2 * chunk_size + hsa_sliding_window + 256

    print(f"Config: B={B}, L={L}, H_q={H_q}, H_k={H_k}, D={D}")
    print(f"  chunk_size={chunk_size}, hsa_sliding_window={hsa_sliding_window}")

    hsa_q_norm = torch.randn((B, L, H_q, D), dtype=dtype, device=device)
    hsa_k_norm = torch.randn((B, L, H_k, D), dtype=dtype, device=device)
    freqs = torch.randn((1, max_seq_len, D // 2), dtype=torch.float32, device=device)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    q_cos_chunk = cos[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :].contiguous()
    q_sin_chunk = sin[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :].contiguous()
    k_cos_chunk = cos[:, :chunk_size, :].contiguous()
    k_sin_chunk = sin[:, :chunk_size, :].contiguous()
    _ = hsa_intra_chunk_rope(hsa_q_norm, hsa_k_norm, q_cos_chunk, q_sin_chunk, k_cos_chunk, k_sin_chunk)
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def pytorch_baseline():
        L_full = hsa_k_norm.shape[1]

        q_cos_chunk_ref = cos[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :]
        q_sin_chunk_ref = sin[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :]
        n_repeats = (L + chunk_size - 1) // chunk_size
        q_cos = q_cos_chunk_ref.repeat(1, n_repeats, 1)[:, :L, :]
        q_sin = q_sin_chunk_ref.repeat(1, n_repeats, 1)[:, :L, :]

        k_positions = torch.arange(L_full, device=hsa_k_norm.device) % chunk_size
        k_cos = cos[:, k_positions, :]
        k_sin = sin[:, k_positions, :]

        # Apply RoPE
        hsa_q_rot = hsa_q_norm.transpose(1, 2)
        hsa_k_rot = hsa_k_norm.transpose(1, 2)
        q_cos_u = q_cos.unsqueeze(1)
        q_sin_u = q_sin.unsqueeze(1)
        k_cos_u = k_cos.unsqueeze(1)
        k_sin_u = k_sin.unsqueeze(1)
        q_out = ((hsa_q_rot * q_cos_u) + (rotate_half_torch(hsa_q_rot) * q_sin_u)).transpose(1, 2).contiguous()
        k_out = ((hsa_k_rot * k_cos_u) + (rotate_half_torch(hsa_k_rot) * k_sin_u)).transpose(1, 2).contiguous()
        return q_out, k_out

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def tilelang_kernel():
        qc = cos[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :].contiguous()
        qs = sin[:, chunk_size + hsa_sliding_window : 2 * chunk_size + hsa_sliding_window, :].contiguous()
        kc = cos[:, :chunk_size, :].contiguous()
        ks = sin[:, :chunk_size, :].contiguous()
        # Kernel apply
        return hsa_intra_chunk_rope(hsa_q_norm, hsa_k_norm, qc, qs, kc, ks)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def tilelang_kernel_only():
        return hsa_intra_chunk_rope(hsa_q_norm, hsa_k_norm, q_cos_chunk, q_sin_chunk, k_cos_chunk, k_sin_chunk)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def benchmark(func, name, num_iters=100, num_warmup=20):
        for _ in range(num_warmup):
            func()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_iters):
            func()
        end_event.record()
        torch.cuda.synchronize()

        avg_time = start_event.elapsed_time(end_event) / num_iters
        print(f"  [{name}] Average Latency: {avg_time:.3f} ms")
        return avg_time

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    print("-" * 50)
    t_baseline = benchmark(pytorch_baseline, "PyTorch path (+apply)")
    t_kernel_e2e = benchmark(tilelang_kernel, "TileLang to (slice+kernel)")
    t_kernel_only = benchmark(tilelang_kernel_only, "TileLang  kernel")

    print("-" * 50)
    print(f"   ( vs to): {t_baseline / t_kernel_e2e:.2f}x")
    print(f"   ( vs kernel): {t_baseline / t_kernel_only:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    test_single_tensor_rope_correctness()

    test_single_tensor_rope_performance()

    # test_hsa_rope_correctness()

    # test_hsa_rope_performance()

    # test_rope_correctness()

    # test_rope_performance()


# python ops/rope_tilelang_fp32.py
