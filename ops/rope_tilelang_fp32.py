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
    B, L, HQ, D = q.shape
    H = k.shape[2]
    kernel_func = rope_fwd_kernel(B, L, HQ, H, D)
    return kernel_func(q, k, cos, sin)


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


if __name__ == "__main__":
    test_single_tensor_rope_correctness()

    test_single_tensor_rope_performance()
    # test_rope_correctness()

    # test_rope_performance()


# python ops/rope_tilelang_fp32.py
