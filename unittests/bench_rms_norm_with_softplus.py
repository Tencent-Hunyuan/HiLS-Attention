"""
Benchmark: Fused RMSNorm+Softplus (TileLang) vs flash_attn rms_norm_fn (Triton).

Usage:
    python ops/bench_rms_norm_with_softplus.py

Compares:
1. flash_attn rms_norm_fn (Triton kernel, no softplus)
2. flash_attn rms_norm_fn + torch softplus (two-kernel baseline)
3. Our fused RMSNorm+Softplus TileLang kernel
4. PyTorch reference (pure torch, no custom kernel)
"""

import time
import torch
import torch.nn.functional as F

from ops.rms_norm_with_softplus import rms_norm_with_softplus, rms_norm_with_softplus_ref

try:
    from flash_attn.ops.triton.layer_norm import rms_norm_fn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("WARNING: flash_attn not available, skipping flash_attn benchmarks")


def benchmark_fn(fn, *args, warmup=20, repeat=100, label=""):
    """Benchmark a function with CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Timed runs
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn(*args)
        end_events[i].record()

    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Remove top/bottom 10% outliers
    trim = repeat // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    avg_ms = sum(trimmed) / len(trimmed)
    min_ms = min(times)
    max_ms = max(times)
    return avg_ms, min_ms, max_ms


def flash_attn_rms_norm_only(x, weight):
    """flash_attn rms_norm (no softplus)."""
    return rms_norm_fn(x, weight, None, eps=1e-6)


def flash_attn_rms_norm_plus_softplus(x, weight):
    """flash_attn rms_norm + separate softplus."""
    normed = rms_norm_fn(x, weight, None, eps=1e-6)
    return F.softplus(normed)


def fused_tilelang(x, weight):
    """Our fused TileLang kernel."""
    return rms_norm_with_softplus(x, weight)


def pytorch_ref(x, weight):
    """Pure PyTorch reference."""
    return rms_norm_with_softplus_ref(x, weight)


def run_benchmark_suite():
    print("\n" + "=" * 80)
    print("  RMSNorm + Softplus Benchmark")
    print("  Comparing: flash_attn (Triton) vs TileLang Fused vs PyTorch Reference")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16

    # Test configurations: (batch*seq, hidden_dim, description)
    configs = [
        # Typical LLM scenarios
        (2 * 16 * 256, 64, "B=2, H=16, L=256, D=64 (typical attn)"),
        (2 * 16 * 1024, 64, "B=2, H=16, L=1024, D=64 (long seq)"),
        (2 * 16 * 4096, 64, "B=2, H=16, L=4096, D=64 (very long)"),
        (2 * 4 * 256, 128, "B=2, H=4, L=256, D=128 (GQA kv)"),
        (2 * 4 * 1024, 128, "B=2, H=4, L=1024, D=128 (GQA long)"),
        (2 * 4 * 4096, 128, "B=2, H=4, L=4096, D=128 (GQA very long)"),
        # Large hidden dim (MLP-like)
        (2 * 1024, 1024, "B=2, L=1024, D=1024 (MLP-like)"),
        (2 * 4096, 1024, "B=2, L=4096, D=1024 (MLP large)"),
    ]

    print(f"\n{'Config':<45} | {'flash_attn RMS':<14} | {'flash+softplus':<14} | {'TileLang Fused':<14} | {'PyTorch Ref':<14} | {'Speedup vs flash+sp'}")
    print("-" * 130)

    for num_rows, dim, desc in configs:
        torch.manual_seed(42)
        x = torch.randn(num_rows, dim, dtype=dtype, device=device)

        # Learnable RMSNorm gain.  Use a non-trivial weight so the kernel
        # actually exercises the per-channel scaling path (not all ones).
        weight = torch.randn(dim, dtype=dtype, device=device) * 0.1 + 1.0

        results = {}

        # 1. flash_attn rms_norm only (baseline, no softplus)
        if HAS_FLASH_ATTN:
            avg, _, _ = benchmark_fn(flash_attn_rms_norm_only, x, weight)
            results["flash_rms"] = avg

        # 2. flash_attn rms_norm + softplus (two kernels)
        if HAS_FLASH_ATTN:
            avg, _, _ = benchmark_fn(flash_attn_rms_norm_plus_softplus, x, weight)
            results["flash_rms_sp"] = avg

        # 3. Our fused TileLang kernel
        avg, _, _ = benchmark_fn(fused_tilelang, x, weight)
        results["tilelang"] = avg

        # 4. PyTorch reference
        avg, _, _ = benchmark_fn(pytorch_ref, x, weight)
        results["pytorch"] = avg

        # Format output
        flash_rms_str = f"{results.get('flash_rms', 0):.3f} ms" if HAS_FLASH_ATTN else "N/A"
        flash_sp_str = f"{results.get('flash_rms_sp', 0):.3f} ms" if HAS_FLASH_ATTN else "N/A"
        tilelang_str = f"{results['tilelang']:.3f} ms"
        pytorch_str = f"{results['pytorch']:.3f} ms"

        if HAS_FLASH_ATTN and results.get("flash_rms_sp", 0) > 0:
            speedup = results["flash_rms_sp"] / results["tilelang"]
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        print(f"{desc:<45} | {flash_rms_str:<14} | {flash_sp_str:<14} | {tilelang_str:<14} | {pytorch_str:<14} | {speedup_str}")

    print()


def run_backward_benchmark():
    """Benchmark backward pass."""
    print("\n" + "=" * 80)
    print("  Backward Pass Benchmark")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16

    configs = [
        (2 * 16 * 256, 64, "B=2, H=16, L=256, D=64"),
        (2 * 16 * 1024, 64, "B=2, H=16, L=1024, D=64"),
        (2 * 4 * 1024, 128, "B=2, H=4, L=1024, D=128"),
        (2 * 4 * 4096, 128, "B=2, H=4, L=4096, D=128"),
    ]

    print(f"\n{'Config':<45} | {'TileLang Fwd+Bwd':<18} | {'PyTorch Fwd+Bwd':<18} | {'Speedup'}")
    print("-" * 100)

    for num_rows, dim, desc in configs:
        torch.manual_seed(42)

        # TileLang fwd+bwd  (weight captured by closure; requires_grad=True
        # so grad_weight is also exercised, matching real training cost)
        def tilelang_fwd_bwd(num_rows=num_rows, dim=dim):
            x = torch.randn(num_rows, dim, dtype=dtype, device=device, requires_grad=True)
            weight = torch.randn(dim, dtype=dtype, device=device, requires_grad=True) * 0.1 + 1.0
            y = rms_norm_with_softplus(x, weight)
            y.sum().backward()

        # PyTorch fwd+bwd
        def pytorch_fwd_bwd(num_rows=num_rows, dim=dim):
            x = torch.randn(num_rows, dim, dtype=dtype, device=device, requires_grad=True)
            weight = torch.randn(dim, dtype=dtype, device=device, requires_grad=True) * 0.1 + 1.0
            y = rms_norm_with_softplus_ref(x, weight)
            y.sum().backward()

        avg_tl, _, _ = benchmark_fn(tilelang_fwd_bwd, warmup=10, repeat=50)
        avg_pt, _, _ = benchmark_fn(pytorch_fwd_bwd, warmup=10, repeat=50)

        speedup = avg_pt / avg_tl if avg_tl > 0 else 0
        print(f"{desc:<45} | {avg_tl:.3f} ms{'':<8} | {avg_pt:.3f} ms{'':<8} | {speedup:.2f}x")

    print()


def run_throughput_benchmark():
    """Measure throughput in GB/s."""
    print("\n" + "=" * 80)
    print("  Memory Throughput Analysis")
    print("=" * 80)

    device = "cuda"
    dtype = torch.bfloat16
    bytes_per_elem = 2  # bf16

    configs = [
        (2 * 16 * 4096, 64, "B=2, H=16, L=4096, D=64"),
        (2 * 4 * 4096, 128, "B=2, H=4, L=4096, D=128"),
        (2 * 4096, 1024, "B=2, L=4096, D=1024"),
    ]

    print(f"\n{'Config':<45} | {'Data Size':<12} | {'TileLang BW':<14} | {'flash+sp BW':<14} | {'Peak BW %'}")
    print("-" * 110)

    for num_rows, dim, desc in configs:
        torch.manual_seed(42)
        x = torch.randn(num_rows, dim, dtype=dtype, device=device)
        weight = torch.randn(dim, dtype=dtype, device=device) * 0.1 + 1.0

        # Data moved: read input + write output = 2 * num_rows * dim * bytes_per_elem
        data_bytes = 2 * num_rows * dim * bytes_per_elem
        data_mb = data_bytes / (1024 * 1024)

        # TileLang
        avg_tl, _, _ = benchmark_fn(fused_tilelang, x, weight, warmup=20, repeat=100)
        bw_tl = data_bytes / (avg_tl * 1e-3) / 1e9  # GB/s

        # flash_attn + softplus
        if HAS_FLASH_ATTN:
            avg_fa, _, _ = benchmark_fn(flash_attn_rms_norm_plus_softplus, x, weight, warmup=20, repeat=100)
            bw_fa = data_bytes / (avg_fa * 1e-3) / 1e9
            fa_str = f"{bw_fa:.1f} GB/s"
        else:
            fa_str = "N/A"

        # Assume ~2TB/s peak for A100/H100
        peak_bw = 2000  # GB/s (adjust for your GPU)
        peak_pct = bw_tl / peak_bw * 100

        print(f"{desc:<45} | {data_mb:.1f} MB{'':<4} | {bw_tl:.1f} GB/s{'':<4} | {fa_str:<14} | {peak_pct:.1f}%")

    print()


if __name__ == "__main__":
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    if HAS_FLASH_ATTN:
        import flash_attn
        print(f"flash_attn: {flash_attn.__version__}")

    run_benchmark_suite()
    run_backward_benchmark()
    run_throughput_benchmark()

    print("=" * 80)
    print("  Benchmark Complete!")
    print("=" * 80)
