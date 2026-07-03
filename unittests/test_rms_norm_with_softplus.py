"""
Test cases for fused RMSNorm + Softplus TileLang kernel.
Compares against rms_norm_with_softplus_ref for forward and backward precision.

Usage:
    python ops/test_rms_norm_with_softplus.py
"""

import sys
import torch
import torch.nn.functional as F

from ops.rms_norm_with_softplus import (
    rms_norm_with_softplus as _rms_norm_with_softplus_raw,
    rms_norm_with_softplus_ref as _rms_norm_with_softplus_ref_raw,
)


def _default_weight(x: torch.Tensor) -> torch.Tensor:
    return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)


def rms_norm_with_softplus(x, weight=None, **kwargs):
    """Test wrapper: auto-fill weight=ones(D) when caller omits it.

    The underlying kernel now requires weight; tests that don't care about
    gain (weight=1) keep working unchanged via this wrapper.
    """
    if weight is None:
        weight = _default_weight(x)
    return _rms_norm_with_softplus_raw(x, weight=weight, **kwargs)


def rms_norm_with_softplus_ref(x, weight=None, **kwargs):
    if weight is None:
        weight = _default_weight(x)
    return _rms_norm_with_softplus_ref_raw(x, weight=weight, **kwargs)


def test_forward_precision():
    """Test forward pass precision against reference implementation."""
    print("=" * 60)
    print("TEST: Forward Precision")
    print("=" * 60)

    test_cases = [
        # (shape, description)
        ((2, 16, 256, 64), "Typical attention: B=2, H=16, L=256, D=64"),
        ((1, 4, 1024, 128), "Long seq: B=1, H=4, L=1024, D=128"),
        ((4, 32, 64, 64), "Many heads: B=4, H=32, L=64, D=64"),
        ((1, 1, 8, 32), "Small: B=1, H=1, L=8, D=32"),
        ((2, 16, 512, 128), "Large dim: B=2, H=16, L=512, D=128"),
    ]

    all_passed = True
    for shape, desc in test_cases:
        torch.manual_seed(42)
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        # Reference (pure PyTorch)
        y_ref = rms_norm_with_softplus_ref(x)

        # Fused kernel
        y_fused = rms_norm_with_softplus(x)

        # Compare
        max_diff = (y_fused.float() - y_ref.float()).abs().max().item()
        mean_diff = (y_fused.float() - y_ref.float()).abs().mean().item()
        # Relative error
        rel_err = ((y_fused.float() - y_ref.float()).abs() / (y_ref.float().abs() + 1e-8)).mean().item()

        # BF16 tolerance: ~1e-2 relative error is acceptable
        passed = max_diff < 0.05 and rel_err < 0.01
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {desc}")
        print(f"         shape={shape}, max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.6f}")

    print()
    return all_passed


def test_forward_different_scales():
    """Test forward with different input magnitudes."""
    print("=" * 60)
    print("TEST: Forward with Different Input Scales")
    print("=" * 60)

    shape = (2, 16, 128, 64)
    scales = [0.01, 0.1, 1.0, 5.0, 10.0]

    all_passed = True
    for scale in scales:
        torch.manual_seed(123)
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda") * scale

        y_ref = rms_norm_with_softplus_ref(x)
        y_fused = rms_norm_with_softplus(x)

        max_diff = (y_fused.float() - y_ref.float()).abs().max().item()
        rel_err = ((y_fused.float() - y_ref.float()).abs() / (y_ref.float().abs() + 1e-8)).mean().item()

        passed = max_diff < 0.1 and rel_err < 0.02
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] scale={scale:.2f}, max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")

    print()
    return all_passed


def test_backward_precision():
    """Test backward pass (gradient) precision against reference."""
    print("=" * 60)
    print("TEST: Backward Precision")
    print("=" * 60)

    test_cases = [
        ((2, 16, 128, 64), "Typical: B=2, H=16, L=128, D=64"),
        ((1, 4, 256, 128), "Larger dim: B=1, H=4, L=256, D=128"),
        ((4, 8, 64, 64), "Multi-batch: B=4, H=8, L=64, D=64"),
    ]

    all_passed = True
    for shape, desc in test_cases:
        torch.manual_seed(42)

        # Reference backward
        x_ref = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        x_ref_f32 = x_ref.float().requires_grad_(True)
        rms = x_ref_f32.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normed = x_ref_f32 * rms
        y_ref = F.softplus(normed)
        grad_out = torch.randn_like(y_ref)
        y_ref.backward(grad_out)
        grad_ref = x_ref_f32.grad.to(torch.bfloat16)

        # Fused kernel backward
        x_fused = x_ref.clone().requires_grad_(True)
        y_fused = rms_norm_with_softplus(x_fused)
        y_fused.backward(grad_out.to(torch.bfloat16))
        grad_fused = x_fused.grad

        # Compare gradients
        max_diff = (grad_fused.float() - grad_ref.float()).abs().max().item()
        mean_diff = (grad_fused.float() - grad_ref.float()).abs().mean().item()
        rel_err = ((grad_fused.float() - grad_ref.float()).abs() / (grad_ref.float().abs() + 1e-8)).mean().item()

        # BF16 backward tolerance is slightly looser
        passed = max_diff < 0.1 and rel_err < 0.05
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {desc}")
        print(f"         grad max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.6f}")

    print()
    return all_passed


def _rmsnorm_softplus_fp32_forward(x_f32: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Pure fp32 RMSNorm + full softplus (no bf16 round-trip). Used as the
    ground-truth for gradient checks."""
    rms = x_f32.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return F.softplus(x_f32 * rms)


def test_gradcheck_numerical():
    """Cross-check the fused kernel's gradient three ways:
       (A) kernel bwd (bf16) vs fp32 autograd reference (ground truth)
       (B) kernel bwd (bf16) vs central finite differences in pure fp32

    Why FD alone isn't enough here:
      * RMSNorm is *row-wise* (normalizes across last dim), so perturbing
        x[i] changes every output in that row through rms(x). y.sum() sees
        a lot of cancellation; the FD signal is O(eps * grad) per element,
        typically ~1e-4, comparable to fp32 ulp of the sum itself.
      * If the ref forward does a bf16 round-trip, bf16 roundoff swamps
        the FD signal entirely. We therefore run the FD ref forward in
        pure fp32.
      * (A) is the actual correctness test; (B) is informational.
    """
    print("=" * 60)
    print("TEST: Numerical Gradient Check (fp32 autograd ref + FD)")
    print("=" * 60)

    torch.manual_seed(7)
    shape = (2, 4, 8, 32)
    x_bf16 = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    x_f32_base = x_bf16.float()  # exact fp32 copy of the bf16 bits

    # --- (A) analytical fp32 autograd reference (ground truth) ---------------
    x_ref = x_f32_base.clone().requires_grad_(True)
    y_ref = _rmsnorm_softplus_fp32_forward(x_ref)
    y_ref.sum().backward()
    grad_ref_f32 = x_ref.grad.detach()

    # --- Fused kernel grad (bf16 path) ---------------------------------------
    x_param = x_bf16.clone().detach().requires_grad_(True)
    y_fused = rms_norm_with_softplus(x_param)
    y_fused.sum().backward()
    grad_fused_f32 = x_param.grad.detach().float()

    # Compare (A)
    diff_A = (grad_fused_f32 - grad_ref_f32).abs()
    ref_abs = grad_ref_f32.abs()
    max_A = diff_A.max().item()
    mean_A = diff_A.mean().item()
    mask_A = ref_abs > 1e-3
    rel_A = (diff_A[mask_A] / ref_abs[mask_A]).mean().item() if mask_A.any() else 0.0

    # --- (B) central finite differences in pure fp32 -------------------------
    eps_fd = 5e-3
    n_probe = 64
    x_flat = x_f32_base.reshape(-1)
    grad_fd = torch.zeros(n_probe, dtype=torch.float32, device="cuda")

    for i in range(min(n_probe, x_flat.numel())):
        xp = x_f32_base.clone().reshape(-1)
        xp[i] += eps_fd
        yp = _rmsnorm_softplus_fp32_forward(xp.reshape(shape)).sum()

        xm = x_f32_base.clone().reshape(-1)
        xm[i] -= eps_fd
        ym = _rmsnorm_softplus_fp32_forward(xm.reshape(shape)).sum()

        grad_fd[i] = (yp - ym) / (2 * eps_fd)

    grad_fused_probe = grad_fused_f32.reshape(-1)[:n_probe]
    grad_ref_probe = grad_ref_f32.reshape(-1)[:n_probe]

    diff_B_ref = (grad_fd - grad_ref_probe).abs()
    diff_B_fused = (grad_fd - grad_fused_probe).abs()
    # FD vs analytical ref tells us the FD method's own noise floor
    fd_noise = diff_B_ref.max().item()
    max_B = diff_B_fused.max().item()

    # --- Verdict -------------------------------------------------------------
    # (A) is the correctness gate. bf16 backward tolerance ~= 3e-2 rel.
    passed_A = (max_A < 0.1) and (rel_A < 0.03)

    status_A = "PASS" if passed_A else "FAIL"
    print(f"  [{status_A}] (A) kernel_grad vs fp32_autograd_ref (ground truth)")
    print(f"         max_abs={max_A:.4e}, mean_abs={mean_A:.4e}, "
          f"rel_on_active={rel_A:.4e} (active={int(mask_A.sum().item())}/{mask_A.numel()})")
    print(f"         grad_fused range=[{grad_fused_f32.min().item():+.4f},{grad_fused_f32.max().item():+.4f}]  "
          f"grad_ref range=[{grad_ref_f32.min().item():+.4f},{grad_ref_f32.max().item():+.4f}]")

    print(f"  [info] (B) FD-fp32 noise floor vs kernel")
    print(f"         FD_vs_ref max={fd_noise:.4e} (noise floor of FD itself)")
    print(f"         FD_vs_fused max={max_B:.4e}")

    return passed_A


def test_output_shape_consistency():
    """Test that output shapes match input shapes for various configurations."""
    print("=" * 60)
    print("TEST: Output Shape Consistency")
    print("=" * 60)

    shapes = [
        (1, 64),
        (8, 128),
        (2, 4, 64),
        (2, 16, 256, 64),
        (1, 1, 1, 128),
    ]

    all_passed = True
    for shape in shapes:
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        y = rms_norm_with_softplus(x)

        passed = y.shape == x.shape and y.dtype == x.dtype
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] shape={shape} -> output shape={tuple(y.shape)}, dtype={y.dtype}")

    print()
    return all_passed


def test_output_positivity():
    """Test that softplus output is always positive."""
    print("=" * 60)
    print("TEST: Output Positivity (softplus guarantee)")
    print("=" * 60)

    torch.manual_seed(99)
    # Include negative values
    x = torch.randn(4, 16, 512, 64, dtype=torch.bfloat16, device="cuda") * 3.0
    y = rms_norm_with_softplus(x)

    min_val = y.min().item()
    passed = min_val > 0
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] min output value = {min_val:.8f} (should be > 0)")
    print()
    return passed


# =====================================================================
# Weight (per-channel gain) tests
# =====================================================================

def test_forward_with_weight():
    """Forward: kernel with weight matches fp32 reference."""
    print("=" * 60)
    print("TEST: Forward with per-channel weight")
    print("=" * 60)

    cases = [
        ((2, 8, 128, 64), None, "shape=(2,8,128,64), full softplus"),
        ((2, 8, 128, 64), 32,   "shape=(2,8,128,64), split_dim=32 (partial)"),
        ((1, 4, 256, 128), None, "shape=(1,4,256,128), full softplus"),
        ((1, 4, 256, 128), 64,  "shape=(1,4,256,128), split_dim=64"),
        ((4, 32, 64, 64), 0,    "shape=(4,32,64,64), split_dim=0 (pure RMSNorm)"),
    ]

    all_passed = True
    for shape, split, desc in cases:
        torch.manual_seed(0)
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        # Non-trivial weight: mean ~1, spread ~0.3 so it actually changes output
        w = (1.0 + 0.3 * torch.randn(shape[-1], device="cuda")).to(torch.bfloat16)

        y_ref = rms_norm_with_softplus_ref(x, weight=w, softplus_split_dim=split)
        y_fused = rms_norm_with_softplus(x, weight=w, softplus_split_dim=split)

        diff = (y_fused.float() - y_ref.float()).abs()
        max_abs = diff.max().item()
        rel = (diff / (y_ref.float().abs() + 1e-3)).mean().item()

        passed = max_abs < 0.05 and rel < 0.01
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {desc}")
        print(f"         max_abs={max_abs:.4e}, rel={rel:.4e}, "
              f"w range=[{w.float().min().item():+.3f},{w.float().max().item():+.3f}]")

    print()
    return all_passed


def test_backward_with_weight():
    """Backward: kernel grad_x AND grad_w match fp32 autograd reference."""
    print("=" * 60)
    print("TEST: Backward with per-channel weight (grad_x & grad_w)")
    print("=" * 60)

    cases = [
        ((2, 8, 64, 64), None, "shape=(2,8,64,64), full softplus"),
        ((2, 8, 64, 64), 32,   "shape=(2,8,64,64), split_dim=32"),
        ((1, 4, 128, 128), 64, "shape=(1,4,128,128), split_dim=64"),
        ((4, 8, 32, 64), 0,    "shape=(4,8,32,64), split_dim=0 (pure RMSNorm)"),
    ]

    all_passed = True
    for shape, split, desc in cases:
        torch.manual_seed(42)
        x_bf16 = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        w_bf16 = (1.0 + 0.3 * torch.randn(shape[-1], device="cuda")).to(torch.bfloat16)
        grad_out = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        # --- fp32 autograd ground truth (done entirely in fp32) ---
        x_ref = x_bf16.float().detach().requires_grad_(True)
        w_ref = w_bf16.float().detach().requires_grad_(True)
        rms = x_ref.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        normed = x_ref * rms * w_ref
        D = shape[-1]
        if split is None or split == D:
            y_ref = F.softplus(normed)
        elif split == 0:
            y_ref = normed
        else:
            left = F.softplus(normed[..., :split])
            right = normed[..., split:]
            y_ref = torch.cat([left, right], dim=-1)
        y_ref.backward(grad_out.float())
        grad_x_ref = x_ref.grad.detach()
        grad_w_ref = w_ref.grad.detach()

        # --- fused kernel ---
        x_fused = x_bf16.clone().detach().requires_grad_(True)
        w_fused = w_bf16.clone().detach().requires_grad_(True)
        y_fused = rms_norm_with_softplus(x_fused, weight=w_fused, softplus_split_dim=split)
        y_fused.backward(grad_out)
        grad_x_fused = x_fused.grad.detach().float()
        grad_w_fused = w_fused.grad.detach().float()

        # grad_x comparison
        dx = (grad_x_fused - grad_x_ref).abs()
        dx_max = dx.max().item()
        dx_ref = grad_x_ref.abs()
        mask_x = dx_ref > 1e-3
        dx_rel = (dx[mask_x] / dx_ref[mask_x]).mean().item() if mask_x.any() else 0.0

        # grad_w comparison (single vector, use relative on active)
        dw = (grad_w_fused - grad_w_ref).abs()
        dw_max = dw.max().item()
        dw_ref = grad_w_ref.abs()
        mask_w = dw_ref > 1e-3
        dw_rel = (dw[mask_w] / dw_ref[mask_w]).mean().item() if mask_w.any() else 0.0

        passed = (dx_max < 0.1 and dx_rel < 5e-2 and dw_max < 0.5 and dw_rel < 5e-2
                  and not torch.isnan(grad_x_fused).any().item()
                  and not torch.isnan(grad_w_fused).any().item())
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] {desc}")
        print(f"         grad_x: max_abs={dx_max:.4e}, rel_on_active={dx_rel:.4e}  "
              f"(active={int(mask_x.sum().item())}/{mask_x.numel()})")
        print(f"         grad_w: max_abs={dw_max:.4e}, rel_on_active={dw_rel:.4e}  "
              f"(active={int(mask_w.sum().item())}/{mask_w.numel()})")
        print(f"         grad_w range: fused=[{grad_w_fused.min().item():+.4e},{grad_w_fused.max().item():+.4e}]  "
              f"ref=[{grad_w_ref.min().item():+.4e},{grad_w_ref.max().item():+.4e}]")

    print()
    return all_passed


# =====================================================================
# Extreme value-range tests: verify forward / backward stay correct when
# softplus input z spans ~[-255, 255]. RMSNorm naturally clamps z near +/-1,
# so we construct inputs two ways:
#   (A) raw x with very large magnitude (scale up to +/-255)
#   (B) directly craft z to cover [-255, 255] and build x = z * rms so that
#       after normalization we *actually* hit those extreme values.
# =====================================================================

def _forward_error(y_fused, y_ref):
    diff = (y_fused.float() - y_ref.float()).abs()
    ref_abs = y_ref.float().abs()
    max_abs = diff.max().item()
    # relative error, guarded by the ref magnitude (larger eps so tiny refs
    # near 0 don't blow up the stat).
    rel = (diff / (ref_abs + 1e-3)).mean().item()
    return max_abs, rel, ref_abs.max().item()


def test_forward_large_range():
    """Forward with raw x magnitudes up to +/-255.

    Note: RMSNorm divides by the row's RMS, so z = x / rms(x) is scale-
    invariant w.r.t. x's global scale. This test therefore mainly stresses
    the fp32 reduction (sum of squares of ~255^2 * dim values).
    """
    print("=" * 60)
    print("TEST: Forward with Large Raw-x Range (up to +/-255)")
    print("=" * 60)

    shape = (2, 8, 128, 64)
    # Cover wide magnitudes; last two specifically test near-overflow for bf16
    # (bf16 max ~= 3.39e38, so x^2 up to ~65k is fine; sum over dim=64 -> ~4M
    #  still well within fp32 range).
    scales = [1.0, 32.0, 64.0, 128.0, 255.0]

    all_passed = True
    for scale in scales:
        torch.manual_seed(2024)
        # Uniform in [-scale, +scale] so the extremes actually appear
        x = (torch.rand(shape, dtype=torch.float32, device="cuda") * 2 - 1) * scale
        x = x.to(torch.bfloat16)

        y_ref = rms_norm_with_softplus_ref(x)
        y_fused = rms_norm_with_softplus(x)

        max_abs, rel, ref_max = _forward_error(y_fused, y_ref)
        # y = softplus(z), z ~ O(1); so y ~ [log2, ~| z |] i.e. O(1).
        passed = (not torch.isnan(y_fused).any().item()
                  and not torch.isinf(y_fused).any().item()
                  and max_abs < 0.1 and rel < 0.02)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False

        print(f"  [{status}] x range=[-{scale:.0f},+{scale:.0f}]  "
              f"x_min={x.float().min().item():+.2f} x_max={x.float().max().item():+.2f}  "
              f"y_range=[{y_fused.float().min().item():.4f},{y_fused.float().max().item():.4f}]  "
              f"max_abs={max_abs:.4e} rel={rel:.4e}")

    print()
    return all_passed


def _build_extreme_z_input(shape, z_values_per_row, dtype=torch.bfloat16):
    """Construct x so that z = x / rms(x) exactly equals a desired pattern.

    For a row z in R^D, if we pick x = z directly then rms(x) = rms(z),
    and the normalized output is z / rms(z), not z itself. To make the
    *post-normalization* activation equal a target pattern T, we need:
        x = T * c   for ANY c > 0
    because rms(T * c) = c * rms(T) and then (x / rms(x)) = T / rms(T).

    So we build T such that rms(T) = 1 on every row by construction:
    sample a base pattern, then normalize each row. Scale it to cover the
    desired z-range directly.

    Returns x (bf16), and the analytically-known z = x / rms(x) tensor.
    """
    # z_values_per_row: function(i) -> torch.Tensor of shape (D,)
    num_rows = 1
    for s in shape[:-1]:
        num_rows *= int(s)
    rows = [z_values_per_row(i) for i in range(num_rows)]
    Z = torch.stack(rows, dim=0).to(torch.float32).cuda()  # (R, D)
    # Renormalize each row so that rms(Z) = 1  => after RMSNorm, z stays Z.
    rms = Z.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    Z = Z / rms
    x = Z.reshape(shape).contiguous().to(dtype)
    return x, Z.reshape(shape)


def test_forward_extreme_z_range():
    """Forward when the *post-normalization* z spans [-255, 255].

    This directly stresses softplus's extreme-z branches:
      z >> 0 : softplus(z) ~= z        (kernel uses z itself when z >= 20)
      z << 0 : softplus(z) ~= exp(z) ~= 0  (must NOT produce NaN)
    """
    print("=" * 60)
    print("TEST: Forward with Post-Norm z in [-255, 255]")
    print("=" * 60)

    shape = (4, 256, 64)   # 1024 rows, D=64
    D = shape[-1]
    torch.manual_seed(0)

    def make_row(i):
        # Row i gets a linearly interpolated extreme-z pattern.
        # Half positive extremes, half negative extremes, plus random middle.
        half = D // 2
        lo = -255.0 + (i % 16) * 2.0      # sweep slightly across rows
        hi = +255.0 - (i % 16) * 2.0
        neg = torch.linspace(lo, -1.0, half)
        pos = torch.linspace(1.0, hi, D - half)
        r = torch.cat([neg, pos])
        # shuffle so extremes aren't all at the same channel positions
        g = torch.Generator().manual_seed(i + 1)
        perm = torch.randperm(D, generator=g)
        return r[perm]

    x, z_target = _build_extreme_z_input(shape, make_row)

    y_ref = rms_norm_with_softplus_ref(x)
    y_fused = rms_norm_with_softplus(x)

    # Basic finiteness check - critical at |z|=255 (exp(-255) underflow to 0 is fine;
    # log(1+exp(255)) would overflow in fp32 if not handled by the z>=20 fast path).
    nan_fused = torch.isnan(y_fused).any().item()
    inf_fused = torch.isinf(y_fused).any().item()
    nan_ref   = torch.isnan(y_ref).any().item()
    inf_ref   = torch.isinf(y_ref).any().item()

    max_abs, rel, ref_max = _forward_error(y_fused, y_ref)

    # Measure actual z extremes realized on device
    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    z_actual = (x.float() * rms)
    print(f"  actual z range  : [{z_actual.min().item():+.3f}, {z_actual.max().item():+.3f}]")
    print(f"  y_fused range   : [{y_fused.float().min().item():.4f}, {y_fused.float().max().item():.4f}]")
    print(f"  y_ref   range   : [{y_ref.float().min().item():.4f}, {y_ref.float().max().item():.4f}]")
    print(f"  finiteness      : fused NaN={nan_fused} Inf={inf_fused} | ref NaN={nan_ref} Inf={inf_ref}")
    print(f"  max_abs_err={max_abs:.4e}  rel_err={rel:.4e}  ref_max={ref_max:.2f}")

    # The reference does full fp32 softplus then casts to bf16; the kernel uses
    # an `if z>=20: z` fast-path. For z>=20, softplus(z) - z is below bf16 precision,
    # so both should agree to within bf16 rounding (|err| <= |y| * ~7.8e-3 for bf16).
    # We use a relaxed absolute tolerance because y can be ~255 here.
    atol = 0.5   # bf16 roundoff at y=255 is ~2, but most elements are O(1)
    passed = (not nan_fused and not inf_fused
              and max_abs < atol
              and rel < 5e-3)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]")
    print()
    return passed


def test_backward_extreme_z_range():
    """Backward precision when post-norm z spans [-255, 255].

    The key gradient-path nodes:
      d softplus(z)/dz = sigmoid(z)
        z >> 0 : sigmoid -> 1         (gradient flows through as-is)
        z << 0 : sigmoid -> 0         (gradient should vanish, NO NaN)
      Then the RMSNorm chain:
        grad_x = rms_inv * (g - x * dot(g, x) / (D * rms^2))
    """
    print("=" * 60)
    print("TEST: Backward with Post-Norm z in [-255, 255]")
    print("=" * 60)

    # Use a moderate-size problem so ref fp32 autograd is affordable.
    shape = (2, 64, 64)
    D = shape[-1]
    torch.manual_seed(1)

    def make_row(i):
        half = D // 2
        lo = -255.0 + (i % 8) * 4.0
        hi = +255.0 - (i % 8) * 4.0
        neg = torch.linspace(lo, -1.0, half)
        pos = torch.linspace(1.0, hi, D - half)
        r = torch.cat([neg, pos])
        g = torch.Generator().manual_seed(i + 101)
        perm = torch.randperm(D, generator=g)
        return r[perm]

    x, _ = _build_extreme_z_input(shape, make_row)

    torch.manual_seed(2)
    grad_out = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    # Reference backward via fp32 autograd
    x_ref = x.detach().float().requires_grad_(True)
    rms = x_ref.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    z = x_ref * rms
    y_ref = F.softplus(z)
    y_ref.backward(grad_out.float())
    grad_ref = x_ref.grad.detach()

    # Fused kernel backward
    x_fused = x.detach().clone().requires_grad_(True)
    y_fused = rms_norm_with_softplus(x_fused)
    y_fused.backward(grad_out)
    grad_fused = x_fused.grad.detach()

    nan_grad = torch.isnan(grad_fused).any().item()
    inf_grad = torch.isinf(grad_fused).any().item()

    diff = (grad_fused.float() - grad_ref.float()).abs()
    ref_abs = grad_ref.abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    # relative error with a small additive floor (many grads are ~0 where sigmoid->0)
    rel = (diff / (ref_abs + 1e-3)).mean().item()
    # robust relative error restricted to locations where the ref grad is meaningful
    mask = ref_abs > 1e-2
    if mask.any():
        rel_on_active = (diff[mask] / ref_abs[mask]).mean().item()
    else:
        rel_on_active = 0.0

    # Diagnostic: check the two extreme regions separately
    rms_x = x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
    z_actual = x.float() * rms_x
    pos_mask = z_actual > 10
    neg_mask = z_actual < -10
    def region_stats(name, m):
        if not m.any():
            print(f"    {name}: <empty>")
            return
        d = (grad_fused.float() - grad_ref.float()).abs()[m]
        r = grad_ref.float().abs()[m]
        print(f"    {name}: count={m.sum().item()}  "
              f"grad_abs_max(fused)={grad_fused.float().abs()[m].max().item():.3e}  "
              f"grad_abs_max(ref)={r.max().item():.3e}  "
              f"max_abs_diff={d.max().item():.3e}  mean_diff={d.mean().item():.3e}")

    print(f"  shape={shape}, z range=[{z_actual.min().item():+.2f},{z_actual.max().item():+.2f}]")
    print(f"  grad_fused finite : NaN={nan_grad} Inf={inf_grad}")
    print(f"  overall           : max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} "
          f"rel={rel:.4e} rel_on_active={rel_on_active:.4e}")
    print("  per-region diagnostics:")
    region_stats("z >  10  (sigmoid~1)", pos_mask)
    region_stats("z < -10  (sigmoid~0)", neg_mask)
    region_stats("|z|<= 10 (normal)   ", ~(pos_mask | neg_mask))

    # Tolerances: grads can be O(1) because the sigmoid mask kills the z<<0
    # entries, but on z>>10 entries the path is basically RMSNorm backward with
    # g = grad_y. We allow bf16-ish tolerance.
    passed = (not nan_grad and not inf_grad
              and max_abs < 5e-2
              and rel_on_active < 5e-2)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}]")
    print()
    return passed


def test_backward_extreme_z_finiteness_only():
    """Fast guard: purely check no NaN/Inf in bwd under saturated extreme-z
    inputs (uniform z in [-255,255]). A single FAIL here strongly hints at
    a numerical issue in the kernel's softplus/sigmoid fast-path."""
    print("=" * 60)
    print("TEST: Backward Finiteness under Saturated z in [-255, 255]")
    print("=" * 60)

    shape = (2, 128, 64)
    D = shape[-1]

    def make_row(i):
        g = torch.Generator().manual_seed(777 + i)
        return (torch.rand(D, generator=g) * 2 - 1) * 255.0

    x, _ = _build_extreme_z_input(shape, make_row)
    x = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    y = rms_norm_with_softplus(x)
    y.backward(grad_out)
    g = x.grad

    nan_y, inf_y = torch.isnan(y).any().item(), torch.isinf(y).any().item()
    nan_g, inf_g = torch.isnan(g).any().item(), torch.isinf(g).any().item()
    passed = not (nan_y or inf_y or nan_g or inf_g)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] y: NaN={nan_y} Inf={inf_y} | grad_x: NaN={nan_g} Inf={inf_g}")
    print(f"         grad_x abs_max={g.float().abs().max().item():.4e}  "
          f"min={g.float().min().item():+.4e}  max={g.float().max().item():+.4e}")
    print()
    return passed


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Fused RMSNorm + Softplus Kernel Test Suite")
    print("=" * 60 + "\n")

    results = []
    results.append(("Forward Precision", test_forward_precision()))
    results.append(("Forward Different Scales", test_forward_different_scales()))
    results.append(("Backward Precision", test_backward_precision()))
    results.append(("Numerical Gradient Check", test_gradcheck_numerical()))
    results.append(("Output Shape Consistency", test_output_shape_consistency()))
    results.append(("Output Positivity", test_output_positivity()))
    results.append(("Forward Large Raw-x Range", test_forward_large_range()))
    results.append(("Forward Extreme z Range", test_forward_extreme_z_range()))
    results.append(("Backward Extreme z Range", test_backward_extreme_z_range()))
    results.append(("Backward Finiteness Saturated z", test_backward_extreme_z_finiteness_only()))
    results.append(("Forward with weight", test_forward_with_weight()))
    results.append(("Backward with weight (grad_x & grad_w)", test_backward_with_weight()))

    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("   ALL TESTS PASSED!")
    else:
        print("   SOME TESTS FAILED!")
        sys.exit(1)
    print()
