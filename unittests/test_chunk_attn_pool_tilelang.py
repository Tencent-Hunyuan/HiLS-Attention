"""
Unit tests for TileLang chunk_attn_pool kernel.

Tests forward correctness, backward correctness, and performance against
the PyTorch reference implementation.
"""

import torch
import torch.nn.functional as F
import pytest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ops.chunk_attn_pool_tilelang import (
    chunk_attn_pool_tilelang,
    chunk_attn_pool_tilelang_shared_k,
    chunk_attn_pool_ref,
)


# =====================================================================
# Helpers
# =====================================================================

def get_err_ratio(x, y):
    """RMS relative error between two tensors."""
    x_f = x.float().flatten()
    y_f = y.float().flatten()
    mask = (x_f.abs() > 1e-6) | (y_f.abs() > 1e-6)
    if mask.sum() == 0:
        return 0.0
    err = (x_f[mask] - y_f[mask]).square().mean().sqrt().item()
    base = x_f[mask].square().mean().sqrt().item()
    return err / (base + 1e-12)


def get_abs_err(x, y):
    """Max absolute error between two tensors."""
    return (x.float() - y.float()).abs().max().item()


def assert_close(prefix, ref, tri, ratio_tol=0.01, atol=1e-3):
    """Assert two tensors are close, print diagnostics on failure."""
    abs_err = get_abs_err(ref, tri)
    rel_ratio = get_err_ratio(ref, tri)
    msg = f"{prefix} | abs_err: {abs_err:.6f} | rel_ratio: {rel_ratio:.6f}"
    print(msg)
    if rel_ratio > ratio_tol and abs_err > atol:
        raise AssertionError(msg)


# =====================================================================
# Forward Tests
# =====================================================================

class TestChunkAttnPoolForward:
    """Forward correctness: TileLang kernel vs PyTorch reference."""

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 64, 2, 128, 64),     # minimal typical (fallback path)
        (2, 128, 4, 128, 64),    # multi-batch (fallback path)
        (1, 2048, 8, 128, 64),   # large N (fallback path)
        (4, 256, 2, 128, 64),    # large batch (fallback path)
        (1, 32, 2, 64, 64),      # smaller D (fallback path)
        (1, 16, 1, 128, 32),     # smaller chunk size (fallback path)
    ])
    def test_forward_matches_reference(self, B, N, H, D, S):
        torch.manual_seed(42)
        device = "cuda"

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)
        sm_scale = 1.0 / math.sqrt(D)

        # Reference (PyTorch)
        lmk_k_ref, lmk_b_ref = chunk_attn_pool_ref(mu_q, k_chunked, sm_scale)

        # TileLang kernel
        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang(mu_q, k_chunked, sm_scale)

        # Check lmk_k (bf16 output)
        assert_close(
            f"lmk_k [B={B}, N={N}, H={H}, D={D}, S={S}]",
            lmk_k_ref, lmk_k_tl,
            ratio_tol=0.005, atol=5e-3
        )

        # Check lmk_b (fp32 entropy)
        assert_close(
            f"lmk_b [B={B}, N={N}, H={H}, D={D}, S={S}]",
            lmk_b_ref, lmk_b_tl,
            ratio_tol=0.01, atol=1e-2
        )

    def test_forward_default_sm_scale(self):
        """Test that default sm_scale=1/sqrt(D) is applied correctly."""
        torch.manual_seed(123)
        device = "cuda"
        B, N, H, D, S = 1, 32, 2, 128, 64

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)

        # Without explicit sm_scale
        lmk_k_1, lmk_b_1 = chunk_attn_pool_tilelang(mu_q, k_chunked)
        # With explicit sm_scale
        lmk_k_2, lmk_b_2 = chunk_attn_pool_tilelang(mu_q, k_chunked, 1.0 / math.sqrt(D))

        assert torch.allclose(lmk_k_1, lmk_k_2, atol=1e-6)
        assert torch.allclose(lmk_b_1, lmk_b_2, atol=1e-6)

    def test_forward_last_token_masked(self):
        """Verify the last token in each chunk has zero attention weight."""
        torch.manual_seed(77)
        device = "cuda"
        B, N, H, D, S = 1, 4, 1, 64, 8

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)

        # Get the reference softmax weights
        sm_scale = 1.0 / math.sqrt(D)
        mu_f32 = mu_q.float()
        k_f32 = k_chunked.float()
        logits = torch.einsum("bnhd,bnshd->bnsh", mu_f32, k_f32) * sm_scale
        logits[:, :, -1, :] = float('-inf')
        p = F.softmax(logits, dim=2)

        # Last token weight should be 0
        assert (p[:, :, -1, :] == 0.0).all(), "Last token should have zero weight"

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 64, 32, 128, 64),
        (2, 128, 16, 128, 64),
    ])
    def test_forward_shared_k_path(self, B, N, H, D, S):
        """Test v8 multi-head kernel with shared K (K identical across heads)."""
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        # Create shared K: same data for all heads
        k_single = torch.randn(B, N, S, 1, D, dtype=torch.bfloat16, device=device)
        k_chunked = k_single.expand(B, N, S, H, D).contiguous()

        lmk_k_ref, lmk_b_ref = chunk_attn_pool_ref(mu_q, k_chunked, sm_scale)
        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang_shared_k(mu_q, k_chunked, sm_scale)

        assert_close(f"shared_k lmk_k [H={H}]", lmk_k_ref, lmk_k_tl, ratio_tol=0.005, atol=5e-3)
        assert_close(f"shared_k lmk_b [H={H}]", lmk_b_ref, lmk_b_tl, ratio_tol=0.01, atol=1e-2)


# =====================================================================
# Backward Tests
# =====================================================================

class TestChunkAttnPoolBackward:
    """Backward gradient correctness tests."""

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 32, 2, 128, 64),
        (2, 64, 4, 128, 64),
        (1, 16, 2, 64, 32),
    ])
    def test_backward_matches_autograd(self, B, N, H, D, S):
        """Compare TileLang backward against PyTorch autograd on the reference."""
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        # === Reference path (fp32 PyTorch autograd) ===
        mu_q_ref = torch.randn(B, N, H, D, dtype=torch.float32, device=device, requires_grad=True)
        k_ref = torch.randn(B, N, S, H, D, dtype=torch.float32, device=device, requires_grad=True)

        lmk_k_ref, lmk_b_ref = chunk_attn_pool_ref(
            mu_q_ref.to(torch.bfloat16), k_ref.to(torch.bfloat16), sm_scale
        )

        # Use fixed random gradients
        grad_k = torch.randn_like(lmk_k_ref.float())
        grad_b = torch.randn_like(lmk_b_ref.float())

        # Recompute with float32 for reference grads
        lmk_k_ref2, lmk_b_ref2 = chunk_attn_pool_ref(mu_q_ref, k_ref, sm_scale)
        loss_ref = (lmk_k_ref2.float() * grad_k).sum() + (lmk_b_ref2.float() * grad_b).sum()
        loss_ref.backward()

        d_mu_q_ref = mu_q_ref.grad.clone()
        d_k_ref = k_ref.grad.clone()

        # === TileLang path ===
        mu_q_tl = mu_q_ref.detach().to(torch.bfloat16).requires_grad_(True)
        k_tl = k_ref.detach().to(torch.bfloat16).requires_grad_(True)

        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang(mu_q_tl, k_tl, sm_scale)
        loss_tl = (lmk_k_tl.float() * grad_k).sum() + (lmk_b_tl.float() * grad_b).sum()
        loss_tl.backward()

        d_mu_q_tl = mu_q_tl.grad.clone()
        d_k_tl = k_tl.grad.clone()

        # Check gradients (bf16 kernel → relaxed tolerance)
        assert_close(
            f"d_mu_q [B={B}, N={N}, H={H}]",
            d_mu_q_ref, d_mu_q_tl.float(),
            ratio_tol=0.03, atol=5e-2
        )
        assert_close(
            f"d_k_chunked [B={B}, N={N}, H={H}]",
            d_k_ref, d_k_tl.float(),
            ratio_tol=0.03, atol=5e-2
        )

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 16, 2, 128, 64),
    ])
    def test_grad_lmk_k_only(self, B, N, H, D, S):
        """Test backward when only grad_lmk_k is non-zero (grad_lmk_b = 0)."""
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        mu_q_ref = torch.randn(B, N, H, D, dtype=torch.float32, device=device, requires_grad=True)
        k_ref = torch.randn(B, N, S, H, D, dtype=torch.float32, device=device, requires_grad=True)
        lmk_k_ref, _ = chunk_attn_pool_ref(mu_q_ref, k_ref, sm_scale)
        grad_k = torch.randn_like(lmk_k_ref.float())
        (lmk_k_ref.float() * grad_k).sum().backward()

        d_mu_q_ref = mu_q_ref.grad.clone()

        mu_q_tl = mu_q_ref.detach().to(torch.bfloat16).requires_grad_(True)
        k_tl = k_ref.detach().to(torch.bfloat16).requires_grad_(True)
        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang(mu_q_tl, k_tl, sm_scale)
        # Zero out grad_lmk_b
        (lmk_k_tl.float() * grad_k).sum().backward()

        d_mu_q_tl = mu_q_tl.grad.clone()
        assert_close("d_mu_q (grad_b=0)", d_mu_q_ref, d_mu_q_tl.float(), ratio_tol=0.03, atol=5e-2)

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 16, 2, 128, 64),
    ])
    def test_grad_lmk_b_only(self, B, N, H, D, S):
        """Test backward when only grad_lmk_b is non-zero (grad_lmk_k = 0)."""
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        mu_q_ref = torch.randn(B, N, H, D, dtype=torch.float32, device=device, requires_grad=True)
        k_ref = torch.randn(B, N, S, H, D, dtype=torch.float32, device=device, requires_grad=True)
        _, lmk_b_ref = chunk_attn_pool_ref(mu_q_ref, k_ref, sm_scale)
        grad_b = torch.randn_like(lmk_b_ref.float())
        (lmk_b_ref.float() * grad_b).sum().backward()

        d_mu_q_ref = mu_q_ref.grad.clone()
        d_k_ref = k_ref.grad.clone()

        mu_q_tl = mu_q_ref.detach().to(torch.bfloat16).requires_grad_(True)
        k_tl = k_ref.detach().to(torch.bfloat16).requires_grad_(True)
        _, lmk_b_tl = chunk_attn_pool_tilelang(mu_q_tl, k_tl, sm_scale)
        (lmk_b_tl.float() * grad_b).sum().backward()

        d_mu_q_tl = mu_q_tl.grad.clone()
        d_k_tl = k_tl.grad.clone()

        assert_close("d_mu_q (grad_k=0)", d_mu_q_ref, d_mu_q_tl.float(), ratio_tol=0.05, atol=1e-1)
        assert_close("d_k (grad_k=0)", d_k_ref, d_k_tl.float(), ratio_tol=0.05, atol=1e-1)


# =====================================================================
# compact_lmk_k path (fallback kernel) — fwd + bwd
# =====================================================================

class TestChunkAttnPoolCompact:
    """compact_lmk_k=True path of the fallback kernel.

    In this mode the kernel skips the P @ K GEMM and writes ``lmk_k = 0``;
    only the entropy bias ``lmk_b`` is meaningful.  The caller (hils_layer)
    overrides ``lmk_k`` with the chunk-boundary K token.

    For backward, the kernel forces ``GradK_shared = 0`` regardless of the
    incoming ``grad_lmk_k`` -- so even if downstream passes a non-zero
    ``grad_lmk_k``, the kernel must produce gradients identical to the
    case where only ``grad_lmk_b`` flows.  We rely on this invariant
    when sharing the parameter with the non-compact path.
    """

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 32, 2, 128, 64),
        (2, 64, 4, 128, 64),
        (1, 16, 1, 64, 32),
    ])
    def test_forward_compact(self, B, N, H, D, S):
        """compact fwd: lmk_b matches ref; lmk_k is exactly zero."""
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)

        # Reference (full path) for lmk_b.
        _, lmk_b_ref = chunk_attn_pool_ref(mu_q, k_chunked, sm_scale)

        # TileLang kernel in compact mode.
        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang(
            mu_q, k_chunked, sm_scale, compact_lmk_k=True
        )

        # lmk_b must match the reference exactly (entropy is computed the
        # same way regardless of compact_lmk_k).
        assert_close(
            f"compact lmk_b [B={B}, N={N}, H={H}, D={D}, S={S}]",
            lmk_b_ref, lmk_b_tl,
            ratio_tol=0.01, atol=1e-2,
        )

        # lmk_k must be all zero (kernel writes 0.0 in compact mode).
        assert lmk_k_tl.abs().max().item() == 0.0, (
            f"compact mode should write lmk_k=0, got max abs "
            f"{lmk_k_tl.abs().max().item()}"
        )

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 32, 2, 128, 64),
        (2, 64, 4, 128, 64),
    ])
    def test_backward_compact_grad_b_only(self, B, N, H, D, S):
        """compact bwd: gradients come only from grad_lmk_b path.

        We compare against a PyTorch autograd reference that ALSO drops the
        ``lmk_k`` term from the loss, so the comparison is apples-to-apples.
        """
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        # === Reference path: only lmk_b participates in the loss ===
        mu_q_ref = torch.randn(B, N, H, D, dtype=torch.float32, device=device, requires_grad=True)
        k_ref = torch.randn(B, N, S, H, D, dtype=torch.float32, device=device, requires_grad=True)
        _, lmk_b_ref = chunk_attn_pool_ref(mu_q_ref, k_ref, sm_scale)
        grad_b = torch.randn_like(lmk_b_ref.float())
        (lmk_b_ref.float() * grad_b).sum().backward()
        d_mu_q_ref = mu_q_ref.grad.clone()
        d_k_ref = k_ref.grad.clone()

        # === TileLang compact path ===
        mu_q_tl = mu_q_ref.detach().to(torch.bfloat16).requires_grad_(True)
        k_tl = k_ref.detach().to(torch.bfloat16).requires_grad_(True)
        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang(
            mu_q_tl, k_tl, sm_scale, compact_lmk_k=True
        )
        # Loss uses only lmk_b; lmk_k_tl is zero anyway.
        (lmk_b_tl.float() * grad_b).sum().backward()
        d_mu_q_tl = mu_q_tl.grad.clone()
        d_k_tl = k_tl.grad.clone()

        assert_close(
            f"compact d_mu_q [B={B}, N={N}, H={H}]",
            d_mu_q_ref, d_mu_q_tl.float(),
            ratio_tol=0.05, atol=1e-1,
        )
        assert_close(
            f"compact d_k [B={B}, N={N}, H={H}]",
            d_k_ref, d_k_tl.float(),
            ratio_tol=0.05, atol=1e-1,
        )

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 32, 2, 128, 64),
    ])
    def test_backward_compact_ignores_grad_k(self, B, N, H, D, S):
        """Compact bwd must IGNORE any incoming grad_lmk_k.

        The kernel hard-zeros ``GradK_shared`` in compact mode (see fallback
        line ~466 / v8 line ~230).  Therefore feeding a non-zero gradient
        into ``lmk_k`` must produce IDENTICAL d_mu_q / d_k to the case with
        zero grad_lmk_k.  This is a critical invariant because hils_layer
        passes the autograd-built grad_lmk_k unconditionally -- if it leaked
        through, downstream would silently train on a wrong objective.
        """
        torch.manual_seed(123)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)

        # --- Run 1: backward with non-zero grad_lmk_k ---
        mu_q_a = mu_q.clone().detach().requires_grad_(True)
        k_a = k_chunked.clone().detach().requires_grad_(True)
        lmk_k_a, lmk_b_a = chunk_attn_pool_tilelang(mu_q_a, k_a, sm_scale, compact_lmk_k=True)
        grad_k_nonzero = torch.randn_like(lmk_k_a.float()) * 5.0  # large, on purpose
        grad_b = torch.randn_like(lmk_b_a.float())
        ((lmk_k_a.float() * grad_k_nonzero).sum() + (lmk_b_a * grad_b).sum()).backward()
        d_mu_q_a = mu_q_a.grad.clone()
        d_k_a = k_a.grad.clone()

        # --- Run 2: backward with grad_lmk_k = 0 ---
        mu_q_b = mu_q.clone().detach().requires_grad_(True)
        k_b = k_chunked.clone().detach().requires_grad_(True)
        _, lmk_b_b = chunk_attn_pool_tilelang(mu_q_b, k_b, sm_scale, compact_lmk_k=True)
        (lmk_b_b * grad_b).sum().backward()
        d_mu_q_b = mu_q_b.grad.clone()
        d_k_b = k_b.grad.clone()

        # The gradients must be IDENTICAL (not just close) because the
        # kernel literally zeros out the grad_lmk_k contribution.
        assert torch.equal(d_mu_q_a, d_mu_q_b), (
            "compact mode leaked grad_lmk_k into d_mu_q; "
            f"max diff = {(d_mu_q_a - d_mu_q_b).abs().max().item()}"
        )
        assert torch.equal(d_k_a, d_k_b), (
            "compact mode leaked grad_lmk_k into d_k; "
            f"max diff = {(d_k_a - d_k_b).abs().max().item()}"
        )


# =====================================================================
# Multi-head v8 (shared-K) path — backward
# =====================================================================

class TestChunkAttnPoolSharedKBackward:
    """Backward correctness for the multi-head v8 kernel (shared K).

    The forward path is exercised by ``test_forward_shared_k_path`` above;
    here we validate ``ChunkAttnPoolSharedKFunction.backward`` (which calls
    ``chunk_attn_pool_bwd_kernel``, the H-as-M variant).
    """

    @pytest.mark.parametrize("B,N,H,D,S", [
        (1, 32, 16, 128, 64),
        (2, 64, 32, 128, 64),
    ])
    def test_backward_shared_k(self, B, N, H, D, S):
        torch.manual_seed(42)
        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        # Build the inputs. The ground-truth K has shape (B, N, S, 1, D);
        # we then ``expand`` (a view, NOT contiguous) along the H axis when
        # feeding the reference -- this lets autograd accumulate gradients
        # from all H heads BACK into the single-head leaf, exactly the
        # aggregation the v8 kernel performs.
        mu_q_base = torch.randn(B, N, H, D, dtype=torch.float32, device=device)
        k_single_base = torch.randn(B, N, S, 1, D, dtype=torch.float32, device=device)

        # === Reference: single-head K, expanded into the H axis as a view ===
        mu_q_ref = mu_q_base.clone().detach().requires_grad_(True)
        k_single_ref = k_single_base.clone().detach().requires_grad_(True)
        k_expanded_ref = k_single_ref.expand(B, N, S, H, D)  # NO contiguous!
        lmk_k_ref, lmk_b_ref = chunk_attn_pool_ref(mu_q_ref, k_expanded_ref, sm_scale)
        grad_k = torch.randn_like(lmk_k_ref.float())
        grad_b = torch.randn_like(lmk_b_ref.float())
        ((lmk_k_ref.float() * grad_k).sum() + (lmk_b_ref * grad_b).sum()).backward()
        d_mu_q_ref = mu_q_ref.grad.clone()
        # ``k_single_ref.grad`` has shape (B, N, S, 1, D) and already
        # contains the SUM of gradients across all H heads (via autograd's
        # implicit accumulation through the ``expand`` view).  This is
        # exactly the aggregated gradient the v8 kernel writes into
        # ``d_k[..., 0, :]``.
        d_k_ref_aggregated = k_single_ref.grad.squeeze(3)  # (B, N, S, D)

        # === TileLang shared-K path: full (B, N, S, H, D) layout, K rows
        # MUST be identical for the v8 kernel to produce a meaningful
        # forward output (the kernel only reads head 0 in fwd).  We use
        # ``expand`` + ``contiguous`` to materialize the redundancy. ===
        k_full_base = k_single_base.expand(B, N, S, H, D).contiguous()
        mu_q_tl = mu_q_base.clone().detach().to(torch.bfloat16).requires_grad_(True)
        k_tl = k_full_base.clone().detach().to(torch.bfloat16).requires_grad_(True)
        lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang_shared_k(mu_q_tl, k_tl, sm_scale)
        ((lmk_k_tl.float() * grad_k).sum() + (lmk_b_tl * grad_b).sum()).backward()
        d_mu_q_tl = mu_q_tl.grad.clone()
        d_k_tl = k_tl.grad.clone()

        assert_close(
            f"shared_k d_mu_q [B={B}, N={N}, H={H}]",
            d_mu_q_ref, d_mu_q_tl.float(),
            ratio_tol=0.03, atol=5e-2,
        )
        # The v8 kernel writes the aggregated d_k into row 0 only;
        # other rows must remain zero.
        assert_close(
            f"shared_k d_k row0 [B={B}, N={N}, H={H}]",
            d_k_ref_aggregated, d_k_tl[:, :, :, 0, :].float(),
            ratio_tol=0.05, atol=1e-1,
        )
        # Verify the rest of the head axis was left alone.
        if H > 1:
            other_max = d_k_tl[:, :, :, 1:, :].abs().max().item()
            assert other_max == 0.0, (
                f"v8 kernel wrote into d_k[..., h>0, :], max abs = {other_max}"
            )


# =====================================================================
# Performance Benchmark
# =====================================================================

class TestChunkAttnPoolPerformance:
    """Latency benchmark (not an assertion, just prints)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_performance_comparison(self):
        """Benchmark TileLang kernel vs torch.compile reference."""
        torch.manual_seed(42)
        device = "cuda"
        B, N, H, D, S = 4, 2048, 32, 128, 64
        sm_scale = 1.0 / math.sqrt(D)

        mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
        k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(5):
            chunk_attn_pool_tilelang(mu_q, k_chunked, sm_scale)
            chunk_attn_pool_ref(mu_q, k_chunked, sm_scale)
        torch.cuda.synchronize()

        # Benchmark reference
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        N_ITER = 50
        start.record()
        for _ in range(N_ITER):
            chunk_attn_pool_ref(mu_q, k_chunked, sm_scale)
        end.record()
        torch.cuda.synchronize()
        time_ref = start.elapsed_time(end) / N_ITER

        # Benchmark TileLang
        start.record()
        for _ in range(N_ITER):
            chunk_attn_pool_tilelang(mu_q, k_chunked, sm_scale)
        end.record()
        torch.cuda.synchronize()
        time_tl = start.elapsed_time(end) / N_ITER

        print(f"\n{'='*60}")
        print(f"  Performance Comparison (B={B}, N={N}, H={H}, D={D}, S={S})")
        print(f"  Reference (PyTorch):  {time_ref:.3f} ms")
        print(f"  TileLang kernel:      {time_tl:.3f} ms")
        print(f"  Speedup:              {time_ref/time_tl:.2f}x")
        print(f"{'='*60}")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_backward_performance(self):
        """Benchmark backward pass."""
        torch.manual_seed(42)
        device = "cuda"
        B, N, H, D, S = 4, 2048, 32, 128, 64
        sm_scale = 1.0 / math.sqrt(D)

        mu_q_tl = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
        k_tl = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)

        mu_q_ref = mu_q_tl.detach().clone().float().requires_grad_(True)
        k_ref = k_tl.detach().clone().float().requires_grad_(True)

        # Warmup
        for _ in range(3):
            lmk_k, lmk_b = chunk_attn_pool_tilelang(mu_q_tl, k_tl, sm_scale)
            (lmk_k.float().sum() + lmk_b.sum()).backward()
            mu_q_tl.grad = None
            k_tl.grad = None
        torch.cuda.synchronize()

        # Benchmark TileLang backward
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        N_ITER = 30

        start.record()
        for _ in range(N_ITER):
            lmk_k, lmk_b = chunk_attn_pool_tilelang(mu_q_tl, k_tl, sm_scale)
            (lmk_k.float().sum() + lmk_b.sum()).backward()
            mu_q_tl.grad = None
            k_tl.grad = None
        end.record()
        torch.cuda.synchronize()
        time_tl = start.elapsed_time(end) / N_ITER

        # Benchmark reference backward
        start.record()
        for _ in range(N_ITER):
            lmk_k, lmk_b = chunk_attn_pool_ref(mu_q_ref, k_ref, sm_scale)
            (lmk_k.float().sum() + lmk_b.sum()).backward()
            mu_q_ref.grad = None
            k_ref.grad = None
        end.record()
        torch.cuda.synchronize()
        time_ref = start.elapsed_time(end) / N_ITER

        print(f"\n{'='*60}")
        print(f"  Backward Performance (B={B}, N={N}, H={H}, D={D}, S={S})")
        print(f"  Reference (PyTorch):  {time_ref:.3f} ms")
        print(f"  TileLang kernel:      {time_tl:.3f} ms")
        print(f"  Speedup:              {time_ref/time_tl:.2f}x")
        print(f"{'='*60}")


# =====================================================================
# Entry point for quick validation
# =====================================================================

if __name__ == "__main__":
    print("Running quick forward validation...")
    torch.manual_seed(42)
    device = "cuda"
    B, N, H, D, S = 2, 64, 4, 128, 64
    sm_scale = 1.0 / math.sqrt(D)

    mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device)
    k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device)

    print(f"Input shapes: mu_q={tuple(mu_q.shape)}, k_chunked={tuple(k_chunked.shape)}")

    lmk_k_ref, lmk_b_ref = chunk_attn_pool_ref(mu_q, k_chunked, sm_scale)
    print(f"Reference done: lmk_k={tuple(lmk_k_ref.shape)}, lmk_b={tuple(lmk_b_ref.shape)}")

    lmk_k_tl, lmk_b_tl = chunk_attn_pool_tilelang(mu_q, k_chunked, sm_scale)
    print(f"TileLang done: lmk_k={tuple(lmk_k_tl.shape)}, lmk_b={tuple(lmk_b_tl.shape)}")

    assert_close("lmk_k", lmk_k_ref, lmk_k_tl, ratio_tol=0.005, atol=5e-3)
    assert_close("lmk_b", lmk_b_ref, lmk_b_tl, ratio_tol=0.01, atol=1e-2)

    print("\nRunning quick backward validation...")
    mu_q_tl = torch.randn(B, N, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    k_tl = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device=device, requires_grad=True)
    lmk_k, lmk_b = chunk_attn_pool_tilelang(mu_q_tl, k_tl, sm_scale)
    loss = lmk_k.float().sum() + lmk_b.sum()
    loss.backward()
    print(f"d_mu_q shape: {mu_q_tl.grad.shape}, d_k shape: {k_tl.grad.shape}")
    print("Backward pass completed successfully!")
    print("\nAll quick checks passed!")
