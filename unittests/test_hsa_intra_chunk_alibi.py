"""Correctness tests for the intra-chunk ALiBi extension of ``HSA_block_M_head``.

Design (mirrors the semantics requested by the user):
    - HSA attends each q to a set of K chunks, each of length S.
    - Intra-chunk positions are "reset" -- every chunk uses positions 0..S-1;
      there is no global accumulation of distance across chunks.
    - The ALiBi bias applied to pre-softmax logits is
          -slope[h] * |q_intra - k_intra|,
      where q_intra is the q's offset inside the current block-M tile
      (0..M-1) and k_intra is the k's offset inside the chunk (0..S-1).

Tests:
    (1) Forward parity: fused kernel vs. PyTorch reference (``hsa_torch_ref``)
        with the same slope tensor.
    (2) Backward parity: DQ / DK / DV / DW via autograd through the
        kernel vs. autograd through the reference.
    (3) Slope == 0 equivalence: the kernel with a zero slope must match
        the kernel with ``slope=None`` (no-ALiBi path).
"""

import os
import sys

import math
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ops.hsa_fwd_bwd_head import (  # noqa: E402
    HSA_block_M_head,
    build_block_indices_block_M,
    hsa_torch_ref,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _abs_err(x, y):
    return (x.float() - y.float()).abs().max().item()


def _rel_err(x, y):
    num = (x.float() - y.float()).square().mean().sqrt().item()
    den = x.float().square().mean().sqrt().item() + 1e-12
    return num / den


def _assert_close(name, ref, got, atol, rtol):
    a = _abs_err(ref, got)
    r = _rel_err(ref, got)
    print(f"  {name:<22s} abs={a:.3e}  rel={r:.3e}  (atol={atol:.1e}, rtol={rtol:.1e})")
    if r > rtol and a > atol:
        raise AssertionError(
            f"{name} mismatch: abs={a:.3e} rel={r:.3e} "
            f"(atol={atol:.1e}, rtol={rtol:.1e})"
        )


def _alibi_slopes(n_heads: int, device, dtype=torch.float32):
    """Standard ALiBi slopes: geometric sequence starting from 2**(-8/n).

    Works for arbitrary n_heads (not only powers of two) by interleaving
    odd-indexed half, matching the reference implementation in the ALiBi
    paper (Press et al., 2021)."""
    def _pow2_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = _pow2_slopes(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = _pow2_slopes(closest_pow2)
        extra = _pow2_slopes(2 * closest_pow2)[0::2][: n_heads - closest_pow2]
        slopes = slopes + extra
    return torch.tensor(slopes, device=device, dtype=dtype)


def _build_case(B, L, HQ, H, D, S, block_size, block_M, device, dtype):
    torch.manual_seed(0xC0FFEE)
    G = HQ // H
    block_indices = build_block_indices_block_M(
        B=B, SEQ_LEN=L, H=H, S=S, block_size=block_size,
        overlap_ratio=0.8, block_M=block_M, device=device, kv_len=L,
    )

    q = torch.randn((B, L, HQ, D), dtype=dtype, device=device).contiguous()
    k = torch.randn((B, L, H, D), dtype=dtype, device=device).contiguous()
    v = torch.randn((B, L, H, D), dtype=dtype, device=device).contiguous()

    logits = torch.randn((B, L, HQ, S), dtype=torch.float32, device=device)
    valid_mask_hq = torch.repeat_interleave(
        (block_indices != -1), repeats=G, dim=2)
    logits = logits.masked_fill(~valid_mask_hq, float("-inf"))
    weights = F.softmax(logits, dim=-1).to(dtype)
    weights = torch.nan_to_num(weights, 0.0).contiguous()

    return q, k, v, weights, block_indices


# ---------------------------------------------------------------------------
# 1. Forward parity: kernel-with-slope vs. torch-ref-with-slope
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "B, L, HQ, H, D, S, block_size, block_M",
    [
        (1, 512,  8, 1,  64, 4, 64,  4),
        (1, 1024, 16, 2, 64, 4, 64,  4),
        (2, 512,  8, 2,  64, 8, 64,  4),
        (1, 512,  8, 1,  64, 4, 64,  1),  # block_M=1 (degenerate tile)
    ],
)
def test_alibi_forward_matches_ref(B, L, HQ, H, D, S, block_size, block_M):
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    q, k, v, w, indices = _build_case(
        B, L, HQ, H, D, S, block_size, block_M, device, dtype,
    )
    slopes = _alibi_slopes(HQ, device)

    print(f"\n[fwd] B={B} L={L} HQ={HQ} H={H} D={D} S={S} "
          f"bs={block_size} M={block_M}")

    o_hsa = HSA_block_M_head(
        q, k, v, w, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=True, slope=slopes,
    )

    o_ref = hsa_torch_ref(
        q.float(), k.float(), v.float(), w.float(), indices,
        chunk_size=block_size, sm_scale=sm_scale, block_q=1,
        mask_last_token=True, slope=slopes.float(),
    )

    _assert_close("fwd o", o_ref, o_hsa, atol=1e-2, rtol=5e-3)


# ---------------------------------------------------------------------------
# 2. Backward parity: DQ / DK / DV / DW with ALiBi
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "B, L, HQ, H, D, S, block_size, block_M",
    [
        (1, 512,  8, 1, 64, 4, 64, 4),
        (1, 1024, 16, 2, 64, 4, 64, 4),
    ],
)
def test_alibi_backward_matches_ref(B, L, HQ, H, D, S, block_size, block_M):
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    q, k, v, w, indices = _build_case(
        B, L, HQ, H, D, S, block_size, block_M, device, dtype,
    )
    slopes = _alibi_slopes(HQ, device)
    do = torch.randn_like(q)

    print(f"\n[bwd] B={B} L={L} HQ={HQ} H={H} D={D} S={S} "
          f"bs={block_size} M={block_M}")

    # --- reference autograd (fp32 through hsa_torch_ref) ---
    q_r = q.float().detach().clone().requires_grad_(True)
    k_r = k.float().detach().clone().requires_grad_(True)
    v_r = v.float().detach().clone().requires_grad_(True)
    w_r = w.float().detach().clone().requires_grad_(True)

    o_ref = hsa_torch_ref(
        q_r, k_r, v_r, w_r, indices,
        chunk_size=block_size, sm_scale=sm_scale, block_q=1,
        mask_last_token=True, slope=slopes.float(),
    )
    o_ref.backward(do.float())

    # --- kernel autograd ---
    q_k = q.detach().clone().requires_grad_(True)
    k_k = k.detach().clone().requires_grad_(True)
    v_k = v.detach().clone().requires_grad_(True)
    w_k = w.detach().clone().requires_grad_(True)

    o_hsa = HSA_block_M_head(
        q_k, k_k, v_k, w_k, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=True, slope=slopes,
    )
    o_hsa.backward(do)

    _assert_close("fwd o",  o_ref, o_hsa, atol=1e-2, rtol=5e-3)
    _assert_close("grad q", q_r.grad, q_k.grad, atol=2e-2, rtol=2e-2)
    _assert_close("grad k", k_r.grad, k_k.grad, atol=2e-2, rtol=2e-2)
    _assert_close("grad v", v_r.grad, v_k.grad, atol=2e-2, rtol=2e-2)
    _assert_close("grad w", w_r.grad, w_k.grad, atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 3. slope=zeros vs. slope=None must match (sanity: enable_alibi path is
#    numerically a no-op when the slope is 0).
# ---------------------------------------------------------------------------

def test_alibi_zero_slope_matches_no_alibi():
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, L, HQ, H, D, S, block_size, block_M = 1, 512, 8, 1, 64, 4, 64, 4
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    q, k, v, w, indices = _build_case(
        B, L, HQ, H, D, S, block_size, block_M, device, dtype,
    )

    print("\n[zero-slope] checking alibi-path with slope=0 == no-alibi path")

    o_no_alibi = HSA_block_M_head(
        q, k, v, w, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=True, slope=None,
    )
    zero_slope = torch.zeros(HQ, device=device, dtype=torch.float32)
    o_zero = HSA_block_M_head(
        q, k, v, w, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=True, slope=zero_slope,
    )
    # Should be bit-wise identical up to kernel compile variations; use a
    # tight tolerance.
    _assert_close("zero-slope fwd", o_no_alibi, o_zero, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. slope > 0 produces a different output than slope == 0 (sanity: the
#    ALiBi term is actually taking effect, not being silently dropped).
# ---------------------------------------------------------------------------

def test_alibi_nonzero_slope_changes_output():
    device = "cuda"
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    B, L, HQ, H, D, S, block_size, block_M = 1, 512, 8, 1, 64, 4, 64, 4
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    q, k, v, w, indices = _build_case(
        B, L, HQ, H, D, S, block_size, block_M, device, dtype,
    )

    o_no_alibi = HSA_block_M_head(
        q, k, v, w, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=True, slope=None,
    )
    strong_slope = torch.full((HQ,), 1.0, device=device, dtype=torch.float32)
    o_strong = HSA_block_M_head(
        q, k, v, w, indices,
        block_size=block_size, sm_scale=sm_scale, block_M=block_M,
        mask_last_token=True, slope=strong_slope,
    )
    diff = (o_no_alibi.float() - o_strong.float()).abs().max().item()
    print(f"\n[nonzero-slope] max|o_no_alibi - o_strong| = {diff:.4e}")
    assert diff > 1e-3, (
        "ALiBi path appears to have no effect: outputs with slope=0 and "
        "slope=1.0 are numerically identical. The ALiBi branch may not be "
        "compiled into the kernel."
    )


if __name__ == "__main__":
    test_alibi_zero_slope_matches_no_alibi()
    test_alibi_nonzero_slope_changes_output()
    test_alibi_forward_matches_ref(1, 512, 8, 1, 64, 4, 64, 4)
    test_alibi_backward_matches_ref(1, 512, 8, 1, 64, 4, 64, 4)
    print("\nAll tests passed ✅")
