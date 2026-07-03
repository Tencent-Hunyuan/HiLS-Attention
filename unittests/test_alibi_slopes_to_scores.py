"""Bit-exact equivalence test: naive Python loop vs. vectorized closed form.

The vectorized ``alibi_slopes_to_scores`` in
``models/FlashHiLS/lhsa_layer_alibi.py`` is a closed-form rewrite of the
``naive_alibi_slopes_to_scores`` reference loop.  This test verifies the two
agree on a sweep of (L, K, h, sliding_window, chunk_size) configurations.

We re-implement ``naive`` locally here (instead of importing it) to keep the
test free of the layer's heavy ``veomni`` / model-side dependencies.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.FlashHiLS.lhsa_layer_alibi import alibi_slopes_to_scores  # noqa: E402


# ---------------------------------------------------------------------------
# Local naive reference (verbatim copy of the in-tree ``naive_*`` semantics,
# without the debug ``print``).  Kept here so this test does not depend on
# importing the full layer module.
# ---------------------------------------------------------------------------
def _naive_alibi_slopes_to_scores(slopes, scores, sliding_window, chunk_size):
    # scores: (N, L, h, K) ; slopes: (h,)
    alibi_slope = slopes.unsqueeze(0).unsqueeze(-1)  # (1, h, 1)
    delta_slopes = torch.zeros_like(scores)
    K = scores.shape[-1]
    for i in range(scores.shape[1]):
        visible_chunks = (i - sliding_window + 1) // chunk_size
        if visible_chunks > 0:
            delta = i - visible_chunks * chunk_size
            chunk_len = min(visible_chunks, K)
            delta_dis = chunk_len - torch.arange(1, chunk_len + 1, device=slopes.device)
            delta_dis = delta_dis.unsqueeze(0).unsqueeze(0) * chunk_size  # (1, 1, chunk_len)
            delta_slopes[:, i, :, :chunk_len] -= alibi_slope * (delta_dis + delta + 1)
            print(f'i: {i} -> {delta_dis + delta + 1}, {delta_dis}, {delta}')
        else:
            print(f'i; {i} -> no visible chunks')
    return delta_slopes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "L, K, h, sliding_window, chunk_size",
    [
        # Typical training config
        (256,  4,  8,  64, 32),
        (512,  8, 16, 128, 64),
        # Short L where most rows fall in the "no visible chunk" regime
        ( 64,  4,  4,  32, 16),
        # K larger than typical V_i to exercise the visibility mask
        (128, 32,  4,  16,  8),
        # sliding_window == 0 edge case
        (128,  8,  4,   0, 16),
        # chunk_size == 1 edge case (each token is its own chunk)
        ( 64,  8,  2,   8,  1),
        # Single head
        (200,  6,  1,  20, 10),
    ],
)
def test_efficient_matches_naive(L, K, h, sliding_window, chunk_size):
    torch.manual_seed(0)
    N = 3  # batch dim only matters for naive; efficient broadcasts over it
    slopes = torch.rand(h) * 0.1            # positive, ALiBi-like magnitudes
    scores = torch.randn(N, L, h, K)         # actual values irrelevant; we
                                             # only compare the additive bias

    naive = _naive_alibi_slopes_to_scores(slopes, scores, sliding_window, chunk_size)
    eff = alibi_slopes_to_scores(
        slopes, L=L, K=K,
        sliding_window=sliding_window, chunk_size=chunk_size,
        dtype=scores.dtype,
    )

    # eff is (1, L, h, K); broadcast-add to scores must equal naive's per-N copy.
    assert eff.shape == (1, L, h, K), f"unexpected eff shape: {eff.shape}"

    # Bit-exact since both paths do only int arithmetic + slopes-mul, and
    # we cast to the same dtype before the multiply.
    torch.testing.assert_close(
        scores + eff, scores + naive,
        rtol=0.0, atol=0.0,
        msg=f"naive vs efficient mismatch at L={L}, K={K}, h={h}, "
            f"W={sliding_window}, S={chunk_size}",
    )


def test_efficient_dtype_promotion():
    """``dtype`` arg should control output dtype regardless of slopes' dtype."""
    L, K, h, W, S = 128, 8, 4, 32, 16
    slopes = torch.rand(h, dtype=torch.float32) * 0.1

    for out_dtype in (torch.float32, torch.float16, torch.bfloat16):
        eff = alibi_slopes_to_scores(slopes, L, K, W, S, dtype=out_dtype)
        assert eff.dtype == out_dtype, f"expected {out_dtype}, got {eff.dtype}"
        assert eff.shape == (1, L, h, K)


def test_efficient_default_dtype_follows_slopes():
    L, K, h, W, S = 64, 4, 2, 16, 8
    for slope_dtype in (torch.float32, torch.float16, torch.bfloat16):
        slopes = torch.rand(h, dtype=slope_dtype) * 0.1
        eff = alibi_slopes_to_scores(slopes, L, K, W, S)
        assert eff.dtype == slope_dtype


if __name__ == "__main__":
    # Run all parametrized configs sequentially when invoked as a script.
    configs = [
        (256, 4, 8, 64, 32),
        (512, 8, 16, 128, 64),
        (64, 4, 4, 32, 16),
        (128, 32, 4, 16, 8),
        (128, 8, 4, 0, 16),
        (64, 8, 2, 8, 1),
        (200, 6, 1, 20, 10),
    ]
    for cfg in configs:
        test_efficient_matches_naive(*cfg)
        print(f"  ✓ efficient == naive  for L={cfg[0]}, K={cfg[1]}, h={cfg[2]}, "
              f"W={cfg[3]}, S={cfg[4]}")
    test_efficient_dtype_promotion()
    print("  ✓ dtype promotion")
    test_efficient_default_dtype_follows_slopes()
    print("  ✓ default dtype follows slopes")
    print("\nAll tests passed ✅")
