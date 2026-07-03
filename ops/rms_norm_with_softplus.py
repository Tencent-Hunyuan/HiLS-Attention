"""
Fused RMSNorm + Softplus kernel using TileLang (High-Performance Version).

Forward:  y = softplus(x / rms(x) * weight)
          where rms(x) = sqrt(mean(x^2, dim=-1) + eps)
          softplus(z) = log(1 + exp(z))

          ``weight`` is a per-channel learnable gain (shape ``(D,)``) applied
          *before* the softplus nonlinearity, matching the standard RMSNorm
          semantics (e.g. LLaMA/Olmo q_norm/k_norm).

          Optional partial-softplus:
              If softplus_split_dim=S (0 < S < D) is given, softplus is applied
              ONLY to the first S channels of the last dim; the remaining
              (D - S) channels are a pure passthrough (y = z, i.e. plain
              RMSNorm output).  rms is still computed across the full last
              dim, and ``weight`` is applied across the full last dim.

Backward: dy/dx and dy/dweight via chain rule through softplus and rmsnorm.
          For channels in passthrough region, the softplus derivative becomes
          1 (identity).

Key optimizations:
- Use T.copy for vectorized shared-memory loads (cp.async on Hopper/Ampere)
- Use T.reduce_sum for efficient block-level row reduction
- Large BLOCK_ROWS per block to saturate memory bandwidth
- Pad num_rows to multiple of BLOCK_ROWS to avoid in-kernel branching
- 2D parallelism (rows, dim) for better warp utilization
- split_dim is a compile-time constant, so the per-channel branch is folded
  away during codegen (no runtime overhead).
"""

from typing import Optional

import torch
import torch.nn.functional as F

import tilelang
from tilelang import language as T


# =====================================================================
# Configuration
# =====================================================================

# Block row configurations by dim to balance shared memory usage.
# Total shared mem per block ~= BLOCK_ROWS * dim * 2 bytes (bf16)
def _choose_block_rows(dim):
    """Choose BLOCK_ROWS based on dim (to keep smem footprint reasonable)."""
    if dim <= 64:
        return 64
    elif dim <= 128:
        return 32
    elif dim <= 256:
        return 16
    elif dim <= 512:
        return 8
    else:
        return 4


def _choose_threads(dim, block_rows):
    """Choose threads based on work per block."""
    # Total work = block_rows * dim elements
    # Want each thread to handle 4-8 elements
    total = block_rows * dim
    t = max(128, min(512, total // 4))
    # round down to multiple of 32
    return (t // 32) * 32


# =====================================================================
# Forward Kernel
# =====================================================================

_fwd_kernel_cache = {}  # (dim, block_rows, threads, num_rows_padded, split_dim) -> kernel


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def _rms_norm_softplus_fwd_kernel(num_rows_padded, dim, block_rows, threads, split_dim):
    """
    Fused RMSNorm + (partial) Softplus forward with per-channel weight.

    Inputs:
        X : (num_rows_padded, dim) bf16
        W : (dim,) bf16
    Output:
        Y : (num_rows_padded, dim) bf16  with  Y = softplus_partial(X/rms(X) * W)

    split_dim: compile-time int in [0, dim]. Channels with index d < split_dim
               get softplus; channels with d >= split_dim are passthrough.
               split_dim == dim -> full softplus.
               split_dim == 0   -> plain RMSNorm * weight.

    Assumption: num_rows_padded is a multiple of block_rows.
    Caller must pad the input if necessary.
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    num_blocks = num_rows_padded // block_rows

    @T.prim_func
    def main(
        X: T.Buffer((num_rows_padded, dim), dtype),
        W: T.Buffer((dim,), dtype),
        Y: T.Buffer((num_rows_padded, dim), dtype),
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            X_shared = T.alloc_shared((block_rows, dim), dtype)
            W_shared = T.alloc_shared((dim,), dtype)
            X_frag = T.alloc_fragment((block_rows, dim), accum_dtype)
            W_frag = T.alloc_fragment((dim,), accum_dtype)
            sq_sum = T.alloc_fragment((block_rows,), accum_dtype)
            rms_inv = T.alloc_fragment((block_rows,), accum_dtype)

            row_start = bx * block_rows
            T.copy(X[row_start:row_start + block_rows, :], X_shared)
            T.copy(W, W_shared)

            for i, d in T.Parallel(block_rows, dim):
                X_frag[i, d] = T.cast(X_shared[i, d], accum_dtype)
            for d in T.Parallel(dim):
                W_frag[d] = T.cast(W_shared[d], accum_dtype)

            sq_frag = T.alloc_fragment((block_rows, dim), accum_dtype)
            for i, d in T.Parallel(block_rows, dim):
                sq_frag[i, d] = X_frag[i, d] * X_frag[i, d]
            T.reduce_sum(sq_frag, sq_sum, dim=1, clear=True)

            for i in T.Parallel(block_rows):
                rms_inv[i] = T.rsqrt(sq_sum[i] / T.cast(dim, accum_dtype) + 1e-6)

            Y_frag = T.alloc_fragment((block_rows, dim), accum_dtype)
            for i, d in T.Parallel(block_rows, dim):
                z = X_frag[i, d] * rms_inv[i] * W_frag[d]
                if d < split_dim:
                    Y_frag[i, d] = T.if_then_else(
                        z >= 20.0,
                        z,
                        T.log(1.0 + T.exp(z))
                    )
                else:
                    Y_frag[i, d] = z

            Y_shared = T.alloc_shared((block_rows, dim), dtype)
            for i, d in T.Parallel(block_rows, dim):
                Y_shared[i, d] = T.cast(Y_frag[i, d], dtype)
            T.copy(Y_shared, Y[row_start:row_start + block_rows, :])

    return main


def _get_fwd_kernel(dim, block_rows, threads, num_rows_padded, split_dim):
    key = (dim, block_rows, threads, num_rows_padded, split_dim)
    if key in _fwd_kernel_cache:
        return _fwd_kernel_cache[key]
    k = _rms_norm_softplus_fwd_kernel(num_rows_padded, dim, block_rows, threads, split_dim)
    _fwd_kernel_cache[key] = k
    return k


# =====================================================================
# Backward Kernel
# =====================================================================

_bwd_kernel_cache = {}


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    })
def _rms_norm_softplus_bwd_kernel(num_rows_padded, dim, block_rows, threads, split_dim):
    """
    Fused RMSNorm + (partial) Softplus backward with per-channel weight.

    Forward recap:  z_tilde = x * rms_inv
                    z       = z_tilde * w
                    y[..., :S]  = softplus(z[..., :S])
                    y[..., S:]  = z[..., S:]

    For channels d < S (softplus applied):   dy/dz = sigmoid(z)
    For channels d >= S (passthrough):       dy/dz = 1
    Let h_d = (sigmoid(z_d) if d<S else 1).
    Then:
        g_tilde_d = grad_y_d * h_d * w_d      (gradient wrt z_tilde)
        grad_w_d += grad_y_d * h_d * z_tilde_d       (accumulated over rows)
        grad_x_i  = rms_inv * (g_tilde_i - x_i * dot(g_tilde, x) / (D * rms^2))

    Outputs:
        GradX     : (num_rows_padded, dim)
        GradWPart : (num_blocks, dim)  -- partial per-block sums of grad_w;
                    the caller must reduce across the first dim to get the
                    final grad_w.
    """
    dtype = "bfloat16"
    accum_dtype = "float32"
    num_blocks = num_rows_padded // block_rows

    @T.prim_func
    def main(
        X: T.Buffer((num_rows_padded, dim), dtype),
        GradY: T.Buffer((num_rows_padded, dim), dtype),
        W: T.Buffer((dim,), dtype),
        GradX: T.Buffer((num_rows_padded, dim), dtype),
        GradWPart: T.Buffer((num_blocks, dim), accum_dtype),
    ):
        with T.Kernel(num_blocks, threads=threads) as bx:
            X_shared = T.alloc_shared((block_rows, dim), dtype)
            GY_shared = T.alloc_shared((block_rows, dim), dtype)
            GX_shared = T.alloc_shared((block_rows, dim), dtype)
            W_shared = T.alloc_shared((dim,), dtype)

            X_f = T.alloc_fragment((block_rows, dim), accum_dtype)
            G_f = T.alloc_fragment((block_rows, dim), accum_dtype)
            H_f = T.alloc_fragment((block_rows, dim), accum_dtype)  # h_d (sigmoid or 1)
            W_f = T.alloc_fragment((dim,), accum_dtype)

            sq_sum = T.alloc_fragment((block_rows,), accum_dtype)
            dot_gx = T.alloc_fragment((block_rows,), accum_dtype)
            rms_inv = T.alloc_fragment((block_rows,), accum_dtype)
            coeff = T.alloc_fragment((block_rows,), accum_dtype)

            tmp_frag = T.alloc_fragment((block_rows, dim), accum_dtype)
            gw_frag = T.alloc_fragment((block_rows, dim), accum_dtype)
            gw_sum = T.alloc_fragment((dim,), accum_dtype)
            out_frag = T.alloc_fragment((block_rows, dim), accum_dtype)

            row_start = bx * block_rows

            T.copy(X[row_start:row_start + block_rows, :], X_shared)
            T.copy(GradY[row_start:row_start + block_rows, :], GY_shared)
            T.copy(W, W_shared)

            for i, d in T.Parallel(block_rows, dim):
                X_f[i, d] = T.cast(X_shared[i, d], accum_dtype)
                G_f[i, d] = T.cast(GY_shared[i, d], accum_dtype)
            for d in T.Parallel(dim):
                W_f[d] = T.cast(W_shared[d], accum_dtype)

            for i, d in T.Parallel(block_rows, dim):
                tmp_frag[i, d] = X_f[i, d] * X_f[i, d]
            T.reduce_sum(tmp_frag, sq_sum, dim=1, clear=True)

            for i in T.Parallel(block_rows):
                rms_inv[i] = T.rsqrt(sq_sum[i] / T.cast(dim, accum_dtype) + 1e-6)

            # h_d = sigmoid(z_d)  for d < S   (z uses full weight)
            #     = 1              for d >= S
            # Numerically stable sigmoid (symmetric to the fwd kernel):
            #   z >= 0:  1 / (1 + exp(-z))     -- exp arg in (-inf, 0]
            #   z <  0:  exp(z) / (1 + exp(z)) -- exp arg in (-inf, 0)
            # Avoids exp(-z) overflow (|z| > ~88.7 in fp32) that could yield
            # NaN under TL_ENABLE_FAST_MATH, which would poison grad_w/grad_x.
            for i, d in T.Parallel(block_rows, dim):
                if d < split_dim:
                    z = X_f[i, d] * rms_inv[i] * W_f[d]
                    H_f[i, d] = T.if_then_else(
                        z >= 0.0,
                        1.0 / (1.0 + T.exp(-z)),
                        T.exp(z) / (1.0 + T.exp(z)),
                    )
                else:
                    H_f[i, d] = 1.0

            # grad_w partial = grad_y * h * z_tilde   (z_tilde = x * rms_inv)
            for i, d in T.Parallel(block_rows, dim):
                gw_frag[i, d] = G_f[i, d] * H_f[i, d] * (X_f[i, d] * rms_inv[i])
            T.reduce_sum(gw_frag, gw_sum, dim=0, clear=True)
            for d in T.Parallel(dim):
                GradWPart[bx, d] = gw_sum[d]

            # g_tilde = grad_y * h * w     (gradient wrt z_tilde)
            for i, d in T.Parallel(block_rows, dim):
                G_f[i, d] = G_f[i, d] * H_f[i, d] * W_f[d]

            # dot(g_tilde, x)
            for i, d in T.Parallel(block_rows, dim):
                tmp_frag[i, d] = G_f[i, d] * X_f[i, d]
            T.reduce_sum(tmp_frag, dot_gx, dim=1, clear=True)

            for i in T.Parallel(block_rows):
                coeff[i] = dot_gx[i] * rms_inv[i] * rms_inv[i] / T.cast(dim, accum_dtype)

            # grad_x_i = rms_inv * (g_tilde_i - x_i * coeff)
            for i, d in T.Parallel(block_rows, dim):
                out_frag[i, d] = rms_inv[i] * (G_f[i, d] - X_f[i, d] * coeff[i])

            for i, d in T.Parallel(block_rows, dim):
                GX_shared[i, d] = T.cast(out_frag[i, d], dtype)
            T.copy(GX_shared, GradX[row_start:row_start + block_rows, :])

    return main


def _get_bwd_kernel(dim, block_rows, threads, num_rows_padded, split_dim):
    key = (dim, block_rows, threads, num_rows_padded, split_dim)
    if key in _bwd_kernel_cache:
        return _bwd_kernel_cache[key]
    k = _rms_norm_softplus_bwd_kernel(num_rows_padded, dim, block_rows, threads, split_dim)
    _bwd_kernel_cache[key] = k
    return k


# =====================================================================
# Helper: pad rows to multiple of BLOCK_ROWS
# =====================================================================

def _pad_rows(x_2d, block_rows):
    num_rows, dim = x_2d.shape
    pad = (block_rows - num_rows % block_rows) % block_rows
    if pad == 0:
        return x_2d, num_rows
    padded = torch.empty((num_rows + pad, dim), dtype=x_2d.dtype, device=x_2d.device)
    padded[:num_rows].copy_(x_2d)
    padded[num_rows:].zero_()
    return padded, num_rows


# =====================================================================
# Autograd Function
# =====================================================================

class RMSNormSoftplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, split_dim: int) -> torch.Tensor:
        orig_shape = x.shape
        dim = orig_shape[-1]
        x_2d = x.reshape(-1, dim).contiguous()

        assert weight is not None, "weight is required"
        assert weight.shape == (dim,), f"weight must be shape ({dim},), got {tuple(weight.shape)}"
        if weight.dtype != x_2d.dtype:
            weight = weight.to(x_2d.dtype)
        weight = weight.contiguous()

        block_rows = _choose_block_rows(dim)
        threads = _choose_threads(dim, block_rows)

        x_padded, real_rows = _pad_rows(x_2d, block_rows)
        num_rows_padded = x_padded.shape[0]

        kernel = _get_fwd_kernel(dim, block_rows, threads, num_rows_padded, split_dim)
        y_padded = kernel(x_padded, weight)
        y_2d = y_padded[:real_rows]

        ctx.save_for_backward(x_2d, weight)
        ctx.orig_shape = orig_shape
        ctx.block_rows = block_rows
        ctx.threads = threads
        ctx.split_dim = split_dim
        return y_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_2d, weight = ctx.saved_tensors
        orig_shape = ctx.orig_shape
        block_rows = ctx.block_rows
        threads = ctx.threads
        split_dim = ctx.split_dim
        dim = orig_shape[-1]

        grad_2d = grad_output.reshape(-1, dim).contiguous()

        x_padded, real_rows = _pad_rows(x_2d, block_rows)
        g_padded, _ = _pad_rows(grad_2d, block_rows)
        num_rows_padded = x_padded.shape[0]
        num_blocks = num_rows_padded // block_rows

        kernel = _get_bwd_kernel(dim, block_rows, threads, num_rows_padded, split_dim)

        grad_x_padded = torch.empty_like(x_padded)
        grad_w_partial = torch.empty(
            (num_blocks, dim), dtype=torch.float32, device=x_padded.device
        )
        kernel(x_padded, g_padded, weight, grad_x_padded, grad_w_partial)
        grad_w = grad_w_partial.sum(dim=0).to(weight.dtype)

        grad_x_2d = grad_x_padded[:real_rows]
        # Returns must match forward signature: (x, weight, split_dim)
        return grad_x_2d.reshape(orig_shape), grad_w, None


def rms_norm_with_softplus(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    softplus_split_dim: Optional[int] = None,
) -> torch.Tensor:
    """Fused RMSNorm + (optionally partial) Softplus with per-channel gain.

    Computes ``y = softplus_partial(x / rms(x) * weight)``.

    Args:
        x: Input tensor of any shape. RMSNorm is applied over the last
           dimension across the FULL dim (independent of ``softplus_split_dim``).
        weight: Required 1-D tensor of shape ``(dim,)`` (standard RMSNorm gain).
        eps: Epsilon for numerical stability (hardcoded to 1e-6 in kernel).
        softplus_split_dim:
            - ``None`` (default): softplus is applied to every channel of the
              last dim.
            - int in ``[0, dim]``: softplus is applied ONLY to the first
              ``softplus_split_dim`` channels of the last dim; the remaining
              channels are passthrough (plain RMSNorm output).
              Typical usage: ``softplus_split_dim = dim // 2``.

    Returns:
        Tensor of the same shape as ``x`` with fused RMSNorm + partial softplus
        applied on the last dim.
    """
    orig_shape = x.shape
    dim = orig_shape[-1]

    if softplus_split_dim is None:
        split_dim = dim
    else:
        if not (0 <= softplus_split_dim <= dim):
            raise ValueError(
                f"softplus_split_dim must be in [0, {dim}], got {softplus_split_dim}"
            )
        split_dim = int(softplus_split_dim)

    assert weight is not None, "weight is required"
    assert weight.shape == (dim,), f"weight must be shape ({dim},), got {tuple(weight.shape)}"

    # Autograd path (need bwd to track both x and weight gradients)
    if x.requires_grad or weight.requires_grad:
        return RMSNormSoftplusFunction.apply(x, weight, split_dim)

    x_2d = x.reshape(-1, dim).contiguous()

    block_rows = _choose_block_rows(dim)
    threads = _choose_threads(dim, block_rows)

    x_padded, real_rows = _pad_rows(x_2d, block_rows)
    num_rows_padded = x_padded.shape[0]

    kernel = _get_fwd_kernel(dim, block_rows, threads, num_rows_padded, split_dim)
    w_contig = weight if weight.dtype == x_2d.dtype else weight.to(x_2d.dtype)
    w_contig = w_contig.contiguous()
    y_padded = kernel(x_padded, w_contig)
    return y_padded[:real_rows].reshape(orig_shape)


# =====================================================================
# Reference implementation for testing
# =====================================================================

def rms_norm_with_softplus_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    softplus_split_dim: Optional[int] = None,
) -> torch.Tensor:
    """Reference (pure PyTorch) implementation for correctness verification.

    Semantics:  y = softplus_partial(x / rms(x) * weight)
    """
    dim = x.shape[-1]
    if softplus_split_dim is None:
        split_dim = dim
    else:
        if not (0 <= softplus_split_dim <= dim):
            raise ValueError(
                f"softplus_split_dim must be in [0, {dim}], got {softplus_split_dim}"
            )
        split_dim = int(softplus_split_dim)

    assert weight is not None, "weight is required"
    assert weight.shape == (dim,), f"weight must be shape ({dim},), got {tuple(weight.shape)}"

    rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    normed = x.float() * rms * weight.float()  # (..., D)

    if split_dim == dim:
        return F.softplus(normed).to(x.dtype)
    if split_dim == 0:
        return normed.to(x.dtype)

    left = F.softplus(normed[..., :split_dim])
    right = normed[..., split_dim:]
    return torch.cat([left, right], dim=-1).to(x.dtype)
