import pytest
import torch

from models.FlashHiLS.modeling_olmo_hils import Olmo3FlashAttnRMSNorm, Olmo3TorchRMSNorm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for RMSNorm kernel tests")


FWD_TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.bfloat16: (2e-2, 2e-2),
}

BWD_TOLERANCES = {
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (3e-2, 3e-2),
}


def _make_module(backend: str, hidden_size: int, eps: float, device: str) -> torch.nn.Module:
    if backend == "baseline":
        return Olmo3TorchRMSNorm(hidden_size, eps=eps).to(device=device)
    if backend == "flash_attn":
        pytest.importorskip("flash_attn.ops.triton.layer_norm")
        return Olmo3FlashAttnRMSNorm(hidden_size, eps=eps).to(device=device)
    if backend == "liger":
        liger_module = pytest.importorskip("liger_kernel.transformers.rms_norm")
        return liger_module.LigerRMSNorm(hidden_size, eps=eps).to(device=device)
    raise ValueError(f"Unknown backend: {backend}")


def _assert_close(name: str, ref: torch.Tensor, actual: torch.Tensor, atol: float, rtol: float) -> None:
    ref_fp32 = ref.float()
    actual_fp32 = actual.float()
    max_diff = (ref_fp32 - actual_fp32).abs().max().item()
    assert torch.allclose(ref_fp32, actual_fp32, atol=atol, rtol=rtol), (
        f"{name} mismatch: max_diff={max_diff:.6e}, atol={atol}, rtol={rtol}"
    )


def _make_input(dtype: torch.dtype, device: str, batch_size: int = 3, seq_len: int = 17, hidden_size: int = 4096):
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    if dtype == torch.bfloat16 and not bf16_supported:
        pytest.skip("bfloat16 is not supported on this GPU")
    return torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)


@pytest.mark.parametrize("backend", ["flash_attn", "liger"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_forward_matches_torch_baseline(backend: str, dtype: torch.dtype):
    torch.manual_seed(0)
    device = "cuda"
    eps = 1e-6
    hidden_size = 4096
    x = _make_input(dtype=dtype, device=device, hidden_size=hidden_size)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    baseline = _make_module("baseline", hidden_size=hidden_size, eps=eps, device=device)
    backend_module = _make_module(backend, hidden_size=hidden_size, eps=eps, device=device)

    baseline.weight.data.copy_(weight)
    backend_module.weight.data.copy_(weight)

    ref = baseline(x)
    actual = backend_module(x)

    atol, rtol = FWD_TOLERANCES[dtype]
    _assert_close(f"{backend} forward", ref, actual, atol=atol, rtol=rtol)


@pytest.mark.parametrize("backend", ["flash_attn", "liger"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_backward_matches_torch_baseline(backend: str, dtype: torch.dtype):
    torch.manual_seed(1)
    device = "cuda"
    eps = 1e-6
    hidden_size = 4096
    x = _make_input(dtype=dtype, device=device, hidden_size=hidden_size)
    grad_out = torch.randn_like(x)
    weight = torch.randn(hidden_size, device=device, dtype=torch.float32)

    baseline = _make_module("baseline", hidden_size=hidden_size, eps=eps, device=device)
    backend_module = _make_module(backend, hidden_size=hidden_size, eps=eps, device=device)

    baseline.weight.data.copy_(weight)
    backend_module.weight.data.copy_(weight)

    x_ref = x.detach().clone().requires_grad_(True)
    x_backend = x.detach().clone().requires_grad_(True)

    out_ref = baseline(x_ref)
    out_backend = backend_module(x_backend)
    out_ref.backward(grad_out)
    out_backend.backward(grad_out)

    atol, rtol = BWD_TOLERANCES[dtype]
    _assert_close(f"{backend} backward output", out_ref, out_backend, atol=atol, rtol=rtol)
    _assert_close(f"{backend} grad_input", x_ref.grad, x_backend.grad, atol=atol, rtol=rtol)
    if backend == "liger":
        pytest.xfail("Known liger_kernel RMSNorm grad_weight mismatch; not fixing kernel here.")
    _assert_close(
        f"{backend} grad_weight",
        baseline.weight.grad,
        backend_module.weight.grad,
        atol=atol,
        rtol=rtol,
    )
