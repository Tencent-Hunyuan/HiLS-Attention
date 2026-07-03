# import unittest
# import torch
# import torch.nn as nn
# from typing import Tuple


# # 标准 HuggingFace RoPE 实现
# def apply_rotary_pos_emb(
#     q: torch.Tensor,
#     k: torch.Tensor,
#     cos: torch.Tensor,
#     sin: torch.Tensor,
#     unsqueeze_dim: int = 1
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """标准 HuggingFace 风格的 RoPE 实现"""
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
    
#     # rotate_half
#     def rotate_half(x):
#         x1 = x[..., : x.shape[-1] // 2]
#         x2 = x[..., x.shape[-1] // 2 :]
#         return torch.cat((-x2, x1), dim=-1)
    
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed

# class TestLigerRope(unittest.TestCase):
        
#     def setUp(self):
#         self.batch_size = 2
#         self.num_heads = 8
#         self.seq_len = 128
#         self.head_dim = 64
#         self.dtype = torch.bfloat16
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#     def _generate_inputs(self):
#         """生成测试输入"""
#         # shape: (batch, num_heads, seq_len, head_dim)
#         query_states = torch.randn(
#             self.batch_size, self.seq_len, self.num_heads, self.head_dim,
#             dtype=self.dtype, device=self.device
#         )
#         key_states = torch.randn(
#             self.batch_size, self.seq_len, self.num_heads, self.head_dim,
#             dtype=self.dtype, device=self.device
#         )
        
#         # cos/sin shape: (seq_len, head_dim)
#         cos = torch.randn(self.seq_len, self.head_dim, dtype=self.dtype, device=self.device)
#         sin = torch.randn(self.seq_len, self.head_dim, dtype=self.dtype, device=self.device)
        
#         return query_states, key_states, cos, sin
    
#     def test_rope_vs_liger_rope(self):
#         """测试标准 RoPE 和 Liger RoPE 的一致性"""
#         try:
#             from liger_kernel.transformers.rope import liger_rotary_pos_emb
#         except ImportError:
#             self.skipTest("Liger Kernel not installed")
        
#         query_states, key_states, cos, sin = self._generate_inputs()
        
#         # 克隆输入（因为 liger 可能是 in-place 操作）
#         q_standard = query_states.clone()
#         k_standard = key_states.clone()
#         q_liger = query_states.clone()
#         k_liger = key_states.clone()
        
#         # 标准实现
#         q_out_standard, k_out_standard = apply_rotary_pos_emb(
#             q_standard, k_standard, cos, sin
#         )
        
#         # Liger 实现
#         q_out_liger, k_out_liger = liger_rotary_pos_emb(
#             q_liger, k_liger, cos, sin
#         )
        
#         # 验证一致性
#         self.assertTrue(
#             torch.allclose(q_out_standard, q_out_liger, rtol=1e-4, atol=1e-4),
#             f"Query 不一致! Max diff: {(q_out_standard - q_out_liger).abs().max()}"
#         )
#         self.assertTrue(
#             torch.allclose(k_out_standard, k_out_liger, rtol=1e-4, atol=1e-4),
#             f"Key 不一致! Max diff: {(k_out_standard - k_out_liger).abs().max()}"
#         )
        
#         print(f"✓ Query max diff: {(q_out_standard - q_out_liger).abs().max():.2e}")
#         print(f"✓ Key max diff: {(k_out_standard - k_out_liger).abs().max():.2e}")

#     def test_rope_vs_liger_rope_half_precision(self):
#         """测试半精度下的一致性 (bfloat16/float16)"""
#         try:
#             from liger_kernel.transformers.rope import liger_rotary_pos_emb
#         except ImportError:
#             self.skipTest("Liger Kernel not installed")
        
#         if not torch.cuda.is_available():
#             self.skipTest("CUDA not available")
        
#         for dtype in [torch.float16, torch.bfloat16]:
#             with self.subTest(dtype=dtype):
#                 self.dtype = dtype
#                 self.device = "cuda"
                
#                 query_states, key_states, cos, sin = self._generate_inputs()
                
#                 q_standard = query_states.clone()
#                 k_standard = key_states.clone()
#                 q_liger = query_states.clone()
#                 k_liger = key_states.clone()
                
#                 q_out_standard, k_out_standard = apply_rotary_pos_emb(
#                     q_standard, k_standard, cos, sin
#                 )
#                 q_out_liger, k_out_liger = liger_rotary_pos_emb(
#                     q_liger, k_liger, cos, sin
#                 )
                
#                 # 半精度允许更大的容差
#                 rtol, atol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-2, 1e-2)
                
#                 self.assertTrue(
#                     torch.allclose(q_out_standard, q_out_liger, rtol=rtol, atol=atol),
#                     f"Query ({dtype}) 不一致! Max diff: {(q_out_standard - q_out_liger).abs().max()}"
#                 )
#                 print(f"✓ {dtype} Query max diff: {(q_out_standard - q_out_liger).abs().max():.2e}")

    # def test_rope_gradient_consistency(self):
    #     """测试反向传播梯度的一致性"""
    #     try:
    #         from liger_kernel.transformers.rope import liger_rotary_pos_emb
    #     except ImportError:
    #         self.skipTest("Liger Kernel not installed")
        
    #     query_states, key_states, cos, sin = self._generate_inputs()
        
    #     # 需要梯度
    #     q_standard = query_states.clone().requires_grad_(True)
    #     k_standard = key_states.clone().requires_grad_(True)
    #     q_liger = query_states.clone().requires_grad_(True)
    #     k_liger = key_states.clone().requires_grad_(True)
        
    #     # Forward
    #     q_out_standard, k_out_standard = apply_rotary_pos_emb(
    #         q_standard, k_standard, cos, sin
    #     )
    #     q_out_liger, k_out_liger = liger_rotary_pos_emb(
    #         q_liger, k_liger, cos, sin
    #     )
        
    #     # Backward
    #     grad_output = torch.randn_like(q_out_standard)
        
    #     (q_out_standard + k_out_standard).sum().backward()
    #     (q_out_liger + k_out_liger).sum().backward()
        
    #     # 验证梯度一致性
    #     self.assertTrue(
    #         torch.allclose(q_standard.grad, q_liger.grad, rtol=1e-4, atol=1e-4),
    #         f"Query 梯度不一致! Max diff: {(q_standard.grad - q_liger.grad).abs().max()}"
    #     )
    #     self.assertTrue(
    #         torch.allclose(k_standard.grad, k_liger.grad, rtol=1e-4, atol=1e-4),
    #         f"Key 梯度不一致! Max diff: {(k_standard.grad - k_liger.grad).abs().max()}"
    #     )
        
    #     print(f"✓ Query grad max diff: {(q_standard.grad - q_liger.grad).abs().max():.2e}")
    #     print(f"✓ Key grad max diff: {(k_standard.grad - k_liger.grad).abs().max():.2e}")

import pytest
import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.transformers.functional import liger_rope
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from liger_kernel.utils import infer_device
from liger_kernel.utils import transformers_version_dispatch

device = infer_device()

SLEEP_SECONDS = 0.1


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        # different q/k heads
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird shapes
        # HuggingFace llama/mistral source code doesn't support odd head dimension
        # so we don't test it here
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(False, reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    dtype,
    expand_position_ids,
    atol,
    rtol,
):
    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).transpose(1, 2).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)

    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)
    # cos = torch.randn(1, seq_len, head_dim, dtype=dtype, device=device)
    # sin = torch.randn(1, seq_len, head_dim, dtype=dtype, device=device)

    # validate forward pass
    hf_q, hf_k = apply_rotary_pos_emb(q1, k1, cos, sin, pos_ids)
    tt_q, tt_k = liger_rotary_pos_emb(q2, k2, cos, sin)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # validate backward pass
    dq, dk = (
        torch.randn_like(hf_q, device=device),
        torch.randn_like(hf_k, device=device).to(dtype),
    )

    q1_grad, k1_grad = torch.autograd.grad((hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 2, 2, 2, 8),
        (1, 2, 1, 2, 8),
        # weird shapes
        (9, 7, 41, 41, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
@pytest.mark.parametrize(
    "expand_position_ids",
    [True, False],
)
def test_functional_correctness(
    bsz,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_dim,
    expand_position_ids,
    dtype,
    atol,
    rtol,
):
    _q = torch.randn((bsz, num_q_heads, seq_len, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, num_kv_heads, seq_len, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)

    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    rotary_emb = transformers_version_dispatch(
        "4.48.0",
        LlamaRotaryEmbedding,
        LlamaRotaryEmbedding,
        before_kwargs={"dim": head_dim, "device": device},
        after_kwargs={"config": LlamaConfig(num_kv_heads=num_kv_heads, head_dim=head_dim), "device": device},
    )

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    if expand_position_ids:
        pos_ids = pos_ids.expand(bsz, -1)
    cos, sin = rotary_emb(k1, pos_ids)

    functional_q, functional_k = liger_rope(q=q1, k=k1, cos=cos, sin=sin)
    class_q, class_k = LigerRopeFunction.apply(q2, k2, cos, sin)

    assert torch.allclose(functional_q, class_q, atol=atol, rtol=rtol)
    assert torch.allclose(functional_k, class_k, atol=atol, rtol=rtol)

    dq, dk = torch.randn_like(functional_q), torch.randn_like(functional_k)

    dq1, dk1 = dq.clone(), dk.clone()
    dq2, dk2 = dq.clone(), dk.clone()

    q1_grad, k1_grad = torch.autograd.grad(
        (functional_q, functional_k),
        (q1, k1),
        (dq1, dk1),
        allow_unused=True,
    )

    q2_grad, k2_grad = torch.autograd.grad(
        (class_q, class_k),
        (q2, k2),
        (dq2, dk2),
        allow_unused=True,
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)
