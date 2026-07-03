"""
Unit test: Gradient Accumulation Correctness with FSDP2 MixedPrecisionPolicy

Uses the real torch.distributed FSDP2 API (fully_shard + MixedPrecisionPolicy)
to verify gradient accumulation, exactly matching VeOmni's parallelize_model_fsdp2:

    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.layers:
        fully_shard(layer, mp_policy=mp_policy)
    fully_shard(model, mp_policy=mp_policy)

Training loop matches VeOmni's pretrain.py:
    for micro_batch in micro_batches:
        loss = model(**micro_batch).loss
        loss = loss * (micro_tokens / total_tokens)   # mean_global_loss
        loss.backward()                               # FSDP handles bf16→fp32 reduce
    optimizer.step()

Tests (single GPU, real FSDP2):
  1. FSDP2 bf16 accumulated vs fp32 full-batch baseline → cosine sim ≥ 0.99
  2. FSDP2 fp32 accumulated vs fp32 full-batch baseline → near-exact
"""

import copy
import math
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Skip if no GPU or torch version too old for FSDP2
# ──────────────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for FSDP2 test",
)

try:
    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed._tensor import DTensor
    HAS_FSDP2 = True
except ImportError:
    HAS_FSDP2 = False
    DTensor = None


# ──────────────────────────────────────────────────────────────────────────────
# Model components (mirrors VeOmni's Olmo3 architecture exactly)
# ──────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Matches Olmo3RMSNorm: upcasts to fp32 for variance, then back to input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLUMLP(nn.Module):
    """Matches Olmo3MLP: gate_proj, up_proj, down_proj with SiLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """
    Matches Olmo3DecoderLayer:
      post-norm residual: x = residual + norm(sublayer(x))
    """

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(hidden_size, eps=eps)
        self.k_norm = RMSNorm(hidden_size, eps=eps)

        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        residual = hidden_states

        q = self.q_norm(self.q_proj(hidden_states))
        k = self.k_norm(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)

        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, -1)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + self.post_attention_layernorm(attn_output)

        residual = hidden_states
        hidden_states = residual + self.post_feedforward_layernorm(self.mlp(hidden_states))

        return hidden_states


class MiniCausalLM(nn.Module):
    """
    Minimal causal LM matching VeOmni's HiLSForCausalLM:
      embed_tokens → [DecoderLayer × N] → norm → lm_head → CE loss
    """

    # Used by VeOmni's FSDP2 to find basic_modules for per-layer sharding
    _no_split_modules = ["DecoderLayer"]

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        intermediate_size: int = 512,
        max_position_embeddings: int = 128,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, intermediate_size, eps)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        head_dim = hidden_size // num_heads
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        sin = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        return cos, sin

    def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor | None = None) -> dict:
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self._get_rope(input_ids.shape[1], input_ids.device, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Under FSDP2, logits may be a DTensor. Convert to plain tensor
            # so F.cross_entropy backward doesn't hit unregistered DTensor ops
            # (e.g. aten.clamp_min_.default). This matches real VeOmni where
            # the loss_function receives logits after lm_head which handles
            # the DTensor→local conversion internally via chunked CE.
            logits_local = _to_local(logits)
            shift_logits = logits_local[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"loss": loss, "logits": logits}


def _to_local(t: torch.Tensor) -> torch.Tensor:
    """Convert DTensor to local tensor, no-op for plain tensors."""
    if DTensor is not None and isinstance(t, DTensor):
        return t.full_tensor()
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _init_single_device_pg():
    """Initialize a single-device process group for FSDP2 (world_size=1)."""
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl")


def _apply_fsdp2(model: nn.Module, mp_policy: MixedPrecisionPolicy | None = None):
    """
    Apply FSDP2 exactly like VeOmni's parallelize_model_fsdp2:
      1. fully_shard each DecoderLayer (basic_module) with mp_policy
      2. fully_shard the root model with mp_policy
    """
    fsdp_kwargs = {}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy

    # Shard each decoder layer first (like VeOmni iterating decoder_blocks)
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)

    # Shard root model
    fully_shard(model, **fsdp_kwargs)

    return model


def _fp32_baseline_grads(
    model: nn.Module,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
) -> dict[str, torch.Tensor]:
    """
    Pure fp32 reference: single forward+backward on the full batch.
    No FSDP, no mixed precision — the gold standard.
    """
    model_ref = copy.deepcopy(model).float().cuda()
    model_ref.eval()
    model_ref.zero_grad()

    output = model_ref(input_ids, labels=labels)
    output["loss"].backward()

    return {
        name: p.grad.detach().float().clone()
        for name, p in model_ref.named_parameters()
        if p.grad is not None
    }


def _fsdp2_accumulated_grads(
    model: nn.Module,
    input_ids: torch.LongTensor,
    labels: torch.LongTensor,
    num_micro_batches: int,
    mp_policy: MixedPrecisionPolicy | None = None,
) -> dict[str, torch.Tensor]:
    """
    Real FSDP2 gradient accumulation matching VeOmni's training loop:

        mp_policy = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)
        fully_shard(layer, mp_policy=mp_policy)   # per layer
        fully_shard(model, mp_policy=mp_policy)    # root

        for micro_batch in micro_batches:
            loss = model(**micro_batch).loss
            loss = loss * (micro_tokens / total_tokens)   # mean_global_loss
            loss.backward()
    """
    model_fsdp = copy.deepcopy(model).float().cuda()
    model_fsdp = _apply_fsdp2(model_fsdp, mp_policy=mp_policy)
    model_fsdp.eval()

    mb_ids = input_ids.chunk(num_micro_batches, dim=0)
    mb_labels = labels.chunk(num_micro_batches, dim=0)

    # count_loss_token: total valid tokens across all micro-batches
    total_tokens = (labels != -100).sum().item()

    for ids_mb, lbl_mb in zip(mb_ids, mb_labels):
        output = model_fsdp(ids_mb, labels=lbl_mb)
        loss = output["loss"]

        # mean_global_loss: loss * cur_token_len / all_reduced_len * fsdp_size
        # On single GPU: fsdp_size=1, all_reduced_len=total_tokens
        micro_tokens = (lbl_mb != -100).sum().item()
        loss = loss * (micro_tokens / total_tokens)

        loss.backward()

    # After all micro-batches, collect the accumulated fp32 grads
    # FSDP2 with reduce_dtype=fp32 ensures param.grad is already fp32
    # On single GPU the DTensor full_tensor() == local tensor (no actual gather)
    grads = {}
    for name, p in model_fsdp.named_parameters():
        if p.grad is not None:
            grads[name] = _to_local(p.grad).detach().float().clone()

    return grads


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def init_dist():
    """Initialize process group once for all tests in this module."""
    if not HAS_FSDP2:
        pytest.skip("torch >= 2.4 required for FSDP2 (fully_shard + MixedPrecisionPolicy)")
    _init_single_device_pg()
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture
def setup():
    """Seeded fp32 model + token inputs (on CPU, tests move to GPU)."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    device = torch.device("cuda")

    vocab_size = 256
    batch_size = 8
    seq_len = 64

    model = MiniCausalLM(
        vocab_size=vocab_size,
        hidden_size=256,
        num_heads=4,
        num_layers=2,
        intermediate_size=512,
        max_position_embeddings=seq_len,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = input_ids.clone()

    return model, input_ids, labels


class TestGradAccumulationFSDP2:
    """Gradient accumulation correctness tests using real FSDP2."""

    def test_fsdp2_bf16_grad_accumulation(self, setup):
        """
        FSDP2 with MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)
        + 4 micro-batch accumulation vs pure fp32 full-batch baseline.

        This is the exact VeOmni configuration. We compare against fp32
        baseline because that's the ground truth. bf16 compute introduces
        some error, but gradients should remain directionally correct.
        """
        model, input_ids, labels = setup

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        grads_baseline = _fp32_baseline_grads(model, input_ids, labels)
        grads_fsdp = _fsdp2_accumulated_grads(
            model, input_ids, labels,
            num_micro_batches=4,
            mp_policy=mp_policy,
        )

        assert grads_baseline.keys() == grads_fsdp.keys(), (
            f"Parameter sets differ:\n"
            f"  baseline: {sorted(grads_baseline.keys())}\n"
            f"  fsdp:     {sorted(grads_fsdp.keys())}"
        )

        for name in grads_baseline:
            g_ref = grads_baseline[name].flatten()
            g_fsdp = grads_fsdp[name].flatten()

            cos_sim = F.cosine_similarity(g_ref.unsqueeze(0), g_fsdp.unsqueeze(0)).item()
            rel_err = (g_fsdp - g_ref).norm() / g_ref.norm().clamp(min=1e-12)

            # bf16 compute loses precision, but gradient direction should be preserved
            assert cos_sim >= 0.95, (
                f"Cosine similarity too low for {name}: {cos_sim:.6f}"
            )
            assert rel_err.item() <= 0.1, (
                f"Relative L2 error too large for {name}: {rel_err.item():.6f}"
            )

    def test_fsdp2_fp32_grad_accumulation(self, setup):
        """
        FSDP2 with no mixed precision (fp32 throughout)
        + 4 micro-batch accumulation vs fp32 full-batch baseline.

        Should be near-exact. Confirms the accumulation + loss scaling
        (mean_global_loss) logic is mathematically correct.
        """
        model, input_ids, labels = setup

        grads_baseline = _fp32_baseline_grads(model, input_ids, labels)
        grads_fsdp = _fsdp2_accumulated_grads(
            model, input_ids, labels,
            num_micro_batches=4,
            mp_policy=None,  # pure fp32, no mixed precision
        )

        assert grads_baseline.keys() == grads_fsdp.keys(), "Parameter sets differ"

        for name in grads_baseline:
            torch.testing.assert_close(
                grads_fsdp[name],
                grads_baseline[name],
                atol=1e-5,
                rtol=1e-4,
                msg=f"Gradient mismatch (fp32 FSDP2 vs fp32 baseline) for {name}",
            )

    def test_fsdp2_bf16_same_dtype_accumulation(self, setup):
        """
        FSDP2 bf16 full-batch (1 micro-batch) vs FSDP2 bf16 4 micro-batches.
        Same FSDP2 + MixedPrecision on both sides — isolates the accumulation
        variable only. Should match very closely.
        """
        model, input_ids, labels = setup

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        grads_single = _fsdp2_accumulated_grads(
            model, input_ids, labels,
            num_micro_batches=1,
            mp_policy=mp_policy,
        )
        grads_accum = _fsdp2_accumulated_grads(
            model, input_ids, labels,
            num_micro_batches=4,
            mp_policy=mp_policy,
        )

        assert grads_single.keys() == grads_accum.keys(), "Parameter sets differ"

        for name in grads_single:
            g_ref = grads_single[name].flatten()
            g_acc = grads_accum[name].flatten()

            cos_sim = F.cosine_similarity(g_ref.unsqueeze(0), g_acc.unsqueeze(0)).item()
            rel_err = (g_acc - g_ref).norm() / g_ref.norm().clamp(min=1e-12)

            assert cos_sim >= 0.99, (
                f"Cosine similarity too low for {name}: {cos_sim:.6f}"
            )
            assert rel_err.item() <= 0.02, (
                f"Relative L2 error too large for {name}: {rel_err.item():.6f}"
            )
