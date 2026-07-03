#!/usr/bin/env python
"""Unit test: verify that the new modeling_olmo_lhsa produces outputs
whose distribution is close (in KL-divergence) to the legacy
modeling_olmo_lhsa, given the same weights and inputs.

Usage:
    # Step 1 – generate reference data in the legacy repo:
    cd /data/workspace/shanwxxxhu_gz/InfiniteLongLM_legacy
    python generate_test_data.py --output_path legacy_test_data.pt

    # Step 2 – run this test in the new repo:
    cd /data/workspace/shanwxxxhu_gz/InfiniteLongLM
    pytest unittests/test_legacy_compatibility.py -v -s

Environment variables:
    LEGACY_DATA_PATH: path to legacy_test_data.pt (default: ../InfiniteLongLM_legacy/legacy_test_data.pt)
    KL_THRESHOLD: KL divergence threshold (default: 1e-3)
"""

import os
import sys
import json
import pytest
import torch
import torch.nn.functional as F

# Ensure repo root is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Disable liger / flash-attn kernels to match the legacy eager-only execution
os.environ["USE_LIGER_KERNEL"] = "1"
os.environ["USE_LIGER_RMSNORM"] = "0"
os.environ["USE_LIGER_ROPE"] = "1"
os.environ["USE_LIGER_SWIGLU"] = "1"
os.environ["USE_FLASH_ATTN_RMSNORM"] = "1"

from models.FlashHiLS.configuration_hsa import HSAConfig
from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM


# ---------------------------------------------------------------------------
# Default paths & thresholds
# ---------------------------------------------------------------------------
DEFAULT_LEGACY_DATA_PATH = os.path.join(
    os.path.dirname(_REPO_ROOT),
    "InfiniteLongLM_legacy",
    "unittests/legacy_test_data.pt",
)
DEFAULT_KL_THRESHOLD = 1e-3  # KL divergence threshold (per-token average)
DEFAULT_GRAD_COS_THRESHOLD = 0.99  # Cosine similarity threshold for gradient comparison
DEFAULT_GRAD_RTOL = 1e-2  # Relative tolerance for gradient comparison
DEFAULT_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def compute_kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """Compute KL(P || Q) where P = legacy, Q = new model.

    Args:
        logits_p: [B, L, V] raw logits from legacy model (reference)
        logits_q: [B, L, V] raw logits from new model

    Returns:
        Scalar tensor: mean KL divergence per token.
    """
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    # KL(P || Q) = sum_v P(v) * (log P(v) - log Q(v))
    kl = (p * (log_p - log_q)).sum(dim=-1)  # [B, L]
    return kl.mean()


def load_legacy_data(path: str):
    """Load the .pt file produced by generate_test_data.py."""
    assert os.path.isfile(path), (
        f"Legacy test data not found at {path}.\n"
        f"Please run `python generate_test_data.py` in the legacy repo first."
    )
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def build_new_model(config_dict: dict, state_dict: dict, device: str, dtype: torch.dtype,
                    train_mode: bool = True):
    """Build the new HiLSForCausalLM and load legacy weights."""
    # Use eager attention to match legacy
    config = HSAConfig(**config_dict)
    model = HiLSForCausalLM(config)

    # Load state dict (may need to handle minor key mismatches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading legacy weights: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading legacy weights: {unexpected}")

    model = model.to(device=device, dtype=dtype)
    if train_mode:
        model.train()
    else:
        model.eval()
    return model


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
class TestLegacyCompatibility:
    """Test that new model output distribution matches legacy model."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Load legacy data once for all tests."""
        legacy_path = os.environ.get("LEGACY_DATA_PATH", DEFAULT_LEGACY_DATA_PATH)
        print('start loading data')
        self.data = load_legacy_data(legacy_path)
        print('loading over')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.kl_threshold = float(os.environ.get("KL_THRESHOLD", DEFAULT_KL_THRESHOLD))
        self.grad_cos_threshold = float(os.environ.get("GRAD_COS_THRESHOLD", DEFAULT_GRAD_COS_THRESHOLD))
        self.grad_rtol = float(os.environ.get("GRAD_RTOL", DEFAULT_GRAD_RTOL))

    def test_kl_divergence_below_threshold(self):
        """Main test: KL(legacy || new) < threshold for all samples."""
        config_dict = self.data["config_dict"]
        state_dict = self.data["model_state_dict"]
        input_ids = self.data["input_ids"]       # [N, L]
        legacy_logits = self.data["logits"]       # [N, L_out, V]
        num_samples = input_ids.shape[0]

        print(f"\n[INFO] Loaded {num_samples} samples from legacy data")
        print(f"[INFO] input_ids shape: {input_ids.shape}")
        print(f"[INFO] legacy_logits shape: {legacy_logits.shape}")
        print(f"[INFO] KL threshold: {self.kl_threshold}")

        # Build new model with legacy weights
        model = build_new_model(config_dict, state_dict, self.device, self.dtype)

        # Run forward pass on new model
        all_new_logits = []
        batch_size = DEFAULT_BATCH_SIZE
        vocab_size = config_dict["vocab_size"]

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_ids = input_ids[i : i + batch_size].to(self.device)
                outputs = model(input_ids=batch_ids, use_cache=False)
                logits = outputs.logits[:, :, :vocab_size].float().cpu()
                all_new_logits.append(logits)

        new_logits = torch.cat(all_new_logits, dim=0)  # [N, L_out, V]

        # Ensure shapes match
        min_len = min(legacy_logits.shape[1], new_logits.shape[1])
        legacy_logits_trimmed = legacy_logits[:, :min_len, :]
        new_logits_trimmed = new_logits[:, :min_len, :]

        assert legacy_logits_trimmed.shape == new_logits_trimmed.shape, (
            f"Shape mismatch: legacy={legacy_logits_trimmed.shape}, new={new_logits_trimmed.shape}"
        )

        # Compute KL divergence
        kl_div = compute_kl_divergence(legacy_logits_trimmed, new_logits_trimmed)
        print(f"[RESULT] Mean KL divergence: {kl_div.item():.6e}")

        assert kl_div.item() < self.kl_threshold, (
            f"KL divergence {kl_div.item():.6e} exceeds threshold {self.kl_threshold:.6e}"
        )

    def test_per_sample_kl_divergence(self):
        """Check per-sample KL divergence to identify outliers."""
        config_dict = self.data["config_dict"]
        state_dict = self.data["model_state_dict"]
        input_ids = self.data["input_ids"]
        legacy_logits = self.data["logits"]
        num_samples = input_ids.shape[0]
        vocab_size = config_dict["vocab_size"]

        model = build_new_model(config_dict, state_dict, self.device, self.dtype)

        all_new_logits = []
        batch_size = DEFAULT_BATCH_SIZE

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                batch_ids = input_ids[i : i + batch_size].to(self.device)
                outputs = model(input_ids=batch_ids, use_cache=False)
                logits = outputs.logits[:, :, :vocab_size].float().cpu()
                all_new_logits.append(logits)

        new_logits = torch.cat(all_new_logits, dim=0)

        min_len = min(legacy_logits.shape[1], new_logits.shape[1])
        legacy_logits_trimmed = legacy_logits[:, :min_len, :]
        new_logits_trimmed = new_logits[:, :min_len, :]

        # Per-sample KL
        log_p = F.log_softmax(legacy_logits_trimmed.float(), dim=-1)
        log_q = F.log_softmax(new_logits_trimmed.float(), dim=-1)
        p = log_p.exp()
        kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # [N, L]
        kl_per_sample = kl_per_token.mean(dim=-1)          # [N]

        max_kl = kl_per_sample.max().item()
        mean_kl = kl_per_sample.mean().item()
        num_exceed = (kl_per_sample > self.kl_threshold).sum().item()

        print(f"\n[RESULT] Per-sample KL stats:")
        print(f"  Mean:    {mean_kl:.6e}")
        print(f"  Max:     {max_kl:.6e}")
        print(f"  #Exceed: {num_exceed}/{num_samples} (threshold={self.kl_threshold:.6e})")

        # Allow at most 5% of samples to exceed the threshold
        max_exceed_ratio = 0.05
        exceed_ratio = num_exceed / num_samples
        assert exceed_ratio <= max_exceed_ratio, (
            f"{num_exceed}/{num_samples} ({exceed_ratio:.1%}) samples exceed KL threshold "
            f"{self.kl_threshold:.6e}, max allowed is {max_exceed_ratio:.0%}"
        )

    def test_logits_max_abs_diff(self):
        """Sanity check: maximum absolute difference in logits."""
        config_dict = self.data["config_dict"]
        state_dict = self.data["model_state_dict"]
        input_ids = self.data["input_ids"]
        legacy_logits = self.data["logits"]
        vocab_size = config_dict["vocab_size"]

        model = build_new_model(config_dict, state_dict, self.device, self.dtype)

        # Just test first 8 samples for speed
        n = min(8, input_ids.shape[0])
        batch_ids = input_ids[:n].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids=batch_ids, use_cache=False)
            new_logits = outputs.logits[:, :, :vocab_size].float().cpu()

        min_len = min(legacy_logits.shape[1], new_logits.shape[1])
        diff = (legacy_logits[:n, :min_len, :] - new_logits[:, :min_len, :]).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n[RESULT] Logits absolute difference (first {n} samples):")
        print(f"  Max:  {max_diff:.6e}")
        print(f"  Mean: {mean_diff:.6e}")

        # For bfloat16 models, max abs diff should be very small if implementations match
        # Use a generous threshold since numerical differences are expected
        abs_threshold = 1e-2
        assert max_diff < abs_threshold, (
            f"Max absolute logits diff {max_diff:.6e} exceeds threshold {abs_threshold:.6e}"
        )

    def test_backward_loss_match(self):
        """Verify that the loss from backward pass matches between legacy and new model."""
        config_dict = self.data["config_dict"]
        state_dict = self.data["model_state_dict"]
        grad_input_ids = self.data["grad_input_ids"]  # [grad_samples, L]
        grad_labels = self.data["grad_labels"]        # [grad_samples, L]
        legacy_loss = self.data["grad_loss"]           # scalar

        print(f"\n[INFO] Backward loss verification")
        print(f"[INFO] Legacy loss: {legacy_loss:.6f}")

        # Build new model in train mode
        model = build_new_model(config_dict, state_dict, self.device, self.dtype,
                                train_mode=True)
        model.zero_grad()

        # Forward + backward
        batch_ids = grad_input_ids.to(self.device)
        batch_labels = grad_labels.to(self.device)
        outputs = model(input_ids=batch_ids, labels=batch_labels, use_cache=False)
        new_loss = outputs.loss

        print(f"[INFO] New loss:    {new_loss.item():.6f}")
        loss_diff = abs(legacy_loss - new_loss.item())
        print(f"[RESULT] Loss absolute diff: {loss_diff:.6e}")

        assert loss_diff < 1e-3, (
            f"Loss mismatch: legacy={legacy_loss:.6f}, new={new_loss.item():.6f}, "
            f"diff={loss_diff:.6e}"
        )

    def test_backward_gradient_cosine_similarity(self):
        """Verify that parameter gradients are close via cosine similarity."""
        config_dict = self.data["config_dict"]
        state_dict = self.data["model_state_dict"]
        grad_input_ids = self.data["grad_input_ids"]
        grad_labels = self.data["grad_labels"]
        legacy_grads = self.data["param_grads"]  # dict: name -> grad tensor

        print(f"\n[INFO] Backward gradient cosine similarity verification")
        print(f"[INFO] Legacy has gradients for {len(legacy_grads)} parameters")
        print(f"[INFO] Cosine similarity threshold: {self.grad_cos_threshold}")

        # Build new model in train mode
        model = build_new_model(config_dict, state_dict, self.device, self.dtype,
                                train_mode=True)
        model.zero_grad()

        # Forward + backward
        batch_ids = grad_input_ids.to(self.device)
        batch_labels = grad_labels.to(self.device)
        outputs = model(input_ids=batch_ids, labels=batch_labels, use_cache=False)
        outputs.loss.backward()

        # Collect new gradients
        new_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                new_grads[name] = param.grad.float().cpu().clone()

        print(f"[INFO] New model has gradients for {len(new_grads)} parameters")

        # Compare gradients
        common_keys = set(legacy_grads.keys()) & set(new_grads.keys())
        print(f"[INFO] Comparing {len(common_keys)} common parameters")

        failed_params = []
        cos_sims = []
        for name in sorted(common_keys):
            lg = legacy_grads[name].flatten()
            ng = new_grads[name].flatten()
            if lg.shape != ng.shape:
                failed_params.append((name, f"shape mismatch: {lg.shape} vs {ng.shape}"))
                continue
            # Skip zero gradients
            if lg.norm() < 1e-12 and ng.norm() < 1e-12:
                cos_sims.append((name, 1.0))
                continue
            cos_sim = F.cosine_similarity(lg.unsqueeze(0), ng.unsqueeze(0)).item()
            cos_sims.append((name, cos_sim))
            if cos_sim < self.grad_cos_threshold:
                failed_params.append((name, f"cosine_sim={cos_sim:.6f}"))

        # Print summary
        all_cos = [c for _, c in cos_sims]
        if all_cos:
            print(f"\n[RESULT] Gradient cosine similarity stats:")
            print(f"  Min:  {min(all_cos):.6f}")
            print(f"  Mean: {sum(all_cos)/len(all_cos):.6f}")
            print(f"  Max:  {max(all_cos):.6f}")

        if failed_params:
            print(f"\n[RESULT] Failed parameters ({len(failed_params)}/{len(common_keys)}):")
            for name, reason in failed_params[:20]:  # Show at most 20
                print(f"  {name}: {reason}")

        assert len(failed_params) == 0, (
            f"{len(failed_params)}/{len(common_keys)} parameters failed gradient cosine "
            f"similarity check (threshold={self.grad_cos_threshold})"
        )

    def test_backward_gradient_relative_error(self):
        """Verify that parameter gradients are close via relative error."""
        config_dict = self.data["config_dict"]
        state_dict = self.data["model_state_dict"]
        grad_input_ids = self.data["grad_input_ids"]
        grad_labels = self.data["grad_labels"]
        legacy_grads = self.data["param_grads"]

        print(f"\n[INFO] Backward gradient relative error verification")
        print(f"[INFO] Relative tolerance: {self.grad_rtol}")

        # Build new model in train mode
        model = build_new_model(config_dict, state_dict, self.device, self.dtype,
                                train_mode=True)
        model.zero_grad()

        # Forward + backward
        batch_ids = grad_input_ids.to(self.device)
        batch_labels = grad_labels.to(self.device)
        outputs = model(input_ids=batch_ids, labels=batch_labels, use_cache=False)
        outputs.loss.backward()

        # Collect new gradients
        new_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                new_grads[name] = param.grad.float().cpu().clone()

        common_keys = set(legacy_grads.keys()) & set(new_grads.keys())

        failed_params = []
        rel_errors = []
        for name in sorted(common_keys):
            lg = legacy_grads[name]
            ng = new_grads[name]
            if lg.shape != ng.shape:
                failed_params.append((name, f"shape mismatch: {lg.shape} vs {ng.shape}"))
                continue
            # Relative error: ||lg - ng|| / max(||lg||, ||ng||, eps)
            diff_norm = (lg - ng).norm().item()
            ref_norm = max(lg.norm().item(), ng.norm().item(), 1e-12)
            rel_err = diff_norm / ref_norm
            rel_errors.append((name, rel_err))
            if rel_err > self.grad_rtol:
                failed_params.append((name, f"rel_error={rel_err:.6e}"))

        # Print summary
        all_rel = [r for _, r in rel_errors]
        if all_rel:
            print(f"\n[RESULT] Gradient relative error stats:")
            print(f"  Min:  {min(all_rel):.6e}")
            print(f"  Mean: {sum(all_rel)/len(all_rel):.6e}")
            print(f"  Max:  {max(all_rel):.6e}")

        if failed_params:
            print(f"\n[RESULT] Failed parameters ({len(failed_params)}/{len(common_keys)}):")
            for name, reason in failed_params[:20]:
                print(f"  {name}: {reason}")

        assert len(failed_params) == 0, (
            f"{len(failed_params)}/{len(common_keys)} parameters failed gradient relative "
            f"error check (rtol={self.grad_rtol})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
