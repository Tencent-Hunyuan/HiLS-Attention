"""HoPE: High-frequency-only / partial Rotary Positional Embedding.

Drop-in replacement for ``Qwen3RotaryEmbedding`` with two key properties:

1. **Partial-RoPE via freq mask.**  In the default HoPE mode, keep only the first
   ``hope_partial_ratio * (head_dim // 2)`` highest-frequency freq-pairs;
   zero out the rest of ``inv_freq``.  Because HF rotary pairs dim ``j``
   with dim ``j + head_dim/2`` (both share ``inv_freq[j]``), zeroing
   ``inv_freq[j]`` makes BOTH head_dim components position-invariant.
   Out of ``D = head_dim`` head components:
       #RoPE-active = 2 * n_keep
       #NoPE        = D - 2 * n_keep
   with NoPE dims at ``[n_keep, D/2) ∪ [D/2 + n_keep, D)``.

2. **Frequency layout via standard RoPE formula, with theta picked so
   the LONGEST kept period exactly equals ``L``.**  We mimic the vanilla
   RoPE formula

       inv_freq[j] = theta ** (-j / rope_dim),   j = 0 .. rope_dim - 1

   (so ``inv_freq[0] = 1`` -- highest frequency, period 2*pi tokens)
   and choose ``theta`` such that

       inv_freq[rope_dim - 1] = 2*pi / L
       <=> theta ** ((rope_dim - 1) / rope_dim) = L / (2*pi)
       <=> theta = (L / (2*pi)) ** (rope_dim / (rope_dim - 1))

   where ``rope_dim = n_keep`` and ``L = hope_context_length``.
   This guarantees every kept freq completes at least one full cycle
   inside the training window L (no out-of-domain rotary angles).

Initialization mirrors the project's existing ``Qwen3RotaryEmbedding`` so
that VeOmni's training framework -- which clears tensors / buffers built
in ``__init__`` -- can rebuild the buffer on the first ``forward`` call:
``self.reinit`` flag + buffer construction logic shared between
``__init__`` and the first ``forward``.

Config knobs (all read via ``getattr`` with sensible defaults):
  * ``hope_partial_ratio`` (float in (0, 1], default 0.5)
  * ``hope_context_length`` (int, default ``config.max_position_embeddings``)
  * ``enable_inrange_rope`` / ``rope_context_length`` /
    ``rope_period_multiplier`` for the standard-RoPE mask mode
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from veomni.utils import logging

logger = logging.get_logger(__name__)


class HoPERotaryEmbedding(nn.Module):
    """High-frequency-only partial RoPE.

    Forward signature matches ``Qwen3RotaryEmbedding`` exactly so it can be
    swapped in without touching the attention module:

        cos, sin = self(x, position_ids)
        # cos / sin: same dtype/device as x, shape [batch, L, head_dim]

    The returned ``cos / sin`` are tiled in the ``cat([freqs, freqs], -1)``
    layout that HF's ``rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2])``
    expects -- i.e. dim ``j`` and dim ``j + D/2`` share the same angle.
    """

    def __init__(self, config, device=None, force_inrange_rope=None):
        super().__init__()

        # ---------------- knob plumbing ----------------
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.config = config

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        self.enable_inrange_rope = (
            bool(getattr(config, "enable_inrange_rope", False))
            if force_inrange_rope is None
            else bool(force_inrange_rope)
        )
        self.rope_context_length = int(
            getattr(config, "rope_context_length", config.max_position_embeddings)
        )
        self.rope_period_multiplier = float(
            getattr(config, "rope_period_multiplier", 1.0)
        )

        self.partial_ratio = float(getattr(config, "hope_partial_ratio", 0.5))
        if not (0.0 < self.partial_ratio <= 1.0):
            raise ValueError(
                f"hope_partial_ratio must be in (0, 1], got {self.partial_ratio}"
            )
        self.context_length = int(
            getattr(config, "hope_context_length", config.max_position_embeddings)
        )
        if self.context_length <= 0:
            raise ValueError(
                f"hope_context_length must be positive, got {self.context_length}"
            )

        # n_keep == "rope_dim" in the docstring: number of RoPE-active
        # freq-pairs (equivalently, half of the RoPE-active head_dim count).
        n_keep = max(1, int(round(self.partial_ratio * self.head_dim_half)))
        n_keep = min(n_keep, self.head_dim_half)
        self.n_keep = n_keep
        self.theta = getattr(config, "hope_theta", None)

        # ``attention_scaling`` is preserved for API compatibility with
        # Qwen3RotaryEmbedding (which multiplies cos/sin by it for some
        # rope-scaling variants).  Default HoPE itself does no scaling.
        self.attention_scaling: float = 1.0

        # Mirror Qwen3RotaryEmbedding's "max seq len" attrs so HF helpers
        # like ``dynamic_rope_update`` don't choke on introspection.
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        # Build inv_freq once at __init__ time.  But VeOmni training will
        # later wipe/replace tensors created here, so on the first forward
        # we rebuild the buffer (see ``self.reinit`` below) -- this matches
        # Qwen3RotaryEmbedding's pattern.
        inv_freq = self._build_current_inv_freq(device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq
        self.reinit = False

        if not self.enable_inrange_rope:
            theta = self._compute_theta()
            shortest_period = 2.0 * math.pi
            longest_period = 2.0 * math.pi if n_keep == 1 else self.context_length
            logger.warning_once(
                f"[HoPE] head_dim={self.head_dim}, partial_ratio={self.partial_ratio}, "
                f"context_length={self.context_length}, "
                f"keep {n_keep} / {self.head_dim_half} freq-pairs "
                f"(active head_dims = {2 * n_keep} / {self.head_dim}); "
                f"theta={theta:.4e}, "
                f"shortest period = {shortest_period:.2f} tokens, "
                f"longest period = {longest_period:.2f} "
                f"tokens (target L = {self.context_length})."
            )

    # ------------------------------------------------------------------
    # Frequency construction
    # ------------------------------------------------------------------
    def _build_current_inv_freq(self, device=None) -> torch.Tensor:
        if self.enable_inrange_rope:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            return self._mask_long_period_(inv_freq)

        self.attention_scaling = 1.0
        return self._build_inv_freq(device=device)

    def _mask_long_period_(self, inv_freq: torch.Tensor) -> torch.Tensor:
        """Zero out standard RoPE freq-pairs whose period exceeds the training window."""
        L = self.rope_context_length
        m = self.rope_period_multiplier
        threshold = (2.0 * math.pi * m) / max(L, 1)
        inv_freq = inv_freq.clone()
        keep = inv_freq >= threshold
        if keep.numel() > 0:
            keep[0] = True

        # Meta tensors are used during empty-weight model construction, where
        # scalar reads and boolean-index assignment are illegal. Defer logging
        # until the first real-device rebuild in forward.
        is_meta = inv_freq.is_meta
        if not is_meta and not hasattr(self, "_inrange_rope_logged"):
            n_keep = int(keep.sum().item())
            self._inrange_rope_logged = True
            longest_kept_period = (
                2 * math.pi / inv_freq[keep][-1].item() if n_keep > 0 else 0.0
            )
            logger.warning_once(
                f"[HoPE inrange-RoPE] head_dim_half={self.head_dim_half}, "
                f"L={L}, m={m}, threshold(inv_freq)={threshold:.4e} rad/token "
                f"-> keep {n_keep} / {self.head_dim_half} freq-pairs; "
                f"longest kept period = {longest_kept_period:.1f} tokens."
            )
        return torch.where(keep, inv_freq, torch.zeros_like(inv_freq))

    def _compute_theta(self) -> float:
        """Pick ``theta`` so that ``inv_freq[n_keep - 1] = 2*pi / L``.

        From ``inv_freq[j] = theta^(-j / n_keep)``:
            inv_freq[n_keep - 1] = theta^(-(n_keep - 1) / n_keep)  =  2*pi / L
            <=> theta^((n_keep - 1) / n_keep) = L / (2*pi)
            <=> theta = (L / (2*pi)) ** (n_keep / (n_keep - 1))

        Degenerate case ``n_keep == 1``: only the highest freq is kept,
        period = 2*pi regardless, and the formula above blows up.  Return
        a placeholder theta = L / (2*pi) which is never actually used (the
        single inv_freq entry is hard-coded to 1.0 in ``_build_inv_freq``).
        """
        n = self.n_keep
        L = self.context_length
        if n <= 1:
            return max(L / (2.0 * math.pi), 1.0)
        return (L / (2.0 * math.pi)) ** (n / (n - 1))

    def _build_inv_freq(self, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Build the ``(head_dim // 2,)`` ``inv_freq`` vector.

        Layout::

            inv_freq[0 : n_keep] = theta ** (-arange(n_keep) / n_keep)
            inv_freq[n_keep : ]  = 0   (NoPE tail)

        with ``theta`` chosen so that ``inv_freq[n_keep - 1] = 2*pi / L``.
        """
        n_total = self.head_dim_half
        n_keep = self.n_keep

        inv_freq = torch.zeros(n_total, dtype=dtype, device=device)

        if n_keep == 1:
            # Single-freq degenerate: keep only the top, period 2*pi.
            inv_freq[0] = 1.0
            return inv_freq

        theta = self._compute_theta() if self.theta is None else self.theta
        # inv_freq[j] = theta ** (-j / n_keep) for j = 0..n_keep-1.
        # Build in float32 for numerical stability before casting.
        j = torch.arange(n_keep, dtype=torch.float32, device=device)
        inv_freq[:n_keep] = (theta ** (-j / n_keep)).to(dtype)
        return inv_freq

    # ------------------------------------------------------------------
    # Forward (matches Qwen3RotaryEmbedding signature)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """Compute ``(cos, sin)`` for the given positions.

        Args:
            x: any tensor on the target device/dtype (only ``x.device`` and
               ``x.dtype`` are consulted).
            position_ids: ``[batch, L]`` integer positions.

        Returns:
            ``(cos, sin)``, each ``[batch, L, head_dim]`` in ``x.dtype``.
        """
        # First-call rebuild: VeOmni's training framework wipes buffers
        # created in __init__, so we re-register on the first forward,
        # binding the buffer to the actual training device.  Mirrors the
        # ``self.reinit`` pattern in Qwen3RotaryEmbedding.
        if not self.reinit:
            self.reinit = True
            inv_freq = self._build_current_inv_freq(device=x.device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
