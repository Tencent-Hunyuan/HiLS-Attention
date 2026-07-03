from __future__ import annotations
from collections import namedtuple
from math import pi
from typing import Optional

import torch
from torch import arange, cat, stack, is_tensor, Tensor
from torch.nn import Module, Parameter

import torch.nn.functional as F

from einops import einsum, rearrange

from torch_einops_utils import slice_right_at_dim

# constants

PolarEmbedReturn = namedtuple('PolarEmbedReturn', ('freqs', 'bias'))

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# applying pope to qk

def apply_pope_to_qk(
    pope: PolarEmbedReturn,
    layer_bias: Optional[Tensor],
    q, k,
    to_magnitude = F.softplus,
    return_complex = False,
    return_nope = False
):
    '''
    q: [B, h, L, d]
    k: [B, h, L, d]
    layer_bias: optional per-layer bias (h, d); if not None, overrides pope.bias.
    '''
    input_dtype = q.dtype
    freqs, bias = pope
    if layer_bias is not None:
        bias = layer_bias

    q_len, k_len, qk_dim, rotate_dim = q.shape[-2], k.shape[-2], q.shape[-1], freqs.shape[-1]

    assert q_len <= k_len and rotate_dim <= qk_dim

    is_partial_rotate = rotate_dim < qk_dim

    if is_partial_rotate:
        q, q_rest = q[..., :rotate_dim], q[..., rotate_dim:]
        k, k_rest = k[..., :rotate_dim], k[..., rotate_dim:]

        if return_complex:
            q_rest = torch.polar(q_rest, torch.zeros_like(q_rest))
            k_rest = torch.polar(k_rest, torch.zeros_like(k_rest))

    if freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    freqs_with_bias = freqs + rearrange(bias, 'h d -> h 1 d')

    # convert q and k to polar magnitudes with activation

    # apply rotations

    freqs = slice_right_at_dim(freqs, q_len, dim = -2)

    device_type = q.device.type if isinstance(q.device.type, str) and q.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        q_nope, k_nope = to_magnitude(q.float()), to_magnitude(k.float())

        if return_complex:
            q = torch.polar(q_nope, freqs_with_bias.float())
        else:
            qcos = freqs_with_bias.float().cos()
            qsin = freqs_with_bias.float().sin()
            q = rearrange([q_nope * qcos, q_nope * qsin], 'two ... d -> ... (d two)')

        if return_complex:
            k = torch.polar(k_nope, freqs.float())
        else:
            kcos = freqs.float().cos()
            ksin = freqs.float().sin()
            k = rearrange([k_nope * kcos, k_nope * ksin], 'two ... d -> ... (d two)')

    q = q.to(input_dtype)
    k = k.to(input_dtype)
    q_nope = q_nope.to(input_dtype)
    k_nope = k_nope.to(input_dtype)

    # concat

    if is_partial_rotate:
        q = cat((q, q_rest), dim = -1)
        k = cat((k, k_rest), dim = -1)

    if return_nope:
        return q, k, cat([q_nope, q_rest], dim=-1), cat([k_nope, k_rest], dim=-1)

    return q, k

def apply_pope_to_q(
    pope: PolarEmbedReturn,
    layer_bias: Optional[Tensor],
    q,
    to_magnitude = F.softplus,
    return_complex = False
):
    '''
    q: [B, h, L, d]
    layer_bias: optional per-layer bias (h, d); if not None, overrides pope.bias.
    '''
    input_dtype = q.dtype
    freqs, bias = pope
    if layer_bias is not None:
        bias = layer_bias

    q_len, qk_dim, rotate_dim = q.shape[-2], q.shape[-1], freqs.shape[-1]

    is_partial_rotate = rotate_dim < qk_dim

    if is_partial_rotate:
        q, q_rest = q[..., :rotate_dim], q[..., rotate_dim:]

        if return_complex:
            q_rest = torch.polar(q_rest, torch.zeros_like(q_rest))

    if freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    freqs_with_bias = freqs + rearrange(bias, 'h d -> h 1 d')

    # convert q and k to polar magnitudes with activation

    # apply rotations

    freqs = slice_right_at_dim(freqs, q_len, dim = -2)

    device_type = q.device.type if isinstance(q.device.type, str) and q.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        q = to_magnitude(q.float())

        if return_complex:
            q = torch.polar(q, freqs_with_bias.float())
        else:
            qcos = freqs_with_bias.float().cos()
            qsin = freqs_with_bias.float().sin()
            q = rearrange([q * qcos, q * qsin], 'two ... d -> ... (d two)')

    q = q.to(input_dtype)

    # concat

    if is_partial_rotate:
        q = cat((q, q_rest), dim = -1)

    return q

# main class

class PoPE(Module):
    apply_pope_to_qk = staticmethod(apply_pope_to_qk)

    def __init__(
        self,
        dim,
        *,
        heads,
        theta = 10000,
        bias_uniform_init = True,
        layer_bias = False,
        bias_learnable = True,
        bias_use_sigmoid = True,
        inv_freqs: Tensor | list[float] | None = None
    ):
        super().__init__()

        # freqs
        self.theta = theta
        self.dim = dim
        if not exists(inv_freqs):
            inv_freqs = theta ** -(arange(dim).float() / dim)

        self.register_buffer('inv_freqs', inv_freqs, persistent=False)

        # the learned bias on the keys
        # `bias_uniform_init`: if True, initialize bias with U(-1, 1); otherwise zeros.
        # `bias_learnable`: if False, freeze bias (requires_grad=False) so it stays constant during training.
        # `bias_use_sigmoid`: if True, map bias via sigmoid(b) * 2pi (constrained to (0, 2pi));
        #                     if False, use raw bias directly as radians (bias=0 means true zero effect).
        # `layer_bias`: if True, PoPE owns no bias (per-layer bias is managed by each attention layer).

        self.bias_use_sigmoid = bias_use_sigmoid

        self.bias = Parameter(torch.zeros(heads, dim)) if not layer_bias else 0
        
        self.reinit = False
        # if bias_uniform_init:
        #     with torch.no_grad():
        #         self.bias.uniform_(-1.0, 1.0)

        if not bias_learnable and self.bias != 0:
            self.bias.requires_grad_(False)

        print(
            f"[PoPE] bias config: layer_bias={layer_bias}, "
            f"shape={tuple(self.bias.shape) if isinstance(self.bias, Parameter) else 'N/A (per-layer)'}, "
            f"uniform_init={bias_uniform_init}, learnable={bias_learnable}, "
            f"use_sigmoid={bias_use_sigmoid}"
        )

    @property
    def device(self):
        return self.inv_freqs.device

    def forward(
        self,
        pos_or_seq_len: Tensor | int,
        offset = 0
    ):
        # get positions depending on input
        # print(f"[PoPE] reinit inv_freqs: {self.inv_freqs}")
        if not self.reinit:
            self.reinit = True
            # NOTE: the buffer is named `inv_freqs` (with an `s`) everywhere
            # else in this class; reinit MUST use the same name to actually
            # overwrite the FSDP-zeroed buffer. Also place it on the current
            # device of this module so downstream einsum does not mismatch.
            device = self.inv_freqs.device
            dtype = self.inv_freqs.dtype
            inv_freqs = (self.theta ** -(arange(self.dim).float() / self.dim)).to(device=device, dtype=dtype)
            self.register_buffer("inv_freqs", inv_freqs, persistent=False)
            

        if is_tensor(pos_or_seq_len):
            pos = pos_or_seq_len
        else:
            seq_len = pos_or_seq_len
            pos = arange(seq_len, device = self.device, dtype = self.inv_freqs.dtype)

        pos = pos + offset

        # freqs - compute in float32 like naive RoPE

        device_type = self.inv_freqs.device.type if isinstance(self.inv_freqs.device.type, str) and self.inv_freqs.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = einsum(pos.float(), self.inv_freqs.float(), '... i, j -> ... i j')

            # the bias: optionally mapped through sigmoid to constrain to (0, 2pi)

            bias = self.bias % (2 * pi)

        return PolarEmbedReturn(freqs, bias)


class PoPERotaryEmbWrapper(Module):
    """Wrapper around PoPE that mimics the Qwen3RotaryEmbedding interface.

    hsa_forward.py calls: position_embeddings = self.rotary_emb(hidden_states, position_ids)
    This wrapper delegates to PoPE.forward(position_ids) and returns PolarEmbedReturn.
    """

    def __init__(self, config, device=None):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        pope_dim = getattr(config, 'pope_dim', head_dim)
        self.pope = PoPE(
            dim=pope_dim,
            heads=config.num_attention_heads,
            theta=getattr(config, 'rope_theta', 10000),
            bias_uniform_init=getattr(config, 'pope_bias_uniform_init', True),
            bias_learnable=getattr(config, 'pope_bias_learnable', True),
            bias_use_sigmoid=getattr(config, 'pope_bias_use_sigmoid', True),
            layer_bias=getattr(config, 'enable_pope_layer_bias', False),
        )

    def forward(self, hidden_states, position_ids):
        """Return PolarEmbedReturn(freqs, bias) compatible with downstream layers."""
        return self.pope(position_ids)
