from typing import Any, Callable, Optional, Tuple, Union
from math import pi
import torch.nn as nn
from einops import rearrange
from veomni.utils import logging
from .pope import apply_pope_to_qk, apply_pope_to_q
import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from ops.flex_attn_tilelang import flex_attn_tl


logger = logging.get_logger(__name__)


def create_chunk_dropout_mask(
    B: int,
    L: int,
    chunk_size: int,
    hsa_sliding_window: int,
    p: float,
    device: torch.device
) -> torch.Tensor:
    num_chunks = L // chunk_size

    # ----------------------------------------------------------------
    # Step 1: generate random scores; set invisible positions to -inf
    # ----------------------------------------------------------------
    rand = torch.rand(B, L, num_chunks, device=device)

    positions   = torch.arange(L, device=device)
    chunk_ids   = torch.arange(num_chunks, device=device)
    chunk_offset = (positions - hsa_sliding_window + 1) // chunk_size                       # (L,)
    available   = chunk_ids[None, :] < chunk_offset[:, None]   # (L, num_chunks)

    rand = rand.masked_fill(~available.unsqueeze(0), float('-inf'))

    # ----------------------------------------------------------------
    # Step 2: sort in descending order
    # ----------------------------------------------------------------
    indices = rand.argsort(dim=-1, descending=True)            # (B, L, num_chunks)

    # ----------------------------------------------------------------
    # Step 3: sample cnt from a binomial distribution
    # cnt ~ Binomial(num_available, p), then clamp(min=1)
    # if no visible chunks, keep cnt as 0
    # ----------------------------------------------------------------
    num_available = available.sum(dim=-1).float()              # (L,)

    # torch.binomial requires count and prob to have the same shape
    prob = torch.full_like(num_available, p)                   # (L,)
    cnt_base = torch.binomial(num_available, prob).long()      # (L,)  ~ Binomial(n_avail, p)

    # mask at least 1 (only for positions with available chunks)
    cnt = torch.where(
        num_available.long() > 0,
        cnt_base.clamp(min=1),
        cnt_base                    # keep 0 when no visible chunks
    )                                                          # (L,)
    
    cnt = cnt.unsqueeze(0).expand(B, -1)                       # (B, L)

    # ----------------------------------------------------------------
    # Step 4: rank_mask: top cnt ranks are True
    # ----------------------------------------------------------------
    rank     = torch.arange(num_chunks, device=device)         # (num_chunks,)
    rank_mask = rank[None, None, :] < cnt.unsqueeze(-1)        # (B, L, num_chunks)

    # ----------------------------------------------------------------
    # Step 5: scatter back to the chunk dimension
    # ----------------------------------------------------------------
    dropout_mask = torch.zeros(B, L, num_chunks, dtype=torch.bool, device=device)
    dropout_mask.scatter_(dim=2, index=indices, src=rank_mask)

    return dropout_mask.to(torch.int32)

# RoPE helper functions removed; PoPE is used instead via apply_pope_to_qk

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def ensure_blhdk_strides(hidden_states: torch.Tensor) -> torch.Tensor:
    """TileLang kernels require canonical strides even when L == 1."""
    if hidden_states.ndim != 4:
        return hidden_states.contiguous()

    _, L, H, D = hidden_states.shape
    expected_stride = (L * H * D, H * D, D, 1)
    if hidden_states.stride() == expected_stride:
        return hidden_states

    out = torch.empty_strided(
        hidden_states.shape,
        expected_stride,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    out.copy_(hidden_states)
    return out

def apply_pope_to_k(
    pope,
    k,
    to_magnitude=F.softplus,
):
    """
    Apply the K-side PoPE rotation only.

    k: [B, h, L, d]
    """
    input_dtype = k.dtype
    freqs, _ = pope

    k_len, qk_dim, rotate_dim = k.shape[-2], k.shape[-1], freqs.shape[-1]
    assert rotate_dim <= qk_dim

    is_partial_rotate = rotate_dim < qk_dim
    if is_partial_rotate:
        k, k_rest = k[..., :rotate_dim], k[..., rotate_dim:]

    if freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    assert freqs.shape[-2] == k_len, (
        f"PoPE K freqs length ({freqs.shape[-2]}) must match K length ({k_len})"
    )

    device_type = k.device.type if isinstance(k.device.type, str) and k.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        if to_magnitude is not None:
            k = to_magnitude(k.float())

        kcos = freqs.float().cos()
        ksin = freqs.float().sin()
        k = rearrange([k * kcos, k * ksin], 'two ... d -> ... (d two)')

    k = k.to(input_dtype)

    if is_partial_rotate:
        k = torch.cat((k, k_rest), dim=-1)

    return k

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LandmarkHSA(nn.Module):
    def __init__(self, config, layer_idx, norm_cls=None):
        super().__init__()
        print(f"[LandmarkHSA] using lhsa_layer_pope_naive path (layer_idx={layer_idx})")
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.head_dim = self.d_model // config.num_attention_heads
        self.d_kv = self.head_dim * config.num_key_value_heads
        self.h_kv = config.num_key_value_heads
        self.h_q = config.num_attention_heads

        assert self.h_q % 4 == 0, "num_attention_heads must be divisible by 4"
        self.hsa_heads = getattr(config, "hsa_heads", self.h_q // 4)
        self.hsa_qk_ratio = getattr(config, "hsa_qk_ratio", 4)
        assert self.hsa_heads % self.hsa_qk_ratio == 0, "hsa_heads must be divisible by hsa_qk_ratio"
        assert self.h_q % self.hsa_heads == 0, "num_attention_heads must be divisible by hsa_heads"
        self.hsa_denom = self.h_q // self.hsa_heads

        assert self.h_kv % self.hsa_denom == 0, "num_key_value_heads must be divisible by hsa_denom"

        self.h_hsa_kv = self.hsa_heads // self.hsa_qk_ratio
        self.unified_retrieval = getattr(config, "unified_retrieval", False)
        self.retrieval_head_num = 1 if self.unified_retrieval else getattr(config, "retrieval_head_num", self.h_hsa_kv)

        self.hsa_q_proj = nn.Linear(self.d_model, self.d_model // self.hsa_denom, bias=False)
        self.hsa_k_proj = nn.Linear(self.d_model, self.h_hsa_kv * self.head_dim, bias=False)
        self.hsa_v_proj = nn.Linear(self.d_model, self.h_hsa_kv * self.head_dim, bias=False)

        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.enable_lmk_q_proj = getattr(config, "enable_lmk_q_proj", False)
        if self.enable_lmk_q_proj:
            self.retrieval_dim = getattr(config, 'retrieval_dim', self.retrieval_head_num * self.head_dim)
            self.lmk_q_proj = nn.Linear(self.d_model, self.retrieval_dim, bias=False)
            self.lmk_q_norm = norm_cls(self.retrieval_dim // self.retrieval_head_num)

        if not self.enable_lmk_q_proj:
            logger.warning_once("Recommend to set enable_lmk_q_proj=True for better performance")

        self.topk = config.hsa_topk
        
        self.chunk_size = config.chunk_size
        self.layerwise_qk_norm = getattr(config, "layerwise_qk_norm", False)
        if not self.layerwise_qk_norm:
            self.q_norm = norm_cls(self.head_dim)
            self.k_norm = norm_cls(self.head_dim)
        else:
            self.q_norm = norm_cls(self.d_model)
            self.k_norm = norm_cls(self.h_hsa_kv * self.head_dim)

        self.scale_lmk_k = getattr(config, "scale_lmk_k", False)
        if self.scale_lmk_k:
            self.lmk_k_scale = nn.Parameter(torch.ones(1))

        self.hsa_visible_window = getattr(config, "hsa_visible_window", -1)
        if getattr(config, 'full_upper_hsa', False) and self.layer_idx >= config.num_hidden_layers // 2:
            self.hsa_visible_window = -1
        if self.hsa_visible_window != -1:
            self.topk = min(self.hsa_visible_window // self.chunk_size, self.topk)
            self.hsa_visible_window += config.sliding_window
        self.scaling = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window
        self.hsa_sliding_window = getattr(config, "hsa_sliding_window", self.sliding_window)
        self.enable_hsa_swa = getattr(config, "enable_hsa_swa", True)
        self.nope_chunkwise_attn = getattr(config, "nope_chunkwise_attn", False)
        
        self.is_causal = True
        
        hsa_mode = config.hsa_mode
        from ops.hsa_fwd_bwd_head import HSA_block_M_head as HSA
        from ops.topk_head_softmax import online_softmax_topk_head as topk_func
        self.topk_func = topk_func
        self.hsa_func = HSA
        self.hsa_mode = hsa_mode

        self.enable_softmax1 = config.enable_softmax1
        self.hsa_dropout_prob = getattr(config, "hsa_dropout_prob", 0.0)
        self.hsa_disturb_prob = getattr(config, "hsa_disturb_prob", 0.0)

        self.enable_pope_layer_bias = getattr(config, "enable_pope_layer_bias", False)
        if self.enable_pope_layer_bias:
            pope_dim = getattr(config, "pope_dim", self.head_dim)
            self.pope_bias = nn.Parameter(torch.zeros(self.h_q, pope_dim))
            # Optional U(0, 2*pi) random init for per-layer bias.
            # Controlled by config.pope_layer_bias_uniform_init (default False for
            # backward compatibility: old checkpoints / configs keep zeros).
            if getattr(config, "pope_layer_bias_uniform_init", False):
                with torch.no_grad():
                    self.pope_bias.uniform_(0.0, 2 * pi)
        else:
            self.pope_bias = None

    def forward(self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=False,
        cache_position=None,
        pope_pos_embeddings=None,
        chunk_pos_embeddings=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        B, L, _ = hidden_states.shape

        # --- HSA heads (lower part) ---
        hsa_q = self.hsa_q_proj(hidden_states)
        if self.layerwise_qk_norm:
            hsa_q = self.q_norm(hsa_q)
        hsa_q = rearrange(hsa_q, 'B L (h d)->B L h d', d=self.head_dim)
    
        if not self.layerwise_qk_norm:
            hsa_q_norm_nope = self.q_norm(hsa_q)  # (B, L, h, d)
        else:
            hsa_q_norm_nope = hsa_q

        if self.enable_lmk_q_proj:
            lmk_q = self.lmk_q_proj(hidden_states)
            lmk_q = rearrange(lmk_q, 'B L (h d)->B L h d', h=self.retrieval_head_num)
            lmk_q_norm = self.lmk_q_norm(lmk_q)  # (B, L, r_head_num, d)
        else:
            assert not self.unified_retrieval, "Unified_retrieval is not supported for w/o lmk q proj"
            lmk_q_norm = hsa_q_norm_nope

        hsa_k = self.hsa_k_proj(hidden_states)
        if self.layerwise_qk_norm:
            hsa_k = self.k_norm(hsa_k)
        hsa_k = rearrange(hsa_k, 'B L (h d)->B L h d', d=self.head_dim)

        if not self.layerwise_qk_norm:
            hsa_k_norm_nope = self.k_norm(hsa_k)
        else:
            hsa_k_norm_nope = hsa_k

        hsa_v = self.hsa_v_proj(hidden_states)
        hsa_v = rearrange(hsa_v, 'B L (h d)->B L h d', d=self.head_dim)

        # Apply PoPE to HSA q/k (always enabled)
        hsa_q_norm_pope, hsa_k_norm_pope, hsa_q_norm_nope, hsa_k_norm_nope = apply_pope_to_qk(
            pope_pos_embeddings,
            self.pope_bias,
            hsa_q_norm_nope.transpose(1, 2),
            hsa_k_norm_nope.transpose(1, 2),
            return_nope=True
        )
        hsa_q_norm_pope = hsa_q_norm_pope.transpose(1, 2).contiguous()
        hsa_k_norm_pope = hsa_k_norm_pope.transpose(1, 2).contiguous()
        hsa_q_norm_nope = hsa_q_norm_nope.transpose(1, 2).contiguous()
        hsa_k_norm_nope = hsa_k_norm_nope.transpose(1, 2).contiguous()
        if not self.enable_lmk_q_proj:
            lmk_q_norm = hsa_q_norm_pope
        else:
            lmk_q_norm = apply_pope_to_q(pope_pos_embeddings, self.pope_bias, lmk_q_norm.transpose(1, 2)).transpose(1, 2).contiguous()
            # (B, L, r_head_num, d)

        # Inference: HSA KV cache.  For nope_chunkwise_attn, the HSA block
        # consumes noPE K, so keep the cache in noPE form and reconstruct the
        # PoPE K view only for TopK/SWA.
        if use_cache and past_key_values is not None:
            hsa_cache_idx = self.layer_idx + self.config.num_hidden_layers
            cache_kwargs = {"cache_position": cache_position}
            if self.nope_chunkwise_attn:
                hsa_k_norm_nope, hsa_v = past_key_values.update(
                    hsa_k_norm_nope, hsa_v, hsa_cache_idx, cache_kwargs
                )
                pope_cache_pos_embeddings = kwargs.get("pope_cache_pos_embeddings", None)
                if pope_cache_pos_embeddings is None:
                    raise RuntimeError(
                        "pope_cache_pos_embeddings is required when using "
                        "PoPE naive + nope_chunkwise_attn + KV cache."
                    )
                if pope_cache_pos_embeddings.freqs.shape[-2] != hsa_k_norm_nope.shape[1]:
                    raise RuntimeError(
                        "PoPE cache freqs length must match HSA K cache length: "
                        f"freqs_len={pope_cache_pos_embeddings.freqs.shape[-2]}, "
                        f"k_len={hsa_k_norm_nope.shape[1]}, "
                        f"layer={self.layer_idx}, cache_position={cache_position}"
                    )
                hsa_k_norm_pope = apply_pope_to_k(
                    pope_cache_pos_embeddings,
                    hsa_k_norm_nope.transpose(1, 2),
                    to_magnitude=None if not self.layerwise_qk_norm else F.softplus,
                ).transpose(1, 2).contiguous()
            else:
                hsa_k_norm_pope, hsa_v = past_key_values.update(
                    hsa_k_norm_pope, hsa_v, hsa_cache_idx, cache_kwargs
                )

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        doc_ids = kwargs.get("doc_ids", None)
        if cu_seq_lens_q is not None and cu_seq_lens_q.shape[0] > 2:
            assert hidden_states.shape[0] == 1, f'cu_seq_lens_q is only supported for batch size 1, but got {hidden_states.shape[0]}'
            assert torch.all((cu_seq_lens_q[:-1] % self.chunk_size) ==0), f'cu_seq_lens_q must be divisible by chunk_size, cu_seq_lens_q: {cu_seq_lens_q}'
        else:
            cu_seq_lens_q = None

        if hidden_states.shape[0] > 1:
            assert doc_ids is None, f'doc_ids is not supported for batch size > 1, but got doc ids {doc_ids}'

        kernel_is_training = past_key_values is None

        # apply pope for hsa part
        if self.enable_hsa_swa:
            swa_o, lse_sum = flex_attn_tl(
                hsa_q_norm_pope.transpose(1, 2), # (B, H, L, d)
                hsa_k_norm_pope.transpose(1, 2), 
                hsa_v.transpose(1, 2), 
                window_size=self.hsa_sliding_window,
                chunk_size=self.chunk_size,
                training=kernel_is_training,
                mask_lmk=True,
                expand_to_chunk=True
            )
            # lse_sum = lse_sum.transpose(1, 2)  # (B, L, h_q)
            lse_sum = lse_sum.contiguous()  # (B, L, hG)
            lse_sum = lse_sum.to(hidden_states.dtype)  # (B, L, h_kv)

        full_seq_len = hsa_k_norm_pope.shape[1]
        if full_seq_len >= self.chunk_size:
            lmk_k: Any = hsa_k_norm_pope[:, self.chunk_size - 1::self.chunk_size, : ,:]  # (B, L // S, hsa_kv, d)
            if self.scale_lmk_k:
                lmk_k = lmk_k * self.lmk_k_scale
            if self.unified_retrieval:
                lmk_k = rearrange(lmk_k, 'B S H D -> B S 1 (H D)')

            B, S,  H, D = lmk_k.shape
            lmk_k = lmk_k.reshape(B, S, H, D)
            # q_offset = int(cache_position[0].item()) if (use_cache and cache_position is not None) else 0
            # 上面的q_offset计算方法依赖外部的GenerateState维护cache_seq_len并正确构造cache_position且传入，容易出错
            q_offset = full_seq_len - L if (use_cache and past_key_values is not None) else 0

            drop_mask = None
            if self.training and self.hsa_dropout_prob > 0.0:
                drop_mask = create_chunk_dropout_mask(
                    B, L, self.chunk_size, self.hsa_sliding_window, self.hsa_dropout_prob, device=hidden_states.device
                )
                if self.hsa_disturb_prob > 0.0:
                    gate = (torch.rand(B, L, 1, device=lmk_q_norm.device) < self.hsa_disturb_prob).to(torch.int32)
                    drop_mask = drop_mask * gate

            assert cu_seq_lens_q is None, "cu_seq_lens_q is not supported for headwise topk"
            # print(f'lse_sum: {lse_sum.shape}')
            # print(f'lmk_q_norm: {lmk_q_norm.shape}, lmk_k: {lmk_k.shape}, lse_sum: {lse_sum.shape}, topk: {self.topk}, chunk_size: {self.chunk_size}, hsa_sliding_window: {self.hsa_sliding_window}, hsa_visible_window: {hsa_visible_window}, q_offset: {q_offset}')
            indices, scores = self.topk_func(
                lmk_q_norm,
                lmk_k,
                lse_sum,
                self.topk,
                block_size=self.chunk_size,
                window_size=self.hsa_sliding_window,
                is_causal=True,
                q_offset=q_offset,
                is_training=kernel_is_training,
                drop_mask=drop_mask,
            )

            if self.enable_hsa_swa:
                if not self.enable_softmax1:
                    cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1)], dim=-1)  # (B, L, h_kv, K + 1)
                    swa_weight_idx = -1
                else:
                    # softmax off by one
                    cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1), torch.zeros(B, L, scores.shape[2], 1, device=hidden_states.device)], dim=-1)
                    swa_weight_idx = -2
                chunk_weights = F.softmax(cat_scores, dim=-1).to(hidden_states.dtype) # (B, L, h_kv, K)
            else:
                valid_row = torch.isfinite(scores).any(dim=-1, keepdim=True)  # (B, L, h, 1)
                safe_scores = scores.masked_fill(~valid_row, 0.0)
                chunk_weights = F.softmax(safe_scores, dim=-1, dtype=torch.float32).to(hidden_states.dtype)  # (B, L, h_kv, K)
                chunk_weights = chunk_weights * valid_row.to(chunk_weights.dtype)

            if self.unified_retrieval:
                chunk_weights = chunk_weights.repeat_interleave(self.h_hsa_kv // self.retrieval_head_num, dim=2).contiguous()
                indices = indices.repeat_interleave(self.h_hsa_kv // self.retrieval_head_num, dim=2).contiguous()

            # Pick which q/k feed into chunk attention:
            #   nope_chunkwise_attn / chunk_pos_embeddings  -> nope branch
            #   otherwise                                   -> pope branch
            # NOTE: chunk_pos_embeddings may be a list [None, None] when
            # enable_intra_chunk_pos=False (see hsa_forward.py), so we must
            # check that its entries are actual tensors.
            has_intra_pos = (
                chunk_pos_embeddings is not None
                and chunk_pos_embeddings[0] is not None
                and chunk_pos_embeddings[1] is not None
            )
            if self.nope_chunkwise_attn or has_intra_pos:
                hsa_q_norm = hsa_q_norm_nope.contiguous()
                hsa_k_norm = hsa_k_norm_nope.contiguous()
            else:
                hsa_q_norm = hsa_q_norm_pope.contiguous()
                hsa_k_norm = hsa_k_norm_pope.contiguous()

            # Intra-chunk additive positional bias (optional).
            #   q_pos : (h_q, d)               -- shared across all positions
            #   k_pos : (S, h_kv, d)           -- per intra-chunk position;
            #                                     position l uses k_pos[l % S]
            if has_intra_pos:
                q_pos, k_pos = chunk_pos_embeddings
                S, L_k = k_pos.shape[0], hsa_k_norm.shape[1]
                pos_idx = torch.arange(L_k, device=hsa_k_norm.device) % S
                # (L_k, h_kv, d) -> (1, L_k, h_kv, d) for broadcast over batch
                k_bias = k_pos[pos_idx].unsqueeze(0)
                hsa_q_norm = hsa_q_norm + q_pos[None, None, :, :]
                hsa_k_norm = hsa_k_norm + k_bias.to(hsa_k_norm.dtype)

            hsa_q_norm = ensure_blhdk_strides(hsa_q_norm)
            hsa_k_norm = ensure_blhdk_strides(hsa_k_norm)
            hsa_v = ensure_blhdk_strides(hsa_v)
            hsa_o = self.hsa_func(
                hsa_q_norm,
                hsa_k_norm,
                hsa_v,
                weights=chunk_weights,
                indices=indices,
                block_size=self.chunk_size,
                mask_last_token=True,
                is_training=kernel_is_training,
            )
            if self.enable_hsa_swa:
                swa_o_weight = chunk_weights[:, :, :, swa_weight_idx]  # (B, L, h_kv)
                # o = hsa_o + swa_o * swa_o_weight.unsqueeze(-1)
                swa_o_weight_expanded = swa_o_weight
                o_lower = torch.addcmul(hsa_o, swa_o, swa_o_weight_expanded.unsqueeze(-1))
            else:
                o_lower = hsa_o
        else:
            o_lower = swa_o
        
        o = o_lower
        o = rearrange(o, 'B L h d->B L (h d)')
        
        return self.o_proj(o), None

# Liger kernel RoPE replacement removed; PoPE is used instead
