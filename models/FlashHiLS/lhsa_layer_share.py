from typing import Any, Callable, Optional, Tuple, Union
import torch.nn as nn
from einops import rearrange
from veomni.utils import logging
import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from utils.flex_attn import flex_attn
from veomni.utils.import_utils import (
    is_liger_kernel_available,
)
if is_liger_kernel_available():
    from liger_kernel.transformers.rope import liger_rotary_pos_emb


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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

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
        self.retrieval_head_num = 1 if self.unified_retrieval else self.h_hsa_kv

        if self.hsa_denom > 1:
            self.q_proj = nn.Linear(self.d_model, self.d_model * (self.hsa_denom - 1) // self.hsa_denom, bias=False)
            self.k_proj = nn.Linear(self.d_model, self.d_kv * (self.hsa_denom - 1) // self.hsa_denom, bias=False)
            self.v_proj = nn.Linear(self.d_model, self.d_kv * (self.hsa_denom - 1) // self.hsa_denom, bias=False)
        self.hsa_q_proj = nn.Linear(self.d_model, self.d_model // self.hsa_denom, bias=False)

        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)

        self.enable_lmk_q_proj = getattr(config, "enable_lmk_q_proj", False)
        self.apply_hsa_rope = getattr(config, "apply_hsa_rope", False)
        if self.enable_lmk_q_proj:
            self.retrieval_dim = getattr(config, 'retrieval_dim', self.d_model // self.hsa_denom // self.hsa_qk_ratio)
            self.lmk_q_proj = nn.Linear(self.d_model, self.retrieval_dim, bias=False)
            self.lmk_q_norm = norm_cls(self.retrieval_dim // self.retrieval_head_num)

        if not self.enable_lmk_q_proj:
            logger.warning_once("Recommend to set enable_lmk_q_proj=True for better performance")

        self.topk = config.hsa_topk
        
        self.chunk_size = config.chunk_size
        self.layerwise_qk_norm = getattr(config, "layerwise_qk_norm", False)
        if not self.layerwise_qk_norm:
            self.q_norm = norm_cls(self.head_dim)
        else:
            self.q_norm = norm_cls(self.d_model)


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
        
        self.is_causal = True
        
        hsa_mode = config.hsa_mode
        from ops.hsa_fwd_bwd_group import HSA_block_M_group as HSA
        from ops.topk_group import online_topk_group as topk_func
        self.topk_func = topk_func
        self.hsa_func = HSA
        self.hsa_mode = hsa_mode

        self.enable_softmax1 = config.enable_softmax1
        self.hsa_dropout_prob = getattr(config, "hsa_dropout_prob", 0.0)
        self.hsa_disturb_prob = getattr(config, "hsa_disturb_prob", 0.0)

    def forward(self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions=None,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        shared_k=None,
        shared_v=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        B, L, _ = hidden_states.shape
        cos, sin = position_embeddings
        if self.hsa_denom > 1:
            assert not self.layerwise_qk_norm, 'not support layerwise_qk_norm with hsa_denom > 1'
            swa_q = self.q_proj(hidden_states)  # (B, L, d)
            swa_q = rearrange(swa_q, 'B L (h d)->B L h d', d=self.head_dim)
            swa_q_norm = self.q_norm(swa_q)  # q_norm
            swa_k = self.k_proj(hidden_states)
            swa_k = rearrange(swa_k, 'B L (h d)->B L h d', d=self.head_dim)  # (B, L, h, d)
            swa_k_norm = self.k_norm(swa_k)
            swa_v = self.v_proj(hidden_states)
            swa_v = rearrange(swa_v, 'B L (h d)->B L h d', d=self.head_dim)  # (B, L, h, d)

            swa_q_norm = swa_q_norm.transpose(1, 2)
            swa_k_norm = swa_k_norm.transpose(1, 2)
            swa_v = swa_v.transpose(1, 2)  # (B, h, L, d)

            # The position embedding should be compatible with kv passing
            assert swa_q_norm.shape[2] == swa_k_norm.shape[2], f'{swa_q_norm.shape} vs {swa_k_norm.shape}'
            swa_q_norm, swa_k_norm = apply_rotary_pos_emb(swa_q_norm, swa_k_norm, cos, sin)

            # Inference: SWA KV cache
            if use_cache and past_key_values is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                swa_k_norm, swa_v = past_key_values.update(
                    swa_k_norm, swa_v, self.layer_idx, cache_kwargs
                )
                if self.sliding_window is not None:
                    kv_item = past_key_values.layers[self.layer_idx]
                    if kv_item.keys is not None and kv_item.keys.shape[-2] > self.sliding_window:
                        kv_item.keys = kv_item.keys[:, :, -self.sliding_window:, :]
                        kv_item.values = kv_item.values[:, :, -self.sliding_window:, :]

            attention_interface: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            o_upper, _ = attention_interface(
                self,
                swa_q_norm,
                swa_k_norm,
                swa_v,
                attention_mask,
                dropout=0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,  # diff with Llama
                **kwargs,
            )  # (B, L, h_q // 2, d)

        hsa_q = self.hsa_q_proj(hidden_states)
        if self.layerwise_qk_norm:
            hsa_q = self.q_norm(hsa_q)
        hsa_q = rearrange(hsa_q, 'B L (h d)->B L h d', d=self.head_dim)
    
        if not self.layerwise_qk_norm:
            hsa_q_norm = self.q_norm(hsa_q)  # （B, L, h, d)
        else:
            hsa_q_norm = hsa_q

        if self.enable_lmk_q_proj:
            lmk_q = self.lmk_q_proj(hidden_states)
            lmk_q = rearrange(lmk_q, 'B L (h d)->B L h d', h=self.retrieval_head_num)
            lmk_q_norm = self.lmk_q_norm(lmk_q)  # (B, L, h_kv * d // 2)
        else:
            assert not self.unified_retrieval, "Unified_retrieval is not supported for w/o lmk q proj"
            lmk_q_norm = hsa_q_norm

        # hsa_k = self.hsa_k_proj(hidden_states)
        # if self.layerwise_qk_norm:
        #     hsa_k = self.k_norm(hsa_k)
        # hsa_k = rearrange(hsa_k, 'B L (h d)->B L h d', d=self.head_dim)

        # if not self.layerwise_qk_norm:
        #     hsa_k_norm = self.k_norm(hsa_k)
        # else:
        #     hsa_k_norm = hsa_k

        # hsa_v = self.hsa_v_proj(hidden_states)
        # hsa_v = rearrange(hsa_v, 'B L (h d)->B L h d', d=self.head_dim)
        hsa_k_norm = shared_k  # (B L h d)
        hsa_v = shared_v  # (B L h d)

        if self.apply_hsa_rope and self.enable_hsa_swa:
            assert not self.unified_retrieval, "Unified_rope is not supported for hsa+RoPE"
            hsa_swa_q_norm, hsa_swa_k_norm = apply_rotary_pos_emb(hsa_q_norm.transpose(1, 2), hsa_k_norm.transpose(1, 2), cos, sin)
            hsa_swa_q_norm = hsa_swa_q_norm.transpose(1, 2).contiguous()
            hsa_swa_k_norm = hsa_swa_k_norm.transpose(1, 2).contiguous()
        else:
            hsa_swa_q_norm = hsa_q_norm
            hsa_swa_k_norm = hsa_k_norm

        # Inference: HSA KV cache
        if use_cache and past_key_values is not None:
            hsa_cache_idx = self.layer_idx + self.config.num_hidden_layers
            hsa_k_cache = hsa_k_norm.transpose(1, 2)  # (B, h, L, d)
            hsa_v_cache = hsa_v.transpose(1, 2)
            cache_kwargs = {"cache_position": cache_position}
            hsa_k_cache, hsa_v_cache = past_key_values.update(
                hsa_k_cache, hsa_v_cache, hsa_cache_idx, cache_kwargs
            )
            hsa_k_norm = hsa_k_cache.transpose(1, 2).contiguous()  # (B, L_full, h, d)
            hsa_v = hsa_v_cache.transpose(1, 2).contiguous()

        cu_seq_lens_q = kwargs.get("cu_seq_lens_q", None)
        doc_ids = kwargs.get("doc_ids", None)
        if cu_seq_lens_q is not None and cu_seq_lens_q.shape[0] > 2:
            assert hidden_states.shape[0] == 1, f'cu_seq_lens_q is only supported for batch size 1, but got {hidden_states.shape[0]}'
            assert torch.all((cu_seq_lens_q[:-1] % self.chunk_size) ==0), f'cu_seq_lens_q must be divisible by chunk_size, cu_seq_lens_q: {cu_seq_lens_q}'
        else:
            cu_seq_lens_q = None

        if hidden_states.shape[0] > 1:
            assert doc_ids is None, f'doc_ids is not supported for batch size > 1, but got doc ids {doc_ids}'

        # apply rope for hsa part
        if self.enable_hsa_swa:
            swa_o, lse_sum = flex_attn(
                hsa_swa_q_norm.transpose(1, 2), 
                hsa_swa_k_norm.transpose(1, 2), 
                hsa_v.transpose(1, 2), 
                window_size=self.hsa_sliding_window,
                chunk_size=self.chunk_size,
                training=(past_key_values is None),
                doc_ids=doc_ids,
            )
            lse_sum = lse_sum.transpose(1, 2)  # (B, L, h_q)
            lse_sum = rearrange(lse_sum, 'B L (h G)->B L h G', h=self.retrieval_head_num)
            lse_sum = lse_sum.logsumexp(dim=-1).contiguous()  # sum along group
            lse_sum = lse_sum.to(hidden_states.dtype)  # (B, L, h_kv)
            swa_o = swa_o.transpose(1, 2)  # (B, L, h_q, hd)

        full_seq_len = hsa_k_norm.shape[1]
        if full_seq_len >= self.chunk_size:
            lmk_k: Any = hsa_k_norm[:, self.chunk_size - 1::self.chunk_size, : ,:]  # (B, L // S, hsa_kv, d)
            if self.unified_retrieval:
                lmk_k = rearrange(lmk_k, 'B S H D -> B S 1 (H D)')
            hsa_visible_window = self.hsa_visible_window if self.training else -1
            B, S,  H, D = lmk_k.shape

            q_offset = int(cache_position[0].item()) if (use_cache and cache_position is not None) else 0

            drop_mask = None
            if self.training and self.hsa_dropout_prob > 0.0:
                drop_mask = create_chunk_dropout_mask(
                    B, L, self.chunk_size, self.hsa_sliding_window, self.hsa_dropout_prob, device=hidden_states.device
                )
                if self.hsa_disturb_prob > 0.0:
                    gate = (torch.rand(B, L, 1, device=lmk_q_norm.device) < self.hsa_disturb_prob).to(torch.int32)
                    drop_mask = drop_mask * gate

            indices, scores = self.topk_func(
                lmk_q_norm, 
                lmk_k, 
                self.topk, 
                block_size=self.chunk_size, 
                window_size=self.hsa_sliding_window,
                is_causal=True,
                is_training=self.training,
                drop_mask=drop_mask,
                q_offset=q_offset,
                cu_seq_lens=cu_seq_lens_q,
            )

            if self.enable_hsa_swa:
                if not self.enable_softmax1:
                    cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1)], dim=-1)  # (B, L, h_kv, K + 1)
                    swa_weight_idx = -1
                else:
                    # softmax off by one
                    cat_scores = torch.cat([scores, lse_sum.unsqueeze(-1), torch.zeros(B, L, self.retrieval_head_num, 1, device=hidden_states.device)], dim=-1)
                    swa_weight_idx = -2
                chunk_weights = F.softmax(cat_scores, dim=-1).to(hidden_states.dtype) # (B, L, h_kv, K)
            else:
                chunk_weights = F.softmax(scores, dim=-1).to(hidden_states.dtype) # (B, L, h_kv, K)

            if self.unified_retrieval:
                chunk_weights = chunk_weights.repeat_interleave(self.h_hsa_kv // self.retrieval_head_num, dim=2).contiguous()
                indices = indices.repeat_interleave(self.h_hsa_kv // self.retrieval_head_num, dim=2).contiguous()

            if self.apply_hsa_rope:
                # Apply chunk-periodic RoPE for HSA block attention:
                # q: all positions rotated to chunk_size (fixed position)
                # k: positions rotate with period chunk_size (pos % chunk_size)
                L_full = hsa_k_norm.shape[1]

                # q positions: periodic within chunk (chunk_size positions, tiled to L)
                q_cos_chunk = cos[:, self.chunk_size + self.hsa_sliding_window : 2 * self.chunk_size + self.hsa_sliding_window, :]  # (B, chunk_size, head_dim)
                q_sin_chunk = sin[:, self.chunk_size + self.hsa_sliding_window : 2 * self.chunk_size + self.hsa_sliding_window, :]  # (B, chunk_size, head_dim)
                n_repeats = (L + self.chunk_size - 1) // self.chunk_size
                q_cos = q_cos_chunk.repeat(1, n_repeats, 1)[:, :L, :]  # (B, L, head_dim)
                q_sin = q_sin_chunk.repeat(1, n_repeats, 1)[:, :L, :]  # (B, L, head_dim)

                # k positions: periodic with chunk_size
                k_positions = torch.arange(L_full, device=hsa_k_norm.device) % self.chunk_size  # (L_full,)
                k_cos = cos[:, k_positions, :]  # (B, L_full, head_dim) -> index by periodic positions
                k_sin = sin[:, k_positions, :]  # (B, L_full, head_dim)

                # hsa_q_norm/hsa_k_norm shape: (B, L, h, d), need transpose to (B, h, L, d) for apply_rotary_pos_emb
                # Apply separately since q and k use different cos/sin
                hsa_q_rot = hsa_q_norm.transpose(1, 2)  # (B, h, L, d)
                hsa_k_rot = hsa_k_norm.transpose(1, 2)  # (B, h, L_full, d)
                q_cos_u = q_cos.unsqueeze(1)  # (B, 1, L, head_dim)
                q_sin_u = q_sin.unsqueeze(1)
                k_cos_u = k_cos.unsqueeze(1)  # (B, 1, L_full, head_dim)
                k_sin_u = k_sin.unsqueeze(1)
                hsa_q_norm = ((hsa_q_rot * q_cos_u) + (rotate_half(hsa_q_rot) * q_sin_u)).transpose(1, 2).contiguous()
                hsa_k_norm = ((hsa_k_rot * k_cos_u) + (rotate_half(hsa_k_rot) * k_sin_u)).transpose(1, 2).contiguous()

            hsa_o = self.hsa_func(hsa_q_norm, hsa_k_norm, hsa_v, weights=chunk_weights, indices=indices, block_size=self.chunk_size, mask_last_token=True, is_training=self.training)
            if self.enable_hsa_swa:
                swa_o_weight = chunk_weights[:, :, :, swa_weight_idx]  # (B, L, h_kv)
                # o = hsa_o + swa_o * swa_o_weight.unsqueeze(-1)
                swa_o_weight_expanded = swa_o_weight.repeat_interleave(
                    self.hsa_qk_ratio, dim=2
                )
                o_lower = torch.addcmul(hsa_o, swa_o, swa_o_weight_expanded.unsqueeze(-1))
            else:
                o_lower = hsa_o
        else:
            o_lower = swa_o
        
        if self.hsa_denom > 1:
            o = torch.cat([o_upper, o_lower], dim=2)
        else:
            o = o_lower
        o = rearrange(o, 'B L h d->B L (h d)')
        
        return self.o_proj(o), None

if is_liger_kernel_available():
    apply_rotary_pos_emb = liger_rotary_pos_emb
    logger.info_rank0("Apply liger kernel to LHSA.")
