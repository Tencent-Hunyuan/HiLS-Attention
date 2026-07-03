from typing import Any, Callable, Optional, Tuple, Union
import torch.nn as nn
import math
from einops import rearrange
from veomni.utils import logging
from .pope import apply_pope_to_qk, apply_pope_to_q
import torch
import os
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from ops.flex_attn_tilelang import flex_attn_tl


logger = logging.get_logger(__name__)


def alibi_slopes_to_scores(
    slopes: torch.Tensor,
    L: int,
    K: int,
    sliding_window: int,
    chunk_size: int,
    *,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Vectorized, batch-broadcastable equivalent of ``naive_alibi_slopes_to_scores``.

    Returns a tensor of shape ``(1, L, h, K)`` (broadcasting over batch) such
    that adding it to ``scores`` reproduces the per-q-head intra-chunk ALiBi
    bias used by HSA topk attention.

    Semantics (matches the naive loop bit-for-bit):
        Let ``W = sliding_window``, ``S = chunk_size``,
        ``V_i = floor((i - W + 1) / S)``, and ``cl_i = min(V_i, K)``
        (number of in-bound chunks for q-position ``i``).  The ``+1`` in
        ``V_i`` matches the ``chunk_offset`` convention used by
        ``create_chunk_dropout_mask`` here and by ``lhsa_layer_pope_naive``,
        i.e. a chunk becomes visible as soon as the q-position has moved at
        least ``W`` tokens past the chunk's last token (inclusive).
        Rank ``k`` of the K topk slots refers to the (``cl_i - 1 - k``)-th
        chunk back from q -- rank ``0`` is the FARTHEST visible chunk and
        rank ``cl_i - 1`` is the CLOSEST one (which sits flush against the
        sliding-window boundary).  Concretely the chunk's start position is::

            chunk_start(i, k) = (V_i - cl_i + 1 + k) * S

        and the bias is::

            bias[i, h, k] = -slopes[h] * (i - chunk_start(i, k))   if k < cl_i
                          = 0                                       otherwise.

        Equivalently, the distance is
        ``(cl_i - 1 - k) * S + (i - V_i * S)``: the closest rank
        ``k = cl_i - 1`` has distance ``i - V_i * S`` (less than ``S``),
        and successive ranks step back ``S`` tokens at a time.

    ``dtype`` controls the output dtype; defaults to ``slopes.dtype``.
    """
    device = slopes.device
    out_dtype = dtype if dtype is not None else slopes.dtype

    # (L,) and (K,) as ints, all math done in int to avoid fp drift.
    i = torch.arange(L, device=device)
    k = torch.arange(K, device=device)

    # V_i = floor((i - W + 1) / S).  Floor-div via ``rounding_mode="floor"``
    # for correct behavior on negative numerators (i < W - 1).  The ``+1``
    # offset matches the project-wide ``chunk_offset`` convention.
    V = (i - sliding_window + 1).div(chunk_size, rounding_mode="floor")  # (L,)
    # cl_i = min(V_i, K), clamped to >= 0 so V_i <= 0 cleanly disables the row.
    cl = V.clamp(min=0, max=K)                                       # (L,)

    # Visibility mask: rank k is in-bound iff k < cl_i.
    visible = k.unsqueeze(0) < cl.unsqueeze(1)                       # (L, K)

    # Actual chunk index for rank k at q-position i: (V_i - cl_i + 1) + k.
    # The trailing ``+1`` (vs. an earlier (V_i - cl_i + k) variant) reflects
    # the naive loop's ``arange(1, cl+1)`` indexing -- rank ``cl_i - 1`` is
    # the chunk immediately preceding the sliding window, NOT the chunk at
    # ``V_i - cl_i``.
    # Distance in tokens: i - chunk_idx * S.  Always non-negative on visible
    # entries because ``chunk_idx * S <= V_i * S <= i + 1 - S <= i`` for any
    # in-bound (visible) entry.
    chunk_idx = (V - cl + 1).unsqueeze(1) + k.unsqueeze(0)           # (L, K)
    dist = i.unsqueeze(1) - chunk_idx * chunk_size + 1               # (L, K)

    dist_f = torch.where(visible, dist, dist.new_zeros(())).to(out_dtype)
    # bias = -slopes[h] * dist[i, k]; broadcast (h,) -> (1, 1, h, 1).
    bias = -slopes.to(out_dtype).view(1, 1, -1, 1) * dist_f.view(1, L, 1, K)
    return bias  # (1, L, h, K) -- broadcasts over batch when added to scores

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
        self.retrieval_head_num = getattr(config, "retrieval_head_num", self.h_hsa_kv)

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
        from ops.hsa_fwd_bwd_head import HSA_block_M_head as HSA
        from ops.topk_head_softmax import online_softmax_topk_head as topk_func
        self.topk_func = topk_func
        self.hsa_func = HSA
        self.hsa_mode = hsa_mode

        self.enable_softmax1 = config.enable_softmax1
        self.hsa_dropout_prob = getattr(config, "hsa_dropout_prob", 0.0)
        self.hsa_disturb_prob = getattr(config, "hsa_disturb_prob", 0.0)

        self.reinited = False

    def _init_alibi_slope(self, device):
        # ------------------------------------------------------------------
        # ALiBi slopes for HSA chunk attention.
        # Spec:
        #   x  = log2(next_power_of_2(topk * chunk_size + sliding_window))
        #   slope[h] = 2 ** ( -(x / h) )  for h = 1, 2, ..., hsa_heads
        # The slope vector has shape (hsa_heads,), matching the ``slope`` arg
        # expected by ``HSA_block_M_head`` (per-q-head).  Stored as a
        # non-persistent buffer because it is fully derived from ``config`` --
        # no need to serialize it into state_dict; it follows the module's
        # device via ``.to(...)`` for free.
        # ------------------------------------------------------------------
        alibi_span = self.topk * self.chunk_size + self.sliding_window
        # next_power_of_2(n): smallest 2**x such that 2**x >= n.  For n <= 1
        # clamp x to 1 to avoid a degenerate zero exponent.
        alibi_x = max(1, math.ceil(math.log2(max(alibi_span, 1))))
        # If alibi_span is exactly a power of 2, ceil(log2) already gives x;
        # otherwise ceil rounds up to the next power's exponent, as required.
        alibi_slopes = torch.tensor(
            [2.0 ** (-(alibi_x / h)) for h in range(1, self.h_q + 1)],
            dtype=torch.float32,
            device=device
        )
        self.register_buffer("alibi_slopes", alibi_slopes, persistent=False)

        # Cache for the (L, K)-dependent additive ALiBi bias applied to topk
        # ``scores``.  In training the seq length is fixed and topk doesn't
        # change, so this cache hits 100% of the time after the first call.
        # The cache value lives on the same device as ``alibi_slopes`` (moved
        # automatically by ``module.to(...)`` because we keep it as a regular
        # attribute, not a parameter; we manage device transitions in the
        # accessor below).
        self._alibi_bias_cache_key: Optional[Tuple[int, int, torch.dtype]] = None
        self._alibi_bias_cache: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _alibi_bias_for_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Return the additive ALiBi bias to be added to ``scores``.

        Shape: ``(1, L, h, K)`` so it broadcasts over the batch dimension of
        ``scores: (N, L, h, K)``.  Identical numerically to
        ``naive_alibi_slopes_to_scores(self.alibi_slopes, scores, ...)``.

        Cached by ``(L, K, dtype)`` -- (slopes, sliding_window, chunk_size)
        are fixed for the layer.  The cache is invalidated automatically if
        any of the cache key components change (e.g. variable seq length at
        eval time).  A device move is detected and the cached tensor is
        relocated lazily; this is virtually free since it's a small tensor.
        """
        L, K = scores.shape[1], scores.shape[-1]
        key = (L, K, scores.dtype)
        cached = self._alibi_bias_cache
        if (
            cached is not None
            and self._alibi_bias_cache_key == key
            and cached.device == scores.device
        ):
            return cached

        bias = alibi_slopes_to_scores(
            self.alibi_slopes,
            L=L, K=K,
            sliding_window=self.sliding_window,
            chunk_size=self.chunk_size,
            dtype=scores.dtype,
        )
        self._alibi_bias_cache = bias
        self._alibi_bias_cache_key = key
        return bias


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
        if not self.reinited:
            # deal with veomni bugs
            self._init_alibi_slope(hidden_states.device)

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


        # Inference: HSA KV cache
        if use_cache and past_key_values is not None:
            hsa_cache_idx = self.layer_idx + self.config.num_hidden_layers
            cache_kwargs = {"cache_position": cache_position}
            hsa_k_norm_nope, hsa_v = past_key_values.update(
                hsa_k_norm_nope, hsa_v, hsa_cache_idx, cache_kwargs
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

        # apply pope for hsa part
        swa_o, lse_sum = flex_attn_tl(
            hsa_q_norm_nope.transpose(1, 2), # (B, H, L, d)
            hsa_k_norm_nope.transpose(1, 2), 
            hsa_v.transpose(1, 2), 
            window_size=self.hsa_sliding_window,
            chunk_size=self.chunk_size,
            training=(past_key_values is None),
            mask_lmk=True,
            expand_to_chunk=True,
            m_h=self.alibi_slopes
        )
        # lse_sum = lse_sum.transpose(1, 2)  # (B, L, h_q)
        lse_sum = lse_sum.contiguous()  # (B, L, hG)
        lse_sum = lse_sum.to(hidden_states.dtype)  # (B, L, h_kv)

        full_seq_len = hsa_k_norm_nope.shape[1]
        if full_seq_len >= self.chunk_size:
            lmk_k: Any = hsa_k_norm_nope[:, self.chunk_size - 1::self.chunk_size, : ,:]  # (B, L // S, hsa_kv, d)

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
            indices, scores = self.topk_func(
                lmk_q_norm,
                lmk_k,
                lse_sum,
                self.topk,
                block_size=self.chunk_size,
                window_size=self.hsa_sliding_window,
                is_causal=True,
                drop_mask=drop_mask,
                q_offset=q_offset,
                bias=self.alibi_slopes * self.sliding_window
            )
            # scores: (N, L, h, K)

            # Apply ALiBi additive bias to the topk scores.  Indices are in
            # ascending order (rank 0 == most-distant chunk, rank K-1 == most
            # recent), which matches the closed-form bias[i, h, k] =
            # -slope[h] * (i - k * chunk_size) used by ``alibi_slopes_to_scores``.
            scores = scores + self._alibi_bias_for_scores(scores)

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


            hsa_q_norm = hsa_q_norm_nope  # (B, L, h_q, d)
            hsa_k_norm = hsa_k_norm_nope    # (B, L, h_kv, d)

            hsa_o = self.hsa_func(
                hsa_q_norm, hsa_k_norm, hsa_v,
                weights=chunk_weights, indices=indices,
                block_size=self.chunk_size, mask_last_token=True,
                slope=self.alibi_slopes,
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
