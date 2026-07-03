"""Shared forward implementations for HiLSModel and HiLSForCausalLM."""

import inspect
from tkinter import N
from typing import Optional, Union

from utils.flex_attn import cu_seqlens_to_doc_ids
import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack

from utils.hsa_cache_utils import HSADynamicLayer

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.utils import logging
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
from .pope import PolarEmbedReturn

logger = logging.get_logger(__name__)


def _build_mask_kwargs(
    mask_fn,
    self,
    attention_mask,
    inputs_embeds,
    cache_position,
    past_key_values,
    position_ids,
):
    """Build kwargs for transformers mask helpers across local version drift."""
    params = inspect.signature(mask_fn).parameters
    kwargs = {
        "config": self.config,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    if "inputs_embeds" in params:
        kwargs["inputs_embeds"] = inputs_embeds
    elif "input_embeds" in params:
        kwargs["input_embeds"] = inputs_embeds
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    return kwargs


def _prepare_causal_mask_mapping(
    self,
    attention_mask,
    inputs_embeds,
    cache_position,
    past_key_values,
    position_ids,
):
    if isinstance(attention_mask, dict):
        return attention_mask

    causal_mask_mapping = {
        "full_attention": create_causal_mask(
            **_build_mask_kwargs(
                create_causal_mask,
                self,
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                position_ids,
            )
        ),
    }

    layer_types = getattr(self.config, "layer_types", None) or ()
    if "sliding_attention" in layer_types or getattr(self.config, "sliding_window", None) is not None:
        causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
            **_build_mask_kwargs(
                create_sliding_window_causal_mask,
                self,
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values,
                position_ids,
            )
        )

    return causal_mask_mapping


def hsa_model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    cache_position=None,
    prepare_attention_mask_fn=None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    """Common forward for HiLSModel (Olmo3 / Qwen3).

    prepare_attention_mask_fn: if provided, called as fn(self, attn_mask, inputs_embeds,
        cache_position, past_key_values, position_ids) -> causal_mask_mapping dict.
        If None, masks are prepared with the same full/sliding causal-mask helpers
        used by HF decoder-only models.
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    logger.warning_once(f"[hsa_model_forward] training={self.training}, use_cache={use_cache}, gradient_checkpointing={self.gradient_checkpointing}")

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

    if inputs_embeds is None:
        lmk_embed = getattr(self, "lmk_embed", None)
        if lmk_embed is not None:
            lmk_id = self.lmk_id  # guaranteed non-None when lmk_embed exists
            lmk_mask = input_ids == lmk_id                                  # (B, L) bool
            safe_ids = input_ids.masked_fill(lmk_mask, 0)
            inputs_embeds = self.embed_tokens(safe_ids)                     # (B, L, D)
            inputs_embeds = torch.where(
                lmk_mask.unsqueeze(-1),                                     # (B, L, 1)
                lmk_embed.to(inputs_embeds.dtype),                          # (D,) broadcast
                inputs_embeds,
            )
        else:
            inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if use_cache and not self.training and isinstance(past_key_values, DynamicCache):
        required_layers = 2 * self.config.num_hidden_layers
        while len(past_key_values.layers) < required_layers:
            idx = len(past_key_values.layers)
            if idx >= self.config.num_hidden_layers:
                past_key_values.layers.append(HSADynamicLayer())
            else:
                past_key_values.layers.append(DynamicLayer())

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # --- update varlen flash attn kwargs due according to lmk position ids adjustments ---
    if input_ids.shape[0] == 1 and position_ids is not None and past_key_values is None and "cu_seq_lens_q" in flash_attn_kwargs:
        (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(position_ids)
        # assert torch.all(cu_q[:-1] % self.chunk_size == 0), f'input pos ids not padding to {self.chunk_size}x, cu_q: {cu_q}, max_q: {max_q}'
        doc_ids = cu_seqlens_to_doc_ids(cu_q, input_ids.shape[1], input_ids.device)
        flash_attn_kwargs = dict(flash_attn_kwargs)
        flash_attn_kwargs.update(
            {"cu_seq_lens_q": cu_q, "cu_seq_lens_k": cu_k, "max_length_q": max_q, "max_length_k": max_k, "doc_ids": doc_ids}
        )

    if prepare_attention_mask_fn is not None:
        causal_mask_mapping = prepare_attention_mask_fn(self, attention_mask, inputs_embeds, cache_position, past_key_values, position_ids)
    else:
        causal_mask_mapping = _prepare_causal_mask_mapping(
            self,
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            position_ids,
        )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    hsa_position_embeddings = None
    hsa_rotary_emb = getattr(self, "hsa_rotary_emb", None)
    if hsa_rotary_emb is not None:
        hsa_position_embeddings = hsa_rotary_emb(hidden_states, position_ids)

    # --- PoPE embeddings (used by fused LandmarkHSA layers) ---
    pope_pos_embeddings = None
    pope_cache_pos_embeddings = None
    if hasattr(self, 'pop_emb'):
        pope_pos_ids = position_ids
        pope_pos_embeddings = self.pop_emb(hidden_states, pope_pos_ids)
        needs_pope_cache = (
            use_cache
            and past_key_values is not None
            and getattr(self.config, "pope_impl", None) == "naive"
            and getattr(self.config, "nope_chunkwise_attn", False)
        )
        if needs_pope_cache:
            freqs = pope_pos_embeddings.freqs
            if freqs.ndim == 2:
                freqs = freqs.unsqueeze(0)

            cached_freqs = getattr(past_key_values, "_pope_freqs", None)
            reset_pope_cache = (
                cached_freqs is None
                or cache_position is None
                or cache_position.numel() == 0
                or int(cache_position[0].item()) == 0
            )
            if reset_pope_cache:
                cached_freqs = freqs.contiguous()
            else:
                cached_freqs = torch.cat(
                    [cached_freqs.to(freqs.device), freqs],
                    dim=-2,
                ).contiguous()
            past_key_values._pope_freqs = cached_freqs
            pope_cache_pos_embeddings = PolarEmbedReturn(
                cached_freqs,
                pope_pos_embeddings.bias,
            )

    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
    if hsa_position_embeddings is not None:
        hsa_position_embeddings = slice_position_embedding(hsa_position_embeddings, dim=1, sp_group=sp_group)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    q_pos = None
    k_pos = None
    raw_q_pos = getattr(self, "q_pos", None)
    raw_k_pos = getattr(self, "k_pos", None)
    if raw_q_pos is not None and raw_k_pos is not None:
        # Decoder layers expect intra-chunk positional bias spanning the FULL
        # head_dim.  Some configs (e.g. olmo with PoPE) only learn the trailing
        # ``head_dim - pope_dim`` channels and leave the leading ``pope_dim``
        # channels as zeros (so PoPE owns the rotated front, intra-chunk-pos
        # owns the un-rotated tail).  Left-pad with zeros to reach head_dim.
        head_dim = getattr(self, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        q_pad = head_dim - raw_q_pos.shape[-1]
        k_pad = head_dim - raw_k_pos.shape[-1]
        assert q_pad >= 0 and k_pad >= 0, (
            f"q_pos/k_pos last-dim ({raw_q_pos.shape[-1]}, "
            f"{raw_k_pos.shape[-1]}) must be <= head_dim ({head_dim})."
        )
        # F.pad on the last dim: (pad_left, pad_right).  We want zeros on the
        # LEFT, so pad_left=q_pad, pad_right=0.
        q_pos = F.pad(raw_q_pos, (q_pad, 0)) if q_pad > 0 else raw_q_pos
        k_pos = F.pad(raw_k_pos, (k_pad, 0)) if k_pad > 0 else raw_k_pos

    for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_flash_attn_kwargs = flash_attn_kwargs
        if pope_cache_pos_embeddings is not None and hasattr(decoder_layer.self_attn, "nope_chunkwise_attn"):
            layer_flash_attn_kwargs = dict(flash_attn_kwargs)
            layer_flash_attn_kwargs["pope_cache_pos_embeddings"] = pope_cache_pos_embeddings
        layer_position_embeddings = position_embeddings
        if hsa_position_embeddings is not None and getattr(decoder_layer, "use_hsa_rotary_embedding", False):
            layer_position_embeddings = hsa_position_embeddings
        attention_type = getattr(
            decoder_layer,
            "attention_type",
            self.config.layer_types[layer_idx] if getattr(self.config, "layer_types", None) is not None else "full_attention",
        )

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping.get(attention_type, causal_mask_mapping["full_attention"]),
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=layer_position_embeddings,
            pope_pos_embeddings=pope_pos_embeddings,
            chunk_pos_embeddings=[q_pos, k_pos],
            **layer_flash_attn_kwargs,
        )
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def hsa_causal_lm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    cache_position=None,
    logits_to_keep=0,
    **kwargs,
) -> CausalLMOutputWithPast:
    """Common forward for HiLSForCausalLM (Olmo3 / Qwen3)."""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

    non_lmk_mask = None  # 仅在 auto_insert_lmk 非 generate 模式下被赋值
    if self.training and self.insert_landmarks:
        if self.adjust_lmk_pos:
            position_ids = create_position_ids_with_landmarks(position_ids, input_ids.shape[1], self.chunk_size, input_ids.device)

        input_ids = insert_special_tokens(input_ids, self.lmk_id, self.chunk_size)

        if labels is not None:
            sp_enabled = get_parallel_state().sp_enabled
            if not sp_enabled:
                labels = torch.roll(labels, shifts=-1, dims=-1)
                labels[:, -1] = -100
                labels = insert_special_tokens(labels, -100, self.chunk_size)
                labels = torch.roll(labels, shifts=1, dims=-1)
            else:
                labels = insert_special_tokens(labels, -100, self.chunk_size)
    elif self.insert_landmarks and self.auto_insert_lmk and not self._gen_state.active:
        position_ids = create_position_ids_with_landmarks(position_ids, input_ids.shape[1], self.chunk_size, input_ids.device)
        input_ids = insert_special_tokens(input_ids, self.lmk_id, self.chunk_size)
        new_seq_len = input_ids.shape[1]
        pos_indices = torch.arange(new_seq_len, device=input_ids.device)
        non_lmk_mask = ~(pos_indices % self.chunk_size == self.chunk_size - 1)

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # 统一的 LMK 过滤：generate decode 边界步 + auto_insert_lmk 非 generate 模式
    hidden_states = self._filter_lmk_hidden_states(hidden_states, non_lmk_mask)

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits, labels=labels, vocab_size=self.vocab_size,
            hidden_states=hidden_states, weights=self.lm_head.weight, **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states)

    return CausalLMOutputWithPast(
        loss=loss, logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )