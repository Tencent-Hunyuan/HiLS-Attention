"""
Automatic chunk prefill module.

Split long sequences into chunks and pass each chunk through all decoder layers
while preserving causality with KV cache. This reduces peak MLP activation
memory from O(seq_len) to O(chunk_size).

Usage:
    In HiLSModel.forward, call chunked_forward instead of the original
    layer-by-layer loop when inference sequence length exceeds the threshold.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from utils.hsa_cache_utils import HSADynamicLayer
from .pope import PolarEmbedReturn

# Hard-coded chunk size in tokens
CHUNK_PREFILL_SIZE = 1024

# Default auto-enable threshold in tokens; sequences above this length use chunk prefill
DEFAULT_CHUNK_PREFILL_THRESHOLD = 128 * 1024  # 128K

def chunked_forward(
    model,  # HiLSModel instance
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    """
    Run all decoder layers on a long sequence chunk by chunk.

    Main idea:
    1. Compute embeddings and position embeddings globally; this has low memory cost.
    2. Split hidden_states by CHUNK_PREFILL_SIZE.
    3. Pass each chunk through all decoder layers while KV cache accumulates.
    4. Concatenate chunk outputs and apply final norm.

    Args:
        model: HiLSModel instance.
        Other arguments match HiLSModel.forward.
    """
    config = model.config

    # KV cache must be enabled internally so later chunks can attend to previous chunks.
    # The return value still follows the caller original use_cache intent.
    caller_use_cache = use_cache if use_cache is not None else config.use_cache
    use_cache = True  # Forced internally

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # 1. Embedding
    if inputs_embeds is None:
        inputs_embeds = model.embed_tokens(input_ids)

    seq_len = inputs_embeds.shape[1]

    # 2. Initialize KV cache
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if use_cache and isinstance(past_key_values, DynamicCache):
        required_layers = 2 * config.num_hidden_layers
        while len(past_key_values.layers) < required_layers:
            idx = len(past_key_values.layers)
            if idx >= config.num_hidden_layers:
                past_key_values.layers.append(HSADynamicLayer())
            else:
                past_key_values.layers.append(DynamicLayer())

    # 3. Compute global cache_position and position_ids
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    # 4. Compute global position embeddings; RoPE has low memory cost
    position_embeddings = model.rotary_emb(inputs_embeds, position_ids)

    # 4.1 Compute global PoPE embeddings for PoPE HSA layers
    pope_pos_embeddings = None
    if hasattr(model, 'pop_emb'):
        if getattr(model, 'routine_pope', False):
            _past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            pope_pos_ids = torch.arange(_past_seen, _past_seen + seq_len, device=inputs_embeds.device) % model.chunk_size
        else:
            pope_pos_ids = position_ids
        pope_pos_embeddings = model.pop_emb(inputs_embeds, pope_pos_ids)

    # 5. Process chunks
    chunk_prefill_size = CHUNK_PREFILL_SIZE
    num_chunks = (seq_len + chunk_prefill_size - 1) // chunk_prefill_size
    all_hidden_chunks = []
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    q_pos = None
    k_pos = None
    if getattr(model, "q_pos", None) is not None and getattr(model, "k_pos", None) is not None:
        head_dim = getattr(model, "head_dim", config.hidden_size // config.num_attention_heads)
        q_pad = head_dim - model.q_pos.shape[-1]
        k_pad = head_dim - model.k_pos.shape[-1]
        assert q_pad >= 0 and k_pad >= 0, (
            f"q_pos/k_pos last-dim ({model.q_pos.shape[-1]}, "
            f"{model.k_pos.shape[-1]}) must be <= head_dim ({head_dim})."
        )
        q_pos = F.pad(model.q_pos, (q_pad, 0)) if q_pad > 0 else model.q_pos
        k_pos = F.pad(model.k_pos, (k_pad, 0)) if k_pad > 0 else model.k_pos

    for c in range(num_chunks):
        start = c * chunk_prefill_size
        end = min((c + 1) * chunk_prefill_size, seq_len)

        # Slice inputs for the current chunk
        chunk_hidden = inputs_embeds[:, start:end, :]
        chunk_position_ids = position_ids[:, start:end]
        chunk_cache_position = cache_position[start:end]
        chunk_pos_emb = (
            position_embeddings[0][:, start:end, :],
            position_embeddings[1][:, start:end, :],
        )

        # Slice PoPE embeddings for the current chunk
        if pope_pos_embeddings is not None:
            freqs = pope_pos_embeddings.freqs
            if freqs.ndim == 3:
                chunk_pope = PolarEmbedReturn(freqs[:, start:end, :], pope_pos_embeddings.bias)
            else:
                chunk_pope = PolarEmbedReturn(freqs[start:end, :], pope_pos_embeddings.bias)
        else:
            chunk_pope = None

        chunk_cache_pope = None
        needs_pope_cache = (
            chunk_pope is not None
            and use_cache
            and past_key_values is not None
            and getattr(config, "pope_impl", None) == "naive"
            and getattr(config, "nope_chunkwise_attn", False)
        )
        if needs_pope_cache:
            chunk_freqs = chunk_pope.freqs
            if chunk_freqs.ndim == 2:
                chunk_freqs = chunk_freqs.unsqueeze(0)

            cached_freqs = getattr(past_key_values, "_pope_freqs", None)
            reset_pope_cache = (
                cached_freqs is None
                or chunk_cache_position.numel() == 0
                or int(chunk_cache_position[0].item()) == 0
            )
            if reset_pope_cache:
                cached_freqs = chunk_freqs.contiguous()
            else:
                cached_freqs = torch.cat(
                    [cached_freqs.to(chunk_freqs.device), chunk_freqs],
                    dim=-2,
                ).contiguous()
            past_key_values._pope_freqs = cached_freqs
            chunk_cache_pope = PolarEmbedReturn(cached_freqs, chunk_pope.bias)

        # Build the causal mask for the current chunk.
        # Pass the chunk cache_position so the mask accounts for existing KV cache.
        mask_kwargs = {
            "config": config,
            "input_embeds": chunk_hidden,
            "attention_mask": attention_mask,
            "cache_position": chunk_cache_position,
            "past_key_values": past_key_values,
            "position_ids": chunk_position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        # Run decoder layers one by one.
        # position_ids is popped in Olmo3Attention.forward and is not passed into flash attention,
        # so cu_seq_lens does not need to be computed here.
        for layer_idx, decoder_layer in enumerate(model.layers[: config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (chunk_hidden,)

            attention_type = getattr(
                decoder_layer.self_attn,
                "attention_type",
                config.layer_types[layer_idx] if config.layer_types is not None else "full_attention",
            )
            layer_flash_attn_kwargs = flash_attn_kwargs
            if chunk_cache_pope is not None and hasattr(decoder_layer.self_attn, "nope_chunkwise_attn"):
                layer_flash_attn_kwargs = dict(flash_attn_kwargs)
                layer_flash_attn_kwargs["pope_cache_pos_embeddings"] = chunk_cache_pope

            layer_outputs = decoder_layer(
                chunk_hidden,
                attention_mask=causal_mask_mapping[attention_type],
                position_ids=chunk_position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=chunk_cache_position,
                position_embeddings=chunk_pos_emb,
                pope_pos_embeddings=chunk_pope,
                chunk_pos_embeddings=[q_pos, k_pos],
                **layer_flash_attn_kwargs,
            )

            chunk_hidden = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                all_self_attns += (layer_outputs[1],)

        all_hidden_chunks.append(chunk_hidden)

    # 6. Concatenate all chunk outputs
    hidden_states = torch.cat(all_hidden_chunks, dim=1)

    # 7. Final norm
    hidden_states = model.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if caller_use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
