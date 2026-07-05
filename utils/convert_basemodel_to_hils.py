"""Convert OLMo3 base model weights to HiLS model weights.

Usage:
    python utils/convert_basemodel_to_hils.py \
        --base_path /path/to/olmo3_base_model \
        --target_config configs/olmo3_7B/olmo3_lhsa_interleave_8KA2K_non_unified.json \
        --output_path /path/to/output/converted_model.pt \
        [--log_path /path/to/log.json]

Flow:
  1. Load src_sd from base model checkpoint.
  2. Load target HSA config, identify HSA layers.
  3. Directly transform src_sd into converted_sd using the current LHSA key layout.
  4. Create HSA model, pad vocab weights, and save converted_sd.
  5. Load converted_sd with strict=False, report missing keys.
"""

import models  # noqa: F401  -- register custom model types
import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from veomni.models import build_foundation_model


# ---------------------------------------------------------------------------
# 1. Load OLMo3 model state dict
# ---------------------------------------------------------------------------

def load_state_dict_from_path(path: str) -> Dict[str, torch.Tensor]:
    """Load state dict from a directory or single file (safetensors / bin)."""
    if os.path.isdir(path):
        safetensors_path = os.path.join(path, "model.safetensors")
        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            return load_file(safetensors_path)

        safetensors_index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(safetensors_index):
            from safetensors.torch import load_file
            with open(safetensors_index, "r", encoding="utf-8") as f:
                index = json.load(f)
            state_dict: Dict[str, torch.Tensor] = {}
            for shard in sorted(set(index["weight_map"].values())):
                state_dict.update(load_file(os.path.join(path, shard)))
            return state_dict

        bin_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(bin_path):
            return torch.load(bin_path, map_location="cpu")

        bin_index = os.path.join(path, "pytorch_model.bin.index.json")
        if os.path.exists(bin_index):
            with open(bin_index, "r", encoding="utf-8") as f:
                index = json.load(f)
            state_dict = {}
            for shard in sorted(set(index["weight_map"].values())):
                state_dict.update(torch.load(os.path.join(path, shard), map_location="cpu"))
            return state_dict

        raise FileNotFoundError(f"No model weight files found in directory: {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)

    return torch.load(path, map_location="cpu")


def load_config_dict_from_path(path: str) -> Dict[str, Any]:
    """Load config.json from a checkpoint path (file or directory)."""
    config_dir = path if os.path.isdir(path) else os.path.dirname(path)
    config_path = os.path.join(config_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2. Identify HSA layers from config
# ---------------------------------------------------------------------------

def get_lhsa_layer_indices(config: Dict[str, Any]) -> List[int]:
    """Determine which layers are LHSA (HSA) layers based on config.

    HSA layers are determined by:
      - replace_full_attention_with_lhsa must be true (default true)
      - full_attn_interleave > 0
      - layer_idx >= num_swa_layers
      - (layer_idx - num_swa_layers) % full_attn_interleave == full_attn_interleave - 1
    """
    replace = config.get("replace_full_attention_with_lhsa", True)
    if not replace:
        return []
    n_layers = config["num_hidden_layers"]
    interleave = config.get("full_attn_interleave", 0)
    num_swa = config.get("num_swa_layers", 0)
    if interleave <= 0:
        return []
    return [
        i for i in range(n_layers)
        if i >= num_swa
        and (i - num_swa) % interleave == interleave - 1
    ]




# ---------------------------------------------------------------------------
# 3. Key normalization: ensure all keys have "model." prefix
# ---------------------------------------------------------------------------

def normalize_key(key: str) -> str:
    """Normalize source key to match HiLSForCausalLM state_dict naming.

    HiLSForCausalLM structure:
      self.model = HiLSModel(...)   -> keys: model.layers.X, model.embed_tokens, model.norm, ...
      self.lm_head = nn.Linear(...) -> key:  lm_head.weight  (NO 'model.' prefix)

    Handles:
      layers.X...          -> model.layers.X...
      embed_tokens.X       -> model.embed_tokens.X
      norm.X               -> model.norm.X
      model.model.layers.X -> model.layers.X...
      model.layers.X...    -> model.layers.X... (unchanged)
      lm_head.weight       -> lm_head.weight    (unchanged, top-level param)
      model.lm_head.weight -> lm_head.weight    (strip incorrect 'model.' prefix)
    """
    # Fix double "model." prefix
    if key.startswith("model.model."):
        return key[len("model."):]  # model.model.X -> model.X

    # lm_head is a top-level param in HiLSForCausalLM (not under self.model)
    if key.startswith("model.lm_head."):
        return key[len("model."):]  # model.lm_head.X -> lm_head.X
    if key.startswith("lm_head."):
        return key  # already correct

    # All other keys should have "model." prefix (they belong to HiLSModel)
    if not key.startswith("model."):
        return "model." + key  # layers.X -> model.layers.X
    return key


# ---------------------------------------------------------------------------
# 4. Build converted_sd directly from src_sd
# ---------------------------------------------------------------------------

def build_converted_sd(
    src_sd: Dict[str, torch.Tensor],
    target_config: Dict[str, Any],
    src_config: Dict[str, Any],
    lhsa_indices: List[int],
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    """Build converted state dict directly from source state dict.

    For non-HSA keys: normalize key prefix and copy tensor as-is.
    For current HiLS layers: copy base q/k/v/o_proj directly; names and shapes match.

    New HiLS-only parameters such as lmk_q_proj and lmk_q_norm have no
    base-model source and are intentionally left randomly initialized.
    """
    converted_sd: Dict[str, torch.Tensor] = {}
    lhsa_set = set(lhsa_indices)
    hils_layer_logs: List[Dict[str, Any]] = []

    # Normalize all source keys first
    normalized_src: Dict[str, torch.Tensor] = {}
    for k, v in src_sd.items():
        nk = normalize_key(k)
        normalized_src[nk] = v

    # Handle tie_word_embeddings: if source ties but target doesn't,
    # we need to create lm_head.weight from embed_tokens.weight
    target_tie = bool(target_config.get("tie_word_embeddings", False))
    src_tie = bool(src_config.get("tie_word_embeddings", False))

    for nk, tensor in normalized_src.items():
        # Check if this key belongs to an LHSA layer's self_attn q/k/v/o_proj
        layer_idx = _extract_layer_idx(nk)
        if layer_idx is not None and layer_idx in lhsa_set:
            suffix = _extract_attn_suffix(nk)
            if suffix is not None:
                # This key is an attention projection in an LHSA layer.
                # It will be handled in the LHSA conversion pass below.
                continue

        # Non-HSA key: copy directly
        converted_sd[nk] = tensor

    # Handle tie_word_embeddings: create lm_head.weight if needed
    if not target_tie and "lm_head.weight" not in converted_sd:
        # Check if lm_head.weight exists under different name
        lm_head_key = None
        for candidate in ("lm_head.weight", "model.lm_head.weight"):
            if candidate in normalized_src:
                lm_head_key = candidate
                break
        if lm_head_key is None:
            # Use embed_tokens as fallback
            for embed_candidate in (
                "model.embed_tokens.weight",
                "model.model.embed_tokens.weight",
            ):
                nk_embed = normalize_key(embed_candidate)
                if nk_embed in normalized_src:
                    converted_sd["lm_head.weight"] = normalized_src[nk_embed].clone()
                    print(f"  [tie_word_embeddings] Created lm_head.weight from {nk_embed}")
                    break

    # Convert LHSA layers
    for layer_idx in lhsa_indices:
        prefix = f"model.layers.{layer_idx}.self_attn."

        # Find source q/k/v/o_proj for this layer
        src_q = normalized_src.get(f"{prefix}q_proj.weight")
        src_k = normalized_src.get(f"{prefix}k_proj.weight")
        src_v = normalized_src.get(f"{prefix}v_proj.weight")
        src_o = normalized_src.get(f"{prefix}o_proj.weight")

        if src_q is None or src_k is None or src_v is None or src_o is None:
            raise KeyError(
                f"Cannot find q/k/v/o_proj weights for LHSA layer {layer_idx}. "
                f"Available keys with this prefix: "
                f"{[k for k in normalized_src if k.startswith(prefix)]}"
            )

        converted_sd[f"{prefix}q_proj.weight"] = src_q
        converted_sd[f"{prefix}k_proj.weight"] = src_k
        converted_sd[f"{prefix}v_proj.weight"] = src_v
        converted_sd[f"{prefix}o_proj.weight"] = src_o

        hils_layer_logs.append({
            "layer_idx": layer_idx,
            "mode": "direct_qkv_reuse",
            "copied": ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"],
        })

    return converted_sd, hils_layer_logs


def _extract_layer_idx(key: str) -> Optional[int]:
    """Extract layer index from a key like 'model.layers.3.self_attn.q_proj.weight'.
    Returns None if key doesn't match the pattern.
    """
    parts = key.split(".")
    try:
        idx = parts.index("layers")
        return int(parts[idx + 1])
    except (ValueError, IndexError):
        return None


def _extract_attn_suffix(key: str) -> Optional[str]:
    """Extract the suffix after 'self_attn.' from a key.
    Returns None if key doesn't contain 'self_attn'.
    Only returns suffix if it's one of q_proj/k_proj/v_proj/o_proj.
    """
    marker = ".self_attn."
    pos = key.find(marker)
    if pos < 0:
        return None
    suffix = key[pos + len(marker):]
    if suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"):
        return suffix
    return None


# ---------------------------------------------------------------------------
# Vocab size padding: handle source vocab_size < target vocab_size
# ---------------------------------------------------------------------------

def pad_vocab_weights(
    converted_sd: Dict[str, torch.Tensor],
    model_sd: Dict[str, torch.Tensor],
    initializer_range: float = 0.02,
) -> List[str]:
    """Resize vocab-dimension weights in converted_sd to match model_sd shapes.

    For keys like embed_tokens.weight and lm_head.weight, if the source has
    fewer vocab entries than the target (e.g. 100278 vs 100288), the extra rows
    are randomly initialized with N(0, initializer_range) to match the model's
    _init_weights behavior (nn.Embedding / nn.Linear use normal_(mean=0, std=std)).
    Returns list of keys that were resized.
    """
    resized_keys = []
    for key in list(converted_sd.keys()):
        if key not in model_sd:
            continue
        src_tensor = converted_sd[key]
        dst_tensor = model_sd[key]
        if src_tensor.shape == dst_tensor.shape:
            continue
        # Check if this is a vocab-related key (embed_tokens or lm_head)
        is_vocab_key = any(vk in key for vk in ("embed_tokens", "lm_head"))
        if not is_vocab_key:
            continue
        # Only resize if first dim differs (vocab_size) and other dims match
        if src_tensor.dim() != dst_tensor.dim():
            continue
        if src_tensor.shape[1:] != dst_tensor.shape[1:]:
            continue
        if src_tensor.shape[0] >= dst_tensor.shape[0]:
            # Source is larger or equal, truncate to target size
            converted_sd[key] = src_tensor[:dst_tensor.shape[0]]
            resized_keys.append(f"{key}: truncated {list(src_tensor.shape)} -> {list(dst_tensor.shape)}")
        else:
            # Source is smaller, randomly init extra rows (like _init_weights in modeling)
            new_tensor = torch.empty_like(dst_tensor)
            new_tensor.normal_(mean=0.0, std=initializer_range)
            new_tensor[:src_tensor.shape[0]] = src_tensor.to(new_tensor.dtype)
            converted_sd[key] = new_tensor
            resized_keys.append(
                f"{key}: extended {list(src_tensor.shape)} -> {list(new_tensor.shape)} "
                f"(extra rows randomly initialized with std={initializer_range})"
            )
    return resized_keys


def init_external_lmk_embed(
    hidden_size: int,
    initializer_range: float = 0.02,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random-init standalone landmark embedding (same schedule as a padded embed row)."""
    lmk_embed = torch.empty(hidden_size, dtype=dtype)
    lmk_embed.normal_(mean=0.0, std=initializer_range)
    return lmk_embed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert OLMo3 base model weights to HiLS model weights"
    )
    parser.add_argument(
        "--base_path", required=True, type=str,
        help="Path to OLMo3 base model checkpoint (file or directory)"
    )
    parser.add_argument(
        "--target_config", required=True, type=str,
        help="Path to target HiLS model config JSON"
    )
    parser.add_argument(
        "--output_path", required=True, type=str,
        help="Output path for converted state dict (.pt)"
    )
    parser.add_argument(
        "--log_path", default=None, type=str,
        help="Optional path to save conversion log (JSON)"
    )
    args = parser.parse_args()

    # ---- Step 1: Load source state dict ----
    print(f"[1/5] Loading OLMo3 base model from: {args.base_path}")
    src_sd = load_state_dict_from_path(args.base_path)
    src_config = load_config_dict_from_path(args.base_path)
    print(f"       Source keys: {len(src_sd)}")

    # ---- Step 2: Load target config & identify HSA layers ----
    print(f"[2/5] Loading target HSA config from: {args.target_config}")
    with open(args.target_config, "r", encoding="utf-8") as f:
        target_config = json.load(f)

    lhsa_indices = get_lhsa_layer_indices(target_config)

    if lhsa_indices:
        print(f"       HSA layers: {lhsa_indices}")
        print("       Mode: direct_qkv_reuse")
    else:
        print("       No HSA layers (all pure Olmo3Attention)")

    # ---- Step 3: Build converted_sd directly from src_sd ----
    print(f"[3/5] Converting weights (directly from src_sd, no dst model needed)...")
    converted_sd, hils_layer_logs = build_converted_sd(
        src_sd, target_config, src_config, lhsa_indices,
    )
    print(f"       Converted keys: {len(converted_sd)}")

    if hils_layer_logs:
        print(f"       HSA layer conversion details:")
        for entry in hils_layer_logs:
            print(f"         Layer {entry['layer_idx']}: mode={entry['mode']}")

    # ---- Step 4: Pad vocab / init external lmk_embed & save converted state dict ----
    # Create HSA model first to get the actual parameter layout.
    enable_external_lmk_embed = bool(target_config.get("enable_external_lmk_embed", False))
    print(f"[4/5] Creating HSA model, preparing vocab/lmk weights, and saving...")
    print(f"       enable_external_lmk_embed={enable_external_lmk_embed}")
    model = build_foundation_model(config_path=args.target_config)

    model_sd = model.state_dict()
    init_range = target_config.get("initializer_range", 0.02)

    if enable_external_lmk_embed:
        # embed_tokens / lm_head stay at config.vocab_size (e.g. 100278).
        # Landmark vector lives in model.lmk_embed, not an extra embedding row.
        ref_key = "model.embed_tokens.weight"
        ref_dtype = converted_sd[ref_key].dtype if ref_key in converted_sd else torch.float32
        if "model.lmk_embed" in model_sd:
            converted_sd["model.lmk_embed"] = init_external_lmk_embed(
                target_config["hidden_size"], init_range, ref_dtype
            )
            print(
                f"       Initialized model.lmk_embed with Normal(0, {init_range}) "
                f"(shape={tuple(converted_sd['model.lmk_embed'].shape)})"
            )
        for vocab_key in ("model.embed_tokens.weight", "lm_head.weight"):
            if vocab_key in converted_sd and vocab_key in model_sd:
                if converted_sd[vocab_key].shape != model_sd[vocab_key].shape:
                    raise ValueError(
                        f"{vocab_key} shape mismatch for enable_external_lmk_embed=true: "
                        f"converted {list(converted_sd[vocab_key].shape)} vs model "
                        f"{list(model_sd[vocab_key].shape)}. "
                        "Do not pad vocab when using external lmk_embed."
                    )
    else:
        # Internal landmark row: resize vocab (e.g. 100278 -> 100288) and init extra rows.
        resized_keys = pad_vocab_weights(converted_sd, model_sd, initializer_range=init_range)
        if resized_keys:
            print(f"       Vocab size resizing applied (initializer_range={init_range}):")
            for rk in resized_keys:
                print(f"         {rk}")

    # Drop keys whose shape doesn't match target model (e.g. layerwise qk_norm
    # [4096] -> per-head qk_norm [128]).  These will be randomly initialized.
    shape_mismatch_keys = []
    for key in list(converted_sd.keys()):
        if key in model_sd and converted_sd[key].shape != model_sd[key].shape:
            shape_mismatch_keys.append(
                f"{key}: src {list(converted_sd[key].shape)} vs target {list(model_sd[key].shape)}"
            )
            del converted_sd[key]
    if shape_mismatch_keys:
        print(f"       Shape mismatch keys dropped (will be randomly initialized):")
        for mk in shape_mismatch_keys:
            print(f"         {mk}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(converted_sd, args.output_path)
    print(f"       Done. File size: {os.path.getsize(args.output_path) / 1e9:.2f} GB")

    # ---- Step 5: Load converted_sd into HSA model, verify missing keys ----
    print(f"[5/5] Loading converted state dict into HSA model for verification...")

    load_result = model.load_state_dict(converted_sd, strict=False)

    missing_keys = load_result.missing_keys
    unexpected_keys = load_result.unexpected_keys

    print(f"\n{'='*60}")
    print(f"Verification Summary:")
    print(f"  Total keys in converted_sd : {len(converted_sd)}")
    print(f"  Total keys in HSA model    : {len(model.state_dict())}")
    print(f"  Missing keys (random init) : {len(missing_keys)}")
    print(f"  Unexpected keys (unused)   : {len(unexpected_keys)}")
    print(f"{'='*60}")

    if missing_keys:
        print(f"\nMissing keys ({len(missing_keys)}) - these will be randomly initialized:")
        for k in missing_keys:
            print(f"  - {k}")

    if unexpected_keys:
        print(f"\nUnexpected keys ({len(unexpected_keys)}) - these are in converted_sd but not in model:")
        for k in unexpected_keys:
            print(f"  - {k}")

    if not missing_keys and not unexpected_keys:
        print(f"\n  All keys matched perfectly!")

    # ---- Save log ----
    if args.log_path:
        report = {
            "summary": {
                "total_src": len(src_sd),
                "total_converted": len(converted_sd),
                "total_model": len(model.state_dict()),
                "missing_keys": len(missing_keys),
                "unexpected_keys": len(unexpected_keys),
                "converted_hils_layers": len(hils_layer_logs),
            },
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "converted_hils_layers": hils_layer_logs,
            "tie_word_embeddings": {
                "source": src_config.get("tie_word_embeddings"),
                "target": target_config.get("tie_word_embeddings"),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.log_path)), exist_ok=True)
        with open(args.log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed log saved to: {args.log_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
