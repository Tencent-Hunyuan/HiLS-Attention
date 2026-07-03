import models
import argparse
import json
import os
from typing import Dict, Tuple, List, Any
from veomni.models import build_foundation_model
import torch


def load_state_dict_from_path(path: str) -> Dict[str, torch.Tensor]:
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
            state_dict = {}
            for shard in sorted(set(index["weight_map"].values())):
                shard_path = os.path.join(path, shard)
                state_dict.update(load_file(shard_path))
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
                shard_path = os.path.join(path, shard)
                state_dict.update(torch.load(shard_path, map_location="cpu"))
            return state_dict

        raise FileNotFoundError(f"No model weight files found in directory: {path}")

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)

    return torch.load(path, map_location="cpu")


def load_config_dict_from_path(path: str) -> Dict[str, Any]:
    config_path = path
    if not os.path.isdir(config_path):
        config_path = os.path.dirname(config_path)

    config_path = os.path.join(config_path, "config.json")
    if not os.path.exists(config_path):
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_src_key(src_sd: Dict[str, torch.Tensor], dst_key: str) -> str:
    if dst_key in src_sd:
        return dst_key
    if dst_key.startswith("model."):
        alt = dst_key[len("model.") :]
        if alt in src_sd:
            return alt
        alt2 = "model.model." + dst_key[len("model.") :]
        if alt2 in src_sd:
            return alt2
    alt3 = "model." + dst_key
    if alt3 in src_sd:
        return alt3
    return ""


def copy_or_slice(dst: torch.Tensor, src: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if dst.shape == src.shape:
        return src.clone().to(dst.dtype), True

    dst_copy = dst.clone()
    src_cast = src.to(dst.dtype)
    min_shape = tuple(min(d, s) for d, s in zip(dst.shape, src_cast.shape))
    slices = tuple(slice(0, m) for m in min_shape)
    dst_copy[slices] = src_cast[slices]
    return dst_copy, False


def resolve_weight_key(state_dict: Dict[str, torch.Tensor], *candidates: str) -> str:
    for key in candidates:
        if key in state_dict:
            return key
    return ""


def get_lhsa_layer_indices(config: Dict[str, Any]) -> List[int]:
    """Determine which layers are LHSA layers based on full_attn_interleave."""
    n_layers = config["num_hidden_layers"]
    interleave = config.get("full_attn_interleave", 0)
    num_swa = config.get("num_swa_layers", 0)
    if interleave <= 0:
        return []
    return [i for i in range(n_layers)
            if (i - num_swa) % interleave == interleave - 1 and i >= num_swa]


def is_innerx_config(config: Dict[str, Any]) -> bool:
    """Check if target config uses InnerX architecture (hsa_heads set)."""
    return config.get("hsa_heads") is not None


def is_full_hsa_config(config: Dict[str, Any]) -> bool:
    """Check if hsa_heads == num_attention_heads and hsa_denom == 1.
    In this case, no SWA q/k/v_proj exist; only hsa_q/k/v_proj are created,
    so we just need a simple rename (q_proj -> hsa_q_proj, etc.)."""
    hsa_heads = config.get("hsa_heads")
    if hsa_heads is None:
        return False
    n_heads = config["num_attention_heads"]
    return hsa_heads == n_heads


# Keys within self_attn that are handled by InnerX conversion (should not be
# copied via the generic path for LHSA layers).
INNERX_HANDLED_SUFFIXES = {
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "hsa_q_proj.weight", "hsa_k_proj.weight", "hsa_v_proj.weight",
}

# Keys that should keep their random initialization (no source equivalent).
INNERX_REINIT_SUFFIXES = {
    "q_norm.weight", "k_norm.weight",
    "lmk_q_proj.weight", "lmk_q_norm.weight",
}

# For full-HSA (hsa_denom==1): hsa_q/k/v_proj exist but no q/k/v_proj.
# We rename base model's q_proj -> hsa_q_proj, etc.
FULL_HSA_HANDLED_SUFFIXES = {
    "hsa_q_proj.weight", "hsa_k_proj.weight", "hsa_v_proj.weight", "o_proj.weight",
}

FULL_HSA_REINIT_SUFFIXES = {
    "q_norm.weight", "k_norm.weight",
    "lmk_q_proj.weight", "lmk_q_norm.weight",
}

# Rename mapping: base model key suffix -> target key suffix
FULL_HSA_RENAME_MAP = {
    "q_proj.weight": "hsa_q_proj.weight",
    "k_proj.weight": "hsa_k_proj.weight",
    "v_proj.weight": "hsa_v_proj.weight",
    "o_proj.weight": "o_proj.weight",
}


def convert_lhsa_layer_full_hsa(
    src_sd: Dict[str, torch.Tensor],
    dst_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    config: Dict[str, Any],
    converted_innerx: List[Dict[str, Any]],
):
    """Simple rename conversion for full-HSA layers (hsa_denom==1).

    When hsa_heads == num_attention_heads, there's no SWA/HSA head split.
    The base model's q/k/v_proj weights are directly renamed to hsa_q/k/v_proj.
    """
    dst_prefix = f"model.layers.{layer_idx}.self_attn"

    def _find_src(proj_name: str) -> torch.Tensor:
        candidates = [
            f"model.layers.{layer_idx}.self_attn.{proj_name}",
            f"layers.{layer_idx}.self_attn.{proj_name}",
            f"model.model.layers.{layer_idx}.self_attn.{proj_name}",
        ]
        for c in candidates:
            if c in src_sd:
                return src_sd[c]
        raise KeyError(f"Cannot find source weight for {proj_name} at layer {layer_idx}")

    renamed = {}
    for src_suffix, dst_suffix in FULL_HSA_RENAME_MAP.items():
        dst_key = f"{dst_prefix}.{dst_suffix}"
        if dst_key not in dst_sd:
            continue
        src_tensor = _find_src(src_suffix)
        new_tensor, _ = copy_or_slice(dst_sd[dst_key], src_tensor)
        dst_sd[dst_key] = new_tensor
        renamed[src_suffix] = dst_suffix

    converted_innerx.append({
        "layer_idx": layer_idx,
        "mode": "full_hsa_rename",
        "renamed": renamed,
    })


def convert_lhsa_layer_innerx(
    src_sd: Dict[str, torch.Tensor],
    dst_sd: Dict[str, torch.Tensor],
    layer_idx: int,
    config: Dict[str, Any],
    converted_innerx: List[Dict[str, Any]],
    no_permute: bool = False,
):
    """Split base model's unified QKV into SWA + HSA projections with head permutation.

    Args:
        no_permute: If True, use fixed head order (first N-hsa_heads streaming,
            last hsa_heads retrieval) instead of headwise_config permutation.
            Useful for PPL ablation to verify permutation correctness.
    """
    head_dim = config["hidden_size"] // config["num_attention_heads"]
    n_heads = config["num_attention_heads"]
    hsa_heads = config["hsa_heads"]
    hsa_qk_ratio = config.get("hsa_qk_ratio", 4)
    h_hsa_kv = hsa_heads // hsa_qk_ratio

    if no_permute:
        # Fixed order: first (n_heads - hsa_heads) streaming, last hsa_heads retrieval
        streaming_idx = list(range(n_heads - hsa_heads))
        retrieval_idx = list(range(n_heads - hsa_heads, n_heads))
        print(f"  [Layer {layer_idx}] no_permute: streaming={streaming_idx[:3]}...{streaming_idx[-1]}, "
              f"retrieval={retrieval_idx[0]}...{retrieval_idx[-1]}")
    else:
        headwise_config = config.get("headwise_config", {})
        mask = headwise_config.get(str(layer_idx))
        if mask is None:
            raise ValueError(
                f"headwise_config missing for layer {layer_idx}. "
                f"Available keys: {list(headwise_config.keys())}"
            )
        retrieval_idx = [i for i, m in enumerate(mask) if m == 1]
        streaming_idx = [i for i, m in enumerate(mask) if m == 0]

    assert len(retrieval_idx) == hsa_heads, (
        f"Layer {layer_idx}: expected {hsa_heads} retrieval heads, got {len(retrieval_idx)}"
    )

    # Determine the dst prefix (model.layers.X.self_attn) and find source keys
    dst_prefix = f"model.layers.{layer_idx}.self_attn"

    def _find_src(proj_name: str) -> torch.Tensor:
        """Find the source weight for a given projection name in the base model."""
        # Base model keys: model.layers.X.self_attn.{q,k,v,o}_proj.weight
        candidates = [
            f"model.layers.{layer_idx}.self_attn.{proj_name}.weight",
            f"layers.{layer_idx}.self_attn.{proj_name}.weight",
            f"model.model.layers.{layer_idx}.self_attn.{proj_name}.weight",
        ]
        for c in candidates:
            if c in src_sd:
                return src_sd[c]
        raise KeyError(f"Cannot find source weight for {proj_name} at layer {layer_idx}")

    def extract_head_rows(weight: torch.Tensor, indices: List[int]) -> torch.Tensor:
        return torch.cat([weight[i * head_dim:(i + 1) * head_dim] for i in indices], dim=0)

    def mean_pool_heads(weight: torch.Tensor, indices: List[int], n_groups: int) -> torch.Tensor:
        """Pool len(indices) heads into n_groups by averaging groups."""
        rows = torch.stack([weight[i * head_dim:(i + 1) * head_dim] for i in indices])
        # rows: (n_heads, head_dim, in_features)
        grouped = rows.reshape(n_groups, len(indices) // n_groups, head_dim, -1)
        return grouped.mean(dim=1).reshape(n_groups * head_dim, -1)

    # --- Q: split into SWA (streaming) and HSA (retrieval) ---
    src_q = _find_src("q_proj")
    dst_sd[f"{dst_prefix}.q_proj.weight"] = extract_head_rows(src_q, streaming_idx).to(dst_sd[f"{dst_prefix}.q_proj.weight"].dtype)
    dst_sd[f"{dst_prefix}.hsa_q_proj.weight"] = extract_head_rows(src_q, retrieval_idx).to(dst_sd[f"{dst_prefix}.hsa_q_proj.weight"].dtype)

    # --- K: SWA gets streaming heads, HSA gets mean-pooled retrieval heads ---
    src_k = _find_src("k_proj")
    dst_sd[f"{dst_prefix}.k_proj.weight"] = extract_head_rows(src_k, streaming_idx).to(dst_sd[f"{dst_prefix}.k_proj.weight"].dtype)
    dst_sd[f"{dst_prefix}.hsa_k_proj.weight"] = mean_pool_heads(src_k, retrieval_idx, h_hsa_kv).to(dst_sd[f"{dst_prefix}.hsa_k_proj.weight"].dtype)

    # --- V: same as K ---
    src_v = _find_src("v_proj")
    dst_sd[f"{dst_prefix}.v_proj.weight"] = extract_head_rows(src_v, streaming_idx).to(dst_sd[f"{dst_prefix}.v_proj.weight"].dtype)
    dst_sd[f"{dst_prefix}.hsa_v_proj.weight"] = mean_pool_heads(src_v, retrieval_idx, h_hsa_kv).to(dst_sd[f"{dst_prefix}.hsa_v_proj.weight"].dtype)

    # --- O: permute columns (streaming first, retrieval last) ---
    src_o = _find_src("o_proj")
    perm_order = streaming_idx + retrieval_idx
    col_indices = []
    for i in perm_order:
        col_indices.extend(range(i * head_dim, (i + 1) * head_dim))
    dst_sd[f"{dst_prefix}.o_proj.weight"] = src_o[:, col_indices].to(dst_sd[f"{dst_prefix}.o_proj.weight"].dtype)

    converted_innerx.append({
        "layer_idx": layer_idx,
        "no_permute": no_permute,
        "streaming_heads": streaming_idx,
        "retrieval_heads": retrieval_idx,
        "q_proj_shape": list(dst_sd[f"{dst_prefix}.q_proj.weight"].shape),
        "hsa_q_proj_shape": list(dst_sd[f"{dst_prefix}.hsa_q_proj.weight"].shape),
        "hsa_k_proj_shape": list(dst_sd[f"{dst_prefix}.hsa_k_proj.weight"].shape),
        "hsa_v_proj_shape": list(dst_sd[f"{dst_prefix}.hsa_v_proj.weight"].shape),
        "o_proj_shape": list(dst_sd[f"{dst_prefix}.o_proj.weight"].shape),
    })


def _is_innerx_attn_key(dst_key: str, lhsa_layer_indices: List[int], full_hsa: bool = False) -> bool:
    """Check if a dst_key belongs to an LHSA layer's self_attn and should be
    handled by InnerX conversion or reinitialization."""
    if full_hsa:
        handled = FULL_HSA_HANDLED_SUFFIXES
        reinit = FULL_HSA_REINIT_SUFFIXES
    else:
        handled = INNERX_HANDLED_SUFFIXES
        reinit = INNERX_REINIT_SUFFIXES
    for layer_idx in lhsa_layer_indices:
        prefix = f"model.layers.{layer_idx}.self_attn."
        if dst_key.startswith(prefix):
            suffix = dst_key[len(prefix):]
            if suffix in handled or suffix in reinit:
                return True
    return False


def main():
    parser = argparse.ArgumentParser("Convert base model params to FlashHSA params")
    parser.add_argument("--base_path", required=True, type=str, help="Base model checkpoint path (file or dir)")
    parser.add_argument("--target_config", required=True, type=str, help="Target FlashHSA config json path")
    parser.add_argument("--output_path", required=True, type=str, help="Output path for converted state_dict")
    parser.add_argument("--log_path", default=None, type=str, help="Optional log path (json)")
    parser.add_argument("--no_permute", action="store_true",
                        help="Disable head permutation for InnerX: use fixed order "
                             "(first N-hsa streaming, last hsa retrieval). "
                             "Useful for PPL ablation to verify permutation correctness.")
    args = parser.parse_args()

    src_sd = load_state_dict_from_path(args.base_path)
    src_config = load_config_dict_from_path(args.base_path)
    with open(args.target_config, "r", encoding="utf-8") as f:
        target_config = json.load(f)


    model = build_foundation_model(config_path=args.target_config)
    dst_sd = model.state_dict()

    # Determine if InnerX conversion is needed
    use_innerx = is_innerx_config(target_config)
    full_hsa = is_full_hsa_config(target_config)
    lhsa_layer_indices = get_lhsa_layer_indices(target_config) if use_innerx else []
    if use_innerx:
        mode_str = "full_hsa_rename" if full_hsa else "innerx_split"
        print(f"[InnerX] Detected hsa_heads={target_config['hsa_heads']}, "
              f"mode={mode_str}, LHSA layers: {lhsa_layer_indices}, no_permute={args.no_permute}")

    used_src_keys = set()
    copied_exact = []
    copied_sliced = []
    missing_in_src = []
    converted_innerx = []
    tie_word_embeddings_fixups = []

    for dst_key, dst_tensor in dst_sd.items():
        # Skip keys that will be handled by InnerX conversion or kept as random init
        if use_innerx and _is_innerx_attn_key(dst_key, lhsa_layer_indices, full_hsa=full_hsa):
            continue

        src_key = find_src_key(src_sd, dst_key)
        if not src_key:
            missing_in_src.append(dst_key)
            continue

        src_tensor = src_sd[src_key]
        new_tensor, is_exact = copy_or_slice(dst_tensor, src_tensor)
        dst_sd[dst_key] = new_tensor
        used_src_keys.add(src_key)
        if is_exact:
            copied_exact.append(dst_key)
        else:
            copied_sliced.append({
                "dst_key": dst_key,
                "dst_shape": tuple(dst_tensor.shape),
                "src_key": src_key,
                "src_shape": tuple(src_tensor.shape),
            })

    # Perform InnerX conversion for LHSA layers
    if use_innerx:
        for layer_idx in lhsa_layer_indices:
            if full_hsa:
                convert_lhsa_layer_full_hsa(
                    src_sd, dst_sd, layer_idx, target_config, converted_innerx,
                )
            else:
                convert_lhsa_layer_innerx(
                    src_sd, dst_sd, layer_idx, target_config, converted_innerx,
                    no_permute=args.no_permute,
                )
            # Mark source keys as used
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                for prefix_fmt in (
                    "model.layers.{}.self_attn.{}.weight",
                    "layers.{}.self_attn.{}.weight",
                    "model.model.layers.{}.self_attn.{}.weight",
                ):
                    candidate = prefix_fmt.format(layer_idx, proj)
                    if candidate in src_sd:
                        used_src_keys.add(candidate)

    target_tie = bool(target_config.get("tie_word_embeddings", False))
    src_tie = bool(src_config.get("tie_word_embeddings", False))
    if not target_tie:
        lm_head_key = resolve_weight_key(dst_sd, "lm_head.weight", "model.lm_head.weight")
        if lm_head_key:
            lm_head_missing = lm_head_key in missing_in_src
            if lm_head_missing:
                embed_src_key = resolve_weight_key(
                    src_sd,
                    "model.embed_tokens.weight",
                    "embed_tokens.weight",
                    "model.model.embed_tokens.weight",
                )
                if not embed_src_key:
                    raise ValueError(
                        "target tie_word_embeddings=false requires lm_head.weight, but source checkpoint "
                        "has neither lm_head.weight nor embed_tokens.weight to use as fallback."
                    )

                new_tensor, is_exact = copy_or_slice(dst_sd[lm_head_key], src_sd[embed_src_key])
                dst_sd[lm_head_key] = new_tensor
                used_src_keys.add(embed_src_key)
                missing_in_src.remove(lm_head_key)
                fixup = {
                    "dst_key": lm_head_key,
                    "src_key": embed_src_key,
                    "src_tie_word_embeddings": src_tie,
                    "reason": "target is untied and source lm_head.weight is missing; initialized from embed_tokens.weight",
                    "copy_type": "exact" if is_exact else "sliced",
                }
                tie_word_embeddings_fixups.append(fixup)
                if is_exact:
                    copied_exact.append(lm_head_key)
                else:
                    copied_sliced.append({
                        "dst_key": lm_head_key,
                        "dst_shape": tuple(dst_sd[lm_head_key].shape),
                        "src_key": embed_src_key,
                        "src_shape": tuple(src_sd[embed_src_key].shape),
                    })
            elif not resolve_weight_key(src_sd, "lm_head.weight", "model.lm_head.weight"):
                tie_word_embeddings_fixups.append({
                    "dst_key": lm_head_key,
                    "src_key": None,
                    "src_tie_word_embeddings": src_tie,
                    "reason": "target is untied but source lm_head.weight is not separately present; existing mapping handled by key fallback",
                    "copy_type": "info",
                })

    extra_in_src = [k for k in src_sd.keys() if k not in used_src_keys]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(dst_sd, args.output_path)

    report = {
        "copied_exact": copied_exact,
        "copied_sliced": copied_sliced,
        "missing_in_src": missing_in_src,
        "extra_in_src": extra_in_src,
        "converted_innerx": converted_innerx,
        "tie_word_embeddings": {
            "source": src_config.get("tie_word_embeddings"),
            "target": target_config.get("tie_word_embeddings"),
            "fixups": tie_word_embeddings_fixups,
        },
        "summary": {
            "total_dst": len(dst_sd),
            "total_src": len(src_sd),
            "copied_exact": len(copied_exact),
            "copied_sliced": len(copied_sliced),
            "missing_in_src": len(missing_in_src),
            "extra_in_src": len(extra_in_src),
            "converted_innerx_layers": len(converted_innerx),
            "no_permute": args.no_permute if use_innerx else None,
            "tie_fixups": len(tie_word_embeddings_fixups),
        },
    }

    print("[Summary]", report["summary"])
    if args.log_path:
        os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
        with open(args.log_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {args.log_path}")


if __name__ == "__main__":
    main()
