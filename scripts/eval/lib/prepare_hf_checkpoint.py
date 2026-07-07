#!/usr/bin/env python3
"""Symlink HF weights + tokenizer and write merged config.json for OpenCompass."""

import json
import os
import sys


def detect_tokenizer_vocab_size(tokenizer_dir, default):
    p = os.path.join(tokenizer_dir, "vocab.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return len(json.load(f))
    p = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            tok = json.load(f)
        vocab = tok.get("model", {}).get("vocab", {})
        if vocab:
            return len(vocab)
    return int(default)


def main():
    hf_dir, hils_path, tokenizer_dir, dst_path = sys.argv[1:5]
    with open(hils_path, "r", encoding="utf-8") as f:
        hils = json.load(f)

    base_cfg_path = os.path.join(hf_dir, "config.json")
    if os.path.exists(base_cfg_path):
        with open(base_cfg_path, "r", encoding="utf-8") as f:
            base = json.load(f)
        config = dict(hils)
        for key in ("torch_dtype", "transformers_version"):
            if key in base and key not in config:
                config[key] = base[key]
    else:
        config = dict(hils)

    config["insert_landmarks"] = bool(config.get("insert_landmarks") or config.get("adjust_lmk_pos"))
    config["adjust_lmk_pos"] = bool(config.get("adjust_lmk_pos", False))
    config["vocab_size"] = detect_tokenizer_vocab_size(tokenizer_dir, config.get("vocab_size", 0))

    if os.path.lexists(dst_path):
        os.remove(dst_path)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"[config] model_type={config.get('model_type')}, arch={config.get('architectures')}, "
        f"vocab_size={config.get('vocab_size')}, insert_lmk={config['insert_landmarks']}, "
        f"adjust_lmk_pos={config['adjust_lmk_pos']}, hils_config={hils_path}"
    )


if __name__ == "__main__":
    main()
