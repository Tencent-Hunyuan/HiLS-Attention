#!/usr/bin/env python3
"""
Inspect tool for olmo3_mix dataloader.

Checks:
  1. YAML parsing: source names, weights, directory paths
  2. Data loading: each source's .data files, token counts
  3. Mix ratio verification: sample N items, count source hits vs expected weights
  4. Token decoding: decode a few samples to verify tokenization is sane

Usage:
  python code_exp/inspect_mix_dataloader.py [--yaml PATH] [--num_samples N] [--seq_len L] [--vocab_dir DIR]
"""

import argparse
import os
import sys
import yaml
import numpy as np
from collections import Counter

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "..", "veomni_src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect olmo3_mix dataloader")
    parser.add_argument(
        "--yaml",
        default=os.path.join(PROJECT_ROOT, "scripts/mid_train/dolmino_mix.yaml"),
        help="Path to dolmino_mix.yaml",
    )
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to draw for ratio check")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length")
    parser.add_argument(
        "--vocab_dir",
        default=os.path.join(PROJECT_ROOT, "configs/olmo3_vocab"),
        help="Path to tokenizer vocab dir (for decoding check)",
    )
    parser.add_argument("--decode_samples", type=int, default=3, help="Number of samples to decode and print")
    parser.add_argument("--skip_load", action="store_true", help="Only check YAML, skip loading .data files")
    parser.add_argument("--phase", type=str, default=None,
                        help="Run specific phases only, e.g. '4' or '1,4,5' (comma-separated)")
    return parser.parse_args()


def check_yaml(yaml_path):
    """Phase 1: Parse and validate YAML config."""
    print("=" * 80)
    print("Phase 1: YAML Validation")
    print("=" * 80)

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg["data_root"]
    sources = cfg["sources"]

    print(f"  data_root: {data_root}")
    print(f"  data_root exists: {os.path.isdir(data_root)}")
    print(f"  num sources: {len(sources)}")

    total_weight = sum(s["weight"] for s in sources)
    print(f"  total weight: {total_weight:.6f} (should be ~1.0)")

    # Check each source
    print(f"\n  {'#':>3s}  {'name':<30s}  {'weight':>8s}  {'dirs':>4s}  {'dirs_exist':>10s}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*8}  {'-'*4}  {'-'*10}")

    missing_dirs = []
    for i, src in enumerate(sources):
        dirs = src["dirs"]
        exist_count = sum(1 for d in dirs if os.path.isdir(os.path.join(data_root, d)))
        status = f"{exist_count}/{len(dirs)}"
        if exist_count < len(dirs):
            status += " ⚠️"
            for d in dirs:
                full = os.path.join(data_root, d)
                if not os.path.isdir(full):
                    missing_dirs.append((src["name"], full))
        print(f"  {i+1:3d}  {src['name']:<30s}  {src['weight']:8.4f}  {len(dirs):4d}  {status:>10s}")

    if missing_dirs:
        print(f"\n  ⚠️  Missing directories ({len(missing_dirs)}):")
        for name, path in missing_dirs[:20]:
            print(f"     {name}: {path}")
        if len(missing_dirs) > 20:
            print(f"     ... and {len(missing_dirs) - 20} more")

    # Category breakdown
    print(f"\n  Category breakdown:")
    cats = {}
    for s in sources:
        n = s["name"]
        w = s["weight"]
        if any(x in n for x in ["dolmino-math", "cranemath", "megamatt", "tinymath", "omr-rewrite"]):
            cat = "Math"
        elif any(x in n for x in ["code-fim", "cranecode"]):
            cat = "Code"
        elif any(x in n for x in ["general-reasoning", "program-verifiable", "math-meta", "code-meta"]):
            cat = "Reasoning"
        elif any(x in n for x in ["nemotron", "wiki-rcqa", "reddit"]):
            cat = "QA+Reddit"
        elif any(x in n for x in ["tulu", "dolmino-flan"]):
            cat = "Instruction"
        elif "pdf" in n:
            cat = "PDF"
        elif "web" in n or "stem" in n:
            cat = "Web+STEM"
        else:
            cat = "Other"
        cats[cat] = cats.get(cat, 0) + w
    for c, w in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {c:<15s} {w*100:6.2f}%")

    print()
    return cfg


def load_mix_dataset(cfg, seq_len):
    """Phase 2: Load all sources, build MixedTextDataset."""
    print("=" * 80)
    print("Phase 2: Loading Data")
    print("=" * 80)

    from data.mix_dataset import MixedTextDataset
    from data.lazy_dataset import LazyChunkedLoader
    from itertools import accumulate
    from tqdm import tqdm

    data_root = cfg["data_root"]
    sources = cfg["sources"]

    loaders = []
    weights = []
    names = []

    for src in sources:
        dirs = src["dirs"]
        full_dirs = [os.path.join(data_root, d) for d in dirs]

        all_files = []
        for d in full_dirs:
            if not os.path.isdir(d):
                continue
            for root, _, files in os.walk(d, followlinks=True):
                for file in files:
                    if file.endswith(".data"):
                        all_files.append(os.path.join(root, file))

        if not all_files:
            print(f"  ⚠️  {src['name']}: no .data files found, skipping")
            continue

        # Build loader manually (same as __init__.py)
        loader = LazyChunkedLoader.__new__(LazyChunkedLoader)
        loader.array_data_type = np.uint32
        loader.is_lazy = True
        loader.split = "train"
        loader.val_ratio = 0.1

        files_ptrs = []
        files_lens = []
        for fpath in tqdm(all_files, desc=f"  {src['name']}", ncols=100, leave=False):
            try:
                fsize = os.path.getsize(fpath)
                if fsize > 0:
                    np_array = np.memmap(fpath, dtype=np.uint32, mode="r")
                    ids_len = np_array.shape[0]
                    end = int(0.9 * ids_len)
                    files_ptrs.append(np_array[:end])
                    files_lens.append(files_ptrs[-1].shape[0])
            except Exception as e:
                print(f"    Error: {fpath}: {e}")

        if not files_ptrs:
            print(f"  ⚠️  {src['name']}: all files empty, skipping")
            continue

        loader.files_ptrs = files_ptrs
        loader.lens = files_lens
        loader.ends = list(accumulate(files_lens))
        loader.total_tokens = loader.ends[-1]

        loaders.append(loader)
        weights.append(src["weight"])
        names.append(src["name"])
        print(f"  ✅ {src['name']:<30s}  {len(files_lens):>5d} files  {loader.total_tokens:>15,d} tokens")

    # Re-normalize
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    total_tokens = sum(ld.total_tokens for ld in loaders)
    print(f"\n  Active sources: {len(loaders)}")
    print(f"  Total tokens:   {total_tokens:,}")
    print(f"  Dataset length: {total_tokens // seq_len:,} (seq_len={seq_len})")

    dataset = MixedTextDataset(loaders, weights, names, seq_len)
    print()
    return dataset, names, weights


def check_mix_ratio(dataset, names, weights, num_samples):
    """Phase 3: Sample N items, verify source distribution matches weights."""
    print("=" * 80)
    print(f"Phase 3: Mix Ratio Verification ({num_samples:,} samples)")
    print("=" * 80)

    import random as py_random

    counter = Counter()
    for idx in tqdm(range(num_samples), desc="  Sampling", ncols=100):
        # Reproduce the same RNG as MixedTextDataset.__getitem__
        rng = py_random.Random(idx)
        np_rng = np.random.RandomState(seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])
        source_idx = dataset._pick_source(np_rng)
        counter[source_idx] += 1

    print(f"\n  {'#':>3s}  {'source':<30s}  {'expected':>8s}  {'observed':>8s}  {'diff':>8s}  {'status':>6s}")
    print(f"  {'-'*3}  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")

    max_diff = 0
    for i, (name, expected_w) in enumerate(zip(names, weights)):
        observed = counter.get(i, 0) / num_samples
        diff = observed - expected_w
        max_diff = max(max_diff, abs(diff))
        status = "✅" if abs(diff) < 0.02 else "⚠️"
        print(
            f"  {i+1:3d}  {name:<30s}  {expected_w*100:7.2f}%  {observed*100:7.2f}%  {diff*100:+7.2f}%  {status:>6s}"
        )

    print(f"\n  Max absolute deviation: {max_diff*100:.3f}%")
    if max_diff < 0.02:
        print(f"  ✅ Mix ratios look correct (max dev < 2%)")
    elif max_diff < 0.05:
        print(f"  ⚠️  Mix ratios have moderate deviation (max dev < 5%), consider more samples")
    else:
        print(f"  ❌ Mix ratios have large deviation (max dev >= 5%), check weights!")
    print()


def check_token_decode(dataset, vocab_dir, num_decode):
    """Phase 4: Decode a few samples to verify tokenization."""
    print("=" * 80)
    print(f"Phase 4: Token Decode Check ({num_decode} samples)")
    print("=" * 80)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(vocab_dir, trust_remote_code=True)
    except Exception as e:
        print(f"  ⚠️  Cannot load tokenizer from {vocab_dir}: {e}")
        print(f"  Skipping decode check.")
        print()
        return

    for sample_idx in range(num_decode):
        tokens = dataset[sample_idx]
        source_name = "unknown"

        # Reproduce RNG to find which source was picked
        import random as py_random

        rng = py_random.Random(sample_idx)
        np_rng = np.random.RandomState(seed=[rng.randint(0, 2**32 - 1) for _ in range(16)])
        source_idx = dataset._pick_source(np_rng)
        source_name = dataset.names[source_idx]

        print(f"\n  --- Sample {sample_idx} (source: {source_name}) ---")
        print(f"  shape: {tokens.shape}, dtype: {tokens.dtype}")
        print(f"  token range: [{tokens.min()}, {tokens.max()}]")

        # Check for invalid tokens
        vocab_size = tokenizer.vocab_size
        invalid_count = int((tokens < 0).sum() + (tokens >= vocab_size).sum())
        if invalid_count > 0:
            print(f"  ❌ {invalid_count} tokens out of vocab range [0, {vocab_size})")
        else:
            print(f"  ✅ All tokens in valid range [0, {vocab_size})")

        # Decode first 200 tokens
        token_list = tokens[:200].tolist()
        decoded = tokenizer.decode(token_list, skip_special_tokens=False)
        print(f"  [First 200 tokens decoded ({len(decoded)} chars)]:")
        # Show first 500 chars
        preview = decoded[:500].replace("\n", "\\n").replace("\r", "\\r")
        print(f"  {preview}")
        if len(decoded) > 500:
            print(f"  ... ({len(decoded) - 500} more chars)")

    print()


def check_flat_vs_mix(cfg, seq_len, num_samples=5000):
    """Phase 5: Compare flat loading ratio (by token count) vs YAML weights."""
    print("=" * 80)
    print("Phase 5: Flat vs Mix Ratio Comparison")
    print("=" * 80)

    data_root = cfg["data_root"]
    sources = cfg["sources"]

    # Count actual tokens per source (quick estimate from file sizes)
    source_tokens = {}
    total_tokens = 0
    for src in sources:
        dirs = src["dirs"]
        src_bytes = 0
        for d in dirs:
            full = os.path.join(data_root, d)
            if not os.path.isdir(full):
                continue
            for root, _, files in os.walk(full, followlinks=True):
                for file in files:
                    if file.endswith(".data"):
                        src_bytes += os.path.getsize(os.path.join(root, file))
        # uint32 = 4 bytes per token, train split = 90%
        src_tokens = int(src_bytes / 4 * 0.9)
        source_tokens[src["name"]] = src_tokens
        total_tokens += src_tokens

    if total_tokens == 0:
        print("  ⚠️  No data found, skipping comparison")
        print()
        return

    print(f"\n  {'source':<30s}  {'YAML wt':>8s}  {'flat wt':>8s}  {'diff':>8s}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}  {'-'*8}")

    for src in sources:
        name = src["name"]
        yaml_w = src["weight"]
        flat_w = source_tokens.get(name, 0) / total_tokens if total_tokens > 0 else 0
        diff = flat_w - yaml_w
        print(f"  {name:<30s}  {yaml_w*100:7.2f}%  {flat_w*100:7.2f}%  {diff*100:+7.2f}%")

    print(f"\n  Total tokens across all sources: {total_tokens:,}")
    print(f"  If these match, flat loading (olmo3) and YAML loading (olmo3_mix) give same ratio.")
    print()


def main():
    args = parse_args()

    print("\n" + "🔍 " * 20)
    print("  olmo3_mix Dataloader Inspector")
    print("🔍 " * 20 + "\n")
    print(f"  YAML:        {args.yaml}")
    print(f"  seq_len:     {args.seq_len}")
    print(f"  num_samples: {args.num_samples}")
    print(f"  vocab_dir:   {args.vocab_dir}")
    print()

    # Parse phases to run
    phases = set()
    if args.phase:
        phases = set(int(p.strip()) for p in args.phase.split(","))
    elif args.skip_load:
        phases = {1, 5}
    else:
        phases = {1, 2, 3, 4, 5}

    # Phase 1: YAML validation
    cfg = check_yaml(args.yaml) if 1 in phases else check_yaml(args.yaml)  # always needed for cfg

    # Phase 2-4 need dataset loaded
    dataset, names, weights = None, None, None
    if phases & {2, 3, 4}:
        dataset, names, weights = load_mix_dataset(cfg, args.seq_len)

    if 3 in phases and dataset is not None:
        check_mix_ratio(dataset, names, weights, args.num_samples)

    if 4 in phases and dataset is not None:
        check_token_decode(dataset, args.vocab_dir, args.decode_samples)

    if 5 in phases:
        check_flat_vs_mix(cfg, args.seq_len)

    print("=" * 80)
    print("Done! All checks completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
