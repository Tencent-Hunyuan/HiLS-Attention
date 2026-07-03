#!/usr/bin/env python3
"""
Merge many small .data files (uint32 numpy) into fewer large shards.

Usage:
    # Merge all source directories under data_root (dry-run first):
    python scripts/merge_data_shards.py \
        --data_root /data/dolmino-midtrain-100B-tokenized \
        --target_shard_gb 2 \
        --dry_run

    # Actually run:
    python scripts/merge_data_shards.py \
        --data_root /data/dolmino-midtrain-100B-tokenized \
        --target_shard_gb 2

    # Only merge specific directories:
    python scripts/merge_data_shards.py \
        --data_root /data/dolmino-midtrain-100B-tokenized \
        --dirs ingredient1-olmocr_science_pdfs-high_quality-crime_law-2e12 \
               ingredient1-common_crawl-high-quality_19_crime_and_law \
        --target_shard_gb 2

What it does:
    1. Scans each source dir for *.data files
    2. Skips dirs that are already "healthy" (few large files)
    3. Concatenates small files into shards of ~target_shard_gb each
    4. Writes merged shards as  merged_shard_XXXXX.data
    5. Moves original small files into  _original_small_files/  (not deleted)
    6. Verifies token count is preserved

The merged files are plain uint32 arrays, fully compatible with LazyChunkedLoader.
"""

import argparse
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_data_files(directory: str) -> list[str]:
    """Collect all .data files (non-recursively for safety, then fall back to walk)."""
    files = []
    for root, _, fnames in os.walk(directory):
        for f in fnames:
            if f.endswith(".data") and not f.startswith("merged_shard_"):
                files.append(os.path.join(root, f))
    files.sort()
    return files


def analyze_directory(directory: str, files: list[str]) -> dict:
    """Get stats for a source directory."""
    sizes = []
    for f in files:
        try:
            sizes.append(os.path.getsize(f))
        except OSError:
            sizes.append(0)
    total_bytes = sum(sizes)
    avg_bytes = total_bytes / len(sizes) if sizes else 0
    return {
        "num_files": len(files),
        "total_bytes": total_bytes,
        "avg_bytes": avg_bytes,
        "min_bytes": min(sizes) if sizes else 0,
        "max_bytes": max(sizes) if sizes else 0,
    }


def should_merge(stats: dict, target_shard_bytes: int) -> bool:
    """Decide whether a directory needs merging."""
    # Skip if few files already
    if stats["num_files"] <= 20:
        return False
    # Skip if average file size is already large (> half target shard size)
    if stats["avg_bytes"] > target_shard_bytes * 0.5:
        return False
    return True


def merge_directory(
    directory: str,
    files: list[str],
    target_shard_bytes: int,
    dry_run: bool = False,
) -> dict:
    """
    Merge small .data files into larger shards.
    Returns stats about the merge.
    """
    stats = analyze_directory(directory, files)

    if not should_merge(stats, target_shard_bytes):
        return {
            "action": "skipped",
            "reason": f"already healthy ({stats['num_files']} files, avg {stats['avg_bytes']/1024/1024:.1f}MB)",
            **stats,
        }

    # Calculate how many shards we'll produce
    num_shards = max(1, int(np.ceil(stats["total_bytes"] / target_shard_bytes)))

    if dry_run:
        return {
            "action": "would_merge",
            "num_shards": num_shards,
            "shard_size_gb": target_shard_bytes / 1024**3,
            **stats,
        }

    # --- Actually merge ---
    backup_dir = os.path.join(directory, "_original_small_files")
    os.makedirs(backup_dir, exist_ok=True)

    shard_idx = 0
    current_arrays = []
    current_bytes = 0
    total_tokens_original = 0
    total_tokens_merged = 0
    merged_files = []

    def flush_shard():
        nonlocal shard_idx, current_arrays, current_bytes, total_tokens_merged
        if not current_arrays:
            return
        merged = np.concatenate(current_arrays)
        total_tokens_merged += len(merged)
        shard_path = os.path.join(directory, f"merged_shard_{shard_idx:05d}.data")
        merged.tofile(shard_path)
        merged_files.append(shard_path)
        shard_idx += 1
        current_arrays = []
        current_bytes = 0

    for fpath in tqdm(files, desc=f"Merging {os.path.basename(directory)}", ncols=120):
        try:
            fsize = os.path.getsize(fpath)
            if fsize == 0:
                continue
            arr = np.fromfile(fpath, dtype=np.uint32)
            total_tokens_original += len(arr)
            current_arrays.append(arr)
            current_bytes += fsize

            if current_bytes >= target_shard_bytes:
                flush_shard()
        except Exception as e:
            print(f"  Warning: failed to read {fpath}: {e}")

    # Flush remaining
    flush_shard()

    # Verify token count
    assert total_tokens_original == total_tokens_merged, (
        f"Token count mismatch! original={total_tokens_original}, merged={total_tokens_merged}"
    )

    # Move originals to backup
    for fpath in files:
        fname = os.path.relpath(fpath, directory)
        dest = os.path.join(backup_dir, fname)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            shutil.move(fpath, dest)
        except Exception as e:
            print(f"  Warning: failed to move {fpath} to backup: {e}")

    return {
        "action": "merged",
        "original_files": stats["num_files"],
        "merged_shards": shard_idx,
        "total_tokens": total_tokens_merged,
        "total_bytes": stats["total_bytes"],
        "tokens_verified": total_tokens_original == total_tokens_merged,
    }


def main():
    parser = argparse.ArgumentParser(description="Merge small .data shards into larger ones")
    parser.add_argument("--data_root", required=True, help="Root directory of tokenized data")
    parser.add_argument("--dirs", nargs="*", default=None,
                        help="Specific subdirectories to merge (default: all)")
    parser.add_argument("--target_shard_gb", type=float, default=2.0,
                        help="Target shard size in GB (default: 2)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print what would be done, don't actually merge")
    args = parser.parse_args()

    target_shard_bytes = int(args.target_shard_gb * 1024**3)
    data_root = args.data_root

    if not os.path.isdir(data_root):
        print(f"Error: {data_root} is not a directory")
        sys.exit(1)

    # Determine which directories to process
    if args.dirs:
        subdirs = [os.path.join(data_root, d) for d in args.dirs]
    else:
        subdirs = sorted([
            os.path.join(data_root, d)
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d)) and not d.startswith("_")
        ])

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Merging shards in {data_root}")
    print(f"Target shard size: {args.target_shard_gb:.1f} GB")
    print(f"Directories to process: {len(subdirs)}\n")

    results = {}
    total_original_files = 0
    total_merged_shards = 0

    for subdir in subdirs:
        name = os.path.basename(subdir)
        files = get_data_files(subdir)
        if not files:
            print(f"  {name}: no .data files, skipping")
            continue

        result = merge_directory(subdir, files, target_shard_bytes, dry_run=args.dry_run)
        results[name] = result

        if result["action"] == "skipped":
            print(f"  ⏭  {name}: {result['reason']}")
        elif result["action"] == "would_merge":
            print(
                f"  📦 {name}: {result['num_files']} files → ~{result['num_shards']} shards "
                f"(total {result['total_bytes']/1024**3:.2f} GB)"
            )
            total_original_files += result["num_files"]
            total_merged_shards += result["num_shards"]
        elif result["action"] == "merged":
            verified = "✓" if result["tokens_verified"] else "✗ MISMATCH"
            print(
                f"  ✅ {name}: {result['original_files']} files → {result['merged_shards']} shards "
                f"({result['total_tokens']:,} tokens, {verified})"
            )
            total_original_files += result["original_files"]
            total_merged_shards += result["merged_shards"]

    print(f"\n{'=' * 60}")
    if args.dry_run:
        print(f"[DRY RUN] Would merge {total_original_files} files → ~{total_merged_shards} shards")
        print(f"Run without --dry_run to execute.")
    else:
        print(f"Done! {total_original_files} files → {total_merged_shards} shards")
        print(f"Originals backed up to _original_small_files/ in each directory.")
        print(f"To delete backups after verifying: find {data_root} -name '_original_small_files' -exec rm -rf {{}} +")


if __name__ == "__main__":
    main()
