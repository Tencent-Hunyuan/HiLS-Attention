"""Print every parameter / buffer key in a checkpoint.

Accepts any of:
    * a single ``pytorch_model.bin``
    * a single ``*.safetensors`` file
    * a directory containing one of:
        - ``model.safetensors``
        - ``model.safetensors.index.json``  (sharded safetensors)
        - ``pytorch_model.bin``
        - ``pytorch_model.bin.index.json``  (sharded torch)

Without flags it prints just keys, one per line, in original (load) order.
With ``--shape`` it also prints dtype + shape; with ``--regex`` it filters
to keys matching the given pattern (``re.search`` semantics).

Examples
--------
    # full key list
    python code_exp/inspect_ckpt_keys.py /path/to/ckpt

    # only ALiBi-related keys, with shape & dtype
    python code_exp/inspect_ckpt_keys.py /path/to/ckpt --regex 'alibi|inv_freqs' --shape

    # sort alphabetically and emit just the count + a sample
    python code_exp/inspect_ckpt_keys.py /path/to/ckpt --sort --head 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Tuple

import torch


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _load_safetensors(path: str) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    return load_file(path)


def _iter_shards_from_index(index_path: str) -> List[str]:
    """Return shard filenames (in stable sorted order) listed by an HF index."""
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    return sorted(set(index["weight_map"].values()))


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a state_dict from a file or a HF-style directory.

    Mirrors the format-detection logic of ``utils/convert_basemodel_to_flashhsa``
    so the inspector accepts whatever the rest of the project produces.
    """
    if os.path.isdir(path):
        candidates: List[Tuple[str, str]] = [
            ("model.safetensors", "safetensors"),
            ("model.safetensors.index.json", "safetensors_index"),
            ("pytorch_model.bin", "torch"),
            ("pytorch_model.bin.index.json", "torch_index"),
        ]
        for fname, kind in candidates:
            full = os.path.join(path, fname)
            if not os.path.exists(full):
                continue
            if kind == "safetensors":
                return _load_safetensors(full)
            if kind == "safetensors_index":
                sd: Dict[str, torch.Tensor] = {}
                for shard in _iter_shards_from_index(full):
                    sd.update(_load_safetensors(os.path.join(path, shard)))
                return sd
            if kind == "torch":
                return torch.load(full, map_location="cpu")
            if kind == "torch_index":
                sd = {}
                for shard in _iter_shards_from_index(full):
                    sd.update(torch.load(os.path.join(path, shard), map_location="cpu"))
                return sd
        raise FileNotFoundError(
            f"No model weight files found under directory: {path}\n"
            f"Looked for: {[c[0] for c in candidates]}"
        )

    if path.endswith(".safetensors"):
        return _load_safetensors(path)
    return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _humanize_numel(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if n < 1000:
            return f"{n:.1f}{unit}" if isinstance(n, float) else f"{n}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def _fmt_tensor(t: torch.Tensor) -> str:
    return f"{str(t.dtype).replace('torch.', ''):>9}  shape={tuple(t.shape)}"


def emit(
    keys: Iterable[str],
    state_dict: Dict[str, torch.Tensor],
    *,
    show_shape: bool,
) -> None:
    if show_shape:
        for k in keys:
            print(f"{k}    {_fmt_tensor(state_dict[k])}")
    else:
        for k in keys:
            print(k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="List every key in a model checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "path",
        help="Path to a checkpoint file (.bin / .safetensors) or to a "
             "directory containing one (HF-style layout).",
    )
    ap.add_argument(
        "--regex", "-r", default=None,
        help="Only show keys matching this regex (re.search semantics, "
             "i.e. unanchored substring match).",
    )
    ap.add_argument(
        "--shape", "-s", action="store_true",
        help="Also print each tensor's dtype and shape.",
    )
    ap.add_argument(
        "--sort", action="store_true",
        help="Sort keys alphabetically (default: original load order).",
    )
    ap.add_argument(
        "--head", type=int, default=None, metavar="N",
        help="Print only the first N keys after filtering / sorting.",
    )
    ap.add_argument(
        "--summary-only", action="store_true",
        help="Skip per-key listing; print only the totals at the end.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    print(f"[inspect_ckpt_keys] loading {args.path}", file=sys.stderr)
    state_dict = load_state_dict(args.path)
    if not isinstance(state_dict, dict):
        raise SystemExit(
            f"[inspect_ckpt_keys] expected dict-like state_dict at top level, "
            f"got {type(state_dict).__name__}.  If this is a training-framework "
            f"ckpt (e.g. {{'model': ..., 'optim': ...}}), pass the model "
            f"sub-dict separately."
        )

    keys = list(state_dict.keys())
    total = len(keys)

    if args.regex is not None:
        try:
            pat = re.compile(args.regex)
        except re.error as e:
            raise SystemExit(f"[inspect_ckpt_keys] invalid regex {args.regex!r}: {e}")
        keys = [k for k in keys if pat.search(k)]

    if args.sort:
        keys.sort()
    if args.head is not None:
        keys = keys[: args.head]

    if not args.summary_only:
        emit(keys, state_dict, show_shape=args.shape)

    # Always print a trailing summary on stderr so it's easy to grep stdout
    # without losing the totals.
    total_params = sum(
        v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
    )
    shown_params = sum(
        state_dict[k].numel() for k in keys if isinstance(state_dict[k], torch.Tensor)
    )
    print(
        f"\n[inspect_ckpt_keys] shown {len(keys)} / {total} keys  "
        f"({_humanize_numel(shown_params)} / {_humanize_numel(total_params)} elems)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
