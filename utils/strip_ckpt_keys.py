"""Strip state-dict keys matching a set of patterns from a checkpoint.

Typical use case: a ``pytorch_model.bin`` was saved with extra/legacy buffers
(e.g. RoPE ``inv_freqs``, persistent ALiBi slopes, or experiment-only side
modules) that you want gone before loading into a refactored model so
``strict=True`` doesn't error out.

Pattern syntax
--------------
Each ``--pattern`` is first tried as a Python regex; if that fails to
compile we fall back to shell-style glob (``fnmatch``).  So both of these
work for "drop everything under layers.3":

    --pattern '.*\\.layers\\.3\\..*'   # regex
    --pattern '*.layers.3.*'           # glob (fallback)

In both cases the pattern is matched against the FULL key (anchored at
both ends).

Usage
-----
    # show which keys would be removed (no file written)
    python utils/strip_ckpt_keys.py path/to/pytorch_model.bin \\
        --pattern '*.inv_freqs' --pattern '*.alibi_slopes' \\
        --dry-run

    # write filtered ckpt next to the input
    python utils/strip_ckpt_keys.py path/to/pytorch_model.bin \\
        --pattern '*.inv_freqs' \\
        --output path/to/pytorch_model.stripped.bin

    # in-place (overwrites input; original kept under .bak suffix unless
    # --no-backup is given)
    python utils/strip_ckpt_keys.py path/to/pytorch_model.bin \\
        --pattern '*.lmk_q_proj.*' \\
        --inplace

The script is single-file checkpoint only -- it does not currently rewrite a
sharded ``pytorch_model.bin.index.json``; for sharded ckpts merge them first
or extend this script.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import sys
from typing import Dict, List, Pattern, Tuple

import torch


def compile_patterns(patterns: List[str]) -> List[Tuple[Pattern[str], str]]:
    """Compile each pattern as a regex, falling back to glob (``fnmatch``).

    Returns a list of ``(compiled_regex, original_text)`` pairs so we can show
    the user exactly what they typed when reporting matches.

    Heuristic (matches the project-wide ``_compile_pattern`` in
    ``utils/training_utils.py``):
        1. try ``re.compile(p)``;
        2. if that raises ``re.error`` (typical for shell-style globs like
           ``*.layers.3.*``), fall back to ``fnmatch.translate(p)``, which
           turns the glob into an anchored regex such as
           ``(?s:.*\\.layers\\.3\\..*)\\Z``.

    Both branches end up matched against the FULL key via ``re.fullmatch``;
    the glob branch is already tail-anchored by ``fnmatch.translate`` so the
    semantics are identical.
    """
    compiled: List[Tuple[Pattern[str], str]] = []
    for p in patterns:
        try:
            regex = re.compile(p)
        except re.error:
            try:
                regex = re.compile(fnmatch.translate(p))
            except re.error as e:
                raise SystemExit(
                    f"[strip_ckpt_keys] invalid pattern {p!r} "
                    f"(neither valid regex nor glob): {e}"
                )
        compiled.append((regex, p))
    return compiled


def filter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    patterns: List[Tuple[Pattern[str], str]],
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[str, str]]]:
    """Return ``(kept_state_dict, removed)`` where ``removed`` is a list of
    ``(key, matching_pattern_text)`` pairs, in original key order."""
    kept: Dict[str, torch.Tensor] = {}
    removed: List[Tuple[str, str]] = []
    for k, v in state_dict.items():
        match_pat = next((text for regex, text in patterns if regex.fullmatch(k)), None)
        if match_pat is None:
            kept[k] = v
        else:
            removed.append((k, match_pat))
    return kept, removed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Remove state-dict keys matching one or more regex "
                    "patterns from a pytorch_model.bin checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="Path to pytorch_model.bin (single file).")
    ap.add_argument(
        "--pattern", "-p", action="append", default=[],
        metavar="PATTERN",
        help="Regex (preferred) or shell-style glob (fallback) matched "
             "against the full key (anchored at both ends). Repeat to add "
             "more patterns. Examples: "
             "--pattern '*.inv_freqs'   (glob, simple)   "
             "--pattern '.*\\.inv_freqs$'   (regex, explicit)",
    )
    ap.add_argument(
        "--patterns-file", default=None,
        help="Optional file with one regex per line (lines starting with "
             "'#' are ignored). Combined with --pattern.",
    )
    out = ap.add_mutually_exclusive_group()
    out.add_argument(
        "--output", "-o", default=None,
        help="Write filtered checkpoint to this path. "
             "Defaults to '<input>.stripped.bin' if neither --output nor "
             "--inplace is given.",
    )
    out.add_argument(
        "--inplace", action="store_true",
        help="Overwrite the input file. A '<input>.bak' copy is kept "
             "unless --no-backup is given.",
    )
    ap.add_argument(
        "--no-backup", action="store_true",
        help="When used with --inplace, skip the .bak backup.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print which keys would be removed and exit without writing.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    patterns_str: List[str] = list(args.pattern)
    if args.patterns_file:
        with open(args.patterns_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    patterns_str.append(s)
    if not patterns_str:
        raise SystemExit(
            "[strip_ckpt_keys] no patterns given; use --pattern or "
            "--patterns-file (at least one is required)."
        )
    patterns = compile_patterns(patterns_str)

    if not os.path.isfile(args.input):
        raise SystemExit(f"[strip_ckpt_keys] not a file: {args.input}")

    print(f"[strip_ckpt_keys] loading {args.input}")
    state_dict = torch.load(args.input, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise SystemExit(
            f"[strip_ckpt_keys] expected a dict-like state_dict at the top of "
            f"{args.input}, got {type(state_dict).__name__}. This script does "
            f"not handle nested checkpoints (e.g. {{'model': ..., 'optim': ...}}); "
            f"extract the model state_dict first."
        )

    kept, removed = filter_state_dict(state_dict, patterns)

    print(f"[strip_ckpt_keys] {len(state_dict)} keys -> "
          f"keep {len(kept)}, remove {len(removed)}")
    if removed:
        print("[strip_ckpt_keys] removed keys (key <- pattern):")
        for k, pat in removed:
            print(f"  - {k}  <-  {pat}")
    else:
        print("[strip_ckpt_keys] no keys matched any pattern; nothing to remove.")

    if args.dry_run:
        print("[strip_ckpt_keys] --dry-run set; not writing anything.")
        return 0

    if args.inplace:
        out_path = args.input
        if not args.no_backup:
            bak_path = args.input + ".bak"
            print(f"[strip_ckpt_keys] backing up original to {bak_path}")
            shutil.copy2(args.input, bak_path)
    else:
        out_path = args.output or (
            os.path.splitext(args.input)[0] + ".stripped.bin"
        )

    if not removed and out_path == args.input:
        # No-op for in-place with no matches: avoid rewriting a giant file.
        print("[strip_ckpt_keys] in-place + no removals -> skipping rewrite.")
        return 0

    print(f"[strip_ckpt_keys] writing {out_path}  ({len(kept)} keys)")
    torch.save(kept, out_path)
    print("[strip_ckpt_keys] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
