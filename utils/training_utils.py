import fnmatch
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch.nn as nn


@dataclass
class FreezeSummary:
    total: int = 0
    frozen: int = 0
    trainable: int = 0
    frozen_names: List[str] = None
    trainable_names: List[str] = None

    def __post_init__(self):
        if self.frozen_names is None:
            self.frozen_names = []
        if self.trainable_names is None:
            self.trainable_names = []


def _strip_shell_quotes(pattern: str) -> str:
    """Remove a single pair of surrounding shell quotes accidentally passed through."""
    s = pattern.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _compile_pattern(pattern: str) -> re.Pattern[str]:
    if not isinstance(pattern, str):
        return pattern
    pattern = _strip_shell_quotes(pattern)
    try:
        return re.compile(pattern)
    except re.error:
        return re.compile(fnmatch.translate(pattern))


def summarize_requires_grad(model: nn.Module) -> FreezeSummary:
    summary = FreezeSummary()
    for name, param in model.named_parameters():
        summary.total += 1
        if param.requires_grad:
            summary.trainable += 1
            summary.trainable_names.append(name)
        else:
            summary.frozen += 1
            summary.frozen_names.append(name)
    return summary


def freeze_parameters(
    model: nn.Module,
    pattern: str,
    *,
    trainable_pattern: Optional[str] = None,
) -> FreezeSummary:
    """Freeze parameters whose names match ``pattern``.

    If ``trainable_pattern`` is provided, only params matching it stay trainable and
    everything else is frozen. This is often clearer than a negative-lookahead freeze regex.
    """
    if trainable_pattern is not None:
        trainable_re = _compile_pattern(trainable_pattern)
        for name, param in model.named_parameters():
            param.requires_grad_(trainable_re.search(name) is not None)
        return summarize_requires_grad(model)

    regex = _compile_pattern(pattern)
    for name, param in model.named_parameters():
        if regex.search(name):
            param.requires_grad_(False)
    return summarize_requires_grad(model)


def format_freeze_summary(summary: FreezeSummary, *, max_names: int = 20) -> str:
    lines = [
        f"total={summary.total} trainable={summary.trainable} frozen={summary.frozen}",
    ]
    if summary.trainable_names:
        shown = summary.trainable_names[:max_names]
        lines.append(f"trainable sample ({len(shown)}/{len(summary.trainable_names)}):")
        lines.extend(f"  {name}" for name in shown)
        if len(summary.trainable_names) > max_names:
            lines.append(f"  ... and {len(summary.trainable_names) - max_names} more")
    return "\n".join(lines)
