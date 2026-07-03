#!/usr/bin/env python3
"""Verify freeze_pattern / trainable-pattern logic for CPT scripts.

Three modes (fast -> faithful):

1) pattern  - CPU-only regex check on parameter names (seconds)
2) eager    - single-GPU full model, one backward, check grads (minutes, needs ~30GB)
3) fsdp2    - same path as training (meta init + FSDP2 + optional weights load)

Examples
--------
    # Quick: regex + expected trainable count only
    python code_exp/verify_freeze_pattern.py --mode pattern

    # Single GPU: real backward pass without FSDP
    CUDA_VISIBLE_DEVICES=0 python code_exp/verify_freeze_pattern.py --mode eager --seq-len 128

    # Closest to CPT training (1-GPU FSDP2 smoke test)
    CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \\
        code_exp/verify_freeze_pattern.py --mode fsdp2 --seq-len 128 \\
        --model-path /path/to/pytorch_model.bin_or_dir
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Optional, Tuple

import models  # noqa: F401
import torch
import torch.distributed as dist

from utils.training_utils import format_freeze_summary, freeze_parameters
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model
from veomni.optim import build_optimizer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


DEFAULT_CONFIG = "configs/olmo3_7B/olmo3_8KA2K_lmk_embed_lora_lmkq_layerwise_lmkq_norm.json"
DEFAULT_FREEZE_PATTERN = r"^(?!.*(?:lmk_embed|lmk_q_proj|lmk_q_norm)).*"
DEFAULT_TRAINABLE_PATTERN = r"(?:lmk_embed|lmk_q_proj|lmk_q_norm)"
EXPECTED_TRAINABLE = 25


def _compile(name: str, pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def _is_trainable(name: str, trainable_pattern: re.Pattern[str]) -> bool:
    return trainable_pattern.search(name) is not None


def _grad_max(param: torch.nn.Parameter) -> Optional[float]:
    grad = param.grad
    if grad is None:
        return None
    if hasattr(grad, "to_local"):
        grad = grad.to_local()
    return float(grad.detach().float().abs().max().item())


def check_pattern_on_names(names: List[str], freeze_pattern: str, trainable_pattern: str) -> int:
    freeze_re = _compile("freeze", freeze_pattern)
    trainable_re = _compile("trainable", trainable_pattern)

    would_freeze = [n for n in names if freeze_re.search(n)]
    would_train = [n for n in names if _is_trainable(n, trainable_re)]
    inconsistent = [n for n in names if freeze_re.search(n) and _is_trainable(n, trainable_re)]

    print("=== pattern mode ===")
    print(f"freeze_pattern:     {freeze_pattern}")
    print(f"trainable_pattern:  {trainable_pattern}")
    print(f"total param keys:   {len(names)}")
    print(f"would freeze:       {len(would_freeze)}")
    print(f"would train:        {len(would_train)}")
    print(f"inconsistent:       {len(inconsistent)}")
    if would_train:
        print("trainable keys:")
        for n in would_train:
            print(f"  {n}")

    ok = len(inconsistent) == 0 and len(would_train) == EXPECTED_TRAINABLE
    if not ok:
        print(f"FAIL: expected {EXPECTED_TRAINABLE} trainable keys, got {len(would_train)}")
        return 1
    print(f"PASS: pattern matches {EXPECTED_TRAINABLE} trainable keys, no overlap with freeze set")
    return 0


def _resolve_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return torch.device(f"{get_device_type()}:{local_rank}")


def _run_backward_check(model: torch.nn.Module, trainable_pattern: str, seq_len: int, vocab_cap: int) -> int:
    device = _resolve_device()
    trainable_re = _compile("trainable", trainable_pattern)

    model.train()
    input_ids = torch.randint(0, vocab_cap, (1, seq_len), device=device)
    labels = input_ids.clone()
    loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    if loss is None:
        print("FAIL: model returned loss=None (labels required for HiLSForCausalLM)")
        return 1
    loss.backward()

    frozen_with_grad: List[Tuple[str, float]] = []
    trainable_without_grad: List[str] = []
    for name, param in model.named_parameters():
        is_trainable = _is_trainable(name, trainable_re)
        gmax = _grad_max(param)
        if is_trainable:
            if gmax is None:
                trainable_without_grad.append(name)
        else:
            if gmax is not None and gmax > 0:
                frozen_with_grad.append((name, gmax))

    print("=== backward grad check ===")
    print(f"frozen params with non-zero grad: {len(frozen_with_grad)}")
    for name, gmax in frozen_with_grad[:20]:
        print(f"  {name}  grad_max={gmax:.3e}")
    if len(frozen_with_grad) > 20:
        print(f"  ... and {len(frozen_with_grad) - 20} more")

    print(f"trainable params without grad: {len(trainable_without_grad)}")
    for name in trainable_without_grad[:20]:
        print(f"  {name}")

    if frozen_with_grad:
        print("FAIL: frozen parameters received gradients")
        return 1
    if len(trainable_without_grad) == len([n for n, _ in model.named_parameters() if _is_trainable(n, trainable_re)]):
        print("FAIL: no trainable parameter received gradients")
        return 1
    if trainable_without_grad:
        print("WARN: some trainable parameters have no grad (may be unused in this layer type)")
    print("PASS: frozen params have zero/no grad after one backward")
    return 0


def _init_dist_if_needed() -> None:
    if dist.is_initialized():
        return
    backend = get_dist_comm_backend()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, device_id=torch.device(f"cuda:{local_rank}"))
    else:
        dist.init_process_group(backend=backend)


def run_eager(config_path: str, freeze_pattern: str, trainable_pattern: str, seq_len: int) -> int:
    print("=== eager mode (no FSDP) ===", flush=True)
    print("[verify_freeze] loading full model on GPU (may take 1-3 min)...", flush=True)
    model = build_foundation_model(
        config_path=config_path,
        torch_dtype="bfloat16",
        init_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    summary = freeze_parameters(model, freeze_pattern)
    print(format_freeze_summary(summary))
    if summary.trainable != EXPECTED_TRAINABLE:
        print(f"FAIL: expected trainable={EXPECTED_TRAINABLE}, got {summary.trainable}")
        return 1
    return _run_backward_check(model, trainable_pattern, seq_len, vocab_cap=1000)


def run_fsdp2(
    config_path: str,
    model_path: Optional[str],
    freeze_pattern: str,
    trainable_pattern: str,
    seq_len: int,
    load_weights: bool,
) -> int:
    _init_dist_if_needed()
    world_size = dist.get_world_size()
    if world_size < 2:
        print(
            "FAIL: fsdp2 mode requires world_size >= 2 (VeOmni sets fsdp_enabled only when fsdp_size > 1).\n"
            "Launch with at least 2 GPUs, e.g.:\n"
            "  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \\\n"
            "    code_exp/verify_freeze_pattern.py --mode fsdp2 --no-load-weights\n"
            "For single-GPU grad check, use: MODE=eager bash scripts/cpt/verify_freeze_pattern.sh",
            flush=True,
        )
        return 1

    rank = dist.get_rank()
    get_torch_device().set_device(f"{get_device_type()}:{int(os.environ.get('LOCAL_RANK', '0'))}")

    init_parallel_state(
        dp_size=world_size,
        dp_replicate_size=1,
        dp_shard_size=world_size,
        tp_size=1,
        ep_size=1,
        pp_size=1,
        cp_size=1,
        ulysses_size=1,
        dp_mode="fsdp2",
    )

    if rank == 0:
        print("=== fsdp2 mode (matches CPT training stack) ===")

    weights_path = model_path if load_weights else None
    model = build_foundation_model(
        config_path=config_path,
        weights_path=weights_path,
        torch_dtype="bfloat16",
        init_device="meta",
    )
    model = build_parallelize_model(
        model,
        init_device="meta",
        weights_path=weights_path,
        enable_full_shard=True,
        enable_mixed_precision=True,
        enable_gradient_checkpointing=False,
        basic_modules=model._no_split_modules + getattr(model, "basic_modules", []),
        enable_reentrant=False,
        enable_forward_prefetch=False,
        broadcast_model_weights_from_rank0=False,
    )

    summary = freeze_parameters(model, freeze_pattern)
    if rank == 0:
        print(format_freeze_summary(summary))

    optimizer = build_optimizer(model, lr=1e-4, weight_decay=0.1, fused=False)
    optim_params = sum(len(g["params"]) for g in optimizer.param_groups)
    if rank == 0:
        print(f"optimizer param tensors: {optim_params} (expect {EXPECTED_TRAINABLE})")

    if summary.trainable != EXPECTED_TRAINABLE:
        if rank == 0:
            print(f"FAIL: expected trainable={EXPECTED_TRAINABLE}, got {summary.trainable}")
        return 1
    if optim_params != summary.trainable:
        if rank == 0:
            print(
                f"FAIL: optimizer has {optim_params} tensors but {summary.trainable} trainable params "
                "(frozen params leaked into optimizer)"
            )
        return 1

    rc = _run_backward_check(model, trainable_pattern, seq_len, vocab_cap=1000)
    optimizer.zero_grad()
    if dist.is_initialized():
        dist.barrier()
    return rc


def run_pattern(config_path: str, freeze_pattern: str, trainable_pattern: str) -> int:
    print("[verify_freeze] building meta skeleton (no weights, should finish in seconds)...", flush=True)
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="meta",
    )
    names = [n for n, _ in model.named_parameters()]
    print(f"[verify_freeze] collected {len(names)} parameter names", flush=True)
    return check_pattern_on_names(names, freeze_pattern, trainable_pattern)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify CPT freeze_pattern behavior.")
    ap.add_argument("--mode", choices=("pattern", "eager", "fsdp2"), default="pattern")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--freeze-pattern", default=DEFAULT_FREEZE_PATTERN)
    ap.add_argument("--trainable-pattern", default=DEFAULT_TRAINABLE_PATTERN)
    ap.add_argument("--model-path", default=None, help="Optional weights for fsdp2 mode")
    ap.add_argument("--no-load-weights", action="store_true", help="fsdp2: random init only (faster)")
    ap.add_argument("--seq-len", type=int, default=128)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.mode == "pattern":
            return run_pattern(args.config, args.freeze_pattern, args.trainable_pattern)
        if args.mode == "eager":
            if not torch.cuda.is_available():
                print("eager mode requires CUDA", file=sys.stderr)
                return 1
            return run_eager(args.config, args.freeze_pattern, args.trainable_pattern, args.seq_len)
        if args.mode == "fsdp2":
            if not torch.cuda.is_available():
                print("fsdp2 mode requires CUDA", file=sys.stderr)
                return 1
            if "LOCAL_RANK" not in os.environ:
                print("fsdp2 mode must be launched with torchrun (needs >=2 GPUs), e.g.:", file=sys.stderr)
                print(
                    "  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 "
                    "code_exp/verify_freeze_pattern.py --mode fsdp2 --no-load-weights",
                    file=sys.stderr,
                )
                return 1
            return run_fsdp2(
                args.config,
                args.model_path,
                args.freeze_pattern,
                args.trainable_pattern,
                args.seq_len,
                load_weights=not args.no_load_weights and args.model_path is not None,
            )
        return 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
