import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List

import torch
import torch.distributed as dist

from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
from data.data_transform import process_sft_example_with_lmk
from utils.arguments import SFTArguments
from veomni.distributed.parallel_state import init_parallel_state
from veomni.models import build_tokenizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
)

logger = helper.create_logger(__name__)


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=SFTArguments)


# ============================================================================
# 在这里硬编码参数，就不需要命令行传 yaml / --args 了
# 直接 python run_sft_tokenizer.py 或 torchrun --nproc_per_node=1 run_sft_tokenizer.py
# ============================================================================
NUM_BATCHES_TO_INSPECT = 3

FAKE_ARGS = [
    # ---- 方式1: 直接指定一个 yaml 配置文件 (取消注释下一行即可) ----
    # "/path/to/your/config.yaml",

    # ---- 方式2: 逐个指定参数 ----
    # model
    "--model.tokenizer_path", "configs/olmo3_vocab",
    "--model.config_path", "configs/olmo3_7B",

    # data
    "--data.chat_template", "olmo3_sft",
    "--data.train_path", "/data/Dolci-Instruct-SFT-parquet",
    "--data.max_seq_len", "8192",
    "--data.datasets_type", "mapping",
    "--data.dataset_name", "mapping",
    # "--data.text_keys", "messages",

    # train
    "--train.micro_batch_size", "1",
    "--train.global_batch_size", "1",
    "--train.dataloader_batch_size", "1",
    "--train.output_dir", "/tmp/",
    "--train.chunk_size", "64",
    "--train.rmpad", "False",
    "--train.rmpad_with_pos_ids", "True",
    "--train.train_steps", "100",
]


def decode_and_inspect(tokenizer, micro_batch, batch_idx, micro_idx, chunk_size=None):
    """Decode a micro_batch and inspect labels (-100 mask) and chunk alignment."""
    input_ids = micro_batch["input_ids"]   # [seq_len] or [bsz, seq_len]
    labels = micro_batch["labels"]         # same shape

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)

    bsz, seq_len = input_ids.shape
    print(f"\n{'='*80}")
    print(f"[Batch {batch_idx} | Micro {micro_idx}] keys={list(micro_batch.keys())}")
    for k, v in micro_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")

    for b in range(min(bsz, 2)):  # only inspect first 2 samples in batch
        ids = input_ids[b]
        lab = labels[b]

        # --- Basic stats ---
        total_tokens = seq_len
        masked_count = (lab == -100).sum().item()
        train_count = total_tokens - masked_count
        print(f"\n  --- Sample {b} ---")
        print(f"  seq_len={total_tokens}, masked(-100)={masked_count}, trainable={train_count} ({100*train_count/total_tokens:.1f}%)")

        # --- Decode full input ---
        decoded_input = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
        print(f"\n  [INPUT (first 500 chars)]:")
        print(f"  {decoded_input[:500]}")

        # --- Decode trainable part (labels != -100) ---
        train_mask = lab != -100
        train_ids = lab[train_mask].tolist()
        decoded_train = tokenizer.decode(train_ids, skip_special_tokens=False)
        print(f"\n  [TRAINABLE LABELS (first 500 chars)]:")
        print(f"  {decoded_train[:500]}")

        # --- Show boundary: where -100 switches to non-(-100) and vice versa ---
        transitions = []
        for i in range(1, len(lab)):
            prev_masked = (lab[i-1].item() == -100)
            curr_masked = (lab[i].item() == -100)
            if prev_masked != curr_masked:
                direction = "MASK->TRAIN" if prev_masked else "TRAIN->MASK"
                # decode a small window around the transition
                start = max(0, i - 3)
                end = min(len(ids), i + 3)
                window_ids = ids[start:end].tolist()
                window_lab = lab[start:end].tolist()
                window_text = tokenizer.decode(window_ids, skip_special_tokens=False)
                transitions.append((i, direction, window_text, window_lab))

        print(f"\n  [LABEL TRANSITIONS] (total {len(transitions)} transitions):")
        for pos, direction, text, lab_window in transitions[:20]:  # show first 20
            lab_str = ["T" if l != -100 else "M" for l in lab_window]
            print(f"    pos={pos:>6d} {direction:<12s} labels=[{','.join(lab_str)}] text='{text}'")
        if len(transitions) > 20:
            print(f"    ... and {len(transitions) - 20} more transitions")

        # --- Check chunk alignment ---
        if chunk_size and chunk_size > 0:
            print(f"\n  [CHUNK ALIGNMENT] chunk_size={chunk_size}")
            # Check if position_ids exist (indicates chunk boundaries)
            if "position_ids" in micro_batch:
                pos_ids = micro_batch["position_ids"]
                if pos_ids.dim() == 1:
                    pos_ids = pos_ids.unsqueeze(0)
                p = pos_ids[b]
                # Find positions where position_ids reset to 0 (chunk boundaries)
                reset_positions = (p == 0).nonzero(as_tuple=True)[0].tolist()
                print(f"    position_ids resets at: {reset_positions[:30]}")
                if len(reset_positions) > 30:
                    print(f"    ... and {len(reset_positions) - 30} more resets")

                # Check if chunk boundaries align with chunk_size
                chunk_size_int = int(chunk_size) - 1
                for i in range(1, len(reset_positions)):
                    gap = reset_positions[i] - reset_positions[i-1]
                    if gap % chunk_size_int != 0:
                        print(f"    WARNING: chunk gap {gap} at pos {reset_positions[i]} (expected {chunk_size_int})")
            else:
                # No position_ids, check transition alignment with chunk boundaries
                chunk_size_int = int(chunk_size)
                for pos, direction, _, _ in transitions:
                    remainder = pos % chunk_size_int
                    aligned = "ALIGNED" if remainder == 0 else f"OFFSET={remainder}"
                    print(f"    transition at pos={pos}, pos%{chunk_size_int}={remainder} -> {aligned}")

        PRINT_TOKEN_NUM = 8192
        print(f"\n  [TOKEN-LEVEL first {PRINT_TOKEN_NUM} tokens]:")
        print(f"  {'pos':>5s} | {'id':>8s} | {'lab':>8s} | {'mask':>4s} | token_text")
        print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*4}-+-{'-'*30}")
        for i in range(min(PRINT_TOKEN_NUM, seq_len)):
            tok_id = ids[i].item()
            lab_val = lab[i].item()
            mask_str = "M" if lab_val == -100 else "T"
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
            tok_text = tok_text.replace('\n', '\\n').replace('\r', '\\r')
            print(f"  {i:>5d} | {tok_id:>8d} | {lab_val:>8d} | {mask_str:>4s} | {tok_text}")

    print(f"{'='*80}\n")


def main():
    # ============================================================
    # 注入 FAKE_ARGS 到 sys.argv, 这样 parse_args 就能读到
    # 如果命令行本身已经传了参数 (len > 1), 则优先用命令行的
    # ============================================================
    if len(sys.argv) == 1:
        # 没有任何命令行参数 -> 用脚本内硬编码的 FAKE_ARGS
        sys.argv = [sys.argv[0]] + FAKE_ARGS
        logger.info("Using hardcoded FAKE_ARGS (no CLI args detected)")
    else:
        logger.info(f"Using CLI args: {sys.argv[1:]}")

    nccl_timeout = os.getenv("NCCL_TIMEOUT", None)
    pg_nccl_timeout = None
    if nccl_timeout is not None and is_nccl_backend():
        pg_nccl_timeout = timedelta(seconds=int(nccl_timeout))
    logger.info(f"Process_group timeout: {nccl_timeout}")
    dist.init_process_group(backend=get_dist_comm_backend(), timeout=pg_nccl_timeout)

    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # ===================== Data pipeline (identical to sft_with_lmk.py) =====================
    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    chat_template = build_chat_template(args.data.chat_template, tokenizer)
    transform = partial(
        process_sft_example_with_lmk,
        chat_template=chat_template,
        max_seq_len=args.data.max_seq_len,
        chunk_size=args.train.chunk_size,
        text_keys=args.data.text_keys,
    )

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        **asdict(args.data),
    )
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping":
        dataset_length = dataset_length / args.train.data_parallel_size
    args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader_type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        seed=args.train.seed,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train.train_steps,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz_margin=args.train.dyn_bsz_margin,
        dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
        num_workers=args.data.num_workers,
        drop_last=args.data.drop_last,
        pin_memory=args.data.pin_memory,
        prefetch_factor=args.data.prefetch_factor,
    )

    # ===================== Iterate and decode =====================
    logger.info_rank0(f"Will inspect {NUM_BATCHES_TO_INSPECT} batches (change NUM_BATCHES_TO_INSPECT in script to adjust)")

    if hasattr(train_dataloader, "set_epoch"):
        train_dataloader.set_epoch(0)

    data_iterator = iter(train_dataloader)
    for batch_idx in range(NUM_BATCHES_TO_INSPECT):
        try:
            micro_batches: List[Dict[str, Any]] = next(data_iterator)
        except StopIteration:
            logger.info(f"Dataloader exhausted after {batch_idx} batches")
            break

        if batch_idx == 0:
            helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

        for micro_idx, micro_batch in enumerate(micro_batches):
            decode_and_inspect(
                tokenizer,
                micro_batch,
                batch_idx=batch_idx,
                micro_idx=micro_idx,
                chunk_size=args.train.chunk_size,
            )

    logger.info_rank0("Done inspecting data samples.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
