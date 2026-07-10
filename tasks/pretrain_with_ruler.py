import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List

import data
import torch
import torch.distributed as dist
import wandb
from tqdm import trange
from utils.training_utils import format_freeze_summary, freeze_parameters

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_dataset,
)
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.optim.optimizer import get_parameter_names
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
    is_nccl_backend,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.utils.loss_utils import count_loss_token, mean_global_loss
from utils.arguments import RULERDataArguments, CPTArguments
from data import RulerSynthesizer, synthesize_ruler_example
import models


logger = helper.create_logger(__name__)
INDEX_TENSOR_KEYS = {"input_ids", "labels", "position_ids", "cache_position", "token_type_ids"}


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=RULERDataArguments)
    train: "TrainingArguments" = field(default_factory=CPTArguments)


def summarize_top_grad_norms(model, topk: int) -> list[tuple[str, float, float, float, str]]:
    grad_stats = []
    for name, param in model.named_parameters():
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        grad_local = grad.to_local() if hasattr(grad, "to_local") else grad
        param_local = param.to_local() if hasattr(param, "to_local") else param
        if grad_local is None:
            continue
        grad_fp32 = grad_local.detach().to(torch.float32)
        param_fp32 = param_local.detach().to(torch.float32)
        grad_stats.append(
            (
                name,
                grad_fp32.norm().item(),
                grad_fp32.abs().max().item(),
                param_fp32.norm().item(),
                str(tuple(grad_local.shape)),
            )
        )
    grad_stats.sort(key=lambda item: item[1], reverse=True)
    return grad_stats[:topk]


def maybe_log_grad_debug(model, args, global_step: int):
    topk = getattr(args.train, "debug_grad_topk", 0)
    max_steps = getattr(args.train, "debug_grad_steps", 0)
    if topk <= 0 or max_steps <= 0 or global_step > max_steps:
        return
    if args.train.global_rank != 0:
        return

    grad_stats = summarize_top_grad_norms(model, topk)
    if not grad_stats:
        logger.info_rank0(f"[grad-debug][step {global_step}] no gradients found")
        return

    lines = [f"[grad-debug][step {global_step}] top {len(grad_stats)} grad norms:"]
    for idx, (name, grad_norm, grad_abs_max, param_norm, shape) in enumerate(grad_stats, start=1):
        lines.append(
            f"  {idx}. {name} shape={shape} grad_norm={grad_norm:.4e} "
            f"grad_abs_max={grad_abs_max:.4e} param_norm={param_norm:.4e}"
        )
    logger.info_rank0("\n".join(lines))


def normalize_model_inputs(batch: Dict[str, Any]) -> Dict[str, Any]:
    normalized_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key in INDEX_TENSOR_KEYS and value.dtype != torch.long:
            normalized_batch[key] = value.to(dtype=torch.long)
        else:
            normalized_batch[key] = value
    return normalized_batch


def main():
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

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

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

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    ruler_synthesizer = RulerSynthesizer(tokenizer)
    transform = partial(
        synthesize_ruler_example,
        ruler_synthesizer=ruler_synthesizer,
        params=args.data.data_type,
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

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="bfloat16" if args.train.enable_mixed_precision else "float32",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=getattr(args.train, "broadcast_model_weights_from_rank0", False),
    )

    if args.train.freeze_pattern is not None:
        freeze_summary = freeze_parameters(model, args.train.freeze_pattern)
        logger.info_rank0(
            f"Applied freeze_pattern before optimizer build (pattern={args.train.freeze_pattern!r}):\n"
            f"{format_freeze_summary(freeze_summary)}"
        )
        if freeze_summary.frozen == 0 and freeze_summary.trainable == freeze_summary.total:
            logger.warning_rank0(
                "freeze_pattern matched zero parameters; check shell quoting "
                "(pattern must not include literal quote characters)."
            )

    param_groups = None
    if getattr(args.train, "include_frozen_params_in_optimizer", False):
        # Keep optimizer param_groups stable across freeze/unfreeze stages by explicitly
        # including all parameters (even requires_grad=False). Frozen params won't have
        # grads, so optimizer state is not initialized for them.
        ps = get_parallel_state()
        if ps.dp_mode == "fsdp2" and ps.ep_enabled:
            raise RuntimeError(
                "include_frozen_params_in_optimizer is not supported for EP+FSDP2 optimizer path "
                "(it filters requires_grad)."
            )

        decay_param_names = set(get_parameter_names(model, args.train.no_decay_modules, args.train.no_decay_params))
        decayed_params = []
        undecayed_params = []
        undecayed_names = []
        for n, p in model.named_parameters():
            if n in decay_param_names:
                decayed_params.append(p)
            else:
                undecayed_params.append(p)
                undecayed_names.append(n)

        param_groups = []
        if decayed_params:
            param_groups.append({"params": decayed_params, "weight_decay": args.train.weight_decay})
        if undecayed_params:
            logger.info_rank0(f"Parameters without weight decay: {undecayed_names}")
            param_groups.append({"params": undecayed_params, "weight_decay": 0.0})

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
        param_groups=param_groups,
        no_decay_modules=args.train.no_decay_modules,
        no_decay_params=args.train.no_decay_params,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                settings=wandb.Settings(console="off"),
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        # save model_assets before training
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        # When doing stage2 unfreezing, optimizer/scheduler param groups can change.
        # Allow resuming model/dataloader/RNG while rebuilding optimizer/scheduler.
        state = {"model": model, "extra_state": {}}  # extra_state cannot be None
        if getattr(args.train, "load_optimizer_state", True):
            state["optimizer"] = optimizer

        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        if getattr(args.train, "load_lr_scheduler_state", True):
            lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        if getattr(args.train, "load_dataloader_state", True):
            train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        if getattr(args.train, "load_environ_meter_state", True):
            environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        if getattr(args.train, "load_rng_state", True):
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

        if args.train.freeze_pattern is not None:
            freeze_summary = freeze_parameters(model, args.train.freeze_pattern)
            logger.info_rank0(
                "Re-applied freeze_pattern after checkpoint load:\n"
                f"{format_freeze_summary(freeze_summary)}"
            )

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()

            micro_batches_token_num = count_loss_token(micro_batches)

            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                micro_batch_token_num = count_loss_token(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("cur_token_num", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                micro_batch = normalize_model_inputs(micro_batch)
                with model_fwd_context:
                    loss = model(**micro_batch, use_cache=False).loss

                loss, _ = mean_global_loss(loss, micro_batch_token_num, micro_batches_token_num)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            maybe_log_grad_debug(model, args, global_step)
            grad_norm = veomni_clip_grad_norm(model, args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.4f}, grad_norm: {grad_norm:.4f}, lr: {lr:.2e}",
                refresh=False,
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            # # ── Dynamic runtime config: hot-reload from {output_dir}/runtime_config.json ──
            # # Supports changing save_steps, attention dropout, etc. without restarting training.
            # # Example runtime_config.json:
            # #   {"save_steps": 250, "attention_dropout": 0.0}
            # _runtime_cfg_path = os.path.join(args.train.output_dir, "runtime_config.json")
            # try:
            #     if os.path.isfile(_runtime_cfg_path):
            #         with open(_runtime_cfg_path, "r") as _f:
            #             _runtime_cfg = json.load(_f)

            #         # 1) save_steps
            #         if "save_steps" in _runtime_cfg:
            #             _new_ss = int(_runtime_cfg["save_steps"])
            #             if _new_ss > 0 and _new_ss != args.train.save_steps:
            #                 logger.info_rank0(f"[runtime] save_steps: {args.train.save_steps} -> {_new_ss}")
            #                 args.train.save_steps = _new_ss

            #         # 2) model config fields (dropout, etc.) — propagate to all layers
            #         _model_cfg_keys = ["attention_dropout"]
            #         _unwrapped = model.module if hasattr(model, "module") else model
            #         for _key in _model_cfg_keys:
            #             if _key in _runtime_cfg:
            #                 _new_val = float(_runtime_cfg[_key])
            #                 _old_val = getattr(model_config, _key, None)
            #                 if _old_val is not None and _new_val != _old_val:
            #                     logger.info_rank0(f"[runtime] {_key}: {_old_val} -> {_new_val}")
            #                     setattr(model_config, _key, _new_val)
            #                     # propagate to all submodules that cache this attribute
            #                     for _m in _unwrapped.modules():
            #                         if hasattr(_m, _key):
            #                             setattr(_m, _key, _new_val)
            # except Exception as _e:
            #     if global_step % 100 == 0:  # don't spam logs
            #         logger.info_rank0(f"[runtime] failed to load {_runtime_cfg_path}: {_e}")

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)

                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
