import argparse
import sys
import os
import json
import math
from contextlib import contextmanager

EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if EVAL_ROOT not in sys.path:
    sys.path.insert(0, EVAL_ROOT)

from data import build_numpy_dataset

import models
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.utils import data
from torch.utils.data import SequentialSampler
from veomni.models import build_foundation_model
from veomni.checkpoint import build_checkpointer
from veomni.distributed.parallel_state import init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from utils.landmark_utils import insert_special_tokens, create_position_ids_with_landmarks
from transformers import AutoConfig, AutoModelForCausalLM

HF_CAUSAL_LM_MODEL_TYPES = frozenset({"olmo3"})


@contextmanager
def auto_chunk_threshold(threshold: int):
    """Disable model-internal auto chunk prefill during manual segment prefill."""
    try:
        import models.FlashHiLS.chunk_prefill as chunk_prefill_module
    except ImportError:
        yield
        return

    saved = chunk_prefill_module.DEFAULT_CHUNK_PREFILL_THRESHOLD
    chunk_prefill_module.DEFAULT_CHUNK_PREFILL_THRESHOLD = threshold
    try:
        yield
    finally:
        chunk_prefill_module.DEFAULT_CHUNK_PREFILL_THRESHOLD = saved


def get_config_model_type(config_path):
    if not config_path or not os.path.exists(config_path):
        return ""
    with open(config_path, "r", encoding="utf-8") as fin:
        return json.load(fin).get("model_type", "")


def read_model_type(config_path=None, checkpoint_path=None):
    for path in (config_path, os.path.join(checkpoint_path, "config.json") if checkpoint_path else None):
        model_type = get_config_model_type(path)
        if model_type:
            return model_type
    return ""


def should_use_hf_model(args):
    if args.use_hf_model:
        return True
    if args.config_path:
        return get_config_model_type(args.config_path) in HF_CAUSAL_LM_MODEL_TYPES
    return False


def should_use_hf_loader(args):
    if should_use_hf_model(args):
        return True
    return "lhsa" in read_model_type(args.config_path, args.checkpoint_path)


def is_lhsa_model(config_path=None, checkpoint_path=None):
    return "lhsa" in read_model_type(config_path, checkpoint_path)


def load_hsa_config(config_path):
    from models.FlashHiLS.configuration_hsa import HSAConfig

    with open(config_path, "r", encoding="utf-8") as fin:
        return HSAConfig.from_dict(json.load(fin))


def load_eval_config(config_path, checkpoint_path=None):
    if is_lhsa_model(config_path, checkpoint_path):
        return load_hsa_config(config_path)
    return AutoConfig.from_pretrained(config_path, trust_remote_code=True)


def _register_with_exist_ok(register_fn, *args):
    try:
        register_fn(*args, exist_ok=True)
    except TypeError:
        try:
            register_fn(*args)
        except ValueError:
            pass
    except ValueError:
        pass


def register_hsa_model(config_path=None, checkpoint_path=None):
    model_type = read_model_type(config_path, checkpoint_path)
    if "lhsa" not in model_type:
        return
    if "olmo" in model_type:
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
        print("Using OLMo LHSA implementation")
    else:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        print("Using Qwen LHSA implementation")

    from models.FlashHiLS.configuration_hsa import HSAConfig

    HiLSForCausalLM.config_class = HSAConfig
    for name in {model_type, "olmo_lhsa", "flash_hsa", "qwen_lhsa"}:
        if not name:
            continue
        _register_with_exist_ok(AutoConfig.register, name, HSAConfig)
    _register_with_exist_ok(AutoModelForCausalLM.register, HSAConfig, HiLSForCausalLM)


def get_config_flag(config_path, key, default=False):
    if not config_path or not os.path.exists(config_path):
        return default
    with open(config_path, "r", encoding="utf-8") as fin:
        return json.load(fin).get(key, default)


def resolve_segment_size(args):
    if get_config_flag(args.config_path, "use_naive_bsa", False):
        return -1
    if getattr(args, "tp_size", 1) > 1:
        return -1
    if not getattr(args, "enable_chunk_prefill", False):
        return -1
    if args.segment_size > 0:
        return args.segment_size
    if args.inference_segment > 0:
        return args.inference_segment
    return -1


def setup_runtime(args):
    tp_size = getattr(args, "tp_size", 1)
    use_hf_tp = tp_size > 1 and should_use_hf_loader(args)
    use_veomni_tp = tp_size > 1 and not should_use_hf_loader(args)

    if use_veomni_tp:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        init_parallel_state(
            dp_size=1,
            tp_size=tp_size,
            cp_size=1,
            dp_mode="fsdp2",
        )
        return device, tp_size, use_hf_tp, use_veomni_tp

    device = torch.device("cuda:0")
    if use_hf_tp and torch.cuda.device_count() < tp_size:
        raise RuntimeError(
            f"--tp_size={tp_size} requires {tp_size} visible GPU(s), "
            f"but only {torch.cuda.device_count()} found in CUDA_VISIBLE_DEVICES"
        )
    return device, tp_size, use_hf_tp, use_veomni_tp


def get_model_input_device(model):
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


class KahanSum:
    """Kahan 求和算法，减少浮点累加误差"""
    def __init__(self):
        self.sum = 0.0
        self.c = 0.0  # 误差补偿

    def add(self, value):
        y = value - self.c
        t = self.sum + y
        self.c = (t - self.sum) - y
        self.sum = t

    def get(self):
        return self.sum


def append_summary_log(summary_log, summary_text):
    log_dir = os.path.dirname(os.path.abspath(summary_log))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(summary_log, "a", encoding="utf-8") as fout:
        fout.write(summary_text)
        fout.write("\n\n")


def get_last_k_tokens_with_landmarks(raw_seq_len, raw_last_k_tokens, chunk_size):
    if raw_last_k_tokens <= 0:
        return raw_last_k_tokens

    raw_last_k_tokens = min(raw_last_k_tokens, raw_seq_len)
    if raw_last_k_tokens <= 0:
        return raw_last_k_tokens

    raw_answer_start = raw_seq_len - raw_last_k_tokens
    total_landmarks = raw_seq_len // (chunk_size - 1)
    landmarks_before_answer = raw_answer_start // (chunk_size - 1)
    return raw_last_k_tokens + (total_landmarks - landmarks_before_answer)


def is_dcp_checkpoint(path):
    return os.path.isdir(path) and os.path.exists(os.path.join(path, ".metadata"))


def load_state_dict_from_path(path):
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


def build_eval_model(args, device, tp_size=1, use_hf_tp=False, use_veomni_tp=False):
    attn_implementation = getattr(args, "attn_implementation", "flash_attention_3")

    if use_veomni_tp:
        if not args.config_path:
            raise ValueError("--config_path is required for VeOmni TP")
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required for VeOmni TP")
        model = build_foundation_model(
            config_path=args.config_path,
            torch_dtype="bfloat16",
            init_device="cuda",
        )
        model = build_parallelize_model(
            model,
            init_device="cuda",
            dtype="bfloat16",
            weights_path=args.checkpoint_path,
            enable_full_shard=False,
            enable_mixed_precision=True,
            enable_gradient_checkpointing=False,
        )
        Checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
        Checkpointer.load(args.checkpoint_path, {"model": model})
        model.eval()
        return model

    if should_use_hf_loader(args):
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when loading HF causal LM")
        register_hsa_model(args.config_path, args.checkpoint_path)
        hf_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "attn_implementation": attn_implementation,
        }
        if use_hf_tp:
            hf_kwargs["device_map"] = "auto"
            print(f"Using HF device_map=auto across {tp_size} GPU(s) for full forward")
        else:
            hf_kwargs["device_map"] = device
        if args.config_path:
            config = load_eval_config(args.config_path, args.checkpoint_path)
            if args.insert_lmk and hasattr(config, "auto_insert_lmk"):
                config.auto_insert_lmk = False
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint_path,
                config=config,
                **hf_kwargs,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint_path,
                **hf_kwargs,
            )
        if args.insert_lmk:
            if hasattr(model, "auto_insert_lmk"):
                model.auto_insert_lmk = False
            if hasattr(model, "config"):
                model.config.auto_insert_lmk = False
        # device_map dispatch may leave the root-level lmk_embed parameter on
        # `meta` or a wrong device (accelerate warns it doesn't match a
        # submodule). Repair it from the checkpoint so it sits on the
        # embed_tokens device; otherwise long-context PPL produces NaN.
        try:
            from models.FlashHiLS.hsa_device_utils import ensure_lmk_embed_materialized
            ensure_lmk_embed_materialized(model, args.checkpoint_path)
        except ImportError:
            pass
        return model

    if not args.config_path:
        raise ValueError("--config_path is required unless --use_hf_model is set")

    model = build_foundation_model(
        config_path=args.config_path,
        torch_dtype="bfloat16",
    )

    if args.checkpoint_path:
        if is_dcp_checkpoint(args.checkpoint_path):
            Checkpointer = build_checkpointer(dist_backend='fsdp2', ckpt_manager='dcp')
            Checkpointer.load(args.checkpoint_path, {"model": model})
        else:
            state_dict = load_state_dict_from_path(args.checkpoint_path)
            incompatible = model.load_state_dict(state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Checkpoint mismatch. missing={incompatible.missing_keys}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )

    model.to(device)
    return model


def prepare_ppl_batch(input_ids, args, tokenizer, device):
    label_ids = input_ids
    pos_ids = None
    orig_seq_len = input_ids.shape[1]
    eval_last_k_tokens = args.last_k_tokens

    if args.insert_lmk:
        input_ids = insert_special_tokens(
            input_ids, fill_id=tokenizer.vocab_size, chunk_size=args.chunk_size
        )
        label_ids = torch.roll(label_ids, shifts=-1, dims=-1)
        label_ids[:, -1] = -100
        label_ids = insert_special_tokens(label_ids, fill_id=-100, chunk_size=args.chunk_size)
        label_ids = torch.roll(label_ids, shifts=1, dims=-1)
        if args.last_k_tokens > 0:
            eval_last_k_tokens = get_last_k_tokens_with_landmarks(
                orig_seq_len,
                args.last_k_tokens,
                args.chunk_size,
            )
        if args.adjust_lmk_pos:
            pos_ids = create_position_ids_with_landmarks(
                None, orig_seq_len, chunk_size=args.chunk_size, device=device
            )

    return {
        "input_ids": input_ids,
        "label_ids": label_ids,
        "position_ids": pos_ids,
        "orig_seq_len": orig_seq_len,
        "eval_last_k_tokens": eval_last_k_tokens,
    }


def _answer_len_for_chunk_prefill(
    seq_len,
    eval_last_k_tokens,
    raw_last_k_tokens,
    insert_lmk,
    orig_seq_len,
    chunk_size,
):
    if eval_last_k_tokens > 0:
        if insert_lmk:
            orig_answer_start = orig_seq_len - raw_last_k_tokens
            answer_start_with_lmk = orig_answer_start + (orig_answer_start // (chunk_size - 1))
            return seq_len - answer_start_with_lmk
        return eval_last_k_tokens
    return seq_len


def forward_full_logits(
    model,
    input_ids,
    position_ids,
    eval_last_k_tokens,
    device,
):
    """Full-sequence forward (same path as eval_ruler_hf)."""
    seq_len = input_ids.shape[1]
    kwargs = {}
    if eval_last_k_tokens > 0:
        kwargs["logits_to_keep"] = eval_last_k_tokens + 1

    cache_pos = torch.arange(0, seq_len, device=input_ids.device)
    with auto_chunk_threshold(0):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            result = model(
                input_ids=input_ids,
                cache_position=cache_pos,
                use_cache=False,
                position_ids=position_ids,
                **kwargs,
            )
    logits = result.logits
    if eval_last_k_tokens > 0:
        logits = logits[:, :-1, :]
    return logits


def forward_chunk_prefill_logits(
    model,
    input_ids,
    position_ids,
    eval_last_k_tokens,
    raw_last_k_tokens,
    segment_size,
    device,
    insert_lmk=False,
    orig_seq_len=None,
    chunk_size=64,
    skip_hsa_prefill=False,
):
    """Segmented chunk prefill (same path as eval_ruler_hf / test_ruler_chunk_prefill_consistency)."""
    seq_len = input_ids.shape[1]
    answer_token_len = _answer_len_for_chunk_prefill(
        seq_len,
        eval_last_k_tokens,
        raw_last_k_tokens,
        insert_lmk,
        orig_seq_len,
        chunk_size,
    )
    if segment_size > 0 and eval_last_k_tokens > 0:
        assert segment_size >= eval_last_k_tokens, (
            f"segment_size={segment_size} must be >= eval_last_k_tokens={eval_last_k_tokens}"
        )

    if eval_last_k_tokens > 0:
        answer_logits_start = seq_len - answer_token_len - 1
        logits_to_extract = answer_token_len
    else:
        answer_logits_start = 0
        logits_to_extract = seq_len

    first_answer_segment = answer_logits_start // segment_size
    num_segments = (seq_len + segment_size - 1) // segment_size

    past_key_values = None
    answer_logits_list = []

    with auto_chunk_threshold(0):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
            for seg_idx in range(num_segments):
                start_idx = seg_idx * segment_size
                end_idx = min((seg_idx + 1) * segment_size, seq_len)

                seg_input_ids = input_ids[:, start_idx:end_idx]
                seg_cache_pos = torch.arange(start_idx, end_idx, device=device)
                seg_pos_ids = position_ids[:, start_idx:end_idx] if position_ids is not None else None
                seg_logits_to_keep = end_idx - start_idx if seg_idx >= first_answer_segment else 1

                extra_kwargs = {}
                if skip_hsa_prefill and seg_idx < first_answer_segment - 1:
                    extra_kwargs["skip_hsa"] = True

                out = model(
                    input_ids=seg_input_ids,
                    cache_position=seg_cache_pos,
                    use_cache=True,
                    past_key_values=past_key_values,
                    logits_to_keep=seg_logits_to_keep,
                    position_ids=seg_pos_ids,
                    **extra_kwargs,
                )
                past_key_values = out.past_key_values
                if seg_idx >= first_answer_segment:
                    answer_logits_list.append(out.logits.cpu())
                del out

    answer_region_logits = torch.cat(answer_logits_list, dim=1)
    del answer_logits_list
    torch.cuda.empty_cache()

    offset_in_region = answer_logits_start - first_answer_segment * segment_size
    answer_logits = answer_region_logits[
        :, offset_in_region : offset_in_region + logits_to_extract, :
    ]
    del answer_region_logits
    return answer_logits.to(device)


def compute_ppl_loss(logits, label_ids, eval_last_k_tokens, insert_lmk, ce_fct):
    if eval_last_k_tokens > 0:
        if insert_lmk:
            pred_logits = logits[:, -eval_last_k_tokens:, :]
            target_labels = label_ids[:, -eval_last_k_tokens:]
            valid_mask = (target_labels != -100).squeeze(0)
            return ce_fct(
                pred_logits[0, valid_mask, :],
                target_labels[0, valid_mask].to(torch.long),
            )

        k = min(eval_last_k_tokens, logits.shape[1])
        pred_logits = logits[:, -k:, :]
        target_labels = label_ids[:, -k:]
        return ce_fct(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            target_labels.reshape(-1).to(torch.long),
        )

    pred_logits = logits[:, :-1, :]
    target_labels = label_ids[:, 1:]
    return ce_fct(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        target_labels.reshape(-1).to(torch.long),
    )


def run_ppl_forward(
    model,
    batch,
    args,
    device,
    segment_size=-1,
):
    input_ids = batch["input_ids"]
    label_ids = batch["label_ids"]
    pos_ids = batch["position_ids"]
    eval_last_k_tokens = batch["eval_last_k_tokens"]
    use_chunk_prefill = segment_size > 0

    if use_chunk_prefill:
        logits = forward_chunk_prefill_logits(
            model=model,
            input_ids=input_ids,
            position_ids=pos_ids,
            eval_last_k_tokens=eval_last_k_tokens,
            raw_last_k_tokens=args.last_k_tokens,
            segment_size=segment_size,
            device=device,
            insert_lmk=args.insert_lmk,
            orig_seq_len=batch["orig_seq_len"],
            chunk_size=args.chunk_size,
            skip_hsa_prefill=args.skip_hsa_prefill,
        )
    else:
        logits = forward_full_logits(
            model=model,
            input_ids=input_ids,
            position_ids=pos_ids,
            eval_last_k_tokens=eval_last_k_tokens,
            device=device,
        )

    ce_fct = nn.CrossEntropyLoss()
    loss = compute_ppl_loss(
        logits,
        label_ids,
        eval_last_k_tokens,
        args.insert_lmk,
        ce_fct,
    )
    return loss, logits


def main(args):
    device, tp_size, use_hf_tp, use_veomni_tp = setup_runtime(args)
    segment_size = resolve_segment_size(args)
    use_chunk_prefill = segment_size > 0
    if get_config_flag(args.config_path, "use_naive_bsa", False) and (
        getattr(args, "enable_chunk_prefill", False) or args.segment_size > 0
    ):
        print("NaiveBSA does not support KV cache; chunk prefill disabled (full inference)")

    print(f"Max Sequence Length for Evaluation: {args.max_seq_len}")
    if tp_size > 1:
        print(f"Parallel mode: TP={tp_size}, full forward (chunk prefill disabled)")
    else:
        print(
            f"Chunk Prefill: {'Enabled' if use_chunk_prefill else 'Disabled (Full Inference)'}"
            + (f", segment_size={segment_size}" if use_chunk_prefill else "")
        )

    dataset = build_numpy_dataset(args.data_path, args.max_seq_len, namespace="test")
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    def vanilla_collate_fn(examples):
        return {
            "input_ids": torch.tensor(examples),
            "labels": torch.tensor(examples),
        }

    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=vanilla_collate_fn,
        sampler=SequentialSampler(dataset),
        num_workers=1,
    )

    model = build_eval_model(
        args,
        device,
        tp_size=tp_size,
        use_hf_tp=use_hf_tp,
        use_veomni_tp=use_veomni_tp,
    )
    model.eval()
    input_device = get_model_input_device(model)

    loss_accum = KahanSum()
    steps = 0
    for inputs in dataloader:
        steps += 1
        for key, value in inputs.items():
            if value is not None and isinstance(value, torch.Tensor):
                inputs[key] = value.to(input_device)

        batch = prepare_ppl_batch(inputs["input_ids"], args, tokenizer, input_device)
        loss, _ = run_ppl_forward(model, batch, args, input_device, segment_size=segment_size)

        loss_accum.add(loss.item())
        if steps % 100 == 0:
            print(f"step: {steps}, mean_loss: {loss_accum.get() / steps}")

        if args.max_samples > 0 and steps >= args.max_samples:
            break

    final_mean_loss = loss_accum.get() / steps if steps > 0 else 0.0
    ppl = math.exp(final_mean_loss)
    summary = (
        f"Test Length: {args.max_seq_len}, Final Mean Loss: {final_mean_loss:.4f}, "
        f"PPL: {ppl:.4f}\nModel: {args.checkpoint_path}"
    )
    print(summary)
    if args.summary_log:
        append_summary_log(args.summary_log, summary)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser("NCR pretraining setup")
    cmd.add_argument("--config_path", required=False, type=str, default=None)
    cmd.add_argument("--vocab_dir", required=True, type=str)
    cmd.add_argument("--data_path", required=True, type=str, help="path to the training corpus")
    cmd.add_argument("--max_seq_len", default=16384, type=int)
    cmd.add_argument("--chunk_size", default=64, type=int)
    cmd.add_argument("--insert_lmk", action="store_true")
    cmd.add_argument("--checkpoint_path", required=False, type=str, help="directory of the checkpoints")
    cmd.add_argument("--use_cache", action="store_true")
    cmd.add_argument("--last_k_tokens", type=int, default=-1)
    cmd.add_argument(
        "--enable_chunk_prefill",
        action="store_true",
        help="Enable segmented chunk prefill (requires --segment_size > 0). Off by default.",
    )
    cmd.add_argument(
        "--segment_size",
        type=int,
        default=-1,
        help="Chunk prefill segment size; only used with --enable_chunk_prefill",
    )
    cmd.add_argument(
        "--inference_segment",
        type=int,
        default=-1,
        help="Deprecated alias of --segment_size",
    )
    cmd.add_argument("--max_samples", default=-1, type=int, help="max samples to eval")
    cmd.add_argument("--parallel_mode", default="fsdp1", type=str)
    cmd.add_argument("--adjust_lmk_pos", action="store_true")
    cmd.add_argument("--use_hf_model", action="store_true")
    cmd.add_argument(
        "--skip_hsa_prefill",
        action="store_true",
        help="Skip HSA layers on non-answer segments during chunk prefill",
    )
    cmd.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_3",
        help="Attention backend for HF models (same as eval_ruler_hf)",
    )
    cmd.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallel size. >1 disables chunk prefill and runs full forward split across GPUs",
    )
    cmd.add_argument("--summary_log", default=None, type=str, help="append final summary to this log file")
    args = cmd.parse_args(sys.argv[1:])
    print(args)
    main(args)
