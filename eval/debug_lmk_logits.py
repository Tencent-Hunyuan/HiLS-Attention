import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.FlashHiLS.configuration_hsa import HSAConfig
from utils.landmark_utils import create_position_ids_with_landmarks, insert_special_tokens


def get_abs_err(x, y):
    return (x - y).abs().max().item()


def get_err_ratio(x, y):
    err = (x - y).float().square().mean().sqrt().item()
    base = x.float().square().mean().sqrt().item()
    return err / max(base, 1e-12)


def resolve_hsa_class(config_path=None, checkpoint_path=None):
    model_type = ""
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            model_type = json.load(f).get("model_type", "")

    if "olmo" in model_type:
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
        return HiLSForCausalLM, "olmo_lhsa"
    if "qwen" in model_type:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        return HiLSForCausalLM, "qwen_lhsa"

    raise ValueError(f"无法识别 model_type: {model_type or '<empty>'}")


def load_model(args, auto_insert_lmk):
    HiLSForCausalLM, resolved_model_type = resolve_hsa_class(args.config_path, args.checkpoint_path)

    DebugHSAConfig = type(
        "DebugHSAConfig",
        (HSAConfig,),
        {"model_type": resolved_model_type},
    )

    AutoConfig.register(resolved_model_type, DebugHSAConfig, exist_ok=True)
    HiLSForCausalLM.config_class = DebugHSAConfig
    AutoModelForCausalLM.register(DebugHSAConfig, HiLSForCausalLM, exist_ok=True)

    config = AutoConfig.from_pretrained(args.config_path or args.checkpoint_path)
    config.auto_insert_lmk = bool(auto_insert_lmk)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_3",
    }
    if auto_insert_lmk:
        model_kwargs["auto_insert_lmk"] = True

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint_path, config=config, **model_kwargs)
    model = model.to(args.device)
    model.eval()
    return model


def build_inputs(tokenizer, text, device):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    return {k: v.to(device) for k, v in encoded.items()}


def insert_attention_mask(attention_mask, chunk_size):
    if attention_mask is None:
        return None
    fill_id = True if attention_mask.dtype == torch.bool else 1
    return insert_special_tokens(attention_mask, fill_id=fill_id, chunk_size=chunk_size)


def run_manual_insert(model, tokenizer, raw_inputs, chunk_size, adjust_lmk_pos):
    input_ids = raw_inputs["input_ids"]
    attention_mask = raw_inputs.get("attention_mask")
    lmk_id = tokenizer.vocab_size

    inserted_input_ids = insert_special_tokens(input_ids, fill_id=lmk_id, chunk_size=chunk_size)
    inserted_attention_mask = insert_attention_mask(attention_mask, chunk_size=chunk_size)
    position_ids = None
    if adjust_lmk_pos:
        position_ids = create_position_ids_with_landmarks(
            input_ids.shape[1], chunk_size=chunk_size, device=input_ids.device
        )

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(
            input_ids=inserted_input_ids,
            attention_mask=inserted_attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits

    pos_indices = torch.arange(inserted_input_ids.shape[1], device=input_ids.device)
    non_lmk_mask = ~(pos_indices % chunk_size == chunk_size - 1)
    filtered_logits = logits[:, non_lmk_mask, :]

    return {
        "inserted_input_ids": inserted_input_ids,
        "inserted_attention_mask": inserted_attention_mask,
        "filtered_logits": filtered_logits.float(),
        "raw_logits": logits.float(),
    }


def run_auto_insert(model, raw_inputs):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**raw_inputs, use_cache=False).logits
    return logits.float()


def summarize_last_tokens(tokenizer, logits_a, logits_b, input_ids, last_k):
    shift_labels = input_ids[:, 1:]
    compare_len = min(last_k, shift_labels.shape[1], logits_a.shape[1] - 1, logits_b.shape[1] - 1)
    if compare_len <= 0:
        return

    a = logits_a[:, -compare_len - 1 : -1, :]
    b = logits_b[:, -compare_len - 1 : -1, :]
    labels = shift_labels[:, -compare_len:]
    pred_a = a.argmax(dim=-1)
    pred_b = b.argmax(dim=-1)

    print(f"\nLast {compare_len} next-token predictions:")
    for i in range(compare_len):
        label_id = labels[0, i].item()
        pa = pred_a[0, i].item()
        pb = pred_b[0, i].item()
        print(
            f"  idx={i:02d} "
            f"label={label_id:>6}({repr(tokenizer.decode([label_id]))}) "
            f"manual={pa:>6}({repr(tokenizer.decode([pa]))}) "
            f"auto={pb:>6}({repr(tokenizer.decode([pb]))})"
        )


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir, trust_remote_code=True)
    raw_inputs = build_inputs(tokenizer, args.text, args.device)
    auto_model = load_model(args, auto_insert_lmk=True)

    chunk_size = int(getattr(auto_model.config, "chunk_size", args.chunk_size))
    adjust_lmk_pos = bool(getattr(auto_model.config, "adjust_lmk_pos", False))

    print(f"seq_len={raw_inputs['input_ids'].shape[1]}, chunk_size={chunk_size}, adjust_lmk_pos={adjust_lmk_pos}")

    auto_logits = run_auto_insert(auto_model, raw_inputs)

    if args.compare_no_attention_mask:
        raw_inputs_no_mask = {"input_ids": raw_inputs["input_ids"]}
        auto_logits_no_mask = run_auto_insert(auto_model, raw_inputs_no_mask)
        abs_err_no_mask = get_abs_err(auto_logits, auto_logits_no_mask)
        err_ratio_no_mask = get_err_ratio(auto_logits, auto_logits_no_mask)
        print(
            "auto(with attention_mask) vs auto(without attention_mask) "
            f"max_abs={abs_err_no_mask:.6f}, err_ratio={err_ratio_no_mask:.6f}"
        )
        summarize_last_tokens(
            tokenizer,
            auto_logits,
            auto_logits_no_mask,
            raw_inputs["input_ids"],
            args.last_k_tokens,
        )

    if args.manual_compare:
        manual_model = load_model(args, auto_insert_lmk=False)
        manual = run_manual_insert(manual_model, tokenizer, raw_inputs, chunk_size, adjust_lmk_pos)

        print(f"manual filtered logits shape: {tuple(manual['filtered_logits'].shape)}")
        print(f"auto logits shape:            {tuple(auto_logits.shape)}")
        if manual["filtered_logits"].shape != auto_logits.shape:
            raise RuntimeError("manual filtered logits 和 auto logits shape 不一致")

        abs_err = get_abs_err(manual["filtered_logits"], auto_logits)
        err_ratio = get_err_ratio(manual["filtered_logits"], auto_logits)
        print(f"manual vs auto logits max_abs={abs_err:.6f}, err_ratio={err_ratio:.6f}")

        summarize_last_tokens(
            tokenizer,
            manual["filtered_logits"],
            auto_logits,
            raw_inputs["input_ids"],
            args.last_k_tokens,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Debug LMK logits alignment")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vocab_dir", type=str, required=True)
    parser.add_argument("--text", type=str, required=True, help="单条测试文本")
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--last_k_tokens", type=int, default=8)
    parser.add_argument("--compare_no_attention_mask", action="store_true")
    parser.add_argument("--manual_compare", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    main(parser.parse_args())
