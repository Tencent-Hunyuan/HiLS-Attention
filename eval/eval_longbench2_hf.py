"""
LongBench-v2 Cloze Evaluation (HF model loading aligned with eval_ruler_hf.py)

Evaluates models on LongBench-v2 multiple-choice (cloze) task using log-probability
scoring via a single forward pass per option.

Usage:
    python eval/eval_longbench2_hf.py \
        --checkpoint_path /path/to/hf_ckpt \
        --vocab_dir ./configs/olmo3_vocab/ \
        --config_path configs/olmo3_7B/olmo3_hils_dropout.json \
        --local_dataset /path/to/longbench_v2.json \
        --max_input_tokens 65000 \
        --save_dir results/longbench2 \
        --n_proc 1
"""

import os
import sys
import json
import math
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from models.FlashHiLS.configuration_hils import HSAConfig


# ============================================================
#  Seed
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
#  Model loading (aligned with eval_ruler_hf.py)
# ============================================================
def resolve_hsa_class(config_path=None, checkpoint_path=None):
    model_type = ""
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            model_type = json.load(f).get("model_type", "")
    if "olmo" in model_type:
        from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM
        print("Using OLMo HiLS implementation")
    else:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        print("Using Qwen HiLS implementation")
    return HiLSForCausalLM


def load_model(args, device):
    HiLSForCausalLM = resolve_hsa_class(args.config_path, args.checkpoint_path)
    AutoConfig.register("olmo_hils", HSAConfig)
    HiLSForCausalLM.config_class = HSAConfig
    AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3',
        'device_map': device,
    }

    if args.checkpoint_path:
        if args.auto_insert_lmk:
            model_kwargs['auto_insert_lmk'] = True
        if args.config_path:
            config = AutoConfig.from_pretrained(args.config_path)
            if args.auto_insert_lmk:
                config.auto_insert_lmk = True
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint_path, config=config, **model_kwargs
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.checkpoint_path, **model_kwargs
            )
    else:
        assert args.config_path is not None, "必须提供 --config_path 或 --checkpoint_path"
        config = AutoConfig.from_pretrained(args.config_path)
        config.auto_insert_lmk = args.auto_insert_lmk
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    model.eval()
    return model


# ============================================================
#  Prompt template
# ============================================================
TEMPLATE = """\
$DOC$

$Q$

"""


# ============================================================
#  Forward-based cloze scoring
# ============================================================
def evaluate_cloze_fwd(model, tokenizer, prompt, options,
                       max_input_tokens=65000, rank=0, pad_to_multiple=1):
    device = torch.device(f"cuda:{rank}")
    option_scores = []

    for option in options:
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0]
        prompt_len = len(prompt_ids)
        option_ids = tokenizer(option, return_tensors="pt", truncation=False).input_ids[0]
        option_len = len(option_ids)
        input_ids = torch.cat([prompt_ids, option_ids], dim=0)

        # Truncate (keep the tail)
        if len(input_ids) > max_input_tokens:
            input_ids = input_ids[-max_input_tokens:]

        # Padding
        original_length = len(input_ids)
        target_length = math.ceil(original_length / pad_to_multiple) * pad_to_multiple
        padding_length = target_length - original_length
        if padding_length > 0:
            padding_tensor = torch.full((padding_length,), -100, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding_tensor])

        actual_prompt_len = min(prompt_len, len(input_ids) - option_len)

        inputs = {
            "input_ids": input_ids.unsqueeze(0).int().to(device),
        }

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        logits = outputs.logits[0]  # [seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        log_prob = 0.0
        input_ids_cpu = input_ids.cpu().numpy()
        total_len = input_ids_cpu.shape[0]
        actual_option_len = len(input_ids) - actual_prompt_len

        for i in range(actual_prompt_len - 1, original_length - 1):
            target_token = int(input_ids_cpu[i + 1])
            neg_index = i - total_len
            log_prob += log_probs[neg_index][target_token]

        avg_log_prob = log_prob / actual_option_len
        option_scores.append(avg_log_prob)

    perplexities = [-score for score in option_scores]

    for i, option in enumerate(options):
        print(f"Option {chr(65 + i)} PPL: {perplexities[i]:.4f} \t{option}")

    best_option_index = perplexities.index(min(perplexities))
    return chr(65 + best_option_index)


# ============================================================
#  Generative cloze scoring (force-decode)
# ============================================================
class ForceOptionLogitsProcessor(torch.nn.Module):
    def __init__(self, tokenizer, prefix_ids):
        super().__init__()
        self.prefix_ids = prefix_ids
        self.step_counter = 0
        self.original_logits = []

    def forward(self, input_ids, scores):
        if self.step_counter < len(self.prefix_ids):
            self.original_logits.append(scores.detach().clone())
            forced_token_id = self.prefix_ids[self.step_counter]
            mask = torch.full_like(scores, float("-inf"))
            mask[:, forced_token_id] = 0
            scores = scores + mask
            self.step_counter += 1
        return scores


def evaluate_cloze_generative(model, tokenizer, prompt, options,
                              max_input_tokens=65000, rank=0):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    seq_length = input_ids.shape[1]

    if seq_length > max_input_tokens:
        inputs["input_ids"] = input_ids[:, -max_input_tokens:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_tokens:]

    device = torch.device(f"cuda:{rank}")
    inputs = inputs.to(device)

    option_scores = []
    for option in options:
        option_id = tokenizer.encode(option)
        option_len = len(option_id)
        option_processor = ForceOptionLogitsProcessor(tokenizer, option_id)

        tqdm.write(f"inputs.shape: {inputs['input_ids'].shape}")
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=option_len,
                logits_processor=[option_processor],
                output_scores=True,
                return_dict_in_generate=True,
                num_beams=1,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
        if not torch.equal(generated_ids.cpu(), torch.tensor(option_id)):
            raise ValueError("Forced generation failed.")

        log_prob = 0.0
        assert len(option_processor.original_logits) == option_len
        for step in range(option_len):
            logits = option_processor.original_logits[step][0]
            prob = torch.log_softmax(logits, dim=-1)
            log_prob += prob[option_id[step]].item()

        avg_log_prob = log_prob / option_len
        option_scores.append(avg_log_prob)

    best_option_index = option_scores.index(max(option_scores))
    return chr(65 + best_option_index)


# ============================================================
#  Per-GPU worker
# ============================================================
def get_pred(data, args, out_path, rank, pad_to_multiple=1):
    device = torch.device(f'cuda:{rank}')
    model = load_model(args, device)
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir, trust_remote_code=True)

    fout = open(out_path, 'a', encoding='utf-8')
    print(f"[GPU {rank}] get_pred len: {len(data)}")

    correct = 0
    total = 0

    with torch.cuda.device(rank):
        for item in tqdm(data, desc=f"GPU {rank}"):
            context = item['context']
            options = [item['choice_A'], item['choice_B'], item['choice_C'], item['choice_D']]
            prompt = TEMPLATE.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip())

            if args.generative:
                pred = evaluate_cloze_generative(
                    model, tokenizer, prompt, options,
                    max_input_tokens=args.max_input_tokens, rank=rank,
                )
            else:
                pred = evaluate_cloze_fwd(
                    model, tokenizer, prompt, options,
                    max_input_tokens=args.max_input_tokens, rank=rank,
                    pad_to_multiple=pad_to_multiple,
                )

            item['pred'] = pred
            item['judge'] = (pred == item['answer'])
            print(f"pred: {pred}, answer: {item['answer']}")

            item['context'] = context[:1000]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()

            total += 1
            if item['judge']:
                correct += 1

            print({
                'acc': f"{correct / total:.2%}" if total > 0 else "0.00%",
                'correct': correct,
                'total': total,
            })

    fout.close()


# ============================================================
#  Main
# ============================================================
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    mp.set_start_method('spawn', force=True)

    out_file = os.path.join(args.save_dir, "cloze_result.jsonl")

    if args.local_dataset is None:
        dataset = load_dataset('THUDM/LongBench-v2', split='train')
    else:
        dataset = json.load(open(args.local_dataset, 'r', encoding='utf-8'))

    fields = [
        "_id", "domain", "sub_domain", "difficulty", "length",
        "question", "choice_A", "choice_B", "choice_C", "choice_D",
        "answer", "context",
    ]
    data_all = [{k: item[k] for k in fields} for item in dataset]
    print(f"total data: {len(data_all)}")

    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    data = [item for item in data_all if item["_id"] not in has_data]
    print(f"total lines to eval: {len(data)}")

    if len(data) == 0:
        print("All samples already evaluated, nothing to do.")
        return

    if args.n_proc == 1:
        get_pred(data, args, out_file, rank=0, pad_to_multiple=args.pad_to_multiple)
    else:
        data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
        processes = []
        for rank in range(args.n_proc):
            p = mp.Process(
                target=get_pred,
                args=(data_subsets[rank], args, out_file, rank, args.pad_to_multiple),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    if os.path.exists(out_file):
        correct = 0
        total = 0
        with open(out_file, encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                total += 1
                if obj.get('judge', False):
                    correct += 1
        print(f"\n{'='*60}")
        print(f"Final Results: {correct}/{total} = {correct/total:.2%}" if total > 0 else "No results")
        print(f"{'='*60}")


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('LongBench-v2 Cloze Evaluation (HF)')

    # --- Model ---
    cmd.add_argument('--checkpoint_path', type=str, required=True, help='Path to HF checkpoint')
    cmd.add_argument('--config_path', type=str, default=None, help='Path to model config (overrides ckpt config)')
    cmd.add_argument('--vocab_dir', type=str, required=True, help='Path to tokenizer vocab dir')
    cmd.add_argument('--auto_insert_lmk', action='store_true', help='Let model internally insert LMK tokens')

    # --- Data ---
    cmd.add_argument('--local_dataset', type=str, default=None, help='Path to local LongBench-v2 JSON')
    cmd.add_argument('--save_dir', '-s', type=str, default='results', help='Output directory')

    # --- Eval ---
    cmd.add_argument('--max_input_tokens', type=int, default=65000, help='Max input tokens (truncate from left)')
    cmd.add_argument('--pad_to_multiple', type=int, default=1, help='Pad sequence length to multiple of this')
    cmd.add_argument('--generative', action='store_true', help='Use generative (force-decode) mode instead of forward')
    cmd.add_argument('--n_proc', '-n', type=int, default=1, help='Number of GPUs for parallel eval')

    args = cmd.parse_args()
    main(args)
