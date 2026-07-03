"""
LongBench-v2 Cloze Evaluation for Base Models (with Chunk Prefill)

For each sample, concatenate prompt + option_text for each of the 4 options,
do a forward pass (with chunk prefill), compute avg log-prob over the option
tokens, and pick the option with lowest perplexity.

This is 4x forward per sample (one per option), suitable for base models
that have NOT been instruction-tuned.

Reports results split by 'length' field: short / medium / long.

Usage:
    python eval/eval_longbench2_cloze.py \
        --checkpoint_path /path/to/hf_ckpt \
        --config_path configs/olmo3_7B/olmo3_lhsa_innerx.json \
        --vocab_dir ./configs/olmo3_vocab/ \
        --segment_size 4096 \
        --save_dir results/longbench2_cloze
"""

import os
import sys
import json
import math
import argparse
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


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
        print("Using OLMo LHSA implementation")
    else:
        from models.FlashHiLS.modeling_qwen_hils import HiLSForCausalLM
        print("Using Qwen LHSA implementation")
    return HiLSForCausalLM


def _need_hsa(config_path=None, checkpoint_path=None):
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            mt = json.load(f).get("model_type", "")
        return "hsa" in mt or "lhsa" in mt
    return False


def load_model(args, device):
    use_hsa = _need_hsa(args.config_path, args.checkpoint_path)

    if use_hsa:
        from models.FlashHiLS.configuration_hsa import HSAConfig
        HiLSForCausalLM = resolve_hsa_class(args.config_path, args.checkpoint_path)
        AutoConfig.register("olmo_lhsa", HSAConfig)
        HiLSForCausalLM.config_class = HSAConfig
        AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3' if use_hsa else 'flash_attention_2',
        'device_map': device,
    }

    if args.checkpoint_path:
        if args.config_path:
            config = AutoConfig.from_pretrained(args.config_path)
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
        model = AutoModelForCausalLM.from_config(config, **model_kwargs).to(device)

    model.eval()
    return model


# ============================================================
#  Prompt template (base model: no chat template, no system prompt)
# ============================================================
TEMPLATE = """\
{context}

{question}

"""


def build_prompt(item):
    return TEMPLATE.format(
        context=item['context'].strip(),
        question=item['question'].strip(),
    )


# ============================================================
#  Cloze scoring with chunk prefill
# ============================================================
def score_one_option(model, prompt_ids, option_ids,
                     max_input_tokens=0, segment_size=0, device=None):
    """
    Compute average log-probability of option_ids given prompt_ids.
    Uses chunk prefill for the prompt part, then scores the option tokens.

    Returns: (avg_log_prob, total_seq_len)
    """
    input_ids = torch.cat([prompt_ids, option_ids], dim=0)  # (seq_len,)
    prompt_len = len(prompt_ids)
    option_len = len(option_ids)

    # Truncate from the left (keep tail)
    if max_input_tokens > 0 and len(input_ids) > max_input_tokens:
        input_ids = input_ids[-max_input_tokens:]
        prompt_len = len(input_ids) - option_len

    input_ids = input_ids.unsqueeze(0).to(device)  # (1, seq_len)
    seq_len = input_ids.shape[1]
    use_chunk = segment_size > 0

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
        if not use_chunk:
            # ==================== Full forward ====================
            # We only need logits for the option region: positions [prompt_len-1, seq_len-1)
            outputs = model(input_ids=input_ids, use_cache=False,
                            logits_to_keep=option_len + 1)
            # outputs.logits shape: (1, option_len+1, vocab_size) — last option_len+1 positions
            logits = outputs.logits[0]  # (option_len+1, vocab_size)
            del outputs
        else:
            # ==================== Chunk Prefill ====================
            # Prefill prompt in chunks (no logits needed), then score option chunk
            num_segments = (seq_len + segment_size - 1) // segment_size
            # Which segment does the option scoring start?
            # We need logits from position (prompt_len - 1) onward
            answer_start = max(0, prompt_len - 1)
            first_answer_seg = answer_start // segment_size

            past_key_values = None
            answer_logits_list = []

            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, seq_len)

                seg_input_ids = input_ids[:, start_idx:end_idx]
                seg_cache_pos = torch.arange(start_idx, end_idx, device=device)

                keep = (end_idx - start_idx) if i >= first_answer_seg else 0

                out = model(
                    input_ids=seg_input_ids,
                    cache_position=seg_cache_pos,
                    use_cache=True,
                    past_key_values=past_key_values,
                    logits_to_keep=keep,
                )
                past_key_values = out.past_key_values

                if i >= first_answer_seg:
                    answer_logits_list.append(out.logits.cpu())

                del out

            del past_key_values

            answer_region_logits = torch.cat(answer_logits_list, dim=1)
            del answer_logits_list

            # Extract the logits we need: from answer_start to seq_len-1
            offset = answer_start - first_answer_seg * segment_size
            logits = answer_region_logits[0, offset:offset + option_len + 1, :]  # (option_len+1, vocab)
            logits = logits.to(device)
            del answer_region_logits

    torch.cuda.empty_cache()

    # Compute log-prob of each option token
    # logits[t] predicts token at position (prompt_len - 1 + t + 1) = (prompt_len + t)
    # The target tokens are input_ids[0, prompt_len : prompt_len + option_len]
    targets = input_ids[0, prompt_len:prompt_len + option_len].to(torch.long)  # (option_len,)
    # logits to use: logits[0:option_len] (predicting positions prompt_len .. prompt_len+option_len-1)
    pred_logits = logits[:option_len, :]  # (option_len, vocab)
    log_probs = F.log_softmax(pred_logits, dim=-1)  # (option_len, vocab)

    token_log_probs = log_probs[torch.arange(option_len), targets.cpu()]  # (option_len,)
    avg_log_prob = token_log_probs.sum().item() / option_len

    return avg_log_prob, seq_len


def evaluate_cloze(model, tokenizer, prompt, options,
                   max_input_tokens=0, segment_size=0, device=None):
    """
    Score each option via cloze (prompt + option_text), pick lowest PPL.
    Returns: (pred_letter, scores_dict, max_seq_len_across_options)
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0]  # (prompt_len,)

    option_scores = {}
    max_seq_len = 0

    for i, option_text in enumerate(options):
        letter = chr(65 + i)  # A, B, C, D
        option_ids = tokenizer(option_text, return_tensors="pt",
                               truncation=False, add_special_tokens=False).input_ids[0]
        avg_lp, seq_len = score_one_option(
            model, prompt_ids, option_ids,
            max_input_tokens=max_input_tokens,
            segment_size=segment_size,
            device=device,
        )
        option_scores[letter] = avg_lp
        max_seq_len = max(max_seq_len, seq_len)

    # Pick highest avg log-prob (= lowest PPL)
    best = max(option_scores, key=option_scores.get)
    return best, option_scores, max_seq_len


# ============================================================
#  Result aggregation by length & difficulty
# ============================================================
def aggregate_results(out_file):
    if not os.path.exists(out_file):
        return

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    with open(out_file, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            length_cat = obj.get('length', 'unknown')
            diff_cat = obj.get('difficulty', 'unknown')
            is_correct = obj.get('judge', False)

            for key in ['all',
                        f'len:{length_cat}',
                        f'diff:{diff_cat}',
                        f'len:{length_cat}|diff:{diff_cat}']:
                cat_total[key] += 1
                if is_correct:
                    cat_correct[key] += 1

    def _print_row(label, key):
        t = cat_total.get(key, 0)
        c = cat_correct.get(key, 0)
        acc = f"{c/t:.2%}" if t > 0 else "N/A"
        print(f"  {label:<28} {c:>6}/{t:<6} {acc:>8}")

    print(f"\n{'='*60}")
    print("  LongBench-v2 Cloze Results")
    print(f"{'='*60}")

    _print_row("Overall", "all")

    print(f"\n  --- By Length ---")
    for cat in ['short', 'medium', 'long']:
        _print_row(cat, f'len:{cat}')

    print(f"\n  --- By Difficulty ---")
    for cat in ['easy', 'medium', 'hard']:
        _print_row(cat, f'diff:{cat}')

    print(f"\n  --- Length × Difficulty ---")
    for l_cat in ['short', 'medium', 'long']:
        for d_cat in ['easy', 'medium', 'hard']:
            key = f'len:{l_cat}|diff:{d_cat}'
            if cat_total.get(key, 0) > 0:
                _print_row(f"{l_cat} × {d_cat}", key)

    known_keys = {'all'}
    for l in ['short', 'medium', 'long']:
        known_keys.add(f'len:{l}')
        for d in ['easy', 'medium', 'hard']:
            known_keys.add(f'diff:{d}')
            known_keys.add(f'len:{l}|diff:{d}')
    extra = sorted(k for k in cat_total if k not in known_keys)
    if extra:
        print(f"\n  --- Other ---")
        for k in extra:
            _print_row(k, k)

    print(f"{'='*60}")


# ============================================================
#  Per-GPU worker
# ============================================================
def get_pred(data, args, out_path, rank):
    device = torch.device(f'cuda:{rank}')
    model = load_model(args, device)

    tokenizer_path = args.vocab_dir if args.vocab_dir else args.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    fout = open(out_path, 'a', encoding='utf-8')
    segment_str = f"segment_size={args.segment_size}" if args.segment_size > 0 else "full_forward"
    print(f"[GPU {rank}] Cloze eval: {len(data)} samples, {segment_str}")

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    eval_start = time.time()

    with torch.cuda.device(rank):
        for item in tqdm(data, desc=f"GPU {rank}"):
            prompt = build_prompt(item)
            options = [item['choice_A'], item['choice_B'], item['choice_C'], item['choice_D']]

            pred, scores, seq_len = evaluate_cloze(
                model, tokenizer, prompt, options,
                max_input_tokens=args.max_input_tokens,
                segment_size=args.segment_size,
                device=device,
            )

            length_cat = item.get('length', 'unknown')
            diff_cat = item.get('difficulty', 'unknown')
            item['pred'] = pred
            item['judge'] = (pred == item['answer'])
            item['scores'] = {k: round(v, 4) for k, v in scores.items()}
            item['seq_len'] = seq_len

            for key in ['all', f'len:{length_cat}', f'diff:{diff_cat}']:
                cat_total[key] += 1
                if item['judge']:
                    cat_correct[key] += 1

            total = cat_total['all']
            correct = cat_correct['all']

            if total % 10 == 0 or total <= 3:
                per_len = " ".join(
                    f"{c}={cat_correct.get(f'len:{c}',0)}/{cat_total.get(f'len:{c}',0)}"
                    for c in ['short', 'medium', 'long'] if cat_total.get(f'len:{c}', 0) > 0
                )
                ppls = {k: f"{-v:.2f}" for k, v in scores.items()}
                print(f"[GPU {rank}] seq_len={seq_len}, pred={pred}, answer={item['answer']}, "
                      f"ppls={ppls}, acc={correct/total:.2%} ({correct}/{total}) | {per_len}")

            item['context'] = item['context'][:1000]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()

    fout.close()
    elapsed = time.time() - eval_start
    total = cat_total['all']
    correct = cat_correct['all']
    print(f"[GPU {rank}] Done. {correct}/{total} = {correct/total:.2%}, "
          f"{elapsed:.1f}s ({elapsed/max(total,1):.1f}s/sample)" if total > 0 else f"[GPU {rank}] No data")


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
    print(f"Total data: {len(data_all)}")

    length_dist = defaultdict(int)
    for item in data_all:
        length_dist[item.get('length', 'unknown')] += 1
    print(f"Length distribution: {dict(length_dist)}")

    # Resume
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    data = [item for item in data_all if item["_id"] not in has_data]
    print(f"Lines to eval: {len(data)}")

    if len(data) == 0:
        print("All samples already evaluated.")
        aggregate_results(out_file)
        return

    if args.n_proc == 1:
        get_pred(data, args, out_file, rank=0)
    else:
        data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
        processes = []
        for rank in range(args.n_proc):
            p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, rank))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    aggregate_results(out_file)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('LongBench-v2 Cloze Evaluation (Base Model)')

    # --- Model ---
    cmd.add_argument('--checkpoint_path', type=str, required=True,
                     help='Path to HF checkpoint')
    cmd.add_argument('--config_path', type=str, default=None,
                     help='Path to model config (overrides ckpt config, required for HSA models)')
    cmd.add_argument('--vocab_dir', type=str, default=None,
                     help='Path to tokenizer (default: use checkpoint_path)')

    # --- Data ---
    cmd.add_argument('--local_dataset', type=str, default=None,
                     help='Path to local LongBench-v2 JSON (default: download from HF)')
    cmd.add_argument('--save_dir', '-s', type=str, default='results',
                     help='Output directory')

    # --- Eval ---
    cmd.add_argument('--max_input_tokens', type=int, default=0,
                     help='Max input tokens (0 = no limit, truncate from left when set)')
    cmd.add_argument('--segment_size', type=int, default=4096,
                     help='Chunk prefill segment size (<=0 = full forward, may OOM on long docs)')
    cmd.add_argument('--n_proc', '-n', type=int, default=1,
                     help='Number of GPUs for parallel eval')

    args = cmd.parse_args()
    main(args)
