"""
LongBench-v2 Direct Choice Evaluation (with Chunk Prefill)

Single forward pass per sample. Look at the logit of A/B/C/D at the last
token position and pick the highest one. Uses chunk prefill to avoid OOM
on long sequences.

Reports results split by 'length' field: short / medium / long.

Supports both:
  - Standard HF models (e.g. Olmo-3-7B-Instruct-SFT)
  - HSA models with custom config (e.g. olmo3_hils_innerx)

Usage:
    python eval/eval_longbench2_direct.py \
        --checkpoint_path /path/to/hf_ckpt \
        --config_path configs/olmo3_7B/olmo3_hils_innerx.json \
        --vocab_dir ./configs/olmo3_vocab/ \
        --segment_size 4096 \
        --save_dir results/longbench2_direct
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
#  Model loading (aligned with eval_ruler_hf.py / eval_longbench2_hf.py)
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


def _need_hsa(config_path=None, checkpoint_path=None):
    """Check if the model is an HSA model (has qwen_hils / olmo_hils model_type)."""
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            mt = json.load(f).get("model_type", "")
        return "hsa" in mt or "hils" in mt
    return False


def load_model(args, device):
    """
    Unified model loading:
      - If config_path points to an HSA config → register HSA class, load with config override
      - Otherwise → standard AutoModelForCausalLM.from_pretrained
\
Document:
{context}

Question: {question}

Choices:
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}

Answer:"""


def build_prompt_chat(tokenizer, item):
    """Build prompt using the model's chat template."""
    user_content = USER_TEMPLATE.format(
        context=item['context'].strip(),
        question=item['question'].strip(),
        choice_A=item['choice_A'],
        choice_B=item['choice_B'],
        choice_C=item['choice_C'],
        choice_D=item['choice_D'],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def build_prompt_plain(item):
    """Fallback plain prompt (no chat template)."""
    return USER_TEMPLATE.format(
        context=item['context'].strip(),
        question=item['question'].strip(),
        choice_A=item['choice_A'],
        choice_B=item['choice_B'],
        choice_C=item['choice_C'],
        choice_D=item['choice_D'],
    )


# ============================================================
#  Direct choice with chunk prefill
# ============================================================
def evaluate_direct_choice(model, tokenizer, prompt, choice_token_ids,
                           max_input_tokens=0, segment_size=0, device=None):
    """
    Single forward pass with optional chunk prefill.
    Returns the letter (A/B/C/D) whose token has the highest logit
    at the last position, plus the seq_len used.

    choice_token_ids: dict  e.g. {'A': 32, 'B': 33, 'C': 34, 'D': 35}
    max_input_tokens: 0 = no limit
    segment_size: >0 = chunk prefill, <=0 = full forward
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_ids = inputs["input_ids"]

    # Truncate from the left (keep tail) only when limit is set
    if max_input_tokens > 0 and input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, -max_input_tokens:]

    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    use_chunk_prefill = segment_size > 0

    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
        if not use_chunk_prefill:
            # ==================== Full forward ====================
            outputs = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
            last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
            del outputs
        else:
            # ==================== Chunk Prefill ====================
            num_segments = (seq_len + segment_size - 1) // segment_size
            past_key_values = None
            last_logits = None

            for i in range(num_segments):
                start_idx = i * segment_size
                end_idx = min((i + 1) * segment_size, seq_len)
                is_last = (i == num_segments - 1)

                seg_input_ids = input_ids[:, start_idx:end_idx]
                seg_cache_pos = torch.arange(start_idx, end_idx, device=device)

                out = model(
                    input_ids=seg_input_ids,
                    cache_position=seg_cache_pos,
                    use_cache=True,
                    past_key_values=past_key_values,
                    logits_to_keep=1 if is_last else 0,
                )
                past_key_values = out.past_key_values

                if is_last:
                    last_logits = out.logits[0, -1, :]  # (vocab_size,)

                del out

            del past_key_values

    torch.cuda.empty_cache()

    # Compare A/B/C/D
    scores = {}
    for letter, tid in choice_token_ids.items():
        scores[letter] = last_logits[tid].item()

    best = max(scores, key=scores.get)
    return best, scores, seq_len


# ============================================================
#  Result aggregation by length & difficulty
# ============================================================
def aggregate_results(out_file):
    """Print overall + per-length + per-difficulty + cross-tab accuracy."""
    if not os.path.exists(out_file):
        return

    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)

    with open(out_file, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            length_cat = obj.get('length', 'unknown')       # short / medium / long
            diff_cat = obj.get('difficulty', 'unknown')      # easy / medium / hard
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
    print("  LongBench-v2 Results")
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

    # Print any unknown categories
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

    # tokenizer: prefer --vocab_dir, fallback to --checkpoint_path
    tokenizer_path = args.vocab_dir if args.vocab_dir else args.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Resolve A/B/C/D token ids from the vocab
    choice_token_ids = {}
    for letter in ['A', 'B', 'C', 'D']:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(ids) == 1, f"Token '{letter}' encodes to multiple ids: {ids}"
        choice_token_ids[letter] = ids[0]
    print(f"[GPU {rank}] Choice token ids: {choice_token_ids}")

    # Check if tokenizer has a chat template
    has_chat_template = (
        hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    ) or (
        hasattr(tokenizer, 'apply_chat_template')
        and os.path.exists(os.path.join(args.checkpoint_path, 'chat_template.jinja'))
    )

    fout = open(out_path, 'a', encoding='utf-8')
    segment_str = f"segment_size={args.segment_size}" if args.segment_size > 0 else "full_forward"
    print(f"[GPU {rank}] Evaluating {len(data)} samples, chat_template={has_chat_template}, {segment_str}")

    # per-category counters for live progress
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    eval_start = time.time()

    with torch.cuda.device(rank):
        for item in tqdm(data, desc=f"GPU {rank}"):
            # Build prompt
            if has_chat_template:
                prompt = build_prompt_chat(tokenizer, item)
            else:
                prompt = build_prompt_plain(item)

            pred, scores, seq_len = evaluate_direct_choice(
                model, tokenizer, prompt, choice_token_ids,
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
                per_diff = " ".join(
                    f"{c}={cat_correct.get(f'diff:{c}',0)}/{cat_total.get(f'diff:{c}',0)}"
                    for c in ['easy', 'medium', 'hard'] if cat_total.get(f'diff:{c}', 0) > 0
                )
                print(f"[GPU {rank}] seq_len={seq_len}, pred={pred}, answer={item['answer']}, "
                      f"acc={correct/total:.2%} ({correct}/{total}) | {per_len} | {per_diff}")

            # Save (truncate context to save disk)
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

    out_file = os.path.join(args.save_dir, "direct_choice_result.jsonl")

    # Load data
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

    # Print length distribution
    length_dist = defaultdict(int)
    for item in data_all:
        length_dist[item.get('length', 'unknown')] += 1
    print(f"Length distribution: {dict(length_dist)}")

    # Resume: skip already evaluated
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

    # Aggregate results by length category
    aggregate_results(out_file)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('LongBench-v2 Direct Choice Evaluation')

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
