"""
LongBench v1 Evaluation — Generate predictions then score with official metrics.

Standard generative evaluation: model.generate() → compare with reference answers
using task-specific metrics (F1, ROUGE-L, classification accuracy, etc.)

Supports:
  - Standard HF models (Olmo, Llama, etc.)
  - HSA models with custom config

Usage:
    python eval/eval_longbench_v1.py \
        --checkpoint_path /path/to/hf_ckpt \
        --vocab_dir ./configs/olmo3_vocab/ \
        --max_length 65536 \
        --segment_size 4096 \
        --save_dir results/longbench_v1 \
        --n_proc 1
"""

import os
import sys
import json
import math
import argparse
import random
import re
import string
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# Optional deps for metrics
try:
    import jieba
except ImportError:
    jieba = None
try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None
try:
    from rouge import Rouge
except ImportError:
    Rouge = None


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


def _need_hsa(config_path=None, checkpoint_path=None):
    path = config_path or (os.path.join(checkpoint_path, "config.json") if checkpoint_path else None)
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            mt = json.load(f).get("model_type", "")
        return "hsa" in mt or "hils" in mt
    return False


def load_model(args, device):
    use_hsa = _need_hsa(args.config_path, args.checkpoint_path)

    if use_hsa:
        from models.FlashHiLS.configuration_hils import HSAConfig
        HiLSForCausalLM = resolve_hsa_class(args.config_path, args.checkpoint_path)
        AutoConfig.register("olmo_hils", HSAConfig)
        HiLSForCausalLM.config_class = HSAConfig
        AutoModelForCausalLM.register(HSAConfig, HiLSForCausalLM)

    model_kwargs = {
        'torch_dtype': torch.bfloat16,
        'attn_implementation': 'flash_attention_3' if use_hsa else 'flash_attention_2',
        'device_map': device,
        'trust_remote_code': True,
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
#  LongBench v1 prompt templates & max generation lengths
# ============================================================
DATASET2PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

DATASET2MAXGEN = {
    "narrativeqa": 128, "qasper": 128, "multifieldqa_en": 64, "multifieldqa_zh": 64,
    "hotpotqa": 32, "2wikimqa": 32, "musique": 32, "dureader": 128,
    "gov_report": 512, "qmsum": 512, "multi_news": 512, "vcsum": 512,
    "trec": 64, "triviaqa": 32, "samsum": 128, "lsht": 64,
    "passage_count": 32, "passage_retrieval_en": 32, "passage_retrieval_zh": 32,
    "lcc": 64, "repobench-p": 64,
}

ALL_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]

# Task categories for reporting
TASK_CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}


# ============================================================
#  Metrics (from official LongBench metrics.py)
# ============================================================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    def white_space_fix(text):
        return "".join(text.split())
    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s)))


def _f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction, ground_truth, **kwargs):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    return _f1_score(pred_tokens, gt_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    assert jieba is not None, "pip install jieba"
    pred_tokens = [normalize_zh_answer(t) for t in jieba.cut(prediction, cut_all=False)]
    gt_tokens = [normalize_zh_answer(t) for t in jieba.cut(ground_truth, cut_all=False)]
    pred_tokens = [t for t in pred_tokens if len(t) > 0]
    gt_tokens = [t for t in gt_tokens if len(t) > 0]
    return _f1_score(pred_tokens, gt_tokens)


def rouge_score(prediction, ground_truth, **kwargs):
    assert Rouge is not None, "pip install rouge"
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    assert jieba is not None, "pip install jieba"
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    return rouge_score(prediction, ground_truth)


def classification_score(prediction, ground_truth, **kwargs):
    all_classes = kwargs["all_classes"]
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in list(em_match_list):
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r'Paragraph (\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r'段落(\d+)'
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth_id))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if str(n) == str(ground_truth))
    return 0.0 if len(numbers) == 0 else right_num / len(numbers)


def code_sim_score(prediction, ground_truth, **kwargs):
    assert fuzz is not None, "pip install fuzzywuzzy"
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


DATASET2METRIC = {
    "narrativeqa": qa_f1_score, "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score, "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score, "2wikimqa": qa_f1_score,
    "musique": qa_f1_score, "dureader": rouge_zh_score,
    "gov_report": rouge_score, "qmsum": rouge_score,
    "multi_news": rouge_score, "vcsum": rouge_zh_score,
    "trec": classification_score, "triviaqa": qa_f1_score,
    "samsum": rouge_score, "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score, "repobench-p": code_sim_score,
}


# ============================================================
#  Truncation: middle truncation (keep head + tail)
# ============================================================
def truncate_middle(tokenizer, prompt, max_length):
    """Truncate from the middle, preserving head and tail, as per LongBench paper."""
    tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized) <= max_length:
        return prompt
    half = max_length // 2
    head = tokenizer.decode(tokenized[:half], skip_special_tokens=True)
    tail = tokenizer.decode(tokenized[-half:], skip_special_tokens=True)
    return head + tail


# ============================================================
#  Prediction: generate for one dataset
# ============================================================
def predict_dataset(model, tokenizer, data, dataset_name, max_length,
                    segment_size, device, out_path, use_chat_template=False):
    """Generate predictions for all samples in a dataset."""
    prompt_template = DATASET2PROMPT[dataset_name]
    max_gen = DATASET2MAXGEN[dataset_name]

    fout = open(out_path, 'a', encoding='utf-8')
    count = 0

    for item in tqdm(data, desc=dataset_name):
        prompt = prompt_template.format(**item)

        # Middle truncation (truncate the raw prompt before wrapping with chat template)
        if max_length > 0:
            prompt = truncate_middle(tokenizer, prompt, max_length)

        # Apply chat template for instruct models
        # Following official LongBench: skip chat template for few-shot and code tasks,
        # as these tasks need direct continuation rather than conversational response.
        SKIP_CHAT_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
        if use_chat_template and dataset_name not in SKIP_CHAT_DATASETS:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
            )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)

        # 清理 generate 状态和 KV cache，释放显存
        if hasattr(model, '_gen_state'):
            model._gen_state.reset()
        torch.cuda.empty_cache()

        result = {
            "pred": pred,
            "answers": item["answers"],
            "all_classes": item["all_classes"],
            "length": item["length"],
        }
        fout.write(json.dumps(result, ensure_ascii=False) + '\n')
        fout.flush()
        count += 1

    fout.close()
    return count


# ============================================================
#  Evaluation: score predictions with official metrics
# ============================================================
def evaluate_predictions(pred_dir):
    """Score all .jsonl files in pred_dir, return per-dataset scores."""
    scores = {}
    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith('.jsonl'):
            continue
        dataset_name = filename.replace('.jsonl', '')
        if dataset_name not in DATASET2METRIC:
            continue

        predictions, answers, all_classes_list = [], [], None
        with open(os.path.join(pred_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                predictions.append(obj["pred"])
                answers.append(obj["answers"])
                all_classes_list = obj.get("all_classes")

        metric_fn = DATASET2METRIC[dataset_name]
        total_score = 0.0
        for pred, ground_truths in zip(predictions, answers):
            # For some tasks, take only the first line
            if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip('\n').split('\n')[0]
            # Max over multiple references
            score = 0.0
            for gt in ground_truths:
                score = max(score, metric_fn(pred, gt, all_classes=all_classes_list))
            total_score += score

        avg_score = round(100 * total_score / max(len(predictions), 1), 2)
        scores[dataset_name] = avg_score
        print(f"  {dataset_name}: {avg_score:.2f}")

    return scores


def print_category_summary(scores):
    """Print scores grouped by task category."""
    print(f"\n{'='*70}")
    print(f"  {'Category':<20} {'Datasets':>10} {'Avg Score':>12}")
    print(f"{'='*70}")

    all_scores = []
    for cat_name, datasets in TASK_CATEGORIES.items():
        cat_scores = [scores[d] for d in datasets if d in scores]
        if cat_scores:
            avg = round(np.mean(cat_scores), 2)
            all_scores.extend(cat_scores)
            detail = ", ".join(f"{d}={scores[d]}" for d in datasets if d in scores)
            print(f"  {cat_name:<20} {len(cat_scores):>10} {avg:>12.2f}")
            print(f"    {detail}")

    if all_scores:
        print(f"{'='*70}")
        print(f"  {'Overall':<20} {len(all_scores):>10} {round(np.mean(all_scores), 2):>12.2f}")
    print(f"{'='*70}")


# Length buckets: (label, lower_bound_inclusive, upper_bound_exclusive)
LENGTH_BUCKETS = [
    ("0-2k",     0,      2048),
    ("2k-4k",    2048,   4096),
    ("4k-8k",    4096,   8192),
    ("8k-16k",   8192,   16384),
    ("16k-32k",  16384,  32768),
    ("32k-64k",  32768,  65536),
    ("64k-128k", 65536,  131072),
    ("128k+",    131072, float('inf')),
]


def _get_bucket(length):
    for label, lo, hi in LENGTH_BUCKETS:
        if lo <= length < hi:
            return label
    return LENGTH_BUCKETS[-1][0]


def evaluate_predictions_by_length(pred_dir):
    """Score predictions grouped by input length bucket.

    Returns:
        bucket_scores: dict of {bucket_label: {dataset_name: (avg_score, count)}}
        category_bucket_scores: dict of {category: {bucket_label: (avg_score, count)}}
    """
    # Collect per-sample scores with length info
    # Structure: {dataset_name: [(score, length), ...]}
    dataset_sample_scores = {}

    for filename in sorted(os.listdir(pred_dir)):
        if not filename.endswith('.jsonl'):
            continue
        dataset_name = filename.replace('.jsonl', '')
        if dataset_name not in DATASET2METRIC:
            continue

        samples = []
        with open(os.path.join(pred_dir, filename), 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                samples.append(obj)

        metric_fn = DATASET2METRIC[dataset_name]
        scored = []
        for obj in samples:
            pred = obj["pred"]
            ground_truths = obj["answers"]
            length = obj.get("length", 0)
            all_classes = obj.get("all_classes")

            if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip('\n').split('\n')[0]

            score = 0.0
            for gt in ground_truths:
                score = max(score, metric_fn(pred, gt, all_classes=all_classes))
            scored.append((score, length))

        dataset_sample_scores[dataset_name] = scored

    # Aggregate by bucket
    bucket_labels = [b[0] for b in LENGTH_BUCKETS]

    # Per-dataset per-bucket
    dataset_bucket = {}  # {ds: {bucket: [scores]}}
    for ds, scored in dataset_sample_scores.items():
        dataset_bucket[ds] = {b: [] for b in bucket_labels}
        for score, length in scored:
            bucket = _get_bucket(length)
            dataset_bucket[ds][bucket].append(score)

    # Per-category per-bucket
    category_bucket = {}
    for cat_name, datasets in TASK_CATEGORIES.items():
        category_bucket[cat_name] = {b: [] for b in bucket_labels}
        for ds in datasets:
            if ds not in dataset_bucket:
                continue
            for b in bucket_labels:
                category_bucket[cat_name][b].extend(dataset_bucket[ds][b])

    # Overall per-bucket
    overall_bucket = {b: [] for b in bucket_labels}
    for ds, buckets in dataset_bucket.items():
        for b, scores in buckets.items():
            overall_bucket[b].extend(scores)

    return dataset_bucket, category_bucket, overall_bucket


def print_length_summary(dataset_bucket, category_bucket, overall_bucket):
    """Print score table grouped by length bucket."""
    bucket_labels = [b[0] for b in LENGTH_BUCKETS]

    def _fmt_cell(scores_list):
        if not scores_list:
            return "   -   "
        avg = 100 * np.mean(scores_list)
        return f"{avg:5.1f}/{len(scores_list):<3d}"

    # Header
    header = f"  {'':.<28s}"
    for b in bucket_labels:
        header += f" {b:>9s}"
    header += f" {'All':>9s}"

    print(f"\n{'='*len(header)}")
    print("  Length-bucketed scores (score / #samples)")
    print(f"{'='*len(header)}")
    print(header)
    print(f"  {'':-<{len(header)-2}s}")

    # Per-category rows
    for cat_name, datasets in TASK_CATEGORIES.items():
        # Category header
        cat_all = []
        for ds in datasets:
            if ds not in dataset_bucket:
                continue
            row = f"    {ds:<26s}"
            ds_all = []
            for b in bucket_labels:
                scores = dataset_bucket[ds][b]
                row += f" {_fmt_cell(scores):>9s}"
                ds_all.extend(scores)
            row += f" {_fmt_cell(ds_all):>9s}"
            cat_all.extend(ds_all)
            print(row)

        # Category summary
        cat_row = f"  [{cat_name}]"
        cat_row = f"  {cat_row:<26s}"
        for b in bucket_labels:
            cat_row += f" {_fmt_cell(category_bucket[cat_name][b]):>9s}"
        cat_row += f" {_fmt_cell(cat_all):>9s}"
        print(cat_row)
        print()

    # Overall
    overall_all = []
    overall_row = f"  {'Overall':<26s}"
    for b in bucket_labels:
        scores = overall_bucket[b]
        overall_row += f" {_fmt_cell(scores):>9s}"
        overall_all.extend(scores)
    overall_row += f" {_fmt_cell(overall_all):>9s}"
    print(f"  {'':-<{len(header)-2}s}")
    print(overall_row)
    print(f"{'='*len(header)}")


# ============================================================
#  Per-GPU worker
# ============================================================
def worker(rank, data_by_dataset, args, pred_dir):
    device = torch.device(f'cuda:{rank}')
    model = load_model(args, device)

    tokenizer_path = args.vocab_dir if args.vocab_dir else args.checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Detect chat template
    use_chat_template = (
        hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    ) or os.path.exists(os.path.join(args.checkpoint_path, 'chat_template.jinja'))

    if args.no_chat_template:
        use_chat_template = False

    print(f"[GPU {rank}] use_chat_template={use_chat_template}")

    for dataset_name, data in data_by_dataset.items():
        out_path = os.path.join(pred_dir, f"{dataset_name}.jsonl")
        print(f"[GPU {rank}] Predicting {dataset_name}: {len(data)} samples")
        predict_dataset(
            model, tokenizer, data, dataset_name,
            max_length=args.max_length,
            segment_size=args.segment_size,
            device=device,
            out_path=out_path,
            use_chat_template=use_chat_template,
        )


# ============================================================
#  Main
# ============================================================
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    pred_dir = os.path.join(args.save_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    print(args)

    mp.set_start_method('spawn', force=True)

    # Select datasets
    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets = ALL_DATASETS

    # Load data
    all_data = {}
    for dataset_name in datasets:
        pred_path = os.path.join(pred_dir, f"{dataset_name}.jsonl")
        # Count already predicted lines for resume
        existing = 0
        if os.path.exists(pred_path):
            with open(pred_path, 'r') as f:
                existing = sum(1 for _ in f)

        data = list(load_dataset('THUDM/LongBench', dataset_name, split='test', trust_remote_code=True))

        if existing >= len(data):
            print(f"  {dataset_name}: already done ({existing}/{len(data)}), skipping prediction")
            continue

        # Skip already evaluated samples
        data = data[existing:]
        if data:
            all_data[dataset_name] = data
            print(f"  {dataset_name}: {len(data)} samples to predict ({existing} done)")

    # Run predictions
    if all_data:
        if args.n_proc == 1:
            worker(0, all_data, args, pred_dir)
        else:
            # Distribute datasets across GPUs (round-robin)
            gpu_tasks = [dict() for _ in range(args.n_proc)]
            for i, (ds_name, ds_data) in enumerate(all_data.items()):
                gpu_tasks[i % args.n_proc][ds_name] = ds_data

            processes = []
            for rank in range(args.n_proc):
                if gpu_tasks[rank]:
                    p = mp.Process(target=worker, args=(rank, gpu_tasks[rank], args, pred_dir))
                    p.start()
                    processes.append(p)
            for p in processes:
                p.join()

    # Evaluate
    print("\n--- Evaluating ---")
    scores = evaluate_predictions(pred_dir)

    # Save results
    result_path = os.path.join(args.save_dir, "result.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(f"\nResults saved to: {result_path}")

    # Print summary
    print_category_summary(scores)

    # Print length-bucketed summary
    dataset_bucket, category_bucket, overall_bucket = evaluate_predictions_by_length(pred_dir)
    print_length_summary(dataset_bucket, category_bucket, overall_bucket)


if __name__ == "__main__":
    cmd = argparse.ArgumentParser('LongBench v1 Evaluation (Generate + Score)')

    # --- Model ---
    cmd.add_argument('--checkpoint_path', type=str, required=True,
                     help='Path to HF checkpoint')
    cmd.add_argument('--config_path', type=str, default=None,
                     help='Path to model config (overrides ckpt config, required for HSA models)')
    cmd.add_argument('--vocab_dir', type=str, default=None,
                     help='Path to tokenizer (default: use checkpoint_path)')

    # --- Data ---
    cmd.add_argument('--datasets', type=str, default=None,
                     help='Comma-separated dataset names (default: all 21 tasks)')
    cmd.add_argument('--save_dir', '-s', type=str, default='results/longbench_v1',
                     help='Output directory')

    # --- Inference ---
    cmd.add_argument('--max_length', type=int, default=65536,
                     help='Max input length (middle truncation, 0=no limit)')
    cmd.add_argument('--no_chat_template', action='store_true',
                     help='Disable chat template even if tokenizer has one (for base models)')
    cmd.add_argument('--segment_size', type=int, default=0,
                     help='Reserved for future chunk prefill (not used for generate)')
    cmd.add_argument('--n_proc', '-n', type=int, default=1,
                     help='Number of GPUs for parallel prediction')

    args = cmd.parse_args()
    main(args)
