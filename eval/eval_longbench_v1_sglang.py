"""
LongBench v1 Evaluation using SGLang offline Engine.

Generate predictions via sgl.Engine (supports HSA models with continuous
batching), then score with official LongBench metrics.

Usage:
    python eval/eval_longbench_v1_sglang.py \
        --hf-path /path/to/resolved_hf_ckpt \
        --save-dir results/longbench_v1_sglang \
        --max-length 65536 \
        --sglang-tp 1 \
        --sglang-page-size 64 \
        --sglang-max-total-tokens 131072 \
        --sglang-mem-fraction-static 0.90 \
        --sglang-batch-size 8 \
        [--sglang-attention-backend hsa] \
        [--datasets hotpotqa,qasper]
"""

# ============================================================
# NOTE: Only stdlib / non-sglang imports at top level.
# SGLang Engine forks subprocesses that re-import this module;
# anything that touches sys.argv or triggers CUDA init must be
# guarded under `if __name__ == "__main__"`.
# ============================================================

import os
import json
import re
import string
import random
from collections import Counter

import numpy as np

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
            if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
                pred = pred.lstrip('\n').split('\n')[0]
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

    bucket_labels = [b[0] for b in LENGTH_BUCKETS]

    dataset_bucket = {}
    for ds, scored in dataset_sample_scores.items():
        dataset_bucket[ds] = {b: [] for b in bucket_labels}
        for score, length in scored:
            bucket = _get_bucket(length)
            dataset_bucket[ds][bucket].append(score)

    category_bucket = {}
    for cat_name, datasets in TASK_CATEGORIES.items():
        category_bucket[cat_name] = {b: [] for b in bucket_labels}
        for ds in datasets:
            if ds not in dataset_bucket:
                continue
            for b in bucket_labels:
                category_bucket[cat_name][b].extend(dataset_bucket[ds][b])

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

    header = f"  {'':.<28s}"
    for b in bucket_labels:
        header += f" {b:>9s}"
    header += f" {'All':>9s}"

    print(f"\n{'='*len(header)}")
    print("  Length-bucketed scores (score / #samples)")
    print(f"{'='*len(header)}")
    print(header)
    print(f"  {'':-<{len(header)-2}s}")

    for cat_name, datasets in TASK_CATEGORIES.items():
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

        cat_row = f"  [{cat_name}]"
        cat_row = f"  {cat_row:<26s}"
        for b in bucket_labels:
            cat_row += f" {_fmt_cell(category_bucket[cat_name][b]):>9s}"
        cat_row += f" {_fmt_cell(cat_all):>9s}"
        print(cat_row)
        print()

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
#  Everything below: ONLY executed as main script.
#  SGLang Engine forks subprocesses that re-import this module,
#  so CLI parsing / CUDA init must NOT happen at import time.
# ============================================================

if __name__ == "__main__":

    import sys
    import argparse
    import time
    from tqdm import tqdm
    from datasets import load_dataset
    from transformers import AutoTokenizer

    random.seed(42)
    np.random.seed(42)

    # ── Args ──
    parser = argparse.ArgumentParser("LongBench v1 — SGLang Engine")

    # Model / SGLang
    parser.add_argument("--hf-path", type=str, required=True,
                        help="Path to resolved HF checkpoint (with config.json + tokenizer)")
    parser.add_argument("--sglang-tp", type=int, default=1)
    parser.add_argument("--sglang-page-size", type=int, default=64)
    parser.add_argument("--sglang-max-total-tokens", type=int, default=131072)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.90)
    parser.add_argument("--sglang-batch-size", type=int, default=1,
                        help="Number of prompts to send in one engine.generate() call")
    parser.add_argument("--sglang-max-running-requests", type=int, default=None,
                        help="Max concurrent requests in SGLang scheduler "
                             "(set to 1 for HSA backend to prevent batch>1 in prefill)")
    parser.add_argument("--sglang-attention-backend", type=str, default=None,
                        help="Attention backend, e.g. 'hsa' for HSA models")
    parser.add_argument("--sglang-port", type=int, default=None,
                        help="Base port for SGLang Engine (avoids port conflicts when "
                             "running multiple engines on the same node)")

    # Data
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names (default: all 21 tasks)")
    parser.add_argument("--save-dir", "-s", type=str, default="results/longbench_v1_sglang")
    parser.add_argument("--max-length", type=int, default=65536,
                        help="Max input length (middle truncation, 0=no limit)")

    # Execution mode
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate predictions, skip evaluation scoring")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate existing predictions, skip generation "
                             "(no SGLang Engine needed)")

    # Sample-level sharding: split samples within each dataset across multiple workers
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards for sample-level parallelism")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="This worker's shard index (0-based)")

    args = parser.parse_args()

    # ── Setup paths ──
    save_dir = args.save_dir
    pred_dir = os.path.join(save_dir, "pred")
    os.makedirs(pred_dir, exist_ok=True)
    print(args)

    # ── Select datasets ──
    if args.datasets:
        datasets_to_run = [d.strip() for d in args.datasets.split(',')]
    else:
        datasets_to_run = ALL_DATASETS

    # ── Tokenizer (needed for both generate and eval-only with truncation) ──
    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)

    # ── Generate predictions ──
    if not args.eval_only:
        import sglang as sgl

        engine_kwargs = dict(
            model_path=args.hf_path,
            tp_size=args.sglang_tp,
            max_total_tokens=args.sglang_max_total_tokens,
            mem_fraction_static=args.sglang_mem_fraction_static,
            log_level="info",
        )
        if args.sglang_attention_backend:
            engine_kwargs["attention_backend"] = args.sglang_attention_backend
            engine_kwargs["page_size"] = args.sglang_page_size
            engine_kwargs["disable_cuda_graph"] = True
            engine_kwargs["disable_overlap_schedule"] = True
            # HSA backend 只支持 bsz=1，限制 prefill 和 running 并发
            engine_kwargs["prefill_max_requests"] = 1
            engine_kwargs["max_running_requests"] = 1

        if args.sglang_port is not None:
            engine_kwargs["port"] = args.sglang_port

        if args.sglang_max_running_requests is not None:
            engine_kwargs["max_running_requests"] = args.sglang_max_running_requests

        print(f"[SGLang] Creating Engine: {engine_kwargs}")
        engine = sgl.Engine(**engine_kwargs)

        # Warmup
        page_size = engine_kwargs.get("page_size", 64)
        warmup_prompt = "Hello " * max(page_size * 2, 128)
        print("[SGLang] Warmup...")
        engine.generate(prompt=warmup_prompt,
                        sampling_params={"max_new_tokens": 3, "temperature": 0.0})
        print("[SGLang] Engine ready.")

        batch_size = args.sglang_batch_size
        total_start = time.time()

        for dataset_name in datasets_to_run:
            if dataset_name not in DATASET2PROMPT:
                print(f"  [WARN] Unknown dataset '{dataset_name}', skipping")
                continue

            # Determine output path: sharded or normal
            if args.num_shards > 1:
                out_path = os.path.join(pred_dir, f"{dataset_name}.shard{args.shard_id}.jsonl")
            else:
                out_path = os.path.join(pred_dir, f"{dataset_name}.jsonl")

            # Resume: count existing predictions
            existing = 0
            if os.path.exists(out_path):
                with open(out_path, 'r') as f:
                    existing = sum(1 for _ in f)

            data = list(load_dataset('THUDM/LongBench', dataset_name,
                                     split='test', trust_remote_code=True))

            # Sample-level sharding: each shard takes a contiguous slice
            if args.num_shards > 1:
                total = len(data)
                shard_size = (total + args.num_shards - 1) // args.num_shards
                shard_start = args.shard_id * shard_size
                shard_end = min(shard_start + shard_size, total)
                data = data[shard_start:shard_end]
                print(f"  [Shard {args.shard_id}/{args.num_shards}] "
                      f"{dataset_name}: samples [{shard_start}:{shard_end}) = {len(data)}")

            if existing >= len(data):
                print(f"  {dataset_name}: already done ({existing}/{len(data)}), skipping")
                continue

            data = data[existing:]
            print(f"\n[{dataset_name}] {len(data)} samples to predict ({existing} done)")

            prompt_template = DATASET2PROMPT[dataset_name]
            max_gen = DATASET2MAXGEN[dataset_name]

            # Build all prompts first
            prompts = []
            metadata = []
            for item in data:
                prompt = prompt_template.format(**item)
                if args.max_length > 0:
                    prompt = truncate_middle(tokenizer, prompt, args.max_length)
                prompts.append(prompt)
                metadata.append({
                    "answers": item["answers"],
                    "all_classes": item["all_classes"],
                    "length": item["length"],
                })

            # Batch generate with SGLang
            fout = open(out_path, 'a', encoding='utf-8')
            ds_start = time.time()

            for batch_start in tqdm(range(0, len(prompts), batch_size), desc=dataset_name):
                batch_prompts = prompts[batch_start:batch_start + batch_size]
                batch_meta = metadata[batch_start:batch_start + batch_size]

                if len(batch_prompts) == 1:
                    results = [engine.generate(
                        prompt=batch_prompts[0],
                        sampling_params={"max_new_tokens": max_gen, "temperature": 0.0},
                    )]
                else:
                    results = engine.generate(
                        prompt=batch_prompts,
                        sampling_params={"max_new_tokens": max_gen, "temperature": 0.0},
                    )

                for res, meta in zip(results, batch_meta):
                    pred = res["text"]
                    result = {
                        "pred": pred,
                        "answers": meta["answers"],
                        "all_classes": meta["all_classes"],
                        "length": meta["length"],
                    }
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()

            fout.close()
            elapsed = time.time() - ds_start
            print(f"  [{dataset_name}] Done in {elapsed:.1f}s "
                  f"({len(prompts) / max(elapsed, 0.1):.1f} samples/s)")

        print("\n[SGLang] Shutting down engine...")
        engine.shutdown()

        total_elapsed = time.time() - total_start
        print(f"\nTotal generation time: {total_elapsed:.1f}s")

    # ── Evaluate ──
    if not args.generate_only:
        # Merge shard files if any exist
        import glob
        for shard_file in sorted(glob.glob(os.path.join(pred_dir, "*.shard*.jsonl"))):
            # Extract dataset name: "hotpotqa.shard0.jsonl" -> "hotpotqa"
            base = os.path.basename(shard_file)
            dataset_name = base.split(".shard")[0]
            merged_path = os.path.join(pred_dir, f"{dataset_name}.jsonl")
            # Append shard content to merged file
            with open(shard_file, 'r') as sf, open(merged_path, 'a') as mf:
                mf.write(sf.read())
            os.remove(shard_file)
            print(f"  Merged {base} -> {dataset_name}.jsonl")

        print("\n--- Evaluating ---")
        scores = evaluate_predictions(pred_dir)

        result_path = os.path.join(save_dir, "result.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        print(f"\nResults saved to: {result_path}")

        print_category_summary(scores)

        # Print length-bucketed summary
        dataset_bucket, category_bucket, overall_bucket = evaluate_predictions_by_length(pred_dir)
        print_length_summary(dataset_bucket, category_bucket, overall_bucket)
