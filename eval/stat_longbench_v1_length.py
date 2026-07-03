"""
统计 LongBench v1 各 dataset 的 length 字段分布

Usage:
    python eval/stat_longbench_v1_length.py
"""

import numpy as np
from datasets import load_dataset

DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]

CATEGORIES = {
    "Single-Doc QA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}

def fmt(x):
    if x >= 1000:
        return f"{x/1000:.1f}k"
    return str(int(x))

def main():
    header = f"{'Dataset':<25} {'N':>5} {'Min':>7} {'Mean':>7} {'Med':>7} {'P90':>7} {'P95':>7} {'Max':>7}"
    sep = "=" * len(header)

    print(header)
    print(sep)

    all_lengths = []
    cat_lengths = {cat: [] for cat in CATEGORIES}

    for ds_name in DATASETS:
        try:
            data = load_dataset("THUDM/LongBench", ds_name, split="test", trust_remote_code=True)
            lengths = [item["length"] for item in data]
            all_lengths.extend(lengths)
            for cat, ds_list in CATEGORIES.items():
                if ds_name in ds_list:
                    cat_lengths[cat].extend(lengths)
            a = np.array(lengths)
            print(f"{ds_name:<25} {len(a):>5} {fmt(a.min()):>7} {fmt(a.mean()):>7} {fmt(np.median(a)):>7} {fmt(np.percentile(a,90)):>7} {fmt(np.percentile(a,95)):>7} {fmt(a.max()):>7}")
        except Exception as e:
            print(f"{ds_name:<25} ERROR: {e}")

    print(sep)
    print("\nBy Category:")
    print(sep)
    for cat, lengths in cat_lengths.items():
        if lengths:
            a = np.array(lengths)
            print(f"{cat:<25} {len(a):>5} {fmt(a.min()):>7} {fmt(a.mean()):>7} {fmt(np.median(a)):>7} {fmt(np.percentile(a,90)):>7} {fmt(np.percentile(a,95)):>7} {fmt(a.max()):>7}")

    print(sep)
    a = np.array(all_lengths)
    print(f"{'TOTAL':<25} {len(a):>5} {fmt(a.min()):>7} {fmt(a.mean()):>7} {fmt(np.median(a)):>7} {fmt(np.percentile(a,90)):>7} {fmt(np.percentile(a,95)):>7} {fmt(a.max()):>7}")

    # 长度分布直方图
    print(f"\nLength Distribution (token count):")
    bins = [0, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 999999]
    labels = ["0-2k", "2k-4k", "4k-8k", "8k-16k", "16k-32k", "32k-64k", "64k-128k", "128k+"]
    counts, _ = np.histogram(a, bins=bins)
    for label, count in zip(labels, counts):
        bar = "█" * (count // 20)
        print(f"  {label:<10} {count:>5}  {bar}")

    # 按类别分长度段统计
    print(f"\nLength Distribution by Category:")
    print(f"{'Category':<25} " + " ".join(f"{l:>8}" for l in labels))
    print("=" * (25 + 9 * len(labels)))
    for cat, lengths in cat_lengths.items():
        if lengths:
            ca = np.array(lengths)
            cnts, _ = np.histogram(ca, bins=bins)
            row = " ".join(f"{c:>8}" for c in cnts)
            print(f"{cat:<25} {row}")
    # Total
    cnts, _ = np.histogram(a, bins=bins)
    row = " ".join(f"{c:>8}" for c in cnts)
    print(f"{'TOTAL':<25} {row}")

    # Code 类别按 dataset 细分长度段
    print(f"\nCode Category - Length Distribution by Dataset:")
    print(f"{'Dataset':<25} " + " ".join(f"{l:>8}" for l in labels))
    print("=" * (25 + 9 * len(labels)))
    for ds_name in CATEGORIES["Code"]:
        try:
            data = load_dataset("THUDM/LongBench", ds_name, split="test", trust_remote_code=True)
            lengths = [item["length"] for item in data]
            ca = np.array(lengths)
            cnts, _ = np.histogram(ca, bins=bins)
            row = " ".join(f"{c:>8}" for c in cnts)
            print(f"{ds_name:<25} {row}")
        except Exception as e:
            print(f"{ds_name:<25} ERROR: {e}")

if __name__ == "__main__":
    main()
