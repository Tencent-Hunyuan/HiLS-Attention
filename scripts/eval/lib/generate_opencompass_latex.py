#!/usr/bin/env python3
"""Build a LaTeX summary table from OpenCompass eval log directories."""

import csv
import io
import os
import re
import sys

DATASET_ORDER = [
    ("mmlu_ppl_ac766d", "MMLU(5-shot)"),
    ("gpqa_few_shot_ppl_4b5a83", "GPQA(5-shot)"),
    ("hellaswag_10shot_ppl_59c85e", "Hellaswag(10-shot)"),
    ("ARC_c_few_shot_ppl", "ARC-c(25-shot)"),
    ("SuperGLUE_BoolQ_few_shot_ppl", "BoolQ(5-shot)"),
    ("race_few_shot_ppl", "Race(3-shot)"),
]
GEN_DATASETS = [
    ("gsm8k_gen", "GSM8K"),
    ("cmath_gen", "CMath"),
    ("humaneval_plus_gen", "HumanEval+"),
    ("mbpp_plus_gen", "MBPP+"),
    ("cruxeval_o_gen", "CruxEval-O"),
]


def find_summary(directory):
    for root, _, files in os.walk(directory):
        for name in sorted(files):
            if name.startswith("summary_") and name.endswith(".txt"):
                return os.path.join(root, name)
    return None


def extract_score(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    idx = text.find("csv format\n")
    if idx < 0:
        return None
    csv_text = text[idx + len("csv format\n") :]
    nl = csv_text.find("\n")
    if nl < 0:
        return None
    csv_text = csv_text[nl + 1 :].lstrip("\n")
    lines = [line for line in csv_text.splitlines() if line.strip() and "," in line]
    if not lines:
        return None
    for row in csv.reader(io.StringIO("\n".join(lines))):
        if row and row[0].strip() == "overall_average":
            for cell in reversed(row):
                try:
                    return float(cell.strip())
                except ValueError:
                    pass
    for row in csv.reader(io.StringIO("\n".join(lines))):
        if row and row[0].strip() != "dataset":
            for cell in reversed(row):
                try:
                    return float(cell.strip())
                except ValueError:
                    pass
    return None


def extract_gen_score(model_dir, ds_tag):
    ds_dir = os.path.join(model_dir, ds_tag)
    if not os.path.isdir(ds_dir):
        return None

    if "humaneval" in ds_tag:
        log_path = os.path.join(model_dir, "rescore_humaneval.log")
        if os.path.isfile(log_path):
            with open(log_path, encoding="utf-8") as f:
                text = f.read()
            for pattern in (r"humaneval_plus_plus_pass_1\s*=\s*([0-9.]+)", r"humaneval_plus_base_pass_1\s*=\s*([0-9.]+)"):
                match = re.search(pattern, text)
                if match:
                    return float(match.group(1))
    elif "mbpp" in ds_tag:
        log_path = os.path.join(model_dir, "rescore_mbpp.log")
        if os.path.isfile(log_path):
            with open(log_path, encoding="utf-8") as f:
                text = f.read()
            for pattern in (r"mbpp_plus_plus_pass_1\s*=\s*([0-9.]+)", r"mbpp_plus_base_pass_1\s*=\s*([0-9.]+)"):
                match = re.search(pattern, text)
                if match:
                    return float(match.group(1))

    summary = find_summary(ds_dir)
    if summary:
        return extract_score(summary)
    return None


def ds_tag(ds_internal):
    return re.sub(r"[^A-Za-z0-9_.\-]", "", ds_internal.replace(",", "_").replace("/", "_"))


def main():
    master = sys.argv[1]
    all_datasets = DATASET_ORDER + GEN_DATASETS
    models = sorted(d for d in os.listdir(master) if os.path.isdir(os.path.join(master, d)))
    scores = {}
    for model in models:
        scores[model] = {}
        model_dir = os.path.join(master, model)
        for ds_internal, _ in DATASET_ORDER:
            summary = find_summary(os.path.join(model_dir, ds_tag(ds_internal)))
            if summary:
                value = extract_score(summary)
                if value is not None:
                    scores[model][ds_internal] = value
        for ds_internal, _ in GEN_DATASETS:
            value = extract_gen_score(model_dir, ds_tag(ds_internal))
            if value is not None:
                scores[model][ds_internal] = value

    display = [name for _, name in all_datasets]
    header = "& " + " & ".join(display + ["AVG"]) + "\\\\"
    lines = [
        "% Auto-generated OpenCompass summary",
        f"% Log dir: {master}",
        "",
        header,
        "\\midrule",
    ]
    for model in models:
        vals, valid = [], []
        for ds, _ in all_datasets:
            value = scores[model].get(ds)
            if value is not None:
                vals.append(f"{value:.2f}")
                valid.append(value)
            else:
                vals.append("-")
        avg = f"{sum(valid) / len(valid):.2f}" if valid else "-"
        vals.append(avg)
        lines.append(f"{model} & " + " & ".join(vals) + "\\\\")

    output = "\n".join(lines) + "\n"
    out_path = os.path.join(master, "summary.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"Summary: {out_path}\n{output}")


if __name__ == "__main__":
    main()
