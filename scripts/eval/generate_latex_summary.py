#!/usr/bin/env python3
"""Parse per-model summary logs and produce LaTeX tables for PPL and RULER."""

import argparse
import json
import os
import re
import sys

_STEP_RE = re.compile(r"^(.+?)-step(\d+)$")
_PPL_RE = re.compile(
    r"Test Length:\s*(\d+),\s*Final Mean Loss:\s*[\d.]+,\s*PPL:\s*([\d.]+)"
)
RULER_TASK_SHORT = {0: "S-N", 1: "MQ-N", 2: "VT"}
_PPL_SEQ_LENS = [64, 128, 512, 8192, 16384, 65536, 131072, 262144, 524288, 1048576]


def _len_label(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


def _step_label(step_str: str) -> str:
    if step_str == "-":
        return "-"
    try:
        n = int(step_str)
    except ValueError:
        return step_str
    if n >= 1000:
        val = n / 1000
        return f"{int(val)}k" if val == int(val) else f"{val:g}k"
    return step_str


def _split_name_step(model_name: str):
    match = _STEP_RE.match(model_name)
    if match:
        return match.group(1), _step_label(match.group(2))
    return model_name, "-"


def parse_ppl_log(path: str) -> dict:
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = _PPL_RE.search(line)
            if match:
                results[int(match.group(1))] = float(match.group(2))
    return results


def parse_ruler_log(path: str) -> dict:
    results = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (int(obj["max_seq_len"]), int(obj["task_id"]))
            results[key] = float(obj["exact_match_rate"])
    return results


def collect(log_dir: str):
    ppl_data, ruler_data, seen_order = {}, {}, []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".summary.log"):
            continue
        fpath = os.path.join(log_dir, fname)
        if fname.endswith(".ppl.summary.log"):
            model_name = fname.rsplit(".ppl.summary.log", 1)[0]
            if model_name not in seen_order:
                seen_order.append(model_name)
            ppl_data[model_name] = parse_ppl_log(fpath)
        elif ".ruler.task" in fname:
            model_name = fname.split(".ruler.task")[0]
            if model_name not in seen_order:
                seen_order.append(model_name)
            ruler_data.setdefault(model_name, {}).update(parse_ruler_log(fpath))
    return ppl_data, ruler_data, seen_order


def make_ppl_table(ppl_data, model_order):
    if not ppl_data:
        return ""
    present_lens = {length for data in ppl_data.values() for length in data}
    all_lens = [length for length in _PPL_SEQ_LENS if length in present_lens or length == 64]
    if not all_lens:
        return ""

    len_labels = [_len_label(length) for length in all_lens]
    lines = [
        "% ── PPL Table ──",
        r"\begin{tabular}{" + "l|l|c|" + "c" * len(all_lens) + "}",
        r"\toprule",
        "Models & Steps & & " + " & ".join(len_labels) + r" \\",
        r"\midrule",
    ]

    grouped, prev_base = [], None
    for model_name in model_order:
        if model_name not in ppl_data:
            continue
        base, step = _split_name_step(model_name)
        if prev_base is None or base != prev_base:
            grouped.append((base, []))
            prev_base = base
        grouped[-1][1].append((model_name, step))

    for base, members in grouped:
        for i, (model_name, step) in enumerate(members):
            data = ppl_data[model_name]
            vals = [f"{data[length]:.2f}" if length in data else "-" for length in all_lens]
            if i == 0 and len(members) > 1:
                base_cell = f"\\multirow{{{len(members)}}}{{*}}{{{base}}}"
            elif i == 0:
                base_cell = base
            else:
                base_cell = ""
            lines.append(f"{base_cell} & {step} & & " + " & ".join(vals) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def make_ruler_table(ruler_data, model_order):
    if not ruler_data:
        return ""
    all_lens = sorted({key[0] for data in ruler_data.values() for key in data})
    all_tasks = sorted({key[1] for data in ruler_data.values() for key in data})
    if not all_lens or not all_tasks:
        return ""

    n_tasks, n_lens = len(all_tasks), len(all_lens)
    task_labels = [RULER_TASK_SHORT.get(task, f"T{task}") for task in all_tasks]
    lines = [
        "% ── RULER Table ──",
        r"\begin{tabular}{" + "l|l|" + "|".join(["c" * n_tasks] * n_lens) + "}",
        r"\toprule",
    ]

    header = ["\\multirow{2}{*}{Models}", "\\multirow{2}{*}{Steps}"]
    for length in all_lens:
        header.append(f"\\multicolumn{{{n_tasks}}}{{c|}}{{{_len_label(length)}}}")
    lines.append(" & ".join(header) + r" \\")

    subheader = ["", ""]
    for _ in all_lens:
        subheader.extend(task_labels)
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")

    grouped, prev_base = [], None
    for model_name in model_order:
        if model_name not in ruler_data:
            continue
        base, step = _split_name_step(model_name)
        if prev_base is None or base != prev_base:
            grouped.append((base, []))
            prev_base = base
        grouped[-1][1].append((model_name, step))

    for base, members in grouped:
        for i, (model_name, step) in enumerate(members):
            data = ruler_data[model_name]
            vals = []
            for length in all_lens:
                for task in all_tasks:
                    key = (length, task)
                    vals.append(f"{int(round(data[key] * 100))}" if key in data else "")
            if i == 0 and len(members) > 1:
                base_cell = f"\\multirow{{{len(members)}}}{{*}}{{{base}}}"
            elif i == 0:
                base_cell = base
            else:
                base_cell = ""
            lines.append(f"{base_cell} & {step} & " + " & ".join(vals) + r" \\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX summary tables")
    parser.add_argument("log_dir", help="Directory containing *.summary.log files")
    parser.add_argument("-o", "--output", default=None, help="Output file (default: LOG_DIR/summary.log)")
    args = parser.parse_args()

    ppl_data, ruler_data, model_order = collect(args.log_dir)
    parts = [
        "% Auto-generated evaluation summary (LaTeX)",
        f"% Log dir: {os.path.abspath(args.log_dir)}",
        "",
    ]
    ppl_table = make_ppl_table(ppl_data, model_order)
    if ppl_table:
        parts.extend([ppl_table, ""])
    ruler_table = make_ruler_table(ruler_data, model_order)
    if ruler_table:
        parts.extend([ruler_table, ""])

    output_text = "\n".join(parts)
    out_path = args.output or os.path.join(args.log_dir, "summary.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(output_text)
    print(f"\nSummary written to: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
