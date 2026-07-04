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
Truncate from the middle, preserving head and tail, as per LongBench paper."""
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
