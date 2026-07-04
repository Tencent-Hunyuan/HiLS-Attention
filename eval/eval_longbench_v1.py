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
Truncate from the middle, preserving head and tail, as per LongBench paper."""
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
