import os
import io
import json
import argparse
import multiprocessing as mp
from typing import List, Tuple

import numpy as np
from transformers import AutoTokenizer


def jsonl_token_generator(path, tokenizer):
    with open(path, "r", encoding="utf-8") as fh:
        accum_context = ""
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                accum_context += line
                obj = json.loads(accum_context)
                accum_context = ""
                encoded = tokenizer.encode(obj["text"])
                if tokenizer.eos_token_id is not None:
                    encoded.append(tokenizer.eos_token_id)
                yield encoded
            except json.JSONDecodeError:
                continue


def worker_tokenize_file_quota(file_path, vocab_dir, out_queue, batch_tokens, token_quota):
    tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
    buffer = []
    total = 0
    for tokens in jsonl_token_generator(file_path, tokenizer):
        if total >= token_quota:
            break
        remain = token_quota - total
        take = tokens if len(tokens) <= remain else tokens[:remain]
        buffer.extend(take)
        total += len(take)
        if len(buffer) >= batch_tokens:
            out_queue.put(buffer)
            buffer = []
    if buffer:
        out_queue.put(buffer)
    out_queue.put(None)


def collect_file_sizes(corpus_dir) -> List[Tuple[str, int]]:
    files = []
    for root, _, filenames in os.walk(corpus_dir):
        for name in filenames:
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size > 0:
                files.append((path, size))
    return files


def compute_quota_per_file(files: List[Tuple[str, int]], token_quota: int, seed: int = 1234):
    sizes = np.array([s for _, s in files], dtype=np.float64)
    total = sizes.sum()
    if total <= 0:
        raise ValueError("Total file size is zero.")
    weights = sizes / total
    base = np.floor(weights * token_quota).astype(np.int64)
    remainder = int(token_quota - base.sum())
    if remainder > 0:
        rng = np.random.default_rng(seed)
        extra = rng.choice(len(files), size=remainder, replace=True, p=weights)
        for idx in extra:
            base[idx] += 1
    return base.tolist()


def concurrent_weighted_quota_tokenize(
    corpus_dir,
    output_file,
    vocab_dir,
    token_quota,
    num_workers=8,
    batch_tokens=65536,
    buffer_tokens=1_000_000,
    seed=1234,
):
    files = collect_file_sizes(corpus_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {corpus_dir}")

    quotas = compute_quota_per_file(files, token_quota, seed=seed)
    file_targets = [(files[i][0], quotas[i]) for i in range(len(files)) if quotas[i] > 0]
    if not file_targets:
        raise ValueError("All per-file quotas are zero.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)

    ctx = mp.get_context("fork")

    def start_worker(path, quota):
        q = ctx.Queue(maxsize=2)
        p = ctx.Process(target=worker_tokenize_file_quota, args=(path, vocab_dir, q, batch_tokens, quota))
        p.daemon = True
        p.start()
        return {"proc": p, "queue": q, "path": path, "quota": quota}

    active = []
    idx = 0
    for _ in range(min(num_workers, len(file_targets))):
        path, quota = file_targets[idx]
        active.append(start_worker(path, quota))
        idx += 1

    total_tokens = 0
    buffer = []

    with open(output_file, "ab") as f_out:
        rr_idx = 0
        while active and total_tokens < token_quota:
            if rr_idx >= len(active):
                rr_idx = 0
            worker = active[rr_idx]
            rr_idx += 1

            batch = worker["queue"].get()
            if batch is None:
                worker["proc"].join()
                worker["queue"].close()
                active.remove(worker)
                if idx < len(file_targets):
                    path, quota = file_targets[idx]
                    active.append(start_worker(path, quota))
                    idx += 1
                continue

            buffer.extend(batch)
            total_tokens += len(batch)

            if len(buffer) >= buffer_tokens:
                np.array(buffer, dtype=np.uint32).tofile(f_out)
                buffer.clear()

            if total_tokens >= token_quota:
                break

        if buffer:
            np.array(buffer, dtype=np.uint32).tofile(f_out)

    for worker in active:
        if worker["proc"].is_alive():
            worker["proc"].terminate()
            worker["proc"].join()
        worker["queue"].close()

    print(f"Weighted-sampled tokens written: {total_tokens} -> {output_file}")


def main():
    parser = argparse.ArgumentParser("Weighted sampling tokenize (by file size) to token quota")
    parser.add_argument("--vocab_dir", required=True, type=str)
    parser.add_argument("--corpus_dir", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--token_quota", required=True, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_tokens", default=65536, type=int)
    parser.add_argument("--buffer_tokens", default=1_000_000, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    args = parser.parse_args()

    concurrent_weighted_quota_tokenize(
        corpus_dir=args.corpus_dir,
        output_file=args.output_file,
        vocab_dir=args.vocab_dir,
        token_quota=args.token_quota,
        num_workers=args.num_workers,
        batch_tokens=args.batch_tokens,
        buffer_tokens=args.buffer_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
