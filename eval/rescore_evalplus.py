#!/usr/bin/env python3
"""Offline re-scoring for HumanEval+ and MBPP+ predictions.

Usage:
    python eval/rescore_evalplus.py <pred.json> --dataset humaneval [--tag NAME]
    python eval/rescore_evalplus.py <pred.json> --dataset mbpp      [--tag NAME]

Takes an OpenCompass-format predictions JSON (the file inside
`.../predictions/hf_with_tokenizer_sglang/{humaneval_plus,mbpp_plus}.json`)
and reports pass@1 for base and plus modes using the SAME post-processing
logic as `eval/configs/datasets/custom_datasets.py`.

Requires `evalplus` to be importable. Run with the venv Python that has it
installed, e.g. /root/sglang/python/.venv/bin/python.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
import tempfile


# ── MBPP post-processor (mirrors MBPPPlusEvaluatorFixed._process_answer) ──

_BEGIN_DONE_PATTERNS = [
    r"\[BEGIN\]\s*'(.*?)'\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*?)'\s*DONE",
    r"\[BEGIN\]\s*(.*?)\s*\[DONE\]",
    r"\[BEGIN\](.*?)\[DONE\]",
    r"\[BEGIN\](.*)",
]


def _strip_quotes(text: str) -> str:
    t = text.strip()
    if t.startswith("'"):
        t = t[1:]
    t_rstrip = t.rstrip()
    if t_rstrip.endswith("'"):
        t = t_rstrip[:-1]
    return t


def process_mbpp_answer(text: str) -> str:
    for pat in _BEGIN_DONE_PATTERNS:
        m = re.search(pat, text, re.DOTALL)
        if m:
            return _strip_quotes(m.group(1)).strip()
    if '[DONE]' in text:
        text = text.split('[DONE]')[0]
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return _strip_quotes(text).strip()


# ── HumanEval post-processor (opencompass humaneval_postprocess_v2 behaviour) ──

def process_humaneval_answer(text: str) -> str:
    """Mirror of `humaneval_postprocess_v2`: keep the completion up to the
    first top-level `\\ndef ` or test-harness boundary."""
    # Split on common stop tokens that mark "model kept going"
    for stop in [
        '\ndef ',      # next function definition
        '\nclass ',
        '\nif __name__',
        '\n# Test',
        '\nprint(',
        '\nassert ',
        '\nTest:',
    ]:
        text = text.split(stop)[0]
    return text


# ── evalplus results parser (mirrors _read_evalplus_results) ──

def read_evalplus_results(samples_jsonl: str, prefix: str):
    results_path = samples_jsonl.replace('.jsonl', '_eval_results.json')
    if not osp.isfile(results_path):
        for cand in [samples_jsonl + '_eval_results.json',
                     samples_jsonl + '.eval_results.json']:
            if osp.isfile(cand):
                results_path = cand
                break
        else:
            print(f'[parse] no _eval_results.json near {samples_jsonl}',
                  file=sys.stderr)
            return None
    with open(results_path) as f:
        data = json.load(f)

    out = {}
    pak = data.get('pass_at_k', {}) or {}
    for mode in ('base', 'plus'):
        for k, v in (pak.get(mode) or {}).items():
            if isinstance(v, (int, float)):
                out[f'{prefix}_{mode}_pass_{k}'] = v * 100

    if not out and isinstance(data.get('eval'), dict):
        ev = data['eval']
        total = len(ev)
        if total > 0:
            base_ok = plus_ok = 0
            for info in ev.values():
                if isinstance(info, dict):
                    b = info.get('base') or []
                    p = info.get('plus') or []
                    if b and isinstance(b[0], (list, tuple)) and \
                            b[0] and b[0][0] == 'success':
                        base_ok += 1
                    if p and isinstance(p[0], (list, tuple)) and \
                            p[0] and p[0][0] == 'success':
                        plus_ok += 1
            out[f'{prefix}_base_pass_1'] = 100 * base_ok / total
            out[f'{prefix}_plus_pass_1'] = 100 * plus_ok / total

    if out:
        primary = out.get(f'{prefix}_plus_pass_1',
                          out.get(f'{prefix}_base_pass_1'))
        if isinstance(primary, (int, float)):
            out['accuracy'] = primary
    return out or None


# ── Build the evalplus samples jsonl from OpenCompass predictions ──

def _load_predictions(pred_path: str) -> dict:
    """Load OpenCompass predictions from either a single JSON file or a
    directory of shard files (``{abbr}_0.json`` ... ``{abbr}_{N-1}.json``)
    produced by NumWorkerPartitioner. When merging shards the keys are
    re-numbered sequentially so downstream sorted-iteration stays stable.
    """
    if osp.isfile(pred_path):
        with open(pred_path) as f:
            return json.load(f)

    if not osp.isdir(pred_path):
        raise FileNotFoundError(f'no such file or directory: {pred_path}')

    # Collect shard files: accept anything like <base>_<digits>.json
    shard_files = []
    for name in sorted(os.listdir(pred_path)):
        if not name.endswith('.json'):
            continue
        stem = name[:-len('.json')]
        if '_' not in stem:
            continue
        head, _, tail = stem.rpartition('_')
        if not tail.isdigit():
            continue
        shard_files.append((int(tail), osp.join(pred_path, name), head))

    if not shard_files:
        raise FileNotFoundError(
            f'no shard json files in {pred_path}; expected '
            f'{{abbr}}_{{i}}.json produced by NumWorkerPartitioner')

    shard_files.sort(key=lambda x: x[0])
    merged: dict = {}
    next_idx = 0
    for _, path, _head in shard_files:
        with open(path) as f:
            d = json.load(f)
        # OpenCompass shard predictions use numeric string keys "0","1",...
        # that restart at 0 in each shard. Re-key sequentially across shards
        # so ordering is preserved.
        for k in sorted(d.keys(), key=lambda x: int(x) if x.isdigit() else x):
            merged[str(next_idx)] = d[k]
            next_idx += 1
    return merged


def _get_all_problem_ids(dataset: str) -> set:
    """Return the full set of task_ids that the installed evalplus expects.

    Tries multiple import paths to handle different evalplus versions.
    Returns an empty set if evalplus is not importable (padding will be
    skipped and we rely on --i-just-wanna-run instead).
    """
    try:
        if dataset == 'humaneval':
            from evalplus.data import get_human_eval_plus
            problems = get_human_eval_plus()
        else:
            from evalplus.data import get_mbpp_plus
            problems = get_mbpp_plus()
        return set(problems.keys())
    except Exception:
        return set()


def _pad_missing_tasks(samples_path: str, dataset: str) -> int:
    """Append dummy (always-failing) entries for any task_ids that evalplus
    expects but are missing from the samples file. Returns the number of
    padded entries added."""
    all_ids = _get_all_problem_ids(dataset)
    if not all_ids:
        return 0

    # Read existing task_ids from the samples file
    existing_ids = set()
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                existing_ids.add(obj['task_id'])

    missing = all_ids - existing_ids
    if not missing:
        return 0

    print(f'[rescore] padding {len(missing)} missing task_ids with dummy '
          f'solutions (will count as failures)')
    with open(samples_path, 'a') as f:
        for task_id in sorted(missing):
            # Write a syntactically-invalid solution that always fails
            f.write(json.dumps({
                'task_id': task_id,
                'solution': '# missing from predictions\nraise NotImplementedError',
            }) + '\n')
    return len(missing)


def build_humaneval_samples(pred_path: str, out_jsonl: str) -> int:
    d = _load_predictions(pred_path)
    n = 0
    with open(out_jsonl, 'w') as f:
        for k in sorted(d.keys(), key=lambda x: int(x) if x.isdigit() else x):
            item = d[k]
            task_id = item['gold']
            prompt = item.get('origin_prompt', '')
            completion = process_humaneval_answer(item.get('prediction', ''))
            # Heuristic: OC `origin_prompt` is the formatted full HUMAN turn
            # ("Complete the following python code:\n<prompt>"). We want the
            # raw problem prompt for evalplus — it's everything after the
            # first occurrence of "Complete the following python code:\n".
            tag = 'Complete the following python code:\n'
            raw_prompt = prompt.split(tag, 1)[1] if tag in prompt else prompt
            f.write(json.dumps({
                'task_id': task_id,
                'solution': raw_prompt + completion,
            }) + '\n')
            n += 1
    # Pad missing problems so evalplus doesn't assert
    n_padded = _pad_missing_tasks(out_jsonl, 'humaneval')
    if n_padded:
        n += n_padded
    return n


def build_mbpp_samples(pred_path: str, out_jsonl: str) -> int:
    d = _load_predictions(pred_path)
    n = 0
    with open(out_jsonl, 'w') as f:
        for k in sorted(d.keys(), key=lambda x: int(x) if x.isdigit() else x):
            item = d[k]
            task_id = item['gold']
            code = process_mbpp_answer(item.get('prediction', ''))
            f.write(json.dumps({
                'task_id': task_id,
                'solution': code,
            }) + '\n')
            n += 1
    # Pad missing problems so evalplus doesn't assert
    n_padded = _pad_missing_tasks(out_jsonl, 'mbpp')
    if n_padded:
        n += n_padded
    return n


# ── Run evalplus ──

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try: st.write(s)
            except Exception: pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try: st.flush()
            except Exception: pass


def _parse_evalplus_stdout(text, prefix):
    pattern = re.compile(
        r'(humaneval|mbpp)(\+?)\s*\([^)]*\)\s*\n\s*pass@(\d+)\s*:\s*([0-9.]+)',
        re.IGNORECASE,
    )
    out = {}
    for m in pattern.finditer(text):
        is_plus = bool(m.group(2))
        k = m.group(3)
        try:
            v = float(m.group(4))
        except ValueError:
            continue
        mode = 'plus' if is_plus else 'base'
        out[f'{prefix}_{mode}_pass_{k}'] = v * 100
    if out:
        primary = out.get(f'{prefix}_plus_pass_1',
                          out.get(f'{prefix}_base_pass_1'))
        if isinstance(primary, (int, float)):
            out['accuracy'] = primary
    return out or None


def run_evalplus(dataset: str, samples_jsonl: str):
    """Returns (rc, captured_stdout).

    Invokes evalplus via `python -m evalplus.evaluate` so we don't depend
    on the unstable Python API signature (which has added/removed kwargs
    across versions — ``noextreme``, ``test_details``, ``gt_time_limit_factor``
    etc.). Streams output to the terminal while also capturing it for score
    parsing.
    """
    import subprocess

    # Verify evalplus is importable (so we fail fast with a clear error).
    try:
        import evalplus  # noqa: F401
    except ImportError as e:
        print(f'ERROR: cannot import evalplus: {e}', file=sys.stderr)
        return 2, ''

    try:
        n_parallel = int(os.environ.get('EVALPLUS_PARALLEL',
                                        min(16, os.cpu_count() or 4)))
    except Exception:
        n_parallel = 4

    cmd = [
        sys.executable, '-m', 'evalplus.evaluate',
        '--dataset', dataset,
        '--samples', samples_jsonl,
        '--parallel', str(n_parallel),
    ]
    # evalplus ships an expanded MBPP+/HumanEval+ problem set; OpenCompass's
    # built-in dataset is the `_sanitized` subset (378 / 164 problems). When
    # the two don't match we hit
    #     AssertionError: Missing problems in samples
    # because evalplus loads more problems than we have completions for.
    # `--noextreme` drops the newly-added "extreme" problems and usually
    # brings the count back down. Toggle via RESCORE_NOEXTREME=0 to disable.
    if os.environ.get('RESCORE_NOEXTREME', '1') != '0':
        cmd.append('--noextreme')
    # Some evalplus versions (>=0.2.0) support --i-just-wanna-run which skips
    # the strict "all problems must be present" assertion. We try it as a
    # safety net; if it's not supported the dummy-padding we did earlier will
    # handle the mismatch.
    try:
        import evalplus.evaluate as _ep_eval
        import inspect
        _src = inspect.getsource(_ep_eval)
        if 'i_just_wanna_run' in _src or 'i-just-wanna-run' in _src:
            cmd.append('--i-just-wanna-run')
    except Exception:
        pass
    # Escape hatch: let user pass extra evalplus args through unchanged.
    extra = os.environ.get('RESCORE_EVALPLUS_EXTRA', '').strip()
    if extra:
        cmd += extra.split()

    print(f'[rescore] running: {" ".join(cmd)}')

    # Stream and capture simultaneously.
    captured_chunks = []
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
    except FileNotFoundError as e:
        print(f'ERROR: cannot launch evalplus CLI: {e}', file=sys.stderr)
        return 2, ''

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_chunks.append(line)
    rc = proc.wait()
    captured = ''.join(captured_chunks)

    if rc != 0:
        print(f'ERROR: evalplus CLI exited with code {rc}', file=sys.stderr)

    return rc, captured


def _call_with_tee(fn, log_path):
    """DEPRECATED — kept for import compatibility; no longer used.

    Previously duplicated the child's stdout to both the terminal and a log
    file via OS-level fd manipulation. We now use `subprocess.Popen` with
    `stdout=PIPE`, tee manually in Python, which is simpler and avoids the
    fragile interaction with evalplus's own multiprocessing fd inheritance.
    """
    raise NotImplementedError(
        "_call_with_tee is no longer used; see run_evalplus.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pred_path', help='OpenCompass predictions JSON')
    ap.add_argument('--dataset', choices=['humaneval', 'mbpp'], required=True)
    ap.add_argument('--tag', default='', help='Optional label for output')
    ap.add_argument('--keep', action='store_true',
                    help='Keep the temp jsonl / results file for inspection')
    args = ap.parse_args()

    if not (osp.isfile(args.pred_path) or osp.isdir(args.pred_path)):
        print(f'ERROR: no such file or directory: {args.pred_path}',
              file=sys.stderr)
        return 2

    tmp_dir = tempfile.mkdtemp(prefix=f'rescore_{args.dataset}_')
    samples_path = osp.join(
        tmp_dir, f'{args.dataset}_samples.jsonl')

    if args.dataset == 'humaneval':
        n = build_humaneval_samples(args.pred_path, samples_path)
        prefix = 'humaneval_plus'
    else:
        n = build_mbpp_samples(args.pred_path, samples_path)
        prefix = 'mbpp_plus'

    print(f'Wrote {n} samples to {samples_path}')
    print(f'Running evalplus (dataset={args.dataset})...')

    rc, captured = run_evalplus(args.dataset, samples_path)
    if rc != 0:
        return rc

    # Prefer stdout — it's evalplus's canonical printed answer. The json
    # results file can have `pass_at_k = 0.0` even when stdout says
    # `pass@1: 0.226` (schema drift across evalplus versions).
    results = _parse_evalplus_stdout(captured, prefix=prefix)
    if not results:
        results = read_evalplus_results(samples_path, prefix=prefix)

    tag = f'[{args.tag}] ' if args.tag else ''
    print()
    print('=' * 60)
    print(f'{tag}Rescored {args.dataset} from {args.pred_path}')
    print('=' * 60)
    if results:
        for k, v in sorted(results.items()):
            print(f'  {k:40s} = {v:.2f}')
    else:
        print('  (no results parsed)')
    print()

    if not args.keep:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f'Kept files in {tmp_dir}')
    return 0 if results else 5


if __name__ == '__main__':
    sys.exit(main())
