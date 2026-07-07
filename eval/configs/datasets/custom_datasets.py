"""Custom OpenCompass dataset implementations for CMATH and CRUxEval-O.

This module must be **eagerly imported** before mmengine Config.fromfile()
processes the corresponding dataset config files — mmengine's lazy-import
machinery cannot execute decorators like @LOAD_DATASET.register_module().

Importing this module triggers:
  - Registration of CMATHDataset / CRUxEvalDataset in LOAD_DATASET
  - Registration of cmath_postprocess / cruxeval_o_postprocess in TEXT_POSTPROCESSORS

The Evaluator classes (CMATHEvaluator, CRUxEvalEvaluator,
HumanEvalPlusEvaluatorFixed) do NOT need registration — they are
referenced by `type=` and resolved by class reference directly.
"""

import os
import re
import ast
import json
import sys
import tempfile
import os.path as osp

from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from opencompass.datasets.base import BaseDataset


# ===========================================================================
# CMATH
# ===========================================================================
@LOAD_DATASET.register_module()
class CMATHDataset(BaseDataset):
    """Chinese elementary school math word problems.

    Source: https://huggingface.co/datasets/weitianwen/cmath
    Columns: question, golden (→ renamed to answer), grade, reasoning_step
    Splits: validation (600), test (1100)
    """

    @staticmethod
    def load(path, **kwargs):
        ds = hf_load_dataset(path)
        # Rename `golden` → `answer` so reader_cfg can use output_column='answer'
        for split in list(ds.keys()):
            cols = ds[split].column_names
            if 'golden' in cols and 'answer' not in cols:
                ds[split] = ds[split].rename_column('golden', 'answer')
        return ds


@TEXT_POSTPROCESSORS.register_module('cmath')
def cmath_postprocess(text: str) -> str:
    """Extract the numeric answer from a CoT generation."""
    # Stop at next question marker
    for stop in ['问题：', '问题:', 'Question:', '题目：', '题目:']:
        text = text.split(stop)[0]
    # Match "答案是X" / "答案为X" / "答案：X" / "\boxed{X}" / "The answer is X"
    patterns = [
        r'答案[是为：:\s]+[^0-9\-\.]*?(\-?\d+(?:\.\d+)?)',
        r'[Tt]he answer is[:\s]*(\-?\d+(?:\.\d+)?)',
        r'\\boxed\{(\-?\d+(?:\.\d+)?)\}',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    # Fallback: last number anywhere in text
    numbers = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    return numbers[-1] if numbers else 'NULL'


@TEXT_POSTPROCESSORS.register_module('cmath_dataset')
def cmath_dataset_postprocess(text: str) -> str:
    """Normalise the reference `golden` answer to a clean number string."""
    text = str(text).strip()
    nums = re.findall(r'\-?\d+\.\d+|\-?\d+', text)
    return nums[0] if nums else text


class CMATHEvaluator(BaseEvaluator):
    """Numeric equality check with 1e-6 tolerance."""

    def is_equal(self, pred, refer):
        try:
            if pred == refer:
                return True
            if abs(float(pred) - float(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
        correct = 0
        details = []
        for pred, ref in zip(predictions, references):
            ok = self.is_equal(pred, ref)
            if ok:
                correct += 1
            details.append({'pred': pred, 'answer': ref, 'correct': ok})
        return {'accuracy': 100 * correct / len(predictions), 'details': details}


# ===========================================================================
# CRUxEval-O
# ===========================================================================
@LOAD_DATASET.register_module()
class CRUxEvalDataset(BaseDataset):
    """CRUxEval output-prediction: given Python code + input, predict output.

    Source: https://huggingface.co/datasets/cruxeval-org/cruxeval
    Columns: id, code, input, output
    Splits: test (800)
    """

    @staticmethod
    def load(path, **kwargs):
        ds = hf_load_dataset(path)
        return ds


@TEXT_POSTPROCESSORS.register_module('cruxeval_o')
def cruxeval_o_postprocess(text: str) -> str:
    """Extract the predicted Python value from the model output."""
    # Stop at next example/question marker
    for stop in ['## Example', '---\n', 'Code:\n', 'Input:', 'Based on the given',
                 '[BEGIN]']:
        text = text.split(stop)[0]
    text = text.strip()

    # Try explicit "output ..." / "result ..." / assert patterns
    patterns = [
        r'[Tt]he output is\s*(.+?)(?:\n|$)',
        r'[Oo]utput\s*(?:is|=|:)\s*(.+?)(?:\n|$)',
        r'[Rr]esult\s*(?:is|=|:)\s*(.+?)(?:\n|$)',
        r'[Rr]eturn(?:s|ed)?\s*(?:is|=|:)?\s*(.+?)(?:\n|$)',
        r'assert\s+.+?==\s*(.+?)(?:\n|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1).strip().rstrip('.')
            if val:
                return val

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r'^(?:#{1,3}|>{1,3}|output\s*:\s*)', '', last,
                      flags=re.IGNORECASE).strip()
        return last
    return ''


def _normalise_value(s: str) -> str:
    s = str(s).strip()
    # Strip surrounding matching quotes if present
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    try:
        return repr(ast.literal_eval(s))
    except Exception:
        return s


class CRUxEvalEvaluator(BaseEvaluator):
    """Python literal equality after normalisation."""

    def is_equal(self, pred, refer):
        if str(pred).strip() == str(refer).strip():
            return True
        try:
            if _normalise_value(pred) == _normalise_value(refer):
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'predictions and references have different length'}
        correct = 0
        details = []
        for pred, ref in zip(predictions, references):
            ok = self.is_equal(pred, ref)
            if ok:
                correct += 1
            details.append({'pred': pred, 'answer': ref, 'correct': ok})
        return {'accuracy': 100 * correct / len(predictions), 'details': details}


# ===========================================================================
# HumanEval+ / MBPP+ stdout capture helpers
# ===========================================================================
# evalplus.evaluate.evaluate() prints the canonical pass@1 results to stdout
# in a stable format like:
#
#     humaneval (base tests)
#     pass@1:    0.012
#     humaneval+ (base + extra tests)
#     pass@1:    0.012
#
# but writes a json file whose schema has drifted across versions (sometimes
# no `pass_at_k` key). We tee stdout to a buffer and parse those printed
# lines as a version-proof fallback.

class _Tee:
    """File-like object that forwards writes to multiple streams."""
    def __init__(self, *streams):
        self._streams = streams
    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)
    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def _run_with_stdout_tee(fn, log_path):
    """Run `fn()` with stdout fd-level duplicated to both original stdout
    and `log_path`. This captures even C-level prints and subprocess output,
    unlike `contextlib.redirect_stdout` which only rebinds `sys.stdout`.

    Returns True on success, False if `fn` raised (re-raises the exception).
    """
    import subprocess
    sys.stdout.flush()
    saved_fd = os.dup(1)
    tee_proc = None
    log_fd = None
    try:
        try:
            tee_proc = subprocess.Popen(
                ['tee', log_path], stdin=subprocess.PIPE, stdout=saved_fd)
            os.dup2(tee_proc.stdin.fileno(), 1)
        except FileNotFoundError:
            # No `tee` — write only to log (terminal won't see live output)
            log_fd = os.open(log_path,
                             os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.dup2(log_fd, 1)
        try:
            fn()
        finally:
            sys.stdout.flush()
            os.dup2(saved_fd, 1)
            if tee_proc is not None:
                try:
                    tee_proc.stdin.close()
                    tee_proc.wait(timeout=5)
                except Exception:
                    pass
            if log_fd is not None:
                os.close(log_fd)
    finally:
        os.close(saved_fd)


def _parse_evalplus_stdout(text: str, prefix: str):
    """Extract pass@1 values from the text printed by evalplus.evaluate.

    Expected patterns:
        humaneval (base tests)     OR   mbpp (base tests)
        pass@1:    0.012
        humaneval+ (base + extra tests)  OR  mbpp+ (base + extra tests)
        pass@1:    0.012
    Returns dict like {'humaneval_plus_base_pass_1': 1.2, ...} scaled ×100,
    plus an 'accuracy' convenience key, or None if nothing matched.
    """
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


# ===========================================================================
# HumanEval+ — robust replacement for opencompass.datasets.HumanEvalPlusEvaluator
# ===========================================================================
# The upstream OpenCompass HumanEvalPlusEvaluator passes a `flags` dict
# positionally to `evalplus.evaluate.evaluate(flags)`. In current evalplus
# versions this silently leaves `samples=None` and the internal
# `assert samples is not None, "No samples provided"` fires. This evaluator
# calls evalplus with explicit keyword arguments and falls back to a direct
# per-sample correctness check if the high-level API is incompatible.

@TEXT_POSTPROCESSORS.register_module('humaneval_plus_robust')
def humaneval_plus_robust_postprocess(text: str) -> str:
    """Strip prose chatter and fenced-code markers from a base-model completion."""
    m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1)
    for stop in ['\nQuestion:', '\nProblem:', '\ndef f(', '\n# Test',
                 '\nassert ', '\n>>>', '\nif __name__', '\nprint(']:
        text = text.split(stop)[0]
    text = re.split(r'\n\n\n+', text)[0]
    return text


class HumanEvalPlusEvaluatorFixed(BaseEvaluator):
    """Drop-in replacement for HumanEvalPlusEvaluator using keyword-style
    invocation of evalplus.evaluate.evaluate, with a sandbox-safe fallback."""

    def __init__(self, k=None, metric='humaneval_plus'):
        self.k = k or [1]
        self.metric = metric
        super().__init__()

    def _run_evalplus_high_level(self, samples_jsonl):
        """Call evalplus.evaluate.evaluate ONCE and parse pass@1 from both:
          (a) the `<samples_base>_eval_results.json` file, and
          (b) the stdout it printed (fallback — format is stable across versions).
        """
        try:
            from evalplus.evaluate import evaluate as _ep_evaluate
        except Exception:
            return None
        try:
            n_parallel = int(os.environ.get('EVALPLUS_PARALLEL',
                                            min(16, os.cpu_count() or 4)))
        except Exception:
            n_parallel = 4

        kwargs_variants = [
            dict(dataset='humaneval', samples=samples_jsonl,
                 base_only=False, parallel=n_parallel,
                 i_just_wanna_run=False,
                 test_details=False, min_time_limit=0.2,
                 gt_time_limit_factor=4.0, mini=False),
            dict(dataset='humaneval', samples=samples_jsonl,
                 parallel=n_parallel),
            dict(dataset='humaneval', samples=samples_jsonl),
        ]

        # Capture stdout at the OS fd level so subprocess / C-level prints
        # from evalplus workers are captured too.
        log_path = tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', prefix='evalplus_he_',
            delete=False).name
        call_ok = False
        for kw in kwargs_variants:
            try:
                _run_with_stdout_tee(lambda: _ep_evaluate(**kw), log_path)
                call_ok = True
                break
            except TypeError:
                continue
            except Exception as e:
                print(f'[HumanEvalPlusEvaluatorFixed] high-level evaluate '
                      f'failed with {type(e).__name__}: {e}')
                break
        try:
            with open(log_path) as f:
                captured = f.read()
        except Exception:
            captured = ''
        try:
            os.unlink(log_path)
        except Exception:
            pass

        if not call_ok:
            # Still try stdout parse in case a partial run printed something
            stdout_result = _parse_evalplus_stdout(captured,
                                                   prefix='humaneval_plus')
            return stdout_result

        # Prefer stdout (it's evalplus's canonical printed answer). The json
        # pass_at_k values have been observed to be 0.0 while stdout shows
        # the correct pass@1 — don't trust the json first.
        out = _parse_evalplus_stdout(captured, prefix='humaneval_plus')
        if out:
            return out
        return self._read_evalplus_results(samples_jsonl,
                                           prefix='humaneval_plus')

    @staticmethod
    def _read_evalplus_results(samples_jsonl, prefix='humaneval_plus'):
        """Parse pass@1 from the `<samples_base>_eval_results.json` file
        that evalplus wrote next to the input jsonl."""
        results_path = samples_jsonl.replace('.jsonl', '_eval_results.json')
        if not osp.isfile(results_path):
            # Fallback: try alternative naming
            base = samples_jsonl[:-6] if samples_jsonl.endswith('.jsonl') \
                else samples_jsonl
            for cand in [base + '_eval_results.json',
                         base + '.eval_results.json',
                         samples_jsonl + '.eval_results.json']:
                if osp.isfile(cand):
                    results_path = cand
                    break
            else:
                print(f'[evalplus parse] no _eval_results.json found near '
                      f'{samples_jsonl}')
                return None

        try:
            with open(results_path) as f:
                data = json.load(f)
        except Exception as e:
            print(f'[evalplus parse] failed to read {results_path}: {e}')
            return None

        # evalplus result schema:
        #   { "date": ..., "hash": ..., "eval": { task_id: {"base":[...],
        #     "plus":[...]}, ... }, "pass_at_k": {"base":{...},"plus":{...}} }
        out = {}
        pak = data.get('pass_at_k', {})
        if isinstance(pak, dict):
            base_pak = pak.get('base', {}) or {}
            plus_pak = pak.get('plus', {}) or {}
            for k, v in base_pak.items():
                if isinstance(v, (int, float)):
                    # Avoid `@` in key names — OpenCompass summary may drop them
                    out[f'{prefix}_base_pass_{k}'] = v * 100
            for k, v in plus_pak.items():
                if isinstance(v, (int, float)):
                    out[f'{prefix}_plus_pass_{k}'] = v * 100

        # Fallback: compute pass@1 manually from per-task `eval` entries
        if not out and isinstance(data.get('eval'), dict):
            ev = data['eval']
            total = len(ev)
            if total > 0:
                base_ok, plus_ok = 0, 0
                for tid, info in ev.items():
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

        # Surface a primary "accuracy" field for the OpenCompass summary
        # table (uses plus pass@1 since that's the harder +mode metric).
        if out:
            primary = out.get(f'{prefix}_plus_pass_1',
                              out.get(f'{prefix}_base_pass_1'))
            if isinstance(primary, (int, float)):
                out['accuracy'] = primary
        return out if out else None

    def _run_evalplus_low_level(self, humaneval_preds):
        try:
            from evalplus.data import get_human_eval_plus
            from evalplus.eval import check_correctness
        except Exception:
            return None
        try:
            problems = get_human_eval_plus()
        except Exception as e:
            print(f'[HumanEvalPlusEvaluatorFixed] get_human_eval_plus failed: {e}')
            return None

        base_pass, plus_pass, total = 0, 0, 0
        details = {}
        for idx, entry in enumerate(humaneval_preds):
            tid = entry['task_id']
            solution = entry['solution']
            if tid not in problems:
                details[str(idx)] = {'task_id': tid, 'error': 'unknown_task'}
                continue
            problem = problems[tid]
            total += 1
            try:
                base_res = check_correctness(
                    dataset='humaneval', completion_id=idx, problem=problem,
                    solution=solution, expected_output=problem.get('base_input'),
                    base_only=True, fast_check=True, identifier=str(idx),
                    min_time_limit=0.2, gt_time_limit_factor=4.0)
                plus_res = check_correctness(
                    dataset='humaneval', completion_id=idx, problem=problem,
                    solution=solution, expected_output=problem.get('plus_input'),
                    base_only=False, fast_check=True, identifier=str(idx),
                    min_time_limit=0.2, gt_time_limit_factor=4.0)
            except Exception as e:
                details[str(idx)] = {'task_id': tid, 'error': f'{type(e).__name__}: {e}'}
                continue
            base_list = base_res.get('base') if isinstance(base_res, dict) else None
            plus_list = plus_res.get('plus') if isinstance(plus_res, dict) else None
            base_ok = bool(base_list) and base_list[0][0] == 'success'
            plus_ok = bool(plus_list) and plus_list[0][0] == 'success'
            base_pass += int(base_ok)
            plus_pass += int(plus_ok)
            details[str(idx)] = {'task_id': tid,
                                 'base_pass': base_ok,
                                 'plus_pass': plus_ok}
        if total == 0:
            return None
        return {
            'humaneval_plus_pass@1': 100 * base_pass / total,
            'humaneval_plus_plus_pass@1': 100 * plus_pass / total,
            'details': details,
        }

    def score(self, predictions, references, test_set):
        if len(predictions) != len(references):
            return {'error': 'preds and refs have different length'}
        if len(predictions) == 0:
            return {'error': 'empty predictions'}

        prompts = [item['prompt'] for item in test_set]
        humaneval_preds = []
        for preds, refer, prompt in zip(predictions, references, prompts):
            if not isinstance(preds, list):
                preds = [preds]
            for pred in preds:
                humaneval_preds.append({'task_id': refer,
                                        'solution': prompt + pred})

        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = osp.join(tmp_dir, 'humaneval_samples.jsonl')
            with open(samples_path, 'w', encoding='utf-8') as f:
                for p in humaneval_preds:
                    f.write(json.dumps(p) + '\n')

            if os.path.getsize(samples_path) == 0:
                return {'error': 'sample file was written empty'}

            out = self._run_evalplus_high_level(samples_path)
            if out is not None and isinstance(out, dict):
                # Results from _read_evalplus_results are already scaled ×100.
                if out:
                    return out

        fallback = self._run_evalplus_low_level(humaneval_preds)
        if fallback is not None:
            return fallback

        return {'error': 'evalplus evaluation failed; '
                        'check evalplus version compatibility'}


# ===========================================================================
# MBPP+ (evalplus native schema) — compatible loader + evaluator
# ===========================================================================
# Your local /apdcephfs_fsgm/.../data/mbpp_plus/mbpp_plus.jsonl uses the
# evalplus-native schema (task_id, prompt, entry_point, canonical_solution,
# base_input, atol, plus_input, contract, assertion) — NOT legacy MBPP
# sanitized format (which has text + test_list). The upstream MBPPPlusDataset
# in OpenCompass fails with `KeyError: 'test_list'` on this file.
#
# We provide MBPPPlusEvalPlusDataset which reads the evalplus schema and
# emits the columns the OpenCompass reader expects: `text`, `test_list`
# (derived from the evalplus `assertion` field, split on newlines), and
# `task_id`.

@LOAD_DATASET.register_module()
class MBPPPlusEvalPlusDataset(BaseDataset):
    """Load MBPP+ from an evalplus-native jsonl file.

    The evalplus jsonl schema we convert:
        task_id (e.g. 'Mbpp/2')
        prompt (docstring with 1 example assert)
        entry_point (function name)
        canonical_solution (reference impl; unused here)
        base_input / plus_input (test inputs for execution; unused here)
        assertion (3 assert statements, like legacy MBPP test_list)

    We emit:
        text        ← natural-language description extracted from `prompt`
        test_list   ← list[str] of assert statements from `assertion`
        task_id     ← unchanged
    """

    @staticmethod
    def load(path, num_repeats: int = 1, **kwargs):
        path = os.path.abspath(path)

        def _process(ex):
            raw = ex.get('prompt', '')
            # Strip triple-quote docstring wrapping, drop the assert line
            text = raw.strip()
            if text.startswith('"""') and text.endswith('"""'):
                text = text[3:-3]
            # Drop `assert ...` line inside the docstring — that's the example
            lines = [l for l in text.splitlines()
                     if not l.strip().startswith('assert ')]
            text = '\n'.join(l for l in lines if l.strip()).strip()

            assertion = ex.get('assertion', '') or ''
            test_list = [l.strip() for l in assertion.splitlines()
                         if l.strip().startswith('assert ')]
            if not test_list:
                test_list = [assertion.strip()]

            return {
                'task_id': ex.get('task_id', ''),
                'text': text,
                'test_list': test_list,
                'test_list_2': '\n'.join(test_list),
                'test_case': test_list,
                'test_column': {'test_list_2': '\n'.join(test_list),
                                'task_id': ex.get('task_id', '')},
            }

        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(_process(json.loads(line)))
        dataset = Dataset.from_list(rows * num_repeats)
        return dataset


# ===========================================================================
# MBPP+ — robust replacement for opencompass.datasets.MBPPEvaluator(metric='MBPPPlus')
# ===========================================================================
# Same `evaluate(flags)` vs `evaluate(**kwargs)` API mismatch as HumanEval+.

class MBPPPlusEvaluatorFixed(BaseEvaluator):
    """Drop-in replacement for MBPPEvaluator(metric='MBPPPlus') that calls
    evalplus.evaluate.evaluate with explicit keyword arguments."""

    _BEGIN_DONE_PATTERNS = [
        r"\[BEGIN\]\s*'(.*?)'\s*\[DONE\]",
        r"\[BEGIN\]\s*'(.*?)'\s*DONE",
        r"\[BEGIN\]\s*(.*?)\s*\[DONE\]",
        r"\[BEGIN\](.*?)\[DONE\]",
        r"\[BEGIN\](.*)",
    ]

    def __init__(self, metric: str = 'MBPPPlus'):
        self.metric = metric
        super().__init__()

    def _process_answer(self, text: str) -> str:
        """Extract the Python function body from a MBPP+ prediction.

        Model output format (following the 3-shot prompt) typically looks
        like:

            <leading space>'def foo(...):
                <body>
            ' \n[DONE] \n\n <next task...>

        Because the prompt ends with `[BEGIN]\n`, the opening `[BEGIN]` is
        NOT in the completion. We strip the leading `'`, cut at the first
        `[DONE]`, then strip the trailing `'`.
        """
        # 1) Prefer [BEGIN]...[DONE] if the full pair appears (rare).
        for pat in self._BEGIN_DONE_PATTERNS:
            m = re.search(pat, text, re.DOTALL)
            if m:
                return self._strip_quotes(m.group(1)).strip()

        # 2) Cut at first [DONE] (model continues to next task after this).
        if '[DONE]' in text:
            text = text.split('[DONE]')[0]
        elif 'DONE' in text:
            # less safe, but some models drop the brackets
            idx = text.find('\nDONE')
            if idx >= 0:
                text = text[:idx]

        # 3) Strip surrounding ```python``` fence if present.
        m = re.search(r'```(?:python)?\s*\n(.*?)```', text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # 4) Strip the MBPP quote wrapping produced by the few-shot format.
        return self._strip_quotes(text).strip()

    @staticmethod
    def _strip_quotes(text: str) -> str:
        """Remove leading/trailing `'` (and whitespace) used in MBPP
        few-shot wrapping like ` 'def foo():\\n  ...' `."""
        t = text.strip()
        # Leading `'` possibly after whitespace
        if t.startswith("'"):
            t = t[1:]
        # Trailing `'` possibly before whitespace/newline
        t_rstrip = t.rstrip()
        if t_rstrip.endswith("'"):
            t = t_rstrip[:-1]
        return t

    def _run_evalplus_high_level(self, samples_jsonl):
        """Same approach as HumanEval+: call evaluate once, capture stdout,
        then try json results + stdout as fallback."""
        try:
            from evalplus.evaluate import evaluate as _ep_evaluate
        except Exception:
            return None
        try:
            n_parallel = int(os.environ.get('EVALPLUS_PARALLEL',
                                            min(16, os.cpu_count() or 4)))
        except Exception:
            n_parallel = 4
        kwargs_variants = [
            dict(dataset='mbpp', samples=samples_jsonl,
                 base_only=False, parallel=n_parallel,
                 i_just_wanna_run=False,
                 test_details=False, min_time_limit=0.2,
                 gt_time_limit_factor=4.0, mini=False),
            dict(dataset='mbpp', samples=samples_jsonl,
                 parallel=n_parallel),
            dict(dataset='mbpp', samples=samples_jsonl),
        ]

        log_path = tempfile.NamedTemporaryFile(
            mode='w', suffix='.log', prefix='evalplus_mbpp_',
            delete=False).name
        call_ok = False
        for kw in kwargs_variants:
            try:
                _run_with_stdout_tee(lambda: _ep_evaluate(**kw), log_path)
                call_ok = True
                break
            except TypeError:
                continue
            except Exception as e:
                print(f'[MBPPPlusEvaluatorFixed] high-level evaluate '
                      f'failed: {type(e).__name__}: {e}')
                break
        try:
            with open(log_path) as f:
                captured = f.read()
        except Exception:
            captured = ''
        try:
            os.unlink(log_path)
        except Exception:
            pass

        if not call_ok:
            return _parse_evalplus_stdout(captured, prefix='mbpp_plus')

        # Prefer stdout (canonical); fall back to json.
        out = _parse_evalplus_stdout(captured, prefix='mbpp_plus')
        if out:
            return out
        return HumanEvalPlusEvaluatorFixed._read_evalplus_results(
            samples_jsonl, prefix='mbpp_plus')

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {'error': 'preds and refs have different length'}
        if len(predictions) == 0:
            return {'error': 'empty predictions'}

        mbpp_preds = []
        for preds, refer in zip(predictions, references):
            if not isinstance(preds, list):
                preds = [preds]
            for pred in preds:
                mbpp_preds.append({'task_id': refer,
                                   'solution': self._process_answer(pred)})

        with tempfile.TemporaryDirectory() as tmp_dir:
            samples_path = osp.join(tmp_dir, 'mbpp_samples.jsonl')
            with open(samples_path, 'w', encoding='utf-8') as f:
                for p in mbpp_preds:
                    f.write(json.dumps(p) + '\n')
            if os.path.getsize(samples_path) == 0:
                return {'error': 'sample file was written empty'}

            out = self._run_evalplus_high_level(samples_path)
            if out is not None and isinstance(out, dict) and out:
                # Results from _read_evalplus_results are already scaled ×100.
                return out

        return {'error': 'evalplus evaluation failed for MBPP+'}


# ===========================================================================
# MATH post-processor: undo double-brace contamination + cut off continuation
# ===========================================================================
@TEXT_POSTPROCESSORS.register_module('math_unbrace')
def math_unbrace_postprocess(text: str) -> str:
    """Clean up a MATH prediction.

    1) Cut off at the next "Problem:" marker (model often continues making up
       new problems after finishing the current one).
    2) Keep only up to the first "Final Answer:" line's newline, since the
       intended answer is in `\\boxed{...}` before that sentence.
    3) Undo `{{` → `{` and `}}` → `}` because the few-shot examples in the
       upstream MATH config contain stray `{{...}}` (historical `str.format`
       escape that safe_format does NOT collapse). Base models learn to
       emit `\\boxed{{45}}` which math_verify.parse cannot match.
    """
    # 1) Stop at next problem boundary
    text = text.split('\nProblem:')[0]
    # 2) Keep through Final Answer sentence (but not beyond)
    m = re.search(r'(.*?Final Answer[^\n]*\n)', text, re.DOTALL)
    if m:
        text = m.group(1)
    # 3) Collapse spurious double braces that leaked in from the prompt
    text = text.replace('{{', '{').replace('}}', '}')
    return text
