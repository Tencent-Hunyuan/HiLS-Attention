"""OpenCompass evaluation using SGLang offline Engine (no server needed).

Usage
-----
    python eval_opencompass_sglang.py \
        --hf-path /path/to/hils_checkpoint \
        --hils-config /path/to/config.json \
        --sglang-tp 1 \
        --sglang-page-size 64 \
        --sglang-max-total-tokens 4096 \
        --sglang-batch-size 32 \
        --datasets gpqa_few_shot_ppl_4b5a83 \
        -w /path/to/workdir \
        [--debug]
"""

import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Ensure opencompass is importable (safe for subprocesses)
# ---------------------------------------------------------------------------

def _ensure_opencompass():
    try:
        import opencompass  # noqa: F401
        return
    except ModuleNotFoundError:
        opencompass_path = os.environ.get("OPENCOMPASS_PATH")
        if opencompass_path:
            opencompass_path = os.path.abspath(opencompass_path)
            if os.path.isdir(os.path.join(opencompass_path, "opencompass")):
                if opencompass_path not in sys.path:
                    sys.path.insert(0, opencompass_path)
                return
        raise ModuleNotFoundError(
            "Cannot import `opencompass`. Please install it or set "
            "`OPENCOMPASS_PATH=/path/to/opencompass_repo`."
        )


_ensure_opencompass()

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS


# ---------------------------------------------------------------------------
# SGLang offline model wrapper for OpenCompass
#
# This class is at module top-level so it can be imported by mmengine Config
# in any subprocess without triggering CLI parsing.
# ---------------------------------------------------------------------------

@MODELS.register_module()
class SGLangModel(BaseModel):
    """OpenCompass model wrapper using SGLang offline Engine."""

    is_api: bool = False

    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = {},
        meta_template: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None,
        batch_size: int = 32,
        tp_size: int = 1,
        page_size: int = 64,
        max_total_tokens: int = 4096,
        mem_fraction_static: float = 0.85,
        enable_prefix_cache: bool = True,
        attention_backend: Optional[str] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=False,
            meta_template=meta_template,
            generation_kwargs=generation_kwargs or {},
        )
        self.batch_size = batch_size
        self._engine = None
        self._engine_kwargs = dict(
            model_path=path,
            tp_size=tp_size,
            max_total_tokens=max_total_tokens,
            mem_fraction_static=mem_fraction_static,
            log_level="info",
        )
        if attention_backend:
            self._engine_kwargs["attention_backend"] = attention_backend
            self._engine_kwargs["page_size"] = page_size
            # HiLS-specific workarounds
            self._engine_kwargs["disable_cuda_graph"] = True
            self._engine_kwargs["disable_overlap_schedule"] = True

        tok_path = tokenizer_path or path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=True, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prefix cache + logprob cache for PPL evaluation
        # Key insight: in causal attention, logprob of prefix tokens is
        # independent of suffix tokens.  So we cache prefix logprobs from
        # the first request and reuse them for subsequent requests that
        # share the same prefix, while SGLang only needs to compute the
        # suffix logprobs (and can reuse KV cache for the prefix).
        self._prev_token_ids = None          # origin token ids of previous request
        self._cached_prefix_logprobs = None  # List[float] logprobs for the common prefix
        self._enable_prefix_cache = enable_prefix_cache

    def _get_engine(self):
        if self._engine is None:
            import sglang as sgl
            print(f"[SGLangModel] Creating Engine: {self._engine_kwargs}")
            self._engine = sgl.Engine(**self._engine_kwargs)
            page_size = self._engine_kwargs.get("page_size", 64)
            warmup_prompt = "Hello " * max(page_size * 2, 1024)
            print("[SGLangModel] Warmup: triggering JIT compilation...")
            self._engine.generate(
                prompt=warmup_prompt,
                sampling_params={"max_new_tokens": 3, "temperature": 0.0},
            )
            print("[SGLangModel] Engine ready.")
        return self._engine

    def __del__(self):
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                pass
            self._engine = None

    # ── PPL evaluation ──

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        all_ce: List[float] = []
        for start in range(0, len(inputs), self.batch_size):
            batch_texts = inputs[start: start + self.batch_size]
            batch_mask = (
                mask_length[start: start + self.batch_size]
                if mask_length is not None else None
            )
            all_ce.extend(self._get_ppl_batch(batch_texts, batch_mask))
        return np.array(all_ce)

    def _get_ppl_batch(self, texts, mask_length=None):
        engine = self._get_engine()

        # ── Determine logprob_start_len per request ──
        #
        # For PPL we need the FULL sequence logprobs (positions 1..N-1).
        # When prefix cache is enabled, SGLang only returns logprobs from
        # logprob_start_len onwards.  We cache the prefix logprobs from the
        # first request and splice them back in for subsequent requests.
        #
        # IMPORTANT: prefix cache only works across sequential requests —
        # the previous request must complete and deposit its KV into the
        # radix tree before the next request can match.  So in prefix-cache
        # mode we send requests ONE BY ONE (not as a batch).
        #
        # Three modes:
        #   1. mask_length provided  → logprob_start = mask_length (only suffix matters)
        #   2. prefix cache enabled  → sequential one-by-one with logprob splice
        #   3. no prefix cache       → logprob_start = 0, batch OK

        if self._enable_prefix_cache and mask_length is None:
            # ── Prefix cache mode: send one-by-one for KV reuse ──
            ce_losses = []
            for i, text in enumerate(texts):
                tids = self.tokenizer.encode(text)

                # Find common prefix with previously cached request
                if self._prev_token_ids is not None and self._cached_prefix_logprobs is not None:
                    cp = 0
                    for a, b in zip(self._prev_token_ids, tids):
                        if a == b:
                            cp += 1
                        else:
                            break
                    cp = min(cp, len(self._cached_prefix_logprobs))
                else:
                    cp = 0
                self._prev_token_ids = tids

                # The logprob at position cp-1 is log P(token_cp | prefix).
                # Since token_cp is the FIRST differing token between the cached
                # request and the current one, we cannot reuse the cached value.
                # So we request SGLang to recompute from position cp-1 onwards,
                # and only splice cached logprobs up to cp-2.
                safe_cp = max(cp - 1, 0)

                # Send single request — previous request's KV is now in radix tree
                res = engine.generate(
                    prompt=text,
                    sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                    return_logprob=True,
                    logprob_start_len=safe_cp,
                    top_logprobs_num=0,
                )

                # Extract valid (non-None) logprobs
                suffix_logprobs = []
                for item in res["meta_info"]["input_token_logprobs"]:
                    lp = item[0] if isinstance(item, (list, tuple)) else item
                    if lp is not None:
                        suffix_logprobs.append(float(lp))

                # Splice: cached prefix logprobs + new suffix logprobs
                # cached[:safe_cp] = [lp_1, ..., lp_{cp-1}] (all within shared prefix, safe to reuse)
                # suffix = [lp_cp, lp_{cp+1}, ..., lp_{N-2}] (freshly computed for this request)
                if safe_cp > 0:
                    prefix_lps = self._cached_prefix_logprobs[:safe_cp]
                    full_logprobs = prefix_lps + suffix_logprobs
                else:
                    full_logprobs = suffix_logprobs

                # Update cache for next request
                self._cached_prefix_logprobs = full_logprobs

                ce_losses.append(-float(np.mean(full_logprobs)) if full_logprobs else 0.0)
            return ce_losses

        # ── Non-prefix-cache mode: batch is fine ──
        if mask_length is not None:
            logprob_starts = list(mask_length)
        else:
            logprob_starts = [0] * len(texts)

        # Call SGLang engine (batch)
        if len(texts) == 1:
            results = [engine.generate(
                prompt=texts[0],
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                logprob_start_len=logprob_starts[0],
                top_logprobs_num=0,
            )]
        else:
            results = engine.generate(
                prompt=texts,
                sampling_params={"max_new_tokens": 1, "temperature": 0.0},
                return_logprob=True,
                logprob_start_len=logprob_starts,
                top_logprobs_num=0,
            )

        ce_losses = []
        for i, res in enumerate(results):
            input_logprobs_raw = res["meta_info"]["input_token_logprobs"]
            logprobs = []
            for item in input_logprobs_raw:
                lp = item[0] if isinstance(item, (list, tuple)) else item
                if lp is not None:
                    logprobs.append(float(lp))

            if mask_length is not None and mask_length[i] is not None:
                ce_losses.append(-float(np.mean(logprobs)) if logprobs else 0.0)
            else:
                ce_losses.append(-float(np.mean(logprobs)) if logprobs else 0.0)
        return ce_losses

    def get_ppl_tokenwise(self, inputs, mask_length=None):
        return self.get_ppl(inputs, mask_length)

    # ── Generation ──

    def generate(
        self,
        inputs: List[str],
        max_out_len: int = 512,
        stopping_criteria: Optional[List[str]] = None,
        min_out_len: Optional[int] = None,
        **kwargs,
    ) -> List[str]:
        """Generate completions.

        `stopping_criteria`: list of string stop tokens. SGLang will stop
        decoding as soon as the output contains any of them. This is the
        OpenCompass GenInferencer hook — it detects the presence of the
        `stopping_criteria` kwarg via `inspect.signature` and passes its
        list through automatically.

        `min_out_len`: minimum number of tokens to generate before EOS can
        fire. GenInferencer also forwards this via signature sniffing.
        Crucial for base models on few-shot prompts: without it, a base
        model that hasn't seen this exact turn-boundary format will often
        emit EOS on the very first token, producing an empty prediction.

        Base (non-chat) models do not know when to stop on their own for
        few-shot prompts. Without explicit stops, they'll keep hallucinating
        additional Q/A pairs until `max_out_len`, wasting compute and
        polluting predictions. Pass e.g. `["\\n\\nQuestion:", "\\nProblem:"]`
        matching the few-shot delimiter.
        """
        engine = self._get_engine()
        sp = {
            "max_new_tokens": max_out_len,
            "temperature": kwargs.get("temperature", 0.0),
        }
        if stopping_criteria:
            # SGLang expects `stop` as List[str]; it will be stripped from
            # the emitted text on match.
            sp["stop"] = list(stopping_criteria)
        if min_out_len is not None and min_out_len > 0:
            sp["min_new_tokens"] = int(min_out_len)
        results = []
        for start in range(0, len(inputs), self.batch_size):
            batch = inputs[start: start + self.batch_size]
            if len(batch) == 1:
                out = engine.generate(prompt=batch[0], sampling_params=sp)
                results.append(out["text"])
            else:
                outs = engine.generate(prompt=batch, sampling_params=sp)
                results.extend(out["text"] for out in outs)
        return results

    # ── Tokenizer utilities ──

    def encode(self, prompt: str):
        return self.tokenizer.encode(prompt, return_tensors="pt")

    def decode(self, tokens):
        import torch
        if isinstance(tokens, torch.Tensor) and tokens.dim() > 1:
            tokens = tokens[0]
        return self.tokenizer.decode(tokens)

    def get_token_len(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))


# ---------------------------------------------------------------------------
# Everything below is ONLY executed when run as main script, NOT on import.
# This is critical because SGLang Engine forks subprocesses that may
# re-import this module.
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ── CLI argument helpers ──

    def _pop_cli_arg(name: str, default=None):
        if name not in sys.argv:
            return default
        i = sys.argv.index(name)
        if i + 1 >= len(sys.argv):
            raise ValueError(f"{name} requires a value")
        value = sys.argv[i + 1]
        del sys.argv[i: i + 2]
        return value

    def _pop_cli_flag(name: str) -> bool:
        if name in sys.argv:
            sys.argv.remove(name)
            return True
        return False

    def _peek_cli_arg(name: str, default=None):
        if name not in sys.argv:
            return default
        i = sys.argv.index(name)
        if i + 1 >= len(sys.argv):
            return default
        return sys.argv[i + 1]

    # ── Parse args ──

    hf_path              = _pop_cli_arg("--hf-path")
    hils_config           = _pop_cli_arg("--hils-config")
    sglang_tp            = int(_pop_cli_arg("--sglang-tp", "1"))
    sglang_page_size     = int(_pop_cli_arg("--sglang-page-size", "64"))
    sglang_max_tokens    = int(_pop_cli_arg("--sglang-max-total-tokens", "4096"))
    sglang_batch_size    = int(_pop_cli_arg("--sglang-batch-size", "32"))
    sglang_mem_fraction  = float(_pop_cli_arg("--sglang-mem-fraction-static", "0.85"))
    sglang_attn_backend  = _pop_cli_arg("--sglang-attention-backend")  # None for default
    enable_prefix_cache  = not _pop_cli_flag("--no-prefix-cache")
    datasets_str         = _pop_cli_arg("--datasets")
    work_dir_base        = _pop_cli_arg("-w", os.path.join("outputs", "sglang_eval"))
    debug                = _pop_cli_flag("--debug")

    _pop_cli_arg("--hf-type")  # discard if present

    if not hf_path:
        print("ERROR: --hf-path is required", file=sys.stderr)
        sys.exit(1)
    if not datasets_str:
        print("ERROR: --datasets is required", file=sys.stderr)
        sys.exit(1)

    # ── Load datasets via mmengine Config.fromfile (preserves lazy imports) ──

    def _load_datasets(ds_str: str) -> list:
        from mmengine.config import Config as MmConfig
        import opencompass
        oc_root = os.path.dirname(opencompass.__file__)
        datasets_root = os.path.join(oc_root, "configs", "datasets")

        # Also search local eval/configs/datasets/ directory for custom configs
        local_datasets_root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "configs", "datasets")

        # Make custom_datasets.py importable as a top-level module, then
        # eagerly import it so that @LOAD_DATASET / @TEXT_POSTPROCESSORS
        # decorators run BEFORE mmengine lazy-parses the config files
        # (mmengine's LazyObject cannot execute decorator calls).
        if os.path.isdir(local_datasets_root):
            if local_datasets_root not in sys.path:
                sys.path.insert(0, local_datasets_root)
            custom_path = os.path.join(local_datasets_root, "custom_datasets.py")
            if os.path.isfile(custom_path):
                try:
                    import custom_datasets  # noqa: F401
                    print(f"[eval_sglang] Pre-imported custom_datasets from "
                          f"{custom_path}")
                except Exception as e:
                    print(f"[eval_sglang] WARNING: failed to pre-import "
                          f"custom_datasets: {e}", file=sys.stderr)

        search_roots = [local_datasets_root, datasets_root]

        all_datasets = []
        for ds_name in ds_str.strip().split():
            target = ds_name + ".py"
            found = None
            for search_root in search_roots:
                if not os.path.isdir(search_root):
                    continue
                # First check flat files directly in the directory
                flat_path = os.path.join(search_root, target)
                if os.path.isfile(flat_path):
                    found = flat_path
                    break
                # Then walk subdirectories
                for root, dirs, files in os.walk(search_root):
                    if target in files:
                        found = os.path.join(root, target)
                        break
                if found:
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Cannot find dataset config '{ds_name}' under "
                    f"{local_datasets_root} or {datasets_root}"
                )
            ds_cfg = MmConfig.fromfile(found, format_python_code=False)
            for key in ds_cfg:
                if key.endswith("_datasets") and isinstance(ds_cfg[key], list):
                    all_datasets.extend(ds_cfg[key])
                    print(f"[eval_sglang] Loaded {len(ds_cfg[key])} dataset(s) "
                          f"from {key} in {found}")
        if not all_datasets:
            raise ValueError(f"No datasets loaded from '{ds_str}'")
        return all_datasets

    # ── Main ──

    from datetime import datetime
    from mmengine.config import Config
    from opencompass.registry import PARTITIONERS, RUNNERS
    from opencompass.utils import get_logger

    logger = get_logger(log_level='DEBUG' if debug else 'INFO')

    repo_root = os.path.abspath(".")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    datasets = _load_datasets(datasets_str)

    model_cfg = dict(
        type=f'{SGLangModel.__module__}.{SGLangModel.__name__}',
        abbr=os.path.basename(hf_path) + '_sglang',
        path=hf_path,
        max_seq_len=sglang_max_tokens,
        batch_size=sglang_batch_size,
        tp_size=sglang_tp,
        page_size=sglang_page_size,
        max_total_tokens=sglang_max_tokens,
        mem_fraction_static=sglang_mem_fraction,
        enable_prefix_cache=enable_prefix_cache,
        attention_backend=sglang_attn_backend,
        run_cfg=dict(num_gpus=0, num_procs=1),
    )

    cfg_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    work_dir = os.path.join(work_dir_base, cfg_time_str)
    os.makedirs(work_dir, exist_ok=True)

    cfg_dict = dict(
        models=[model_cfg],
        datasets=datasets,
        work_dir=work_dir,
        infer=dict(
            partitioner=dict(
                type='opencompass.partitioners.NumWorkerPartitioner',
                num_worker=1,
                out_dir=os.path.join(work_dir, 'predictions/'),
            ),
            runner=dict(
                type='opencompass.runners.LocalRunner',
                max_num_workers=1,
                task=dict(type='opencompass.tasks.OpenICLInferTask'),
                debug=debug,
            ),
        ),
        eval=dict(
            partitioner=dict(
                type='opencompass.partitioners.NumWorkerPartitioner',
                num_worker=1,
                out_dir=os.path.join(work_dir, 'results/'),
            ),
            runner=dict(
                type='opencompass.runners.LocalRunner',
                max_num_workers=1,
                task=dict(type='opencompass.tasks.OpenICLEvalTask'),
                debug=debug,
            ),
        ),
    )

    cfg = Config(cfg_dict)
    logger.info(f'Work dir: {work_dir}')
    logger.info(f'Models: {[m["abbr"] for m in cfg.models]}')
    logger.info(f'Datasets: {[d.get("abbr", d.get("type", "?")) for d in cfg.datasets]}')

    # ── Infer ──
    logger.info('Starting inference...')
    partitioner = PARTITIONERS.build(cfg.infer.partitioner)
    tasks = partitioner(cfg)
    runner = RUNNERS.build(cfg.infer.runner)
    runner(tasks)

    # ── Eval ──
    logger.info('Starting evaluation...')
    eval_partitioner = PARTITIONERS.build(cfg.eval.partitioner)
    eval_tasks = eval_partitioner(cfg)
    logger.info(f'Eval tasks: {len(eval_tasks)}')
    if eval_tasks:
        eval_runner = RUNNERS.build(cfg.eval.runner)
        eval_runner(eval_tasks)
    else:
        logger.warning('No eval tasks found — check if predictions exist in the right location.')

    # ── Summary ──
    try:
        from opencompass.summarizers import DefaultSummarizer
        summarizer = DefaultSummarizer(cfg)
        summarizer.summarize()
    except Exception as e:
        logger.warning(f'Summary failed: {e}')

    logger.info(f'Done! Results in: {work_dir}')
