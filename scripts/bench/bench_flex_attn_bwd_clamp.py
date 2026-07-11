"""Before/after benchmark for the flex-attention backward window-clamp fix (commit
bc77fd1). Times the O(L^2) pre-fix kernels against the O(L*W) fixed ones and checks
that the gradients are unchanged.

The pre-fix baseline is synthesized in memory by reverting the four symbolic clamps
back to their Python-`if` form (or loaded from git via --baseline-ref). Only the
backward is timed (the forward is frozen/untimed), via tilelang `do_bench`.

Benchmarks the three HiLS-Attention paper configs (345M / 1.4B / 7B).

    python scripts/bench/bench_flex_attn_bwd_clamp.py [--seq-len N] [--path both]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
KERNEL_FILE = REPO_ROOT / "ops" / "flex_attn_tilelang.py"
LN2 = 0.6931471805599453

# The four symbolic clamps the fix introduced, and their pre-fix Python-`if` form.
# Each `find` must appear exactly once in the kernel file (asserted).
_BASELINE_REVERTS = [
    ("            q_hi = T.min(kv_hi + window_size + chunk_size, seq_len)",
     "            q_hi = kv_hi + window_size + chunk_size\n"
     "            if q_hi > seq_len:\n"
     "                q_hi = seq_len"),
    ("            kv_lo = T.max(kv_lo_raw, 0)",
     "            kv_lo = kv_lo_raw\n"
     "            if kv_lo < 0:\n"
     "                kv_lo = 0"),
    ("            kv_hi = T.min(q_hi, seq_len)",
     "            kv_hi = q_hi\n"
     "            if kv_hi > seq_len:\n"
     "                kv_hi = seq_len"),
    ("            q_hi = T.min(q_hi_raw, seq_len)",
     "            q_hi = q_hi_raw\n"
     "            if q_hi > seq_len:\n"
     "                q_hi = seq_len"),
]


# --- The three HiLS-Attention paper configs, as the flex-kernel shape of each
# scale: heads_q/heads_kv, head_dim, SWA window, chunk. Model-level params (layers,
# hidden, vocab, top-K, LoRA) do not affect this kernel. OLMo3-7B sliding layers
# run expand_to_chunk=False (window edge not expanded to the chunk start). ---
def _wl(id, heads_q, heads_kv, seq_len, window_size, chunk_size,
        dim_qk=128, dim_v=128, mask_lmk=True, expand_to_chunk=True, seed=0):
    return dict(id=id, batch=1, heads_q=heads_q, heads_kv=heads_kv, seq_len=seq_len,
                dim_qk=dim_qk, dim_v=dim_v, window_size=window_size, chunk_size=chunk_size,
                mask_lmk=mask_lmk, expand_to_chunk=expand_to_chunk, seed=seed)


def paper_workloads(seq_len):
    return [
        _wl("paper_small_345M",  16, 2,  seq_len, 512, 64, dim_qk=64,  dim_v=64,  seed=345),
        _wl("paper_medium_1p4B", 32, 4,  seq_len, 512, 64, dim_qk=64,  dim_v=64,  seed=1400),
        _wl("paper_large_7B",    32, 32, seq_len, 512, 64, dim_qk=128, dim_v=128,
            expand_to_chunk=False, seed=7000),
    ]


# --- Load current (fixed) and pre-fix (baseline) kernels as separate modules ---
def _load_module(src: str, name: str):
    tmp = tempfile.NamedTemporaryFile("w", suffix=f"_{name}.py", delete=False)
    tmp.write(src)
    tmp.close()
    spec = importlib.util.spec_from_file_location(name, tmp.name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _synthesize_baseline(fixed_src: str) -> str:
    src = fixed_src
    for find, replace in _BASELINE_REVERTS:
        if src.count(find) != 1:
            raise SystemExit(
                f"[baseline] expected exactly one occurrence of:\n    {find!r}\n"
                f"in {KERNEL_FILE}. The clamp-fix source drifted; update "
                f"_BASELINE_REVERTS (or use --baseline-ref).")
        src = src.replace(find, replace)
    return src


def load_versions(baseline_ref: str | None):
    fixed_src = KERNEL_FILE.read_text()
    fixed = _load_module(fixed_src, "flex_attn_fixed")
    if baseline_ref:
        baseline_src = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "show", f"{baseline_ref}:ops/flex_attn_tilelang.py"],
            text=True)
        note = f"git:{baseline_ref}"
    else:
        baseline_src = _synthesize_baseline(fixed_src)
        note = "synthesized (Python-if revert)"
    return fixed, _load_module(baseline_src, "flex_attn_baseline"), note


# --- Backward paths (kernel sequences mirror the repo's autograd backward) ---
def _dims(q, v):
    B, L, Hq, D_qk = q.shape
    H_kv, D_v = v.shape[-2], v.shape[-1]
    return B, L, Hq, H_kv, D_qk, D_v, Hq // H_kv


def frozen_forward(mod, q, k, v, cfg):
    """Untimed forward -> (o, lse[log2], fully-masked-row mask). Forward is unchanged
    by the fix, so either module works."""
    B, L, Hq, _, D_qk, D_v, groups = _dims(q, v)
    kern = mod.flex_attn_fwd(B, Hq, L, L, D_qk, D_v, cfg["window_size"], cfg["chunk_size"],
                             cfg["mask_lmk"], cfg["expand_to_chunk"], 128, 64, groups,
                             use_cache=False, sm_scale=cfg["sm_scale"])
    o, lse = kern(q, k, v, torch.tensor([0], device=q.device, dtype=torch.int32))
    return o, lse, ~torch.isfinite(lse)


def _delta(mod, o, do, dlse):
    B, L, Hq, D_v = o.shape
    delta = mod.flex_attn_bwd_preprocess(B, Hq, L, D_v)(o, do)  # [B, Hq, L] fp32
    if dlse is not None:
        delta = delta - dlse.permute(0, 2, 1).contiguous().to(delta.dtype) / LN2
    return delta


def bwd_atomic(mod, q, k, v, o, lse, do, dlse, cfg):
    B, L, Hq, H_kv, D_qk, D_v, groups = _dims(q, v)
    dq_len_padded = ((L + 7) // 8) * 8
    delta = _delta(mod, o, do, dlse)
    kern = mod.flex_attn_bwd(B, Hq, L, D_qk, D_v, cfg["window_size"], cfg["chunk_size"],
                             cfg["mask_lmk"], cfg["expand_to_chunk"], 64, 64,
                             threads=128, num_stages=1, groups=groups, sm_scale=cfg["sm_scale"])
    dq = torch.zeros([B, dq_len_padded, Hq, D_qk], dtype=torch.float32, device=q.device)
    dk = torch.zeros([groups, B, L, H_kv, D_qk], dtype=torch.float32, device=q.device)
    dv = torch.zeros([groups, B, L, H_kv, D_v], dtype=torch.float32, device=q.device)
    kern(q, k, v, do, lse, delta, dq, dk, dv)
    dq = mod.flex_attn_bwd_postprocess(B, Hq, L, dq_len_padded, D_qk)(dq)
    return dq, dk.sum(0).bfloat16(), dv.sum(0).bfloat16()


def bwd_two_phase(mod, q, k, v, o, lse, do, dlse, cfg):
    B, L, Hq, _, D_qk, D_v, groups = _dims(q, v)
    delta = _delta(mod, o, do, dlse)
    dq = mod.flex_attn_bwd_dq(B, Hq, L, D_qk, D_v, cfg["window_size"], cfg["chunk_size"],
                              cfg["mask_lmk"], cfg["expand_to_chunk"], 128, 128,
                              threads=256, num_stages=1, groups=groups,
                              sm_scale=cfg["sm_scale"])(q, k, v, do, lse, delta)
    dk, dv = mod.flex_attn_bwd_dkdv(B, Hq, L, D_qk, D_v, cfg["window_size"], cfg["chunk_size"],
                                    cfg["mask_lmk"], cfg["expand_to_chunk"], 128, 64,
                                    threads=256, num_stages=0, groups=groups,
                                    sm_scale=cfg["sm_scale"])(q, k, v, do, lse, delta)
    return dq, dk, dv


_RUNNERS = {"atomic": bwd_atomic, "two_phase": bwd_two_phase}


# --- Timing / correctness ---
def _do_bench(fn):
    from tilelang.profiler import do_bench
    torch.cuda.synchronize()
    ms = do_bench(fn, warmup=25.0, rep=100.0, return_mode="median")
    torch.cuda.synchronize()
    return float(ms) * 1000.0  # microseconds


def _max_rel(a, b):
    a = torch.nan_to_num(a.detach().float())
    b = torch.nan_to_num(b.detach().float())
    denom = a.abs().max().item()
    return (a - b).abs().max().item() / denom if denom > 0 else 0.0


def run_workload(cfg, fixed, baseline, paths, use_dlse):
    cfg = {**cfg, "sm_scale": 1.0 / math.sqrt(cfg["dim_qk"])}
    torch.manual_seed(cfg["seed"])
    B, L, Hq, Hkv = cfg["batch"], cfg["seq_len"], cfg["heads_q"], cfg["heads_kv"]
    q = torch.randn(B, L, Hq, cfg["dim_qk"], device="cuda", dtype=torch.bfloat16) * 0.5
    k = torch.randn(B, L, Hkv, cfg["dim_qk"], device="cuda", dtype=torch.bfloat16) * 0.5
    v = torch.randn(B, L, Hkv, cfg["dim_v"], device="cuda", dtype=torch.bfloat16) * 0.5
    o, lse, invalid = frozen_forward(fixed, q, k, v, cfg)

    torch.manual_seed(cfg["seed"] + 1)
    do = torch.randn_like(o)
    dlse = torch.randn_like(lse) if use_dlse else None
    do[invalid] = 0.0  # zero seeds on fully-masked rows so neither version produces NaN
    if dlse is not None:
        dlse[invalid] = 0.0

    rows = []
    for path in paths:
        run = _RUNNERS[path]
        args = (q, k, v, o, lse, do, dlse, cfg)
        gb, gf = run(baseline, *args), run(fixed, *args)
        rows.append(dict(
            id=cfg["id"], path=path,
            heads_q=Hq, heads_kv=Hkv, seq_len=L,
            window_size=cfg["window_size"], chunk_size=cfg["chunk_size"],
            baseline_us=_do_bench(lambda: run(baseline, *args)),
            fixed_us=_do_bench(lambda: run(fixed, *args)),
            grad_max_rel=max(_max_rel(b, f) for b, f in zip(gb, gf)),
            grad_finite=all(torch.isfinite(t).all().item() for t in gf),
        ))
        rows[-1]["speedup"] = rows[-1]["baseline_us"] / rows[-1]["fixed_us"]
    return rows


# --- Reporting ---
def print_table(rows, note):
    print(f"\nflex_attn backward window-clamp fix — before/after  (baseline = {note})")
    print(f"device: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    hdr = (f"{'workload':<26} {'HqxHkv':>7} {'L':>6} {'W/C':>9} {'path':>10} "
           f"{'base us':>10} {'fixed us':>10} {'speedup':>8} {'grad rel':>10}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        heads = f"{r['heads_q']}x{r['heads_kv']}"
        wc = f"{r['window_size']}/{r['chunk_size']}"
        flag = "" if r["grad_finite"] else "  !!NONFINITE"
        print(f"{r['id']:<26} {heads:>7} {r['seq_len']:>6} {wc:>9} {r['path']:>10} "
              f"{r['baseline_us']:>10.1f} {r['fixed_us']:>10.1f} "
              f"{r['speedup']:>7.2f}x {r['grad_max_rel']:>10.2e}{flag}")

    for path in sorted({r["path"] for r in rows}):
        s = [r["speedup"] for r in rows if r["path"] == path]
        if s:
            geo = math.exp(sum(map(math.log, s)) / len(s))
            print(f"\n  [{path}] geomean speedup = {geo:.3f}x  (n={len(s)})")
    print(f"\n  worst grad max-rel-diff (fixed vs baseline) = {max(r['grad_max_rel'] for r in rows):.2e}"
          "  -> gradients numerically unchanged.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-len", type=int, default=16384,
                    help="context length for the three paper configs (default 16384)")
    ap.add_argument("--path", choices=["two_phase", "atomic", "both"], default="two_phase",
                    help="backward path to benchmark (default: two_phase = shipped path)")
    ap.add_argument("--dlse", action="store_true",
                    help="seed a non-zero dlse (exercise the lse-fold-into-Delta path)")
    ap.add_argument("--baseline-ref",
                    help="load the O(L^2) baseline from a git revision (e.g. bc77fd1^)")
    ap.add_argument("--json-out", help="append per-row results as JSONL here")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (TileLang kernels).")

    paths = ["two_phase", "atomic"] if args.path == "both" else [args.path]
    workloads = paper_workloads(args.seq_len)

    t0 = time.perf_counter()
    print(f"loading fixed + baseline kernels from {KERNEL_FILE} ...")
    fixed, baseline, note = load_versions(args.baseline_ref)

    rows = []
    for i, wl in enumerate(workloads, 1):
        print(f"\n[{i}/{len(workloads)}] {wl['id']} "
              f"(L={wl['seq_len']}, W={wl['window_size']}, C={wl['chunk_size']}, paths={paths}) ...")
        try:
            rows.extend(run_workload(wl, fixed, baseline, paths, args.dlse))
        except Exception as e:
            import traceback
            print(f"  [ERROR] {wl['id']}: {type(e).__name__}: {e}")
            traceback.print_exc()

    if rows:
        print_table(rows, note)
    if args.json_out and rows:
        with open(args.json_out, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
        print(f"\n  wrote {len(rows)} rows to {args.json_out}")
    print(f"\ndone in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
