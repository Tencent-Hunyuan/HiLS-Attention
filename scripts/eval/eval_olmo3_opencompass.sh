#!/usr/bin/env bash
# ============================================================================
# eval_olmo3_opencompass_bsz.sh — Batch-size Transformers OpenCompass evaluation
#
# Usage:
#   bash scripts/eval/eval_olmo3_opencompass_bsz.sh
#
# This is a Transformers-backend (HuggingFace) evaluation script modelled on
# scripts/eval/eval_sglang_all4.sh but using eval/eval_opencompass.py instead of
# the SGLang variant. Each (model, dataset) pair runs on a single GPU with a
# configurable `--batch-size` and `--max-out-len`. A GPU queue keeps every card
# busy: all (model, dataset) pairs are flattened into one work list.
# ============================================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

# ── Paths ──
TOKENIZER_PATH=${TOKENIZER_PATH:-configs/olmo3_vocab}
OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-}
PYTHON_BIN=${PYTHON_BIN:-python}
export PYTHONPATH="${REPO_ROOT}:${OPENCOMPASS_PATH}${PYTHONPATH:+:$PYTHONPATH}"


if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "Neither python nor python3 is available, please activate the runtime environment first" >&2
        exit 1
    fi
fi

lmk_token_tuning_config=configs/olmo3_7B/olmo3_8KA2K_lmk_token_tuning.json
lmk_token_tuning_ckpt=../../checkpoints/olmo3_8KA2K_lmk_token_tuning/global_step_1192/hf_ckpt

olmo3_8KA2K_HoPE_LoRA_config=configs/olmo3_7B/olmo3_8KA2K_HoPE_LoRA.json
olmo3_8KA2K_HoPE_LoRA_ckpt=../../checkpoints/olmo3_8KA2K_HoPE_LoRA/global_step_13000/hf_ckpt

# ── Models ── (parallel arrays)
MODEL_NAMES=(
    lmk_token_tuning
    olmo3_8KA2K_HoPE_LoRA_step13000
)
HILS_CONFIGS=(
    $lmk_token_tuning_config
    $olmo3_8KA2K_HoPE_LoRA_config
)
HF_PATHS=(
    $lmk_token_tuning_ckpt
    $olmo3_8KA2K_HoPE_LoRA_ckpt
)

MAX_PREFETCH_RETRIES=0
PREFETCH_OK=0
for i in $(seq 1 $MAX_PREFETCH_RETRIES); do
    echo "[Prefetch] Attempt $i/$MAX_PREFETCH_RETRIES ..."
    if python code_exp/prefetch.py ${HILS_CONFIGS[0]}; then
        echo "[Prefetch] Success on attempt $i."
        PREFETCH_OK=1
        break
    fi
    if [ $i -eq $MAX_PREFETCH_RETRIES ]; then
        echo "[Prefetch] Failed after $MAX_PREFETCH_RETRIES attempts, continuing anyway." >&2
    else
        echo "[Prefetch] Attempt $i failed, retrying..."
        sleep 2
    fi
done
if [ $PREFETCH_OK -eq 0 ]; then
    echo "[Prefetch] Warning: checkpoint may not be warmed; eval will proceed." >&2
fi
# ── Datasets ──
DATASET_LIST=(
    gpqa_few_shot_ppl_4b5a83
    mmlu_ppl_ac766d
    hellaswag_10shot_ppl_59c85e
    ARC_c_few_shot_ppl
    SuperGLUE_BoolQ_few_shot_ppl
    race_few_shot_ppl
    gsm8k_gen
    cmath_gen
    humaneval_plus_gen
    mbpp_plus_gen
    cruxeval_o_gen
)

# ── GPUs & eval settings ──
GPU_IDS=(0 1 2 3 4 5 6 7)
# Number of GPUs each (model, dataset) pair uses. When >1, OpenCompass's
# NumWorkerPartitioner splits the dataset into shards and LocalRunner
# runs them concurrently on those GPUs, then OpenICLEvalTask aggregates
# the `{abbr}_{part}.json` predictions into a single summary.
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_OUT_LEN=${MAX_OUT_LEN:-1024}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-}
DEBUG=${DEBUG:-1}


LOCAL_STAGE_DIR=${LOCAL_STAGE_DIR:-}
# If set (default on), warm the OS page cache with a single sequential
# read of the weight shards before forking the infer subprocesses.
WARMUP_PAGE_CACHE=${WARMUP_PAGE_CACHE:-}

# ── Validation ──
if [ "${#MODEL_NAMES[@]}" -ne "${#HILS_CONFIGS[@]}" ] || \
   [ "${#MODEL_NAMES[@]}" -ne "${#HF_PATHS[@]}" ]; then
    echo "MODEL_NAMES / HILS_CONFIGS / HF_PATHS length mismatch" >&2; exit 1
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS must not be empty" >&2; exit 1
fi

if [ "$GPUS_PER_TASK" -lt 1 ]; then
    echo "GPUS_PER_TASK must be >= 1" >&2; exit 1
fi

if [ "$GPUS_PER_TASK" -gt "${#GPU_IDS[@]}" ]; then
    echo "GPUS_PER_TASK (${GPUS_PER_TASK}) > #GPU_IDS (${#GPU_IDS[@]}), will clamp to ${#GPU_IDS[@]}" >&2
    GPUS_PER_TASK="${#GPU_IDS[@]}"
fi

if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "TOKENIZER_PATH does not exist: $TOKENIZER_PATH" >&2; exit 1
fi

if [ ! -d "$OPENCOMPASS_PATH/opencompass" ]; then
    echo "OPENCOMPASS_PATH does not exist or is not an OpenCompass repository: $OPENCOMPASS_PATH" >&2; exit 1
fi

# Kill any lingering burner processes from previous runs.
if ! pkill -f "burner" >/dev/null 2>&1; then
    echo "No existing burner process found."
fi

# ── Output ──
RUN_TAG=$(date +%Y%m%d_%H%M%S)
MASTER_LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_opencompass_bsz_${RUN_TAG}"
mkdir -p "$MASTER_LOG_DIR"

echo "=============================================="
echo " Transformers OpenCompass Multi-GPU Evaluation (batch mode)"
echo "=============================================="
echo " Models:       ${MODEL_NAMES[*]}"
echo " Datasets:     ${DATASET_LIST[*]}"
echo " GPUs:         ${GPU_IDS[*]}"
echo " GPUs/task:    ${GPUS_PER_TASK}"
echo " Batch size:   ${BATCH_SIZE}"
echo " Max out len:  ${MAX_OUT_LEN}"
echo " Log dir:      $MASTER_LOG_DIR"
echo "=============================================="

# ── Prepare resolved HF path (symlink weights + tokenizer + write config.json) ──
prepare_hf_path() {
    local hf_path="$1" hils_config="$2" resolved_hf_path="$3"
    rm -rf "$resolved_hf_path"
    mkdir -p "$resolved_hf_path"

    shopt -s nullglob dotglob
    for src in "$hf_path"/*; do
        local base
        base=$(basename "$src")
        case "$base" in
            config.json|config_*.json|generation_config.json|tokenizer.json|tokenizer_config.json|vocab.json|merges.txt|special_tokens_map.json)
                continue
                ;;
        esac
        ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$base"
    done
    for src in "$TOKENIZER_PATH"/*; do
        ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")"
    done
    shopt -u nullglob dotglob

    "$PYTHON_BIN" - "$hf_path" "$hils_config" "$TOKENIZER_PATH" "$resolved_hf_path/config.json" <<'PYEOF'
import json, os, sys

hf_dir, hils_path, tokenizer_dir, dst_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

with open(hils_path, "r", encoding="utf-8") as f:
    hils = json.load(f)

def detect_tokenizer_vocab_size(tokenizer_dir, default):
    p = os.path.join(tokenizer_dir, "vocab.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return len(json.load(f))
    p = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            tok = json.load(f)
        v = tok.get("model", {}).get("vocab", {})
        if v: return len(v)
    return int(default)

# Always build config from the requested hils_config (e.g. config_hf.json for
# Olmo3ForCausalLM), optionally overlay checkpoint-only fields we still need.
base_cfg_path = os.path.join(hf_dir, "config.json")
if os.path.exists(base_cfg_path):
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    config = dict(hils)
    for key in ("torch_dtype", "transformers_version"):
        if key in base and key not in config:
            config[key] = base[key]
else:
    config = dict(hils)

config["insert_landmarks"] = bool(config.get("insert_landmarks") or config.get("adjust_lmk_pos"))
config["adjust_lmk_pos"]   = bool(config.get("adjust_lmk_pos", False))
config["vocab_size"]       = detect_tokenizer_vocab_size(tokenizer_dir, config.get("vocab_size", 0))

if os.path.lexists(dst_path):
    os.remove(dst_path)

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2); f.write("\n")
print(f"[config] model_type={config.get('model_type')}, arch={config.get('architectures')}, "
      f"vocab_size={config.get('vocab_size')}, insert_lmk={config['insert_landmarks']}, "
      f"adjust_lmk_pos={config['adjust_lmk_pos']}, hils_config={hils_path}")
PYEOF
}

# ── Summary augmentation: append overall_average row ──
append_overall_average_to_summary() {
    local summary_file="$1"
    "$PYTHON_BIN" - "$summary_file" <<'PY'
import csv, io, sys
summary_file = sys.argv[1]
with open(summary_file, "r", encoding="utf-8") as fin:
    text = fin.read()
if "overall_average" in text:
    raise SystemExit(0)
try:
    from tabulate import tabulate
except ImportError:
    tabulate = None
csv_marker = "csv format\n"
divider_marker = "$" * 124
table_marker = "tabulate format\n"
csv_start = text.find(csv_marker)
if csv_start < 0:
    raise SystemExit(0)
csv_after_marker = text[csv_start + len(csv_marker):]
csv_caret_end = csv_after_marker.find("\n")
if csv_caret_end < 0:
    raise SystemExit(0)
csv_preamble = text[: csv_start + len(csv_marker) + csv_caret_end + 1]
csv_text = csv_after_marker[csv_caret_end + 1 :].lstrip("\n")
csv_lines = []
for line in csv_text.splitlines():
    if not line.strip():
        continue
    if "," not in line:
        if csv_lines:
            break
        continue
    csv_lines.append(line)
if not csv_lines:
    raise SystemExit(0)
reader = csv.reader(io.StringIO("\n".join(csv_lines)))
rows = list(reader)
if len(rows) < 2:
    raise SystemExit(0)
header = rows[0]
data_rows = rows[1:]
value_start = 4
if len(header) <= value_start:
    raise SystemExit(0)
column_sums = [0.0] * (len(header) - value_start)
column_counts = [0] * (len(header) - value_start)
for row in data_rows:
    if not row or row[0] == "overall_average":
        continue
    for i in range(value_start, min(len(row), len(header))):
        cell = row[i].strip()
        if not cell:
            continue
        try:
            value = float(cell)
        except ValueError:
            continue
        column_sums[i - value_start] += value
        column_counts[i - value_start] += 1
metric = next((row[2] for row in data_rows if len(row) > 2 and row[0] != "overall_average"), "average")
mode = next((row[3] for row in data_rows if len(row) > 3 and row[0] != "overall_average"), "average")
overall_row = ["overall_average", "-", metric, mode]
for total, count in zip(column_sums, column_counts):
    overall_row.append(f"{(total / count):.2f}" if count else "")
csv_rows = rows + [overall_row]
csv_output = io.StringIO()
writer = csv.writer(csv_output, lineterminator="\n")
writer.writerows(csv_rows)
csv_block = csv_output.getvalue().rstrip("\n")
table_start = text.find(table_marker)
if table_start >= 0:
    table_after_marker = text[table_start + len(table_marker):]
    table_caret_end = table_after_marker.find("\n")
    if table_caret_end >= 0:
        table_preamble = text[: table_start + len(table_marker) + table_caret_end + 1]
        divider_start = text.find(divider_marker)
        if divider_start >= 0:
            if tabulate is not None:
                table_block = tabulate(
                    data_rows + [overall_row],
                    headers=header,
                    tablefmt="simple",
                    stralign="left",
                    numalign="right",
                )
            else:
                table_rows = [header] + data_rows + [overall_row]
                widths = [0] * len(header)
                for row in table_rows:
                    for i, cell in enumerate(row):
                        widths[i] = max(widths[i], len(str(cell)))
                def format_table_row(row):
                    parts = []
                    for i, cell in enumerate(row):
                        cell = str(cell)
                        if i >= value_start:
                            parts.append(cell.rjust(widths[i]))
                        else:
                            parts.append(cell.ljust(widths[i]))
                    return "  ".join(parts)
                separator = "  ".join("-" * width for width in widths)
                table_block = "\n".join(
                    [format_table_row(header), separator]
                    + [format_table_row(row) for row in data_rows + [overall_row]]
                )
            text = table_preamble + table_block + "\n" + text[divider_start:]
csv_start = text.find(csv_marker)
csv_after_marker = text[csv_start + len(csv_marker):]
csv_caret_end = csv_after_marker.find("\n")
csv_preamble = text[: csv_start + len(csv_marker) + csv_caret_end + 1]
text = csv_preamble + csv_block + "\n"
with open(summary_file, "w", encoding="utf-8") as fout:
    fout.write(text)
PY
}

# ── GPU queue management (supports multi-GPU allocation per task) ──
available_gpus=()
active_pids=()
active_gpu_groups=()  # comma-separated GPU ids per active job
active_labels=()
REAPED_GPU_GROUP=""
queue_failed=0

# Pop GPUS_PER_TASK ids from available_gpus and return them comma-joined in
# REAPED_GPU_GROUP. Assumes there are at least GPUS_PER_TASK gpus available.
pop_gpu_group() {
    local take="$GPUS_PER_TASK"
    local -a taken=("${available_gpus[@]:0:take}")
    available_gpus=("${available_gpus[@]:take}")
    REAPED_GPU_GROUP=$(IFS=,; echo "${taken[*]}")
}

reap_one_job() {
    local finished_pid wait_status=0
    wait -n -p finished_pid "${active_pids[@]}" || wait_status=$?
    [ "$wait_status" -ne 0 ] && queue_failed=1
    for idx in "${!active_pids[@]}"; do
        if [ "${active_pids[$idx]}" = "$finished_pid" ]; then
            local released_group="${active_gpu_groups[$idx]}"
            # Return GPUs to the available pool one by one
            local -a released_list=()
            IFS=',' read -r -a released_list <<< "$released_group"
            available_gpus+=("${released_list[@]}")
            echo "[$(date '+%F %T')] Done: ${active_labels[$idx]} (gpus=${released_group}, status=${wait_status})" \
                | tee -a "$MASTER_LOG_DIR/queue.log"
            unset 'active_pids[idx]' 'active_gpu_groups[idx]' 'active_labels[idx]'
            active_pids=("${active_pids[@]}"); active_gpu_groups=("${active_gpu_groups[@]}"); active_labels=("${active_labels[@]}")
            return
        fi
    done
}

drain_all() { while [ "${#active_pids[@]}" -gt 0 ]; do reap_one_job; done; }

# ── Single (model, dataset) evaluation ──
run_one() {
    local gpu_group="$1" model_name="$2" hils_config="$3" hf_path="$4" dataset="$5" work_dir="$6"
    local resolved_hf_path="$work_dir/hf_with_tokenizer"
    local run_log="$work_dir/run.log"

    # Count GPUs in the comma-separated group — this is also the shard count
    # we ask NumWorkerPartitioner to split the dataset into.
    local -a gpu_list=()
    IFS=',' read -r -a gpu_list <<< "$gpu_group"
    local num_gpus="${#gpu_list[@]}"

    prepare_hf_path "$hf_path" "$hils_config" "$resolved_hf_path" 2>&1 | tee -a "$run_log"

    local cmd=(
        "$PYTHON_BIN" eval/eval_opencompass.py
        --datasets "$dataset"
        --hf-type base
        --hf-path "$resolved_hf_path"
        --hils-config "$hils_config"
        --batch-size "$BATCH_SIZE"
        --max-out-len "$MAX_OUT_LEN"
        --max-num-workers "$num_gpus"
        --hf-num-gpus 1
        -w "$work_dir"
    )
    [ -n "$MAX_SEQ_LEN" ] && cmd+=(--max-seq-len "$MAX_SEQ_LEN")

    # Pass insert_landmarks via model kwargs, matching eval_olmo3_opencompass_all3.sh
    local insert_lmk
    insert_lmk=$("$PYTHON_BIN" -c "import json,sys;c=json.load(open(sys.argv[1]));print('1' if c.get('insert_landmarks') or c.get('adjust_lmk_pos') else '0')" "$hils_config")

    cmd+=(--model-kwargs torch_dtype=torch.bfloat16 attn_implementation=flash_attention_3)
    [ "$insert_lmk" = "1" ] && cmd+=(auto_insert_lmk=True)
    # NOTE: do NOT pass --debug when num_gpus > 1; in debug mode LocalRunner
    # runs shards serially on the same GPU set, defeating the parallelism.
    if [ "$DEBUG" = "1" ] && [ "$num_gpus" -eq 1 ]; then
        cmd+=(--debug)
    fi

    {
        echo "[$(date '+%F %T')] Start: model=${model_name}, dataset=${dataset}, gpus=${gpu_group} (shards=${num_gpus})"
        echo "[$(date '+%F %T')] Work dir: ${work_dir}"
        echo "[$(date '+%F %T')] HF path:  ${hf_path}"
        echo "[$(date '+%F %T')] Config:   ${hils_config}"
        echo "[$(date '+%F %T')] Batch:    ${BATCH_SIZE}  MaxOut: ${MAX_OUT_LEN}  MaxSeq: ${MAX_SEQ_LEN:-<model default>}"
        echo "[$(date '+%F %T')] Command:  CUDA_VISIBLE_DEVICES=${gpu_group} ${cmd[*]}"
    } | tee -a "$run_log"

    if ! CUDA_VISIBLE_DEVICES="$gpu_group" "${cmd[@]}" 2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] ERROR: model=${model_name}, dataset=${dataset} failed" | tee -a "$run_log"
        return 1
    fi

    local latest_summary_file=""
    latest_summary_file=$(find "$work_dir" -maxdepth 4 -path '*/summary/summary_*.txt' | sort | tail -n 1 || true)
    if [ -n "$latest_summary_file" ]; then
        append_overall_average_to_summary "$latest_summary_file"
        echo "[$(date '+%F %T')] Added overall average to summary: ${latest_summary_file}" | tee -a "$run_log"
    fi

    echo "[$(date '+%F %T')] Finished: model=${model_name}, dataset=${dataset}, gpus=${gpu_group}" | tee -a "$run_log"
}

# ── Main loop: flatten all (model, dataset) pairs into one GPU-queued work list ──
available_gpus=("${GPU_IDS[@]}")
active_pids=(); active_gpu_groups=(); active_labels=()

# ── Stage a checkpoint to local disk once per model (optional) ──
# Copies weight shards + tokenizer + config files into $LOCAL_STAGE_DIR/<model>
# so per-dataset work dirs can symlink from local NVMe instead of pulling the
# same 10+GB shards over shared storage for every shard process.
stage_model_to_local() {
    local model_name="$1" src_hf_path="$2"
    local staged_dir="$LOCAL_STAGE_DIR/$model_name"
    if [ -d "$staged_dir" ] && [ -f "$staged_dir/.stage_done" ]; then
        echo "[stage] Reusing existing staged checkpoint: $staged_dir"
        echo "$staged_dir"
        return 0
    fi
    mkdir -p "$staged_dir"
    echo "[stage] Copying $src_hf_path -> $staged_dir ..."
    # Copy EVERY file (resolves symlinks via -L); safetensors + tokenizer.
    cp -Lr --reflink=auto "$src_hf_path"/. "$staged_dir"/ 2>/dev/null \
        || cp -Lr "$src_hf_path"/. "$staged_dir"/
    touch "$staged_dir/.stage_done"
    echo "[stage] Done: $staged_dir"
    echo "$staged_dir"
}

# Warm OS page cache for weight shards to avoid N shard processes all cold-
# missing on the same bytes simultaneously (effective on local disk too).
warmup_weights() {
    local hf_path="$1"
    local f
    shopt -s nullglob
    for f in "$hf_path"/*.safetensors "$hf_path"/*.bin; do
        # Use dd to push bytes through the page cache. 32MiB blocks.
        dd if="$f" of=/dev/null bs=32M status=none || true
    done
    shopt -u nullglob
}

for model_idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$model_idx]}"
    hils_config="${HILS_CONFIGS[$model_idx]}"
    hf_path="${HF_PATHS[$model_idx]}"
    model_log_dir="$MASTER_LOG_DIR/$model_name"
    mkdir -p "$model_log_dir"

    # Optionally stage to local disk (one-time per model)
    effective_hf_path="$hf_path"
    if [ -n "$LOCAL_STAGE_DIR" ]; then
        mkdir -p "$LOCAL_STAGE_DIR"
        effective_hf_path=$(stage_model_to_local "$model_name" "$hf_path" | tail -n 1)
    fi

    # Warm page cache sequentially (cheap vs shard storms later)
    if [ "$WARMUP_PAGE_CACHE" = "1" ]; then
        echo "[warmup] Warming page cache for $effective_hf_path"
        warmup_weights "$effective_hf_path"
    fi

    for dataset in "${DATASET_LIST[@]}"; do
        # Wait until we have enough GPUs for this task
        while [ "${#available_gpus[@]}" -lt "$GPUS_PER_TASK" ]; do
            reap_one_job
        done
        pop_gpu_group; gpu_group="$REAPED_GPU_GROUP"

        ds_tag=$(printf '%s' "$dataset" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
        work_dir="$model_log_dir/$ds_tag"
        mkdir -p "$work_dir"

        echo "[$(date '+%F %T')] Launch: model=${model_name}, dataset=${dataset}, gpus=${gpu_group}" \
            | tee -a "$MASTER_LOG_DIR/queue.log"

        run_one "$gpu_group" "$model_name" "$hils_config" "$effective_hf_path" "$dataset" "$work_dir" &
        active_pids+=($!); active_gpu_groups+=("$gpu_group"); active_labels+=("${model_name}/${dataset}")
    done
done
drain_all
echo "[$(date '+%F %T')] All models and datasets done."

# ── Re-score HumanEval+ / MBPP+ with evalplus ──
# OpenCompass generates predictions but doesn't always run evalplus scoring
# correctly. Use our rescore script to compute pass@1 from predictions.
echo ""
echo "=============================================="
echo " Re-scoring HumanEval+ / MBPP+ predictions"
echo "=============================================="
RESCORE_PY="$REPO_ROOT/eval/rescore_evalplus.py"
if [ -f "$RESCORE_PY" ]; then
    for model_idx in "${!MODEL_NAMES[@]}"; do
        model_name="${MODEL_NAMES[$model_idx]}"
        model_log_dir="$MASTER_LOG_DIR/$model_name"

        # Find and score humaneval_plus predictions
        while IFS= read -r pred_file; do
            echo "[rescore] Scoring HumanEval+: $pred_file"
            "$PYTHON_BIN" "$RESCORE_PY" "$pred_file" --dataset humaneval --tag "$model_name" \
                2>&1 | tee -a "$model_log_dir/rescore_humaneval.log" || true
        done < <(find "$model_log_dir" -path "*/humaneval_plus_gen/*/predictions/*/humaneval_plus*.json" 2>/dev/null | sort -u)

        # Find and score mbpp_plus predictions
        while IFS= read -r pred_file; do
            echo "[rescore] Scoring MBPP+: $pred_file"
            "$PYTHON_BIN" "$RESCORE_PY" "$pred_file" --dataset mbpp --tag "$model_name" \
                2>&1 | tee -a "$model_log_dir/rescore_mbpp.log" || true
        done < <(find "$model_log_dir" -path "*/mbpp_plus_gen/*/predictions/*/mbpp_plus*.json" 2>/dev/null | sort -u)
    done
    echo "[$(date '+%F %T')] Evalplus re-scoring done."
else
    echo "[WARNING] $RESCORE_PY not found, skipping evalplus re-scoring." >&2
fi

# ── Generate LaTeX summary ──
"$PYTHON_BIN" - "$MASTER_LOG_DIR" <<'PYEOF'
import csv, io, os, sys, re

master = sys.argv[1]
DATASET_ORDER = [
    ("mmlu_ppl_ac766d",              "MMLU(5-shot)"),
    ("gpqa_few_shot_ppl_4b5a83",     "GPQA(5-shot)"),
    ("hellaswag_10shot_ppl_59c85e",  "Hellaswag(10-shot)"),
    ("ARC_c_few_shot_ppl",           "ARC-c(25-shot)"),
    ("SuperGLUE_BoolQ_few_shot_ppl", "BoolQ(5-shot)"),
    ("race_few_shot_ppl",            "Race(3-shot)"),
]

# Generation-based datasets whose scores come from OpenCompass summary or
# evalplus rescore logs rather than PPL-style summary files.
GEN_DATASETS = [
    ("gsm8k_gen",          "GSM8K"),
    ("cmath_gen",          "CMath"),
    ("humaneval_plus_gen", "HumanEval+"),
    ("mbpp_plus_gen",      "MBPP+"),
    ("cruxeval_o_gen",     "CruxEval-O"),
]

def find_summary(d):
    for r, _, fs in os.walk(d):
        for f in sorted(fs):
            if f.startswith("summary_") and f.endswith(".txt"):
                return os.path.join(r, f)
    return None

def extract_score(path):
    with open(path) as f: text = f.read()
    idx = text.find("csv format\n")
    if idx < 0: return None
    csv_text = text[idx+len("csv format\n"):]
    nl = csv_text.find("\n")
    if nl < 0: return None
    csv_text = csv_text[nl+1:].lstrip("\n")
    lines = [l for l in csv_text.splitlines() if l.strip() and "," in l]
    if not lines: return None
    for row in csv.reader(io.StringIO("\n".join(lines))):
        if row and row[0].strip() == "overall_average":
            for cell in reversed(row):
                try: return float(cell.strip())
                except: pass
    for row in csv.reader(io.StringIO("\n".join(lines))):
        if row and row[0].strip() != "dataset":
            for cell in reversed(row):
                try: return float(cell.strip())
                except: pass
    return None

def extract_gen_score(model_dir, ds_tag):
    """Extract score for generation datasets from OpenCompass summary or rescore log."""
    ds_dir = os.path.join(model_dir, ds_tag)
    if not os.path.isdir(ds_dir):
        return None

    # For evalplus datasets, try rescore log first
    if 'humaneval' in ds_tag:
        log_path = os.path.join(model_dir, "rescore_humaneval.log")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                text = f.read()
            # Look for "plus_pass_1" score
            m = re.search(r'humaneval_plus_plus_pass_1\s*=\s*([0-9.]+)', text)
            if m:
                return float(m.group(1))
            # Fallback: look for base_pass_1
            m = re.search(r'humaneval_plus_base_pass_1\s*=\s*([0-9.]+)', text)
            if m:
                return float(m.group(1))
    elif 'mbpp' in ds_tag:
        log_path = os.path.join(model_dir, "rescore_mbpp.log")
        if os.path.isfile(log_path):
            with open(log_path) as f:
                text = f.read()
            m = re.search(r'mbpp_plus_plus_pass_1\s*=\s*([0-9.]+)', text)
            if m:
                return float(m.group(1))
            m = re.search(r'mbpp_plus_base_pass_1\s*=\s*([0-9.]+)', text)
            if m:
                return float(m.group(1))

    # For all gen datasets, try the OpenCompass summary file
    sf = find_summary(ds_dir)
    if sf:
        v = extract_score(sf)
        if v is not None:
            return v
    return None

models = sorted(d for d in os.listdir(master) if os.path.isdir(os.path.join(master, d)))
scores = {}
for m in models:
    scores[m] = {}
    # PPL datasets
    for ds_internal, _ in DATASET_ORDER:
        ds_tag = re.sub(r"[^A-Za-z0-9_.\-]", "", ds_internal.replace(",","_").replace("/","_"))
        sf = find_summary(os.path.join(master, m, ds_tag))
        if sf:
            v = extract_score(sf)
            if v is not None: scores[m][ds_internal] = v
    # Generation datasets
    for ds_internal, _ in GEN_DATASETS:
        ds_tag = re.sub(r"[^A-Za-z0-9_.\-]", "", ds_internal.replace(",","_").replace("/","_"))
        v = extract_gen_score(os.path.join(master, m), ds_tag)
        if v is not None:
            scores[m][ds_internal] = v

all_datasets = DATASET_ORDER + GEN_DATASETS
display = [dn for _, dn in all_datasets]
header = "& " + " & ".join(display + ["AVG"]) + "\\\\"
lines = ["% Auto-generated Transformers OpenCompass batch-size summary", f"% Log dir: {master}", "", header, "\\midrule"]
for m in models:
    vals, valid = [], []
    for ds, _ in all_datasets:
        v = scores[m].get(ds)
        if v is not None: vals.append(f"{v:.2f}"); valid.append(v)
        else: vals.append("-")
    avg = f"{sum(valid)/len(valid):.2f}" if valid else "-"
    vals.append(avg)
    lines.append(f"{m} & " + " & ".join(vals) + "\\\\")

out = "\n".join(lines) + "\n"
p = os.path.join(master, "summary.log")
with open(p, "w") as f: f.write(out)
print(f"Summary: {p}\n{out}")
PYEOF

echo ""
echo "=============================================="
echo " All done! Results in: $MASTER_LOG_DIR"
echo "=============================================="

[ "$queue_failed" -ne 0 ] && { echo "Some jobs failed." >&2; exit 1; }
