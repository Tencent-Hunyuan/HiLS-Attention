#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

# ============================================================
#  Signal handling: make Ctrl+C actually stop everything.
#  The original script spawns background subshells that each loop
#  over `python ... | tee ...`; without a trap, SIGINT kills the
#  current python, but `if ! ... ; then continue; fi` swallows the
#  failure and the subshell happily launches the next python.
# ============================================================
SHUTDOWN=0

# Recursively send <sig> to every descendant of <pid>, deepest first.
kill_descendants() {
    local sig="${1:-TERM}"
    local parent="${2:-$$}"
    local kids
    kids=$(pgrep -P "$parent" 2>/dev/null || true)
    local pid
    for pid in $kids; do
        kill_descendants "$sig" "$pid"
        kill -"$sig" "$pid" 2>/dev/null || true
    done
}

shutdown_handler() {
    local sig="${1:-INT}"
    if [ "$SHUTDOWN" -ne 0 ]; then
        return
    fi
    SHUTDOWN=1
    trap - INT TERM
    echo "" >&2
    echo "[$(date '+%F %T')] Received SIG${sig}: terminating all child processes..." >&2
    kill_descendants TERM "$$"
    local i
    for i in 1 2 3 4 5; do
        if ! pgrep -P "$$" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    if pgrep -P "$$" >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] Some children still alive, sending SIGKILL..." >&2
        kill_descendants KILL "$$"
    fi
    exit 130
}

trap 'shutdown_handler INT'  INT
trap 'shutdown_handler TERM' TERM

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

cd "$REPO_ROOT"
export PYTHONPATH=./

if [ -n "${PYTHON_BIN:-}" ]; then
    true
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
else
    echo "Neither python nor python3 is available, please activate the runtime environment first" >&2
    exit 1
fi

# ============================================================
#  Eval mode: ppl / ruler / all
# ============================================================
EVAL_MODE="${EVAL_MODE:-all}"

# ============================================================
#  Models: each entry is  "name|config|ckpt_path|ruler_max_seq_len"
#  ruler_max_seq_len is optional (0 or omitted = no limit)
# ============================================================

lmk_token_tuning_config=configs/olmo3_7B/olmo3_8KA2K_lmk_token_tuning.json
lmk_token_tuning_ckpt=../../checkpoints/olmo3_8KA2K_lmk_token_tuning/global_step_1192/hf_ckpt

olmo3_8KA2K_HoPE_LoRA_config=configs/olmo3_7B/olmo3_8KA2K_HoPE_LoRA.json
olmo3_8KA2K_HoPE_LoRA_ckpt=../../checkpoints/olmo3_8KA2K_HoPE_LoRA/global_step_13000/hf_ckpt

# ============================================================
#  Models: "name|config|ckpt_path"
#  config can be empty for standard HF models
# ============================================================
MODELS=(
    "lmk_token_tuning|$lmk_token_tuning_config|$lmk_token_tuning_ckpt"
    "olmo3_8KA2K_HoPE_LoRA_step13000|$olmo3_8KA2K_HoPE_LoRA_config|$olmo3_8KA2K_HoPE_LoRA_ckpt"
)



# ============================================================
#  GPU pool
# ============================================================
GPU_IDS=(
0 1 2 3 4 5 6 7
)

# ============================================================
#  PPL settings
# ============================================================
PPL_SEQ_LEN_LIST=(
    64
    128
    512
    $((8 * 1024))
    $((16 * 1024))
    $((64 * 1024))
    $((128 * 1024))
    $((256 * 1024))
    # $((512 * 1024))
    # $((1024 * 1024))
)
PPL_DATA_PATH=../../data/dolma3_mix-6T-1025-partial-tokenized
PPL_MAX_SAMPLES=100
PPL_LAST_K_TOKENS=512
PPL_TP_SIZE=-1
PPL_TP_MIN_LEN=$((64 * 1024))
PPL_TP_GPU_IDS=(0 1 2 3 4 5 6 7)
PPL_TP_EXCLUSIVE=0
PPL_SEGMENT_SIZE=-1
PPL_CHUNK_PREFILL_MIN_LEN=$((64 * 1024))

# ============================================================
#  RULER settings
# ============================================================
RULER_SEQ_LEN_LIST=(
    $((8 * 1024))
    $((16 * 1024))
    $((32 * 1024))
    $((128 * 1024))
    # $((256 * 1024))
    # $((512 * 1024))
    # $((1024 * 1024))
)
RULER_TASK_IDS=(0 1 2)
RULER_CORPUS_PATH=../../data/dolma3_mix-6T-1025-partial-tokenized/
RULER_MAX_SAMPLES=50
RULER_PRINT_EVERY=1
RULER_SEGMENT_SIZE=-1
RULER_TP_SIZE=-1
RULER_TP_MIN_LEN=$((64 * 1024))
RULER_TP_GPU_IDS=(0 1 2 3 4 5 6 7)
RULER_TP_EXCLUSIVE=1
RULER_CHUNK_PREFILL_MIN_LEN=$((64 * 1024))
RULER_TP_SEGMENT_SIZE=16384

# Allow overriding TASK_IDS from command line args
if [ "$#" -gt 0 ]; then
    RULER_TASK_IDS=("$@")
fi

VOCAB_DIR=./configs/olmo3_vocab/

# ============================================================
#  Parse MODELS array into parallel arrays
# ============================================================
MODEL_NAMES=()
MODEL_CONFIGS=()
CKPT_PATHS=()
RULER_MAX_SEQ_LENS=()    # per-model ruler cap (0 = no limit)

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name config path ruler_cap <<< "$entry"
    MODEL_NAMES+=("$name")
    MODEL_CONFIGS+=("$config")
    CKPT_PATHS+=("$path")
    RULER_MAX_SEQ_LENS+=("${ruler_cap:-0}")
done

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "MODELS must not be empty" >&2
    exit 1
fi

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "GPU_IDS must not be empty" >&2
    exit 1
fi

if [ "$PPL_TP_SIZE" -gt 1 ] && [ "${#PPL_TP_GPU_IDS[@]}" -lt "$PPL_TP_SIZE" ]; then
    echo "PPL_TP_GPU_IDS count (${#PPL_TP_GPU_IDS[@]}) is less than PPL_TP_SIZE=${PPL_TP_SIZE}" >&2
    exit 1
fi

if [ "$RULER_TP_SIZE" -gt 1 ] && [ "${#RULER_TP_GPU_IDS[@]}" -lt "$RULER_TP_SIZE" ]; then
    echo "RULER_TP_GPU_IDS count (${#RULER_TP_GPU_IDS[@]}) is less than RULER_TP_SIZE=${RULER_TP_SIZE}" >&2
    exit 1
fi

# ============================================================
#  Logging
# ============================================================
LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_${EVAL_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Eval mode: ${EVAL_MODE}"
echo "Logs will be saved to: $LOG_DIR"
if [ "$PPL_TP_SIZE" -gt 1 ] && [ "${PPL_TP_EXCLUSIVE:-0}" = "1" ] && { [ "$EVAL_MODE" = "ppl" ] || [ "$EVAL_MODE" = "all" ]; }; then
    echo "PPL mode: sequential TP${PPL_TP_SIZE} on GPUs ${PPL_TP_GPU_IDS[*]} (${#MODEL_NAMES[@]} model(s), no queue)"
elif [ "$RULER_TP_SIZE" -gt 1 ] && [ "${RULER_TP_EXCLUSIVE:-0}" = "1" ] && { [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ]; }; then
    echo "RULER mode: sequential TP${RULER_TP_SIZE} on GPUs ${RULER_TP_GPU_IDS[*]} (${#MODEL_NAMES[@]} model(s), no queue)"
else
    echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"
fi

# ============================================================
#  Helper: read model config flags
# ============================================================
parse_model_config() {
    local model_config="$1"
    read -r insert_lmk adjust_lmk_pos chunk_size < <("$PYTHON_BIN" - "$model_config" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fin:
    config = json.load(fin)

insert_lmk = bool(config.get("insert_landmarks", False) or config.get("adjust_lmk_pos", False))
adjust_lmk_pos = bool(config.get("adjust_lmk_pos", False))
chunk_size = int(config.get("chunk_size", 64))
print(f'{"1" if insert_lmk else "0"} {"1" if adjust_lmk_pos else "0"} {chunk_size}')
PY
)
    echo "$insert_lmk $adjust_lmk_pos $chunk_size"
}

build_eval_args() {
    local insert_lmk="$1"
    local adjust_lmk_pos="$2"
    local args=""
    if [ "$insert_lmk" = "1" ]; then
        args+=" --insert_lmk"
    fi
    if [ "$adjust_lmk_pos" = "1" ]; then
        args+=" --adjust_lmk_pos"
    fi
    echo "$args"
}

# ============================================================
#  PPL evaluation for one model
# ============================================================
run_ppl_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local model_config="$3"
    local ckpt_path="$4"
    local run_log="$LOG_DIR/${model_name}.ppl.run.log"
    local summary_log="$LOG_DIR/${model_name}.ppl.summary.log"
    local had_failures=0

    local insert_lmk adjust_lmk_pos chunk_size
    read -r insert_lmk adjust_lmk_pos chunk_size < <(parse_model_config "$model_config")
    local eval_args=()
    [ "$insert_lmk" = "1" ] && eval_args+=(--insert_lmk)
    [ "$adjust_lmk_pos" = "1" ] && eval_args+=(--adjust_lmk_pos)

    : > "$run_log"
    : > "$summary_log"

    {
        echo "[$(date '+%F %T')] [PPL] Start model=${model_name}, gpu=${gpu_id}"
        echo "[$(date '+%F %T')] Eval args: ${eval_args[*]:-<none>}"
        echo "[$(date '+%F %T')] Chunk size: ${chunk_size}"
    } | tee -a "$run_log"

    for max_seq_len in "${PPL_SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} max_seq_len=${max_seq_len}: chunk_size=${chunk_size}" | tee -a "$run_log"
            continue
        fi

        local segment_args=()
        local tp_args=()
        local cuda_devices="$gpu_id"
        local mode_label="full"
        local use_tp=0

        if [ "$PPL_TP_SIZE" -gt 1 ]; then
            if [ "${PPL_TP_EXCLUSIVE:-0}" = "1" ] || [ "$max_seq_len" -gt "$PPL_TP_MIN_LEN" ]; then
                use_tp=1
            fi
        fi

        if [ "$use_tp" -eq 1 ]; then
            tp_args+=(--tp_size "$PPL_TP_SIZE")
            cuda_devices=$(IFS=,; echo "${PPL_TP_GPU_IDS[*]}")
            mode_label="tp${PPL_TP_SIZE}"
        elif [ "$PPL_SEGMENT_SIZE" -gt 0 ] && [ "$max_seq_len" -gt "$PPL_CHUNK_PREFILL_MIN_LEN" ]; then
            segment_args+=(--enable_chunk_prefill --segment_size "$PPL_SEGMENT_SIZE")
            mode_label="chunk${PPL_SEGMENT_SIZE}"
        fi

        local last_k_tokens=$((max_seq_len - 1))
        if [ "$max_seq_len" -gt "$PPL_LAST_K_TOKENS" ]; then
            last_k_tokens="$PPL_LAST_K_TOKENS"
        fi

        echo "[$(date '+%F %T')] GPU ${cuda_devices} | ${model_name} | PPL | max_seq_len=${max_seq_len} last_k_tokens=${last_k_tokens} mode=${mode_label}" | tee -a "$run_log"

        if ! CUDA_VISIBLE_DEVICES="$cuda_devices" "$PYTHON_BIN" eval/eval_ppl.py \
            --config_path "$model_config" \
            --vocab_dir "$VOCAB_DIR" \
            --checkpoint_path "$ckpt_path" \
            --data_path "$PPL_DATA_PATH" \
            --max_seq_len "$max_seq_len" \
            --max_samples "$PPL_MAX_SAMPLES" \
            --last_k_tokens "$last_k_tokens" \
            "${tp_args[@]}" \
            "${segment_args[@]}" \
            "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[$(date '+%F %T')] ERROR ${model_name} PPL max_seq_len=${max_seq_len} failed" | tee -a "$run_log"
            continue
        fi
    done

    echo "[$(date '+%F %T')] [PPL] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"
    [ "$had_failures" -ne 0 ] && return 1
    return 0
}

# ============================================================
#  RULER evaluation for one (model, task_id) combo
# ============================================================
run_ruler_eval() {
    local gpu_id="$1"
    local task_id="$2"
    local model_name="$3"
    local model_config="$4"
    local ckpt_path="$5"
    local ruler_cap="${6:-0}"
    local log_prefix="${model_name}.ruler.task${task_id}"
    local run_log="$LOG_DIR/${log_prefix}.run.log"
    local summary_log="$LOG_DIR/${log_prefix}.summary.log"
    local had_failures=0

    local insert_lmk adjust_lmk_pos chunk_size
    read -r insert_lmk adjust_lmk_pos chunk_size < <(parse_model_config "$model_config")
    local eval_args=()
    [ "$insert_lmk" = "1" ] && eval_args+=(--insert_lmk)
    [ "$adjust_lmk_pos" = "1" ] && eval_args+=(--adjust_lmk_pos)
    [ "${RULER_VERBOSE:-0}" = "1" ] && eval_args+=(--verbose)

    : > "$run_log"
    : > "$summary_log"

    {
        echo "[$(date '+%F %T')] [RULER] Start model=${model_name}, gpu=${gpu_id}, task_id=${task_id}"
        echo "[$(date '+%F %T')] Eval args: ${eval_args[*]:-<none>}"
        echo "[$(date '+%F %T')] Chunk size: ${chunk_size}"
    } | tee -a "$run_log"

    for max_seq_len in "${RULER_SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} task_id=${task_id} max_seq_len=${max_seq_len}: chunk_size=${chunk_size}" | tee -a "$run_log"
            continue
        fi

        if [ "$ruler_cap" -gt 0 ] && [ "$max_seq_len" -gt "$ruler_cap" ]; then
            echo "[$(date '+%F %T')] Skip ${model_name} task_id=${task_id} max_seq_len=${max_seq_len}: exceeds ruler_max_seq_len=${ruler_cap}" | tee -a "$run_log"
            continue
        fi

        local tp_args=()
        local cuda_devices="$gpu_id"
        local mode_label="full"
        local use_tp=0
        local segment_size_arg="$RULER_SEGMENT_SIZE"

        if [ "$RULER_TP_SIZE" -gt 1 ]; then
            if [ "${RULER_TP_EXCLUSIVE:-0}" = "1" ] || [ "$max_seq_len" -gt "$RULER_TP_MIN_LEN" ]; then
                use_tp=1
            fi
        fi

        if [ "$use_tp" -eq 1 ]; then
            tp_args+=(--tp_size "$RULER_TP_SIZE")
            cuda_devices=$(IFS=,; echo "${RULER_TP_GPU_IDS[*]}")
            # Under PP (device_map=auto) prefer chunk-prefill: full forward over
            # 100K+ tokens has a length-dependent HSA retrieval bug, while the
            # segmented KV-cache path is correct (and the per-layer cache shards
            # across the PP GPUs). Set RULER_TP_SEGMENT_SIZE<=0 to force the old
            # (broken at long seq) full-forward behavior.
            if [ "${RULER_TP_SEGMENT_SIZE:-0}" -gt 0 ]; then
                segment_size_arg="$RULER_TP_SEGMENT_SIZE"
                mode_label="tp${RULER_TP_SIZE}+chunk${RULER_TP_SEGMENT_SIZE}"
            else
                segment_size_arg=-1
                mode_label="tp${RULER_TP_SIZE}-full"
            fi
        elif [ "$RULER_SEGMENT_SIZE" -gt 0 ] && [ "$max_seq_len" -gt "$RULER_CHUNK_PREFILL_MIN_LEN" ]; then
            segment_size_arg="$RULER_SEGMENT_SIZE"
            mode_label="chunk${RULER_SEGMENT_SIZE}"
        fi

        echo "[$(date '+%F %T')] GPU ${cuda_devices} | ${model_name} | RULER task=${task_id} | max_seq_len=${max_seq_len} mode=${mode_label}" | tee -a "$run_log"

        if ! PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" HSA_DEBUG_NAN="${RULER_HSA_DEBUG_NAN:-0}" CUDA_VISIBLE_DEVICES="$cuda_devices" "$PYTHON_BIN" eval/eval_ruler_hf.py \
            --config_path "$model_config" \
            --vocab_dir "$VOCAB_DIR" \
            --corpus_path "$RULER_CORPUS_PATH" \
            --checkpoint_path "$ckpt_path" \
            --task_id "$task_id" \
            --segment_size "$segment_size_arg" \
            --max_seq_len "$max_seq_len" \
            --max_samples "$RULER_MAX_SAMPLES" \
            --print_every "$RULER_PRINT_EVERY" \
            "${tp_args[@]}" \
            "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[$(date '+%F %T')] ERROR ${model_name} RULER task=${task_id} max_seq_len=${max_seq_len} failed" | tee -a "$run_log"
            continue
        fi
    done

    echo "[$(date '+%F %T')] [RULER] Finished model=${model_name}, gpu=${gpu_id}, task_id=${task_id}" | tee -a "$run_log"
    [ "$had_failures" -ne 0 ] && return 1
    return 0
}

# ============================================================
#  GPU scheduler
# ============================================================
failed=0
available_gpus=("${GPU_IDS[@]}")
active_pids=()
active_gpus=()
active_jobs=()
REAPED_GPU_ID=""

pop_gpu() {
    REAPED_GPU_ID="${available_gpus[0]}"
    available_gpus=("${available_gpus[@]:1}")
}

reap_one_job() {
    local finished_pid
    local wait_status=0
    local active_idx

    wait -n -p finished_pid "${active_pids[@]}" || wait_status=$?
    if [ "$wait_status" -ne 0 ]; then
        failed=1
    fi

    for active_idx in "${!active_pids[@]}"; do
        if [ "${active_pids[$active_idx]}" = "$finished_pid" ]; then
            local finished_gpu="${active_gpus[$active_idx]}"
            local finished_job="${active_jobs[$active_idx]}"
            available_gpus+=("$finished_gpu")
            echo "[$(date '+%F %T')] Slot released: gpu=${finished_gpu}, job=${finished_job}, status=${wait_status}" | tee -a "$LOG_DIR/queue.log"

            unset 'active_pids[active_idx]'
            unset 'active_gpus[active_idx]'
            unset 'active_jobs[active_idx]'
            active_pids=("${active_pids[@]}")
            active_gpus=("${active_gpus[@]}")
            active_jobs=("${active_jobs[@]}")
            REAPED_GPU_ID="$finished_gpu"
            return
        fi
    done
}

dispatch_job() {
    # dispatch_job <job_name> <func> <arg1> <arg2> ...
    # The first arg to <func> will be gpu_id (auto-assigned).
    local job_name="$1"
    shift

    if [ "${#available_gpus[@]}" -eq 0 ]; then
        reap_one_job
    fi

    pop_gpu
    local gpu_id="$REAPED_GPU_ID"
    local func="$1"
    shift

    echo "[$(date '+%F %T')] Launch job=${job_name} on gpu=${gpu_id}" | tee -a "$LOG_DIR/queue.log"

    # Each background subshell sets its own trap so that SIGINT/SIGTERM
    # aborts the inner for-loop instead of falling through to `continue`.
    ( trap 'kill_descendants TERM $BASHPID; exit 130' INT TERM
      "$func" "$gpu_id" "$@" ) &

    active_pids+=($!)
    active_gpus+=("$gpu_id")
    active_jobs+=("$job_name")
}

# ============================================================
#  Dispatch jobs
# ============================================================

# --- PPL ---
if [ "$EVAL_MODE" = "ppl" ] || [ "$EVAL_MODE" = "all" ]; then
    if [ "$PPL_TP_SIZE" -gt 1 ] && [ "${PPL_TP_EXCLUSIVE:-0}" = "1" ]; then
        echo "--- Running PPL sequentially (TP${PPL_TP_SIZE}, one model at a time) ---"
        for idx in "${!MODEL_NAMES[@]}"; do
            model_name="${MODEL_NAMES[$idx]}"
            echo "[$(date '+%F %T')] Start PPL model=${model_name} (TP${PPL_TP_SIZE})" | tee -a "$LOG_DIR/queue.log"
            if ! run_ppl_eval "tp${PPL_TP_SIZE}" \
                "$model_name" \
                "${MODEL_CONFIGS[$idx]}" \
                "${CKPT_PATHS[$idx]}"; then
                failed=1
            fi
            echo "[$(date '+%F %T')] Finished PPL model=${model_name} (TP${PPL_TP_SIZE})" | tee -a "$LOG_DIR/queue.log"
        done
    else
        echo "--- Queueing PPL jobs ---"
        for idx in "${!MODEL_NAMES[@]}"; do
            model_name="${MODEL_NAMES[$idx]}"
            dispatch_job "${model_name}.ppl" \
                run_ppl_eval \
                "$model_name" \
                "${MODEL_CONFIGS[$idx]}" \
                "${CKPT_PATHS[$idx]}"
        done
    fi
fi

# --- RULER ---
if [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ]; then
    if [ "$RULER_TP_SIZE" -gt 1 ] && [ "${RULER_TP_EXCLUSIVE:-0}" = "1" ]; then
        echo "--- Running RULER sequentially (TP${RULER_TP_SIZE}, one job at a time) ---"
        for task_id in "${RULER_TASK_IDS[@]}"; do
            for idx in "${!MODEL_NAMES[@]}"; do
                model_name="${MODEL_NAMES[$idx]}"
                echo "[$(date '+%F %T')] Start RULER model=${model_name} task=${task_id} (TP${RULER_TP_SIZE})" | tee -a "$LOG_DIR/queue.log"
                if ! run_ruler_eval "tp${RULER_TP_SIZE}" \
                    "$task_id" \
                    "$model_name" \
                    "${MODEL_CONFIGS[$idx]}" \
                    "${CKPT_PATHS[$idx]}" \
                    "${RULER_MAX_SEQ_LENS[$idx]}"; then
                    failed=1
                fi
                echo "[$(date '+%F %T')] Finished RULER model=${model_name} task=${task_id} (TP${RULER_TP_SIZE})" | tee -a "$LOG_DIR/queue.log"
            done
        done
    else
        echo "--- Queueing RULER jobs ---"
        for task_id in "${RULER_TASK_IDS[@]}"; do
            for idx in "${!MODEL_NAMES[@]}"; do
                model_name="${MODEL_NAMES[$idx]}"
                dispatch_job "${model_name}.ruler.task${task_id}" \
                    run_ruler_eval \
                    "$task_id" \
                    "$model_name" \
                    "${MODEL_CONFIGS[$idx]}" \
                    "${CKPT_PATHS[$idx]}" \
                    "${RULER_MAX_SEQ_LENS[$idx]}"
            done
        done
    fi
fi

# --- Wait for remaining ---
while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

# ============================================================
#  Aggregate summary logs
# ============================================================
if compgen -G "$LOG_DIR/*.summary.log" > /dev/null; then
    cat "$LOG_DIR"/*.summary.log > "$LOG_DIR/all_models.summary.log"
fi

# Generate LaTeX summary tables (PPL + RULER)
"$PYTHON_BIN" "$SCRIPT_DIR/generate_latex_summary.py" "$LOG_DIR" \
    -o "$LOG_DIR/summary.log" || echo "Warning: LaTeX summary generation failed" >&2

{
    echo "eval_mode=${EVAL_MODE}"
    echo "log_dir=${LOG_DIR}"
    echo "models=${MODEL_NAMES[*]}"
    [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ] && printf 'ruler_task_ids=%s\n' "${RULER_TASK_IDS[*]}"
} > "$LOG_DIR/index.log"

echo ""
echo "All logs saved to: $LOG_DIR"

if [ "$failed" -ne 0 ]; then
    echo "Some evaluation jobs failed." >&2
    exit 1
fi
