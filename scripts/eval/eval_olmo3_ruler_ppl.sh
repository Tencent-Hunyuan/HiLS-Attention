#!/usr/bin/env bash
# PPL + RULER evaluation. Usage: bash scripts/eval/eval_olmo3_ruler_ppl.sh [ruler_task_ids...]
# EVAL_MODE=ppl|ruler|all (default: all)

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ensure_bash
setup_eval_env
GPU_QUEUE_USE_TRAP=1

# Ctrl+C: stop background eval loops
SHUTDOWN=0
kill_descendants() {
    local sig="${1:-TERM}" parent="${2:-$$}" kids pid
    kids=$(pgrep -P "$parent" 2>/dev/null || true)
    for pid in $kids; do
        kill_descendants "$sig" "$pid"
        kill -"$sig" "$pid" 2>/dev/null || true
    done
}
shutdown_handler() {
    local sig="${1:-INT}"
    [ "$SHUTDOWN" -ne 0 ] && return
    SHUTDOWN=1
    trap - INT TERM
    echo "[$(date '+%F %T')] SIG${sig}: stopping children..." >&2
    kill_descendants TERM "$$"
    for _ in 1 2 3 4 5; do pgrep -P "$$" >/dev/null 2>&1 || break; sleep 1; done
    pgrep -P "$$" >/dev/null 2>&1 && kill_descendants KILL "$$"
    exit 130
}
trap 'shutdown_handler INT' INT
trap 'shutdown_handler TERM' TERM

EVAL_MODE="${EVAL_MODE:-all}"

# Models: "name|config|ckpt_path|ruler_max_seq_len" (ruler cap optional)
MODELS=(
    "lmk_token_tuning|configs/olmo3_7B/olmo3_8KA2K_lmk_token_tuning.json|../../checkpoints/olmo3_8KA2K_lmk_token_tuning/global_step_1192/hf_ckpt"
    "olmo3_8KA2K_HoPE_LoRA_step13000|configs/olmo3_7B/olmo3_8KA2K_HoPE_LoRA.json|../../checkpoints/olmo3_8KA2K_HoPE_LoRA/global_step_13000/hf_ckpt"
)

GPU_IDS=(0 1 2 3 4 5 6 7)
VOCAB_DIR=./configs/olmo3_vocab/

PPL_SEQ_LEN_LIST=(64 128 512 $((8*1024)) $((16*1024)) $((64*1024)) $((128*1024)) $((256*1024)))
PPL_DATA_PATH=../../data/dolma3_mix-6T-1025-partial-tokenized
PPL_MAX_SAMPLES=100
PPL_LAST_K_TOKENS=512
PPL_TP_SIZE=-1
PPL_TP_MIN_LEN=$((64*1024))
PPL_TP_GPU_IDS=(0 1 2 3 4 5 6 7)
PPL_TP_EXCLUSIVE=0
PPL_SEGMENT_SIZE=-1
PPL_CHUNK_PREFILL_MIN_LEN=$((64*1024))

RULER_SEQ_LEN_LIST=($((8*1024)) $((16*1024)) $((32*1024)) $((128*1024)))
RULER_TASK_IDS=(0 1 2)
RULER_CORPUS_PATH=../../data/dolma3_mix-6T-1025-partial-tokenized/
RULER_MAX_SAMPLES=50
RULER_PRINT_EVERY=1
RULER_SEGMENT_SIZE=-1
RULER_TP_SIZE=-1
RULER_TP_MIN_LEN=$((64*1024))
RULER_TP_GPU_IDS=(0 1 2 3 4 5 6 7)
RULER_TP_EXCLUSIVE=1
RULER_CHUNK_PREFILL_MIN_LEN=$((64*1024))
RULER_TP_SEGMENT_SIZE=16384
[ "$#" -gt 0 ] && RULER_TASK_IDS=("$@")

MODEL_NAMES=() MODEL_CONFIGS=() CKPT_PATHS=() RULER_MAX_SEQ_LENS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name config path ruler_cap <<< "$entry"
    MODEL_NAMES+=("$name")
    MODEL_CONFIGS+=("$config")
    CKPT_PATHS+=("$path")
    RULER_MAX_SEQ_LENS+=("${ruler_cap:-0}")
done
[ "${#MODEL_NAMES[@]}" -gt 0 ] || { echo "MODELS must not be empty" >&2; exit 1; }
[ "${#GPU_IDS[@]}" -gt 0 ] || { echo "GPU_IDS must not be empty" >&2; exit 1; }
[ "$PPL_TP_SIZE" -le 1 ] || [ "${#PPL_TP_GPU_IDS[@]}" -ge "$PPL_TP_SIZE" ] || {
    echo "PPL_TP_GPU_IDS too small for PPL_TP_SIZE=${PPL_TP_SIZE}" >&2; exit 1; }
[ "$RULER_TP_SIZE" -le 1 ] || [ "${#RULER_TP_GPU_IDS[@]}" -ge "$RULER_TP_SIZE" ] || {
    echo "RULER_TP_GPU_IDS too small for RULER_TP_SIZE=${RULER_TP_SIZE}" >&2; exit 1; }

LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_${EVAL_MODE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Mode: $EVAL_MODE | Logs: $LOG_DIR"

parse_model_config() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    c = json.load(f)
insert = bool(c.get("insert_landmarks") or c.get("adjust_lmk_pos"))
adjust = bool(c.get("adjust_lmk_pos", False))
chunk = int(c.get("chunk_size", 64))
print(f"{'1' if insert else '0'} {'1' if adjust else '0'} {chunk}")
PY
}

run_ppl_eval() {
    local gpu_id="$1" model_name="$2" model_config="$3" ckpt_path="$4"
    local run_log="$LOG_DIR/${model_name}.ppl.run.log" summary_log="$LOG_DIR/${model_name}.ppl.summary.log"
    local insert_lmk adjust_lmk_pos chunk_size had_failures=0
    read -r insert_lmk adjust_lmk_pos chunk_size < <(parse_model_config "$model_config")
    local -a eval_args=()
    [ "$insert_lmk" = "1" ] && eval_args+=(--insert_lmk)
    [ "$adjust_lmk_pos" = "1" ] && eval_args+=(--adjust_lmk_pos)
    : > "$run_log"; : > "$summary_log"

    for max_seq_len in "${PPL_SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then
            echo "[skip] ${model_name} len=${max_seq_len} <= chunk=${chunk_size}" | tee -a "$run_log"
            continue
        fi

        local -a tp_args=() segment_args=()
        local cuda_devices="$gpu_id" mode_label="full" use_tp=0
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
        [ "$max_seq_len" -gt "$PPL_LAST_K_TOKENS" ] && last_k_tokens="$PPL_LAST_K_TOKENS"
        echo "[$(date '+%F %T')] ${model_name} PPL len=${max_seq_len} mode=${mode_label}" | tee -a "$run_log"

        if ! CUDA_VISIBLE_DEVICES="$cuda_devices" "$PYTHON_BIN" eval/eval_ppl.py \
            --config_path "$model_config" --vocab_dir "$VOCAB_DIR" --checkpoint_path "$ckpt_path" \
            --data_path "$PPL_DATA_PATH" --max_seq_len "$max_seq_len" --max_samples "$PPL_MAX_SAMPLES" \
            --last_k_tokens "$last_k_tokens" "${tp_args[@]}" "${segment_args[@]}" "${eval_args[@]}" \
            --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[ERROR] ${model_name} PPL len=${max_seq_len}" | tee -a "$run_log"
        fi
    done
    [ "$had_failures" -eq 0 ]
}

run_ruler_eval() {
    local gpu_id="$1" task_id="$2" model_name="$3" model_config="$4" ckpt_path="$5" ruler_cap="${6:-0}"
    local prefix="${model_name}.ruler.task${task_id}"
    local run_log="$LOG_DIR/${prefix}.run.log" summary_log="$LOG_DIR/${prefix}.summary.log"
    local insert_lmk adjust_lmk_pos chunk_size had_failures=0
    read -r insert_lmk adjust_lmk_pos chunk_size < <(parse_model_config "$model_config")
    local -a eval_args=()
    [ "$insert_lmk" = "1" ] && eval_args+=(--insert_lmk)
    [ "$adjust_lmk_pos" = "1" ] && eval_args+=(--adjust_lmk_pos)
    [ "${RULER_VERBOSE:-0}" = "1" ] && eval_args+=(--verbose)
    : > "$run_log"; : > "$summary_log"

    for max_seq_len in "${RULER_SEQ_LEN_LIST[@]}"; do
        if [ "$insert_lmk" = "1" ] && [ "$max_seq_len" -le "$chunk_size" ]; then continue; fi
        if [ "$ruler_cap" -gt 0 ] && [ "$max_seq_len" -gt "$ruler_cap" ]; then continue; fi

        local -a tp_args=()
        local cuda_devices="$gpu_id" mode_label="full" use_tp=0 segment_size_arg="$RULER_SEGMENT_SIZE"
        if [ "$RULER_TP_SIZE" -gt 1 ]; then
            if [ "${RULER_TP_EXCLUSIVE:-0}" = "1" ] || [ "$max_seq_len" -gt "$RULER_TP_MIN_LEN" ]; then
                use_tp=1
            fi
        fi
        if [ "$use_tp" -eq 1 ]; then
            tp_args+=(--tp_size "$RULER_TP_SIZE")
            cuda_devices=$(IFS=,; echo "${RULER_TP_GPU_IDS[*]}")
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

        echo "[$(date '+%F %T')] ${model_name} RULER task=${task_id} len=${max_seq_len} mode=${mode_label}" | tee -a "$run_log"
        if ! PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
            HILS_DEBUG_NAN="${RULER_HILS_DEBUG_NAN:-0}" \
            CUDA_VISIBLE_DEVICES="$cuda_devices" "$PYTHON_BIN" eval/eval_ruler_hf.py \
            --config_path "$model_config" --vocab_dir "$VOCAB_DIR" --corpus_path "$RULER_CORPUS_PATH" \
            --checkpoint_path "$ckpt_path" --task_id "$task_id" --segment_size "$segment_size_arg" \
            --max_seq_len "$max_seq_len" --max_samples "$RULER_MAX_SAMPLES" --print_every "$RULER_PRINT_EVERY" \
            "${tp_args[@]}" "${eval_args[@]}" --summary_log "$summary_log" 2>&1 | tee -a "$run_log"; then
            had_failures=1
            echo "[ERROR] ${model_name} RULER task=${task_id} len=${max_seq_len}" | tee -a "$run_log"
        fi
    done
    [ "$had_failures" -eq 0 ]
}

failed=0
gpu_queue_init "$LOG_DIR/queue.log" "${GPU_IDS[@]}"

if [ "$EVAL_MODE" = "ppl" ] || [ "$EVAL_MODE" = "all" ]; then
    if [ "$PPL_TP_SIZE" -gt 1 ] && [ "${PPL_TP_EXCLUSIVE:-0}" = "1" ]; then
        for idx in "${!MODEL_NAMES[@]}"; do
            run_ppl_eval "tp${PPL_TP_SIZE}" "${MODEL_NAMES[$idx]}" "${MODEL_CONFIGS[$idx]}" "${CKPT_PATHS[$idx]}" || failed=1
        done
    else
        for idx in "${!MODEL_NAMES[@]}"; do
            gpu_queue_dispatch "${MODEL_NAMES[$idx]}.ppl" run_ppl_eval \
                "${MODEL_NAMES[$idx]}" "${MODEL_CONFIGS[$idx]}" "${CKPT_PATHS[$idx]}"
        done
    fi
fi

if [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ]; then
    if [ "$RULER_TP_SIZE" -gt 1 ] && [ "${RULER_TP_EXCLUSIVE:-0}" = "1" ]; then
        for task_id in "${RULER_TASK_IDS[@]}"; do
            for idx in "${!MODEL_NAMES[@]}"; do
                run_ruler_eval "tp${RULER_TP_SIZE}" "$task_id" "${MODEL_NAMES[$idx]}" \
                    "${MODEL_CONFIGS[$idx]}" "${CKPT_PATHS[$idx]}" "${RULER_MAX_SEQ_LENS[$idx]}" || failed=1
            done
        done
    else
        for task_id in "${RULER_TASK_IDS[@]}"; do
            for idx in "${!MODEL_NAMES[@]}"; do
                gpu_queue_dispatch "${MODEL_NAMES[$idx]}.ruler.task${task_id}" run_ruler_eval \
                    "$task_id" "${MODEL_NAMES[$idx]}" "${MODEL_CONFIGS[$idx]}" \
                    "${CKPT_PATHS[$idx]}" "${RULER_MAX_SEQ_LENS[$idx]}"
            done
        done
    fi
fi

gpu_queue_wait_all
[ "$GPU_QUEUE_FAILED" -ne 0 ] && failed=1

if compgen -G "$LOG_DIR/*.summary.log" > /dev/null; then
    cat "$LOG_DIR"/*.summary.log > "$LOG_DIR/all_models.summary.log"
fi
"$PYTHON_BIN" "$SCRIPT_DIR/generate_latex_summary.py" "$LOG_DIR" -o "$LOG_DIR/summary.log" \
    || echo "Warning: LaTeX summary generation failed" >&2

{
    echo "eval_mode=${EVAL_MODE}"
    echo "log_dir=${LOG_DIR}"
    echo "models=${MODEL_NAMES[*]}"
    [ "$EVAL_MODE" = "ruler" ] || [ "$EVAL_MODE" = "all" ] && printf 'ruler_task_ids=%s\n' "${RULER_TASK_IDS[*]}"
} > "$LOG_DIR/index.log"

echo "All logs saved to: $LOG_DIR"
[ "$failed" -eq 0 ] || { echo "Some jobs failed." >&2; exit 1; }
