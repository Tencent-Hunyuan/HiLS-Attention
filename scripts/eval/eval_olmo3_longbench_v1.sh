#!/usr/bin/env bash
# LongBench v1 evaluation (generate + score).
# Usage: bash scripts/eval/eval_olmo3_longbench_v1.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ensure_bash
setup_eval_env

# Models: "name|config|ckpt_path" (config empty for standard HF models)
MODELS=(
    "lmk_token_tuning|configs/olmo3_7B/olmo3_8KA2K_lmk_token_tuning.json|../../checkpoints/olmo3_8KA2K_lmk_token_tuning/global_step_1192/hf_ckpt"
    "olmo3_8KA2K_HoPE_qcal_step13000|configs/olmo3_7B/olmo3_8KA2K_HoPE_qcal.json|../../checkpoints/olmo3_8KA2K_HoPE_qcal/global_step_13000/hf_ckpt"
)

GPU_IDS=(0 1 2 3 4 5 6 7)
MAX_LENGTH=65536
VOCAB_DIR=./configs/olmo3_vocab/
DATASETS=""   # empty = all tasks; or comma-separated e.g. "hotpotqa,qasper"
SAVE_ROOT="$SCRIPT_DIR/logs/eval_longbench_v1_$(date +%Y%m%d_%H%M%S)"

MODEL_NAMES=() MODEL_CONFIGS=() CKPT_PATHS=()
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name config path <<< "$entry"
    MODEL_NAMES+=("$name")
    MODEL_CONFIGS+=("$config")
    CKPT_PATHS+=("$path")
done
[ "${#MODEL_NAMES[@]}" -gt 0 ] || { echo "MODELS must not be empty" >&2; exit 1; }

mkdir -p "$SAVE_ROOT"
echo "Logs: $SAVE_ROOT (${#GPU_IDS[@]} GPUs, ${#MODEL_NAMES[@]} models)"

run_longbench_v1_eval() {
    local gpu_id="$1" model_name="$2" model_config="$3" ckpt_path="$4"
    local save_dir="$SAVE_ROOT/${model_name}" run_log="$SAVE_ROOT/${model_name}.run.log"
    local -a extra_args=()

    mkdir -p "$save_dir"
    : > "$run_log"
    echo "[$(date '+%F %T')] Start ${model_name} on gpu=${gpu_id}" | tee -a "$run_log"

    [ -n "$model_config" ] && extra_args+=(--config_path "$model_config" --vocab_dir "$VOCAB_DIR")
    [ -n "$DATASETS" ] && extra_args+=(--datasets "$DATASETS")

    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_longbench_v1.py \
        --checkpoint_path "$ckpt_path" \
        --save_dir "$save_dir" \
        --max_length "$MAX_LENGTH" \
        --n_proc 1 \
        "${extra_args[@]}" \
        2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] ERROR ${model_name} failed" | tee -a "$run_log"
        return 1
    fi
    echo "[$(date '+%F %T')] Finished ${model_name}" | tee -a "$run_log"
}

gpu_queue_init "$SAVE_ROOT/queue.log" "${GPU_IDS[@]}"
for idx in "${!MODEL_NAMES[@]}"; do
    gpu_queue_dispatch "${MODEL_NAMES[$idx]}.longbench_v1" run_longbench_v1_eval \
        "${MODEL_NAMES[$idx]}" "${MODEL_CONFIGS[$idx]}" "${CKPT_PATHS[$idx]}"
done
gpu_queue_wait_all

echo "All results saved to: $SAVE_ROOT"
[ "$GPU_QUEUE_FAILED" -eq 0 ] || { echo "Some jobs failed." >&2; exit 1; }
