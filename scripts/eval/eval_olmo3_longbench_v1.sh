#!/usr/bin/env bash
# ============================================================
# LongBench v1 evaluation (generate + score with official metrics)
#
# Usage:
#   bash scripts/eval/eval_olmo3_longbench_v1.sh
# ============================================================

if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi

set -euo pipefail

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
#  Models: "name|config|ckpt_path"
#  config can be empty for standard HF models
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
#  LongBench v1 settings
# ============================================================
MAX_LENGTH=65536                           # middle truncation limit
VOCAB_DIR=./configs/olmo3_vocab/           # tokenizer for HSA models
DATASETS=""                                # empty = all 21 tasks; or comma-separated, e.g. "hotpotqa,qasper"
SAVE_ROOT="$SCRIPT_DIR/logs/eval_longbench_v1_$(date +%Y%m%d_%H%M%S)"

# ============================================================
#  Parse MODELS
# ============================================================
MODEL_NAMES=()
MODEL_CONFIGS=()
CKPT_PATHS=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r name config path <<< "$entry"
    MODEL_NAMES+=("$name")
    MODEL_CONFIGS+=("${config}")
    CKPT_PATHS+=("$path")
done

if [ "${#MODEL_NAMES[@]}" -eq 0 ]; then
    echo "MODELS must not be empty" >&2
    exit 1
fi

# ============================================================
#  Logging
# ============================================================
mkdir -p "$SAVE_ROOT"
echo "Logs / results will be saved to: $SAVE_ROOT"
echo "Queue size: ${#GPU_IDS[@]} GPU(s), ${#MODEL_NAMES[@]} model(s)"

# ============================================================
#  Eval function for one model
# ============================================================
run_longbench_v1_eval() {
    local gpu_id="$1"
    local model_name="$2"
    local model_config="$3"
    local ckpt_path="$4"
    local save_dir="$SAVE_ROOT/${model_name}"
    local run_log="$SAVE_ROOT/${model_name}.run.log"

    mkdir -p "$save_dir"
    : > "$run_log"

    echo "[$(date '+%F %T')] [LongBench-v1] Start model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"

    local extra_args=()

    if [ -n "$model_config" ]; then
        extra_args+=(--config_path "$model_config")
        extra_args+=(--vocab_dir "$VOCAB_DIR")
    fi

    if [ -n "$DATASETS" ]; then
        extra_args+=(--datasets "$DATASETS")
    fi

    if ! CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" eval/eval_longbench_v1.py \
        --checkpoint_path "$ckpt_path" \
        --save_dir "$save_dir" \
        --max_length "$MAX_LENGTH" \
        --n_proc 1 \
        "${extra_args[@]}" \
        2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] [ERROR] ${model_name} failed" | tee -a "$run_log"
        return 1
    fi

    echo "[$(date '+%F %T')] [LongBench-v1] Finished model=${model_name}, gpu=${gpu_id}" | tee -a "$run_log"
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

    wait -n -p finished_pid "${active_pids[@]}" || wait_status=$?
    if [ "$wait_status" -ne 0 ]; then
        failed=1
    fi

    for active_idx in "${!active_pids[@]}"; do
        if [ "${active_pids[$active_idx]}" = "$finished_pid" ]; then
            local finished_gpu="${active_gpus[$active_idx]}"
            local finished_job="${active_jobs[$active_idx]}"
            available_gpus+=("$finished_gpu")
            echo "[$(date '+%F %T')] Slot released: gpu=${finished_gpu}, job=${finished_job}, status=${wait_status}" | tee -a "$SAVE_ROOT/queue.log"

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
    local job_name="$1"
    shift

    if [ "${#available_gpus[@]}" -eq 0 ]; then
        reap_one_job
    fi

    pop_gpu
    local gpu_id="$REAPED_GPU_ID"
    local func="$1"
    shift

    echo "[$(date '+%F %T')] Launch job=${job_name} on gpu=${gpu_id}" | tee -a "$SAVE_ROOT/queue.log"

    "$func" "$gpu_id" "$@" &

    active_pids+=($!)
    active_gpus+=("$gpu_id")
    active_jobs+=("$job_name")
}

# ============================================================
#  Dispatch
# ============================================================
echo "--- Queueing LongBench v1 jobs ---"
for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    dispatch_job "${model_name}.longbench_v1" \
        run_longbench_v1_eval \
        "$model_name" \
        "${MODEL_CONFIGS[$idx]}" \
        "${CKPT_PATHS[$idx]}"
done

# Wait for all
while [ "${#active_pids[@]}" -gt 0 ]; do
    reap_one_job
done

echo ""
echo "All results saved to: $SAVE_ROOT"

if [ "$failed" -ne 0 ]; then
    echo "Some evaluation jobs failed." >&2
    exit 1
fi
