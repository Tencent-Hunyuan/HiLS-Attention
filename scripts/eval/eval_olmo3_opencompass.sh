#!/usr/bin/env bash
# Transformers OpenCompass evaluation with GPU queue.
# Usage: bash scripts/eval/eval_olmo3_opencompass.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

ensure_bash
setup_eval_env

TOKENIZER_PATH=${TOKENIZER_PATH:-configs/olmo3_vocab}
OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-/apdcephfs_tj5/share_300719894/user/guhao/opencompass}
# OPENCOMPASS_PATH=${OPENCOMPASS_PATH:-}
export PYTHONPATH="${REPO_ROOT}:${OPENCOMPASS_PATH}${PYTHONPATH:+:$PYTHONPATH}"

MODEL_NAMES=(
            # lmk_token_tuning
            olmo3_8KA2K_HoPE_qcal
)
HILS_CONFIGS=(
    # configs/olmo3_7B/olmo3_8KA2K_lmk_token_tuning.json
    configs/olmo3_7B/olmo3_8KA2K_HoPE_qcal.json
)
HF_PATHS=(
    # ../../checkpoints/olmo3_8KA2K_lmk_token_tuning/global_step_1192/hf_ckpt
    # ../../checkpoints/olmo3_8KA2K_HoPE_LoRA/global_step_13000/hf_ckpt
    /apdcephfs_tj5/share_300719894/user/guhao/checkpoints/olmo3_8KA2K_lmk_embed_HoPE_stage2/checkpoints/global_step_13000/hf_ckpt
)

DATASET_LIST=(
    gpqa_few_shot_ppl_4b5a83 
    mmlu_ppl_ac766d hellaswag_10shot_ppl_59c85e
    ARC_c_few_shot_ppl SuperGLUE_BoolQ_few_shot_ppl race_few_shot_ppl
    gsm8k_gen cmath_gen humaneval_plus_gen mbpp_plus_gen cruxeval_o_gen
)

GPU_IDS=(0 1 2 3 4 5 6 7)
GPUS_PER_TASK=${GPUS_PER_TASK:-1}
DEBUG=${DEBUG:-1}
LOCAL_STAGE_DIR=${LOCAL_STAGE_DIR:-}
WARMUP_PAGE_CACHE=${WARMUP_PAGE_CACHE:-}

PREPARE_PY="$SCRIPT_DIR/lib/prepare_hf_checkpoint.py"
APPEND_AVG_PY="$SCRIPT_DIR/lib/append_oc_summary_avg.py"
LATEX_PY="$SCRIPT_DIR/lib/generate_opencompass_latex.py"

[ "${#MODEL_NAMES[@]}" -eq "${#HILS_CONFIGS[@]}" ] && \
[ "${#MODEL_NAMES[@]}" -eq "${#HF_PATHS[@]}" ] || { echo "Model array length mismatch" >&2; exit 1; }
[ "${#GPU_IDS[@]}" -gt 0 ] || { echo "GPU_IDS must not be empty" >&2; exit 1; }
[ "$GPUS_PER_TASK" -ge 1 ] || { echo "GPUS_PER_TASK must be >= 1" >&2; exit 1; }
[ "$GPUS_PER_TASK" -le "${#GPU_IDS[@]}" ] || GPUS_PER_TASK="${#GPU_IDS[@]}"
[ -d "$TOKENIZER_PATH" ] || { echo "TOKENIZER_PATH not found: $TOKENIZER_PATH" >&2; exit 1; }
[ -d "$OPENCOMPASS_PATH/opencompass" ] || {
    echo "OPENCOMPASS_PATH not found: $OPENCOMPASS_PATH (run install_opencompass.sh)" >&2; exit 1; }

pkill -f "burner" >/dev/null 2>&1 || true

MASTER_LOG_DIR="$SCRIPT_DIR/logs/eval_olmo3_opencompass_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"
echo "Log dir: $MASTER_LOG_DIR"
echo "Models: ${MODEL_NAMES[*]}"
echo "GPUs: ${GPU_IDS[*]} (${GPUS_PER_TASK}/task)"

prepare_hf_path() {
    local hf_path="$1" hils_config="$2" resolved_hf_path="$3"
    rm -rf "$resolved_hf_path"
    mkdir -p "$resolved_hf_path"

    shopt -s nullglob dotglob
    for src in "$hf_path"/*; do
        case "$(basename "$src")" in
            config.json|config_*.json|generation_config.json|tokenizer.json|tokenizer_config.json|vocab.json|merges.txt|special_tokens_map.json) ;;
            *) ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")" ;;
        esac
    done
    for src in "$TOKENIZER_PATH"/*; do
        ln -sfn "$(readlink -f "$src")" "$resolved_hf_path/$(basename "$src")"
    done
    shopt -u nullglob dotglob

    "$PYTHON_BIN" "$PREPARE_PY" "$hf_path" "$hils_config" "$TOKENIZER_PATH" "$resolved_hf_path/config.json"
}

stage_model_to_local() {
    local model_name="$1" src_hf_path="$2"
    local staged_dir="$LOCAL_STAGE_DIR/$model_name"
    if [ -d "$staged_dir" ] && [ -f "$staged_dir/.stage_done" ]; then
        echo "[stage] Reuse $staged_dir"
        echo "$staged_dir"
        return
    fi
    mkdir -p "$staged_dir"
    echo "[stage] Copy $src_hf_path -> $staged_dir"
    cp -Lr --reflink=auto "$src_hf_path"/. "$staged_dir"/ 2>/dev/null || cp -Lr "$src_hf_path"/. "$staged_dir"/
    touch "$staged_dir/.stage_done"
    echo "$staged_dir"
}

warmup_weights() {
    local hf_path="$1" f
    shopt -s nullglob
    for f in "$hf_path"/*.safetensors "$hf_path"/*.bin; do
        dd if="$f" of=/dev/null bs=32M status=none 2>/dev/null || true
    done
    shopt -u nullglob
}

run_one() {
    local gpu_group="$1" model_name="$2" hils_config="$3" hf_path="$4" dataset="$5" work_dir="$6"
    local resolved_hf_path="$work_dir/hf_with_tokenizer" run_log="$work_dir/run.log"
    local -a gpu_list=() cmd=()
    IFS=',' read -r -a gpu_list <<< "$gpu_group"
    local num_gpus="${#gpu_list[@]}"

    prepare_hf_path "$hf_path" "$hils_config" "$resolved_hf_path" 2>&1 | tee -a "$run_log"

    cmd=(
        "$PYTHON_BIN" eval/eval_opencompass.py
        --datasets "$dataset"
        --hf-type base
        --hf-path "$resolved_hf_path"
        --hils-config "$hils_config"
        --batch-size 1
        -w "$work_dir"
    )

    local insert_lmk
    insert_lmk=$("$PYTHON_BIN" -c "import json,sys;c=json.load(open(sys.argv[1]));print('1' if c.get('insert_landmarks') or c.get('adjust_lmk_pos') else '0')" "$hils_config")
    cmd+=(--model-kwargs torch_dtype=torch.bfloat16 attn_implementation=flash_attention_3)
    [ "$insert_lmk" = "1" ] && cmd+=(auto_insert_lmk=True)
    [ "$DEBUG" = "1" ] && [ "$num_gpus" -eq 1 ] && cmd+=(--debug)

    {
        echo "[$(date '+%F %T')] Start ${model_name}/${dataset} gpus=${gpu_group}"
        echo "CUDA_VISIBLE_DEVICES=${gpu_group} ${cmd[*]}"
    } | tee -a "$run_log"

    if ! CUDA_VISIBLE_DEVICES="$gpu_group" "${cmd[@]}" 2>&1 | tee -a "$run_log"; then
        echo "[$(date '+%F %T')] ERROR ${model_name}/${dataset}" | tee -a "$run_log"
        return 1
    fi

    local summary
    summary=$(find "$work_dir" -maxdepth 4 -path '*/summary/summary_*.txt' | sort | tail -n 1 || true)
    if [ -n "$summary" ]; then
        "$PYTHON_BIN" "$APPEND_AVG_PY" "$summary"
        echo "[$(date '+%F %T')] Summary updated: $summary" | tee -a "$run_log"
    fi
}

gpu_group_queue_init "$MASTER_LOG_DIR/queue.log" "$GPUS_PER_TASK" "${GPU_IDS[@]}"

for model_idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$model_idx]}"
    hils_config="${HILS_CONFIGS[$model_idx]}"
    hf_path="${HF_PATHS[$model_idx]}"
    model_log_dir="$MASTER_LOG_DIR/$model_name"
    mkdir -p "$model_log_dir"

    effective_hf_path="$hf_path"
    if [ -n "$LOCAL_STAGE_DIR" ]; then
        mkdir -p "$LOCAL_STAGE_DIR"
        effective_hf_path=$(stage_model_to_local "$model_name" "$hf_path" | tail -n 1)
    fi
    [ "$WARMUP_PAGE_CACHE" = "1" ] && warmup_weights "$effective_hf_path"

    for dataset in "${DATASET_LIST[@]}"; do
        ds_tag=$(printf '%s' "$dataset" | tr ',/' '__' | tr -cd '[:alnum:]_.-')
        work_dir="$model_log_dir/$ds_tag"
        mkdir -p "$work_dir"
        gpu_group_queue_dispatch "${model_name}/${dataset}" run_one \
            "$model_name" "$hils_config" "$effective_hf_path" "$dataset" "$work_dir"
    done
done
gpu_group_queue_wait_all

RESCORE_PY="$REPO_ROOT/eval/rescore_evalplus.py"
if [ -f "$RESCORE_PY" ]; then
    for model_name in "${MODEL_NAMES[@]}"; do
        model_log_dir="$MASTER_LOG_DIR/$model_name"
        while IFS= read -r pred_file; do
            "$PYTHON_BIN" "$RESCORE_PY" "$pred_file" --dataset humaneval --tag "$model_name" \
                2>&1 | tee -a "$model_log_dir/rescore_humaneval.log" || true
        done < <(find "$model_log_dir" -path "*/humaneval_plus_gen/*/predictions/*/humaneval_plus*.json" 2>/dev/null | sort -u)
        while IFS= read -r pred_file; do
            "$PYTHON_BIN" "$RESCORE_PY" "$pred_file" --dataset mbpp --tag "$model_name" \
                2>&1 | tee -a "$model_log_dir/rescore_mbpp.log" || true
        done < <(find "$model_log_dir" -path "*/mbpp_plus_gen/*/predictions/*/mbpp_plus*.json" 2>/dev/null | sort -u)
    done
fi

"$PYTHON_BIN" "$LATEX_PY" "$MASTER_LOG_DIR"
echo "All done: $MASTER_LOG_DIR"
[ "$GPU_GROUP_QUEUE_FAILED" -eq 0 ] || { echo "Some jobs failed." >&2; exit 1; }
