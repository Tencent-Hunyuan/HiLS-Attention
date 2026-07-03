export MODEL_CONFIG="configs/olmo3_7B/olmo3_8KA2K_HoPE_LoRA.json"
export BASE_MODEL_DIR="../../Models/OLMo-stage1-step999000-base"
export OUTPUT_DIR="../../checkpoints/olmo3_8KA2K_HoPE_LoRA/pytorch_model.bin"
export LOG_PATH="./olmo3_8KA2K_HoPE_LoRA.log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

python3 utils/convert_basemodel_to_hsa.py \
    --target_config $MODEL_CONFIG \
    --base_path $BASE_MODEL_DIR \
    --output_path $OUTPUT_DIR \
    --log_path $LOG_PATH