export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="mlx5_bond"
export NCCL_IB_GID_INDEX="3"
export NCCL_NET_GDR_LEVEL="0"
export NCCL_NET_GDR_READ="0"
export NCCL_IB_QPS_PER_CONNECTION="4"
export NCCL_IB_TC="136"
export NCCL_IB_TIMEOUT="22"
export NCCL_IB_RETRY_CNT="13"
export NCCL_SOCKET_IFNAME="bond1"
export NCCL_BUFFSIZE="8388608"
export NCCL_NVLS_ENABLE="0"
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

export MODEL_CONFIG="${MODEL_CONFIG:-configs/olmo3_7B/olmo3_8KA2K_HoPE_qcal.json}"

export HF_CKPT="${HF_CKPT:-../../checkpoints/olmo3_8KA2K_HoPE_qcal/pytorch_model.bin}"
export MODEL_PATH="${MODEL_PATH:-${HF_CKPT}}"
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"

export CORPUS_PATH="${CORPUS_PATH:-../../data/dolma3_mix-6T-1025-500B/}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
export WANDB_NAME="${WANDB_NAME:-olmo3_8KA2K_HoPE_qcal}"
export OUTPUT_DIR="${OUTPUT_DIR:-../../checkpoints/olmo3_8KA2K_HoPE_qcal}"
export TOKEN_CNT=500_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=2e-4
export MIN_LR=2e-5
export MAX_STEPS=13000
export WARMUP_STEPS=1000
export SAVE_STEPS=1000

export LR_WARMUP_RATIO=$(awk "BEGIN { printf \"%.18g\", ${WARMUP_STEPS} / ${MAX_STEPS} }")

if [ "${MAX_STEPS}" -le 0 ]; then
    echo "ERROR: MAX_STEPS must be positive, got ${MAX_STEPS}" >&2
    exit 1
fi
if [ "${WARMUP_STEPS}" -ge "${MAX_STEPS}" ]; then
    echo "ERROR: WARMUP_STEPS (${WARMUP_STEPS}) must be < MAX_STEPS (${MAX_STEPS})" >&2
    exit 1
fi
if [ ! -d "${HF_CKPT}" ]; then
    echo "ERROR: HF_CKPT not found: '${HF_CKPT}'" >&2
    exit 1
fi

export EXTRA_ARGS="\
  --train.load_optimizer_state false \
  --train.load_lr_scheduler_state false \
  --train.load_dataloader_state false \
  --train.load_rng_state false \
  --train.include_frozen_params_in_optimizer false \
  --train.lr_decay_style cosine \
  --train.lr_decay_ratio 1.0 \
  --train.no_decay_params norm bias embed \
"

COSINE_STEPS=$((MAX_STEPS - WARMUP_STEPS))
echo "Stage2 from step 0: HF weights=${HF_CKPT}"
echo "MAX_STEPS=${MAX_STEPS}, warmup ${WARMUP_STEPS} steps (0->${MAX_LR}), cosine ${COSINE_STEPS} steps (${MAX_LR}->${MIN_LR})"
echo "LR_WARMUP_RATIO=${LR_WARMUP_RATIO}, LOAD_CHECKPOINT_PATH='${LOAD_CHECKPOINT_PATH}' (empty = no DCP resume)"

MAX_PREFETCH_RETRIES=10
for i in $(seq 1 $MAX_PREFETCH_RETRIES); do
    echo "[Prefetch] Attempt $i/$MAX_PREFETCH_RETRIES ..."
    python code_exp/prefetch.py $MODEL_CONFIG
    if [ $? -eq 0 ]; then
        echo "[Prefetch] Success on attempt $i."
        break
    fi
    if [ $i -eq $MAX_PREFETCH_RETRIES ]; then
        echo "[Prefetch] Failed after $MAX_PREFETCH_RETRIES attempts, aborting." >&2
        exit 1
    fi
    echo "[Prefetch] Attempt $i failed, retrying..."
    sleep 2
done

pkill -f "burner"

bash scripts/cpt/CPT_dist.sh 2>&1 | tee "$LOG_FILE"
