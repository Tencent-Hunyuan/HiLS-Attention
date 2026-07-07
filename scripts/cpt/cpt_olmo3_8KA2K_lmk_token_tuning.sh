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

export MODEL_CONFIG="${MODEL_CONFIG:-configs/olmo3_7B/olmo3_8KA2K_lmk_token_tuning.json}"

CEPH_MODEL_PATH="${CEPH_MODEL_PATH:-../../checkpoints/olmo3_8KA2K_lmk_token_tuning}"
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"
export MODEL_PATH="${MODEL_PATH:-${CEPH_MODEL_PATH}}"
export CORPUS_PATH="${CORPUS_PATH:-../../data/dolma3_mix-6T-1025-500B/}"
export MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
export WANDB_NAME="${WANDB_NAME:-olmo3_8KA2K_lmk_token_tuning}"
export OUTPUT_DIR="${OUTPUT_DIR:-../../checkpoints/olmo3_8KA2K_lmk_token_tuning}"
export TOKEN_CNT=500_000_000_000
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe.yaml"
export MAX_LR=2e-4
export MIN_LR=2e-5
export MAX_STEPS=1_192  # ~10B tokens (512 * 8192 = 4,194,304 tokens/step)
export SAVE_STEPS=1_000
export LR_WARMUP_RATIO=0.02  # 2% warmup, then cosine decay to MIN_LR over remaining steps

if [ "$MAX_STEPS" = "auto" ] || [ "$MAX_STEPS" = "-1" ]; then
    token_cnt_clean=${TOKEN_CNT//_/}
    global_batch_size_clean=${GLOBAL_BATCH_SIZE//_/}
    max_seq_len_clean=${MAX_SEQ_LEN//_/}
    tokens_per_step=$((global_batch_size_clean * max_seq_len_clean))

    if [ "$tokens_per_step" -le 0 ]; then
        echo "ERROR: GLOBAL_BATCH_SIZE * MAX_SEQ_LEN must be > 0, got ${GLOBAL_BATCH_SIZE} * ${MAX_SEQ_LEN}" >&2
        exit 1
    fi

    export MAX_STEPS=$((token_cnt_clean / tokens_per_step))

    if [ "$MAX_STEPS" -le 0 ]; then
        echo "ERROR: computed MAX_STEPS=${MAX_STEPS} from TOKEN_CNT=${TOKEN_CNT}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MAX_SEQ_LEN=${MAX_SEQ_LEN}" >&2
        exit 1
    fi
fi



# Do not put freeze_pattern in EXTRA_ARGS: unquoted $EXTRA_ARGS expansion leaves literal quotes
# in the regex (matches nothing -> trainable=380). Pass via TRAIN_FREEZE_PATTERN instead.
export TRAIN_FREEZE_PATTERN='^(?!.*(?:lmk_embed|lmk_q_proj|lmk_q_norm)).*'

export EXTRA_ARGS="\
  --train.load_optimizer_state false \
  --train.load_lr_scheduler_state false \
  --train.load_dataloader_state false \
  --train.load_rng_state false \
  --train.include_frozen_params_in_optimizer false \
  --train.lr_decay_style cosine \
  --train.no_decay_params norm bias embed \
"

echo "Using MAX_STEPS=${MAX_STEPS} (TOKEN_CNT=${TOKEN_CNT}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MAX_SEQ_LEN=${MAX_SEQ_LEN})"

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
