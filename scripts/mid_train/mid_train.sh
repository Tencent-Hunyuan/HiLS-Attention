export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ "${USE_LOCAL_VEOMNI_SRC:-1}" = "1" ]; then
    export PYTHONPATH="${PROJECT_ROOT}/../veomni_src:${PROJECT_ROOT}:${PYTHONPATH}"
else
    export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
fi

export MODEL_CONFIG="configs/olmo3_7B/olmo3_param_reuse.json"
export MODEL_PATH="/checkpoints/olmo3_cpt_64gpu/checkpoints/global_step_13000/hf_ckpt/"
export CORPUS_PATH="${PROJECT_ROOT}/scripts/mid_train/dolmino_mix.yaml"
export MAX_SEQ_LEN=8192
export WANDB_NAME="olmo3_param_reuse_mid_train"
export OUTPUT_DIR="/checkpoints/olmo3_param_reuse_mid_train"
export TOKEN_CNT=100_000_000_000  # 100B tokens
export BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=512
export TRAINING_RECIPE="configs/olmo3_7B/training_recipe_64gpu.yaml"
export MAX_LR=2e-4
export MIN_LR=0
export MAX_STEPS=23_841
export SAVE_STEPS=1_000
export LR_WARMUP_RATIO=$(python - << 'PY'
import os
max_steps = int(os.environ["MAX_STEPS"])
print(1000 / max_steps)
PY
)

export USE_LIGER_KERNEL=1

export USE_LIGER_RMSNORM=0
export USE_FLASH_ATTN_RMSNORM=1
export USE_LIGER_ROPE=1
export USE_LIGER_SWIGLU=1
export USE_LIGER_CE=1

export EXTRA_ARGS="--train.lr_decay_style cosine --train.no_decay_params norm bias embed"
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

echo "Using MAX_STEPS=${MAX_STEPS} (TOKEN_CNT=${TOKEN_CNT}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}, MAX_SEQ_LEN=${MAX_SEQ_LEN})"

MAX_PREFETCH_RETRIES=10
for i in $(seq 1 $MAX_PREFETCH_RETRIES); do
    echo "[Prefetch] Attempt $i/$MAX_PREFETCH_RETRIES ..."
    python code_exp/flash_hsa_run.py
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

export DATASETS_TYPE="olmo3_mix"

bash scripts/cpt/CPT_dist.sh 2>&1 | tee "$LOG_FILE"

