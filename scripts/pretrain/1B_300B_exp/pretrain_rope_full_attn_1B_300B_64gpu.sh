export PYTHONPATH=./



export MODEL_CONFIG="configs/swan_gpt_tiny/config_rope_full_theta10000_1B.json"
export CORPUS_PATH="${CORPUS_PATH:-data/dolma3_mix-6T-1025-500B}"
export MAX_SEQ_LEN=8192
export WANDB_NAME="rope-full-attn-swangpt-1B-300B-64gpu"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/rope-full-attn-rope-swangpt-1B-300B-64gpu}"
export MAX_STEPS=143000
export SAVE_STEPS=5000
export TRAIN_SIZE=300000000000
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=256


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

bash scripts/pretrain/pretrain_ruler_task_5per_dist_300B.sh
