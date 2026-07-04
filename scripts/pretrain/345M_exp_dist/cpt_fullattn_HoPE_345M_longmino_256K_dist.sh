export PYTHONPATH=./



export MODEL_CONFIG="configs/fullattn_tiny/config_fullattn_HoPE_345M_256K.json"
export MODEL_PATH="${MODEL_PATH:-outputs/checkpoints/fullattn_HOPE_nolmktoken/checkpoints/global_step_30000/hf_ckpt}"
export CORPUS_PATH="${CORPUS_PATH:-data/dolma3_long_tokenized}"
export MAX_SEQ_LEN=262144
export WANDB_NAME="fullattn_HOPE_theta10000_345M_256K_dist"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/fullattn_HOPE_theta10000_345M_256K_dist}"
export GRADIENT_CKPT=true
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=32
export TRAIN_SIZE=10000000000
export MAX_STEPS=1193
export SAVE_STEPS=500
export TRAINING_RECIPE="configs/baselines/full_attn_tiny_cos.yaml"
export WANDB_PROJECT="345M_long"

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

bash scripts/pretrain/cpt_ruler_task_5per_345M_dist.sh
