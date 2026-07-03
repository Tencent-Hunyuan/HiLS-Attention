export PYTHONPATH=./

export MODEL_CONFIG="configs/flash_hsa/config_hsa_8KA2K_NoPE_345M_wloralmkq_loradim64.json"
export CORPUS_PATH="${CORPUS_PATH:-data/dolma3_mix-6T-1025-partial-tokenized}"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hsa_8KA2K_NoPE_full_345M_wloralmkq_loradim64"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/hsa_8KA2K_NoPE_full_345M_dist_wloralmkq_loradim64}"
export GRADIENT_CKPT=false
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128
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
bash scripts/pretrain/pretrain_ruler_task_5per_345M_dist.sh
