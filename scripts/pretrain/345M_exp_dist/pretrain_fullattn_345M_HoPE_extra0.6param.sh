export PYTHONPATH=./


export MODEL_CONFIG="configs/fullattn_tiny/config_fullattn_HoPE_345M_extra0.6param.json"
export CORPUS_PATH="${CORPUS_PATH:-data/dolma3_mix-6T-1025-partial-tokenized}"
export MAX_SEQ_LEN=8192
export WANDB_NAME="fullattn_HOPE_extra0.6param"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/fullattn_HOPE_nolmktoken_extra0.6param}"
export GRADIENT_CKPT=false
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128
bash scripts/pretrain/pretrain_ruler_task_5per_345M_dist.sh