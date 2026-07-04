export PYTHONPATH=./



export MODEL_CONFIG="configs/fullattn_tiny/config_rope_full_theta10000000_345M_256K.json"
export MODEL_PATH="${MODEL_PATH:-outputs/checkpoints/rope_full_theta10000_345M_dist/checkpoints/global_step_30000/hf_ckpt}"
export CORPUS_PATH="${CORPUS_PATH:-data/dolma3_long_tokenized}"
export MAX_SEQ_LEN=262144
export WANDB_NAME="rope_full_theta10000000_345M_256K_dist"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/rope_full_theta10000000_345M_256K_dist}"
export GRADIENT_CKPT=true
export MICRO_BATCH_SIZE=1
export GLOBAL_BATCH_SIZE=32
export TRAIN_SIZE=10000000000
export MAX_STEPS=1193
export SAVE_STEPS=500
export TRAINING_RECIPE="configs/training_recipes/cpt_345M_256K_longmino.yaml"
export WANDB_PROJECT="345M_long"

bash scripts/pretrain/cpt_ruler_task_5per_345M_dist.sh
