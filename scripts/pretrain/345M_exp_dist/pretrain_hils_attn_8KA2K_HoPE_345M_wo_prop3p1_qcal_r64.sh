export PYTHONPATH=./



export MODEL_CONFIG="configs/hils_attention/config_hils_attn_8KA2K_HoPE_345M_wo_prop3p1_qcal_r64.json"
export CORPUS_PATH="${CORPUS_PATH:-data/dolma3_mix-6T-1025-partial-tokenized}"
export MAX_SEQ_LEN=8192
export WANDB_NAME="hils_attn_8KA2K_HoPE_345M_wo_prop3p1_qcal_r64"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/hils_attn_8KA2K_HoPE_345M_wo_prop3p1_qcal_r64}"
export GRADIENT_CKPT=false
export MICRO_BATCH_SIZE=4
export GLOBAL_BATCH_SIZE=128
bash scripts/pretrain/pretrain_ruler_task_5per_345M_dist.sh
