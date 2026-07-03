export PYTHONPATH=./

MODEL_CONFIG=${MODEL_CONFIG:-"configs/flash_hsa/config_lsa_unified.json"}                # TODO: fill model config path
MODEL_PATH=${MODEL_PATH:-""}                    # TODO: fill model weights path
CORPUS_PATH=${CORPUS_PATH:-"/data/Dolci-Instruct-SFT-parquet"}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-32768}
CHUNK_SIZE=${CHUNK_SIZE:-64}
WANDB_NAME=${WANDB_NAME:-"sft-lmk"}
OUTPUT_DIR=${OUTPUT_DIR:-"/checkpoints/sft-lmk-debug"}

bash train.sh tasks/sft_with_lmk.py configs/sft/sft_lmk.yaml \
    --model.config_path $MODEL_CONFIG \
    --data.train_path $CORPUS_PATH \
    --data.max_seq_len $MAX_SEQ_LEN \
    --train.chunk_size $CHUNK_SIZE \
    --train.use_wandb true \
    --train.wandb_name $WANDB_NAME \
    --train.enable_gradient_checkpointing true \
    --train.micro_batch_size 1 \
    --train.global_batch_size 8 \
    --train.lr 2e-5 \
    --train.lr_min 2e-6 \
    --train.save_steps 500 \
    --train.max_steps 3000 \
    --train.load_checkpoint_path auto \
    --train.output_dir $OUTPUT_DIR
