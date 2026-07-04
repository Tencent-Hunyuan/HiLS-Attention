export PYTHONPATH=./

DATA_TYPE=${DTYPE:-ruler_0.05}
MAX_STEPS=${MAX_STEPS:-200000}
SAVE_STEPS=${SAVE_STEPS:-5000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
TRAIN_SIZE=${TRAIN_SIZE:-500000000000}

bash train_dist.sh tasks/pretrain_with_ruler.py configs/training_recipes/pretrain_1.4B_8K_300B_64gpu.yaml \
    --model.config_path $MODEL_CONFIG \
    --data.train_path $CORPUS_PATH \
    --data.max_seq_len $MAX_SEQ_LEN \
    --data.train_size $TRAIN_SIZE \
    --data.data_type $DATA_TYPE \
    --data.datasets_type olmo3 \
    --data.sort_files true \
    --data.enable_ruler_plus true \
    --data.num_workers 16 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.enable_gradient_checkpointing true \
    --train.rmpad false \
    --train.wandb_project ruler_pretrain_1B_300B_64gpu \
    --train.wandb_name $WANDB_NAME \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size $MICRO_BATCH_SIZE \
    --train.global_batch_size $GLOBAL_BATCH_SIZE \
    --train.lr 3e-4 \
    --train.lr_min 3e-5 \
    --train.no_decay_params norm bias embed \
    --train.ulysses_parallel_size 1 \
    --train.save_steps $SAVE_STEPS \
    --train.max_steps $MAX_STEPS \
    --train.load_checkpoint_path auto \
    --train.output_dir $OUTPUT_DIR
