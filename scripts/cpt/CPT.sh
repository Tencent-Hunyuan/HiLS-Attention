export PYTHONPATH="./:${PYTHONPATH}"

DATA_TYPE=${DTYPE:-ruler_0.05}

USE_WANDB_ENV=${USE_WANDB:-true}
WANDB_ARGS=()
if [ "$USE_WANDB_ENV" = "true" ]; then
    WANDB_ARGS+=(--train.use_wandb true)
    if [ -n "$WANDB_NAME" ]; then
        WANDB_ARGS+=(--train.wandb_name "$WANDB_NAME")
    fi
else
    WANDB_ARGS+=(--train.use_wandb false)
fi

bash train.sh tasks/pretrain_with_ruler.py $TRAINING_RECIPE \
    --model.config_path $MODEL_CONFIG \
    --model.model_path $MODEL_PATH \
    --data.train_path $CORPUS_PATH \
    --data.max_seq_len $MAX_SEQ_LEN \
    --data.train_size $TOKEN_CNT \
    --data.data_type $DATA_TYPE \
    --data.datasets_type olmo3 \
    --data.num_workers 16 \
    --train.init_device meta \
    "${WANDB_ARGS[@]}" \
    --train.enable_gradient_checkpointing true \
    --train.rmpad false \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size $BATCH_SIZE \
    --train.global_batch_size $GLOBAL_BATCH_SIZE \
    --train.lr $MAX_LR \
    --train.lr_min $MIN_LR \
    --train.lr_warmup_ratio $LR_WARMUP_RATIO \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 10000 \
    --train.max_steps $MAX_STEPS \
    --train.load_checkpoint_path auto \
    --train.output_dir $OUTPUT_DIR \
    $EXTRA_ARGS
