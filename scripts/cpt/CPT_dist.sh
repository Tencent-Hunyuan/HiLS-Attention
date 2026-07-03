export PYTHONPATH="./:${PYTHONPATH}"

DATA_TYPE=${DTYPE:-ruler_0.05}
LOAD_CKPT_ARGS=()
if [ -n "${LOAD_CHECKPOINT_PATH+x}" ] && [ -n "${LOAD_CHECKPOINT_PATH}" ]; then
    LOAD_CKPT_ARGS+=(--train.load_checkpoint_path "$LOAD_CHECKPOINT_PATH")
elif [ -z "${LOAD_CHECKPOINT_PATH+x}" ]; then
    LOAD_CKPT_ARGS+=(--train.load_checkpoint_path auto)
fi

# Pass freeze regex via env to avoid shell quoting bugs in EXTRA_ARGS (literal quotes in pattern).
FREEZE_PATTERN_ARGS=()
if [ -n "${TRAIN_FREEZE_PATTERN:-}" ]; then
    FREEZE_PATTERN_ARGS=(--train.freeze_pattern "${TRAIN_FREEZE_PATTERN}")
fi

bash train_dist.sh tasks/pretrain_with_ruler.py $TRAINING_RECIPE \
    --model.config_path $MODEL_CONFIG \
    --model.model_path $MODEL_PATH \
    --data.train_path $CORPUS_PATH \
    --data.max_seq_len $MAX_SEQ_LEN \
    --data.train_size $TOKEN_CNT \
    --data.data_type $DATA_TYPE \
    --data.datasets_type ${DATASETS_TYPE:-olmo3} \
    --data.num_workers 16 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.enable_gradient_checkpointing true \
    --train.rmpad false \
    --train.wandb_name $WANDB_NAME \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size $BATCH_SIZE \
    --train.global_batch_size $GLOBAL_BATCH_SIZE \
    --train.lr $MAX_LR \
    --train.lr_min $MIN_LR \
    --train.lr_warmup_ratio $LR_WARMUP_RATIO \
    --train.ulysses_parallel_size 1 \
    --train.save_steps $SAVE_STEPS \
    --train.max_steps $MAX_STEPS \
    --train.output_dir $OUTPUT_DIR \
    "${LOAD_CKPT_ARGS[@]}" \
    "${FREEZE_PATTERN_ARGS[@]}" \
    $EXTRA_ARGS
