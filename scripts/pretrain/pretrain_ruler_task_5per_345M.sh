export PYTHONPATH=./

DATA_TYPE=${DTYPE:-ruler_0.05}

bash train.sh tasks/pretrain_with_ruler.py configs/training_recipes/pretrain_345M_8K_30B.yaml \
    --model.config_path $MODEL_CONFIG \
    --data.train_path $CORPUS_PATH \
    --data.max_seq_len $MAX_SEQ_LEN \
    --data.train_size 10000000000 \
    --data.data_type $DATA_TYPE \
    --data.datasets_type olmo3 \
    --data.sort_files true \
    --data.enable_ruler_plus true \
    --data.num_workers 16 \
    --train.init_device meta \
    --train.use_wandb true \
    --train.enable_gradient_checkpointing $GRADIENT_CKPT \
    --train.rmpad false \
    --train.wandb_project ruler_pretrain_5per_345M \
    --train.wandb_name $WANDB_NAME \
    --train.rmpad_with_pos_ids false \
    --train.enable_mixed_precision \
    --train.micro_batch_size 16 \
    --train.global_batch_size 128 \
    --train.lr 3e-4 \
    --train.lr_min 3e-5 \
    --train.no_decay_params norm bias embed \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 5000 \
    --train.max_steps 30000 \
    --train.load_checkpoint_path auto \
    --train.output_dir $OUTPUT_DIR
