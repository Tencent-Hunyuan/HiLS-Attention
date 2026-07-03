export PYTHONPATH=./

bash train.sh tasks/sft.py configs/baselines/full_attn_tiny_cos.yaml \
    --data.data_type conversation \
    --data.chat_template chatml \
    --data.chunk_size 64 \
    --data.max_seq_len 8064 \
    --data.train_size 10000000000 \
    --train.init_device meta \
    --train.use_wandb True \
    --train.enable_gradient_checkpointing true \
    --train.enable_mixed_precision true \
    --train.rmpad false \
    --train.rmpad_with_pos_ids true \
    --train.micro_batch_size 1 \
    --train.global_batch_size 128 \
    --train.lr 1e-5 \
    --train.lr_min 1e-6 \
    --train.lr_decay_style cosine \
    --train.ulysses_parallel_size 1 \
    --train.save_steps 1000 \
    --train.max_steps 3000 \
    --train.wandb_name hsa-sft \
    --train.wandb_project hsa-sft \
    "$@"
