python preprocess/build_olmo3_datasets_weighted.py \
  --vocab_dir configs/olmo3_vocab \
  --corpus_dir /data/dolma3_mix-6T-1025 \
  --output_file /data/dolma3_mix-6T-1025-500B.numpy \
  --token_quota 500000000000 \
  --num_workers 32 \
  --batch_tokens 65536 \
  --buffer_tokens 1000000 \
  --seed 1234