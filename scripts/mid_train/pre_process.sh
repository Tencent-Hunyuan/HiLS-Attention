mkdir -p /data/dolmino_ingredient1_raw
for d in /data/dolma3_dolmino_mix-100B-1125/data/ingredient1-*; do                                                                                             
    ln -sf "$d" /data/dolmino_ingredient1_raw/$(basename "$d")
done                                                                                                                                                                                                
                                                                
# 2. Tokenize
python preprocess/build_olmo3_datasets.py \
    --vocab_dir /configs/olmo3_vocab \
    --corpus_dir /data/dolmino_ingredient1_raw/ \
    --output_dir /data/dolmino-midtrain-100B-tokenized/ \
    --num_workers 128
