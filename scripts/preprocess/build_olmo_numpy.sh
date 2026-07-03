# python preprocess/build_olmo3_datasets.py --vocab_dir configs/olmo3_vocab --corpus_dir /data/dolma3_longmino_mix-100B-1125 --output_dir /data/dolma3_long_tokenized --num_workers 256

# # long context
# python preprocess/build_olmo3_datasets.py --vocab_dir configs/olmo3_vocab --corpus_dir /data/dolma3_longmino_mix-100B-1125/data/olmocr_science_pdfs-high_quality-science_tech-length_2e15 --output_dir /data/dolma3_long_tokenized/data/olmocr_science_pdfs-high_quality-science_tech-length_2e15 --num_workers 256

# long context
python preprocess/build_olmo3_datasets.py --vocab_dir configs/olmo3_vocab --corpus_dir /data/dolma3_mix-6T-1025-7B --output_dir /data/dolma3_mix-6T-1025-partial-tokenized --num_workers 128

# olmocr_science_pdfs-high_quality-finance_business-length_2e15
# program_verifiable
# olmocr_science_pdfs-high_quality-science_tech-length_2e15
