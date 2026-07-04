#!/bin/bash

SAVE_DIR=${1:-"/data/dolma3_dolmino_mix-100B-1125"}
REPO="allenai/dolma3_dolmino_mix-100B-1125"

mkdir -p "$SAVE_DIR"

export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download \
    "$REPO" \
    --repo-type dataset \
    --local-dir "$SAVE_DIR" \
    --resume-download \
    --token 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
