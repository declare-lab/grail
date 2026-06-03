#!/bin/bash

MODELS=(
    "qwen3-4b-grpo-run1/checkpoint-200"
    "qwen3-4b-grail-run1/checkpoint-200"
)
INPUT_PATHS=()

for MODEL in "${MODELS[@]}"; do
    FILE_PATH="./results/${MODEL}/eval_summary.json"
    INPUT_PATHS+=("$FILE_PATH")
done

python convert_to_excel.py \
    --output_file "./results/combined_report_2.xlsx" \
    --input_files "${INPUT_PATHS[@]}"
