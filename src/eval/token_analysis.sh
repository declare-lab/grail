# !/bin/bash
# This script computes token-level statistics for the checkpoints of a specified model, and generates plots to visualize these statistics.
# Run this script after completing the checkpoint evaluations using the run_eval.sh script.

export CUDA_VISIBLE_DEVICES=""
MODEL_BASE_PATH="<Path to your models directory here>"

GRAIL_MODEL="qwen3-4b-grail-run1"
GRAIL_RESULTS="results/${GRAIL_MODEL}"

python compute_checkpoint_stats.py \
    --checkpoints_dir "${MODEL_BASE_PATH}/${GRAIL_MODEL}" \
    --results_dir "${GRAIL_RESULTS}" \
    --output_dir "token_analysis_stats/${GRAIL_MODEL}" \
    --batch_size 1

python aggregate_and_plot.py \
    --stats_dir "token_analysis_stats/${GRAIL_MODEL}" \
    --output_dir "token_analysis_plots/${GRAIL_MODEL}"
