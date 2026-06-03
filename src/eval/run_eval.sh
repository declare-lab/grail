#!/bin/bash
# Start the VLLM servers for the specified model checkpoints, and then run the evaluation script to compute the evaluation metrics for each checkpoint on the specified datasets. Finally, it generates summary reports for each model.

MODEL_BASE_PATH="<Path to your models directory here>"

model_names=(
    "qwen3-4b-grail-run1/checkpoint-200"
)

model_ports=("8000")

dataset_list=("aime24" "math" "college_math" "minerva_math" "olympiadbench" "amc23")

pids=()

for i in "${!model_names[@]}"; do
    (
        model_name="${model_names[$i]}"
        model_port="${model_ports[$i]}"
        for dataset in "${dataset_list[@]}"; do
            python run_eval.py \
                --policy_model_path "${MODEL_BASE_PATH}/${model_name}" \
                --data "$dataset" \
                --output_dir "./results/${model_name}/$dataset" \
                --api_base "http://localhost:$model_port/v1" \
                --temperature 0.6 \
                --top_p 0.95 \
                --top_k 20 \
                --max_workers 128 #64 #32
        done

        python eval_results_base.py \
            --results_dir results/${model_name}

    ) &
    pids+=($!)
done

trap "kill ${pids[*]}; exit" SIGINT SIGTERM

wait
