#!/bin/bash
export HF_TOKEN="Your HF Token Here"

MODEL_BASE_PATH="<Path to your models directory here>"

model_names=(
    "qwen3-4b-grail-run1/checkpoint-200"
)

model_ports=("8000")

GPUS=("0,1,2,3")

for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    model_port="${model_ports[$i]}"
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} vllm serve "${MODEL_BASE_PATH}/${model_name}" \
        --tensor-parallel-size 2 \
        --data-parallel-size 2 \
        --port ${model_port} \
        --host 0.0.0.0 \
        --max-model-len 16384 \
        --max-num-seqs 1024 \
        --gpu-memory-utilization 0.95 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --disable-fastapi-docs &
done

wait
