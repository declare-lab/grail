#!/bin/bash
export TORCH_CPP_LOG_LEVEL=WARNING
export NCCL_DEBUG=ERROR
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export TORCH_DISTRIBUTED_DEBUG=INFO
export CUDA_VISIBLE_DEVICES="0,1,2,3"

mkdir -p logs

export WANDB_API_KEY="Your Wandb API Key Here"
export HF_TOKEN="Your HF Token Here"


ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 25679 \
--config_file training_configs/deepspeed_zero2.yaml \
--num_processes $(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c | awk '{print $1+1}') \
grpo.py \
training_configs/qwen3_grpo.yaml 2>&1 | tee logs/qwen3_grpo_$(date +%Y%m%d_%H%M%S).log

echo "Training Done"
