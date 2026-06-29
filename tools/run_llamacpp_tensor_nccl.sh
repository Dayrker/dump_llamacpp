#!/usr/bin/env bash
set -euo pipefail

# 模式 A: 多卡张量并行，AllReduce 走 NCCL 库。
model_path=${MODEL_PATH:-/home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf}

export LD_LIBRARY_PATH=/home/zhangchen/PTX/dump_llamacpp/nccl/build/lib:${LD_LIBRARY_PATH:-}

cd llama.cpp

CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=nccl \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
