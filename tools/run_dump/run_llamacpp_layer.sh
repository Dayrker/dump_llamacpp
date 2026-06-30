#!/usr/bin/env bash
set -euo pipefail

# 模式 C: 多卡层切分 pipeline，无 AllReduce。
model_path=${MODEL_PATH:-/home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf}

cd llama.cpp

CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
./build/bin/llama-cli -m "$model_path" -ngl all
