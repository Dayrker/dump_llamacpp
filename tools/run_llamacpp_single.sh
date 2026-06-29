#!/usr/bin/env bash
set -euo pipefail

# 模式 D: 单卡。
model_path=${MODEL_PATH:-/home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf}

cd llama.cpp

CUDA_VISIBLE_DEVICES=0 \
./build/bin/llama-cli -m "$model_path" -ngl all
