#!/usr/bin/env bash
set -euo pipefail
model_path=${MODEL_PATH:-/home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf}

cd llama.cpp

# # 模式 B1: 双卡张量并行 — AllReduce 走 llama.cpp / ggml 自带 CUDA kernel (allreduce.cu), 不依赖 NCCL
# CUDA_VISIBLE_DEVICES=2,3 \
# GGML_CUDA_P2P=1 \
# GGML_CUDA_ALLREDUCE=internal \
# ./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor


# # B1只支持 2 卡张量并行, 如果要多卡张量并行, 会默认lower到B2模式;     # #
# # 走llama.cpp自带的butterfly reduction, 不依赖NCCL, 但速度会慢一些. # #


# 模式 B2: 多卡张量并行 — AllReduce 走 llama.cpp / ggml 自带 meta-backend 的 butterfly reduction, 不依赖 NCCL
# -sm tensor 会自动开启llama.cpp自研 flash_attn (见 llama-context.cpp:3513),不用手动加参数。
CUDA_VISIBLE_DEVICES=2,3,4,5 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=none \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
