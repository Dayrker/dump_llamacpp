#!/usr/bin/env bash
set -euo pipefail
model_path=${MODEL_PATH:-/home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf}

# # dump 逻辑
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
allreduce=${GGML_CUDA_ALLREDUCE:-none}
if [[ "${1:-}" == "--dump" || "${1:-}" == "--dump-only" ]]; then
  dump_target=${2:-$allreduce} # none / internal / all
  [[ "$dump_target" == "all" ]] || allreduce=$dump_target
  MODEL_PATH="$model_path" "$script_dir/dump_llamacpp_tensor.sh" "$dump_target"
  if [[ "$1" == "--dump-only" ]]; then
    exit 0
  fi
fi

case "$allreduce" in
  internal) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-2,3} ;;
  none)     cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-2,3,4,5} ;;
  *)        echo "error: GGML_CUDA_ALLREDUCE must be internal or none" >&2; exit 1 ;;
esac

cd "$script_dir/../llama.cpp"

# # 模式 B1: 双卡张量并行 — AllReduce 走 llama.cpp / ggml 自带 CUDA kernel (allreduce.cu), 不依赖 NCCL
# CUDA_VISIBLE_DEVICES=2,3 \
# GGML_CUDA_P2P=1 \
# GGML_CUDA_ALLREDUCE=internal \
# ./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor


# # B1只支持 2 卡张量并行, 如果要多卡张量并行, 会默认lower到B2模式;     # #
# # 走llama.cpp自带的butterfly reduction, 不依赖NCCL, 但速度会慢一些. # #


# 默认模式 B2: 多卡张量并行 — AllReduce 走 llama.cpp / ggml 自带 meta-backend 的 butterfly reduction, 不依赖 NCCL
# 如需 dump: `bash tools/run_llamacpp_tensor.sh --dump-only none|internal|all`
# -sm tensor 会自动开启llama.cpp自研 flash_attn (见 llama-context.cpp:3513),不用手动加参数。
CUDA_VISIBLE_DEVICES="$cuda_visible_devices" \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE="$allreduce" \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
