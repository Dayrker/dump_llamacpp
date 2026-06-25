# model path
model_path=/home/zhangchen/PTX/model/Qwen3-8B/Qwen3-8B-bf16.gguf

# NCCL 运行时库 (项目内子模块 nccl/ 编译产物 libnccl.so)
# 仅「模式 A: NCCL AllReduce」需要; 其余模式 (internal / layer / 单卡) 可忽略
export LD_LIBRARY_PATH=/home/zhangchen/PTX/dump_llamacpp/nccl/build/lib:${LD_LIBRARY_PATH:-}

cd llama.cpp


# # 模式 A: 多卡张量并行 — AllReduce 走 NCCL 库 (libnccl.so)
# CUDA_VISIBLE_DEVICES=2,3 \
# GGML_CUDA_P2P=1 \
# GGML_CUDA_ALLREDUCE=nccl \
# ./build/bin/llama-cli -m $model_path -ngl all -sm tensor

# 模式 B1: 双卡张量并行 — AllReduce 走 llama.cpp / ggml 自带 CUDA kernel (allreduce.cu), 不依赖 NCCL
# -sm tensor 会自动开启 flash_attn
CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=internal \
./build/bin/llama-cli -m $model_path -ngl all -sm tensor

# # 模式 B1 设置为多卡时，会自动 lower 到 B2 的 GGML_CUDA_ALLREDUCE=none 模式
# # 模式 B2: 多卡张量并行 — AllReduce 走 llama.cpp / ggml 自带 meta-backend 的 butterfly reduction, 不依赖 NCCL
# # -sm tensor 会自动开启llama.cpp自研 flash_attn (见 llama-context.cpp:3513),不用手动加参数。
# CUDA_VISIBLE_DEVICES=2,3,4,5 \
# GGML_CUDA_P2P=1 \
# GGML_CUDA_ALLREDUCE=none \
# ./build/bin/llama-cli -m $model_path -ngl all -sm tensor

# # 模式 C: 多卡层切分 (pipeline, 无 reduce)
# CUDA_VISIBLE_DEVICES=2,3 \
# GGML_CUDA_P2P=1 \
# ./build/bin/llama-cli -m $model_path -ngl all

# # 模式 D: 单卡
# CUDA_VISIBLE_DEVICES=0 \
# ./build/bin/llama-cli -m $model_path -ngl all
