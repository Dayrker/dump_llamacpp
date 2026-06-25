# model path
model_path=/home/zhangchen/PTX/model/Qwen3-8B/Qwen3-8B-bf16.gguf

# NCCL 运行时库 (项目内子模块 nccl/ 编译产物 libnccl.so)
export LD_LIBRARY_PATH=/home/zhangchen/PTX/dump_llamacpp/nccl/build/lib:${LD_LIBRARY_PATH:-}

cd llama.cpp


# # run model multi-card -> tensor parallel (NCCL AllReduce)
# CUDA_VISIBLE_DEVICES=2,3 \
# GGML_CUDA_P2P=1 \
# GGML_CUDA_ALLREDUCE=nccl \
# ./build/bin/llama-cli -m $model_path -ngl all -sm tensor

# run model multi-card -> layer split (pipeline, 无 reduce)
CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
./build/bin/llama-cli -m $model_path -ngl all

# # run model single-card
# CUDA_VISIBLE_DEVICES=0 \
# ./build/bin/llama-cli -m $model_path -ngl all