# model path
model_path=/home/zhangchen/PTX/model/Qwen3-8B/Qwen3-8B-bf16.gguf

cd llama.cpp


# run model multi-card
CUDA_VISIBLE_DEVICES=0,1 \
GGML_CUDA_P2P=1 \
CUDA_SCALE_LAUNCH_QUEUES=4x \
./build/bin/llama-cli -m $model_path -ngl all

# # run model single-card
# CUDA_VISIBLE_DEVICES=0 \
# ./build/bin/llama-cli -m $model_path -ngl all