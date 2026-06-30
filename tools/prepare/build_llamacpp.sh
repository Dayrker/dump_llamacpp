# conda activate torch251
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# NCCL 子模块 (绝对路径)
nccl_root=/home/zhangchen/PTX/dump_llamacpp/nccl


# # ---- NCCL: 本地源码编译 (A100 = sm_80) ----
# # 源码 v2.21.5-1 (CUDA 12.1 可编的版本), 产物: $nccl_root/build/{lib/libnccl.so, include/nccl.h}
# make -C $nccl_root -j 64 src.build \
#   CUDA_HOME=/usr/local/cuda \
#   NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"


# ---- llama.cpp: 本地源码链接nccl编译 (A100 = sm_80) ----
cd llama.cpp
# Cmake (首次 / 改配置时执行)
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DGGML_CUDA_NCCL=ON \
  -DNCCL_ROOT=$nccl_root/build
# Build
cmake --build build --config Release -j 64