# conda activate torch251
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}


cd llama.cpp

# # Cmake
# cmake -B build \
#   -DGGML_CUDA=ON \
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
#   -DCMAKE_CUDA_ARCHITECTURES=80 \
#   -DGGML_CUDA_NCCL=ON

cmake --build build --config Release -j 64