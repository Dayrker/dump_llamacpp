cd llama.cpp

# python convert_hf_to_gguf.py /home/zhangchen/PTX/model/Qwen3-8B \
#   --outfile /home/zhangchen/PTX/model/Qwen3-8B/Qwen3-8B-bf16.gguf \
#   --outtype bf16

python convert_hf_to_gguf.py /home/zhangchen/PTX/model/Qwen3-32B \
  --outfile /home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf \
  --outtype bf16