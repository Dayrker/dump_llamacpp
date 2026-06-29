# dump_llama.cpp

基于 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的工作区，用于在 NVIDIA CUDA 平台上**从源码编译 llama.cpp、转换模型、运行推理**，以便 dump / 分析其 CUDA kernel 的 PTX 与链接函数（link funcs）。

llama.cpp 以 git 子模块的形式引入，所有自定义脚本集中在 [tools/](tools/) 目录，对子模块本身零侵入。

---

## 目录结构

```
dump_llamacpp/
├── llama.cpp/              # git 子模块，上游 ggml-org/llama.cpp
├── tools/                  # 自定义辅助脚本
│   ├── build_llamacpp.sh   # 编译 llama.cpp（CUDA）
│   ├── run_safeT_to_gguf.sh          # HF safetensors → GGUF 模型转换
│   ├── run_llamacpp_tensor.sh        # 模式 B：多卡张量并行（不依赖 NCCL）
│   ├── run_llamacpp_tensor_nccl.sh   # 模式 A：多卡张量并行（NCCL）
│   ├── run_llamacpp_layer.sh         # 模式 C：多卡层切分
│   └── run_llamacpp_single.sh        # 模式 D：单卡
├── .gitmodules            # 子模块配置
├── LICENSE
└── README.md
```

> 当前子模块固定版本：`00139b660`（`gguf-v0.19.0-733-g00139b660`）。

---

## 环境要求

- NVIDIA GPU + 驱动（脚本默认目标架构 **sm_80**，即 A100 / A800）
- CUDA Toolkit（默认安装在 `/usr/local/cuda`）
- CMake、C/C++ 编译器
- Python 环境（用于模型转换，脚本默认 conda 环境 `torch251`）
- NCCL 模式：需编译时开启 `-DGGML_CUDA_NCCL=ON`

---

## 快速开始

### 1. 克隆仓库（含子模块）

```bash
git clone --recursive <本仓库地址>
# 若已克隆但未拉取子模块：
git submodule update --init --recursive
```

### 2. 编译 llama.cpp

脚本 [tools/build_llamacpp.sh](tools/build_llamacpp.sh) 分两步：

- **首次编译**：需先取消脚本中 `cmake -B build ...` 配置段的注释，先做 CMake 配置（生成 `build/`），再编译。
- **后续编译**：只需 `cmake --build`（脚本默认行为），无需重复配置。

```bash
bash tools/build_llamacpp.sh
```

关键 CMake 选项：

| 选项 | 含义 |
| --- | --- |
| `-DGGML_CUDA=ON` | 开启 CUDA 后端 |
| `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc` | 指定 nvcc |
| `-DCMAKE_CUDA_ARCHITECTURES=80` | 目标架构 sm_80（A100/A800），换卡需修改 |
| `-DGGML_CUDA_NCCL=ON` | 多卡 NCCL 支持 |

### 3. 模型转换（HF → GGUF）

用 [tools/run_safeT_to_gguf.sh](tools/run_safeT_to_gguf.sh) 调用子模块里的 `convert_hf_to_gguf.py`，把 HuggingFace safetensors 权重转成 GGUF：

```bash
bash tools/run_safeT_to_gguf.sh
```

脚本内通过 `--outtype bf16` 指定精度。**模型输入/输出路径目前为硬编码**，使用前请按需修改。

### 4. 运行推理

运行脚本会调用编译产物 `./build/bin/llama-cli`，按场景直接选一个脚本：

```bash
bash tools/run_llamacpp_tensor.sh       # 模式 B：多卡张量并行，不依赖 NCCL
bash tools/run_llamacpp_tensor_nccl.sh  # 模式 A：多卡张量并行，AllReduce 走 NCCL
bash tools/run_llamacpp_layer.sh        # 模式 C：多卡层切分 pipeline
bash tools/run_llamacpp_single.sh       # 模式 D：单卡
```

多卡张量并行（`-sm tensor`）的核心区别在 **AllReduce 后端**，由环境变量 `GGML_CUDA_ALLREDUCE` 控制：

| 模式 | 脚本 | 切分方式 | AllReduce 后端 | 是否依赖 NCCL | PTX 可见性 |
| --- | --- | --- | --- | --- | --- |
| A | [tools/run_llamacpp_tensor_nccl.sh](tools/run_llamacpp_tensor_nccl.sh) | 张量并行 `-sm tensor` | `nccl`：NVIDIA NCCL 库 | 是 | AllReduce 在闭源 `libnccl.so`，**dump 不到** |
| **B2（常用）** | [tools/run_llamacpp_tensor.sh](tools/run_llamacpp_tensor.sh) | 张量并行 `-sm tensor` | `none`：meta-backend butterfly reduction | 否 | 不走 NCCL，相关 CUDA 算子可 dump |
| B1 | [tools/run_llamacpp_tensor.sh](tools/run_llamacpp_tensor.sh) | 张量并行 `-sm tensor` | `internal`：llama.cpp/ggml 自带 CUDA kernel（`allreduce.cu`） | 否 | AllReduce 为自研 kernel，**可 dump** |
| C | [tools/run_llamacpp_layer.sh](tools/run_llamacpp_layer.sh) | 层切分（pipeline） | 无 reduce | 否 | 无 AllReduce |
| D | [tools/run_llamacpp_single.sh](tools/run_llamacpp_single.sh) | 单卡 | 无 | 否 | 无 AllReduce |

> [tools/run_llamacpp_tensor.sh](tools/run_llamacpp_tensor.sh) 默认启用 B2：`CUDA_VISIBLE_DEVICES=2,3,4,5` + `GGML_CUDA_ALLREDUCE=none`。如需测试 B1，可改成脚本中注释掉的双卡 `internal` 配置：`CUDA_VISIBLE_DEVICES=2,3` + `GGML_CUDA_ALLREDUCE=internal`。

公共要点：

- `-ngl all`：将全部层 offload 到 GPU。
- `GGML_CUDA_P2P=1`：开启 GPU 间 P2P（多卡模式需要）。
- `-sm tensor` 会**自动开启 flash_attn**（`SPLIT_MODE_TENSOR` 的硬性要求，见 [llama-context.cpp:3513](llama.cpp/src/llama-context.cpp#L3513)），无需手动加参数。该 flash attention 是 **llama.cpp 自研的 CUDA kernel**（`fattn*.cu/cuh`），**非**外部 Dao-AILab flash-attention 库，因此也在可 dump 范围内。
- `LD_LIBRARY_PATH` 指向 `nccl/build/lib` 仅模式 A 需要；模式 B/C/D 可忽略。
- `model_path` 在脚本中给了默认值，也可运行时用 `MODEL_PATH=/path/to/model.gguf bash tools/run_llamacpp_tensor.sh` 覆盖。

---

## 脚本说明

| 脚本 | 作用 |
| --- | --- |
| [tools/build_llamacpp.sh](tools/build_llamacpp.sh) | CUDA 配置 + 编译 llama.cpp |
| [tools/run_safeT_to_gguf.sh](tools/run_safeT_to_gguf.sh) | safetensors → GGUF 模型转换 |
| [tools/run_llamacpp_tensor.sh](tools/run_llamacpp_tensor.sh) | llama-cli 推理：多卡张量并行（不依赖 NCCL） |
| [tools/run_llamacpp_tensor_nccl.sh](tools/run_llamacpp_tensor_nccl.sh) | llama-cli 推理：多卡张量并行（NCCL） |
| [tools/run_llamacpp_layer.sh](tools/run_llamacpp_layer.sh) | llama-cli 推理：多卡层切分 |
| [tools/run_llamacpp_single.sh](tools/run_llamacpp_single.sh) | llama-cli 推理：单卡 |

> 注意：脚本中的模型路径、`conda activate torch251`、CUDA 路径等均为当前环境的硬编码值，迁移到其它机器时需相应调整。

---

## 子模块维护

```bash
# 查看子模块当前提交
git submodule status

# 更新 llama.cpp 到上游最新
cd llama.cpp && git fetch && git checkout <tag-or-commit> && cd ..
git add llama.cpp && git commit -m "bump llama.cpp"
```

---

## License

见 [LICENSE](LICENSE)。llama.cpp 子模块遵循其自身的开源协议。
