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
│   ├── run_safeT_to_gguf.sh# HF safetensors → GGUF 模型转换
│   └── run_llamacpp.sh     # 用 llama-cli 跑推理（单卡 / 多卡）
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
- 多卡场景：NCCL（编译时开启 `-DGGML_CUDA_NCCL=ON`）

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

用 [tools/run_llamacpp.sh](tools/run_llamacpp.sh) 调用编译产物 `./build/bin/llama-cli`：

```bash
bash tools/run_llamacpp.sh
```

脚本提供两种模式（默认开启多卡，单卡为注释段，可按需切换）：

- **单卡**：`CUDA_VISIBLE_DEVICES=0`
- **多卡**：`CUDA_VISIBLE_DEVICES=0,1` 并配合以下环境变量：
  - `GGML_CUDA_P2P=1`：开启 GPU 间 P2P
  - `CUDA_SCALE_LAUNCH_QUEUES=4x`：扩展 launch queue

`-ngl all` 表示将全部层 offload 到 GPU。**`model_path` 为硬编码**，使用前请改成你自己的 GGUF 路径。

---

## 脚本说明

| 脚本 | 作用 |
| --- | --- |
| [tools/build_llamacpp.sh](tools/build_llamacpp.sh) | CUDA 配置 + 编译 llama.cpp |
| [tools/run_safeT_to_gguf.sh](tools/run_safeT_to_gguf.sh) | safetensors → GGUF 模型转换 |
| [tools/run_llamacpp.sh](tools/run_llamacpp.sh) | llama-cli 推理（单卡/多卡） |

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
