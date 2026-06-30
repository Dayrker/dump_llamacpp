#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/dump_llamacpp_tensor.sh [internal|none|all]

Environment:
  MODEL_PATH             Model path recorded into the manifest.
  CUDA_VISIBLE_DEVICES   Device list recorded into the manifest.
  LLAMACPP_DIR           llama.cpp checkout; default: ./llama.cpp.
  DUMP_ROOT              Dump output root; default: ./dump/llamacpp_tensor.
  CUOBJDUMP              cuobjdump path; default: /usr/local/cuda/bin/cuobjdump.

Outputs:
  dump/llamacpp_tensor/internal/...
  dump/llamacpp_tensor/none/...
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
llamacpp_dir=${LLAMACPP_DIR:-"$repo_root/llama.cpp"}
dump_root=${DUMP_ROOT:-"$repo_root/dump/llamacpp_tensor"}
model_path=${MODEL_PATH:-/home/zhangchen/PTX/model/Qwen3-32B/Qwen3-32B-bf16.gguf}
mode_arg=${1:-${GGML_CUDA_ALLREDUCE:-none}}

case "$mode_arg" in
  -h|--help)
    usage
    exit 0
    ;;
  internal|none|all)
    ;;
  *)
    die "mode must be one of: internal, none, all"
    ;;
esac

[[ -d "$llamacpp_dir" ]] || die "llama.cpp directory not found: $llamacpp_dir"
[[ -f "$llamacpp_dir/build/compile_commands.json" ]] || die "missing $llamacpp_dir/build/compile_commands.json; build llama.cpp first"

cuobjdump=${CUOBJDUMP:-/usr/local/cuda/bin/cuobjdump}
[[ -x "$cuobjdump" ]] || die "cuobjdump not found or not executable: $cuobjdump"

git_head() {
  git -C "$llamacpp_dir" describe --always --dirty 2>/dev/null || git -C "$llamacpp_dir" rev-parse --short HEAD 2>/dev/null || echo unknown
}

devices_for_mode() {
  local mode=$1
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    printf '%s\n' "$CUDA_VISIBLE_DEVICES"
  elif [[ "$mode" == "internal" ]]; then
    printf '2,3\n'
  else
    printf '2,3,4,5\n'
  fi
}

compile_command_for() {
  local source=$1
  local output=$2

  if command -v jq >/dev/null 2>&1; then
    jq -r --arg file "$source" '
      .[] | select(.file == $file) |
      "directory=\(.directory)\ncommand=\(.command)"
    ' "$llamacpp_dir/build/compile_commands.json" > "$output"
  else
    python3 - "$llamacpp_dir/build/compile_commands.json" "$source" > "$output" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

for item in data:
    if item.get("file") == sys.argv[2]:
        print(f"directory={item.get('directory', '')}")
        print(f"command={item.get('command', '')}")
        break
PY
  fi

  [[ -s "$output" ]] || die "compile command not found for $source"
}

object_for() {
  local source_base=$1
  local object
  object=$(find "$llamacpp_dir/build" -path '*ggml-cuda*' -type f -name "${source_base}.o" | head -n 1)
  [[ -n "$object" ]] || die "object not found for $source_base under $llamacpp_dir/build"
  printf '%s\n' "$object"
}

write_entries() {
  local ptx=$1
  local entries=$2
  local demangled=$3

  awk '
    /^[[:space:]]*\.entry[[:space:]]+/ {
      name = $2
      sub(/\(.*/, "", name)
      print name
    }
  ' "$ptx" > "$entries"

  if command -v c++filt >/dev/null 2>&1; then
    c++filt < "$entries" > "$demangled"
  else
    cp "$entries" "$demangled"
  fi
}

write_numbered_snippet() {
  local source=$1
  local start=$2
  local end=$3
  local output=$4

  awk -v start="$start" -v end="$end" '
    NR >= start && NR <= end {
      printf "/* L%-5d */ %s\n", NR, $0
    }
  ' "$source" > "$output"
}

write_symbols() {
  local object=$1
  local all_symbols=$2
  local filtered=$3
  local pattern=$4

  if command -v nm >/dev/null 2>&1; then
    nm -a "$object" > "$all_symbols" 2>/dev/null || true
  else
    : > "$all_symbols"
  fi

  if [[ -s "$all_symbols" ]]; then
    grep -E "$pattern" "$all_symbols" > "$filtered" || true
  else
    : > "$filtered"
  fi
}

write_manifest() {
  local mode=$1
  local source=$2
  local object=$3
  local ptx=$4
  local manifest=$5
  local devices=$6

  {
    printf 'mode=%s\n' "$mode"
    printf 'generated_at=%s\n' "$(date -Is)"
    printf 'repo_root=%s\n' "$repo_root"
    printf 'llamacpp_dir=%s\n' "$llamacpp_dir"
    printf 'git_head=%s\n' "$(git_head)"
    printf 'model_path=%s\n' "$model_path"
    printf 'GGML_CUDA_ALLREDUCE=%s\n' "$mode"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$devices"
    printf 'source=%s\n' "$source"
    printf 'object=%s\n' "$object"
    printf 'ptx=%s\n' "$ptx"
    printf 'cuobjdump=%s\n' "$cuobjdump"
  } > "$manifest"
}

write_run_command() {
  local mode=$1
  local devices=$2
  local output=$3

  {
    printf 'cd "%s"\n' "$llamacpp_dir"
    printf 'CUDA_VISIBLE_DEVICES=%s \\\n' "$devices"
    printf 'GGML_CUDA_P2P=1 \\\n'
    printf 'GGML_CUDA_ALLREDUCE=%s \\\n' "$mode"
    printf './build/bin/llama-cli -m %q -ngl all -sm tensor\n' "$model_path"
  } > "$output"
}

write_internal_docs() {
  local dir=$1

  cat > "$dir/IMPLEMENTATION.md" <<'EOF'
# internal AllReduce CUDA 实现

主要源码文件：

- `src/allreduce.cu`：两卡 internal AllReduce pipeline 的实现。
- `src/allreduce.cuh`：internal AllReduce 对外声明。
- `src/ggml-cuda-comm.cu`：带源码行号注释的 CUDA backend communication dispatch 片段。

关键 CUDA/runtime 函数：

- `ggml_backend_cuda_comm_init`：读取 `GGML_CUDA_ALLREDUCE`。
- `ggml_backend_cuda_comm_init_internal`：创建 internal pipeline。
- `ggml_backend_cuda_comm_allreduce_internal`：检查 tensor 的 shape/type/device 状态。
- `ggml_cuda_ar_allreduce`：internal AllReduce 的顶层 dispatcher。
- `ggml_cuda_ar_kernel<T_dst, T_wire>`：小规模或分块 reduction 使用的 CUDA kernel。
- `ggml_cuda_ar_allreduce_copy_impl<T_src, T_dst>`：D2H/H2D copy-engine 路径。
- `ggml_cuda_ar_add_kernel<T_dst, T_src>`：copy path 中使用的 device-local add kernel。

PTX:

- `ptx/allreduce.ptx`：从 CUDA object 中提取出的 PTX。
- `ptx/allreduce.ptx.entries`：mangled PTX entry functions。
- `ptx/allreduce.ptx.entries.demangled`：如果存在 `c++filt`，这里保存 demangled PTX entry functions。
EOF

  cat > "$dir/LINKAGE.md" <<'EOF'
# llama.cpp Tensor AllReduce 链路（internal）

## 实际 llama.cpp / ggml 链路

1. `tools/run_llamacpp_tensor.sh` 用 `-sm tensor` 和 `GGML_CUDA_ALLREDUCE=internal` 启动 `llama.cpp/build/bin/llama-cli`。
2. `common/arg.cpp` 把 `-sm tensor` 解析成 `LLAMA_SPLIT_MODE_TENSOR`。
3. `common/common.cpp` 把 CLI split mode 写入 `llama_model_params`。
4. `llama-model.cpp` 创建 tensor-split CUDA buffer types，并记录 model 的 tensor split 状态。
5. Tensor split mode 会在多个 CUDA backends 之上使用 ggml 的 meta backend。
6. `ggml_backend_meta_context` 通过 `ggml_backend_dev_get_proc_address` 解析 CUDA communication hooks：`ggml_backend_comm_init`、`ggml_backend_comm_free` 和 `ggml_backend_comm_allreduce_tensor`。
7. `ggml_backend_cuda_comm_init` 读取 `GGML_CUDA_ALLREDUCE=internal`，并安装 `ggml_backend_cuda_comm_try_allreduce_internal`。
8. 每次 tensor allreduce 时，`ggml_backend_cuda_comm_allreduce_tensor` 会调用这个 function pointer。
9. `ggml_backend_cuda_comm_allreduce_internal` 检查两卡、contiguous、F32/F16/BF16 tensor set，然后调用 `ggml_cuda_ar_allreduce`。
10. `ggml_cuda_ar_allreduce` 在两条 CUDA 路径中选择：小规模 reduction 启动 `ggml_cuda_ar_kernel<T_dst, T_wire>`；大规模 reduction 使用 `cudaMemcpyAsync` D2H/H2D chunks，再配合 `ggml_cuda_ar_add_kernel<T_dst, T_src>`。
11. 这些 kernel 生成的 PTX entries 位于 `ptx/allreduce.ptx`。

## Torch / ATen / runtime 关系

这里的 PTX 不是通过 PyTorch、Torch dispatcher 或 ATen 触达的。它是编译进 CUDA backend 的 llama.cpp / ggml 原生代码。实际归属链路是：

`llama-cli -> llama.cpp graph -> ggml scheduler/meta backend -> ggml CUDA backend -> CUDA runtime API -> CUDA driver -> GPU kernel PTX/SASS`.

作为对照，典型 PyTorch CUDA op 链路是：

`torch Python API -> c10 dispatcher -> ATen native CUDA implementation -> cudaLaunchKernel/cudaMemcpyAsync in libcudart -> libcuda driver -> GPU kernel PTX/SASS`.

如果是 PyTorch distributed allreduce，常见链路是：

`torch.distributed.all_reduce -> c10d ProcessGroup -> ProcessGroupNCCL or another backend -> NCCL/CUDA runtime -> CUDA driver -> GPU`.

所以 `torch -> ATen -> runtime` 这条链路可以作为对照，但它不是本次 dump 的 PTX 实际使用链路。
EOF
}

write_none_docs() {
  local dir=$1

  cat > "$dir/IMPLEMENTATION.md" <<'EOF'
# none / Meta-Backend Butterfly 实现

`GGML_CUDA_ALLREDUCE=none` 会关闭 CUDA backend 的专用 AllReduce 实现。CUDA comm hook 返回 `false`，于是 meta backend 会执行通用 butterfly reduction。

主要源码文件：

- `src/ggml-cuda-none-comm.cu`：`none` 映射到 false-returning hook 的 CUDA comm path。
- `src/ggml-backend-meta-allreduce.cpp`：meta-backend fallback allreduce 实现。
- `src/binbcast.cu`：`GGML_OP_ADD` 使用的 CUDA binary broadcast/add kernels。
- `src/binbcast.cuh`：binary broadcast 声明。
- `src/ggml-cuda-op-dispatch.cu`：把 `GGML_OP_ADD` dispatch 到 `ggml_cuda_op_add` 的 CUDA op dispatcher。

关键 CUDA/runtime 组件：

- `ggml_backend_tensor_copy_async`：fallback 使用的 cross-backend tensor copy。
- `GGML_OP_ADD`：用于累加 peer data 的辅助 graph node。
- `ggml_cuda_op_add`：`GGML_OP_ADD` 选中的 CUDA 实现。
- 带 `op_add` 的 `k_bin_bcast` / `k_bin_bcast_unravel`：普通 CUDA add kernels。

PTX:

- `ptx/binbcast.ptx`：CUDA binary broadcast/add object 的 PTX。
- `ptx/binbcast.ptx.entries`：所有 mangled PTX entry functions。
- `ptx/binbcast.ptx.entries.demangled`：如果存在 `c++filt`，这里保存所有 demangled PTX entry functions。
- `ptx/binbcast.ptx.add-entries*`：筛选到 `op_add` specializations 的 entries。

这个模式没有专用 AllReduce PTX；真正相关的 PTX 是 meta-backend fallback 调用的普通 CUDA add path。
EOF

  cat > "$dir/LINKAGE.md" <<'EOF'
# llama.cpp Tensor AllReduce 链路（none）

## 实际 llama.cpp / ggml 链路

1. `tools/run_llamacpp_tensor.sh` 用 `-sm tensor` 和 `GGML_CUDA_ALLREDUCE=none` 启动 `llama.cpp/build/bin/llama-cli`。
2. `common/arg.cpp` 把 `-sm tensor` 解析成 `LLAMA_SPLIT_MODE_TENSOR`。
3. `common/common.cpp` 把 CLI split mode 写入 `llama_model_params`。
4. `llama-model.cpp` 创建 tensor-split CUDA buffer types，并记录 model 的 tensor split 状态。
5. Tensor split mode 会在多个 CUDA backends 之上使用 ggml 的 meta backend。
6. `ggml_backend_meta_context` 通过 `ggml_backend_dev_get_proc_address` 解析 CUDA communication hooks：`ggml_backend_comm_init`、`ggml_backend_comm_free` 和 `ggml_backend_comm_allreduce_tensor`。
7. `ggml_backend_cuda_comm_init` 读取 `GGML_CUDA_ALLREDUCE=none`，并安装 `ggml_backend_cuda_comm_try_allreduce_butterfly`。
8. `ggml_backend_cuda_comm_try_allreduce_butterfly` 按设计返回 `false`。
9. meta backend 开始执行 `allreduce_fallback`。
10. fallback 用 `ggml_backend_tensor_copy_async` 把每个 shard copy 到 butterfly peer，然后在 destination backend 上构造一个只有 `GGML_OP_ADD` 的辅助 graph。
11. CUDA dispatch 通过 `ggml_cuda_op_add` 处理这个 add，并启动使用 `op_add` 的 `k_bin_bcast` / `k_bin_bcast_unravel` specializations。
12. 相关 ordinary-add PTX 位于 `ptx/binbcast.ptx`；这个模式没有专用 allreduce CUDA kernel。

## Torch / ATen / runtime 关系

这里的 PTX 不是通过 PyTorch、Torch dispatcher 或 ATen 触达的。它是编译进 CUDA backend 的 llama.cpp / ggml 原生代码。实际归属链路是：

`llama-cli -> llama.cpp graph -> ggml scheduler/meta backend -> ggml CUDA backend -> CUDA runtime API -> CUDA driver -> GPU kernel PTX/SASS`.

作为对照，典型 PyTorch CUDA op 链路是：

`torch Python API -> c10 dispatcher -> ATen native CUDA implementation -> cudaLaunchKernel/cudaMemcpyAsync in libcudart -> libcuda driver -> GPU kernel PTX/SASS`.

如果是 PyTorch distributed allreduce，常见链路是：

`torch.distributed.all_reduce -> c10d ProcessGroup -> ProcessGroupNCCL or another backend -> NCCL/CUDA runtime -> CUDA driver -> GPU`.

所以 `torch -> ATen -> runtime` 这条链路可以作为对照，但它不是本次 dump 的 PTX 实际使用链路。
EOF
}

dump_internal() {
  local mode=internal
  local dir="$dump_root/$mode"
  local source="$llamacpp_dir/ggml/src/ggml-cuda/allreduce.cu"
  local header="$llamacpp_dir/ggml/src/ggml-cuda/allreduce.cuh"
  local object
  local devices

  object=$(object_for "$(basename "$source")")
  devices=$(devices_for_mode "$mode")

  mkdir -p "$dir/src" "$dir/build" "$dir/ptx" "$dir/symbols"
  rm -f "$dir/src/ggml-cuda-comm.txt"
  rm -f "$dir/ptx/allreduce.entries.txt" "$dir/ptx/allreduce.entries.demangled.txt"

  cp "$source" "$dir/src/allreduce.cu"
  cp "$header" "$dir/src/allreduce.cuh"
  write_numbered_snippet "$llamacpp_dir/ggml/src/ggml-cuda/ggml-cuda.cu" 1150 1445 "$dir/src/ggml-cuda-comm.cu"

  compile_command_for "$source" "$dir/build/allreduce.compile-command.txt"
  printf 'source=%s\nobject=%s\nextracted_with=%s --dump-ptx\n' "$source" "$object" "$cuobjdump" > "$dir/build/allreduce.ptx-source.txt"

  "$cuobjdump" --dump-ptx "$object" > "$dir/ptx/allreduce.ptx"
  write_entries "$dir/ptx/allreduce.ptx" "$dir/ptx/allreduce.ptx.entries" "$dir/ptx/allreduce.ptx.entries.demangled"
  write_symbols "$object" "$dir/symbols/allreduce.symbols.txt" "$dir/symbols/allreduce.filtered-symbols.txt" 'ggml_cuda_ar_|allreduce'

  write_manifest "$mode" "$source" "$object" "$dir/ptx/allreduce.ptx" "$dir/MANIFEST.txt" "$devices"
  write_run_command "$mode" "$devices" "$dir/RUN_COMMAND.txt"
  write_internal_docs "$dir"
}

dump_none() {
  local mode=none
  local dir="$dump_root/$mode"
  local source="$llamacpp_dir/ggml/src/ggml-cuda/binbcast.cu"
  local header="$llamacpp_dir/ggml/src/ggml-cuda/binbcast.cuh"
  local object
  local devices

  object=$(object_for "$(basename "$source")")
  devices=$(devices_for_mode "$mode")

  mkdir -p "$dir/src" "$dir/build" "$dir/ptx" "$dir/symbols"
  rm -f "$dir/src/ggml-cuda-none-comm.txt" "$dir/src/ggml-backend-meta-allreduce.txt" "$dir/src/ggml-cuda-op-dispatch.txt"
  rm -f "$dir/ptx/binbcast.entries.txt" "$dir/ptx/binbcast.entries.demangled.txt"
  rm -f "$dir/ptx/binbcast.add-entries.txt" "$dir/ptx/binbcast.add-entries.demangled.txt"

  cp "$source" "$dir/src/binbcast.cu"
  cp "$header" "$dir/src/binbcast.cuh"
  write_numbered_snippet "$llamacpp_dir/ggml/src/ggml-cuda/ggml-cuda.cu" 1324 1438 "$dir/src/ggml-cuda-none-comm.cu"
  write_numbered_snippet "$llamacpp_dir/ggml/src/ggml-backend-meta.cpp" 2069 2217 "$dir/src/ggml-backend-meta-allreduce.cpp"
  write_numbered_snippet "$llamacpp_dir/ggml/src/ggml-cuda/ggml-cuda.cu" 2818 2835 "$dir/src/ggml-cuda-op-dispatch.cu"

  compile_command_for "$source" "$dir/build/binbcast.compile-command.txt"
  printf 'source=%s\nobject=%s\nextracted_with=%s --dump-ptx\n' "$source" "$object" "$cuobjdump" > "$dir/build/binbcast.ptx-source.txt"

  "$cuobjdump" --dump-ptx "$object" > "$dir/ptx/binbcast.ptx"
  write_entries "$dir/ptx/binbcast.ptx" "$dir/ptx/binbcast.ptx.entries" "$dir/ptx/binbcast.ptx.entries.demangled"
  grep 'op_add' "$dir/ptx/binbcast.ptx.entries" > "$dir/ptx/binbcast.ptx.add-entries" || true
  if command -v c++filt >/dev/null 2>&1; then
    c++filt < "$dir/ptx/binbcast.ptx.add-entries" > "$dir/ptx/binbcast.ptx.add-entries.demangled"
  else
    cp "$dir/ptx/binbcast.ptx.add-entries" "$dir/ptx/binbcast.ptx.add-entries.demangled"
  fi
  write_symbols "$object" "$dir/symbols/binbcast.symbols.txt" "$dir/symbols/binbcast.filtered-symbols.txt" 'k_bin_bcast|op_add|ggml_cuda_op_add'
  grep 'op_add' "$dir/symbols/binbcast.symbols.txt" > "$dir/symbols/binbcast.add-symbols.txt" || true

  write_manifest "$mode" "$source" "$object" "$dir/ptx/binbcast.ptx" "$dir/MANIFEST.txt" "$devices"
  write_run_command "$mode" "$devices" "$dir/RUN_COMMAND.txt"
  write_none_docs "$dir"
}

mkdir -p "$dump_root"

case "$mode_arg" in
  internal)
    dump_internal
    ;;
  none)
    dump_none
    ;;
  all)
    dump_internal
    dump_none
    ;;
esac

echo "dump written to: $dump_root"
