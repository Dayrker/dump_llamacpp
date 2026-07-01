# llama.cpp 分布式 / 多卡路径梳理

本文记录当前仓库里 llama.cpp 多卡推理的主要路径，重点是 `-sm tensor`
张量并行和 AllReduce 后端选择。这里的“分布式”主要指 **单进程多 GPU /
多 backend device**，不是 PyTorch DDP 那种多进程训练框架。

## 1. 模式对照

当前脚本把多卡运行分成几类：

| 模式 | 脚本 | llama.cpp 参数 | AllReduce 后端 | 说明 |
| --- | --- | --- | --- | --- |
| A | `tools/run_dump/run_llamacpp_tensor_nccl.sh` | `-ngl all -sm tensor` | `GGML_CUDA_ALLREDUCE=nccl` | 张量并行，AllReduce 走 NCCL |
| B1 | `tools/run_dump/run_llamacpp_tensor.sh` | `-ngl all -sm tensor` | `GGML_CUDA_ALLREDUCE=internal` | 张量并行，AllReduce 走 ggml CUDA 自带 `allreduce.cu`，只支持 2 卡 |
| B2 | `tools/run_dump/run_llamacpp_tensor.sh` | `-ngl all -sm tensor` | `GGML_CUDA_ALLREDUCE=none` | 张量并行，禁用 CUDA 专用 AllReduce，走 meta-backend generic butterfly reduction |
| C | `tools/run_dump/run_llamacpp_layer.sh` | `-ngl all` | 无 | 默认 `layer` split，按层切分 / pipeline，不需要 AllReduce |
| D | `tools/run_dump/run_llamacpp_single.sh` | `-ngl all` | 无 | 单卡 |

### 1.1 run_llamacpp_tensor.sh
`tools/run_dump/run_llamacpp_tensor.sh` 里 B1/B2 模式的转换：
```
# B1只支持 2 卡张量并行, 如果要多卡张量并行, 会默认lower到B2模式;
internal AllReduce init failed -> falling back to meta-backend butterfly
```

核心源码位置：

- `llama.cpp/ggml/src/ggml-cuda/allreduce.cu:396`：
  <a href="#appendix-b1-two-gpu-limit" style="color:#0969da;">`ggml_cuda_ar_pipeline_init()`</a> 要求 `n_devices == 2`。
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1349`：
  `internal` 初始化失败后调用
  <a href="#appendix-internal-fallback" style="color:#0969da;">`ggml_backend_cuda_comm_init_none()`</a>。
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1329`：
  `GGML_CUDA_ALLREDUCE=none` 会让 CUDA backend 返回 `false`，把 AllReduce
  交给 <a href="#appendix-none-triggers-meta-fallback" style="color:#0969da;">Meta backend</a>。
- `llama.cpp/ggml/src/ggml-backend-meta.cpp:2069`：
  backend-specific AllReduce 不可用时，使用
  <a href="#appendix-meta-allreduce-fallback" style="color:#0969da;">generic fallback</a>。
- `llama.cpp/ggml/src/ggml-backend-meta.cpp:2146`：
  generic fallback 的核心是
  <a href="#appendix-meta-butterfly" style="color:#0969da;">butterfly reduction</a>。

## 2. split mode 入口

命令行参数在 `llama.cpp/common/arg.cpp` 里解析：

```text
-sm, --split-mode {none,layer,row,tensor}
```

对应枚举在 `llama.cpp/include/llama.h`：

```cpp
LLAMA_SPLIT_MODE_NONE   = 0, // single GPU
LLAMA_SPLIT_MODE_LAYER  = 1, // split layers and KV across GPUs
LLAMA_SPLIT_MODE_ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
LLAMA_SPLIT_MODE_TENSOR = 3,
```

几个模式的差异：

- `none`：只用一张 GPU，模型不跨 GPU 切分。
- `layer`：默认模式，按 layer 分配到多张 GPU，KV 也跟随切分，整体更像 pipeline。
- `row`：旧路径，使用 backend 提供的 split buffer type，主要把部分矩阵按行拆。
- <span style="color:red; font-weight:700;">`tensor` 是当前张量并行路径。它会创建一个 Meta device，把多个真实 GPU device 包装成一个逻辑设备。</span>

## 3. `-sm tensor` 如何创建 Meta device

`-sm tensor` 的入口在 `llama.cpp/src/llama.cpp:124` 附近的
`llama_prepare_model_devices()`。

当 `params.split_mode == LLAMA_SPLIT_MODE_TENSOR` 时：

1. 收集可用 GPU backend device。
2. 跳过 CPU backend。
3. 调用 `ggml_backend_meta_device(...)` 创建 Meta device。
4. 把 `llama_meta_device_get_split_state` 作为回调传给 Meta device，用于描述每个 tensor 怎么切。

简化调用链：

```text
llama-cli -sm tensor
  -> common/arg.cpp 设置 LLAMA_SPLIT_MODE_TENSOR
  -> llama_prepare_model_devices()
  -> ggml_backend_meta_device(devs, n_devs, llama_meta_device_get_split_state, ...)
  -> Meta backend 包装多个 CUDA backend
```

脚本里的 `CUDA_VISIBLE_DEVICES=2,3,4,5` 会影响 llama.cpp 能看到的 CUDA
backend device 数量，因此也会影响 Meta device 里的 `n_devs`。

## 4. tensor split 状态

张量并行的切分规则集中在
`llama.cpp/src/llama-model.cpp:329` 的
`llama_meta_device_get_split_state()`。

这个函数按 tensor 名称匹配不同规则，例如：

- attention Q/K/V 权重、bias。
- fused QKV 权重、bias。
- KV cache。
- attention output。
- FFN up/gate/down。
- output weight/bias。

它返回 `ggml_backend_meta_split_state`，关键字段在
`llama.cpp/ggml/include/ggml-backend.h:376`：

- `axis`：沿哪个维度切，或是否 mirrored / partial。
- `ne[]`：每个 segment、每个 device 的元素数。
- `nr[]`：segment 重复次数。
- `n_segments`：有些 fused tensor 会拆成多个 segment 分别切。

常见 split axis：

| split axis | 含义 |
| --- | --- |
| `GGML_BACKEND_SPLIT_AXIS_0..3` | 沿某个 tensor 维度切片 |
| `GGML_BACKEND_SPLIT_AXIS_MIRRORED` | 每张卡保留完整副本 |
| `GGML_BACKEND_SPLIT_AXIS_PARTIAL` | 每张卡持有部分和，后续需要 reduce |

`tensor_split` 参数用于控制各 GPU 的比例。如果用户没有传，llama.cpp 会在
`llama.cpp/src/llama-model.cpp:1238` 根据各 device 的 free memory 自动计算比例。

## 5. `-sm tensor` 强制 flash attention

`llama.cpp/src/llama-context.cpp:3513` 有硬性检查：

```cpp
if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        return nullptr;
    }
}
```

所以 `-sm tensor` 下不用手动加 flash-attn 参数；如果用户显式禁用
flash attention，context 初始化会失败。

## 6. Meta backend 的执行和 AllReduce

Meta backend 位于 `llama.cpp/ggml/src/ggml-backend-meta.cpp`。

Meta backend 的思路是：把一个计算图拆成多个子图，在每个真实 backend 上执行；
需要跨卡汇总时，对每张卡上的对应 tensor 做 AllReduce。

关键执行逻辑在 `ggml_backend_meta_graph_compute()`：

```text
for each subgraph:
  for each backend:
    compute subgraph on that backend

  if need reduce:
    try backend-specific allreduce
    if failed / unavailable:
      run generic fallback allreduce
```

源码对应：

- `llama.cpp/ggml/src/ggml-backend-meta.cpp:1634`：
  Meta context 初始化时尝试从 backend registry 取 `ggml_backend_comm_init`。
- `llama.cpp/ggml/src/ggml-backend-meta.cpp:2069`：
  优先使用 backend-specific AllReduce。
- `llama.cpp/ggml/src/ggml-backend-meta.cpp:2205`：
  调用 `comm_allreduce(...)`。
- `llama.cpp/ggml/src/ggml-backend-meta.cpp:2208`：
  如果返回 false，就进入 `allreduce_fallback(...)`。

generic fallback 的做法：

1. 对 zero-sized / inactive slice 先写 0。
2. 通过 `ggml_backend_tensor_copy_async()` 在 backend 之间拷贝。
3. 在目标 backend 上用 `GGML_OP_ADD` 累加。
4. 对非 2 的幂的设备数，先 fold excess，再做 butterfly。
5. 最后把 reduce 结果拷回 excess backend。

butterfly 逻辑在 `llama.cpp/ggml/src/ggml-backend-meta.cpp:2146`：

```text
for offset_j = highest_power_of_two_half; offset_j >= 1; offset_j /= 2:
  j_other = j ^ offset_j
  push_data(j, j_other)
```

这就是 B2 的核心路径。

## 7. CUDA AllReduce 后端选择

CUDA backend 在 `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu` 注册了 Meta
backend 可调用的通信接口：

- `ggml_backend_comm_init`
- `ggml_backend_comm_free`
- `ggml_backend_comm_allreduce_tensor`

注册位置在 `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:5616`。

选择由环境变量 `GGML_CUDA_ALLREDUCE` 控制：

| `GGML_CUDA_ALLREDUCE` | 初始化函数 | 实际含义 |
| --- | --- | --- |
| 未设置 | Linux 默认尝试 NCCL，其它平台默认 internal | 平台默认链 |
| `nccl` | `ggml_backend_cuda_comm_init_nccl` | 使用 NCCL |
| `internal` | `ggml_backend_cuda_comm_init_internal` | 使用 ggml CUDA 自带 AllReduce pipeline |
| `none` | `ggml_backend_cuda_comm_init_none` | CUDA backend 不接管 AllReduce，交给 Meta fallback |

源码位置是 `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1404`。

`none` 不是真的“不 reduce”，而是：

```cpp
static bool ggml_backend_cuda_comm_try_allreduce_butterfly(...) {
    return false;
}
```

这个 `false` 会让 Meta backend 进入 generic butterfly fallback。

## 8. B1: internal AllReduce 为什么只支持 2 卡

B1 使用 `GGML_CUDA_ALLREDUCE=internal`，会进入：

```text
ggml_backend_cuda_comm_init_internal()
  -> ggml_cuda_ar_pipeline_init(dev_ids, n_devices)
```

硬限制在 `llama.cpp/ggml/src/ggml-cuda/allreduce.cu:396`：

```cpp
if (n_devices != 2) {
    GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                   "falling back\n", __func__, n_devices);
    return nullptr;
}
```

后面的执行函数也显式假设 2 卡：

```cpp
const int n = p->n_devices;
GGML_ASSERT(n == 2);

const int peer = 1 - i;  // valid for n == 2 only
```

也就是说，`internal` 不是一个通用 N 卡 AllReduce 实现。它现在是一个双卡
pipeline，主要依赖 CUDA kernel、host-mapped pinned memory、copy-engine
路径等机制做两个 peer 之间的 reduce。

如果你用 4 卡跑：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
GGML_CUDA_ALLREDUCE=internal \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
```

`ggml_cuda_ar_pipeline_init()` 会返回 `nullptr`，然后
`ggml_backend_cuda_comm_init_internal()` 打印 warning 并降到：

```text
meta-backend butterfly
```

对应源码在 `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1349`：

```cpp
ret->ar_pipeline = ggml_cuda_ar_pipeline_init(...);
if (ret->ar_pipeline) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
    return;
}

GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
              "falling back to meta-backend butterfly\n");
ggml_backend_cuda_comm_init_none(ret);
```

这就是脚本里“B1 只支持 2 卡；多卡会 lower 到 B2”的精确来源。

## 9. A: NCCL 路径

A 使用：

```bash
GGML_CUDA_ALLREDUCE=nccl
```

对应 `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1363` 的
`ggml_backend_cuda_comm_init_nccl()`：

```text
ncclCommInitAll(...)
  -> 成功：try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl
  -> 失败：fallback to internal AllReduce
```

真正 reduce 在 `ggml_backend_cuda_comm_allreduce_nccl()`：

- 小 tensor 直接按 FP32 reduce。
- 大 tensor 先转 BF16，用 NCCL BF16 reduce，再转回 FP32。
- inactive shard 会先置 0，保证 reduce 语义一致。

因为核心通信在 `libnccl.so` 里，A 模式下 AllReduce 内部 kernel 不属于
llama.cpp/ggml 源码，通常无法通过本仓库的 PTX dump 直接看到。

## 10. B2: none / Meta butterfly 路径

B2 使用：

```bash
GGML_CUDA_ALLREDUCE=none
```

含义是 CUDA backend 不提供专用 `comm_allreduce`，于是 Meta backend 自己做
generic fallback。这个路径不依赖 NCCL，也不使用 `allreduce.cu` 的 internal
pipeline。

优点：

- 支持多卡，不限 2 卡。
- 不依赖 NCCL。
- 跨 backend copy 和 add 是 ggml / CUDA backend 自己的路径，更容易 dump 和追踪。

代价：

- 通信模式是 generic fallback，不如 NCCL 或专门优化过的 internal 双卡路径快。
- reduce 会拆成多次 backend copy + ADD，而不是一个统一的高性能 collective。

## 11. 推荐追源码顺序

如果要从零追一遍 llama.cpp 多卡张量并行，可以按这个顺序看：

1. `llama.cpp/common/arg.cpp`：`-sm tensor` 参数解析。
2. `llama.cpp/include/llama.h`：`llama_split_mode` 枚举。
3. `llama.cpp/src/llama.cpp`：`llama_prepare_model_devices()` 创建 Meta device。
4. `llama.cpp/src/llama-model.cpp`：`llama_meta_device_get_split_state()` 决定 tensor 怎么切。
5. `llama.cpp/src/llama-context.cpp`：`SPLIT_MODE_TENSOR` 强制 flash attention。
6. `llama.cpp/ggml/src/ggml-backend-meta.cpp`：Meta backend 执行图和 generic butterfly AllReduce。
7. `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`：`GGML_CUDA_ALLREDUCE` 选择 NCCL / internal / none。
8. `llama.cpp/ggml/src/ggml-cuda/allreduce.cu`：internal 双卡 AllReduce pipeline。

## 12. 实验命令

B1 双卡 internal：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=internal \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
```

B2 多卡 butterfly：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=none \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
```

A 双卡 NCCL：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
GGML_CUDA_ALLREDUCE=nccl \
./build/bin/llama-cli -m "$model_path" -ngl all -sm tensor
```

层切分：

```bash
CUDA_VISIBLE_DEVICES=2,3 \
GGML_CUDA_P2P=1 \
./build/bin/llama-cli -m "$model_path" -ngl all
```

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 \
./build/bin/llama-cli -m "$model_path" -ngl all
```


---
---

# 附录：关键源码片段

> 正文到这里结束。下面是给上文链接用的源码摘录区，只保留关键代码和最短结论，方便核对具体实现。


## <a id="appendix-b1-two-gpu-limit"></a>附录 A：B1 internal AllReduce 只支持双卡

位置：`llama.cpp/ggml/src/ggml-cuda/allreduce.cu:396`

```cpp
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(const int * devices, size_t n_devices) {

    if (n_devices != 2) {
        GGML_LOG_DEBUG("%s: internal AllReduce only supports n_devices=2 (got %zu); "
                       "falling back\n", __func__, n_devices);
        return nullptr;
    }

    ...
}
```

后面的执行函数也写死了双卡 peer 关系：

```cpp
bool ggml_cuda_ar_allreduce(
        ggml_cuda_ar_pipeline * p,
        ggml_backend_t        * backends,
        ggml_tensor           ** tensors) {
    GGML_ASSERT(p != nullptr);

    const int n = p->n_devices;
    GGML_ASSERT(n == 2);

    ...

    for (int i = 0; i < n; ++i) {
        const int peer = 1 - i;  // valid for n == 2 only
        ...
    }

    return ok;
}
```

结论：B1 的 `internal` AllReduce 当前是双卡 pipeline，不是通用 N 卡 collective。

## <a id="appendix-internal-fallback"></a>附录 B：internal 初始化失败后 fallback 到 Meta butterfly

位置：`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1349`

```cpp
static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}
```

结论：当 `GGML_CUDA_ALLREDUCE=internal` 但设备数不是 2 时，`ar_pipeline`
初始化失败，CUDA backend 会切到 `none` 路径，让 Meta backend 做 butterfly。

## <a id="appendix-none-triggers-meta-fallback"></a>附录 C：none 模式如何触发 Meta fallback

位置：`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1329`

```cpp
static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}
```

位置：`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1345`

```cpp
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}
```

位置：`llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1418`

```cpp
} else if (env_str == "none") {
    ggml_backend_cuda_comm_init_none(ret);
}
```

结论：`GGML_CUDA_ALLREDUCE=none` 不是跳过 reduce，而是让 CUDA 专用
AllReduce 返回 `false`，从而触发 Meta backend 的 generic fallback。

## <a id="appendix-meta-allreduce-fallback"></a>附录 D：Meta backend 优先尝试专用 AllReduce

位置：`llama.cpp/ggml/src/ggml-backend-meta.cpp:2069`

```cpp
// Preferentially use backend-specific allreduce_tensor_async (e.g. NCCL for CUDA), use a generic fallback if unavailable:
auto allreduce_fallback = [&](size_t i) -> ggml_status {
    ...
};
```

位置：`llama.cpp/ggml/src/ggml-backend-meta.cpp:2195`

```cpp
if (n_backends > 1 && i < backend_ctx->n_subgraphs - 1) {
    bool backend_allreduce_success = false;
    if (backend_ctx->comm_ctx) {
        std::vector<ggml_tensor *> nodes;
        nodes.reserve(n_backends);
        for (size_t j = 0; j < n_backends; j++) {
            auto & bcj = backend_ctx->backend_configs[j];
            ggml_cgraph * cgraph_ij = bcj.cgraphs[i].cgraph_main;
            nodes.push_back(cgraph_ij->nodes[cgraph_ij->n_nodes-1]);
        }
        backend_allreduce_success = backend_ctx->comm_allreduce(backend_ctx->comm_ctx, nodes.data());
    }

    if (!backend_allreduce_success) {
        const ggml_status status = allreduce_fallback(i);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
}
```

结论：Meta backend 会先尝试 CUDA backend 注册的专用 AllReduce；只要专用
AllReduce 不存在或返回 `false`，就进入 generic fallback。

## <a id="appendix-meta-butterfly"></a>附录 E：generic fallback 的 butterfly reduction

位置：`llama.cpp/ggml/src/ggml-backend-meta.cpp:2135`

```cpp
// If n_backends is not a power of 2, fold in the excess prior to butterfly reduction:
for (size_t j_src = 2*offset_j_max; j_src < n_backends; j_src++) {
    const size_t j_dst = j_src - 2*offset_j_max;
    push_data(j_src, j_dst, i_buf);
    const ggml_status status = ggml_backend_graph_compute_async(backend_ctx->backend_configs[j_dst].backend, step_cgraphs[j_dst]);
    if (status != GGML_STATUS_SUCCESS) {
        return status;
    }
    i_buf = 1;
}

// Butterfly reduction:
for (; offset_j >= 1; offset_j /= 2) {
    std::fill(step_cgraphs.begin(), step_cgraphs.end(), nullptr);

    for (size_t j = 0; j < 2*offset_j_max; j++) {
        const size_t j_other = j ^ offset_j;
        if (j_other >= n_backends) {
            continue;
        }
        push_data(j, j_other, i_buf);
    }

    for (size_t j = 0; j < 2*offset_j_max; j++) {
        if (step_cgraphs[j] == nullptr) {
            continue;
        }
        auto & bcj = backend_ctx->backend_configs[j];
        const ggml_status status = ggml_backend_graph_compute_async(bcj.backend, step_cgraphs[j]);
        if (status != GGML_STATUS_SUCCESS) {
            return status;
        }
    }
    i_buf++;
}
```

结论：B2 的核心不是 `allreduce.cu`，而是 Meta backend 用
`ggml_backend_tensor_copy_async()` + `GGML_OP_ADD` 拼出来的 generic butterfly
reduction。
