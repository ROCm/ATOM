# MiniMax-M3 C32：官方 v0.1.26 Gluon 方案

当前方案以 **官方 AITER v0.1.26 的 Gluon unified attention 和 split 优化**为实现来源，
只补 ATOM 原生 KV 格式兼容。target 保持 per-token FP8 scales，draft 保持 scalar scale，
Q 保持 BF16。无需转换整个 KV 池，也不创建连续 KV workspace。

## 代码来源

- ATOM 分支：`mengqing/minimax-m3-gluon-c32`。
- 官方 tag：`v0.1.26`，提交 `84077b3c0461dac1f0f3f8b9590a9282e2035a2f`。
- 实验 AITER 分支：`ROCm/aiter:mengqing/m3-gluon-v0126-native`。
- CI 固定 AITER 提交：`4fe2800a5c252adff8ee28d199bea66f8b18efac`。

代码分成两个提交，方便审查：

1. `86f7d57f`：从官方 tag **逐字节原样导入** unified dispatcher、gfx950 Gluon kernel、
   Triton attention/reduce kernel，以及 gfx950 的 Gluon/Triton unified-attention 配置，共五个文件。
2. `4fe2800a`：增加必要的 KV 格式兼容和数值测试。

其他 AITER 算子保持原基线 `5eb866a95d66a38d64eecf151ac4ef3567a07ac5`，其中包含此前的
scatter 越界修复。因此这是采用官方 attention 实现的定向替换，并未整体升级其他算子。

旧实验固定的 M128、整页 tile、单缓冲配置已移除。split 数量、tile、warps 和缓冲数量均由
官方代码与配置选择。单 token query 使用官方 decode 网格；这里说的是 cached-prefill 内部的
kernel 网格，ATOM 普通 decode 的 FlyDSL 路径继续沿用原配置。

## 必要的 KV 格式适配

官方 gfx950 Gluon 原接口要求 Q/KV 同 dtype，只提供 scalar descale。ATOM target 实际使用
BF16 Q、FP8 KV 和按物理 token 存储的 scales，所以增加 `k_scale_cache`、`v_scale_cache`：

```text
scale shape = [physical block, KV head, token in block]
```

适配范围限定在 M3 的 shuffled KV、head size 128、page size 128。输入不支持或 Gluon 不可用时
直接报错，避免静默切回其他 attention。

`ScaledKVLoader` 包装官方整页或子页 async loader。它按逻辑 token 查真实 block table，读取
物理 page/token 的 scale，再把 FP8 数据反量化为 BF16 操作数。tile 可以小于物理 page，
例如 tile64 对应 page128，不再强制 tile 等于 page。

双缓冲循环仅额外传入 scale 索引所需的 `tile_index`。循环边界、barrier、split 边界、
attention 数学计算和 reduce 保持官方实现。draft 的 scalar 使用零 stride 视图，底层仍然是
一个 4 字节 FP32 值。KV writer、量化方式和原生 KV 池均不改变。

## C32 对照配置

性能基线仍是 [gather sweep 39034151209](https://github.com/ROCm/ATOM/actions/runs/39034151209)
的 C32，job `117400829563`，ATOM 提交 `ebeb0f925289619cb0630b3d70a6f600412af6bb`。
该运行作为性能对照；新 kernel 的实现来源是官方 v0.1.26。

| 配置 | 值 |
| --- | --- |
| Case | `minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32` |
| 镜像 | `rocm/atom-dev:nightly_202610141644` |
| 拓扑 | P TP4 + D TP4，PP1，共 8 张 GPU |
| Draft | `Inferact/MiniMax-M3-EAGLE3`，3 个 speculative tokens |
| KV | FP8，block size 128，原有池布局与容量预算 |
| 服务 | 80 sequences，32768 batched tokens，GPU utilization 0.95 |
| LMCache | 每个 prefill rank 256 GiB |
| 负载 | AgentX C32，1M context limit，3600 秒，每 lane 预热 10 次 |
| 评估 | `run_eval=false` |
| 调度 | `atomesh-cicd`、`amd-spur`，原有自动选节点规则 |

替换 target/draft 的 cached dense prefill。未命中缓存的 prefill、profiling、普通 decode、
router、显存配置和负载保持该 C32 基线。matrix 门禁要求除了 AITER attention 源码 SHA 外，
所有配置都与归档基线一致。

## 代码与验证

- [ATOM 适配层](../../atom/model_ops/gluon_prefill.py)。
- [attention 接入位置](../../atom/model_ops/attention_mha.py)。
- [固定源码与 GPU 门禁](../../.github/scripts/atomesh/prepare_scatter_aiter_source.sh)。
- [官方源码一致性检查](../../.github/scripts/atomesh/check_m3_gluon_upstream.py)。
- [官方源码指纹](../../.github/scripts/atomesh/m3_gluon_v0126_source.json)。
- [实际 writer/adapter 数值检查](../../.github/scripts/atomesh/check_m3_gluon_adapter.py)。

官方源码检查在 GPU 排队前和 GPU 作业中各执行一次，检查调优配置、split、attention 数学计算、
双缓冲循环和 reduce。循环比对仅允许额外传入 `tile_index`，防止旧实验调优重新混入。

本地已通过：6 组实际 adapter/dispatcher 的 gfx950 离线编译、16 项 CPU adapter 测试、
4 项 CPU 页 fixture 测试、ATOMesh CPU 测试（194 项，1 项跳过），以及新增源码门禁回归检查。
编译参数确认使用 tile64、双缓冲，覆盖单 token/prefill 网格、per-token/scalar scales，
以及小池和超过 2 GiB 池的 buffer/global loads。

这些结果不代表 GPU 数值验证已完成。CI 在加载模型前执行 17 项 public-wrapper 和 9 项实际
ATOM writer/adapter 数值检查。当前还没有新方案的端到端性能结果。
