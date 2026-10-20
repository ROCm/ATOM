# MiniMax-M3 Gluon C32 实验方案

本文是 [英文方案](minimax-m3-gluon-c32.md) 的独立中文版。实验分支为
`mengqing/minimax-m3-gluon-c32`。

## 实验目标与基线

对比 Gluon unified attention 与原有 gather 路径在 **命中 KV cache 的稠密注意力 prefill** 场景下的表现，覆盖主模型（target）和 EAGLE3 草稿模型（draft）。基线使用
[gather sweep 39034151209](https://github.com/ROCm/ATOM/actions/runs/39034151209)
中的 C32 实验，job 为 `117400829563`。实验分支从该基线运行使用的精确 ATOM 提交
`ebeb0f925289619cb0630b3d70a6f600412af6bb` 建立。

本次实验只改变上述 cached-prefill 的 attention 实现；模型、并行拓扑、负载和资源配置与基线保持一致。最终通过端到端 AgentX C32 结果评估收益。

## 接入方式

原有 cached-prefill 路径先从分页 KV cache 中 gather 出连续的 BF16 上下文，再执行 attention。新路径由 Gluon 直接读取原生分页 FP8 KV，以及 writer 返回的缩放系数：target 使用按物理 token 位置存储的 per-token scales，draft 使用标量 scale。适配层直接使用真实的 block table 和调度器提供的请求元数据，无需转换整个 KV 池。

对于单 token 或较短的 speculative query，当 query 维度的计算网格不足以充分利用 GPU 时，Gluon 可以沿上下文维度拆分计算（context split），增加可并行执行的工作量。

主要接入代码：

- [gluon_prefill.py](../../atom/model_ops/gluon_prefill.py)：Gluon cached-prefill 适配层。
- [attention_mha.py](../../atom/model_ops/attention_mha.py)：attention 路径的接入位置。

切换范围仅为主模型与草稿模型的 cached-prefill。Decode 继续使用原有 FlyDSL 路径；未命中缓存的 prefill 和显存 profiling 也继续沿用原实现。

### AITER 版本与内核组成

AITER 固定到 `ROCm/aiter:mengqing/m3-gluon-scaled-unified` 分支的提交：

```text
5f1da21fe6e89620babf1450847783b8d551ccef
```

相对于基线 AITER 提交 `5eb866a95d66a38d64eecf151ac4ef3567a07ac5`，变更范围限于 unified attention、相关 Gluon 配置及 attention 测试。该版本组合了：

- 已通过 GPU 验证的 per-token-scale 操作数加载逻辑，来自提交 `76743b4b`。
- 上游 [aiter #6336](https://github.com/ROCm/aiter/pull/6336) 中面向 speculative query 的上下文拆分优化，以及拆分边界修复。

M3 使用已验证的配置：`BLOCK_M=128`、4 warps、MFMA32、单缓冲、page size 128。单 token 的 cached-prefill 同样使用 BF16 prefill 操作数布局。

## 控制变量

| 配置项 | 基线与本实验的共同配置 |
| --- | --- |
| Case | `minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32` |
| 容器镜像 | `rocm/atom-dev:nightly_202610141644` |
| ATOM 源码 | 挂载工作区 checkout |
| 并行拓扑 | 一个 prefill TP4 实例 + 一个 decode TP4 实例，PP1，共 8 张 GPU |
| 草稿模型 | `Inferact/MiniMax-M3-EAGLE3`，3 个 speculative tokens |
| Decode | 原有 `ATOM_PA_FLYDSL=1` 实现 |
| KV cache | FP8，block size 128，沿用原有 KV 池布局和容量预算方式 |
| 服务容量参数 | 最大 80 条序列，最大 32768 batched tokens，GPU utilization 0.95 |
| LMCache | 每个 prefill rank 256 GiB，共 4 个 rank |
| 工作负载 | AgentX C32，上下文上限 1M，运行 3600 秒，每条并发 lane 预热 10 次 |
| Mesh / router | 原镜像中的实现和原有 proxy 配置 |
| 调度配置 | `atomesh-cicd`、`amd-spur`，自动选择节点，沿用原有排除列表 |
| 评估方式 | 原有性能测试配置，`run_eval=false` |

基线 artifact `12101761714` 中归档的 matrix cell 已保存到
[m3_gluon_c32_baseline.json](../../.github/scripts/atomesh/m3_gluon_c32_baseline.json)。
提交 GPU 作业前，[check_m3_gluon_c32.py](../../.github/scripts/atomesh/check_m3_gluon_c32.py)
会将生成的完整 matrix cell 与归档基线逐项比较。

该检查只允许 AITER 源码 SHA 不同；这个 SHA 会同步出现在 common、prefill、decode 和 router 的环境配置中。模型、拓扑、显存参数、工作负载、proxy、镜像发生变化，或者额外选择了其他 case，都会导致检查失败。

原有运行时监测和 bounded-gather overlay 仍然生效；Gluon 适配层会绕过 cached-prefill 的 gather 分支。去除 gather 后造成的运行时显存用量和吞吐量变化属于本次内核替换的实验结果。**实际 GPU blocks 数量需要从启动日志核实，不能仅根据配置推断。**

## 验证方式与已知限制

### 初次接入时完成的本地验证

以下为英文方案记录的初次接入验证结果，不代表后续 CI 已完成：

- ATOMesh unittest：185 个用例，测试通过，其中 1 个跳过。
- 适配层和 bounded-gather 检查：7 项通过。
- Gluon 公共入口 dispatch 测试：18 项通过，未回退到 Triton attention。
- gfx950 离线编译：5 项通过，覆盖单 token prefill、拆分与不拆分的计算网格，以及 buffer/global KV 加载路径，包括大于 2 GiB 的 KV 池。
- 生成的 matrix 与基线精确对比：通过。
- Ruff、shell 语法、workflow YAML 和 runtime overlay 兼容性检查：通过。

离线编译使用本地 `rocm/atom-dev:latest` 镜像和 Triton 3.7.0。本地 GPU 为 gfx942，因此离线编译结果不能替代在固定 CI 镜像、gfx950 硬件上的数值正确性验证。

### GPU CI 数值检查

在加载 prefill 或 decode 模型实例之前，CI 必须执行 17 个数值测试，用独立的 Torch 稠密因果 attention 实现作为参考。覆盖范围包括：

- GQA 比例 1–64，单 token 和短 speculative query。
- 混合 query tile 边界、长上下文。
- 带 stride 的 scale、共享页面。
- 不完整页面中填入异常值的无效位置，以及输出缓冲区保护区。
- 实际的 buffer/global 加载路径。

检查要求真正执行 Gluon 的拆分与不拆分路径，并拒绝回退到 Triton attention。任意一项失败都将终止作业。拆分结果的归约（split reduction）仍由 Triton 实现，这是正常执行路径的一部分。

### 性能结论的边界

当前方案不预设性能一定提升。此前经过验证的 Gluon 操作数加载实现在一个较大的 draft-prefill 微基准中慢于 gather 路径，因此必须以端到端 C32 实验结果判断实际效果。

## 运行 39280266360 失败后的修复

### GPU 测试数据修复

[该次运行](https://github.com/ROCm/ATOM/actions/runs/39280266360) 的前 14 个 GPU 数值测试通过，随后短上下文用例失败。原因是测试数据错误地让两个请求共享了一个不完整页面：给其中一个请求的无效位置填入异常值时，覆盖了另一个请求的有效 token。

测试数据现已改为只共享完整的前缀页面，并在启动内核前检查参考结果全部为有限值。新增 4 个 CPU 回归测试，在分配 GPU 之前的 hosted 配置任务中执行。17 个 GPU 数值检查仍然必须全部通过。此次 AITER 修复仅涉及测试，未修改内核实现。

### FlyDSL 能力探测与监控修复

共享的 FlyDSL 能力探测改为显式导入模块，避免依赖包级懒加载导出的函数。新增 4 个在独立进程中执行的导入测试，覆盖两种导入顺序，以及对旧版 optional-plan API 的拒绝行为。模型启动之前还会对固定版本的实际源码执行能力探测。

Benchmark 监控也已补充对 `ModelRunner` 退出的识别，即使 HTTP 前端仍能响应，也可以识别模型执行进程已经终止。

## 运行 39351509328 失败后的修复

17 个 AITER GPU 数值用例全部通过，但真实 EAGLE3 draft 在 cached-prefill 中失败。原因是基线的 `ATOM_FORCE_ATTN_TRITON=1` 使 draft 进入仅包含 RoPE 的 writer 分支，返回单个 FP32 `kv_scale`；原适配层却要求 per-token scales。池中预分配的 scale planes 并未由这个 writer 写入，不能用它们替代标量。

适配层现在分别接受 K/V 的标量或 per-token scale。对标量使用 `scale.view(()).expand(blocks, heads, page_size)` 建立零步长视图：所有位置读取同一个标量，底层指针不变，即使有 53602 个 draft blocks，存储仍为 4 字节。这里不复制 scale 数据、不读取标量到 CPU，也不改变原有量化方式。固定 AITER 的 BF16 Q + FP8 KV Gluon 路径只通过 scale-cache 接口开放，直接改传 `k_descale/v_descale` 仍不足以保证进入该 Gluon 路径。

CI 在 GPU 排队前运行 CPU adapter 回归，其中执行真实 `rope_cache` 的 Python 分支，仅替换 GPU writer。模型加载前新增 9 个真实 writer → adapter → Gluon 数值用例，覆盖 target Gemma/per-token、draft RoPE/scalar、scale=1 和 0.375、单 token/短 query/大 query、物理页面乱序、split/unsplit 以及 buffer/global 读取。原有 17 个 kernel 用例继续执行；所有数值检查必须通过，并且禁止 attention 回退到 Triton。

监控同时识别 `Process ModelRunner3/4:` 这样的未捕获异常头，即使日志分多次读取也能发现；普通可恢复错误的 traceback 不会被直接当作 worker 死亡。这样可避免本次 HTTP 前端仍存活时额外空等 900 秒。

AITER 固定版本、C32 实验矩阵及 draft 量化方式均保持原样。新增 GPU 数值用例的通过情况以目标 gfx950 CI 结果为准。

本次本地验证：16 个 CPU adapter 用例通过；193 个 ATOMesh unittest 中 192 个通过、1 个跳过；6 组真实 adapter/dispatcher 参数通过 gfx950 离线编译，覆盖 scalar/per-token、split/unsplit、buffer/global 路径。重新生成的矩阵与 39351509328 完全一致，并通过原始 gather C32 基线比较。Ruff、Black、Shell 语法与 workflow YAML 检查通过。新增 GPU 数值检查尚未在 gfx950 上运行，当前没有新的性能结果。

## 补充审查：门禁导入与真实追加写入

独立门禁脚本原先可能导入镜像中的 `/app/ATOM`。容器复现显示，`python -c` 会找到 mounted checkout，但 `python /path/script.py` 将脚本目录放在首位，随后找到镜像的 editable install。门禁现已在任何 ATOM import 之前固定当前 checkout，并新增从其他目录、带冲突 `PYTHONPATH` 启动的 CPU 回归。

同时，AITER 的新旧导入路径在特定顺序下会替换模块实例。门禁现在根据 ATOM 实际使用的 `unified_attention` callable 选择监控模块，避免 launch 计数、禁止回退和强制 global-load 的检查作用于旧实例。

数值用例改为先写入缓存前缀，再仅追加本轮 query token，最后通过真实 `dispatch_backend` 选择 attention。旧用例一次写入全部上下文，会遗漏 draft 在新 token 数为 1/4 时使用的 `BLOCK_H=1` writer 分支。对真实 target/draft 两阶段 writer 的 meta-tensor 跟踪及 8 组 gfx950 离线编译已通过；另有 3 组 meta-tensor 检查从真实 ATOM `prefill_attention` 方法出发，验证 53602-block 池的标量 scale 与门禁捕获的 Gluon 调度一致。这些不属于 GPU 数值验证。

本轮还核对了真实 prefix metadata 的零起点、padding 截取、decode 路径以及 runtime overlay 顺序；Gluon 返回位于 bounded-gather overlay 之前。在这些已检查路径中未发现其他确定的运行时缺陷。目标 gfx950 上的新增数值验证仍待执行。

补充审查的完整 CPU 回归结果：194 项中 193 项通过、1 项跳过，进程退出码为 0。Ruff、Black、Shell 语法和空白检查通过。本轮修改仍在本地，未提交、推送或触发 CI。
