# DSV4-Pro FlyDSL mono-kernel 接入记录

更新：2026-09-29。当前分支接入了 DSV4-Pro 的每层 `mono_kernel_forward`，已验证 **bs1、seq1/2/3/4、local TP4/TP8**。单层功能矩阵覆盖 hash/bias routing、HCA/CSA、改输入 CUDA Graph replay、HCState 和压缩 cache。整模型生成、MTP accept/reject 和 bs2–8 仍待验证。

配套实现：[FlyDSL codex/dsv4-a8w4-monokernel](https://github.com/ROCm/FlyDSL/tree/codex/dsv4-a8w4-monokernel)。完整实现进展、功能/性能命令、数值门限和优化路线见 [FlyDSL DSV4 PROGRESS.md](https://github.com/ROCm/FlyDSL/blob/codex/dsv4-a8w4-monokernel/kernels/monokernel/dsv4/PROGRESS.md)。

## 接口与配置

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_V4_USE_TRITON_FUSION=0
export ATOM_DSV4_MONOKERNEL=1
export PYTHONPATH=/path/to/FlyDSL:/path/to/ATOM:/path/to/aiter:/path/to/flydsl032
```

`ATOM_DSV4_MONOKERNEL=1` 在支持范围内启用完整层 forward；仅需要 MoE 时使用 `ATOM_DSV4_MOE_MONOKERNEL=1`。开关默认关闭。FlyDSL 运行时需 0.3.2，GPU 为 gfx950；模型配置必须是 Pro。

```python
state = block.mono_kernel_forward(state, positions)
# unfused=True：保留同一 attention/mHC，将 MoE 分成五个 GPU 阶段。
```

调用方必须提供真实 forward context、input IDs、attention metadata 和已绑定 cache。显式接口拒绝范围外调用，普通 `Block.forward` 保留原生 fallback。当前不支持多请求、prefill、EP/DP/PP/PCP/DCP/TBO、在线 requantization 和其他通信融合 MoE backend。

每层内部为 attention mHC→indexer/attention→FFN mHC→MoE。Indexer 已属于 CSA attention 的内部流程，cache 只更新一次，完整 delayed `HCState` 保留。Routed 使用 A8W4/per32 E8M0，shared 保持原生 A8W8/128×128 E8M0。每层有多次 launch，MoE 是一个 resident launch。

五阶段路径是同一 FlyDSL MoE 按 router、route、quant、up、down 拆开的 5 次 launch，用于验证融合/调度；它不是 ATOM 默认算子路径。真实 ATOM 默认对照由 `tools.compare --atom-module --atom-profile default` 提供；`stable-rne` 是另外标注的精度稳定配置。当前 1.16–1.30× 的结果以五阶段为基线。

## 权重、精度与资源生命周期

- `Dsv4MoeMono.prepare()` 在子模块 shuffle 前执行。Mono 借用已加载的 GU-interleaved routed Parameters；四个 query bucket 共用权重，各自持有 scratch 和 IPC。
- Routed SwiGLU 使用 FP32→FP8 native fused exponent；shared 保留 BF16 RNE。支持的完整层 bucket 使用局部稳定 Torch BF16 compressor projection，避免 BF16 atomic split-K 引起 cache 漂移。
- `ModelRunner.exit()` 在 TP teardown 前调用 `close_monokernels`，覆盖主模型和存在的 draft model。该接入点已实现；完整 ModelRunner 端到端退出/生成仍属于下一步验证。
- Graph 使用结束后集体关闭对应 runtime。准备后替换权重、跨 stream 并发复用同一个 runtime 不在当前合同内。

## 功能测试方法

从配套 FlyDSL checkout 运行。以下示例假设环境变量已设置，checkpoint 路径为 `/path/to/DeepSeek-V4-Pro`。

```bash
# 完整层：TP4 HCA，mono 与五阶段 MoE，共用相同 attention/mHC。
torchrun --standalone --nproc-per-node=4 \
  -m kernels.monokernel.dsv4.tools.monokernel \
  --checkpoint /path/to/DeepSeek-V4-Pro --tp 4 --layer-idx 3 \
  --batch-size 1 --seq-lens 1 2 3 4 --check --replays 3 --baseline-repeats 3 \
  --output results/layer-l3-tp4.json

# CPU policy 单测；cache 布局单测需可见 gfx950 和配套 ATOM/AITER。
python3 -m pytest -q /path/to/ATOM/tests/test_dsv4_monokernel_policy.py \
  /path/to/FlyDSL/tests/unit/test_dsv4_cache_observation.py
```

将 TP 改为 8 并使用 8 个 worker，层号覆盖 0/2/3/4，即构成完整层验收矩阵。`tools.compare` 的 `--atom-module` 用于真实 ATOM MoE 模块的 enabled/disabled 对照；覆盖 L0/L3×TP4/TP8。使用 `tools.accuracy_config` 生成 TP 专属 profile，再显式传 `--atom-profile stable-rne --atom-config-dir ...`，可复现稳定 router/CK RNE shared GEMM1 对照。默认 native 和 stable/RNE 的结果必须分开记录。

检查所有 worker 的退出码和 `*.rankN.json`：每个 shape/replay 的 `passed` 必须为 true。MoE mono/五阶段以及 module mono/standalone 要逐位一致；每个实现独立对 oracle 的 NRMSE 默认不超过 1.5%。完整层同时检查 HCState、逻辑 cache、数值区域外的 backing bytes 和重复漂移。所有 shape guard 在 cache 写入前执行。两个入口均拒绝 bs2、seq0、seq5。

可选添加 `--bench --warmup 5 --repeats 30 --graph-iters 10` 比较性能；完整层添加 `--profile-launches` 获取真实 device kernel 数量。精度失败的 case 不计时。性能需相同输入、seed、初始 cache 和空闲 GPU；以三 seed 的最慢 rank median 汇总。

## 当前结果与后续工作

本轮 36 项测试全部通过，MoE 对独立 oracle 最大 NRMSE 为 0.378549%；8 组完整层的 HCState/cache 差异为 0。新的多 token 阶段调度在 seq4 上相对旧 mono 降低完整层延迟约 15.2%–23.0%，相对五阶段路径约 1.16–1.30×。HCA TP4/TP8 为 29/28 次 launch，CSA 为 48/47 次，均比五阶段对照少 4 次。

提交源码另通过 5 项单测和 L0/TP4、L3/TP8 的 seq1–4 MoE 功能/计时入口回归。Standalone MoE benchmark 的默认 stream capture 问题已修复。服务器回收时停止后续完整层重复测试，保留此前完整层矩阵的验证记录。

这些结果来自 position128、私有初始零 cache 的单层 fixture。默认 native ATOM 曾出现 router/shared split-K 相关精度失败，保留独立记录，不能用 stable/RNE 的通过结果替代。

下一步先验证整模型的初始化、graph capture、生成、MTP accept/reject 和退出，再覆盖非零历史/长 context 与 cache page/ring 边界。bs2–8 扩展需要明确 request/token 映射、metadata、hash IDs、scratch 和 IPC bucket 合同。后续性能工作包括 GEMM1 K-wave/tile、GEMM2 权重与 scale 双缓冲/LDS 重叠，以及减少 indexer/attention/mHC launch；每项先验证数值和 cache，再比较性能。
