# Qwen3.8-Flash-Next TP8 普通 decode Mono 整网复现

## 状态与范围

2026-10-16，分支 `yifehuan/qwen38_flash_mono`。这是实验接入复现，**不是生产默认启用说明**。
生产 `config.ENABLED_TP=(1,)`、`ENABLED_WIDTHS=(1,)` 保持不变；仅设置
`ATOM_QWEN_MONO_ENABLE=1` 不会自动启用 TP8。下述 bootstrap 在进程启动时显式打开
TP8 实验 gate，不改生产配置，也不提高资源限额。

当前应区分三种路径：

| 宽度 | 接入路径 | 当前结论 |
| --- | --- | --- |
| S1 | TP8 GDN K1/K2、QSA K1/K2 + deferred peer pull | 已测明显时延收益，冻结参考；完整输出质量仍待验收 |
| S2 | native attention/前置 HC + paired HC/router/shared G1、逐行 routed G1/G2 + deferred pull | partial-Mono，同配置单次 A/B 快约1.7%，可靠收益与质量待验收 |
| S4/S8 | native fallback | 不计作 Mono 泛化成绩 |
| TP2/TP4、S16 | 未扩展/未验收 | 不支持本复现 |

本说明不评估 MTP、verify width、TBO、DP/PP/CP 或生产多样 prompt 吞吐。

## 环境

已测环境为 080 的 8×MI355X/gfx950，ROCm7.2、Torch2.10、FlyDSL0.3.4.1。
GDR composition emitter 已随本 PR 放入 ATOM，无需 AITER 自定义分支或 CSV 调优。
历史验证使用 AITER `2eb02e58dbc47243393773d79828b059ab0960b0`；本地 emitter
保留相同 recurrence 与 BF16 beta 语义，使用 AITER 上游的 FlyDSL 基础工具。
模型 `/shared_nfs/huggingface_models/Qwen/Qwen3.8-Flash-Next-FP8`。
命令在可使用 8 GPU 的现有环境内执行；不自行申请或取消作业。

ATOM 仓库路径示例 `/home/yifehuan/glm5_tileRT/atom`，AITER 示例
`/home/yifehuan/glm5_tileRT/aiter`。脚本自动解析 ATOM 路径，AITER 可用 `GEN_AITER` 指定。
日志、JSON、编译缓存写入仓外 `GEN_ROOT`；各轮使用唯一 `GEN_SUFFIX`，避免覆盖结果。

## 启动与 benchmark

先确认 GPU 无竞争、无旧 server 占用 HTTP8000。进入 ATOM 仓库后启动 native：

```bash
export GEN_ROOT=/home/yifehuan/data/qwen38_mono_repro
export GEN_AITER=/home/yifehuan/glm5_tileRT/aiter
export GEN_MAX_SEQS=2 GEN_CAPTURE_SIZES='[1,2]'
export GEN_SUFFIX=A1
bash tools/qwen38_mono_repro/start_server.sh 8 native
```

启动前台运行，另一个同环境终端执行；**等日志 Application startup complete 且 health 成功**：

```bash
curl -fsS http://127.0.0.1:8000/health
GEN_ROOT=/home/yifehuan/data/qwen38_mono_repro GEN_SUFFIX=A1 GEN_CONCS=1,2 \
  python3 tools/qwen38_mono_repro/benchmark.py 8 native
```

当前 CLI 的 `--port 8123` 是 engine 配置，实际 OpenAI HTTP 服务默认监听 **8000**，
以启动日志为准。benchmark 默认 HTTP8000，可用 `GEN_URL` 覆盖。

完成后 Ctrl-C 停止这次前台 server，确认其进程退出及显存释放。不要 kill 容器、作业或其他人的进程。
使用相同 max-seqs、capture-sizes、cache、模型等配置启动 B1：

```bash
export GEN_SUFFIX=B1
export GEN_TP8_FINISH_PAIR=1
unset GEN_TP8_GDN_WIDTH GEN_RESOURCE_AUDIT GEN_HYBRID
bash tools/qwen38_mono_repro/start_server.sh 8 mono
```

另一个终端 health 成功后：

```bash
GEN_ROOT=/home/yifehuan/data/qwen38_mono_repro GEN_SUFFIX=B1 GEN_CONCS=1,2 \
  python3 tools/qwen38_mono_repro/benchmark.py 8 mono
```

此配置 S1 保持原 TP8 Mono，S2 使用 partial-Mono paired finish。
只复现 S1 / S4/S8 fallback 时，取消 `GEN_TP8_FINISH_PAIR`，使用
`GEN_MAX_SEQS=8 GEN_CAPTURE_SIZES='[1,2,4,8]'` 与 `GEN_CONCS=1,2,4,8`，native 必须同配置重跑。
严禁把 `GEN_TP8_GDN_WIDTH=1` 的串行扩宽 GDN 候选作为最佳配置：它曾使 C2 退化。
`paired_shared_g2` 是默认关闭的算子 probe 候选，复测无稳定收益，整网脚本不启用。

正式性能判定重复 A1/B1/A2/B2 四次独立启动。预热、测量时不要并行跑 GPU probe。
启动/编译耗时不计入 decode 指标，但要记录 TTFT 和端到端吞吐。

## 指标与路径证据

固定 19-token prompt、每请求128 output tokens、greedy、ignore_eos、无 prefix cache/MTP。
每档一轮预热 + 七轮正式测量，每轮提交 C 个独立并发 HTTP 请求，不能用一个请求的 n=C 代替。

- client TPOT = `(request wall - server TTFT) / 127`，每轮取最慢请求，再取七轮中位数；
  server TTFT 有舍入误差，另存 server TPOT，不冒充纯 GPU kernel 时间。
- decode 总吞吐 = `C * 1000 / TPOT_ms`，每 GPU 吞吐再除8。
- end-to-end output throughput = `C * 128 / round wall`，包含 prefill；
  不是 `(输入+输出)/wall` 的 Total token throughput，不能混用两个口径。
- `vram_c*.json` 记录各 GPU 设备显存；utilization0.92 下预分配 KV/state pool，
  不等于请求增量显存。`telemetry_c*.json` 另记录 Torch allocated/reserved/peak。

`benchmark.json` 保存每轮请求 usage、文本 SHA256、wall、两种 TPOT、两种吞吐。
`telemetry_<pid>.json`/`telemetry_c*.json` 保存实际 `(scheduled width, running requests, graph)`
计数与 `mono_prepared`。快照周期127步，尾部可能未计全；结合 server 日志和各 rank 检查，
不能仅凭 env enabled 或请求 C 判定实际 Mono。

S2 应看到 `GEN_PATH S2=partial-Mono native-attention ...`、所有 rank
`GEN_MONO_PREPARE ... width=2 ok=True`；真实 decode 主要为 `(2,2,True)`，尾部 S1 单列另记。
S4/S8 没有对应成功 prepare 就是 fallback；若准备失败，不得将其延迟列为 Mono。

## 已有性能记录与验收边界

历史同节点普通 decode（ms），不是本轮四启动 ABAB：

| C | native TPOT | 对照 TPOT | 实际路径 | 结论 |
| ---: | ---: | ---: | --- | --- |
| 1 | 6.8831 | 4.6044 | Mono S1 | 1.495× 性能参考 |
| 2 | 7.1369 | 6.9862 | partial-Mono paired finish | 约1.022×，配置未完全配对，不能正式判正收益 |
| 4 | 7.8757 | 8.0422 | native fallback | 不是 Mono 泛化 |
| 8 | 8.4369 | 8.5599 | native fallback | 不是 Mono 泛化 |

历史 C1 冻结 P12 另次 6.8652→4.5813 ms，1.4985×。TP1 的 S2/S4 审计路径
分别慢于 native，且超旧 SGPR cap，不是生产验收。不能总结为“TP1/TP8 的 S1/S2 均有收益”。

算子资源/数值/replay 通过不等于整网正确性通过。当前生成文本 hash 变化，尚缺固定
checkpoint、相同 cache/state 初态下 final hidden/logits/argmax 与长生成验收，
所以 **TP8 生产 gate 保持关闭**。S1 是性能基线通过，不是生产质量放行；
S2 是 partial 接入，还不是完整 TP8 Mono specialization。

主机回归：`python3 -m pytest tests/test_qwen4_exp_mono.py -q`，本轮64 passed。
8-rank shared G2 双列 probe 四类输出相对原 S1 参考 relL2=0，1000 replay exact，
private scratch/VGPR spill0；复测84.059→83.993µs，无可辨认收益，不进入默认整网。
更多候选与资源账本见 [TP8 方案](qwen38_flash_next_tp8_monokernel_plan.md)。

### 本轮同配置 A/B 复测

080、Job217242、max-seqs2、capture[1,2]，native/partial-Mono 各一次启动，
每档1预热+7测量。未完成四启动 ABAB，因此 S2 不标为可靠正收益验收通过。

| C / 实际宽度 | native TPOT ms | Mono TPOT ms | 比率 | native / Mono decode tok/s | native / Mono tok/s/GPU |
| --- | ---: | ---: | ---: | --- | --- |
| 1 / 1 | 6.8870 | 4.5979 | 1.4979× | 145.20 / 217.49 | 18.15 / 27.19 |
| 2 / 2 | 7.1332 | 7.0132 | 1.0171× | 280.38 / 285.18 | 35.05 / 35.65 |

所有8个TP worker累计 C1/C2 dispatch 各1024次 FULL graph；Mono prepare S1/S2
均 true，native prepare为空。C2 是明确的 partial-Mono，不是 fallback。
每请求128 token完成；生成 hash 仍有变化，质量验收未完成。
原始仓外记录：`/home/yifehuan/data/qwen38_generalization_20261016/` 下
`tp8_native_accept_native` 与 `tp8_mono_accept_mono` 的 benchmark/telemetry/vram JSON。
复测摘要同时保存在 `tools/qwen38_mono_repro/results_20261016.json`。
完整主机/runtime/plan/shard回归153 passed，3项已有fork deprecation warning。

## 整网精度验证进展

GDN 已修复 BF16 卷积逐项乘积舍入和递推 beta 的 BF16 边界。
300 题配对评测及同状态 GPU 对照已完成，全量 1319 题正在运行。
同状态对照曾出现一次宽度 2 的 argmax 分歧，仍需定位；当前不宣告质量通过。
结果、限制和复现工具见 [精度验证记录](qwen38_flash_next_accuracy.md)。
