# DSV4.1 C256 实际运行配置核对

2026-10-29。本次读取了公开 C256 测点的实际 summary artifact，不再仅用 catalog 默认值推断。

**ATOM random dashboard 的这个 C256 点是 TP4 / EP1 / DP1，关闭 DP attention，不是 DEP。**

- [实际运行](https://github.com/ROCm/ATOM/actions/runs/40724946700)，attempt 1，cell `deepseek-v41-flash-dspark-8192-1024-256-0.8`。
- [公开 summary artifact](https://nightly.link/ROCm/ATOM/actions/artifacts/12853757650.zip)，10,617 bytes，SHA256 `58028d07bbf4881a8a319e0e689667b0ae47fc047d9210b14370ff0645a382cf`，与 GitHub API artifact digest 一致。
- [保存的实际 config](../../validation/dashboard_whole_20261029/public_config/c256_config.json)。`config.json`、`summary.json`、`validation.json` 的摘要均与包内 manifest 核对一致；没有下载 607 MB 的完整 bundle。

| 字段 | 实际记录 |
|---|---|
| 并行 | TP4、EP1、DP1、PP1、`dp_attention=false` |
| GPU | 使用 4 张 MI355X；runner 节点有 8 张卡 |
| KV / index | FP8 / FP4 |
| 投机 | DSpark5，`synthetic=false` |
| 负载 | random 8192/1024、range .8、C256、seed0、ignore-eos、chat template |
| 请求数 | 2560 formal、512 warm |
| 实际 ATOM | `646da00fe322d28cb7971082d2a8af6905c57cd8` |
| 实际 AITER | `c90842add033fb3f9dbe0c6c53187a023faff1ee` |
| 实际镜像 | `sha256:01e76c6c20e3db4ed867bae44879e9ad61e523d4ebd855c51b4ba0aac1ebc397` |
| 模型 revision | `314113f8a7dadb388215396ba5fbee2c610a04d6` |

实际 `server_argv` 的模型配置部分是：

```text
--kv_cache_dtype fp8 -tp 4 --index_cache_dtype fp4
--method dspark --num-speculative-tokens 5
```

它没有 `--enable-dp-attention` 或 `--enable-expert-parallel`。实际提交的 `Config/EngineArgs` 默认值也已复核：max-num-seqs512、max-batched-tokens16384、attention chunk16384、memory .9、prefix on、confidence_schedule=false。KV block-size 的通用默认16仍会被 V4.1 运行时解析为128，不能将通用默认直接当作该模型的最终值。

必须纠正旧方案中的一处来源混淆：dashboard data.js 最新条目的 commit 标签是 `bdec6812`，但这个测点的 run、workflow、实际 artifact 的 ATOM/harness SHA 都是 `646da00f`。页面聚合条目的提交不能代替测点运行源码。

旧页面值为 output4816.42 tok/s、TPOT50.55 ms、AL2.44。包内按请求窗口重新聚合的 output 是4816.5203 tok/s，窗口488.83049秒；两种计时口径不能混算。包的 `validation.complete=false` 是缺少 model.precision 元数据，并标记不可提交；这不妨碍核对明确记录的启动配置，也不能将该包宣称为通过完整发布验收。

DEP 来自另一条测试线：[ATOM AgentX nightly catalog](https://github.com/ROCm/ATOM/blob/bdec681249254846f73b833d1a5112f86730ca34/.github/benchmark/models_agentic_nightly.json) 有 DTP2/DEP2、C128、合成 AL3.51。DEP2 的参数明确同时包含 `-tp 2 --enable-dp-attention --enable-expert-parallel`。它不是本次 random C256 点。SemiAnalysis 当前 ATOM AgentX 源矩阵则是 TP2/TP4、EP1、无 DPA；尚未逐一验明它的发布包。

本地已完成 TP4/FP8/真实 DSpark5/C256 的四臂同配置整网对照，见[完整基线](dsv41_c256_baseline.md)：合并+0.54%，两对方向相反，未接受为稳定加速；实际执行行数与冷编译均已记录。完整FFN内部净成本仍待诊断。旧 EP4/owner 的 M768、完成曲线与成本占比不能直接代入。低并发测试依照用户最新优先级暂不启动。

本地受控比较使用已有 pinned image/AITER 和 ebd5 main/f2d3 branch；其软件版本、观察器和593条独立请求预热与公开包不同。因此它用于本地分支比较，不是上述公开结果的原样复现。

## 低并发补充核对（06:31 UTC）

继续核对同一 run40724946700 的 C1/4/8/16 实际 summary 包：均为 TP4/EP1/DP1、无 DPA、FP8 KV、真实 DSpark5；实际 ATOM/AITER/镜像与上面的 C256 点一致。四个 zip 的 SHA256 均与 GitHub artifact digest 相符，包内 config/summary/validation 均与 manifest 相符。见 [逐点配置与证据](../../validation/dashboard_whole_20261029/public_config/low_concurrency_artifacts.json)。

下表统一采用 dashboard 的发布计时口径，未与包内重新聚合的 throughput 混算：

| 并发 | output tok/s | TPOT ms | 发布 AL |
|---|---:|---:|---:|
| C1 | 574.73 | 1.62 | 2.67 |
| C4 | 971.30 | 3.86 | 2.44 |
| C8 | 1153.34 | 6.60 | 2.60 |
| C16 | 2042.34 | 7.38 | 2.22 |
| C256 | 4816.42 | 50.55 | 2.44 |

[发布值及来源](../../validation/dashboard_whole_20261029/public_config/low_concurrency_dashboard.json) 保留 dashboard 内容摘要和逐点 run 链接。以上是公开测点，不是本地主线/分支配对，也不是 SemiAnalysis AgentX。没有启动本地低并发测试。产品入口的 `dashboard_profile.json` reference 已从页面聚合提交改为实际测点提交646da00f，并记录 artifact 摘要；engine/benchmark 参数未变，R7 与 fault_r1 的冻结 profile 未改。
