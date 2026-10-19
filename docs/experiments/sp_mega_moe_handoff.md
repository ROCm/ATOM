# SP4 MegaMoE 换机性能复测交接

交接日期：2026-10-13。用户已在另一台机器确认 MegaMoE 有性能提升，本机后续只做接入问题排查与 profiling。以下同配置性能 A/B 步骤保留供后续换机复测使用；AITER 继续使用**不打任何本地补丁的 main**。

当前分支已按用户要求删除 ATOM 前端和 MegaMoE 输出端的运行时 watchdog 检查，tail 的数据与 scale 用一次 `copy_tail` 搬运。受支持的完整 prefill 区域为 `sp_rs_norm → MegaMoE → copy_tail`，不增加替代检查；AITER kernel 内部的数据同步保持原样。接入已整合至 PR #2364 的 `hexwang/atom_sp`，并 rebase 到 ATOM main `2c948339`。rebase 后在纯 AITER main 上通过 209 项定向回归、81 项无 AITER/Triton 的 CPU 检查、9 项 GPU 拷贝/graph 检查、整模型 level 3 / FULL graph 捕获、2K/4K 四卡 trace 和短 decode。每个 rank 各有 57 个前端、56 个 FP8 tail、1 个 BF16 final tail、57 个 copy kernel，watchdog 检查 kernel 为 0。这些是正确性和 profiling 结果，未重新测量端到端吞吐或 GSM8K。 完整 CPU CI suite 也已重跑：8,453 passed、883 skipped、3 xfailed、0 failed；原版 ATOM main `2c948339` 对照为 8,290 passed、871 skipped、3 xfailed、0 failed。

2026-10-16 更新：AITER 固定提交换为 [PR #6514](https://github.com/ROCm/aiter/pull/6514) 的 head（见第 1 节），其 `MegaMoeTP` 在本地行数 ≥ 2048（SP4 下全局 8192 token）时走 big-M kernel。ATOM 侧去掉 tile 的环境变量，每次 launch 的本地行数自动取 `min(ceil(max_num_batched_tokens / SP), 8192)`，SP 为实际的 SP group 大小（MXFP8 激活的上限为 `256 / SP`，SP4 下为 64），并修复 `copy_tail` 在超过 4096 行的 tile 上只拷前 4096 个 FP8 scale 的问题。32K budget 的 A/B 与 GSM8K 见 3.1 节。分支随后 rebase 到 ATOM main `ffd113bc`。

2026-10-19 更新：AITER 固定提交换为 PR #6514 的新 head `70c93048`。该版本为 big-M kernel 加了磁盘缓存，初始化时编译全部档位，不再在服务中途编译；SDMA 设备函数改由 MORI 提供，需要 MORI main 包含 [ROCm/mori#786](https://github.com/ROCm/mori/pull/786)（已验证 `eb211db5`）。3.1 节的 A/B 与 GSM8K 已在本分支实现上重测。分支随后 rebase 到 ATOM main `53e4fb52`。

## 1. 拉取版本和依赖

ATOM 远程分支：[`hexwang/atom_sp`](https://github.com/ROCm/ATOM/tree/hexwang/atom_sp)。rebase 后的全融合接入提交：[`7b231f15ee4981c1efacb3cc1acd65c65bc257db`](https://github.com/ROCm/ATOM/commit/7b231f15ee4981c1efacb3cc1acd65c65bc257db)。当前实现还包含上述 watchdog 删除与 tail 拷贝合并，使用随分支更新的 [experiment-source.json](../../scripts/experiments/sp_mega_moe/experiment-source.json) 校验实现树。

AITER 固定在 [PR #6514](https://github.com/ROCm/aiter/pull/6514) 的 head [`70c93048e57fa40008707b803d35732d44fc3a3b`](https://github.com/ROCm/aiter/commit/70c93048e57fa40008707b803d35732d44fc3a3b)（尚未合入；其祖先包含此前使用的 main `cc709933`），两组都使用干净源码。tile 小于 2048 时（如默认 A/B 的 2K / 4K budget）在 main `cc709933` 上同样可用；tile 达到 2048（budget ≥ 8192，包括 ATOM 默认的 16384）需要 big-M。big-M 的 SDMA 需要以 `BUILD_CCO_SDMA=ON` 编译、包含 ROCm/mori#786 的 MORI main（已验证 `eb211db5c9cd5a39ee771b5c6799db96ab8cf1c0`，更早的 main 缺少 `cco_sdma_put_ptr_rsig__thread`）：`AITER_MORI_PATH` 指向其安装目录；环境里已装有其他 MORI 时，把该目录加入 `PYTHONPATH`（`run_ab.py` 保留调用方的 `PYTHONPATH`）。遵照用户最后确认，**不安装 tiled sorting 或其他本地补丁**。ATOM 会根据 AITER API 自动关闭 tiled sorting 扩展，native 和小 batch 回退走 main 自带的路径。

以下路径均可换成新机器的空闲目录；执行后沿用这些 shell 变量：

```bash
export SP_WORK=/app/sp-mega-retest
export SP_MODEL=/home/models/MiniMax-M3-MXFP4
mkdir -p "$SP_WORK"
git clone --single-branch --branch hexwang/atom_sp \
  https://github.com/ROCm/ATOM.git "$SP_WORK/ATOM"
export SP_ATOM="$SP_WORK/ATOM"
export SP_TOOLS="$SP_ATOM/scripts/experiments/sp_mega_moe"
git -C "$SP_ATOM" merge-base --is-ancestor \
  7b231f15ee4981c1efacb3cc1acd65c65bc257db HEAD

git clone https://github.com/ROCm/aiter.git "$SP_WORK/aiter"
export SP_AITER="$SP_WORK/aiter"
git -C "$SP_AITER" fetch origin pull/6514/head
git -C "$SP_AITER" switch --detach 70c93048e57fa40008707b803d35732d44fc3a3b
git -C "$SP_AITER" submodule update --init --recursive
git -C "$SP_AITER" status --short
```

优先沿用兼容的 ROCm 开发环境，在其中安装源码。已验证环境为 Python 3.12、PyTorch `2.10.0+rocm7.2.4.lw.git3d3aa833`、Triton `3.8.0+amd.rocm7.2.0.git7cb7b059`、FlyDSL `0.3.4.1`、Transformers `5.16.1`。旧机器的容器镜像 digest 未归档；新机器记录实际镜像 digest、驱动与 ROCm 版本。ROCm Torch / Triton 来自环境中的 AMD 构建，按上面版本核对，避免用 PyPI 通用包替换。

```bash
# 在已具备 ROCm Torch / Triton 的 Python 环境执行。
python -m pip install 'setuptools_scm[toml]>=6.2' wheel
python -m pip install -r "$SP_AITER/requirements.txt"
python -m pip install --no-build-isolation --no-deps -e "$SP_AITER"
python -m pip install --no-build-isolation -e "$SP_ATOM"
export PYTHONPATH="$SP_ATOM"
export HIP_VISIBLE_DEVICES=0,1,2,3
export GPU_ARCHS=gfx950 HSA_ENABLE_IPC_MODE_LEGACY=1
export GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1 AITER_LOG_LEVEL=WARNING
python "$SP_TOOLS/verify_source.py" --model "$SP_MODEL" \
  > "$SP_WORK/source-check.json"
```

`verify_source.py` 校验 ATOM 实现树、AITER 固定 SHA 与干净工作区、模型配置和 tokenizer 等文件的 SHA256，以及 59 个权重分片的大小；发现 AITER 本地修改就拒绝开始测试。权重未做全文件 hash；仍需使用同一来源的 MiniMax-M3-MXFP4 checkpoint。具体清单在 [experiment-source.json](../../scripts/experiments/sp_mega_moe/experiment-source.json)。AITER editable 包的日期版本后缀变化是预期的；其他包版本差异应先对齐，或明确记录为跨机器比较的额外变量。

完成条件：源码校验 `errors=[]`，模型齐全，Python 实际指向上述源码。新机器首轮复测固定此 AITER SHA；换用其他 AITER 提交放在后续独立实验中。

## 2. 检查机器和已知正确性边界

使用四张支持 P2P/IPC 的 gfx950 GPU，保留当前 SP4 / TP1 / DP1 / PP1 配置。MegaMoE persistent kernel 需要完整 GPU 执行资源，测试期间四卡独占；同一时刻只启动一个实验服务。

```bash
python "$SP_TOOLS/telemetry.py" --mapping > "$SP_WORK/gpu-mapping.json"
python "$SP_TOOLS/telemetry.py" --output "$SP_WORK/idle-telemetry.jsonl" --samples 3
rocm-smi --showproductname --showdriverversion --showtopo > "$SP_WORK/rocm-system.txt"
rocminfo > "$SP_WORK/rocminfo.txt"
python -m pip freeze > "$SP_WORK/pip-freeze.txt"
```

mapping 文件同时记录 HIP local rank 和 ROCm SMI index 对应的 PCI BDF。**按 PCI 匹配，不能假设 rank、SMI index、ROCTracer agent ID 相同。** telemetry 每两秒记录温度、gfx clock、功耗和 throttle 累计计数，并计算相邻采样的 residency 百分比。绑定来自 `$ROCM_PATH/libexec/rocm_smi`（默认 `/opt/rocm`）；如果系统改用其他监控接口，替换采集方法并保留 PCI、UTC、温度、频率、功耗与限频状态后再测试。

运行本地 CPU 回归和短算子正确性检查：

```bash
cd "$SP_ATOM"
python -m pytest -q tests/model_ops/test_sp_mega_fusion.py \
  tests/model_ops/test_sp_mega_moe.py tests/model_ops/test_m3_tp_o_proj.py \
  tests/model_ops/test_sp_moe_dispatch.py tests/distributed/test_m3_tp_o_proj_config.py \
  > "$SP_WORK/cpu-tests.log" 2>&1
unset AITER_MEGAMOE_TP_LB_MT GPTOSS_SWIGLU_MXFP4_BF16_BOUND
torchrun --standalone --nproc-per-node=4 -m atom.benchmarks.benchmark_sp_mega_moe \
  --model "$SP_MODEL" --full-fusion --tokens 64 256 260 2048 4096 8196 \
  --max-relative-l2 0.12 --max-fusion-relative-l2 0.02 \
  --output "$SP_WORK/operator-fp8.json" > "$SP_WORK/operator-fp8.log" 2>&1
torchrun --standalone --nproc-per-node=4 -m atom.benchmarks.benchmark_sp_mega_moe \
  --model "$SP_MODEL" --full-fusion --no-fp8-output --tokens 64 256 260 2048 8196 \
  --max-relative-l2 0.12 --max-fusion-relative-l2 0.02 \
  --output "$SP_WORK/operator-bf16.json" > "$SP_WORK/operator-bf16.log" 2>&1
```

这是随机权重/输入的完整 MoE 区域测试，包含 eager 与改变输入后的 graph replay；不加载 checkpoint 权重，不代表端到端吞吐或 GSM8K。交接前已在干净 AITER main 下重新通过 142 个 CPU tests（3.03s）；之前上述 11 个 GPU case 使用带 tiled sorting 扩展的 AITER，因此纯 main 的 GPU case 必须重新检查。完成条件：新机器无数值/graph/watchdog 错误，GPU 映射和遥测可用；若失败，先保留日志排查再进入性能比较。

默认保留 native activation 选择：**全局 padded tokens <256 时回退原生 BF16 activation MoE**。`ATOM_SP_MOE_MEGA_ALLOW_ACTIVATION_CHANGE=0` 保持此行为；设置为 1 的早期诊断曾影响精度。这个开关指 MoE 中间激活量化选择，并非把整个 MoE、router、权重和通信都变成同一种 dtype。

## 3. 同配置交替 A/B，首先复现原负载

此前“2K / 4K”指 `max_num_batched_tokens=2048 / 4096`，**请求长度仍为 131072（128K）输入 +100 输出，C16**。每个 case 16 次 warmup、160 个正式请求、seed=0、ignore EOS、禁用 prefix cache。不要改成 2K 短 prompt 后与旧数值比较。

[run_ab.py](../../scripts/experiments/sp_mega_moe/run_ab.py) 固定启动参数，两组只改变 `ATOM_SP_MOE_MEGA=0/1`。默认三轮，每轮两个 token budget，每个 budget 两组；顺序为 native→mega、mega→native、native→mega，共 12 次服务启动。每次从新进程开始，保留编译缓存，由不同 cache key 隔离 native/Mega 图；模型使用默认 level=3 / FULL graph。AITER MT override 被清除，使用已验证的 MT6 默认值。

```bash
python "$SP_TOOLS/run_ab.py" --model "$SP_MODEL" \
  --output "$SP_WORK/ab-output100" --dry-run > "$SP_WORK/ab-plan.json"
# 检查展开后的环境和参数，再执行同一个计划。
python -u "$SP_TOOLS/run_ab.py" --model "$SP_MODEL" \
  --output "$SP_WORK/ab-output100" > "$SP_WORK/ab-driver.log" 2>&1
python "$SP_TOOLS/summarize.py" "$SP_WORK/ab-output100" \
  > "$SP_WORK/ab-output100-summary.json"
```

runner 检查依赖和模型，保存 plan、版本、PCI mapping、每组 server/client/telemetry 日志，等待服务就绪后执行客户端。它校验 160 个请求全部成功、实际输入/输出长度一致，在退出路径上停止本次启动的进程组；输出目录必须是新目录，失败结果不会被覆盖。中断或失败后依据 `events.jsonl`、端口和进程/显存核查清理情况。脚本已完成 CPU dry-run、模拟 HTTP 服务的成功/失败清理及结果验收检查、只读 GPU mapping/telemetry 采集；trace 分析器也已读旧档案核对过四卡计数。新机器首次执行才是这个可移植 runner 的完整 GPU 实跑。

正常服务日志应包含 `SP MegaMoE full fusion`、`57 layers`、`tile=1024`（2K budget 为 `tile=512`）、`activation=MXFP4`、`comm=fp8`、`lb_mt=6`，且 native 组没有 MegaMoE 启用日志。若继承了其他 ATOM/AITER 环境变量，先根据 plan 对齐实验配置。

独立的 prefill 测量使用同样流程，只把输出长度改为 1：

```bash
python -u "$SP_TOOLS/run_ab.py" --model "$SP_MODEL" --output-len 1 \
  --output "$SP_WORK/ab-output1" > "$SP_WORK/ab-output1-driver.log" 2>&1
python "$SP_TOOLS/summarize.py" "$SP_WORK/ab-output1" \
  > "$SP_WORK/ab-output1-summary.json"
```

output=1 的输入吞吐用 `total_input_tokens / duration`；output=100 同时报告总吞吐、输出吞吐、TTFT、TPOT、ITL 和 p99。两种 workload 独立汇报。每轮配对算 `(mega/native - 1)*100%`，报告三轮原始值、中位数和范围。吞吐正值是改善，延迟负值是改善。保留完整持续负载窗口，结合 client 日志的请求完成进度和 telemetry 检查后半程漂移。

完成条件：两个 budget 都有三对有效样本；所有请求及长度验收通过；每组全程有健康遥测；吞吐采集未开启 profiler。若仍有明显限频，保留该结果但不据此给出健康硬件上的加速结论。

### 3.1 32K budget（big-M）

`max_num_batched_tokens=32768` 时每个 rank 的 prefill 为 8192 行，tile 自动为 8192；本地行数 ≥ 2048 的 tile 走 big-M。负载与上文相同（128K 输入 + 100 输出，C16，160 个请求）。

```bash
python -u "$SP_TOOLS/run_ab.py" --model "$SP_MODEL" --batches 32768 --rounds 1 \
  --output "$SP_WORK/ab-32k" > "$SP_WORK/ab-32k-driver.log" 2>&1
python "$SP_TOOLS/summarize.py" "$SP_WORK/ab-32k" > "$SP_WORK/ab-32k-summary.json"
```

mega 服务日志应包含 `tile=8192`。2026-10-19 在本分支实现（ATOM `f2314f45`）、AITER `70c93048` 与 MORI `eb211db5` 上先跑一轮 warmup A/B，再跑一轮正式 A/B，正式轮均值：

| 指标 | native | mega | 变化 |
|---|---:|---:|---:|
| 总吞吐 (tok/s) | 51,245.14 | 65,179.51 | +27.19% |
| 输出吞吐 (tok/s) | 39.07 | 49.69 | +27.19% |
| TTFT (ms) | 22,668.53 | 17,477.55 | -22.90% |
| TPOT (ms) | 184.12 | 148.18 | -19.52% |
| ITL (ms) | 182.28 | 146.70 | -19.52% |
| E2EL (ms) | 40,896.18 | 32,147.34 | -21.39% |

warmup 轮总吞吐 +24.59%。每个 32K prefill batch：native 0.587 s，mega 0.460 s。GSM8K（完整 1319 题，lm_eval local-completions，strict-match）：3-shot 89.16%（native 87.79%），72-shot 93.71%（native 93.40%）；72-shot 的 prefill token 99.4% 走 big-M。2026-10-16 在 AITER `9cdcf70d`、tail 用 torch 拷贝的实验分支上，正式轮总吞吐为 +19.49%。

big-M 编译：初始化时编译全部档位并写入 `FLYDSL_RUNTIME_CACHE_DIR/aiter_bigm`，首次启动约 320 s，之后约 75 s 就绪；各档位首次出现时不再停顿。budget 16384（tile 4096）同样验证通过。

已知问题：
- big-M 本地行数上限为 8192。16384 行（全局 65536 token）在 forward 时报 32 位偏移越界，tile 因此不超过 8192。
- 间歇性单 rank 停顿：29 次 mega 启动中，一次 32K warmup 用了 183 s（通常 9–13 s），之后服务正常；72-shot GSM8K 中有一次 big-M 调用停顿 7.6 s，其余 rank 的看门狗超时，该层结果无效。期间没有编译。ATOM 不读取 MegaMoE 的看门狗位，此类超时不会报错。

## 4. 单独抓 trace，验收融合并观察 step 跳变

吞吐完成后，在独立服务上抓 trace。以下在一个终端启动服务；在第二个终端沿用第 1 节变量运行客户端和 telemetry。服务前台运行，结束后停止该服务并核查四个 worker 已释放 GPU，再切另一组。

```bash
# 终端 A：先检查完整 2K 融合。输出目录使用新名字。
python -u "$SP_TOOLS/run_ab.py" --model "$SP_MODEL" --serve mega --batches 2048 \
  --output "$SP_WORK/profile-fusion" --profile-dir "$SP_WORK/profile-fusion/traces" \
  > "$SP_WORK/profile-fusion-server.log" 2>&1
```

```bash
# 终端 B：先等待 /v1/models 返回模型，另起遥测，再发送请求。
curl --fail http://127.0.0.1:18080/v1/models
python "$SP_TOOLS/telemetry.py" --output "$SP_WORK/profile-telemetry.jsonl" &
SP_TELEMETRY_PID=$!
python "$SP_TOOLS/profile_client.py" --model "$SP_MODEL" --tokens 2048 \
  --output "$SP_WORK/profile-fusion-response.json"
kill "$SP_TELEMETRY_PID"
wait "$SP_TELEMETRY_PID"
python "$SP_TOOLS/analyze_trace.py" "$SP_WORK/profile-fusion/traces" \
  --chunks 1 --output "$SP_WORK/profile-fusion-analysis"
```

客户端先做未采样 warmup，再 `/start_profile`→一次请求→`/stop_profile`。token ID 参数必须用 `prompt_token_ids`，不是 `prompt`。launcher 显式传 `--torch-profiler-dir`；仅设置 `ATOM_TORCH_PROFILER_DIR` 会被 CLI 默认 None 覆盖。

**单个完整 2K prefill，四个 rank 各应有：57 个 `sp_rs_norm_h6144_tp1...` 前端、56 个 `_tn1` FP8 tail、1 个 `_tn2` BF16 final tail。** 全模型仍有前三个 dense 层和其他 norm；验收聚焦 57 个 MoE 层。

复现连续 step 跳变：停止上一服务，换新 trace 目录，以同样 `--batches 2048` 启动 mega，将 profile 客户端改为 `--tokens 131072 --warmups 4`。一个请求应产生 64 个 2K prefill chunk；分析时用 `--chunks 64`。随后停止 mega，用 `--serve native` 和另一新目录重复，分析时加 `--mode native`。可按同法复测 4K budget（32 chunks）。

`analyze_trace.py` 输出每 rank 的融合计数、kernel 名称计数、按 chunk 的 marker 耗时和 UTC 对齐时间。Mega marker 包含 MoE 计算/通信/tail，native marker 只是 QuickReduceScatter，二者用于观察各自随时间的漂移，不能直接相除当作整区间加速比。完整区域速度比较使用 `benchmark_sp_mega_moe --full-fusion` 或对齐 trace 边界的所有 kernels。

针对跳变，对同一 rank、同一 kernel/shape/layer 比较 chunk 1–4、7–10 和后半段；同时观察普通 dense GEMM、融合 prefix、MegaMoE，以及其他 rank。使用 `baseTimeNanoseconds/1e9 + ts/1e6` 对齐 telemetry。重复 token 1 的请求只是固定 shape 的诊断输入；端到端吞吐以第 3 节为准。Profiler 的 GPU `dur` 单位是 **µs**，400–500 µs 即 0.4–0.5 ms。

完成条件：四个 rank 融合数量符合预期，保留 native/mega 持续 64-chunk trace，并说明跳变是否复现、是否伴随局部计算变慢与限频；分析失败时保留原始 trace，先核对请求、chunk 和 kernel 名称，不跳过计数验收。

## 5. 如何判断硬件与接入问题

旧机器有明确限频证据：HIP rank 1 对应 PCI `05:00.0`，热点约 100°C、gfx clock 597–912 MHz，thermal residency 约 97.62%。2K trace 中 rank 1 MegaMoE 从约 400 µs 变为 1417 µs，普通 dense GEMM 也从约 53 µs 变为 148 µs；native 同卡 dense 从约 53 µs 变为 145 µs。rank 3 dense 仍约 53 µs，但 MegaMoE 从约 564 µs 变为 1552 µs，符合等待慢 peer 的表现。它支持“限频卡拖累 collective”的解释，仍需新机器同配置复测判断剩余差异。

旧版（尚未接全前端和 tail）的 4K / 2K 端到端吞吐分别低于 native 17.46% / 9.69%。这些不是当前全融合版本的性能结果，不能用来计算本次代码的加速。

- 新机器频率稳定、无显著 throttle，普通 GEMM 不漂移，Mega 的跳变消失：支持旧机器问题影响了结果；再根据配对吞吐判断实际收益。
- native 和 mega 同一张卡的普通 GEMM 都变慢，其他 rank 主要在 collective 中等待：继续调查该卡/散热/功耗/负载，保留所有轮次。
- 健康机器上仅 Mega 慢：保存完整融合区域与 native 分解耗时、四卡时序、tile/MT 配置，继续分析通信、调度和融合开销；不能继续把回归归因于机器。

## 6. 已完成的精度与实现参考

当前 ATOM 接入提交在 **AITER main + 当时本地 tiled sorting 扩展**上，全模型 level=3 / FULL graph 加载和捕获已通过，2K trace 四卡均满足第 4 节融合计数；完整 GSM8K 为 **1246/1319 = 94.47% strict-match**，flexible-extract 为 1245/1319 =94.39%。此前 Mega 为 1250/1319 =94.77%，历史 native 为 1249/1319 =94.69%。本次比前一 Mega 少 4 题，但同时换了 AITER main，因此不是 fusion 单变量精度归因。**纯 AITER main 的完整 GSM8K 尚未重跑；不能把此分数标为纯 main 结果。**

评测口径：同一 checkpoint、5-shot、并发 32、max_gen_toks=16384、batch budget=32768、完整 1319 题无 limit。本次切到纯 main，性能测试外应补齐 native/mega 配对精度验证。用 `--serve mega --batches 32768` 启动同配置服务后执行，完成后停止服务，换 native 和独立输出目录重复：

```bash
# 已验证 lm_eval=0.4.13、datasets=5.0.1；评测按需在性能采集之外运行。
lm_eval --model local-chat-completions \
  --model_args "model=$SP_MODEL,base_url=http://127.0.0.1:18080/v1/chat/completions,num_concurrent=32,max_gen_toks=16384,timeout=1800" \
  --tasks gsm8k --num_fewshot 5 --batch_size 65 --apply_chat_template \
  --fewshot_as_multiturn --output_path "$SP_WORK/gsm8k-mega" --log_samples \
  > "$SP_WORK/gsm8k-mega.log" 2>&1
```

完整 GSM8K 的 raw JSONL 含两个 filter，各 1319 条；分别计分。读取用文件迭代或 `.split("\n")`，模型文本中含 Unicode 分隔符，`.splitlines()` 会误拆记录。

实现入口是 [sp_mega_moe.py](../../atom/model_ops/sp_mega_moe.py) 和 [M3 model.py](../../atom/models/minimax_m3/model.py)。MoE 前端把 residual、Gemma RMSNorm、BF16 gate/logits、sigmoid/bias/top4/shared route 融合为 `SpRsNorm`；当前 SP attention 输出已经是完整本地 token 行，前端使用 singleton TP1 group，不能换成 SP4 reduce-scatter。主体使用 SP4 MegaMoE，57 层共用 runtime；tail 融合 RS、residual、下一层 norm/FP8，最后一层直接输出 final norm 的 BF16。

每个 local tile 最多 `min(ceil(max_num_batched_tokens / SP), 8192)` 行，SP 为实际的 SP group 大小（MXFP8 激活的上限为 `256 / SP`），即一次覆盖最大的本地分片；prefix 的最后不足 16 行使用原有 kernels，小 token 的 BF16 MoE 回退复用已算好的路由。tail 的不支持形状也保留普通 residual/norm 路径。AITER tail 将 norm 结果先舍入 BF16 再转 FP8，与旧 ATOM 直接从 FP32 norm 转 FP8 并非逐位等价。

用户给的 [AITER PR #5937 benchmark 链接](https://github.com/ROCm/aiter/pull/5937/changes/9aeac9a60251c79290973ce33f52d18b265da64c) 对应删除 benchmark 的提交，接入对照的是其父提交 `9aeac9a60251c79290973ce33f52d18b265da64c^` 的 `op_tests/op_benchmarks/flydsl/bench_mega_moe_TP.py`；实际 API 来自上述固定 main。

旧机器原始精度/trace/源码证据分别在 `/app/artifacts/atom-sp/mega-moe-full-fusion-20261013/` 和 `/app/artifacts/atom-sp/mega-moe-step-jump-20261013/`。大文件未上传到 Git；这些路径只供旧机追溯，新机执行不依赖它们。

## 7. 最终交付

在新结果目录写 `REPORT.md`，记录实际 ATOM/AITER SHA 与 AITER 干净工作区证据、软件/驱动/镜像、checkpoint 来源、GPU PCI 映射、各轮两个 budget 的吞吐/延迟、健康遥测、前后段漂移、四卡融合计数和跳变判断。标明算子结果、模型吞吐和精度各自的范围；保留 raw result JSON、请求错误/长度、server/client 日志、telemetry、trace，以及服务清理状态。

新机器本地测试按上述步骤继续。若改走远程 GitHub Actions/Slurm，沿用用户约定：CI 仅用于 GPU 性能实验，每批最多 2 个 workflow、批次间隔至少 1 小时；准备和只读检查在本地完成。
