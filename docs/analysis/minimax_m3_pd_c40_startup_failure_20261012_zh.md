# MiniMax M3 C40 PD 启动失败分析与规避修复（2026-10-12）

[失败任务](https://github.com/ROCm/ATOM/actions/runs/37867633847/job/113618310487)：
`minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c40`。

结论：P、D 都在 32768 token 的启动预热中触发 AiTer MXFP4 MoE scatter
内核非法访存，随后 EngineCore 初始化失败。AgentX 尚未执行，因此这次运行
没有给出并发 40 下的性能或精度结果。已加入用例级调优回退，绕开该 scatter
分支；尚未在 MI355X 上验证修复结果。

## 证据与失败顺序

原始资料来自该 run 的 `atomesh-model-benchmark-*`、matrix 和 latency artifacts。
模型 artifact 包含 `slurm-5973.out`、`slurm-5973.err`、
`slurm_job-5973/logs/prefill-rank-0.log` 和 `decode-rank-0.log`。
公开下载入口：[nightly.link](https://nightly.link/ROCm/ATOM/actions/runs/37867633847)。

| 项目 | 实际记录 |
| --- | --- |
| ATOM commit | `c02697c92d0c4c567ae91de8140698722d974152` |
| 镜像 | `rocm/atom-dev:nightly_202610081455` |
| 镜像 digest | `sha256:4d2381a2fb75dc36aa48e054ee5e65b70455e28f7460114b8c2fe838693f9e66` |
| Slurm | job 5973，节点 `pit2-p03-g10` |
| 拓扑 | 单机 1P1D，P TP4 / D TP4，GPU 0–3 / 4–7 |
| 调度预算 | 两端 max sequences 80，batched tokens 32768 |
| 权重加载 | target 加载和在线量化完成，峰值 59.02 GB；draft 加载后峰值 60.51 GB |
| 预热 | `warmup_max_tokens=32768 < max_model_len=1048576`，1 条长 32768 的序列 |
| 首次退出 | 日志时间 P 03:47:12，D 03:47:17 |

Prefill 日志第 194 行记录 rank 0 开始上述预热，第 221 行开始 GPU fault，
第 224 行给出内核名，第 240 行给出 HSA 错误；Decode 第 194、225、241 行
有对应证据。原始内核名：

```text
_ZN5aiter9mxfp4_moe18moe_scatter_reduce26scatter_reduce_kernel_implILi6144ELi5ELi8ELb1EEEviPK14__hip_bfloat16PKiPKfPS3_
grid=[768, 32768, 1], workgroup=[128, 1, 1]
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:
The agent attempted to access memory beyond the largest legal address. code: 0x29
```

该模板参数对应 hidden size 6144、top-k 5、每线程 8 列、NT=true。
随后出现的 `Received unexpected SHUTDOWN signal ... during initialization`
是 GPU worker 退出的结果。无 benchmark metrics / latency 文件也是启动失败
的后果。`job-result.json` 记录 scheduler 为 FAILED、workload 为 UNVERIFIED。

这份日志能定位故障内核，但不能确定错误地址最初来自哪个输入或先前内核；
也没有证据将它归因为显存 OOM、LMCache 容量、RDMA 或 C40 压测过载。
GPU core dump 因 `/coredumps` 不可写而未生成。

## 回退依据与改动

上游 [AiTer commit 7be37e8 / #5812](https://github.com/ROCm/aiter/commit/7be37e87a56ac63d7c9eda4fcc1425d1ba65e775)
在 2026-10-08 将 M3 调优表中的 4 行切换为 scatter：
token 16384 / 32768 × intermediate size 768 / 384。
其中 TP4 对应 intermediate size 384，与本任务形状相符。
这为绕开故障提供了明确的配置回退点；仅凭这些日志，还不能证明该提交
是唯一根因，也未从 artifact 获取到镜像内 AiTer 的精确 git revision。

| TP4 配置 | #5812 切换后的表项 | 本次固定的切换前表项 |
| --- | --- | --- |
| token 桶 | 16384、32768 | 16384、32768 |
| MoE 排序 block_m | 128 | 64 |
| GEMM2 | `t128x256x128_scatter_sbm128` | `t64x256x128_reduce_sbm64` |
| GEMM1 | 128 行 tile | 同步恢复上游配对的 64 行 tile |

具体修改：

- 新增 [.github/benchmark/aiter/minimax_m3_fp4_reduce_tuned_fmoe.csv](../../.github/benchmark/aiter/minimax_m3_fp4_reduce_tuned_fmoe.csv)，
  原样复制 #5812 父提交 `b1cdc19ebf3a52b101221fee5b24a3ef7753c244` 的完整 M3 表。
- 在 [models_atomesh.yaml](../../.github/benchmark/models_atomesh.yaml) 的 C40 anchor 中
  设置 `AITER_CONFIG_FMOE`，C48 和 C40 GSM8K 用例继承；其他 M3 用例不设置该覆盖。
- 配置文件通过现有 `/workspace/ATOM` 挂载提供，两端使用同一张表。
  显式覆盖会替换 AiTer 默认合并的表，因此保留了全部 32 行，覆盖小 token decode 桶。
- 新增配置回归检查，检查用例继承范围、GPU 分配、32768 预算、表项覆盖及
  GEMM1 / GEMM2 / sorting 的布局一致性；workflow 路径过滤包含新增调优目录。

修复保留 C40、1P1D、TP4/TP4、32768 batched tokens、FULL graph、EAGLE3 K=3、
fp8 KV、indexer CP1、mono0、Mooncake RDMA 和原有 LMCache 配置。
精度用例继续禁用固定接受长度；性能用例保留 2.78。
更换底层内核会影响耗时，后续性能结果需要建立新基线。

## 复跑与验收

需要在包含这些修改的新提交上发起 workflow_dispatch。
直接点击旧 run 的 Re-run 会继续使用旧提交，不能验证本次回退。

先跑 [C40 GSM8K 精度用例](../../recipes/MiniMax-M3-PD-accuracy.md)；
该文档包含完整命令。为控制镜像变量，在精度命令中也加入下面的
`atomesh_image` digest 参数。随后用同一镜像跑原 C40 AgentX 用例：

```bash
gh workflow run atomesh-benchmark.yaml \
  --repo ROCm/ATOM \
  --ref mengqing/test_m3_pd \
  -f suite=nightly \
  -f case_names=minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c40 \
  -f run_all_models=false \
  -f run_model_benchmark=true \
  -f atom_source=mounted \
  -f atomesh_image=rocm/atom-dev@sha256:4d2381a2fb75dc36aa48e054ee5e65b70455e28f7460114b8c2fe838693f9e66 \
  -f atomesh_slurm_submit_runner=atomesh-cicd \
  -f atomesh_single_node=auto \
  -f benchmark_concurrency=40 \
  -f publish_dashboard=false
```

验收顺序：

1. 两端完成 32768 token 预热、CUDA graph capture，服务通过健康检查。
2. GSM8K 在真实 EAGLE3 验证下完成全量评测；初始门槛为 0.93。
   此门槛来自现有 TP8 测试，尚非 TP4 PD 实测基线。
3. AgentX C40 真正开始并完成，产出有效性能指标；检查两端日志无 GPU fault。
4. 若仍崩溃，保留首次 fault 的 kernel/shape，使用可写 core dump 目录定位；
   若仍为相同 scatter 内核，先确认进程实际加载的调优表和命中的表项。

当前验证范围是 CPU 配置与启动脚本检查。开发机为 MI308X（gfx942），
未进行需要 MI355X（gfx950）的模型启动、GSM8K 或 AgentX GPU 复跑。

本地检查结果：

- `python3 -m unittest discover -s tests -p 'test_atomesh_*.py' -q`：68 项通过。
- C40 AgentX 和 C40 GSM8K 提交脚本 dry-run 通过，确认 common / prefill /
  decode 三组导出都带上了同一调优表路径，没有提交 Slurm 作业。
- 新 CSV 与上游父提交逐字节一致，32 行均通过 AiTer 内核名解析和排序布局检查。
- `git diff --check` 与相关 shell 脚本的 `bash -n` 检查通过。
