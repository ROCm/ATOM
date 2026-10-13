# PR #2425：2026-10-13 修复说明与 Jasen review 回复

本次基于 `mengqing/kv_transfer_by_chunk_rm_d_sched` 的 `9e53e0ce22b0ba776924525729e197829f6e2eec` 修改。对应 [Jasen 对 9e53e0ce2 的复审](https://github.com/ROCm/ATOM/pull/2425#pullrequestreview-5487812387) 和 [Actions run 38199018822](https://github.com/ROCm/ATOM/actions/runs/38199018822)。

**代码层面的 review 修改与 CI 默认镜像选择修复已完成。GPU warmup 是否恢复，以及 c32/c48 的性能对照，尚未验证；不能将本次 CPU/Rust 结果视为 GPU 性能验收通过。** 本文提供逐项回复和可复现命令；回复与 PR 描述草稿尚未发布到 GitHub。

## 1. 本次改动

### 1.1 容量回收：持续阻塞、最早完成优先、一次一个

原实现只要 P 的 `can_allocate()` 失败，就取消所有已完成但未被 consumer 认领的 chunked prefill。现在由 scheduler 选择候选并向 worker 下发 `{request_id: transfer_id}`：

1. 同一队首请求持续 **1 秒**无法通过真实的容量检查后，才允许回收。成功准入、chunked prefill 完成、源发送占用释放都会重置观察时间；换一个等待请求也重新计时。
2. 维护按 prefill 完成顺序排列的候选队列，每轮最多取最早完成的一项。所有 TP/PP rank 收到同一个候选，避免各 rank 因局部 reader 状态不同而选择不同请求。
3. worker 先分发已经就绪的 consumer 请求，再在状态锁内调用 `cancel_if_unclaimed()`。未完成的 prefill、活跃 reader、已经部分完成的 fan-out 仍受保护。
4. `pd_reclaim_done` 经 TP/PP 汇总后确认本次尝试。全部成功时还要等 `pd_source_safe` 才能退休发送占用；某个 rank 已经认领或源已不存在时，汇总失败结果清除尝试状态，再等待一个观察周期后才考虑下一个候选。
5. scheduler 在下一次回收前重新执行 `can_allocate()`。队首请求已经放得下就正常准入，不再继续回收剩余源。源块的真正释放仍由已有的 deferred-free 和所有 connector 的占用判断决定。

该策略保留解除逆序准入等待环的能力，同时避免短暂调度错位引发批量丢弃。它仍可能让 D 重算，因此 1 秒门限的性能效果需要实测，不能仅由状态机正确性推断。

### 1.2 保持未启用 chunked transfer 时的行为

- `EngineCore` 和 `PPEngineCoreProc` 的 idle dispatch 只对原有 offload 或显式启用 chunked transfer 的 producer 执行；普通 Mooncake/MoRIIO P/D 不再被扩展到这条路径。MultiConnector 汇总 producer 的开关。
- `_consume_failed_remote_kv()` 中释放失败目标分配、清除远端首 token/drafts 的新增行为限定于 chunked 请求。移除独立的 `tests/test_pd_recv_failure.py`，legacy receive-failure 修复不包含在本 PR 中，也没有另行创建 PR。chunked 的失败后本地 prefill 流程继续由真实 scheduler/block manager 回归覆盖。
- 恢复 legacy `_has_slot_regions` 判断。只有当前请求为 chunked 且存在 SWA regions 时，consumer 才额外选择最终状态传输路径；保留 SWA-only chunked 模型的最终 slot 搬移测试。
- 将原来的 decode YAML 锚点放回 1P1D TP4 case，为 chunked case 使用新的 CPP4+DCP4 decode 环境锚点。解析后与 PR 基线 `2479251d31f4c745b43d64213cd2657f60fa7856` 对比，**60 个已有命名 case 均保持一致**，包括三个 1P3D c12/c16/c24 case。

### 1.3 简化协议实现

- 删除 `_execute_transfer()` 内不可达的 chunked 分支，以及 `_execute_chunked_transfer()` 内等待源状态建立的备用路径。后者只接受 dispatcher 已认领的 `state`，退出时无条件 `release()`。
- dispatcher 用独立失败列表收集需要通知的请求；`None` 不再同时表示两种状态。测试通过真实队列/dispatcher 进入传输路径。
- `ChunkedPrefill.readers` 从 `len(claims) - completed_readers` 派生，去掉需要同步维护的重复计数器。
- MultiConnector 对新增 base hooks 直接调用；去掉多余的 `getattr/callable` 和 `send_finished` fallback。
- `_notify_transfer_result()` 直接传 `handoff=request_data.get("prefill_handoff")`。
- 直接访问初始化保证存在的 `_reclaim_unclaimed_prefills`、`_pending_handoffs`，补齐绕过 `__init__` 的测试 fixtures。
- 新增轻量的 `mooncake/endpoint.py`，复用 producer 地址、handshake port、PP 和 page geometry 字段以及默认端口常量。能力发现不再携带每请求 flags 或可推导的 `hash_block_size`；atomesh 构造请求时写入 flags。legacy 响应继续保留兼容字段。
- 删除旧内部日志 `docs/pd_chunked_transfer_review_20260928.md`，恢复两处无关日志字符串格式，修正 workflow case 列表逗号与分组。

## 2. Actions 38199018822 的诊断和修复边界

Jasen 的复审记录：decode ModelRunner 在 warmup 的 RCCL all-to-all kernel 中发生 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`，日志没有 `Reclaiming unclaimed chunked prefill`。本次能读取公开运行摘要和 review，但未能获得需要登录的完整 job 日志。以下 kernel 异常信息引用 review，不声称是本次独立重现。

发现并修复了一个可独立验证的 CI 配置缺陷：`latest_nightly_for()` 原先接受所有 `nightly_日期` 前缀标签，再按发布时间排序。它会把 ROCm 10 和自定义构建混入默认 runtime line。仓库 `docker-release.yaml` 明确规定无后缀 `latest/nightly_<timestamp>` 属于默认发布线，ROCm 10 使用 `-rocm10.0` 后缀。

公开 Docker Hub 标签元数据证实：

| 标签 | 发布时间 UTC | 含义 |
| --- | --- | --- |
| `nightly_202610111450` | 2026-10-11 14:53:30 | 当日默认发布线 |
| `nightly_202610111453-rocm10.0` | 2026-10-11 14:59:47 | 当日较晚发布的 ROCm 10 变体；review 记录的失败镜像 |
| `nightly_202610081455` | 2026-10-08 14:58:41 | review 记录的上次通过镜像 |

修复后默认 `rocm/atom-dev:latest` 只解析 `nightly_[0-9]{12}` 的完整标签。没有当日默认线构建时，沿用原有前一天回退；显式指定固定版本或 ROCm 10 镜像不经过该筛选。没有把整个 benchmark 工作流永久锁到旧版本。

此修改消除了默认镜像意外跨 runtime line 的问题。**尚不能证明 RCCL 资源不足只由镜像引起，也不能宣称 run 已重新通过。** 当前环境 `gh` 未登录、公开 API 限流，没有 Slurm 工具或所需集群访问资料；本机是 MI308X，未找到 GLM-5.2 模型，也没有目标测试镜像。需要在原 MI355X 环境复跑。

运行摘要还包含 runner 剩余 18 MB、artifact retention 被限制到 15 天及指标不完整等警告。当前工作流的相关 upload 步骤已使用默认 retention；本次未操作远端 runner 磁盘。GPU 进程 warmup 失败时产生不完整指标不是有效性能结果；复跑前仍需确认该 runner 的可用磁盘。

## 3. 验证结果

- chunked、TP/PP、DCP、aggregator、MultiConnector、API 和 connector contract 组合：**426 passed**。随后新增两个 legacy 隔离参数用例；最终 chunked 文件：**69 passed**。
- ATOMesh 配置/启动脚本及新增镜像选择、锚点测试：**63 passed**。同镜像 baseline/chunked c32/c48 矩阵 dry-run 也已通过。
- Rust chunked 过滤集：**5 passed**（其中 1 项是已有 parser 的 chunked 输入测试，4 项是本特性路由/adapter 测试）。
- Rust ATOM adapter 与完整 P/D router 集：**22 + 38 = 60 passed**。
- 最终非 GPU CI 范围测试：**7,603 passed、836 skipped、3 xfailed，95 subtests passed**，耗时 286.59 秒。使用无 GPU 设备的 ROCm Torch 容器，移除 aiter、flydsl、LMCache、Triton 等非 CPU CI 依赖。
- Black、Ruff、`git diff --check`：通过。

组合回归使用本机已有 `rocm/atom-dev:nightly_202609191458` 镜像，以 CPU buffer 替代 NIC/GPU，未映射 GPU/RDMA 设备。完整测试使用只含已跟踪文件与本次新文件的隔离源码快照，避免混入工作区原有未跟踪脚本。第一次全量运行保留 Triton 却卸载 aiter，结果为 7,719 passed / 1 failed，唯一失败是 DeepSeek-v4.1 测试跨入缺失 aiter 的 import；随后移除 GPU 专属依赖以匹配非 GPU CI。没有修改该无关模型测试。

复现组合回归：

```bash
python3 -m pytest -q \
  tests/test_pd_chunked_transfer.py tests/test_pd_pp.py \
  tests/test_pd_source_claim.py tests/test_dcp_sharded_transfer.py \
  tests/test_deepseek_v4_transfer_regions.py tests/test_kv_aggregator.py \
  tests/test_kv_drain_liveness.py tests/test_pp_kv_status.py \
  tests/test_multi_connector.py tests/entrypoints/test_pd_prompt_token_ids.py \
  tests/test_kv_connector_contract.py
python3 -m unittest discover -s tests -p 'test_atomesh_*.py'
cd atom/mesh
cargo test --lib core::placement::tests::f_atom_adapter
cargo test --lib routers::http_pd_router
```

## 4. GPU 对照测试的复现参数与待验收指标

提交推送到 PR 分支后，在有权限的环境触发以下 workflow。使用已有 case 的 concurrency override 同时覆盖 c32/c48，无需新增影响 nightly 范围的 case：

```bash
gh workflow run atomesh-benchmark.yaml --repo ROCm/ATOM \
  --ref mengqing/kv_transfer_by_chunk \
  -f suite=nightly \
  -f run_all_models=false \
  -f case_names=glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-1m-c48,glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-chunked-1m-c32 \
  -f benchmark_concurrency=32,48 \
  -f atomesh_image=rocm/atom-dev:nightly_202610081455 \
  -f atom_source=mounted \
  -f atomesh_slurm_submit_runner=atomesh-cicd \
  -f publish_dashboard=false
```

上次通过镜像的 Docker Hub digest 为 `sha256:4d2381a2fb75dc36aa48e054ee5e65b70455e28f7460114b8c2fe838693f9e66`；复跑时应记录实际 digest、代码 SHA、节点和完整参数。baseline/chunked 必须使用同一镜像、模型、硬件和 workload。必要时再对 `nightly_202610111453-rocm10.0` 做同镜像 baseline，判别 runtime 与 chunked 因素。

每个并发档需要记录：warmup 是否成功、完成/失败请求数、TTFT p50/p90/p99、output throughput/tokens/s/GPU，以及以下两类计数：

- P 回收：从 `Reclaiming unclaimed chunked prefill <transfer_id>` 按 transfer ID 去重；不能把不同 TP/PP rank 的日志相加当成请求数。可同时报告 raw rank-event 数，区分部分 PP 回收。
- D 本地 fallback：从新增 `Chunked P/D receive failed for <request_id>; falling back to local prefill` 按 D 请求 ID 去重，按测试的预热/测量时间窗划分。这是已消费的 chunked receive-failure fallback 次数，其他失败原因也可能触发，不能直接等同于回收次数。

**上述 c32/c48 表格目前没有本次测量值。** 旧 c32 表格属于较早提交，不能作为本 commit 的结果；1 秒门限也尚未经过吞吐和尾延迟调参。

## 5. 可直接粘贴到 review 的英文回复

### 5.1 Reclaim aggressiveness and missing performance data

> The reclaim is now gated on the same head request remaining capacity-blocked for one second, with admission, completed chunked prefills and source-release progress resetting that interval. The scheduler selects one oldest completed source and sends that exact target to every TP/PP rank. Workers still atomically reject any source that has been claimed. A TP/PP-aggregated attempt result prevents overlapping attempts; successful cancellation also waits for the existing source-safe completion before retiring ownership. The scheduler rechecks actual capacity before considering another candidate, so reclamation stops when the head request fits.
>
> Added regressions for the grace period, oldest-first selection, partial TP/PP acknowledgements, differing claims across PP stages, and retaining other completed sources. The exhausted-pool opposite-admission regression still completes, now after the grace period rather than the transfer timeout. The requested same-image c32/c48 GPU comparison remains pending: this environment has no authenticated GitHub/Slurm access to the benchmark cluster. No performance claim is made for this revision.

### 5.2 Existing benchmark environments

> Restored the original decode anchor to the 1P1D TP4 case and introduced a separate CPP4+DCP4 decode anchor for the chunked case. Resolved YAML comparison against the PR base shows all 60 existing named cases unchanged, including the 1P3D c12/c16/c24 cases. Added regressions for those environments and the baseline/chunked env equivalence.

### 5.3 Idle dispatch outside the opt-in path

> Restored the gate: idle connector-only dispatch runs for the existing offload path or an explicitly enabled chunked producer. Legacy Mooncake and MoRIIO P/D retain their previous behavior. MultiConnector exposes the enabled producer capability. Both single-stage and PP gating are covered by tests.

### 5.4 Legacy receive-failure and SWA-only changes

> Removed the legacy receive-failure fix and its standalone test file from this PR. Allocation cleanup and generation-metadata clearing are now guarded by the request's chunked flag. Also restored the legacy slot-region classification; SWA-only dispatch is enabled specifically for chunked requests. The chunked fallback and final SWA-state regressions remain. A separate legacy-fix PR has not been opened as part of this change.

### 5.5 Internal review log and formatting noise

> Removed `docs/pd_chunked_transfer_review_20260928.md` and restored the unrelated log-string formatting. The new resolution document records the final behavior, validation boundaries, and reply drafts requested for this follow-up; it does not retain the old uncommitted-review narrative.

### 5.6 Dead transfer paths and redundant state

> Removed the unreachable chunked branch in `_execute_transfer` and the fallback state lookup/wait in `_execute_chunked_transfer`. The latter now takes `(request_data, state)` with a reader already claimed by the dispatcher and unconditionally releases it in `finally`. The dispatcher keeps failures separately. `readers` is now derived from `len(claims) - completed_readers`; tests enter through the actual dispatcher queue.

### 5.7 Base hooks, handoff argument and fixture guards

> MultiConnector directly invokes the inherited hooks. `_notify_transfer_result` passes `handoff=request_data.get("prefill_handoff")`. The initialized attributes are accessed directly, and fixtures that bypass `__init__` now initialize the fields they use.

### 5.8 Capability discovery duplication

> Added a shared, dependency-light Mooncake endpoint helper and handshake-port default, used by both legacy completion metadata and `/kv_transfer_info`. Discovery now carries connection/page geometry only; the adapter adds the request flags itself. The derived hash-block field is omitted from discovery while the legacy response preserves compatibility. Python discovery and Rust request-injection tests cover the contract.

### 5.9 Workflow list and stale PR description

> Restored the missing comma and grouped the chunked case with the 1P1D entries. The updated description draft below advertises the retained c32 case and explicitly identifies the older benchmark numbers as historical. c48 can be exercised through the existing concurrency override. The live PR description has not been edited in this session.

### 5.10 Failed validation image

> Found a separate workflow issue: default nightly resolution accepted suffixed ROCm 10 and custom tags, then selected whichever was published last. On October 11 this selects `nightly_202610111453-rocm10.0` over the default-line `nightly_202610111450`. Default resolution now matches only the complete unsuffixed nightly tag; explicit image overrides still work. Added tests using those release timestamps. This fixes unintended runtime-line selection, but the reported RCCL failure still needs a GPU rerun to establish whether the runtime change explains the crash.

## 6. PR 描述更新草稿

```markdown
Adds opt-in chunked Mooncake P/D KV transfer, including pipeline-parallel prefill. Atomesh dispatches P and D concurrently using a shared transfer ID; D waits for all KV writes and the final generation handoff before decoding.

Producer source blocks remain pinned until TP/PP readers drain, including cancellation. Sustained capacity stalls can relinquish one oldest completed, unclaimed source at a time; D then recomputes that request locally. This fallback is gated by a grace period and rechecks capacity after each attempt.

Includes one GLM-5.2 CPP4+DCP4 LMCache chunked benchmark case at c32. Higher concurrency is selectable through the workflow override. Existing benchmark environments and legacy P/D behavior are preserved. Default benchmark nightly selection stays on the unsuffixed runtime line; alternate runtime images require an explicit override.

Validation: CPU protocol/scheduler and ATOMesh configuration regressions, plus Rust P/D routing tests. The earlier c32 performance table belongs to an older revision. Same-image c32/c48 performance and recovery from run 38199018822 remain pending; this revision has no new GPU performance result.
```
