# DSV4 PAGE/SLOT 修改记录

> 这是本轮重构的持续修改记录。后续 review 提出的变更继续追加到本文件，统一完成后再跑测试、提交和推送。

## 当前状态

- 分支：`feature/dsv4-lmcache-page-slot`
- 基线提交：`3675567 refactor(offload): remove legacy hybrid compatibility layer`
- 本轮状态：review correctness、结构 cleanup 与统一验证均已完成。
- 提交目标：本记录与实现、测试一起提交到当前 feature 分支。

## 2026-08-13：全量 review 修复与结构收口

本轮按 merge-base 之后的完整净 diff 复核并处理了以下问题：

- Scheduler 将 load/save/send 建模为独立 deferred-release obligations；任一异步
  terminal 都不能越过另一个仍在读写 PAGE/SLOT 的 operation 提前 deallocate。
- Dense save 全面使用 `SaveOperationId`；Multi 对同一 request 保留所有 exact
  `SendOperationId`，迟到旧 generation 不再覆盖当前 generation。
- checkpoint lease 批量申请中途失败时，以 `preserve=False` 回滚本轮已取得 lease。
- `KVConnectorOutput.pending_work` 经 worker、Multi、TP aggregator 传到 Scheduler；
  可完成 fence 维持 idle polling，永久 quarantine 不造成 busy-spin。
- `DSV4CheckpointStore` 成为 corruption fence 的唯一 owner，connector 不再保存
  第二份 unresolved-corruption 状态。
- SLOT fingerprint 使用 immutable codec snapshot、稳定 semantic region role 与
  versioned layout schema；相同 byte geometry 的 plane 换序也会改变 fingerprint。
- Triton capability 在 codec 初始化时实际探测，并在 DSV4 worker startup fail-fast；
  增加 1023/1024/1025/2049-byte tile contract/round-trip 覆盖。
- 配置对未知 `offload_layout`、未知 `lmcache.*` override 和 bool/float/string integer
  geometry 全部 fail-fast；scheduler 只把 lookup client 创建视为 optional，配置、
  metadata 与 DSV4 profile 校验错误不再被 broad exception 吞掉。
- Dense/DSV4 共用 `OffloadWorkerMixin` 与 `OffloadSchedulerMixin`；移除重复 lookup
  unpin、profiling、handoff、frontier 与 save-pending 实现。
- Scheduler 通过 connector metadata 的通用 checkpoint protocol 选择 copy，不再读取
  `slot_save_spec`；Multi 通过 protocol owner 路由，Scheduler 按 connector 声明的
  checkpoint channel 消费 terminal，不再硬编码 DSV4 staging channel。metadata、
  scheduler、worker 三端的 protocol channel 必须完全一致。
- Multi 对 child-local `SaveOperationId` 增加 `connector_idx` wire namespace；两个
  saving child 即使都从 generation 0 开始，也不会互相命中 pending/tombstone，
  completion 只回调对应 child。
- 删除 test-only `__new__` lazy initializer、destructive pairs-only copy API、未调用的
  composite plan/reference-span API、codec alias/unused fields 和恒等 geometry check；
  DSV4 worker 的 completion/checkpoint collections 只允许由 `__init__` 建立，不再在
  热路径用 `hasattr/getattr` 补造半初始化状态。
- raw-pointer codec 严格要求 `reverse_indexed` 为原生 `bool`；numpy bool、Tensor、
  `0/1` 不能通过 truthiness 改变 PAGE/SLOT 地址方向。
- DeepSeek-V4 不再读取 `KVConnectorFactory._registry`；connector alias/capability 与
  Multi topology validation 统一由 factory 负责。

当前聚焦验证结果：

```text
最终聚焦回归矩阵                             642 passed
Triton/Store/Policy 扩展矩阵                    77 passed, 7 skipped
全仓非插件/非服务集成单测                     1769 passed, 107 skipped
Black --check --fast（全仓）                      PASS
Ruff（本次全部 Python 变更）                     PASS
compileall                                       PASS
git diff --check                                 PASS
```

全仓 Ruff 仍报告 19 个不在本分支净 diff 中的既有错误；GitHub workflow 使用
`reviewdog -filter-mode=diff_context`，本轮变更行无 Ruff violation。

## 2026-08-13：移除 MultiConnector 中的 DSV4 completion 语义

### Review 问题

`MultiConnector` 直接维护了以下 DSV4/offload 专用状态：

```text
sidecar_saved / sidecar_failed
checkpoint_staged / checkpoint_aborted
checkpoint local-reader expected/terminal fan-in
```

这会让通用 composite connector 理解 DSV4 的 SLOT checkpoint、storage publish 和 freelist lease 语义，分层不合理。

### 修改后的边界

新增通用事件：

```text
ConnectorCompletion
├── channel       opaque owner namespace
├── operation_id  TP rank 间的同一操作标识
└── succeeded     成功/失败；聚合时失败优先
```

新的调用链：

```text
DSV4 worker
  │ emit ConnectorCompletion(channel, id, succeeded)
  v
MultiConnector
  │ 只校验 channel owner + union；不理解 DSV4
  v
KVOutputAggregator
  │ 按 (channel, operation_id) 等待全部 TP rank
  │ 任一 rank 失败 => aggregated failure
  v
Scheduler
  ├── atom.state_checkpoint.staging
  │      -> release/invalidate native checkpoint lease
  └── atom.dsv4.checkpoint.save
         -> 路由回 DSV4 scheduler 更新 checkpoint commit 状态
```

### 唯一 owner 约束

每个 completion channel 在一个 `MultiConnector` 中只能由一个 child 声明：

```text
MultiConnector
├── moriio                 channels = {}
└── lmcache_offload/dsv4   channels = {
                               atom.state_checkpoint.staging,
                               atom.dsv4.checkpoint.save,
                           }
```

如果两个 child 声明同一 channel，初始化直接失败。native checkpoint copies、issued hook 和 abort hook 只路由给 staging channel 的唯一 owner。因此本 rank 的 local-child fan-in 被配置约束降为 1，只保留必要的 TP-rank fan-in。

安全约束：

1. scheduler 和 worker 的 channel-owner 映射必须完全一致；不一致时拒绝执行。
2. 只有 staging channel owner 能收到 leased `state_checkpoint_copies`。
3. 非 owner sub-metadata 如果携带 checkpoint copies，worker 拒绝执行。
4. child 只能上报自己声明并拥有的 channel；unknown/non-owner event 被丢弃并记录错误。
5. staging completion 的 `operation_id` 必须是精确 `int copy_id`；错误类型不会释放 lease。
6. GPU completion 无法确认时 DSV4 仍不发事件，lease 继续 quarantine，不会提前复用。
7. public offload shell 如果声明 staging channel，却缺少 start/issued/abort hook，直接失败，不再静默吞掉。

### Metadata ownership

`MultiConnectorMetadata` 新增 completion channel owner 映射，并提供：

```text
requests_for_completion_channel(channel)
```

scheduler 在申请 checkpoint lease 时，只使用 staging owner sub-metadata 中的 request。这样非 owner 的 `slot_save_spec` 不会错误申请一个永远没人完成的 lease。

connector metadata 只携带真正取得 offload lease 的 copy；完整 native D2D copy plan 仍由 `ScheduledBatch.state_checkpoint_copies` 独立携带。

### Generic async-save contract

`MultiConnector` 原来还会直接检查 child request 的 `save_spec` 和
`slot_save_spec`，用来决定 send/save block-lifetime pairing。这也是 backend
语义泄漏，现改为 `ConnectorMetadata` 的通用 opt-in：

```text
iter_async_save_operations()
    -> ((request_id, completion_id), ...)
```

普通 connector 默认返回空；`LMCacheOffloadMetadata` 根据自己的 PAGE/SLOT
request descriptor 生成结果。Multi 只消费通用 `(request_id, completion_id)`，
不再引用 DSV4、sidecar、PAGE/SLOT spec 或 boundary/source-group 字段。

### 本次涉及文件

```text
atom/kv_transfer/disaggregation/types.py
atom/kv_transfer/disaggregation/__init__.py
atom/kv_transfer/disaggregation/aggregator.py
atom/kv_transfer/disaggregation/multi/multi_connector.py
atom/kv_transfer/offload/connector.py
atom/kv_transfer/offload/metadata.py
atom/kv_transfer/offload/hybrid/dsv4/connector.py
atom/model_engine/scheduler.py
tests/（迁移到 generic completion API，尚未执行）
```

### 删除的通用层专用接口

```text
KVConnectorOutput.finished_sidecar_saving
KVConnectorOutput.failed_sidecar_saving
KVConnectorOutput.finished_checkpoint_staging
KVConnectorOutput.aborted_checkpoint_staging

MultiConnector._checkpoint_staging_expected
MultiConnector._checkpoint_staging_terminal
MultiConnector._terminal_checkpoint_staging_order
MultiConnector._terminal_checkpoint_staging

MultiConnectorScheduler.sidecar_save_finished
MultiConnectorScheduler.sidecar_save_failed
LMCacheOffloadConnectorScheduler.sidecar_save_finished
LMCacheOffloadConnectorScheduler.sidecar_save_failed
```

DSV4 内部仍可使用 sidecar/checkpoint 的实现名称；它们不再泄漏到 Multi、TP aggregator 或公共 `KVConnectorOutput` 字段。

### Multi save completion 去重收拢

Review 继续指出 Multi 中不应裸露：

```text
_terminal_save_order: deque[SaveOperationId]
_terminal_save: set[SaveOperationId]
```

这两项不是 DSV4 payload，而是通用 send/save pairing 防止 late duplicate 的
bounded completed-operation memory，不能直接删除，否则重复 metadata 可能重新注册一个已经完成的 save 并长期压住 send。当前已封装为：

```text
_completed_save_operations = _CompletedOperationWindow(limit=4096)
```

`MultiConnector` 不再直接维护 terminal deque/set，也不再出现 terminal
checkpoint collections。通用 helper 内部提供 O(1) membership 和 FIFO bounded
eviction；DSV4 checkpoint/store completion 仍全部走 opaque channel。

### TP aggregator 状态收拢

Review 继续指出 `KVOutputAggregator` 把相同机制展开成多组 `_seen_*`、
`_terminal_*_order` 和 `_terminal_*`。现统一为通用 `_TPCompletionGroup`：

```text
KVOutputAggregator
├── _sending
├── _receiving
├── _saving
├── _loading
└── _connector_completions
          |
          v
_TPCompletionGroup
├── reports[key][worker_idx] = succeeded
├── unique-worker TP quorum
├── sticky failure (False wins duplicate/contradictory report)
├── optional exact-operation completed window
└── drain() -> (succeeded_keys, failed_keys)
```

语义保持不变：

- 全部 worker output 先 ingest，再统一 drain，避免同一轮后出现的 failure 被提前 success 覆盖。
- 同一 worker 重复上报不增加 quorum；同一 worker success/failure 冲突时 failure 保持。
- raw receive/request IDs 不 tombstone。
- 只有精确 `SendOperationId`、`SaveOperationId`、`LoadOperationId` 进入 bounded completed window。
- `ConnectorCompletionKey` 始终按精确 channel + operation identity 去重。
- `reset()`、`pending_count` 和现有 tombstone count 查询保持公共行为。

### 只读安全复核结论

本轮未执行代码，但对 lease 生命周期做了只读调用链复核：

```text
local child readers = 1 (unique owner invariant)
        x
all TP ranks terminal (KVOutputAggregator)
        -> scheduler release/invalidate lease
```

未发现正常路径提前释放或永久泄漏。unknown channel、non-owner event、非法 staging ID 和无法确认的 GPU completion 都不会释放 lease。

该阶段记录的 fatal-path 待办已在后续全量 review 中关闭：
`_attach_state_checkpoint_plan()` 连续申请多个 lease 时，如果中途 acquire 失败，
会对本轮已取得 lease 逐个执行 `preserve=False` rollback。

## 统一验证待办

以下是当时约定的统一验证清单；现已在后续全量 review 阶段执行：

```text
1. DSV4 PAGE/SLOT 聚焦单测
2. MultiConnector 与 KVOutputAggregator 单测
3. scheduler/state checkpoint lease 单测
4. dense offload 回归
5. 完整 non-GPU 测试
6. Ruff 0.16.2
7. Black check
8. compileall
9. git diff --check
10. GitHub Pre Checkin
```

## 2026-08-13：首次 review push 后的 Codecheck 修复

提交 `7e48636` 推送后，GitHub Pre Checkin 首先在代码风格阶段失败：

```text
Black 26.5.1: formatting differences
Ruff 0.16.2: import ordering / __all__ ordering / TRY004 / RUF012
```

处理内容：

- 使用与 CI 相同的 Black `26.5.1` 格式化本次 Python 变更。
- 使用 Ruff `0.16.2` 修复 import、`__all__`、异常类型和测试 class attribute。
- 聚焦测试发现 `_attach_state_checkpoint_plan()` 在没有 exact checkpoint
  identity match 时也提前要求 staging channel owner；现改为先计算
  `matching_checkpoints`，只有确实准备申请 lease 时才校验 owner。
- 更新旧测试契约：完整 native D2D copy plan 只保留在
  `ScheduledBatch.state_checkpoint_copies`；connector metadata 只携带真正取得
  offload lease 的 subset。

本地验证结果：

```text
Black 26.5.1 --check --diff .                       PASS
Ruff 0.16.2（本次全部 Python 变更）                 PASS
聚焦 connector/scheduler tests                     445 passed
完整 non-GPU CI 等价范围                            1694 passed, 103 skipped
compileall                                          PASS
git diff --check                                    PASS
```
