# DSV4 PAGE/SLOT 修改记录

## 2026-08-17：适配统一 PAGE/SLOT backing 与 PAGE-backed checkpoint

本分支已 rebase 到包含 DSV4 统一物理内存和 PAGE-backed state checkpoint 的最新
`origin/main`。本轮适配的目标是让 LMCache offload 使用新的内存与 checkpoint
边界，不再依赖旧的 SLOT checkpoint lease/staging 协议。

### 数据面

- `DSV4PageSlotCodec` 继续把 PAGE 与 SLOT 作为两类 typed copy plan：PAGE 以
  physical block ID 正向寻址，SLOT 以 request group ID 反向寻址。
- PAGE 与 SLOT region 允许来自同一个底层 HBM allocation，也允许对应 plane 使用
  相同 semantic role；它们的 kind、unit geometry 和索引方向仍保持独立。
- 新增 CPU reference-span 测试，验证 PAGE scatter 不改变 SLOT span，SLOT scatter
  不改变 PAGE span。

### Native checkpoint

- DSV4 native state checkpoint 由 `PagedStateCheckpointCoordinator` 保存到 PAGE
  units。
- Active SLOT 只服务 live request；native checkpoint copy 与 LMCache sidecar 不
  共用 lease、staging row 或 completion channel。
- Model runner 和 block manager 继续独立完成 native checkpoint store/restore。

### LMCache AOS1 sidecar

- Scheduler 只传递 `SlotSaveSpec`，其中 `source_group` 指向当前 live Active SLOT。
- Worker 在 `start_load_kv()` 中预留 connector-owned staging row，在 current CUDA
  stream 上 gather live SLOT 并记录 event。
- metadata 在 batch forward 前派发；snapshot 与 forward 在同一 stream 上有序，
  后续 forward 可以安全修改统一 backing。
- 后台线程等待 event，再从 staging row D2H、编码和存储；它不会在 forward 后读取
  live SLOT。
- staging admission/snapshot 失败时 SLOT fail closed，但 PAGE save 可继续。临时 GPU
  row 在 D2H 后立即 release，publication 不长期占用它。
- DSV4 只声明 `atom.dsv4.checkpoint.save` completion channel。

### Scheduler 与统计

- 移除等待 sidecar publication 才继续 partial prefill 的暂停策略；source-safe
  snapshot 已经切断 live SLOT mutation race。
- Dense 与 DSV4 scheduler 共用 exact-operation offload 统计：load/save token、失败和
  pending 数量均按 operation identity 维护。
- sidecar-only save 计为一次 0 PAGE-token save；重复或迟到 exact completion 不重复
  计数。
- request cleanup 会移除已放弃 load operation 的 pending 统计。
- public `lmcache_offload` scheduler shell 委托 `get_statistics()`。

### 通用 connector 层

- Multi connector 仍按唯一 owner 路由 opaque completion channel，并按
  `iter_async_save_operations()` 做 producer send/save 生命周期配对。
- TP aggregator 继续按 `(channel, operation_id)` 聚合，失败优先；它不理解 DSV4
  checkpoint 语义。
- 删除测试中对旧 native SLOT staging hook 的假实现，保留 generic completion owner、
  owner mismatch、non-owner drop 和 TP quorum 覆盖。

### 验证范围

```text
核心 DSV4/offload/Multi/scheduler/API 回归             515 passed
MTP deferred-proposal GPU 回归                         10 passed
PAGE-backed checkpoint 与 offload 扩展回归            368 passed, 1 skipped
PAGE/SLOT Triton CPU contract 与 GPU round-trip         15 passed
compileall                                               PASS
git diff --check                                         PASS
旧 SLOT lease/staging API 全仓扫描                        无残留
```
