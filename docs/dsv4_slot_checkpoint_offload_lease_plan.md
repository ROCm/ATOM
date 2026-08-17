# DSV4 SLOT checkpoint offload：当前实现

> 本文已按统一 PAGE/SLOT 物理内存后的实现重写。早期方案曾让 LMCache
> sidecar 借用 native checkpoint 的 SLOT copy/staging 生命周期；该方案已经废弃，
> 当前实现中两条 checkpoint 路径彼此独立。

## 结论

DSV4 现在同时存在两套互不依赖的 checkpoint：

1. native HBM checkpoint 由 `PagedStateCheckpointCoordinator` 管理，保存到 PAGE
   units；它不占用 Active SLOT，也不向 offload connector 出借临时副本。
2. LMCache checkpoint 把 token-indexed PAGE 保存到标准 LMCache object，并把完整
   request SLOT 保存为 AOS1 sidecar。sidecar 直接从当前 live Active SLOT 做 D2D
   snapshot，不读取 native checkpoint 的 destination。

因此，native checkpoint 的保留、驱逐和恢复不会控制 LMCache sidecar 的保存时机；
LMCache publication 也不会延长 native checkpoint 的生命周期。

## 统一 backing 的地址模型

最新 DSV4 可以让 PAGE 和 SLOT region 指向同一块底层 HBM allocation，但两者的
索引方向不同：

```text
allocation low address
  ├─ PAGE block 0
  ├─ PAGE block 1
  ├─ ...                         forward indexed
  │
  │   unused / dynamically shared capacity
  │
  ├─ ...                         reverse indexed
  ├─ SLOT group 1
  └─ SLOT group 0
allocation high address
```

`DSV4PageSlotCodec` 分别保存 PAGE 与 SLOT 的 immutable region geometry：

- PAGE unit address：`base + block_id * page_unit_bytes`；
- SLOT unit address：`base + total_bytes - (group_id + 1) * slot_unit_bytes`；
- PAGE 与 SLOT 可以具有相同的 `base_addr`、`total_bytes` 和 semantic plane role；
- region 是否重复只在各自 PAGE/SLOT 集合内判断，不能因为共用 backing 就合并；
- PAGE gather/scatter 只能访问 PAGE spans，SLOT gather/scatter 只能访问 SLOT spans。

## LMCache sidecar 保存流程

Scheduler 产生 `SlotSaveSpec(boundary_tokens, boundary_block_hash,
source_group)`。metadata 在本轮 forward 前派发到 worker，worker 的
`start_load_kv()` 执行：

```text
reserve connector-owned SLOT staging row
  -> current CUDA stream 上从 SlotSaveSpec.source_group gather live Active SLOT
  -> 同一 stream 记录 ready event
  -> 后台 save executor 等待 ready event
  -> staging row D2H 到独立 CPU frame
  -> 立即 release/quarantine staging row
  -> 保存/验证 PAGE coverage
  -> 编码并写入 AOS1 sidecar
  -> 聚合 atom.dsv4.checkpoint.save completion
```

snapshot 与随后 forward 位于同一 CUDA stream：snapshot 先读取旧 boundary 的
完整 SLOT，forward 后修改统一 backing。后台线程只读取 connector-owned staging
row，不会在 forward 后再次读取 live SLOT。

若 staging admission 或 snapshot 失败，SLOT sidecar fail closed；同一 operation 的
PAGE 部分仍可独立保存。PAGE object 没有对应成功 sidecar 时只是未提交缓存，不能被
当成可恢复的 stateful boundary。

## LMCache restore 流程

恢复严格按 PAGE+SLOT 执行：

```text
retrieve PAGE into allocated PAGE blocks
  -> fetch and validate AOS1 header/fingerprint/CRC
  -> H2D into connector-owned staging row
  -> scatter into newly allocated destination Active SLOT group
  -> synchronize
  -> release/quarantine staging row
  -> report load success
```

任何 PAGE retrieve、sidecar lookup、decode、H2D、scatter 或 synchronize 失败都会让
整个 stateful load 失败，scheduler 使用已经分配的 PAGE/SLOT 重新计算，不发布
PAGE-only prefix。

## 生命周期约束

- sidecar save 使用 exact `SaveOperationId`，迟到 generation 不能完成当前 save；
- `atom.dsv4.checkpoint.save` 是 DSV4 connector 唯一 completion channel；
- native PAGE checkpoint 的 copy operation 由 model runner 自己执行；
- sidecar staging row 只持有到 D2H 完成，不等待 PAGE/NVMe publication；
- GPU completion 无法确认时 staging row 必须 quarantine，不能回到 admission pool；
- request cleanup 必须清除已放弃 operation 的统计/pending state；
- Multi connector 与 TP aggregator 只传输 opaque completion，不解释 PAGE/SLOT。

## 主要实现与测试

```text
atom/model_engine/page_unit_checkpoint.py
atom/model_engine/block_manager.py
atom/kv_transfer/offload/hybrid/dsv4/codec.py
atom/kv_transfer/offload/hybrid/dsv4/connector.py
tests/test_dsv4_page_slot_codec.py
tests/test_lmcache_offload_v4_page_slot.py
```

几何回归测试使用一个模拟 backing address space，同时建立 PAGE forward region 和
SLOT reverse region，验证两类 scatter 只改变各自 span。worker 测试验证 snapshot
来源是 `SlotSaveSpec.source_group`、event 在 current stream 记录、后台 D2H 不再读取
live SLOT，以及 staging admission 失败时 PAGE 仍可保存而 SLOT fail closed。
