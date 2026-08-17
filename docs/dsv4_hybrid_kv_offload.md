# DeepSeek V4 混合 KV Cache Offload 说明

DeepSeek V4 的 offload 将一个可复用检查点定义为不可拆分的组合：

```text
检查点 = PAGE 数据 + 一个完整的 SLOT sidecar
```

PAGE 和 Active SLOT 共用主 HBM allocation，但仍是不同的逻辑对象。对于
有状态的 DSV4 请求，只命中 PAGE 不足以恢复执行。

## HBM 布局

`DeepseekV4AttentionMetadataBuilder.allocate_per_req_cache()` 只创建一个字节
allocation，再从中切出主 NoPE plane 和可选的 RoPE plane。每个 plane 包含
PAGE 地址空间和预留的 Active SLOT 容量。共用物理 allocation 不代表 Active
SLOT 也是一个 PAGE ID。

```text
一个物理 allocation: per_req_pool

  低地址                                                       高地址
     |                                                            |
     v                                                            v

  NoPE plane
  +------------------------------------------+--------------------------+
  | PAGE 地址空间                            | Active SLOT 容量         |
  | PAGE 0 | PAGE 1 | ... | PAGE N-1        | ... | group 1 | group 0 |
  +------------------------------------------+--------------------------+
    BlockPool 管理                              StateGroupPool 管理
    正向索引                                    反向索引

  启用 FP8 KV 时的 RoPE plane
  +------------------------------------------+--------------------------+
  | PAGE 地址空间                            | Active SLOT 容量         |
  | PAGE 0 | PAGE 1 | ... | PAGE N-1        | ... | group 1 | group 0 |
  +------------------------------------------+--------------------------+
```

CSA indexer KV 在物理上仍是独立 tensor `v4_csa_idx_kv`，但与主 KV 共用同一
套 `BlockPool` block ID。一个逻辑 PAGE 由多个物理 region 共同组成：

```text
Logical PAGE i
+-------------------------+
| NoPE plane region       |  per_req_pool
+-------------------------+
| RoPE plane region       |  per_req_pool，启用 FP8 KV 时存在
+-------------------------+
| CSA indexer layer 0     |  v4_csa_idx_kv
+-------------------------+
| CSA indexer layer 1     |  v4_csa_idx_kv
+-------------------------+
| ...                     |
+-------------------------+
```

这里是按同一个 `block_id` 组合 region，不是单独的 indexer allocator。
`BlockPool` 分配、复用或释放 PAGE `i` 时，上述 region 作为同一个逻辑 PAGE
一起生效；offload 和原生 PAGE checkpoint 也按这个完整 PAGE unit 搬运。

空闲 PAGE unit 仍位于左侧 PAGE 地址空间。`BlockPool` 记录每个 PAGE ID 的
逻辑状态，物理位置本身不表示它当前是否空闲：

```text
PAGE ID       0          1          2          3          4
          +----------+----------+----------+----------+----------+
状态      | 在用 KV  | 空闲     | 前缀缓存 | 状态检查 | 空闲     |
          |          |          |          | 点 unit  |          |
          +----------+----------+----------+----------+----------+
```

右侧只用于预留请求持有的 Active SLOT。右侧的空闲 group 可以分配给新的活跃
请求，但它不是空闲 PAGE，也不用于保存 PAGE-backed 状态检查点。

因此，一个逻辑 PAGE 包含压缩 KV envelope 以及对应的 CSA indexer region。
一个 Active SLOT 包含请求的全部可变状态：

```text
+------------------+-----------------------+-------------------+
| compressor state | 所有层的 SWA ring     | MTP 额外条目      |
+------------------+-----------------------+-------------------+
```

MTP 不需要单独的 offload 路径，因为 ring 宽度已经包含 speculative token：

```text
win_with_spec = window_size + mtp_k
```

`get_kv_transfer_tensors()` 导出两个逻辑视图：

```text
block_regions       PAGE，按物理 block ID 正向索引
swa_block_regions   完整 Active SLOT，按请求 group ID 反向索引
staging_region      仅含 compressor state 的 P/D staging，不能作为 sidecar 来源
```

对应的地址 ABI 为：

```text
PAGE address(block_id):
    base + block_id * unit_bytes

Active SLOT address(group_id):
    base + total_bytes - (group_id + 1) * unit_bytes
```

Active SLOT 使用反向索引，因此 SLOT 侧向 PAGE 侧扩展时，已有 group 的地址
仍能保持稳定。

### Active SLOT、原生检查点与 offload sidecar

三者的存储职责不同：

```text
活跃请求状态
------------

Active SLOT group G
+-----------------------------+
| compressor state + SWA ring |
+-----------------------------+
       右侧连续地址


原生 HBM 状态检查点
-------------------

Active SLOT group G
          |
          | BlockPool.reserve_units(K)
          | segmented copy
          v
   +--------+  +---------+       +---------+
   | PAGE 7 |  | PAGE 31 |  ...  | PAGE 92 |
   +--------+  +---------+       +---------+
       左侧 PAGE 地址空间中的任意 PAGE ID，不要求连续


LMCache SLOT offload
--------------------

Active SLOT group G
          |
          v
GPU staging row --> pinned CPU frame --> CPU/NVMe AOS1 sidecar
```

`PagedStateCheckpointCoordinator` 从 `BlockPool` 申请任意空闲 PAGE unit 来
保存原生状态检查点，不会把检查点留在空闲 Active SLOT group 中。

LMCache sidecar 路径不长期占用 HBM PAGE。它只使用有界的临时 GPU staging
row，随后把完整 SLOT 持久化到 CPU 或 NVMe。

## Offload 存储格式

`DSV4PageSlotCodec` 同时持有 PAGE 和 SLOT 的 geometry，并通过 Triton
gather/scatter kernel 搬运原始字节。

```text
get_kv_transfer_tensors()
          |
          +--> block_regions --------------------+
          |                                      |
          +--> swa_block_regions ----------------+
                                                 v
                                      +----------------------+
                                      | DSV4PageSlotCodec   |
                                      | PAGE plan           |
                                      | SLOT plan           |
                                      +----------+-----------+
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v

              LMCache PAGE 对象                              SLOT sidecar
              token/chunk 寻址                               boundary 寻址

  +------------------------------------------+     +----------------------------+
  | block 0                                  |     | AOS1 header，128 bytes     |
  |   NoPE | RoPE | indexer layer 0 | ...    |     | boundary tokens/hash       |
  +------------------------------------------+     | layout fingerprint         |
  | block 1                                  |     | TP size/rank               |
  |   NoPE | RoPE | indexer layer 0 | ...    |     | payload size + CRC32       |
  +------------------------------------------+     +----------------------------+
                                                   | 完整 SLOT payload          |
                                                   | NoPE slot | RoPE slot       |
                                                   +----------------------------+
```

PAGE 对象继续使用 LMCache 根据 token 生成的 chunk key。SLOT 使用内容寻址
key，其中包含链式 boundary hash、layout fingerprint 和 TP 身份。

SLOT payload 不保存源 `group_id`。恢复时总是写入新请求刚分配的
`destination_group`。

## 保存路径

请求到达对齐的检查点边界后，scheduler 生成 `SaveSpec` 和 `SlotSaveSpec`。
connector metadata 会在下一次 forward 前下发，因此 worker 可以先在当前
stream 上快照可变的 Active SLOT，再允许后续 forward 修改它。

```text
Scheduler                         Worker 当前 stream
---------                         -----------------

prefill 到达边界 B
        |
        | build_connector_meta()
        | SaveSpec + SlotSaveSpec
        v
                              上一次 forward 完成
                                        |
                                        v
                              gather_slot(source_group)
                              HBM SLOT --> GPU staging row
                                        |
                                  记录 CUDA event
                                        |
                                        +------> 下一次 forward 可修改 SLOT


后台保存线程
------------

等待 CUDA event
      |
      v
GPU staging row --D2H--> pinned CPU AOS1 frame
      |
      +--> 释放 GPU staging row
      |
      v
gather 已完成的 PAGE blocks
      |
      v
Triton pack --> D2H --> LMCache.store(PAGE chunks)
      |
      v
等待 PAGE 在边界 B 前全部可见
      |
      v
写入 AOS1 header 和 CRC
      |
      v
StorageManager.put(SLOT sidecar)
      |
      v
等待 sidecar 可见
      |
      v
所有 TP rank 上报同一个 SaveOperationId
      |
      v
scheduler 提交 boundary hash B
```

可见性顺序不能颠倒。只有边界 B 之前的 PAGE 全部可见后才能发布 sidecar，
否则 sidecar 可能错误地授权一个不完整检查点。

## 加载路径

scheduler 首先查询普通 LMCache PAGE prefix。对于有状态的 DSV4 请求，它会
把 PAGE hit 收缩到当前 scheduler session 中所有 TP rank 已提交 sidecar 的
最新对齐边界。

```text
Scheduler
---------

相同 prompt
    |
    v
LMCache lookup --> PAGE hit H
    |
    v
选择最新已提交的 PAGE+SLOT 边界 B
    |
    +--> 没有已提交 SLOT：miss，重新计算
    |
    v
分配目标 PAGE blocks 和 Active SLOT group
    |
    v
生成 LoadSpec + SlotLoadSpec(destination_group)
    |
    v
请求进入 WAITING_FOR_REMOTE_KVS


Worker
------

LMCache.retrieve(PAGE)
        |
        v
CPU MemoryObj --> 有界 GPU staging
        |
        v
Triton 按物理 block ID scatter PAGE
        |
        +--> PAGE 缺失：失败
        |
        v
借用并校验 AOS1 sidecar
        |
        | magic/version、boundary、payload size、
        | fingerprint、TP 身份和 CRC32
        |
        +--> sidecar 缺失或无效：失败
        |
        v
CPU payload --> GPU staging --> scatter_slot(destination_group)
        |
        v
所有 TP rank 完成
        |
        v
唤醒请求，继续 suffix prefill/decode
```

worker 先恢复 PAGE，再恢复 SLOT。只有两者都成功，composite load 才成功；
否则请求回退到重新计算。

## 主要代码入口

- `atom/model_ops/attentions/deepseek_v4_attn.py`
  - `allocate_per_req_cache()` 创建共享 plane，并把 state arena 嵌入每个完整
    Active SLOT。
  - `get_kv_transfer_tensors()` 导出正向 PAGE region 和反向 SLOT region。
- `atom/model_ops/attentions/v4_pool_geometry.py`
  - `UnifiedPoolGeometry` 统一维护 PAGE/SLOT 地址计算。
- `atom/model_engine/block_pool.py`
  - `BlockPool` 管理在用、缓存、空闲和状态检查点 PAGE unit。
- `atom/model_engine/page_unit_checkpoint.py`
  - `PagedStateCheckpointCoordinator` 为原生 HBM 状态检查点申请 PAGE unit，
    并将其作为一个原子对象释放。
- `atom/kv_transfer/offload/hybrid/dsv4/codec.py`
  - `DSV4PageSlotCodec` 构造 PAGE/SLOT copy plan。
  - `DSV4CheckpointCodec` 实现 AOS1 frame。
  - `DSV4CheckpointStore` 通过 LMCache storage tier 持久化 sidecar。
- `atom/kv_transfer/offload/hybrid/dsv4/triton_page_slot.py`
  - 实现正向和反向的原始字节 gather/scatter。
- `atom/kv_transfer/offload/hybrid/dsv4/connector.py`
  - `DSV4OffloadScheduler` 选择边界并构造请求 metadata。
  - `DSV4OffloadConnector` 执行快照、PAGE 传输、sidecar 提交、校验和恢复。
- `atom/kv_transfer/offload/_block_gpu_connector.py`
  - 通过有界 GPU staging 在 LMCache `MemoryObj` 和 ATOM PAGE block 之间传输。

## 当前安全约束

- 有状态 DSV4 必须在同一边界同时恢复 PAGE 和 SLOT。
- 当实际 HBM prefix floor 非零时，跳过有状态 offload load；版本 1 尚不支持
  把 HBM 状态检查点与更晚边界的 SLOT sidecar 合并。
- SLOT commit 只在当前 scheduler session 内有效。
- 尚不支持 FP4 indexer offload，也不支持 `pipeline_parallel_size > 1`。
