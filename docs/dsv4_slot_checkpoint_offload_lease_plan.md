# DSV4 SLOT interval checkpoint offload lease 修改计划

状态：**Implemented / Ready for Review**
实现状态：**第一版已于 2026-08-13 按本文方案落地并完成聚焦回归**

生产代码已收敛到 `atom/kv_transfer/offload/hybrid/dsv4/` 五文件 package；
PAGE/SLOT 使用统一 codec，SLOT 仅按 interval 保存，checkpoint destination 通过
all-TP staging completion 管理 lease，GPU temp row 在 D2H 后立即释放。第一版仍然
使用标准 PAGE chunks 与独立 AOS1 checkpoint object。跨 scheduler restart 的 SLOT
发现保持 session-local fail-closed，本版未实现 persistent manifest/contains adapter。

## 1. 目标

在不改变 GPU/HBM 中 PAGE 与 SLOT 物理布局的前提下：

1. PAGE 继续按 LMCache chunk（当前默认 256 tokens）增量 offload。
2. 完整 SLOT snapshot 只按照配置的 state checkpoint interval 保存。
3. interval checkpoint 从 `StateGroupPool` free list 取得一个 destination group，
   并对它建立短生命周期 lease。
4. lease 只持有到该 checkpoint 的内容已经完整进入 connector GPU temp
   buffer；之后立即把 group 放回 free list，不等待 D2H、PAGE publication、
   AOS1 put 或 StorageManager 可见性确认。
5. PAGE 与 SLOT 继续使用独立存储对象和联合命中/提交语义；本计划不做
   PAGE+SLOT bundle 合包。

### 1.1 一眼看懂：哪些内容被复制，哪个资源先释放

```text
                         GPU / HBM

  +------------------+       D2D copy       +---------------------------+
  | live SLOT        | -------------------> | checkpoint group dst      |
  | src (request用)  |                      | LEASED: 不在 free list     |
  +------------------+                      +-------------+-------------+
                                                        |
                                                        | gather full SLOT
                                                        v
                                          +---------------------------+
                                          | connector GPU temp row    |
                                          +-------------+-------------+
                                                        |
                                   gather event E done  |
                              +-------------------------+------------------+
                              |                                            |
                              v                                            v
               all TP ranks confirm E                         D2H: GPU temp -> CPU
                              |                                            |
                              v                                            v
               release dst to free list                       +-------------------+
               (保留 checkpoint hash)                         | CPU AOS1 frame    |
                                                              +---------+---------+
                                                                        |
                                                                        | put
                                                                        v
  PAGE chunks --store--> PAGE visible -----------------------> SLOT object visible
                                                                        |
                                                                        v
                                                        scheduler commits boundary B
```

图中最重要的是中间的分叉：`dst` 在 SLOT bytes 进入 GPU temp 后即可释放；右边
的 D2H、PAGE visibility 和 StorageManager put 可以继续慢慢执行。

## 2. 非目标

- 不改变模型 forward 使用的 active/live SLOT 布局。
- 不改变 PAGE 的标准 LMCache key、chunk codec、去重和查找路径。
- 不把 SLOT 变成 256-token 增量；每次保存的仍是完整 state snapshot。
- 不用远端 offload 成功与否决定 native HBM checkpoint 是否保留。
- 本计划不扩展当前 DSV4 PAGE+SLOT 的 PP 支持范围。

## 3. 名词和资源

| 名称 | 含义 |
|---|---|
| `src` | 请求当前正在使用的 live state group。 |
| `dst` | 从 `StateGroupPool` free list 取得的 native checkpoint group。 |
| `copy_id` | scheduler 生命周期内单调递增、唯一的 checkpoint lease ID。 |
| checkpoint lease | 在 GPU copy/gather 期间阻止 `dst` 被重新分配、adopt 或 retire。 |
| GPU temp row | connector 自己管理的完整 SLOT staging row。 |
| CPU frame | AOS1 header + SLOT payload 的 host buffer。 |

## Connector 端到端实现（重点）

本节把 connector 内部从“拿到 block/group”一直画到“调用 LMCache 存取”完整展开。
当前实现已经删除独立的 `ATOMPageRegionCodec` 和 `ATOMSlotSidecarCodec`，PAGE 与
SLOT 均由 DSV4 专用的 `DSV4PageSlotCodec` 计算 typed plan 并执行 gather/scatter。

### C1. 启动时：模型先把可搬运的 GPU 地址注册给 connector

`DeepseekV4AttentionMetadataBuilder.get_kv_transfer_tensors()` 不把大 tensor 本身
交给 connector，而是交一组稳定的 region descriptor：

```text
KVTransferTensors
|
+-- block_regions[]       PAGE，forward-indexed by block_id
|     region.unit_addr(b) = base_addr + b * unit_bytes
|
+-- swa_block_regions[]   完整 SLOT，reverse-indexed by state group
|     region.unit_addr(g) = base_addr + total_bytes - (g + 1) * unit_bytes
|
+-- num_blocks            scheduler-visible PAGE pool size
+-- num_slots             StateGroupPool group count
+-- expected_full_slot_region_count
```

PAGE region 覆盖 compressed KV planes 和 CSA indexer planes；完整 SLOT region
覆盖同一个 state group 的 compressor state 与 SWA rows。历史字段名
`swa_block_regions` 容易误导，但这里装的是完整 SLOT，不只是 SWA。

启动链路如下：

```text
ModelRunner.allocate_kv_cache()
        |
        v
attention builder creates KVTransferTensors
        |
        v
set_kv_cache_data(..., transfer_tensors, num_blocks)
        |
        v
DSV4OffloadConnector.register_kv_caches()
        |
        +--> DSV4PageSlotCodec(
        |               page_regions=block_regions,
        |               slot_regions=swa_block_regions)
        |       page_bytes_per_block = sum(PAGE region.unit_bytes)
        |       slot_bytes           = sum(SLOT region.unit_bytes)
        |       bytes_per_block      = page_bytes_per_block
        |                              # LMCache PAGE protocol compatibility
        |
        +--> BlockGPUConnector(unified_codec, block_size, chunk_size)
        |
        +--> ATOMRawBytesLMCacheMetadata
        |       dtype  = uint8
        |       shape  = nblocks * bytes_per_block
        |       groups = 1
        |
        +--> LMCacheEngineBuilder.get_or_create(..., BlockGPUConnector, ...)
        |       engine.fmt = KV_2LTD       # 只作为 allocator 可接受的容器格式
        |       engine.post_init()
        |
        +--> same unified_codec supplies SLOT plan/staging methods
        +--> DSV4StagingAdmission(staging_rows)
        +--> DSV4CheckpointStore(engine.storage_manager)
```

已经完成的迁移关系是：

```text
REMOVED                                      CURRENT

ATOMPageRegionCodec   ----\
                             +------------> DSV4PageSlotCodec
ATOMSlotSidecarCodec  ----/                  one region/plan/copy engine
```

LMCache 看到的是一个 opaque `uint8` group。它负责 token chunk/key、MemoryObj、
CPU/NVMe tier 和 eviction；ATOM codec 负责解释“这些 bytes 在 GPU 哪个地址”。

对应代码入口：

- `atom/model_ops/attentions/deepseek_v4_attn.py::get_kv_transfer_tensors`
- `atom/model_engine/model_runner.py::allocate_kv_cache`
- `atom/kv_transfer/offload/hybrid/dsv4/connector.py::register_kv_caches`
- `atom/kv_transfer/offload/_offload_common.py::build_offload_engine`

### C2. Scheduler 如何把 block table 交给 worker

connector 不从 LMCache 猜 GPU block。scheduler 已经拥有每个请求真实的
`seq.block_table`，构造 `LMCacheReqMeta` 时直接做快照：

```text
BlockPool
   |
   +--> prefix hit: reuse an indexed physical block ID
   +--> cache miss: pop/allocate a fresh physical block ID
   |
   v
BlockManager.allocate() / may_append()
   |
   v
Sequence
|
+-- token_ids:   [t0, t1, ..., tB-1]
+-- block_table: [42, 7, 91, 13, ...]
                  ^ values are GPU PAGE pool block IDs
                  ^ list order is logical prefix order
```

```text
DSV4OffloadScheduler.build_connector_meta()
        |
        +--> token_ids = seq.token_ids[:token_end]
        +--> block_ids = list(seq.block_table)
        +--> SaveSpec(skip_leading_tokens=saved)
        +--> or LoadSpec(hbm_cached_tokens, lmcache_cached_tokens)
        +--> optional SlotSaveSpec(B, hash, source_group)
        +--> optional SlotLoadSpec(B, hash, destination_group)
        |
        v
LMCacheOffloadMetadata.requests[]
        |
        v
EngineCore -> ModelRunner.process_kvconnector_output()
        |
        v
DSV4OffloadConnector.start_load_kv_with_state_checkpoints()
```

`block_ids` 的列表下标是逻辑 block number，列表里的值才是 PAGE region 的
物理 pool block ID。例如 `virtual_block_size=32`、LMCache range 为
`[256, 512)`：

```text
start_block = 256 / 32       = 8
end_block   = ceil(512 / 32) = 16

chunk block IDs = req.block_ids[8:16]
                = 8 physical PAGE pool IDs
```

该转换由 `BlockGPUConnector._range_block_ids()` 完成。它同时拒绝未对齐的
`start`、越过 block table 的 range，以及缺失的 `block_ids`。

因此 connector **不拥有也不分配 PAGE block**：BlockManager 先完成 allocation，
connector 只消费 metadata 中的 block table，并按 LMCache 给出的 token range 对
它做 slice。

### C3. PAGE save：GPU PAGE regions -> LMCache MemoryObj -> storage tiers

scheduler 只发送已完成的 PAGE frontier，并通过 `skip_leading_tokens` 避免重复保存
旧 chunk。worker 的入口是：

```python
self._engine.store(
    torch.tensor(token_ids),
    mask=mask,                 # 已保存 prefix 为 False，新 chunk 为 True
    block_ids=req.block_ids,   # ATOM 扩展参数，传给自定义 GPU connector
    req_id=str(req.req_id),
)
```

完整调用链：

```text
DSV4OffloadConnector._do_save_req
        |
        | engine.store(tokens, mask, block_ids, req_id)
        v
LMCache ChunkedTokenDatabase
        |
        +--> split selected tokens into chunk ranges [start, end)
        +--> derive rolling content key for each chunk
        +--> allocate one uint8 MemoryObj per chunk
        |
        v
BlockGPUConnector.batched_from_gpu(memory_objs, starts, ends, block_ids=...)
        |
        +--> _range_block_ids(): token range -> physical block IDs
        |
        +--> group a bounded number of chunks into GPU staging batches
        |
        +--> PAGE gather on pack_stream
        |       DSV4PageSlotCodec.page_plan(block_ids)
        |       Triton PAGE gather(PAGE regions -> GPU staging)
        |
        +--> D2H/copy on copy_stream
                GPU staging slices -> LMCache MemoryObj tensors
        |
        v
LMCache StorageManager.batched_put(...)
        |
        +--> LocalCPU tier
        +--> optional LocalDisk/NVMe tier
        +--> configured remote tiers
```

PAGE gather 的 byte layout 是 block-major、region-minor：

```text
GPU source regions

region0:  ... [block42 bytes] ... [block7 bytes] ...
region1:  ... [block42 bytes] ... [block7 bytes] ...
region2:  ... [block42 bytes] ... [block7 bytes] ...

                  Triton gather_copy_plan
                           |
                           v
one LMCache chunk MemoryObj payload

+----------+----------+----------+----------+----------+----------+
| b42.r0   | b42.r1   | b42.r2   | b7.r0    | b7.r1    | b7.r2    |
+----------+----------+----------+----------+----------+----------+
```

对每个 `(block_id, region)`，copy plan 保存：

```text
src_addr  = region.base_addr + block_id * region.unit_bytes
dst_offset = previous payload bytes
nbytes     = region.unit_bytes
```

LMCache 不读取这个 layout；它把 MemoryObj 当 opaque bytes 存储。`KV_2LTD` 不是
真实 PAGE tensor layout，只是让 LMCache allocator 接受这个 `uint8` MemoryObj。

#### C3.1 LMCache 原生编排 + ATOM custom GPU connector

这里不是绕开 LMCache，而是使用 LMCache 官方提供的 custom GPU connector 扩展点。
切分边界如下：LMCache 继续拥有 token/chunk/key、MemoryObj 和存储层；ATOM connector
只负责把 ATOM 特有的 PAGE 地址布局转换成 LMCache 可保存的连续 bytes。

```text
                         SAVE / offload

 scheduler metadata
 tokens + mask + block_ids
          |
          v
+-----------------------------------------------------------------------+
| LMCache 原生逻辑（保持不改）                                          |
|                                                                       |
| ChunkedTokenDatabase                                                  |
|   tokens -> 256-token chunks -> content keys                          |
|                                                                       |
| MemoryObj allocator                                                   |
|   one opaque uint8 payload per chunk                                  |
+-------------------------------+---------------------------------------+
                                | official custom GPU connector call
                                | batched_from_gpu(memory_objs,
                                |                  ranges, block_ids)
                                v
+-----------------------------------------------------------------------+
| ATOM BlockGPUConnector（layout adapter）                              |
|                                                                       |
| block_ids + KVTransferRegion[]                                        |
|        |                                                              |
|        +--> address = region.base + block_id * region.unit_bytes      |
|        |                                                              |
|        +--> build copy plan                                           |
|        |      [b42.r0][b42.r1][b42.r2][b7.r0][b7.r1][b7.r2]           |
|        |                                                              |
|        +--> Triton gather: PAGE regions -> GPU staging                |
|        +--> D2H: GPU staging -> MemoryObj payload                     |
+-------------------------------+---------------------------------------+
                                |
                                | connector returns filled MemoryObj
                                v
+-----------------------------------------------------------------------+
| LMCache 原生逻辑（保持不改）                                          |
|                                                                       |
| StorageManager.batched_put(content_key, MemoryObj)                    |
|        +--> CPU                                                       |
|        +--> NVMe                                                      |
|        +--> remote tier                                               |
|                                                                       |
| StorageManager 只存 opaque bytes，不解析 b42.r0 等内部布局             |
+-----------------------------------------------------------------------+
```

load 完全沿相同边界反向执行：

```text
                         LOAD / restore

 LMCache token/chunk key lookup
          |
          v
 StorageManager.get -> opaque MemoryObj
          |
          | official custom GPU connector call
          | batched_to_gpu(memory_objs, ranges, new_block_ids)
          v
 ATOM BlockGPUConnector
          |
          +--> H2D: MemoryObj -> GPU staging
          +--> rebuild the same copy plan for new_block_ids
          +--> Triton scatter:
          |       staging[b42.r1]
          |          -> region1.base + new_block42 * region1.unit_bytes
          v
 ATOM PAGE regions restored
```

因此扩展点的含义是：

```text
LMCache knows:  token range, chunk key, MemoryObj lifetime, storage tiers
ATOM knows:     block_id -> CSA/PAGE region address, gather/scatter layout

                         custom GPU connector
LMCache control plane  <---------------------->  ATOM data-layout adapter
```

如果要求 LMCache “原生”理解这个 PAGE layout，就必须把 `KVTransferRegion[]`、
region/group shape、地址公式和对应 gather/scatter kernel 都上游到 LMCache，改动的不再
只是配置。保持 custom connector 可以让 PAGE layout 演进留在 ATOM 内部，同时复用
LMCache 已有的 token/key/storage/eviction 行为，接口面和回归面最小。

#### C3.2 一个 DSV4 专用的 `DSV4PageSlotCodec`

这里的“合并”分三层，本计划第一版只强制第一层：

```text
layer 1: one codec class
         PAGE + SLOT address/plan/gather/scatter 统一       [FIRST VERSION]

layer 2: one GPU staging transaction
         [PAGE payload][SLOT payload] 一次 gather/scatter   [RESERVED API]

layer 3: one Storage object
         [header][PAGE][SLOT] 一个 bundle key/put/get       [NOT FIRST VERSION]
```

第一层应该立即做，因为当前两个 codec 的核心已经相同：它们都生成 `PageCopy[]`，
再由相同的 `build_copy_tiles(..., 1024)` 和 Triton raw-byte kernel 搬运。差别只是
index domain 和地址方向：

```text
PAGE RegionSet
  index      = physical block_id
  direction  = forward
  unit_addr  = base + block_id * unit_bytes

SLOT RegionSet
  index      = state group
  direction  = reverse
  unit_addr  = base + total_bytes - (group + 1) * unit_bytes
```

计划中的单一对象：

```text
DSV4PageSlotCodec
|
+-- page_regions: RegionSet
|     +-- C128/HCA Main PAGE plane bytes
|     +-- C4/CSA Main PAGE plane bytes
|     +-- C4/CSA Indexer PAGE regions
|
+-- slot_regions: RegionSet
|     +-- C4/C128 compressor state
|     +-- Dense/C128/C4 SWA
|     +-- optional state-carried window + padding
|
+-- page_plan(block_ids, dst_offset=0)
+-- slot_plan(group, dst_offset=0)
+-- checkpoint_plan(block_ids, group)
|
+-- gather(plan, gpu_staging, stream)
+-- scatter(gpu_staging, plan, stream)
|
+-- page_bytes_per_block
+-- slot_bytes
+-- bytes_per_block == page_bytes_per_block
```

##### C3.2.1 目录结构

这个 codec 是 DeepSeek-V4 GPU layout 的 vertical package，不放在看起来通用的
`hybrid/page_slot_codec.py`：

```text
atom/kv_transfer/offload/
|
+-- _block_gpu_connector.py
|     generic LMCache GPUConnectorInterface
|     MemoryObj grouping + PAGE staging pipeline
|     does not know C4/C128/state/SWA
|
+-- atom_lmcache_staging.py
|     generic staging buffers, streams and CUDA events
|
+-- dense/
|     +-- connector.py
|     +-- kv_byte_codec.py
|     +-- triton_kv_staging.py
|           dense chunk/segment-major kernels only
|
+-- hybrid/
      |
      +-- dsv4/
            +-- __init__.py
            |     export DSV4 connector/profile/codec stable surface
            |
            +-- connector.py
            |     PAGE/SLOT orchestration, interval, lease, completion,
            |     admission, D2H and LMCache calls
            |
            +-- policy.py
            |     DSV4OffloadProfile + config parsing
            |     boundary/hash/fingerprint/commit policy
            |     staging-row admission/quarantine
            |
            +-- codec.py
            |     DSV4PageSlotCodec: GPU layout/plan/gather/scatter
            |     DSV4CheckpointCodec: AOS1 key/header/fingerprint/CRC
            |     DSV4CheckpointStore: StorageManager ownership
            |     private RegionSet / payload plan types
            |
            +-- triton_page_slot.py
                  DSV4 region gather/scatter kernels and launch wrappers
```

因此，原先分散的实现已经按下图收拢：

```text
REMOVED                                      CURRENT

hybrid/sidecar_format.py      ---\
                                 +------->   hybrid/dsv4/codec.py
hybrid/page_region_codec.py   ---/           three focused classes in one file
hybrid/slot_codec.py          ---/
hybrid/store.py               ----------->   hybrid/dsv4/codec.py
planned dsv4/layout.py        ---\
                                 +------->   do not create; keep inside codec.py
planned dsv4/copy_plan.py     ---/
hybrid/policy.py              ---\
                                 +------->   hybrid/dsv4/policy.py
hybrid/profiles/dsv4.py       ---/
hybrid/profiles/base.py       ----------->   removed
hybrid/connector.py           ----------->   root offload/connector.py imports dsv4 directly
hybrid/gpu_connector.py       ----------->   dsv4 uses shared BlockGPUConnector directly
hybrid/admission.py           ----------->   hybrid/dsv4/policy.py
offload/copy_plan.py          ----------->   removed; typed plan stays in dsv4/codec.py
```

`sidecar_format.py` 不再作为独立文件保留，它和 PAGE/SLOT GPU codec 一起收进
`dsv4/codec.py`，但不能合成一个大 class。文件内维持三个明确 class：

```text
dsv4/codec.py
|
+-- DSV4PageSlotCodec
|     GPU region geometry
|     PAGE/SLOT typed plans
|     gather/scatter launch
|     BlockByteCodec PAGE API
|
+-- DSV4CheckpointCodec
|     AOS1 key/header encode/decode
|     fingerprint/size/CRC validation
|     no StorageManager ownership
|
+-- DSV4CheckpointStore
      StorageManager allocate/put/get/contains/remove
      MemoryObj borrow/refcount ownership
```

`layout.py` 和 `copy_plan.py` 也不创建。它们原本只是 codec 的 private geometry 和
plan DTO，没有第二个 consumer；拆文件只会增加 import 和跳转。最终在
`codec.py` 内按下列顺序组织：

```text
codec.py
|
+-- constants / AOS1 structs
+-- private geometry
|     _AddressMode
|     _RegionSnapshot
|     _RegionSet
|
+-- typed plan DTOs
|     DSV4PayloadKind
|     DSV4PayloadSection
|     DSV4CopyPlan
|
+-- DSV4PageSlotCodec
+-- DSV4CheckpointCodec
+-- DSV4CheckpointStore
```

这样减少文件数量，同时仍保留 class-level single responsibility。connector 同时持有
三个 class 实例；`DSV4CheckpointStore` 只调用 `DSV4CheckpointCodec`，不调用 GPU
plan/gather API。

如果将来 PAGE+SLOT 真正升级成单 bundle，则由
`DSV4CheckpointCodec` 从 AOS1 演进到新版本/新 magic；不需要修改
`DSV4PageSlotCodec` 的 GPU address/kernel contract。

`profiles/dsv4.py` 当前也没有跨模型价值，和 `policy.py` 一起收进
`dsv4/policy.py`。当前 `DSV4OffloadProfile` 只有 DSV4 使用时，将其收紧为
`DSV4OffloadProfile`；以后出现第二种 hybrid 模型时，由它自己的 model package
提供 policy/profile，再在根 connector 做显式 registry/factory，而不是提前保留
一个只有单实现的 `profiles/` 层。

DSV4 package 内部依赖固定为：

```text
dsv4/connector.py
|
+-- imports policy.py                  profile/config/cadence/boundary/hash
+-- imports codec.DSV4PageSlotCodec    GPU layout movement
+-- imports codec.DSV4CheckpointCodec  persisted bytes validation
+-- imports codec.DSV4CheckpointStore  LMCache StorageManager ownership

codec.DSV4PageSlotCodec
+-- lazy imports triton_page_slot.py only on GPU execution

codec.DSV4CheckpointStore
+-- calls DSV4CheckpointCodec
+-- lazy imports LMCache runtime types in __init__

forbidden reverse dependencies:
  codec.py             -X-> connector/policy
  triton_page_slot.py -X-> connector/LMCache/StorageManager
```

测试继续遵循 PR #1683 的 flat `tests/test_*.py` 习惯，不为少量文件再造深层 package：

```text
tests/
+-- test_dsv4_page_slot_codec.py
|     typed API, BlockByteCodec contract, validation, composite plan
|
+-- test_dsv4_page_slot_triton.py
|     GPU BF16/FP8 gather-scatter round trip, multi-tile, non-default stream
|
+-- test_deepseek_v4_transfer_regions.py
|     real model region registration -> codec integration
|
+-- test_lmcache_offload_v4_page_slot.py
|     connector/admission/event/D2H/store lifecycle
|
+-- test_dsv4_checkpoint_codec.py
+-- test_dsv4_checkpoint_format.py
+-- test_dsv4_checkpoint_store.py
|     AOS1 wire and StorageManager ownership
|
+-- test_dsv4_staging_admission.py
+-- test_dsv4_policy.py
      temp-row ownership/quarantine and checkpoint-grid policy
```

这沿用 PR #1683 最重要的职责边界：

```text
connector orchestration
        |
        v
model profile / source selection
        |
        v
DSV4 GPU layout codec
        |
        v
Triton raw-byte movement

AOS1/wire validation -------- separate
StorageManager ownership ---- separate
```

不直接照搬 PR #1683 的 `BundleCodec`：那个类只负责 CPU container/header/CRC，
不负责 GPU source address 或 gather。这里的 `DSV4PageSlotCodec` 明确是 GPU layout
codec；AOS1 和 storage 由同一 `codec.py` 中另外两个 class 负责，不混入这个 GPU
class 的方法和依赖。

##### C3.2.2 public class 与 private geometry

`codec.py` 包含三个 public class。GPU class surface 计划如下：

```python
class DSV4PageSlotCodec:
    def __init__(
        self,
        page_regions: Sequence[KVTransferRegion],
        slot_regions: Sequence[KVTransferRegion],
        *,
        num_blocks: int,
        num_slots: int,
        device: torch.device | str,
    ) -> None: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def num_blocks(self) -> int: ...

    @property
    def num_slots(self) -> int: ...

    @property
    def page_bytes_per_block(self) -> int: ...

    @property
    def bytes_per_block(self) -> int:
        # BlockByteCodec / LMCache PAGE contract.
        # MUST equal page_bytes_per_block; never includes SLOT.
        ...

    @property
    def slot_bytes(self) -> int: ...

    @property
    def has_fused_chunk_major_staging(self) -> bool:
        # Compatibility property for BlockGPUConnector.
        ...

    def page_plan(
        self,
        block_ids: Sequence[int],
        *,
        buffer_offset: int = 0,
    ) -> DSV4CopyPlan: ...

    def slot_plan(
        self,
        group: int,
        *,
        buffer_offset: int = 0,
    ) -> DSV4CopyPlan: ...

    def checkpoint_plan(
        self,
        block_ids: Sequence[int],
        group: int,
        *,
        buffer_offset: int = 0,
    ) -> DSV4CopyPlan: ...

    def gather(
        self,
        plan: DSV4CopyPlan,
        dst: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None: ...

    def scatter(
        self,
        src: torch.Tensor,
        plan: DSV4CopyPlan,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None: ...

    # Exact BlockByteCodec compatibility; PAGE only.
    def gpu_to_chunk_major_device_buffer(
        self, device_buf, block_id_groups, stream=None
    ) -> None: ...

    def chunk_major_device_buffer_to_gpu(
        self, device_buf, block_id_groups, stream=None
    ) -> None: ...

    # Typed SLOT convenience; buffers are caller-owned.
    def gather_slot(self, group, dst, *, stream=None) -> None: ...
    def scatter_slot(self, src, group, *, stream=None) -> None: ...
```

同一文件中的 persisted-object class：

```python
class DSV4CheckpointCodec:
    def __init__(self, *, fingerprint: bytes, tp_size: int, tp_rank: int): ...

    def make_key(self, *, boundary_block_hash: int) -> DSV4CheckpointKey: ...

    def frame_size(self, *, payload_bytes: int) -> int: ...

    def finalize_tensor_(
        self,
        framed: torch.Tensor,
        *,
        boundary_tokens: int,
        boundary_block_hash: int,
    ) -> torch.Tensor: ...

    def decode_tensor(
        self,
        framed: torch.Tensor,
        *,
        expected_boundary_tokens: int,
        expected_boundary_block_hash: int,
        expected_payload_bytes: int,
    ) -> tuple[DSV4CheckpointHeader, torch.Tensor]: ...
```

`DSV4CheckpointKey`、`DSV4CheckpointHeader` 和 `DSV4CheckpointError` 也定义在
`codec.py`。

同一文件中的 StorageManager adapter：

```python
class DSV4CheckpointStore:
    def __init__(self, engine, *, checkpoint_codec: DSV4CheckpointCodec): ...

    def put(self, key: DSV4CheckpointKey, framed: torch.Tensor) -> bool: ...

    @contextmanager
    def borrow(self, key: DSV4CheckpointKey): ...

    def contains(self, key: DSV4CheckpointKey) -> bool: ...

    def invalidate(self, key: DSV4CheckpointKey) -> bool: ...
```

`DSV4CheckpointStore` 只负责 StorageManager/MemoryObj ownership，不解析 GPU plan；
header/CRC 校验全部委托给 `DSV4CheckpointCodec`。

同文件不会强制 CPU format 用户加载 Triton。`codec.py` 顶层禁止 import
`triton_page_slot`；只有 `DSV4PageSlotCodec.gather/scatter` 第一次执行 GPU path 时
做 local/lazy import。这样 scheduler、store 和纯 CPU AOS1 测试只使用
`DSV4CheckpointCodec` 时不触发 Triton JIT/module load。

同理，`codec.py` 顶层不 import `lmcache`；只有
`DSV4CheckpointStore.__init__()` local import `CacheEngineKey` 和 `MemoryFormat`。
因此 import DSV4 GPU codec 或纯 CPU checkpoint codec 时，不要求 LMCache runtime。

`codec.py` 内部的 private immutable geometry section：

```text
_AddressMode
  FORWARD    # PAGE
  REVERSE    # SLOT

_RegionSnapshot
  base_addr
  total_bytes
  unit_bytes
  address_mode

_RegionSet
  payload_kind
  item_count
  regions[]
  bytes_per_item
  unit_addr(region, item_id)
```

构造 codec 时立即 snapshot `KVTransferRegion`，之后不再读取可变 descriptor；同时
验证：

- PAGE regions 全部 forward indexed，且至少覆盖 `num_blocks * unit_bytes`；
- SLOT regions 全部 reverse indexed，且至少覆盖 `num_slots * unit_bytes`；
- base/total/unit 正数并满足 device/address alignment contract；
- PAGE/SLOT region 顺序进入 layout fingerprint，不能静默重排。

`codec.py` 内部的 public plan types 是 semantic section，不在 PAGE 高频路径为每个
`(block, region, tile)` 永久物化 Python object：

```python
class DSV4PayloadKind(Enum):
    PAGE = "page"
    SLOT = "slot"


@dataclass(frozen=True)
class DSV4PayloadSection:
    kind: DSV4PayloadKind
    item_ids: tuple[int, ...]       # block IDs or one SLOT group
    buffer_offset: int
    nbytes: int


@dataclass(frozen=True)
class DSV4CopyPlan:
    sections: tuple[DSV4PayloadSection, ...]
    payload_bytes: int
    required_buffer_bytes: int
```

CPU 单测/debug 可以通过 `iter_reference_spans(plan)` 展开成：

```text
DSV4CopySpan
  item_id
  region_index
  device_addr        # gather source / scatter destination
  buffer_offset
  nbytes
```

`device_addr` 不再叫旧 `PageCopy.src_addr`，因为 scatter 时它是 destination，SLOT
使用它时也不是 PAGE block。

##### C3.2.3 DSV4 专用 Triton kernel

`dsv4/triton_page_slot.py` 属于 codec package，并由 `DSV4PageSlotCodec` 唯一调用：

```text
codec.py / DSV4PageSlotCodec
        |
        | DSV4CopyPlan + compiled RegionSet metadata
        v
triton_page_slot.py
        +-- _gather_region_items_kernel
        +-- _scatter_region_items_kernel
        +-- gather_region_items(...)
        +-- scatter_region_items(...)
```

“DSV4 专用”是指它实现 DSV4 PAGE-forward/SLOT-reverse region ABI，不表示把
`Nc=30`、`Nh=31`、C4/C128 tensor shape 写死在 kernel。C4/C128/state/SWA 已经由
attention builder 注册的 region 顺序和 `unit_bytes` 表达；kernel 只搬 raw bytes。

静态 region/tile metadata 在 codec 初始化时构造一次：

```text
_DeviceRegionPlan
  region_base_ptrs_d[R]       int64
  region_total_bytes_d[R]     int64
  region_unit_bytes_d[R]      int64

  tile_region_d[T]            int32
  tile_unit_offset_d[T]       int64
  tile_output_offset_d[T]     int64
  tile_valid_bytes_d[T]       int32

  bytes_per_item
  reverse_indexed
```

`T` 只包含真实存在的 tile。这样不会用最大 region 长度给小 C4-indexer region
制造大量空 CTA，也避免当前 hot path 每次展开全部 absolute `PageCopy`、再创建三组
device metadata tensors。每次 PAGE 调用只需要传动态 block IDs；SLOT 只传一个
group ID。

gather kernel 的逻辑：

```text
grid = (num_items, tiles_per_item)

item_pos = program_id(0)
tile_id  = program_id(1)
item_id  = item_ids[item_pos]
region   = tile_region[tile_id]

if FORWARD/PAGE:
    unit_base = region_base[region]
              + item_id * region_unit_bytes[region]

if REVERSE/SLOT:
    unit_base = region_base[region]
              + region_total_bytes[region]
              - (item_id + 1) * region_unit_bytes[region]

src = unit_base + tile_unit_offset[tile_id] + lane
dst = output_base
    + item_pos * bytes_per_item
    + tile_output_offset[tile_id]
    + lane

dst[...] = src[...]                    # raw uint8
```

`REVERSE` 作为 Triton `constexpr`，PAGE/SLOT 各编译一个 specialization，kernel
内部没有每个 byte 的 runtime branch。scatter 完全反向：相同 metadata、相同 offset，
交换 raw buffer 与 device region 的读写方向。

第一版保留当前已验证的 `TILE_BYTES = 1024`：

```text
one registered region unit

+---------+---------+---------+-------+
| tile 0  | tile 1  | tile 2  | tail  |
| 1024 B  | 1024 B  | 1024 B  | mask  |
+---------+---------+---------+-------+
```

后续只在 MI300/MI355 benchmark 后评估 2 KiB/4 KiB 和 warp 数；不在结构重构时
同时改变 tile 性能参数。

kernel wrapper 的同步 contract：

```text
wrapper may:
  validate CUDA uint8 contiguous buffer
  enqueue metadata/id copy on supplied stream
  enqueue Triton kernel on supplied stream

wrapper must not:
  stream.synchronize()
  create/record completion event
  release staging buffer
  call LMCache or StorageManager
```

PAGE 的 ready/free event 仍由 `_block_gpu_connector.py` 的双-stream pipeline 管理；
SLOT gather event 仍由 `DSV4OffloadConnector` 在 native `src -> dst` 之后记录。这样
codec/kernel 不会偷偷扩大 lease 生命周期。

composite `checkpoint_plan()` 不需要第三套 monolithic kernel。在同一个 stream 上按
section 顺序 enqueue 两个 specialization，最后由 caller 记录一个 event：

```text
one staging buffer

+---------------- PAGE ----------------+--------------- SLOT ---------------+
| gather_region_items(REVERSE=False)   | gather_region_items(REVERSE=True)   |
+--------------------------------------+--------------------------------------+
                                        |
same stream: PAGE enqueue -> SLOT enqueue -> caller records one completion event
```

一次物理 staging/bundle 不要求只有一次 kernel launch；保持 PAGE/SLOT specialization
可以让地址方向和 metadata 更简单。

##### C3.2.4 staging ownership 与依赖方向

新 codec 不再像当前 `ATOMSlotSidecarCodec` 一样在内部
`torch.empty((staging_slots, slot_bytes))`。staging 必须由 caller 持有：

```text
PAGE staging owner
  BlockGPUConnector / atom_lmcache_staging pipeline

SLOT temp owner
  DSV4OffloadConnector + DSV4StagingAdmission

DSV4PageSlotCodec
  validates caller buffer and enqueues copies only
```

这样 SLOT temp 才能在 D2H 完成后由 connector 立即释放或在 completion 不确定时
quarantine，而不是把资源生命周期藏进 layout class。

严格依赖方向：

```text
deepseek_v4_attn.py
  registers DSV4 PAGE/SLOT regions
             |
             v
hybrid/dsv4/DSV4PageSlotCodec
  layout -> plan -> triton_page_slot
             ^
             |
BlockGPUConnector (PAGE)       DSV4OffloadConnector (SLOT)
             |                            |
LMCache token chunks             D2H -> AOS1 -> StorageManager
```

`codec.py` 虽然集中三个 class，依赖仍按 class 单向约束：

```text
DSV4PageSlotCodec
  may use private layout/plan types
  may lazy-import triton_page_slot
  must not use LMCache/StorageManager/policy

DSV4CheckpointCodec
  may use struct/hashlib/zlib/torch CPU tensor helpers
  must not use Triton/LMCache/StorageManager/policy

DSV4CheckpointStore
  may call DSV4CheckpointCodec
  may lazy-import LMCache runtime types
  must not use GPU layout plan/Triton/policy
```

整个 `codec.py` 顶层禁止 import Triton 和 LMCache；`connector.py` 可以单向 import
codec 和 policy，反向依赖禁止。这样三个 class 在一个文件里，但 CPU-only import、
GPU layout 测试和 StorageManager ownership 仍能分开验证。

不能把 `slot_bytes` 加到 `bytes_per_block`。`BlockGPUConnector` 和 LMCache raw
metadata 用 `bytes_per_block` 为每个 256-token PAGE chunk 固定分配 MemoryObj：

```text
WRONG
  bytes_per_block = page_bytes_per_block + slot_bytes

  every 256-token PAGE MemoryObj
  +--------------------------+-------------------------+
  | useful PAGE bytes        | unused full SLOT bytes  |
  +--------------------------+-------------------------+

RIGHT
  bytes_per_block = page_bytes_per_block
  slot_bytes      = separate typed property
```

统一 codec 内部使用一个通用 region-plan builder，但 public API 必须是 typed 的，
不能只有含义不清的 `copy_plan(id)`：

```text
page_plan([b42, b7], dst_offset=0)

  for block_id in [b42, b7]:
      for region in page_regions:
          append PageCopy(
              src_addr  = region.base + block_id * region.unit,
              dst_offset = running_offset,
              nbytes     = region.unit)


slot_plan(group=3, dst_offset=0)

  for region in slot_regions:
      append PageCopy(
          src_addr  = region.base + region.total - 4 * region.unit,
          dst_offset = running_offset,
          nbytes     = region.unit)
```

`checkpoint_plan()` 只是把两个 typed plan 串起来，不重新解释 C4/C128/state/SWA：

```text
checkpoint_plan(page_block_ids=[b42, b7], slot_group=3)

offset 0
   |
   v
+---------------- PAGE section ----------------+
| b42.page.r0 | b42.page.r1 | ...               |
| b7.page.r0  | b7.page.r1  | ...               |
+-----------------------------------------------+
                                                ^ page_end / slot_start
+---------------- SLOT section ----------------+
| group3.slot.plane0 | optional slot.plane1     |
+-----------------------------------------------+
                                                ^ total_end
```

建议的数据结构：

```text
PayloadSection
  kind       = PAGE | SLOT
  offset     = section byte offset
  nbytes     = section byte length
  copies[]   = PageCopy entries

PayloadPlan
  sections[]
  total_bytes
```

正常 PAGE 和 interval SLOT 都由同一个 codec 处理，但第一版仍使用独立 transaction：

```text
normal PAGE frontier, every 256 tokens

engine.store
   -> BlockGPUConnector
   -> unified_codec.page_plan(block_ids)
   -> unified_codec.gather(plan, PAGE staging)
   -> LMCache PAGE MemoryObj
```

```text
interval SLOT checkpoint

native src -> leased dst
   -> unified_codec.slot_plan(dst_group)
   -> unified_codec.gather(plan, SLOT temp row)
   -> ready event
   -> all TP staging completion
   -> release leased dst
   -> D2H/AOS1/StorageManager
```

load 对称使用同一个实例：

```text
engine.retrieve
   -> unified_codec.page_plan(new_block_ids)
   -> unified_codec.scatter(PAGE staging, plan)

sidecar get/validate
   -> unified_codec.slot_plan(destination_group)
   -> unified_codec.scatter(SLOT temp, plan)

all PAGE + SLOT + TP ranks succeed
   -> scheduler resumes request
```

第一版不强制“一次 composite GPU gather”，原因不是 codec 做不到，而是两个资源的
最早安全时机不同：

```text
SLOT
  native checkpoint copy issued
      -> 必须立即 gather dst -> connector temp
      -> 尽早释放 StateGroup lease

PAGE
  当前由稍后的 LMCache engine.store callback
      -> 根据 chunk ranges/block_ids gather
```

要改成真正一次 gather，需要在 checkpoint-copy hook 里同时提前 gather PAGE，随后把
prepacked PAGE bytes 交给 LMCache callback，或者完全绕过 engine；这会新增 prepacked
handoff、staging 容量、并发、retry、cleanup 和 quarantine 状态，不放进 lease 第一版。

如果以后启用第三层“一个 Storage object”，统一 codec 已能用
`checkpoint_plan()` 生成 composite GPU payload，但 wire format 必须升级成独立 bundle：

```text
checkpoint bundle

+-------------------------------------------------------+
| header                                                |
| boundary/hash/page_offset/page_bytes                  |
| slot_offset/slot_bytes/fingerprint/TP/CRC             |
+-------------------------------------------------------+
| PAGE payload                                          |
+-------------------------------------------------------+
| SLOT payload                                          |
+-------------------------------------------------------+
```

这个 bundle 不能直接复用当前 `engine.store/retrieve` callback：

- callback 收到的 MemoryObj 已按固定 PAGE `bytes_per_block` 分配，不能追加可变 SLOT；
- PAGE callback 没有 `source_group/destination_group`；
- LMCache 已经决定 token key、chunk 数和 token DB publication；
- retrieve callback 无法从一个 bundle 同时 scatter 到新 `block_ids` 和 state group。

因此单对象 bundle 必须使用独立 key，直接调用 `StorageManager.put/get`。而 bundle
究竟包含完整 `PAGE[0:B)` 还是最后一个 PAGE delta 仍是独立 storage policy：完整前缀
可独立恢复但会在每个 interval 重复；delta 不重复但依赖普通 PAGE prefix，并需要处理
token lookup 连续性。只建议将单对象 bundle 用于稀疏/terminal checkpoint，不阻塞
第一版的 unified codec 和 lease 生命周期改造。

##### C3.2.5 256-token PAGE 与 8K SLOT cadence 如何共存

这里必须明确：**第一版合并的是 codec/code path，不是 save cadence 或 Storage
object。** `checkpoint_plan()` 是为以后 bundle 预留的 API，第一版正常路径不会用它
把每个 PAGE chunk 和 SLOT 拼成同一 MemoryObj。

当前粒度：

```text
PAGE chunk size         = 256 tokens
SLOT checkpoint spacing = 8192 tokens

8192 / 256 = 32 PAGE chunks per SLOT checkpoint interval
```

时间轴：

```text
token boundary

   0    256    512    768          7680   7936   8192   8448
   |-----|------|------|---- ... ----|------|------|------|

PAGE     P1     P2     P3           P30    P31    P32    P33
SLOT     .      .      .             .      .    S@8K     .
                                                     ^
                                                     |
                              only here a stable checkpoint dst
                              enters the offload lease/gather path
```

到 `B=8192` 为止，同一个 `DSV4PageSlotCodec` 被调用 32 次 PAGE plan 和 1 次
SLOT plan，但这些是不同 transaction；到 8448 后才是第 33 次 PAGE plan：

```text
every 256 tokens

  engine.store(tokens[B-256:B])
      -> codec.page_plan(block_ids_for_this_chunk)
      -> PAGE gather -> PAGE MemoryObj/key


only at B = 8192

  native checkpoint copy live src -> leased dst
      -> codec.slot_plan(dst_group)
      -> SLOT gather -> SLOT temp/AOS1/key
```

`B=8192` 的 logical checkpoint 由两个条件配对，不要求物理合并 bytes：

```text
PAGE condition
  engine.lookup(tokens[:8192]) >= 8192
  means P1 ... P32 are visible through the standard token DB

SLOT condition
  checkpoint object S@8K is visible
  and its header carries boundary_tokens=8192 + boundary_block_hash

TP condition
  every TP rank reports matching SLOT publication

                         all true
                            |
                            v
              commit logical checkpoint @8K
```

load 同样组合两条路径：

```text
checkpoint @8K
    |
    +--> engine.retrieve PAGE P1...P32
    |       -> codec.page_plan(new_block_ids)
    |       -> scatter PAGE
    |
    +--> checkpoint_store.get S@8K
            -> codec.slot_plan(new_destination_group)
            -> scatter SLOT

all PAGE + SLOT + TP ranks succeed -> resume at 8K
```

不能每 256 tokens 顺便保存 SLOT，原因是 256 boundary 没有一个冻结的 native
checkpoint `dst`：live group 仍在被下一步 forward 修改。若强制每 256 tokens 创建
state copy，会把 free-list checkpoint、D2D copy 和完整 SLOT offload 成本放大 32 倍，
也改变当前 8K state checkpoint policy。

也不能把 PAGE 延迟到 8K 再统一保存，否则会失去：

- LMCache 256-token incremental lookup/dedup；
- PAGE 的渐进 offload 和更早可见性；
- 与现有 token DB/chunk key 的兼容。

若强制“8K 时 PAGE+SLOT 一个物理对象”，只有三种选择，均不是零成本：

```text
1. full-prefix bundle
   Bundle@8K  = PAGE[0:8K)  + SLOT@8K
   Bundle@16K = PAGE[0:16K) + SLOT@16K

   + self-contained, one get
   - every checkpoint duplicates all earlier PAGE bytes


2. interval-delta bundle
   Bundle@8K  = PAGE[0:8K)   + SLOT@8K
   Bundle@16K = PAGE[8K:16K) + SLOT@16K + parent=@8K

   + no repeated PAGE bytes
   - load/GC/eviction require an intact parent chain
   - no longer standard LMCache continuous token lookup


3. duplicate composite bundle
   keep standard P1...P32
   additionally store [P1...P32 + SLOT@8K]

   + preserves standard LMCache lookup
   - PAGE transfer/storage is duplicated
```

因此第一版的解决方案不是强行把两个 cadence 拉平，而是：

```text
one DSV4 codec implementation
two typed plans
two cadences
two storage object families (many PAGE chunks + one SLOT at B)
one logical checkpoint commit identity
```

这里的 AOS1 checkpoint object 可以理解为 PAGE prefix 的 atomic commit record：它
携带 boundary/hash/fingerprint，并只在 PAGE 已经 visible through B 后发布。这样仍然
避免“有 SLOT、缺 PAGE”的 checkpoint 被 scheduler 使用，同时保留 PAGE 的 256-token
粒度和 SLOT 的 8K 粒度。

##### C3.2.6 Boundary checkpoint 的 kernel 与 stream 拓扑

在第一版的“两 cadence、两 transaction”方案里，触发 `B=8K` 时至少有两个 DSV4
Triton gather launch，但不是同一个 stream：

```text
current compute stream

  forward writes PAGE/state
        |
        v
  native copy_state_entries(src -> leased dst)
        |
        v
  SLOT gather kernel
    gather_region_items(REVERSE=True,
                        group=dst,
                        dst=slot_temp)
        |
        v
  record E_slot_ready
```

```text
LMCache BlockGPUConnector thread state

  pack_stream
    PAGE gather kernel
      gather_region_items(REVERSE=False,
                          block_ids=boundary_chunk_blocks,
                          dst=page_staging)
          |
          | page_staging.ready_event
          v
  copy_stream
    PAGE staging -> CPU MemoryObj
```

所以“存 PAGE”和“存 checkpoint”不是各自一个完整 kernel 就结束；准确分层是：

```text
SLOT transaction
  native state D2D copy
  + one SLOT Triton gather launch
  + event
  + later D2H
  + AOS1 encode/put

PAGE transaction
  one PAGE Triton gather launch per bounded transfer group
  + pipeline event handoff
  + D2H to LMCache MemoryObj
  + LMCache batched_put
```

如果一次 `engine.store()` 的 PAGE chunks 超过 bounded staging group capacity，PAGE
pipeline 可能有多个 gather launch；因此不能把“一个 PAGE save”等同于严格一个
kernel launch。对边界最后一个 256-token chunk，在常见单-group情况下是一个 PAGE
gather launch。

第一版选择不同 stream 的原因：

- SLOT gather 必须精确排在 native `src -> dst` 后面，它的 event 是释放 state-group
  lease 的依据；
- PAGE gather 必须留在 `BlockGPUConnector` 的 bounded PAGE staging pipeline 中，
  继续复用其 ready/free event 和 MemoryObj chunk grouping；
- 把 PAGE 强行搬到 compute stream 会绕过/复制 LMCache staging ownership，并阻塞
  后续 compute stream work；
- codec 本身不持有 stream，`gather(plan, dst, stream=...)` 由两个 caller 各自传入。

当前最简单的依赖是 save executor 先确认 `E_slot_ready`，再进入 PAGE
`engine.store()`。因为 `E_slot_ready` 记录在 compute stream 上，它同时覆盖之前的
PAGE/state producer writes 和 native state copy：

```text
compute stream:
  PAGE/state writes -> native state copy -> SLOT gather -> E_slot_ready
                                                        |
save executor:                                           | wait/confirm
                                                        v
page pack_stream:                              PAGE gather -> page ready event
                                                               |
page copy_stream:                                              D2H
```

这里不依赖两个独立 stream 的隐含顺序；跨 stream 可见性来自显式 event/host-confirmed
completion。后续若把 SLOT gather 迁到 connector-owned checkpoint stream，则必须：

```text
compute stream records E_native_done
checkpoint stream waits E_native_done
checkpoint stream gathers SLOT
checkpoint stream records E_slot_ready
```

禁止在另一个 stream 上无 `wait_event` 直接读取 `dst`。

只有未来启用真正的单 bundle/composite transaction 时，PAGE 和 SLOT 才计划在同一个
checkpoint pack stream 上执行。即使那时也建议是两个 specialization launch，而不是
一个带方向分支的大 kernel：

```text
compute stream
  producers/native copy -> E_native_done
                              |
checkpoint_pack_stream        | wait_event
                              v
  gather PAGE section  (REVERSE=False, dst_offset=page_off)
        |
        v
  gather SLOT section  (REVERSE=True,  dst_offset=slot_off)
        |
        v
  record E_bundle_ready
```

它们在同一个 stream 上严格有序，共享一个 staging buffer，并由一个最终 event 表示
“PAGE+SLOT GPU staging 全部完成”。一次 bundle 不要求只有一个 Triton launch；两个
方向 specialization 更容易验证，也避免每个 tile 的 runtime address-mode branch。

#### C3.3 C4、C128 PAGE 到底 gather 了哪些 bytes

先区分四类数据；它们不是一个 tensor，也不都走 PAGE：

```text
compress_ratio=4   = CSA
compress_ratio=128 = HCA

                       checkpoint data paths
                    +--------------------------+
CSA Main compressed |                          |
HCA Main compressed +--> PAGE block regions   +--> LMCache token chunks
CSA Indexer         |                          |
                    +--------------------------+

CSA/HCA compressor tail state                  \
SWA rings for Dense/HCA/CSA layers              +--> full SLOT regions
optional state-carried draft/SWA window         /    --> AOS1 sidecar
```

DSV4 的原始 token block 固定为 256 tokens。因此一个物理 block 在每层产生：

```text
C128 / HCA: 256 / 128 =  2 compressed rows per HCA layer
C4   / CSA: 256 /   4 = 64 compressed rows per CSA layer
```

设 `Nh` 是 HCA layer 数、`Nc` 是 CSA layer 数。共享 row space 的一个 PAGE
block envelope 在每个 KV plane 内都是下列顺序；顺序由
`_ENVELOPE_ORDER = (HCA_RATIO, CSA_RATIO)` 固定：

```text
one physical PAGE block b, one plane p

       C128 / HCA Main: Nh * 2 rows
       <------------------------------------>
+------+-------+-----+------+-------+-----+-------------------------------+
| H0.0 | H0.1  | ... | H1.0 | H1.1  | ... | ... H(Nh-1), rows 0..1 ...   |
+------+-------+-----+------+-------+-----+-------------------------------+
                                                                      |
       C4 / CSA Main: Nc * 64 rows                                    |
       <--------------------------------------------------------------+---->
+------------------------+------------------------+-----+-------------------+
| C0, rows 0..63         | C1, rows 0..63         | ... | C(Nc-1), 0..63    |
+------------------------+------------------------+-----+-------------------+

unit_bytes[p] = (Nh * 2 + Nc * 64) * plane_row_bytes[p]
```

plane 数和 row width 取决于 KV dtype：

```text
BF16 build
  plane0: one complete 512-element KV row
          plane_row_bytes[0] = 512 * 2 = 1024 B

FP8 2-buffer build
  plane0: packed NoPE FP8 row       = 512 B
  plane1: RoPE BF16 row             =  64 * 2 = 128 B

Both planes use the same row index:
  plane0[row I] and plane1[row I] are the two parts of the same logical KV row.
```

CSA Indexer 不在共享 plane 里。每个 CSA layer 单独注册一个 region，每个 block
是 8448 bytes：

```text
one CSA Indexer region for layer Ci, block b

+------------------------------------------+------------------+
| 64 rows * 128 FP8 data bytes = 8192 B    | 64 FP32 scales   |
|                                          | = 64 * 4 = 256 B |
+------------------------------------------+------------------+

unit_bytes = 64 * (128 + 4) = 8448 B
```

这里 data 和 scale 是 block 内的两个连续 sub-regions，不按
`[row data][row scale]` 交错解释。offload codec 仍然只做 8448-byte raw copy，
不解析其中的 FP8 或 FP32 值。FP4 Indexer 当前不会注册这个不完整的 region map；
启用 transfer/offload 时会直接拒绝启动，避免漏存 scale pool。

`get_kv_transfer_tensors()` 最终按下列顺序注册 PAGE regions：

```text
block_regions[]

  BF16: [shared_plane0]
  FP8 : [NoPE_plane0, RoPE_plane1]

  then append:
        [CSA_indexer_layer0,
         CSA_indexer_layer1,
         ...,
         CSA_indexer_layer(Nc-1)]
```

因此，对 LMCache 一个 chunk 映射出的 block ID 顺序 `[b42, b7]`，最终
MemoryObj payload 是严格的 block-major、region-minor：

```text
BF16 example

+-------------------------- block 42 ---------------------------+
| shared plane0                                              |
| [all C128 Main][all C4 Main]                               |
+----------------+----------------+-----+-----------------------+
| C4 idx layer0  | C4 idx layer1  | ... | C4 idx layer(Nc-1)   |
+----------------+----------------+-----+-----------------------+
+--------------------------- block 7 ---------------------------+
| shared plane0                                              |
| [all C128 Main][all C4 Main]                               |
+----------------+----------------+-----+-----------------------+
| C4 idx layer0  | C4 idx layer1  | ... | C4 idx layer(Nc-1)   |
+----------------+----------------+-----+-----------------------+

FP8 example adds RoPE immediately after NoPE inside each block:

[b42.NoPE(C128,C4)] [b42.RoPE(C128,C4)] [b42.C4idx0] ...
[ b7.NoPE(C128,C4)] [ b7.RoPE(C128,C4)] [ b7.C4idx0] ...
```

实际 gather 不是逐元素 Python copy，也没有 transpose。codec 先生成 raw-byte
`PageCopy`：

```text
for block_id in requested_block_ids:          # caller order
    for region in block_regions:              # registration order
        plan.append(
            src_addr  = region.base + block_id * region.unit_bytes,
            dst_offset = running_payload_offset,
            nbytes     = region.unit_bytes,
        )
```

然后 `build_copy_tiles(..., tile_bytes=1024)` 把每个连续 range 切成至多 1024-byte
tile。Triton 每个 program 处理一个 tile：

```text
job j
  source_ptr = plan[j].src_addr + tile_offset
  dest_ptr   = gpu_staging      + plan[j].dst_offset + tile_offset
  load uint8 bytes from source_ptr
  store the same uint8 bytes to dest_ptr

No dtype conversion
No row reordering inside one registered region
No C4/C128 recomputation
```

save 的两条 stream 是：

```text
pack_stream                         copy_stream

PAGE regions                        bounded GPU staging
     |                                      |
     +-- Triton gather_copy_plan ---------->|
                                            +-- copy_ --> CPU MemoryObj
```

一个 transfer group 可以包含多个 LMCache MemoryObj；GPU staging 中依次放每个
MemoryObj 的完整 payload，再按 chunk 的 `nbytes` 切回各自的 CPU MemoryObj。

load 不需要在 payload 中保存物理 block ID。LMCache key 找到 bytes 后，scheduler
已经给新请求分配了新的 `block_ids`；codec 对这些新 ID 重建同样的 plan：

```text
CPU MemoryObj -> GPU staging
                        |
                        +-- payload offset 0 --> new_b0.region0 address
                        +-- next bytes       --> new_b0.region1 address
                        +-- ...
                        +-- next block       --> new_b1.region0 address

scatter destination = region.base + new_block_id * region.unit_bytes
```

所以 C4/C128 的恢复是 byte-identical relocation：逻辑内容和 region 内部顺序不变，
只把旧 block 的 bytes 写进新请求 block table 指定的新物理 block。

### C4. PAGE load：LMCache MemoryObj -> GPU staging -> PAGE regions

load metadata 已经包含新请求刚分配好的 `block_ids`。worker 只加载
`[hbm_cached_tokens, lmcache_cached_tokens)`，并调用：

```python
ret_mask = self._engine.retrieve(
    torch.tensor(token_ids[:lmc]),
    mask=mask,                 # HBM已有 prefix 为 False，需要加载部分为 True
    block_ids=req.block_ids,   # 这次请求的新物理 PAGE destinations
    req_id=str(req.req_id),
)
```

```text
LMCache storage tier
        |
        v
MemoryObj(s) selected by token chunk keys
        |
        v
BlockGPUConnector.batched_to_gpu(memory_objs, starts, ends, block_ids=...)
        |
        +--> _range_block_ids(): range -> new request's destination block IDs
        |
        +--> copy_stream: MemoryObj tensors -> bounded GPU staging
        |
        +--> pack_stream:
                DSV4PageSlotCodec.page_plan + scatter
                Triton PAGE scatter(GPU staging -> PAGE region addresses)
        |
        v
ret_mask[hbm:lmc].all()
        |
        +--> true: PAGE local load succeeded
        +--> false: composite load fails and scheduler recomputes
```

scatter 使用与 save 完全相同的 copy plan，只是方向反过来：payload 中的
`b42.r1` bytes 写到 `region1.base + block42 * unit_bytes`。这样 save/load 不对
FP8/BF16 值做 reinterpret，要求的是 byte-identical round trip。

### C5. SLOT save：state group -> GPU temp -> AOS1 -> StorageManager

SLOT 不使用 `LMCacheEngine.store()` 的 token chunk codec，因为它是 boundary 上的
完整 request state，不是 token-indexed PAGE。它直接使用相同 engine 内部的
`StorageManager`。

SLOT source group 有两种来源：

```text
regular interval keeper:
    StateCheckpointCopy.destination_group  -> source group for SLOT gather
    group held by checkpoint lease

interval terminal without successor batch:
    SlotSaveSpec.source_group               -> stable live group fallback
```

regular keeper 的 piggyback 实际跨两个 EngineCore 调用：

```text
RPC / step A: dispatch connector metadata

  EngineCore
      -> process_kvconnector_output(meta, checkpoint records)
      -> connector records deferred save intent
      -> no SLOT bytes are read yet

RPC / step B: execute forward batch

  current compute stream
      -> attention builder executes copy_state_entries(src -> dst)
      -> builder returns
      -> ModelRunner calls state_checkpoint_copies_issued(records)
      -> connector executes snapshot_to_staging(dst -> GPU temp)
      -> connector records ready_event E

  save executor
      -> waits E
      -> takes over PAGE/SLOT publication work
```

这里 connector 拿到 `dst` 的方式不是自行查询 free list，而是 scheduler 把
`StateCheckpointCopy(copy_id, request, B, hash, src, dst)` 随 batch 传到 worker。
connector 只在 exact identity 匹配后使用 record 中的 `destination_group`。

完整 SLOT gather 是按 plane-region 顺序拼接的一个 group snapshot。当前 DSV4
BF16 build 有一个 full-slot region，FP8 2-buffer build 有两个，不是每个 state
field 或每层 SWA 各注册一个 region：

```text
reverse-indexed SLOT regions for group g

region0 / plane0 addr = base0 + total0 - (g + 1) * unit0
optional region1 / plane1 addr = base1 + total1 - (g + 1) * unit1

                    DSV4PageSlotCodec.slot_plan + gather
                    Triton SLOT gather
                              |
                              v
connector GPU temp row

+-------------------------------+-------------------------------+
| group g, plane0 full SLOT     | optional plane1 full SLOT     |
+-------------------------------+-------------------------------+
```

#### C5.1 SLOT 内部的 C4/C128 state 与 SWA layout

当前 DSV4 sidecar 的 `swa_block_regions` 是历史命名，实际含义是
**full SLOT regions**。它不是“只保存 SWA”，而是把同一个 request group 的：

```text
1. CSA/C4 Main compressor kv_state + score_state
2. CSA/C4 Indexer compressor kv_state + score_state
3. HCA/C128 Main compressor kv_state + score_state
4. every layer's SWA ring
5. optional state-carried draft/SWA window
6. alignment padding
```

一起原样保存。正常 SWA 与 compressor state 已经物理合并在共享 plane 的同一个
slot 中，因此这里不需要分别收集六个 state tensor 和每层 SWA tensor。

一个 logical group `g` 先映射成 reverse-indexed physical slot。对每个 plane：

```text
physical_slot(g) = slot_positions - 1 - g

region.base  = lowest live SLOT address in this plane
region.total = num_slots * slot_bytes_in_this_plane
region.unit  = slot_bytes_in_this_plane

slot_address(g)
    = region.base + region.total - (g + 1) * region.unit
```

所以 group 0 位于最高地址，group 越大越靠近 PAGE/SLOT 之间的 gap：

```text
low address                                                        high address

PAGE blocks ---> | gap | SLOT group N-1 | ... | SLOT group 2 | group 1 | group 0
                         ^
                         region.base

gather group 1:
  src = region.base + total - 2 * unit
```

每个 plane 内一个 SLOT 是连续的，先放 compressor state arena，再放各类 SWA
window rows：

```text
one SLOT in plane p

+-----------------------------------------------------------------------+
| state arena assigned to plane p                                      |
|                                                                       |
| a subset of, in declaration order:                                   |
|   C4 main kv_state                                                    |
|   C4 main score_state                                                 |
|   C4 indexer kv_state                                                 |
|   C4 indexer score_state                                              |
|   C128 main kv_state                                                  |
|   C128 main score_state                                               |
|   optional state_window                                               |
|                                                                       |
| each field starts on its required alignment; padding is copied too    |
+-----------------------------------------------------------------------+
| Dense-class SWA rows                                                  |
+-----------------------------------------------------------------------+
| C128 / HCA-class SWA rows                                             |
+-----------------------------------------------------------------------+
| C4 / CSA-class SWA rows                                               |
+-----------------------------------------------------------------------+
| slot alignment padding                                                |
+-----------------------------------------------------------------------+

slot_bytes[p] = slot_rows * plane_row_bytes[p]
```

FP8 2-buffer 时，state fields 由 `plan_field_planes()` 分配到 NoPE/packed plane
或 RoPE plane。算法最小化“两个 plane 中较大的所需 row 数”，然后两个 plane 物理上
统一预留这个 `arena_rows`；它不保证两个 plane 中实际 state bytes 相等。一个 field
不会横跨两个 plane。BF16 时只有一个 plane，所有 state fields 都在该 plane 的 slot
前部。sidecar 不需要理解某个 field 被分到哪个 plane，因为它把每个 plane 的整个
slot 都复制走。

compressor state 每层的逻辑 shape 是：

```text
ring_extra = max_spec_steps

C4 Main kv_state    : [8 + ring_extra, 1024] fp32 per CSA layer
C4 Main score_state : [8 + ring_extra, 1024] fp32 per CSA layer

C4 Index kv_state    : [8 + ring_extra, 256] fp32 per CSA layer
C4 Index score_state : [8 + ring_extra, 256] fp32 per CSA layer

C128 Main kv_state    : [128 + ring_extra, 512] fp32 per HCA layer
C128 Main score_state : [128 + ring_extra, 512] fp32 per HCA layer
```

这些 shape 在 state arena 内按 field、再按 layer 排列，并带 256-byte alignment。
但 sidecar gather 的粒度仍然是“整个 plane slot”，不会为每个 shape 单独发一次
StorageManager put。

SWA 的逻辑 ring 长度为：

```text
Wspec = sliding_window + max_spec_steps
q     = absolute_token_position % Wspec
```

slot 中 SWA class 顺序由 `_ENTRY_ORDER = (Dense, HCA, CSA)` 固定。为了让同一个
index 公式同时服务该 class 的所有 layer，HCA/CSA 的 window 不是简单
`[layer0 whole ring][layer1 whole ring]`；它按该 class 的 block-row stride 分 run：

```text
physical row in plane p
  = slot_start_row(g)
  + arena_rows
  + entry_offset[class]
  + row_within_class(layer, q)

Dense class: ring_stride = Wspec
  row_within_class(layer, q) = layer * Wspec + q

HCA / C128 class: ring_stride = 2
  row_within_class(layer, q) = layer * 2
                              + (q // 2) * (Nh * 2)
                              + (q % 2)

CSA / C4 class: ring_stride = 64
  row_within_class(layer, q) = layer * 64
                              + (q // 64) * (Nc * 64)
                              + (q % 64)
```

当 `Wspec = 128` 时可以直观看成：

```text
Dense SWA
  [D0 q0..127] [D1 q0..127] ...

C128 / HCA SWA
  run0: [H0 q0..1] [H1 q0..1] ... [H(Nh-1) q0..1]
  run1: [H0 q2..3] [H1 q2..3] ... [H(Nh-1) q2..3]
  ...
  run63:[H0 q126..127] ...

C4 / CSA SWA
  run0: [C0 q0..63]   [C1 q0..63]   ... [C(Nc-1) q0..63]
  run1: [C0 q64..127] [C1 q64..127] ... [C(Nc-1) q64..127]
```

注意这只是 SLOT plane 内部的现有 GPU layout。sidecar gather 不重新排列上述
state/SWA rows；它把 `slot_address(g)` 开始的整个 `slot_bytes[p]` 当作 opaque
uint8 range。

最终 GPU temp 和 AOS1 wire layout 是：

```text
BF16 build, one full-slot region

GPU temp payload
+---------------------------------------------------------------+
| plane0 SLOT g                                                 |
| [state arena][Dense SWA][C128 SWA][C4 SWA][padding]           |
+---------------------------------------------------------------+

FP8 2-buffer build, two full-slot regions

GPU temp payload
+--------------------------------+-------------------------------+
| NoPE/packed plane SLOT g       | RoPE plane SLOT g             |
| [state subset][all SWA rows]   | [state subset][all SWA rows]  |
+--------------------------------+-------------------------------+

Pinned CPU AOS1 frame
+------------------+---------------------------------------------+
| 128-byte header  | exact GPU temp payload bytes                |
| magic/version/B  | plane0 full SLOT | optional plane1 full SLOT|
| hash/size/CRC/TP |                                             |
+------------------+---------------------------------------------+
```

AOS1 v1 header 不保存 plane offset/size table。codec 按 region 注册顺序和
`unit_bytes` 前缀和隐式重建 payload offset：

```text
payload_offset(plane p) = sum(unit_bytes[k] for k < p)

BF16: p0 offset = 0
FP8 : p0 / NoPE offset = 0
      p1 / RoPE offset = unit_bytes[p0]
```

header 中的 fingerprint 和 `expected_payload_bytes` 用于阻止 incompatible layout
被错误恢复；它不是一个能让任意新 codec 自描述旧 payload 的 schema。

当前 `ATOMSlotSidecarCodec._copy_plan(g)`、计划中的
`DSV4PageSlotCodec.slot_plan(g)` 都只生成“一 plane 一 range”的 plan：

```text
dst_offset = 0
for region in full_slot_regions:             # plane order
    src_addr = region.base + region.total - (g + 1) * region.unit
    nbytes   = region.unit
    plan.append(src_addr, dst_offset, nbytes)
    dst_offset += nbytes
```

它同样被切成 1024-byte Triton tiles，直接 gather 到 connector-owned GPU temp row。
load 时以新请求的 `destination_group` 重建相同 plan，然后逐 plane raw-byte scatter：

```text
SAVE
  old group g, full plane slots
      -- gather_copy_plan --> GPU temp row
      -- D2H -------------> AOS1 payload
      -- one put ----------> LMCache StorageManager

LOAD
  LMCache StorageManager
      -- one get ----------> AOS1 payload
      -- H2D -------------> GPU temp row
      -- scatter_copy_plan -> new destination group g'
```

这里还有一个容易混淆的旧接口：`staging_region + gather_slot/scatter_slot` 是
PD disaggregation 的 compressor-only staging，它只把 split state arena 的各 plane
entry 拼进 `v4_state_pool`，**不包含普通 SWA rows**。SLOT sidecar offload 不能使用
它；offload 使用的是 `swa_block_regions` 中的 full-slot raw ranges：

```text
PD compressor staging
  [C4/C128 compressor state only]              no normal SWA

checkpoint SLOT sidecar
  [compressor state + every normal SWA row]    complete request SLOT
```

worker save 链路：

```text
start_load_kv_with_state_checkpoints(metadata, checkpoint_copies)
        |
        +--> match (request, boundary, hash, source_group)
        +--> CURRENT: record a lightweight deferred intent
        +--> PLAN: pre-reserve save credit + GPU temp staging_id
        +--> wait until native src -> dst copy is issued
        |
        v
state_checkpoint_copies_issued()
        |
        +--> gather_slot(dst, staging_row, current_compute_stream)
        +--> ready_event.record(current_compute_stream)
        |
        v
save executor waits ready_event
        |
        +--> report ConnectorCompletion(
        |        atom.state_checkpoint.staging, copy_id, succeeded=True)
        |            all TP ranks -> release StateGroupPool lease
        |
        +--> engine.store(PAGE chunks) if PAGE is due
        |
        +--> D2H: GPU temp row -> pinned CPU
        |                   [128-byte header | payload]
        +--> release GPU temp row immediately after D2H
        |
        +--> finalize_checkpoint_tensor_()
        |       magic/version/boundary/hash/size/fingerprint/TP/CRC32
        |
        +--> wait engine.lookup(tokens[:B], pin=False) >= B
        |       PAGE coverage must be visible before SLOT publication
        |
        +--> DSV4CheckpointStore.put(key, CPU frame)
        +--> poll DSV4CheckpointStore.contains(key)
        |
        v
worker reports ConnectorCompletion(
    atom.dsv4.checkpoint.save, save_operation, succeeded=True)
        |
        v
all TP ranks succeed -> scheduler commits boundary B
```

当前草稿和本计划预期的顺序差别如下：

```text
CURRENT
  event -> engine.store(PAGE) -> wait PAGE visible -> D2H SLOT
        -> finalize -> put SLOT -> wait SLOT visible -> release temp in finally

PLAN
  event -> report checkpoint staged -> engine.store(PAGE) -> D2H SLOT
        -> release temp -> finalize -> wait PAGE visible -> put SLOT
        -> wait SLOT visible
```

计划把 D2H 放在 PAGE store submission 之后、PAGE visibility polling 之前；这样
PAGE 后端 publication 与 SLOT D2H 可以重叠，而 temp row 不会被慢速
NVMe/visibility poll 长时间占用。若预留必须发生在 metadata dispatch，当前
`_DeferredCheckpointSave` 还需要增加 save-credit/staging ownership 字段，不能只
记录 request 与 checkpoint identity。

### C6. SLOT 如何调用 LMCache StorageManager

SLOT key 先由 boundary identity 构造：

```text
DSV4CheckpointKey
|
+-- boundary_block_hash
+-- layout fingerprint
+-- tp_size
+-- tp_rank
|
+--> canonical string
     atom-slot-v1:<tp_size>:<tp_rank>:<boundary_hash>:<fingerprint>
|
+--> BLAKE2b-8 -> nonnegative 63-bit storage_hash
```

`DSV4CheckpointStore` 再把它包装成 LMCache `CacheEngineKey`：

```text
CacheEngineKey(
    model_name = PAGE engine model namespace,
    world_size = tp_size,
    worker_id  = tp_rank,
    chunk_hash = DSV4CheckpointKey.storage_hash(),
    dtype      = uint8,
)
```

实际 put：

```text
CPU AOS1 frame: uint8[N]
        |
        v
storage_manager.allocate(shape=(1, 1, N), dtype=uint8, fmt=KV_2LTD)
        |
        v
copy frame -> MemoryObj.tensor
        |
        v
storage_manager.batched_put([CacheEngineKey], [MemoryObj], location=store_location)
        |
        +--> normal return: ownership transferred to StorageManager
        |                   connector must NOT ref_count_down again
        |
        +--> exception: connector ref_count_down(memory_obj), report failure
```

`batched_put()` 正常返回只表示提交被接受，不表示 NVMe 或远端 tier 已 durable
flush。因此 connector 还用 `storage_manager.contains(..., pin=False)` 做本 session
的 publication probe。

### C7. SLOT load 与 scatter

composite load 固定先 PAGE、后 SLOT；PAGE 成功但 SLOT 失败时整个 boundary 都不
发布给 scheduler。

```text
_do_load_req(req)
        |
        +--> _load_page(req)
        |       engine.retrieve -> PAGE scatter
        |       miss/failure ------------------------------> recompute
        |
        v
_load_slot(req, staging reservation)
        |
        +--> DSV4CheckpointStore.borrow(DSV4CheckpointKey)
        |       contains/locate -> storage_manager.get()
        |       keep MemoryObj borrowed through H2D+scatter
        |
        +--> decode_checkpoint_tensor()
        |       validate magic/version/reserved bytes
        |       validate B/hash/fingerprint/TP/payload size/CRC32
        |
        +--> CPU payload -> GPU temp row                  (H2D stream)
        |
        +--> DSV4PageSlotCodec.slot_plan(destination_group)
        |         + scatter(temp, plan)
        |       Triton SLOT scatter(temp -> all SLOT regions)
        |
        +--> stream.synchronize()
        +--> MemoryObj.ref_count_down()
        +--> release staging row
        |
        v
all TP ranks PAGE + SLOT succeeded
        |
        v
scheduler publishes loaded PAGE prefix and resumes suffix prefill
```

SLOT scatter 的目标不是保存时的 group ID。scheduler 为新请求从
`StateGroupPool` 分配 `seq.per_req_cache_group`，把它作为
`SlotLoadSpec.destination_group` 传给 worker；codec 用该 group 重新计算每个
reverse-indexed region 的目标地址。

### C8. PAGE 与 SLOT 两条 LMCache 路径的区别

```text
                         PAGE                              SLOT
                +-------------------+             +----------------------+
index unit      | token chunk       |             | checkpoint boundary  |
GPU identity    | block_ids[]       |             | state group ID       |
codec           | same unified      |             | same unified         |
                | codec.page_plan   |             | codec.slot_plan      |
LMCache API     | engine.store /    |             | storage_manager      |
                | engine.retrieve   |             | allocate/put/get     |
key owner       | ChunkedTokenDB    |             | DSV4CheckpointKey       |
payload object  | one MemoryObj per |             | one AOS1 MemoryObj   |
                | token chunk       |             | per boundary/rank    |
                +-------------------+             +----------------------+
                         |                                  |
                         +------------- require both ------+
                                        at boundary B
```

这种拆分让 PAGE 保留标准 LMCache 256-token lookup/dedup，同时允许 SLOT 只在较
稀疏的 interval 保存完整 state。

#### C8.1 PAGE hit 与 SLOT hit 的交集由谁计算

LMCache 原生 lookup 只认识 token database 中的 PAGE chunks：

```text
LMCache LookupClient.lookup(token_ids)
        |
        v
page_hit_tokens = longest continuous PAGE token-chunk prefix
```

它不认识 ATOM 的：

- `DSV4CheckpointKey`；
- 8K SLOT boundary；
- state-group snapshot；
- PAGE/SLOT 必须来自同一 boundary hash；
- 每个 TP rank 都必须有对应 checkpoint object。

所以 LMCache 没有原生的“两个 KV 类型自动取交集”。当前分支由 ATOM scheduler-side
connector 做 composite hit gate。

交集不是简单的 `min(page_hit_tokens, slot_hit_tokens)`，因为 SLOT hit 是离散 boundary
集合，不是连续 token count。定义：

```text
P = LMCache PAGE continuous-prefix hit tokens

S = {
      B |
      B is a valid resume boundary,
      checkpoint hash(B) is committed,
      every TP rank published SLOT@B
    }

effective_hit = max({B in S | B <= P}, default=0)
```

当前 scheduler 的实际适配流程：

```text
1. page_hit = lookup_client.lookup(prompt_tokens)

2. if page_hit == full_prompt:
       page_hit -= 1
   # 保留至少一个 token 由正常执行路径处理

3. candidate = floor(page_hit / resume_alignment) * resume_alignment

4. compute the same chained prefix hash used by BlockManager:
       boundary_hash = chained_hashes[candidate]

5. walk candidate downward until:
       boundary_hash in _committed_sidecar_hashes

6. effective_hit = matched candidate
   pending SLOT load = (candidate, boundary_hash)

7. no matching committed SLOT:
       reject the stateful remote hit and recompute
```

例子：

```text
PAGE continuous hit     committed SLOT boundaries      effective stateful hit

16K                     {8K}                           8K
20K                     {8K, 16K}                      16K
12K                     {8K, 16K}                      8K
20K                     {}                             0 / recompute
20K                     {8K, 16K with wrong hash}      8K or 0
```

`_committed_sidecar_hashes` 不是 LMCache 原生索引，而是 ATOM 的 bounded
scheduler-session LRU set。它只在同一 save generation 的所有 TP workers 都报告
SLOT publication 成功后加入：

```text
TP0 SLOT@B success ----+
TP1 SLOT@B success ----+
...                    +--> KVOutputAggregator all-rank quorum
TPN SLOT@B success ----+             |
                                     v
                    scheduler adds hash(B) to committed set
```

任何一个 rank 失败都不会 commit 该 boundary，因此查询侧不需要再对 session 内的
每个 rank 单独取交集；all-TP aggregator 已经把它压成一个全局 committed bit。

当前 LMCache PAGE lookup server 只查询 rank 0（`lookup_server_worker_ids = [0]`）。
安全性依赖保存阶段各 TP rank 锁步 publication、all-TP SLOT commit，以及 load 阶段
每个 rank 再实际 retrieve/validate：

```text
fast admission lookup
  rank0 PAGE token lookup
  + scheduler all-TP committed SLOT bit
        |
        v
candidate effective boundary

authoritative load
  every TP rank retrieves its PAGE shard
  + every TP rank gets/validates its SLOT object
        |
        v
all-rank success or whole load fails
```

因此 rank0 lookup 可以产生随后 load 才发现的 stale/partial candidate，但不会让缺少
任一 TP shard 的 checkpoint 被最终发布。

query gate 后，worker load 仍做第二层 fail-closed 验证：

```text
load effective boundary B
        |
        +--> PAGE engine.retrieve/scatter through B
        |       failure ------------------------------> whole load fails
        |
        +--> checkpoint_store.borrow(SLOT@B, tp_rank)
                validate B/hash/fingerprint/TP/size/CRC
                H2D + SLOT scatter
                failure ------------------------------> whole load fails

all PAGE + SLOT + TP ranks succeed -> publish prefix B
```

因此 scheduler-side intersection 只是 admission optimization，不能替代 worker 的
真实 get/CRC/scatter 校验。对象可能在 lookup 后被 evict 或损坏，load 必须再次验证。

HBM prefix 还有一个额外 gate。当前 stateful v1 不能把“HBM 中某个较短 boundary 的
request state”和“远端较长 boundary 的完整 SLOT”合并：

```text
stateful request:
  HBM cached tokens == 0
      -> allowed to load PAGE+SLOT at effective_hit

  HBM cached tokens > 0
      -> v1 rejects remote composite load and recomputes
      -> avoids PAGE state from one boundary + SLOT state from another boundary
```

这不是 LMCache 限制，而是 ATOM 当前完整 request-level SLOT protocol 的一致性限制。

当前适配的已知持久化限制：`_committed_sidecar_hashes` 是 session-local。scheduler
重启以后：

```text
StorageManager may still contain:
  PAGE chunks + AOS1 SLOT objects

new scheduler knows from LMCache lookup:
  PAGE hit P

new scheduler does not automatically know:
  which deterministic SLOT keys exist on every TP rank

result:
  stateful hit fails closed to 0/recompute
```

如果第一版要求跨 scheduler restart 复用 SLOT，需要再加 ATOM lookup adapter，而不是
期待 LMCache token lookup 自动处理。推荐算法：

```text
page_hit P from native LMCache lookup
        |
        v
enumerate valid DSV4 boundaries B <= P, newest first
        |
        v
derive deterministic DSV4CheckpointKey(B, hash(B), fingerprint, tp_rank)
        |
        v
batched checkpoint_store.contains() on every TP rank
        |
        v
first B present on all ranks = effective_hit
```

这个查询应通过 worker-side batched RPC 或持久 checkpoint manifest/index 完成，并带
session positive/negative cache，避免每个 request 对每个 8K boundary 发大量远端
`contains`。即使加入该 adapter，最终 load 仍必须做 PAGE retrieve + SLOT get/CRC 的
二次验证。

### C9. Completion、ownership 与当前已知边界

PAGE 与 SLOT 的“函数返回”都不能直接等同于 checkpoint 已经 durable：

```text
PAGE engine.store returns
    = PAGE GPU gather/copy 已结束
    + LMCache batched_put 已提交
    != 所有异步 storage tier 已 durable flush

SLOT DSV4CheckpointStore.put returns True
    = StorageManager 接受/提交 MemoryObj
    != 当前 retrieve policy 已经能看到对象

SLOT contains returns location
    = 本 session 按 retrieve_locations 可见
    != 所有 backend 都完成 durable flush
```

因此 composite save 的逻辑 commit 仍然是：

```text
PAGE lookup visible through B
        -> SLOT put accepted
        -> SLOT contains visible
        -> every TP rank reports sidecar success
        -> scheduler adds boundary hash to session commit set
```

当前 `_committed_sidecar_hashes` 是 scheduler-session-local；进程重启后不会自动
枚举 StorageManager 中已有 AOS1 对象。持久对象仍在，但第一版不会把它们自动
恢复为已提交 boundary。

MemoryObj ownership 也要区分：

- PAGE MemoryObj 完全由 `LMCacheEngine.store/retrieve` 分配、pin、put/get 和释放；
  `BlockGPUConnector` 只在 callback 生命周期里借用其中的 flat `uint8` tensor。
- SLOT adapter 自己直接调用 StorageManager，所以 `put` 异常必须自己
  `ref_count_down`；`batched_put` 正常返回后由 StorageManager 接管，不能再减一次。
- SLOT load 使用 `borrow()`，MemoryObj 必须一直持有到 H2D 和 scatter stream
  synchronize 完成后才能 `ref_count_down`。

需要在实现/测试时额外审计一个第三方边界：当前 vendor LMCache 的 PAGE
`store/retrieve` 如果自定义 GPU callback 中途抛异常，部分已 allocate/get 的
MemoryObj 是否都能在异常路径 unpin/ref-count-down。ATOM 能 quarantine 自己的
GPU staging，但不能释放不属于自己的 LMCache MemoryObj；必要时应补 wrapper、
上游修复或明确的失败回归测试。

## 4. Boundary 策略

### 4.1 Regular boundary

沿用当前 DSV4 profile 的对齐规则：

```text
hash_block_size  = kv_cache_block_size * dcp_size
resume_alignment = lcm(LMCACHE_CHUNK_SIZE, hash_block_size)
snapped_interval = floor(state_checkpoint_interval_tokens / hash_block_size)
                   * hash_block_size
slot_interval    = lcm(snapped_interval, resume_alignment)
```

只有 `B % slot_interval == 0` 的有效 boundary 才产生 regular SLOT offload。

- PAGE 仍可在每个 LMCache chunk frontier 保存。
- native demand checkpoint 如果不在 SLOT interval 上，不触发 SLOT offload。
- decode 中因 spacing 形成但不在 regular interval 上的 checkpoint，不触发本
  SLOT offload 路径。

例如 `PAGE chunk = 256`、`SLOT interval = 8192` 时：

```text
token boundary
    0    256    512    768   ...   7936   8192   8448   ...  16128  16384
    |-----|------|------|----- ... ---|------|------|---- ... ---|------|

PAGE     P256   P256   P256          P256   P256   P256        P256   P256
SLOT      .      .      .             .    SLOT@8K  .           .   SLOT@16K
                                               ^                       ^
                                               |                       |
                                      only these boundaries acquire a lease

Legend:
  P256 = 新增的 256-token PAGE chunk
  SLOT@B = boundary B 的完整 SLOT snapshot，不是增量 state
```

### 4.2 Terminal boundary

推荐默认语义：

- terminal 恰好位于有效 SLOT interval：允许保存。
- terminal 不在 SLOT interval：不额外生成 SLOT checkpoint。
- terminal 位于 interval、但已经没有后继 batch 可承载 deferred native copy：
  允许从仍被 terminal-save 生命周期保护的 live group 直接 gather。

如果 review 决定 terminal 必须一律经过 free-list `dst`，则需要增加一个只执行
`copy_state_entries + gather` 的 maintenance RPC；不建议为此构造空模型 forward。

## 5. 核心状态机

### 5.1 正常链路

```text
interval boundary B reached
        |
        v
checkpoint intent (request, B, boundary_hash, src)
        |
        v
StateGroupPool.pop() -> dst
create lease(copy_id -> dst); dst stays off free list
        |
        v
compute stream: copy_state_entries(src -> dst)
        |
        v
same compute stream: gather_slot(dst -> GPU temp row)
        |
        v
record completion event E
        |
        v
worker confirms E completed
        |
        v
all TP ranks terminal for copy_id
        |
        v
release lease; return dst to checkpointed free list
        |
        +---- GPU temp -> CPU -> PAGE visible -> AOS1 put continues async
```

`dst` 返回 free list 时保留 `boundary_hash`。它仍然是可复用的 native HBM
checkpoint；只有后续正常 allocation/LRU eviction 才会 invalidate 它。

### 5.2 `dst` 在 free list 中的状态变化

```text
                        pop() + acquire lease(copy_id)
  +------------------+ --------------------------------------+
  | FREE             |                                       v
  | vacant or old    |                           +-------------------------+
  | checkpoint       |                           | LEASED_WAIT_COPY        |
  +--------+---------+                           | dst not allocatable     |
           ^                                     +-----------+-------------+
           |                                                 |
           |                                      src -> dst copy valid
           |                                                 v
           |                                     +-------------------------+
           |                                     | LEASED_CHECKPOINT_VALID |
           |                                     | hash=B, gather may read |
           |                                     +-----+--------------+----+
           |                                           |              |
           |                         all TP gather done |              | GPU completion
           |                                           |              | cannot be proven
           |                                           v              v
           |                               +-------------------+  +-------------+
           +-------------------------------| CHECKPOINTED_FREE |  | QUARANTINED |
                 release, preserve hash    | reusable / LRU    |  | never reuse |
                                           +-------------------+  +-------------+

  Native copy abort before valid bytes:

       LEASED_WAIT_COPY --invalidate hash + release--> VACANT_FREE
```

这里的 `FREE` 表示“可被 pool 分配”，不表示内容一定为空。正常成功路径最终进入
`CHECKPOINTED_FREE`，所以 native HBM checkpoint 仍然可被后续请求命中。

## 6. Scheduler 侧选择和加锁

### 6.1 精确匹配

不是所有 `StateCheckpointCopy` 都建立 offload lease。只锁定本次 connector
metadata 中存在匹配 `SlotSaveSpec` 的记录：

```text
(request_id,
 boundary_tokens,
 boundary_block_hash,
 source_group)
```

匹配后使用 `StateCheckpointCopy.copy_id` 作为 lease ID，并验证：

- `destination_group` 当前仍属于相同 `boundary_block_hash`；
- group 尚未被其他 lease 重用；
- 同一个 `copy_id` 不可指向另一个 group；
- resume copy 没有 keeper identity，不能进入该路径。

### 6.2 StateGroupPool lease

计划新增概念性状态：

```text
_offload_leases: dict[copy_id, (destination_group, boundary_hash)]
_offload_pins_by_group: dict[destination_group, set[copy_id]]
```

external/offload pin 必须参与以下判断：

- `is_pinned(group)`；
- resume 时是否允许 adopt/write 该 checkpoint；
- `retire_top()` 是否可回收该 group；
- scheduled reader pin 释放时，是否真的可以把 group 放回 free list；
- checkpoint hash 被 KV index orphan/evict 时，只 invalidate hash，不能在 lease
  完成前重新分配该 group。

lease completion 必须幂等。迟到或重复的 `copy_id` completion 不能释放已经重用
的 group。

## 7. Worker 侧 capture 顺序

### 7.1 Metadata dispatch

`start_load_kv_with_state_checkpoints()`：

1. 用精确 identity 找到对应 deferred SLOT save。
2. 尽早预留 save admission credit。
3. 尽早预留 connector GPU temp row。
4. 保存轻量 intent；此时不读取 `dst`。

如果 admission 或 GPU temp row 不可用，sidecar save 标记失败，但 native
`src -> dst` copy 仍可能已经属于本 batch；因此 lease 不能在 copy 完成前直接
释放。

### 7.2 Native copy 与 gather

`ModelRunner` 中的顺序必须是：

```text
copy_state_entries(src -> dst)
gather_slot(dst -> GPU temp)
event.record(current_compute_stream)
```

禁止在不同 stream 上无依赖地执行 gather。若未来改为专用 copy stream，必须先
显式 `wait_event(native_copy_done)`。

### 7.3 Staging completion

后台 finalizer 等待 event：

- event 成功：worker 报告该 `copy_id` 已经不再读取 `dst`；随后 D2H 和存储
  独立继续。
- event 失败但能通过 stream synchronize 确认所有访问已经结束：允许报告 lease
  terminal，但 sidecar save 失败。
- 无法确认 GPU completion：不得把 group 放回 free list；该 lease/group 进入
  quarantine，通常同时将 worker/engine 视为不健康。

## 8. Completion 协议

当前通过通用 `ConnectorCompletion` 表达独立于 checkpoint publication 的完成语义：

```text
channel      = atom.state_checkpoint.staging
operation_id = copy_id
succeeded    = True   # copy valid，GPU 已不再读 dst
succeeded    = False  # native copy 无效，需 invalidate 后释放
```

说明：

- gather/admission 失败但 `src -> dst` 已成功时，仍报告 staging success；native
  checkpoint 有效，只是 remote checkpoint save 失败。
- native copy 在发出前 abort，或无法建立有效 `dst` 内容时，属于
  staging failure。
- 无法确认仍在飞的 GPU 工作时，不属于任何可释放集合。

`KVOutputAggregator` 按 `copy_id` 等待所有 TP workers terminal：

```text
all ranks finished -> preserve hash and release lease
any rank aborted, after all ranks terminal -> invalidate hash and release lease
rank still non-terminal -> keep lease
```

多 TP 情况下，scheduler 只管理一个逻辑 group index，因此必须等最慢的 rank：

```text
                  copy_id = 42, destination_group = 7

  TP rank 0:  dst[7] -> temp[0] -> event done -----------+
  TP rank 1:  dst[7] -> temp[0] -> event done -----------+---> TP aggregator
  TP rank 2:  dst[7] -> temp[0] -> event pending --------+          |
  TP rank 3:  dst[7] -> temp[0] -> event done -----------+          |
                                                                    | rank 2未完成
                                                                    v
                                                             KEEP LEASED

  later:

  TP rank 2:  event done ------------------------------------------> all terminal
                                                                    |
                                    +-------------------------------+----------------+
                                    |                                                |
                              all finished                                    any aborted
                                    |                                                |
                                    v                                                v
                         preserve hash + release                         invalidate + release
```

某个 rank 的 event 完成不能单独释放 group；否则其他 rank 仍可能从同一个逻辑
group index 读数据，而 scheduler 已把它交给了新请求。

`MultiConnector` 只需要 union offload child 的 staging completion；send/save pairing
不应延迟 checkpoint lease 的释放。

## 9. 三层资源释放点

下面用一条时间轴表示三个资源为什么不能在同一时刻释放：

```text
time -------------------------------------------------------------------------->

             T0             T1               T2              T3       T4      T5
             pop dst        src->dst done    dst->temp done  D2H done put     visible
             |              |                |               |        |       |

checkpoint  [================ LEASED ========================)
group dst                                             release at all-TP T2

GPU temp          [================ RESERVED =======================)
row                                                          release at T3

CPU frame                                                   [========= OWNED ========)
                                                                      StorageManager owns

remote SLOT                                                                     COMMITTED
                                                                                at T5

PAGE path    store PAGE chunks ---------------------------> PAGE visible
```

`T2` 是每个 rank 的本地 gather event；真正释放 `dst` 的时间是
`max(T2_rank0, ..., T2_rankN)` 经 aggregator 回传之后。它不需要等 `T3/T4/T5`。

| 资源 | 最早安全释放点 | 不需要等待 |
|---|---|---|
| `StateGroupPool dst` | 所有 TP rank 完成 `dst -> GPU temp`，或确认本 rank没有再读 `dst` | D2H、PAGE lookup、AOS1 put、StorageManager visibility |
| connector GPU temp row | `GPU temp -> CPU frame` D2H 完成 | PAGE visibility、AOS1 put |
| CPU frame / MemoryObj | StorageManager 接管 ownership 或 put 终止 | scheduler sidecar commit callback |

PAGE physical blocks继续遵循现有 PAGE save completion/deferred-free 生命周期，不和
checkpoint lease 混用。

## 10. 失败和竞态矩阵

失败时先判断“checkpoint bytes 是否有效”以及“GPU 是否还可能读取 group”：

```text
                         native src -> dst valid?
                                  |
                         +--------+--------+
                         |                 |
                        NO                YES
                         |                 |
                         v                 v
             GPU access definitely done?  gather / cleanup completion proven?
                         |                 |
                    +----+----+       +----+----+
                    |         |       |         |
                   YES        NO     YES        NO
                    |         |       |         |
                    v         v       v         v
             invalidate   quarantine  preserve  quarantine
             hash+release  group      hash+release group
                                      |
                                      +--> sidecar success/failure is independent
```

核心原则只有一句：**无法证明 GPU 已经不再访问时，宁可 quarantine，也不能把
group 放回 free list。**

| 场景 | checkpoint group | GPU temp | 远端 SLOT |
|---|---|---|---|
| State free list 无 group | 不创建 lease | 不预留 | 跳过/失败；PAGE 可继续 |
| 无匹配 interval `SlotSaveSpec` | 不创建 offload lease | 不预留 | 不触发 |
| save admission/temp row 失败，native copy 未发出 | abort 后 invalidate/release | 无 | 失败 |
| admission 失败，但 native copy 会随 batch 发出 | 等 native-copy event 后保留 hash并release | 无 | 失败 |
| native copy/build abort | all-rank terminal 后 invalidate/release | release | 失败 |
| gather 失败，但 copy 有效且 GPU completion 已确认 | 保留 hash并release | release或quarantine | 失败 |
| gather/event completion 无法确认 | quarantine，不复用 | quarantine | 失败/worker unhealthy |
| save executor 拒绝 | 先等 event，再release lease | D2H未开始则在event后release | 失败 |
| D2H 失败 | 已release | completion确认后release/quarantine | 失败 |
| PAGE visibility/AOS1 put/storage失败 | 已release | 已release | 不commit |
| sequence finish/preempt | lease独立存在，不能由seq deallocate提前释放 | 正常完成 | 按save结果 |
| late/duplicate completion | copy_id幂等忽略 | 无影响 | 无影响 |
| 一个TP失败、其他TP成功 | 等所有TP terminal；最终invalidate或保守处理 | 各rank独立清理 | 全局不commit |

## 11. 实现后的修改点

### Unified codec

- 新增 `atom/kv_transfer/offload/hybrid/dsv4/` vertical package：
  - `__init__.py`：导出 DSV4 connector/profile/codec stable surface；
  - `connector.py`：从当前 hybrid connector 迁入 DSV4 orchestration；
  - `policy.py`：`DSV4OffloadProfile`、config parsing、boundary/hash/fingerprint/commit，
    以及 connector-owned staging row 的 admission/quarantine；
  - `codec.py`：`DSV4PageSlotCodec` + `DSV4CheckpointCodec` +
    `DSV4CheckpointStore`，并内含 private RegionSet/typed plan；
  - `triton_page_slot.py`：DSV4 region gather/scatter kernels；
  - `bytes_per_block` 保持 PAGE-only，另暴露 `slot_bytes`。
- `atom/kv_transfer/offload/connector.py` 直接构造
  `hybrid/dsv4/connector.py` 的 worker/scheduler；不再保留 hybrid 根层 facade。
- profile、format/store、policy 已分别收拢到 `dsv4/policy.py` 和
  `dsv4/codec.py`；`hybrid/profiles/`、`sidecar_format.py`、`store.py`、
  `policy.py` 均已删除。
- PAGE-only/SLOT-only adapter 和 codec-owned staging pool 已删除；生产和测试
  均直接使用 `DSV4PageSlotCodec`。
- `atom/kv_transfer/offload/copy_plan.py` 已删除；DSV4 typed geometry/plan 只存在于
  `dsv4/codec.py`。
- dense chunk/segment-major kernel 已迁至
  `atom/kv_transfer/offload/dense/triton_kv_staging.py`；旧 raw copy-plan kernel
  随 adapter 一起删除。
- dense 和 DSV4 PAGE 直接使用共享 `BlockGPUConnector`，不再保留两个零行为
  family facade。
- `atom/kv_transfer/offload/hybrid/dsv4/triton_page_slot.py`
  - 同一个 region-copy kernel body 以 `REVERSE: tl.constexpr` 生成两个 specialization；
  - PAGE launch 固定 `REVERSE=False`，SLOT launch 固定 `REVERSE=True`；
  - 不实现每 tile runtime 判断 PAGE/SLOT 的 monolithic branch kernel；
  - wrapper 只 enqueue 到 caller 传入的 stream，不 synchronize、不记录 event。
- `DSV4PageSlotCodec.gather(checkpoint_plan, ..., stream=S)` 的 bundle contract：
  - 在同一个 `S` 上先 enqueue PAGE specialization；
  - 再在同一个 `S` 上 enqueue SLOT specialization；
  - 两个 section 写入同一 staging buffer 的不同 offset；
  - codec 返回后由 caller 在 `S` 上记录唯一 `E_bundle_ready`；
  - 一次 bundle 等于 one buffer + ordered two launches + one final event，不要求
    one monolithic launch。
- `atom/kv_transfer/offload/hybrid/dsv4/connector.py`
  - 启动时只构造一个 `DSV4PageSlotCodec`；
  - PAGE engine 的 `BlockGPUConnector` 和 SLOT admission/store 引用同一实例；
  - LMCache native lookup 只得到 PAGE hit；connector 用 committed SLOT boundary/hash
    计算 `effective_hit = max(B <= page_hit)`；
  - 无 matching SLOT、hash mismatch 或 stateful nonzero HBM floor 时 fail closed；
  - connector 自己持有 SLOT GPU temp rows/admission/quarantine；
  - 迁移期可令 `self._slot_codec = self._codec`，随后移除双字段。

实现没有修改 `BlockGPUConnector`、LMCache raw metadata、`engine.store/retrieve` 与
checkpoint store 的 StorageManager ownership 语义；它们只切换到 DSV4 package 和
unified codec 的 typed entrypoint。

### Scheduler / state ownership

- `atom/model_engine/state_pool.py`
  - checkpoint lease map；
  - external pin 与 free-list/index/resize 交互；
  - idempotent preserve-release / invalidate-release。
- `atom/model_engine/block_manager.py`
  - lease acquire/complete 封装。
- `atom/model_engine/scheduler.py`
  - `SlotSaveSpec` 与 `StateCheckpointCopy` 精确匹配；
  - staging completion 驱动 lease release。

### Worker / completion transport

- `atom/model_engine/model_runner.py`
  - 保证 native copy 后立刻调用 gather hook；
  - copy abort 与 completion event 路径。
- `atom/kv_transfer/offload/hybrid/dsv4/connector.py`
  - 预留 temp row；
  - staging event terminal notification；
  - GPU temp row 在 D2H 后提前释放。
- `atom/kv_transfer/disaggregation/types.py`
  - staging completion fields。
- `atom/kv_transfer/disaggregation/aggregator.py`
  - all-TP staging terminal aggregation。
- `atom/kv_transfer/disaggregation/multi/multi_connector.py`
  - completion passthrough/union。
- `atom/kv_transfer/offload/connector.py`
  - worker shell hook passthrough。

## 12. 测试计划

### PAGE/SLOT hit intersection

- PAGE hit 16K、只有 committed SLOT@8K -> effective hit 8K；
- PAGE hit 20K、committed SLOT@8K/@16K -> effective hit 16K；
- PAGE hit 小于最新 SLOT boundary -> 回退到不超过 PAGE hit 的前一个 committed boundary；
- PAGE hit 存在但无 committed SLOT -> stateful hit 为 0并清理 lookup pin/state；
- boundary token 相同但 chained prefix hash 不同 -> 不匹配；
- full-prompt PAGE hit 先减一 token，再选择 prior resumable boundary；
- 只有 all-TP SLOT save quorum 才进入 committed index；任一 rank失败不命中；
- committed LRU eviction 或 scheduler restart 后默认 fail closed；
- worker PAGE retrieve 成功但 SLOT get/CRC/scatter失败 -> whole load失败并recompute；
- SLOT 成功但 PAGE retrieve失败 -> whole load失败；
- stateful nonzero HBM floor -> v1拒绝远端 PAGE+SLOT混合；
- 如果实现 persistent checkpoint lookup adapter：batched all-rank contains、negative
  cache、eviction race 和 load-time revalidation 都要覆盖。

### Unified codec

- PAGE `page_plan()` 与当前 block-major/region-minor golden layout byte-identical；
- SLOT `slot_plan()` 与当前 reverse-indexed plane-order golden layout byte-identical；
- `checkpoint_plan()` section offset 等于 PAGE/SLOT plan 的精确前缀和；
- PAGE/SLOT 共享 gather/scatter kernel 后仍分别 round trip；
- PAGE forward region 拒绝 reverse descriptor，SLOT reverse region 拒绝 forward descriptor；
- block ID 与 group ID 的越界/duplicate 校验互不混淆；
- `bytes_per_block == page_bytes_per_block`，SLOT bytes 不进入 LMCache PAGE shape；
- BF16 one-plane 和 FP8 two-plane SLOT 均覆盖；
- 旧 codec import adapter 与 unified codec 产生相同 plan，迁移完成后再删除 adapter 测试；
- checkpoint composite plan 只验证 codec/layout，不在第一版声称 Storage bundle 已发布。
- bundle launch-order test：同一个 stream object 上严格调用 PAGE(False) 后
  SLOT(True)，offset 分别落在 PAGE/SLOT section；
- bundle event test：caller 只在两个 launch 都 enqueue 后记录一个 event，等待该
  event 后 PAGE/SLOT staging bytes 都完整；
- kernel contract test：PAGE/SLOT 使用两个 constexpr specialization，禁止
  runtime per-tile address-mode branch；
- composite scatter 对称地在同一 stream 上先后恢复 PAGE/SLOT，并做 byte-identical
  round trip。

### StateGroupPool

- acquire lease 后 group 不在 free list；
- resume/adopt 不能写入 leased checkpoint；
- `retire_top()` 跳过 leased top group；
- preserve-release 后 group 回到 checkpointed free list；
- invalidate-release 后回到 vacant free list；
- hash orphan 时 lease 仍阻止重分配；
- late/duplicate/wrong-group `copy_id` 不误释放。

### Worker connector

- `copy -> gather -> event` 顺序；
- event 完成后先报告 staging completion，再进行/完成 storage publication；
- checkpoint group completion 不等待 PAGE visibility；
- GPU temp row 只持有到 D2H 完成；
- admission、snapshot、event、executor、D2H、put 各失败点的 cleanup/quarantine。

### TP aggregator / MultiConnector

- 所有 rank 完成才释放；
- 一 rank abort 导致全局 invalidating release；
- partial completion 不释放；
- late duplicate completion 被 tombstone/幂等逻辑忽略；
- MultiConnector 不用 send/save pairing 延迟 staging completion。

### End-to-end scheduler

- PAGE 每 256 tokens 保存，SLOT 只在配置 interval 保存；
- demand/off-grid terminal 不产生 SLOT；
- interval terminal live fallback；
- free-list 压力下 SLOT miss 不影响 PAGE 保存和请求正确性；
- offload 后 native checkpoint 可被正常命中或按 LRU 淘汰。

## 13. Review 时需要确认的决策

1. lease 释放后是否保留 native checkpoint hash：**第一版已采用：保留**。
2. terminal 不在 interval 时是否跳过 SLOT：**第一版已采用：跳过**。
3. interval terminal 无后继 batch 时是否允许 live-group fallback：**第一版已采用：允许**。
4. 无法确认 GPU completion 时是否 quarantine：**第一版已采用：保留 lease、禁止复用**。
5. 是否需要 maintenance RPC 强制所有 terminal 也经过 free-list `dst`：
   **第一版已采用：不做**。
6. PAGE 与 SLOT 是否合并为一个 GPU layout codec：**已确认，第一版合并为
   DSV4 专用的 `DSV4PageSlotCodec`**；PAGE/SLOT typed plan 共用 copy engine。
7. 是否同时强制一次 composite gather 和一个 Storage bundle：**第一版不做**；
   保留 `checkpoint_plan()`，等 lease 生命周期稳定后只为稀疏/terminal checkpoint
   单独评审 bundle storage policy。
8. 如果启用 bundle，PAGE/SLOT 是否合成一个带 runtime 方向分支的大 kernel：
   **已确认不合成**。同一个 checkpoint stream 上依次运行 PAGE forward specialization
   和 SLOT reverse specialization，共用一个 staging buffer，最后只记录一个
   `E_bundle_ready`。
9. scheduler restart 后是否必须发现 StorageManager 中已有 SLOT checkpoint：
   **第一版不要求，采用 session-local fail-closed**。当前实现使用 session-local
   committed hash index，LMCache native token lookup 不会发现 SLOT；若后续要求跨
   restart 复用，需要新增 all-TP batched checkpoint contains/manifest lookup adapter。

上述决策确认后才进入实现、回归、提交和推送阶段。
