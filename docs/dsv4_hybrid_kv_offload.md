# DeepSeek V4 Hybrid KV Cache Offload

DeepSeek V4 offload treats a reusable checkpoint as an atomic pair:

```text
checkpoint = PAGE data + one complete SLOT sidecar
```

PAGE and SLOT share the main HBM allocation, but remain separate logical and
storage objects. A PAGE-only hit is not sufficient to resume a stateful DSV4
request.

## HBM layout

`DeepseekV4AttentionMetadataBuilder.allocate_per_req_cache()` creates one byte
allocation and carves the main NoPE and optional RoPE planes from it. Each plane
contains a PAGE address space and pre-reserved Active SLOT capacity. Sharing the
physical allocation does not make an Active SLOT a PAGE ID.

```text
One physical allocation: per_req_pool

  Low address                                                   High address
       |                                                             |
       v                                                             v

  NoPE plane
  +------------------------------------------+--------------------------+
  | PAGE address space                       | Active SLOT capacity     |
  | PAGE 0 | PAGE 1 | ... | PAGE N-1        | ... | group 1 | group 0 |
  +------------------------------------------+--------------------------+
    managed by BlockPool                       managed by StateGroupPool
    forward indexed                            reverse indexed

  RoPE plane, when FP8 KV is enabled
  +------------------------------------------+--------------------------+
  | PAGE address space                       | Active SLOT capacity     |
  | PAGE 0 | PAGE 1 | ... | PAGE N-1        | ... | group 1 | group 0 |
  +------------------------------------------+--------------------------+

  Separate PAGE-only allocation
  +--------------------+--------------------+--------------------+
  | CSA indexer PAGE 0 | CSA indexer PAGE 1 |        ...         |
  +--------------------+--------------------+--------------------+
```

Free PAGE units remain inside the PAGE address space. `BlockPool` tracks their
logical state; their position in this diagram does not imply their state:

```text
PAGE ID       0          1          2          3          4
          +----------+----------+----------+----------+----------+
state     | live KV  | free     | prefix   | state    | free     |
          |          |          | cached   | ckpt unit|          |
          +----------+----------+----------+----------+----------+
```

The right side contains only capacity for request-owned Active SLOT groups. A
free group there is available for admitting another active request; it is not a
free PAGE unit and is not where a retained PAGE-backed checkpoint lives.

One PAGE contains compressed KV envelopes and the corresponding CSA indexer
regions. One SLOT contains the complete mutable request state:

```text
+------------------+-----------------------+-------------------+
| compressor state | SWA rings, all layers | MTP extra entries |
+------------------+-----------------------+-------------------+
```

MTP is covered by the normal SLOT path because the ring width is
`win_with_spec = window_size + mtp_k`.

`get_kv_transfer_tensors()` exports the two logical views:

```text
block_regions       PAGE, indexed forward by physical block ID
swa_block_regions   full SLOT, indexed backward by request group ID
staging_region      compressor-only P/D staging; not a sidecar source
```

The address ABI is:

```text
PAGE address(block_id):
    base + block_id * unit_bytes

Active SLOT address(group_id):
    base + total_bytes - (group_id + 1) * unit_bytes
```

Reverse Active SLOT indexing keeps existing groups at stable addresses when the
SLOT side grows toward the PAGE side.

### Active SLOT, native checkpoint, and offload sidecar

The three storage roles are distinct:

```text
Active request state
--------------------

Active SLOT group G
+-----------------------------+
| compressor state + SWA ring |
+-----------------------------+
       contiguous, right side


Native HBM state checkpoint
---------------------------

Active SLOT group G
          |
          | BlockPool.reserve_units(K)
          | segmented copy
          v
   +--------+  +---------+       +---------+
   | PAGE 7 |  | PAGE 31 |  ...  | PAGE 92 |
   +--------+  +---------+       +---------+
      arbitrary PAGE IDs in the left-side PAGE address space


LMCache SLOT offload
--------------------

Active SLOT group G
          |
          v
GPU staging row --> pinned CPU frame --> CPU/NVMe AOS1 sidecar
```

`PagedStateCheckpointCoordinator` stores native state checkpoints in arbitrary
free PAGE units obtained from `BlockPool`; it does not keep them in free Active
SLOT groups. The LMCache sidecar path does not retain HBM PAGE units at all. It
uses a bounded temporary GPU row and then persists the complete SLOT in CPU or
NVMe storage.

## Storage representation

`DSV4PageSlotCodec` owns both geometry descriptions and uses Triton gather and
scatter kernels for raw byte movement.

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

              PAGE objects in LMCache                         SLOT sidecar
              token/chunk addressed                           boundary addressed

  +------------------------------------------+     +----------------------------+
  | block 0                                  |     | AOS1 header, 128 bytes     |
  |   NoPE | RoPE | indexer layer 0 | ...    |     | boundary tokens/hash       |
  +------------------------------------------+     | layout fingerprint         |
  | block 1                                  |     | TP size/rank               |
  |   NoPE | RoPE | indexer layer 0 | ...    |     | payload size + CRC32       |
  +------------------------------------------+     +----------------------------+
                                                   | full SLOT payload          |
                                                   | NoPE slot | RoPE slot       |
                                                   +----------------------------+
```

PAGE objects keep LMCache's normal token-derived chunk keys. A SLOT uses a
content-derived key containing the chained boundary hash, layout fingerprint,
and TP identity. The payload does not store the source group ID; restore always
targets the new request's allocated destination group.

## Save path

At an aligned checkpoint boundary, the scheduler emits `SaveSpec` and
`SlotSaveSpec`. Connector metadata is dispatched before the next forward, so
the worker snapshots the mutable SLOT on the current stream before that forward
can update it.

```text
Scheduler                         Worker current stream
---------                         ---------------------

prefill reaches boundary B
        |
        | build_connector_meta()
        | SaveSpec + SlotSaveSpec
        v
                              previous forward finished
                                        |
                                        v
                              gather_slot(source_group)
                              HBM SLOT --> GPU staging row
                                        |
                                  record CUDA event
                                        |
                                        +------> next forward may mutate SLOT


Background save worker
----------------------

wait CUDA event
      |
      v
GPU staging row --D2H--> pinned CPU AOS1 frame
      |
      +--> release GPU staging row
      |
      v
gather completed PAGE blocks
      |
      v
Triton pack --> D2H --> LMCache.store(PAGE chunks)
      |
      v
wait until PAGE is visible through boundary B
      |
      v
finalize AOS1 header and CRC
      |
      v
StorageManager.put(SLOT sidecar)
      |
      v
wait until sidecar is visible
      |
      v
all TP ranks report the same SaveOperationId
      |
      v
scheduler commits boundary hash B
```

The visibility order is deliberate. Publishing the sidecar before all PAGE
chunks are visible could authorize an incomplete checkpoint.

## Load path

The scheduler first queries the normal LMCache PAGE prefix. For a stateful DSV4
request, it reduces that hit to the newest aligned boundary whose sidecar was
committed by all TP ranks in the current scheduler session.

```text
Scheduler
---------

same prompt
    |
    v
LMCache lookup --> PAGE hit H
    |
    v
select newest committed PAGE+SLOT boundary B
    |
    +--> no committed SLOT: miss and recompute
    |
    v
allocate destination PAGE blocks and SLOT group
    |
    v
emit LoadSpec + SlotLoadSpec(destination_group)
    |
    v
park request in WAITING_FOR_REMOTE_KVS


Worker
------

LMCache.retrieve(PAGE)
        |
        v
CPU MemoryObj --> bounded GPU staging
        |
        v
Triton scatter PAGE by physical block ID
        |
        +--> missing PAGE: fail
        |
        v
borrow and validate AOS1 sidecar
        |
        | magic/version, boundary, payload size,
        | fingerprint, TP identity, and CRC32
        |
        +--> missing or invalid sidecar: fail
        |
        v
CPU payload --> GPU staging --> scatter_slot(destination_group)
        |
        v
all TP ranks finish
        |
        v
wake request and continue suffix prefill/decode
```

The worker restores PAGE first and SLOT second. The composite load succeeds
only when both succeed; otherwise the request falls back to recomputation.

## Main code paths

- `atom/model_ops/attentions/deepseek_v4_attn.py`
  - `allocate_per_req_cache()` creates the shared planes and embeds the state
    arena in each complete SLOT.
  - `get_kv_transfer_tensors()` exports forward PAGE and reverse SLOT regions.
- `atom/model_ops/attentions/v4_pool_geometry.py`
  - `UnifiedPoolGeometry` owns all PAGE/SLOT address arithmetic.
- `atom/model_engine/block_pool.py`
  - `BlockPool` tracks live, cached, free, and raw-checkpoint PAGE units.
- `atom/model_engine/page_unit_checkpoint.py`
  - `PagedStateCheckpointCoordinator` reserves PAGE units for native HBM state
    checkpoints and releases them as one atomic object.
- `atom/kv_transfer/offload/hybrid/dsv4/codec.py`
  - `DSV4PageSlotCodec` builds PAGE/SLOT copy plans.
  - `DSV4CheckpointCodec` implements the AOS1 frame.
  - `DSV4CheckpointStore` persists sidecars through LMCache storage tiers.
- `atom/kv_transfer/offload/hybrid/dsv4/triton_page_slot.py`
  - Implements forward and reverse raw-byte gather/scatter.
- `atom/kv_transfer/offload/hybrid/dsv4/connector.py`
  - `DSV4OffloadScheduler` selects boundaries and builds request metadata.
  - `DSV4OffloadConnector` performs snapshot, PAGE transfer, sidecar commit,
    validation, and restore.
- `atom/kv_transfer/offload/_block_gpu_connector.py`
  - Bridges LMCache MemoryObjs and ATOM PAGE blocks through bounded GPU staging.

## Current safety constraints

- Stateful DSV4 restore requires both PAGE and SLOT at the same boundary.
- A stateful offload load is skipped when the real HBM prefix floor is nonzero;
  version 1 does not merge an HBM state checkpoint with a later SLOT sidecar.
- SLOT commits are scheduler-session-local.
- FP4 indexer offload and pipeline parallelism greater than one are rejected.
