# Local KV Cache Offload (CPU DRAM / NVMe)

This module adds a *local* second/third-tier KV cache to ATOM's standalone
serving stack — CPU DRAM as L2, optional NVMe as L3, sitting behind the HBM
paged prefix cache. It's the standalone-mode counterpart to the
[ATOM-as-vLLM-plugin + LMCache recipe](../../../recipes/atom_vllm/LMCache-KV-Cache-Offload.md):
that path borrows vLLM's `KVConnectorV1` so LMCache plugs into vLLM, but
leaves ATOM standalone without any offload story. This module fills that
gap.

## When to use

Same workload profile as the plugin-mode recipe:

* Multi-turn / agentic serving where conversations reuse 90–97 % of their
  prefix across turns (Claude Code, Cursor, Devin-style agents).
* Long prompts (20K–150K tokens), short outputs.
* Many concurrent users contending for HBM KV cache.

Low-concurrency setups whose working set already fits in HBM should stay on
ATOM's native prefix cache — offload adds D2H/H2D copy overhead that's not
amortized when there's no eviction pressure.

## Architecture

```
+-----------------------------+
|   Scheduler (PD-style hooks)|
|  get_num_new_matched_tokens |  --> OffloadConnectorScheduler.lookup_external_hits
|  update_state_after_alloc   |  --> OffloadConnectorScheduler.queue_load
|  postprocess (hash_blocks)  |  --> OffloadConnectorScheduler.queue_save
|  build_connector_meta       |  --> OffloadConnectorMetadata
+-----------------------------+
              |
              | ScheduledBatch.connector_meta_output
              v
+-----------------------------+
|         EngineCore          |
|  process_kvconnector_output |
+-----------------------------+
              |
              v
+------- worker (per TP rank) -----+
| OffloadConnector.start_load_kv   |  --> H2D loads + D2H saves
| OffloadConnector.get_finished    |  --> (done_saving, done_loading)
+----------------------------------+
              |
              v   KVConnectorOutput aggregated across ranks
              |   via KVOutputAggregator (already wired by engine_core)
              v
+-----------------------------+
| Scheduler._update_from_     |
| kv_xfer_finished            |  --> finished_recving wakes
|                             |      WAITING_FOR_REMOTE_KVS seqs
+-----------------------------+
```

Key differences from the existing `atom/kv_transfer/disaggregation/` path:

| Concern | PD disaggregation | OFFLOAD |
|---|---|---|
| Store location | Remote node (RDMA via Mooncake/MoRIIO) | Local CPU DRAM / NVMe |
| Transfer protocol | RDMA push (producer) or pull (consumer) | HIP `memcpy` / pinned buffers |
| Block free policy | Producer defers free until send completes | Defer only while an async D2H save is pending |
| `KVConnectorRole` | `PD_PRODUCER` / `PD_CONSUMER` | `OFFLOAD` |
| Per-block hash key | Anchored by sequence id + remote tag | `BlockManager.compute_hash` (same as in-HBM prefix cache) |
| Metadata shape | `ConnectorMetadata` with `remote_host`/`port` | `OffloadConnectorMetadata` with `block_ids`/`hashes` only |

The scheduler's role split is preserved (`get_num_new_matched_tokens`,
`update_state_after_alloc`, `build_connector_meta`, `request_finished`,
`_update_from_kv_xfer_finished`), so adding offload doesn't introduce a
second scheduling path — only branching on `KVConnectorRole`.

## Save completion and block free

`queue_save(request_id, published)` only records scheduler-side metadata.
The actual D2H save starts later on each worker when
`EngineCore.process_kvconnector_output` dispatches the
`OffloadConnectorMetadata` snapshot to `LMCacheOffloadConnector.start_load_kv`.

The worker judges save completion with a CUDA event:

1. `start_load_kv` enqueues `dst.copy_(src, non_blocking=True)` on the
   connector's copy stream.
2. After all K/V layer copies for a request are enqueued, it records
   `torch.cuda.Event()` on that stream and stores `(req_id, "save", evt)`.
3. `get_finished()` polls the event via `evt.query()`. Completed save
   events become `done_save`.
4. `ModelRunner.async_proc_aggregation()` returns `done_save` as
   `KVConnectorOutput.finished_sending`.
5. `Scheduler._update_from_kv_xfer_finished()` treats offload
   `finished_sending` as the point where a pending save is complete and
   any deferred GPU blocks may be returned to `BlockManager`.

This means a finished request with no queued offload save still frees its
blocks immediately, but a request with a pending async D2H save keeps its
source GPU blocks alive until `finished_sending` arrives.

## Implementations

* `atom.kv_transfer.offload.lmcache.LMCacheOffloadConnector` — wraps
  [LMCache](https://github.com/LMCache/LMCache) (built from source with
  `BUILD_WITH_HIP=1` so its HIP `c_ops` backend is used for KV transfer).
  Bypasses LMCache's `LMCacheConnectorV1` to avoid pulling in vLLM type
  dependencies — talks directly to the engine-level API + `lmcache.c_ops`.

## Cache key consistency

The OFFLOAD connector uses the *same* xxhash64 chain that ATOM's HBM
prefix cache uses (`BlockManager.compute_hash`), so HBM hits and offload
hits live in the same hash space. The scheduler-side lookup order:

1. `BlockManager.can_allocate(seq)` — HBM hit, no copy required.
2. On miss, `OffloadConnectorScheduler.lookup_external_hits` — CPU/NVMe
   hit triggers H2D.
3. On double miss, real prefill runs; finalized blocks are queued for
   save via `queue_save` after `hash_blocks` publishes their hash.

This means TP workers must produce identical hash chains for identical
prompts. The xxhash chain is deterministic, but `PYTHONHASHSEED=0` is
still required at engine start to avoid Python-level hash randomization
leaking into any auxiliary data structure the connector may use; the
LMCache connector enforces this at `__init__`.
