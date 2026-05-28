# KV Cache Offload (CPU DRAM) — ATOM Standalone

This guide covers the OFFLOAD connector subsystem at
`atom/kv_transfer/offload/`, which adds a CPU-DRAM second-tier KV cache
to **ATOM standalone serving**
(`python -m atom.entrypoints.openai_server`). The first-tier HBM prefix
cache (driven by `BlockManager`) stays in place; the OFFLOAD connector
sits behind it and remembers blocks that have already aged out of HBM,
so a repeat request can H2D-load them instead of re-running prefill.

If you came from the
[LMCache plugin recipe](../recipes/atom_vllm/LMCache-KV-Cache-Offload.md):
that one is for **ATOM-as-vLLM-plugin** mode, where vLLM owns the
`KVConnectorV1` interface and LMCache plugs into vLLM — ATOM itself
doesn't participate in offload. **This guide is the standalone-mode
counterpart**, and the two paths can coexist (a deployment can pick
either form without affecting the other).

## When to enable

OFFLOAD pays off when **HBM eviction pressure is real**:

- Multi-turn / agentic serving where conversations reuse 90 %+ of their
  prefix across turns (Claude Code, Cursor, Devin-style agents).
- Long prompts (20K–150K tokens), short outputs.
- Many concurrent users contending for the HBM KV pool.

If your working set already fits in HBM, the in-engine prefix cache
alone serves every hit for free. Adding OFFLOAD then only adds D2H copy
overhead with no return. Keep it off in that regime.

## Architecture

```
┌────────── Scheduler (engine process) ──────────┐
│  schedule():                                   │
│   ├─ get_num_new_matched_tokens(seq) ─────┐    │
│   │   • walk BlockManager.compute_hash    │    │
│   │     chain for the prompt              │    │
│   │   • count HBM hits (free)             │    │
│   │   • then count OFFLOAD hits (mirror)  │    │
│   │   • return external-only token count  │    │
│   ├─ BlockManager.can_allocate / allocate │    │
│   └─ update_state_after_alloc(seq) ───────┤    │
│       • queue H2D load for fresh slots    │    │
│         beyond the HBM-hit prefix         │    │
│                                           │    │
│  postprocess(...):                        │    │
│   ├─ if not seq.prefix_hashes_published:  │    │  scheduler-side
│   │    published = hash_blocks(seq, n)    │    │  connector
│   │    queue_save(req_id, published) ─────┼──→ LMCacheOffloadConnectorScheduler
│   │                                       │    │  • saved_hashes (mirror)
│   ├─ append/stream/free                   │    │  • _pending_save / _pending_load
│   └─ build_connector_meta() ──────────────┼──→ build_connector_meta()
│                                           │    │  returns OffloadConnectorMetadata
│  _update_from_kv_xfer_finished(out):◄─────┘    │
│   ├─ finished_recving → WAITING_FOR_REMOTE_KVS │
│   │   transitions back to WAITING              │
│   └─ finished_sending → clear pending save     │
│       and release deferred GPU blocks          │
└────────────────┬───────────────────────────────┘
                 │ ScheduledBatch.connector_meta_output
                 ▼
        EngineCore.process_kvconnector_output
                 │  (IPC to each TP worker)
                 ▼
┌────────── Worker (per TP rank) ────────────────┐
│  LMCacheOffloadConnector                       │
│   • register_kv_caches → alloc pinned CPU pool │
│     (cpu_bytes_per_rank / bytes_per_block      │
│      slots), capture per-layer K and V tensor  │
│     handles                                    │
│   • start_load_kv(meta):                       │
│       copy_stream.wait_stream(compute) ────── ensures D2H reads only
│       for save block:                          after the prefill's
│         for layer in 0..L:                     attention write has
│           gpu_block.bytes ─copy_→ cpu_pool     landed in HBM
│       for load block:                          (no race on prefill KV)
│         for layer in 0..L:
│           cpu_pool ─copy_→ gpu_block.bytes
│       compute.wait_stream(copy_stream) ─────── ensures the next forward
│       record CUDA event per request            sees fully-loaded KV
│   • get_finished() → poll events → return     │
│     (done_save, done_load) request id sets    │
│   • hash_to_slot: dict (worker truth)          │
│                                                │
│  KVOutputAggregator (existing, reused from PD) │
│   • aggregate per-rank KVConnectorOutput,      │
│     vote, return to scheduler                  │
└────────────────────────────────────────────────┘
```

The factory registry, ConnectorMetadata flow, and EngineCore dispatch
are the same machinery used by the existing `disaggregation/`
Mooncake/MoRIIO PD connectors. OFFLOAD is differentiated by a new
`KVConnectorRole` enum on the shared base — see the role table in
[`base.py`](../atom/kv_transfer/disaggregation/base.py).

## Quickstart

```bash
# Inside the container with ATOM + LMCache (BUILD_WITH_HIP=1) installed
export PYTHONHASHSEED=0   # mandatory — TP-rank hash consistency
export AITER_LOG_LEVEL=WARNING

python -m atom.entrypoints.openai_server \
  --model <model_path> --kv_cache_dtype fp8 -tp <N> --trust-remote-code \
  --gpu-memory-utilization 0.78 \
  --kv-transfer-config '{
      "kv_connector": "lmcache_offload",
      "kv_role":      "offload",
      "cpu_bytes":    17179869184
  }'
```

You should see, once per worker rank at startup:

```
LMCacheOffloadConnector initialized (rank-local pool=8.00 GB, ...)
LMCacheOffload: 8456 slots × 1015808 bytes (pool=8.00 GB, num_layers=62, ...)
```

…and once in the scheduler process:

```
LMCacheOffloadConnectorScheduler initialized (chunk_size=256)
```

Health-check + a sample completion:

```bash
curl --noproxy '*' -s http://127.0.0.1:8000/health
curl --noproxy '*' -s http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model_path>","prompt":"...","max_tokens":8,"temperature":0}'
```

Run the same prompt twice; the second time the scheduler log should
show an HBM hit (the OFFLOAD save fires too but HBM serves the request
without needing an H2D):

```
Scheduled prefill batch: 1 reqs, 300 new tokens (cached: [0],   new: [300]), req_ids: (0,)
Scheduled prefill batch: 1 reqs,  12 new tokens (cached: [288], new: [12]),  req_ids: (1,)
```

## Configuration

All knobs live in the `kv-transfer-config` JSON (dict in code).

| Key                    | Default                | Meaning                                                                 |
|------------------------|------------------------|-------------------------------------------------------------------------|
| `kv_connector`         | required               | Set to `"lmcache_offload"` to select this backend                        |
| `kv_role`              | `"offload"`            | Free-form label, future-reserved for sub-modes                          |
| `cpu_bytes`            | `8 * 1024**3` (8 GiB)  | **Server-wide** CPU pool budget; split evenly across TP ranks           |
| `cpu_bytes_per_rank`   | `cpu_bytes / TP`       | Explicit per-rank override (use when split is not even)                 |
| `chunk_size`           | `256`                  | Reserved for the future LMCacheEngine wrapper; ignored in this impl     |
| `disk_path`            | unset                  | Reserved for the future NVMe (L3) tier; currently warns + ignores       |

Geometry of one slot (filled at `register_kv_caches`):

- `bytes_per_block = Σ layers (sizeof(K block) + sizeof(V block))`
- `num_slots = cpu_bytes_per_rank // bytes_per_block`

So if you size `cpu_bytes` to N times the in-HBM block budget, you can
keep roughly N working sets warm in CPU. Typical: 1–4× of the HBM
pool for prefix-heavy workloads.

## Operational requirements

1. **`PYTHONHASHSEED=0` is mandatory.** Without it, TP-rank Python dict
   hashing diverges and the scheduler's optimistic mirror disagrees
   with the workers. The connector `__init__` raises a `RuntimeError`
   if the env var is unset or set to anything else.

2. **LMCache must be the HIP-built `c_ops` backend.** The PyPI wheel
   ships CUDA-linked binaries that don't load on ROCm. Build from
   source:

   ```bash
   git clone --depth 1 https://github.com/LMCache/LMCache.git /tmp/LMCache
   cd /tmp/LMCache
   BUILD_WITH_HIP=1 pip install -e . --no-build-isolation --no-deps
   pip install aiofile aiofiles blake3 msgspec sortedcontainers redis pyzmq \
               prometheus_client numba nvtx httptools setuptools_scm 'numpy<=2.2.6'
   ```

   Verify: `python -c "import lmcache" 2>&1 | grep backend` should show
   `Using backend: lmcache.c_ops`. The connector `__init__` rejects
   `lmcache.python_ops_fallback`.

3. **Enable ATOM's prefix cache** (default on). OFFLOAD piggybacks on
   the same xxhash64 block-hash chain that `BlockManager` already
   computes. Disabling prefix caching disables OFFLOAD too.

4. **Don't share the pool across processes.** Each TP rank allocates
   its own pinned CPU pool; sizing assumes per-rank ownership.

## Cache key consistency

OFFLOAD uses `BlockManager.compute_hash` (xxhash64 chain) as the cache
key, so HBM hits and OFFLOAD hits live in the same hash space. The
scheduler lookup order in `get_num_new_matched_tokens`:

1. **HBM** — `BlockManager.hash_to_block_id` (free if hit; no H2D
   needed).
2. **OFFLOAD** — connector's `saved_hashes` mirror (H2D from CPU pool
   if hit).
3. **Re-prefill** the residual on a miss.

The chain stops at the first OFFLOAD miss because xxhash chains
forward — a hash mismatch on block N means blocks N+1, N+2, … have
different content downstream too. This matches the in-HBM behavior of
`BlockManager.can_allocate`.

## A latent ATOM bug also fixed

This change includes a one-line behavioral fix for ATOM's in-HBM prefix
cache when running in deferred-output mode (the only mode currently
used —`tokenIDProcessor.is_deferred_out` is hard-coded `True`).

The old hash-publish gate was `seq.type == SequenceType.PREFILL`. In
deferred mode this never fires for the prefill step's seq, because the
prefill output surfaces one engine step later than the prefill forward,
and by then the next-step `schedule()` has flipped `seq.type` to
`DECODE`. The result: `BlockManager.hash_to_block_id` was never
populated for prompt blocks, and the HBM prefix cache was a silent
no-op (every request reported `cached:[0]` regardless).

The new gate uses a per-seq `Sequence.prefix_hashes_published` flag
that fires exactly once per seq at the first postprocess where `idx`
is not `None`, in both deferred and non-deferred modes. Side effect:
ATOM's existing HBM prefix cache now actually works in default
deployment — same-prefix requests show `cached:[288]` (or whatever the
chain count is) and TTFT drops accordingly, independent of whether
OFFLOAD is enabled.

If for any reason you want the old gate back, set the per-seq flag to
`True` immediately after `Sequence.__init__` — the gate becomes a
permanent no-op.

## Layer-by-layer vs batched

The current implementation is **batch-level**: one `start_load_kv`
call per engine step, with per-layer copy loops inside. CUDA events
are recorded once per request, not per layer. This keeps the
implementation small and avoids touching `AiterBackend` /
`AiterMLABackend`.

Layer-pipelined save/load (analogous to vLLM's
`wait_for_layer_load` / `save_kv_layer` hooks) is **not** done here.
It's a worthwhile follow-up if a benchmark shows D2H/H2D on the
critical path, but the current impl already overlaps the per-step
copy stream with the next-step compute stream via `wait_stream` —
the only thing missing is intra-step overlap with the same forward's
attention layers.

## Save completion and block lifetime

OFFLOAD saves are asynchronous D2H copies, so the scheduler must not
return a finished request's GPU KV blocks to `BlockManager` until the
worker has confirmed the save completed.

The completion signal is a CUDA event recorded by the worker-side
connector after all K/V block bytes for a request have been enqueued:

1. `LMCacheOffloadConnector.start_load_kv()` issues `dst.copy_(src,
   non_blocking=True)` on the dedicated copy stream.
2. It records `torch.cuda.Event()` on that same stream and stores
   `(req_id, "save", event)` in `_pending`.
3. Each engine step calls `ModelRunner.async_proc_aggregation()`, which
   calls `connector.get_finished()`.
4. `get_finished()` polls each event with `evt.query()`. If the event
   has completed and `kind == "save"`, the request id is returned in
   `done_save`.
5. `ModelRunner.async_proc_aggregation()` maps `done_save` to
   `KVConnectorOutput.finished_sending`.
6. `EngineCore` forwards that output to
   `Scheduler._update_from_kv_xfer_finished()`.
7. For `KVConnectorRole.OFFLOAD`, the scheduler removes the request id
   from `offload_pending_save_req_ids`; if the finished sequence was
   waiting in `deferred_free_blocks`, it finally calls
   `block_manager.deallocate(seq)`.

This matches the same lifetime invariant vLLM relies on: a paged GPU KV
block may be reused only after the connector has either finished saving
it or otherwise holds an extra reference preventing allocator reuse.

## Files

| Path                                                       | What                                                |
|------------------------------------------------------------|-----------------------------------------------------|
| `atom/kv_transfer/disaggregation/base.py`                  | Adds `KVConnectorRole` enum + `role` class attr     |
| `atom/kv_transfer/offload/base.py`                         | `OffloadConnectorBase` + `OffloadConnectorSchedulerBase` |
| `atom/kv_transfer/offload/types.py`                        | `OffloadReqMeta` + `OffloadConnectorMetadata`       |
| `atom/kv_transfer/offload/__init__.py`                     | Re-exports + factory registration                   |
| `atom/kv_transfer/offload/lmcache/lmcache_connector.py`    | Concrete `LMCacheOffloadConnector` (worker + scheduler) |
| `atom/kv_transfer/offload/README.md`                       | Module-local overview (this guide is the user-facing one) |
| `atom/model_engine/block_manager.py`                       | `hash_blocks` returns the published `(bid, h)` list |
| `atom/model_engine/scheduler.py`                           | Role-aware dispatch helpers + OFFLOAD queue_save hook + pending-save deferred free |
| `atom/model_engine/sequence.py`                            | `Sequence.prefix_hashes_published` flag             |
| `atom/utils/forward_context.py`                            | Lazy `import atom.kv_transfer.offload` so factory registers in worker subprocesses |
| `tests/test_offload_connector.py`                          | Covers factory wiring, save-queue + mirror, block lifetime, lookup HBM-vs-external dedup, update_state_after_alloc, worker pool, env checks |

## Known limitations

- **NVMe (L3) tier** not implemented. `disk_path` accepted but warns
  and ignores. Slated for a follow-up via the LMCacheEngine wrapper.
- **Optimistic-mirror eviction races.** If the worker FIFO-evicts a
  slot between save and load, a subsequent load lookup reports
  done-with-empty-block (logged as `WARN: %d/%d H2D load misses`).
  Mitigation: size `cpu_bytes` comfortably above the working set;
  surface the miss back to the scheduler in a follow-up.
- **No per-layer pipelining** (see Layer-by-layer vs batched above).
- **`get_finished` is per-request**, not per-block. Coarser than the
  vLLM equivalent; matches the scheduler's per-request
  `WAITING_FOR_REMOTE_KVS` granularity.

## Related guides

- [`docs/vllm_plugin_backend_guide.md`](vllm_plugin_backend_guide.md) —
  the ATOM-as-vLLM-plugin counterpart deployment form
- [`docs/scheduling_kv_cache_guide.md`](scheduling_kv_cache_guide.md) —
  BlockManager + scheduler internals this connector hooks into
- [`atom/kv_transfer/disaggregation/README.md`](../atom/kv_transfer/disaggregation/README.md) —
  PD-disaggregation (Mooncake/MoRIIO), the sibling connector family
- [`recipes/atom_vllm/LMCache-KV-Cache-Offload.md`](../recipes/atom_vllm/LMCache-KV-Cache-Offload.md) —
  the plugin-mode LMCache recipe this guide complements
