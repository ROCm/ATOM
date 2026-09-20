# KV cache routing rollout

See the [validation report](kv_cache_routing_validation.md) for image versions,
test results, the GLM routing smoke and outstanding performance acceptance.

The native PAGE namespace is version 4. It includes the complete PP layer
partition, TP size, DCP size and interleave, effective index dtype, HF cache
geometry and speculative configuration. Existing version 3 CPU/disk objects
remain isolated and are cold misses after upgrading.

Both frontends expose `/server_info`; the Python frontend also exposes
`/kv_transfer_info`. These include PP/TP/DCP geometry and the global hash span.
Placement is optional in `kv_transfer_config.routing_topology`:

```json
{
  "execution_id": "prefill-a/dp-0",
  "ranks": [
    {"pp": 0, "tp": 0, "host_id": "host-a", "device_id": 0},
    {"pp": 1, "tp": 0, "host_id": "host-a", "device_id": 1},
    {"pp": 2, "tp": 0, "host_id": "host-b", "device_id": 0},
    {"pp": 3, "tp": 0, "host_id": "host-b", "device_id": 1}
  ],
  "cache_bindings": [
    {"storage_domain_id": "native-prefill-a", "rank_set": [[0, 0], [1, 0], [2, 0], [3, 0]]}
  ]
}
```

Every PP x TP rank must appear exactly once when placement is supplied. DCP
reuses TP GPUs: TP4/DCP2 has token shards 0,1,0,1. Host placement is independent
of byte compatibility. The connector handshake still determines actual transfer
support and reachability; a placement manifest does not enable a new layout
conversion. P and D may have different physical namespaces.

For GLM-5.2, keep FP8 index cache and DCP interleave 1. Preserve the exact PP
partition used for each deployment (`20,20,20,18` and `18,20,20,20` differ).
Correctness runs must disable forced MTP acceptance.

## Enable native HBM + CPU observation

This milestone supports one dense, prefix-cache-enabled execution per serving
endpoint. PP and TP/DCP workers collectively own that execution. Multi-DP
endpoints, recurrent-state models, sliding-window state, multimodal identities,
LoRA and cache salt do not supply exact reuse in this version. Unsupported
sources or failed observation use the existing load policy.

Use a runtime containing the LMCache native `residency_snapshot` and
`residency_events` APIs. HBM observation does not require CPU offload. CPU
observation uses the existing non-MP `lmcache_offload` connector and requires
all workers' CPU objects to be readable. It never calls lookup, pins objects or
creates another KV data store.

Each execution needs a unique ID and Catalog listen address. The same JSON must
reach its scheduler and every PP/TP worker. Use immutable weight, tokenizer and
template revisions; local paths and mutable model names are insufficient.

```bash
export ATOM_CACHE_ROUTING_CONFIG='{
  "execution_id": "p-a",
  "catalog_url": "http://127.0.0.1:18610",
  "namespace_manifest": {
    "model_revision": "weights-sha256:REPLACE",
    "tokenizer_revision": "tokenizer-sha256:REPLACE",
    "template_revision": "template-sha256:REPLACE",
    "kv_semantics": "glm52-fp8-index-fp8-mtp1",
    "adapter_revision": null,
    "cache_salt": null,
    "multimodal_identity": null
  },
  "canonical_block_size": 16,
  "max_entries": 200000,
  "max_log_bytes": 16777216,
  "stale_seconds": 3.0
}'
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=2
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MIN_LOAD_TOKENS=0
```

CPU memory is allocated **per worker**. For example, 2 GiB on each of four PP
workers consumes up to 8 GiB, separate from catalog metadata and GPU memory.
The canonical block must divide the HBM hash span and native CPU chunk size.
Use the same semantic namespace on compatible P/D executions, even when their
physical layouts differ. With observation enabled, execution identity also
isolates native lookup IPC sockets for multiple instances on one host.

`catalog_url` must be reachable from workers and routers. On multiple hosts,
replace loopback with the scheduler's reachable address. The metadata service
is an internal HTTP control plane; expose it on the deployment's internal
network. Tensor transfer remains on the existing connector data path.

Both frontends expose:

- `/server_info` for discovery.
- POST `/v1/completions/render` (one result per prompt) and
  `/v1/chat/completions/render` (one object) for exact token IDs using the same
  template preparation as inference. The pair policy supports one prompt.
- `/v1/cache/info`, `/load`, `/snapshot`, `/events` for observation and recovery.

The scheduler Catalog also accepts internal worker POST `/v1/cache/cpu`.
`info` includes actual native codec plane manifests, indexed by worker, after
CPU registration. Index and MTP cache planes are included in their stored order.
A missing PP stage or DCP shard, eviction, worker restart, report gap or stale
source revokes the affected CPU benefit independently of HBM residency.

## Recovery and budgets

This first delivery uses HTTP polling with versioned replay and snapshot
recovery. It does not add the design's optional ZMQ transport or llm-d exporter.
Existing ATOM KV event publication continues to work independently.

A source has a boot epoch and monotonically increasing decimal-string cursor.
Snapshot pages share one immutable cut and must be followed by replay from
`cut_seq`. An expired cursor or snapshot returns HTTP 410. Snapshot leases last
10 seconds, have at most two concurrent readers and are released on the final
page; log overflow invalidates unrecoverable leases. Native CPU recovery clears
one worker scope and rebuilds it in pages of at most 256 objects / approximately
1 MiB. Partial recovery underestimates residency.

The router polls asynchronously every 100 ms, concurrently across workers. A
failed refresh immediately removes positive cache credit. Sources and load
samples older than 3 seconds cannot score; a failed replay requires a complete
snapshot. Memory is capped at 200,000 entries per router source, and HTTP pages
at 16 MiB. These defaults are conservative bounds, not a claim of meeting the
design's sub-100-ms propagation or 1% overhead performance targets.

## Calibrate and run Atomesh

Use `--backend atom --pd-disaggregation --policy kv_cache_aware` with arbitrary
numbers of `--prefill` and `--decode` endpoints. Selection checks model,
namespace, layout, verified transfer path, available capacity and idx2idx rank
constraints before scoring P/D pairs. CPU benefits apply only to P prefill;
D HBM can reduce transfer. D CPU preload remains a later milestone.

Costs must be measured for the deployed execution and link. Missing calibration
or a request outside the measured range falls back to load routing. Discover
each current `layout_id` through `/v1/cache/info`, measure context-dependent
prefill work, H2D, decode step and each legal P-to-D transfer, then provide a JSON
file through `ATOM_CACHE_ROUTING_CALIBRATION`. Restart only the router to reload
it. Each entry is bound to an execution ID and physical layout. The engine may
alternatively publish these fields in `kv_transfer_config.routing_topology`.

The following illustrates the schema; **the numbers are examples, not GLM
measurements**. Points are `[tokens, elapsed_ms]`, monotonic, with linear
interpolation inside the measured range. Prefill curves are selected by the
smallest `max_context_tokens` covering the request.

```json
{
  "executions": {
    "p-a": {
      "layout_id": "COPY-P-LAYOUT-FROM-INFO",
      "costs": {
        "prefill_curves": [{"max_context_tokens": 4096,
                            "points": [[256, 5.0], [4096, 70.0]]}],
        "h2d_curve": [[256, 0.8], [4096, 6.0]]
      },
      "transfer_paths": [{
        "verified": true,
        "protocol": "mooncake",
        "destination_execution_id": "d-a",
        "source_layout_id": "COPY-P-LAYOUT-FROM-INFO",
        "destination_layout_id": "COPY-D-LAYOUT-FROM-INFO",
        "transfer_curve": [[256, 1.0], [4096, 9.0]]
      }]
    },
    "d-a": {
      "layout_id": "COPY-D-LAYOUT-FROM-INFO",
      "costs": {"decode_step_ms": 4.0}
    }
  }
}
```

Set `verified` only after executing that layout/connector pair successfully.
Placement metadata alone does not demonstrate reachability. Keep independent
calibration for distinct P/D pairs. At nonzero CPU copy load, the source must
supply `copy_queue_ms` or the router declines CPU credit.

```bash
export ATOM_CACHE_ROUTING_CALIBRATION=/configs/cache-costs.json
atomesh --backend atom --pd-disaggregation --policy kv_cache_aware \
  --prefill http://127.0.0.1:18510 --prefill http://127.0.0.1:18511 \
  --decode http://127.0.0.1:18520 --decode http://127.0.0.1:18521 \
  --host 0.0.0.0 --port 18000
```

The router renders once before retries and uses a local metadata index for
selection. It sends an `auto` or `skip` CPU-load hint to P and `skip` to D.
The native connector rechecks actual availability and retains its existing
failure/recompute behavior. Dispatch reservations include unsampled queued work
and capacity; sampled accepted dispatch IDs reconcile them. Response completion,
cancellation or failure releases the reservation, with a 30-second recovery TTL.

Disable `ATOM_CACHE_ROUTING_CONFIG` to remove observation and select another
router policy to revert routing. Existing v3 native objects remain separate
from v4; changing a configured semantic revision creates a new native namespace.
