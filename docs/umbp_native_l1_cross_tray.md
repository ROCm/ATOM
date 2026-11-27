# Native L1 sharing across trays

This opt-in path imports a peer's FABRIC-exportable HBM and copies directly
into ATOM's held KV slots. Attention reads the completed local bytes. There
is no staging buffer, SSD, or lower-tier KV pool in this path.

The `mori.umbp_l1.NativeL1Memory` helper runs inside the ATOM worker and owns
allocation, export, registration, and safe detach. ATOM supplies the KV layout,
reports completed owner writes, and prevents slot reuse on unknown completion.

## Requirements

Install the matching MORI Python package and native extension from
[MORI #1074](https://github.com/ROCm/mori/pull/1074) before enabling FABRIC.
The base IPC connector alone is insufficient. The deployment needs:

- `mori.umbp_l1.NativeL1Memory` and `mori_umbp_l1` with
  `NATIVE_L1_FABRIC_VERSION=2`, FABRIC export, and the `commit(timeout_ms=...)`
  interface;
- MORI's FABRIC allocator library and a PyTorch version supporting
  `torch.cuda.MemPool` and `torch.cuda.use_mem_pool`;
- FABRIC-capable GPUs in the same supported fabric domain;
- one native L1 service per tray, with the other service configured as a
  static peer. Use MORI's `tools/umbp_l1_fabric_server.py` and its FABRIC copy
  kernel. The gRPC endpoint carries control traffic; KV bytes use FABRIC.

The connector supports DeepSeek-V4.1 with one runner per engine rank and
attention TP=1. Use data-parallel attention; TP/EP collectives for each model
replica remain local to its tray. IPC remains the default transport.

## Configuration

Add this configuration to the existing ATOM model and parallelism options:

```json
{
  "kv_connector": "umbp_pool",
  "kv_connector_extra_config": {
    "address": "10.0.0.10:29620",
    "namespace": "atom",
    "transport": "fabric",
    "node_id": "tray-a",
    "group": "deepseek-v41-flash-revision-1",
    "share_ratio": 1.0,
    "fetch": true
  }
}
```

Bind each L1 service to its peer-reachable internal address and configure its
local ATOM workers with that address. On tray B, use `address: "10.0.0.11:29620"`,
`node_id: "tray-b"`, and the same `group`. Replace the example addresses for your
deployment. Each engine sharing a service must have a distinct namespace. Pool
IDs include node, namespace, DP rank, and TP rank.

FABRIC requires an explicit group identifying the same model and draft-model
revisions and tokenizer/hash conventions. ATOM appends a digest of the resolved
KV/index dtypes, index layout, block size, online quantization, and speculative
configuration. Local model paths are excluded so identical checkpoints mounted
at different paths can share. The digest protects against format mismatches;
it does not compare checkpoint contents or prove weight equality. Operators
must use compatible model revisions and layer/shard layouts. MORI additionally
checks per-region block sizes. IPC's existing group behavior is unchanged.

`fetch: false` exports the engine's KV while disabling its own peer fetches.
`share_ratio` selects the fraction of slots included in the shared index;
the remaining slots are not indexed for sharing. This does not restrict a
trusted peer's descriptor access to the whole backing allocation.
The helper scopes only KV allocation to the FABRIC allocator
and zeroes it in place. Missing allocator/export support fails startup.

The FABRIC arena uses PyTorch's caching allocator through a scoped `MemPool`
and is counted in `allocated_bytes.all.current`. Check accounting on the
deployment runtime with an idle supported GPU:

```bash
python -m pytest -q -s tests/test_umbp_fabric_memory_gpu.py
```

This compares IPC and FABRIC allocations against the runner's 3% tolerance.
It needs no L1 service or peer; missing ROCm GPUs or optional native modules
are skipped.

## Completion and shutdown

The scheduler reserves destination slots before starting a background fetch.
MORI returns a completed continuous prefix; ATOM publishes those blocks and
releases the remaining reservations. A drained failed copy may return an
empty prefix. An exception or invalid completion reply is treated as unknown
completion in FABRIC mode: ATOM keeps the slots held and stops serving.
A gRPC `UNAVAILABLE` response does not prove that no copy was submitted.

Completed owner writes are reported through Commit. FABRIC retries a failed
Commit with interruptible backoff from 10 ms to 1 second, retaining only the
highest completed stamp. Each FABRIC Commit has a 1-second RPC deadline so
the reporter cannot delay shutdown for the client's default 120 seconds.
IPC retains its existing behavior: a new completed forward supplies the next
Commit after failure. A Commit timeout is not a DMA-completion guarantee.

The connector stops and joins the reporter before native memory close, while
the runner retains its KV views. The helper synchronizes owner GPU work and
requires a confirmed Detach after remote reads and local writes have drained.
An unconfirmed close raises `L1MemoryInDoubtError` and retains the arena and
allocator. Normal KV/builder teardown must not continue after that error.

For normal shutdown, stop admission on both trays and wait for requests, peer
reads, and owner writes to finish. Keep the L1 services and owners alive until
each pool's Detach is confirmed, then stop services or containers. Configure
`ATOM_SHUTDOWN_TIMEOUT_S` for the deployment's drain budget; for example,
`ATOM_SHUTDOWN_TIMEOUT_S=180` allows more time than the default 5 seconds for
GPU synchronization, Detach, and worker cleanup. This is a normal-shutdown
budget, not a safety guarantee after a failed Detach.

The current process controller can still terminate workers on errors or when
the shutdown deadline expires. Python references preserve allocations only
while their owner process lives. There is no automatic recovery from unknown
completion, owner death, or an uncoordinated service restart; preserving
affected owners and draining peers requires external coordination. Increasing
the timeout, cancelling an RPC, or observing worker exit does not prove that
remote operations have completed.
