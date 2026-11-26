# Native L1 sharing across trays

This opt-in path keeps ATOM's native paged KV allocation and slot management,
using the MORI native L1 extension in PR #1074. It imports the peer's
FABRIC-exportable HBM and copies directly into ATOM's held destination slots.
There is no host/device staging buffer, SSD, or lower-tier KV pool in this path.
Attention reads the completed local KV bytes as before.

The `mori.umbp_l1.NativeL1Memory` helper runs inside the ATOM worker and returns
the final KV tensor. UMBP owns FABRIC allocation, descriptor export,
registration, and safe detach. ATOM's connector supplies KV size/device/layout,
reports completed owner writes, and prevents slot reuse on unknown fetch
completion; it does not implement the FABRIC memory lifecycle.

## Dependencies

This change is based directly on ATOM `main`; the base connector from
[#3482](https://github.com/ROCm/ATOM/pull/3482) is already merged. The
2026-11-25 integrated validation used:

| Repository | Source | Commit |
| --- | --- | --- |
| ATOM base | `main` | `031d4da2671e72b724b41111edf979ce62e615c0` |
| ATOM adapter | [#3828](https://github.com/ROCm/ATOM/pull/3828) | `e1178843de1bb919527145075167b48d36059105` |
| MORI native L1 | [#1074](https://github.com/ROCm/mori/pull/1074) | `ef41bf906732f5ee9e1b2d45ea4d5ced299f6d2d` |

MORI documentation snapshot `256630648a6fb37124edc7e31a99222003d13396`
contains the same validated production implementation, with subsequent changes
limited to deployment and validation documentation. These results do not cover
a later ATOM `main` merge or subsequent adapter changes.

FABRIC runtime requires the companion MORI #1074 native L1 cross-tray change.
**MORI #1045 alone is insufficient.** Install the companion
Python helper and compiled extension together; startup requires
`NATIVE_L1_FABRIC_VERSION=1` and `export_fabric_memory`. No merge of #1024 or the
general KV-sharing branch is required. IPC remains the default.

## Configuration

Both trays need the matching native L1 MORI build, including:

- the MORI Python package providing `mori.umbp_l1.NativeL1Memory` and its
  matching native `mori_umbp_l1` extension; UMBP's compatibility check rejects
  missing FABRIC export/version support at connector startup;
- `mori.io.make_fabric_mem_pool`, its compiled allocator library, and a PyTorch
  version supporting `torch.cuda.MemPool`/`torch.cuda.use_mem_pool`;
- fabric-capable GPUs in the same supported fabric domain;
- one local native L1 service per tray with the other service configured as a
  static peer. Use MORI's `tools/umbp_l1_fabric_server.py`, backed by
  `mori_umbp_l1.start_fabric_server(address, peers, kernel_path)`, and the compiled
  FABRIC copy kernel. This is a pure L1 service; no standalone tier server is
  needed. The gRPC endpoint is control traffic; KV bytes go through FABRIC.

Use the existing ATOM model and parallelism options, adding this connector configuration:

```json
{
  "kv_connector": "umbp_pool",
  "kv_connector_extra_config": {
    "address": "127.0.0.1:29620",
    "namespace": "atom",
    "transport": "fabric",
    "node_id": "tray-a",
    "group": "deepseek-v41-flash-revision-kvformat-v1",
    "share_ratio": 1.0,
    "fetch": true
  }
}
```

Bind each L1 service to a peer-reachable network address; ATOM can connect through
loopback on the same port. On the second tray, use `node_id: "tray-b"` and the
same `group`. Each engine on one tray must also have a distinct namespace.
Pool IDs include node, namespace, DP rank, and TP rank. An explicit group is
mandatory: operators must use the
same model revision, tokenizer/hash conventions, KV/index dtypes, block size,
layer/shard layout, and speculative configuration on its members. Matching
region sizes alone does not prove these semantic properties. The group is used
verbatim; it does not automatically distinguish TP shards. Put only replicas
of the same shard in one group, even when different shards have identical sizes.
`fetch: false` is also valid: the engine contributes exportable KV while
disabling its own peer fetches.

The UMBP helper scopes only KV allocation to the fabric allocator; other torch
allocations keep their usual allocator. An in-place elementwise operation
zero-initializes this arena directly, preserving ATOM's allocation contract
without relying on FABRIC `hipMemset` support. A missing allocator/export
capability fails startup, with no fallback to staging or ordinary HIP IPC.
Existing local deployments keep the default `transport: "ipc"` behavior. Native L1
restrictions remain, including one runner per engine rank and attention TP=1;
attention tensor parallelism within a rank is unsupported.

The FABRIC arena uses `torch.empty` inside a scoped PyTorch `MemPool` backed by
MORI's pluggable allocator; it does not replace the process-global allocator.
The campaign's PyTorch source revision
[`c9ab2693`](https://github.com/pytorch/pytorch/blob/c9ab269368d46a826d64705192fef55594c8b698/c10/cuda/CUDACachingAllocator.cpp#L1832)
records these allocations in the caching allocator's aggregate allocated-byte
statistics. This supports keeping `counted_by_torch = True` for the runner's
allocation-mismatch diagnostic. The dedicated 2026-11-26 measurement confirmed
that `torch.cuda.memory_stats()["allocated_bytes.all.current"]` increased by
exactly 67,108,864 bytes for each 64 MiB IPC and FABRIC allocation on both
j07-16 and j07-14. All four cases passed with zero accounting error on the
tested PyTorch runtime.

On an idle FABRIC-capable GPU, with the matching MORI package and extension,
run the opt-in accounting regression:

```bash
RUN_UMBP_FABRIC_GPU_TESTS=1 python -m pytest -q -s \
  tests/test_umbp_fabric_memory_gpu.py
```

It compares ordinary IPC and FABRIC allocations of 64 MiB after warming the
zeroing paths, records the runtime and before/after counters, and checks the
runner's 3% tolerance. It requires no L1 service or peer and skips by default.

The test image required `LD_PRELOAD=/opt/rocm/lib/libamd_smi.so.27` to align
AMD-SMI loading with the host library. Before this environment correction,
AMD-SMI and PyTorch reported zero GPUs while HIP reported four. Afterwards all
three agreed, and each owner retained a single SDK core HIP/HSA runtime. This
is an environment requirement of that image, not an override of device counts.

## Completion and shutdown

The scheduler reserves destination slots before submitting a background fetch.
MORI returns a completed continuous prefix; ATOM publishes those blocks and
releases the rest. A drained failed copy can return an empty prefix.

An exception or an invalid completion reply is treated as unknown completion
in fabric mode. ATOM keeps the destination slots held, does not publish them,
and stops serving instead of turning the error into a cache miss. This is
deliberately conservative, including RPC failures where the reply was lost.

The UMBP helper retains the KV tensor and its allocator before sending Attach.
On normal shutdown, the connector stops and joins its progress reporter before
calling the helper's `close()`, while the runner still retains its KV views.
FABRIC must wait for an in-flight Commit to finish before closing native memory.
MORI's `L1Client::Commit()` sets a 120-second gRPC deadline, so that RPC is
bounded; this is not a bound on the entire shutdown, which also includes GPU
synchronization and Detach. A short reporter join followed by arena release
would not establish safe completion. The helper synchronizes
owner GPU work and requires a confirmed Detach after remote reads and local
writes have drained. An unconfirmed close raises `L1MemoryInDoubtError` and
keeps the allocation retained inside UMBP; it does not authorize freeing or
restarting that owner.

Normal shutdown requires coordinated drain: stop admission on both trays and
wait for requests, peer reads, and owner writes to finish. Keep the L1 services
and peer owners alive until each pool's Detach is confirmed; only then stop
services or containers. A timeout or worker exit is not a completion proof.

The current ATOM process controller uses `ATOM_SHUTDOWN_TIMEOUT_S` (default
5 seconds) as a shared worker-exit deadline, after which it can terminate
workers. This is shorter than an in-flight Commit's RPC deadline. Configure
enough time for a coordinated normal drain, but increasing the timeout does not
guarantee that an owner remains alive after a failed or timed-out Detach:
exception and shutdown-timeout paths can still terminate workers. The helper's
Python references protect memory
only while its process lives. There is no automatic recovery from unknown
completion, owner death, or an uncoordinated service restart; the coordinator
must preserve affected owners and drain peers before teardown.

MORI currently retains up to 65,536 peer operation receipts/tombstones per
`PoolManager` lifetime, shared across its pools. Release and Detach/reattach do
not reset this cumulative budget; exhaustion rejects new remote read grants.
Each remote read window of up to 128 keys consumes one receipt.
There is no automatic reclamation, so plan a coordinated drain and service
restart before exhausting it.

## Validation scope

The completed 2026-11-25 campaign on j07-16 ↔ j07-14 used the exact ATOM
`e1178843` and MORI `ef41bf90` revisions above. The
[published MORI validation record](https://github.com/ROCm/mori/blob/256630648a6fb37124edc7e31a99222003d13396/docs/umbp/native-l1-cross-tray.md#ownership-and-shutdown)
summarizes the cross-tray results:

| Layer | Result |
| --- | --- |
| ATOM pool/connector/scheduler/lifecycle/DP tests | 471 passed, two existing skips; dedicated UMBP scheduler and native lifecycle tests ran |
| MORI native tests | 81 C++ cases with default `KEEP_COPIES=1`, 69 pool/peer cases with `KEEP_COPIES=0`, 12 Python helper cases, and copy-planner checks passed |
| GPU import orders | Native-first and IO-first both passed using one compatible HIP runtime |
| Native cross-tray copies | 32 directed GPU paths, 128 full-arena byte comparisons plus three smokes, 52 confirmed Detaches, and 28 owner processes exited successfully |
| Real ATOM adapter | Eight bidirectional GPU paths, 16 full/partial prefix checks, Event/Commit, payload/suffix/guards, source-arena checks, physical slot reuse, and 16 confirmed Detaches |
| Real model | 11 HTTP requests; three remotely reused prefixes matched local reuse, and one unseeded miss matched a cold request |

Native coverage included prefixes crossing the 128-key window, reversed
destination slots, the full 844-slot arena, and a missing key within a prefix.
The ATOM adapter copied 4,000 slots / 16,532,000 bytes in total. The two ATOM
skips were the intentionally fake-only native copy-failure injection and an
existing scheduler test module with an obsolete import path.

The model used DeepSeek-V4.1-Flash, attention DP4/TP1, MoE EP4, eager execution,
BF16 KV, FP8 index, and block size 128. Each tray ran its own model replica;
TP/EP collectives were local. Model sharing was tested in one direction,
j07-14 → j07-16. Remote reuse of 512/1536/2304 cached tokens matched the
corresponding local reuse in output token IDs, full text, finish reason, and
usage. The unseeded DP3 request had zero remote copies and matched the source's
cold request. Remote model copies totaled 34 pages / 12,577,280 bytes.
All eight model owners confirmed Detach and empty service stats.

The model comparison preserves ATOM's existing bounded-replay semantics;
prefix reuse can differ from cold prefill. These finite correctness checks do
not establish model quality, TTFT, throughput, physical link bandwidth,
saturation, long-term stability, crash recovery, or additional topologies.
Existing AITER destructor exceptions were retained in the logs, alongside
independently confirmed L1 Detach and resource cleanup.

The 2026-11-26 review follow-up tested a frozen working tree based on ATOM
`e1178843`, with local adapter comments, allocation/layout and shutdown
regressions, and documentation corrections. Its CPU suite passed 454 cases;
two opt-in GPU cases and one existing fake-only copy-failure case were skipped
in that CPU run. The two GPU cases were then run on each tray and all four
passed, as described in the accounting section above.

The real ATOM adapter regression also passed all eight bidirectional GPU
paths, 16 full/partial prefix and complete-arena guard checks, eight real
Event/Commit fences, clear/slot reuse, and 16 confirmed Detaches. It copied
4,000 slots / 16,532,000 bytes. Raw owner replies, the loaded adapter/helper/
extension hashes, and service copy counters were independently checked. All
16 owners exited successfully; all 16 services exited and both test ports
were clear before the dedicated empty containers were stopped.

This follow-up validates allocation accounting and adapter functionality.
It did not repeat the model or native GPU matrix from the earlier campaign,
and the saved load baseline was busy. It does not establish performance or
bandwidth under idle conditions.
