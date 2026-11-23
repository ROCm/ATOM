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
2026-11-25 alignment uses:

| Repository | Source | Commit |
| --- | --- | --- |
| ATOM | `main` | `031d4da2671e72b724b41111edf979ce62e615c0` |
| MORI | [#1074](https://github.com/ROCm/mori/pull/1074) | `36c430bab74b724843f78a4fbbec76c349155b93` |

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

Use the existing ATOM model/455 options, adding this connector configuration:

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

## Completion and shutdown

The scheduler reserves destination slots before submitting a background fetch.
MORI returns a completed continuous prefix; ATOM publishes those blocks and
releases the rest. A drained failed copy can return an empty prefix.

An exception or an invalid completion reply is treated as unknown completion
in fabric mode. ATOM keeps the destination slots held, does not publish them,
and stops serving instead of turning the error into a cache miss. This is
deliberately conservative, including RPC failures where the reply was lost.

The UMBP helper retains the KV tensor and its allocator before sending Attach.
On normal shutdown, the connector stops its progress reporter and calls the
helper's `close()` before the runner releases KV views. The helper synchronizes
owner GPU work and requires a confirmed Detach after remote reads and local
writes have drained. An unconfirmed close raises `L1MemoryInDoubtError` and
keeps the allocation retained inside UMBP; it does not authorize freeing or
restarting that owner.

Normal shutdown requires coordinated drain: stop admission on both trays and
wait for requests, peer reads, and owner writes to finish. Keep the L1 services
and peer owners alive until each pool's Detach is confirmed; only then stop
services or containers. A timeout or worker exit is not a completion proof.

The current ATOM process controller does not guarantee that an owner remains
alive after a failed or timed-out Detach: exception and shutdown-timeout paths
can still terminate workers. The helper's Python references protect memory
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

The 2026-11-25 alignment to `main` passed 414 pool, connector, scheduler,
lifecycle, multi-connector and DP-routing tests, with one intentional native
fault-injection skip. The native tests used a fresh `mori_umbp_l1` build from
MORI `36c430ba`. GPU transfers and model inference were not rerun for this
alignment; the results below describe the earlier PR revision.

The 2026-11-23 run on g13-12 ↔ g13-10 used ATOM `3e37722ad` and MORI
`965a6061`. It passed 134 ATOM
pool/connector/lifecycle tests (one intentional native fault-injection skip).
The capacity regression exercises real startup prompt limits and scheduler
watermarks for full sharing, partial sharing and a private reserved null block.
The matching MORI build passed 57 C++ tests, copy planner, 19 keep-copies peer
tests and 12 Python helper tests.

Real GPUs passed all 32 directed paths (160 cases), both import orders and
16 GPU Event/Commit cases. DeepSeek-V4.1-Flash Fetch-off/on on the same consumer
matched inputs, complete output text and token counts for all 64 requests per
arm including warmup. All 16 measured remote hits confirmed 96 slots /
18,309,120 bytes copied each; true misses and local-hot requests had no remote
copy. Each node ran attention DP4/TP1 and MoE EP4, without cross-node TP.
Baseline, shared consumer and publisher each confirmed normal Detach.

The test image `rocm/fw-bringup-mlperf:gfx1250-atom-20261116` needed the two
Python modules from AITER #7416 (`11db23cc1d6f`) and the capability-query patch
from #8086 (`ff2f0ce274e7`) for this newer ATOM base. Its existing gfx1250
kernels were preserved. This greedy workload does not validate fused stochastic
sampling, which requires additional gfx1250 configuration in that image.

The earlier 2026-11-22 bases also passed slot reuse and a ten-minute bidirectional
soak. These are bounded workload and normal-shutdown results, not full model
quality or crash-recovery guarantees. An earlier GSM8K truncation remains
unresolved and also reproduced with Fetch disabled. Saturated model throughput
and topologies beyond the tested two trays have not been established.
