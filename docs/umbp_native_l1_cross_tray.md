# Native L1 sharing across trays

This opt-in path keeps ATOM's native paged KV allocation and slot management,
using the MORI native L1 extension based on PR #1045. It imports the peer's
FABRIC-exportable HBM and copies directly into ATOM's held destination slots.
There is no host/device staging buffer, SSD, or lower-tier KV pool in this path.
Attention reads the completed local KV bytes as before.

The `mori.umbp_l1.NativeL1Memory` helper runs inside the ATOM worker and returns
the final KV tensor. UMBP owns FABRIC allocation, descriptor export,
registration, and safe detach. ATOM's connector supplies KV size/device/layout,
reports completed owner writes, and prevents slot reuse on unknown fetch
completion; it does not implement the FABRIC memory lifecycle.

## Dependencies

The 2026-11-23 alignment uses these upstream PR heads:

| Repository | Base PR | Commit |
| --- | --- | --- |
| ATOM | [#3482](https://github.com/ROCm/ATOM/pull/3482) | `db27ec2198711f2d77dafff8ed0d5621af59d9fb` |
| MORI | [#1045](https://github.com/ROCm/mori/pull/1045) | `82996281c5800ec63541cff7c793e7f86fc9907c` |

Both PRs were open at alignment time. This ATOM change builds on #3482;
FABRIC runtime additionally requires the companion native L1 cross-tray change
on top of MORI #1045 (tested MORI commit
`13847e7eabde85dfa9dd6e817eaf2c114a21721a`). **MORI #1045 alone is insufficient.** Install the companion
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
There is no automatic reclamation, so plan a coordinated drain and service
restart before exhausting it.

## Validation scope

The earlier implementation (ATOM base `bd897325f`, MORI base `b946dd65`) was
validated on 2026-11-22 between g13-12 and g13-10, with four gfx1250 GPUs per tray
in one active fabric domain:

- CPU pool/lifecycle tests cover confirmed partial prefixes, unknown-Fetch
  quarantine, GPU-event progress ordering, and failed Commit retries. Follow-up
  CPU regressions also cover IPC/FABRIC reporter shutdown, repeated close, and
  preservation of adapter state after an unconfirmed Detach. See
  [ATOM lifecycle tests](../tests/test_umbp_fabric_lifecycle.py).
- GPU tests passed all 32 directed device paths with strong byte patterns,
  guards and final tensor-pointer checks, plus real GPU Event/Commit ordering,
  slot reuse, and a ten-minute, four-pair bidirectional soak. Owners completed
  normal Detach before services stopped.
- DeepSeek-V4.1-Flash ran with Fetch disabled/enabled on the same consumer:
  64 requests per arm including warmup had matching inputs, output text and
  output tokens. The 16 measured remote-hit requests each confirmed actual KV
  copying; true misses and local-hot requests had no remote copy.

After alignment to the PR heads above, a fresh native extension/kernel build,
ATOM's four pool/connector/lifecycle suites (128 passed, one intentional native
fault-injection skip), MORI's pool/peer/RPC suites (56 passed), copy planner,
and Python helper suite (12 passed) succeeded. All 18 peer tests also passed
with `UMBP_L1_KEEP_COPIES=1`. The new regressions cover Fetch completion across
cache clears, IPC Detach behavior, and remote read pins in the kept eviction
list. GPU transfers and model inference were not rerun on these updated heads.

These are bounded workload and normal-shutdown results, not full model-quality
or crash-recovery guarantees. An earlier GSM8K truncation remains unresolved
and also reproduced with Fetch disabled. Saturated model throughput and
topologies beyond the tested two trays have not been established.
