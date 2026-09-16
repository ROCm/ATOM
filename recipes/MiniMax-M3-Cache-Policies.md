# MiniMax-M3 CPU-offload cache policies

Improve prefix reuse under CPU-only offload without changing model weights,
KV precision, context length, or cache-hit accounting.

## Changes

- **All-rank lookup:** honor explicit lookup-worker configuration. Each TP
  rank updates its own cache recency and pins its KV shard; the minimum hit
  length across ranks is the common restorable prefix. This avoids trusting
  rank 0 when another rank has already evicted its shard.
- **Pin lifecycle:** release unused lookup pins on misses, errors, skipped
  loads, and cancellation, including aborts before HBM allocation. Retain
  pins for pending loads and avoid duplicate lookup ownership or idle-drain
  loops with no dispatchable work.
- **Optional SLRU:** new data enters probation; reused data enters protection.
  Evict probationary data first, demoting older protected entries when the
  limit is exceeded. HBM protection is capped at the configured fraction of
  total pool blocks; CPU protection targets half the resident chunks.
  Protected data remains evictable, but referenced/pinned data does not.
  Separate CPU queues avoid scanning the protected segment for every
  probationary eviction; pinned entries may still require scanning.

## Configuration

Set these before starting a TP4 server with LMCache offload enabled:

```bash
export ATOM_PREFIX_CACHE_POLICY=slru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
export LMCACHE_CACHE_POLICY=ATOM_SLRU
export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3
```

Without explicit settings, HBM/CPU retain LRU and lookup defaults to rank 0.
The shared pin-lifecycle fix applies independently of SLRU. Only synchronous
lookup is supported: enabling `LMCACHE_ENABLE_ASYNC_LOADING` is rejected at
startup.

## Results and validation

MiniMax-M3 FP4, TP4, FP8 KV, 48 concurrency, 1800s profiling,
256 GiB CPU cache per rank, no NVMe, and synthetic acceptance 0.5933.
Both runs enabled HBM/CPU SLRU and the pin fix; **only lookup scope changed**.

| Metric | Rank 0 only | All-rank |
|---|---:|---:|
| Total prompt cache hit rate | 82.86% | 95.48% |
| Total throughput (tokens/s) | 86,770.56 | 171,373.84 |
| Total throughput per GPU (tokens/s/GPU) | 21,692.64 | 42,843.46 |
| P90 ITL (ms) | 66.94 | 24.35 |

These are final AIPerf exports; the all-rank submission was valid. The
comparison demonstrates the lookup-scope benefit, **not SLRU's independent
contribution**. GPU measurements predate the rebase and review fixes and
have not been rerun.

**604 regression tests passed; 1 known environment-dependent test was
deselected** because it assumes LMCache is absent. Coverage includes cache
retention, pin safety, allocation-wait cancellation, async-mode rejection,
and bounded probationary-eviction scans.
