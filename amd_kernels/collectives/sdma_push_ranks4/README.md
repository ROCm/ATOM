# Four-rank pushes on the SDMA engines, ordered by the GPU stream

## What it does

`SdmaPush` moves bytes between four ranks of one node with the copy (SDMA) engines instead of CU kernels: no
workgroup reads a peer or spins on a flag, so a collective can run next to another stream's or DBO microbatch's
compute without taking its CUs. The GPU orders it: the copies start once the caller's stream reaches the push, and
the receiver's stream waits with a command-processor wait. The host blocks only to recycle a signal-ring slot `RING`
(32) pushes later. Bytes are copied unchanged (no arithmetic, no `reference.py`).
Caller: `SdmaSeqParallelHandoffs` (`collectives/seq_parallel_handoffs_h6144_q2048_ranks4`), M3.1-Flash TP4 SP.

## Entry points

- `SdmaPush`: a collective on a gloo group; receive regions in one IPC pool the peers map, an uncached flag word per
  (region, writer); `region` gives a region's uint8 view.
- `SdmaPush.push`: copies items (peer, source address, offset in the peer's region, bytes) once the stream reaches the
  push, holding the given source tensors alive; returns the push's sequence number.
- `SdmaPush.wait`: the stream waits until every peer's push of that sequence into this rank's region has landed; by
  default it drops the held sources of every push but the latest.
- `SdmaPush.drain`: the host waits for every push's copies and drops the held sources.

The caller's part: every rank issues the same pushes and waits on a region in the same order. An engine reads a source
after the push's stream position, outside stream order, so keep sources alive (releasing on `wait` is right when a peer
pushes again only after its wait for this rank's push). The next push into a region overwrites it: its consumer must
run before this rank's next push that every peer waits for, so two pushes into one region need a phase in between.
Eager only (stream memory operations, host-side sequence state; no graph capture). Nothing compiles; the engine choice
and the per-push steps are in `transport.py`.

## Performance

vLLM nightly image: 143-155 GB/s out per GPU to 3 peers at 4096-24576 rows of 6 KB;
next to a bf16 GEMM loop 95-98% of the copy time is hidden, the GEMM +0.7-3.6%. `bench_engines.py` (3 x 64 MB per
rank at once): 1 / 2 / 4 engines per peer 155.8 / 137 / 165 GB/s, so the xGMI links bound it. Serving: M3.1-Flash
TP4, c64 AgentX render, lone fresh prefills, step time from the iteration log, DBO from 16384 tokens:

| tokens | DBO on the RCCL TP path (ms) | DBO on `SdmaSeqParallelHandoffs` (ms) |
|---:|---:|---:|
| 16384 | 302 | 218 |
| 24576 | 389 | 305 |

## Tests

```bash
python3 -m pytest tests/collectives/sdma_push_ranks4 -q                       # 4 GPUs
torchrun --nproc-per-node 4 benchmarks/collectives/sdma_push_ranks4/bench_engines.py 64
```

`test_transport.py`: gather / reduce-scatter-shaped pushes, 80 rounds, a slowed producer, sources dropped early.
