# aiter top-k `calc_grid_dim` spins forever above 1M context → PD pipeline deadlock

Status: **root-caused, fixed, repro'd both ways** (2026-08-08).

A deterministic host-side infinite loop in the aiter per-row top-k grid-size
search. On a 256-CU MI355X it triggers for any row length above **1,048,576**,
wedges the PP stage that hit it, and — because the PD consumer waits on all four
prefill stages — takes the whole disaggregated pipeline down with it. It is what
made `results/js_c16_m24_r14` abort with 6/52 failed requests.

## How it presents

Nothing looks like a crash. `curl /health` keeps returning 200 on both nodes.

| where | symptom |
|---|---|
| prefill log | last `Scheduled prefill batch` at 03:27:28, then only `Request N arrived` for 30 min |
| decode log | `PD backpressure: parked=1, waiting=1, running=0, resident=1/24` — 55,653 times |
| decode pools | index_cache **3%** used, 254,633 pages free, `running=0` — completely idle |
| mesh | 30 minutes later, `status_code=503 latency=1800411662` (its 1800 s timeout) |
| aiperf | `6/52 profiling requests failed (11.5%)`, `ProfileAborted` |

The decode side is a red herring: it is idle and healthy, just waiting for a KV
transfer that can never complete.

## The chain

`[PRODUCER]` accounting for the request that wedged it (req 223) versus a
healthy neighbour (req 222):

| line | req 222 | req 223 |
|---|---|---|
| `Received write_request` | 16 | **12** |
| `_execute_transfer` | 16 | **12** |
| `block RDMA write` | 16 | **12** |
| `write-done sent` | 16 | **12** |
| `served 4 consumers … awaiting release` | 4 | **3** |

12 = 3 PP stages × 4 decode ranks. **PP stage 0 never even received the
write_request** — its process was already stuck, so its handshake listener never
ran. Stages 1-3 sat in "awaiting release" holding the shared page table, the
decode never reached 4/4 write-dones, and its request stayed parked forever.

Stage 0 hung during the batch immediately before:

```
Scheduled prefill batch: 2 reqs, 3842 new tokens
  (cached: [464224, 663853], new: [659, 3183]), req_ids: (226, 227)
```

## Localizing it

Stage 0's `EngineCore_PP0` and its `ATOM::TP0` were still alive 30 minutes
later, spinning at 131% CPU. `py-spy dump` on the worker:

```
top_k_per_row_prefill (aiter/ops/topk.py:361)
sparse_attn_indexer (deepseek_v2.py:1552)
```

`rocgdb -p <tp0> -ex "info dispatches"` returned **`No dispatches are currently
active`** — nothing was running on the GPU at all — while the PC sat in:

```
#0 aiter::mb::calc_grid_dim<float, int, 11, 1024, false, Phase(0)>(int, int, int)
#1 aiter::mb::standalone_stable_radix_topk<float, int, false, true, Phase(0)>(...)
#2 topk_mb_workspace_size(long, long, long, bool)
```

Two PC samples minutes apart differed (`…d12` → `…d49`), so it was spinning, not
blocked. `topk_mb_workspace_size` is a pure host-side sizing query — no kernel,
no stream, nothing to deadlock against. The loop itself was the bug.

## Root cause

`csrc/kernels/topk_per_row_kernels.cu`, the `calc_grid_dim` overload that
carries the resident-capacity cap (aiter `28cb66b`, "cap persistent radix grid
at resident block capacity"):

```c
const IdxT max_num_blocks =
    std::min(ceildiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize),
             max_resident_blocks);
for(int num_waves = 1;; ++num_waves)
{
    IdxT num_blocks = std::min(max_num_blocks, max(num_waves*active_blocks/batch_size, 1));
    IdxT items_per_thread = ceildiv<IdxT>(len, num_blocks * BlockSize);
    items_per_thread      = alignTo<IdxT>(items_per_thread, VECTORIZED_READ_SIZE / sizeof(T));
    num_blocks            = ceildiv<IdxT>(len, items_per_thread * BlockSize);   // <- recomputed
    ...
    if(tail_wave_penalty < 0.15) break;
    if(num_blocks == max_num_blocks) break;        // <- the only unconditional exit
}
```

`items_per_thread` is rounded **up**, so the recomputed `num_blocks` is always
`<=` the count the search asked for. Before the cap, `max_num_blocks` was
exactly `ceildiv(len, VEC/sizeof(T) * BlockSize)` — the value the round-trip
reproduces — so the exit test could fire. The cap breaks that identity: once
`max_resident_blocks` is the binding bound, the round-trip lands strictly below
it, `num_blocks == max_num_blocks` is never true, the penalty is frozen at
whatever the saturated block count gives, and `num_waves` counts up forever.

The cap binds exactly when

```
len > VECTORIZED_READ_SIZE / sizeof(T) * BlockSize * active_blocks
    = 4 * 1024 * (occupancy 1 × 256 CU)
    = 1,048,576          on MI355X, float, BlockSize 1024
```

and it hangs whenever the frozen penalty also happens to be `>= 0.15`.

Confirmed directly — `aiter.topk_mb_workspace_size(rows, stride, 2048, False)`:

| stride | before | after |
|---|---|---|
| ≤ 1,048,576 | returns | returns (identical sizes) |
| 1,100,000 | **hangs** | returns |
| 1,500,000 / 2,000,000 | **hangs** | returns |

The second `calc_grid_dim` overload further down the file keeps the uncapped,
self-consistent bound and is not affected.

## Fix

End the search on saturation of the **requested** count rather than the
recomputed one — applied by `.claude/scratch/port_sparsekv_aiter.py`
(`patch_topk_grid_dim`, idempotent, re-applied on every server start alongside
the existing `__threadfence` fix):

```c
const IdxT requested_blocks = std::min(max_num_blocks, ...);
IdxT num_blocks = requested_blocks;
...
if(requested_blocks == max_num_blocks) break;
```

Where the old code terminated it terminates on the same iteration with the same
answer: without the cap, `requested_blocks == max_num_blocks` implies
`num_blocks == max_num_blocks`. Verified no change in returned workspace sizes
across 39 shapes.

The prebuilt `aiter/jit/module_top_k_per_row.so` must be deleted so the JIT
rebuilds (backup kept at `/root/module_top_k_per_row.so.bak`).

## Why it took a 1M-token prompt to show up

The trace corpus tops out around 700K input tokens per request, below the
threshold on its own. `stride0` is the top-k row stride for the whole scheduled
batch, and the batch that wedged stage 0 held two requests at 464,883 and
667,036 context. The decode node caps context at `--max-model-len 1048576`;
the prefill node has no such cap, so it is the side that crosses the line.
