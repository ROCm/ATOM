# Qwen3.5 LMCache offload on gfx1250

This recipe extends `rocm/fw-bringup-mlperf:gfx1250-atom-20261031-qwen35`
and its `/app/qwen35_work/README.md` c512 benchmark. It targets standalone
ATOM, Qwen3.5-35B-A3B-MXFP4, four gfx1250 GPUs, DP4 attention / EP4 MoE,
MTP-3, and BF16 KV. The ATOM base is `662a4f8f5071587b3190e5b21bb5592147f8b4d6`.

## Build

Mount this checkout at `/work/ATOM` and the model directory at `/data` in the
supplied image, with the devices, IPC and network options in the image README.
Inside that container:

```bash
bash /work/ATOM/recipes/mlperf_endpoints_qwen35/install_lmcache_gfx1250.sh
```

The installer builds LMCache commit
`f22dec2832124ea4ffeb464553b819a6369d4a30` with HIP for `gfx1250`, using the
image's ROCm and PyTorch ABI. The existing published gfx942/gfx950 wheel is
not used. It creates `/work/lmcache-venv` with system site packages and adds
the pinned dependencies in `lmcache-gfx1250-requirements.txt`. Torch, Triton,
NumPy and AITER remain those supplied by the image. This is a qualified
embedded CPU-offload environment, not a general installation of every
optional LMCache backend; the image's NumPy/setuptools versions also fall
outside LMCache's declared dependency ranges.

The build sets `BUILD_WITH_HIP=1`, `CXX=hipcc`,
`PYTORCH_ROCM_ARCH=gfx1250`, and `MAX_JOBS=8`. Its wheel is written under
`/work/wheels`. The installer checks both native-extension imports.

## Run

```bash
RUNS=/work/offload_c512 \
  bash /work/ATOM/recipes/mlperf_endpoints_qwen35/run_lmcache_c512.sh
```

The wrapper uses the image's c512 harness and workload, with `STOP_FIRST=0`
so completing a trajectory after exhausting the launch pool cannot end the
load before the 10-minute cap. The original recipe defaults to `STOP_FIRST=1`;
one trial consequently stopped issuing early. It adds `ATOM_KV_OFFLOAD=lmcache`
and these per-worker settings:

| Setting | Default | Purpose |
|---|---:|---|
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | 128 GiB | Shared pinned CPU pool for KV and state |
| `OFFLOAD_STATE_CPU_SIZE` | 80 GiB | Maximum state allocation within that pool |
| `OFFLOAD_STATE_GPU_SIZE` | 64 GiB | Soft budget for GPU checkpoint images |
| `LMCACHE_CHUNK_SIZE` | 1024 tokens | KV transfer granularity |
| `LMCACHE_ENABLE_ASYNC_LOADING` | False | Embedded synchronous lookup path |
| `LMCACHE_SAVE_DECODE_CACHE` | False | Store prompt prefixes |
| `PYTHONHASHSEED` | 0 | Stable Python hashing across processes |

The four CPU pools reserve approximately 512 GiB in total. Leave additional
RAM for the image, server, benchmark client and its tokenization/reporting
queues. CPU state capacity is part of that 512 GiB, not an additional pool.
No local-disk tier is enabled for this benchmark.

`RUNS`, `CAP_MIN`, `STOP_FIRST`, `LMCACHE_VENV` and the numeric cache settings can be
overridden. The DP prefill balance weight remains 28000 and decode interval
remains 16, matching the supplied baseline. `OFFLOAD_STATE_GPU_SIZE=0`
disables the new GPU checkpoint retention policy.

## Why bound GPU checkpoints

Qwen3.5 prefixes require both full-attention KV and the recurrent GDN state
at a matching boundary. Its state images are about 32.81 MiB each in this
configuration. Ordinary checkpoint reclamation waits for the block pool's
free list to empty, but that list includes reusable KV. Consequently, state
snapshots can displace KV even after LMCache has stored their bytes.

The new optional policy releases old GPU checkpoint images after successful
offload, while retaining their CPU index and KV blocks. It uses existing
pin and READY-state rules: it never reclaims a GPU writer, restore source,
or active offload source. CPU misses retain the existing recompute fallback.

## Validation and measurement

The image's smoke/tool-call and repeated greedy-prefix checks run before
each benchmark. Relevant regression checks are:

```bash
cd /work/ATOM
OMP_NUM_THREADS=4 LMCACHE_TRACK_USAGE=false /work/lmcache-venv/bin/python -m pytest -q \
  tests/test_page_unit_checkpoint.py tests/test_page_unit_checkpoint_leases.py \
  tests/test_state_offload_index.py tests/test_block_manager.py tests/test_state_checkpoint.py
```

GPU transfer checks (run separately from throughput measurement):

```bash
OMP_NUM_THREADS=4 LMCACHE_TRACK_USAGE=false /work/lmcache-venv/bin/python -m pytest -q \
  tests/test_lmcache_offload_gpu_disk_e2e.py tests/test_gdn_page_checkpoint.py
```

After the benchmark has drained, with no other requests being submitted,
verify model output across a forced CPU state restore:

```bash
python3 recipes/mlperf_endpoints_qwen35/verify_lmcache_state_restore.py \
  --fillers 2200 --gpu-checkpoints 1985 \
  --server-log /work/offload_c512/c512/server.log \
  --output /work/state-restore-probe.json
```

The probe uses one DP session, checks cold and GPU-cached greedy output,
then fills more unique checkpoints than fit in that worker's GPU state
budget. Fillers run sequentially so their CPU stores can drain. It requires
enough GPU evictions, substantial prefix reuse on the next anchor request,
identical restored output, and an increased completed CPU state-load counter.
The log combines worker counters without rank labels, so the probe conservatively
requires a new high-water mark; use a fresh server if an older count on another
worker prevents this check. For other budgets,
choose a filler count above the GPU image capacity and below the CPU image
capacity printed at server startup. CPU capacity is an upper limit within
the shared allocator, not a reservation protected from KV eviction.

Use the image's `engine_tput.py` on `RUNS/c512/server.log` to compare minute
buckets 1–9. Do not include startup or the drain after the 10-minute cap.
Inspect `state_offload_loads_completed`, failures, `checkpoints_evicted`,
prefix reuse and MTP acceptance as well as throughput. A loaded extension
or successful CPU store alone does not demonstrate useful offload.

The original harness's client reporting stalled while draining its tokenization
queue in these runs. Request completion and server throughput were collected
separately. After requests drained, the metrics service was terminated to flush
its snapshot; the resulting nonzero harness exit is not a successful client
report. Do not interpret its tokenization-derived throughput as validated.

## Measured result, 2026-10-31

Both measurements used the supplied image, model, DP4/EP4, MTP-3 and target
concurrency 512. The baseline used the original recipe. The final run used the
defaults above and reached the client's explicit 600,000 ms issue limit.
The table follows the supplied `engine_tput.py`: sum over four engines,
minute buckets 1–9 relative to `BENCH`, excluding the initial ramp and drain.

| Generation, tokens/s | Baseline | LMCache + GPU state budget |
|---|---:|---:|
| Average, buckets 1–9 | 24,100 | 24,903 |
| Bucket 6 | 26,625 | 24,946 |
| Bucket 7 | 26,082 | 23,935 |
| Bucket 8 | 17,264 | 24,609 |
| Bucket 9 | 13,318 | 21,781 |

The observed sharp late collapse did not recur: the last bucket improved
63.5%, and the measured average improved 3.3%. Throughput still softened
late; this is evidence for this ten-minute workload, not a claim of flat
25k tokens/s indefinitely. CPU offload alone with an unbounded GPU state
cache did not fix the collapse. A 16 GiB GPU state budget was too aggressive;
64 GiB retained a useful GPU working set. The 1,024-token chunks avoided the
CPU-fill pause seen in the 256-token trials.

The main benchmark's benefit was checkpoint retention and preserving KV
capacity. Most reuse remained in GPU memory; successful stores alone must
not be described as successful external loads. The separate forced-restore
probe and GPU byte round trips check the offload path itself. The relevant
regression suite passed 439 tests, and the GPU/disk round-trip suite passed
three cases, including the production block/chunk geometry.

The final forced probe evicted 2,195 GPU checkpoints, reused 6,336 of 6,337
anchor prompt tokens, and returned identical cold, GPU-cached and restored
greedy output. The completed CPU state-load counter increased from 0 to 1.
MTP acceptance during the benchmark was 68.9%.

The final benchmark's `BENCH` marker was `2026-10-31T13:58:38Z`;
the client issued from `13:58:47.689Z` to its cap at `14:08:47.688Z`.
Raw logs, per-minute summaries, package/build receipts and restore-probe
results accompany the patch in the experiment directory. The benchmark
qualifies the CPU tier; the disk tests check byte correctness, not NVMe
serving performance or long-duration stability.
