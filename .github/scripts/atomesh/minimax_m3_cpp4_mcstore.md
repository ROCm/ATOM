# MiniMax M3: direct-paged versus grouped-gather C72

Case: `minimax-m3-fp4-1p1d-cpp4-tp4-eagle3-agentic-mcstore-1m-c72`
under suite `mooncake_store` in `models_atomesh.yaml`.

This branch (`mengqing/minimax-m3-paged-c72-compare`) reads cached dense
MHA/GQA K/V directly from the existing pool, including EAGLE3. The AITER
kernel follows the block table and applies FP8 per-token scales in the
kernel; no full-context BF16 K/V gather or request grouping is performed.
Uncached prefill, sparse attention, PP transport and Store page layouts keep
the baseline behavior.

The comparison base is `d04323cdf` on
`mengqing/minimax-m3-gather-concurrency-sweep`. Its only change from gather
CI run 39203564775 (`d3ff348ce`) is cumulative-length naming/documentation.
The paged adapter is ported from ATOM `5bed482f4`, preserved on
`mengqing/draft-prefill-kv-workspace`. AITER is pinned to the numerically and
GSM8K-validated `af88b4020521e25b677c5c6e5ff1d33d40395cb3` on
`mengqing/paged-prefill-kv`. Later asynchronous / scale-after-dot candidates
are not enabled.

Both C72 runs need to complete before drawing a performance conclusion.

| Setting | Value |
| --- | --- |
| Placement | Single node, 8 GPUs, 1P1D |
| Prefill | PP4 x TP1, 15 target layers/stage, EAGLE3 on last stage, eager |
| Decode | TP4, DCP1, original EAGLE3 settings |
| P / D batched-token budget | 8192 / 2048 |
| CPU tier | Shared Mooncake Store, 2 NUMA owners x 768 GiB = 1536 GiB |
| Dense gather workspace | Paged: 0; baseline: prefill 32768 MiB, decode 8192 MiB |
| Performance | AgentX 1M, C72, 3600 s profiling, 10 warmup requests/lane |
| Allocation | 8 hours, TW `amd-spur`, default Slurm account, automatic node |

Budgets and Store size follow
`https://github.com/SemiAnalysisAI/InferenceX/pull/4045`, inspected at
`9f774fe3f040329eb38b9fbde85e0e8d3ce818bc`.
That GLM recipe also uses DCP4; this M3 experiment retains DCP1. The Store
owners use the existing TW NIC mapping (`rdma4,rdma5` and `rdma6,rdma7`).
The selected image is the successful gather baseline's
`rocm/atom-dev:nightly_202610141644`; the launcher checks for Store binaries
and its Python client before model startup. The baseline AITER is
`5eb866a95d66a38d64eecf151ac4ef3567a07ac5`; the paged pin descends from it
and changes only unified attention and its numerical tests. The scatter
bounds fix is retained.

## Compatibility changes

- Port the requested native gather commits and their initial implementation
  dependency. Remove the CI monkey-patch gather module and its injection.
- Save all M3 PAGE regions through the Store, including K/V, their scales,
  sparse index cache, and the last stage's draft pages; isolate the codec
  version in the persistent key namespace.
- Carry EAGLE3 auxiliary captures between PP stages. Permit producer-only
  q1 input preparation with PP and speculative decoding enabled.
- Match P/D PAGEs by global target layer or draft identity. A TP1 producer
  sends each TP4 consumer its head slice; sparse KV uses eight physical
  page-16 groups inside a scheduler page-128, dense/draft KV uses one.
  The index cache is replicated. DCP relayout is not enabled by this change.

## Ordered validation

The case opts into `eval_before_benchmark`:

1. The pinned AITER standalone suite checks 33 numerical configurations,
   including partial pages, mixed lengths and GQA ratios 1 through 64. CPU
   adapter/dispatch tests also run in the GPU image with AITER available.
2. A real GPU/Store round trip checks target, index, scale and draft-like PAGE
   payloads. A skip is a failure of this gate.
3. Full GSM8K, 20-shot, EAGLE3 real acceptance, concurrency 32, completions
   endpoint, 8192 max generated tokens. The flexible-extract score must be
   at least 0.90. This is an experiment gate, not an already measured score.
4. Fresh services and a fresh Store run C72 AgentX. The two phases use separate
   container lifetimes and ports; accuracy cache warmup is not carried into
   performance. An accuracy failure prevents the performance phase.

CPU tests cover full PP auxiliary propagation, production RDMA descriptor
construction with reordered layers, TP head slicing and replicated index
bytes, all-plane Store codec creation, and phase sequencing. Actual model,
GPU Store, GSM8K and performance validation run only on ATOMESH benchmark CI.

## Memory tradeoff and measurements

TP1 EAGLE3 has 64 KV heads of width 128. One 1,048,576-token request needs
32 GiB of BF16 K+V if gathered. The grouped-gather baseline reserves
32 / 0 / 0 / 32 GiB across the PP stages before sizing persistent KV:
stage 0 has the target's dense layers, and stage 3 owns EAGLE3.

Paged dense layers do not declare a gather workspace, so this experiment
has no dense-gather deduction. The configured budget is explicitly zero
for both roles. The existing measured activation peak, CUDA graph estimate,
2% safety margin, and connector reservations remain in place. Paged attention
still allocates output and, for split-K, query-proportional FP32 partial
output/max/sum buffers. It is not allocation-free.

At 34,656 bytes/token on the final PP stage, releasing 32 GiB alone could
add about 991,434 tokens before block rounding, changes in measured peak
memory, and the minimum across all ranks. This is an upper-bound scenario,
not a measured capacity increase. The shared 1536 GiB Store capacity is
unchanged: at most 18,837,504 complete payload tokens (256-token chunks),
at 87,552 bytes/token across the four stages.

Record each rank's memory budget and final common KV block count. Compare
C72 successful requests, output/total token throughput, TTFT p50/p90/p99,
ITL p90, and E2E-normalized interactivity on the full profiling window.
Use engine cache reads and offload load/save/eviction evidence to distinguish
GPU retention and CPU-tier hits from theoretical trace prefix reuse.

The two expanded cases differ only in the pinned AITER, gather budget and
the paged numerical preflight. Image, model, P/D budgets, Store size, workload,
accuracy gate, node exclusions and Slurm settings match. Automatic allocation
may select different nodes; retain node identity and memory measurements when
interpreting a single A/B result.

## Previous C32 evidence (different topology)

With TP4 prefill and LMCache, validated paged CI 39012235215 completed with
3847 successful requests and zero errors, following full GSM8K 20-shot
CI 39004723476: 1273/1319 flexible-extract correct (96.5125%).

| Metric | Successful gather | Validated paged | Change |
| --- | ---: | ---: | ---: |
| Output tok/s | 1062.880 | 1060.789 | -0.20% |
| Total tok/s | 168199.321 | 167550.031 | -0.39% |
| TTFT p90 (ms) | 1836.056 | 1997.204 | +8.78% |
| TTFT p99 (ms) | 6099.750 | 9315.472 | +52.72% |

Those historical ATOM revisions also differed outside attention; they do
not isolate the kernel's causal effect. Their C32 result cannot determine
which memory/compute tradeoff wins at PP4 + Mooncake Store + C72.
