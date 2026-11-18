# DeepSeek-V4.1 Flash standalone FP4 Mono indexer

`ATOM_DSV41_MONO_INDEXER=1` (the default) moves the indexer into its own
persistent launch on supported gfx950 deployments. The native decode hook and
the ordinary low-concurrency layer Mono route share the same kernel. Set
`ATOM_DSV41_MONO_INDEXER=0` at process startup to retain the previous embedded
indexer in layer Mono and the native projection/scoring chain in other decode
forwards. `ATOM_MONO_ENABLE=0` disables both Mono routes.

The launch reads attention-normalized BF16 hidden rows and the MXFP8 query
latent, then performs the replicated MXFP8 query projection, GPT-J tail RoPE,
head-weight projection, FP4 query quantization, paged scoring and stable
Top512/Top2048 selection. FULL layers use the request PAGE table; REINDEX
layers use the earlier candidate-block table and lift the selection back to
compressed row IDs. Both the OPUS `mfma` and eight-row `preshuffle` FP4 cache
layouts are supported.

The layer route exports the normalized hidden rows and query latent from K1,
runs compression and the standalone indexer, then passes the selected physical
rows to K2. Token-parallel K1 gathers every rank's query latent before the
standalone launch. Every rank computes its replicated indexer rows. Query,
scale and head-weight outputs use uncached device memory because projection
and scoring exchange them across XCDs within the same launch. A single-query
FULL task avoids computing the five padding queries in a six-query score walk.

Admission requires 256 gfx950 CUs, shuffled `(16,16)` MXFP8 query weights,
BF16 RoPE tables and the Flash geometry (32 index heads, dimension 128,
Top512, eight-row blocks and Top2048 candidate blocks). The standalone kernel
accepts at most 64 query rows. Other widths, prefill, incompatible inputs and
refused builds use the existing path. Builds with private scratch are refused
collectively before launch. The existing layer route retains its 48-row cap.
DP attention, FP8 index caches and the DSpark HI route retain their previous
implementation.

At TP2 without DP attention, FP4 decode widths of 40 rows or more use native
layers with the standalone Mono indexer. The measured layer-Mono route is
slower at that width; this guard avoids preparing unused layer buffers while
keeping the independent indexer active. TP4 and TP8 retain layer Mono for
supported widths.

Qualification on TP4 uses the local DeepSeek-V4.1-Flash checkpoint, BF16 main
cache and FP4 index cache. The GPU differential test covers 8K and 160K,
FULL/candidate/REINDEX layers, both layouts, empty padding rows, short context,
six-query groups, partial tails, the 64-row cap and CUDA Graph replay. Scoring
and selection are compared exactly against the native scorer on identical
quantized queries and head weights. Query projection and head weights are
checked separately: the native multi-row BF16 GEMM rounds partial sums while
Mono accumulates FP32, so complete projection chains can move near-boundary
TopK membership. The test reports that membership separately.

The native row-group scorer's long-context fast path mis-scores a partial
final six-query group in the tested aiter version. Differential tests qualify
those tails against its one-query path using the same cache pages. No aiter
source changes are required by this implementation.

Manual GPU qualification:

```bash
PYTHONPATH=/path/to/ATOM:/path/to/aiter \
ATOM_DSV41_CHECKPOINT=/data/models/deepseek-ai/DeepSeek-V4.1-Flash \
pytest -qs tests/models/deepseek_v41/test_mono_indexer.py
```

## Qualification results

On a gfx950 GPU, the final standalone-indexer matrix uses real checkpoint
projection weights with synthetic hidden/query/cache inputs, FP4 `mfma`,
contexts of 8192 and 163840 tokens, and FULL/candidate/REINDEX paths. Each
point uses seven alternating A/B samples, with 250 calls per sample in CUDA
Graphs. The qlen1 matrix covers sequence batches 1/8/16/24/32/40; qlen6 covers
batches 1/8, or 6/48 query rows. All measured builds have zero private scratch.

Of 48 points, 47 are faster than native and one is within timing noise:
160K FULL, qlen6, batch8 measures 61.619 us native versus 61.609 us Mono.
The 36 qlen1 points reduce indexer latency by 2.925%–43.009%. The 8K FULL
reductions at batches 1/8/16/24/32/40 are
16.438%/8.640%/5.195%/4.797%/5.305%/8.476%.

Real-checkpoint decode-forward measurements use synthetic window-sized
capture metadata, FULL CUDA Graphs and compilation level0. They exclude
prefill, scheduling, TTFT and serving throughput. Each result is the median
of seven paired samples, with five warmups and twenty measured replays, using
the slowest TP rank. The default route reduces latency by:

| TP | Batch1 | Batch8 | Batch40 |
|---:|---:|---:|---:|
| 2 | 46.231% | 40.705% | 0.824% |
| 4 | 49.588% | 43.012% | 17.878% |
| 8 | 53.185% | 44.210% | 5.902% |

TP2 batch40 uses native layers with the standalone indexer; the other entries
use layer Mono with the independent indexer. These larger gains include
existing layer fusion. Replacing only the native indexer reduces whole-model
decode-forward latency by about 0.5%–1.0% in this window-sized benchmark.

Validation includes 17 GPU indexer differential tests, 150 exact TopK cases,
437 CPU tests, 65 build-key/dispatch tests and 21 TP4 real-model generation
cases. Both Mono routes check all eight indexer layers' live head-weight
projections. Greedy output sequences are not all token-identical to native.
Formal GSM8K, compilation level1 and actual 8K/160K whole-model serving
performance remain unqualified by these results.
