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

This branch establishes functionality and includes initial performance
measurements. Its standalone indexer remains slower than the native chain in
the measured 8K/160K matrix; those measurements do not establish an overall
throughput improvement or GSM8K accuracy.
