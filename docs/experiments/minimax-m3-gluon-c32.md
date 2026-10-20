# MiniMax-M3 C32 with official v0.1.26 Gluon attention

The experiment uses the official AITER v0.1.26 unified attention code and
split optimization. A small KV-format extension reads ATOM's existing FP8
cache with BF16 queries. The target keeps per-token scales; the draft keeps
its scalar scale. There is no full-pool conversion or dense KV workspace.

## Source and scope

- ATOM branch: `mengqing/minimax-m3-gluon-c32`.
- Official AITER tag: `v0.1.26`, commit
  `84077b3c0461dac1f0f3f8b9590a9282e2035a2f`.
- Experiment AITER branch: `ROCm/aiter:mengqing/m3-gluon-v0126-native`.
- Fixed AITER commit: `4fe2800a5c252adff8ee28d199bea66f8b18efac`.

Commit `86f7d57f` copies five files byte for byte from the official tag: the
unified dispatcher, gfx950 Gluon kernel, Triton attention/reduce kernel, and
gfx950 Gluon/Triton unified-attention configs. Other AITER operators stay at
baseline `5eb866a95d66a38d64eecf151ac4ef3567a07ac5`, including its scatter fix.
Commit `4fe2800a` adds the KV-format extension and numerical tests.

The old experimental M128, whole-page tile and single-buffer overrides are
removed. The official dispatcher selects split counts, tile sizes, warps and
buffers. Single-token queries use its decode grid. This describes the grid
inside cached prefill; ATOM's ordinary decode path still uses FlyDSL.

## Native KV adaptation

Official gfx950 Gluon accepts matching Q/KV dtypes and scalar descales. ATOM's
target instead supplies BF16 Q and FP8 KV with physical per-token scales.
The extension adds `k_scale_cache` and `v_scale_cache`, each shaped
`[physical block, KV head, token in block]`. It supports the M3 shuffled
head-128/page-128 format and refuses unsupported input or attention fallback.

`ScaledKVLoader` wraps the official whole-page or sub-page async loader.
For each logical token it reads the real physical block ID, loads the scale,
and converts the FP8 value to a BF16 operand. A tile can be smaller than a
physical page. Loop bounds, barriers, split boundaries and attention math
remain upstream code. Draft scalar scales use a zero-stride view backed by
one FP32 value. Quantization, writers and the native KV pool do not change.

The only loop-call extension is the logical tile index needed to load scales.
The official split reduce remains the Triton `reduce_segments` kernel.

## Controlled C32 configuration

The comparison is the C32 cell of
[gather sweep 39034151209](https://github.com/ROCm/ATOM/actions/runs/39034151209),
job `117400829563`, ATOM commit `ebeb0f925289619cb0630b3d70a6f600412af6bb`.
It remains the performance baseline, while the new kernel comes from the
official v0.1.26 source.

| Setting | Value |
| --- | --- |
| Case | `minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32` |
| Image | `rocm/atom-dev:nightly_202610141644` |
| Topology | One TP4 prefill + one TP4 decode, PP1, eight GPUs |
| Draft | `Inferact/MiniMax-M3-EAGLE3`, three speculative tokens |
| KV | FP8, block size 128, original pool and capacity budgeting |
| Service | 80 sequences, 32768 batched tokens, GPU utilization 0.95 |
| LMCache | 256 GiB per prefill rank |
| Workload | AgentX C32, 1M context limit, 3600 seconds, 10 warmups per lane |
| Evaluation | `run_eval=false` |
| Placement | `atomesh-cicd`, `amd-spur`, original automatic node selection |

Only target/draft cached dense prefill changes its attention implementation.
Uncached prefill, profiling, ordinary decode, router, memory settings and
workload stay at the C32 baseline. The generated matrix must match the
archived baseline except for the AITER attention source SHA.

## Code and validation

- [ATOM adapter](../../atom/model_ops/gluon_prefill.py).
- [ATOM attention entry](../../atom/model_ops/attention_mha.py).
- [Pinned source and GPU gates](../../.github/scripts/atomesh/prepare_scatter_aiter_source.sh).
- [Official-source check](../../.github/scripts/atomesh/check_m3_gluon_upstream.py).
- [Source fingerprints](../../.github/scripts/atomesh/m3_gluon_v0126_source.json).
- [Real writer/adapter gate](../../.github/scripts/atomesh/check_m3_gluon_adapter.py).

The source check runs before GPU allocation and again in the GPU job. It
checks official configs, split helpers, attention math, buffer loops and
reduce. It permits only the added `tile_index` keyword in the buffer loops.

Local checks passed: six real adapter/dispatcher gfx950 offline compilations;
16 CPU adapter tests; four CPU page-fixture tests; the ATOMesh CPU suite
(194 tests, one skipped); and the new source-guard regression. The captured
launches use tile64, two buffers, both single-token and prefill grids, and
buffer/global loads for small and greater-than-2-GiB pools.

These checks are not GPU numerical validation. Before model startup, CI runs
17 public-wrapper and nine actual ATOM writer/adapter numerical cases. The
new implementation has no confirmed end-to-end performance result yet.
