# MiniMax-M3 Gluon C32 experiment

Compare cached dense prefill using Gluon unified attention with the C32 cell in
[gather sweep 39034151209](https://github.com/ROCm/ATOM/actions/runs/39034151209),
job 117400829563. This branch starts at that run's exact ATOM commit,
`ebeb0f925289619cb0630b3d70a6f600412af6bb`.

The treatment is the cached-prefill attention implementation for both target
and EAGLE3 draft: native paged FP8 KV plus physical per-token scales are consumed
by Gluon instead of gathering a dense BF16 context. The adapter uses real block
tables and scheduled request metadata without converting the KV pool. Short
queries can split the context when the query grid cannot fill the GPU.

The AITER source is pinned to `a2e22688ace23667aa717df389d53b7177d0e334` on
`ROCm/aiter:mengqing/m3-gluon-scaled-unified`. Relative to baseline
`5eb866a95d66a38d64eecf151ac4ef3567a07ac5`, its changes are confined to unified
attention, its Gluon configuration, and attention tests. It combines the
previously GPU-validated per-token-scale operand loader (`76743b4b`) with
[aiter #6336](https://github.com/ROCm/aiter/pull/6336)'s speculative-query context
splitting and split-boundary fixes. M3 uses the validated M128 / 4-warps /
MFMA32 / single-buffer / page128 configuration. A single-token cached prefill
also uses the BF16 prefill operand layout.

## Controlled configuration

| Setting | Baseline and experiment |
| --- | --- |
| Case | `minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32` |
| Image | `rocm/atom-dev:nightly_202610141644` |
| ATOM source | Mounted checkout |
| Topology | One prefill TP4 + one decode TP4, PP1, eight GPUs |
| Draft | `Inferact/MiniMax-M3-EAGLE3`, three speculative tokens |
| Decode | Original `ATOM_PA_FLYDSL=1` implementation |
| KV | FP8, block size 128, original pool layout and budgeting |
| Service capacity | 80 sequences, 32768 batched tokens, GPU utilization 0.95 |
| LMCache | 256 GiB per prefill rank, four ranks |
| Workload | AgentX C32, 1M context limit, 3600 seconds, 10 warmups per lane |
| Mesh/router | Original image implementation and proxy configuration |
| Placement | `atomesh-cicd`, `amd-spur`, automatic node selection, original exclusions |
| Evaluation | Original performance recipe, `run_eval=false` |

The archived matrix cell from artifact 12101761714 is checked in at
`.github/scripts/atomesh/m3_gluon_c32_baseline.json`. Before GPU submission,
`check_m3_gluon_c32.py` compares the entire generated cell to it. Only the
AITER source SHA (replicated across common/prefill/decode/router environments)
may differ. Changing the model, topology, memory settings, workload, proxy,
image, or selecting additional cases fails this check.

Uncached prefill and memory profiling follow the original implementation.
The original runtime instrumentation and bounded-gather overlay still apply;
the cached gather branch is bypassed by the Gluon adapter. Runtime memory and
throughput differences caused by removing gather are outcomes of the kernel
change. Actual GPU block counts must be verified from startup logs.

## Validation

Completed locally:

- 185 ATOMesh unittest cases: passed, one skipped.
- Seven adapter and bounded-gather checks: passed.
- 18 public Gluon dispatch tests: passed, with no Triton attention fallback.
- Five offline gfx950 compilations: passed, covering one-token prefill,
  split and unsplit grids, and buffer/global KV loads including >2 GiB pools.
- Exact generated-matrix comparison: passed.
- Ruff, shell syntax, workflow YAML, and runtime-overlay compatibility: passed.

Offline compilation used the local `rocm/atom-dev:latest` image with Triton
3.7.0. It cannot substitute for numerical validation in the pinned CI image
on gfx950 hardware; the local GPU is gfx942.

Before either model replica is loaded, CI executes 17 numerical cases against
an independent Torch dense causal-attention reference. These cover GQA 1–64,
single-token and short speculative queries, mixed query tiles, long contexts,
strided scales, shared pages, poisoned partial pages, output guard regions,
and actual buffer/global loads. The gate requires real Gluon split and unsplit
launches and rejects Triton attention fallback. Any failure stops the job.
The normal split reduction is still implemented in Triton.

This experiment does not imply a performance gain. The earlier validated
Gluon operand loader was slower than gather for a large draft-prefill
microbenchmark; the end-to-end C32 result is needed to assess the tradeoff.
