# MiniMax-M3 Gluon C32 experiment

Compare cached dense prefill using Gluon unified attention with the C32 cell in
[gather sweep 39034151209](https://github.com/ROCm/ATOM/actions/runs/39034151209),
job 117400829563. This branch starts at that run's exact ATOM commit,
`ebeb0f925289619cb0630b3d70a6f600412af6bb`.

The treatment is the cached-prefill attention implementation for both target
and EAGLE3 draft: Gluon consumes native paged FP8 KV with the scales returned
by each writer (target per-token, draft scalar), instead of gathering a dense
BF16 context. The adapter uses real block
tables and scheduled request metadata without converting the KV pool. Short
queries can split the context when the query grid cannot fill the GPU.

The AITER source is pinned to `5f1da21fe6e89620babf1450847783b8d551ccef` on
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

## CI repair after run 39280266360

The first 14 GPU cases passed. The short-context case incorrectly shared a
partial page and poisoned valid tokens in the other request. The fixture now
shares only full prefix pages and checks that its reference is finite before
launch. Four CPU regressions run in the hosted configuration job before GPU
allocation; the 17-case GPU gate remains required. Kernel code is unchanged.

The shared FlyDSL capability probe now explicitly imports the module instead
of the package's lazy function export. Four fresh-process import tests cover
both import orders and rejection of the old optional-plan API. The pinned
source is also probed before model startup. Benchmark monitoring recognizes
ModelRunner death even if the HTTP frontend still responds.

## CI repair after run 39351509328

All 17 AITER GPU cases passed, but real EAGLE3 cached prefill failed because
`ATOM_FORCE_ATTN_TRITON=1` selects a rotary-only cache writer that returns a
single FP32 `kv_scale`. The earlier adapter incorrectly required per-token
scales even though this writer leaves the preallocated scale planes unused.

The adapter now accepts the writer's scalar or per-token scale independently
for K and V. A scalar becomes a zero-stride `(blocks, heads, page_size)` view
through `scale.view(()).expand(...)`: it retains the same pointer and four
bytes of storage, even for 53602 draft blocks. No scalar readback, scale-plane
initialization, cache conversion, or change to the writer is involved. This
uses the pinned Gluon BF16/FP8 operand loader, preserving BF16 dequantization
as in gather. Passing only `k_descale/v_descale` would not suffice: that pinned
Gluon dispatch supports BF16 Q / FP8 KV only through its scale-cache path.

CPU adapter regressions run before GPU allocation, including the actual draft
Python `rope_cache` branch with a stub GPU writer and poisoned unused scale
planes. In addition to the original 17 kernel cases, a nine-case GPU gate calls
real ATOM `rope_cache` and `prefill_attention`: target Gemma/per-token and draft
rotary/scalar, scale 1.0 and 0.375, single-token, short and large queries,
permuted physical pages, split/unsplit and buffer/global loads. It compares
against Torch dense causal attention and rejects Triton attention fallback.
These GPU cases must pass in the target CI image before loading the models.

The benchmark guard also recognizes the multiprocessing
`Process ModelRunner3/4:` uncaught-exception header, including split log reads.
It does not treat arbitrary logged tracebacks as worker death. This prevents
the observed 900-second stall wait while the HTTP frontend remains healthy.
AITER stays at `5f1da21fe6e89620babf1450847783b8d551ccef`; the experiment matrix
and original draft quantization remain unchanged.

Local validation for this repair: 16 CPU adapter cases passed; 193 ATOMesh
unittests completed with one skip and no failures. Six actual adapter/dispatch
argument sets compiled offline for gfx950, covering scalar/per-token,
split/unsplit and buffer/global loads. The regenerated matrix is identical to
run 39351509328 and still passes the gather-C32 baseline comparison. Ruff,
Black, shell syntax and workflow YAML checks pass. The new GPU numerical gate
has not yet run on gfx950; no new performance result is available.

## Additional preflight review

The standalone adapter gate now pins this checkout before importing ATOM.
A container reproduction showed that `python -c` used the mounted checkout,
while a standalone script imported the image's editable `/app/ATOM` instead.
A CPU regression runs the gate from another cwd with a conflicting `PYTHONPATH`.
The gate also resolves its monitoring hooks from ATOM's actual unified-attention
callable: importing AITER's canonical module before its legacy alias can leave
a stale module object whose patches would not observe ATOM's launches.

Each numerical case now writes the cached prefix first, then appends only the
scheduled query tokens and uses the real `dispatch_backend` result. The earlier
fixture wrote all context tokens in one call; it missed the draft writer's
`BLOCK_H=1` path for 1/4 new tokens. Offline meta-tensor tracing reached the real
target and draft prefix/append kernels, and all eight writer launches compiled
for gfx950. Three additional meta-tensor checks exercised the actual ATOM
`prefill_attention` method with scalar scales and a 53602-block pool; the gate
captured the expected Gluon launches. These are routing and compilation checks,
not GPU numerical results.

The native zero-based prefix metadata, padded request slicing, unchanged decode
path, and runtime overlay order were reviewed. The Gluon return precedes the
bounded-gather overlay. No further deterministic runtime defect was found in
these checked paths. Target gfx950 numerical validation remains pending.

Review validation completed: 194 ATOMesh tests, 193 passed and one skipped;
process exit code 0. Ruff, Black, shell syntax and whitespace checks pass.
