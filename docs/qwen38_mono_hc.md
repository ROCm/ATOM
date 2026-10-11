# Qwen3.8 Flash-Next decode specializations

This work adds gfx950 FlyDSL kernels for the repeated hyper-connection (HC)
path and an opt-in indexer-only context partition for long QSA decode. These
are component specializations, not a single-launch decoder layer. No Qwen
Triton HC variant is retained. QSA still uses ATOM's existing paged scorer,
AITER stable top-k, and sparse attention; index CP partitions that scorer's
work over the TP group without partitioning the KV cache or GDN/PLE state.

## FlyDSL hyper-connection

Set `ATOM_QWEN38_MONO_HC=1` before starting ATOM. The default is off. Dispatch
requires gfx950, contiguous BF16 data and weights, FP32 norm weights, and the
Flash-Next geometry: HC=4, hidden=2560, low rank=320. Mix uses three FlyDSL
launches at 1–16 tokens; combine uses one launch at 1–128 tokens. Larger and
unsupported shapes use the native operators.
The HC kernels require
[ROCm/FlyDSL#1265](https://github.com/ROCm/FlyDSL/pull/1265), including its typed
`fx.rocdl.dot2_bf16` intrinsic wrapper. The installed FlyDSL 0.2.4 predates
this compiler API and cannot compile the specialization.

The four stages are grouped Gemma RMSNorm, down projection with scaled SiLU,
up projection with gated mean, and injection with residual update. The
normalization output is kept because both the up and injection projections
consume it. BF16 rounding at projection and scaled-activation boundaries is
preserved. Each CTA owns its output; there is no persistent scratch or
cross-CTA synchronization, so changed-input CUDA Graph replay is safe.

`python -m kernels.monokernel.qwen38.tools.benchmark --repeats 5` benchmarks an
actual ATOM `Qwen4ExpHyperConnection` with synthetic BF16 weights on one idle
MI355X. Times are median graph GPU microseconds for one mix+combine pair:

| Decode tokens | Native | FlyDSL | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 33.29 | 12.59 | 2.64× |
| 2 | 35.06 | 13.46 | 2.61× |
| 4 | 35.58 | 15.71 | 2.27× |
| 8 | 36.41 | 20.08 | 1.81× |
| 16 | 36.17 | 28.07 | 1.29× |

The 16-token FlyDSL stage times were 2.58 µs norm, 7.08 µs down, 12.88 µs
up, and 3.16 µs combine. The up projection is the larger-batch bottleneck.
For 16/32/64/128 tokens, the combine stage alone measured 3.34/3.39/3.48/
3.95 µs versus 9.92/9.93/10.15/10.08 µs native, or 2.98/2.93/2.92/2.55×.
The 32–128-token mix remains native.

### Generated gfx950 ISA

The benchmark's `--dump-ir-dir` option saves the final AMDGCN assembly. The
one-token build at `/tmp/qwen38-rocdl-dot2-isa2` yielded:

| Stage | VGPR | LDS bytes | `v_dot2c_f32_bf16` | DPP moves | Buffer loads | Barriers | Scratch ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Norm | 21 | 16 | 0 | 6 | 15 | 1 | 0 |
| Down + SiLU | 45 | 16 | 20 | 6 | 40 | 1 | 0 |
| Up + gated mean | 13 | 0 | 12 | 24 | 19 | 0 | 0 |
| Injection + combine | 49 | 16 | 20 | 6 | 50 | 2 | 0 |

The BF16 GEMVs use packed `v_dot2c_f32_bf16` through a typed FlyDSL wrapper;
the HC implementation contains no inline assembly or raw LLVM operations.
There is no MFMA at one token.
No stage spills. The up stage's four gate reductions account for its 24 DPP
moves and four exponentials/reciprocals; it is compute and reduction heavy as
batch size rises. Down and combine use one wave reduction plus a CTA barrier
to join partials. The combine grid grows with rows, which keeps it effective
through 128 tokens.
The separately dumped 128-token combine build still has 49 VGPR, 20 dot2
instructions, 32 bytes LDS, and no scratch operations.

## QSA long context and index CP

`tools/bench_qwen38_qsa_longctx.py` tests an unsplit QSA layer at 128k and
256k with a permuted page table and eight decode rows. It checks the last
row's selected groups against an independent PyTorch score/top-k, every row's
sparse attention against a PyTorch attention reference, and changed-input
graph replay. The maximum attention NRMSE was 0.00277 at 128k and 0.00270 at
256k. Unsplit selection took 52.85/71.13 µs and sparse attention 13.21/13.24
µs respectively. These are synthetic component times.

Set `ATOM_QWEN38_QSA_CP=1` for the experimental indexer-only TP2 or TP4 path.
The flag defaults off. Each rank scores a contiguous slice of compressed
groups in its replicated index cache, keeps the full local top-k, all-gathers
bounded score/id candidates, merges with stable top-k, then runs normal local
TP sparse attention. The selected tokens matched the unsplit path exactly at
128k and 256k, including graph replay after query and cache changes. AITER's
stable top-k needs contiguous candidate scores; a strided view produced wrong
selections during bring-up, so the merge materializes a compact score row.

The fixed exchange cost is a loss at small batches. Production dispatch uses
the native selection below 16 million row-context tokens on TP4 and 32
million on TP2. The following TP4 graph benchmark gives every row a distinct
128k or 256k request and a disjoint, permuted physical page range:

| Context | Rows | Unsplit selection | Index CP selection | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128k | 64 | 170.28 µs | 134.31 µs | 1.27× |
| 128k | 128 | 314.94 µs | 171.96 µs | 1.83× |
| 256k | 64 | 329.15 µs | 171.27 µs | 1.92× |
| 256k | 128 | 619.24 µs | 257.82 µs | 2.40× |

For 256k/128 rows, including local TP sparse attention measured 688.76 µs
unsplit versus 330.95 µs with CP (2.08×). Sparse attention outputs were exact
matches. TP2 selection at 256k/128 rows measured 1.61×, or 1.41× including
sparse attention.
Ragged requests, graph padding, and all-zero score ties were also checked for
exact parity.
The QSA CP path reuses the M3-style registered all-gather when available and
warms both transport choices before graph capture. It is separate from full
PCP/DCP, whose Qwen backend guard remains in place because those modes need
partitioned cache/page metadata, recurrent state and attention contracts.

## Validation scope

`tests/model_ops/test_qwen38_mono_hyperconnection.py` checks 1/2/4/8/16-token
mix+combine parity, 32/64/128-token combine parity, fallback, and graph replay.
`tools/bench_qwen38_qsa_cp.py` checks exact selection and changed-input graph
replay on two or four GPUs. No local Flash-Next checkpoint was available, so
whole-layer accuracy, real-checkpoint generation, and InferenceMAX throughput
remain unmeasured. The 2–3× figures above apply to the named HC or QSA
component and measured shape, not to the full model.
