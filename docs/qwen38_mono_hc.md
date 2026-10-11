# Qwen3.8 Flash-Next decode specializations

This work adds gfx950 FlyDSL kernels for the repeated hyper-connection (HC)
path and an opt-in indexer-only context partition for long QSA decode. These
are component specializations, not a single-launch decoder layer. This PR
adds no Triton HC kernel; the existing Triton HC path remains the fallback.
QSA still uses ATOM's existing paged scorer,
AITER stable top-k, and sparse attention; index CP partitions that scorer's
work over the TP group without partitioning the KV cache or GDN/PLE state.

## FlyDSL hyper-connection

Set `ATOM_QWEN38_MONO_HC=1` before starting ATOM. The default is off. Dispatch
requires gfx950, contiguous BF16 data and weights, FP32 norm weights, and the
Flash-Next geometry: HC=4, hidden=2560, low rank=320. The served decoder calls
`mix_fused`, which uses three FlyDSL launches at exactly 4 or 8 decode tokens:
deferred combine plus grouped norm, fused down and injection projection, then
up projection plus gated mean. It keeps the existing two-launch path at 1–2
tokens and native GEMMs at 16 or more tokens, where the FlyDSL path did not
improve latency. The standalone `mix` and `combine` calls used by the MTP
drafter use three launches at 1–16 tokens and one launch at 1–128 tokens,
respectively. Unsupported shapes use the native operators. The kernel and its
packed BF16 dot helper live in ATOM and use the FlyDSL build already used by
ATOM's mono kernels. No FlyDSL source change or companion PR is required.

The four stages are grouped Gemma RMSNorm, down projection with scaled SiLU,
up projection with gated mean, and injection with residual update. The
normalization output is kept because both the up and injection projections
consume it. BF16 rounding at projection and scaled-activation boundaries is
preserved. Each CTA owns its output; there is no persistent scratch or
cross-CTA synchronization, so changed-input CUDA Graph replay is safe.

`python tools/bench_qwen38_hc.py --repeats 5` benchmarks an
actual ATOM `Qwen4ExpHyperConnection` with synthetic BF16 weights on one idle
MI355X. Times are median graph GPU microseconds for one standalone
mix+combine pair, which the served decoder does not call:

| Decode tokens | Native | FlyDSL | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 33.58 | 12.98 | 2.59× |
| 2 | 35.44 | 13.90 | 2.55× |
| 4 | 36.10 | 16.44 | 2.20× |
| 8 | 36.31 | 20.53 | 1.77× |
| 16 | 36.31 | 28.52 | 1.27× |

For the served `mix_fused` call with a deferred block output and injection
logits, the same benchmark measured:

| Decode tokens | Native | FlyDSL | Speedup | Production dispatch |
| ---: | ---: | ---: | ---: | :--- |
| 1 | 8.74 | 10.44 | 0.84× | Native |
| 2 | 10.73 | 12.50 | 0.86× | Native |
| 4 | 26.45 | 14.67 | 1.80× | FlyDSL |
| 8 | 26.31 | 18.94 | 1.39× | FlyDSL |
| 16 | 26.37 | 27.00 | 0.98× | Native |

These are HC component times with synthetic weights. The production path's
three launches at 4–8 tokens are part of a decoder layer with separate
attention, MoE, communication, state update, and QSA launches; the table does
not measure whole-layer latency or model throughput.

The 16-token standalone FlyDSL stage times were 2.69 µs norm, 7.08 µs down,
13.27 µs up, and 3.18 µs combine. The up projection is the larger-batch
bottleneck.
For 16/32/64/128 tokens, the combine stage alone measured 3.34/3.39/3.48/
3.95 µs versus 9.92/9.93/10.15/10.08 µs native, or 2.98/2.93/2.92/2.55×.
The 32–128-token mix remains native.

### Generated gfx950 ISA

The benchmark's `--dump-ir-dir` option saves the final AMDGCN assembly. The
one-token build at `/tmp/qwen38-atom-internal-isa` yielded:

| Stage | VGPR | LDS bytes | `v_dot2c_f32_bf16` | DPP moves | Buffer loads | Barriers | Scratch ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Norm | 21 | 16 | 0 | 6 | 15 | 1 | 0 |
| Down + SiLU | 45 | 16 | 20 | 6 | 40 | 1 | 0 |
| Up + gated mean | 13 | 0 | 12 | 24 | 19 | 0 | 0 |
| Injection + combine | 49 | 16 | 20 | 6 | 50 | 2 | 0 |

The four-token served variant with a pending combine and next injection has
three compiled stages:

| Stage | VGPR | LDS bytes | Packed BF16 dot2 | Barriers | Scratch ops |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Combine + norm | 26 | 32 | 0 | 1 | 0 |
| Down + injection | 65 | 32 | 40 | 1 | 0 |
| Up + gated mean | 15 | 0 | 24 | 0 | 0 |

The down and up stages retain `v_dot2c_f32_bf16`; none of the served stages
spill. The norm stage recomputes the rounded combined values after its
reduction, avoiding a read of data it has just written through a buffer
resource. That read produced incorrect normalized values during bring-up.

The BF16 GEMVs use packed `v_dot2c_f32_bf16` through an ATOM-side typed helper;
the Qwen HC kernel contains no inline assembly or raw LLVM operations.
There is no MFMA at one token.
No stage spills. The up stage's four gate reductions account for its 24 DPP
moves and four exponentials/reciprocals; it is compute and reduction heavy as
batch size rises. Down and combine use one wave reduction plus a CTA barrier
to join partials. The combine grid grows with rows, which keeps it effective
through 128 tokens.
The separately dumped 128-token combine build still has 49 VGPR, 20 dot2
instructions, 32 bytes LDS, and no scratch operations.
At 16 tokens, the grouped down CTA uses 96 VGPR, 80 packed dot2 instructions,
100 buffer loads, and no scratch spills. The up CTA uses 15 VGPR, 24 dot2
instructions, 26 buffer loads, eight exponentials and eight reciprocals, with
no LDS or spills. It takes 13.27 µs of the 28.52 µs pair; giving each up CTA
four rows raised VGPR to 25 and made the up stage slower (13.61 µs), so the
two-row schedule is retained. Direct FlyDSL mix probes at 32/64/128 rows took
38.0/67.8/142.9 µs versus 25.3/27.5/30.1 µs native; mix therefore falls
back above 16 rows while the fused combine remains enabled through 128.

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
| 128k | 256 | 635.20 µs | 285.07 µs | 2.23× |
| 128k | 512 | 1061.81 µs | 435.51 µs | 2.44× |
| 256k | 64 | 329.15 µs | 171.27 µs | 1.92× |
| 256k | 128 | 619.24 µs | 257.82 µs | 2.40× |
| 256k | 256 | 1132.38 µs | 423.08 µs | 2.68× |
| 256k | 512 | 2074.04 µs | 676.34 µs | 3.07× |

For 256k/128 rows, including local TP sparse attention measured 688.76 µs
unsplit versus 330.95 µs with CP (2.08×). At 256k/512 rows, it measured
2355.56 versus 937.29 µs (2.51×). At 128k/512 rows it measured 1303.68 versus
678.47 µs (1.92×). Sparse attention outputs were exact matches. TP2 selection
at 256k/128 rows measured 1.61×, or 1.41× including sparse attention.
The larger-batch improvement budgets the score workspace against each rank's
local shard width, allowing fewer row chunks and candidate exchanges.
Ragged requests, graph padding, and all-zero score ties were also checked for
exact parity.
The QSA CP path reuses the M3-style registered all-gather when available and
warms both transport choices before graph capture. It is separate from full
PCP/DCP, whose Qwen backend guard remains in place because those modes need
partitioned cache/page metadata, recurrent state and attention contracts.

## Validation scope

`tests/model_ops/test_qwen38_mono_hyperconnection.py` checks 1/2/4/8/16-token
mix+combine parity, 32/64/128-token combine parity, served `mix_fused` parity
at 4/8 tokens including the final mixer without injection, fallback, and
changed-input graph replay.
`tools/bench_qwen38_qsa_cp.py` checks exact selection and changed-input graph
replay on two or four GPUs. No local Flash-Next checkpoint was available, so
whole-layer accuracy, real-checkpoint generation, and InferenceMAX throughput
remain unmeasured. The 2–3× figures above apply to the named HC or QSA
component and measured shape, not to the full model.
