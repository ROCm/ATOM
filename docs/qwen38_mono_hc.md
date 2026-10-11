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

A one-launch resident FlyDSL HC experiment for 1–2 rows matched the native
outputs and graph replay, but took 15.8/20.6 µs versus the native two-launch
path's 8.5/10.6 µs. Its cross-CTA norm/down/up handoffs cost more than the
saved launch, even with zero ISA spills. That experiment is not dispatched or
included in this PR; 1–2 row HC remains native.

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

The candidate exchange now uses three small FlyDSL kernels: pack local
score/id pairs, unpack rank-major gathered pairs into contiguous row-major
scores and ids, then map merged offsets to global ids. They replace multiple
PyTorch gather, mask, stack, permute, copy, fill and gather launches. AITER's
stable top-k and the existing QSA scorer and sparse attention remain in use.
The generated 48-row gfx950 kernels use 6/6/5 VGPR, no LDS, and no spills.
TP4 uses the registered gather through 1 MiB of per-rank payload, where it
still beat RCCL by 12–22 µs; TP2 retains its 512 KiB limit. Both transports
are warmed before native decode graph capture.

The measured dispatch gates are 24/48 rows at the 128k score bucket for
TP4/TP2, and 2/16 rows at the 256k bucket. Smaller contexts retain native
selection. These gates include distinct, shared-request, ragged and padded
rows; the native path remains selected where the exchange can lose.

Median captured graph times on MI355X with a distinct permuted request per
row, including both selection and local TP sparse attention:

| TP | Context | Rows | Unsplit | Index CP | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 128k | 24 | 92.72 µs | 83.32 µs | 1.11× |
| 4 | 128k | 48 | 164.66 µs | 105.18 µs | 1.57× |
| 4 | 128k | 96 | 286.83 µs | 150.04 µs | 1.91× |
| 4 | 256k | 2 | 77.66 µs | 67.37 µs | 1.15× |
| 4 | 256k | 32 | 209.49 µs | 110.71 µs | 1.89× |
| 4 | 256k | 48 | 276.87 µs | 135.88 µs | 2.04× |
| 4 | 256k | 128 | 651.31 µs | 273.44 µs | 2.38× |
| 4 | 256k | 160 | 817.14 µs | 338.36 µs | 2.42× |
| 2 | 128k | 48 | 165.56 µs | 131.85 µs | 1.26× |
| 2 | 256k | 16 | 120.56 µs | 106.19 µs | 1.14× |
| 2 | 256k | 48 | 281.07 µs | 191.61 µs | 1.47× |
| 2 | 256k | 128 | 650.24 µs | 411.25 µs | 1.58× |

The selected tokens and sparse outputs matched the unsplit path exactly on
all listed shapes. TP4 256k/513-row selection, split into 512+1-row workspace
chunks, was exact under changed-input graph replay and took 2240.05 to
693.94 µs (3.23×) with `--cp-logits-workspace-mib 32`; sparse attention was
not included in that case. Ragged,
padded and all-zero tie cases were also exact at 128k/256k on TP2 and TP4.
The QSA CP path is separate from full PCP/DCP, whose Qwen backend guard
remains because those modes require partitioned cache, page metadata and
recurrent-state contracts.

## Validation scope

`tests/model_ops/test_qwen38_mono_hyperconnection.py` checks 1/2/4/8/16-token
mix+combine parity, 32/64/128-token combine parity, served `mix_fused` parity
at 4/8 tokens including the final mixer without injection, fallback, and
changed-input graph replay.
`tools/bench_qwen38_qsa_cp.py` checks exact selection and changed-input graph
replay on two or four GPUs; the additional 128k/256k measurements above used
the same QSA component with local sparse attention. Trained HC weights from
the official checkpoint were separately parity checked. The full checkpoint
is still incomplete locally, so whole-layer accuracy, generation accuracy,
and InferenceMAX throughput remain unmeasured. M3's one-launch sparse layer
also includes index selection, attention, both TP reductions and MoE; this PR
does not yet implement that layer-level fusion for Qwen's GDN/QSA and block-FP8
MoE. The 2–3× figures above are QSA component measurements, not full-model
speedups.
