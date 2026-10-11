# Qwen3.8 Flash-Next decode specializations

This work adds gfx950 FlyDSL kernels for the repeated hyper-connection (HC)
path, an all-visible short-context QSA selection, and an opt-in indexer-only
context partition for long QSA decode. These
are component specializations, not a single-launch decoder layer. This PR
adds no Triton HC kernel; the existing Triton HC path remains the fallback.
Long QSA still uses ATOM's existing paged scorer, AITER stable top-k, and
sparse attention; index CP partitions that scorer's work over the TP group
without partitioning the KV cache or GDN/PLE state.

## FlyDSL hyper-connection

Set `ATOM_QWEN38_MONO_HC=1` before starting ATOM. The default is off. Dispatch
requires gfx950, contiguous BF16 data and weights, FP32 norm weights, and the
Flash-Next geometry: HC=4, hidden=2560, low rank=320. The served decoder calls
`mix_fused`, which uses three FlyDSL launches at 4 and 8 decode tokens:
deferred combine plus grouped norm, fused down and injection projection, then
up projection plus gated mean. At 16 tokens it uses the same first two FlyDSL
launches, the existing AITER CK MFMA up GEMM, and a FlyDSL gated mean. It keeps
the existing two-launch path at 1–2 tokens and native GEMMs above 16 tokens.
The standalone `mix` and `combine` calls used by the MTP
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
| 16 | 26.29 | 20.97 | 1.25× | FlyDSL norm/down/mean, CK MFMA up |

These are HC component times with synthetic weights. The production path's
three launches at 4–8 tokens and four at 16 are part of a decoder layer with separate
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

At 4–8 rows, the down and up stages retain `v_dot2c_f32_bf16`; none of the
served FlyDSL stages spill. The norm stage recomputes the rounded combined values after its
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
100 buffer loads, and no scratch spills. The standalone `mix` up CTA uses 15 VGPR, 24 dot2
instructions, 26 buffer loads, eight exponentials and eight reciprocals, with
no LDS or spills. It takes 13.27 µs of the 28.52 µs pair; giving each up CTA
four rows raised VGPR to 25 and made the up stage slower (13.61 µs), so the
two-row schedule is retained for standalone `mix`. The served 16-row path
instead uses CK's existing MFMA up GEMM and a FlyDSL gated mean. The latter's
gfx950 ISA has 14 VGPR, zero LDS or scratch, eight buffer loads, four
exponentials and four reciprocals. That changed the isolated served HC from
26.29 to 20.97 µs and the complete 128k/16 QSA layer from about 276 to
265 µs. Direct FlyDSL mix probes at 32/64/128 rows took
38.0/67.8/142.9 µs versus 25.3/27.5/30.1 µs native; mix therefore falls
back above 16 rows while the fused combine remains enabled through 128.

## QSA when the context fits the sparse budget

Set `ATOM_QWEN38_QSA_SHORT=1` to use a one-launch FlyDSL selection for gfx950
decode when `max_seq_len <= indexer_budget` (2,048 tokens in this checkpoint).
The flag defaults off until full-model validation is available. Every visible
complete group wins top-k in this case, and stable top-k returns them in group
order. The kernel directly writes logical token IDs through the query position
and `-1` padding. The indexer's cache update still runs, so later long-context
steps see the same pooled keys. Unsupported dtypes, layouts, and devices use
the existing scorer and top-k path.

`python tools/bench_qwen38_qsa_short.py` measures synthetic selection and
selection plus local TP sparse attention as captured graphs on MI355X. Both
selected IDs and sparse outputs matched bit for bit; tests also cover ragged
and padded rows and changed-input graph replay. Times below are medians in µs:

| Context | Rows | Selection native → shortcut | Selection gain | QSA native → shortcut | QSA gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | 5.89 → 1.50 | 3.92× | 14.41 → 10.26 | 1.41× |
| 512 | 8 | 6.39 → 1.63 | 3.93× | 13.19 → 8.81 | 1.50× |
| 2,048 | 1 | 6.12 → 1.59 | 3.85× | 14.99 → 10.69 | 1.40× |
| 2,048 | 2 | 6.07 → 1.64 | 3.71× | 15.19 → 10.78 | 1.41× |
| 2,048 | 32 | 7.04 → 1.66 | 4.24× | 23.90 → 18.25 | 1.31× |

At 2,048 tokens with 32 distinct requests, QSA was 27.88 → 22.37 µs
(1.25×); at 48/64/128 distinct requests it was 35.64 → 29.63 µs (1.20×),
46.60 → 39.97 µs (1.17×), and 82.04 → 72.10 µs (1.14×). The selection
kernel's generated gfx950 ISA has 6 VGPR, zero LDS,
zero scratch, and three buffer loads and one buffer store per output thread.
These are QSA component timings, not decoder-layer timings.

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

The measured dispatch gates are 16/48 rows at the 128k score bucket for
TP4/TP2, and 2/16 rows at the 256k bucket. Smaller contexts retain native
selection. The TP4 128k gate was lowered from 24 to 16 after the complete
16-row layer improved from 290.69 to 275.73 µs when CP and HC were enabled;
its selected token IDs matched the unsplit path exactly. The other gates
include distinct, shared-request, ragged and padded rows; the native path
remains selected where the exchange can lose.

Captured graph times on MI355X with a distinct permuted request per row,
including both selection and local TP sparse attention. The 1/2/48/128-row
cases at both long contexts were rerun together with 100 graph replays per
case. The table reports the measured time per replay, in microseconds:

| TP | Context | Rows | Unsplit | Index CP | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 128k | 1 | 46.43 µs | 56.17 µs | 0.83× |
| 4 | 128k | 2 | 47.79 µs | 57.57 µs | 0.83× |
| 4 | 128k | 24 | 92.72 µs | 83.32 µs | 1.11× |
| 4 | 128k | 48 | 161.83 µs | 104.15 µs | 1.55× |
| 4 | 128k | 96 | 286.83 µs | 150.04 µs | 1.91× |
| 4 | 128k | 128 | 359.39 µs | 187.46 µs | 1.92× |
| 4 | 256k | 1 | 68.78 µs | 65.17 µs | 1.06× |
| 4 | 256k | 2 | 76.64 µs | 66.38 µs | 1.15× |
| 4 | 256k | 32 | 209.49 µs | 110.71 µs | 1.89× |
| 4 | 256k | 48 | 268.07 µs | 131.11 µs | 2.04× |
| 4 | 256k | 128 | 640.11 µs | 269.82 µs | 2.37× |
| 4 | 256k | 160 | 817.14 µs | 338.36 µs | 2.42× |
| 2 | 128k | 1 | 46.68 µs | 64.84 µs | 0.72× |
| 2 | 128k | 2 | 48.45 µs | 65.38 µs | 0.74× |
| 2 | 128k | 48 | 164.31 µs | 130.03 µs | 1.26× |
| 2 | 128k | 128 | 357.37 µs | 263.36 µs | 1.36× |
| 2 | 256k | 1 | 68.96 µs | 72.09 µs | 0.96× |
| 2 | 256k | 2 | 76.82 µs | 74.00 µs | 1.04× |
| 2 | 256k | 16 | 120.56 µs | 106.19 µs | 1.14× |
| 2 | 256k | 48 | 271.44 µs | 185.81 µs | 1.46× |
| 2 | 256k | 128 | 645.60 µs | 407.13 µs | 1.59× |

The new long-context run checks exact selected token IDs and sparse attention
outputs against the unsplit path on every 1/2/48/128-row case, then changes
queries and compressed keys and checks captured graph replay again. The
intermediate 16/24/32/96/160-row cases are from the earlier crossover run.
The production gate retains native selection at 128k/1–2 rows and at
256k/TP2/1–2 rows; TP4/256k/1 row gained only 1.06× in this run, so the
current two-row gate avoids a marginal dispatch. TP4 256k/513-row selection,
split into 512+1-row workspace
chunks, was exact under changed-input graph replay and took 2240.05 to
693.94 µs (3.23×) with `--cp-logits-workspace-mib 32`; sparse attention was
not included in that case. Ragged,
padded and all-zero tie cases were also exact at 128k/256k on TP2 and TP4.
The QSA CP path is separate from full PCP/DCP, whose Qwen backend guard
remains because those modes require partitioned cache, page metadata and
recurrent-state contracts.

## GDN recurrent decode on gfx950

ATOM now permits its existing AITER FlyDSL GDN decode path on gfx950, with the
same shape and layout checks used on gfx942. This changes only ATOM's dispatch
gate; it requires no FlyDSL source modification. The kernel consumes the
physical VK state layout used by Qwen decode and writes the next state in its
existing slot. Tests check BF16 and FP32 state, 1–128 rows, padded rows, and
changed-input graph replay against the native recurrent path. The output and
state NRMSE on synthetic BF16 cases were approximately 0.001–0.002.

`tools/bench_qwen38_gdn_decode.py` compares captured graph times of the GDN
**recurrent core** at Qwen's TP2/TP4 head geometry. Both sides receive
precomputed state indices and query starts. The convolution, projections,
output norm, HC, MoE, and TP communication are outside the timed region.

| State | TP | Rows | Native | AITER FlyDSL | Speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 4 | 1 | 5.21 µs | 2.28 µs | 2.28× |
| BF16 | 4 | 2 | 5.41 µs | 2.36 µs | 2.29× |
| BF16 | 4 | 48 | 13.66 µs | 7.69 µs | 1.78× |
| BF16 | 4 | 128 | 26.66 µs | 18.15 µs | 1.47× |
| BF16 | 2 | 1 | 5.37 µs | 2.35 µs | 2.29× |
| BF16 | 2 | 2 | 5.48 µs | 2.52 µs | 2.18× |
| BF16 | 2 | 48 | 20.77 µs | 14.25 µs | 1.46× |
| BF16 | 2 | 128 | 49.24 µs | 32.29 µs | 1.53× |

FP32 state
also improved at 1–2 rows; TP2/128 rows was approximately even at 1.02×.
The generated gfx950 one-row and 48-row variants use 69 and 123 VGPR, no LDS
allocation, and no scratch spills. These are core-only results, not full GDN
layer or model speedups.

## Complete decoder-layer measurements

`tools/bench_qwen38_full_layer.py` constructs a real ATOM QSA or GDN decoder
layer with HC, MoE, and both TP reductions on four MI355X GPUs using TP4 and
EP4. It uses a local model config and `load_dummy="xavier"`; no checkpoint
tensors are read or downloaded. It captures each complete layer as a GPU
graph and averages 100 replays. The baseline disables the Qwen HC, QSA short,
QSA index CP, and merged projection paths; GDN also uses the native recurrent
path. The tuned run enables eligible paths. PLE, which exists on only one of
the 48 model layers, is excluded. The numbers are layer latency, not model
throughput.

The new QSA projection path joins local QKV `[3584, 2560]` and replicated
index Q/K `[640, 2560]` weights into one block-FP8 GEMM. Both branches use
the same activation and dynamic quantization. A cached packed weight and
scale are built after loading; the original parameters remain available for
checkpoint semantics. Dispatch is opt-in with
`ATOM_QWEN38_QSA_FUSED_PROJ=1`, limited to gfx950 BF16 decode at 4/8/16 rows,
TP4's exact FP8 shapes, and preshuffled FP32 block scales. The index cache
update remains in place. The two projection outputs matched the unfused
outputs bit for bit in every complete-layer run. A separate graph microbench
measured 32.9 → 20.4 µs at 4 rows and 35.6 → 21.5 µs at 16 rows.

| Layer | Context | Rows | Baseline → tuned | Gain |
| :--- | ---: | ---: | ---: | ---: |
| QSA | 2k | 4 | 213.35 → 181.69 µs | 1.17× |
| QSA | 2k | 8 | 224.77 → 197.32 µs | 1.14× |
| QSA | 128k | 4 | 261.28 → 220.04 µs | 1.19× |
| QSA | 128k | 8 | 276.35 → 242.03 µs | 1.14× |
| QSA | 128k | 16 | 305.13 → 264.85 µs | 1.15× |
| QSA | 256k | 4 | 284.72 → 238.06 µs | 1.20× |
| QSA | 256k | 8 | 305.73 → 255.32 µs | 1.20× |
| QSA | 256k | 16 | 370.02 → 279.17 µs | 1.33× |
| GDN, FP32 state | — | 4 | 165.51 → 142.86 µs | 1.16× |
| GDN, FP32 state | — | 8 | 174.13 → 157.90 µs | 1.10× |
| GDN, FP32 state | — | 16 | 202.94 → 180.97 µs | 1.12× |

The QSA scorer now uses 256 rather than 64 column programs on gfx950 for
4/8-row, four-head, 128-dimension decode when a rank scores at least 16,384
compressed groups. It uses the existing scorer kernel and changes only its
grid. Full-layer measurements with the same synthetic setup showed tuned
128k/4, 128k/8, 256k/4 and 256k/8 latency move from approximately
231/248/244/268 µs to 220/241/238/255 µs. The scorer's score values and
the selected IDs are unchanged because each program still writes disjoint
columns. At 128k/16, 256 programs did not improve the complete layer; its
index-CP scorer remains at 64 programs. Before the new 16-row HC up schedule,
disabling CP with 256 scorer programs gave 278.20 µs versus 275.58 µs with
CP, so CP remains selected.
Independent PyTorch index top-k still matched at both contexts for 4 and 8
rows, with changed-input graph replay; sparse-attention NRMSE against the
reference was at most 0.0028.

At 128k/16, enabling index CP after merged projection improved the complete
layer from 290.69 to 284.45 µs. The original 16-row FlyDSL HC path reached
275.73 µs; the MFMA-up and FlyDSL-mean schedule reached 264.85 µs. The tuned
layer still spends about 131 µs in the QSA sublayer, 79 µs in MoE, 25 µs in
each HC stage, and 18 µs in the attention TP
reduction when those stages are captured separately. Those stage times need
not add to the complete-layer time because they use separate graphs and
inputs.

QSA selected token IDs were identical for the measured TP4 CP cases at
128k/16 and 256k/4/8/16, and merged projection outputs were exact. Complete QSA
layer stream/pending output NRMSE was at most 0.0028 on these synthetic
weights. GDN/4's recurrent state NRMSE was 0.00004 and attention output NRMSE
was 0.00148, but its deferred MoE output NRMSE was 0.032. The top-10 routed
experts were identical before and after tuning; the synthetic constant FP8
expert weights amplify small changes in the MoE input. At GDN/16, one row
had 9/10 routing overlap and the deferred MoE output NRMSE was 0.0048.
Checkpoint-weight
full-layer accuracy and generation remain unverified.

Direct GEMM replacement was rejected: for the QSA `(M,N,K)=(4/8,3584,2560)`
projection the existing CK path took about 14–15 µs, ASM 20–25 µs, and CK-Tile
34 µs with the same operands. The merged projection uses the existing CK
backend. No AITER or FlyDSL source change was made. The complete-layer gains
are below the requested 2× target; MoE, the QSA scorer, and the separate TP/HC
handoffs remain the main costs.

## Why the complete layer is still far from 2×

At 128k/8, the current complete layer is about 241 µs; the pre-tuning layer
was about 283 µs, a 1.17× improvement across those runs. A 2× layer would
need to finish near 141 µs, which means removing another roughly 100 µs.
Two separately captured HC stages take about 24 µs each, attention about
116 µs, attention TP reduction about 15 µs, and MoE about 77 µs. Those stage
graphs do not sum exactly to the complete layer, but show why removing one
quantization and GEMM cannot deliver a 2× layer. A single-replay kernel
profile of 128k/4 without index CP placed stable QSA top-k at about 24 µs,
the merged CK projection at 20 µs, MRoPE at 13 µs, and QSA scoring at 12 µs;
MoE's separate graph was about 71 µs. The scorer grid change saves a few
microseconds. The large remaining costs are selection, HC up projections,
MoE routing/experts, and cross-rank handoffs.
Forcing AITER's HIP top-k instead of its existing stable FlyDSL top-k raised
the 128k/4 tuned layer from about 220 to 237 µs; that backend switch was
rejected.

MiniMax-M3's sparse layer has one persistent launch (`mono/runner.py` and
`mono/kernels/post_attn.py`). Its fused K1 performs normalization, QKV/index
projection and cache insertion; K4 then selects, attends, projects, reduces
across TP, routes experts, runs MoE, and reduces across TP again through
mailboxes. It supports at most 16 rows per step. Its first three dense layers
still use a pre-kernel, native attention, and post-kernel. At 128k, M3 scores
1,024 index blocks of 128 tokens and keeps 16 blocks, or 2,048 attention
keys. Qwen scores 32,768 compressed groups of four tokens and keeps 512
groups, also 2,048 tokens. Qwen therefore ranks 32 times as many candidates
for the same sparse-attention budget. It also sums four per-head rectified
scores, while M3's TP4 rank owns one index head.

DeepSeek V4.1's `mono/runner.py` uses K1 (`attn_pre`) and a fused K2
(`layer_post`) per layer. K2 joins index selection, sparse attention,
output projection, the TP reduction, HC postprocessing, the MoE and its TP
reduction; its attention-to-MoE row is published through a mailbox without
a launch boundary. The compressor and Engram can still run between K1 and
K2. V4.1's selection ranks 512 winners through an in-kernel segment tree of
8,192-column segments (`mono/index_plan.py`); it is not a stand-alone AITER
top-k over a materialized Qwen score row. Its maximum step width is 48 rows,
versus M3's 16.

Qwen currently calls independent HC, QSA or GDN, collective, HC and MoE
operators from `Qwen4ExpDecoderLayer.forward`. The QSA path materializes
scores and selected IDs; index CP additionally exchanges candidates and
performs a second top-k. Its MoE has 512 routed experts, top-10 plus shared,
320-wide block-FP8 experts, and EP4. M3 has 128 routed experts/top-4 and
V4.1 has 384/top-6 with TP-sharded expert intermediates. Their expert and
communication stages cannot be reused directly for Qwen's EP layout.

A Qwen layer kernel needs its own row-width-specific expert schedule and
mailbox handoffs: index selection into sparse attention, attention output
into TP reduction and HC, then local expert work into the EP reduction.
The GDN path also needs its recurrent state writeback in the schedule. That
is the structural work needed to approach a 2× complete-layer result; it is
not implemented by the component kernels in this PR.

## MiniMax-M3 techniques applicable to Qwen

These M3 choices give concrete starting points for Qwen's own layer kernel:

| M3 choice | Qwen status and constraint |
| :--- | :--- |
| Skip scoring when every sparse block fits the budget | Added for Qwen contexts up to 2,048 tokens with exact IDs and sparse output. |
| Fuse indexer, sparse attention, output projection, TP reduction, router, and MoE | The main QKV and index Q/K projections now share one quantization and GEMM at 4/8/16 rows. The QSA scorer/top-k workspace, sparse attention, collective, and MoE remain separate. |
| Use a narrow MoE schedule for 1–4 rows and expert-grouped work from 5 rows | Open. Qwen has a single-token special case and AITER's general sorted MoE, but no persistent row-width schedule joining it to attention and HC. Its 512-expert top-10 block-FP8 format needs its own stage. |
| Change TP reduction from one-hop exchange to reduce-scatter/all-gather at wider batches | Open; M3 switches at 12 rows. Qwen's collective choice needs separate TP2/TP4 crossover measurements for its 2,560-wide output. |
| Reuse an earlier layer's selected blocks | Not applicable to Qwen's checkpoint: each of its 12 QSA layers owns a different indexer and compressed-key cache. Reusing another layer's IDs would change the model. |
| Give recurrent or special layers a seam | Qwen has 36 GDN layers and one PLE layer. Its existing GDN recurrent kernel is now enabled on gfx950; the convolution, projections, HC/MoE/TP stages still need a fused layer schedule. Like V4.1's two-part layer, a first implementation can keep the state update as a seam while fusing the HC/MoE/TP suffix; a true one-launch GDN layer must include its state transition. |

M3 also uses width-specific task placements, non-temporal loads for weights
that stream once, and a rank-shared mailbox buffer. Those are useful tuning
tools after the Qwen layer's math and state handoffs are bound; copying M3's
expert kernel or reusing its selected IDs would be incorrect.

## Validation scope

`tests/model_ops/test_qwen38_mono_hyperconnection.py` checks 1/2/4/8/16-token
mix+combine parity, 32/64/128-token combine parity, served `mix_fused` parity
at 4/8/16 tokens including the final mixer without injection, fallback, and
changed-input graph replay.
`tests/model_ops/test_qwen38_qsa_short.py` checks exact all-visible selection
for 64/512/2,048-token contexts with ragged/padded rows and changed metadata
under graph replay. `tools/bench_qwen38_qsa_short.py` compares selection and
selection plus sparse attention, including distinct-request batches.
`tools/bench_qwen38_qsa_cp.py` checks exact selection and changed-input graph
replay on two or four GPUs; the additional 128k/256k measurements above used
the same QSA component with local sparse attention. Trained HC weights from
the official checkpoint were separately parity checked.
`tests/test_aiter_flydsl_gdn.py` checks gfx950 recurrent GDN dispatch, state
writeback, padded decode, and graph replay.
The full checkpoint is incomplete locally. Synthetic full-layer timing
and numerical comparisons are reported above; checkpoint-weight full-layer
accuracy, generation accuracy, and InferenceMAX throughput remain unmeasured.
M3's one-launch sparse layer also includes index selection, attention, both TP
reductions and MoE; this PR does not yet implement that layer-level fusion for
Qwen's GDN/QSA and block-FP8 MoE. The 2–3× figures above are QSA component
measurements, not full-model speedups.
