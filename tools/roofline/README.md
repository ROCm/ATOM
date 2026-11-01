# tools/roofline — operator roofline from ATOM markers

A **thermometer** for "which operators are far from the hardware ceiling, and how
much time could plausibly be recovered". Magnitudes and bound categories are
meaningful; absolute numbers are not. It is not a simulator and not a substitute
for a profiler.

## Why markers

ATOM already labels every module with a `record_function` whose name carries the
operator identity, the layer index, and — for GEMMs — the problem shape and the
input/weight/output dtypes:

```
layers.18.attn.indexer.wq_b[M=1,N=8192,K=1536,a=torch.float8_e4m3fn,w=…,o=…]
```

The profiler emits the same label GPU-side as a `gpu_user_annotation` carrying a
real duration. **One trace therefore contains identity, shape, dtype and measured
time.** No kernel-name matching, no positional heuristics, no landmark anchors, no
hand-maintained shape tables — those all exist to reconstruct information the
marker already states.

Cross-checked against the HuggingFace config (DSV4-Pro, TP4): `K=7168` =
`hidden_size`; `ffn.gate N=384` = `n_routed_experts`; shared-expert
`gate_up N=1536` = `2 × 3072/4` and `w2 K=768` = `3072/4` = `moe_intermediate_size`
sharded by TP4; `attn.wq_b N=16384` = `128 heads × 512 head_dim / 4`;
`indexer.wq_b N=8192` = `index_n_heads 64 × index_head_dim 128`. Every shape in
the marker is independently derivable — the annotation is trustworthy.

## A marker names a module, not a kernel

This is the one thing to get right before reading any number. Measured on the
2026-07-30 c1/TP4 prefill trace, the `attn.wo_b` marker contains:

| | share |
|---|---|
| `ncclDevKernel_Generic_1` — the tensor-parallel all-reduce | **74.1%** |
| `ck::kernel_gemm_xdl_cshuffle_v3` — the actual GEMM | 24.2% |
| `aiter dynamic_per_group_scaled_quant` | 1.2% |
| a triton copy | 0.4% |

Charging the module's whole wall time against a pure-GEMM ceiling puts the wrong
number in the denominator and reports a 15x gap for a healthy GEMM. So kernels
inside a marker are classified (`comm` / `quant` / `gemm` / `other`) and **only the
matmul time is measured against the matmul ceiling**; the rest is reported in its
own column rather than hidden inside the efficiency. With that fix `attn.wo_b`
lands at ~3.7x, a plausible number for a block-scaled fp8 GEMM.

Two consequences worth stating plainly:

- The `comm_us` column is a finding in its own right — **at TP4 prefill, the
  attention output projection spends three quarters of its time communicating.**
- An annotation's own duration is a **span**, and how much of it is real GPU work
  varies by operator: `attn.wo_b` is 1.00 busy/span (the all-reduce fills it),
  `ffn.shared_experts.w2` is **0.04**. `measured_us` is therefore the sum of the
  enclosed kernels, never the annotation's `dur`. `span_us` is kept beside it for
  diagnostics — a low `busy_us / span_us` means the module is waiting, not working.

Kernel classification order is load-bearing: **gemm outranks quant**, because
ck_tile names its fused kernel `QuantGemmMultiD…` — a GEMM that quantizes inline,
not a quantizer. Getting that backwards charged `ffn.shared_experts.w2` only its
0.39 ms Tensile tail instead of its 3.76 ms main kernel and reported **300% of the
fp8 peak**. There is now a guard: any operator whose achieved TFLOP/s exceeds its
ceiling is flagged `over_peak` and shouted about in the run log, because exceeding
a ceiling is never a finding — it is always a bug on our side, either a
mis-classified kernel starving the denominator or a wrong peak.

## Where a cost comes from

| the marker | treatment |
|---|---|
| carries a shape annotation | priced from it: `2·M·N·K`, bytes for A+B+C |
| names an operator with a closed form | priced from the **model config** — see `formulas.py` |
| neither | left blank: measured time counted, no bound claimed |

The middle row is what makes the tool useful. A fused attention kernel has a
perfectly well-defined FLOP and byte count; what it lacks is an annotation saying
so. `formulas.py` states that count from the config instead — the marker still
supplies the identity (which operator, which layer, how many firings) and the
config supplies the dimensions.

Coverage of measured GPU time on a c1/TP4 run:

| | shape in marker | config formula | neither |
|---|---|---|---|
| prefill | 13% | +85% | 2% |
| decode | 19% | +76% | 5% |

What stays blank stays blank on purpose: layout copies whose byte count depends on
transient tensor shapes the config does not describe. A guessed formula there
would be worse than an honest gap.

**The fused operators are modelled by counting passes, not by hunting for a
shape.** `csa_core_attn` (the `fused_compress_attn` kernel) runs QK and AV in one
launch, so it takes 4 passes over the selected KV entries; `attn_logits` and
`attn_reduce` are the same scan at 2 passes each. The KV entry count is where the
two layer families diverge — CSA takes `min(kv_seq/4, index_topk)` after the
Lightning Indexer selects, HCA reads `kv_seq/128` densely.

## Three regimes, and one of them is neither ceiling

`bound` says which of the two ceilings is closer. It is the wrong question for an
operator that is too small to reach either. No kernel in the 2026-08-26 run runs
faster than **4.11 µs** however little work it does — roughly 1 µs of kernel
launch plus per-kernel fixed cost (cold-access latency, wave ramp on a kernel too
small to fill the machine). An operator whose entire roofline time falls below
that floor is reported as **latency**, not as "memory-bound, efficiency 0.00":

```
DECODE  (bs=1)   84.0% latency · 11.0% memory · 5.0% no formula
PREFILL (7237)   67.4% memory · 28.6% compute · 2.4% no formula · 1.5% latency
```

Decode at batch 1 is bound by neither bandwidth nor compute. The lever is batching
and fusion; a bandwidth argument about it would be answering the wrong question.
The floor is measured from the run itself, so a faster stack or a larger batch
moves it without a code change.

## A floor above its own measurement is a broken floor

A roofline is a lower bound, so a row measured *faster* than its own floor is
flagged and kept out of every total. Two very different things produce it:

* **The formula overcounts.** Usually a width error — the worst one in this run
  charged the HCA compressor's rope for all M tokens when it runs on the M/128
  compressed entries, 13× too high.
* **HBM was never the ceiling.** MI355X carries 256 MB of Infinity Cache and this
  roofline prices every byte at HBM bandwidth by construction, so an operator
  whose working set fits in cache legitimately beats the HBM roof.

Working-set size separates the two well enough to print, and collectives are
excluded from the cache explanation entirely — their bytes cross the fabric.

## Usage

```bash
T=<run_trace>.json.gz
C=<model>/config.json                 # the HuggingFace config, NOT hand-written
R=~/roofline-runs/deepseek-v4-pro/c1  # anywhere outside the repo

cd "$R" && python3 <atom>/tools/roofline/marker_roofline.py "$T" \
    --phase decode --capture-trace <capture>/bs_1_rank0.json.gz \
    --model-config "$C" \
    --model deepseek-v4-pro --framework atom --platform mi355x \
    --methodology deepseek-v4-pro \
    --tp 4 --dp 1 --dp-attn off --kv-cache-dtype fp8 --index-cache-dtype fp4

python3 <atom>/tools/roofline/render_report.py \
    roofline_deepseek-v4-pro_prefill_c1.json \
    --decode roofline_deepseek-v4-pro_decode_c1.json -o report.html
python3 <atom>/tools/roofline/selfcheck.py      # needs no trace
```

Flags are positional `trace` plus `--capture-trace`, `--model` / `--framework`
/ `--platform`, a `--methodology` validator, and an `--output` that derives its
own name, so a run can be named by what it is rather than by the caller.

**Run outputs do not belong in the repo.** They are derived from a trace, one is
about a megabyte, and they go stale the moment a build or a formula changes --
committing them would put an expired number somewhere it reads as a fact. Write
them anywhere convenient; the tool takes paths and assumes nothing. What IS in
the repo is hand-written: code, `configs/models/*.yaml`, `peaks/*.yaml` and
`module_map.yaml`. A model's own `config.json` is NOT vendored -- `selfcheck.py`
fetches it from HuggingFace and caches it under `~/.cache/atom-roofline/`,
because a copy of a model config inside a framework repo is a second source of
truth that nothing refreshes.

### The flags that carry meaning

| flag | why it exists |
|---|---|
| `--model` | Picks `configs/models/<model>.yaml` and is recorded in `identity`. The name is the ATOM benchmark catalog's `prefix`, so it is the same string the benchmark uses. |
| `--methodology` | Hard-fails when the layer taxonomy resolved from markers diverges from the declaration. Layer families come from marker CONTENT, which follows the model but can drift silently on a rename -- and a sparse-attention formula on a dense family is a category error. |
| `--tp` / `--dp` / `--dp-attn` | Stated, not guessed. The filename is the fallback and it is the weakest source in this tool; the benchmark catalog states `tp=8` while a trace name may say otherwise. |
| `--kv-cache-dtype` / `--index-cache-dtype` | No HF config carries a cache precision -- ATOM takes them as server flags, so the tool has to be told. |
| `--capture-trace` | Decode replays a CUDA graph and markers do not fire during replay, so decode identity comes from the per-bs capture file. Use the full capture (13 MB, thousands of launches), not a CPU-only export: check with `zcat <capture>.json.gz \| grep -c '"cat": "cuda_runtime"'`. |

Anything not stated is inferred from the trace filename, and `identity.source`
records which was which, per field. Run the server with `--mark-trace` and
`--torch-profiler-dir` or there are no markers to read.

Each run writes `.json`, `.csv` and `.xlsx`. The json is the only seam: it carries
its own `identity`, `peaks`, `layer_counts` and rows, so a report never needs the
model config to draw a chart about the model.

## Coverage is not accuracy, and there is no "how long prefill should take"

Three different numbers, and running them together overstates what the ceiling reaches
(measured on the 2026-07-30 c1/TP4 prefill trace):

```
measured GPU busy                       2231.8 ms
 |- in operators we can price             749.7 ms  (33.6%)
 |    |- GEMM kernels  <- the ceiling's real reach   401.8 ms  (18.0%)
 |    '- comm / quant inside the same markers        347.9 ms
 '- in gap-tier operators, no closed form           1482.1 ms  (66.4%)

sum of t_roofline over those GEMM kernels            110.9 ms   -> gap 3.62x
```

So the honest sentence is: **of 2231.8 ms measured, a matrix ceiling can speak to 402 ms,
and for that slice the floor is 111 ms.** The remaining 1830 ms has no floor at all.

A floor for 18% of the time is **not** a floor for the phase. Do not sum `t_roofline`
across the table and present it as what prefill "should" cost -- that silently prices the
other 82% at zero.

Full coverage does not change this. On a c1 decode run every operator has a cost
model, and the sum is still 1.7 ms against a 16.3 ms step -- because most of that
step sits on operators whose entire roofline time is below the per-kernel floor
the run itself measures. Table 1 therefore reports the step as four terms that
add up to the wall clock rather than as one ratio:

```
             work     issue    stall    idle        = wall
decode c1    1.73     7.81     6.10     0.64 ms       16.27 ms
              11%      48%      37%       4%
```

`work` is the roofline; `issue` is the per-kernel floor where the work does not
cover it; `stall` is achieved rate, dependency stalls and wave ramp-up; `idle` is
wall minus the sum of kernel durations. At this operating point the ceiling
explains 11% of the step, which is the finding, not a defect in the model.

## Ceilings

`peaks/mi355x.yaml`, theoretical peaks with **no derate**, so
`gap = measured / t_roofline ≥ 1` by construction. A large-GEMM gap of 1.1–1.3×
is normal and is not a finding. Achieved-vs-theoretical efficiency lives in the
gap on purpose rather than being baked into the ceiling.

The compute ceiling is **dtype-dependent** — one horizontal line per dtype, not
one line. Ridge points on MI355X: fp4 2517, fp8 629, bf16 315 FLOP/byte.
Consequence worth internalising: dropping precision **doubles** arithmetic
intensity but **quadruples** the fp8→fp4 ceiling, so lower precision moves an
operator *left* relative to its ridge. Low precision buys compute, not bandwidth.

## Not done yet

- **The collective model is standard-form, not read from AITER.** An all-reduce is
  priced as reduce-scatter + all-gather, `2 x message / N` per link. The previous
  one-stage model (whole message per link) was falsified by measurement --
  `ffn.combine_outputs` moves 103.7 MB in 970 us, which is 107 GB/s over a
  76.8 GB/s link, and no implementation beats its own wire. But two-stage is the
  textbook form, not something read out of `aiter.dist`; confirm it there before
  quoting a comm efficiency. (An earlier note here claimed the two collectives
  were different operations. They are not: TBO is off, both call
  `tensor_model_parallel_all_reduce`. The 2x spread between them is rank skew --
  an NCCL kernel blocks until every peer arrives -- which is a scheduling signal
  and stays in the gap.)
- **Prefill is one step, and it is the compile step.** 2000 ms wall against 497 ms
  busy — 25% occupancy. Every prefill per-step number here is warm-up, not steady
  state. Needs a trace with several prompts.
- **Layout copies stay blank.** `triton_poi_fused_as_strided_clone_*` and the
  smallest indexer glue kernels are 2% of prefill and 5% of decode. Their byte
  counts depend on transient tensor layouts the config does not describe; a guessed
  formula would be worse than the gap.
- **MAF.** `peaks.maf` is empty, so only the theoretical ceiling is drawn. Populate
  from an Empirical Roofline Toolkit sweep to get the second, achievable ceiling.

## Open, must verify before publishing a chart

**Resolved 2026-08-31** (AMD's MI355X product page): `matrix_fp8 = 5033` is the
DENSE figure (5.0 PF dense, 10.1 PF with 2:4 sparsity) and `matrix_bf16 = 2516` is
dense. `matrix_fp4` was 20133 and is now **10066** — MXFP4 peaks at 10.1 PF per
package, and the 20.1 PF figure in circulation is the 8-GPU platform total. Every
fp4 operator had been judged against a ceiling 2x too high.

Still open:

1. **fp8 GEMMs cluster at 0.22-0.28; the bf16 ones do not.**
   `attn.compressor.wkv_gate` (bf16) reaches 0.587 and `attn.indexer.weights_proj`
   (bf16, memory-bound) 0.39, while every fp8 GEMM — `wq_b` K=1536, `wo_b` K=4096,
   `wqkv_a` and `gate_up` K=7168, `w2` K=768 — lands in a tight band despite K
   differing by 9x. A band that tight across that much shape variation is a property
   of the path, not of the kernels. The sparsity explanation is now dead (the peaks
   are confirmed dense), which leaves one candidate: the trace was captured with
   `--level 0`, the setting this repo's own capture instructions mandate for V4-Pro
   to dodge an Inductor `cluster_dims` bug on AMD — so these are **untuned** kernels.
   Re-run one trace at the normal optimisation level before quoting any fp8
   efficiency from this table.

2. **`kv_seq_len` is derived, not measured.** ATOM's decode label is
   `decode[bs=1 tok=1 d=1]` and carries no context length, so it is reconstructed as
   prefill `ctx` plus half the decode steps (7237 + 866/2 = 7670 on this run) and
   printed with its derivation everywhere it is used. It scales the attention terms
   only. Adding `ctx=` to the decode label would remove the assumption entirely.

3. **Hash layers are invisible to markers.** The config declares 3; they carry the
   same operator signatures as their neighbours, so the CSA/HCA families absorb
   them — a CSA-family operator fires in 30 layers (29 CSA + 1 hash) and an
   HCA-family one in 31 (29 HCA + 2). The family map itself is 29/29/3 and matches
   the config; only per-operator `n_layers` shows the overlap. Reconciling the two
   is worth ~1.3% on a per-layer extrapolation.

## Maintenance contract

Three kinds of file, and they fail differently.

**`configs/models/<model>.yaml` — one per benchmark catalog entry.** Layer
families, the `methodology` expectation, and which marker map the model's
operators come from. Named for the catalog's `prefix`, so `--model glm-5-2-fp8`
is the string the benchmark already uses. Quantisation variants are separate
catalog entries and so separate files, but they inherit through `extends` rather
than repeating structure -- a precision does not change a layer taxonomy.

**`module_map.yaml` and `peaks/*.yaml` — shared data.** `stages` groups operators
for display, which is the same job whatever the model, so it stays here. Peaks are
per platform. Renaming a module in ATOM should cost one line, not a code change.

**`formulas.py` — cost models plus one marker map per architecture.** The formulas
are shared (a GEMM is a GEMM); the maps are not. `@mark_trace` lives in
`atom/model_ops`, but the name it emits comes from the caller's `prefix=`, so
`attn.wo_b` is DeepSeek-V4's word for an output projection and Llama will call it
something else. Adding a model is a new map, not new formulas.

Rules learned the hard way, each from a bug that shipped:

* **No catch-all kernel needle.** A marker names a MODULE and a module emits
  several kernels. `("attn.compressor.fused_compress_attn", None, csa_core_attn)`
  made an update-states kernel inherit a full core-attention scan -- 7.6x its
  measured time. Match the kernel, or return nothing.
* **Match the marker path first, the kernel second.** Marker paths survived a
  build that renamed kernels wholesale; kernel names did not.
* **A kernel name is a clue, not evidence.** `triton_poi_fused_add_all_reduce` is
  a local add, not a collective (27x). `QuantGemmMultiD` is a GEMM, not a
  quantiser. `fused_compress_attn` performs no attention. Read the source.
* **One operator, several backends.** The same GEMM reaches the trace as an aiter
  fused kernel, an aiter mono-tile kernel and rocBLAS/Tensile's `Cijk_...`, which
  contains no "gemm" at all. After adding a needle, sweep for the others -- the
  tool now reports unpriced GEMM-named kernels on every run so this is not left
  to eyesight.
* **A floor above its own measurement is a broken floor**, and gets flagged and
  excluded rather than published.

`selfcheck.py` is deliberately not named `test_*.py` and not under `tests/`, so
ATOM's CI does not collect it. Run it by hand; it needs no trace, only the model
config, which it takes from `--model-config`, then `~/.cache/atom-roofline/`,
then HuggingFace -- and if none of those works it EXITS rather than skipping the
33 values it reads from it. It once guarded those behind `if cfg:` and printed
"all checks passed" with a quarter of the suite silently absent. Every check in
it is a bug that reached a rendered page before it reached a check.

Gates before proposing a change:

```bash
black --check --target-version py310 tools/roofline/*.py
ruff check --target-version py310 tools/roofline/
python3 tools/roofline/selfcheck.py
```

`--target-version py310` is not optional: ATOM sets `requires-python = ">=3.10"`,
there is no `[tool.black]` section to default it, and an f-string that used a
backslash escape -- 3.12-only syntax -- once made the renderer fail to import on
the version the project actually targets.
