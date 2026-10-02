# Kimi-K3 MI455 B0 — Performance Optimization Record

This document is the review log for performance work derived from
[`Kimi-K3-455-wideEP.md`](./Kimi-K3-455-wideEP.md). It separates measured
results from hypotheses and keeps enough detail to reproduce each experiment
or turn a validated change into a focused PR.

Last updated: 2026-10-02.

## Scope and success criteria

Primary target: **AgentX agentic interactivity p90**.

Optimization target set on 2026-10-02:

- topology and workload are fixed to B0 Kimi-K3 DP16/EP16, AgentX con32;
- measured 900-second baseline: `2.9708 tok/s/user` p90 interactivity;
- required improvement: at least 50%, i.e. `>=4.456 tok/s/user`;
- con64/con128 results remain diagnostic only and are not optimization targets.

Secondary metrics:

- aggregate and per-GPU total token throughput;
- effective concurrency;
- TTFT and inter-token latency;
- prefix-cache read rate;
- error rate, metric coverage, and 16-rank stability.

A candidate is accepted only if:

1. all 16 ranks initialize and remain alive;
2. KV budget is positive;
3. AgentX reports `submission_valid=true`;
4. request errors remain at or below 10%;
5. the primary metric improves in a same-duration, same-seed comparison.

The next scheduler sweep changes one dimension at a time:

```text
long_prefill_token_threshold:     0 -> 4096 -> 2048
ATOM_PREFILL_DELAYER_TARGET_FILL: 0.9 -> 0.5 -> 0.25
ATOM_PREFILL_DECODE_INTERVAL:       0 -> 1 -> 2 -> 4
max_num_batched_tokens:         16384 -> 8192 -> 4096  # only after the above
```

The intent is to bound how long prefill can block decode. TTFT and total
throughput may regress, but error rate, coverage, and the rank-health gates do
not change. `long_prefill_token_threshold` is tested first because it limits
one request's contribution without also changing KV budget, graph sizing, or
the MegaMoE communication arena.

## PR review summary

The following claims are supported strongly enough for review:

1. **Enable Mori V2 fused in the MI455 B0 recipe.** Its isolated fixed-shape
   A/B improved median output throughput by `24.8%`, reduced TPOT by `12.9%`,
   and reduced TTFT by `41.8%`. This is a recipe/configuration change; the
   implementation already exists.
2. **Guard K3 ptpc SiTUv2 fusion by the AITER shape limit.** The shipping
   gfx1250 kernel aborts for `D > 16376`; K3 dense MLP uses `D=33792`, while
   shared experts use `D=6144`. The candidate uses the safe non-fused path for
   the dense layer and preserves the `1.80-1.87x` activation-plus-quant kernel
   speedup for shared experts. This is a correctness fix with retained local
   performance benefit, not yet an AgentX E2E claim.
3. **Add the B0 reproduction record and health gates.** The launch/AgentX
   harness pins the validated DP16/EP16 con32 setup, rejects missing PR #2380
   support and obsolete AITER combine APIs before model allocation, and keeps
   machine credentials out of the repository.

Do not submit the local `merge_attn_states` change as a new fix: the identical
change is already merged upstream as ATOM PR #2378. It appears in this
worktree only because the recipe branch predates that merge and should
disappear when the branch is rebased onto current `main`.

The following remain experimental and should not be advertised as E2E gains:

- the production-safe BF16 table has strong isolated kernel results
  (`64/75` shapes at least 3% faster, `58.5%` median), but no con32 AgentX A/B;
- `ptpc_fp8` passes its single-GPU operator stack, but still needs full-EP16
  GSM8K and AgentX validation;
- checkpoint `interval=-1,demand=0` passes 259 functional tests, but has no
  measured cache-hit or AgentX benefit;
- MXFP8/MXFP4 combine passes the single-rank gfx1250 accuracy gate only on
  AITER #5176 or newer; it still needs a physical multi-rank wire test;
- scheduler threshold/delayer candidates have no valid four-node result yet.

## Test environment

| Item | Value |
|---|---|
| Nodes | C13 `.14`, C14 `.25`, C16 `.12`, C17 `.18` |
| GPUs | 4 nodes × 4 MI455 B0, 432 GiB/GPU |
| Fabric | One PPOD/VPOD, UALink active |
| Image | `rocm/fw-bringup:gfx1250-atom-20260918-ep8` |
| Parallelism | TP1, DP16, EP16, DP attention |
| Model | `/mnt/k3/Kimi-K3`, 96 shards on every node |
| ATOM branch | `recipe-k3-455-wideep` |
| Base branch commit at start | `dd8e3da91` |
| Required runtime patch | ATOM PR #2380 |
| AgentX | aiperf 0.12.0, seed 42, public Weka trace dataset |

## Hardware roofline and optimization targets

Source: `amd-smi static` and `rocminfo` on C7 B0.

| Hardware item | Value |
|---|---:|
| Compute units | 256 |
| Maximum gfx clock | 2.4 GHz |
| HBM bus width | 24,576 bit |
| Reported maximum HBM bandwidth | 23,347 GB/s |
| HBM capacity | 442,368 MB |

Derived matrix-compute engineering roofs use 8,192 BF16 FLOPs/CU/cycle and
the expected 2×/4× FP8/FP4 rate multipliers:

| Datatype | Derived theoretical peak | Promotion target |
|---|---:|---:|
| BF16 | 5.033 PFLOPS | >=85% = 4.278 PFLOPS |
| FP8 | 10.066 PFLOPS | >=80% = 8.053 PFLOPS |
| FP4 | 20.133 PFLOPS | >=75% = 15.100 PFLOPS |
| HBM streaming | 23.347 TB/s | >=80% = 18.678 TB/s |

These compute figures are derived engineering roofs, not a board marketing
claim. The best large-M Opus BF16 tuner result was 2.299 PFLOPS, or only
**45.7%** of the derived BF16 roof. Large-M GEMM therefore remains a primary
optimization target.

Optimization policy:

- a compute-bound GEMM already above its target is not tuned further unless it
  can be fused with a neighboring kernel;
- a pure copy/elementwise kernel should reach at least 80% of HBM roof or be
  eliminated;
- communication kernels are judged against measured UALink transfer roof, not
  HBM or matrix FLOPS;
- no operator result is promoted unless the projected and measured con32
  AgentX p90 interactivity reaches the E2E target.

The decode profile's largest single category, elementwise/copy at 26.15%, has
an Amdahl ceiling of only 1.35× even if removed completely. A 1.5× E2E target
therefore requires scheduler improvements plus reductions in more than one of
MoE communication, routing, quantization, or copies.

## Reproduction scripts

The parameterized scripts used by this work are:

- [`../experiments/kimi_k3_b0/launch_server.sh`](../experiments/kimi_k3_b0/launch_server.sh)
- [`../experiments/kimi_k3_b0/run_agentx.sh`](../experiments/kimi_k3_b0/run_agentx.sh)

`launch_server.sh` keeps the validated architecture, attention, MoE, and
communication environment fixed. The experiment variables are passed through:

- `MAX_NUM_SEQS`
- `MAX_NUM_BATCHED_TOKENS`
- `GPU_MEMORY_UTILIZATION`
- `ENABLE_PREFIX_CACHING`
- `FAKE_EPLB`
- `MORI_FUSED`
- `SESSION_AFFINITY`

No credentials are stored in either script.

## Configurations

### Accuracy / fixed-length baseline

- `max_num_seqs=8`
- `max_num_batched_tokens=2048`
- `gpu_memory_utilization=0.90`
- prefix caching off
- fake EPLB off

### AgentX baseline

- `max_num_seqs=8`
- `max_num_batched_tokens=16384`
- `gpu_memory_utilization=0.94`
- prefix caching on
- `ATOM_DP_SESSION_AFFINITY=1`
- fake EPLB on
- `ATOM_MORI_V2_FUSED=1`

Measured startup budget:

```text
total_gpu=432.00GB
utilization=0.94
budget=406.08GB
peak_torch=236.08GB
cudagraph_est=9.40GB
available_for_kv=118.81GB
```

## Fixed-length serving sweep

Workload: 1,024 input tokens, 256 output tokens, one full concurrency wave,
ignore EOS. Profiler disabled.

| Concurrency | Output tok/s | TPOT mean | TTFT mean |
|---:|---:|---:|---:|
| 16 | 449.61 | 33.72 ms | 508.96 ms |
| 32 | 861.60 | 33.59 ms | 936.93 ms |
| 64 | 1,583.01 | 35.23 ms | 1,358.90 ms |
| 128 | 2,758.97 | 39.03 ms | 1,911.10 ms |

Result: con16 → con128 increased output throughput by **6.14×** while TPOT
increased by **15.8%**. The engine has substantial synthetic decode batching
headroom. This does not prove AgentX will create enough simultaneous decode
work; AgentX effective concurrency must be measured separately.

## Torch profiler results

### Long-prefill profile

Workload: 14,336 input / 32 output, concurrency 32, 32 requests.

Result:

```text
32/32 successful
duration=12.04s
output=85.08 tok/s
total=38,202.37 tok/s
TTFT mean=8,277.70ms
TPOT mean=72.32ms
```

Sixteen rank traces were produced. Average summed GPU kernel time per rank was
8.15s.

| Kernel category | GPU-time share |
|---|---:|
| Elementwise / copy | 38.13% |
| MoE dispatch | 16.87% |
| BF16 bandwidth GEMM | 12.18% |
| MoE combine | 6.30% |
| MXFP4 expert GEMM | 5.78% |
| Attention / KDA | 5.68% |
| Norm / quant | 5.25% |
| rocPRIM scan / sort | 2.65% |

Direct operator attribution showed the generic `fill`, `copy`, `ne`, `sub`,
`cumsum`, and `where` work concentrated in fused-MoE routing and KDA buffer
preparation.

### Decode-focused profile

Workload: 1,024 input / 256 output, concurrency 16, 16 requests.

Result:

```text
16/16 successful
duration=17.99s
output=227.70 tok/s
TTFT mean=8,058.08ms   # profiler overhead included
TPOT mean=37.11ms
```

Sixteen rank traces were produced. Average summed GPU kernel time per rank was
12.03s.

| Kernel category | GPU-time share |
|---|---:|
| Elementwise / copy | 26.15% |
| BF16 GEMM / reduce | 17.54% |
| MoE communication | 12.48% |
| Norm / quant | 11.61% |
| Attention / KDA | 9.68% |
| MXFP4 expert GEMM | 8.54% |
| MoE routing | 8.21% |
| Activation | 2.30% |
| rocPRIM scan / sort | 2.04% |

The largest single decode kernel was the gfx1250 bandwidth-bound BF16 GEMM.
Logs also showed missing tuned configurations for small-M decode shapes,
including `M=2, N=163840, K=7168`.

### Profiler-only instability

Long con32/con64 trace collection triggered `double free or corruption` in a
ModelRunner. The same con32 serving workload completed 32/32 with profiling
disabled:

```text
output=861.60 tok/s
TPOT=33.59ms
```

Conclusion: this failure is currently classified as a profiler / trace-pressure
issue, not a normal serving failure.

Trace artifacts are retained on each test node:

```text
/root/k3ep16-logs/profile_prefill_con32
/root/k3ep16-logs/profile_decode_con16
/root/k3ep16-logs/profile_server_con64_crash.log
/root/k3ep16-logs/profile_server_decode_con32_crash.log
```

## AgentX sweep

Common settings:

- Config B above;
- aiperf duration 900s for tuning;
- warmup requests per lane 3;
- seed 42;
- same public dataset;
- server restarted between runs to clear cache state.

### Results

| Metric | con32 | con64 |
|---|---:|---:|
| Submission valid | yes | yes |
| p90 interactivity | **2.97 tok/s/user** | 0.91 tok/s/user |
| Total throughput | 2,157.64 tok/s | **3,982.39 tok/s** |
| Per-GPU total throughput | 134.85 tok/s | **248.90 tok/s** |
| Effective concurrency avg | 7.05 | **12.75** |
| Effective concurrency p90 | 11 | **22** |
| TTFT p90 | **185,935 ms** | 267,015 ms |
| ITL p90 | **1,659 ms** | 2,167 ms |
| Prefix-cache read | 89.46% | **94.22%** |
| Request error rate | 3.45% | not emitted; no error summary |
| TTFT coverage | 82.89% | 49.31% |
| ITL coverage | 100.00% | 98.45% |

Interpretation:

- con64 increased total throughput by 84.6%;
- con64 reduced the primary p90 interactivity metric by 69.3%;
- con32 is the current winner for the requested interactive AgentX objective;
- the sampled trace mix differed: con64 completed longer prompts on average, so
  the exact magnitude is not a pure engine-only effect.

Artifacts:

```text
/root/agentx/artifacts-opt/baseline-c32-d900
/root/agentx/artifacts-opt/sweep-c64-d900
```

## con128 failure and fix

### Failure

The pre-fix con128 AgentX warmup drove long cached-prefix prefill. C16 was the
first node to fail:

```text
atom/model_ops/attentions/triton_merge_attn_states.py:54
RuntimeError: Triton Error [HIP]: Code: 200,
Messsage: device kernel image is invalid
```

The failing request had approximately 15,544 new tokens and more than 122k
cached tokens. After the first ModelRunner exited, Gloo/TCPStore peers closed
and the whole 16-rank service shut down.

### Root cause

The image declared `prefill_tokens_with_context` as `tl.constexpr`. This
compiled a distinct merge kernel for each batch token count inside the serving
window. Under con128, the compile storm eventually produced a code object that
failed at `load_binary`.

ATOM upstream commit:

```text
acda72f0d fix(mla): pass merge_attn_states'
           prefill_tokens_with_context at runtime (#2378)
```

changes the argument to a runtime integer and explicitly marks it
`do_not_specialize`.

### Backport

Backported in:

```text
atom/model_ops/attentions/triton_merge_attn_states.py
```

The patch is intentionally identical to upstream #2378.

### Validation completed

1. `tests/test_merge_attn_states.py`: **22 passed** on MI455.
2. Cold-cache production-shape test:
   `T=15544, H=128, D=128`.
3. Runtime context counts tested:
   `15544`, `15543`, `7772`, `0`, and `15544` again.
4. Every case completed, synchronized, and produced no NaN.
5. A 131,072-token serving request completed twice:
   `TTFT=29.76s`, total throughput `4,404 tok/s`.
6. A patched con128 warmup ran for 15 minutes without reproducing
   `device kernel image is invalid`.

### Validation still required

- complete a full con128 AgentX warmup and profiling window;
- verify the post-patch GPU page fault described below is not caused by another
  software kernel;
- compare in-window merge-kernel compile counts before and after the backport.

### Post-backport con128 blocker

The original Triton code-object failure did not recur after the backport.
However, the patched con128 warmup exposed a different failure on C17 GPU2:

```text
GCVM no-retry page fault
Faulty UTCL2 client ID: TCP
PERMISSION_FAULTS: 0x5
Process python3 ... ModelRunner exitcode=-6
```

There was no `merge_attn_states` traceback and no
`device kernel image is invalid` in this run. The rank aborted roughly
96 seconds after the page fault, then the remaining ranks failed their Gloo
state synchronization.

This is tracked separately. The evidence is consistent with a GPU virtual
memory/runtime fault, but it is not yet proven to be hardware-only; a different
kernel issuing the bad access remains possible.

## External scheduling interference

The B0 rack is shared. An external DSV4 campaign periodically deleted the Kimi
containers and created its own containers. Any interrupted run is excluded
from comparisons.

One MORI fused baseline screen produced two complete samples before external
takeover, but their large variance means they are not used for an A/B decision.

## Mori fused A/B

To remove prefix-cache bias, this screen disabled prefix caching and kept every
other server variable fixed. Workload: 1,024 input / 256 output, concurrency
128, 128 requests. Each mode used a fresh server and three repetitions.

| Mode | Output tok/s runs | Median output tok/s | Median TPOT | Median TTFT |
|---|---|---:|---:|---:|
| `ATOM_MORI_V2_FUSED=1` | 2266, 2565, 2568 | **2565** | **41.33 ms** | **2.22 s** |
| `ATOM_MORI_V2_FUSED=0` | 2051, 2060, 2056 | 2056 | 47.45 ms | 3.81 s |

The first fused run included cold JIT effects; the next two converged. The
non-fused runs were internally consistent.

Keeping the fused path provides:

- **+24.8%** median output throughput;
- **-12.9%** TPOT;
- **-41.8%** TTFT.

Decision: retain `ATOM_MORI_V2_FUSED=1`.

## Single-node small-M BF16 GEMM tuning

The decode profile and con32 failure logs showed an untuned projection:

```text
M={1,2,4,8,16}, N=163840, K=7168
dtype=bf16, output=bf16, bias=false
```

These shapes were tuned on one C14 GPU with
`csrc/gemm_a16w16/gemm_a16w16_tune.py`. The tuner evaluated 2,095 tasks across
Triton and 417 Opus candidates per shape. FlyDSL had no gfx1250 candidates and
the image had no gfx1250 ASM catalog for this family.

Initial tuner selections:

| M | Selected backend | Tuner time |
|---:|---|---:|
| 1 | Opus kid 20201 | 152.19 µs |
| 2 | Triton default | 151.56 µs |
| 4 | Opus kid 20012 | 151.95 µs |
| 8 | Opus kid 20012 | 152.96 µs |
| 16 | Opus kid 20004 | 156.39 µs |

Production-operator validation rejected the full table:

- M=1's selected kid was not in the default production Opus module;
- M=2 was effectively unchanged;
- M=8 improved less than 3%;
- M=16 regressed.

M=4 was the only plausible candidate. It was rerun in one persistent container
with interleaved default/tuned measurements, 50 warmups and 200 iterations:

| M=4 run | Default | Tuned |
|---:|---:|---:|
| 1 | 155.11 µs | 152.73 µs |
| 2 | 155.12 µs | 152.87 µs |
| 3 | 154.86 µs | 152.56 µs |
| Median | **155.11 µs** | **152.73 µs** |

The repeated production gain was **1.53%**, below the 3% promotion threshold.

Decision: **do not add a tuned GEMM CSV**. This avoids shipping an uncompiled
M=1 kid and regressions for M=16 for a sub-threshold M=4 gain.

### Large-M prefill BF16 GEMM

The most expensive recurring untuned prefill projection was screened at:

```text
M={8192,16384}, N=67584, K=7168, bf16 -> bf16
```

The initial tuner selected gfx1250 Opus kid 21316 with zero numerical error:

| M | Default Triton tuner time | Opus tuner time | Tuner speedup |
|---:|---:|---:|---:|
| 8192 | 3,970.33 µs | 3,452.29 µs | 1.15× |
| 16384 | 8,138.19 µs | 7,197.64 µs | 1.13× |

After correcting the BF16 roof to 5.033 PFLOPS, the 8192 shape was retuned
with hipBLASLt enabled and an expanded Opus catalog:

| Backend | Time | Throughput | BF16 roof |
|---|---:|---:|---:|
| Opus kid 21317 | **3,232.14 µs** | **2.456 PFLOPS** | **48.8%** |
| Opus kid 21316 | 3,286.46 µs | 2.415 PFLOPS | 48.0% |
| Default Triton | 3,970.33 µs | 1.998 PFLOPS | 39.7% |
| hipBLASLt best | 20,233.05 µs | 0.392 PFLOPS | 7.8% |

The expanded Opus candidate is 22.8% faster than default Triton at the kernel
level, but it remains well below the 85% BF16 roof target. Since BF16
GEMM/reduce accounts for 17.54% of decode-profile GPU time, its isolated Amdahl
ceiling is approximately a 4% E2E improvement at this measured kernel delta.

This is a high-potential candidate because the shape is common in 16k chunked
prefill and the tuner delta exceeds 3%. It is **not a default yet**:
`gemm_a16w16_tune.py --run_config` uses ROCTracer, and both default and tuned
production-operator runs triggered a gfx1250 no-retry page fault before
reporting a result. The direct tuner path remained stable.

Decision: promote the stable, default-compiled Opus kid 21316 to a rack A/B
candidate, not to production defaults. The two-shape experiment file is:

```text
experiments/kimi_k3_b0/kimik3_bf16_gfx1250_candidate.csv
```

Validate by injecting it through `AITER_CONFIG_GEMM_BF16` in a real con32
serving A/B, where the normal server path does not enable ROCTracer. The faster
kid 21317 remains excluded until its production code-object availability is
proven.

### Full hot-shape BF16 sweep

Five recurring `(N,K)` families were swept over
`M={1,2,4,...,16384}`: 75 shapes total, using four GPUs and comparing Triton,
Opus, FlyDSL, torch, and hipBLASLt where each backend had candidates.

Raw tuner selection chose 73 Opus rows and two Triton rows. Before promotion,
the result was filtered against gfx1250's 228-kid default production compile
set. Any selected Opus kid outside that set was replaced by the fastest
numerically valid, deployable candidate for the same shape.

Production-safe result:

```text
75 shapes
69 default-compiled Opus
6 Triton
34 raw selections replaced because their kid was not production-compiled
64/75 shapes improve >=3% over default Triton
median kernel gain: 58.5%
maximum kernel gain: 248.9%
all selected rows pass the tuner error threshold
```

Candidate file:

```text
experiments/kimi_k3_b0/tuning/k3_bf16_hot_gfx1250_production_safe.csv
```

Using the decode profile's 17.54% BF16 GEMM/reduce share, a uniform 1.585×
category speedup has an Amdahl projection of approximately **6.9% E2E**.
Actual gain depends on the runtime shape distribution and requires the fixed
DP16/EP16 AgentX con32 serving A/B.

## Additional single-node candidate screening

### KDA prefill output rebind

Candidate: replace the prefill-only
`out.copy_(kda_out.squeeze(0))` with a view rebind before `rmsnorm_gated`.

Production-shape copy microbenchmark:

```text
shape=(16384, 56, 128), bf16
copy median=0.0687ms
read+write=0.4698GB
effective bandwidth=6.84TB/s
69-layer saving=4.74ms per prefill chunk
```

The actual `rmsnorm_gated` consumer was bit-identical and did not mutate the
input. A dedicated GPU regression passed. The measured end-to-end ceiling is
still below the 3% promotion threshold, so the source change and test were
removed rather than carrying an unpromoted micro-optimization.

### Split DP gathers instead of cat/split copies

The eager MoE DP gather concatenates hidden states and router logits before one
collective, then materializes both strided splits. Replacing it with two
collectives would remove the local copies.

Single-GPU price of the removable work:

```text
local_tokens=2048
gathered_tokens=32768
hidden=1792
router=896
median local copy work=0.0758ms/layer
61-layer saving=4.62ms per prefill chunk
```

Source-path review subsequently showed that this block is not reached by the
validated configuration. `dp_gather_hidden_and_router` requires
`not use_all2all_kernels`, while DP16/EP16 selects the Mori all-to-all backend.
The measured local-copy cost is real for the collective fallback but irrelevant
to this serving path.

Decision: close this candidate; do not schedule a rack A/B unless Mori is
disabled deliberately.

### Fused routed RMSNorm + MXFP4 quant

`ATOM_USE_TRITON_GEMM=1` disables the routed RMSNorm+MXFP4 quant fusion because
the Triton FP4 path selects an M-dependent scale layout. A single-GPU operator
comparison measured:

| M | Separate RMSNorm + quant | Fused | Speedup | Saved/layer |
|---:|---:|---:|---:|---:|
| 1 | 29.44 µs | 18.61 µs | 1.58× | 10.84 µs |
| 2 | 30.67 µs | 18.87 µs | 1.63× | 11.80 µs |
| 4 | 30.41 µs | 18.57 µs | 1.64× | 11.84 µs |
| 8 | 30.67 µs | 18.93 µs | 1.62× | 11.74 µs |
| 16 | 30.61 µs | 18.71 µs | 1.64× | 11.90 µs |
| 32 | 30.13 µs | 18.65 µs | 1.62× | 11.48 µs |

The paths are not bit-identical: approximately 0.6-1.3% of packed FP4 value
bytes differed in this random-input screen. The current fused kernel also emits
one scale layout while the Triton consumer changes layout at M=32.

A prototype added the M-dependent layout and benchmarked the complete
RMSNorm+quant+Triton-GEMM pair:

| M | Full-pair speedup | Saved | Output mismatch | Max absolute diff |
|---:|---:|---:|---:|---:|
| 1 | 1.031× | 29.8 µs | 27.18% | 7.56 |
| 2 | 1.026× | 25.1 µs | 16.07% | 8.09 |
| 4 | 1.032× | 30.0 µs | 24.53% | 14.38 |
| 8 | 1.030× | 27.8 µs | 28.23% | 15.50 |
| 16 | 1.036× | 33.7 µs | 20.52% | 13.72 |
| 32 | 1.027× | 25.9 µs | 26.12% | 19.00 |

`Output mismatch` uses `rtol=5%, atol=0.5` on the bf16 GEMM output. The complete
pair improved only 2.6-3.6% while producing unacceptable numerical drift.

Decision: reject and revert the prototype. Do not flip
`ATOM_USE_TRITON_GEMM` or enable this fusion. Any future attempt needs a
numerically aligned fused quant kernel and must pass GSM8K before performance
promotion.

### FP16 KDA temporal state

`ATOM_GDN_SSM_DTYPE=fp16` halves temporal-state storage and state traffic while
the recurrence still accumulates in fp32. The existing fused KDA decode kernel
was benchmarked at the production `H=56, K=V=128` shape:

| Decode batch M | FP32 state | FP16 state | Speedup | Saved |
|---:|---:|---:|---:|---:|
| 1 | 33.77 µs | 33.95 µs | 1.00× | -0.18 µs |
| 2 | 34.83 µs | 34.29 µs | 1.02× | 0.54 µs |
| 4 | 34.71 µs | 33.65 µs | 1.03× | 1.06 µs |
| 8 | 39.02 µs | 34.73 µs | 1.12× | 4.29 µs |

Output mismatch was 0% under `rtol=5%, atol=0.5`; max absolute differences were
0.043-0.050 on this one-step random test. Long recurrent accumulation across
69 KDA layers remains an accuracy risk.

Decision: do not promote for the measured AgentX operating point, which is
usually about one decode sequence per rank. Retain as a high-batch or
state-memory-pressure candidate, gated on full GSM8K.

### gfx1250 MoE tuner coverage gap

The production profile uses gfx1250 MegaMoE kernels such as
`a8w4_tdm_fp4_*_e56`. Three AITER tuning paths were tested against the actual
DP16/EP16 K3 shape (`model_dim=3584`, `inter_dim=3072`,
`experts_per_rank=56`, `topk=16`):

| Tuning path | Result |
|---|---|
| normal `gemm_moe_tune.py` | `QuantType.per_1x32 is not supported on gfx1250` |
| A4W4 blockscale tuner | `tuning is not supported in this chip gfx1250` |
| `--mxfp4-flydsl` | auxiliary codegen emitted illegal DPP broadcast instructions for gfx1250 |

No candidate was produced. This is not evidence that the production kernel is
optimal; it is a tooling/codegen coverage gap. Tuning the profiled MegaMoE
GEMM1/GEMM2 kernels requires adding a gfx1250-specific tuner for the TDM
implementation or fixing the MXFP4 auxiliary code generator before another
search.

## MI355/gfx950 recipe portability audit

The MI355 AgentX recipe was diffed against the B0 DP16/EP16 launch and the
source gates behind each option.

### Con32 is not a DSpark comparison

The MI355 con32 throughput band disables speculative decoding. Its advantage
over the MI455 con32 baseline therefore comes from topology, cache, precision,
scheduler, and kernel differences:

| Setting | MI355 con32 | MI455 B0 con32 baseline |
|---|---|---|
| GPUs | 8 | 16 |
| Parallelism | TP8, DCP8 | TP1, DCP1, DP16/EP16 |
| Speculative decode | off | off |
| Prefix tier | GPU + 128 GiB LMCache | GPU only |
| `max_num_seqs` | 64 | 8 |
| `max_num_batched_tokens` | 8192 | 16384 |
| GPU utilization | 0.90 | 0.94 |
| Online quant | `ptpc_fp8` | none |
| State checkpoint interval | -1 | 8192 |
| State checkpoint demand | off | on |
| MLA | gfx950 AITER/FlyDSL, block 128 | gfx1250 Triton MLA, block 16, unfused gather fallback |

DCP8 is the largest structural difference for long contexts and is not
portable to the TP1 topology. The current no-DSpark optimization stack should
therefore focus on the portable rows: prefill scheduling, ptpc_fp8, checkpoint
policy, combine wire, and tuned gfx1250 GEMMs.

### High-value portable candidates

1. **DSpark speculative decoding, DCP=1.** The gfx950 assertion applies only
   when speculative decode is combined with `decode_context_parallel_size>1`.
   The B0 topology is TP1/DCP1, so DSpark itself is not architecture-gated.
   With forced acceptance length 3.0, it is the only single feature expected to
   exceed the 50% p90-interactivity target. Required first gate: a q>1 verify
   pass through the gfx1250 Triton MLA + PR #2380 fallback.
2. **`ptpc_fp8` online quantization.** No gfx1250 architecture gate. It targets
   attention, dense MLP, and shared experts while leaving routed MXFP4 experts
   excluded. Estimated 5-12%; must pass GSM8K.
3. **ReplaySSM and checkpoint policy.** `ATOM_ENABLE_REPLAYSSM=1`,
   `--state-checkpoint-interval-tokens -1`, and
   `ATOM_STATE_CHECKPOINT_DEMAND=0` are architecture-neutral co-requisites for
   speculative verify and long-context state-pool capacity.
4. **Prefill scheduling.** `--long-prefill-token-threshold`,
   `ATOM_PREFILL_DELAYER_TARGET_FILL`, and
   `ATOM_PREFILL_DECODE_INTERVAL` directly target the ~50x gap between
   synthetic TPOT and AgentX ITL. They are expected to contribute more to p90
   interactivity than any individual kernel.
5. **`ATOM_MEGA_COMBINE_WIRE={fp8,fp4}`.** Implemented on the active
   `ATOM_MORI_V2_FUSED=1` path and guarded to prefill/ragged steps. It requires
   ROCm/AITER #5176 (`22ab77eb19d3`) or newer; the current 2026-09-18 image
   predates that implementation. Expected low-single-digit standalone impact;
   requires GSM8K because the wire is lossy.

DSpark was explicitly deferred from the current optimization scope on
2026-10-02. Its checkpoint is retained for later work, but it is not part of
the current con32 candidate stack.

The public DSpark draft was not present on the B0 rack. A single copy is being
downloaded to:

```text
/mnt/k3/Kimi-K3-DSpark
```

then it will be replicated locally to every serving node.

### Single-node functional gates for portable candidates

These gates use one C7 MI455 B0 GPU. They establish gfx1250 operator support
without claiming an EP16 or AgentX performance result.

`ptpc_fp8`:

- the existing online-quant config and K3 layout suite passed 99 tests;
- fused RMSNorm + per-token FP8, fused attention sigmoid-mul + per-token FP8,
  SiTUv2 + per-token FP8, and a K3-size `7168 x 7168` Triton A8W8 projection
  passed at `M={1,4,32}`;
- fused/dequantized operator cosine similarity was at least `0.99964`; the
  end-to-end quantized projection was at least `0.99929`;
- the AITER gfx1250 `situv2_and_mul_quant` kernel terminates the process when
  `D > 16376`. K3's layer-0 dense MLP uses `D=33792`, while every shared
  expert uses `D=6144`;
- the ATOM candidate now disables this fusion for unsupported MLP dimensions
  and retains the fused path for shared experts. The `D=6144` fused operator
  was `1.80-1.87x` faster than activation followed by standalone quant for
  `M=1..32`; numerical checks had zero elements outside `rtol=5%, atol=0.5`.

Checkpoint policy:

- `test_state_checkpoint`, `test_page_unit_checkpoint`, `test_kda_layout_id`,
  and `test_gdn_state_relocation` passed `259/259`;
- this validates `interval=-1`, demand-rung behavior, page-unit geometry, KDA
  layout identity, and state-slot relocation. It does not measure the
  long-context cache-hit benefit, which still needs the full EP16 workload.

Quantized combine wire:

- the shipping image's AITER revision exposes neither constructor nor forward
  `combine_quant`; enabling fp8/fp4 there would fail during model startup;
- an isolated checkout of upstream AITER #5176 passed the gfx1250
  `MegaMoEGfx1250` single-rank accuracy gate at K3 dimensions
  (`H=7168`, `I=3072`, top-k 16, 32 tokens);
- MXFP8 combine passed with logits difference `0.020819` below its automatic
  `0.032735` tolerance and `120.875 us` median per layer;
- MXFP4 combine passed with logits difference `0.027262` below its automatic
  `0.042692` tolerance and `123.694 us` median per layer;
- this proves the gfx1250 kernels execute correctly, but a multi-rank EP test
  is still required to exercise the physical return wire and quantify benefit.
  The launch harness now fails before model allocation when a quantized wire is
  requested against an older AITER.

### Not directly portable

- DCP=8: TP1 cannot satisfy `tp % dcp == 0`, and persistent DCP MLA is gfx950-only.
- FlyDSL `gather_kv_b_proj`: gfx950/fp8-weight path; K3 uses an unquantized
  projection and gfx1250 requires PR #2380's unfused fallback.
- `AITER_SITUV2_A4W4` and FlyDSL stage2: gfx950 implementation; gfx1250 uses
  MegaMoE TDM kernels.
- block size 128 segmented MLA: available only off the mandatory gfx1250
  Triton-MLA path.
- INT4 quick-reduce: TP all-reduce optimization; the B0 target is TP1.

### Reproduction-harness correction

`ATOM_UNFUSED_GATHER_KV_B_PROJ` is absent on stock image ATOM and would
otherwise be a silent no-op. `experiments/kimi_k3_b0/launch_server.sh` now
checks for the PR #2380 runtime symbol before model allocation and fails
immediately when the patch is missing.

## Final 1800-second validation attempt

Best candidate:

- AgentX concurrency 32;
- Config B;
- `ATOM_MORI_V2_FUSED=1`;
- runtime merge specialization fix applied.

Warmup completed successfully:

```text
130/130 completed
0 errors
elapsed=914.39s
```

The 1800-second profiling window did not produce sufficient data in time. It
completed four trajectories early, then entered long trace idle intervals and
had not produced another completed trajectory when the run was manually paused
at the operator's request. No valid final coverage or result JSON was emitted,
so the partial metrics are not reported as a benchmark result.

Preserved log:

```text
/root/agentx_opt_final_c32_1800.log
```

All four Kimi containers were removed after the pause. Verified idle state on
C13/C14/C16: no running containers and approximately 0.2 GiB VRAM used per
GPU. Host-side logs, model data, patch files, and AgentX artifacts remain.

### Second 1800-second attempt on a replacement node set

To avoid the earlier C17 fault, the retry used C13/C14/C16/C8. All nodes passed
the fabric/resource gate, 16 ranks initialized, and the KV budget remained
positive (`available_for_kv=118.73GB`).

The warmup reached 30/130, then stopped making progress. C13 lost DP1:

```text
C13 0002:01:00.0 [gfxhub0] no-retry page fault
Process python3 pid 25997
Faulty UTCL2 client ID: TCP
ModelRunner exitcode=-6
```

This occurred without `device kernel image is invalid` and without a Python
traceback identifying `merge_attn_states`. The remaining 15 ranks stayed
alive but could not make collective progress.

The run was terminated and all four Kimi containers were removed. Preserved
artifacts:

```text
/root/agentx_opt_final2_c32_1800.log
/root/k3ep16-logs/agentx_final2_server.log
C13 dmesg at 2026-09-30 15:22:50
```

This is the second con128/con32 high-pressure run to expose a gfx1250 no-retry
page fault on a different node. It is now a platform/runtime blocker for final
AgentX certification, independent of the fixed Triton specialization bug.

The 2026-10-02 12:22 resource poll found C13 with no containers and 0% allocated
VRAM, but GPUs 1/2 were repeatedly reporting `MES ring buffer is full` and GPU3
reported expired SDMA fence fallback timers. C13 is therefore excluded from
performance runs until it is reset and passes a clean kernel-health window;
free VRAM alone is not a sufficient admission gate.

## Current optimization priorities

1. On the next stable four-node window, run the fixed con32 scheduler A/B with
   only `long_prefill_token_threshold=4096` changed.
2. Run full-EP16 GSM8K and AgentX gates for `ptpc_fp8`; the single-node
   operator stack now passes and the dense SiTU dimension crash is guarded.
3. Diagnose the recurring gfx1250 no-retry page fault before another
   1800-second certification attempt; changing nodes did not remove it.
4. Upgrade AITER to #5176 or newer before any fp8/fp4 combine-wire A/B, then
   validate it multi-rank before GSM8K and AgentX.
5. Inject the production-safe BF16 tuned CSV in an isolated con32 A/B.
6. Keep con32 for the interactivity objective; con64 is a throughput mode.
7. Keep `ATOM_MORI_V2_FUSED=1`; its isolated A/B passed the 3% threshold.
8. Fuse MoE routing `fill/copy/cumsum/where` bookkeeping.

## Proposed PR split

### PR 1 — MI455 B0 K3 recipe and reproducibility

- Rebase the existing recipe branch onto current `main`, thereby taking merged
  PR #2378 and removing the duplicate local diff.
- Add the validated `ATOM_MORI_V2_FUSED=1` result and fixed DP16/EP16 con32
  configuration.
- Add the parameterized launch/AgentX harness and this evidence record if
  `experiments/` is an acceptable repository location; otherwise keep the
  scripts in the recipe appendix.

### PR 2 — K3 ptpc SiTU shape guard

- Limit the AITER fused path to `D <= 16376` and `D % 8 == 0`.
- Preserve fusion for K3 shared experts (`D=6144`) and use the safe fallback
  for the dense MLP (`D=33792`).
- Add regression coverage for both dimensions and the empty-token case.
- Claim crash prevention and the measured shared-expert kernel speedup only;
  do not claim con32 E2E improvement before full-EP16 validation.

### Hold for later PRs

- BF16 gfx1250 tuning CSV: promote to AITER only after con32 E2E A/B.
- ptpc_fp8 recipe enablement: after EP16 GSM8K and AgentX.
- quantized combine wire: after AITER upgrade and multi-rank correctness/A-B.
- checkpoint policy and scheduler knobs: after valid AgentX comparisons.
- fused MoE routing bookkeeping: no implementation or isolated result yet.

The recipe and correctness fix should remain separate so each PR has one
reviewable claim.
