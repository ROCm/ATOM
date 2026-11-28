# AtomRL release validation

Results behind the reference values in
[`examples/docs/08-release.md`](../examples/docs/08-release.md) §8.5 and the catalog in
[`examples/README.md`](../examples/README.md). Every result below was judged by
`examples/run.sh` (before 2026-11-17 by its predecessor `release/run_example.sh`, with the
same references and tolerances) on 8× MI355X (gfx950) nodes, seed 10086, DAPO metrics read
at step 1.

Reproduce: `bash examples/run.sh suite` (one node), plus `--site <two-host site>` for the
two-node tier.

## Current release

| | |
|---|---|
| Image | `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125` (91.6 GB uncompressed) |
| Image ID | `sha256:4ede265ad0f1…` (local build) |
| Stack | ROCm 7.2.4, torch 2.10.0, vLLM 0.31.1.dev0 (`gdb9527a46`), transformers 5.16.1, Triton 3.8.0, flydsl 0.5.0, AITER `c813348a`, MORI 1.2.6.dev20261119, megatron-core 0.18.2, TransformerEngine `6e541a10`, flash-attention 2.8.3 (CK), Apex 1.10.0 (`751f5dd5`, see below) |
| Base | `rocm/atom-dev:vllm-v0.31.0-nightly_20261119` |
| Source | ATOM `4f40b7e87` (`ATOM_COMMIT`): main `62999fb21` plus the release-recipe fixes |
| Hardware | 2 nodes × 8 MI355X, RoCE between them |
| Date | 2026-11-26 |

The image is the recipe's first 261125 build (`4fd0c73007fb`, source `cfdfffec0`) with layers
added rather than a rebuild: flash-attention 2.8.3 built against its torch, the files that
changed up to `4f40b7e87`, and `ATOM_COMMIT`. The recipe now does all of it itself. The 16
results below ran on the image before its last layer (`809111e8d0bb`); that layer changes
only Megatron native's attention backend (below), so example 7, the one Megatron-native
example, ran again on the final image.

### Results

All 16 pass, one run each, the ATOM examples without `--atom-src`.

| Example | Verdict | `k3_kl` | Reference | Δ | Wall clock |
|---|---|---|---|---|---|
| `unit-tests` | PASS | 1073 passed, 9 skipped | | | 5.0 min |
| 1 DAPO Qwen3-8B, vLLM BF16 | PASS | 0.00109 | 0.00110 ±30% | −1.0% | 3.1 min |
| 2 DAPO Qwen3-8B, vLLM FP8 rollout | PASS | 0.00449 | 0.00481 ±30% | −6.7% | 2.8 min |
| 3 DAPO Qwen3-8B, FP8 end to end | PASS | 0.00350 | 0.00410 ±35% | −14.7% | 3.5 min |
| 4 DAPO Qwen3-8B, ATOM FP8 | PASS | 0.00261 | 0.00399 ±50% | −34.5% | 10.9 min |
| 5 DAPO Qwen3-8B, ATOM BF16 | PASS | 0.000889 | 0.000954 ±50% | −6.8% | 8.0 min |
| 6 DAPO Qwen3-30B-A3B, vLLM | PASS | 0.00167 | 0.00144 ±50% | +15.9% | 8.0 min |
| 7 DAPO Qwen3-30B-A3B, Megatron EP=8 | PASS | 0.00176 | 0.00161 ±50% | +9.3% | 8.2 min |
| 9 DAPO Qwen3-30B-A3B, ATOM | PASS | 0.00158 | 0.00153 ±50% | +3.2% | 15.8 min |
| DAPO fully async, 4 + 4 GPUs | PASS | 0.00173 | 0.00157, ×3 band | +9.9% | 12.8 min |
| DAPO fully async, two nodes, vLLM | PASS | 0.00168 | 0.00156, ×3 band | +7.6% | 15.0 min |
| DAPO fully async, two nodes, ATOM | PASS | 0.00169 | 0.00157, ×3 band | +7.6% | 22.4 min |
| 8 GRPO Qwen3-30B-A3B, two nodes, vLLM | PASS | 0.000817 | 0.00082, ×3 band | −0.4% | 20.5 min |
| GRPO Qwen3-30B-A3B, two nodes, ATOM | PASS | 0.00184 | 0.0017, ×3 band | +8.3% | 28.4 min |
| SDDD Eagle3 smoke | PASS | loss 36.2, position 0: 12.3 | bounds 29–60, 10–20 | | 0.8 min |
| Quick start, Qwen3-0.6B GRPO, one GPU | PASS | `grad_norm` 0 at step 2 (every sample wrong) | | | 13.8 min |

GRPO with ATOM rollout, two nodes, 5 steps: steps 2–5 take 176–189 s (244–261 s on 261119),
generation 78–80 s (103–107 s), training 72–81 s, weight sync 4.5–4.9 s.

Example 7's entropy reference moved to 0.68 ±60%, the band of example 6, on 2026-11-26: its
step-1 entropy moves with sampling (0.686, 0.986 and 1.243 on the pinned model; 0.626, 0.818,
0.597, 0.700 and 0.650 on this stack), while `k3_kl` stays put. Under the old 0.986 ±32% it failed
twice on this stack.

**What this stack changes.** Against 261119, five things needed launcher, config or code
changes, all on this branch:

- *`expandable_segments` now takes effect.* `run_grpo.sh` exported
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `run_dapo.sh` defaults to it; on
  261119's torch it was a no-op on HIP. With it, a sleeping colocated ATOM no longer returns
  its memory (a colocated Qwen3-8B Megatron GRPO step at 2048 × 4096 runs out of memory),
  and `hipIpcGetMemHandle` rejects a TP2 custom all-reduce's buffers (vLLM
  `custom_all_reduce_hip.cuh:167 'invalid argument'`). `run_grpo.sh` now keeps a caller's
  empty value, and the GRPO two-node examples pass one.
- *vLLM 0.31 checks routing replay.* `enable_return_routed_experts` needs prefix caching;
  example 8 turns it on (every weight update already resets the cache). vLLM 0.31 also keeps
  FusedMoE's weights in a nested `RoutedExperts`, which the RDMA receiver's completeness
  check counted as 96 missing parameters out of 435; it now uses the router's aliases.
- *AITER `c813348a` asserts in a custom all-gather* when the custom all-reduce is off, which
  AtomRL does for ATOM TP > 1; the GRPO example with ATOM sets `ATOMRL_DISABLE_CUSTOM_AR=0`.
- *No `flash_attn` in the base.* Without it Megatron's `attention_backend: flash` falls back
  to unfused attention, which routing replay refuses (example 8) and which made GRPO with ATOM
  train about 15% slower; the packed log-prob path also loses its fused cross-entropy.
- *TransformerEngine on ROCm prefers flash_attn's backend* whenever the package is
  installed. For Qwen3-30B-A3B's packed causal attention it is half as fast as TE's fused
  attention (fwd + bwd 2.60 vs 1.33 ms per 8192 tokens), which made Megatron native training
  14% slower once flash-attention was in the image. Megatron native now asks for fused
  attention there; `megatron_cfg.te_attention_backend: auto` restores TE's choice.

The first build, without flash-attention or these changes, passed 13 of 16: example 7 on its
old entropy check, and both GRPO two-node examples.

Not fixed: the pinned Apex `daed8525` is built as an egg that the base's Apex 1.10.0 shadows,
so `import apex` loads the base's (261119 loaded the pinned 1.14.0a0).

## Previous release: 261119

| | |
|---|---|
| Image | `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119` (13.4 GB compressed) |
| Digest (linux/amd64) | `sha256:ced66f1525e7…` |
| Stack | ROCm 7.2.3, torch 2.10.0, vLLM 0.23.0, transformers 5.12.0, Triton 3.8.0, flydsl 0.5.0, AITER `2037a7711`, MORI `9c42e3240` |
| Source | ATOM `2a5440e5b` (`ATOM_COMMIT`) |
| Hardware | 2 nodes × 8 MI355X, RoCE between them |
| Date | 2026-11-19 |

### Results

All 16 pass, one run each. The full suite ran on an image built from `04acce33c` with the
same pins; the paths that the later Lumen fixes (`f6c43d7b3`) touch -- the unit tests, the
FP8-training examples 3 and 4, and the quick start -- were run again on the published image,
and those are the values below. The ATOM examples run without `--atom-src`: the image's
ATOM includes the RDMA weight receiver of ROCm/ATOM#3058.

| Example | Verdict | `k3_kl` | Reference | Δ | Wall clock |
|---|---|---|---|---|---|
| `unit-tests` | PASS | 1090 passed, 10 skipped | | | 6.2 min |
| 1 DAPO Qwen3-8B, vLLM BF16 | PASS | 0.00108 | 0.00110 ±30% | −1.7% | 3.8 min |
| 2 DAPO Qwen3-8B, vLLM FP8 rollout | PASS | 0.00470 | 0.00481 ±30% | −2.3% | 2.6 min |
| 3 DAPO Qwen3-8B, FP8 end to end | PASS | 0.00348 | 0.00410 ±30% | −15.0% | 4.0 min |
| 4 DAPO Qwen3-8B, ATOM FP8 | PASS | 0.00236 | 0.00399 ±50% | −40.9% | 12.3 min |
| 5 DAPO Qwen3-8B, ATOM BF16 | PASS | 0.000867 | 0.000954 ±50% | −9.1% | 7.4 min |
| 6 DAPO Qwen3-30B-A3B, vLLM | PASS | 0.00157 | 0.00144 ±50% | +8.9% | 10.1 min |
| 7 DAPO Qwen3-30B-A3B, Megatron EP=8 | PASS | 0.00159 | 0.00161 ±50% | −1.0% | 8.7 min |
| 9 DAPO Qwen3-30B-A3B, ATOM | PASS | 0.00157 | 0.00153 ±50% | +2.5% | 16.3 min |
| DAPO fully async, 4 + 4 GPUs | PASS | 0.00152 | 0.00157, ×3 band | −3.0% | 13.4 min |
| DAPO fully async, two nodes, vLLM | PASS | 0.00160 | 0.00156, ×3 band | +2.4% | 17.0 min |
| DAPO fully async, two nodes, ATOM | PASS | 0.00163 | 0.00157, ×3 band | +3.6% | 26.9 min |
| 8 GRPO Qwen3-30B-A3B, two nodes, vLLM | PASS | 0.000821 | 0.00082, ×3 band | +0.1% | 25.0 min |
| GRPO Qwen3-30B-A3B, two nodes, ATOM | PASS | 0.00171 | 0.0017, ×3 band | +0.5% | 25.2 min |
| SDDD Eagle3 smoke | PASS | loss 42.8, position 0: 14.4 | bounds 29–60, 10–20 | | 0.9 min |
| Quick start, Qwen3-0.6B GRPO, one GPU | PASS | `grad_norm` 0.55 at step 2 | | | 14.2 min |

Example 4 ran three times on this stack: 0.00204, 0.00256 and 0.00236 (−48.9%, −35.7%, −40.9%).

GRPO on Qwen3-30B-A3B, two nodes, 5 steps:

| | vLLM rollout (example 8) | ATOM rollout |
|---|---|---|
| Time per step | 196–239 s | 199–234 s |
| Generation per step | 99–104 s | 103–107 s |
| Weight sync, trainer → 4 rollout replicas | 3.9–4.4 s | 4.5–5.0 s |

ATOM's generation now matches vLLM's. On the 261117 image it took 145–150 s per step, with an
AITER (`ff745fb09`) that lacks ROCm/aiter#7640; ATOM itself also changed between the two
images, so the gain is not attributed to #7640 alone.

## Earlier release: 261117

| | |
|---|---|
| Image | `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261117` (13.3 GB compressed) |
| Multi-architecture index | `sha256:9e12afe84fd2…` |
| linux/amd64 manifest | `sha256:fd2bf0e17658…` |
| Stack | ROCm 7.2.3, torch 2.10.0, vLLM 0.23.0, transformers 5.12.0, Triton 3.8.0, flydsl 0.4.2, AITER `ff745fb09`, MORI `9c42e3240` |
| Source | ATOM `3f4492da1` (`ATOM_COMMIT`), a pre-merge commit of PR #3355 |
| Hardware | 2 nodes × 8 MI355X, RoCE between them; the image pulled from Docker Hub and its digest checked |
| Date | 2026-11-17 |

These results describe source commit `3f4492da1`; they do not establish parity
with the current merge candidate. Runtime fixes made after that commit require
the release suite to be re-recorded before merge-candidate validation can be
claimed.

### Results

15 examples and the unit-test suite pass; one run each. Example 8 with ATOM rollout comes from
the two-node rerun below.

| Example | Verdict | `k3_kl` | Reference | Δ | Wall clock |
|---|---|---|---|---|---|
| `unit-tests` | PASS | 1073 passed, 14 skipped | | | 6.4 min |
| 1 DAPO Qwen3-8B, vLLM BF16 | PASS | 0.000950 | 0.00110 ±30% | −13.7% | 4.0 min |
| 2 DAPO Qwen3-8B, vLLM FP8 rollout | PASS | 0.00469 | 0.00481 ±30% | −2.5% | 2.7 min |
| 3 DAPO Qwen3-8B, FP8 end to end | PASS | 0.00351 | 0.00410 ±30% | −14.3% | 3.0 min |
| 4 DAPO Qwen3-8B, ATOM FP8 | PASS | 0.00253 | 0.00399 ±50% | −36.5% | 12.0 min |
| 5 DAPO Qwen3-8B, ATOM BF16 | PASS | 0.00102 | 0.000954 ±50% | +7.3% | 6.4 min |
| 6 DAPO Qwen3-30B-A3B, vLLM | PASS | 0.00147 | 0.00144 ±50% | +2.3% | 9.1 min |
| 7 DAPO Qwen3-30B-A3B, Megatron EP=8 | PASS | 0.00162 | 0.00161 ±50% | +0.8% | 8.7 min |
| 9 DAPO Qwen3-30B-A3B, ATOM | PASS | 0.00169 | 0.00153 ±50% | +10.3% | 16.2 min |
| DAPO fully async, 4 + 4 GPUs | PASS | 0.00153 | 0.00157, ×3 band | −2.5% | 12.8 min |
| DAPO fully async, two nodes, vLLM | PASS | 0.00148 | 0.00156, ×3 band | −5.1% | 17.7 min |
| DAPO fully async, two nodes, ATOM¹ | PASS | 0.00154 | 0.00157, ×3 band | −1.9% | 31.7 min |
| 8 GRPO Qwen3-30B-A3B, two nodes, vLLM | PASS | 0.000815 | 0.00082, ×3 band | −0.6% | 24.1 min |
| GRPO Qwen3-30B-A3B, two nodes, ATOM¹ | PASS | 0.00168 | first run | | 28.9 min |
| SDDD Eagle3 smoke | PASS | loss 44.2, position 0: 15.0 | bounds 29–60, 10–20 | | 0.8 min |
| Quick start, Qwen3-0.6B GRPO, one GPU | PASS | `grad_norm` 0.81 at step 2 | | | 12.4 min |

¹ With ATOM from [ROCm/ATOM#3058](https://github.com/ROCm/ATOM/pull/3058) (`--atom-src`), which adds the RDMA weight receiver.

Every run exited 0 with no `Traceback`, `OutOfMemory`, `CUDA error` or `HSA_STATUS`, and
every weight-sync bucket reported `skipped=0`.

Not covered by this release run: the verl plugin (validated in its own image, not run
through `run.sh`), the vime plugin (no published image), and the long tier (FlowGRPO,
DeepSeek-V4-Flash GRPO, Kimi K2.5 SDDD).

### Two-node rerun on the final series

The two-node tier again on the same image with the final local series (`9d5966da6` plus
example 8's two-node ATOM config), after the history was regrouped:

| Example | Verdict | `k3_kl`, step 1 | Steps 2–N | Weight sync | Wall clock |
|---|---|---|---|---|---|
| DAPO fully async, two nodes, vLLM | PASS | 0.00155 | 52–74 s, trainer wait ≤ 11 s | 3.2 s | 16.5 min |
| DAPO fully async, two nodes, ATOM¹ | PASS | 0.00165 | 54–118 s, trainer wait ≤ 52 s | 3.7 s | 31.3 min |
| 8 GRPO, two nodes, vLLM | PASS | 0.000819 | 199–216 s | 3.9–4.5 s | 24.3 min |
| GRPO, two nodes, ATOM¹ | PASS | 0.00168 | 244–261 s | 4.6–4.9 s | 28.9 min |

Example 8 with ATOM previously ran out of trainer memory on a config written for one
colocated node. Its new config is the vLLM two-node config with the rollout on ATOM and
R3 off, since ATOM does not support routing replay; that is also why its `k3_kl` is about
twice the vLLM example's.

## Stability across builds

DAPO `k3_kl` on every image built so far from the ATOM repository, against the reference:

| # | Reference (n runs) | 2026-11-16, local build | 2026-11-17, local build | 2026-11-17, published | 2026-11-19, published | 2026-11-26, local build |
|---|---|---|---|---|---|---|
| 1 | 0.00110 (3) | 0.00108 | 0.000999 | 0.000950 | 0.00108 | 0.00109 |
| 2 | 0.00481 (3) | | 0.00470 | 0.00469 | 0.00470 | 0.00449 |
| 3 | 0.00410 (3) | | 0.00352 | 0.00351 | 0.00348 | 0.00350 |
| 4 | 0.00399 (5) | 0.00268 | 0.00267 | 0.00253 | 0.00204 / 0.00256 / 0.00236 | 0.00261 |
| 5 | 0.000954 (3) | 0.000941 | 0.00111 | 0.00102 | 0.000867 | 0.000889 |
| 6 | 0.00144 (3) | | 0.00146 | 0.00147 | 0.00157 | 0.00167 |
| 7 | 0.00161 (3) | | 0.00179 | 0.00162 | 0.00159 | 0.00176 |
| 9 | 0.00153 (3) | 0.00153 | 0.00147 | 0.00169 | 0.00157 | 0.00158 |

AtomRL also ran unchanged in the previous private Lumen-RL validation image:
example 1 at 0.00115 (+4.1%), example 5 at 0.00103 (+8.2%).

## Reference values

### DAPO

Established on 2026-09-21 on the previous private Lumen-RL validation image:
26 runs, all PASS. Each reference is the mean of its runs; spread is the worst
deviation of a run from that mean.

| # | Runs of `k3_kl` | Reference | Spread | `entropy` reference | Tolerance (`k3_kl` / `entropy`) |
|---|---|---|---|---|---|
| 1 | 0.00112 / 0.00107 / 0.00112 | 0.00110 | 2.6% | 0.628 | ±30% / ±25% |
| 2 | 0.00471 / 0.00485 / 0.00485 | 0.00481 | 2.0% | 0.791 | ±30% / ±25% |
| 3 | 0.00481 / 0.00373 / 0.00375 | 0.00410 | 17.2% | 0.790 | ±30% / ±25% |
| 4 | 0.00460 / 0.00339 / 0.00426 / 0.00499 / 0.00271 | 0.00399 | 32.1% | 0.599 | ±50% / ±50% |
| 5 | 0.000841 / 0.000993 / 0.00103 | 0.000954 | 11.8% | 0.667 | ±50% / ±60% |
| 6 | 0.00136 / 0.00157 / 0.00138 | 0.00144 | 9.2% | 0.620 | ±50% / ±60% |
| 7 | 0.00166 / 0.00160 / 0.00157 | 0.00161 | 3.2% | 0.68 | ±50% / ±60% |
| 9 | 0.00158 / 0.00146 / 0.00154 | 0.00153 | 4.7% | 0.683 | ±50% / ±60% |

Example 7's entropy reference is 0.68 since 2026-11-26 (Current release above).

`rollout_corr/k3_kl` is the primary criterion; `entropy` is a coarse secondary check;
`rollout_corr/kl` must stay within a factor of 10 of its reference. A run also needs
`skipped=0` on every weight-sync bucket, since a rollout that misses weights otherwise
exits cleanly. Full criteria: [§8.5.2](../examples/docs/08-release.md#852-criteria).

### Other examples

New examples take a factor-3 band around their first runs until enough runs establish
a spread: fully async one node 0.00157, two nodes 0.00156 (vLLM) and 0.00157 (ATOM),
example 8 0.00082 (runs 0.000818 / 0.000820 / 0.000815 / 0.000819), example 8 with ATOM
0.0017 (one run, 0.00168). The SDDD smoke is bounded
around ln(vocab) = 11.9 nats per position, because the draft's random initialisation
moves its starting loss by more than a tolerance allows: four runs gave 12.2–15.0 at
position 0 and 36.0–44.2 in total.

## History

| Date | Image | Change | Result |
|---|---|---|---|
| 2026-11-26 | `atomrl:…-261125` (local build) | flash-attention 2.8.3 and `ATOM_COMMIT` added to the first build; GRPO two-node launcher and config changes, the RDMA reload check, example 7's entropy band; Megatron native keeps TE's fused attention | 16 / 16 PASS (example 7 again on the final image) |
| 2026-11-25 | `atomrl:…-261125` (local build) | Base `vllm-v0.31.0-nightly_20261119` (ROCm 7.2.4, vLLM 0.31), AITER `c813348a`, megatron-core 0.18.2; four recipe fixes (base pin, `MORI_SHA`, entrypoint, megatron-core) | 13 / 16 PASS (first build); example 7 entropy; GRPO two-node: no `flash_attn`, `expandable_segments`, AITER custom all-gather |
| 2026-11-19 | `atomrl:…-261119` (published) | AITER `2037a7711` (ROCm/aiter#7640) and flydsl 0.5.0; ATOM with ROCm/ATOM#3058 baked in, so the ATOM two-node examples need no `--atom-src` | 16 / 16 PASS |
| 2026-11-17 | `atomrl:…-261117` (published) | Two-node rerun on the final series; example 8 with ATOM on a two-node config | 4 / 4 PASS |
| 2026-11-17 | `atomrl:…-261117` (published) | First published AtomRL image; `run.sh` default | 14 examples + unit tests PASS |
| 2026-11-17 | local build of 2026-11-16 | First `examples/run.sh` suite, one and two nodes; three defects found and fixed (SDDD HF teacher embedding, OpenMathInstruct-2 field names, verl plugin model path) | 13 PASS, plus two-node ATOM with #3058; quick start fixed afterwards |
| 2026-11-16 | local build | First build from the ATOM repository: in-tree ATOM and Lumen, official AITER `ff745fb09`, Triton 3.8.0 | 4 / 4 PASS (examples 1, 4, 5, 9) |
| 2026-09-21 | `lumen-rl:…-260921b` | ATOM PR #2267 merged; AITER rebased; reference table re-measured with 3–5 runs per example | 26 / 26 PASS; references moved −8.9% to +10.5% |
| 2026-09-17 | `lumen-rl:…-260917` | ATOM `main` baked in, measured with nothing mounted | 11 / 11 PASS |
| 2026-09-16 | `lumen-rl:…-260910` | ATOM re-pinned to `main` after ROCm/ATOM#2028 merged; rollout sleep releases memory | 15 / 15 clean exits |
| 2026-09-10 | `lumen-rl:…-260910` | ATOM re-pinned to the PR #2028 head; four latent defects on `main` fixed | 8 / 8 PASS, `k3_kl` within ±8.2% |

Findings from those rounds that still apply:

- **Releasing rollout memory during sleep costs about 1.1 s per step** (+2–3%) on examples
  4 and 9. Example 9 ran 48 balanced release/resume cycles with no KV-pool or memory faults.
- **`ATOM_FORCE_ATTN_TRITON=1` is set for every ATOM mode.** ATOM's assembly paged-decode
  kernel returns wrong output for a context of exactly 16 pages with a partial last one;
  on example 4 it produced a `chi2_token` of 5761.69 at step 3, against at most 5.39 with
  the Triton path. Step-1 `k3_kl` cannot see this defect and is unchanged by the setting.
- **Single runs mislead.** Example 4's five runs span a factor of 1.8, and two earlier
  three-run clusters looked like systematic shifts that further runs dissolved. References
  are multi-run means for this reason.

## Limits

- References are smoke-scale (1–3 steps): they show that a stack reproduces, not that it
  trains well. Training quality is measured by the long runs in
  [asynchronous training](../atomrl-docs/docs/source/advance/async_training.md#results) and
  [FlowGRPO](../examples/FlowGRPO/README.md).
- The current release was run once per example; spreads come from 2026-09-21.
- Example 4 sat below the 0.00271–0.00499 range of its reference runs on every image built
  from the ATOM repository before 261125: 0.00253–0.00268 with AITER `ff745fb09` and
  0.00204–0.00256 over three runs with `2037a7711`; with `c813348a` it gave 0.00282 and 0.00261, inside
  the range. One run does not settle whether the reference should move.
- gfx950 only. Cold `docker pull` time and logging to a live W&B project were not measured.
