# MiniMax-M3.1 TP4 + DSpark AgentX: per-concurrency recipes, the PRs that provide them, and what was measured

This is the recipe for the AgentX replay (`inferencex-agentx-mvp`) of MiniMax-M3.1-Flash (MXFP4 experts, native MXFP8
attention, NVFP4 KV and index caches) with a DSpark draft at TP4 on MI355X, using only ATOM and aiter (no amd-kernels
import, no overlay, no private tuned CSV). Every flag is off by default; a flag works once the PR named next to it is
merged. Numbers are from the stack builds described in "Reproducing the stack"; read the caveats first.

## Caveats (read before quoting any number)

1. Every number is a **single 900 s run** (seed 42, synthetic acceptance length 4.04 at k=7, k=3 at c64). Measured
   noise on the same node is about 0.2-0.5% tok/s/GPU at c32 and about 1.5% on ix_p90; the c64 noise is not known
   (a repeat is queued). Differences under ~1% tok/s or ~6% ix_p90 between arms are not distinguishable.
2. The reference ("C17r") is the private production stack re-run the same day on the same node, 900 s windows, not the
   3600 s official numbers.
3. Interactivity `ix_p90 = 1000 / ITL p90 (ms)`. **Throughput reaches 85% of the reference at every concurrency, but
   interactivity does not at c32 and c64** (77% and 64-74%); the remaining gap is the decode step (+1.5 ms) at c32 and
   prefill/queueing at c64. This recipe is not yet the user's target of >= 85% on both.
4. The GSM8K gate margin is thin (pooled 8 runs 0.9629 against a 0.9626 line, see "Accuracy gate").
5. #3510/#3511 are cherry-picked onto public main (they sit on a 55-commit private branch); the recipe needs one
   extra one-line fix and one env entry that are not in any PR yet (see "Commits needed from #3510/#3511").
6. The aiter side is the image's aiter plus the listed PRs; three of them (#8548 stack, #8555, #8584) carry
   conflict resolutions that the stack builder did by hand (see "Divergences from the PR heads").
7. aiperf reports `grace_timeout=True` at c64 for these arms and for the reference (in-flight requests at cutoff,
   `leaked_state` 0); the score is still produced.

## Environment

| Item | Value |
|---|---|
| Image | `rocm/atom-dev:latest` (torch 2.10 rocm 7.2.4, triton 3.8, flydsl 0.5.3, aiter at `/app/aiter-test`; ATOM main calls `init_dist_env(device_communicators=...)`: the 2026-11-17 nightly fails at startup) |
| aiter | the image's `/app/aiter-test` tree with the aiter PRs below merged on top of aiter main `c7d4f9fe1`. The image's JIT `.so` of modules whose C++ changed are **not** rebuilt automatically: delete the stale ones in the copied tree (`custom_all_reduce`, `hipbsolgemm`, `fused_qknorm_idxrqknorm`, `mla_nvfp4`, `kpool_compress`); the first boot rebuilds them (~5-6 min). Never boot two servers on one writable tree. |
| ATOM | ATOM main `d0dec1480` plus the ATOM PRs below |
| Server | as `recipes/MiniMax-M3-Agentic-InferenceX.md` with `--kv_cache_dtype nvfp4 --index-cache-dtype nvfp4 --method dspark --draft-model <model>/dspark --num-speculative-tokens 7` (3 at c64), `--max-num-seqs 64`, `--gpu-memory-utilization 0.85`, `--cudagraph-capture-sizes [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]`, `--spec-decode-kv-cache-dtype fp8`, `AITER_LOG_LEVEL=WARNING`, `ATOM_FORCE_ATTN_TRITON=1`, `PYTHONHASHSEED=0` |
| Client | AIPerf `inferencex-agentx-mvp`, seed 42, 900 s per point |

## The two recipes

Common to both (the F set, all provided by PRs listed below): `ATOM_PREFILL_DECODE_INTERVAL=1`,
`ATOM_PREFILL_DELAYER_TARGET_FILL=0.225`, `ATOM_SCHED_DECODE_PREFILL_CAP=16384`, `ATOM_SCHED_IDLE_PREFILL_CAP=16384`,
`ATOM_SCHED_FINISH_HOLD_LEFT=128`, `ATOM_SCHED_FINISH_HOLD_MAX_MS=500`, `ATOM_SCHED_FINISH_HOLD_MAX_TOKENS=400`,
`ATOM_SCHED_ITL_RISK_MS=4.5`, `ATOM_SCHED_ITL_RISK_MAX_MS=150`, `ATOM_PREFIX_DEMOTE_DECODE=1`,
`ATOM_M3_PREFILL_TOKEN_GRAPH=1`, `ATOM_TOKEN_GRAPH_SIZES=256,512,1024`, `ATOM_MONO_MAX_ROWS=8`,
`ATOM_DSPARK_BLOCK_BS_LADDER=1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64`, `ATOM_LM_HEAD_RING=1`,
`ATOM_DSPARK_FC_RING=1`, `ATOM_M3_FUSED_INDEX_SELECT=1`, `AITER_FLY_AR_PULL=1`, `ATOM_M3_PREFILL_TOPK_FLYDSL=0`,
`ATOM_M3_SP_PREFILL=1`.

### c8 / c16 / c32: "FsW"

Common set plus:

```
ATOM_M3_SP_SHARED_LOCAL=1      # owner-local shared expert (#3805)
ATOM_M3_QKV_CHAIN=1            # QKV front chain, 17-128 decode rows (#3809/#3810, aiter #8606/#8607/#8610/#8611)
ATOM_M3_DECODE_MOE=1           # decode MoE chain (#3787, aiter #8548-#8556 incl. the w13_split commit of #8555)
--spec-decode-kv-cache-dtype fp8
```

`#3807` (SP glue fusion) has no flag; it is active whenever SP prefill runs.

### c64: "FsK"

Common set plus `--spec-decode-kv-cache-dtype fp8` and **`ATOM_M3_DECODE_MOE=0`**, with `--num-speculative-tokens 3`;
no shared-local, no QKV chain, no #3807 (the FsK tree is "Fs": SP prefill without #3805/#3807/#3809/#3810).
FsW also runs at c64 (below), but FsK is the better c64 point.

## Measured (c32 and c64 are 900 s; "%" is of the same-day C17r)

| Point | Arm | tok/s/GPU | % of C17r | ix_p90 | % of C17r | ITL p50/p90/p99 (ms) | TTFT p50/p90 (s) | n | err |
|---|---|---|---|---|---|---|---|---|---|
| c8 | FsW | 13,124 | 99.0 | 434.3 | 87 | 1.67/2.30/4.47 | 0.12/0.28 | 445 | 0 |
| c8 | C17r | 13,262 | 100 | 498.4 | 100 | 1.52/2.01/3.79 | 0.11/0.22 | 450 | 0 |
| c16 | FsW | 21,234 | 100.1 | 279.6 | 90 | 2.44/3.58/8.42 | 0.21/0.67 | 601 | 0 |
| c16 | C17r | 21,216 | 100 | 309.2 | 100 | 2.27/3.23/7.83 | 0.18/0.58 | 601 | 0 |
| c32 | FsW | 57,825 | **96.5** | 175.7 | **77.1** | 3.61/5.69/8.68 | 0.42/1.52 | 1638 | 0 |
| c32 | C17r | 59,946 | 100 | 227.9 | 100 | 3.09/4.39/8.46 | 0.31/1.21 | 1677 | 0 |
| c64 | FsK | 50,093 | **91.9** | 50.8 | 73.6 | 10.50/19.69/38.3 | 3.6/23.1 | 1670 | 0 |
| c64 | FsW | 46,592 | 85.5 | 44.2 | 64.1 | 10.73/22.61/43.8 | 4.2/46.9 | 1575 | 0 |
| c64 | C17r | 54,523 | 100 | 69.0 | 100 | 7.57/14.49 | 2.8/28.2 | 1766 | 0 |

The c64 ix_p90 values are computed from the measured ITL p90 (1000 / 19.69, 1000 / 22.61, 1000 / 14.49). c8 and c16 are load-limited points (the offered load, not the server, bounds tok/s). The public-main baseline with
the image's aiter and no flags (P0r) scores 51,753 at c32 (ix_p90 102.0); P0r is not a recipe.

Memory and KV pool (the c64 result depends on them):

| Arm | KV blocks | weight peak per GPU (GB) | note |
|---|---|---|---|
| C17r (reference) | 89,425 | 68.2 | |
| FsW | 89,431 (c32), 89,425 (c64) | 69.2 | |
| FsK | 92,634 | 63.7 | DECODE_MOE=0, fp8 draft KV |
| Fs (SP, decode MoE on the old head) | 66,627 | 98.1 | the old #3787 head kept a second copy of w13 |

## What each flag costs in memory

| Flag | Price | Evidence |
|---|---|---|
| `ATOM_M3_DECODE_MOE=1` with the old #3787 head | +34 GB/GPU (second re-laid w13 copy), -18.5k KV blocks (90% of the loss to the reference) | Fs boot log |
| `ATOM_M3_DECODE_MOE=1` with #3787 `a779e6967` + aiter #8555 w13_split | +2.4 GB peak (routed w13 is read in place; 36 MiB held per MoE layer and rank) | FsW boot log: peak_torch 76.39 GB |
| `ATOM_M3_SP_PREFILL=1` | +3.1 GB/GPU (`o_proj_full`, an unsharded o_proj copy), -1.3k KV blocks | Fs vs Fm |
| `ATOM_M3_SP_SHARED_LOCAL=1` | +3.1 GiB/GPU (unsharded shared expert, 57 layers), -1.6k KV blocks | Fs3 vs Fs |
| `--spec-decode-kv-cache-dtype fp8` | draft KV entry 1,986,560 -> 1,827,840 bytes: +8.7% KV blocks (public flag, nothing to port) | FsD/FmK boot logs |
| `ATOM_M3_QKV_CHAIN=1` | the QKV weight must be preshuffled (layers where it is not run the old path); no separate copy measured | FsW weight peak 69.2 GB |
| #3807 | none | |

FsK is the c64 recipe because DECODE_MOE=0 and the fp8 draft KV give it a 92.6k-block pool and a 89.5% prefix-cache hit
rate (FsW 83.8%, C17r 89.3%).

## Accuracy gate (GSM8K, strict)

A fixed threshold on 4 runs does not discriminate (run-to-run sd 0.003-0.004). The rule used: an arm is reportable if
its pooled strict mean over all runs is >= 0.9626 (the baseline 0.9671 minus 0.0045), or if a paired per-question
comparison against the pooled TP-prefill control shows p > 0.1 and a mean difference > -0.003.

| Arm | runs | pooled strict | note |
|---|---|---|---|
| Fm (no SP, control) | 4 | 0.9636 | |
| Fs (SP prefill) | 4 | 0.9629 | paired vs Fm -0.0008, p 0.82 |
| FsW (SP + shared-local + #3807 + chain + fp8 draft KV + single-copy MoE) | 8 | 0.9629 | gate4 0.9634, gate4b 0.9625; paired vs Fm -0.0007, p 0.79; margin +0.0003 |
| FsK | 8 | 0.9635 | gate4 0.9621, gate4b 0.9649; paired vs Fm -0.0001, p 1.0 |

The gate was run with the SP flags, the chain and the fp8 draft KV on. It is not a per-flag proof: arms in between
(Fs3, Fs3b) sit within noise of the line and pass by the paired rule.

## Reproducing the stack

All PRs are against `main` unless noted. Merge order (stack bases first):

aiter (aiter main `c7d4f9fe1` as the base of the builds):

1. `#8548 -> #8551 -> #8552 -> #8553 -> #8555` (decode MoE chain; #8555's head is `#8553`'s plus the `w13_split` commit
   `9e02b0320`), then `#8556`.
2. `#8554` (small-M BF16 ring GEMM), `#8573` (draft fc ring), `#8574` (pull all-reduce + Gemma RMSNorm).
3. `#8575 -> #8577` (sparse decode attention merge, index top-16 in the attention).
4. `#8576 -> #8580` (NVFP4 insert, DPP insert).
5. `#8578 -> #8582 -> #8584` (o_proj_full 8-wave picks, persistent 4-wave GEMM, 4-wave backend with the inference-tensor
   scale-cache-key fix).
6. `#8581` (prefill top-16 selector; present, flag off).
7. `#8606 -> #8607 -> #8610 -> #8611` (MXFP8 shuffled quantizer, 17-64 row QKV chain, 65-128 row split GEMM and
   reduce-insert).

Not part of the recipe: #8587 (pull collectives: no measurable gain, `ATOM_M3_SP_COMM` unset), #8604 + ATOM #3806 (attention
quant-out: no gain), #8598/#8599 (tuned CSV rows, kernel-level only).

ATOM (main `d0dec1480`):

1. `#3721`, `#3722`, `#3723`, `#3724` (any order; each carries the others' `dspark.py` resolution).
2. `#3775` (scheduler policies), `#3776` (LM head ring, needs aiter #8554), `#3783` (prefill token graphs), `#3784`
   (mono row cap), `#3785` (fused index select, needs aiter #8577), `#3786` (draft fc ring, needs #8573), `#3789`
   (draft block batch ladder), `#3788` (prefill top-16; code present, `ATOM_M3_PREFILL_TOPK_FLYDSL=0`).
3. `#3787` (decode MoE chain; head `a779e6967`, needs aiter #8548-#8556).
4. SP prefill: **#3510 then #3511** (see below), then `#3805` (shared-local), then `#3807` (glue).
5. `#3809`, `#3810` (QKV chain; need aiter #8606/#8607/#8610/#8611).
6. This document (`#3790`).

### Commits needed from #3510/#3511

Public main does not have the 55-commit branch the two PRs are stacked on. The recipe uses exactly two commits of it:
`2e833df17` (#3510, prefill DBO under pure TP; code only, not enabled) and `ad44f6d0f` (#3511, sequence-parallel
prefill under pure TP), cherry-picked in that order onto main. Two things are needed on top of them and are in no PR
yet:

- **`moe.fhmoe` getattr fix**: `atom/models/minimax_m3/sp_prefill.py` `_expert_tables` reads `moe.fhmoe` but the FHMoE
  integration commits are not on main, so the attribute does not exist. The one-line fix is
  `if getattr(moe, "fhmoe", None) is not None:` (without it the SP warm-up crashes with an `AttributeError`).
- **`ATOM_M3_INDEXER_CP_MIN_BATCH`**: the cherry-picked #3511 commit needs this entry in `atom/utils/envs.py` (it is
  defined on the 55-commit base branch): `int(os.environ[...]) if "..." in os.environ else None`.
- #3805's cherry-pick also drops two `fhmoe` context lines (the changed lines are identical to the PR head).

### Divergences from the PR heads (what the measured trees contain that a clean merge of the PR heads would not)

The FsW tree is main + the merges above. The following are not byte-identical merges of the current PR heads:

| Where | Divergence |
|---|---|
| ATOM #3510/#3511 | cherry-picks (not the PR commits); #3511 commit carries the extra envs entry above; local commit `db9e767b6` (getattr fix) is in no PR |
| ATOM #3805 | cherry-pick, `fhmoe` context lines dropped; changed lines identical to `6ec5724d5` |
| ATOM #3807 | identical patch to `8633fa102` |
| ATOM #3809/#3810 | net diff `3598884c3..03c20fecc` applied with a union resolution in the SP files; changed lines identical to the PR heads, `qkv_chain.py`/tests byte-identical |
| ATOM #3787 | the tree has the old head merged, then the net diff to `a779e6967`; `decode_moe.py` byte-identical to the new head |
| ATOM #3788 | tree has `b1583fd10`; the PR is now `d6e9e2ad6` (flag off in both recipes) |
| aiter #8584 | merged twice (`aa4d246e0` with `-X theirs` on `minimax_m3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv`, then the fix `77cedd329`); the tree's CSV equals #8584's CSV, which supersedes #8578's 8-wave rows |
| aiter #8555 | the tree merges the old head `8e5592a81`, then cherry-picks the w13_split commit (patch-id identical to `9e02b0320`) |
| aiter chain | #8606/#8607/#8610/#8611 applied as one net diff `41200e689..773f8736e`; the five chain files are byte-identical to `773f8736e` |
| aiter tree | also contains `aiter/ops/triton/comms/pull_collective.py` and `_triton_kernels/comms/pull_collective.py` copied from #8587 for an earlier arm (inert without `ATOM_M3_SP_COMM=pull`) |

## Flags by PR (the full list)

ATOM PRs (ROCm/ATOM):

| Flag | Value used | Needs |
|---|---|---|
| `ATOM_PREFILL_DECODE_INTERVAL`, `ATOM_PREFILL_DELAYER_TARGET_FILL` | `1`, `0.225` | on main |
| `ATOM_SCHED_DECODE_PREFILL_CAP`, `ATOM_SCHED_IDLE_PREFILL_CAP` | `16384`, `16384` | #3775 |
| `ATOM_SCHED_FINISH_HOLD_LEFT`, `_MAX_MS`, `_MAX_TOKENS` | `128`, `500`, `400` | #3775 |
| `ATOM_SCHED_ITL_RISK_MS`, `ATOM_SCHED_ITL_RISK_MAX_MS` | `4.5`, `150` | #3775 |
| `ATOM_PREFIX_DEMOTE_DECODE` | `1` | #3775 |
| `ATOM_M3_PREFILL_TOKEN_GRAPH`, `ATOM_TOKEN_GRAPH_SIZES` | `1`, `256,512,1024` | #3783 |
| `ATOM_MONO_MAX_ROWS` | `8` | #3784 |
| `ATOM_DSPARK_BLOCK_BS_LADDER` | `1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64` | #3789 |
| `ATOM_LM_HEAD_RING` | `1` | #3776 and aiter #8554 |
| `ATOM_DSPARK_FC_RING` | `1` | #3786 and aiter #8573 |
| `ATOM_M3_FUSED_INDEX_SELECT` | `1` | #3785 and aiter #8577 |
| `ATOM_M3_DECODE_MOE` | `1` (c8-c32), `0` (c64) | #3787 and aiter #8548-#8556 |
| `ATOM_M3_SP_PREFILL` | `1` | #3510/#3511 |
| `ATOM_M3_SP_SHARED_LOCAL` | `1` (c8-c32 only) | #3805 |
| `ATOM_M3_QKV_CHAIN` | `1` (c8-c32 only) | #3809, #3810 and aiter #8606/#8607/#8610/#8611 |
| `ATOM_M3_PREFILL_TOPK_FLYDSL` | `0` | #3788 and aiter #8581 (measured: no end-to-end gain) |
| `--spec-decode-kv-cache-dtype` | `fp8` | public flag |

No flag: #3721, #3722, #3723 (DSpark block, always on), #3724 (step watchdog 5 s and GPU guard, constants in
`atom/utils/gpu_guard.py`), #3807 (SP glue fusion).

aiter-only switch: `AITER_FLY_AR_PULL=1` selects the exact-bf16 pull all-reduce + Gemma RMSNorm (aiter #8574); the M3
decoder already calls it.

## What was measured for the individual PRs (c32, single runs, same node; "Fm" = all PRs without SP)

| Change | Result |
|---|---|
| SP prefill (#3510/#3511), Fm -> Fs | +5.3% tok/s, +34% ix_p90 (the largest single lever) |
| + shared-local (#3805), Fs -> Fs3 | +0.4% tok/s, +6% ix_p90 |
| + SP glue (#3807, kernel: -4.3 ms per 16K chunk) | not separately measured end to end |
| + QKV chain, fp8 draft KV, single-copy MoE (FsW) | the c32 point above (96.5%) |
| pull collectives (aiter #8587) | no measurable gain, not in the recipe |
| attention quant-out (aiter #8604, ATOM #3806) | no gain, not in the recipe |
| prefill top-16 selector (#3788/#8581) | no end-to-end gain, flag off |
| `--enable-tbo` for prefill | -0.7%, not used |

## Not covered

The production line additionally uses kernels that are not in aiter yet: the pipelined slots reduce-scatter with the
pull collectives, the MoE gemm tiles for prefill, the fused router + quant, the prefill attention at the production
tile, and decode attention fusions of ~0.1 ms per step each. Remaining gap at c32 (FsW vs C17r): decode step p50 10.0
vs 8.5 ms, prefill forward per request +27%.

`AITER_CONFIG_GEMM_BF16` / `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE`: the MXFP8 M3 rows are in aiter main's
`model_configs` plus the tables of aiter #8578/#8584; the lm_head N=50016 bf16 rows are in aiter #8598 (not in the
recipe; kernel-level -33..-48% on that shape).
