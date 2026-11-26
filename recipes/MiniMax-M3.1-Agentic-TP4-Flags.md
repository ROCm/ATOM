# MiniMax-M3.1 TP4 + DSpark AgentX: per-concurrency recipes, the PRs that provide them, and what was measured

This is the recipe for the AgentX replay (`inferencex-agentx-mvp`) of MiniMax-M3.1-Flash (MXFP4 experts, native MXFP8
attention and shared expert, i.e. no `--online_quant_config`; NVFP4 KV and index caches) with a DSpark draft at TP4 on MI355X, using only ATOM and aiter (no amd-kernels
import, no overlay, no private tuned CSV). Every opt-in switch (the `ATOM_M3_*` and ring/graph on-off flags)
is off by default; the numeric knobs are not all zero by default (`ATOM_PREFILL_DELAYER_TARGET_FILL` defaults to 0.9,
`ATOM_TOKEN_GRAPH_SIZES` to `256,512,1024,2048,3072`, `ATOM_SCHED_FINISH_HOLD_MAX_MS`/`_MAX_TOKENS` to 500/400,
`ATOM_SCHED_ITL_RISK_MAX_MS` to 300), and the recipe sets them explicitly. A flag works once the PR named next to it is
merged. Numbers are from the stack builds described in "Reproducing the stack" (the PR numbers there are the
live ones as of 2026-11-25; several PRs the builds used were later merged into others, see "PRs the builds used that
are closed"); read the caveats first.

## Caveats (read before quoting any number)

1. Every number is a **single 900 s run** (seed 42, synthetic acceptance length 4.04 at k=7; at c64 k=3 with 2.88, the vLLM-official forced value, because
   `--spec-decode-acceptance-length` must be at most k+1 = 4). Measured
   noise on the same node is about 0.2-0.5% tok/s/GPU at c32 and about 1.5% on ix_p90; the c64 noise is not known
   (a repeat is queued). Differences under ~1% tok/s or ~6% ix_p90 between arms are not distinguishable.
2. The reference ("C17r") is the private production stack re-run the same day on the same node, 900 s windows, not the
   3600 s official numbers.
3. Interactivity `ix_p90 = 1000 / ITL p90 (ms)`. **Throughput reaches 85% of the reference at every concurrency, but
   interactivity does not at c32 and c64** (77% and 64-74%); the remaining gap is the decode step (+1.5 ms) at c32 and
   prefill/queueing at c64. This recipe is not yet the user's target of >= 85% on both.
4. The GSM8K gate margin is thin (pooled 8 runs 0.9629 against a 0.9626 line, see "Accuracy gate").
5. The measured trees carry #3510/#3511 as cherry-picks of the earlier 55-commit-branch commits (`2e833df17`,
   `ad44f6d0f`), whose SP MoE chain imports a private `atom.model_ops.minimax_m3.flydsl_moe` package that public main
   does not have. The PR heads are now single commits on main (#3510 `e7c63d7a9`, #3511 `7c403a3e5`) that import the
   chain from aiter (`aiter.ops.flydsl.kernels.minimax_m3_moe_prefill`, ROCm/aiter#8699, draft) instead; the numbers
   below were not repeated with those ported imports (see "Commits needed from #3510/#3511").
6. The aiter side is the image's aiter plus the listed PRs; three of them (#8548 stack, #8555, #8584 now #8582) carry
   conflict resolutions that the stack builder did by hand (see "Divergences from the PR heads").
7. aiperf reports `grace_timeout=True` at c64 for these arms and for the reference (in-flight requests at cutoff,
   `leaked_state` 0); the score is still produced.

## Environment

| Item | Value |
|---|---|
| Image | `rocm/atom-dev:latest` (torch 2.10 rocm 7.2.4, triton 3.8, flydsl 0.5.3, aiter at `/app/aiter-test`; ATOM main calls `init_dist_env(device_communicators=...)`: the 2026-11-17 nightly fails at startup) |
| aiter | the image's `/app/aiter-test` tree with the aiter PRs below merged on top of aiter main `c7d4f9fe1`. The image's JIT `.so` of modules whose C++ changed are **not** rebuilt automatically: delete the stale ones in the copied tree (`custom_all_reduce`, `hipbsolgemm`, `fused_qknorm_idxrqknorm`, `mla_nvfp4`, `kpool_compress`); the first boot rebuilds them (~5-6 min). Never boot two servers on one writable tree. |
| ATOM | ATOM main `d0dec1480` plus the ATOM PRs below |
| Server | as `recipes/MiniMax-M3-Agentic-InferenceX.md` **except** that (a) `--online_quant_config` (ptpc_fp8 attention) is **omitted**: the measured arms ran native MXFP8 attention and shared-expert weights; (b) `--spec-decode-acceptance-rate 0.5933` is **replaced** by `--spec-decode-acceptance-length 4.04` at k=7 (the rate would give `1 + 7 * 0.5933 = 5.15`; the two options are mutually exclusive) and by `--spec-decode-acceptance-length 2.88` at k=3 (c64; the accepted range is `[1, k+1]`, so 4.04 is rejected at k=3); with `--kv_cache_dtype nvfp4 --index-cache-dtype nvfp4 --method dspark --draft-model <model>/dspark --num-speculative-tokens 7` (3 at c64), `--max-num-seqs 64`, `--gpu-memory-utilization 0.85`, `--cudagraph-capture-sizes [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]`, `--spec-decode-kv-cache-dtype fp8`, `AITER_LOG_LEVEL=WARNING`, `ATOM_FORCE_ATTN_TRITON=1`, `PYTHONHASHSEED=0` |
| Client | AIPerf `inferencex-agentx-mvp`, seed 42, 900 s per point |

## The two recipes

Common to both (the F set, all provided by PRs listed below): `ATOM_PREFILL_DECODE_INTERVAL=1`,
`ATOM_PREFILL_DELAYER_TARGET_FILL=0.225`, `ATOM_SCHED_DECODE_PREFILL_CAP=16384`, `ATOM_SCHED_IDLE_PREFILL_CAP=16384`,
`ATOM_SCHED_FINISH_HOLD_LEFT=128`, `ATOM_SCHED_FINISH_HOLD_MAX_MS=500`, `ATOM_SCHED_FINISH_HOLD_MAX_TOKENS=400`,
`ATOM_SCHED_ITL_RISK_MS=4.5`, `ATOM_SCHED_ITL_RISK_MAX_MS=150`, `ATOM_PREFIX_DEMOTE_DECODE=1`,
`ATOM_M3_PREFILL_TOKEN_GRAPH=1`, `ATOM_TOKEN_GRAPH_SIZES=256,512,1024`, `ATOM_MONO_MAX_ROWS=8`,
`ATOM_DSPARK_BLOCK_BS_LADDER=1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64`, `ATOM_LM_HEAD_RING=1`,
`ATOM_DSPARK_FC_RING=1` (both ring flags belong to the retired ATOM #3776/#3786, see the flag table),
`ATOM_M3_FUSED_INDEX_SELECT=1`, `AITER_FLY_AR_PULL=1`, `ATOM_M3_PREFILL_TOPK_FLYDSL=0`, `ATOM_M3_SP_PREFILL=1`.

### c8 / c16 / c32: "FsW"

Common set plus:

```
ATOM_M3_SP_SHARED_LOCAL=1      # owner-local shared expert (#3805)
ATOM_M3_QKV_CHAIN=1            # QKV front chain, 17-128 decode rows (#3809/#3810, aiter #8607; the measured tree used the
                               # pre-merge chain #8606/#8607/#8610/#8611, see "PRs the builds used that are closed")
ATOM_M3_DECODE_MOE=1           # decode MoE chain (old #3787 head, aiter #8548-#8556 incl. the w13_split commit of #8555;
                               # #3787 is being rewritten onto aiter #8676 fused_moe(shared_folded=True) and the flag goes away)
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
| `ATOM_M3_DECODE_MOE=1` with #3787 `a779e6967` + aiter #8555 w13_split (the interface #8676 replaces) | +2.4 GB peak (routed w13 is read in place; 36 MiB held per MoE layer and rank) | FsW boot log: peak_torch 76.39 GB |
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

aiter (aiter main `c7d4f9fe1` as the base of the builds; open PRs as of 2026-11-25):

1. `#8548 -> #8553 -> #8555 -> #8556` (decode MoE chain; #8548 now includes the decode router GEMM work of the closed
   #8551/#8552; the builds' #8555 head is `#8553`'s plus the `w13_split` commit `9e02b0320`). #8676 sits on top of
   these four and is the final decode-MoE interface (`fused_moe(shared_folded=True)`, tuned rows, `--decode-chain`
   tuner); the old #3787 head used the mega_moe_tp ops directly.
2. `#8675` (small-M BF16 ring GEMM, policies `rg` for lm_head and `rk` for the DSpark draft fc, selected by `tgemm`
   tuned rows; only the `rk` rows are reached from ATOM today: the `rg` rows need a preshuffled lm_head weight passed
   with `is_shuffled=True`, and `ParallelLMHead.local_logits` still passes the row-major weight, so the lm_head ring
   needs a separate thin ATOM change that is not in any PR; it replaces the closed #8554/#8573), `#8574` (pull all-reduce + Gemma RMSNorm).
3. `#8575` (sparse decode attention: all-wave merge, in-kernel index top-16, MXFP8 `quant_out`; it contains the closed
   #8577 and #8604).
4. `#8472` (HIP `fused_qknorm_idxrqknorm`: DPP exchanges, xor-atomic V nibbles, hardware converts, NVFP4 insert, prefill
   variant; it contains the closed #8473, #8576 and #8580).
5. `#8578 -> #8582` (o_proj_full 8-wave picks; persistent 4-wave GEMM, tuner backend and o_proj_full rows; #8582
   contains the closed #8584, including the inference-tensor scale-cache-key fix), (`#8599`, MXFP8 small-M config rows, is not in the measured trees and not in the recipe: see below).
6. `#8581` (prefill top-16 selector inside `topk_per_row_small_k`; present, flag off).
7. `#8607` (draft) for the QKV front chain: one op, 1-64 rows one launch and 65-128 rows split-K, reading plain
   row-major activation codes from `per_group_quant_hip`. The closed #8606 (shuffled quantizer), #8610 and #8611 are
   folded into it (#8606 is withdrawn: the front no longer needs its layout).

Not part of the recipe: #8587 (HIP general `all_to_all` / `all_to_all_buffer`, MXFP8 all-to-all; used by ATOM #3812
only, c32 +0.1..0.6%, no flag), attention quant-out (ATOM #3806 with the `quant_out` of #8575: no gain), #8598 (merged,
lm_head tuned rows) / #8599 (tuned CSV rows, kernel-level only; the 069 stack tree `stack069/aiter-F` has no #8599 merge, so no measured arm used its rows).

### PRs the builds used that are closed

The measured trees were built before these were folded; every one maps to a live PR and the content is the same
kernel family:

| Closed PR | What it was | Now in |
|---|---|---|
| aiter #8473, #8576, #8580 | NVFP4 converts, FlyDSL NVFP4 insert, DPP / xor-atomic V in `fused_qknorm_idxrqknorm` | #8472 |
| aiter #8551, #8552 | decode router GEMM split-K, 128-row n-blocks kernel | #8548 |
| aiter #8554, #8573 | small-M BF16 ring GEMM (lm_head), DSpark fc ring | #8675 (draft fc: ATOM needs no call site, the `rk` rows are picked by `tgemm`; lm_head: the `rg` rows need a preshuffled weight and ATOM passes the row-major one, so the lm_head ring is not reached; ATOM #3776 and #3786 are closed) |
| aiter #8577, #8604 | index top-16 in the attention, `quant_out` | #8575 |
| aiter #8584 | 4-wave backend, o_proj_full rows | #8582 |
| aiter #8606, #8610, #8611 | MXFP8 shuffled quantizer, split-3 GEMM, reduce + insert | #8607 (#8606 withdrawn) |

ATOM (main `d0dec1480`):

1. `#3721`, `#3722`, `#3723`, `#3724` (any order; each carries the others' `dspark.py` resolution).
2. `#3775` (scheduler policies), `#3783` (prefill token graphs), `#3784` (mono row cap), `#3785` (fused index select,
   needs aiter #8575), `#3789` (draft block batch ladder), `#3788` (prefill top-16; code present,
   `ATOM_M3_PREFILL_TOPK_FLYDSL=0`, needs aiter #8581). `#3776` (LM head ring) and `#3786` (draft fc ring) are closed:
   the draft fc ring is reached through aiter #8675's `rk` tuned rows, while the lm_head ring (`rg`) is not reached
   from ATOM until it keeps the lm_head weight shuffled with `is_shuffled=True`; `ATOM_LM_HEAD_RING` /
   `ATOM_DSPARK_FC_RING` exist only in the measured trees, not in a live PR. Note the measured trees called the draft fc ring through `fc_taps`
   (#3786); #3723's `write_context_kv` calls `self.fc` directly and bypassed `fc_taps`, so the ring was not on that
   fused path there. Whether the tuned-row route covers that call is not measured.
3. `#3787` (decode MoE chain; the measured head `a779e6967` needs aiter #8548-#8556; the rewrite onto aiter #8676
   drops `ATOM_M3_DECODE_MOE`).
4. SP prefill: **#3510 then #3511** (see below), then `#3805` (shared-local), then `#3807` (glue).
5. `#3809`, `#3810` (QKV chain; draft, need aiter #8607; planned to merge into one thin PR over it).
   Also `#3811` (`getattr` fix for `moe.fhmoe`), `#3808` (norm + quant hand-off, no end-to-end gain at c32, not in the
   recipe) and `#3812` (o_proj all-to-all through aiter `all_to_all`, needs aiter #8587, no flag) are stacked on the SP
   prefill base.
6. This document (`#3790`).

### Commits needed from #3510/#3511

What the measured trees used: two cherry-picked commits of the 55-commit private branch, `2e833df17` (#3510, prefill
DBO under pure TP; code only, not enabled) and `ad44f6d0f` (#3511, sequence-parallel prefill under pure TP), in that
order, on a tree that also contains the private `atom/model_ops/minimax_m3/flydsl_moe` package that `ad44f6d0f`
imports (public main has no such package: those two commits alone fail at the first eligible SP prefill).

What a public build should use instead: the current PR heads, #3510 `e7c63d7a9` then #3511 `7c403a3e5` (single commits
on main). #3511 ports the imports to aiter's `minimax_m3_moe_prefill` package and needs **ROCm/aiter#8699** (draft; it
is not in the aiter list above because no measured arm contained it). The dependent patches (#3805, #3807, #3811,
#3812) are stacked on the old private base, so before reproducing the numbers they have to be re-applied on the new
heads; that rebase and a re-measurement with the ported imports have not been done.

- **`moe.fhmoe`**: `_expert_tables` in `sp_prefill.py` read `moe.fhmoe`, which main does not have (the FHMoE
  integration commits are not on main). ATOM #3811 is the one-line `getattr(moe, "fhmoe", None)` fix, a published PR;
  the measured trees had it as a local commit.
- `ATOM_M3_INDEXER_CP_MIN_BATCH` needs no entry: nothing in main + #3510 + #3511 reads it (ATOM #3811's body corrects
  the earlier note that said it did).
- #3805's cherry-pick also drops two `fhmoe` context lines (the changed lines are identical to the PR head).

### Divergences from the PR heads (what the measured trees contain that a clean merge of the PR heads would not)

The FsW tree is main + the merges above. The following are not byte-identical merges of the current PR heads:

| Where | Divergence |
|---|---|
| ATOM #3510/#3511 | cherry-picks of the old private-branch commits (not the current PR heads); local commit `db9e767b6` is the `moe.fhmoe` getattr fix, now ATOM #3811 |
| ATOM #3805 | cherry-pick, `fhmoe` context lines dropped; changed lines identical to `6ec5724d5` |
| ATOM #3807 | identical patch to `8633fa102` |
| ATOM #3809/#3810 | net diff `3598884c3..03c20fecc` applied with a union resolution in the SP files; changed lines identical to the PR heads, `qkv_chain.py`/tests byte-identical |
| ATOM #3787 | the tree has the old head merged, then the net diff to `a779e6967`; `decode_moe.py` byte-identical to the new head |
| ATOM #3788 | tree has `b1583fd10`; the PR is now `d6e9e2ad6` (flag off in both recipes) |
| aiter #8584 (closed, now #8582) | merged twice (`aa4d246e0` with `-X theirs` on `minimax_m3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv`, then the fix `77cedd329`); the tree's CSV equals #8584's CSV, which supersedes #8578's 8-wave rows |
| aiter #8555 | the tree merges the old head `8e5592a81`, then cherry-picks the w13_split commit (patch-id identical to `9e02b0320`) |
| aiter chain (closed, now #8607) | #8606/#8607/#8610/#8611 applied as one net diff `41200e689..773f8736e`; the five chain files are byte-identical to `773f8736e` |
| aiter tree | also contains `aiter/ops/triton/comms/pull_collective.py` and `_triton_kernels/comms/pull_collective.py` copied from #8587 for an earlier arm (inert without `ATOM_M3_SP_COMM=pull`) |

## Flags by PR (the full list)

ATOM PRs (ROCm/ATOM):

| Flag | Value used | Needs |
|---|---|---|
| `ATOM_PREFILL_DECODE_INTERVAL`, `ATOM_PREFILL_DELAYER_TARGET_FILL` | `1`, `0.225` | on main (defaults 0 and 0.9) |
| `ATOM_SCHED_DECODE_PREFILL_CAP`, `ATOM_SCHED_IDLE_PREFILL_CAP` | `16384`, `16384` | #3775 |
| `ATOM_SCHED_FINISH_HOLD_LEFT`, `_MAX_MS`, `_MAX_TOKENS` | `128`, `500`, `400` | #3775 |
| `ATOM_SCHED_ITL_RISK_MS`, `ATOM_SCHED_ITL_RISK_MAX_MS` | `4.5`, `150` | #3775 |
| `ATOM_PREFIX_DEMOTE_DECODE` | `1` | #3775 |
| `ATOM_M3_PREFILL_TOKEN_GRAPH`, `ATOM_TOKEN_GRAPH_SIZES` | `1`, `256,512,1024` | #3783 (`ATOM_TOKEN_GRAPH_SIZES` is on main, default `256,512,1024,2048,3072`) |
| `ATOM_MONO_MAX_ROWS` | `8` | #3784 |
| `ATOM_DSPARK_BLOCK_BS_LADDER` | `1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64` | #3789 |
| `ATOM_LM_HEAD_RING` | `1` (measured trees only) | ATOM #3776 is closed; aiter #8675 has `rg` rows but they need a preshuffled lm_head weight, which ATOM does not pass yet: no live replacement |
| `ATOM_DSPARK_FC_RING` | `1` (measured trees only) | ATOM #3786 is closed; aiter #8675 tuned rows (`rk`) replace it, no ATOM flag |
| `ATOM_M3_FUSED_INDEX_SELECT` | `1` | #3785 and aiter #8575 |
| `ATOM_M3_DECODE_MOE` | `1` (c8-c32), `0` (c64) | old #3787 head and aiter #8548-#8556; the #3787 rewrite on aiter #8676 drops the flag |
| `ATOM_M3_SP_PREFILL` | `1` | #3510/#3511 |
| `ATOM_M3_SP_SHARED_LOCAL` | `1` (c8-c32 only) | #3805 |
| (no flag) | SP glue fusion #3807; o_proj all-to-all through aiter `all_to_all` #3812 (needs aiter #8587; the earlier `ATOM_M3_SP_COMM` knob is removed there) | |
| `ATOM_M3_QKV_CHAIN` | `1` (c8-c32 only) | #3809, #3810 and aiter #8607 (draft) |
| `ATOM_M3_PREFILL_TOPK_FLYDSL` | `0` | #3788 and aiter #8581 (measured: no end-to-end gain; off by default) |
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
| pull collectives (aiter #8587; ATOM #3812 now uses only the all-to-all) | +0.1..0.6% at c32 on the earlier version, not in the recipe |
| attention quant-out (aiter #8575's `quant_out`, ATOM #3806) | no gain (-1.1% at c32), not in the recipe |
| prefill top-16 selector (#3788/aiter #8581) | no end-to-end gain, flag off |
| `--enable-tbo` for prefill | -0.7%, not used |

## Not covered

The production line additionally uses kernels that are not in aiter yet: the pipelined slots reduce-scatter with the
pull collectives, the MoE gemm tiles for prefill, the fused router + quant, the prefill attention at the production
tile, and decode attention fusions of ~0.1 ms per step each. Remaining gap at c32 (FsW vs C17r): decode step p50 10.0
vs 8.5 ms, prefill forward per request +27%.

`AITER_CONFIG_GEMM_BF16` / `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE`: the MXFP8 M3 rows are in aiter main's
`model_configs` plus the tables of aiter #8578/#8582; the lm_head N=50016 bf16 rows are in aiter #8598 (merged;
kernel-level -33..-48% on that shape).
