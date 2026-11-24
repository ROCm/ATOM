# MiniMax-M3.1 TP4 + DSpark AgentX: the opt-in flags and the PRs that provide them

This is the flag set for the AgentX replay (`inferencex-agentx-mvp`) of
MiniMax-M3.1-Flash (MXFP4 experts, native MXFP8 attention, NVFP4 KV and index
caches) with a DSpark draft at TP4 on MI355X, using only ATOM and aiter. Every
flag is off by default; a flag works once the PR named next to it is merged.

## Environment

| Item | Value |
|---|---|
| Image | `rocm/atom-dev:latest` (aiter 9f3cba25e or newer: ATOM main calls `init_dist_env(device_communicators=...)`; the 2026-11-17 nightly fails at startup) |
| Server | as `recipes/MiniMax-M3-Agentic-InferenceX.md` with `--kv_cache_dtype nvfp4 --index-cache-dtype nvfp4 --method dspark --draft-model <model>/dspark --num-speculative-tokens 7`, `--max-num-seqs 64`, `--gpu-memory-utilization 0.85`, `--cudagraph-capture-sizes [1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,56,64]`, `AITER_LOG_LEVEL=WARNING`, `ATOM_FORCE_ATTN_TRITON=1`, `PYTHONHASHSEED=0` |
| Client | AIPerf `inferencex-agentx-mvp`, concurrency 32, seed 42 |

## Flags by PR

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

No flag: #3721, #3722, #3723 (DSpark block, always on) and #3724 (step
watchdog 5 s and GPU guard, constants in `atom/utils/gpu_guard.py`).

aiter-only switch: `AITER_FLY_AR_PULL=1` selects the exact-bf16 pull
all-reduce + Gemma RMSNorm (aiter #8574); the M3 decoder already calls it.

Merge order for #3721, #3722, #3723: any (each branch carries the others'
resolution of the shared `dspark.py` regions).

## What was measured

Same node, public main 659c3ba49, AgentX c32, 900 s, seed 42, synthetic
acceptance length 4.04, one run per arm, only atom and aiter:

| Arm | tok/s/GPU | ix_p90 | TTFT p50 / p90 (s) |
|---|---|---|---|
| No flag | 51,640 | 102.1 | 0.37 / 1.00 |
| #3775 flags + #3783 token graphs | 52,103 | 117.3 | 0.62 / 1.83 |

The other rows of the table are not measured in this document; the stacked
benchmark of all PRs is tracked separately.

## Not covered

The production line additionally uses kernels that are not in aiter yet
(MoE tiles and fused block, QKV insert + GEMM, prefill sparse attention,
SP-prefill collectives, decode attention insert). A recipe that reaches the
production throughput needs those aiter kernels and their ATOM call sites;
this file lists only what exists as a PR.

`AITER_CONFIG_GEMM_BF16` / `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE`:
the MXFP8 M3 rows are in aiter main's `model_configs`; the bf16 table used in
development has 6462 rows of which only 604 (13 N,K shapes) are in aiter main's
`bf16_tuned_gemm.csv` (the lm_head N=50016, K=6144 shape is). The effect of the
missing rows has not been isolated.
