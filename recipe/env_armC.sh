# ARM C = p1/integrated6 "full" (sourced): arm B's flags on the int6 tree + the round-9 changes below.
# Tree: /shared_nfs/ygan/work/p1-integrated6 @ f5c123c7b = p1/integrated5 5f26263a8 + ttft-sched itl_risk.patch
#       + cliff17 ak_wide.patch + mono-stall watchdog_kind.patch + prefill-sp-tbo r9/sp_on_int5.diff (one commit).
# 1. everything arm B sets (int1 base, round-3..int5 flags, mono cap 8, finish hold, capture list, fp8 draft KV)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armB.sh"
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 2. scheduler = ttft-sched NOTES "next steps" arm P2: duty OFF (unset -> code default 1.0), delayer interval 1
#    (>= 1: 0 disables the delayer in TP mode), fill 0.225 and finish hold 128/500/400 as arm B, ITL-risk hold
#    theta 4.5 ms / at most 300 ms per hold (PF cost constants at their defaults 31.6 ms + 18.65 ms/1K tokens,
#    calibrated on arm B's TP prefill steps).
unset ATOM_SCHED_PREFILL_DUTY
export ATOM_PREFILL_DECODE_INTERVAL=1 ATOM_P1_SCHED_ITL_RISK_MS=4.5 ATOM_P1_SCHED_ITL_RISK_MAX_MS=300
# 3. cliff17: AK decode block (split4 top-16 + amd-kernels attention + QOUT) also at 17-32 requests (k7)
export ATOM_P1_AK_WIDE=1
# 4. prefill-sp-tbo R9.1/R9.2: sequence-parallel prefill for prefill-only steps >= 2048 tokens (pull hand-offs,
#    replicated o_proj = +2.9 GB/rank of weights, loaded before KV sizing), TP token graphs kept only <= 1024 so
#    2K/3K steps take eager SP (R9.2: -17% at 2K). Exactly the env of r9 run T3.
export ATOM_M3_SP_PREFILL=1 ATOM_M3_SP_PREFILL_COMM=pull ATOM_M3_SP_PREFILL_OPROJ=full ATOM_M3_SP_PREFILL_QGATHER=1 \
       ATOM_P1_SP_HANDOFFS=1 ATOM_P1_SP_QG_AK=1 ATOM_P1_SP_A2A_UNPACK=1 ATOM_P1_SP_OPROJ_AK=1 ATOM_P1_SP_MOE_SLOTS=1 \
       ATOM_M3_SP_PREFILL_MIN_TOKENS=2048 ATOM_TOKEN_GRAPH_SIZES=256,512,1024
# 5. mono-stall root cause (prefill-step VRAM exhaustion at 0.85): gpu-memory-utilization 0.80 (serve.sh reads MEM;
#    milestone B used KV 10.7% avg / 33.6% max of the 0.85 pool, so 0.80 costs no capacity at c32)
export MEM=0.80
# 6. step watchdog (watchdog_kind.patch: per-kind long-forward / kick counters): explicit, values = code defaults
export ATOM_STEP_WATCHDOG_S=5 ATOM_STEP_KICK_S=5
ARM=C
ARM_REPO_EXPECT=f5c123c7b
# all C variants share one node-local flydsl cache (same tree)
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$NLOC/flydsl_armC}
