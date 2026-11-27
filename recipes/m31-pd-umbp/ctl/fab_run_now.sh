#!/bin/bash
# fab_run_now.sh <tag> <conc> <duration s>: start P and D with the fabric config, then router + gate + AgentX on P.
# Env (optional): AL (4.04; 0 = real acceptance, which the gate needs), LMC_GB + LAZY=1 (P's LMCache lazy tier, GiB
# per rank), P_TP / P_PP / P_CP / P_GPU_MEM / P_DELAYER / P_MAX_BATCHED, P_EXTRA_ENV / D_EXTRA_ENV ("VAR=value ..."),
# D_DPEP=1 (D: DP attention + EP), D_CP, D_DELAYER, SPEC_K, WARMUP_PER_LANE.
cd "$(dirname "$0")" && . ./common.sh
T=$1 C=$2 D=$3 K=$KIT_REMOTE
ts() { TZ=Asia/Shanghai date '+%T'; }
echo "[$(ts)] starting P ($P_HOST) and D ($D_HOST), tag $T, CONC $C, AL=${AL:-4.04} LMC_GB=${LMC_GB:-off} LAZY=${LAZY:-0} UMBP_GB=${UMBP_GB:-off}" \
  "P_TP=${P_TP:-4} P_PP=${P_PP:-1} P_CP=${P_CP:-1} P_GPU_MEM=${P_GPU_MEM:-default} D_DPEP=${D_DPEP:-0} D_CP=${D_CP:-default}"
on "$P_HOST" "SPEC=${P_SPEC:-} SPEC_K=${SPEC_K:-} AL=${AL:-4.04} LAZY=${LAZY:-0} LMC_CHUNK=${LMC_CHUNK:-} PD_LMCACHE_GB=${LMC_GB:-} PD_UMBP_GB=${UMBP_GB:-} UMBP_HUGEPAGES=${UMBP_HUGEPAGES:-0} DELAYER=${P_DELAYER:-} MAX_BATCHED=${P_MAX_BATCHED:-} GPU_MEM=${P_GPU_MEM:-} TP=${P_TP:-} PP=${P_PP:-} PD_CP=${P_CP:-} $K/fab_up.sh p $T $C ATOM_MORIIO_TRANSFER_STATS=1 ${P_EXTRA_ENV:-}" 3000 > "pd_fab_p.out" 2>&1 & pp=$!
on "$D_HOST" "DELAYER=${D_DELAYER:-0} SPEC=${D_SPEC:-} PD_CP=${D_CP:-} DPEP=${D_DPEP:-0} SPEC_K=${SPEC_K:-} AL=${AL:-4.04} $K/fab_up.sh d $T $C ${D_EXTRA_ENV:-}" 3000 > "pd_fab_d.out" 2>&1 & pd=$!
wait $pp; rp=$?; wait $pd; rd=$?
tail -n 2 pd_fab_p.out pd_fab_d.out
[ $rp = 0 ] && [ $rd = 0 ] || { echo "[$(ts)] a server failed (P rc=$rp, D rc=$rd)"; exit 1; }
echo "[$(ts)] both healthy; router + gate + AgentX ${D}s"
on "$P_HOST" "AL=${AL:-4.04} WARMUP_PER_LANE=${WARMUP_PER_LANE:-10} $K/fab_go.sh $T $C $D" $((D + 3600))
echo "[$(ts)] done rc=$?"
