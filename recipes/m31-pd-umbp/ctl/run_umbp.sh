#!/bin/bash
# run_umbp.sh [verify] [gsm8k] [c192] [agentx]: M3.1 1P1D with P's DRAM tier chosen by ARM (LLM-Booster #826):
#   ARM=umbp (default): umbp_store lazy over a UMBP server of UMBP_GB GiB (640 on 1 TB trays; 1720 on 2 TB);
#   ARM=lmc: LMCache lazy, LMC_GB GiB per PP stage (160 on 1 TB trays: 4 x 160 = the UMBP arm's 640; 430 on 2 TB);
#   ARM=none: no DRAM tier, HBM only, with the HBM prefix policy the tier arms run (SLRU, half protected).
#   VERIFY_MEM: P's gpu_mem for the verify step only (default PP4_MEM): a smaller HBM makes C16 evict into the tier.
#   VERIFY_S: the verify step's AgentX seconds (300).
#   P = PP4 x TP1, CP off, eager, gpu_mem 0.92, LMCache lazy 430 GiB per rank, quick PP4 overlays (kit/pp4_tune_quick).
#   D = -tp 4 --enable-dp-attention --enable-expert-parallel (DP4 x TP1, EP4, MoRI EP v2 high-throughput) + DSpark K=7.
#   verify  AgentX C16 300 s at AL=0 with ATOM_MORIIO_VERIFY=1 on P and D; needs GATE PASS, aiperf rc=0, verify bad=0.
#   gsm8k   GSM8K 5-shot (1319) through the router at AL=0; stops under GSM8K_MIN (0.945).
#   c192    AgentX C192 900 s at forced AL=4.04; then kit/summarize.py.
#   agentx  AgentX at CONC (192) for DUR s (900) at forced AL=4.04 (e.g. CONC=300 DUR=3600); then kit/summarize.py.
# Run from the control host (see common.sh). Logs on the trays in $WORK/logs/<tag>.
cd "$(dirname "$0")" && . ./common.sh || exit 1
STEPS=${*:-verify gsm8k c192}
ts() { TZ=Asia/Shanghai date '+%F %T'; }
log() { echo "[$(ts)] $PFX: $*" | tee -a run_umbp.log; }
ARM=${ARM:-umbp}
PFX=${TAG_PREFIX:-$ARM}
GSM8K_MIN=${GSM8K_MIN:-0.945}
WORK_REMOTE=$(on "$P_HOST" "cd $KIT_REMOTE && . ./fab_env.sh >/dev/null && echo \$WORK" 30)
[ -n "$WORK_REMOTE" ] || { echo "cannot read WORK from $P_HOST:$KIT_REMOTE/fab_env.sh (site.env?)"; exit 1; }
W=$WORK_REMOTE/logs

ARM_ENV=
case $ARM in
  umbp) COMMON="UMBP_GB=${UMBP_GB:-640} P_DELAYER=0 D_DPEP=1 D_CP=0" ;;
  lmc) COMMON="LAZY=1 LMC_GB=${LMC_GB:-160} P_DELAYER=0 D_DPEP=1 D_CP=0" ;;
  # The tier arms set SLRU (server.sh for LMCache, fab_up.sh for UMBP); without a tier the kit would leave LRU.
  none) COMMON="P_DELAYER=0 D_DPEP=1 D_CP=0"; ARM_ENV="ATOM_PREFIX_CACHE_POLICY=slru ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5" ;;
  *) echo "ARM must be umbp, lmc or none"; exit 2 ;;
esac
PP4="P_TP=1 P_PP=4 P_CP=0 P_GPU_MEM=${PP4_MEM:-0.92}"
# ATOM_KV_AGG_STALL_S=100000: no SIGUSR1 stack dump on a KV aggregation stall (the dump preceded a P stage-0 SIGSEGV).
P_ENV_PP="${ARM_ENV:+$ARM_ENV }ATOM_DSPARK_FC_TP=1 ATOM_DSPARK_CTX_KV_ONLY=1 ATOM_PREFILL_ORDER=cached_first ATOM_MORIIO_WRITE_THREAD=1 ATOM_MORIIO_TRANSFER_STATS=1 ATOM_PP_LAYER_PARTITION=kv_balanced ATOM_PP_PARTITION_MAX_EXTRA_LAYERS=2 ATOM_OFFLOAD_PP_SPLIT_BY_KV_BYTES=1 ATOM_MINIMAX_M3_PA_MXFP8_OUT=0 ATOM_MORIIO_PUSH_SEGMENT_BATCH=262144 ATOM_KV_AGG_STALL_S=100000 ATOM_STEP_MEM_LOG=1 NCCL_DEBUG=WARN"
# Quick PP4 overlays: an AITER_CONFIG_* list replaces aiter's default list, so the mxscale list repeats the image's own
# files before the overlay; the grouped-MoE list is the image's M3.1 overlay plus the quick rows.
_C=/app/aiter/aiter/configs
MX="$_C/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
for m in dsv41 dsv4 kimik3 minimax_m3; do MX+=":$_C/model_configs/${m}_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"; done
[ "${PP4_TUNE:-1}" = 1 ] && P_ENV_PP+=" AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=$MX:/kit/pp4_tune_quick/pp4_mxscale_overlay.csv AITER_CONFIG_GROUPED_FMOE=/app/m3_moe_overlay/m31_moe_overlay_p3cl32k.csv:/kit/pp4_tune_quick/pp4_moe_rows.csv"
# ROCm/ATOM #590 DPEP env: MoRI EP v2 (v1 is wave64-only), compact FP4 dispatch + fused stage 1/2, replicated LM head,
# no shared-expert fusion (GSM8K 0.78 with it), DP session affinity (a multi-turn prefix stays on one DP rank).
DPEP_ENV="ATOM_MORI_V2=1 ATOM_MEGA_STAGE1_FUSED=1 ATOM_MEGA_STAGE2_FUSED=1 MEGA_DISPATCH_WIRE=fp4 ATOM_DP_LM_HEAD_MODE=default ATOM_USE_CUSTOM_ALL_GATHER=0 ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=0 ATOM_DP_SESSION_AFFINITY=1 MORI_AMD_SMI_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib/libamd_smi.so.27"
D_ENV="ATOM_MORIIO_INCREMENTAL_PUSH=1 ATOM_DSPARK_CTX_KV_ONLY=1 ATOM_MINIMAX_M3_PA_MXFP8_OUT=0 $DPEP_ENV ${D_MORE_ENV:-}"

run() { local tag=$1 conc=$2 dur=$3; shift 3
  ./fab_down.sh > /dev/null; log "start $tag C$conc ${dur}s"
  env AL=0 WARMUP_PER_LANE=1 $COMMON "$@" ./fab_run_now.sh "$tag" "$conc" "$dur" > "fab_run_$tag.log" 2>&1
  tail -n 3 "fab_run_$tag.log"
  log "$tag p: $(on "$P_HOST" "echo faults=\$(grep -ciE 'Traceback|Memory access fault|GPU Hang' $W/$tag/server-p.log)" 60)"
  log "$tag d: $(on "$D_HOST" "echo faults=\$(grep -ciE 'Traceback|Memory access fault|GPU Hang' $W/$tag/server-d.log); grep -m1 -o 'Enable dp attention.*' $W/$tag/server-d.log" 60 | tr '\n' ' ')"; }

for step in $STEPS; do case $step in
verify)
  G=$PFX-gate-verify
  run $G 16 ${VERIFY_S:-300} $PP4 P_GPU_MEM=${VERIFY_MEM:-${PP4_MEM:-0.92}} "P_EXTRA_ENV=$P_ENV_PP ATOM_MORIIO_VERIFY=1" "D_EXTRA_ENV=$D_ENV ATOM_MORIIO_VERIFY=1"
  grep -q "GATE PASS" "fab_run_$G.log" && grep -q "aiperf rc=0" "fab_run_$G.log" || { log "$G: gate or AgentX failed; servers left up"; exit 1; }
  DL=$W/$G/server-d.log
  bad=$(on "$D_HOST" "grep -hc 'moriio-verify.*\(differ from\|written by none\|sent no PAGE\|sent no state\|unchanged after the\|source blocks changed\)' $DL" 60)
  ok=$(on "$D_HOST" "grep -hc 'moriio-verify.*match' $DL" 60)
  held_bad=$(on "$D_HOST" "grep -hc 'moriio-verify.*locally held prefix blocks changed since their push landed' $DL" 60)
  log "$G verify ok=${ok:-?} bad=${bad:-?} held_bad=${held_bad:-?}"
  on "$P_HOST" "curl -s -m 20 localhost:8010/metrics > $W/$G/p_metrics_end.txt; grep -E '^atom:lmcache_(loaded|saved)_tokens_total|^atom:lmcache_lazy_evicted' $W/$G/p_metrics_end.txt | tr '\n' ' '" 60 | sed 's/^/  P counters: /' | tee -a run_umbp.log
  [ "${bad:-1}" = 0 ] && [ "${ok:-0}" -gt 0 ] && [ "${held_bad:-1}" = 0 ] || { log "$G: KV verify failed"; exit 1; } ;;
gsm8k)
  G=$PFX-gsm8k
  run $G 10 0 $PP4 "P_EXTRA_ENV=$P_ENV_PP" "D_EXTRA_ENV=$D_ENV" && grep -q "GATE PASS" "fab_run_$G.log" || { log "$G: servers or gate failed"; exit 1; }
  on "$P_HOST" "cd $KIT_REMOTE && . ./fab_env.sh && CONTAINER=$CTR API_PORT=8000 TAG=$G ./gsm8k.sh" 10800 | tee "gsm8k_$G.txt"
  acc=$(on "$P_HOST" "awk -F'|' '\$4 ~ /flexible-extract/ {gsub(/ /, \"\", \$8); print \$8}' $W/$G/gsm8k.log" 60 | head -1)
  log "$G flexible-extract ${acc:-none} (min $GSM8K_MIN)"
  awk -v a="${acc:-0}" -v m="$GSM8K_MIN" 'BEGIN { exit !(a >= m) }' || { log "GSM8K under $GSM8K_MIN"; ./fab_down.sh; exit 1; } ;;
c192|agentx)
  conc=192 dur=900
  [ "$step" = agentx ] && conc=${CONC:-192} dur=${DUR:-900}
  t=$PFX-c$conc
  run $t "$conc" "$dur" AL=4.04 $PP4 "P_EXTRA_ENV=$P_ENV_PP" "D_EXTRA_ENV=$D_ENV" && grep -q "aiperf rc=0" "fab_run_$t.log" \
    && on "$P_HOST" "curl -s -m 20 localhost:8010/metrics > $W/$t/p_metrics_end.txt; grep -E 'OFFLOAD-MISS-ACCOUNT' $W/$t/server-p.log | tail -1 > $W/$t/p_miss_end.txt; docker exec $CTR python3 /kit/summarize.py /logs/$t" 300 | tee "summary_$t.txt" ;;
*) echo "unknown step $step (verify | gsm8k | c192 | agentx)"; exit 2 ;;
esac; done
./fab_down.sh > /dev/null
log "done; P/D down"
