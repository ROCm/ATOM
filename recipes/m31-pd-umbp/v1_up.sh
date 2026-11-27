#!/bin/bash
# v1_up.sh <tag>: check V1 of recipes/m31-pd-umbp. On this tray alone, P's prefill configuration served aggregated:
# PP4 x TP1, eager, MODE=full flags, umbp_store (save_policy lazy) over this tray's UMBP server; no P/D producer and
# no DSpark (a single-node PP4 runs no draft). API on port 8010; waits until it is healthy.
# Env: UMBP_GB (256), GPU_MEM (0.92; lower it to shrink HBM's KV for the read-back), CONC (16), V1_ENV, V1_ARGS.
set -uo pipefail
cd "$(dirname "$0")" && . ./fab_env.sh && . ./config.env && . ./lib.sh && . ./pd.env
TAG=${1:?usage: v1_up.sh <tag>}
export CONC=${CONC:-16} SPEC=none
. ./pd_flags.sh
export SPEC=none CP=0 TP=1 PP=4 GPU_MEM=${GPU_MEM:-0.92} DELAYER=${DELAYER:-0} LMCACHE_GB=
export API_PORT=$P_PORT ENGINE_PORT=$P_ENGINE_PORT CONTAINER
UMBP_GB=${UMBP_GB:-256}
say() { echo "[$(date '+%F %T')] $(hostname -s) v1: $*"; }
gpus_idle || { say "GPUs are in use; not starting"; exit 1; }
./prep_container.sh > /dev/null || exit 1
mkdir -p "$WORK/logs/$TAG"
docker exec -e UMBP_GB="$UMBP_GB" -e GPUS= "$CONTAINER" bash /kit/umbp_server.sh "/logs/$TAG/umbp-server.log" \
  || { say "the UMBP server did not start: $WORK/logs/$TAG/umbp-server.log"; exit 1; }
say "UMBP server up (${UMBP_GB} GiB DRAM)"
KV='{"kv_connector":"umbp_store","kv_role":"offload","kv_connector_extra_config":{"umbp_store.address":"unix:///tmp/umbp/server.sock","umbp_store.chunk_tokens":128,"umbp_store.save_policy":"lazy","umbp_store.lookup_threads":1}}'
_C=/app/aiter/aiter/configs
MX="$_C/a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
for m in dsv41 dsv4 kimik3 minimax_m3; do MX+=":$_C/model_configs/${m}_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"; done
# P's env of run_dpep_base.sh (P_ENV_PP) without its DSpark and MoRIIO parts, plus fab_up.sh's PD_UMBP_GB additions.
P_ENV="ATOM_PREFILL_ORDER=cached_first ATOM_PP_LAYER_PARTITION=kv_balanced ATOM_PP_PARTITION_MAX_EXTRA_LAYERS=2"
P_ENV+=" ATOM_MINIMAX_M3_PA_MXFP8_OUT=0 ATOM_KV_AGG_STALL_S=100000 ATOM_STEP_MEM_LOG=1 NCCL_DEBUG=WARN"
P_ENV+=" AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE=$MX:/kit/pp4_tune_quick/pp4_mxscale_overlay.csv"
P_ENV+=" AITER_CONFIG_GROUPED_FMOE=/app/m3_moe_overlay/m31_moe_overlay_p3cl32k.csv:/kit/pp4_tune_quick/pp4_moe_rows.csv"
P_ENV+=" ATOM_PREFIX_CACHE_POLICY=slru ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5 ATOM_OFFLOAD_MISS_ACCOUNTING=1"
export EXTRA_ENV="$PD_ENV $P_ENV ${V1_ENV:-}" EXTRA_ARGS="--kv-transfer-config $KV ${V1_ARGS:-}"
SLOG=$TAG/server.log
say "PP4 x TP1 + umbp_store lazy, GPU_MEM=$GPU_MEM, log $WORK/logs/$SLOG"
docker exec -e CONC -e MAX_SEQS -e GPU_MEM -e SPEC -e DELAYER -e ATOM_PATCH -e AITER_PATCH -e EXTRA_ENV -e EXTRA_ARGS \
  -e KV_NVFP4 -e INDEX_DTYPE -e CG_LADDER=sparse -e GPUS= -e TP -e PP -e CP -e LMCACHE_GB -e API_PORT -e ENGINE_PORT \
  -e MODEL="$MODEL" "$CONTAINER" bash /kit/server.sh "/logs/$SLOG" || { say "server.sh failed"; exit 1; }
for _ in $(seq 1 90); do
  curl -sf -m 5 "http://127.0.0.1:$API_PORT/health" > /dev/null && { say healthy; exit 0; }
  if grep -qE 'Traceback|GPU Hang|HW Exception|Memory access fault|proc died unexpectedly' "$WORK/logs/$SLOG"; then
    say "error lines in $SLOG:"; grep -m5 -E 'Error|error' "$WORK/logs/$SLOG" | cut -c1-300; exit 1
  fi
  sleep 30
done
say "not healthy within 45 min"
exit 1
