#!/bin/bash
# GLM-5.2 PP4×TP1 Prefill + TP4 Decode (PD), SparseKV with the GPU cold tier.
#
# Same topology as start_glm52_pp4pd_sparsekv.sh. The addition is a second cold
# pool in spare HBM (ATOM_SPARSEKV_GPU_COLD_PAGES): prefill still RDMAs a
# request's whole KV into the host cold pool, then the decode worker promotes
# what fits into the GPU tier and hands those host pages back. Scheduler
# admission gates on both pools, so the batch ceiling becomes host+GPU instead
# of host alone.
#
# Container: atom_pp4pd_test
# GPUs:      0-3 = prefill (PP4×TP1, no SparseKV), 4-7 = decode (TP4, SparseKV)
# Ports:     8010 = prefill API, 8020 = decode API, 30000 = mesh proxy
#
# Usage:
#   docker exec -it atom_pp4pd_test bash \
#     /it-share/yajizhan/code/ATOM/scripts/start_glm52_pp4pd_gpucold.sh
#
# Tunables (env overrides):
#   GPU_UTIL       --gpu-memory-utilization (default 0.85). THIS sizes the GPU
#                  cold tier: whatever HBM is left once the model, index cache
#                  and hot buffer are allocated becomes cold tier, leaving
#                  (1 - GPU_UTIL) of the device as headroom.
#   GPU_COLD       1 = auto-size the tier (default), 0 = kill-switch (host-only
#                  two-tier build, for the A/B baseline)
#   MAX_NUM_SEQS   decode concurrent seqs        (default 20)
#   MAX_MODEL_LEN  decode context cap            (default 1048576)
#   HOT_BUFFER     per-req resident hot tokens   (default 8192)
#   RATIO          host cold pool multiple       (default 16; 32 hangs at startup)
#   PREFETCH       IndexShare group prefetch     (default 1)
#
# The image's installed atom (/app/ATOM) predates this feature, so PYTHONPATH
# shadows it with this checkout. The aiter swap ops are ported into the image's
# /app/aiter-test by port_sparsekv_aiter.py (idempotent, run below).

set -euo pipefail

ATOM_SRC=/it-share/yajizhan/code/ATOM
AITER_PORT_SCRIPT="$ATOM_SRC/.claude/scratch/port_sparsekv_aiter.py"

MODEL=/mnt/models/GLM-5.2-MXFP4
PREFILL_PORT=8010
DECODE_PORT=8020
MESH_PORT=30000
HANDSHAKE_PORT=6301

GPU_COLD="${GPU_COLD:-1}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-20}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048576}"
HOT_BUFFER="${HOT_BUFFER:-8192}"
PREFETCH="${PREFETCH:-1}"
RATIO="${RATIO:-16}"
# The prefill node is the throughput ceiling for the agentic trace (both c48
# rounds computed 94.2M prompt tokens at 26.0K tok/s), and 81% of its requests
# evict cached blocks, so how much of its HBM goes to the prefix cache decides
# how many of those tokens it has to compute at all.
PREFILL_GPU_UTIL="${PREFILL_GPU_UTIL:-0.85}"
PREFILL_CHUNK="${PREFILL_CHUNK:-8192}"
# atomesh defaults to 1800s, which is shorter than the agentic trace's own tail:
# its longest outputs run to ~32K tokens, and at a decode ITL of 100-200 ms that
# is 3200-6600s of legitimate streaming. The default cut one such request in
# results/js_c16_m24_r14_rerun and reported it as a failure. Raise it past the
# workload rather than let a slow-but-healthy request look like an error; the
# cost is that a genuine hang takes this long to surface at the client (the
# decode-log hang signatures catch those sooner).
MESH_REQUEST_TIMEOUT="${MESH_REQUEST_TIMEOUT:-7200}"

# -1 tells the worker to auto-size from free HBM; 0 disables the tier.
GPU_COLD_PAGES=$([ "$GPU_COLD" = "0" ] && echo 0 || echo -1)

# ── put this checkout ahead of the image's installed atom ────────────
export PYTHONPATH="${ATOM_SRC}${PYTHONPATH:+:$PYTHONPATH}"
RESOLVED=$(python3 -c 'import atom, os; print(os.path.dirname(atom.__file__))' 2>/dev/null || echo "")
if [ "$RESOLVED" != "${ATOM_SRC}/atom" ]; then
  echo "ERROR: atom resolves to '${RESOLVED}', expected '${ATOM_SRC}/atom'."
  echo "       The servers would run the image's build, without the GPU cold tier."
  exit 1
fi
echo ">>> atom: $RESOLVED"

# ── port the sparsekv aiter ops into the image (idempotent) ─────────
if [ -f "$AITER_PORT_SCRIPT" ]; then
  echo ">>> Porting sparsekv aiter ops into /app/aiter-test ..."
  python3 "$AITER_PORT_SCRIPT"
else
  echo "ERROR: aiter port script not found at $AITER_PORT_SCRIPT"
  exit 1
fi
python3 - <<'PY' || exit 1
import importlib.util, sys
if importlib.util.find_spec("aiter.ops.sparsekv_swap") is None:
    sys.exit("ERROR: aiter.ops.sparsekv_swap not importable after port")
print(">>> aiter sparsekv ops OK")
PY

# ── cleanup ──────────────────────────────────────────────────────────
# The workers rename themselves (ATOM::TP0, ATOM::EngineCore_PP1, ...), so
# `pkill -f openai_server` alone leaves them holding VRAM and the new decode
# server then OOMs on allocation. Kill both patterns and verify the VRAM drops.
pkill -f openai_server 2>/dev/null || true
pkill -f atomesh 2>/dev/null || true
pkill -f '^ATOM::' 2>/dev/null || true
sleep 5
pkill -9 -f openai_server 2>/dev/null || true
pkill -9 -f '^ATOM::' 2>/dev/null || true
sleep 3

VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oP 'VRAM%\):\s*\K[0-9]+' | sort -rn | head -1)
if [ -n "$VRAM_MAX" ] && [ "$VRAM_MAX" -gt 10 ]; then
  echo "ERROR: ${VRAM_MAX}% VRAM still allocated after killing the servers."
  echo "       Orphaned workers are holding it; restart the container to clear:"
  echo "         docker restart atom_pp4pd_test"
  exit 1
fi

# Stale compile cache silently mismatches the new code after a source change.
rm -rf /root/.cache/atom/*

echo "================================================================"
echo "  GLM-5.2 PP4×PD + SparseKV GPU cold tier"
if [ "$GPU_COLD_PAGES" = "0" ]; then
  echo "    gpu_cold_tier   = OFF (kill-switch: host-only baseline)"
else
  echo "    gpu_cold_tier   = auto from --gpu-memory-utilization ${GPU_UTIL}"
fi
echo "    max_num_seqs    = ${MAX_NUM_SEQS}"
echo "    gpu_util        = ${GPU_UTIL}"
echo "    max_model_len   = ${MAX_MODEL_LEN}"
echo "    hot_buffer_size = ${HOT_BUFFER}"
echo "    host ratio      = ${RATIO}"
echo "    prefetch        = ${PREFETCH}"
echo "================================================================"

# ── prefill: PP4×TP1 on GPU 0-3 (no SparseKV) ───────────────────────
echo ">>> Starting prefill (PP4×TP1) on GPU 0-3 ..."
AITER_LOG_LEVEL=WARNING \
ATOM_LOG_PREFIX_CACHE_PER_REQ=1 \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
VLLM_PP_LAYER_PARTITION=18,20,20,20 \
HIP_VISIBLE_DEVICES=0,1,2,3 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$PREFILL_PORT" --trust-remote-code \
  -pp 4 -tp 1 --level 3 --enforce-eager \
  --max-num-batched-tokens $PREFILL_CHUNK \
  --kv_cache_dtype fp8 --gpu-memory-utilization $PREFILL_GPU_UTIL \
  --enable_prefix_caching \
  ${PREFILL_PROFILER_DIR:+--torch-profiler-dir $PREFILL_PROFILER_DIR} \
  --kv-transfer-config "{\"kv_role\":\"kv_producer\",\"kv_connector\":\"mooncake\",\"handshake_port\":$HANDSHAKE_PORT,\"proxy_ip\":\"127.0.0.1\"}" \
  > /tmp/prefill.log 2>&1 &

# ── decode: TP4 on GPU 4-7, SparseKV host + GPU cold tiers ──────────
echo ">>> Starting decode (TP4, SparseKV + GPU cold tier) on GPU 4-7 ..."
AITER_LOG_LEVEL=WARNING \
ATOM_LOG_PREFIX_CACHE_PER_REQ=1 \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
AITER_USE_FLYDSL_MOE_SORTING=1 \
ATOM_SPARSEKV_ENABLE=1 \
ATOM_SPARSEKV_HOT_BUFFER_SIZE=$HOT_BUFFER \
ATOM_SPARSEKV_PREFETCH=$PREFETCH \
ATOM_SPARSEKV_HOST_TO_DEVICE_RATIO=$RATIO \
ATOM_SPARSEKV_GPU_COLD_PAGES=$GPU_COLD_PAGES \
HIP_VISIBLE_DEVICES=4,5,6,7 \
nohup python -m atom.entrypoints.openai_server \
  --model "$MODEL" --server-port "$DECODE_PORT" --trust-remote-code \
  -tp 4 --level 3 \
  --max-num-seqs $MAX_NUM_SEQS --max-model-len $MAX_MODEL_LEN \
  --kv_cache_dtype fp8 --gpu-memory-utilization $GPU_UTIL \
  --enable_prefix_caching \
  ${TORCH_PROFILER_DIR:+--torch-profiler-dir $TORCH_PROFILER_DIR} \
  --kv-transfer-config "{\"kv_role\":\"kv_consumer\",\"kv_connector\":\"mooncake\",\"handshake_port\":$HANDSHAKE_PORT,\"proxy_ip\":\"127.0.0.1\"}" \
  > /tmp/decode.log 2>&1 &

# ── wait for both servers ────────────────────────────────────────────
wait_health() {  # name port max_tries
  local name=$1 port=$2 tries=$3 code
  echo ">>> Waiting for $name (port $port) ..."
  for _ in $(seq 1 "$tries"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && { echo "    $name READY"; return 0; }
    sleep 5
  done
  echo "ERROR: $name not ready; see /tmp/${name}.log"
  return 1
}
wait_health prefill "$PREFILL_PORT" 120
wait_health decode "$DECODE_PORT" 180

# /health can answer before the model is resident, so confirm on the GPUs.
echo ">>> Confirming VRAM is actually in use ..."
rocm-smi --showmemuse 2>/dev/null | grep -E "GPU\[|Memory" | head -20 || true

# ── confirm the tier actually came up ────────────────────────────────
echo ">>> SparseKV coordinator init:"
grep -E "SparseKV (GPU cold tier|coordinator)|cold_pool=" /tmp/decode.log | tail -5 || echo "    (no SparseKV lines — check /tmp/decode.log)"

# ── mesh proxy ───────────────────────────────────────────────────────
echo ">>> Starting mesh proxy on port $MESH_PORT ..."
nohup atomesh launch --pd-disaggregation \
  --prefill "http://127.0.0.1:$PREFILL_PORT" "$HANDSHAKE_PORT" \
  --decode "http://127.0.0.1:$DECODE_PORT" \
  --backend atom --port "$MESH_PORT" --log-level info \
  --request-timeout-secs "$MESH_REQUEST_TIMEOUT" \
  > /tmp/mesh_gpucold.log 2>&1 &

wait_health mesh "$MESH_PORT" 30

echo ""
echo "=== All services up ==="
echo "  prefill log: /tmp/prefill.log"
echo "  decode  log: /tmp/decode.log   (grep 'GPU cold tier' for the auto-sized GB)"
echo "  mesh    log: /tmp/mesh_gpucold.log"
echo "  mesh API:    http://127.0.0.1:$MESH_PORT/v1/chat/completions"
