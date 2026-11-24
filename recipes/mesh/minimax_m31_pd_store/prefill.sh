#!/usr/bin/env bash
# PP4 x TP1 prefill (Mooncake producer) with the Mooncake Store as its prefix tier,
# on GPUs 0-3 (NUMA0). Run inside the prefill container.
#   FORCE_AL=4.04  benchmark runs (InferenceX golden AL), empty for real acceptance
#   P_MEM          --gpu-memory-utilization (default 0.9)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OWNER_NICS=${OWNER_NICS:-rdma4,rdma5,rdma6,rdma7}
rm -rf /root/.cache/atom/*
export HIP_VISIBLE_DEVICES=0,1,2,3
# 60 target layers; the last stage also holds the DSpark draft's KV.
export VLLM_PP_LAYER_PARTITION=16,16,16,12
export ATOM_KV_OFFLOAD=mooncake_store
export ATOM_KV_OFFLOAD_EXTRA_CONFIG="{\"mooncake_store.master\":\"${NODE_IP}:${STORE_MASTER_PORT}\",\"mooncake_store.metadata\":\"http://${NODE_IP}:${STORE_METADATA_PORT}/metadata\",\"mooncake_store.owner_rdma_devices\":\"${OWNER_NICS}\",\"mooncake_store.local_hostname\":\"${NODE_IP}\",\"mooncake_store.protocol\":\"rdma\"}"
export OFFLOAD_LOAD_WORKERS=1 OFFLOAD_MIN_LOAD_TOKENS=0 OFFLOAD_PROFILE=1
export MC_MAX_MR_SIZE=1073741824 MC_TCP_BIND_ADDRESS=${NODE_IP}
# Both servers share the host network: give each its own distributed ports.
export ATOM_DP_MASTER_PORT=29500 ATOM_DP_BASE_PORT=29600
env | grep -E '^(ATOM_|AITER_|MC_|OFFLOAD_|VLLM_|HIP_|NCCL_|RCCL_|GLOO_|MORI_)' | sort > "${RUN}/prefill.env"
# shellcheck disable=SC2086
exec python3 -m atom.entrypoints.openai_server \
  --model "${MODEL}" --trust-remote-code --host 0.0.0.0 \
  --server-port "${P_PORT}" --port 18006 \
  -tp 1 --pipeline-parallel-size 4 --enforce-eager \
  --kv_cache_dtype "${KV_DTYPE}" --index-cache-dtype "${KV_DTYPE}" --block-size 128 \
  --max-model-len 1048576 --max-num-seqs 128 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization "${P_MEM:-0.9}" \
  ${SPEC_ARGS} ${AL_ARGS} \
  --kv-transfer-config '{"kv_connector":"mooncake","kv_role":"kv_producer","handshake_port":6301,"protocol":"rdma","enable_chunked_transfer":true,"chunked_admission_timeout_s":1500}' \
  --timeout-keep-alive 900 ${P_EXTRA:-} > "${RUN}/prefill.log" 2>&1
