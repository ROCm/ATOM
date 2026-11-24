#!/usr/bin/env bash
# Mooncake Store for the prefill's mooncake_store offload tier: one master and one
# owner per NIC, owners bound to a NUMA node's DRAM. Run inside the store container
# (no GPU). Usage: store.sh start|stop|status
#   OWNERS   "<numa>:<GiB>:<rdma>;..." (default 4 x 128 GiB on rdma4-7, both NUMA nodes).
#            Owners use NICs no prefill stage uses (stages take their GPU's NIC, rdma0-3).
#   DROP_DIRS page cache evicted before the owners pin huge pages (default: the model only)
#   COMPACT_TIMEOUT seconds each NUMA node's compaction may take (default 1800)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
OWNERS=${OWNERS:-0:128:rdma4;0:128:rdma5;1:128:rdma6;1:128:rdma7}
DROP_DIRS=${DROP_DIRS:-${MODEL}}
COMPACT_TIMEOUT=${COMPACT_TIMEOUT:-1800}
S=/workspace/ATOM/.github/scripts/atomesh
LOG=${RUN}/store
mkdir -p "${LOG}"

stop() {
  pkill -f '[m]ooncake_client' || true
  pkill -f '[m]ooncake_master' || true
}

status() {
  curl -s "http://${NODE_IP}:${STORE_METRICS_PORT}/metrics" \
    | grep -E '^master_(total_capacity_bytes|allocated_bytes|key_count|put_start_requests|get_replica_list_requests|batch_put_start_requests|batch_get_replica_list_requests|eviction)' || true
}

start() {
  stop
  local -a pins=() numa=() gib=() dev=()
  local entry n g d
  IFS=';' read -r -a entries <<< "${OWNERS}"
  for entry in "${entries[@]}"; do
    IFS=':' read -r n g d <<< "${entry}"
    numa+=("${n}"); gib+=("${g}"); dev+=("${d}"); pins+=("${n}:${g}")
  done
  echo "[store] dropping the page cache of ${DROP_DIRS}"
  python3 "${S}/drop_page_cache.py" ${DROP_DIRS//:/ }
  # Refuses owners that do not fit their NUMA node (128 GiB to spare), then faults
  # each node's share as transparent huge pages once, so the owners find them free.
  python3 "${S}/numa_memory_budget.py" --reserve-gib 128 --compact \
    --compact-timeout "${COMPACT_TIMEOUT}" "${pins[@]}" 2>&1 | tee "${LOG}/numa-budget.log"
  HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 setsid python3 "${S}/numa_exec.py" 1 \
    mooncake_master --rpc_port="${STORE_MASTER_PORT}" --rpc_thread_num=32 \
    --enable_http_metadata_server=true --http_metadata_server_host="${NODE_IP}" \
    --http_metadata_server_port="${STORE_METADATA_PORT}" --metrics_port="${STORE_METRICS_PORT}" \
    --default_kv_lease_ttl=10000 --eviction_high_watermark_ratio=0.88 --eviction_ratio=0.02 \
    --allocation_strategy=random --memory_allocator=offset --client_ttl=10 \
    > "${LOG}/master.log" 2>&1 < /dev/null &
  until curl -s -o /dev/null --max-time 5 "http://${NODE_IP}:${STORE_METADATA_PORT}/metadata?key=probe"; do sleep 1; done
  local i want=0
  for i in "${!numa[@]}"; do
    # THP-backed segment (glibc's hugetlb=1 malloc): 4 KiB pages exhaust the NIC's
    # registration budget. 64 GiB memory regions register in about a second each.
    HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 GLIBC_TUNABLES=glibc.malloc.hugetlb=1 \
    MC_MAX_MR_SIZE=68719476736 MC_TCP_BIND_ADDRESS="${NODE_IP}" \
      setsid python3 "${S}/numa_exec.py" "${numa[i]}" mooncake_client \
      --host="${NODE_IP}" --port=$((STORE_MASTER_PORT + 1 + i)) \
      --master_server_address="${NODE_IP}:${STORE_MASTER_PORT}" \
      --metadata_server="http://${NODE_IP}:${STORE_METADATA_PORT}/metadata" \
      --protocol=rdma --device_names="${dev[i]}" \
      --global_segment_size=$((gib[i] * 1024 * 1024 * 1024)) \
      --local_buffer_size=0 --threads=4 \
      > "${LOG}/owner${i}-numa${numa[i]}-${dev[i]}.log" 2>&1 < /dev/null &
    want=$((want + gib[i] * 1024 * 1024 * 1024))
  done
  local deadline=$((SECONDS + COMPACT_TIMEOUT + 600)) have=0
  while :; do
    have=$(curl -s --max-time 5 "http://${NODE_IP}:${STORE_METRICS_PORT}/metrics" \
      | awk '$1 == "master_total_capacity_bytes" {print int($2)}')
    [[ -n "${have}" && "${have}" -ge "${want}" ]] && break
    if grep -l -E 'Failed to setup client|Failed to mount|Check failed|FATAL' "${LOG}"/owner*.log 2>/dev/null; then
      echo "[store][FAIL] an owner failed, see ${LOG}" >&2; exit 1
    fi
    (( SECONDS < deadline )) || { echo "[store][FAIL] capacity ${have:-0} < ${want} bytes" >&2; exit 1; }
    sleep 2
  done
  echo "[store] ready: capacity ${have} bytes, master ${NODE_IP}:${STORE_MASTER_PORT}"
  for pid in $(pgrep -f '[m]ooncake_client'); do
    echo "[store] owner pid ${pid} $(grep -h AnonHugePages "/proc/${pid}/smaps_rollup" 2>/dev/null)"
  done
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: store.sh start|stop|status" >&2; exit 2 ;;
esac
