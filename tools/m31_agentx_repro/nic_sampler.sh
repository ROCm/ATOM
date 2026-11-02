#!/usr/bin/env bash
# Sample every ionic RDMA port's byte/retransmit/congestion counters every 10 s.
# Output: one CSV line per sample: epoch,nic,tx_bytes,rx_bytes,tx_retx_bytes,rx_cnp_pkts,rx_ecn_pkts
BASE="${BASE:?set BASE}"; mkdir -p "${BASE}/nic"
out="${1:-${BASE}/nic/nic-$(date -u +%Y%m%d%H%M%S).csv}"
echo "epoch,nic,tx_bytes,rx_bytes,tx_retx_bytes,rx_cnp_pkts,rx_ecn_pkts" > "${out}"
while [[ ! -f ${BASE}/STOP_NIC ]]; do
  now=$(date +%s)
  for d in /sys/class/infiniband/rdma*; do
    c="${d}/ports/1/hw_counters"
    echo "${now},$(basename "$d"),$(cat "$c/tx_rdma_ucast_bytes"),$(cat "$c/rx_rdma_ucast_bytes"),$(cat "$c/tx_rdma_retx_bytes"),$(cat "$c/rx_rdma_cnp_pkts"),$(cat "$c/rx_rdma_ecn_pkts")"
  done >> "${out}"
  sleep 10
done
