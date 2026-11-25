#!/usr/bin/env bash
# nic_sampler.sh <out.csv> <stop-file> [interval_s]: every RDMA port's byte, retransmit and
# congestion counters every interval (default 10 s) until <stop-file> exists. Runs ON a node.
# CSV: epoch,nic,tx_bytes,rx_bytes,tx_retx_bytes,rx_cnp_pkts,rx_ecn_pkts
out="${1:?out csv}"; stop="${2:?stop file}"; every="${3:-10}"
echo "epoch,nic,tx_bytes,rx_bytes,tx_retx_bytes,rx_cnp_pkts,rx_ecn_pkts" > "${out}"
rd() { cat "$1" 2>/dev/null || echo 0; }
while [[ ! -e "${stop}" ]]; do
  now=$(date +%s)
  for d in /sys/class/infiniband/*; do
    c="${d}/ports/1/hw_counters"
    echo "${now},$(basename "$d"),$(rd "$c/tx_rdma_ucast_bytes"),$(rd "$c/rx_rdma_ucast_bytes"),$(rd "$c/tx_rdma_retx_bytes"),$(rd "$c/rx_rdma_cnp_pkts"),$(rd "$c/rx_rdma_ecn_pkts")"
  done >> "${out}"
  sleep "${every}"
done
