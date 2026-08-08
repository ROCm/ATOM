#!/bin/bash
# Run the SparseKV host-mapping probe on all four decode GPUs at once.
#
# One rank alone pins ~247 GB; the decode server pins that four times over, so a
# limit that only bites in aggregate is invisible to a single-rank probe. Run
# with the servers stopped.
#
#   docker exec -w /it-share/yajizhan/code/ATOM atom_pp4pd_test \
#     bash scripts/probe_sparsekv_host_mapping.sh

set -uo pipefail

ATOM_SRC=/it-share/yajizhan/code/ATOM
export PYTHONPATH="${ATOM_SRC}${PYTHONPATH:+:$PYTHONPATH}"

HOST_PAGES="${HOST_PAGES:-344106}"
OUT_DIR="${OUT_DIR:-${ATOM_SRC}/results/host_mapping_probe}"
mkdir -p "$OUT_DIR"

pids=()
for gpu in 4 5 6 7; do
  HIP_VISIBLE_DEVICES=$gpu HOST_PAGES=$HOST_PAGES TAG="gpu$gpu" \
    python3 "${ATOM_SRC}/scripts/probe_sparsekv_host_mapping.py" \
    > "${OUT_DIR}/gpu${gpu}.log" 2>&1 &
  pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done

echo "=== probe summary (rc=$rc) ==="
tail -3 "${OUT_DIR}"/gpu*.log
exit $rc
