#!/bin/bash
# Pull every log a finished aiperf round produced out of the container into its
# OUTPUT_DIR, then print the summary row.
#
# The server logs live on the container's /tmp and are overwritten by the next
# start_glm52_pp4pd_gpucold.sh, so they have to come out before the next round.
# Copies are made by streaming through `docker exec cat` rather than `cp` inside
# the container, so the files land owned by the host user on the shared mount.
#
#   bash scripts/collect_run_logs.sh results/js_c32_m48_r14

set -uo pipefail

CONTAINER="${CONTAINER:-atom_pp4pd_test}"
OUT_DIR="${1:?usage: collect_run_logs.sh <output_dir>}"
mkdir -p "$OUT_DIR"

for f in decode prefill mesh_gpucold; do
  if docker exec "$CONTAINER" test -f "/tmp/${f}.log"; then
    docker exec "$CONTAINER" cat "/tmp/${f}.log" > "${OUT_DIR}/${f}.log"
    echo "  ${f}.log       $(wc -l < "${OUT_DIR}/${f}.log") lines"
  else
    echo "  ${f}.log       MISSING in container"
  fi
done

# Machine state at collection time: VRAM still held, workers still alive, and
# whether the fault handler managed to write a GPU core dump this round.
docker exec "$CONTAINER" rocm-smi --showmemuse > "${OUT_DIR}/rocm_smi.txt" 2>&1
docker exec "$CONTAINER" bash -c 'ps -eo pid,etime,rss,comm | grep -E "ATOM::|openai_server" | grep -v grep' \
  > "${OUT_DIR}/procs.txt" 2>&1
docker exec "$CONTAINER" bash -c 'ls -la /coredumps 2>&1' > "${OUT_DIR}/coredumps.txt" 2>&1

echo ""
python3 "$(dirname "$0")/summarize_joint_sizing_run.py" "$OUT_DIR" \
  | tee "${OUT_DIR}/summary.txt"
