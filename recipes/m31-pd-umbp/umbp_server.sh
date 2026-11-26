#!/bin/bash
# umbp_server.sh <log> (in the prefill container): start this tray's MORI-UMBP standalone server with a DRAM tier for
# umbp_store, and return once it serves. ATOM's prefill (all PP stages) connects to it at unix://$UMBP_SOCK.
# The server and the ATOM workers must share this container (one PID namespace) and its GPU list.
# Env: UMBP_GB (DRAM tier in GiB, required), UMBP_SOCK (/tmp/umbp/server.sock), GPUS (ROCR_VISIBLE_DEVICES, as the
#   server's; empty = all), UMBP_NUMA (0,1: NUMA nodes the DRAM tier spreads over), UMBP_HUGEPAGES (0; 1 = 2 MiB
#   pages, reserved beforehand on the host), UMBP_METRICS_PORT (9092; Prometheus), UMBP_READY_S (900).
set -uo pipefail
LOG=${1:?usage: umbp_server.sh <log>}
: "${UMBP_GB:?set UMBP_GB}"
SOCK=${UMBP_SOCK:-/tmp/umbp/server.sock}
BIN=$(python3 -c 'import pathlib, mori.umbp; print(pathlib.Path(mori.umbp.__file__).resolve().parents[1] / "umbp_standalone_server")') \
  || { echo "this image's mori has no UMBP (build it with BUILD_UMBP=ON)" | tee "$LOG"; exit 1; }
[ -x "$BIN" ] || { echo "no $BIN" | tee "$LOG"; exit 1; }
pkill -INT -f '[u]mbp_standalone_server' 2> /dev/null && sleep 5
mkdir -p "$(dirname "$SOCK")"
rm -f "$SOCK"*
env_args=(
  UMBP_DRAM_CAPACITY=$((UMBP_GB * (1 << 30)))
  UMBP_DRAM_NUMA_NODE="${UMBP_NUMA:-0,1}"
  UMBP_DRAM_USE_HUGEPAGES="${UMBP_HUGEPAGES:-0}"
  UMBP_DRAM_PREFAULT_THREADS=16
  UMBP_SSD_ENABLED=0
  UMBP_STANDALONE_METRICS_PORT="${UMBP_METRICS_PORT:-9092}"
  MORI_UMBP_LOG_LEVEL=INFO
  # Transfer buffers off the mmap path (what ROCm/mori#1048 does in code), for a mori without it.
  MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=1073741824 MALLOC_TOP_PAD_=268435456
)
[ -n "${GPUS:-}" ] && env_args+=(ROCR_VISIBLE_DEVICES="$GPUS")
echo "$(date '+%F %T') umbp_standalone_server unix://$SOCK ${env_args[*]}" > "$LOG"
setsid nohup env "${env_args[@]}" "$BIN" "unix://$SOCK" >> "$LOG" 2>&1 < /dev/null &
pid=$!
# Ready only once a real client connects: the socket comes up before the DRAM tier is prefaulted and registered.
for _ in $(seq 1 $(( ${UMBP_READY_S:-900} / 5 ))); do
  kill -0 "$pid" 2> /dev/null || { echo "umbp_standalone_server exited" | tee -a "$LOG"; exit 1; }
  if [ -S "$SOCK" ] && python3 - "$SOCK" > /dev/null 2>&1 <<'PY'
import sys
import mori.umbp as u
c = u.UMBPConfig(); p = u.UMBPStandaloneProcessConfig()
p.address = "unix://" + sys.argv[1]; p.auto_start = False; p.startup_timeout_ms = 10000
c.standalone_process = p
u.UMBPClient(c)
PY
  then
    echo "$(date '+%F %T') UMBP server ready (pid $pid, ${UMBP_GB} GiB DRAM)" | tee -a "$LOG"
    exit 0
  fi
  sleep 5
done
echo "UMBP server not ready in ${UMBP_READY_S:-900} s" | tee -a "$LOG"
exit 1
