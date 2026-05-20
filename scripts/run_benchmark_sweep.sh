#!/bin/bash
# Run benchmark across multiple concurrency levels
# Usage: bash run_benchmark_sweep.sh MODEL PORT ISL OSL [CONC_LIST]
#
# Examples:
#   bash run_benchmark_sweep.sh /data/Kimi-K2.5-MXFP4 8000 1024 1024
#   bash run_benchmark_sweep.sh /data/Kimi-K2.5-MXFP4 8000 1024 1024 "4 8 16 32 64 128"
#   bash run_benchmark_sweep.sh /data/DeepSeek-R1-0528 8000 1024 1024 "1 2 4 8 16 32 64 128 256"

set -uo pipefail

MODEL="${1:?Usage: $0 MODEL PORT ISL OSL [CONC_LIST]}"
PORT="${2:-8000}"
ISL="${3:-1024}"
OSL="${4:-1024}"
CONC_LIST="${5:-4 8 16 32 64 128}"

echo "========================================"
echo " Benchmark Sweep"
echo "========================================"
echo " Model:     $MODEL"
echo " ISL/OSL:   ${ISL}/${OSL}"
echo " Conc:      $CONC_LIST"
echo "========================================"

for CONC in $CONC_LIST; do
  echo ""
  echo "================================================================"
  echo "=== CONC=$CONC ==="
  echo "================================================================"
  bash /app/ATOM/scripts/run_benchmark.sh "$MODEL" "$PORT" "$ISL" "$OSL" "$CONC"
done

echo ""
echo "=== Sweep complete ==="
