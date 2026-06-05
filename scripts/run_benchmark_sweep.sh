#!/bin/bash
# Run benchmark across multiple concurrency levels.
#
# Per-step output: each concurrency saves a copy of `/app/logs_claude/benchmark.log`
# (which `run_benchmark.sh` always overwrites) under
# `$SWEEP_DIR/c<CONC>.log`. SWEEP_DIR defaults to a timestamped subdir of
# `/app/logs_claude/sweeps/`; override via the SWEEP_DIR env var to bucket
# multiple sweeps under a stable name.
#
# Final summary table is printed to stdout AND saved to `$SWEEP_DIR/summary.txt`.
#
# Usage: bash run_benchmark_sweep.sh MODEL PORT ISL OSL [CONC_LIST]
#
# Examples:
#   bash run_benchmark_sweep.sh /data/Kimi-K2.5-MXFP4 8000 1024 1024
#   bash run_benchmark_sweep.sh /data/Kimi-K2.5-MXFP4 8000 1024 1024 "4 8 16 32 64 128"
#   bash run_benchmark_sweep.sh /data/DeepSeek-R1-0528 8000 1024 1024 "1 2 4 8 16 32 64 128 256"
#   SWEEP_DIR=/app/logs_claude/sweeps/v4-pro-noMTP \
#     bash run_benchmark_sweep.sh /data/DeepSeek-V4-Pro 30000 1024 1024 "4 8 16 32 64 128 256"

set -uo pipefail

MODEL="${1:?Usage: $0 MODEL PORT ISL OSL [CONC_LIST]}"
PORT="${2:-8000}"
ISL="${3:-1024}"
OSL="${4:-1024}"
CONC_LIST="${5:-4 8 16 32 64 128}"

SWEEP_DIR="${SWEEP_DIR:-/app/logs_claude/sweeps/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$SWEEP_DIR"
SUMMARY="$SWEEP_DIR/summary.txt"

echo "========================================"
echo " Benchmark Sweep"
echo "========================================"
echo " Model:     $MODEL"
echo " ISL/OSL:   ${ISL}/${OSL}"
echo " Conc:      $CONC_LIST"
echo " Output:    $SWEEP_DIR"
echo "========================================"

for CONC in $CONC_LIST; do
  echo ""
  echo "================================================================"
  echo "=== CONC=$CONC ==="
  echo "================================================================"
  bash /app/ATOM/scripts/run_benchmark.sh "$MODEL" "$PORT" "$ISL" "$OSL" "$CONC"
  # Preserve per-step result before the next iteration overwrites it.
  cp -f /app/logs_claude/benchmark.log "$SWEEP_DIR/c${CONC}.log"
done

# Compose summary table: one row per concurrency.
{
  printf "%-12s %-25s %-25s %-15s\n" \
    "Concurrency" "Output Throughput(tok/s)" "Total Throughput(tok/s)" "Mean TPOT(ms)"
  for CONC in $CONC_LIST; do
    LOG="$SWEEP_DIR/c${CONC}.log"
    OUT_TP=$(grep "Output token throughput" "$LOG" 2>/dev/null | awk '{print $NF}')
    TOT_TP=$(grep "Total Token throughput" "$LOG" 2>/dev/null | awk '{print $NF}')
    TPOT=$(grep "Mean TPOT" "$LOG" 2>/dev/null | awk '{print $NF}')
    printf "%-12s %-25s %-25s %-15s\n" \
      "$CONC" "${OUT_TP:--}" "${TOT_TP:--}" "${TPOT:--}"
  done
} | tee "$SUMMARY"

echo ""
echo "=== Sweep complete ==="
echo "Per-step logs + summary: $SWEEP_DIR"
