#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test: 3-step run to verify the Qwen3-30B-A3B GRPO/DAPO pipeline works.
# Run this BEFORE launching full experiments.
#
# Usage:
#   bash examples/GRPO/smoke_test.sh
#   MODEL_PATH=/path/to/Qwen3-30B-A3B bash examples/GRPO/smoke_test.sh
# ═══════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODE=smoke
export STEPS="${STEPS:-3}"

exec bash "$SCRIPT_DIR/run_grpo.sh"
