#!/bin/bash
# Launch ATOM under rocm-debug-agent so a GPU memory fault dumps wave state +
# code-object disassembly. Intended for debugging async kernel races (Memory
# access fault, MEMORY_VIOLATION, ASSERT_TRAP, silent GPU livelocks).
#
# Two modes:
#
#   Server mode (default):
#     bash scripts/run_debug_agent.sh [MODEL] [TP] [PORT] [EXTRA_ARGS...]
#       MODEL       default /data/DeepSeek-V4-Pro
#       TP          default 8
#       PORT        default 8000
#       EXTRA_ARGS  forwarded to start_atom_server.sh
#
#   Offline simple_inference mode:
#     bash scripts/run_debug_agent.sh --simple [MODEL] [TP] [EXTRA_ARGS...]
#       MODEL       default /data/DeepSeek-V4-Pro
#       TP          default 8
#       (no PORT — simple_inference is offline)
#       EXTRA_ARGS  forwarded to start_simple_inference.sh
#       Default log path: /app/logs_claude/simple_inference_debug_agent.log
#       (override with LOG_FILE=...).
#
# REQUIRED in EXTRA_ARGS for either mode: --enforce-eager --level 0
# (graph mode + Inductor are incompatible with the agent's no-caching
# allocator). The script does NOT inject these — pass them via EXTRA_ARGS so
# the model-specific launch matches your repro.
#
# Output: the agent prints wave dumps to stderr; both launchers redirect into
# /app/logs_claude/<their log file>. Code objects (one per faulting kernel,
# ~4 MB each) land in /app/logs_claude/debug_run/.
#
# After fault triggers in server mode, the agent keeps running. Stop with:
#   bash scripts/stop_atom_server.sh
# In simple mode the launcher exits when generation finishes (or faults).

set -uo pipefail

# Resolve script dir to an ABSOLUTE path before any `cd` below. The launch step
# cd's into $DEBUG_DIR for code-object dumps, so a relative dirname would break
# the path to the sibling launcher scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="server"
if [ "${1:-}" = "--simple" ]; then
  MODE="simple"
  shift
fi

MODEL="${1:-/data/DeepSeek-V4-Pro}"
TP="${2:-8}"

if [ "$MODE" = "server" ]; then
  PORT="${3:-8000}"
  shift 3 2>/dev/null || true
else
  shift 2 2>/dev/null || true
fi

# 1) no gpucore files (a single ROCm fault dumps 30-50 GB per rank)
ulimit -c 0

# 2) clean cwd for code-object dumps; --save-code-objects writes
#    `memory___<pid>_offset_<hex>_size_<n>` files into the process cwd.
DEBUG_DIR="/app/logs_claude/debug_run"
mkdir -p "$DEBUG_DIR"
cd "$DEBUG_DIR"
rm -f memory_* *.s

# 3) debug agent env (must be exported so spawn workers inherit)
export HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2
export HSA_ENABLE_DEBUG=1
export ROCM_DEBUG_AGENT_OPTIONS="--save-code-objects"

# 4) ATOM-side default; caller may override via env before invocation
export AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}"

# 5) launch (model-specific env like ATOM_USE_TRITON_MOE must come from caller)
if [ "$MODE" = "server" ]; then
  exec bash "$SCRIPT_DIR/start_atom_server.sh" \
    "$MODEL" "$TP" "$PORT" "$@"
else
  export LOG_FILE="${LOG_FILE:-/app/logs_claude/simple_inference_debug_agent.log}"
  exec bash "$SCRIPT_DIR/start_simple_inference.sh" \
    "$MODEL" "$TP" "$@"
fi
