#!/bin/bash
# Profile one run of either model with VeOmni's built-in torch profiler.
#
#   profile.sh <qwen|wan> [VeOmni CLI overrides...]      # LUMEN_PATCH as exported
#
# A thin wrapper over run_10steps.sh / run_wan_10steps.sh in custom mode, so a
# profiled run is environmentally identical to the timed runs: --train.profile.*
# is ordinary VeOmni configuration. WAN_TASK passes through.
#
# PROF_START/PROF_END pick the window; the default 4..6 skips step 1, which
# carries kernel autotune and, on a first FlyDSL run, JIT compilation.
# WITH_STACK=true adds Python frames -- use it for attribution only, never for
# absolute milliseconds (it inflates the step and manufactures idle time).
# Traces land in $OUT_DIR/trace/<RUN_NAME>; read them with kernel_diff.py,
# trace_report.py, kernel_blame.py and opt_phase.py.
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/env.sh"

MODEL="${1:?usage: profile.sh <qwen|wan> [VeOmni CLI overrides...]}"
shift
case "$MODEL" in
    qwen) RUNNER="$HERE/run_10steps.sh" ;;
    wan)  RUNNER="$HERE/run_wan_10steps.sh" ;;
    *) echo "unknown model: $MODEL (qwen|wan)" >&2; exit 2 ;;
esac

PROF_START="${PROF_START:-4}"
PROF_END="${PROF_END:-6}"
TAG="${RUN_NAME:-prof-$MODEL}${WITH_STACK:+-stack}"
TRACE_DIR="$OUT_DIR/trace/$TAG"
mkdir -p "$TRACE_DIR"

echo "=== profiling $MODEL steps $PROF_START..$PROF_END, LUMEN_PATCH=${LUMEN_PATCH:-<none>} -> $TRACE_DIR"
RUN_NAME="$TAG" bash "$RUNNER" custom \
    --train.profile.enable true \
    --train.profile.start_step "$PROF_START" \
    --train.profile.end_step "$PROF_END" \
    --train.profile.trace_dir "$TRACE_DIR" \
    --train.profile.record_shapes true \
    --train.profile.with_modules true \
    --train.profile.with_stack "${WITH_STACK:-false}" \
    --train.profile.profile_memory false \
    --train.profile.rank0_only true \
    "$@"
rc=$?
echo "EXIT_CODE=$rc  trace -> $TRACE_DIR"
ls -la "$TRACE_DIR"
exit $rc
