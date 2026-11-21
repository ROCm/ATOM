#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# rocprofv3 per-rank wrapper for torchrun launched with `--no-python`.
#
# Profiles GLOBAL rank 0 only (keeps trace size manageable); every other rank
# runs unprofiled. torchrun invokes this as:
#
#     rocprof_wrap.sh python <script.py> <args...>
#
# so "$@" is the full command to execute. On rank 0 we prepend rocprofv3.
#
# Env knobs (forwarded from the launcher):
#   LUMEN_ROCPROF_DIR   output directory          (default /results/rocprof)
#   LUMEN_ROCPROF_NAME  output basename           (default trace_rank0)
#   LUMEN_ROCPROF_OPTS  rocprofv3 trace options   (default -r = runtime-trace:
#                       HIP runtime + RCCL + memory copies + kernel dispatch)
set -euo pipefail

RANK="${RANK:-${LOCAL_RANK:-0}}"

# Non-zero ranks: run the command unprofiled.
if [ "${RANK}" != "0" ]; then
    exec "$@"
fi

OUT_DIR="${LUMEN_ROCPROF_DIR:-/results/rocprof}"
OUT_NAME="${LUMEN_ROCPROF_NAME:-trace_rank0}"
OPTS="${LUMEN_ROCPROF_OPTS:--r}"
mkdir -p "${OUT_DIR}"

echo "[rocprof_wrap] rank0 -> rocprofv3 ${OPTS} -f pftrace -d ${OUT_DIR} -o ${OUT_NAME} -- $*"
# shellcheck disable=SC2086
exec rocprofv3 ${OPTS} -f pftrace -d "${OUT_DIR}" -o "${OUT_NAME}" -- "$@"
