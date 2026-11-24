#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Container entrypoint. Puts the source trees on PYTHONPATH in the order the
# examples expect, then execs whatever was asked for.
#
# Order matters: $AITER_DIR must precede site-packages so `import aiter` resolves
# to the pinned source tree and not the base image's amd-aiter wheel. The wheel
# pins flydsl<0.1.5 while the source needs >=0.2.4, so getting this backwards
# surfaces as "cannot import name 'fly_values' from flydsl.compiler.protocol".
set -uo pipefail

RL_ROOT=${RL_ROOT:-/opt/atomrl}
export LUMEN_DIR=${LUMEN_DIR:-$RL_ROOT/ATOM/atomrl/lumen}
export AITER_DIR=${AITER_DIR:-$RL_ROOT/aiter}
export MORI_DIR=${MORI_DIR:-$RL_ROOT/mori}
export ATOM_DIR=${ATOM_DIR:-$RL_ROOT/ATOM}
export PYTHONPATH="$RL_ROOT/ATOM/atomrl:$AITER_DIR:$MORI_DIR/python:${PYTHONPATH:-}"

# HIP's allocator has no expandable_segments; leaving it set makes the trainer
# fail in ways that point at the model rather than at the allocator.
unset PYTORCH_CUDA_ALLOC_CONF

# Compiled kernels live outside the source tree so that bind-mounting a
# different aiter checkout does not silently reuse kernels built from another
# revision. That mismatch shows up as
# "module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'".
export AITER_JIT_DIR=${AITER_JIT_DIR:-/opt/atomrl/aiter-jit}
mkdir -p "$AITER_JIT_DIR"

if [ "${ATOMRL_ENTRYPOINT_QUIET:-0}" != "1" ]; then
  echo "AtomRL release image"
  echo "  RL_ROOT      $RL_ROOT"
  echo "  AITER_JIT_DIR $AITER_JIT_DIR"
  for d in ATOM aiter mori; do
    if [ -d "$RL_ROOT/$d/.git" ]; then
      printf "  %-11s %s\n" "$d" "$(git -C "$RL_ROOT/$d" rev-parse --short=12 HEAD 2>/dev/null)"
    elif [ "$d" = ATOM ] && [ -n "${ATOM_COMMIT:-}" ]; then
      printf "  %-11s %s (baked)\n" "$d" "${ATOM_COMMIT:0:12}"
    fi
  done
fi

exec "$@"
