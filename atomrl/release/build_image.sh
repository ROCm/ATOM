#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build the AtomRL release image from release/versions.env.
#
#   bash atomrl/release/build_image.sh                    # full image, incl. example 7
#   WITH_MEGATRON=0 bash atomrl/release/build_image.sh    # skip Megatron/Apex/TE
#
# Run from the ATOM repository root. Expect 60-90 min for the full build; the long
# pole is TransformerEngine's AOTriton step.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
# shellcheck disable=SC1091
source "$HERE/versions.env"

TAG=${TAG:-atomrl:release-$(date +%Y%m%d)}
ARCH=${PYTORCH_ROCM_ARCH:-gfx950}
WITH_MEGATRON=${WITH_MEGATRON:-1}

echo "==> building $TAG"
echo "    base        $BASE_IMAGE"
echo "    arch        $ARCH"
echo "    aiter       $AITER_SHA (official ROCm)"
echo "    mori        $MORI_SHA (official ROCm)"
echo "    ATOM/AtomRL current checkout"
echo "    Lumen       in-tree atomrl/lumen"
echo "    megatron    $WITH_MEGATRON"

cd "$REPO_ROOT"
ATOM_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
git diff --quiet HEAD 2>/dev/null || ATOM_COMMIT="$ATOM_COMMIT-dirty"
echo "    commit      $ATOM_COMMIT"
docker build -f atomrl/docker/Dockerfile.release -t "$TAG" \
  --build-arg ATOM_COMMIT="$ATOM_COMMIT" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg PYTORCH_ROCM_ARCH="$ARCH" \
  --build-arg AITER_REPO="$AITER_REPO" \
  --build-arg AITER_SHA="$AITER_SHA" \
  --build-arg APEX_SHA="$APEX_SHA" \
  --build-arg TE_SHA="$TE_SHA" \
  --build-arg MEGATRON_COMMIT="$MEGATRON_COMMIT" \
  --build-arg WITH_MEGATRON="$WITH_MEGATRON" \
  .

echo
echo "==> built $TAG"
echo "    Next, bake the aiter JIT kernels (needs GPUs, ~25 min once):"
echo "      TAG=$TAG bash atomrl/release/precompile_kernels.sh"
