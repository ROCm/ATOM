#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build AtomRL Docker images.
#
# Usage:
#   bash atomrl/docker/build.sh              # extend ATOM vLLM latest
#   bash atomrl/docker/build.sh --dev        # same image, tagged as atomrl:dev
#   bash atomrl/docker/build.sh --base IMG   # use a validated base image override
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOMRL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ATOMRL_ROOT}/.." && pwd)"

DEV_MODE=false
BASE_IMAGE="${ATOMRL_BASE_IMAGE:-rocm/atom-dev:vllm-latest}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dev) DEV_MODE=true; shift ;;
        --base) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

cd "${REPO_ROOT}"

if [ ! -d "atomrl/lumen" ]; then
    echo "ERROR: unified Lumen source not found at atomrl/lumen"
    exit 1
fi
if [ ! -d "atom" ]; then
    echo "ERROR: run this script from an ATOM checkout containing atom/"
    exit 1
fi

DOCKERFILE="atomrl/docker/Dockerfile.dev"
if [ "${DEV_MODE}" = true ]; then
    TAG="atomrl:dev"
else
    TAG="atomrl:latest"
fi

echo "Building ${TAG} from ${DOCKERFILE}..."
docker build \
    --pull \
    --no-cache \
    -f "${DOCKERFILE}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${TAG}" \
    .

echo "Done. Run with:"
echo "  docker run --rm -it --device /dev/kfd --device /dev/dri \\"
echo "    -v /dev/shm:/dev/shm --shm-size=256g \\"
echo "    --network=host ${TAG}"
