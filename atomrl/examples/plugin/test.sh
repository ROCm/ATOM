#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ZhangDanyang-AMD
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Build isolated, pinned plugin validation layers on top of the public base.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DOCKERFILE="${ROOT}/atomrl/examples/plugin/Dockerfile.plugin"

for framework in verl vime miles slime; do
    image="atomrl-plugin-test-${framework}:local"
    docker build \
        --network=host \
        --file "${DOCKERFILE}" \
        --target "${framework}-test" \
        --tag "${image}" \
        "${ROOT}"
    docker rmi -f "${image}" >/dev/null 2>&1 || true
done
