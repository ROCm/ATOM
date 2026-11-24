#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Launch AtomRL on a single node with Ray.
#
# Usage:
#   bash scripts/launch_single_node.sh configs/grpo_dense_fp8.yaml [overrides...]
#
# Example:
#   bash scripts/launch_single_node.sh configs/grpo_dense_fp8.yaml \
#       policy.model_name=Qwen/Qwen3-8B logger.wandb_enabled=true

set -euo pipefail

CONFIG="${1:?Usage: launch_single_node.sh <config.yaml> [overrides...]}"
shift

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)

echo "=== AtomRL Single-Node Launch ==="
echo "Config:   ${CONFIG}"
echo "GPUs:     ${NUM_GPUS}"
echo "Overrides: $*"
echo "=================================="

ray stop --force 2>/dev/null || true
ray start --head --num-gpus="${NUM_GPUS}"

python examples/run_grpo.py --config "${CONFIG}" \
    cluster.gpus_per_node="${NUM_GPUS}" \
    cluster.num_nodes=1 \
    "$@"

ray stop --force
