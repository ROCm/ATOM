#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Multi-node AtomRL launch via SLURM.
#
# Usage:
#   bash scripts/launch_slurm.sh <num_nodes> <config.yaml> [overrides...]
#
# Example:
#   bash scripts/launch_slurm.sh 2 configs/grpo_dense_fp8.yaml \
#       policy.model_name=Qwen/Qwen3-8B

set -euo pipefail

NUM_NODES="${1:?Usage: launch_slurm.sh <num_nodes> <config.yaml> [overrides...]}"
CONFIG="${2:?Usage: launch_slurm.sh <num_nodes> <config.yaml> [overrides...]}"
shift 2

GPUS_PER_NODE=8

echo "=== AtomRL SLURM Launch ==="
echo "Nodes:        ${NUM_NODES}"
echo "GPUs/node:    ${GPUS_PER_NODE}"
echo "Config:       ${CONFIG}"
echo "Overrides:    $*"
echo "============================"

COMMAND="python examples/run_grpo.py --config ${CONFIG} \
    cluster.num_nodes=${NUM_NODES} \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    $*"

sbatch \
    --nodes="${NUM_NODES}" \
    --gres=gpu:"${GPUS_PER_NODE}" \
    --ntasks-per-node=1 \
    --job-name=atomrl \
    --export=ALL,COMMAND="${COMMAND}" \
    scripts/ray.sub
