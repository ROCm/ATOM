#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Benchmark Eagle3 Phase 1 model using vLLM with speculative decoding on MI350

set -uo pipefail

DOCKER_IMAGE="atomrl-vllm-mi350:latest"
CONTAINER_NAME="kimi_k25_eagle3_v2_benchmark"
DRAFT_MODEL="/dev/shm/Kimi_K25_eagle3_v2_phase1_HF"
BASE_MODEL="/dev/shm/Kimi-K2.5-MXFP4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOMRL_DIR="${ATOMRL_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
BENCH_SCRIPT="/workspace/ATOM/atomrl/examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/tools/bench_eagle3_vllm.py"
OUTPUT_DIR="/workspace/ATOM/atomrl/examples/SDDD/Kimi_K25_SDDD_MI350_ATOM/benchmark_results"

# Kill any existing benchmark container
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

echo "Starting benchmark container: ${CONTAINER_NAME}"
echo "Docker image: ${DOCKER_IMAGE}"
echo "Draft model: ${DRAFT_MODEL}"
echo "Base model: ${BASE_MODEL}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --network host \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 64G \
    -v /dev/shm:/dev/shm \
    -v "${ATOMRL_DIR}:/workspace/ATOM/atomrl" \
    -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    "${DOCKER_IMAGE}" \
    bash -c "
set -e

echo '=== Starting vLLM server with Eagle3 speculative decoding ==='
vllm serve ${BASE_MODEL} \
    --speculative-config '{\"model\": \"${DRAFT_MODEL}\", \"method\": \"eagle3\", \"num_speculative_tokens\": 3}' \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --host 0.0.0.0 \
    --port 8000 &

SERVER_PID=\$!

echo 'Waiting for server to be ready...'
for i in \$(seq 1 180); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo 'Server ready!'
        break
    fi
    if ! kill -0 \$SERVER_PID 2>/dev/null; then
        echo 'ERROR: Server process died'
        exit 1
    fi
    sleep 5
done

if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo 'ERROR: Server failed to start after 15 minutes'
    exit 1
fi

echo '=== Running benchmarks ==='
pip install requests pyarrow 2>/dev/null
rm -rf ~/.cache/benchmark_data

python3 ${BENCH_SCRIPT} \
    --base-url http://localhost:8000 \
    --output-dir ${OUTPUT_DIR} \
    --benchmarks all 2>&1

echo '=== All benchmarks complete ==='
kill \${SERVER_PID} 2>/dev/null || true
"

echo "Container started. Monitor with: docker logs -f ${CONTAINER_NAME}"
