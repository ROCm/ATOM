#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# ═══════════════════════════════════════════════════════════════════════════════
# Kimi K2.5 Eagle3 Two-Phase Training — Docker Launch (vLLM+ATOM, MI350)
#
# Uses self-contained Docker image with:
#   - vLLM 0.19.1 + extract_hidden_states patches
#   - ATOM (vLLM plugin, MXFP4 online quantization) from the current checkout
#   - AITER (GPU kernels, shared by ATOM and AtomRL) from ROCm/aiter
#   - AtomRL (FSDP2 training, aiter-patched) from Lumen
#   - MORI (MoE all-to-all) from ROCm/mori
#   - Mooncake Transfer Engine (source-built HIP, TCP)
#
# GPU split (8x MI350):
#   GPUs 0-3: FSDP2 draft model training (BF16, AtomRL + aiter)
#   GPUs 4-7: vLLM + ATOM teacher inference (TP=4, MXFP4, aiter)
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

PHASE2_ONLY=false
SMOKE_TEST=false
for arg in "$@"; do
    case "${arg}" in
        --phase2-only) PHASE2_ONLY=true ;;
        --smoke-test) SMOKE_TEST=true ;;
    esac
done

DOCKER_IMAGE="${DOCKER_IMAGE:-atomrl-vllm-mi350:latest}"
CONTAINER_NAME="kimi_k25_eagle3_v2_phase1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOMRL_DIR="${ATOMRL_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
EX=examples/SDDD/Kimi_K25_SDDD_MI350_ATOM

# Select run command
if [ "${SMOKE_TEST}" = true ]; then
    SMOKE_MODEL="${SMOKE_MODEL:-/dev/shm/Qwen3-8B}"
    RUN_CMD="CUDA_VISIBLE_DEVICES=0 python3 -m atomrl.trainer.main --config ${EX}/configs/smoke_test_hf.yaml \
policy.model_name=${SMOKE_MODEL} algorithm.teacher.model_name=${SMOKE_MODEL}"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Eagle3 SDDD Smoke Test (Docker) — HF teacher, 1 GPU, 2 steps"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  Teacher:  ${SMOKE_MODEL} (HF forward, no vLLM/ATOM/Mooncake)"
    echo "═══════════════════════════════════════════════════════════════"
elif [ "${PHASE2_ONLY}" = true ]; then
    RUN_CMD="bash ${EX}/run_full_training.sh --phase2-only"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K2.5 Eagle3 Training (Docker) — vLLM+ATOM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Phase:    Phase 2 only"
    echo "═══════════════════════════════════════════════════════════════"
else
    RUN_CMD="bash ${EX}/run_full_training.sh"
    echo "═══════════════════════════════════════════════════════════════"
    echo "  Kimi K2.5 Eagle3 Training (Docker) — vLLM+ATOM, MI350"
    echo "  Image:    ${DOCKER_IMAGE}"
    echo "  GPUs:     8x MI350 (0-3 training, 4-7 inference)"
    echo "  Phase:    Phase 1 (run again with --phase2-only for Phase 2)"
    echo "═══════════════════════════════════════════════════════════════"
fi

# Clean stale lock files and compile caches that can deadlock AITER JIT
find /tmp -name '*.lock' -path '*aiter*' -delete 2>/dev/null || true
find /tmp -name '*.lock' -path '*hiprtc*' -delete 2>/dev/null || true
rm -rf /root/.cache/atom/* 2>/dev/null || true

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --network host \
    --ipc host \
    --shm-size 64G \
    --device /dev/kfd \
    --device /dev/dri \
    --group-add video \
    --group-add render \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e PYTORCH_ROCM_ARCH=gfx950 \
    -e HIP_FORCE_DEV_KERNARG=1 \
    -e PYTHONUNBUFFERED=1 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e ATOMRL_LOG_LEVEL=INFO \
    -e NCCL_TIMEOUT=7200 \
    -e ATOMRL_TEACHER_READY_TIMEOUT_SECONDS=1800 \
    -e ATOM_WARMUP_MAX_TOKENS=8192 \
    -e ATOM_USE_TRITON_MOE=1 \
    -e GLOG_minloglevel=3 \
    -e MOONCAKE_LOG_LEVEL=FATAL \
    -e MOONCAKE_VLOG_LEVEL=-1 \
    -e WANDB_MODE=disabled \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    --log-opt max-size=500m \
    --log-opt max-file=2 \
    -v /dev/shm:/dev/shm \
    -v "${ATOMRL_DIR}/atomrl:/root/atomrl/atomrl" \
    -v "${ATOMRL_DIR}/examples:/root/atomrl/examples" \
    -v "${ATOMRL_DIR}/output:/root/atomrl/output" \
    -v "${ATOMRL_DIR}/../atom:/root/ATOM/atom" \
    -w /root/atomrl \
    "${DOCKER_IMAGE}" \
    bash -c "find /root/aiter -name 'amd_buffer_addressing_builtins.hpp' -exec sed -i 's/#if __clang_major__ >= 21 && __clang_major__ < 23/#if 0/' {} \; 2>/dev/null; rm -rf /root/aiter/aiter/jit/build/module_moe_ck2stages* 2>/dev/null; ${RUN_CMD}"

EXIT_CODE=$?
if [ ${EXIT_CODE} -eq 0 ]; then
    echo ">>> Docker run (vLLM+ATOM, MI350) completed successfully."
else
    echo ">>> Docker run (vLLM+ATOM, MI350) failed with exit code ${EXIT_CODE}." >&2
fi
exit ${EXIT_CODE}
