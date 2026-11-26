#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# ═══════════════════════════════════════════════════════════════════════════════
# DeepSeek-V4-Flash GRPO RL — Megatron TP4/PP4/EP4 + vLLM FP8 (MI300X)
#
# MILES-aligned: GRPO 32x8, 4k response, global batch 256, no KL, TP4/PP4/EP4
# vLLM FP8 per-block rollout weights + FP8 KV cache + AITER kernels
#
# Usage:
#   MODE=smoke   STEPS=3    bash examples/GRPO/run_grpo_dsv4.sh   # smoke test
#   MODE=longrun STEPS=3000 bash examples/GRPO/run_grpo_dsv4.sh   # full training
#   MODE=longrun STEPS=30   bash examples/GRPO/run_grpo_dsv4.sh   # 30-step health check
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail
ulimit -n 524288 2>/dev/null || true
: "${RL_ROOT:?需要设置 RL_ROOT}"; : "${DATA_ROOT:?需要设置 DATA_ROOT}"

MODE="${MODE:-smoke}"; STEPS="${STEPS:-3}"
ATOMRL_DIR="${ATOMRL_DIR:-$RL_ROOT/ATOM/atomrl}"
LUMEN_DIR="${LUMEN_DIR:-$ATOMRL_DIR/lumen}"

# Repository fallback
if [ ! -f "$LUMEN_DIR/lumen/config.py" ]; then
  LUMEN_DIR="$ATOMRL_DIR/lumen"
fi

MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/DeepSeek-V4-Flash-BF16}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_ROOT/data/dsv4-flash/dapo-math-17k/dapo-math-17k.jsonl}"
VAL_FILE="${VAL_FILE:-$DATA_ROOT/data/dsv4-flash/aime-2024/aime-2024.jsonl}"

RUN_ID="${RUN_ID:-dsv4-flash-grpo-${MODE}-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG:-$DATA_ROOT/logs/${RUN_ID}.log}"
CKPT_DIR="${CKPT_DIR:-$DATA_ROOT/ckpts/dsv4-flash/${MODE}}"

cd "$ATOMRL_DIR"

# ═══════════════════════════════════════════════════════════════════════
# Environment variables (DSV4-Flash: BF16 training + vLLM FP8 rollout)
# ═══════════════════════════════════════════════════════════════════════
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-/opt/rocm/lib}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=7200 NCCL_CUMEM_ENABLE=0
# Cross-node weight broadcast uses the MI300X Broadcom RoCE path.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-benic7p1}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-ionic_0}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
export NCCL_DMABUF_ENABLE="${NCCL_DMABUF_ENABLE:-0}"
# Keep production logs free of per-collective transport noise. PyTorch/Ray
# watchdog exceptions and process failures remain visible independently.
export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT}"
export HIP_FORCE_DEV_KERNARG=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0 RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Global NOSET: required for vLLM rollout (all GPUs visible).
# Training workers isolate to 1 GPU via base_worker.py module-level code.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1
export HF_HOME="$DATA_ROOT/hf_home" WANDB_DIR="$DATA_ROOT/wandb"
export WANDB_CONSOLE="${WANDB_CONSOLE:-off}"
export ATOMRL_LOG_LEVEL=INFO MODEL_NAME="$MODEL_PATH"
export ATOMRL_DEBUG="${ATOMRL_DEBUG:-0}"
export LUMEN_DISABLE_HF_ATTN_PATCH=1

# DSV4-specific env vars
# vLLM uses AITER kernels for FP8 inference on MI300X
export VLLM_ROCM_USE_AITER="${VLLM_ROCM_USE_AITER:-1}"
# Disable torch dynamo for DSV4 training stability
export TORCHDYNAMO_DISABLE=1
# MI300X GFX version override
export HSA_OVERRIDE_GFX_VERSION=9.4.2
# Long runs must allow ROCm to reclaim kernel scratch/events. Forcing
# HSA_NO_SCRATCH_RECLAIM=1 eventually exhausts HSA queue resources even when
# VRAM is available (observed after 15 full training steps on MI308X).
export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-0}"
export HSA_DISABLE_FRAGMENT_ALLOCATOR="${HSA_DISABLE_FRAGMENT_ALLOCATOR:-0}"

# Apex selects AITER fused RoPE independently of vLLM and warns that it has
# lower precision. Keep Megatron's RoPE on the standard BF16 path so R3 only
# needs to reconcile expert routing, not a second attention-position mismatch.
export USE_ROCM_AITER_ROPE_BACKEND=0

# vLLM v1. Keep expected layerwise FP8 reload and shared-memory wait warnings
# out of production logs; process failures still surface at ERROR level.
export VLLM_USE_V1=1 VLLM_ENABLE_V1_MULTIPROCESSING=1 VLLM_LOGGING_LEVEL=ERROR
export ATOM_DISABLE_VLLM_PLUGIN=1

# PYTHONPATH — unified AtomRL/Lumen source + miles shim for bootstrap compatibility
export PYTHONPATH="$ATOMRL_DIR:$ATOMRL_DIR/miles_shim:${PYTHONPATH:-}"

# The cached ROCm flash-attn wheel contains the older 21-argument native
# varlen ABI while its Python wrapper includes the newer trailing num_splits.
# Apply the idempotent compatibility fix before Ray workers import it.
python3 - <<'PY'
from pathlib import Path

path = Path("/opt/venv/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py")
if path.exists():
    source = path.read_text()
    old = "        None,\n        num_splits,\n    )"
    if old in source:
        path.write_text(source.replace(old, "        None,\n    )", 1))
        print("[run_grpo_dsv4] patched flash-attn ROCm varlen ABI")
PY

# W&B key (optional: $RL_ROOT/wandb.key format WANDB_API_KEY=xxxx)
for _wandb_key in "$RL_ROOT/wandb.key" "$RL_ROOT/../wandb.key"; do
  if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$_wandb_key" ]; then
    export WANDB_API_KEY="$(cut -d= -f2- "$_wandb_key" | tr -d '[:space:]')"
  fi
done

# ═══════════════════════════════════════════════════════════════════════
# Config selection (vLLM only — no ATOM backend for DSV4)
# ═══════════════════════════════════════════════════════════════════════
if [ "$MODE" = "longrun" ] || [ "${ATOMRL_WEIGHT_SYNC_VALIDATE:-0}" = "1" ]; then
  CONFIG=examples/GRPO/configs/grpo_dsv4_flash_vllm_longrun.yaml
else
  CONFIG=examples/GRPO/configs/grpo_dsv4_flash_vllm_smoke.yaml
fi
CONFIG="${CONFIG_OVERRIDE:-$CONFIG}"

echo "$LOG" > /tmp/run_grpo_dsv4_log.txt
echo "═══════════════════════════════════════════════════════════════"
echo "  DeepSeek-V4-Flash GRPO RL — Megatron TP4/PP4/EP4 + vLLM FP8"
echo "  MODE=$MODE  STEPS=$STEPS  CONFIG=$CONFIG"
echo "  MODEL=$MODEL_PATH"
echo "  CKPT=$CKPT_DIR  LOG=$LOG"
echo "═══════════════════════════════════════════════════════════════"

# Multi-node: one Ray cluster across the nodes, started from this script so
# every raylet -- and so every Ray worker -- carries the environment set above.
#   RAY_ROLE=head    start the Ray head on NODE_IP, then exit
#   RAY_ROLE=worker  join HEAD_IP from NODE_IP, then exit
#   RAY_ROLE=driver  run the trainer against that cluster (keeps it running)
# Unset: the existing behaviour, governed by ATOMRL_KEEP_RAY_CLUSTER.
case "${RAY_ROLE:-}" in
  head|worker)
    ray stop --force >/dev/null 2>&1 || true
    if [ "$RAY_ROLE" = head ]; then
      ray start --head --node-ip-address="${NODE_IP:?}" --port="${RAY_PORT:-6379}" \
        --num-gpus="${GPUS_PER_NODE:-8}" --dashboard-host=0.0.0.0
    else
      ray start --address="${HEAD_IP:?}:${RAY_PORT:-6379}" --node-ip-address="${NODE_IP:?}" \
        --num-gpus="${GPUS_PER_NODE:-8}"
    fi
    exit $? ;;
  driver) export ATOMRL_KEEP_RAY_CLUSTER=1 ;;
  "") ;;
  *) echo "unknown RAY_ROLE=$RAY_ROLE (head|worker|driver)" >&2; exit 2 ;;
esac

# ═══════════════════════════════════════════════════════════════════════
# Cleanup stale processes (vLLM Ray server + orphan EngineCore)
# ═══════════════════════════════════════════════════════════════════════
if [ "${ATOMRL_KEEP_RAY_CLUSTER:-0}" = "1" ]; then
  echo "[run_grpo_dsv4] ATOMRL_KEEP_RAY_CLUSTER=1 -> preserve existing Ray cluster"
else
  ray stop --force >/dev/null 2>&1 || true
fi
python3 - <<'PY'
import os, signal, subprocess
patterns = (
    "atomrl.trainer.main", "VLLMRayServer",
    "VLLM::EngineCore", "EngineCore", "spawn_main",
    "torch/_inductor/compile_worker", "multiprocessing.resource_tracker",
)
skip = {os.getpid(), os.getppid()}
out = subprocess.check_output(["ps", "-eo", "pid,ppid,stat,cmd"], text=True)
for line in out.splitlines()[1:]:
    parts = line.strip().split(None, 3)
    if len(parts) < 4:
        continue
    pid, stat, cmd = int(parts[0]), parts[2], parts[3]
    if pid in skip or "Z" in stat:
        continue
    if any(p in cmd for p in patterns):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
sleep 8

# ═══════════════════════════════════════════════════════════════════════
# Launch training
# ═══════════════════════════════════════════════════════════════════════
EXTRA_OVERRIDES=()
if [ -n "${WANDB_RUN_NAME:-}" ]; then
  EXTRA_OVERRIDES+=("logger.wandb.name=$WANDB_RUN_NAME")
fi
if [ -n "${RESUME_OVERRIDE:-}" ]; then
  EXTRA_OVERRIDES+=("checkpointing.resume=$RESUME_OVERRIDE")
fi
if [ -n "${CHECKPOINT_SAVE_STEPS:-}" ]; then
  EXTRA_OVERRIDES+=("checkpointing.save_steps=$CHECKPOINT_SAVE_STEPS")
fi
if [ -n "${CHECKPOINT_SAVE_TOTAL_LIMIT:-}" ]; then
  EXTRA_OVERRIDES+=("checkpointing.save_total_limit=$CHECKPOINT_SAVE_TOTAL_LIMIT")
fi
if [ -n "${WEIGHT_SYNC_BACKEND:-}" ]; then
  EXTRA_OVERRIDES+=("weight_sync.backend=$WEIGHT_SYNC_BACKEND")
fi
if [ -n "${FP8_QUANTIZATION_LOCATION:-}" ]; then
  EXTRA_OVERRIDES+=("weight_sync.fp8_quantization_location=$FP8_QUANTIZATION_LOCATION")
fi
ENTRYPOINT=(python3 -u -m atomrl.trainer.main)
if [ "${ATOMRL_WEIGHT_SYNC_VALIDATE:-0}" = "1" ]; then
  ENTRYPOINT=(python3 -u examples/GRPO/dsv4/validate_weight_sync.py)
fi
"${ENTRYPOINT[@]}" --config "$CONFIG" \
  policy.model_name="$MODEL_PATH" \
  reward.dataset="$TRAIN_FILE" \
  val_dataset="$VAL_FILE" \
  checkpointing.checkpoint_dir="$CKPT_DIR" \
  num_training_steps="$STEPS" \
  seed=10086 \
  "${EXTRA_OVERRIDES[@]}" ${EXTRA_OVERRIDE:-} > "$LOG" 2>&1
EXIT_CODE=$?
echo "=== exit=$EXIT_CODE ==="
exit $EXIT_CODE
