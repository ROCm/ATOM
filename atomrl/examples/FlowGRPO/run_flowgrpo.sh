#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# verl-omni's official recipe examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh (Qwen-Image
# full-parameter FlowGRPO, Qwen3-VL-8B OCR reward, 300 steps, validation and checkpoint every 30) on one 8-GPU node.
# Overrides: local paths, 8 GPUs, ROCm attention backends, and actor offload off (it fits in 288 GB; host-RAM bound
# otherwise). LUMEN=1 adds the Lumen actor patches from this checkout's in-tree atomrl/lumen by default.
# Usage (container): RUN_NAME=<name> [LUMEN=1] [WANDB_API_KEY=...] bash run_flowgrpo.sh [extra Hydra overrides]
# Output: $WORK/runs/<RUN_NAME>/ (train.log, metrics.jsonl, validation/, checkpoints/).
set -euo pipefail
: "${WORK:?}"
: "${RUN_NAME:?set RUN_NAME}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ATOMRL_ROOT=$(cd "$HERE/../.." && pwd)
RUN_DIR=$WORK/runs/$RUN_NAME
mkdir -p "$RUN_DIR"

export WORKSPACE=$WORK NUM_GPUS=${NUM_GPUS:-8} NUM_NODES=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VERL_FILE_LOGGER_PATH=$RUN_DIR/metrics.jsonl WANDB_DIR=$RUN_DIR
if [ -n "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=${WANDB_MODE:-online}; else export WANDB_MODE=offline; fi

LUMEN_ARGS=()
if [ "${LUMEN:-0}" = 1 ]; then
  [ -f "$ATOMRL_ROOT/atomrl/lumen/config.py" ] || { echo "atomrl.lumen not found under $ATOMRL_ROOT" >&2; exit 1; }
  E=+ray_kwargs.ray_init.runtime_env.env_vars
  LUMEN_ARGS=("$E.VERL_USE_EXTERNAL_MODULES=lumen_training_patch" "$E.PYTHONPATH=$HERE:$ATOMRL_ROOT"
              "$E.LUMEN_PATCH=${LUMEN_PATCH:-rmsnorm_fuse:local_adamw}")
fi

cd "$WORK/src/verl-omni"
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh \
  data.train_files="$WORK/data/ocr/train.parquet" data.val_files="$WORK/data/ocr/test.parquet" data.seed=42 \
  actor_rollout_ref.model.path="$WORK/models/Qwen-Image" \
  reward.reward_model.model_path="$WORK/models/Qwen3-VL-8B-Instruct" \
  actor_rollout_ref.model.attn_backend=native \
  actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.seed=42 \
  actor_rollout_ref.actor.veomni_config.param_offload=False \
  actor_rollout_ref.actor.veomni_config.optimizer_offload=False \
  trainer.nnodes=1 trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.resume_mode=disable trainer.val_before_train=True trainer.max_actor_ckpt_to_keep=2 \
  trainer.default_local_dir="$RUN_DIR/checkpoints" trainer.validation_data_dir="$RUN_DIR/validation" \
  trainer.logger='[console,wandb,file]' trainer.project_name="${WANDB_PROJECT:-qwen_image_flowgrpo}" \
  trainer.experiment_name="$RUN_NAME" \
  "${LUMEN_ARGS[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
