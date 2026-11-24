#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Shared helpers for native GRPO finetune inner scripts (source, do not execute).

# Finetune defaults: GBS=256, seq_len=4096 (Miles GRPO smoke: 32×8 rollout samples).
# Megatron arg files still use GBS=8 / SEQ_LEN=2048 for pretrain-only paths; this
# helper overrides those when sourcing finetune inner scripts. Set DSV4_KEEP_GBS=1 or
# DSV4_KEEP_SEQ_LEN=1 to preserve explicit smoke overrides (e.g. GBS=8, SEQ_LEN=512).
dsv4_apply_finetune_batch_defaults() {
    GBS="${GBS:-256}"
    if [[ "${GBS}" == "8" && "${DSV4_KEEP_GBS:-0}" != "1" ]]; then
        GBS=256
    fi
    MBS="${MBS:-1}"
    SEQ_LEN="${SEQ_LEN:-4096}"
    if [[ "${SEQ_LEN}" == "2048" && "${DSV4_KEEP_SEQ_LEN:-0}" != "1" ]]; then
        SEQ_LEN=4096
    fi
    export GBS MBS SEQ_LEN
}

# Resolve torch_dist checkpoint: prefer ${MODEL_NAME}_torch_dist, optional hc suffix fallback.
dsv4_resolve_finetune_ckpt() {
    local model_name="${1:?model_name}"
    local hc_mult="${2:-4}"
    local models_dir="${3:-/root/models}"
    CKPT="${models_dir}/${model_name}_torch_dist"
    if [[ ! -f "${CKPT}/latest_checkpointed_iteration.txt" ]]; then
        local fallback="${models_dir}/${model_name}_torch_dist_hc${hc_mult}"
        if [[ -f "${fallback}/latest_checkpointed_iteration.txt" ]]; then
            echo "[prepare] using fallback checkpoint ${fallback}"
            CKPT="${fallback}"
        fi
    fi
    export CKPT
}

# Host-side ckpt path for run_dsv4.sh banners (before Docker).
dsv4_host_resolve_finetune_ckpt() {
    local model_name="${1:?model_name}"
    local hc_mult="${2:-4}"
    dsv4_resolve_finetune_ckpt "${model_name}" "${hc_mult}" "${MODEL_DIR}"
    CKPT_HOST="${CKPT}"
    export CKPT_HOST
}

# Multinode: place rollout on shared DATA_ROOT so head/worker with different MODEL_DIR agree.
dsv4_resolve_fake_rollout_path() {
    if [[ -n "${FAKE_ROLLOUT_DATA:-}" ]]; then
        export FAKE_ROLLOUT_DATA
        return
    fi
    if [[ "${NNODES:-1}" -gt 1 && -n "${DATA_ROOT:-}" ]]; then
        FAKE_ROLLOUT_DATA="${DATA_ROOT}/models/fake_rollout.pt"
    else
        FAKE_ROLLOUT_DATA="/root/models/fake_rollout.pt"
    fi
    export FAKE_ROLLOUT_DATA
}

dsv4_prepare_rollout_on_shared_fs() {
    dsv4_resolve_fake_rollout_path
    local rollout_path="${FAKE_ROLLOUT_DATA}"
    mkdir -p "$(dirname "${rollout_path}")"
    export PYTHONPATH="/workspace/ATOM/atomrl:${PYTHONPATH:-}"

    if [[ "${NODE_RANK:-0}" == "0" ]]; then
        PYTHONPATH="/workspace/ATOM/atomrl/examples/dsv4/tools:${PYTHONPATH:-}" \
            python -c "from gen_fake_rollout_data import prepare; prepare()"
        return
    fi

    echo "[finetune] node_rank=${NODE_RANK} — waiting for shared rollout ${rollout_path} ..."
    local _i
    for _i in $(seq 1 120); do
        if [[ -f "${rollout_path}" ]]; then
            echo "[finetune] rollout ready (${_i}x5s)"
            return
        fi
        sleep 5
    done
    echo "[finetune] ERROR: rollout not found after wait: ${rollout_path}"
    exit 1
}

dsv4_finetune_recompute_args() {
    RECOMPUTE_ARGS=()
    if [[ "${DSV4_ENABLE_RECOMPUTE:-1}" == "1" ]]; then
        RECOMPUTE_ARGS=(
            --recompute-granularity full
            --recompute-method uniform
            --recompute-num-layers 1
        )
    fi
}

# GRPO training log hint (rollout / step / perf per GRPO step).
dsv4_print_finetune_log_hint() {
    local tag="${1:-finetune}"
    echo "[${tag}] training logs (GRPO format):"
    echo "[${tag}]   rollout {id}: rollout/num_samples, rollout/advantages, ..."
    echo "[${tag}]   step {id}: train/loss, train/pg_loss, train/grad_norm, train/lr-pg_*"
    echo "[${tag}]   perf {id}: perf/actor_train_time, perf/actor_train_tok_per_s"
}

dsv4_print_finetune_launch_info() {
    local tag="${1:-finetune}"
    echo "[${tag}] Megatron path: ${MEGATRON_PATH:-unset}"
    echo "[${tag}] GRPO steps   : ${NUM_ROLLOUT} (DEBUG_TRAIN_ONLY=${DEBUG_TRAIN_ONLY:-1})"
    echo "[${tag}] hc_mult      : ${DSV4_HC_MULT}"
    echo "[${tag}] batch        : GBS=${GBS} MBS=${MBS} seq_len=${SEQ_LEN}"
    echo "[${tag}] rollout      : ${FAKE_ROLLOUT_DATA:-(unset)}"
    if [[ -n "${NNODES:-}" && "${NNODES:-1}" != "1" ]]; then
        echo "[${tag}] nodes        : ${NNODES}×${NPROC_PER_NODE:-8} node_rank=${NODE_RANK:-0}"
        echo "[${tag}] parallel     : TP=${TP} PP=${PP} EP=${EP}"
        if [[ -n "${OPTIMIZER_OFFLOAD_FRACTION:-}" ]]; then
            echo "[${tag}] optimizer    : CPU offload fraction=${OPTIMIZER_OFFLOAD_FRACTION}"
        fi
    fi
    dsv4_print_finetune_log_hint "${tag}"
}

# shellcheck disable=SC2034
DSV4_FINETUNE_TORCHRUN_ARGS=(
    --transformer-impl local
    --disable-jit-fuser
    --moe-router-freeze-gate
    --freeze-e-score-correction-bias
    --sequence-parallel
    --mock-data
    --split 100,0,0
    --bf16
    --no-gradient-accumulation-fusion
    --accumulate-allreduce-grads-in-fp32
    --use-distributed-optimizer
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --clip-grad 1.0
    --log-interval 1
    --save-interval 1000000
    --eval-interval 1000000
    --eval-iters 0
    --no-load-optim
    --no-load-rng
    --distributed-backend nccl
)
