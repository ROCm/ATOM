#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# LLaMA 3.1 pretraining — unified launcher.
#
# Supports two backends via the BACKEND environment variable:
#   BACKEND=megatron  — Megatron-LM-AMD (TP/PP/CP/VP/SP)  [default]
#   BACKEND=fsdp      — PyTorch FSDP + HuggingFace
#
# Usage:
#   bash run_pretrain.sh                                       # default config
#   CONFIG=config_MI355X_1x8x1_8b.sh bash run_pretrain.sh     # MI355X config
#   GBS=64 TRAIN_STEPS=500 bash run_pretrain.sh                # override vars

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LUMEN_SOURCE_ROOT="${ATOM_ROOT}/atomrl/atomrl/lumen"
LUMEN_EXAMPLES_ROOT="${ATOM_ROOT}/atomrl/lumen"

# ---- Load configuration -----------------------------------------------------
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI355X_1x8x1.sh"}
source "${CONFIG}"

# ---- Performance tuning (model-agnostic, from common module) -----------------
source "${LUMEN_SOURCE_ROOT}/models/perf_env.sh"
export MORI_ENABLE_SDMA="${USE_SDMA}"

# ---- Compute EVAL_INTERVAL from EVAL_EVERY if not explicitly set -------------
if [ "${EVAL_EVERY}" -gt 0 ] && [ "${EVAL_INTERVAL}" -eq 0 ]; then
    EVAL_INTERVAL=$(( (EVAL_EVERY + GBS - 1) / GBS ))
fi

# ---- Conditional command suffix (Docker run_and_time.sh compat) --------------
CMD_SUFFIX=""

if [ "${USE_CKPT}" -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --use-ckpt"
    if [ "${FROM_HF}" -gt 0 ]; then
        CMD_SUFFIX="${CMD_SUFFIX} --resume-from-hf"
    fi
fi

if [ "${SAVE_CKPT}" -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --save-ckpt"
fi

if [ -n "${TAG}" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --tag ${TAG}"
fi

if [ "${FP8_PARAMS}" -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --fp8-params"
fi

if [ "${USE_SDMA}" -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --use-sdma"
fi


###############################################################################
# MEGATRON BACKEND
###############################################################################

run_megatron() {
    python -c "import megatron" 2>/dev/null || {
        echo "ERROR: Megatron-LM is not installed."
        echo "  pip install git+https://github.com/ROCm/Megatron-LM.git"
        exit 1
    }

    MEGATRON_ROOT="${MEGATRON_ROOT:-${MEGATRON_PATH:-/workspace/megatron_lm}}"
    PATCH_SCRIPT="${LUMEN_EXAMPLES_ROOT}/examples/dsv4/patch_megatron_source.py"
    if [[ -f "${PATCH_SCRIPT}" && -d "${MEGATRON_ROOT}" ]]; then
        PYTHONPATH="${ATOM_ROOT}/atomrl" python3 "${PATCH_SCRIPT}" "${MEGATRON_ROOT}" --tag llama
    fi

    LUMEN_ATTN_BACKEND=${LUMEN_ATTN_BACKEND:-"aiter_csrc"}
    LUMEN_FP8_QUANT=${LUMEN_FP8_QUANT:-"blockwise"}

    # LLaMA 3.1 architecture
    case "${SIZE}" in
        8b|8B)
            NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=14336
            HEADS=32; KV_HEADS=8; VOCAB=128256
            ROPE_THETA=500000
            ;;
        *)
            echo "ERROR: Unknown SIZE=${SIZE}. Supported: 8b"
            exit 1 ;;
    esac

    GQA_ARGS="--group-query-attention --num-query-groups ${KV_HEADS}"

    VP_ARGS=""; [ "${VP}" -gt 0 ] && VP_ARGS="--num-layers-per-virtual-pipeline-stage ${VP}"
    # Megatron-LM-AMD requires sequence parallelism whenever tensor parallelism is
    # used (validate_args enforces this unconditionally for TP > 1).
    if [ "${TP}" -gt 1 ] && [ "${SP}" -ne 1 ]; then
        echo "WARNING: TP=${TP} > 1 requires SP=1 (Megatron-LM-AMD enforce). Enabling automatically."
        SP=1
    fi
    SP_ARGS=""; [ "${SP}" -eq 1 ] && SP_ARGS="--sequence-parallel"

    LORA_ARGS=""
    if [ "${LORA_RANK}" -gt 0 ]; then
        LORA_ARGS="--lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA} --lora-dropout ${LORA_DROPOUT}"
        [ "${LORA_A2A}" -eq 1 ] && LORA_ARGS="${LORA_ARGS} --lora-a2a"
    fi

    FP8_ARGS=""
    if [ "${FP8_TRAINING}" -eq 1 ]; then
        FP8_ARGS="--linear-fp8 --fp8-format ${FP8_FORMAT} --linear-fp8-scaling ${FP8_SCALING} --linear-fp8-block-size ${FP8_BLOCK_SIZE}"
        FP8_ARGS+=" --linear-fp8-amax-algo ${FP8_AMAX_ALGO} --linear-fp8-amax-history ${FP8_AMAX_HISTORY}"
        [ "${FP8_REDUCE_AMAX}" = "1" ] && FP8_ARGS+=" --linear-fp8-reduce-amax"
        [ "${FP8_ACTIVATION}" = "0" ] && FP8_ARGS+=" --no-linear-fp8-activation"
        [ "${FP8_WGRAD:-1}" = "0" ] && FP8_ARGS+=" --no-linear-fp8-wgrad"
        [ -n "${GRAD_QUANT_TYPE}" ] && FP8_ARGS+=" --grad-quant-type ${GRAD_QUANT_TYPE}"
        if [ "${FIRST_LAST_BF16:-0}" = "1" ]; then
            FP8_ARGS+=" --first-last-layers-bf16"
            FP8_ARGS+=" --num-layers-at-start-in-bf16 ${BF16_LAYERS_START:-1}"
            FP8_ARGS+=" --num-layers-at-end-in-bf16 ${BF16_LAYERS_END:-1}"
        fi
    fi

    LUMEN_RMSNORM=${LUMEN_RMSNORM:-0}
    LUMEN_RMSNORM_ARGS=""
    [ "${LUMEN_RMSNORM}" -eq 1 ] && LUMEN_RMSNORM_ARGS="--lumen-rmsnorm"

    LUMEN_LINEAR=${LUMEN_LINEAR:-0}
    LUMEN_LINEAR_ARGS=""
    [ "${LUMEN_LINEAR}" -eq 1 ] && LUMEN_LINEAR_ARGS="--lumen-linear"

    LUMEN_CROSS_ENTROPY=${LUMEN_CROSS_ENTROPY:-0}
    LUMEN_CE_ARGS=""
    [ "${LUMEN_CROSS_ENTROPY}" -eq 1 ] && LUMEN_CE_ARGS="--lumen-cross-entropy"

    WARMUP_ARGS=""; [ "${WARMUP_STEPS}" -gt 0 ] && WARMUP_ARGS="--warmup-steps ${WARMUP_STEPS}"
    EARLY_STOP_ARGS=""; [ -n "${VAL_LOSS_TARGET}" ] && EARLY_STOP_ARGS="--val-loss-target ${VAL_LOSS_TARGET}"

    VALID_ARGS=""
    [ -n "${VALID_DATA}" ] && VALID_ARGS="--valid-data-path ${VALID_DATA}"

    echo "================================================================"
    echo "LLaMA 3.1 Pretrain — MEGATRON backend"
    echo "  Config:   ${CONFIG}"
    echo "  Model:    ${SIZE} | TP=${TP} PP=${PP} CP=${CP} VP=${VP} SP=${SP}"
    echo "  GPUs:     ${NGPU}x${NNODES}"
    echo "  Batch:    MBS=${MBS} GBS=${GBS} | seq_len=${SEQ_LEN}"
    echo "  LR:       max=${MAX_LR} min=${MIN_LR} warmup=${LR_WARMUP_STEPS}"
    echo "  Lumen attn: ${LUMEN_ATTN_BACKEND} (fp8_quant=${LUMEN_FP8_QUANT}) rmsnorm=${LUMEN_RMSNORM}"
    echo "  FP8:      training=${FP8_TRAINING} format=${FP8_FORMAT} algo=${FP8_AMAX_ALGO} hist=${FP8_AMAX_HISTORY}"
    echo "  SDMA:     use=${USE_SDMA} mori=${MORI_ENABLE_SDMA}"
    echo "  LoRA:     rank=${LORA_RANK}"
    echo "  Target:   log_ppl=${TARGET_LOG_PPL} step_atol=${STEP_TIME_ATOL}"
    echo "  Ckpt:     use=${USE_CKPT} save=${SAVE_CKPT} fp8_params=${FP8_PARAMS} start_step=${CKPT_START_STEP}"
    echo "  Eval:     every=${EVAL_EVERY}seqs (${EVAL_INTERVAL}steps) start_at=${START_EVAL_AT}"
    echo "================================================================"

    torchrun --nproc_per_node=${NGPU} --nnodes=${NNODES} \
        "${SCRIPT_DIR}/pretrain_llama31.py" \
        --backend megatron \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN} \
        --ffn-hidden-size ${FFN_HIDDEN} \
        --num-attention-heads ${HEADS} \
        ${GQA_ARGS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings 131072 \
        --use-rotary-position-embeddings \
        --rotary-base ${ROPE_THETA} \
        --no-position-embedding \
        --normalization RMSNorm \
        --swiglu \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --no-masked-softmax-fusion \
        --attention-softmax-in-fp32 \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --context-parallel-size ${CP} \
        ${VP_ARGS} ${SP_ARGS} \
        --micro-batch-size ${MBS} \
        --global-batch-size ${GBS} \
        --train-iters ${TRAIN_STEPS} \
        --lr ${MAX_LR} --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --lr-warmup-iters ${LR_WARMUP_STEPS} \
        --weight-decay ${WEIGHT_DECAY} \
        --clip-grad ${GRADIENT_CLIP} \
        --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-5 \
        --${PRECISION} \
        --no-gradient-accumulation-fusion \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${TOKENIZER} \
        --train-data-path ${TRAIN_DATA} \
        ${VALID_ARGS} \
        --split 100,0,0 \
        --save ${SAVE_DIR} \
        --seed ${SEED} \
        --eval-iters ${EVAL_ITERS} \
        --eval-interval ${EVAL_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} --log-interval ${LOG_INTERVAL} \
        --lumen-attn-backend ${LUMEN_ATTN_BACKEND} \
        --lumen-fp8-quant-type ${LUMEN_FP8_QUANT} \
        --mxfp8-block-m-fwd ${MXFP8_BLOCK_M_FWD} \
        --mxfp8-block-n-fwd ${MXFP8_BLOCK_N_FWD} \
        --mxfp8-block-m-dq-bwd ${MXFP8_BLOCK_M_DQ_BWD} \
        --mxfp8-block-n-dq-bwd ${MXFP8_BLOCK_N_DQ_BWD} \
        --mxfp8-block-m-dkv-bwd ${MXFP8_BLOCK_M_DKV_BWD} \
        --mxfp8-block-n-dkv-bwd ${MXFP8_BLOCK_N_DKV_BWD} \
        --mxfp8-quant-block-size ${MXFP8_QUANT_BLOCK_SIZE} \
        --continual-ckpt-path ${CONTINUAL_CKPT} \
        --target-log-ppl ${TARGET_LOG_PPL} \
        --step-time-atol ${STEP_TIME_ATOL} \
        --ckpt-start-step ${CKPT_START_STEP} \
        --eval-every ${EVAL_EVERY} \
        --start-eval-at ${START_EVAL_AT} \
        ${LORA_ARGS} ${FP8_ARGS} ${WARMUP_ARGS} ${EARLY_STOP_ARGS} ${LUMEN_RMSNORM_ARGS} \
        ${LUMEN_LINEAR_ARGS} ${LUMEN_CE_ARGS} \
        ${CMD_SUFFIX}
}


###############################################################################
# FSDP BACKEND
###############################################################################

run_fsdp() {
    python -c "import transformers" 2>/dev/null || {
        echo "ERROR: HuggingFace Transformers is not installed."
        echo "  pip install transformers peft"
        exit 1
    }

    CMD="torchrun --nproc_per_node=${NGPU}"
    CMD+=" ${SCRIPT_DIR}/pretrain_llama31.py"
    CMD+=" --backend fsdp"
    CMD+=" --model-name-or-path ${MODEL}"
    CMD+=" --tokenizer-name-or-path ${TOKENIZER}"
    CMD+=" --seq-length ${SEQ_LEN}"
    CMD+=" --micro-batch-size ${MBS}"
    CMD+=" --gradient-accumulation-steps ${GRAD_ACCUM}"
    CMD+=" --max-steps ${TRAIN_STEPS}"
    CMD+=" --lr ${MAX_LR}"
    CMD+=" --min-lr ${MIN_LR}"
    CMD+=" --lr-warmup-steps ${LR_WARMUP_STEPS}"
    CMD+=" --weight-decay ${WEIGHT_DECAY}"
    CMD+=" --max-grad-norm ${MAX_GRAD_NORM}"
    CMD+=" --log-interval ${LOG_INTERVAL}"
    CMD+=" --save-interval ${SAVE_INTERVAL}"
    CMD+=" --eval-interval ${EVAL_INTERVAL}"
    CMD+=" --save-dir ${SAVE_DIR}"
    CMD+=" --num-workers ${NUM_WORKERS}"
    CMD+=" --train-data-path ${TRAIN_DATA}"
    CMD+=" --sharding-strategy ${SHARDING}"

    [ -n "${VALID_DATA}" ] && CMD+=" --val-data-path ${VALID_DATA}"
    [ -n "${TRAIN_SAMPLES}" ] && CMD+=" --train-samples ${TRAIN_SAMPLES}"
    [ -n "${VAL_SAMPLES}" ] && CMD+=" --val-samples ${VAL_SAMPLES}"

    if [ "${LORA_RANK}" -gt 0 ]; then
        CMD+=" --lora-rank ${LORA_RANK} --lora-alpha ${LORA_ALPHA} --lora-dropout ${LORA_DROPOUT}"
    fi

    if [ "${FP8_TRAINING}" = "1" ]; then
        CMD+=" --linear-fp8"
        CMD+=" --linear-fp8-format ${FP8_FORMAT} --linear-fp8-scaling ${FP8_SCALING}"
        CMD+=" --linear-fp8-block-size ${FP8_BLOCK_SIZE} --linear-fp8-amax-algo ${FP8_AMAX_ALGO}"
        CMD+=" --linear-fp8-amax-history ${FP8_AMAX_HISTORY}"
        [ "${FP8_REDUCE_AMAX}" = "1" ] && CMD+=" --linear-fp8-reduce-amax"
        [ "${FP8_ACTIVATION}" = "0" ] && CMD+=" --no-linear-fp8-activation"
        [ "${FP8_WGRAD:-1}" = "0" ] && CMD+=" --no-linear-fp8-wgrad"
        [ -n "${GRAD_QUANT_TYPE}" ] && CMD+=" --grad-quant-type ${GRAD_QUANT_TYPE}"
        if [ "${FIRST_LAST_BF16:-0}" = "1" ]; then
            CMD+=" --first-last-layers-bf16"
            CMD+=" --num-layers-at-start-in-bf16 ${BF16_LAYERS_START:-1}"
            CMD+=" --num-layers-at-end-in-bf16 ${BF16_LAYERS_END:-1}"
        fi
    fi

    [ "${WARMUP_STEPS}" -gt 0 ] && CMD+=" --warmup-steps ${WARMUP_STEPS}"
    [ -n "${VAL_LOSS_TARGET}" ] && CMD+=" --val-loss-target ${VAL_LOSS_TARGET}"

    CMD+=" --continual-ckpt-path ${CONTINUAL_CKPT}"
    CMD+=" --target-log-ppl ${TARGET_LOG_PPL}"
    CMD+=" --step-time-atol ${STEP_TIME_ATOL}"
    CMD+=" --ckpt-start-step ${CKPT_START_STEP}"
    CMD+=" --eval-every ${EVAL_EVERY}"
    CMD+=" --start-eval-at ${START_EVAL_AT}"
    CMD+=" ${CMD_SUFFIX}"

    echo "================================================================"
    echo "LLaMA 3.1 Pretrain — FSDP backend"
    echo "  Config:     ${CONFIG}"
    echo "  Model:      ${MODEL}"
    echo "  GPUs:       ${NGPU}"
    echo "  Batch:      MBS=${MBS} x accum=${GRAD_ACCUM} | seq_len=${SEQ_LEN}"
    echo "  Sharding:   ${SHARDING}"
    echo "  FP8:        training=${FP8_TRAINING} format=${FP8_FORMAT}"
    echo "  SDMA:       use=${USE_SDMA} mori=${MORI_ENABLE_SDMA}"
    echo "  LoRA:       rank=${LORA_RANK}"
    echo "  Target:     log_ppl=${TARGET_LOG_PPL} step_atol=${STEP_TIME_ATOL}"
    echo "  Ckpt:       use=${USE_CKPT} save=${SAVE_CKPT} fp8_params=${FP8_PARAMS} start_step=${CKPT_START_STEP}"
    echo "  Eval:       every=${EVAL_EVERY}seqs (${EVAL_INTERVAL}steps) start_at=${START_EVAL_AT}"
    echo "================================================================"

    eval ${CMD}
}


###############################################################################
# DISPATCH
###############################################################################

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${start_fmt} (backend=${BACKEND})"

set -x

ret_code=0
case "${BACKEND}" in
    megatron) run_megatron || ret_code=$? ;;
    fsdp)     run_fsdp || ret_code=$? ;;
    *)
        echo "ERROR: Unknown BACKEND=${BACKEND}. Use 'megatron' or 'fsdp'."
        exit 1 ;;
esac

if [[ ${ret_code} != 0 ]]; then exit ${ret_code}; fi

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${end_fmt}"
result=$(( end - start ))
result_name="LLM_PRETRAINING"
echo "RESULT,${result_name},,${result},AMD,${start_fmt}"

exit 0
