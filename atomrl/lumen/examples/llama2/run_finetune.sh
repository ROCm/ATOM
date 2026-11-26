#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

#
# LLaMA2 SFT fine-tuning — unified launcher.
#
# Supports two backends via the BACKEND environment variable:
#   BACKEND=megatron  — Megatron-LM-AMD (TP/PP/CP/VP/SP)  [default]
#   BACKEND=fsdp      — PyTorch FSDP + HuggingFace
#
# Usage:
#   bash run_finetune.sh                                       # default config
#   CONFIG=config_MI300X_1x8x1.sh bash run_finetune.sh         # MI300X config
#   TP=2 TRAIN_STEPS=500 bash run_finetune.sh                  # override vars

set -euo pipefail

export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATOM_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
# Host checkouts live at <ATOM>/atomrl/lumen/examples/llama2 (four parents is
# the ATOM root). The FSDP launcher mounts the examples tree at
# /workspace/Lumen, which makes those four parents "/" — fall back to the
# checkout mount the other example containers use.
if [ ! -f "${ATOM_ROOT}/atomrl/atomrl/lumen/models/perf_env.sh" ] \
    && [ -f /workspace/ATOM/atomrl/atomrl/lumen/models/perf_env.sh ]; then
    ATOM_ROOT=/workspace/ATOM
fi
LUMEN_SOURCE_ROOT="${ATOM_ROOT}/atomrl/atomrl/lumen"

# ---- Load configuration -----------------------------------------------------
CONFIG=${CONFIG:-"${SCRIPT_DIR}/config_MI355X_1x8x1.sh"}
source "${CONFIG}"

# ---- Data path fallback: .jsonl -> .npy when only .npy exists -----------------
for _var in TRAIN_DATA VALID_DATA; do
    _path="${!_var}"
    [ -z "${_path}" ] && continue
    if [ ! -f "${_path}" ] && [[ "${_path}" == *.jsonl ]]; then
        _npy="${_path%.jsonl}.npy"
        if [ -f "${_npy}" ]; then
            echo "WARNING: ${_var}='${_path}' not found; using ${_npy}"
            export "${_var}=${_npy}"
        fi
    fi
done

# ---- Validate tokenizer path (fall back to script-relative path) -------------
# An empty examples/llama2/tokenizer directory still exists; only a real
# tokenizer_config.json counts. The fallback uses the public NousResearch mirror.
if [ ! -f "${TOKENIZER:-}/tokenizer_config.json" ]; then
    _FALLBACK="${SCRIPT_DIR}/tokenizer"
    source "${SCRIPT_DIR}/../scripts/ensure_tokenizer.sh"
    if example_tokenizer llama2 "${_FALLBACK}"; then
        echo "WARNING: TOKENIZER='${TOKENIZER:-}' has no tokenizer_config.json; using ${_FALLBACK}"
        export TOKENIZER="${_FALLBACK}"
    else
        echo "ERROR: Tokenizer directory not found: ${TOKENIZER:-<unset>}"
        echo "  Set TOKENIZER=/path/to/tokenizer or allow ${_FALLBACK}"
        echo "  to be fetched from NousResearch/Llama-2-7b-hf."
        exit 1
    fi
fi

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

if [ "${FP8_PARAM_STORAGE:-0}" -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --fp8-param-storage"
fi

if [ "${FP8_CACHE_FROZEN:-0}" = "1" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --linear-fp8-cache-frozen-weight"
fi

if [ "${FP8_ACT_STORE:-0}" = "1" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --lumen-fp8-activation-store"
fi

if [ "${USE_SDMA}" -gt 0 ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --use-sdma"
fi

if [ "${LUMEN_FUSED_ROPE:-0}" = "1" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --lumen-fused-rope"
fi

if [ "${LUMEN_FUSED_MLP:-0}" = "1" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --lumen-fused-mlp"
fi

if [ "${LUMEN_MANUAL_GC:-0}" = "1" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --manual-gc"
fi

if [ "${LUMEN_CROSS_ENTROPY:-0}" = "1" ]; then
    CMD_SUFFIX="${CMD_SUFFIX} --lumen-cross-entropy"
    if [ -n "${LUMEN_CE_CHUNK_ROWS:-}" ] && [ "${LUMEN_CE_CHUNK_ROWS}" -gt 0 ]; then
        CMD_SUFFIX="${CMD_SUFFIX} --lumen-ce-chunk-rows ${LUMEN_CE_CHUNK_ROWS}"
    fi
fi

if [ -n "${LUMEN_LOG_INTERVAL:-}" ]; then
    LOG_INTERVAL="${LUMEN_LOG_INTERVAL}"
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

    case "${MODEL_SIZE}" in
        llama2-7B|llama2-7b)
            NUM_LAYERS=32; HIDDEN=4096; FFN_HIDDEN=11008; HEADS=32; KV_HEADS=32 ;;
        llama2-13B|llama2-13b)
            NUM_LAYERS=40; HIDDEN=5120; FFN_HIDDEN=13824; HEADS=40; KV_HEADS=40 ;;
        llama2-70B|llama2-70b)
            NUM_LAYERS=80; HIDDEN=8192; FFN_HIDDEN=28672; HEADS=64; KV_HEADS=8 ;;
        *)
            echo "ERROR: Unknown MODEL_SIZE=${MODEL_SIZE} (use llama2-7B/13B/70B)"
            exit 1 ;;
    esac

    GQA_ARGS=""
    if [ "${KV_HEADS}" -ne "${HEADS}" ]; then
        GQA_ARGS="--group-query-attention --num-query-groups ${KV_HEADS}"
    fi

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
        LORA_ARGS="${LORA_ARGS} --lora-target-modules ${LORA_TARGET_MODULES:-all}"
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
        [ "${FP8_CHECKPOINT:-0}" = "1" ] && FP8_ARGS+=" --lumen-fp8-checkpoint"
    fi

    LUMEN_ATTN_ARGS="--lumen-attn-backend ${LUMEN_ATTN_BACKEND}"
    case "${LUMEN_ATTN_BACKEND}" in
        aiter_triton_fp8|aiter_csrc_fp8|aiter_asm_fp8)
            LUMEN_ATTN_ARGS+=" --lumen-fp8-quant-type ${LUMEN_FP8_QUANT}"
            if [ "${LUMEN_FP8_QUANT}" = "mxfp8" ]; then
                LUMEN_ATTN_ARGS+=" --mxfp8-block-m-fwd ${MXFP8_BLOCK_M_FWD}"
                LUMEN_ATTN_ARGS+=" --mxfp8-block-n-fwd ${MXFP8_BLOCK_N_FWD}"
                LUMEN_ATTN_ARGS+=" --mxfp8-block-m-dq-bwd ${MXFP8_BLOCK_M_DQ_BWD}"
                LUMEN_ATTN_ARGS+=" --mxfp8-block-n-dq-bwd ${MXFP8_BLOCK_N_DQ_BWD}"
                LUMEN_ATTN_ARGS+=" --mxfp8-block-m-dkv-bwd ${MXFP8_BLOCK_M_DKV_BWD}"
                LUMEN_ATTN_ARGS+=" --mxfp8-block-n-dkv-bwd ${MXFP8_BLOCK_N_DKV_BWD}"
                LUMEN_ATTN_ARGS+=" --mxfp8-quant-block-size ${MXFP8_QUANT_BLOCK_SIZE}"
            fi
            ;;
    esac

    LUMEN_RMSNORM_ARGS=""
    [ "${LUMEN_RMSNORM}" -eq 1 ] && LUMEN_RMSNORM_ARGS="--lumen-rmsnorm"
    [ "${LUMEN_NORM:-0}" -eq 1 ] && LUMEN_RMSNORM_ARGS="${LUMEN_RMSNORM_ARGS} --lumen-norm"

    LUMEN_LINEAR_ARGS=""
    [ "${LUMEN_LINEAR:-0}" -eq 1 ] && LUMEN_LINEAR_ARGS="--lumen-linear"

    RECOMPUTE_ARGS=""
    if [ -n "${RECOMPUTE_GRANULARITY:-}" ]; then
        RECOMPUTE_ARGS="--recompute-granularity ${RECOMPUTE_GRANULARITY} --recompute-method ${RECOMPUTE_METHOD} --recompute-num-layers ${RECOMPUTE_NUM_LAYERS}"
    fi

    WARMUP_ARGS=""; [ "${WARMUP_STEPS}" -gt 0 ] && WARMUP_ARGS="--warmup-steps ${WARMUP_STEPS}"
    EARLY_STOP_ARGS=""; [ -n "${VAL_LOSS_TARGET}" ] && EARLY_STOP_ARGS="--val-loss-target ${VAL_LOSS_TARGET}"
    MEMORY_ARGS=""; [ "${LOG_MEMORY:-0}" -eq 1 ] && MEMORY_ARGS="--log-memory-to-tensorboard"
    LR_DECAY_ARGS=""; [ -n "${LR_DECAY_ITERS:-}" ] && LR_DECAY_ARGS="--lr-decay-iters ${LR_DECAY_ITERS}"

    DIST_OPT_ARGS=""
    [ "${USE_DIST_OPTIMIZER:-0}" = "1" ] && DIST_OPT_ARGS="--use-distributed-optimizer"
    [ "${OVERLAP_GRAD_REDUCE:-0}" = "1" ] && DIST_OPT_ARGS="${DIST_OPT_ARGS} --overlap-grad-reduce"
    [ "${LUMEN_HIP_GRAPHS:-0}" = "1" ] && DIST_OPT_ARGS="${DIST_OPT_ARGS} --lumen-hip-graphs"

    RESET_ARGS="--reset-position-ids --reset-attention-mask --eod-mask-loss"
    if [ "${DISABLE_RESET_FLAGS:-0}" = "1" ]; then
        RESET_ARGS=""
        echo "[CONFIG] reset_position_ids/reset_attention_mask/eod_mask_loss DISABLED (MLPerf alignment)"
    fi

    echo "================================================================"
    echo "LLaMA2 SFT — MEGATRON backend"
    echo "  Config:   ${CONFIG}"
    echo "  Model:    ${MODEL_SIZE} | TP=${TP} PP=${PP} CP=${CP} VP=${VP} SP=${SP}"
    echo "  GPUs:     ${NGPU}x${NNODES}"
    echo "  Batch:    MBS=${MBS} GBS=${GBS} | seq_len=${SEQ_LEN}"
    echo "  Lumen attn: ${LUMEN_ATTN_BACKEND}$(case "${LUMEN_ATTN_BACKEND}" in *fp8) echo " (fp8_quant=${LUMEN_FP8_QUANT})";; esac) rmsnorm=${LUMEN_RMSNORM}"
    echo "  LoRA:     rank=${LORA_RANK} a2a=${LORA_A2A}"
    echo "  FP8:      training=${FP8_TRAINING} format=${FP8_FORMAT} scaling=${FP8_SCALING} algo=${FP8_AMAX_ALGO} hist=${FP8_AMAX_HISTORY}"
    echo "  FP8 det:  block_size=${FP8_BLOCK_SIZE} reduce_amax=${FP8_REDUCE_AMAX} activation=${FP8_ACTIVATION} wgrad=${FP8_WGRAD:-1}"
    echo "  Memory:   log=${LOG_MEMORY:-0}"
    echo "  SDMA:     use=${USE_SDMA} mori=${MORI_ENABLE_SDMA}"
    echo "  Target:   log_ppl=${TARGET_LOG_PPL} step_atol=${STEP_TIME_ATOL}"
    echo "  Ckpt:     use=${USE_CKPT} save=${SAVE_CKPT} fp8_params=${FP8_PARAMS} start_step=${CKPT_START_STEP}"
    echo "  Eval:     every=${EVAL_EVERY}seqs (${EVAL_INTERVAL}steps) start_at=${START_EVAL_AT}"
    echo "================================================================"

    torchrun --nproc_per_node=${NGPU} --nnodes=${NNODES} \
        "${SCRIPT_DIR}/finetune_llama2.py" \
        --backend megatron \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN} \
        --ffn-hidden-size ${FFN_HIDDEN} \
        --num-attention-heads ${HEADS} \
        ${GQA_ARGS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --use-rotary-position-embeddings \
        --no-position-embedding \
        --normalization RMSNorm \
        --swiglu \
        --untie-embeddings-and-output-weights \
        --make-vocab-size-divisible-by ${MAKE_VOCAB_DIVISIBLE_BY:-128} \
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
        --lr ${LR} --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --lr-warmup-iters ${LR_WARMUP_STEPS} \
        ${LR_DECAY_ARGS} \
        --weight-decay ${WEIGHT_DECAY} \
        --clip-grad ${GRADIENT_CLIP} \
        --adam-beta1 ${ADAM_BETA1:-0.9} --adam-beta2 ${ADAM_BETA2:-0.999} --adam-eps ${ADAM_EPS:-1e-8} \
        --${PRECISION} \
        --no-gradient-accumulation-fusion \
        ${RESET_ARGS} \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model ${TOKENIZER} \
        --train-data-path ${TRAIN_DATA} \
        --valid-data-path ${VALID_DATA} \
        --split 100,0,0 \
        --load ${CKPT_DIR} \
        --save ${SAVE_DIR} \
        --seed ${SEED} \
        --finetune --no-load-optim --no-load-rng --auto-detect-ckpt-format \
        --eval-iters ${EVAL_ITERS} --eval-interval ${EVAL_INTERVAL} \
        --save-interval ${SAVE_INTERVAL} --log-interval ${LOG_INTERVAL} \
        --continual-ckpt-path ${CONTINUAL_CKPT} \
        --target-log-ppl ${TARGET_LOG_PPL} \
        --step-time-atol ${STEP_TIME_ATOL} \
        --ckpt-start-step ${CKPT_START_STEP} \
        --eval-every ${EVAL_EVERY} \
        --start-eval-at ${START_EVAL_AT} \
        ${LUMEN_ATTN_ARGS} ${LUMEN_RMSNORM_ARGS} ${LUMEN_LINEAR_ARGS} \
        ${RECOMPUTE_ARGS} \
        ${LORA_ARGS} ${FP8_ARGS} ${WARMUP_ARGS} ${EARLY_STOP_ARGS} \
        ${DIST_OPT_ARGS} \
        --distributed-timeout-minutes ${DIST_TIMEOUT_MINUTES:-120} \
        ${MEMORY_ARGS} ${CMD_SUFFIX}
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
    if [ "${LUMEN_ROCPROF:-0}" = "1" ]; then
        # Profile each rank through a wrapper that runs rocprofv3 on rank 0 only.
        # --no-python makes torchrun exec the wrapper directly with the full cmd.
        CMD+=" --no-python ${SCRIPT_DIR}/scripts/rocprof_wrap.sh python ${SCRIPT_DIR}/finetune_llama2.py"
    else
        CMD+=" ${SCRIPT_DIR}/finetune_llama2.py"
    fi
    CMD+=" --backend fsdp"
    CMD+=" --model-name-or-path ${MODEL}"
    CMD+=" --tokenizer-name-or-path ${TOKENIZER}"
    CMD+=" --seq-length ${SEQ_LEN}"
    CMD+=" --micro-batch-size ${MBS}"
    CMD+=" --gradient-accumulation-steps ${GRAD_ACCUM}"
    CMD+=" --max-steps ${TRAIN_STEPS}"
    CMD+=" --lr ${LR}"
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
    CMD+=" --train-samples ${TRAIN_SAMPLES}"
    CMD+=" --val-samples ${VAL_SAMPLES}"
    CMD+=" --sharding-strategy ${SHARDING}"
    [ -n "${FSDP_VERSION:-}" ] && CMD+=" --fsdp-version ${FSDP_VERSION}"
    [ "${NO_GRADIENT_CHECKPOINTING:-0}" = "1" ] && CMD+=" --no-gradient-checkpointing"
    # FSDP2-only: store the LoRA-frozen blockwise2d base weight as FP8 and
    # all-gather it as FP8 (no per-step re-quant).
    [ "${FSDP_FP8_PARAM_STORAGE:-0}" = "1" ] && CMD+=" --fsdp-fp8-param-storage"

    [ -n "${VALID_DATA}" ] && CMD+=" --val-data-path ${VALID_DATA}"

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

    [ "${LUMEN_NORM:-0}" -eq 1 ] && CMD+=" --lumen-norm"

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
    echo "LLaMA2 SFT — FSDP backend"
    echo "  Config:     ${CONFIG}"
    echo "  Model:      ${MODEL}"
    echo "  GPUs:       ${NGPU}"
    echo "  Batch:      MBS=${MBS} x accum=${GRAD_ACCUM} | seq_len=${SEQ_LEN}"
    echo "  Sharding:   ${SHARDING}"
    echo "  LoRA:       rank=${LORA_RANK}"
    echo "  FP8:        training=${FP8_TRAINING} format=${FP8_FORMAT}"
    echo "  SDMA:       use=${USE_SDMA} mori=${MORI_ENABLE_SDMA}"
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
echo "STARTING LLAMA2 FINETUNE AT ${start_fmt} (backend=${BACKEND})"

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
echo "ENDING LLAMA2 FINETUNE AT ${end_fmt}"
result=$(( end - start ))
echo "RESULT,LLM_FINETUNING,,${result},AMD,${start_fmt}"

exit 0
