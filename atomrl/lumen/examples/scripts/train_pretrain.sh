#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
# Megatron pretrain body for the precision-comparison benchmark, shared by
# every model and by the docker and native launchers.
#
# One body on purpose. The point of the benchmark is to compare BF16, FP8
# delayed and MXFP4 at ms/step, so the only thing allowed to differ between
# arms is the quantization recipe — anything else that varies (a second mock
# corpus, a different eval cadence, a launcher that forgets an env switch)
# lands directly in the number being reported.
#
# Everything is read from the environment so the launchers differ only in how
# they set up paths. Required:
#
#   LUMEN_ROOT      Lumen checkout
#   MEGATRON_ROOT   Megatron-LM checkout (needed on PYTHONPATH and for patching)
#   RESULTS_ROOT    writable dir for logs and mock data
#   TOKENIZER_PATH  HuggingFace tokenizer directory
#
# Optional: MODEL PRECISION NUM_LAYERS TAIL_BF16 MBS GBS SEQ_LEN TRAIN_STEPS
#           SEED NPROC TRAIN_JSONL VALID_JSONL LOG_TAG EVAL_INTERVAL EVAL_ITERS
#           SPLIT DRY_RUN WANDB_PROJECT WANDB_NAME WANDB_ENTITY
#
# DRY_RUN=1 prints the torchrun command and exits without touching a GPU. Use it
# to diff two arms' command lines before spending hours measuring them.
###############################################################################
set -euo pipefail

: "${LUMEN_ROOT:?}"
: "${MEGATRON_ROOT:?}"
: "${RESULTS_ROOT:?}"
: "${TOKENIZER_PATH:?}"

MODEL="${MODEL:-qwen3_8b}"
PRECISION="${PRECISION:-mxfp4}"        # bf16 | fp8 | mxfp4

# Per-model defaults, in one table so the benchmark matrix is readable at a
# glance. MBS x SEQ_LEN is 16384 for all three comparison arms.
#
# VOCAB_MAX only bounds the random token ids in the mock corpus; it tracks each
# tokenizer's vocabulary so no id lands outside the embedding table.
case "${MODEL}" in
    llama2_7b)
        MODEL_DIR="llama2"
        ARGS_FILE="${LUMEN_ROOT}/examples/llama2/scripts/llama2_7b_common_args.sh"
        : "${NUM_LAYERS:=32}" "${MBS:=4}" "${GBS:=256}" "${SEQ_LEN:=4096}"
        : "${VOCAB_MAX:=31999}"
        ;;
    llama31_8b)
        MODEL_DIR="llama31"
        ARGS_FILE="${LUMEN_ROOT}/examples/llama31/scripts/llama31_8b_common_args.sh"
        : "${NUM_LAYERS:=32}" "${MBS:=2}" "${GBS:=128}" "${SEQ_LEN:=8192}"
        : "${VOCAB_MAX:=127999}"
        ;;
    qwen3_8b)
        MODEL_DIR="qwen3"
        ARGS_FILE="${LUMEN_ROOT}/examples/qwen3/scripts/qwen3_8b_common_args.sh"
        : "${NUM_LAYERS:=36}" "${MBS:=2}" "${GBS:=128}" "${SEQ_LEN:=8192}"
        : "${VOCAB_MAX:=151999}"
        ;;
    *)
        echo "ERROR: unknown MODEL='${MODEL}' (llama2_7b | llama31_8b | qwen3_8b)"
        exit 1
        ;;
esac

# Last ~15% of layers stay BF16 under MXFP4: both FP4 papers find the tail
# layers are the sensitive ones, and 8B diverged around step 1300 without this
# (docs/mxfp4_training_report.md §1.5, §6.3). 5 of 32 and 5 of 36 are both that
# fraction, so one default covers all three models.
TAIL_BF16="${TAIL_BF16:-5}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
SEED="${SEED:-1234}"
NPROC="${NPROC:-8}"

# Evaluate ~10 times over the run instead of once at the end: a single end-of-run
# eval gives wandb one data point per val metric, which it can only render as a bar,
# and one point cannot show where the curves separate.
EVAL_INTERVAL="${EVAL_INTERVAL:-$(( TRAIN_STEPS / 10 > 0 ? TRAIN_STEPS / 10 : 1 ))}"
# Each eval pass consumes EVAL_ITERS × GBS fresh sequences from VALID_JSONL, and
# the validation dataset does not wrap, so the whole run dies with a StopIteration
# mid-training once the evals have drained the file. The budget is
# (TRAIN_STEPS / EVAL_INTERVAL + 1) × EVAL_ITERS × GBS sequences of SEQ_LEN + 1
# tokens; size VALID_JSONL above that before raising either knob. Keep it small
# anyway: eval time is pure overhead, and the eval batch must stay identical
# across runs for the comparison to be paired.
EVAL_ITERS="${EVAL_ITERS:-2}"
# Kept only because Megatron requires it; PretrainTextDataset ignores it.
SPLIT="${SPLIT:-98,1,1}"

DATA_DIR="${RESULTS_ROOT}/mock_data"
# Point TRAIN_JSONL at a real corpus (one {"text": ...} object per line) to skip
# the mock generator below. LOG_TAG then keeps the two logs apart.
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/mock_train_s${SEQ_LEN}_g${GBS}_t${TRAIN_STEPS}.jsonl}"
# PretrainTextDataset builds each of train/valid/test from its own path starting at
# chunk 0 and ignores --split entirely, so leaving VALID_JSONL at the train file
# makes "validation loss" a measurement on data the model also trains on. Point it
# at a disjoint file to get a real held-out number.
VALID_JSONL="${VALID_JSONL:-${TRAIN_JSONL}}"
LOG_FILE="${RESULTS_ROOT}/lumen_${MODEL}${LOG_TAG:+_${LOG_TAG}}_${PRECISION}.log"
mkdir -p "${DATA_DIR}"

# Megatron builds its wandb writer only when --wandb-project is non-empty, and it
# writes from the last rank rather than rank 0. Without WANDB_PROJECT the metrics
# exist only in LOG_FILE; examples/qwen3/scripts/wandb_backfill_megatron_log.py
# can replay one of those logs after the fact.
#
# --tensorboard-dir is not optional here. Every wandb_writer.log() call in
# Megatron's training_log() sits inside `if writer and ...`, where `writer` is
# the *TensorBoard* writer, which exists only when --tensorboard-dir is set. Ask
# for wandb without it and the run appears in the project, connects, and uploads
# nothing but a runtime -- an empty workspace with no way to tell it apart from a
# run that never started.
WANDB_ARGS=()
if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME:-megatron-${PRECISION}-${MODEL}${LOG_TAG:+-${LOG_TAG}}-${TRAIN_STEPS}}"
        --wandb-save-dir "${RESULTS_ROOT}/wandb"
        --tensorboard-dir "${RESULTS_ROOT}/tensorboard/${WANDB_NAME:-${PRECISION}${LOG_TAG:+-${LOG_TAG}}}"
    )
    if [ -n "${WANDB_ENTITY:-}" ]; then
        WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
    fi
fi

export PYTHONPATH="${MEGATRON_ROOT}:${LUMEN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import megatron" || { echo "ERROR: megatron not importable"; exit 1; }

# Lumen expects the verified official AITER checkout used by the image. It must
# be installed rather than overlaid through PYTHONPATH so its compiled
# extensions remain available for hipBLASLt and MXFP4 GEMMs.
# `|| true` so a failed import reaches the message below instead of tripping
# `set -e` on the command substitution's exit status.
AITER_DIR="$(python -c 'import aiter,os;print(os.path.dirname(aiter.__file__))' 2>/dev/null || true)"
[ -n "${AITER_DIR}" ] || { echo "ERROR: aiter not importable"; exit 1; }

# Missing these surfaces as a ModuleNotFoundError from deep inside the
# atomrl.lumen.models.megatron import chain.
AITER_REQUIRED=(
    ops/triton/cross_entropy.py
    ops/triton/_triton_kernels/cross_entropy.py
)
# Missing these still trains, which is exactly the problem. fast_transpose_fp8
# falls back to .t().contiguous() and warns per call: the first FP8 baseline ran
# that way and paid 166 ms/step plus 709k warnings into a 78 MB log, with nothing
# in the run to say so. The quant kernels are reached only by the blockwise and
# mxfp8 scaling types.
AITER_OPTIONAL=(
    ops/triton/quant/fast_transpose.py
    ops/triton/_triton_kernels/quant/fast_transpose.py
    ops/triton/_triton_kernels/quant/quant_fp8_blockwise.py
    ops/triton/_triton_kernels/quant/quant_mxfp8.py
)
aiter_missing() {
    local rel
    for rel in "$@"; do
        [ -f "${AITER_DIR}/${rel}" ] || echo "${rel}"
    done
}
aiter_copy_hint() {
    echo "  reinstall ROCm/aiter at adbb6ac8b84110134e0cf4dc513b250d2cc0a75c or newer"
}

mapfile -t MISSING_REQUIRED < <(aiter_missing "${AITER_REQUIRED[@]}")
if [ ${#MISSING_REQUIRED[@]} -gt 0 ]; then
    echo "ERROR: the installed aiter (${AITER_DIR}) is missing required kernels:"
    aiter_copy_hint "${MISSING_REQUIRED[@]}"
    exit 1
fi

mapfile -t MISSING_OPTIONAL < <(aiter_missing "${AITER_OPTIONAL[@]}")
if [ ${#MISSING_OPTIONAL[@]} -gt 0 ]; then
    echo "[setup] WARNING: the installed aiter (${AITER_DIR}) is missing optional kernels."
    echo "[setup] Training will run, but FP8 transposes fall back to .t().contiguous()"
    echo "[setup] and any step time measured here is not comparable to a complete run:"
    aiter_copy_hint "${MISSING_OPTIONAL[@]}"
fi

if ! python -c "from aiter.ops.triton.activation import swiglu_fwd, swiglu_bwd" 2>/dev/null; then
    echo "[setup] WARNING: the installed aiter (${AITER_DIR}) has no swiglu kernels."
    echo "[setup] LUMEN_FUSED_SWIGLU will use the unfused fallback."
fi

# The RMSNorm layer-spec patch rewrites Megatron to use apex's FusedRMSNorm, so
# it only applies where apex exists. Without apex Megatron already falls back to
# WrappedTorchNorm, and applying the patch would inject an unsatisfiable
# `import apex` into the RMSNorm path.
if python -c "import apex" 2>/dev/null; then
    python "${LUMEN_ROOT}/examples/llama2/scripts/patch_gpt_layer_specs.py" "${MEGATRON_ROOT}"
    ROPE_ARGS=()
else
    echo "[setup] apex not installed: skipping FusedRMSNorm patch, disabling rope fusion"
    # Megatron defaults apply_rope_fusion=True and asserts on a fused kernel that
    # only TE or Lumen's apex bridge can supply. Note this flag does not reach the
    # Lumen backend, which overwrites apply_rope_fusion from --lumen-fused-rope.
    ROPE_ARGS=(--no-rope-fusion)
fi

# The Lumen backend ignores both branches above -- it sets apply_rope_fusion from
# its own flag, which routes to AITER's fused kernel instead of apex's or TE's.
# Without it the rotate-half runs as separate neg/cat/mul kernels: 137 ms/step at
# this shape, measured (§5.7).
ROPE_ARGS+=(--lumen-fused-rope)

# Mock corpus of random token ids. Repetitive enough that the model memorises it
# within ~50 steps, which is what makes the loss curve usable as a regression
# signal. Generation is the same as the BF16/FP8 reference runs so the curves
# stay comparable; it is skipped when the file is already there.
if [ ! -s "${TRAIN_JSONL}" ]; then
    SEQ_LEN="${SEQ_LEN}" GBS="${GBS}" TRAIN_STEPS="${TRAIN_STEPS}" SEED="${SEED}" \
    VOCAB_MAX="${VOCAB_MAX}" TRAIN_JSONL="${TRAIN_JSONL}" python - <<'PYEOF'
import json
import os
import random

seq = int(os.environ["SEQ_LEN"])
gbs = int(os.environ["GBS"])
steps = int(os.environ["TRAIN_STEPS"])
vocab_max = int(os.environ["VOCAB_MAX"])
need_chunks = gbs * (steps + 5)
need_tokens = int(need_chunks * (seq + 1) * 1.2)
random.seed(int(os.environ["SEED"]))
path = os.environ["TRAIN_JSONL"]
words_per_doc = 4000
docs = need_tokens // words_per_doc + 1
with open(path, "w") as f:
    for _ in range(docs):
        toks = [str(random.randint(1, vocab_max)) for _ in range(words_per_doc)]
        f.write(json.dumps({"text": " ".join(toks)}) + "\n")
print(f"[mock-data] wrote {docs} docs to {path}")
PYEOF
else
    echo "[data] using existing ${TRAIN_JSONL}"
fi

# ---- quantization flags -----------------------------------------------------
# MXFP4 has its own --linear-fp4 enable switch. The last TAIL_BF16 layers stay BF16: both
# FP4 papers find the tail layers are the sensitive ones, and 8B diverged around
# step 1300 without this (docs/mxfp4_training_report.md §1.5, §6.3).
#
# --lumen-linear swaps Megatron's ColumnParallelLinear / RowParallelLinear for the
# Lumen native ones. It was banned here until 8/10 because
# enable_fp8_for_parallel_linear configured them from the --linear-fp8-scaling
# string ("blockwise") rather than the resolved recipe, so the run silently
# executed FP8 blockwise. With that fixed, the 200-step A/B says the native path
# is worth having: 1429.0 vs 1513.4 ms/step (-5.6%) and 0.6414 vs 0.6556 mem, with
# the loss curve inside the run-to-run noise floor (two identical baselines differ
# by 0.25% mean, the native arm by 0.37%). Numbers from before this date were
# measured on the patched-Megatron path and are 5.6% slower for that reason alone.
#
# The FP8 arm reproduces the recipe the BF16/FP8 reference report measured:
# delayed scaling, amax history 1024, algo max. FP8_FORMAT is overridable because
# `hybrid` is what that report used but it dies in the first backward on this
# stack (mixed-dtype hipb_mm, .claude/tmp-training-bugs.md), so a run here has to
# be able to fall back to fp8_e4m3 and say so.
QUANT_ARGS=()
case "${PRECISION}" in
    mxfp4)
        QUANT_ARGS=(
            --linear-fp4
            --lumen-linear
            --first-last-layers-bf16
            --num-layers-at-start-in-bf16 0
            --num-layers-at-end-in-bf16 "${TAIL_BF16}"
        )
        ;;
    fp8)
        QUANT_ARGS=(
            --linear-fp8
            --linear-fp8-format "${FP8_FORMAT:-hybrid}"
            --linear-fp8-scaling delayed
            --linear-fp8-amax-algo max
            --linear-fp8-amax-history 1024
        )
        # Off by default so the FP8 arm stays the recipe the reference report
        # measured, which predates the native path. FP8_LUMEN_LINEAR=1 exists to
        # answer whether FP8's missing speedup on this machine is the recipe or the
        # integration: MXFP4 gets the native linear and FP8 does not, so the two
        # arms differ by more than their number format until this is set.
        [ "${FP8_LUMEN_LINEAR:-0}" = "1" ] && QUANT_ARGS+=(--lumen-linear)
        # The FP8 fusion switches the reference report's §7 credits for taking
        # Llama2-7B from ~10.8s to 8.19s. They are FP8-only: the MXFP4 path has
        # its own fused quantizer and the BF16 arm reaches none of this code.
        export LUMEN_PREFER_HIPBLASLT=1
        export LUMEN_FUSED_QUANT_TRANSPOSE_CPP=1
        export LUMEN_FUSED_QUANT_AMAX=1
        export LUMEN_FUSED_QUANT_SCALE=1
        export LUMEN_FUSED_CAST_TRANSPOSE=1
        export LUMEN_FUSED_CAST_TRANSPOSE_V2=1
        export LUMEN_FUSED_SWIGLU_QUANT=1
        export LUMEN_FUSED_NORM_QUANT=1
        export LUMEN_FUSED_NORM_QUANT_V2=1
        export LUMEN_TRANSPOSE_CACHE=1
        export LUMEN_FAST_QUANT_DISPATCH=1
        export LUMEN_WEIGHT_QUANT_ONCE=1
        ;;
    bf16)
        ;;
    *)
        echo "ERROR: unknown PRECISION='${PRECISION}' (bf16 | fp8 | mxfp4)"
        exit 1
        ;;
esac

source "${ARGS_FILE}"

# Word-split on purpose: EXTRA_ARGS carries whole flags, e.g. profiler options.
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

cd "${LUMEN_ROOT}/examples/llama31"

TORCHRUN_ARGS=(
    --nproc_per_node="${NPROC}" --nnodes=1 pretrain_llama31.py
    --backend megatron
    "${COMMON_ARGS[@]}"
    --lumen-attn-backend csrc
    "${ROPE_ARGS[@]}"
    "${QUANT_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${EXTRA_ARGS_ARR[@]}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%s\n' "torchrun ${TORCHRUN_ARGS[*]}"
    exit 0
fi

set -x
torchrun "${TORCHRUN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
