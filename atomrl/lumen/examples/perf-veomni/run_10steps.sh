#!/bin/bash
# One 10-step Qwen-Image DiT SFT run on 8 GPUs.
#
#   run_10steps.sh baseline    [VeOmni CLI overrides...]   # no Lumen patch
#   run_10steps.sh flydsl      [...]                       # vae_bf16,vae_conv
#   run_10steps.sh bf16-only   [...]                       # vae_bf16
#   run_10steps.sh custom      [...]                       # LUMEN_PATCH as exported
#
# Every mode uses the same entry point, config, dataset, seed, parallelism and
# interpreter environment; what differs is LUMEN_PATCH and whatever overrides
# follow the mode. Overrides are appended after this script's own arguments, and
# VeOmni takes the last occurrence of a flag, so e.g. --data.train_path given
# here wins. That is how ladder.sh expresses the configuration rungs
# (--train.empty_cache_steps 500 ...) without editing qwen_image_1024.yaml.
#
# RES picks the resolution and defaults to 1024, which is what the official
# configs/dit/qwen_image_sft.yaml trains at; RES=256 uses the smoke config.
# RUN_NAME replaces the run label (default <mode>-<res>); RUN_SUFFIX appends a
# repeat tag, e.g. RUN_SUFFIX=r1. Writes:
#   $LOG_DIR/<run>.log        the training log
#   $LOG_DIR/<run>.meta.txt   command, commit, exit code, wall clock
#   $OUT_DIR/wandb-<run>/     offline wandb, full-precision per-step metrics
set -o pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MODE="${1:?usage: run_10steps.sh <baseline|flydsl|bf16-only|custom> [VeOmni CLI overrides...]   (RES=1024|256)}"
shift
case "$MODE" in
    baseline)  PATCH="" ;;
    # vae_bf16 is not decoration. VeOmni loads the VAE in FP32 and the FlyDSL
    # kernel is BF16-only, so without the cast Lumen demotes every call to torch
    # and no FlyDSL code runs. See README "Why the FlyDSL run also casts to BF16".
    flydsl)    PATCH="vae_bf16,vae_conv" ;;
    bf16-only) PATCH="vae_bf16" ;;
    custom)    PATCH="${LUMEN_PATCH:-}" ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

RES="${RES:-1024}"
case "$RES" in
    1024) CONFIG="$EXAMPLE_DIR/qwen_image_1024.yaml" ;;
    256)  CONFIG="$EXAMPLE_DIR/qwen_image_smoke.yaml" ;;
    *) echo "unknown RES: $RES (expected 1024 or 256)" >&2; exit 2 ;;
esac
RUN="${RUN_NAME:-$MODE-$RES}${RUN_SUFFIX:+-$RUN_SUFFIX}"

STEPS="${STEPS:-10}"
NPROC="${NPROC_PER_NODE:-8}"
JSONL="$DATA_DIR/train_$((STEPS * NPROC)).jsonl"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# Refuse to start if another run of this script is already going. Two concurrent
# 8-GPU runs share the same GPUs and the same $VEOMNI_DIR/log.txt, which corrupts
# both the timings and the logs without failing either run -- it just looks like
# a slow, confusing result. Caught this the hard way while validating the
# runbook: a doubled launch put the baseline at 6.5 s/step instead of 3.9.
LOCK="$LOG_DIR/.run.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "FATAL: a run is already going as pid $(cat "$LOCK")." >&2
    echo "       Wait for it, or remove $LOCK if that pid is stale." >&2
    exit 3
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

# S optimizer steps on D ranks needs >= S*D records. The DiT path forces
# dyn_bsz=False, so each rank gets floor(N/dp_size) batches and the trainer
# breaks out of the epoch when they run out -- silently, without an error.
if [ ! -f "$JSONL" ]; then
    echo "generating $((STEPS * NPROC)) records -> $JSONL"
    python3 "$EXAMPLE_DIR/make_data.py" $((STEPS * NPROC)) "$JSONL" || exit 1
fi
HAVE=$(wc -l < "$JSONL")
if [ "$HAVE" -lt $((STEPS * NPROC)) ]; then
    echo "FATAL: $JSONL has $HAVE records, need >= $((STEPS * NPROC))" >&2
    exit 2
fi

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NPROC_PER_NODE="$NPROC"
export PYTORCH_ROCM_ARCH=gfx950
export MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK=1
export FLA_TILELANG=0
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1

# The tqdm postfix rounds to 2 decimals, useless for a loss of order 1e-2.
# Offline wandb records full precision and needs no network or key.
export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb-$RUN"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"

# Held identical across modes on purpose, including for the baseline: the
# sidecar flydsl shadows the image's, which makes aiter unimportable, and
# diffusers probes aiter when choosing an attention backend. Keeping the
# environment fixed keeps that out of the comparison.
export PYTHONPATH="$LUMEN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "$PATCH" ]; then
    export LUMEN_PATCH="$PATCH"
else
    unset LUMEN_PATCH
fi

TASK="$EXAMPLE_DIR/train_dit_lumen.py"
META="$LOG_DIR/$RUN.meta.txt"

{
    echo "=== run: $RUN ==="
    echo "started      : $(date -Is)"
    echo "VeOmni       : $VEOMNI_DIR @ $(git -C "$VEOMNI_DIR" rev-parse --short HEAD 2>/dev/null)"
    echo "VeOmni dirty : $(git -C "$VEOMNI_DIR" status --short 2>/dev/null | wc -l) files"
    echo "Lumen        : $LUMEN_DIR @ $(git -C "$LUMEN_DIR" -c safe.directory='*' rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$LUMEN_DIR" -c safe.directory='*' rev-parse --short HEAD 2>/dev/null)"
    echo "Lumen dirty  : $(git -C "$LUMEN_DIR" -c safe.directory='*' status --short 2>/dev/null | wc -l) files"
    echo "config       : $CONFIG   (RES=$RES)"
    echo "train_jsonl  : $JSONL ($HAVE records)"
    echo "max_steps    : $STEPS   nproc: $NPROC"
    echo "LUMEN_PATCH  : ${LUMEN_PATCH:-<unset, baseline>}"
    echo "PYTHONPATH   : $PYTHONPATH"
    echo "extra args   : $*"
} | tee "$META"

cd "$VEOMNI_DIR" || exit 1
rm -f "$VEOMNI_DIR/log.txt"
START=$(date +%s)
bash train.sh "$TASK" "$CONFIG" \
    --model.model_path "$QWEN_IMAGE_DIR/transformer" \
    --model.condition_model_path "$QWEN_IMAGE_DIR" \
    --data.train_path "$JSONL" \
    --data.train_sample "$HAVE" \
    --train.max_steps "$STEPS" \
    --train.num_train_epochs 1 \
    --train.wandb.enable true \
    --train.wandb.project Qwen-Image-FlyDSL \
    --train.wandb.name "$RUN" \
    --train.checkpoint.output_dir "$OUT_DIR/ckpt-$RUN" \
    "$@"
rc=$?
END=$(date +%s)

# train.sh writes its log to log.txt in the VeOmni root and overwrites it every
# run, so it has to be copied out before the next mode starts.
cp -f "$VEOMNI_DIR/log.txt" "$LOG_DIR/$RUN.log" 2>/dev/null
{
    echo "finished     : $(date -Is)"
    echo "EXIT_CODE    : $rc"
    echo "WALL_SECONDS : $((END - START))"
    echo "log          : $LOG_DIR/$RUN.log"
    echo "wandb        : $WANDB_DIR"
} | tee -a "$META"

if [ "$rc" -eq 0 ] && [ -n "$LUMEN_PATCH" ]; then
    echo
    echo "--- what the patch reported (grep '[lumen]' in the log for all of it) ---"
    grep -a "\[lumen\]" "$LOG_DIR/$RUN.log" | head -12
fi
exit $rc
