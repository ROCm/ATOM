#!/bin/bash
# One 10-step Wan2.1-T2V-1.3B full-parameter SFT run on 8 GPUs.
#
#   run_wan_10steps.sh baseline    [VeOmni CLI overrides...]  # no Lumen patch
#   run_wan_10steps.sh flydsl      [...]   # vae_bf16,vae_conv_video (T>1 included)
#   run_wan_10steps.sh bf16-only   [...]   # vae_bf16, convolutions still torch
#   run_wan_10steps.sh custom      [...]   # LUMEN_PATCH as exported
#
# The video counterpart of run_10steps.sh. Same discipline: one entry point, one
# config, one dataset, one seed, one parallelism, one interpreter environment;
# only LUMEN_PATCH and the overrides after the mode differ. Overrides are
# appended after this script's own arguments and VeOmni keeps the last
# occurrence of a flag, so --train.training_task or --data.train_path given here
# wins over the ones below. RUN_NAME replaces the run label.
#
# Two things this is for that the image run is not. The clip makes the VAE a real
# share of the step rather than 0.07% of it, and it exercises the T>1 path, where
# the T=1 causal identity does not apply and the kernel runs under the module's
# own padding instead.
#
# RUN_SUFFIX names a repeat, e.g. RUN_SUFFIX=rep1. Use it. A single A/B cannot
# resolve this: repeats of one mode spread about +-6% on this node, which is
# wider than most of what is being measured.
#
# Writes:
#   $LOG_DIR/wan-<mode>[-<suffix>].log       the training log
#   $LOG_DIR/wan-<mode>[-<suffix>].meta.txt  command, commit, exit code, wall clock
#   $OUT_DIR/wandb-wan-<mode>[-<suffix>]/    offline wandb, full-precision metrics
set -o pipefail
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MODE="${1:?usage: run_wan_10steps.sh <baseline|flydsl|bf16-only|custom> [VeOmni CLI overrides...]}"
shift
case "$MODE" in
    baseline)  PATCH="" ;;
    # vae_bf16 is not decoration. VeOmni loads this VAE in FP32
    # (modeling_wan_condition.py) and the FlyDSL kernel is BF16-only, so without
    # the cast Lumen demotes every call to torch and no FlyDSL code runs. On the
    # video path there is no conv3d->conv2d rewrite to fall back on either, so
    # FP32 + vae_conv_video is a measured no-op.
    flydsl)    PATCH="vae_bf16,vae_conv_video" ;;
    bf16-only) PATCH="vae_bf16" ;;
    custom)    PATCH="${LUMEN_PATCH:-}" ;;
    *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

CONFIG="${CONFIG:-$EXAMPLE_DIR/wan_video.yaml}"
STEPS="${STEPS:-10}"
NPROC="${NPROC_PER_NODE:-8}"

# WAN_TASK=offline_embedding runs VeOmni's own pre-embedding task instead of
# training: it builds no DiT, no optimizer and does no backward
# (dit_trainer.py:290 and :236), so a step is the condition model's forward and
# nothing else. That is the workload this kernel is actually for, and it makes
# the VAE's share of a step large instead of marginal. Same eight ranks, same
# clips, same modes.
TASK="${WAN_TASK:-online_training}"
case "$TASK" in
    online_training)   RUN="wan-$MODE" ;;
    offline_embedding) RUN="wanemb-$MODE" ;;
    *) echo "unknown WAN_TASK: $TASK (online_training|offline_embedding)" >&2; exit 2 ;;
esac
RUN="${RUN_NAME:-$RUN}${RUN_SUFFIX:+-$RUN_SUFFIX}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

# Refuse to start if another run is going: two concurrent 8-GPU jobs share the
# GPUs and $VEOMNI_DIR/log.txt, which corrupts both timings and logs without
# failing either run.
LOCK="$LOG_DIR/.run.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "FATAL: a run is already going as pid $(cat "$LOCK")." >&2
    echo "       Wait for it, or remove $LOCK if that pid is stale." >&2
    exit 3
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

# S optimizer steps on D ranks needs >= S*D rows, and the DiT path forces
# dyn_bsz=False, so each rank takes floor(N/dp_size) batches and the trainer
# leaves the epoch when they run out -- silently, exit code 0.
NEED=$((STEPS * NPROC))
ROWS=$(python3 - "$WAN_DATA_DIR" <<'PY'
import glob
import sys

import pyarrow.parquet as pq

print(sum(pq.read_metadata(f).num_rows for f in glob.glob(f"{sys.argv[1]}/*.parquet")))
PY
)
if [ "${ROWS:-0}" -lt "$NEED" ]; then
    echo "FATAL: $WAN_DATA_DIR has ${ROWS:-0} rows, need >= $NEED for $STEPS steps on $NPROC ranks" >&2
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

# The tqdm postfix rounds to 2 decimals and quantises time to the second.
# Offline wandb records full precision and a per-step timestamp, and needs no
# network or key. summarize.py reads these directories.
export WANDB_MODE=offline
export WANDB_DIR="$OUT_DIR/wandb-$RUN"
export WANDB_SILENT=true
mkdir -p "$WANDB_DIR"

# Held identical across modes on purpose, including for the baseline: the sidecar
# flydsl shadows the image's, which makes aiter unimportable, and diffusers
# probes aiter when choosing an attention backend.
export PYTHONPATH="$LUMEN_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "$PATCH" ]; then
    export LUMEN_PATCH="$PATCH"
else
    unset LUMEN_PATCH
fi

ENTRY="$EXAMPLE_DIR/train_dit_lumen.py"
META="$LOG_DIR/$RUN.meta.txt"

{
    echo "=== run: $RUN ==="
    echo "started      : $(date -Is)"
    echo "VeOmni       : $VEOMNI_DIR @ $(git -C "$VEOMNI_DIR" rev-parse --short HEAD 2>/dev/null)"
    echo "VeOmni dirty : $(git -C "$VEOMNI_DIR" status --short 2>/dev/null | wc -l) files"
    echo "Lumen        : $LUMEN_DIR @ $(git -C "$LUMEN_DIR" -c safe.directory='*' rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$LUMEN_DIR" -c safe.directory='*' rev-parse --short HEAD 2>/dev/null)"
    echo "Lumen dirty  : $(git -C "$LUMEN_DIR" -c safe.directory='*' status --short 2>/dev/null | wc -l) files"
    echo "config       : $CONFIG"
    echo "task         : $TASK"
    echo "model        : $WAN_DIR"
    echo "data         : $WAN_DATA_DIR ($ROWS rows)"
    echo "max_steps    : $STEPS   nproc: $NPROC"
    echo "LUMEN_TIME_VAE: ${LUMEN_TIME_VAE:-<unset>}"
    echo "LUMEN_PATCH  : ${LUMEN_PATCH:-<unset, baseline>}"
    echo "PYTHONPATH   : $PYTHONPATH"
    echo "extra args   : $*"
} | tee "$META"

cd "$VEOMNI_DIR" || exit 1
rm -f "$VEOMNI_DIR/log.txt"
START=$(date +%s)
bash train.sh "$ENTRY" "$CONFIG" \
    --model.model_path "$WAN_DIR/transformer" \
    --model.condition_model_path "$WAN_DIR" \
    --data.train_path "$WAN_DATA_DIR" \
    --data.offline_embedding_save_dir "$OUT_DIR/embed-$RUN" \
    --train.training_task "$TASK" \
    --train.max_steps "$STEPS" \
    --train.num_train_epochs 1 \
    --train.wandb.enable true \
    --train.wandb.project Wan2.1-FlyDSL \
    --train.wandb.name "$RUN" \
    --train.checkpoint.output_dir "$OUT_DIR/ckpt-$RUN" \
    "$@"
rc=$?
END=$(date +%s)

# train.sh writes log.txt in the VeOmni root and overwrites it every run, so it
# has to be copied out before the next mode starts.
cp -f "$VEOMNI_DIR/log.txt" "$LOG_DIR/$RUN.log" 2>/dev/null
{
    echo "finished     : $(date -Is)"
    echo "EXIT_CODE    : $rc"
    echo "WALL_SECONDS : $((END - START))"
    echo "log          : $LOG_DIR/$RUN.log"
} | tee -a "$META"

if [ "$rc" -eq 0 ]; then
    # A patched run that demoted to torch looks exactly like a successful one,
    # so the backend line is the only thing that separates them.
    echo
    echo "--- what the patch reported (grep '[lumen]' in the log for all of it) ---"
    grep -a "\[lumen\]" "$LOG_DIR/$RUN.log" | head -12
fi
exit $rc
