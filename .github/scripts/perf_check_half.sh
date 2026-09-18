#!/usr/bin/env bash
# Run one half of the PR perf check pairing.
#
#   perf_check_half.sh <commit-sha> <warmup|base|head|base2>
#
# The container is already running and has the workspace bind-mounted at
# /workspace, so checking out a commit on the host is immediately visible
# inside it. That is the whole reason both halves can share one container: the
# pairing then holds machine, image and container instance fixed, and the only
# thing that differs between the two measurements is the commit.
#
# Everything heavy is delegated to atom_test.sh, the same entry point the
# nightly benchmark uses. Reimplementing launch/benchmark here would fork the
# definition of "run a benchmark" and drift silently the first time upstream
# tunes it.
#
# Expects from the caller's environment: CONTAINER, MODEL_PATH, ARGS,
# RESULT_FILENAME, plus whatever atom_test.sh itself reads. DUMMY_WEIGHT_ARGS
# is optional and appended to ARGS for every phase (see below).

set -euo pipefail

COMMIT="${1:?commit sha required}"
HALF="${2:?half must be one of warmup|base|head|base2}"

# warmup  discarded; exists only to fill the JIT/autotune caches that persist
#         inside the shared container. Without it the first measured half pays
#         for populating them and the second reads 6-9% faster on identical
#         code -- larger than the regression threshold, and in the direction
#         that hides regressions rather than inventing them.
# base2   a second measurement of the base commit, after head. Its distance
#         from the first is the drift the pairing accumulated; without it a
#         head-vs-base delta cannot be told apart from time passing.
case "$HALF" in
  warmup|base|head|base2) ;;
  *) echo "ERROR: half must be warmup|base|head|base2, got '$HALF'" >&2; exit 2 ;;
esac

: "${CONTAINER:?CONTAINER must be set}"
: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set}"
# atom_test.sh reads CONC for every phase; the warmup additionally derives its
# shortened prompt count from it. Checked here so a missing value fails with a
# name rather than "unbound variable" from wherever it is first dereferenced.
: "${CONC:?CONC must be set}"
: "${ISL:?ISL must be set}"
: "${OSL:?OSL must be set}"
: "${RANDOM_RANGE_RATIO:?RANDOM_RANGE_RATIO must be set}"

OUT_DIR="perf-pair/${HALF}"
mkdir -p "$OUT_DIR"

if [ "$HALF" = "warmup" ]; then
  # Same prompt count as the measurement. A tenth of it was tried first, on
  # the theory that the caches are filled by compiling and tuning kernels
  # rather than by request volume. That holds on MI308 (1.35% residual) and
  # does not on MI355X, where run 34233036220 still showed +4.19/+5.65/+3.10%
  # at c=64/128/256 across three machines. Launch and model load dominate a
  # phase (~10.4 of 17.6 min at c=64), so matching the measurement costs about
  # 14% of job wall clock rather than a whole extra phase. Under dummy weights
  # that split moves the other way -- load falls out and the benchmark is what
  # is left -- so a full-length warmup costs proportionally more of the job.
  export NUM_PROMPTS_OVERRIDE="${WARMUP_PROMPTS:-$(( CONC * ${WARMUP_MULT:-10} ))}"
  echo "warmup: NUM_PROMPTS_OVERRIDE=${NUM_PROMPTS_OVERRIDE} (results discarded)"
fi

echo "========== ${HALF}: reclaiming workspace ownership =========="
# The container runs as root against a bind-mounted workspace, so anything it
# writes (__pycache__, build output) ends up owned by root. The checkout and
# clean below run as the runner user and fail with EACCES on those files, as
# does the next job's actions/checkout. Hand them back before touching the
# tree; the container is already up, so this costs nothing.
docker exec "$CONTAINER" bash -lc \
  "chown -R $(id -u):$(id -g) /workspace" || true

echo "========== ${HALF}: checking out ${COMMIT} =========="
# Discard whatever the previous half left behind before moving: a dirty tree
# makes the checkout fail, and a half-applied one would measure neither commit.
git checkout --force --detach "$COMMIT"
git clean -fdx --exclude=perf-pair --exclude=.git
git --no-pager log -1 --format='%H %s'

# Weight-loading flags, appended to ARGS for EVERY phase. Deliberately applied
# here rather than in the matrix cell: the pairing is only valid if base, head
# and warmup load weights the same way, and a single append point makes a
# half-applied setting impossible to express.
#
# What this buys, measured on the MI355X runners this workflow uses (ATOM
# Benchmark run 35121706150, 2026-09-16, 31 successful jobs, grepped from
# `Model load done: ... (weights loaded in Xs)`):
#
#     DeepSeek-V4-Pro     297s median (282-341, n=8)
#     GLM-5.2-FP8         344s        (311-425, n=3)
#     Kimi-K3             635s        (n=1)
#     MiniMax-M3-MXFP8    156s        (147-237, n=3)
#     gpt-oss-120b         37s        (35-37,   n=6)
#
# against a 369s benchmark for the same DeepSeek cell -- model load is 76% of
# the measurement it precedes, and this workflow pays it three times per job.
# `--load_dummy=xavier` drops it to ~0.1s (no tensor is materialized).
#
# It is not free. Measured locally (MI308X, DeepSeek-V4-Flash, tp=8) dummy
# weights ran 20.2% slower -- 4542 vs 5425 tok/s, TPOT 121.0 vs 100.6ms --
# because `--fake-eplb` installs a deterministic synthetic routing table that
# lights all 256 experts every decode step, where learned routing is skewed
# and lights fewer. Net on the DeepSeek cell: 297 + ~30s of shorter cudagraph
# capture saved, 0.202 x 369 = 75s of longer benchmark paid, about +4 min per
# phase and +12 min per job. Break-even is a ~44s load, which every model here
# except gpt-oss-120b clears by 3x or more.
#
# `--fake-eplb` is required, not optional: `--load_dummy` alone leaves the
# recycled `e_score_correction_bias` in place, which collapses every token onto
# one EP rank (see moe.py:2899). MXFP4 is the validated dummy path
# (loader.py:179); FP8 is made finite but is not distribution-realistic.
#
# TWO THINGS THIS CHANGES FOR THE READER, both handled in the workflow:
#   - absolute numbers are NOT comparable to the dashboard, so the history
#     baseline check is switched off when this is set;
#   - sensitivity to a real regression under dummy weights has NOT been
#     measured against a known injection. The pairing still holds base and
#     head to the same footing, so a delta is still a delta, but its magnitude
#     may not match what real weights would have reported.
#
# Set DUMMY_WEIGHT_ARGS to the empty string to measure on real weights.
LAUNCH_ARGS="${ARGS:-}${DUMMY_WEIGHT_ARGS:+ ${DUMMY_WEIGHT_ARGS}}"
if [ -n "${DUMMY_WEIGHT_ARGS:-}" ]; then
  echo "${HALF}: dummy weights active -- ${DUMMY_WEIGHT_ARGS}"
fi

if [ -n "${MODEL_HOST_ROOT:-}" ]; then
  model_path="/models/${MODEL_PATH}"
else
  model_path="${MODEL_PATH}"
fi

echo "========== ${HALF}: launching server =========="
# Piped through stdin so the container's bash parses the quoting in ARGS
# exactly once. Substituting ARGS into a `bash -lc "..."` string instead
# collides single-quoted JSON values with the outer quotes and strips them
# (argparse then rejects the value). Mirrors benchmark-tmpl.yml.
echo ".github/scripts/atom_test.sh launch ${model_path} ${LAUNCH_ARGS}" \
  | docker exec -i "$CONTAINER" bash -l

echo "========== ${HALF}: running benchmark =========="
# ISL/OSL/CONC/RANDOM_RANGE_RATIO are read by atom_test.sh itself, under
# `set -u`, so a missing one aborts the benchmark after the model has already
# loaded. In CI the container is started with them injected (via
# atom-bench-container's container-env), which makes passing them here look
# redundant -- until the container is started any other way and the run fails
# with "ISL: unbound variable". Passing them explicitly removes the dependency
# on how the container was created.
docker exec \
  -e RESULT_FILENAME="${RESULT_FILENAME}" \
  -e SERVER_ARGS="${LAUNCH_ARGS}" \
  -e BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}" \
  -e ISL="${ISL}" \
  -e OSL="${OSL}" \
  -e CONC="${CONC}" \
  -e RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO}" \
  -e NUM_PROMPTS_OVERRIDE="${NUM_PROMPTS_OVERRIDE:-}" \
  -e MP="$model_path" \
  "$CONTAINER" bash -lc '.github/scripts/atom_test.sh benchmark "$MP"'

echo "========== ${HALF}: stopping server =========="
# Must complete before the next half checks out: atom_test.sh stop waits for
# the GPUs to actually free, and starting the next server against a partially
# released device measures the teardown, not the commit.
docker exec "$CONTAINER" bash -lc '.github/scripts/atom_test.sh stop'

if [ "$HALF" = "warmup" ]; then
  # Deliberately not collected: a warmup result that reached the judge would be
  # indistinguishable from a measurement.
  rm -f "${RESULT_FILENAME}"*.json
  echo "warmup: results discarded"
  exit 0
fi

echo "========== ${HALF}: collecting results =========="
# Copy out before the next checkout wipes the tree. Nothing is judged here --
# the pairing only has to produce two comparable sets of result JSON.
found=0
while IFS= read -r -d '' f; do
  cp "$f" "$OUT_DIR/"
  found=$((found + 1))
done < <(find . -maxdepth 2 -name "${RESULT_FILENAME}*.json" -not -path './perf-pair/*' -print0)

if [ "$found" -eq 0 ]; then
  # Leave the directory empty rather than inventing a result. A missing half
  # makes the pair unjudgeable, which the judge reports as insufficient
  # coverage -- the one thing it must never do is read as a pass.
  echo "::warning::${HALF}: no result JSON matching ${RESULT_FILENAME}*.json"
else
  echo "${HALF}: collected ${found} result file(s)"
fi
