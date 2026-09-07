#!/usr/bin/env bash
# Run one half of the PR perf check pairing.
#
#   perf_check_half.sh <commit-sha> <base|head>
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
# RESULT_FILENAME, plus whatever atom_test.sh itself reads.

set -euo pipefail

COMMIT="${1:?commit sha required}"
HALF="${2:?half must be 'base' or 'head'}"

case "$HALF" in
  base|head) ;;
  *) echo "ERROR: half must be 'base' or 'head', got '$HALF'" >&2; exit 2 ;;
esac

: "${CONTAINER:?CONTAINER must be set}"
: "${MODEL_PATH:?MODEL_PATH must be set}"
: "${RESULT_FILENAME:?RESULT_FILENAME must be set}"

OUT_DIR="perf-pair/${HALF}"
mkdir -p "$OUT_DIR"

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
echo ".github/scripts/atom_test.sh launch ${model_path} ${ARGS:-}" \
  | docker exec -i "$CONTAINER" bash -l

echo "========== ${HALF}: running benchmark =========="
docker exec \
  -e RESULT_FILENAME="${RESULT_FILENAME}" \
  -e SERVER_ARGS="${ARGS:-}" \
  -e BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}" \
  -e MP="$model_path" \
  "$CONTAINER" bash -lc '.github/scripts/atom_test.sh benchmark "$MP"'

echo "========== ${HALF}: stopping server =========="
# Must complete before the next half checks out: atom_test.sh stop waits for
# the GPUs to actually free, and starting the next server against a partially
# released device measures the teardown, not the commit.
docker exec "$CONTAINER" bash -lc '.github/scripts/atom_test.sh stop'

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
