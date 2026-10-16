#!/usr/bin/env bash
# Re-measure GSM8K for DeepSeek-V4.1-Flash on the vLLM plugin at the EXACT
# working point the inherited 0.9242 was taken at, so the two are comparable.
#
# Recipe recovered from the original run's results JSON
# (results_2026-09-22T17-50-37.055730.json, ATOM c91d8c22c, lm_eval 0.4.13).
# Every value below that differs from a default is load-bearing; the ones that
# bite are called out inline. Run INSIDE the container.
set -euo pipefail

TREE=${TREE:-/workspace/ATOM-dsv41-nolmc}
MODEL=${MODEL:-/data/amd_int/models/DeepSeek-V4.1-Flash}
NAME=DeepSeek-V4.1-Flash
PORT=${PORT:-8011}
# Outside the tree by default. Writing run artefacts into $TREE is how two
# `.gsm8k_*` fingerprint directories ended up committed.
OUT=${OUT:-/tmp/v41_gsm8k_$(date +%m%d_%H%M)}
mkdir -p "$OUT"

# --- what is under test -------------------------------------------------
# PYTHONPATH, not an editable install: the tree on disk IS the thing measured,
# so record its sha AND prove the interpreter imports it rather than the
# image's own atom. A mismatch here silently measures the wrong code.
export PYTHONPATH="$TREE"

# Refuse to measure a tree that is behind main. The number goes on a PR, and a
# PR is merged as of main's tip, not as of whatever this branch was cut from --
# so measuring a stale tree produces a number for code nobody will run. This is
# a gate rather than a habit on purpose: the window between "GPUs are free" and
# "launch" is exactly where remembering fails.
# The fetch needs credentials this container does not have, so the launcher
# does it on the host and passes the timestamp in. Split, but not weakened:
# a missing or stale stamp is fatal here, so "nobody fetched" cannot pass as
# "nothing to fetch". The comparison itself stays on this side of the launch,
# which is the half that has to guard it.
: "${BASE_FETCHED_AT:?FATAL: launcher must fetch origin/main and export BASE_FETCHED_AT}"
age=$(( $(date +%s) - BASE_FETCHED_AT ))
if [ "$age" -lt 0 ] || [ "$age" -gt 900 ]; then
    echo "FATAL: origin/main was fetched ${age}s ago; re-fetch before measuring" >&2
    exit 1
fi
if ! git -C "$TREE" merge-base --is-ancestor origin/main HEAD; then
    echo "FATAL: origin/main ($(git -C "$TREE" rev-parse --short origin/main)) is not an" >&2
    echo "ancestor of HEAD ($(git -C "$TREE" rev-parse --short HEAD)). Rebase first." >&2
    git -C "$TREE" log --oneline HEAD..origin/main >&2
    exit 1
fi
git -C "$TREE" rev-parse origin/main > "$OUT/base.sha"
git -C "$TREE" rev-parse HEAD > "$OUT/tree.sha"
git -C "$TREE" status --porcelain >> "$OUT/tree.sha"
# A flags=1 .pyc does not revalidate against its source, so a stale one can sit
# behind a correct `atom.__file__` and be the thing that actually runs. The
# path assertion below does not see that; purging does.
find "$TREE" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
# A fingerprint of the code under test, not just its commit: an edit made
# while this runs changes the thing being measured, and `git rev-parse` alone
# would not notice an uncommitted one. Re-checked after the eval; a run whose
# tree moved under it is void, and says so rather than reporting a number.
fingerprint() { find "$TREE/atom" -name '*.py' -type f -exec sha1sum {} + | sort | sha1sum | cut -d' ' -f1; }
fingerprint > "$OUT/tree.fingerprint"

python3 -c 'import atom, sys; print(atom.__file__)' > "$OUT/atom_import.txt"
grep -q "^$TREE/" "$OUT/atom_import.txt" || {
    echo "FATAL: imported atom is $(cat "$OUT/atom_import.txt"), not $TREE" >&2
    exit 1; }

# --- server -------------------------------------------------------------
# Readability from inside this container, not existence on the host: a lost
# supplementary group turns this into an HFValidationError about a repo id.
[ -r "$MODEL/config.json" ] || { echo "FATAL: $MODEL/config.json unreadable here" >&2; exit 1; }
stat -c 'dev=%d inode=%i %n' "$MODEL/config.json" > "$OUT/model_identity.txt"

export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-4,5,6,7}
export AITER_LOG_LEVEL=WARNING
export VLLM_WORKER_MULTIPROC_METHOD=spawn      # fork re-inits CUDA and crashes
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1  # server side: offline
rm -rf /root/.cache/atom/*                     # stale compile cache = silent failures

vllm serve "$MODEL" \
    --served-model-name "$NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --dtype auto --kv-cache-dtype auto \
    --tensor-parallel-size 4 --distributed-executor-backend mp \
    --trust-remote-code --tokenizer-mode deepseek_v4 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 --max-num-seqs 64 \
    --no-enable-prefix-caching \
    > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT

# Block on "answers or died" -- never a bare sleep with no failure branch.
for _ in $(seq 1 240); do
    kill -0 $SERVER 2>/dev/null || {
        echo "server exited during startup; see $OUT/server.log" >&2
        tail -40 "$OUT/server.log" >&2; exit 1; }
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" > "$OUT/models.json" || {
    echo "server never became ready; see $OUT/server.log" >&2; exit 1; }

# `--gpu-memory-utilization 0.9` is not a portable description of a KV pool:
# it is on record that 0.29 began charging cudagraph capture to the KV budget,
# so the same 0.9 buys different pools either side of that change. Record the
# vLLM version AND the pool the server actually got, so a later reader can
# decide whether two runs share a working point instead of inferring it from
# the flag. (Accuracy at 8192/64 should not be pool-sensitive -- but "should
# not" is the assumption this artifact exists to let someone check.)
python3 -c 'import vllm; print(vllm.__version__)' > "$OUT/vllm_version.txt" 2>/dev/null || true
# The image ships xgrammar 0.2.8 against apache-tvm-ffi 0.1.10, which cannot
# import -- and vLLM's startup path goes through it. The launcher pins 0.2.7
# (what #2357's dockerfile does). Record what was actually resolved: this is a
# deviation from the image, so it belongs in the artifacts rather than in my
# memory of what I typed.
pip show xgrammar apache-tvm-ffi 2>/dev/null | grep -E "^(Name|Version):" > "$OUT/env_pins.txt" || true
# `vllm-latest` is a floating daily tag, so the tag names nothing reproducible.
# The launcher passes the digest it actually resolved; record it.
echo "image=${IMAGE_DIGEST:-<unset>}" >> "$OUT/env_pins.txt"
# torch/torchvision too: pinning xgrammar WITHOUT --no-deps once replaced torch
# here, which surfaces as `operator torchvision::nms does not exist` at import
# but, had it happened later, would have been a plausible-looking wrong number.
python3 -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)' \
    >> "$OUT/env_pins.txt" 2>/dev/null || true
grep -iE "GPU KV cache size|KV cache size|# GPU blocks|available KV cache memory|Available KV" \
    "$OUT/server.log" | sort -u > "$OUT/kv_pool.txt" || true

# `--dtype auto` is not self-describing: record what it resolved to, because
# the reference run also used `auto` and "both said auto" is not "both ran the
# same dtype". Expected torch.bfloat16.
grep -oE "dtype=torch\.[a-z0-9]+" "$OUT/server.log" | sort -u > "$OUT/resolved_dtype.txt" || true

# --- client -------------------------------------------------------------
# HF_*_OFFLINE=0 here, unlike the server: lm_eval fetches the gsm8k dataset.
# tokenizer_backend=none + tokenized_requests=False are load-bearing, not
# stylistic -- this checkpoint ships no HF tokenizer, so lm_eval building one
# fails outright.
# num_concurrent=32 and the DEFAULT max_gen_toks=256 are part of the working
# point: 0.9242 was measured under a 256-token generation budget, and changing
# it makes the comparison invalid rather than better.
HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 lm_eval \
  --model local-completions \
  --model_args "model=$NAME,base_url=http://127.0.0.1:$PORT/v1/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=none" \
  --tasks gsm8k --num_fewshot 5 \
  --output_path "$OUT/lm_eval" \
  2>&1 | tee "$OUT/lm_eval.log"

# The tree must not have moved under the measurement.
if [ "$(fingerprint)" != "$(cat "$OUT/tree.fingerprint")" ]; then
    echo "VOID: the source tree changed while this run was in flight; the" >&2
    echo "number above is not attributable to any one tree. Re-run." >&2
    exit 1
fi

# `config.num_fewshot` lands as null in the results JSON even when 5 took
# effect, so comparing that field matches falsely on both sides. The evidence
# is lm_eval's own overwrite line and the results table's n-shot column.
grep -i "num_fewshot" "$OUT/lm_eval.log" > "$OUT/nshot_evidence.txt" || true

echo "--- reference: strict-match 0.9241849886277483 +/- 0.0072912057231626 (n=1319, lm_eval 0.4.13, ATOM c91d8c22c)"
python3 -c "import lm_eval, sys; print('this run lm_eval', lm_eval.__version__)" 2>/dev/null || true
