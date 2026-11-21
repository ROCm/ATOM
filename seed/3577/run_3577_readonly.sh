#!/usr/bin/env bash
# Fixed sequential entrypoint ONLY; no ownership detection, lease or process killing.
# Invoke solely as the cleared common supervisor's admitted child.
set -euo pipefail
case "$LEG" in
  A-pp2) SOURCE_SHA=fb1304722a9f79cd16e7383b8b504ccb29851d77 ;;
  B-pp2|B-accuracy) SOURCE_SHA=8e354a56986bb6342139b657bb9e45e460171385 ;;
  *) exit 2 ;;
esac
export ATOM_SRC=/app/atom-3577/source
export VALIDATION=/app/atom-3577/input
export EVIDENCE=/app/atom-3577/evidence/$LEG
export MODEL=/models/moonshotai/Kimi-K3.1-Pro
export DRAFT=/models/Inferact/Kimi-K3.1-DSpark
export PYTHONPATH="$ATOM_SRC:$VALIDATION"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export ATOM_DOCKER_IMAGE=rocm/atom-dev@sha256:ea74fa0b05d8c7a062ae235b901cb2b1bc0c169b0a5edd7ca07bfa3b967ca462
export ATOM_SERVER_PORT=8000 ATOM_BENCHMARK_CAPTURE=1 ATOM_ENABLE_REPLAYSSM=1
export ATOM_PROFILER_RECORD_SHAPES=1 ATOM_PROFILER_WITH_STACK=1 ATOM_PROFILER_PROFILE_MEMORY=1
export ATOM_SERVER_LOG="$EVIDENCE/server.log"
test -d "$EVIDENCE"
cd "$ATOM_SRC"
test "$(git rev-parse HEAD)" = "$SOURCE_SHA"
python3 "$VALIDATION/verify_inputs.py"
git show -s --format='%H%n%T%n%P' HEAD > "$EVIDENCE/source.txt"
git diff HEAD --binary > "$EVIDENCE/validation-overlay.diff"
# Output CWD is evidence; immutable helper scripts are a reviewed read-only submount.
WORK="$EVIDENCE/work"
mkdir -p "$WORK"
test -d "$WORK/scripts"
cd "$WORK"
trap 'result=$?; printf "{\"entry_exit\":%d}\n" "$result" > "$EVIDENCE/entry-exit.json"; exit "$result"' EXIT
COMMON=(--trust-remote-code --kv_cache_dtype bf16 --index-cache-dtype fp8
  --max-num-seqs 16 --gpu-memory-utilization 0.9 --block-size 128
  --method dspark --draft-model "$DRAFT" --num-speculative-tokens 7)
if [ "$LEG" != B-accuracy ]; then
  export ATOM_MONO_ENABLE=0 ATOM_PP_SEND_ALLGATHER=1
  # Explicit tensor metadata comes from the observer; avoid profiler shape
  # recording retaining bank tensor references and perturbing their lifetimes.
  export ATOM_PROFILER_RECORD_SHAPES=0
  export ATOM_3577_OBSERVER_DIR="$EVIDENCE/pp-observer"
  export SERVER_ARGS="-tp 4 --pipeline-parallel-size 2 --decode-context-parallel-size 4"
  ENABLE_TORCH_PROFILER=0 timeout --signal=TERM --kill-after=10s 1500s \
    bash "$ATOM_SRC/.github/scripts/atom_test.sh" launch "$MODEL" "${COMMON[@]}" \
    -tp 4 --pipeline-parallel-size 2 --decode-context-parallel-size 4 \
    --enforce-eager --max-model-len 131072 --max-num-batched-tokens 16384 \
    --enable_prefix_caching --torch-profiler-dir "$EVIDENCE/traces"
  export ISL=49152 OSL=64 CONC=2 RANDOM_RANGE_RATIO=1
  export NUM_PROMPTS_OVERRIDE=8 ATOM_BENCHMARK_MAX_MINUTES=5 ATOM_BUNDLE_REQUIRE_FULL=1
  export RESULT_FILENAME="$LEG" ATOM_BUNDLE_WORK="$EVIDENCE/benchmark-work"
  export ATOM_BUNDLE_OUTPUT="$EVIDENCE/benchmark-bundle"
  export ATOM_CLIENT_LOG="$EVIDENCE/benchmark-client.log"
  export ATOM_3577_PROMPTS="$EVIDENCE/benchmark-prompts.jsonl"
  export BENCH_EXTRA_ARGS="--num-warmups 0 --random-prefix-len 0 --seed 357701 --save-detailed"
  ENABLE_TORCH_PROFILER=0 timeout --signal=TERM --kill-after=10s 360s \
    bash "$ATOM_SRC/.github/scripts/atom_test.sh" benchmark "$MODEL"
  cp "$LEG.json" "$EVIDENCE/benchmark-detailed.json"
  timeout --signal=TERM --kill-after=10s 660s bash "$VALIDATION/exact_trace.sh"
else
  export ATOM_MONO_ENABLE=1
  unset ATOM_3577_OBSERVER_DIR
  case "${TRACE_COMPILED:-0}" in 0|1) ;; *) exit 2 ;; esac
  printf '{"requested":%d}\n' "${TRACE_COMPILED:-0}" > "$EVIDENCE/compiled-trace-request.json"
  ENABLE_TORCH_PROFILER=0 timeout --signal=TERM --kill-after=10s 1500s \
    bash "$ATOM_SRC/.github/scripts/atom_test.sh" launch "$MODEL" "${COMMON[@]}" \
    -tp 8 --decode-context-parallel-size 8 --torch-profiler-dir "$EVIDENCE/traces"
  test ! -e accuracy_test_results
  export ATOM_CLIENT_LOG="$EVIDENCE/accuracy-client.log" ACCURACY_TIMEOUT_MINUTES=45
  export CLIENT_COMMAND='bash /app/atom-3577/input/lm_eval_receipt.sh --model local-completions --model_args model=${MODEL_PATH},base_url=http://localhost:8000/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True --tasks gsm8k --num_fewshot 3 --output_path ${OUTPUT_PATH} --log_samples'
  timeout --signal=TERM --kill-after=10s 2760s bash "$ATOM_SRC/.github/scripts/atom_test.sh" accuracy "$MODEL"
  cp -a accuracy_test_results "$EVIDENCE/"
  # One optional compiler trace on this already-loaded server; never another evaluation.
  if [ "${TRACE_COMPILED:-0}" = 1 ]; then
    timeout --signal=TERM --kill-after=10s 660s bash "$VALIDATION/exact_trace.sh"
    printf '{"status":"completed"}\n' > "$EVIDENCE/compiled-trace-outcome.json"
  else
    printf '{"status":"not_requested"}\n' > "$EVIDENCE/compiled-trace-outcome.json"
  fi
fi
# Server and telemetry collector deliberately remain descendants for the common
# supervisor's pidfd cleanup, even on success. No stock "stop" or broad PID kill.
