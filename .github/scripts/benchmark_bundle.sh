#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Sourced by atom_test.sh after selecting the client. Never changes server args.

start_benchmark_bundle() {
  local kind=random
  [[ "${BENCH_KIND:-random}" == aiperf_agentic ]] && kind=agentic
  ATOM_BUNDLE_WORK="${ATOM_BUNDLE_WORK:-$PWD/benchmark-evidence/${RESULT_FILENAME}}"
  ATOM_BUNDLE_OUTPUT="${ATOM_BUNDLE_OUTPUT:-$PWD/benchmark-bundles/${RESULT_FILENAME}}"
  mkdir -p "$ATOM_BUNDLE_WORK"
  export ATOM_BENCHMARK_REQUESTS="$ATOM_BUNDLE_WORK/requests.jsonl"
  if [[ "$kind" == random ]]; then
    python3 -m atom.benchmarks.results capture-launch \
      --output "$ATOM_BUNDLE_WORK/client-launch.json" -- "${BENCH_CMD[@]}"
  fi
  if [[ "$kind" == agentic ]]; then
    local -a harness_identity=()
    mapfile -t harness_identity < <(PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" "${AIPERF_VENV}/bin/python" -c \
      'from atom.benchmarks.results.metadata import installed_package; h=installed_package("aiperf"); print(h["version"] or ""); print(h["sha"] or "")')
    export ATOM_HARNESS_VERSION="${harness_identity[0]:-}"
    export ATOM_HARNESS_SHA="${harness_identity[1]:-}"
  fi
  python3 -m atom.benchmarks.results capture-config \
    --model "$MODEL_PATH" --kind "$kind" --concurrency "$CONC" \
    --launch /tmp/atom-benchmark-launch.json \
    --output "$ATOM_BUNDLE_WORK/config.json"
  python3 -m atom.benchmarks.results collect \
    --config "$ATOM_BUNDLE_WORK/config.json" --output "$ATOM_BUNDLE_WORK/telemetry" \
    --server-url "http://localhost:${ATOM_SERVER_PORT}/metrics" \
    > "$ATOM_BUNDLE_WORK/collector.log" 2>&1 &
  ATOM_BUNDLE_COLLECTOR_PID=$!
  trap 'finish_benchmark_bundle $?' EXIT
  trap 'exit 143' TERM
  trap 'exit 130' INT
}

stop_benchmark_client() {
  # The client owns a process group, including tee and harness subprocesses.
  # Both drain failure and the exit trap use this same bounded cleanup.
  [[ -n "${CLIENT_PID:-}" ]] || return 0
  kill -TERM -- -"$CLIENT_PID" 2>/dev/null || true
  local grace=5 i
  # Allow the replay wrapper's bounded stop_profile request to flush traces.
  [[ "${ENABLE_TORCH_PROFILER:-0}" == 1 ]] && grace=135
  for ((i=0; i<grace; i++)); do
    # The group leader can exit before the profiler child finishes flushing.
    # Ignore zombies, which cannot perform cleanup and may await reaping.
    ps -eo pgid=,stat= | awk -v group="$CLIENT_PID" \
      '$1 == group && $2 !~ /^Z/ { alive=1 } END { exit !alive }' || break
    sleep 1
  done
  kill -KILL -- -"$CLIENT_PID" 2>/dev/null || true
  wait "$CLIENT_PID" 2>/dev/null || true
  unset CLIENT_PID
}

finish_benchmark_bundle() {
  local benchmark_rc="$1" bundle_rc=0
  trap - EXIT TERM INT
  set +e
  stop_benchmark_client
  # Collect one final GPU sample beyond the last completed request.
  # Stop the sampling process and flush its final records before packaging.
  kill -TERM "$ATOM_BUNDLE_COLLECTOR_PID" 2>/dev/null
  wait "$ATOM_BUNDLE_COLLECTOR_PID"
  local -a args=(
    --config "$ATOM_BUNDLE_WORK/config.json"
    --output "$ATOM_BUNDLE_OUTPUT"
    --telemetry-dir "$ATOM_BUNDLE_WORK/telemetry"
    --client-log "$ATOM_CLIENT_LOG" --server-log "${ATOM_SERVER_LOG:-/tmp/atom_server.log}"
    --exit-code "$benchmark_rc"
    --client-launch "$ATOM_BUNDLE_WORK/client-launch.json"
    --collector-log "$ATOM_BUNDLE_WORK/collector.log"
    --harness-summary "${RESULT_FILENAME}.json"
  )
  if [[ "${BENCH_KIND:-random}" == aiperf_agentic ]]; then
    args+=(--record-format aiperf --records "$AGENTIC_OUT_DIR/profile_export.jsonl" --raw-dir "$AGENTIC_OUT_DIR")
  else
    args+=(--records "$ATOM_BENCHMARK_REQUESTS")
  fi
  [[ "${ATOM_BUNDLE_REQUIRE_FULL:-0}" == 1 ]] && args+=(--require-full)
  python3 -m atom.benchmarks.results build "${args[@]}" || bundle_rc=$?
  # Original failures remain failures even when diagnostic packaging succeeds.
  if [[ "$benchmark_rc" -ne 0 ]]; then
    exit "$benchmark_rc"
  fi
  exit "$bundle_rc"
}
