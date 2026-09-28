#!/usr/bin/env bash
# AIPerf agentic trace-replay benchmark for the SINGLE-NODE pipeline
# (.github/scripts/atom_test.sh, BENCH_KIND=aiperf_agentic). Mirrors InferenceX's
# benchmarks/single_node/agentic recipe.
#
# Source it, then call `run_aiperf_agentic`. Everything it needs comes from the
# AIPERF_* variables below, so a caller only overrides what it wants to differ.
#
# Single-node workload defaults and installation policy come from run
# 35834179304. PD keeps its own topology/dataset settings; both use the
# shared aiperf_dashboard.py exporter. Sourcing leaves shell options intact.

AIPERF_DIR="${AIPERF_DIR:-/tmp/atom-aiperf}"
AIPERF_VENV="${AIPERF_VENV:-/tmp/atom-aiperf-venv}"
AIPERF_COMMIT="${AIPERF_COMMIT:-754356e9a39acc6cc6afb242d123bb57c3fb6f75}"
AIPERF_SCENARIO="${AIPERF_SCENARIO:-inferencex-agentx-mvp}"
AIPERF_PUBLIC_DATASET="${AIPERF_PUBLIC_DATASET:-semianalysis_cc_traces_weka_062126}"
# Unset by default: the single-node recipe passes no --max-context-length.
# `-` not `:-` so a caller can blank it back out explicitly.
AIPERF_MAX_CONTEXT_LENGTH="${AIPERF_MAX_CONTEXT_LENGTH-}"
AIPERF_NUM_DATASET_ENTRIES="${AIPERF_NUM_DATASET_ENTRIES:-393}"
AIPERF_BENCHMARK_DURATION="${AIPERF_BENCHMARK_DURATION:-3600}"
AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
# `--agentic-warmup-grace-period`, NOT `--warmup-grace-period`: aiperf
# synthesizes the agentic warmup from the profiling phase rather than a
# user-declared one, and its own help says the plain flag is only honoured
# alongside `--warmup-duration` -- which an agentic run never sets. Passing
# the plain one is therefore inert, and unset means the warmup barrier waits
# INDEFINITELY for every primed trajectory to return (aiperf b7b16cf,
# src/aiperf/config/flags/cli_config.py:2310-2330).
AIPERF_AGENTIC_WARMUP_GRACE_PERIOD="${AIPERF_AGENTIC_WARMUP_GRACE_PERIOD:-1800}"
AIPERF_TRAJECTORY_START_MIN_RATIO="${AIPERF_TRAJECTORY_START_MIN_RATIO:-0.25}"
AIPERF_TRAJECTORY_START_MAX_RATIO="${AIPERF_TRAJECTORY_START_MAX_RATIO:-0.75}"
AIPERF_FAILED_REQUEST_THRESHOLD="${AIPERF_FAILED_REQUEST_THRESHOLD:-0.10}"
AIPERF_SLICE_DURATION="${AIPERF_SLICE_DURATION:-1.0}"
AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT="${AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT:-300}"
AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT:-900000}"
AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES:-0}"
AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT:-1800}"
AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT:-1800}"
AIPERF_UNSAFE_OVERRIDE="${AIPERF_UNSAFE_OVERRIDE:-}"

# Lowest aiperf the agentic flags require. `--warmup-requests-per-lane` and
# `--agentic-warmup-grace-period` do not exist before this, and a run started
# with them would die on an unknown-argument error rather than measure anything.
AIPERF_MIN_VERSION="${AIPERF_MIN_VERSION:-0.12.0}"

_aiperf_version_ge() {
  # $1 >= $2, dotted-decimal. `sort -V` puts the lower one first, so $1 wins
  # when the head of the sorted pair is $2 (or the two are equal).
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]
}

ensure_aiperf() {
  # Prefer the aiperf the image already ships: the Dockerfile pins it to the
  # same commit as AIPERF_COMMIT, so a matching image saves a clone plus an
  # editable install per cell (~26s). Only the VERSION is checked, not the
  # commit -- an image that is merely a few commits off still runs the same
  # flags, while one that predates AIPERF_MIN_VERSION cannot.
  #
  # Note this is the server's own venv, so the client shares its site-packages;
  # InferenceX isolates the two deliberately. That is a property of the image
  # (it installs aiperf into /opt/venv at build time), not something reusing it
  # introduces -- but it is why the fallback below builds a separate venv.
  local img_bin img_ver
  img_bin="$(command -v aiperf 2>/dev/null || true)"
  if [[ -n "${img_bin}" ]]; then
    img_ver="$("${img_bin}" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    if [[ -n "${img_ver}" ]] && _aiperf_version_ge "${img_ver}" "${AIPERF_MIN_VERSION}"; then
      AIPERF_VENV="$(dirname "$(dirname "${img_bin}")")"
      echo "[aiperf] using the image's aiperf ${img_ver} (${img_bin})"
      return
    fi
    echo "[aiperf] image ships aiperf ${img_ver:-<unknown>}, below the required" \
         "${AIPERF_MIN_VERSION}; building our own"
  else
    echo "[aiperf] no aiperf on PATH; building our own"
  fi

  local current_commit=""
  if [[ -d "${AIPERF_DIR}/.git" ]]; then
    current_commit="$(git -C "${AIPERF_DIR}" rev-parse HEAD 2>/dev/null || true)"
  fi
  if [[ -x "${AIPERF_VENV}/bin/aiperf" && "${current_commit}" == "${AIPERF_COMMIT}" ]]; then
    return
  fi

  echo "[aiperf] preparing ${AIPERF_DIR} @ ${AIPERF_COMMIT}"
  mkdir -p "$(dirname "${AIPERF_DIR}")" "$(dirname "${AIPERF_VENV}")"
  if [[ ! -d "${AIPERF_DIR}/.git" ]]; then
    rm -rf "${AIPERF_DIR}"
    git clone https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_DIR}"
  fi
  git -C "${AIPERF_DIR}" fetch https://github.com/SemiAnalysisAI/aiperf.git "${AIPERF_COMMIT}"
  git -C "${AIPERF_DIR}" checkout --detach "${AIPERF_COMMIT}"
  rm -rf "${AIPERF_VENV}"
  python3 -m venv "${AIPERF_VENV}"
  "${AIPERF_VENV}/bin/python" -m pip install --upgrade pip
  "${AIPERF_VENV}/bin/python" -m pip install -e "${AIPERF_DIR}"
  "${AIPERF_VENV}/bin/aiperf" --version
}

# The scenario locks its own invariants; a duration under its 900s floor is a
# ScenarioLockError without this. Overridden runs are stamped
# `submission_valid=false`, so they are smoke tests, never published numbers.
aiperf_unsafe_args() {
  if (( AIPERF_BENCHMARK_DURATION < 900 )) \
    || [[ "${AIPERF_UNSAFE_OVERRIDE}" == "1" || "${AIPERF_UNSAFE_OVERRIDE}" == "true" ]]; then
    printf '%s\n' --unsafe-override
  fi
}

# agentic_prepare
#
# Everything a single-node agentic cell needs before the replay: the labels the
# dashboard payload carries, the session-affinity headers, the aiperf install,
# and the supervision budget. Sets AGENTIC_OUT_DIR and BENCH_MAX_MIN for the
# caller. Kept here rather than in the driver so the topology knowledge lives
# with the runner that acts on it.
agentic_prepare() {
  export BENCHMARK_KIND="aiperf_agentic"
  export MODEL_NAME="${MODEL_NAME:-$MODEL_PATH}"
  export TOPOLOGY="${TOPOLOGY:-single-node}"

  if [[ "${SERVER_ARGS:-}" == *"--enable-dp-attention"* ]]; then
    export DISPLAY_TOPOLOGY="${DISPLAY_TOPOLOGY:-single-node-dpa}"
    # Session affinity. ATOM's DPA router reads `x-dynamo-session-id` (falling
    # back to `x-correlation-id`, which AIPerf always sends) and
    # `x-dynamo-parent-session-id` -- see `_get_dp_session_affinity_ids` in
    # atom/entrypoints/openai/api_server.py. Only the Dynamo option sends the
    # PARENT id, which carries a forked agent tree's lineage; the generic
    # `X-Session-ID` one sends a header ATOM does not read, kept in case a
    # router is ever put in front.
    export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
  else
    export DISPLAY_TOPOLOGY="${DISPLAY_TOPOLOGY:-single-node}"
    unset AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID
    unset AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID
  fi

  AGENTIC_OUT_DIR="${AGENTIC_OUT_DIR:-./aiperf-artifacts-c${CONC}}"
  ensure_aiperf
  local metadata_key
  for metadata_key in ${!AIPERF_@}; do export "$metadata_key"; done

  # The replay is only part of the wall clock, and the rest is not small: on a
  # c=48 cell, dataset configuration plus the warmup that primes every lane's
  # prefix cache took 23.5 min BEFORE profiling started, and warmup scales with
  # the lane count. Both bracketing phases carry their own 1800s timeouts, so
  # budget 90 min around the replay rather than let the drain supervisor cut a
  # healthy run. Must stay BELOW the workflow step's `timeout-minutes`, so an
  # overrun is killed there -- with a reason in the log -- rather than by the
  # runner, which takes the whole step down silently.
  BENCH_MAX_MIN=$(( AIPERF_BENCHMARK_DURATION / 60 + 90 ))

  echo "Agentic replay: ${AIPERF_BENCHMARK_DURATION}s at concurrency ${CONC}"
  echo "  scenario=${AIPERF_SCENARIO} dataset=${AIPERF_PUBLIC_DATASET}"
  echo "  artifacts=${AGENTIC_OUT_DIR} drain budget=${BENCH_MAX_MIN}min"
  if (( AIPERF_BENCHMARK_DURATION < 900 )); then
    echo "  WARNING: below the scenario's 900s floor -- --unsafe-override is"
    echo "           active and results carry submission_valid=false."
  fi
}

# run_aiperf_agentic <url> <conc> <out_dir> [server_metrics_url ...]
#
# One replay against one endpoint. The caller owns the topology: it decides the
# URL (a PD router, or a single-node server) and which /metrics endpoints to
# scrape, and it names the output directory. Writes
# `<out_dir>/profile_export_aiperf.json` and returns non-zero if that file is
# missing, which is the only reliable signal that a run produced nothing --
# aiperf itself exits 0 on a run where every request failed.
run_aiperf_agentic() {
  local url="$1" conc="$2" out_dir="$3"
  shift 3

  local -a server_metrics_args=()
  if (( $# > 0 )); then
    server_metrics_args=(--server-metrics "$@")
  fi

  local -a unsafe_args=()
  mapfile -t unsafe_args < <(aiperf_unsafe_args)

  # Optional: the PD topologies pin it, the single-node recipe deliberately does
  # not, and passing an empty value is not the same as omitting the flag.
  local -a ctx_args=()
  [[ -n "${AIPERF_MAX_CONTEXT_LENGTH}" ]] \
    && ctx_args=(--max-context-length "${AIPERF_MAX_CONTEXT_LENGTH}")

  mkdir -p "${out_dir}"
  # Weka intentionally skips inputs.json. Isolate the actual HF inputs for a
  # content ledger; a reused AIPerf mmap cache would bypass loading these files.
  local dataset_cache=""
  if [[ -n "${ATOM_BUNDLE_WORK:-}" ]]; then
    dataset_cache=$(mktemp -d /tmp/atom-benchmark-dataset.XXXXXX)
    export HF_DATASETS_CACHE="$dataset_cache"
    export AIPERF_DATASET_MMAP_CACHE_ENABLED=false
  fi
  local -a AIPERF_COMMAND
  AIPERF_COMMAND=("${AIPERF_VENV}/bin/aiperf" profile \
    "${unsafe_args[@]}" \
    --scenario "${AIPERF_SCENARIO}" \
    --url "${url}" \
    --endpoint /v1/chat/completions \
    --endpoint-type chat \
    --streaming \
    --model "${AIPERF_MODEL:-${MODEL_PATH}}" \
    --concurrency "${conc}" \
    --benchmark-duration "${AIPERF_BENCHMARK_DURATION}" \
    --stats-interval 30 \
    --random-seed 42 \
    --failed-request-threshold "${AIPERF_FAILED_REQUEST_THRESHOLD}" \
    --trajectory-start-min-ratio "${AIPERF_TRAJECTORY_START_MIN_RATIO}" \
    --trajectory-start-max-ratio "${AIPERF_TRAJECTORY_START_MAX_RATIO}" \
    --warmup-requests-per-lane "${AIPERF_WARMUP_REQUESTS_PER_LANE}" \
    --trace-idle-gap-cap-seconds "${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS}" \
    --agentic-warmup-grace-period "${AIPERF_AGENTIC_WARMUP_GRACE_PERIOD}" \
    --use-server-token-count \
    --no-gpu-telemetry \
    --tokenizer "${MODEL_PATH}" \
    --tokenizer-trust-remote-code \
    "${ctx_args[@]}" \
    --num-dataset-entries "${AIPERF_NUM_DATASET_ENTRIES}" \
    --slice-duration "${AIPERF_SLICE_DURATION}" \
    "${server_metrics_args[@]}" \
    --output-artifact-dir "${out_dir}" \
    --public-dataset "${AIPERF_PUBLIC_DATASET}")
  if [[ -n "${ATOM_BUNDLE_WORK:-}" ]]; then
    python3 -m atom.benchmarks.results capture-launch \
      --output "$ATOM_BUNDLE_WORK/client-launch.json" -- "${AIPERF_COMMAND[@]}"
  fi
  local -a replay_command=("${AIPERF_COMMAND[@]}")
  if [[ "${ENABLE_TORCH_PROFILER:-0}" == 1 ]]; then
    replay_command=(python3 "$(dirname -- "${BASH_SOURCE[0]}")/profile_agentic_replay.py" \
      --url "$url" --output "$out_dir/profiler-window.json" \
      --seconds "${ATOM_AGENTIC_PROFILE_SECONDS:-30}" -- "${AIPERF_COMMAND[@]}")
  fi
  AIPERF_UI_REALTIME_METRICS_ENABLED=true "${replay_command[@]}" 2>&1 \
    | tee "${out_dir}/aiperf.log" || return $?

  if [[ ! -f "${out_dir}/profile_export_aiperf.json" ]]; then
    echo "[aiperf][FAIL] ${out_dir}/profile_export_aiperf.json was not produced" >&2
    return 1
  fi
  if [[ -n "$dataset_cache" ]]; then
    python3 -m atom.benchmarks.results capture-hf-dataset \
      --cache-dir "$dataset_cache" --export "${out_dir}/profile_export_aiperf.json" \
      --output "${out_dir}/dataset-identity.json" \
      || echo '[bundle] HF dataset capture failed; completeness validation will report missing identity' >&2
    rm -rf -- "$dataset_cache"
  fi
}

write_aiperf_dashboard_json() {
  python3 "$(dirname -- "${BASH_SOURCE[0]}")/aiperf_dashboard.py" "$@" --single-node
}
