#!/usr/bin/env bash
# AIPerf agentic trace-replay benchmark for the SINGLE-NODE pipeline
# (.github/scripts/atom_test.sh, BENCH_KIND=aiperf_agentic). Mirrors InferenceX's
# benchmarks/single_node/agentic recipe.
#
# Source it, then call `run_aiperf_agentic`. Everything it needs comes from the
# AIPERF_* variables below, so a caller only overrides what it wants to differ.
#
# DELIBERATE DUPLICATION: `.github/scripts/atomesh/pd_server_atom.sh` carries its
# own copy of this logic for the PD-disaggregated topologies. That script is
# ATOMesh-nightly-critical and is intentionally left untouched, so the two are
# kept separate rather than shared. The defaults below therefore track the
# single-node recipe, NOT the PD one -- they differ on dataset
# (`..._062126` vs `..._062126_256k`), failed-request threshold (0.10 vs 0.50)
# and max context length (unset vs 262144). If you change behaviour here, decide
# consciously whether the PD copy wants the same change; nothing links them.
#
# This file is sourced, not executed: it deliberately sets no `set -euo
# pipefail` of its own and leaves the caller's shell options alone.

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
  AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT="${AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT}" \
  AIPERF_HTTP_TCP_USER_TIMEOUT="${AIPERF_HTTP_TCP_USER_TIMEOUT}" \
  AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES}" \
  AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AIPERF_DATASET_CONFIGURATION_TIMEOUT}" \
  AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT}" \
  AIPERF_UI_REALTIME_METRICS_ENABLED=true \
    "${AIPERF_VENV}/bin/aiperf" profile \
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
    --public-dataset "${AIPERF_PUBLIC_DATASET}" \
    2>&1 | tee "${out_dir}/aiperf.log"

  if [[ ! -f "${out_dir}/profile_export_aiperf.json" ]]; then
    echo "[aiperf][FAIL] ${out_dir}/profile_export_aiperf.json was not produced" >&2
    return 1
  fi
}

write_aiperf_dashboard_json() {
  local aiperf_json="$1"
  local out_json="$2"
  local conc="$3"
  python3 - "${aiperf_json}" "${out_json}" "${conc}" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
conc = int(sys.argv[3])
data = json.loads(src.read_text(encoding="utf-8"))


def avg(name):
    value = data.get(name)
    if isinstance(value, dict):
        return value.get("avg")
    return value


def pct(name, key):
    value = data.get(name)
    if isinstance(value, dict):
        return value.get(key)
    return None


def _tp_from_server_args():
    m = re.search(r"(?:^|\s)(?:-tp|--tensor-parallel-size)\s+(\d+)", os.environ.get("SERVER_ARGS", ""))
    return int(m.group(1)) if m else 1


def total_tokens(name):
    """Return one of AIPerf's profiling-only aggregate token counters."""
    value = avg(name)
    return int(value) if isinstance(value, (int, float)) else None


# These aggregates contain successful profiling records only: AIPerf excludes
# its internal warmup and requests cancelled during grace-period draining.
cache_hit_tokens = total_tokens("total_usage_prompt_cache_read_tokens")
cache_total_tokens = total_tokens("total_usage_prompt_tokens")

payload = {
    "benchmark_backend": "atom",
    # Directory holding this run's profile_export.jsonl, so process_result.py can
    # find the per-request records it needs for p90 e2e normalized interactivity
    # without reconstructing the directory name.
    "aiperf_artifact_dir": src.parent.name,
    "benchmark_model_name": os.environ.get("MODEL_NAME")
    or data.get("model")
    or data.get("model_id"),
    "backend": "atom",
    "benchmark_kind": os.environ.get("BENCHMARK_KIND") or "aiperf_agentic",
    "scenario": os.environ.get("AIPERF_SCENARIO"),
    "public_dataset": os.environ.get("AIPERF_PUBLIC_DATASET"),
    "topology": os.environ.get("TOPOLOGY") or data.get("topology"),
    "display_topology": os.environ.get("DISPLAY_TOPOLOGY")
    or data.get("display_topology"),
    "precision": os.environ.get("PRECISION") or data.get("precision"),
    "random_input_len": int(
        data.get("max_context_length")
        or os.environ.get("AIPERF_MAX_CONTEXT_LENGTH")
        or 0
    ),
    "random_output_len": 1024,
    "max_concurrency": conc,
    # The divisor for every per-chip figure. Taken from the server args because
    # the agentic path skips the random branch's post-processing, which is where
    # this used to be filled in.
    "tensor_parallel_size": _tp_from_server_args(),
    "output_sequence_length": avg("output_sequence_length"),
    "input_sequence_length": avg("input_sequence_length"),
    "random_range_ratio": "",
    "request_throughput": avg("request_throughput"),
    "mean_ttft_ms": avg("time_to_first_token"),
    "median_ttft_ms": pct("time_to_first_token", "p50"),
    "p90_ttft_ms": pct("time_to_first_token", "p90"),
    "p99_ttft_ms": pct("time_to_first_token", "p99"),
    "mean_itl_ms": avg("inter_token_latency"),
    "median_itl_ms": pct("inter_token_latency", "p50"),
    # p90 ITL is not decoration: InferenceX's headline interactivity number is
    # `1 / p90(inter_token_latency)` (utils/agentic/aggregation/request_metrics.py).
    # Without this field the metric can only be recovered by scraping AIPerf's
    # console table out of the job log, which is what we had to do.
    "p90_itl_ms": pct("inter_token_latency", "p90"),
    "p99_itl_ms": pct("inter_token_latency", "p99"),
    "mean_e2el_ms": avg("request_latency"),
    "median_e2el_ms": pct("request_latency", "p50"),
    "p90_e2el_ms": pct("request_latency", "p90"),
    "p99_e2el_ms": pct("request_latency", "p99"),
    "input_throughput": avg("input_token_throughput"),
    "output_throughput": avg("output_token_throughput"),
    "total_token_throughput": avg("total_token_throughput"),
    "successful_requests": avg("request_count"),
    "completed": avg("request_count"),
    "benchmark_duration_s": avg("benchmark_duration")
    or data.get("benchmark_duration_s"),
    "total_input_tokens": avg("total_usage_prompt_tokens"),
    "total_output_tokens": avg("total_usage_completion_tokens"),
    "cache_hit_tokens": cache_hit_tokens,
    "cache_total_tokens": cache_total_tokens,
    "cache_hit_rate": (
        round(cache_hit_tokens / cache_total_tokens, 4)
        if cache_hit_tokens is not None and cache_total_tokens
        else None
    ),
}

payload = {key: value for key, value in payload.items() if value is not None}
dst.write_text(json.dumps(payload, indent=2), encoding="utf-8")
if cache_hit_tokens is not None and cache_total_tokens:
    print(
        f"[aiperf] prefix cache hit: {cache_hit_tokens}/{cache_total_tokens} "
        f"tokens ({cache_hit_tokens / cache_total_tokens:.2%})"
    )
else:
    print(
        "[aiperf] prefix cache hit: unavailable "
        "(AIPerf profiling cache-read counters were not produced)"
    )
print(f"[aiperf] dashboard json: {dst}")
PY
}
