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
AIPERF_COMMIT="${AIPERF_COMMIT:-b7b16cf851885567988a643282266bce74e34437}"
AIPERF_SCENARIO="${AIPERF_SCENARIO:-inferencex-agentx-mvp}"
AIPERF_PUBLIC_DATASET="${AIPERF_PUBLIC_DATASET:-semianalysis_cc_traces_weka_062126}"
# Unset by default: the single-node recipe passes no --max-context-length.
# `-` not `:-` so a caller can blank it back out explicitly.
AIPERF_MAX_CONTEXT_LENGTH="${AIPERF_MAX_CONTEXT_LENGTH-}"
AIPERF_NUM_DATASET_ENTRIES="${AIPERF_NUM_DATASET_ENTRIES:-393}"
AIPERF_BENCHMARK_DURATION="${AIPERF_BENCHMARK_DURATION:-3600}"
AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="${AIPERF_TRACE_IDLE_GAP_CAP_SECONDS:-300}"
AIPERF_WARMUP_GRACE_PERIOD="${AIPERF_WARMUP_GRACE_PERIOD:-1800}"
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

ensure_aiperf() {
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
    --warmup-grace-period "${AIPERF_WARMUP_GRACE_PERIOD}" \
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
    "random_range_ratio": "",
    "request_throughput": avg("request_throughput"),
    "mean_ttft_ms": avg("time_to_first_token"),
    "median_ttft_ms": pct("time_to_first_token", "p50"),
    "p99_ttft_ms": pct("time_to_first_token", "p99"),
    "mean_itl_ms": avg("inter_token_latency"),
    "median_itl_ms": pct("inter_token_latency", "p50"),
    "p99_itl_ms": pct("inter_token_latency", "p99"),
    "mean_e2el_ms": avg("request_latency"),
    "median_e2el_ms": pct("request_latency", "p50"),
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
