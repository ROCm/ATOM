#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${SCENARIO:-pd-chat}"
BENCHMARK_NAME="${BENCHMARK_NAME:-${SCENARIO}}"
DURATION="${DURATION:-20s}"
KILL_AFTER="${KILL_AFTER:-30s}"
PRODUCER_THREADS="${PRODUCER_THREADS:-1}"
CONSUMER_THREADS="${CONSUMER_THREADS:-8}"
PREFILL_WORKERS="${PREFILL_WORKERS:-1}"
DECODE_WORKERS="${DECODE_WORKERS:-1}"
POLICY="${POLICY:-round_robin}"
RESULT_DIR="${RESULT_DIR:-atomesh-mocker-results}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MESH_DIR="${REPO_ROOT}/atom/mesh"
MOCKER_DIR="${MESH_DIR}/mocker"
MOCKER_TARGET_DIR="${MOCKER_DIR}/target/mocker"
MESH_TARGET_DIR="${MOCKER_DIR}/target/mesh"
ATOMESH_BIN="${MESH_TARGET_DIR}/release/atomesh"
MOCKER_BIN="${MOCKER_TARGET_DIR}/release/atomesh-mocker"
LOG_DIR="${RESULT_DIR}/logs/${BENCHMARK_NAME}"
FIXTURE="${MOCKER_DIR}/fixtures/http_pd_chat.json"
ROUTER_MODE="pd"
WORKERS=$((PREFILL_WORKERS + DECODE_WORKERS))

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

if [[ "${SCENARIO}" != "pd-chat" ]]; then
  echo "Unsupported SCENARIO=${SCENARIO}; this benchmark script only runs pd-chat" >&2
  exit 2
fi

if (( PREFILL_WORKERS < 1 || DECODE_WORKERS < 1 )); then
  echo "PREFILL_WORKERS and DECODE_WORKERS must both be >= 1" >&2
  exit 2
fi

duration_seconds() {
  local value="$1"
  local number
  local multiplier
  case "${value}" in
    *s) number="${value%s}"; multiplier=1 ;;
    *m) number="${value%m}"; multiplier=60 ;;
    *h) number="${value%h}"; multiplier=3600 ;;
    *[!0-9]* | "") echo "Invalid duration '${value}'; use an integer with s, m, or h" >&2; return 2 ;;
    *) number="${value}"; multiplier=1 ;;
  esac

  if [[ ! "${number}" =~ ^[0-9]+$ || "${number}" -eq 0 ]]; then
    echo "Invalid duration '${value}'; use a positive integer with s, m, or h" >&2
    return 2
  fi

  echo "$((number * multiplier))"
}

DURATION_SECONDS="$(duration_seconds "${DURATION}")"
BENCH_TIMEOUT="$((DURATION_SECONDS + 30))s"

pick_ports() {
  python3 - <<'PY'
import socket

def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

print(free_port(), free_port(), free_port())
PY
}

wait_http() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 100); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  echo "${name} did not become ready at ${url}" >&2
  return 1
}

cleanup() {
  local status=$?
  if [[ -n "${ROUTER_PID:-}" ]]; then
    kill -INT "${ROUTER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${WORKER_PID:-}" ]]; then
    kill -INT "${WORKER_PID}" 2>/dev/null || true
  fi
  wait "${ROUTER_PID:-}" 2>/dev/null || true
  wait "${WORKER_PID:-}" 2>/dev/null || true
  exit "${status}"
}
trap cleanup EXIT

read -r ROUTER_PORT WORKER_BASE_PORT PROMETHEUS_PORT < <(pick_ports)

if [[ ! -x "${MOCKER_BIN}" || ! -x "${ATOMESH_BIN}" ]]; then
  echo "Missing release binaries. Build them before running this benchmark script." >&2
  echo "  MOCKER_BIN=${MOCKER_BIN}" >&2
  echo "  ATOMESH_BIN=${ATOMESH_BIN}" >&2
  exit 2
fi

echo "=== Starting virtual workers for ${BENCHMARK_NAME} (${PREFILL_WORKERS}P${DECODE_WORKERS}D) ==="
"${MOCKER_BIN}" virtual-workers \
  --ip 127.0.0.1 \
  --base-port "${WORKER_BASE_PORT}" \
  --workers "${WORKERS}" \
  "${FIXTURE}" \
  > "${LOG_DIR}/virtual-workers.log" 2>&1 &
WORKER_PID=$!
for index in $(seq 0 $((WORKERS - 1))); do
  wait_http "http://127.0.0.1:$((WORKER_BASE_PORT + index))/health" "virtual worker ${index}"
done

echo "=== Starting Atomesh router (${ROUTER_MODE}) ==="
COMMON_ROUTER_ARGS=(
  launch
  --host 127.0.0.1
  --port "${ROUTER_PORT}"
  --policy "${POLICY}"
  --worker-startup-timeout-secs 10
  --worker-startup-check-interval 1
  --request-timeout-secs 30
  --disable-retries
  --disable-circuit-breaker
  --health-check-interval-secs 300
  --prometheus-port "${PROMETHEUS_PORT}"
  --log-level warn
)

pd_worker_args=(--pd-disaggregation)
for index in $(seq 0 $((PREFILL_WORKERS - 1))); do
  pd_worker_args+=(--prefill "http://127.0.0.1:$((WORKER_BASE_PORT + index))")
done
for index in $(seq 0 $((DECODE_WORKERS - 1))); do
  pd_worker_args+=(--decode "http://127.0.0.1:$((WORKER_BASE_PORT + PREFILL_WORKERS + index))")
done

"${ATOMESH_BIN}" "${COMMON_ROUTER_ARGS[@]}" \
  "${pd_worker_args[@]}" \
  --prefill-policy "${POLICY}" \
  --decode-policy "${POLICY}" \
  > "${LOG_DIR}/atomesh.log" 2>&1 &
ROUTER_PID=$!
wait_http "http://127.0.0.1:${ROUTER_PORT}/health" "Atomesh router"

echo "=== Running request benchmark ${BENCHMARK_NAME} for ${DURATION} ==="
BENCH_LOG="${LOG_DIR}/benchmark-request.log"
set +e
timeout --signal=TERM --kill-after="${KILL_AFTER}" "${BENCH_TIMEOUT}" \
  "${MOCKER_BIN}" benchmark-request \
    --base-url "http://127.0.0.1:${ROUTER_PORT}" \
    --duration "${DURATION}" \
    --producer-threads "${PRODUCER_THREADS}" \
    --consumer-threads "${CONSUMER_THREADS}" \
    "${FIXTURE}" \
    > "${BENCH_LOG}" 2>&1
bench_status=$?
set -e

if [[ "${bench_status}" -ne 0 && "${bench_status}" -ne 124 && "${bench_status}" -ne 130 ]]; then
  echo "benchmark-request failed with status ${bench_status}" >&2
  exit "${bench_status}"
fi

echo "=== Parsing benchmark metrics ==="
RESULT_JSON="${RESULT_DIR}/${BENCHMARK_NAME}.json"
ACTION_JSON="${RESULT_DIR}/${BENCHMARK_NAME}-benchmark-action.json"
SUMMARY_MD="${RESULT_DIR}/${BENCHMARK_NAME}.md"

python3 - <<'PY' \
  "${BENCH_LOG}" "${RESULT_JSON}" "${ACTION_JSON}" "${SUMMARY_MD}" \
  "${SCENARIO}" "${FIXTURE}" "${ROUTER_MODE}" "${DURATION}" \
  "${PRODUCER_THREADS}" "${CONSUMER_THREADS}" "${WORKERS}" "${POLICY}" \
  "${BENCHMARK_NAME}" "${PREFILL_WORKERS}" "${DECODE_WORKERS}"
from datetime import UTC, datetime
import json
import os
import re
import sys
from pathlib import Path

(
    bench_log,
    result_json,
    action_json,
    summary_md,
    scenario,
    fixture,
    router_mode,
    duration,
    producer_threads,
    consumer_threads,
    workers,
    policy,
    benchmark_name,
    prefill_workers,
    decode_workers,
) = sys.argv[1:]

text = Path(bench_log).read_text(encoding="utf-8", errors="replace")
metric_lines = [
    line for line in text.splitlines()
    if re.match(r"^all\s+\d+\s+\d+\s+\d+\s+", line)
]
if not metric_lines:
    print(text)
    raise SystemExit("No aggregate metrics line found in benchmark log")

fields = metric_lines[-1].split()
total = int(fields[1])
success = int(fields[2])
failed = int(fields[3])
avg_ms = float(fields[4])
p99_ms = float(fields[5])
p999_ms = float(fields[6])
one_second_qps = float(fields[8])
one_minute_qps = float(fields[10])
five_minute_qps = float(fields[12])

seconds_match = re.match(r"^(\d+)([smh]?)$", duration)
duration_seconds = None
if seconds_match:
    value = int(seconds_match.group(1))
    unit = seconds_match.group(2) or "s"
    duration_seconds = value * {"s": 1, "m": 60, "h": 3600}[unit]

request_throughput = (
    success / duration_seconds
    if duration_seconds and duration_seconds > 0
    else one_minute_qps
)

payload = {
    "date": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
    "benchmark_backend": "Atomesh-Mocker",
    "dashboard_backend": "Atomesh-Mocker",
    "benchmark_model_name": benchmark_name,
    "benchmark_name": benchmark_name,
    "scenario": scenario,
    "fixture": str(Path(fixture).name),
    "router_mode": router_mode,
    "connection_mode": "http",
    "policy": policy,
    "producer_threads": int(producer_threads),
    "consumer_threads": int(consumer_threads),
    "workers": int(workers),
    "prefill_workers": int(prefill_workers),
    "decode_workers": int(decode_workers),
    "duration_seconds": duration_seconds,
    "completed": success,
    "failed": failed,
    "request_throughput": request_throughput,
    "output_throughput": request_throughput,
    "total_token_throughput": request_throughput,
    "avg_latency_ms": avg_ms,
    "mean_ttft_ms": avg_ms,
    "mean_tpot_ms": avg_ms,
    "p99_latency_ms": p99_ms,
    "p999_latency_ms": p999_ms,
    "one_second_qps": one_second_qps,
    "one_minute_qps": one_minute_qps,
    "five_minute_qps": five_minute_qps,
    "total": total,
}
Path(result_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

run_url = ""
server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
repository = os.environ.get("GITHUB_REPOSITORY")
run_id = os.environ.get("GITHUB_RUN_ID")
if repository and run_id:
    run_url = f"{server_url}/{repository}/actions/runs/{run_id}"

extra_parts = [
    f"cell={benchmark_name}",
    f"router={router_mode}",
    f"policy={policy}",
    f"workers={workers}",
    f"prefill={prefill_workers}",
    f"decode={decode_workers}",
    f"producers={producer_threads}",
    f"consumers={consumer_threads}",
    f"duration_seconds={duration_seconds}",
    f"request_number={success}",
]
if run_url:
    extra_parts.append(f"Run: {run_url}")
extra = " ".join(extra_parts)

entries = []
for metric_name, unit, value in [
    ("request throughput", "req/s", request_throughput),
    ("avg latency", "ms", avg_ms),
    ("p99 latency", "ms", p99_ms),
    ("p999 latency", "ms", p999_ms),
    ("failed requests", "count", failed),
]:
    entries.append(
        {
            "name": f"Atomesh-Mocker::{benchmark_name} {metric_name}",
            "unit": unit,
            "value": round(float(value), 2),
            "extra": extra,
        }
    )
Path(action_json).write_text(json.dumps(entries, indent=2), encoding="utf-8")

summary = f"""### Atomesh Mocker Benchmark: {benchmark_name}

| Metric | Value |
| --- | ---: |
| scenario | {scenario} |
| router mode | {router_mode} |
| workers | {workers} |
| prefill/decode workers | {prefill_workers}/{decode_workers} |
| producer/consumer threads | {producer_threads}/{consumer_threads} |
| completed | {success} |
| failed | {failed} |
| request throughput | {request_throughput:.2f} req/s |
| avg latency | {avg_ms:.3f} ms |
| p99 latency | {p99_ms:.3f} ms |
| p999 latency | {p999_ms:.3f} ms |
"""
Path(summary_md).write_text(summary, encoding="utf-8")
print(summary)
PY

echo "Result JSON: ${RESULT_JSON}"
