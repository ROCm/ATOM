#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_virtual_requests.sh [base_url] [producer_threads] [consumer_threads] [fixture]

Arguments:
  base_url          Atomesh base URL. Defaults to http://127.0.0.1:30000.
  producer_threads  Number of request producer tasks. Defaults to 1.
  consumer_threads  Number of request consumer tasks. Defaults to 4.
  fixture           Fixture JSON path. Defaults to fixtures/http_regular_chat.json.

Environment overrides:
  BASE_URL, PRODUCER_THREADS, CONSUMER_THREADS, QUEUE_CAPACITY, FIXTURE, HOST_HEADER
  TLS_CA_CERT_PATH, TLS_ACCEPT_INVALID_CERTS

Pass extra atomesh-mocker args after "--":
  ./start_virtual_requests.sh http://127.0.0.1:30000 1 8 -- --host example.local

Examples:
  ./start_virtual_requests.sh
  ./start_virtual_requests.sh http://127.0.0.1:30000 2 16
  TLS_ACCEPT_INVALID_CERTS=1 ./start_virtual_requests.sh https://127.0.0.1:30000
  TLS_CA_CERT_PATH=./fullchain.pem ./start_virtual_requests.sh https://127.0.0.1:30000
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

POSITIONAL=()
EXTRA_ARGS=()
while (($#)); do
  if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
    break
  fi
  POSITIONAL+=("$1")
  shift
done

BASE_URL="${POSITIONAL[0]:-${BASE_URL:-http://127.0.0.1:30000}}"
PRODUCER_THREADS="${POSITIONAL[1]:-${PRODUCER_THREADS:-1}}"
CONSUMER_THREADS="${POSITIONAL[2]:-${CONSUMER_THREADS:-4}}"
FIXTURE="${POSITIONAL[3]:-${FIXTURE:-fixtures/http_regular_chat.json}}"
QUEUE_CAPACITY="${QUEUE_CAPACITY:-4096}"

if [[ "$FIXTURE" != /* ]]; then
  FIXTURE="$MOCKER_DIR/$FIXTURE"
fi

request_args=(
  benchmark-request
  --base-url "$BASE_URL"
  --producer-threads "$PRODUCER_THREADS"
  --consumer-threads "$CONSUMER_THREADS"
  --queue-capacity "$QUEUE_CAPACITY"
)

if [[ -n "${HOST_HEADER:-}" ]]; then
  request_args+=(--host "$HOST_HEADER")
fi

if [[ -n "${TLS_CA_CERT_PATH:-}" ]]; then
  request_args+=(--tls-ca-cert-path "$TLS_CA_CERT_PATH")
fi

if [[ "${TLS_ACCEPT_INVALID_CERTS:-0}" == "1" || "${TLS_ACCEPT_INVALID_CERTS:-}" == "true" ]]; then
  request_args+=(--tls-accept-invalid-certs)
fi

echo "Starting virtual request benchmark"
echo "Base URL: $BASE_URL"
echo "Producer threads: $PRODUCER_THREADS"
echo "Consumer threads: $CONSUMER_THREADS"
echo "Queue capacity: $QUEUE_CAPACITY"
echo "Fixture: $FIXTURE"

exec cargo run --manifest-path "$MOCKER_DIR/Cargo.toml" -- \
  "${request_args[@]}" \
  "${EXTRA_ARGS[@]}" \
  "$FIXTURE"
