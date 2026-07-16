#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_atomesh.sh [workers] [base_port] [port] [worker_ip] [host]

Arguments:
  workers    Number of virtual workers Atomesh should route to. Defaults to 1.
  base_port  First virtual worker port. Defaults to 30010.
  port       Atomesh listen port. Defaults to 30000.
  worker_ip  Virtual worker IP or host. Defaults to 127.0.0.1.
  host       Atomesh bind host. Defaults to 127.0.0.1.

Environment overrides:
  WORKERS, BASE_PORT, ATOMESH_PORT, WORKER_IP, ATOMESH_HOST, POLICY
  TLS_CERT_PATH, TLS_KEY_PATH

Pass extra Atomesh args after "--":
  ./start_atomesh.sh 2 -- --log-level debug

Examples:
  ./start_atomesh.sh
  ./start_atomesh.sh 4
  TLS_CERT_PATH=./fullchain.pem TLS_KEY_PATH=./privkey.pem ./start_atomesh.sh 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MESH_DIR="$(cd "$MOCKER_DIR/.." && pwd)"

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

WORKERS="${POSITIONAL[0]:-${WORKERS:-1}}"
BASE_PORT="${POSITIONAL[1]:-${BASE_PORT:-30010}}"
ATOMESH_PORT="${POSITIONAL[2]:-${ATOMESH_PORT:-30000}}"
WORKER_IP="${POSITIONAL[3]:-${WORKER_IP:-127.0.0.1}}"
ATOMESH_HOST="${POSITIONAL[4]:-${ATOMESH_HOST:-127.0.0.1}}"
POLICY="${POLICY:-cache_aware}"

worker_urls=()
for ((index = 0; index < WORKERS; index++)); do
  worker_urls+=("http://$WORKER_IP:$((BASE_PORT + index))")
done

tls_args=()
if [[ -n "${TLS_CERT_PATH:-}" || -n "${TLS_KEY_PATH:-}" ]]; then
  if [[ -z "${TLS_CERT_PATH:-}" || -z "${TLS_KEY_PATH:-}" ]]; then
    echo "TLS_CERT_PATH and TLS_KEY_PATH must be provided together" >&2
    exit 1
  fi
  tls_args=(--tls-cert-path "$TLS_CERT_PATH" --tls-key-path "$TLS_KEY_PATH")
fi

echo "Starting Atomesh"
echo "Bind: $ATOMESH_HOST:$ATOMESH_PORT"
echo "Policy: $POLICY"
echo "Worker URLs: ${worker_urls[*]}"

exec cargo run --manifest-path "$MESH_DIR/Cargo.toml" -- launch \
  --host "$ATOMESH_HOST" \
  --port "$ATOMESH_PORT" \
  --policy "$POLICY" \
  --worker-urls "${worker_urls[@]}" \
  "${tls_args[@]}" \
  "${EXTRA_ARGS[@]}"
