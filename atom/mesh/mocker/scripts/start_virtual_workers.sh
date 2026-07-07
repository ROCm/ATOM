#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  start_virtual_workers.sh [workers] [base_port] [fixture] [ip]

Arguments:
  workers    Number of virtual workers to start. Defaults to 1.
  base_port  First worker port. Workers use base_port + index. Defaults to 30010.
  fixture    Fixture JSON path. Defaults to fixtures/http_regular_chat.json.
  ip         Bind IP or host. Defaults to 127.0.0.1.

Environment overrides:
  WORKERS, BASE_PORT, FIXTURE, IP

Examples:
  ./start_virtual_workers.sh
  ./start_virtual_workers.sh 4
  ./start_virtual_workers.sh 2 30010 fixtures/http_regular_chat.json 127.0.0.1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOCKER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKERS="${1:-${WORKERS:-1}}"
BASE_PORT="${2:-${BASE_PORT:-30010}}"
FIXTURE="${3:-${FIXTURE:-fixtures/http_regular_chat.json}}"
IP="${4:-${IP:-127.0.0.1}}"

if [[ "$FIXTURE" != /* ]]; then
  FIXTURE="$MOCKER_DIR/$FIXTURE"
fi

echo "Starting $WORKERS virtual worker(s)"
echo "Bind IP: $IP"
echo "Base port: $BASE_PORT"
echo "Fixture: $FIXTURE"

exec cargo run --manifest-path "$MOCKER_DIR/Cargo.toml" -- virtual-workers \
  --ip "$IP" \
  --base-port "$BASE_PORT" \
  --workers "$WORKERS" \
  "$FIXTURE"
