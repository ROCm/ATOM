#!/usr/bin/env bash
# Local/CI preflight: syntax-check scatter_bounds verify + its embedded inner script.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
outer="${root}/.github/scripts/atomesh/run_scatter_bounds_verify.sh"
echo "bash -n outer: ${outer}"
bash -n "${outer}"
echo "OK outer"

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
# Extract the quoted heredoc body written as scatter_bounds_inner.sh
awk '
  /^cat > .*<<'\''INNER_EOF'\''$/ { flag=1; next }
  /^INNER_EOF$/ { flag=0; next }
  flag { print }
' "${outer}" > "${tmp}"
lines="$(wc -l < "${tmp}" | tr -d ' ')"
echo "bash -n inner (${lines} lines)"
if [[ "${lines}" -lt 20 ]]; then
  echo "ERROR: extracted inner script too short (${lines} lines)" >&2
  exit 1
fi
bash -n "${tmp}"
echo "OK inner"
echo "preflight bash -n PASSED"
