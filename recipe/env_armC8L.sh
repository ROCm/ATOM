# ARM C8 + ATOM_P1_SP_TBO_HOSTLOG=1 (milestone5 e2e, round 18): identical to C8 except that rank 0 logs one
# "[sp-tbo host]" line per SP prefill forward (kind off/gen, rows, host wall/cpu ms, alloc retries, reserved GiB;
# no device sync) so the TBO step count / ratio can be read from the server log. Used on ONE C8 point only.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env_armC8.sh"
export ATOM_P1_SP_TBO_HOSTLOG=1
