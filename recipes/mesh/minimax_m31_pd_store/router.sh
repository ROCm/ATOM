#!/usr/bin/env bash
# atomesh P/D router. Run inside the decode container (the image ships atomesh).
#   POLICY=random     ATOM's CoreManager places each request on a decode DP rank
#                     (--dp-load-balance least_requests)
#   POLICY=dp_sticky  the router discovers the decode's 4 DP ranks (--dp-aware) and keeps a
#                     session (X-Session-ID) on one rank: multi-turn traffic
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
POLICY=${POLICY:-random}
DP=()
[[ "${POLICY}" == dp_sticky ]] && DP=(--dp-aware)
exec atomesh launch --host 0.0.0.0 --port "${ROUTER_PORT}" --pd-disaggregation \
  --prefill "http://${NODE_IP}:${P_PORT}" --decode "http://${NODE_IP}:${D_PORT}" \
  --policy "${POLICY}" "${DP[@]}" --backend atom --log-level info \
  --disable-circuit-breaker --prometheus-port 29100 > "${RUN}/router-${POLICY}.log" 2>&1
