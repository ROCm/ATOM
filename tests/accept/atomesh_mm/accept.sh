#!/usr/bin/env bash
# Acceptance harness for ATOMesh multimodal requests.
#
#   bash tests/accept/atomesh_mm/accept.sh <model> <topology> <check>
#
# Sources models/<model>.env (image, model, nodes and GPUs per role, server
# arguments) and then checks/<check>.env when the check has one (what the
# check needs changed in the profile), brings the deployment up with
# topologies/<topology>.sh, runs checks/<check>.py (dashes read as
# underscores) in a client container against it, and removes its containers
# at exit. stdout carries one JSON line and then PASS or FAIL <reason>; the
# JSON is also kept as $ACCEPT_DIR/results/<model>/<topology>/<check>.json,
# where later checks read their references, beside the run's logs.
set -euo pipefail

if (($# != 3)); then
  echo "usage: $0 <model> <topology> <check>" >&2
  exit 2
fi
MODEL=$1 TOPOLOGY=$2 CHECK=$3
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
ACCEPT_ISSUE=${ACCEPT_ISSUE:-2}
CHECK_PY=$HERE/checks/${CHECK//-/_}.py

report() { # report <reason>: the JSON line and FAIL for a run that ended before its check
  python3 -c 'import json, sys; print(json.dumps(dict(zip(["model", "topology", "check", "reason"], sys.argv[1:]), passed=False)))' \
    "$MODEL" "$TOPOLOGY" "$CHECK" "$1"
  echo "FAIL $1"
}

for f in "$HERE/models/$MODEL.env" "$HERE/topologies/$TOPOLOGY.sh" "$CHECK_PY"; do
  [[ -f $f ]] || { report "no such file: ${f#"$HERE"/}"; exit 1; }
done

source "$HERE/models/$MODEL.env"
CHECK_ENV=${CHECK_PY%.py}.env
[[ ! -f $CHECK_ENV ]] || source "$CHECK_ENV"
source "$HERE/lib.sh"
REF_TOPOLOGY=""
source "$HERE/topologies/$TOPOLOGY.sh"

RESULTS_DIR=$ACCEPT_DIR/results
RUN_DIR=$RESULTS_DIR/$MODEL/$TOPOLOGY/$CHECK.$(date +%Y%m%d-%H%M%S)
mkdir -p "$RUN_DIR"
log "run directory $RUN_DIR"

# A failing step exits through set -e; the check reports for itself.
STAGE="$TOPOLOGY deployment"
on_exit() {
  local status=$?
  ((status == 0)) || [[ -z $STAGE ]] || report "$STAGE failed; logs in $RUN_DIR"
  cleanup_containers
}
trap on_exit EXIT

topology_up
STAGE="client container"
start_client
STAGE=""
in_role client \
  -e "ACCEPT_MODEL=$MODEL" -e "ACCEPT_TOPOLOGY=$TOPOLOGY" -e "ACCEPT_CHECK=$CHECK" \
  -e "ENDPOINT=$ENDPOINT" -e "SERVED_MODEL=$SERVED_MODEL" -e "RUN_DIR=$RUN_DIR" \
  -e "RESULTS_DIR=$RESULTS_DIR" -e "REF_TOPOLOGY=$REF_TOPOLOGY" -e "LIMIT=${LIMIT:-}" \
  -e "MMMU_PRO_REF_STANDARD=${MMMU_PRO_REF_STANDARD:-}" -e "MMMU_PRO_REF_VISION=${MMMU_PRO_REF_VISION:-}" \
  -- python3 "$CHECK_PY"
