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
# where later checks read their references, beside the run's logs. A
# topology that defines topology_counters adds their change over the check
# to that line; one that also defines topology_findings adds the fields it
# prints, and fails the run on its "failures".
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
RUN_CHECK=(in_role client
  -e "ACCEPT_MODEL=$MODEL" -e "ACCEPT_TOPOLOGY=$TOPOLOGY" -e "ACCEPT_CHECK=$CHECK"
  -e "ENDPOINT=$ENDPOINT" -e "SERVED_MODEL=$SERVED_MODEL" -e "RUN_DIR=$RUN_DIR"
  -e "RESULTS_DIR=$RESULTS_DIR" -e "REF_TOPOLOGY=$REF_TOPOLOGY" -e "LIMIT=${LIMIT:-}"
  -e "MMMU_PRO_REF_STANDARD=${MMMU_PRO_REF_STANDARD:-}" -e "MMMU_PRO_REF_VISION=${MMMU_PRO_REF_VISION:-}"
  -- python3 "$CHECK_PY")
if ! declare -F topology_counters >/dev/null; then
  "${RUN_CHECK[@]}"
  exit
fi

# A topology with counters: read them around the check and add their
# differences to its JSON line as counter_deltas, with the topology's
# findings on them (a JSON object of fields, its "failures" a list of
# reasons), in the files it recorded too (in the client container, which
# owns them). A counter or finding that cannot be read fails the run.
STAGE="$TOPOLOGY counter read"
BEFORE=$(topology_counters)
STATUS=0
OUTPUT=$("${RUN_CHECK[@]}") || STATUS=$?
AFTER=$(topology_counters)
DELTAS=$(python3 -c '
import json, sys
before, after = map(json.loads, sys.argv[1:])
print(json.dumps({k: after[k] - before[k] for k in after}))' "$BEFORE" "$AFTER")
FINDINGS={}
if declare -F topology_findings >/dev/null; then
  FINDINGS=$(topology_findings "$DELTAS")
fi
STAGE=""
MERGE='
import json, sys
from pathlib import Path
deltas, findings, results_dir, output = sys.argv[1:]
line, *verdict = output.splitlines()
result = json.loads(line)
result["counter_deltas"] = json.loads(deltas)
findings = json.loads(findings)
failures = findings.pop("failures", [])
result.update(findings)
if failures:
    reasons = [result["reason"]] if result["reason"] else []
    result["passed"] = False
    result["reason"] = "; ".join(reasons + failures)
    verdict = ["FAIL " + result["reason"]]
line = json.dumps(result)
recorded = Path(results_dir, result["model"], result["topology"], result["check"] + ".json")
for path in (Path(result["run_dir"], "result.json"), recorded):
    if path.exists() and json.loads(path.read_text())["run_dir"] == result["run_dir"]:
        path.write_text(line + "\n")
print(line, *verdict, sep="\n")
sys.exit(0 if result["passed"] else 1)
'
in_role client python3 -c "$MERGE" "$DELTAS" "$FINDINGS" "$RESULTS_DIR" "$OUTPUT" || STATUS=$?
exit "$STATUS"
