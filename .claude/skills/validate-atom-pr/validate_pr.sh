#!/usr/bin/env bash
# validate-atom-pr: base-vs-head pytest comparison, the whole of its scope. No performance
# scraping, no independent shape grid -- those are review-pr/fetch.sh's own territory (the perf
# fallback recipe it prints) or explicitly out of scope (see report_schema.json's descriptions
# on correctness_s1_grid / execution_receipt). What it does: apply --patch onto the --repo
# checkout's current HEAD (the "base"), run --target before and after, and say which of the
# four base/head outcomes happened. See SKILL.md for the verdict table.
#
# usage: validate_pr.sh --repo <worktree> --patch <diff> --head-sha <sha> --target <pytest
#                        target> --label <str> --out <report.json>
# fetch.sh is the only caller this needs to match; see .claude/skills/review-pr/fetch.sh around
# lines 400-460 for exactly how it invokes this and consumes the report.
#
# Contract this validator keeps no matter what goes wrong: BLOCK/NEEDS_WORK/INCONCLUSIVE all
# still write a report worth consuming by fetch.sh's gate -- only a missing file at --out means
# there was nothing to read. Exit code: 0 PASS, 2 INCONCLUSIVE, 1 otherwise (BLOCK/NEEDS_WORK),
# matching fetch.sh's own expected_exit contract exactly.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

REPO=""
PATCH=""
HEAD_SHA=""
TARGET=""
LABEL=""
OUT=""
# Passthrough args fetch.sh's auto-validation path can add (REVIEW_EXPECTED_ROUTE etc.) when
# set in its environment. validate-atom-pr does no route/grid instrumentation, so these are
# accepted (never a hard CLI error if the caller passes them) and, if given, are threaded
# straight into test_selection -- recorded as a caller-asserted hint, not a claim this
# validator itself verified. Without them, test_selection honestly says "not-derived".
EXPECTED_ROUTE=""
SHAPE_VARS=""
SHAPE_ENV=""
SHAPE_ARG=""
SHAPE_ARGNAMES=""
GRID=""

while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --patch) PATCH=$2; shift 2 ;;
    --head-sha) HEAD_SHA=$2; shift 2 ;;
    --target) TARGET=$2; shift 2 ;;
    --label) LABEL=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --expected-route) EXPECTED_ROUTE=$2; shift 2 ;;
    --shape-vars) SHAPE_VARS=$2; shift 2 ;;
    --shape-env) SHAPE_ENV=$2; shift 2 ;;
    --shape-arg) SHAPE_ARG=$2; shift 2 ;;
    --shape-argnames) SHAPE_ARGNAMES=$2; shift 2 ;;
    --grid) GRID=$2; shift 2 ;;
    *) echo "validate_pr.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

for pair in REPO:--repo PATCH:--patch HEAD_SHA:--head-sha TARGET:--target LABEL:--label OUT:--out; do
  var=${pair%%:*}
  flag=${pair#*:}
  if [ -z "${!var}" ]; then
    echo "validate_pr.sh: $flag is required" >&2
    exit 2
  fi
done

if [ ! -d "$REPO/.git" ] && ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  echo "validate_pr.sh: --repo $REPO is not a git checkout" >&2
  exit 2
fi
if [ ! -r "$PATCH" ]; then
  echo "validate_pr.sh: --patch $PATCH is not readable" >&2
  exit 2
fi

STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/validate-atom-pr-XXXXXX")
trap 'rm -rf "$WORK"' EXIT

BASE_SHA=$(git -C "$REPO" rev-parse HEAD)
PATCH_SHA256=$(python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$PATCH")

REPO_KIND="atom"
[ -f "$REPO/atom/__init__.py" ] || REPO_KIND="unknown"

write_report() {  # <python-args-json-file>
  python3 - "$HERE" "$1" "$OUT" <<'PY'
import importlib.util
import json
import sys

here, facts_path, out_path = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("report_lib", f"{here}/report_lib.py")
report_lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report_lib)

with open(facts_path) as fh:
    facts = json.load(fh)

if facts["kind"] == "inconclusive":
    report = report_lib.build_inconclusive_report(
        label=facts["label"],
        started_utc=facts["started_utc"],
        finished_utc=facts["finished_utc"],
        repo=facts["repo"],
        runtime_identity=facts["runtime_identity"],
        test_selection=facts["test_selection"],
        stages=facts["stages"],
        reason=facts["reason"],
        blocking_stage=facts["blocking_stage"],
        degraded_mode=facts.get("degraded_mode"),
    )
else:
    report = report_lib.build_report(
        label=facts["label"],
        started_utc=facts["started_utc"],
        finished_utc=facts["finished_utc"],
        repo=facts["repo"],
        runtime_identity=facts["runtime_identity"],
        test_selection=facts["test_selection"],
        stages=facts["stages"],
        base_stats=facts["base_stats"],
        head_stats=facts["head_stats"],
        gpu_arch=facts["gpu_arch"],
        base_missing=facts.get("base_missing", False),
    )

with open(out_path, "w") as fh:
    json.dump(report, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(f"verdict: {report['verdict']} (exit {report['process_exit_code']})")
sys.exit(report["process_exit_code"])
PY
}

bail_inconclusive() {  # <reason> <blocking_stage> <extra-stages-json> [degraded_mode]
  local reason=$1 stage=$2 extra_stages=$3 degraded=${4:-null}
  python3 -c "
import json, sys
json.dump({
    'kind': 'inconclusive',
    'label': sys.argv[1],
    'started_utc': sys.argv[2],
    'finished_utc': sys.argv[3],
    'repo': json.loads(sys.argv[4]),
    'runtime_identity': json.loads(sys.argv[5]),
    'test_selection': json.loads(sys.argv[6]),
    'stages': json.loads(sys.argv[7]),
    'reason': sys.argv[8],
    'blocking_stage': sys.argv[9],
    'degraded_mode': json.loads(sys.argv[10]),
}, open(sys.argv[11], 'w'))
" "$LABEL" "$STARTED_UTC" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$REPO_JSON" "$RUNTIME_IDENTITY_JSON" "$TEST_SELECTION_JSON" "$extra_stages" \
    "$reason" "$stage" "$degraded" "$WORK/facts.json"
  write_report "$WORK/facts.json"
  exit 2
}

REPO_JSON=$(python3 -c "
import json, sys
json.dump({
    'worktree': sys.argv[1], 'base': sys.argv[2], 'head': sys.argv[3],
    'kind': sys.argv[4], 'patch_sha256': sys.argv[5],
}, sys.stdout)
" "$REPO" "$BASE_SHA" "$HEAD_SHA" "$REPO_KIND" "$PATCH_SHA256")

TEST_SELECTION_JSON=$(python3 -c "
import json, sys
d = {'target': sys.argv[1], 'runner': 'pytest', 'runner_reason':
     'validate-atom-pr always invokes --target as a pytest node/file, per its --target contract with review-pr/fetch.sh'}
if sys.argv[2]:
    d['expected_route'] = sys.argv[2]
if sys.argv[3]:
    d['shape_vars'] = sys.argv[3]
if sys.argv[4]:
    d['shape_env'] = sys.argv[4]
if sys.argv[5]:
    d['shape_arg'] = sys.argv[5]
if sys.argv[6]:
    d['shape_argnames'] = sys.argv[6]
if sys.argv[7]:
    d['grid'] = sys.argv[7]
json.dump(d, sys.stdout)
" "$TARGET" "$EXPECTED_ROUTE" "$SHAPE_VARS" "$SHAPE_ENV" "$SHAPE_ARG" "$SHAPE_ARGNAMES" "$GRID")

# ---------------------------------------------------------------------------------------
# gpu_claim -- claim a device and record which one, honestly. rocm-smi's product-name lookup
# is broken on this box ("Error when calling libdrm", device name comes back empty), so arch
# is read from torch (gcnArchName) instead. Idleness is measured with `rocm-smi --showuse`
# when that call itself succeeds; if it does not, idleness_basis is "unknown" rather than an
# invented classification.
# ---------------------------------------------------------------------------------------
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
GPU_INDEX=""
IDLE_BASIS="unknown"
GFX_PCT="null"
LOCK_DIR="${TMPDIR:-/tmp}/validate-atom-pr-gpu-locks"
mkdir -p "$LOCK_DIR"

ACT_JSON=""
if command -v rocm-smi >/dev/null 2>&1; then
  ACT_JSON=$(rocm-smi --showuse --json 2>/dev/null) || ACT_JSON=""
fi

if [ "$GPU_COUNT" -gt 0 ]; then
  i=0
  while [ "$i" -lt "$GPU_COUNT" ]; do
    exec {LOCKFD}>"$LOCK_DIR/gpu$i.lock"
    if flock -n "$LOCKFD"; then
      busy=""
      if [ -n "$ACT_JSON" ]; then
        busy=$(python3 -c "
import json, sys
d = json.loads(sys.argv[1])
card = d.get(f'card{sys.argv[2]}', {})
print(card.get('GPU use (%)', ''))
" "$ACT_JSON" "$i" 2>/dev/null)
      fi
      if [ -z "$ACT_JSON" ] || [ "$busy" = "0" ]; then
        GPU_INDEX=$i
        [ -n "$ACT_JSON" ] && IDLE_BASIS="activity" && GFX_PCT=$busy
        # keep LOCKFD open (held) for the lifetime of this process; released at process exit
        break
      fi
    fi
    exec {LOCKFD}>&-
    i=$((i + 1))
  done
fi

DEGRADED_MODE="null"
if [ -z "$GPU_INDEX" ]; then
  DEGRADED_MODE='"NO_GPU"'
  GPU_INDEX=0  # still export visibility to a device below; no claim of idleness is made
fi

export HIP_VISIBLE_DEVICES="$GPU_INDEX"
export ROCR_VISIBLE_DEVICES="$GPU_INDEX"
GPU_ARCH=$(python3 -c "
import torch
try:
    print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])
except Exception:
    print('')
" 2>/dev/null)

GPU_CLAIM_STATUS="pass"
[ "$DEGRADED_MODE" = '"NO_GPU"' ] && GPU_CLAIM_STATUS="fail"
GPU_STAGE_JSON=$(python3 -c "
import json, sys
json.dump({
    'status': sys.argv[1], 'hip_index': int(sys.argv[2]), 'arch': sys.argv[3],
    'idleness_basis': sys.argv[4],
    'gfx_activity_pct': (None if sys.argv[5] == 'null' else int(sys.argv[5])),
}, sys.stdout)
" "$GPU_CLAIM_STATUS" "$GPU_INDEX" "$GPU_ARCH" "$IDLE_BASIS" "$GFX_PCT")

if [ "$GPU_CLAIM_STATUS" != "pass" ]; then
  STAGES_JSON=$(python3 -c "import json,sys; json.dump({'gpu_claim': json.loads(sys.argv[1])}, sys.stdout)" "$GPU_STAGE_JSON")
  RUNTIME_IDENTITY_JSON='{"module_path": "", "python_executable": "", "native_artifacts": []}'
  bail_inconclusive "no idle GPU could be claimed on this box" "gpu_claim" "$STAGES_JSON" "$DEGRADED_MODE"
fi

# ---------------------------------------------------------------------------------------
# runtime_compat -- does `import atom` resolve inside --repo, not some other installed copy?
# ---------------------------------------------------------------------------------------
RUNTIME_COMPAT_OK=0
MODULE_PATH=""
PYTHON_EXECUTABLE=$(command -v python3)
if MODULE_PATH=$(cd "$REPO" && python3 -c "
import atom, pathlib, sys
mod = pathlib.Path(atom.__file__).resolve()
root = pathlib.Path(sys.argv[1]).resolve()
if str(mod).startswith(str(root)):
    print(mod)
else:
    sys.exit(1)
" "$REPO" 2>/dev/null); then
  RUNTIME_COMPAT_OK=1
fi
RUNTIME_IDENTITY_JSON=$(python3 -c "
import json, sys
json.dump({
    'module_path': sys.argv[1], 'python_executable': sys.argv[2],
    'python_version': sys.argv[3], 'native_artifacts': [],
}, sys.stdout)
" "$MODULE_PATH" "$PYTHON_EXECUTABLE" "$(python3 -c 'import platform; print(platform.python_version())')")

if [ "$RUNTIME_COMPAT_OK" != "1" ]; then
  STAGES_JSON=$(python3 -c "
import json, sys
json.dump({
    'gpu_claim': json.loads(sys.argv[1]),
    'runtime_compat': {'status': 'fail', 'note': 'import atom did not resolve inside --repo'},
}, sys.stdout)
" "$GPU_STAGE_JSON")
  bail_inconclusive "import atom did not resolve inside the supplied --repo checkout" \
    "runtime_compat" "$STAGES_JSON"
fi

run_target() {  # <phase: base|head> -> writes $WORK/<phase>.json
  local phase=$1
  local junit="$WORK/$phase.junit.xml"
  local start end dur exit_code
  start=$(date +%s.%N)
  (cd "$REPO" && python3 -m pytest "$TARGET" --junit-xml="$junit" -q) \
    >"$WORK/$phase.log" 2>&1
  exit_code=$?
  end=$(date +%s.%N)
  dur=$(python3 - "$end" "$start" <<'PY'
import sys
print(round(float(sys.argv[1]) - float(sys.argv[2]), 3))
PY
)
  python3 -c "
import json, sys
import xml.etree.ElementTree as ET

exit_code, junit, dur = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
stats = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0, 'executed': 0}
try:
    root = ET.parse(junit).getroot()
    suite = root if root.tag == 'testsuite' else root.find('testsuite')
    if suite is not None:
        tests = int(suite.get('tests', 0))
        failures = int(suite.get('failures', 0))
        errors = int(suite.get('errors', 0))
        skipped = int(suite.get('skipped', 0))
        stats.update(
            tests=tests, failures=failures, errors=errors, skipped=skipped,
            executed=tests - skipped - errors,
        )
except (FileNotFoundError, ET.ParseError):
    pass
stats['duration_seconds'] = dur
json.dump({'exit': exit_code, 'stats': stats}, sys.stdout)
" "$exit_code" "$junit" "$dur" > "$WORK/$phase.json"
}

stat_clean() {  # <stats-json-file> -> prints 1/0: exit0 (via caller) and failures==0 and errors==0 and executed>=1
  python3 -c "
import json, sys
s = json.load(open(sys.argv[1]))['stats']
ok = s.get('executed', 0) >= 1 and s.get('failures', 0) == 0 and s.get('errors', 0) == 0
print(1 if ok else 0)
" "$1"
}

# ---------------------------------------------------------------------------------------
# baseline_control -- run the target at base, BEFORE the candidate patch is applied. This is
# the control: without it, "head fails" and "was already failing" are indistinguishable.
#
# A target --target names can be new in this PR (fetch.sh's own triage picks exactly this
# case: the one test file the PR *adds*). When the target file does not exist at base, there
# is no baseline to run -- baseline_control is marked 'skip' with a reason, not a fabricated
# pass or a hollow fail, and BASE_MISSING carries that fact into the verdict table so a
# failing new test reads as this PR's own regression (BLOCK), never as "pre-existing"
# (NEEDS_WORK).
# ---------------------------------------------------------------------------------------
TARGET_FILE="${TARGET%%::*}"
BASE_MISSING=0
if [ ! -e "$REPO/$TARGET_FILE" ]; then
  BASE_MISSING=1
  echo '{"exit": 1, "stats": {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "executed": 0, "duration_seconds": 0}}' \
    > "$WORK/base.json"
  BASELINE_STAGE_JSON='{"status": "skip", "note": "target does not exist at base (added by this PR); no baseline comparison is possible"}'
else
  run_target base
  BASE_EXIT=$(python3 -c "import json;print(json.load(open('$WORK/base.json'))['exit'])")
  BASE_STATS_JSON=$(python3 -c "import json;print(json.dumps(json.load(open('$WORK/base.json'))['stats']))")
  BASELINE_STAGE_JSON=$(python3 -c "
import json, sys
json.dump({'status': 'pass', 'exit': int(sys.argv[1]), 'stats': json.loads(sys.argv[2])}, sys.stdout)
" "$BASE_EXIT" "$BASE_STATS_JSON")
fi

# ---------------------------------------------------------------------------------------
# merge_sim -- apply the candidate patch onto base. The patch is always reverted before this
# script exits (trap below), whether the run that follows succeeds or not, so the worktree
# fetch.sh handed over comes back out at the same base it went in on.
# ---------------------------------------------------------------------------------------
PATCH_APPLIED=0
revert_patch() {
  if [ "$PATCH_APPLIED" = "1" ]; then
    git -C "$REPO" apply -R --whitespace=nowarn "$PATCH" 2>/dev/null || \
      git -C "$REPO" checkout -- . 2>/dev/null || true
    PATCH_APPLIED=0
  fi
}
trap 'revert_patch; rm -rf "$WORK"' EXIT

if git -C "$REPO" apply --whitespace=nowarn "$PATCH" 2>"$WORK/apply.err"; then
  PATCH_APPLIED=1
  MERGE_SIM_STATUS="pass"
else
  MERGE_SIM_STATUS="fail"
fi

if [ "$MERGE_SIM_STATUS" != "pass" ]; then
  STAGES_JSON=$(python3 -c "
import json, sys
json.dump({
    'gpu_claim': json.loads(sys.argv[1]),
    'runtime_compat': {'status': 'pass'},
    'baseline_control': json.loads(sys.argv[2]),
    'merge_sim': {'status': 'fail', 'note': open(sys.argv[3]).read()[:2000]},
}, sys.stdout)
" "$GPU_STAGE_JSON" "$BASELINE_STAGE_JSON" "$WORK/apply.err")
  bail_inconclusive "candidate patch does not apply onto base; see merge_sim.note" \
    "merge_sim" "$STAGES_JSON"
fi

# ---------------------------------------------------------------------------------------
# test_policy -- the target collects at least one test item at head (after the patch), the
# authoritative check that this run can produce anything at all. Checked here, after
# merge_sim and not before, because a target this PR *adds* cannot collect before the patch
# is applied -- that is BASE_MISSING above, not a test_policy failure.
# ---------------------------------------------------------------------------------------
COLLECTED=$(cd "$REPO" && python3 -m pytest "$TARGET" --collect-only -q 2>"$WORK/collect.err" \
  | grep -Eo '^[0-9]+ tests? collected' | grep -Eo '^[0-9]+' || true)
if [ -z "$COLLECTED" ] || [ "$COLLECTED" -lt 1 ]; then
  STAGES_JSON=$(python3 -c "
import json, sys
json.dump({
    'gpu_claim': json.loads(sys.argv[1]),
    'runtime_compat': {'status': 'pass'},
    'baseline_control': json.loads(sys.argv[2]),
    'merge_sim': {'status': 'pass'},
    'test_policy': {'status': 'fail', 'note': 'target collected no test items at head'},
}, sys.stdout)
" "$GPU_STAGE_JSON" "$BASELINE_STAGE_JSON")
  bail_inconclusive "--target $TARGET collected no test items at head" "test_policy" "$STAGES_JSON"
fi

# ---------------------------------------------------------------------------------------
# correctness_repo_tests -- run the target at head (base + candidate patch).
# ---------------------------------------------------------------------------------------
run_target head
HEAD_EXIT=$(python3 -c "import json;print(json.load(open('$WORK/head.json'))['exit'])")
HEAD_STATS_JSON=$(python3 -c "import json;print(json.dumps(json.load(open('$WORK/head.json'))['stats']))")
HEAD_CLEAN=$(stat_clean "$WORK/head.json")
HEAD_STAGE_STATUS="fail"
[ "$HEAD_EXIT" = "0" ] && [ "$HEAD_CLEAN" = "1" ] && HEAD_STAGE_STATUS="pass"

revert_patch
trap 'rm -rf "$WORK"' EXIT

BASE_STATS_JSON=$(python3 -c "import json;print(json.dumps(json.load(open('$WORK/base.json'))['stats']))")

STAGES_JSON=$(python3 -c "
import json, sys
json.dump({
    'gpu_claim': json.loads(sys.argv[1]),
    'runtime_compat': {'status': 'pass'},
    'baseline_control': json.loads(sys.argv[2]),
    'merge_sim': {'status': 'pass'},
    'test_policy': {'status': 'pass', 'note': sys.argv[3] + ' test item(s) collected at head'},
    'correctness_repo_tests': {'status': sys.argv[4], 'exit': int(sys.argv[5]), 'stats': json.loads(sys.argv[6])},
}, sys.stdout)
" "$GPU_STAGE_JSON" "$BASELINE_STAGE_JSON" "$COLLECTED" "$HEAD_STAGE_STATUS" "$HEAD_EXIT" "$HEAD_STATS_JSON")

FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python3 -c "
import json, sys
json.dump({
    'kind': 'report',
    'label': sys.argv[1], 'started_utc': sys.argv[2], 'finished_utc': sys.argv[3],
    'repo': json.loads(sys.argv[4]), 'runtime_identity': json.loads(sys.argv[5]),
    'test_selection': json.loads(sys.argv[6]), 'stages': json.loads(sys.argv[7]),
    'base_stats': json.loads(sys.argv[8]), 'head_stats': json.loads(sys.argv[9]),
    'gpu_arch': sys.argv[10], 'base_missing': bool(int(sys.argv[11])),
}, open(sys.argv[12], 'w'))
" "$LABEL" "$STARTED_UTC" "$FINISHED_UTC" "$REPO_JSON" "$RUNTIME_IDENTITY_JSON" \
  "$TEST_SELECTION_JSON" "$STAGES_JSON" "$BASE_STATS_JSON" "$HEAD_STATS_JSON" "$GPU_ARCH" \
  "$BASE_MISSING" \
  "$WORK/facts.json"

write_report "$WORK/facts.json"
RC=$?
exit "$RC"
