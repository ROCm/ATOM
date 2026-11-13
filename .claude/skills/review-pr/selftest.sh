#!/usr/bin/env bash
# Guard the bot behaviours a refactor breaks silently, because none of them show up until a
# review fails at 2am:
#
#   1. Failure triage. Every agent failure used to collapse into `fail glm`, so a review that
#      merely ran out of wall clock paged the model owner about a backend that was answering in
#      under a second. A timeout must route to flow; only a real backend fault routes to glm.
#   2. The refuter-timeout downgrade. The worker's card is finished work; a refuter timeout must
#      publish it rather than discard it -- but only by admitting it on the card, which is what
#      the independent gate enforces. If the card edit drifts, the downgrade would start hiding
#      unrefuted findings behind a normal-looking review line.
#   3. The watchdog's alarm and, just as importantly, its two silences -- see [watchdog] below.
#
# Usage: bash .claude/skills/review-pr/selftest.sh
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ok=0; bad=0
t() {  # <name> <got> <want>
  if [ "$2" = "$3" ]; then echo "  ✅ $1"; ok=$((ok + 1))
  else echo "  ❌ $1 — got '$2', want '$3'"; bad=$((bad + 1)); fi
}

defaults() {  # emit run_one.sh's own default assignments, so a harness cannot drift from them
  sed -n '/^: "${ATOM_RUN_BUDGET/p; /^: "${ATOM_AGENT_TIMEOUT/p' "$S/run_one.sh"
}

echo "[failure triage]"
# Run agent_fail straight out of run_one.sh with fail() stubbed, so the test reads the shipping
# code rather than a copy of it.
route() {  # <rc> -> the failure class agent_fail picked
  { echo 'set -euo pipefail'
    defaults
    echo 'fail() { echo "$1"; exit 0; }'
    sed -n '/^agent_fail()/,/^}/p' "$S/run_one.sh"
    echo 'agent_fail worker "$1" 2'
  } | bash -s -- "$1"
}
t "a timeout routes to flow" "$(route 124)" "flow"
t "a backend fault routes to glm" "$(route 1)" "glm"

# The line above only proves agent_fail routes a 124 correctly. It says nothing about whether
# run_agent still PRODUCES a 124 on a timeout -- and if that regresses to a plain 1, every
# timeout silently becomes a backend fault again while these checks stay green. Drive a real
# timeout through run_agent instead of trusting its return path.
real_timeout() {
  local out; out=$(mktemp)
  { echo 'set -uo pipefail'
    echo 'say() { :; }'
    echo 'PROJ=/tmp'
    echo 'DEADLINE=$(( $(date +%s) + 3600 ))'    # plenty of budget; the agent itself times out
    echo 'ATOM_AGENT_TIMEOUT=1'
    defaults
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    printf 'run_agent probe /dev/null %s sh -c "sleep 5"; echo $?\n' "$out"
  } | bash
  rm -f "$out"
}
t "run_agent itself reports 124 on a real timeout" "$(real_timeout)" "124"

echo "[run budget]"
# AGENT_TIMEOUT x RETRIES x (worker+refuter) can exceed the workflow's timeout-minutes, and a run
# killed at the job cap dies with no classified status at all. run_agent must refuse to start an
# attempt that cannot fit, and must report it as the timeout it is.
budget_guard() {
  { echo 'set -uo pipefail'
    echo 'say() { :; }'
    echo 'PROJ=/tmp'
    echo 'DEADLINE=0'                  # the whole budget is already spent
    echo 'ATOM_AGENT_TIMEOUT=2400'
    defaults
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    echo 'run_agent probe /dev/null /dev/null true; echo $?'
  } | bash
}
t "an attempt that cannot fit the budget reports a timeout" "$(budget_guard)" "124"

# Out-of-budget and ran-out-of-clock both return 124 so neither gets retried, but they have
# opposite fixes. Telling someone to raise ATOM_AGENT_TIMEOUT for a budget refusal makes the
# refusal fire sooner -- the message must name the knob that actually helps.
fail_message() {  # <DEADLINE> <ATOM_AGENT_TIMEOUT> <cmd...> -> the text the owner is paged with
  { echo 'set -uo pipefail'
    echo 'say() { :; }'; echo 'PROJ=/tmp'
    printf 'DEADLINE=%s\n' "$1"; printf 'ATOM_AGENT_TIMEOUT=%s\n' "$2"; shift 2
    defaults
    echo 'fail() { echo "$3"; exit 0; }'
    sed -n '/^run_agent()/,/^}/p' "$S/run_one.sh"
    sed -n '/^agent_fail()/,/^}/p' "$S/run_one.sh"
    printf 'rc=0; run_agent probe /dev/null /tmp/_st_out %s || rc=$?\n' "$*"
    echo 'agent_fail probe "$rc" 2'
  } | bash
}
budget_msg=$(fail_message 0 2400 true)
clock_msg=$(fail_message "$(( $(date +%s) + 3600 ))" 1 sh -c '"sleep 5"')
t "a budget refusal names the run budget" \
  "$(printf '%s' "$budget_msg" | grep -c ATOM_RUN_BUDGET)" "1"
t "a budget refusal does not blame the agent timeout" \
  "$(printf '%s' "$budget_msg" | grep -c 'raise it in the runner .env')" "0"
t "a real timeout still names the agent timeout" \
  "$(printf '%s' "$clock_msg" | grep -c ATOM_AGENT_TIMEOUT)" "1"

echo "[gates call site]"
# Same bug as the refuter had: `python3 _gates.py ...` bare under `set -euo pipefail` exits the
# script on a red gate, so the EXIT trap reports "exited unexpectedly with no classified
# failure" -- while the log says plainly which gate went red. A reader gets the useless half.
gates_site() {  # <rc from _gates.py> -> what the call site reported
  local W; W=$(mktemp -d)
  # the fixture must not be the file the code tees to: tee truncates it on open, and the
  # stub's cat then races against that -- a test that passes or fails by timing is worse than none
  printf '======== SEVEN GATES ========\n  \342\234\205 answers\n  \342\235\214 card         exit=1\n' > "$W/fixture.out"
  { echo 'set -euo pipefail'
    echo 'say() { :; }'
    echo 'fail() { echo "$1 | $3"; exit 0; }'
    printf 'W=%s\nSKILL=%s\nPROJ=/tmp\n' "$W" "$S"
    printf 'python3() { cat "%s/fixture.out"; return %s; }\n' "$W" "$1"
    sed -n '/^# 4) gates + collect/,/^python3 "\$SKILL\/_collect.py"/p' "$S/run_one.sh"
    echo 'echo "REACHED-END"'
  } | bash
  rm -rf "$W"
}
t "a red gate is reported as a classified failure" \
  "$(gates_site 1 | grep ' | ' | cut -d'|' -f1 | tr -d ' ')" "flow"
t "the report names which gate went red" \
  "$(gates_site 1 | grep ' | ' | grep -c card)" "1"
t "green gates fall through to collect" "$(gates_site 0 | tail -1)" "REACHED-END"

# A gate that prints more than one line carrying `exit=` must not be named twice: a reader
# shown "card,card,ledger" learns less than one shown "card,ledger".
gates_multi() {
  local W; W=$(mktemp -d)
  printf '   \342\235\214 card    exit=1\n   \342\235\214 card    exit=1\n   \342\235\214 ledger  exit=1\n' > "$W/fixture.out"
  { echo 'set -euo pipefail'
    echo 'say() { :; }'
    echo 'fail() { echo "$3"; exit 0; }'
    printf 'W=%s\nSKILL=%s\nPROJ=/tmp\n' "$W" "$S"
    printf 'python3() { cat "%s/fixture.out"; return 1; }\n' "$W"
    sed -n '/^# 4) gates + collect/,/^python3 "\$SKILL\/_collect.py"/p' "$S/run_one.sh"
  } | bash | grep ' gate'
  rm -rf "$W"
}
t "a gate red on several lines is named once" \
  "$(gates_multi | grep -oE 'by the [a-z,]+ gate' | head -1)" "by the card,ledger gate"

echo "[refuter call site]"
# The harnesses above run run_agent in isolation under `set -uo pipefail`. The shipping script
# runs under `set -euo pipefail`, where a bare `cmd; rc=$?` exits before rc is ever read -- so
# proving the function returns 124 says nothing about whether the caller survives to act on it.
# Drive the call site itself, errexit on, exactly as run_one.sh has it.
refuter_site() {  # <rc run_agent returns> [card's first line] -> what the call site actually did
  local W; W=$(mktemp -d)
  # `Review:` is the shape SKILL.md's card format specifies and the one real cards carry.
  # These fixtures used `Review (advisory):`, a Validation-line shape, so the downgrade path
  # was exercised against a card that cannot occur -- which is why a sed anchored on the
  # wrong prefix passed here while doing nothing in production.
  printf '%s\n\342\232\240 a finding\n' "${2:-Review: \342\232\240 NEEDS WORK}" > "$W/card.md"
  { echo 'set -euo pipefail'
    echo 'say() { :; }'
    echo 'bash() { :; }'                       # stub out render.sh
    echo 'agent_fail() { echo "agent_fail $1 $2"; exit 0; }'
    printf 'run_agent() { return %s; }\n' "$1"
    printf 'SKILL=%s\nW=%s\n' "$S" "$W"
    echo 'REFUTER_CMD=(true)'
    defaults
    sed -n '/^  bash "\$SKILL\/render.sh" refuter/,/^  fi$/p' "$S/run_one.sh"
    echo 'grep -q "not independently refuted" "$W/card.md" && echo downgraded || echo no-downgrade'
  } | bash
  rm -rf "$W"
}
t "a refuter timeout reaches the downgrade" "$(refuter_site 124)" "downgraded"
# The card is model-generated and the downgrade is a sed anchored to one literal shape. If the
# worker bolds or indents that line the sed matches nothing, exits 0, and the independent gate
# then discards a card that was known-good -- the one case the downgrade exists to rescue.
t "a card the sed cannot match is still downgraded" \
  "$(refuter_site 124 '**Review:** \342\232\240 NEEDS WORK')" "downgraded"
t "a refuter fault reaches the triage" "$(refuter_site 1)" "agent_fail refuter 1"

echo "[refuter-timeout downgrade]"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
printf 'NONE AVAILABLE -- the refuter did not finish within 2400s on this box\n' > "$tmp/independent.txt"

# The downgrade applies exactly the sed run_one.sh uses.
printf 'Review: \342\232\240 NEEDS WORK\n\342\232\240 a reported finding\n' > "$tmp/card.md"
printf '\nReview (not independently refuted): the refuter did not finish within 2400s\n' >> "$tmp/card.md"
rc=0; python3 "$S/triage.py" independent "$tmp/independent.txt" "$tmp/card.md" >/dev/null 2>&1 || rc=$?
t "downgraded card passes the independent gate" "$rc" "0"

# And the guard still bites: an unavailable refutation the card does not admit must stay red.
printf 'Review (advisory): \342\232\240 NEEDS WORK\n\342\232\240 a reported finding\n' > "$tmp/card.md"
rc=0; python3 "$S/triage.py" independent "$tmp/independent.txt" "$tmp/card.md" >/dev/null 2>&1 || rc=$?
t "a silent downgrade is rejected" "$rc" "1"

echo "[watchdog]"
# The watchdog is the one guard that cannot be tested by running it: `issue_comment` workflows
# only ever run from the default branch, so GitHub will not execute that job until this merges.
# Drive the shipping script against a fake API and a fake clock instead.
wd=$(python3 "$S/watchdog_test.py" 2>&1); wd_rc=$?
printf '%s
' "$wd" | grep -E '^  (✅|❌)' || true
wok=$(printf '%s
' "$wd" | grep -c '✅' || true)
wbad=$(printf '%s
' "$wd" | grep -c '❌' || true)
ok=$((ok + wok)); bad=$((bad + wbad))
if [ "$wd_rc" -ne 0 ] && [ "$wbad" -eq 0 ]; then
  why=$(printf '%s
' "$wd" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ watchdog_test.py did not run — ${why:-exit $wd_rc}"; bad=$((bad + 1))
fi

echo "[queue gate]"
# Same reason as the watchdog: the gate only runs from the default branch, so GitHub will not
# execute this until it merges. Drive the shipping script against a fake API and a fixed clock.
qg=$(python3 "$S/queue_gate_test.py" 2>&1); qg_rc=$?
printf '%s\n' "$qg" | grep -E '^  (✅|❌)' || true
qok=$(printf '%s\n' "$qg" | grep -c '✅' || true)
qbad=$(printf '%s\n' "$qg" | grep -c '❌' || true)
ok=$((ok + qok)); bad=$((bad + qbad))
if [ "$qg_rc" -ne 0 ] && [ "$qbad" -eq 0 ]; then
  why=$(printf '%s\n' "$qg" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ queue_gate_test.py did not run — ${why:-exit $qg_rc}"; bad=$((bad + 1))
fi

echo "[notify handoff]"
# The report job stays quiet when _notify.py says it already spoke, so that flag must mean a
# comment actually landed. _notify.py returns 0 without posting in four cases -- no status
# file, no token, a POST that threw, and a POST GitHub answered 2xx without creating a comment
# -- and claiming "notified" in any of them would trade a visible failure for a silent one.
# Two of the four need no network and are checked here; _notify_test.py drives main() over a
# fake transport for the two that do.
notified_flag() {  # <status file contents or empty> -> what _notify.py wrote to GITHUB_OUTPUT
  local w o; w=$(mktemp -d); o=$(mktemp)
  [ -n "$1" ] && printf '%s\n' "$1" > "$w/.atom-review-status"
  ( cd "$w" && GITHUB_WORKSPACE="$w" GITHUB_OUTPUT="$o" ATOM_BOT_TOKEN= GITHUB_REPOSITORY= \
      python3 "$S/_notify.py" 42 >/dev/null 2>&1 )
  local n; n=$(grep -c notified "$o" 2>/dev/null || true)   # grep -c exits 1 on zero matches
  echo "${n:-0}"
  rm -rf "$w" "$o"
}
t "no status to report claims nothing" "$(notified_flag '')" "0"
t "a report it could not send claims nothing" "$(notified_flag 'flow\tsomething broke')" "0"

echo "[lost review]"
# The other end of the watchdog: a review that started and then vanished. Same reason it cannot
# be exercised here -- the job only runs from the default branch.
lr=$(python3 "$S/lost_review_test.py" 2>&1); lr_rc=$?
printf '%s\n' "$lr" | grep -E '^  (✅|❌)' || true
lok=$(printf '%s\n' "$lr" | grep -c '✅' || true)
lbad=$(printf '%s\n' "$lr" | grep -c '❌' || true)
ok=$((ok + lok)); bad=$((bad + lbad))
if [ "$lr_rc" -ne 0 ] && [ "$lbad" -eq 0 ]; then
  why=$(printf '%s\n' "$lr" | grep -m1 -E 'Error|Traceback|No such file' | sed 's/^ *//' | cut -c1-110)
  echo "  ❌ lost_review_test.py did not run — ${why:-exit $lr_rc}"; bad=$((bad + 1))
fi

echo "[applies.txt recheck]"
# run_one.sh re-checks `git apply` on the merge-target worktree (checked out at the real BASE_SHA)
# because fetch.sh's own check ran against PROJECT_ROOT's current checkout instead. That recheck
# must not clobber a richer diagnosis fetch.sh already wrote -- PR#2705 lost fetch.sh's
# double-application explanation to the generic "needs a rebase" text this way once already.
applies_recheck() {  # <git-apply exit code> <applies.txt fetch.sh already wrote> -> final applies.txt
  local W; W=$(mktemp -d)
  mkdir -p "$W/merge-target"
  echo "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef" > "$W/merge_target.txt"
  printf '%s\n' "$2" > "$W/applies.txt"
  { echo 'set -uo pipefail'
    echo 'say() { :; }'
    printf 'W=%s\n' "$W"
    printf 'git() { return %s; }\n' "$1"
    sed -n '/^if \[ -d "\$W\/merge-target" \]; then/,/^fi$/p' "$S/run_one.sh"
    echo 'cat "$W/applies.txt"'
  } | bash 2>/dev/null
  rm -rf "$W"
}
double_app_msg="STALE: no longer applies to merge target abc -- but this PR was ALREADY
  MERGED via xyz, and abc is downstream of that merge commit. This
  is not drift needing a rebase -- it is double-application: ..."
t "recheck STALE keeps fetch.sh's double-application diagnosis" \
  "$(applies_recheck 1 "$double_app_msg" | grep -c 'double-application')" "1"
t "recheck STALE does not also write the generic rebase text over a double-application one" \
  "$(applies_recheck 1 "$double_app_msg" | grep -c 'needs a rebase,')" "0"
t "recheck STALE still writes the generic text when fetch.sh's own verdict was generic" \
  "$(applies_recheck 1 'STALE: no longer applies to merge target abc -- the PR needs a rebase,' \
     | grep -c 'needs a rebase,')" "1"
t "recheck APPLIES overwrites with the rechecked-tree message regardless" \
  "$(applies_recheck 0 "$double_app_msg" | grep -c 'rechecked on the merge-target worktree')" "1"

echo "[GLM health probe]"
# ANTHROPIC_BASE_URL promises nothing about shape. An on-box OpenAI-compatible GLM answers the
# first probe; an Anthropic-style gateway (serves /v1/messages) needs the second shape tried
# before concluding anything -- and a 404 on BOTH is a probe/shape mismatch, not proof the
# backend is dead, so it must warn and continue rather than page the model owner (class glm).
probe() {  # <1st curl http_code> <2nd curl http_code, or none> -> combined stdout+stderr
  local c1="$1" c2="${2:-000}" cnt; cnt=$(mktemp)
  { echo 'set -uo pipefail'
    echo 'say() { echo "SAY:$*"; }'
    echo 'fail() { echo "FAIL:$1:$3"; exit 5; }'
    echo 'ATOM_GLM_PROBE_TIMEOUT=1'
    echo 'ANTHROPIC_BASE_URL=http://example.invalid'
    # _code / _code2 are each captured via `$(curl ...)`, which forks a subshell -- a plain
    # counter variable bumped inside curl() would reset every call instead of advancing. A
    # file survives the fork.
    printf '_cnt=%q; _c1=%q; _c2=%q\n' "$cnt" "$c1" "$c2"
    echo 'curl() { local n; n=$(( $(cat "$_cnt") + 1 )); echo "$n" > "$_cnt"; if [ "$n" = 1 ]; then printf "%s" "$_c1"; else printf "%s" "$_c2"; fi; }'
    sed -n '/^if \[ -n "\${ANTHROPIC_BASE_URL/,/^fi$/p' "$S/run_one.sh"
  } | ( echo 0 > "$cnt"; bash ) 2>&1
  rm -f "$cnt"
}
t "OpenAI shape 200 on the first try: healthy, no second probe needed" \
  "$(probe 200 | grep -c 'GLM ok')" "1"
t "OpenAI shape 404, Anthropic shape 200: healthy via the second shape" \
  "$(probe 404 200 | grep -c 'GLM ok')" "1"
t "both shapes 404: warns and proceeds, does not abort" \
  "$(probe 404 404 | grep -c '^FAIL:')" "0"
t "both shapes 404: the warning says probe/shape mismatch, not glm class" \
  "$(probe 404 404 | grep -c 'probe/shape mismatch')" "1"
t "connection failure then 404: still aborts (class glm) -- not both-404" \
  "$(probe 000 404 | grep -c '^FAIL:glm:')" "1"
t "both shapes 500: still aborts (class glm) -- a real backend fault, not a 404" \
  "$(probe 500 500 | grep -c '^FAIL:glm:')" "1"

# Every rule id derive() can emit must be in ALL_RULES. Nothing checked this, and a rename
# proved why it matters: the P->N section rename updated rules.md, MAPPING.md and every
# hit() call, but ALL_RULES is one space-separated string and kept the old ids. Selftest
# stayed 93 green. ALL_RULES is the fallback list handed to the worker when derivation
# fails, so the stale version would have shipped five ids with no body in rules.md --
# a broken fallback that only fires on the path nobody exercises.
_rule_ids_consistent() {
  python3 - "$S" <<'PYEOF'
import inspect, re, sys
sys.path.insert(0, sys.argv[1])
import triage
src = inspect.getsource(triage.derive)
emitted = set()
# hit("family-name", "A1 B2 ...") -- the rule list is the second argument, and rule ids
# are an uppercase letter run followed by digits, so the fragments a looser pattern
# picked up (IE, ule) cannot match.
for m in re.finditer(r"hit\([^)]*?,\s*.((?:[A-Z]+[0-9]+[a-z]?(?: |(?=.)))+).\)", src):
    emitted |= set(m.group(1).split())
missing = sorted(emitted - set(triage.ALL_RULES.split()))
print("ok" if emitted and not missing else ("empty" if not emitted else "missing:" + ",".join(missing)))
PYEOF
}
t "every id derive() emits is in ALL_RULES" "$(_rule_ids_consistent)" "ok"

# derive() must not read this skill's own source. rules.md and SKILL.md quote the tokens
# they describe, so a diff that only edits the reviewer used to derive the dtype, Triton,
# stream and GPU-buffer families for a diff with none of that surface: 20 rules on
# ATOM#2762, every one inapplicable, and a worker can mistake a noise adjudication for a
# real finding. The mixed case is the one that matters -- skill prose must not pollute a
# diff that also touches real code.
_rules_of() {  # <diff> -> sorted rule ids (what pollution actually changes)
  printf '%s' "$1" | python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
fams = triage.derive(triage.parse(sys.stdin.read()))
print(" ".join(sorted({r for _, rr in fams for r in rr.split() if r != "NONE"})))' "$S"
}
_fams2() {  # <diff> -> family names
  printf '%s' "$1" | python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
print(" ".join(f for f, _ in triage.derive(triage.parse(sys.stdin.read()))))' "$S"
}
_kern='diff --git a/atom/model_ops/moe.py b/atom/model_ops/moe.py
--- a/atom/model_ops/moe.py
+++ b/atom/model_ops/moe.py
@@ -1,2 +1,3 @@
+    num_warps = 4
'
# A .py under .claude/, NOT a .md: a markdown path matches INFRA on its extension alone,
# so testing with rules.md would pass with the .claude/ prefix rule removed.
# A .py under .claude/, NOT a .md: a markdown path matches INFRA on its extension alone,
# so a .md fixture would pass with the .claude/ prefix rule removed.
_skillcode='diff --git a/.claude/skills/review-pr/triage.py b/.claude/skills/review-pr/triage.py
--- a/.claude/skills/review-pr/triage.py
+++ b/.claude/skills/review-pr/triage.py
@@ -1,2 +1,3 @@
+# prose quoting tl.program_id and fp8_e8m0 and num_warps and cudaMemcpy
'
t "a skills-only diff derives no rules" "$(_fams2 "$_skillcode")" "infra-only"

# A diff too large for one pass must SAY so, above the findings. Seven consecutive reviews
# of ATOM#2762 (47 files / 16.6k added) each reported five findings -- the same count a
# 385-line PR gets. The number saturates, so five reads as "these are the problems" when
# it means "five of them". Thresholds are the measured top of 40 merged ATOM PRs (max 19
# files, 3369 added), not a guess.
_cov() {  # <files> <added-lines-per-file> -> the coverage verdict word
  python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
n, per = int(sys.argv[2]), int(sys.argv[3])
d = "".join(
    f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1,1 +1,{per} @@\n"
    + "".join(f"+line{j}\n" for j in range(per))
    for i in range(n)
)
line = triage.coverage_line(triage.parse(d))
print("BEYOND" if "BEYOND ONE PASS" in line else "OK")' "$S" "$1" "$2"
}
t "a normal-sized diff is within one pass"   "$(_cov 6 50)"    "OK"
t "too many files is flagged as a sample"    "$(_cov 40 10)"   "BEYOND"
t "too many added lines is flagged too"      "$(_cov 5 1000)"  "BEYOND"

# Tests do not count toward reviewability, on either axis. A change of 300 source lines
# with 900 lines of table-driven tests is not what a reviewer loses track of, and counting
# them would flag precisely the PRs that are best covered. The two counts exclude tests
# together so they cannot disagree -- three source files beside nine test files is three
# files of review surface.
_cov_mixed() {  # <src files> <src lines> <test files> <test lines> -> OK|BEYOND
  python3 -c '
import sys; sys.path.insert(0, sys.argv[1]); import triage
sf, sl, tf, tl = (int(x) for x in sys.argv[2:6])
def blk(p, n):
    return f"diff --git a/{p} b/{p}\n--- a/{p}\n+++ b/{p}\n@@ -1,1 +1,{n} @@\n" + "".join(f"+l{i}\n" for i in range(n))
d = "".join(blk(f"atom/f{i}.py", sl) for i in range(sf)) + "".join(blk(f"tests/test_{i}.py", tl) for i in range(tf))
line = triage.coverage_line(triage.parse(d))
print("BEYOND" if "BEYOND ONE PASS" in line else "OK")' "$S" "$1" "$2" "$3" "$4"
}
t "a test-heavy diff is judged on its source only" "$(_cov_mixed 1 300 1 900)" "OK"
t "many test files do not make a diff unreviewable" "$(_cov_mixed 3 10 9 10)" "OK"
t "source lines alone can still exceed the bar"     "$(_cov_mixed 1 600 0 0)" "BEYOND"

# Measuring the diff is not enough: the card has to SAY it. ATOM#3034's card carried the
# coverage line and ATOM#2762's did not -- same worker, same instruction -- and the one
# that dropped it was the 47-file diff where "a sample, not a sweep" mattered most. So the
# card gate enforces it against what Step 1 measured, including the softening case: a card
# that reports a BEYOND ONE PASS diff as an ordinary one is worse than omitting the line.
_cov_gate() {  # <rules.txt coverage line> <card coverage line or -> problem kind or ok
  python3 -c '
import sys, tempfile, pathlib
sys.path.insert(0, sys.argv[1]); import triage
d = tempfile.mkdtemp()
pathlib.Path(d, "rules.txt").write_text("  files=9\n" + sys.argv[2] + "\n")
card = "## T\n\nReview: OK\n" + ("" if sys.argv[3] == "-" else sys.argv[3] + "\n")
p = triage._check_coverage_line(card, d)
print(p[0][0] if p else "ok")' "$S" "$1" "$2"
}
_BIG="  coverage: BEYOND ONE PASS -- 47 files (no ATOM PR merged in the last 40 exceeded 19)"
_SMALL="  coverage: 5 files / 49 added lines -- within one review pass"
t "a card that drops the coverage line is refused" "$(_cov_gate "$_BIG" -)" "MISSING-COVERAGE"
t "a card that softens BEYOND ONE PASS is refused" \
  "$(_cov_gate "$_BIG" 'Coverage: 47 files / 16647 added lines')" "SOFTENED-COVERAGE"
t "a card carrying the measured coverage passes" \
  "$(_cov_gate "$_SMALL" 'Coverage: 5 files / 49 added lines -- within one review pass')" "ok"

# The pollution case needs REAL skill prose, not a one-line stand-in: a synthetic hunk
# trips no family and the test passes with the filter removed -- it did, until this was
# measured. testdata/skill_prose.diff is three files cut from ATOM#2762's own diff, and
# with the content filter off they add 18 rule ids (Triton, stream, dtype) to a diff
# whose only real code is one kernel line.
t "skill prose does not pollute a real-code diff" \
  "$(_rules_of "$_kern$(cat "$S/testdata/skill_prose.diff")")" \
  "$(_rules_of "$_kern")"

# Rejected validation evidence must degrade the review, not delete it. The regression this
# pins is one that actually happened on the first live auto-validation run: the consumer ran
# under errexit, so one missing optional import aborted fetch.sh and produced no card at all
# -- indistinguishable, to the human waiting on it, from the bot being down. _degrade_test.sh
# extracts the real consumer region out of fetch.sh by line markers (not a replica of it) and
# drives it with a report for the wrong head, which must be refused AND survived.
t "rejected validation evidence degrades instead of aborting" \
  "$(bash "$S/_degrade_test.sh" "$S" 2>&1 | tail -1)" "ok"

# A fetch that dies must say why, in the log, in terms the runner can act on. The live run
# this pins: a spent GitHub API quota ended the log mid-sentence after "fetch..." -- errexit
# aborted inside the pipeline before the classification could run, and the exit trap wrote a
# status file while printing nothing. "Check gh auth/token" would also have been the wrong
# advice: the token was fine and a quota was empty, which resets on a clock.
t "a dead fetch names the cause the runner can act on" \
  "$(bash "$S/_fetchfail_test.sh" "$S" 2>&1 | tail -1)" "ok"

# check_citations.py's main() is executed here. The 34 pytest cases covered check(); main()
# was called by nothing, so a NameError in it survived review and a green gate -- citations.txt
# carried a traceback, and the publish step read that as "nothing flagged" and anchored freely.
t "the citation checker runs end to end" \
  "$(bash "$S/_citations_test.sh" "$S" 2>&1 | tail -1)" "ok"

# check_prompts must work outside a git checkout. It asked git for the repo root, and
# `git rev-parse --show-toplevel` prints an empty string when there is no repo -- making
# the SKILL.md path RELATIVE to the caller's cwd, so the drift check died with
# FileNotFoundError on a `git archive HEAD` export. That export is exactly how a reviewer
# is told to verify, and every trial that passed had been inside a checkout.
_drift_outside_a_repo() {
  # The whole skill tree, at the same depth it sits in the repo, with no .git anywhere --
  # what `git archive HEAD | tar -x` produces. Checks whose inputs are absent skip and
  # still exit 0; what must not happen is the crash that an empty repo-root produced.
  # Reconstruct what the check actually needs -- the skill tree at its repo depth, plus
  # the workflow it reads -- in a directory with no .git. Deliberately NOT `git archive`:
  # this suite is itself meant to run from an export, where there is no repo to archive
  # from, and a guard that only works inside a checkout cannot test leaving one.
  local t; t=$(mktemp -d)
  mkdir -p "$t/.claude/skills" "$t/.github/workflows"
  cp -r "$S" "$t/.claude/skills/review-pr"
  rm -rf "$t/.claude/skills/review-pr/reports"
  cp "$S"/../../../.github/workflows/*review-bot.yml "$t/.github/workflows/" 2>/dev/null
  ( cd /tmp && python3 "$t/.claude/skills/review-pr/check_prompts.py" >/dev/null 2>&1; echo $? )
  rm -rf "$t"
}
t "the drift check runs outside a git checkout" "$(_drift_outside_a_repo)" "0"


# ATOM_AGENT_TIMEOUT and ATOM_RUN_BUDGET are coupled and nothing else enforces it. A review
# runs two agents in series and the budget gate refuses to START one that cannot finish, so
# raising the per-agent timeout alone leaves the refuter unable to start: the worker runs for
# an hour, the independent gate reds, the review is withheld, and the failure is reported as a
# timeout whose named fix makes it worse. Hit for real while re-reviewing ATOM#2762.
_budget_pair() {  # <agent> <budget> -> rc of run_one's precondition
  ( ATOM_AGENT_TIMEOUT=$1 ATOM_RUN_BUDGET=$2 \
    timeout 20 bash "$S/run_one.sh" 0 x/y >/dev/null 2>&1; echo $? )
}
t "a per-agent timeout that starves the refuter is refused" "$(_budget_pair 5400 7200)" "2"

# A cross-family refuter must be reachable by doing what the comment says. The hook has
# existed since the port and nothing had ever run through it: the refuter fell back to the
# worker's own binary on every recorded run (41+ findings judged, zero killed), and a
# same-family refutation produces artifacts identical to a cross-family one, so no reader
# could tell. The wrapper exists because the value must be ONE executable -- run_one
# appends its own flags -- which a bare `claude-glm --backend kimi` silently is not.
t "the cross-family refuter entrypoint is executable" \
  "$([ -x "$S/refuter-kimi" ] && echo yes || echo no)" "yes"
t "SAME_FAMILY flips when a different refuter is configured" \
  "$(ATOM_REVIEW_AGENT=a ATOM_REVIEW_REFUTER_AGENT=b bash -c '
      AGENT="${ATOM_REVIEW_AGENT:-claude-glm}"
      REFUTER_BIN="${ATOM_REVIEW_REFUTER_AGENT:-$AGENT}"
      [ "$REFUTER_BIN" = "$AGENT" ] && echo 1 || echo 0')" "0"
t "SAME_FAMILY is set when the refuter is left unconfigured" \
  "$(ATOM_REVIEW_AGENT=a bash -c '
      AGENT="${ATOM_REVIEW_AGENT:-claude-glm}"
      REFUTER_BIN="${ATOM_REVIEW_REFUTER_AGENT:-$AGENT}"
      [ "$REFUTER_BIN" = "$AGENT" ] && echo 1 || echo 0')" "1"
echo "[publish]"
# The publish path's guards are pytest cases (_publish_test.py): the --post closure, the
# anchoring rule (span anchors and the permalinks for what cannot be anchored included), and
# the degrade-to-summary fallback. Run them from here so they are not a
# file nobody executes -- a test suite that only runs when someone remembers it is the same
# kind of silence this script exists to break. pytest is not part of the runner's baseline,
# so its absence is reported out loud rather than passed over: a check that did not run is
# not a check that passed, and it is counted red in neither direction here, it is SAID.
if python3 -c "import pytest" 2>/dev/null; then
  pt=$(cd "$S" && python3 -m pytest _publish_test.py -q 2>&1); pt_rc=$?
  n=$(printf '%s\n' "$pt" | grep -oE '[0-9]+ passed' | grep -oE '^[0-9]+' || echo 0)
  if [ "$pt_rc" -eq 0 ]; then
    echo "  ✅ _publish_test.py: ${n:-0} passed (--post closure, anchoring + spans + permalinks, degrade-to-summary)"
    ok=$((ok + 1))
  else
    printf '%s\n' "$pt" | grep -E '^(FAILED|ERROR)' | sed 's/^/  ❌ /' || true
    echo "  ❌ _publish_test.py failed — run: python3 -m pytest $S/_publish_test.py"
    bad=$((bad + 1))
  fi
else
  echo "  ⏭️  _publish_test.py NOT RUN — no pytest on this box. The --post closure, the"
  echo "      anchoring rule (spans, permalinks) and the degrade-to-summary fallback are"
  echo "      unverified here."
fi

echo "=== $ok green / $bad red ==="
[ "$bad" -eq 0 ]
