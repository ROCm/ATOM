#!/usr/bin/env bash
# Headless review of one PR: fetch -> GLM worker -> GLM refuter -> gates -> collect.
# Lives in the review-pr skill dir and drives the rest of the skill (calls its fetch.sh and
# triage.py). Pure GLM on the box (claude-glm), no claude-mix. Publishing is a separate step
# (_publish.py); this script does not publish.
#
# This is the backend of the atom-bot review workflow, and can also be run by hand.
#
# ⚠️ Must run in a plain terminal / the runner, NOT inside an auto-mode Claude session — the
#    auto-mode classifier blocks a headless sub-agent launched with
#    --dangerously-skip-permissions as "Create Unsafe Agents". A runner is a plain shell.
#
# ⚠️ Cross-family strength: worker and refuter are both GLM (same family) here. The refuter's
#    value comes from being a different family (Opus) catching the GLM worker's mistakes. Pure
#    GLM is weaker adversarially; point REFUTER_CMD at Opus to restore the cross-family check.
set -euo pipefail

PR="${1:?usage: run_one.sh <pr> [owner/repo]}"
REPO="${2:-ROCm/ATOM}"
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .claude/skills/review-pr
PROJ="$(git -C "$SKILL" rev-parse --show-toplevel)"     # repo root
STATUS="${GITHUB_WORKSPACE:-$PROJ}/.atom-review-status"
rm -f "$STATUS"   # fresh run: never inherit a previous run's verdict
# GUARANTEE a responsible person is always identified: any non-zero exit OR a signal kill (the
# agent/job timeout kills with SIGTERM, which skips a plain EXIT trap) that did NOT classify
# itself still leaves a status so _notify.py routes it -- to the bot owner (flow) for triage --
# instead of dying silently. Trap both the exit and the terminating signals.
_on_exit() { local ec=$?; [ "$ec" -ne 0 ] && [ ! -f "$STATUS" ] && printf "flow\trun_one exited unexpectedly (code %s) with no classified failure -- see the job log\n" "$ec" > "$STATUS"; _drop_worktrees; return 0; }
_on_signal() { [ -f "$STATUS" ] || printf "flow\trun_one was killed by a signal (likely the job or agent timeout) -- see the job log\n" > "$STATUS"; _drop_worktrees; exit 143; }
# Keeping WORK on failure is deliberate (see the success-path cleanup at the end) --
# a human paged about a failure needs verdicts.txt and card.md to see what the run
# actually produced. But the two git worktrees under it are ~99.5% of its size AND
# leave registrations in $PROJ that nothing prunes on a failure path: on a long-lived
# self-hosted runner shared with aiter's bot, that grows without bound. Drop the
# checkouts (reproducible), keep the ~200K of text artifacts (not reproducible).
_drop_worktrees() {
  [ -n "${W:-}" ] || return 0
  for wt in "$W/merge-target" "$W/head"; do
    [ -d "$wt" ] && git -C "$PROJ" worktree remove --force "$wt" 2>/dev/null || true
  done
  git -C "$PROJ" worktree prune 2>/dev/null || true
}
trap _on_exit EXIT
trap _on_signal TERM INT

# The workflow's timeout-minutes is the real ceiling: AGENT_TIMEOUT x RETRIES x (worker+refuter)
# can exceed it, and a run killed at the job cap dies mid-agent with no classified status. Cap the
# retries against a deadline that sits inside it instead. Anchored here, at the start: the fetch,
# the prompt check and the health probe all spend the same wall clock the agents do, and a budget
# that starts counting after them can still overrun the job cap.
: "${ATOM_RUN_BUDGET:=7200}"      # 120 min, under the job's timeout-minutes of 150
: "${ATOM_AGENT_TIMEOUT:=2400}"   # 40 min per agent attempt; one default, read everywhere below
# Wider than aiter's 100/120: validate-atom-pr (P2) stands up two model-serving instances inside
# the same run, and that extra wall clock has to fit inside the budget alongside the review itself.
DEADLINE=$(( $(date +%s) + ATOM_RUN_BUDGET ))

# GLM-5.3 is not in Claude's model catalog; disable the unknown-model window enforcement.
# The container runs as root; declare the docker sandbox so headless tools work.
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1 CLAUDE_GLM_QUIET=1 IS_SANDBOX=1 PYTHONUTF8=1
# Kernel validation off unless a GPU of the matching arch is present; report the gap honestly.
export REVIEW_AUTO_VALIDATE="${REVIEW_AUTO_VALIDATE:-0}"

# gh's own login may be broken (GraphQL 401 observed) while ~/.git-credentials is fine. If so,
# hold the token in a PLAIN var and hand it only to fetch.sh's env below -- never `export` it, or
# the headless review agent (--dangerously-skip-permissions on untrusted PR content) inherits a
# GitHub token from its environment. RUNNER-SETUP.md's GH_CONFIG_DIR path avoids needing this.
GH_FALLBACK_TOK=""
if ! gh api repos/"$REPO" --jq .full_name >/dev/null 2>&1; then
  GH_FALLBACK_TOK=$(sed -n 's#https://\([^:]*\):\([^@]*\)@github.com#\2#p' ~/.git-credentials | head -1)
fi

# The box's Claude entrypoint. Default is claude-glm (resolves a remote GLM over a tunnel);
# on a box that hosts GLM itself, set ATOM_REVIEW_AGENT to a local `claude` pointed at the
# on-box endpoint (ANTHROPIC_BASE_URL=http://localhost:<port>), so no tunnel/wrapper is needed.
AGENT="${ATOM_REVIEW_AGENT:-claude-glm}"
WORKER_CMD=("$AGENT" -p --dangerously-skip-permissions)
# Cross-family (restores the refuter's value): set ATOM_REVIEW_REFUTER_AGENT to an Opus entrypoint.
REFUTER_CMD=("${ATOM_REVIEW_REFUTER_AGENT:-$AGENT}" -p --dangerously-skip-permissions)

# Whether the refuter is a different model family from the worker, decided here in the
# shell so no agent can report it. aiter's own note at the top of this file says the
# refuter's value comes from being a different family catching the worker's mistakes,
# and that pure GLM is weaker adversarially -- but a same-family refutation and a
# cross-family one produce identical artifacts, so a reader cannot tell which they got.
# Recorded as evidence, and said on the card, for the same reason the card admits when
# no independent reader was available at all.
REFUTER_BIN="${ATOM_REVIEW_REFUTER_AGENT:-$AGENT}"
if [ "$REFUTER_BIN" = "$AGENT" ]; then
  SAME_FAMILY=1
else
  SAME_FAMILY=0
fi
say() { echo "[run_one #$PR] $*"; }
# Classify a failed review and expose it so the notify step @-mentions the RIGHT owner:
#   atom -> GLM/backend | flow -> this review pipeline | env -> runner box.
# Who that resolves to is _notify.py's OWNERS table's job, not this comment's -- it drifted once
# naming a person by hand; see _notify.py's OWNERS for the live class -> owner map.
# Writes <class>TAB<message> to the status file the workflow reads, plus an ::error annotation.
fail() {  # <class> <exit-code> <message...>
  local cls="$1" code="$2"; shift 2
  say "$*"
  echo "::error title=rocm-atom-bot::[$cls] $*"
  printf '%s\t%s\n' "$cls" "$*" > "$STATUS"
  exit "$code"
}

# Fail fast if the prompts have drifted from SKILL.md (they quote it verbatim).
python3 "$SKILL/check_prompts.py" >/dev/null || fail flow 4 "prompts drifted from SKILL.md -- regenerate the prompt copies (check_prompts.py)"

# GLM health-gate: a review is worthless if the backend is down, and a dead GLM otherwise
# hangs ~40s per agent call and dies silently mid-review. When a direct endpoint is configured
# (ANTHROPIC_BASE_URL, i.e. an on-box GLM), deep-probe it with a REAL 1-token inference — not
# the shallow /health, which stays green while inference is wedged — and fail fast with a clear
# signal. claude-glm mode resolves its own endpoint, so there is nothing to probe here.
if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
  say "GLM health probe..."
  # ANTHROPIC_BASE_URL names where to reach the backend, not what shape it speaks: an on-box
  # GLM is usually OpenAI-compatible (/v1/chat/completions, tried first, below), but some boxes
  # put an Anthropic-style endpoint behind this same var (serves /v1/messages, its own headers)
  # -- e.g. a corporate gateway inherited into the environment. Forward any gateway-specific
  # headers the operator configured (ANTHROPIC_CUSTOM_HEADERS, "Name: value, Name2: value2" --
  # same format Claude Code itself reads) to both probe shapes.
  _hdrs=(-H 'Content-Type: application/json')
  if [ -n "${ANTHROPIC_CUSTOM_HEADERS:-}" ]; then
    while IFS= read -r _h; do
      [ -n "$_h" ] && _hdrs+=(-H "$_h")
    done < <(printf '%s' "$ANTHROPIC_CUSTOM_HEADERS" | tr ',' '\n' | sed 's/^ *//')
  fi
  _code=$(curl -s -m "${ATOM_GLM_PROBE_TIMEOUT:-30}" --noproxy '*' \
    "$ANTHROPIC_BASE_URL/v1/chat/completions" "${_hdrs[@]}" \
    -d "{\"model\":\"${ANTHROPIC_MODEL:-/models/GLM-5.3}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
    -o /dev/null -w '%{http_code}' 2>/dev/null || true)
  _code2=""
  if [ "$_code" != "200" ]; then
    _code2=$(curl -s -m "${ATOM_GLM_PROBE_TIMEOUT:-30}" --noproxy '*' \
      "$ANTHROPIC_BASE_URL/v1/messages" "${_hdrs[@]}" \
      -H "x-api-key: ${ANTHROPIC_API_KEY:-}" -H 'anthropic-version: 2023-06-01' \
      -d "{\"model\":\"${ANTHROPIC_MODEL:-claude-sonnet-5}\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
      -o /dev/null -w '%{http_code}' 2>/dev/null || true)
    [ "$_code2" = "200" ] && _code="$_code2"
  fi
  if [ "$_code" != "200" ]; then
    # A 404 on BOTH shapes means the endpoint answered HTTP and simply doesn't recognize
    # either path we tried -- a probe-shape/gateway-routing problem, not proof the backend is
    # dead. A connection failure/timeout (curl reports "000" or empty) or a 5xx on either
    # attempt is the real "wedged or down" signal this probe exists to catch, so that still
    # aborts hard below -- only the specific both-404 case is ambiguous enough to not be worth
    # paging the model owner over. If this were escalated at all it would be class `env` (a
    # box/gateway configuration fact), never `glm` (the model owner didn't break this) -- but
    # it doesn't warrant failing the run either, so this path warns and continues instead.
    if [ "$_code" = "404" ] && [ "$_code2" = "404" ]; then
      say "WARN: GLM health probe got 404 on both OpenAI- and Anthropic-shape endpoints at $ANTHROPIC_BASE_URL -- treating as a probe/shape mismatch, not a dead backend. Proceeding (a genuinely dead backend will still surface as an agent timeout/failure below)."
    else
      fail glm 5 "GLM backend unavailable at $ANTHROPIC_BASE_URL (deep inference probe returned '$_code'${_code2:+ / '$_code2'})"
    fi
  else
    say "GLM ok"
  fi
fi

# 1) fetch (the skill's own Step-1 fetcher) -> WORK dir
say "fetch..."
FL="$(mktemp)"
if [ -n "$GH_FALLBACK_TOK" ]; then
  (cd "$PROJ" && GH_TOKEN="$GH_FALLBACK_TOK" GITHUB_TOKEN="$GH_FALLBACK_TOK" bash "$SKILL/fetch.sh" "$PR" "$REPO") 2>&1 | tee "$FL"
else
  (cd "$PROJ" && bash "$SKILL/fetch.sh" "$PR" "$REPO") 2>&1 | tee "$FL"
fi
W="$(grep -oE 'WORK=[^[:space:]]+/review-pr-[A-Za-z0-9]+' "$FL" | tail -1 | cut -d= -f2)"
rm -f "$FL"
[ -n "$W" ] && [ -d "$W" ] || fail env 1 "fetch produced no WORK dir -- check gh auth/token, network, git and disk on the runner"

# applies.txt fix: the skill's fetch.sh checks `git apply` against PROJECT_ROOT's worktree but
# reports it as the merge target; re-check on the merge-target worktree checked out at BASE_SHA.
if [ -d "$W/merge-target" ]; then
  BASE=$(grep -oE '[0-9a-f]{40}' "$W/merge_target.txt" 2>/dev/null | head -1)
  if git -C "$W/merge-target" -c core.fileMode=false apply --check "$W/pr.diff" 2>"$W/.err2"; then
    echo "APPLIES: the diff still applies to merge target $BASE (rechecked on the merge-target worktree)" > "$W/applies.txt"
  elif grep -q 'double-application' "$W/applies.txt" 2>/dev/null; then
    # fetch.sh already diagnosed this as double-application: the PR is already merged and
    # BASE_SHA is downstream of the PR's own merge commit, so re-applying the diff conflicts
    # by construction -- not drift. That is a fact about commit ancestry (merge commit vs.
    # BASE_SHA), not about which working tree `git apply --check` ran against, so it still
    # holds when the recheck (on the correct merge-target tree) also comes back STALE. This
    # recheck exists to correct WHICH TREE was tested, not to discard a correct, more
    # specific diagnosis for the generic "needs a rebase" one.
    say "applies.txt: recheck on merge-target agrees (STALE) -- keeping fetch.sh's double-application diagnosis"
  else
    { echo "STALE: no longer applies to merge target $BASE -- the PR needs a rebase,"
      echo "  and any CI result on it describes a tree that has moved"
      sed 's/^/  /' "$W/.err2"; } > "$W/applies.txt"
  fi
fi
say "WORK=$W"

# The GLM backend can be slow or time out on a shared box. Give each agent ATOM_AGENT_TIMEOUT
# (default 2400s / 40min). A *timeout* is never retried -- a slow generation is slow on retry too;
# only a transient failure (dropped connection / 5xx) is retried, up to ATOM_REVIEW_RETRIES
# (default 2) attempts, requiring a non-empty output file to count the attempt as success.
run_agent() {  # <label> <prompt-file> <out-file> <cmd...>
  local label="$1" pf="$2" out="$3"; shift 3
  local n=0 max="${ATOM_REVIEW_RETRIES:-2}" rc
  while :; do
    # Never start an attempt that cannot finish inside the run budget. The job cap would kill it
    # mid-flight, and a run killed there dies with no classified status at all -- worse than the
    # timeout it would have reported. Out of budget is a timeout, so say 124 and let the caller
    # treat it like one.
    if [ $(( $(date +%s) + ${ATOM_AGENT_TIMEOUT} )) -gt "$DEADLINE" ]; then
      # Out of budget is reported as a timeout so the caller does not retry it, but the two have
      # opposite fixes -- record which one happened so the page names the knob that helps.
      AGENT_TIMEOUT_CAUSE=budget
      say "$label: not enough run budget left to start an attempt"; return 124
    fi
    n=$((n + 1)); rm -f "$out"
    (cd "$PROJ" && timeout "${ATOM_AGENT_TIMEOUT}" "$@" "$(cat "$pf")"); rc=$?
    [ "$rc" -eq 0 ] && [ -s "$out" ] && return 0
    if [ "$rc" -eq 124 ]; then
      AGENT_TIMEOUT_CAUSE=agent
      say "$label hit the ${ATOM_AGENT_TIMEOUT}s timeout -- not retrying a timeout"; return 124
    fi
    if [ "$n" -ge "$max" ]; then say "$label failed after $max attempts (GLM error?)"; return 1; fi
    say "$label attempt $n failed (rc=$rc, transient?); retrying in $((n * 10))s"; sleep $((n * 10))
  done
}

# Triage an agent failure. A timeout is this pipeline's own budget, not a backend fault, so it
# routes to flow -- a review that needs longer must never page the model owner.
agent_fail() {  # <label> <rc> <exit-code-for-a-backend-fault>
  if [ "$2" -eq 124 ]; then
    if [ "${AGENT_TIMEOUT_CAUSE:-}" = budget ]; then
      fail flow 6 "the $1 never started: ATOM_RUN_BUDGET=${ATOM_RUN_BUDGET}s for the whole run was already spent. Raising ATOM_AGENT_TIMEOUT makes this refusal fire sooner, not later -- raise the budget and the job's timeout-minutes together"
    fi
    fail flow 6 "the $1 did not finish within ATOM_AGENT_TIMEOUT=${ATOM_AGENT_TIMEOUT}s; the backend answered normally -- raise it in the runner .env"
  fi
  fail glm "$3" "the GLM $1 failed -- the backend errored or returned nothing"
}

# 2) worker (headless GLM), with retry on a transient GLM failure
say "worker (GLM)..."
bash "$SKILL/render.sh" worker "$W" > "$W/_pw.txt"
run_agent "worker" "$W/_pw.txt" "$W/card.md" "${WORKER_CMD[@]}" || agent_fail worker $? 2

# 3) refuter (headless GLM) -- Step 7.7; or the NONE line for a 0-finding card
say "refuter..."
if grep -qiE '(NO FINDINGS|✅)' "$W/card.md" && ! grep -qE '^(🔴|⚠️|📝)' "$W/card.md"; then
  printf 'NONE AVAILABLE -- 0 findings on the card (NO FINDINGS); nothing for an independent reader to refute\n' > "$W/independent.txt"
else
  bash "$SKILL/render.sh" refuter "$W" "$W/card.md" > "$W/_prf.txt"
  # errexit is on: a bare `run_agent ...; rc=$?` exits the script on the first non-zero return,
  # so everything below -- the downgrade and the triage both -- would never run.
  rc=0; run_agent "refuter" "$W/_prf.txt" "$W/independent.txt" "${REFUTER_CMD[@]}" || rc=$?
  if [ "$rc" -eq 124 ]; then
    # card.md is already written and the findings are real work; a refuter timeout must not throw
    # it away. Degrade onto the gate's own "no independent reader" path, which only passes if the
    # card admits it -- so the downgrade reaches the reader instead of hiding in the log.
    say "refuter hit the ${ATOM_AGENT_TIMEOUT}s timeout -- publishing the card unrefuted"
    printf 'NONE AVAILABLE -- the refuter did not finish within %ss on this box\n' "${ATOM_AGENT_TIMEOUT}" > "$W/independent.txt"
    sed -i '0,/^Review (advisory):/s//Review (advisory, not independently refuted):/' "$W/card.md"
    # That sed is anchored to one literal shape of a model-generated line. If the card bolds or
    # indents it the edit silently does nothing, the gate finds no admission, and a known-good
    # card is discarded -- the exact loss this branch exists to prevent. Say it unconditionally.
    grep -q 'not independently refuted' "$W/card.md" \
      || sed -i '1i Review (advisory, not independently refuted): the refuter did not finish' "$W/card.md"
  elif [ "$rc" -ne 0 ]; then
    agent_fail refuter "$rc" 3
  fi
fi

# 3b) apply the refuter's verdicts: drop KILLED findings from the card before the gates, or the
# independent gate red-fails every review whose refuter did its job (kills a finding). No-op on a
# 0-finding card (NONE AVAILABLE) or a count mismatch.
python3 "$SKILL/_apply_refutation.py" "$W/card.md" "$W/independent.txt"

# 3c) citation check. Measured on seven real ATOM reviews: the deterministic half (does
# the cited file resolve in the diff, is the line within the file) never fired -- zero
# findings in that batch cited a file or line that does not exist -- while the proximity
# half caught a real defect, a finding citing envs.py:263 for a construct that lives at
# :253. Proximity is a heuristic and it misfires: a finding legitimately quotes symbols
# that live elsewhere (the remedy it proposes, a caller, a related flag), and two rounds
# of tightening it traded one class of false alarm for another. It ran 1-for-2 on real
# cards. So this is not a gate: it writes its verdict where the collector picks it up and
# the job log shows it, and a human deciding whether to trust a card can see it. Making a
# 50%-precision heuristic withhold reviews would cost more than the defect it catches.
python3 "$SKILL/check_citations.py" "$W" 2>&1 | tee "$W/citations.txt" || true

# Record who ran the independent refutation and what it did. Two separate things get
# called "refutation" here and they are not equally strong: Step 7.6 is the worker
# killing its own candidates -- same model, same context, useful filtering but not a
# second opinion -- and Step 7.7 is a separate invocation reading the finished card.
# Only the second is adversarial, and REFUTER_CMD falls back to the worker's own
# entrypoint when ATOM_REVIEW_REFUTER_AGENT is unset, which is the default. aiter's own
# note at the top of this file says the refuter's value comes from being a DIFFERENT
# family; same-family and cross-family produce identical artifacts, so a reader cannot
# tell which they got. Decided in the shell, where no agent reports on itself.
IND_KILLED=$(grep -cE '^KILLED' "$W/independent.txt" 2>/dev/null || true)
IND_SURVIVED=$(grep -cE '^SURVIVED' "$W/independent.txt" 2>/dev/null || true)
IND_KILLED=${IND_KILLED:-0}; IND_SURVIVED=${IND_SURVIVED:-0}
IND_JUDGED=$(( IND_KILLED + IND_SURVIVED ))
{ printf 'worker:              %s\n' "$AGENT"
  printf 'independent refuter: %s\n' "$REFUTER_BIN"
  if [ "$SAME_FAMILY" = 1 ]; then
    printf 'cross-family: NO -- same entrypoint as the worker, so the independent pass is\n'
    printf '  same-family and weaker adversarially. Point ATOM_REVIEW_REFUTER_AGENT at a\n'
    printf '  different model to restore it.\n'
  else
    printf 'cross-family: YES\n'
  fi
  printf 'independent verdicts: %s judged, %s killed, %s defended\n' \
    "$IND_JUDGED" "$IND_KILLED" "$IND_SURVIVED"
} > "$W/agents.txt"

# Say it on the card. An artifact is only read by someone who already went looking, and
# a stage that never kills anything looks exactly like a stage that is not there.
if [ -f "$W/card.md" ] && ! grep -q 'Independent refutation:' "$W/card.md"; then
  if [ "$SAME_FAMILY" = 1 ]; then
    _fam="same-family (\`$(basename "$AGENT")\` reviewed its own output)"
  else
    _fam="cross-family (\`$(basename "$REFUTER_BIN")\` vs \`$(basename "$AGENT")\`)"
  fi
  printf '\n_Independent refutation: %s judged, %s killed, %s defended — %s. Step 7.6 above\nis the worker filtering its own candidates, which is not a second opinion._\n' \
    "$IND_JUDGED" "$IND_KILLED" "$IND_SURVIVED" "$_fam" >> "$W/card.md"
fi

# 4) gates + collect (call the python directly; no thin shell wrappers)
say "gates..."
# errexit is on: a bare call exits here on a red gate, and the EXIT trap then reports "exited
# unexpectedly with no classified failure" -- while the output right above it says exactly which
# gate went red. Classify it instead, and name the gate, so the notice is actionable: a red gate
# is a review that was withheld on purpose, and re-running the worker usually clears it.
rc=0; python3 "$SKILL/_gates.py" "$W" "$PROJ" | tee "$W/gates.out" || rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  # `|| true` is not optional: a pipeline that finds nothing returns non-zero under pipefail,
  # and a command substitution that fails is a fatal assignment under errexit -- the same trap
  # this block exists to fix. The red lines are the ones carrying `exit=`.
  # `!seen[$2]++` because a gate can red on more than one line, and "card,card,ledger" tells a
  # reader less than "card,ledger" does.
  red=$(awk '/exit=/ && !seen[$2]++ {print $2}' "$W/gates.out" 2>/dev/null | paste -sd, - || true)
  fail flow 7 "the review was withheld by the ${red:-?} gate -- the card did not meet the bar, which is usually cleared by running it again; re-comment to retry"
fi
say "collect..."
python3 "$SKILL/_collect.py" "$W"

# Clean up the scratch WORK dir + its worktrees on success (set -e keeps it on failure for debug).
for wt in "$W/merge-target" "$W/head"; do [ -d "$wt" ] && git -C "$PROJ" worktree remove --force "$wt" 2>/dev/null || true; done
rm -rf "$W"
git -C "$PROJ" worktree prune 2>/dev/null || true

say "done. report in $SKILL/reports/PR-$PR/ (uncommitted, unpublished)"
