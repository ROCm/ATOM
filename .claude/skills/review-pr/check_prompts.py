"""Guard against prompt / SKILL.md drift.

prompts/worker.md and prompts/refuter.md quote SKILL.md verbatim (the Step 8 Output rules and
the Step 7.7 text). If SKILL.md changes and the copies do not, they drift silently and the
agents start working from a stale standard. This check extracts each quoted block from SKILL.md
(between unique start/end anchors) and asserts it appears verbatim in the prompt. Any drift, or
a moved anchor, exits non-zero. Run it in preflight and at the start of run_one.sh.

Layout-independent: the prompts sit in `prompts/` next to this file, wherever it lives, and
SKILL.md is always at <repo>/.claude/skills/review-pr/SKILL.md.
"""

import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _skill_md():
    root = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    p = pathlib.Path(root, ".claude", "skills", "review-pr", "SKILL.md")
    return p.read_text(encoding="utf-8")


# (prompt file relative to HERE, start anchor, end anchor) — anchors are unique lines in SKILL.md.
CHECKS = [
    (
        "prompts/refuter.md",
        "## Step 7.7 — Independent refutation",
        "`rules.md` § Refutation has why.",
    ),
    (
        "prompts/worker.md",
        "**Output rules (strictly enforced):**",
        '- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion',
    ),
]


def extract(text, start, end):
    i = text.find(start)
    if i < 0:
        return None
    j = text.find(end, i)
    if j < 0:
        return None
    return text[i : j + len(end)]


def _missing(*paths):
    """Return the subset of paths that do not exist. Used to make checks tolerant of
    not-yet-built infrastructure (run_one.sh, the workflow YAML, _notify.py, watchdog.py,
    lost_review.py, queue_gate.py): a missing file means 'not yet time to check', not a
    drift. A file that DOES exist but is internally inconsistent must still fail loudly --
    that mismatch is the entire reason each check exists."""
    return [p for p in paths if not p.exists()]


def check_fail_classes():
    """run_one.sh tags each failure with `fail <class> ...`; _notify.py owns the class->owner
    map. Assert every class run_one can emit is known to _notify.py, so a failure never routes
    to nobody and run_one.sh / _notify.py cannot drift apart."""
    import importlib.util

    miss = _missing(HERE / "run_one.sh", HERE / "_notify.py")
    if miss:
        print(f"⏭️  check_fail_classes: skipped — not yet built: "
              f"{[p.name for p in miss]}")
        return 0

    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    emitted = set(re.findall(r"\bfail\s+([a-z]+)\s+[0-9]", run_one))
    spec = importlib.util.spec_from_file_location("_notify", HERE / "_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    known = set(mod.OWNERS)
    unknown = emitted - known
    if unknown:
        print(f"\u274c run_one.sh emits fail classes unknown to _notify.py: {sorted(unknown)}")
        return 1
    print(f"\u2705 fail classes consistent: run_one {sorted(emitted)} \u2286 _notify.py {sorted(known)}")
    return 0


WORKFLOW = HERE.parents[2] / ".github" / "workflows" / "atom-review-bot.yml"


def _notify_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_notify", HERE / "_notify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_budget_fits():
    """The run budget and the job's timeout-minutes live in different files, and a budget that
    does not fit is silently useless: the job is killed mid-agent with no classified status --
    the exact failure the budget was added to prevent. Keep the two in step."""
    miss = _missing(HERE / "run_one.sh", WORKFLOW)
    if miss:
        print(f"⏭️  check_budget_fits: skipped — not yet built: "
              f"{[p.name for p in miss]}")
        return 0
    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    wf = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r'^: "\$\{ATOM_RUN_BUDGET:=(\d+)\}"', run_one, re.M)
    j = re.search(r"^  review:.*?^    timeout-minutes: (\d+)", wf, re.M | re.S)
    if not m or not j:
        print("\u274c cannot read ATOM_RUN_BUDGET or the review job's timeout-minutes")
        return 1
    budget, cap = int(m.group(1)), int(j.group(1)) * 60
    if budget >= cap:
        print(f"\u274c ATOM_RUN_BUDGET={budget}s does not fit the review job's {cap}s cap")
        return 1
    print(f"\u2705 run budget fits the job cap: {budget}s < {cap}s")
    return 0


def check_owner_fallbacks():
    """watchdog.py, lost_review.py and queue_gate.py each hardcode their own fallback owner
    rather than reading owners.txt: they exist to speak precisely when the review job (and
    _notify.py with it) never ran at all, so they cannot depend on anything that only works
    inside that job. That duplication is deliberate, but it drifts from owners.txt in silence
    if nobody checks it: the next owner updates owners.txt and still never hears that the
    runner is down, or that a review vanished.

    Since Task 5 moved the map into owners.txt, there is no longer a repo-variable override
    channel for these three scripts' fallback -- Task 7 removed the workflow's OWNER_OVERRIDE
    injection along with it, so this only compares each hardcoded login against owners.txt."""
    miss = _missing(HERE / "_notify.py", HERE / "watchdog.py",
                     HERE / "lost_review.py", HERE / "queue_gate.py")
    if miss:
        print(f"\u23ed\ufe0f  check_owner_fallbacks: skipped \u2014 not yet built: "
              f"{[p.name for p in miss]}")
        return 0
    owners = _notify_mod().OWNERS
    rows = [
        ("watchdog.py", "env",
         r'OWNER_OVERRIDE"\) or ""\)\.strip\(\) or "([A-Za-z0-9-]+)"'),
        ("lost_review.py", "flow",
         r'OWNER_OVERRIDE"\) or ""\)\.strip\(\) or "([A-Za-z0-9-]+)"'),
        ("queue_gate.py", "env",
         r'OWNER_OVERRIDE"\) or ""\)\.strip\(\) or "([A-Za-z0-9-]+)"'),
    ]
    bad = 0
    for filename, cls, pattern in rows:
        m = re.search(pattern, (HERE / filename).read_text(encoding="utf-8"))
        default = owners[cls][0]
        if not m:
            print(f"\u274c cannot read {filename}'s owner fallback")
            bad += 1
            continue
        if m.group(1) != default:
            print(f"\u274c {filename} pages '{m.group(1)}', but owners.txt's {cls} class is "
                  f"'{default}'")
            bad += 1
            continue
        print(f"\u2705 {filename} matches owners.txt's {cls} class: {default}")
    return 1 if bad else 0


def check_no_self_trigger():
    """The bot's own comments are issue_comment events like any other. Without an explicit
    exclusion every card and queue notice starts a run of this workflow, which is gated off a
    moment later -- but only after taking a hosted-runner slot the real requests are waiting
    for. The guard is one line and leaves no trace when removed, so assert it."""
    miss = _missing(WORKFLOW)
    if miss:
        print(f"\u23ed\ufe0f  check_no_self_trigger: skipped \u2014 not yet built: "
              f"{[p.name for p in miss]}")
        return 0
    wf = WORKFLOW.read_text(encoding="utf-8")
    if not re.search(r"github\.event\.comment\.user\.login != 'rocm-atom-bot'", wf):
        print("\u274c the gate does not exclude the bot's own comments; every notice it posts "
              "will start a run that competes with real requests")
        return 1
    print("\u2705 the gate excludes the bot's own comments")
    return 0


def check_notified_handoff():
    """The report job stays quiet when the review job says it already spoke. That answer travels
    _notify.py -> step output -> job output -> the report job's env, and every link is a string
    that something else has to spell the same way. Break one and nothing errors: the flag simply
    arrives empty, the report fires on every failed review, and the duplicate notice looks like
    the bot being broken rather than the wiring being broken."""
    miss = _missing(WORKFLOW, HERE / "_notify.py")
    if miss:
        print(f"\u23ed\ufe0f  check_notified_handoff: skipped \u2014 not yet built: "
              f"{[p.name for p in miss]}")
        return 0
    wf = WORKFLOW.read_text(encoding="utf-8")
    notify = (HERE / "_notify.py").read_text(encoding="utf-8")
    links = [
        ("_notify.py writes the flag", 'fh.write("notified=true' in notify),
        ("the review job has a step to write it", re.search(r"^\s+id: notify\s*$", wf, re.M)),
        ("the review job exports it",
         re.search(r"^\s+notified: \$\{\{ steps\.notify\.outputs\.notified \}\}", wf, re.M)),
        ("the report job reads it",
         re.search(r"^\s+NOTIFIED: \$\{\{ needs\.review\.outputs\.notified \}\}", wf, re.M)),
    ]
    broken = [name for name, found in links if not found]
    if broken:
        print("\u274c the notified handoff is broken at: %s" % "; ".join(broken))
        return 1
    print("\u2705 notified handoff intact: _notify.py -> step -> job output -> report")
    return 0


def check_runner_label_exclusive():
    """boxBusy() decides 'is the box busy' by looking only at this workflow's runs. That holds
    only while nothing else targets the same runner label -- otherwise a healthy box doing other
    work reads as idle, and the watchdog pages the owner about it."""
    miss = _missing(WORKFLOW)
    if miss:
        print(f"\u23ed\ufe0f  check_runner_label_exclusive: skipped \u2014 not yet built: "
              f"{[p.name for p in miss]}")
        return 0
    wf = WORKFLOW.read_text(encoding="utf-8")
    m = re.search(r"runs-on: \[self-hosted, ([A-Za-z0-9_-]+)\]", wf)
    if not m:
        print("\u274c cannot read the review job's runner label")
        return 1
    label = m.group(1)
    others = sorted(q.name for q in WORKFLOW.parent.glob("*.y*ml")
                    if q != WORKFLOW and label in q.read_text(encoding="utf-8"))
    if others:
        print(f"\u274c runner label '{label}' is also targeted by {others}; the watchdog's busy "
              "check only sees atom-review-bot runs and would page about a busy, healthy box")
        return 1
    print(f"\u2705 runner label '{label}' is exclusive to atom-review-bot.yml")
    return 0


def check_publish_needs_post_flag():
    """_publish.py's only closure against posting with excess write access is its missing
    `--post` flag: on the self-hosted box it falls back to ~/.git-credentials, a human PAT
    with repo write access, and posts over plain curl -- so GITHUB_TOKEN permission scoping
    does not stop it from posting at all. Today the workflow grants the review job only
    `contents: read`, so that fallback never has write access to use. If the workflow is ever
    changed to grant `issues: write` or `pull-requests: write` anywhere, GITHUB_TOKEN stops
    being irrelevant to _publish.py's reach, and the invocation MUST pass --post explicitly
    -- or there is a silent gap between what the token in scope can do and what the one call
    site that would use it was actually told to do."""
    miss = _missing(WORKFLOW, HERE / "_publish.py", HERE / "run_one.sh")
    if miss:
        print(f"⏭️  check_publish_needs_post_flag: skipped — not yet built: "
              f"{[p.name for p in miss]}")
        return 0
    wf = WORKFLOW.read_text(encoding="utf-8")
    run_one = (HERE / "run_one.sh").read_text(encoding="utf-8")
    granted = re.search(r"^\s*(issues|pull-requests):\s*write\s*$", wf, re.M)
    invocations = re.findall(r"[^\n]*_publish\.py[^\n]*", wf) \
        + re.findall(r"[^\n]*_publish\.py[^\n]*", run_one)
    has_post = any("--post" in line for line in invocations)
    if granted and not has_post:
        print(f"❌ atom-review-bot.yml grants '{granted.group(0).strip()}' somewhere, "
              f"but _publish.py is never invoked with --post -- GITHUB_TOKEN scoping is no "
              f"longer the only thing standing between a job and a post, and _publish.py's "
              f"missing-flag default (dry-run) must now be an explicit match, not an "
              f"accident")
        return 1
    if granted:
        print(f"✅ atom-review-bot.yml grants write, and _publish.py is invoked with "
              f"--post to match it")
        return 0
    print(f"✅ atom-review-bot.yml grants no issues/pull-requests write anywhere; "
          f"_publish.py's missing --post is its only closure, as documented")
    return 0


def main():
    skill = _skill_md()
    bad = 0
    for prompt, start, end in CHECKS:
        block = extract(skill, start, end)
        prompt_text = (HERE / prompt).read_text(encoding="utf-8")
        if block is None:
            print(f"❌ {prompt}: SKILL.md anchors not found — {start!r} .. {end!r}")
            bad += 1
        elif block in prompt_text:
            print(f"✅ {prompt}: quoted block matches SKILL.md verbatim ({len(block)} chars)")
        else:
            print(f"❌ {prompt}: quoted block DRIFTED from SKILL.md — re-copy the section verbatim")
            bad += 1
    bad += check_fail_classes()
    bad += check_budget_fits()
    bad += check_owner_fallbacks()
    bad += check_no_self_trigger()
    bad += check_notified_handoff()
    bad += check_runner_label_exclusive()
    bad += check_publish_needs_post_flag()
    print(f"{'OK' if bad == 0 else 'DRIFT'}: {bad} drift(s)")
    return bad


if __name__ == "__main__":
    sys.exit(main())
