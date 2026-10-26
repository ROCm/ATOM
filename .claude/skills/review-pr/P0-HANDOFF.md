# P0 handoff — atom-review-bot

This is the handoff from the agent session that built P0 to whoever runs the remaining
steps on the `srdc-24` host, in a plain terminal. Everything in this skill directory was
written and verified from inside a container; the steps below need a real shell and, in
two places, your own GitHub credentials — neither is something an agent session can do.

## 1. What P0 delivers, and what it deliberately does not

P0 is the **review chain**, end to end: `fetch.sh` collects evidence from a real PR, a
headless worker agent writes a review card, an independent refuter tries to kill every
finding, seven gates verify each step actually happened (not just claimed), and
`_collect.py` archives the run. Run it against a real ATOM PR and you get a `card.md` —
a structured, human-readable review with a one-line summary, tagged findings, and a
verdict.

**P0 posts nothing.** There is no PR comment, no claim reaction, no GitHub write of any
kind. `_publish.py --post` exists in the codebase but is not wired into the workflow's
review step, and the repo secret it would need (`ATOM_BOT_TOKEN`) has no bot account
behind it yet — `rocm-atom-bot` does not exist. Wiring that up is **P0.5**, and it needs
you to mint the bot account first (see §5).

Until P0.5, the dry-run workflow renders the card and logs it; nothing lands on the PR.

## 2. The acceptance run

This is the one step this session could not perform. The brief's acceptance test is to
run the pipeline against a real ATOM PR, and `run_one.sh` launches a headless sub-agent
with `--dangerously-skip-permissions`. This environment's auto-mode classifier blocks
exactly that pattern as "Create Unsafe Agents" — it is not a false positive, it is the
documented boundary of what an auto-mode session is allowed to launch. `run_one.sh`
itself carries the same warning in its header. Run it in a plain terminal (SSH or
console on `srdc-24`, not inside any Claude session), where there is no such classifier.

```bash
cd /mnt/raid0/zufa/atom-bot-work
export ATOM_REVIEW_AGENT=/usr/local/bin/claude-glm
PR=<pick a PR — see below>
time bash .claude/skills/review-pr/run_one.sh "$PR" ROCm/ATOM 2>&1 | tee /tmp/run_one.log
echo "exit=${PIPESTATUS[0]}"
```

**Picking a PR**: a merged ATOM PR that changes code, under about ten files, works best
for a first run — small enough to read the card against the diff by eye. PRs #2697,
#2705, #2678 and #2612 have all been used successfully during development and are known
to work; any PR of similar shape is fine too.

**What success looks like**:

```bash
grep -A 12 "SEVEN GATES" /tmp/run_one.log
```

All seven gate lines `✅`, and the last line reads `======== 7 green / 0 red ========`.
Then:

```bash
cat .claude/skills/review-pr/reports/PR-$PR/card.md
```

`card.md` should exist and be non-empty — that file existing, with the gates green, is
the acceptance bar.

While you're at it, capture token usage and wall clock — this is the first real
opportunity to measure the gateway-quota risk the design spec has carried as "unmeasured"
(see §3 below):

```bash
grep -iE "token|usage|elapsed|real" /tmp/run_one.log | tail -20
```

## 3. How to read the card

Open `card.md` and check three things by hand:

1. **"What it does"** is exactly one sentence, and it reads like something you'd say to
   a reviewer who has not opened the diff — not a restatement of the PR title, not a
   list of changed files.
2. **Every finding has three parts**: problem, impact, and an action that ends in a verb
   phrase ("Author must fix X" / "Reviewer should ask Y"). A finding missing any of the
   three, or ending without a verb, is a gap in the worker's output, not something to
   wave through.
3. **The `Validation (deterministic)` line never states a fabricated verdict.** P0 ships
   no validator (that's `validate-atom-pr`, P2), so this line must read either
   `N/A — no runtime surface changed` (the PR touches nothing with a runtime surface) or
   `NOT RUN — <reason>` (it does, but nothing ran it — an environment gap, not a PR
   defect). Anything that looks like an actual pass/fail validation result is wrong and
   means something upstream regressed.

## 4. What to do when a gate goes red

A red gate means the review was **withheld on purpose** — that step did not happen, or
didn't happen to the standard the gate checks for. Read the rejection reason it prints;
it names the specific thing missing (an unanswered question, a rule with no verdict, a
finding that reached the card without being attacked, etc.).

- **First red**: re-run the worker. Most reds are the model not following the output
  format that particular time, not a real defect in the pipeline.
- **Same gate red three times running**: that is not bad luck, that is the gate — or
  `SKILL.md`'s instruction to the worker — being wrong. Treat it as a code fix: find the
  corresponding task in the plan (`docs/superpowers/plans/2026-10-23-atom-review-bot-p0.md`)
  and fix the instruction or the gate's check.
- **Never relax a gate to make it pass.** A gate exists to catch a specific way the
  model can cheat the review; loosening it to clear a red defeats the one thing gates
  are for.

## 5. What's left for you

Two things remain, both requiring a real shell on the host and your own credentials —
an agent session cannot do either:

- **Register the runner** — `RUNNER-SETUP.md` §1. Minting a GitHub Actions runner
  registration token is a credential operation; no agent session is allowed to do this
  on your behalf.
- **Install it as a systemd service** — `RUNNER-SETUP.md` §5. `config.sh` and
  `sudo ./svc.sh install` need a real systemd, which does not exist inside this
  container (`PID 1` is `bash`, `systemctl is-system-running` returns `offline`). Do the
  install from the host, logged in directly, and add the `Restart=always` drop-in in
  that section — the unit `svc.sh` writes on its own survives a reboot but not a crash.

Once both are done, `bash .claude/skills/review-pr/preflight.sh` should show all green
except the `ATOM_BOT_TOKEN` warning (expected until P0.5 mints a bot account) and the
persistence check should now pass instead of showing the current, correct-for-today red.

## 6. What's deliberately left undone

- **P0.5 — publishing**: wire up the claim reaction (👀), `_publish.py --post`, and real
  PR comments. Needs the `rocm-atom-bot` account and its PAT in `ATOM_BOT_TOKEN`.
- **P1 — ATOM family axis and rule rewrite**: replace the P0 placeholder rule deriver
  with ATOM's real family axis, and rewrite `rules.md`'s rule content from roughly the
  last six months of ATOM's own review comments across all channels.
- **P2 — `validate-atom-pr`**: a deterministic validator that runs GSM8K base/head and
  reports the delta, with a GPU-subset lock and noise-floor measurement so it knows what
  counts as a real regression versus run-to-run noise.
- **P3 — migrate to box308, register against `ROCm/ATOM`**: needs repo admin on the real
  ATOM repo, and needs box308's GPU contention (its 8 cards are already held at 98–99%
  VRAM by a long-running GLM-5.3 server) resolved first — see the design spec §5.3 for
  the candidate solutions, none of which is picked yet.
