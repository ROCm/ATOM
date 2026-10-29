# Runner setup — atom-review-bot

The `atom-bot review` workflow (`.github/workflows/atom-review-bot.yml`) runs on a
self-hosted runner labeled `atom-bot`. This is what that runner must provide. Verify the
box with `bash .claude/skills/review-pr/preflight.sh` — it checks every item below and
prints a fix hint for each red. This doc explains how to make those checks pass.

## Two boxes, two phases — read this before copying a command

This setup runs in two phases, on two different boxes, and the facts that matter invert
between them. Do not read one box's section and assume it generalizes to the other.

| | **srdc-24 — testing, now** | **box308 — production, later** |
|---|---|---|
| systemd | absent in the agent container: `PID 1` is `bash`, `systemctl is-system-running` returns `offline` | real: `PID 1` is `systemd`, `is-system-running` returns `running` |
| `/mnt/raid0` | 100% full (126G free of a much larger disk, all spoken for) | 9.3T free |
| runner install path | `/mnt/md0` instead (~9.6T free, persistent volume) | `/mnt/raid0/zufa/`, alongside aiter's own runner |
| GPUs (8× MI308X) | all idle | all held by a local GLM-5.3 server (VRAM 98–99%) |
| model endpoint | reached over the network (AMD gateway) | served locally, `localhost:30000` |
| other runners on the box | none | aiter's `actions.runner.ROCm-aiter.srdc-8-aiter-review.service` |

**Why both boxes still exist**: srdc-24 is where P0 is being built and verified, with
idle GPUs and no neighbor runner to worry about. box308 is where this eventually has to
live for real, because that is where aiter's bot already lives and where the box has
spare GPU-review capacity *in principle* — except that in practice its GPUs are all
taken (see below). The label `atom-bot` is deliberately machine-neutral so it survives
the move from one to the other without a rename.

### The GPU constraint this creates (live, not hypothetical, on box308)

On srdc-24 the model is reached over the network, so all eight GPUs stay free and
`validate-atom-pr` (arriving in P2) has somewhere to run. On box308 the model is served
*locally* by a GLM-5.3 that already holds all eight GPUs — so when `validate-atom-pr`
arrives in P2, there is **no GPU on this box for it to use**. This is not a future
problem to flag later; it is true today, as soon as review moves to box308.

This is exactly the situation the design spec's **section 5.3** anticipates, and it
lists the candidate resolutions: shrink the GLM's TP to free some GPUs, accept
`NOT RUN — no idle GPU` on box308 (the gate supports this status), or run validation on
a different box (srdc-24) while review runs on box308. **This document does not pick
one** — that is P2's decision to make, once validate exists to make it about. What keeps
all three options open is that `validate_pr.sh` is independently invocable and must
stay that way: nothing in this setup may assume review and validate run on the same
box.

### Why `/mnt/md0` on srdc-24, and `/mnt/raid0/zufa/` on box308

The common reason, on both boxes, is the same: each review checks out the repo and
builds two worktrees, and a runner whose volume fills up mid-review fails mid-review,
often silently if nobody is watching the disk. Only the path changes, because only the
disk changes: srdc-24's `/mnt/raid0` has no room, so the runner goes on `/mnt/md0`
instead; box308's `/mnt/raid0` has 9.3T free, and aiter's own runner is already there
under `/mnt/raid0/zufa/aiter-runner/`, so ATOM's runner goes alongside it at
`/mnt/raid0/zufa/atom-runner/` for the same reason aiter's does — plenty of room, a
persistent volume, consistent with the neighbor already running there.

### The container/host split — do sections 1 and 5 from the host, not from here

Agent sessions on srdc-24 (including whichever session produced this file) run inside a
Docker container: `PID 1` is `bash`, `systemctl is-system-running` returns `offline`,
and `/.dockerenv` exists. There is no systemd there, and no amount of `nohup` or a
hand-rolled keeper loop fixes that — see the warning in section 5. box308 is different:
it has real systemd (`is-system-running` returns `running`), so `svc.sh install` works
there directly. What stays true on **both** boxes is that an agent session running
inside a container cannot register a runner or install a systemd service — that still
has to be done by a human, logged in to the box itself (SSH or console), not from
inside this or any other agent container. The runner's **files** can be prepared from
a container, because the install path (`/mnt/md0` on srdc-24, `/mnt/raid0/zufa/` on
box308) is reachable from both the container and the host; registration (section 1) and
service install (section 5) cannot.

### Two runners on one box (box308 only)

A second runner, not a shared one. aiter's unit is registered against `ROCm/aiter`,
and a repository-scoped runner can only ever serve the repository it was registered
to — there is no way to point it at `ROCm/ATOM` as well. Sharing one registration
across both repos would require re-registering aiter's runner at the `ROCm`
*organization* level, which changes a setup that is working today; that is a decision
for whoever owns aiter's bot, not a prerequisite for this one. What ATOM needs is its
own registration token for `ROCm/ATOM` and its own runner process on the same box.

box308 already runs aiter's reviewer as a systemd unit named
`actions.runner.ROCm-aiter.srdc-8-aiter-review.service`. GitHub names these units
`actions.runner.<org>-<repo>.<name>.service`, so ATOM's runner — registered against
`ROCm/ATOM` — produces `actions.runner.ROCm-ATOM.<name>.service`. The repo segment
differs (`ROCm-aiter` vs `ROCm-ATOM`), so there is no name collision regardless of what
`<name>` is chosen; pick something in aiter's style, e.g. `srdc-8-atom-review`, so the
two units read as a matched pair in `systemctl list-unit-files`.

`preflight.sh`'s persistence check greps `systemctl list-unit-files` for
`actions.runner.*ATOM*` (falling back to `--user` scope). That pattern matches
`actions.runner.ROCm-ATOM.srdc-8-atom-review.service` (contains `ATOM`) and does
**not** match aiter's unit `actions.runner.ROCm-aiter.srdc-8-aiter-review.service`
(contains `aiter`, not `ATOM`, and `glob`/regex matching here is case-sensitive) —
confirmed by inspection of both unit strings against the pattern; a check that reported
green by finding the neighbor's runner instead of its own would be exactly the kind of
false pass this project keeps finding, so this is worth re-checking by eye any time
either runner's name changes.

The two runners also compete for something more consequential than CPU and disk: **the
same GLM**. Both bots run `claude-glm`, which resolves one remote GLM over one tunnel,
and neither side takes a lock — there is no `flock`, no cross-repo concurrency group,
nothing. Each runner takes one job at a time and each job drives one stream (worker and
refuter run in sequence), so the ceiling is two concurrent streams against that GLM.

What this costs is unmeasured, and the honest statement is that nobody has looked. Two
facts make it worth looking before trusting the box with both bots:

- GLM's latency already has a long tail on its own. In a seven-PR batch on this branch,
  one 7-file PR spent over 40 minutes in the worker alone and the run died in the
  refuter. The run budget is 7200 s; a second concurrent stream eats into headroom that
  a single stream has already been observed to exhaust.
- The health gate cannot see this. It deep-probes with a real one-token inference, which
  catches a wedged backend, but it does not time the response — so a GLM that is healthy
  and congested is indistinguishable from one that is healthy and idle. The failure mode
  it would produce is a timeout an hour later, not a refusal up front.

`queue_gate.py` does not close this either: it reasons about ATOM's own queue ("the box
runs one review at a time") and has no visibility into the other repo's runner. Its ETA
is about when a job *starts*, which the second runner does not delay; the contention
lands on how long the job *takes*.

Measure it before building a mechanism — run both bots against real PRs at the same time
and compare wall clock to the single-bot numbers already recorded on this branch. If the
cost is real, the options are a box-level `flock` around the agent calls (needs a change
on aiter's side too), or a latency check in the health gate that widens the budget when
the backend is slow. Picking one now would be guessing.

CPU and disk contention exists too, and gets worse once P2's validation step needs a GPU
(see the constraint above).

---

## 1. Register the runner (host only — needs your GitHub credentials)

### On srdc-24 (testing)

```bash
sudo mkdir -p /mnt/md0/atom-runner && sudo chown "$(whoami)" /mnt/md0/atom-runner
cd /mnt/md0/atom-runner
curl -o actions-runner.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
tar xzf actions-runner.tar.gz && rm actions-runner.tar.gz
```

**>>> Credential step — run this yourself, from the host, logged in as you <<<**
Mint a registration token for the fork used during P0 development (self-serve, no ROCm org
admin needed — `rocm-atom-bot` does not exist yet and P0 needs no bot token for this step):

```bash
env -u GH_TOKEN -u GITHUB_TOKEN gh api -X POST \
  repos/zufayu/ATOM/actions/runners/registration-token --jq .token
```

(When this moves to `ROCm/ATOM` for real, the same call needs **repo admin** on that repo — a
`maintain` or `push` role gets a 403 from this endpoint.)

Register with that token and the label this workflow expects:

```bash
cd /mnt/md0/atom-runner
./config.sh --url https://github.com/zufayu/ATOM --token <token-from-above> \
  --labels atom-bot --name srdc24-atom-review --work _work --unattended --replace
```

### On box308 (production)

Same binary, same label, different path and a name that pairs with aiter's
`srdc-8-aiter-review`:

```bash
sudo mkdir -p /mnt/raid0/zufa/atom-runner && sudo chown "$(whoami)" /mnt/raid0/zufa/atom-runner
cd /mnt/raid0/zufa/atom-runner
curl -o actions-runner.tar.gz -L \
  https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64.tar.gz
tar xzf actions-runner.tar.gz && rm actions-runner.tar.gz

env -u GH_TOKEN -u GITHUB_TOKEN gh api -X POST \
  repos/ROCm/ATOM/actions/runners/registration-token --jq .token

cd /mnt/raid0/zufa/atom-runner
./config.sh --url https://github.com/ROCm/ATOM --token <token-from-above> \
  --labels atom-bot --name srdc-8-atom-review --work _work --unattended --replace
```

This registers as `actions.runner.ROCm-ATOM.srdc-8-atom-review.service` once installed
(section 5) — distinct from aiter's `actions.runner.ROCm-aiter.srdc-8-aiter-review.service`.

## 2. Box config via the runner `.env` (loaded into every job)

The workflow's review step sets no environment on purpose, so box-specific config lives
in the runner's own `.env` (`/mnt/md0/atom-runner/.env` on srdc-24,
`/mnt/raid0/zufa/atom-runner/.env` on box308), which the runner injects into every job.

**srdc-24** (tunnel mode — model reached over the network):

```bash
mkdir -p /mnt/md0/atom-runner/tmp /mnt/md0/atom-runner/gh-config
cat > /mnt/md0/atom-runner/.env <<'EOF'
# Machine facts only. Policy (budgets, owner map, allowlist) lives in the repo --
# see section 3.2 of the design spec.
TMPDIR=/mnt/md0/atom-runner/tmp
GH_CONFIG_DIR=/mnt/md0/atom-runner/gh-config
PATH=/usr/local/bin:/usr/bin:/bin
EOF
```

Deliberately **not** set here: `ATOM_REVIEW_AGENT`. Leaving it unset is what keeps this
box in tunnel mode (`run_one.sh` defaults `AGENT` to `claude-glm`, which resolves the
AMD gateway on its own).

**box308** (direct-agent mode — model served locally at `localhost:30000`):

```bash
mkdir -p /mnt/raid0/zufa/atom-runner/tmp /mnt/raid0/zufa/atom-runner/gh-config
cat > /mnt/raid0/zufa/atom-runner/.env <<'EOF'
TMPDIR=/mnt/raid0/zufa/atom-runner/tmp
GH_CONFIG_DIR=/mnt/raid0/zufa/atom-runner/gh-config
PATH=/usr/local/bin:/usr/bin:/bin
ATOM_REVIEW_AGENT=/usr/local/bin/claude   # or wherever the local claude binary lives
ANTHROPIC_BASE_URL=http://localhost:30000
EOF
```

On box308, `ATOM_REVIEW_AGENT` and `ANTHROPIC_BASE_URL` must be set **together** —
`preflight.sh`'s "headless agent" check takes the direct-agent branch only when
`ATOM_REVIEW_AGENT` is set, and then requires `ANTHROPIC_BASE_URL` too. Setting one
without the other is exactly the mismatch this section exists to prevent: do not copy
aiter's `.env` verbatim without checking which branch it puts this box on.

Also **not** set on either box, because they are policy, not machine fact, and already
live in the repo: `ATOM_RUN_BUDGET`, `ATOM_AGENT_TIMEOUT`, `ATOM_REVIEW_RETRIES`, any
`*_OWNER`. A value parked in `.env` outlives the reason it was added, and nothing
reviews it — follow `run_one.sh`'s own defaults and failure-message hints instead of
pre-guessing an override here.

## 2b. Two repo *variables* (Settings -> Secrets and variables -> Actions -> Variables)

Distinct from the `.env` above: these are repo-level, read by the free `gate` and
`watchdog` jobs that run on `ubuntu-latest` and never see the box's `.env`.

| name | unset behaviour |
|---|---|
| `ATOM_REVIEW_AUTHORIZED` | falls back to `authorized.txt` on the default branch -- the intended steady state. Set it only to change the allowlist without a commit. |
| `ATOM_MAX_WAIT_HOURS` | the watchdog uses its built-in default. |

Neither is required for a first run.

## 3. gh auth off the job environment

`fetch.sh` calls `gh` to read the PR. Authenticate it under `GH_CONFIG_DIR` rather than exporting
`GH_TOKEN`, so the headless review agent never inherits a token from its environment. Substitute
the runner's own `GH_CONFIG_DIR` from section 2 (`/mnt/md0/atom-runner/gh-config` on srdc-24,
`/mnt/raid0/zufa/atom-runner/gh-config` on box308):

```bash
GH_CONFIG_DIR=<gh-config-dir> gh auth login --with-token < <token-file>
GH_CONFIG_DIR=<gh-config-dir> gh api repos/ROCm/ATOM --jq .full_name
```

Expected output: `ROCm/ATOM` (or `zufayu/ATOM` while registered against the fork).

## 4. Bot identity

`rocm-atom-bot` was registered on 2026-10-29 (`public_repo` scope, `pull` on the repo — enough
to comment on a public repo, though **not yet exercised against `ROCm/ATOM`**; see
`ADMIN-HANDOFF.md` §4 for the interaction-limit check that could silently block it).
Set the repo secret `ATOM_BOT_TOKEN` to that account's PAT. The workflow binds it as a step env var at five
places (gate's queue step, the watchdog job, review's claim step, review's notify step, and the
report job) — including steps that run inside the review job. The one step that is never given
it is `run_one.sh`'s own "Review (headless GLM)" step: that step carries no `ATOM_BOT_TOKEN` env,
and `run_one.sh` never reads or exports that name itself, so the headless review agent that runs
over untrusted PR content has no path to the token even though other steps in the same job do.

## 5. Keep it running across reboots *and* container restarts (host only)

Registering a runner does not keep it running, and on srdc-24 there are **two** ways to lose
it, not aiter's one (box308, having real systemd and no container in the loop, only has the
first):

- **Reboot** (aiter's failure mode, and srdc-24's too): a hand-started runner (`./run.sh`,
  `nohup`, a keeper loop) dies on the next reboot and never comes back. aiter's box lost its
  runner to exactly this and sat silently offline for days before anyone noticed — nothing
  pages anyone, because with no runner the job just sits `queued` instead of failing.
- **Container lifecycle** (srdc-24's own extra way to lose it, while it is still a container):
  if the runner process is ever started from inside an agent container instead of the host, it
  dies the moment that container stops or is rebuilt — same silent `queued` outcome, on a
  schedule nobody controls. This does not apply to box308, which is not containerized.

A systemd unit, installed from the host, is the fix:

```bash
cd <atom-runner-dir>   # /mnt/md0/atom-runner on srdc-24, /mnt/raid0/zufa/atom-runner on box308
sudo ./svc.sh install "$(whoami)"
sudo ./svc.sh start
U=$(systemctl list-unit-files 'actions.runner.*ATOM*' --no-legend | awk '{print $1; exit}')
[ -n "$U" ] || { echo "no runner unit -- svc.sh install did not take"; exit 1; }
sudo mkdir -p "/etc/systemd/system/$U.d"
printf '[Service]\nRestart=always\nRestartSec=10\n' | sudo tee "/etc/systemd/system/$U.d/restart.conf"
sudo systemctl daemon-reload && sudo systemctl restart "$U"
systemctl is-enabled "$U"; systemctl show -p Restart "$U"
```

On box308, confirm `$U` resolved to ATOM's own unit
(`actions.runner.ROCm-ATOM.srdc-8-atom-review.service`) and not aiter's — the `*ATOM*` glob
does not match aiter's `actions.runner.ROCm-aiter.srdc-8-aiter-review.service`, but it is cheap
to eyeball `$U` before proceeding, especially the first time both units exist side by side.

Expected: `enabled`, `Restart=always`. The unit `svc.sh` writes on its own carries no `Restart=`,
so it survives a reboot but not a crash — the drop-in above is the other half.

Do **not** hand-roll a unit or a keeper loop, and do **not** attempt this from inside a
container on srdc-24 — there is no systemd to install into there, and anything you start that
way inherits the container's own lifetime, which is the second failure mode above, not a fix
for the first.

## 6. Verify

```bash
cd /mnt/raid0/zufa/atom-bot-work && bash .claude/skills/review-pr/preflight.sh; echo "exit=$?"
```

All green except one expected warning (no `ATOM_BOT_TOKEN` yet — see section 4) means
`atom-bot review` can run end to end. `preflight.sh` also runs `selftest.sh`, which holds
`run_one.sh` to its guards (timeout routes to `flow` rather than paging the model owner, an
attempt that cannot fit the run budget is refused, a refuter timeout still publishes the card
only by admitting it) — run it directly to see which guard broke if it's red:

```bash
bash .claude/skills/review-pr/selftest.sh
```
