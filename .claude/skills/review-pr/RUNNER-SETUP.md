# Runner setup — atom-review-bot

The `@rocm-atom-bot review` workflow (`.github/workflows/atom-review-bot.yml`) runs on a
self-hosted runner labeled `atom-bot-24`, on host `srdc-24`. This is what that runner must
provide. Verify the box with `bash .claude/skills/review-pr/preflight.sh` — it checks every item
below and prints a fix hint for each red. This doc explains how to make those checks pass.

## Why this box, and what must not be copied from aiter's setup

aiter's box serves its own GLM locally, which takes all eight GPUs. Copying that arrangement
here would leave nothing for validation. On this box the review agent reaches its model over
the network instead — the AMD gateway during development, box308's already-running GLM-5.3 in
the end — and the eight local GPUs stay free. Concretely, that means **tunnel mode**: the
runner's `.env` leaves `ATOM_REVIEW_AGENT` unset, so `run_one.sh` falls through to its own
`claude-glm` default, and `preflight.sh`'s "headless agent" check takes the tunnel-mode branch
(claude-glm on PATH + endpoint config), not the direct-agent branch aiter's identical-looking
`.env` recipe would trigger. If you move this setup to a box that serves its own model, say so
in this file, redo the GPU arithmetic, and switch to direct-agent mode deliberately (see
section 2) — do not let it happen by accident because `.env` carried over an aiter-shaped line.

## Install on a data volume, not the root fs

`/mnt/raid0` on this box is 100% full (126G free of a much larger disk, all of it already
spoken for). The runner lives on `/mnt/md0` instead, which has ~9.6T free and — importantly —
is a persistent volume that survives container rebuilds, unlike the container's own filesystem.
Each review checks out the repo and builds two worktrees; a box whose volume fills up mid-review
fails mid-review, silently if nobody is watching the disk.

## The container/host split — do sections 1 and 5 from the host, not from here

Agent sessions on this box (including whichever session produced this file) run inside a Docker
container: `PID 1` is `bash`, `systemctl is-system-running` returns `offline`, and `/.dockerenv`
exists. There is no systemd here, and no amount of `nohup` or a hand-rolled keeper loop fixes
that — see the warning in section 5. The runner's **files** can live on `/mnt/md0` because that
volume is reachable from both the container and the host. But **registering the runner
(section 1) and installing it as a systemd service (section 5) must be done from the host**,
logged in directly (SSH or console), not from inside this or any other agent container.

The reason this matters more than it looks: a runner process started by hand inside a container
dies the moment that container stops or is rebuilt, and nothing reports it. GitHub does not see
a crashed runner as a failure — it sees no runner, so the next `@rocm-atom-bot review` job just
sits in `queued` forever. That state fails nothing, notifies nobody, and looks from the PR like
"review still in progress," not "broken." aiter's own box lost its runner to a plain reboot this
same way and sat silently offline for days before anyone noticed. A systemd unit on the host,
with the `Restart=always` drop-in in section 5, is what prevents both the reboot case (aiter's
failure mode) and the container-lifecycle case (this box's extra failure mode, since aiter's
box was never containerized).

All of sections 1–5 below are **host-side, copy-pasteable commands for the user to run**, not
something an agent container should attempt — `config.sh` and `svc.sh install` need a real
systemd and, for registration, API credentials this environment's policy does not let an agent
mint on your behalf. Section 2 (`.env`) and section 6 (`preflight.sh`) can be run from either
side once the files exist, since they touch only `/mnt/md0`.

---

## 1. Register the runner (host only — needs your GitHub credentials)

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
  --labels atom-bot-24 --name srdc24-atom-review --work _work --unattended --replace
```

## 2. Box config via the runner `.env` (loaded into every job)

The workflow's review step sets no environment on purpose, so box-specific config lives in
`/mnt/md0/atom-runner/.env`, which the runner injects into every job:

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

Deliberately **not** set here: `ATOM_REVIEW_AGENT`. Leaving it unset is what keeps this box in
tunnel mode (`run_one.sh` defaults `AGENT` to `claude-glm`, which resolves the AMD gateway /
box308 endpoint on its own) — see "Why this box" above. Only set `ATOM_REVIEW_AGENT` (to a local
`claude` binary) together with `ANTHROPIC_BASE_URL` if this box is later reconfigured to serve
its own model directly; doing one without the other is exactly the mismatch this file exists to
prevent.

Also **not** set here, because they are policy, not machine fact, and already live in the repo:
`ATOM_RUN_BUDGET`, `ATOM_AGENT_TIMEOUT`, `ATOM_REVIEW_RETRIES`, any `*_OWNER`. A value parked in
`.env` outlives the reason it was added, and nothing reviews it — follow `run_one.sh`'s own
defaults and failure-message hints instead of pre-guessing a override here.

## 3. gh auth off the job environment

`fetch.sh` calls `gh` to read the PR. Authenticate it under `GH_CONFIG_DIR` rather than exporting
`GH_TOKEN`, so the headless review agent never inherits a token from its environment:

```bash
GH_CONFIG_DIR=/mnt/md0/atom-runner/gh-config gh auth login --with-token < <token-file>
GH_CONFIG_DIR=/mnt/md0/atom-runner/gh-config gh api repos/ROCm/ATOM --jq .full_name
```

Expected output: `ROCm/ATOM` (or `zufayu/ATOM` while registered against the fork).

## 4. Bot identity

P0 needs no token for this (`rocm-atom-bot` doesn't exist yet). When it does: set the repo secret
`ATOM_BOT_TOKEN` to the bot account's PAT. The workflow uses it only in the claim and publish
steps — never in the review step — so the review agent cannot read it.

## 5. Keep it running across reboots *and* container restarts (host only)

Registering a runner does not keep it running, and on this box there are **two** ways to lose
it, not aiter's one:

- **Reboot** (aiter's failure mode): a hand-started runner (`./run.sh`, `nohup`, a keeper loop)
  dies on the next reboot and never comes back. aiter's box lost its runner to exactly this and
  sat silently offline for days before anyone noticed — nothing pages anyone, because with no
  runner the job just sits `queued` instead of failing.
- **Container lifecycle** (this box's own extra way to lose it): if the runner process is ever
  started from inside an agent container instead of the host, it dies the moment that container
  stops or is rebuilt — same silent `queued` outcome, on a schedule nobody controls.

A systemd unit, installed from the host, is the fix for both:

```bash
cd /mnt/md0/atom-runner
sudo ./svc.sh install "$(whoami)"
sudo ./svc.sh start
U=$(systemctl list-unit-files 'actions.runner.*ATOM*' --no-legend | awk '{print $1; exit}')
[ -n "$U" ] || { echo "no runner unit -- svc.sh install did not take"; exit 1; }
sudo mkdir -p "/etc/systemd/system/$U.d"
printf '[Service]\nRestart=always\nRestartSec=10\n' | sudo tee "/etc/systemd/system/$U.d/restart.conf"
sudo systemctl daemon-reload && sudo systemctl restart "$U"
systemctl is-enabled "$U"; systemctl show -p Restart "$U"
```

Expected: `enabled`, `Restart=always`. The unit `svc.sh` writes on its own carries no `Restart=`,
so it survives a reboot but not a crash — the drop-in above is the other half.

Do **not** hand-roll a unit or a keeper loop, and do **not** attempt this from inside a
container — there is no systemd to install into, and anything you start that way inherits the
container's own lifetime, which is the second failure mode above, not a fix for the first.

## 6. Verify

```bash
cd /mnt/raid0/zufa/atom-bot-work && bash .claude/skills/review-pr/preflight.sh; echo "exit=$?"
```

All green except one expected warning (no `ATOM_BOT_TOKEN` yet — see section 4) means
`@rocm-atom-bot review` can run end to end. `preflight.sh` also runs `selftest.sh`, which holds
`run_one.sh` to its guards (timeout routes to `flow` rather than paging the model owner, an
attempt that cannot fit the run budget is refused, a refuter timeout still publishes the card
only by admitting it) — run it directly to see which guard broke if it's red:

```bash
bash .claude/skills/review-pr/selftest.sh
```
