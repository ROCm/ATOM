# What an ATOM admin has to do to turn this bot on

Everything in this file needs **admin rights on `ROCm/ATOM`**. Nothing else in the
project does — the code, the runner install, and the box config are all done or can be
done without them. Hand this page to whoever has those rights.

There are three things to create and one to check.

---

## 0. Already done — do not redo

| | state |
|---|---|
| machine account | `rocm-atom-bot` exists (registered 2026-10-29), `public_repo` scope, `pull` on `ROCm/ATOM` |
| its PAT | held by the ATOM side; hand it to the admin only for step 2 |
| box308 runner install path | planned at `/mnt/raid0/zufa/atom-runner/`, alongside aiter's |
| the workflow | `.github/workflows/atom-review-bot.yml`, on the PR branch, CI green |

**ATOM needs its own runner — aiter's cannot be shared.** aiter's unit is registered
against `ROCm/aiter`; a repository-scoped runner serves only the repo it was registered
to. Sharing one registration would mean re-registering aiter's at the `ROCm` org level,
which changes a setup that works today. Don't. See `RUNNER-SETUP.md` §"Two runners on
one box".

---

## 1. A runner registration token  ← the blocker

```bash
env -u GH_TOKEN -u GITHUB_TOKEN gh api -X POST \
  repos/ROCm/ATOM/actions/runners/registration-token --jq .token
```

or: repo **Settings → Actions → Runners → New self-hosted runner** (the page prints the
token inside the `./config.sh` line).

**It expires in one hour and is single-use.** So either run the registration yourself on
box308, or hand the token over and have the other side register within the hour. The
full install sequence is in `RUNNER-SETUP.md` §1 "On box308 (production)"; the part that
must match exactly is:

```
--labels atom-bot --name srdc-8-atom-review
```

The label `atom-bot` is what `runs-on: [self-hosted, atom-bot]` selects, and
`check_prompts.py` enforces that this label appears in no other workflow, so ATOM jobs
cannot land on aiter's runner or vice versa.

---

## 2. The repo secret `ATOM_BOT_TOKEN`

**Settings → Secrets and variables → Actions → New repository secret**

- name: `ATOM_BOT_TOKEN`
- value: the `rocm-atom-bot` PAT

This is what the bot posts comments with. The workflow itself declares only
`contents: read` and is granted no `issues`/`pull-requests` write anywhere, so this
secret is the single path to a published comment. It is bound at five steps and is
deliberately **not** given to the step that runs the headless review agent over
untrusted PR content.

---

## 3. Two repo variables (both optional)

**Settings → Secrets and variables → Actions → Variables**

| name | effect if unset |
|---|---|
| `ATOM_REVIEW_AUTHORIZED` | falls back to `authorized.txt` on the default branch — which is the intended steady state. Set it only to override the allowlist without a commit. |
| `ATOM_MAX_WAIT_HOURS` | the watchdog uses its built-in default. |

Neither blocks a first run.

---

## 4. Check one thing that could silently block every comment

`rocm-atom-bot` is **not** a member of the `ROCm` org. If the repo or the org has
interaction limits set to collaborators/members only, every comment the bot posts will
be rejected — and this cannot be checked without admin rights:

```bash
env -u GH_TOKEN -u GITHUB_TOKEN gh api repos/ROCm/ATOM/interaction-limits
env -u GH_TOKEN -u GITHUB_TOKEN gh api orgs/ROCm/interaction-limits
```

An empty `{}` from both means nothing is blocking. If a limit is set, either lift it for
this repo or add `rocm-atom-bot` as a repo collaborator with read access.

This is the one requirement that has **not** been verified end to end: the bot's ability
to comment on `ROCm/ATOM` specifically has never been exercised, because testing it would
mean posting to a real PR. Everything else in the pipeline has been run.

---

## What happens after

The PR stays a draft until a maintainer has read one of the review cards it produces
(`.claude/skills/review-pr/reports/PR-*/card.md` on the branch). Seven real ATOM PRs have
been run through the pipeline; no human has read a card yet, and no model's opinion
substitutes for that.
