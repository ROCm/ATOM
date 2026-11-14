<!-- Maintained by the ATOM team. Please open an issue or ask a listed contact to change this page. -->

# Contributing to ATOM

> **DRAFT** — this page is being trialed and may change. Feedback is welcome in an issue.

Thank you for your interest in ATOM. This guide explains how to set up a development environment, what a
pull request needs before it can merge, and who to contact.

## Ways to Contribute

* **Bug reports and feature requests**: open a [GitHub issue](https://github.com/ROCm/ATOM/issues).
* **Code**: model support, engine and scheduling improvements, kernels integration, plugins, recipes, docs.
* **Code review**: review pull requests and give constructive feedback. See [REVIEWERS.md](REVIEWERS.md) for how
  to help review.

## Development Setup

Follow the Installation section of the [README](README.md). The nightly image `rocm/atom-dev:latest` has AITER and
ATOM ready to use; to work on ATOM itself, install your clone in editable mode inside the container:

```bash
git clone https://github.com/<your-account>/ATOM.git && cd ATOM
pip install -e .
```

## Code Quality

Every pull request runs **Pre Checkin**: Black, Ruff, catalog schema validation and the non-GPU unit tests.
Run the same checks locally before pushing:

```bash
pip install black ruff
black .
ruff check .
bash .github/scripts/run_unit_tests.sh
```

## Pull Request Process

### Before submitting

* Target `main`. Pull requests into other branches are not picked up for review.
* Keep the pull request focused on one change; split large work into a series of pull requests.
* Describe what changes and why, and how you validated it (models, hardware, parallelism, accuracy).
* For performance changes, include before/after numbers with the hardware and exact configuration.
* Add or update tests for behavior changes.

A pull request is ready for review when it is no longer a draft, its title is not marked WIP or DO NOT MERGE,
and Pre Checkin passes.

### CI

* **Pre Checkin** runs automatically on every pull request.
* **Heavy model accuracy tests** (native ATOM, vLLM and SGLang) run after the pull request is approved and Pre
  Checkin passes. They can be requested earlier with labels: `ci:full`, `ci:atom`, `ci:vllm`, `ci:sglang`.

### Merge requirements

`main` only accepts squash merges. Before a pull request can merge:

* It has at least one approving review.
* A new push dismisses earlier approvals, so batch your fixes instead of pushing one at a time.
* Every review conversation is resolved.
* The required model accuracy checks pass.

The pull request title is used for the squashed commit, so keep it accurate when the scope changes.

### Review timeline

A reviewer is usually assigned within one working day after the pull request is ready for review.
Most pull requests get a first review within a few working days. The exact time depends on the size and risk of
the change and on current project priorities; fixes for regressions and release blockers come first.

If nobody has responded for about a week, mention the contact for the area listed in
[CONTACTS.md](CONTACTS.md), then the backup contact.

On your side, keep CI green and reply to review comments. Pull requests with no activity for an extended period
(around six weeks) may be closed; you are welcome to reopen them.

### Pull requests from forks

GitHub requires a maintainer to approve workflow runs for first-time contributors, and the GPU accuracy tests
need repository resources that forks do not have. If your CI has not started, or an accuracy run is needed,
mention the area contact and a maintainer can help run it.

### Large changes

For new model families, new public APIs or changes to the engine core, open an issue first to agree on the
approach before writing the code.

## Code of Conduct

We follow the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). Please be respectful
and constructive in all interactions.
