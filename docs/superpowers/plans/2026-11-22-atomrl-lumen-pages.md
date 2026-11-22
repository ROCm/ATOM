# AtomRL and Lumen Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically build and publish the in-tree AtomRL and Lumen Sphinx documentation at `/ATOM/atomrl/` and `/ATOM/lumen/`.

**Architecture:** A dedicated GitHub Actions workflow builds both sites before changing branches, then uses ATOM's shared `gh-pages-deploy` lock to replace only the two owned directories on `gh-pages`. Existing static-page and dashboard publishers remain unchanged.

**Tech Stack:** GitHub Actions, Python 3.12, Sphinx, GitHub Pages, Git.

---

### Task 1: Add the documentation deployment workflow

**Files:**
- Create: `.github/workflows/deploy-atomrl-docs.yml`

- [ ] **Step 1: Add focused triggers and permissions**

Create a workflow triggered by `workflow_dispatch` and pushes to `main` that touch:

```yaml
paths:
  - ".github/workflows/deploy-atomrl-docs.yml"
  - "atomrl/atomrl-docs/**"
  - "atomrl/lumen/lumen-docs/**"
```

Give the deployment job `contents: write`, run it on Ubuntu, and serialize it with:

```yaml
concurrency:
  group: gh-pages-deploy
  cancel-in-progress: false
```

- [ ] **Step 2: Build both Sphinx sites before deployment**

Set up Python 3.12, install both documentation requirement files, and run:

```bash
sphinx-build -W --keep-going -b html \
  atomrl/atomrl-docs/docs/source /tmp/atomrl-docs
sphinx-build -W --keep-going -b html \
  atomrl/lumen/lumen-docs/docs/source /tmp/lumen-docs
test -f /tmp/atomrl-docs/index.html
test -f /tmp/lumen-docs/index.html
```

- [ ] **Step 3: Publish only the owned subdirectories**

Configure the Actions bot identity, preserve the source SHA, fetch and check out
`gh-pages`, replace only `atomrl/` and `lumen/`, create `.nojekyll`, and push a
commit only when `git diff --cached` is non-empty. Return to the source SHA at
the end.

- [ ] **Step 4: Validate workflow syntax**

Run:

```bash
actionlint .github/workflows/deploy-atomrl-docs.yml
```

Expected: exit code 0 with no findings.

- [ ] **Step 5: Commit the workflow**

```bash
git add .github/workflows/deploy-atomrl-docs.yml
git commit -m "ci(docs): publish AtomRL and Lumen Pages"
```

### Task 2: Verify the documentation builds and deployment boundaries

**Files:**
- Verify: `.github/workflows/deploy-atomrl-docs.yml`
- Verify: `atomrl/atomrl-docs/docs/source`
- Verify: `atomrl/lumen/lumen-docs/docs/source`

- [ ] **Step 1: Create an isolated documentation environment**

```bash
python -m venv .venv-docs
.venv-docs/bin/python -m pip install \
  -r atomrl/atomrl-docs/docs/requirements.txt \
  -r atomrl/lumen/lumen-docs/docs/requirements.txt
```

Expected: dependencies install successfully.

- [ ] **Step 2: Build AtomRL documentation with warnings as errors**

```bash
.venv-docs/bin/sphinx-build -W --keep-going -b html \
  atomrl/atomrl-docs/docs/source .docs-build/atomrl
test -f .docs-build/atomrl/index.html
```

Expected: Sphinx exits 0 and `index.html` exists.

- [ ] **Step 3: Build Lumen documentation with warnings as errors**

```bash
.venv-docs/bin/sphinx-build -W --keep-going -b html \
  atomrl/lumen/lumen-docs/docs/source .docs-build/lumen
test -f .docs-build/lumen/index.html
```

Expected: Sphinx exits 0 and `index.html` exists.

- [ ] **Step 4: Verify safe deployment scope**

Confirm the workflow:

```text
uses concurrency group gh-pages-deploy
removes only atomrl and lumen
does not use git clean, reset --hard, or force-push
creates .nojekyll
skips an empty commit
```

- [ ] **Step 5: Run repository checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended plan/workflow changes are present.
