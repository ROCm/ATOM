# AtomRL and Lumen GitHub Pages design

## Goal

Build the two in-tree Sphinx documentation projects after relevant changes land
on `main`, and publish them without disturbing the other content already stored
on ATOM's `gh-pages` branch:

- `atomrl/atomrl-docs/docs/source` → `https://rocm.github.io/ATOM/atomrl/`
- `atomrl/lumen/lumen-docs/docs/source` → `https://rocm.github.io/ATOM/lumen/`

The workflow must also support manual dispatch.

## Workflow boundary

Add a dedicated `.github/workflows/deploy-atomrl-docs.yml`. Keep
`.github/workflows/deploy-pages.yml`: it still owns the static content under
`.github/pages`, so it is not an obsolete trigger.

The new workflow triggers only for:

- changes under either Sphinx documentation tree;
- changes to either documentation requirements file;
- changes to the documentation deployment workflow itself;
- manual `workflow_dispatch`.

It does not add PR preview deployments or GPU jobs.

## Build

Use Ubuntu and Python 3.12. Install both documentation requirement sets, then
build both sites with:

```text
sphinx-build -W --keep-going -b html <source> <output>
```

Build into temporary output directories before touching `gh-pages`. Any Sphinx
warning or build failure prevents publication of both sites, avoiding a partial
update.

## Deployment

Use `contents: write` and the repository's existing `gh-pages-deploy`
concurrency group with `cancel-in-progress: false`. This serializes the write
with static-page and benchmark-dashboard publishers.

After both builds succeed:

1. Save the generated output outside the checkout.
2. fetch and check out `origin/gh-pages`;
3. replace only the `atomrl/` and `lumen/` directories;
4. ensure the branch root contains `.nojekyll`;
5. commit and push only when the generated content changed.

All unrelated files and directories on `gh-pages` remain untouched.

## Validation

Before push:

- build both Sphinx sites locally with warnings treated as errors;
- run Actionlint against the new workflow;
- verify both output directories contain `index.html`;
- verify the workflow uses the shared concurrency group and limits deletion to
  `atomrl/` and `lumen/`.

After merge, the first successful run publishes both URLs. GitHub Pages must
continue serving from the repository's `gh-pages` branch.
