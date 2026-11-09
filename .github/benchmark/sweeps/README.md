# Benchmark sweeps

Add or modify **one** `NAME.yaml` file in this directory in a PR to `main`.
When the PR is ready for review, `ATOM PR Benchmark Sweep` runs its selected
recipes at the PR's exact head commit. Updates to the PR replace its pending
sweep. Each configuration describes one model, hardware, workload and serving
mode; submit separate configuration PRs for ATOM (single-node aggregated) and
ATOMesh (PD disaggregated) curves.
Alternatively, use **Actions → ATOM PR Benchmark Sweep → Run workflow** and
enter the configuration's path. A manual run uses the selected branch's exact
commit and produces the same curve format without requiring a PR. The workflow
must exist on the default branch before GitHub exposes manual dispatch.

```yaml
version: 1
model: GLM-5.2-MXFP4
hardware: mi355x
mode: pd
image: rocm/atom-dev@sha256:REPLACE_WITH_THE_64_CHARACTER_IMAGE_DIGEST
suite: mooncake_store
cases:
  - glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c48
  - glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c56
  - glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c72
  - glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c96
  - glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c112
  - glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-mcstore-1m-c128
```

Use a real pinned image digest and recipe names from `models_atomesh.yaml`.
`mi355x` selects TW; `mi350x` selects Spur. `mode` is `pd` or `aggregated`;
the planner checks it against every selected recipe. PD can run on a single
physical node, so serving mode and node placement are recorded separately.

Each recipe uses its configured concurrency list. An optional
`concurrency: [16, 32, 64]` overrides that list for every selected recipe;
use it with a single recipe rather than repeating the same configuration under
several `-cN` names. Only AgentX / `aiperf_agentic` recipes are accepted because
the chart requires both P90 interactivity definitions.

The workflow reuses the existing ATOMesh benchmark runner for both serving
modes and creates a separate job for **each concurrency point**. Jobs are
serialized within a recipe, with at most two recipe groups active at once.
Its `results_bmk-attempt-N` artifact contains:

- `agg_bmk.json`: measured points using the InferenceX AgentX field layout,
  with PR number, tested commit, recipe fingerprint, run URL and run attempt.
- `sweep_manifest.json`: the selected configuration, expected and measured
  point counts, missing points, retained point attempts, and `complete` or
  `partial` status.
- `sweep-jobs.json`: the point job outcomes used to validate the assembly.

A missing or failed point makes the sweep partial and fails the completeness
check after uploading available results. Points are never filled from another
run, commit or recipe. The workflow has read permissions and does not deploy
to GitHub Pages. Download `results_bmk-attempt-N` from the completed attempt,
then import the curve locally:

```sh
python tools/preview_benchmark_dashboard.py --sweep path/to/agg_bmk.json
```

ATOM's sweep artifact already contains each point's run URL, so `--run-url`
is unnecessary. The dashboard labels the source with its PR/run, distinguishes
ATOM and ATOMesh series, and computes their Pareto frontiers independently.
Use **Source → Compare sources** to overlay imported results and published CI
results for the same model, hardware, workload and precision.

It does not run a GPU sweep for changes to this README or dashboard code alone.

## Retry a partial sweep

The logical curve belongs to one Actions **run ID**. On a transient failure,
resolve the cause and select **Re-run failed jobs** in that run. There is no
automatic retry loop. Healthy sibling jobs continue after a point fails.
The point job itself validates both P90 metrics, so missing metrics fail that
job rather than only failing the final collector.

The collector reads job outcomes across attempts and accepts a cell's latest
successful execution only when that cell's latest job succeeded. A failed,
cancelled or missing latest execution stays missing even if an older result
exists. It verifies the saved plan, exact commit, recipe, concurrency, run ID
and point attempt before retaining a measurement. Artifacts and trace names
include the attempt to avoid collisions on retries.

For example, attempt 1 succeeds at c48 but fails at c56. After **Re-run failed
jobs**, c56 succeeds in attempt 2. The assembly contains c48 from attempt 1
and c56 from attempt 2; each point keeps its original `run_attempt` and run URL,
while both carry `sweep_run_attempt: 2`. The dashboard displays one assembled
source and marks retained points in the inspector. All planned points and jobs
must pass before its status becomes `complete`.

If only artifact download or collection failed, retrying that job can reuse
the successful point artifacts. If older artifacts have expired, **Re-run all
jobs** in the same run to regenerate them. Fixes requiring a changed commit or
recipe produce a new run and a separate curve. GitHub may also require a
maintainer to approve a fork PR's workflow before allocating runners.
