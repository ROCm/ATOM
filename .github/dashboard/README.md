# Benchmark dashboard

The disaggregated performance explorer defaults to ATOM CI data and Pareto-only
view. Select a model, source and workload, then switch between P90 Interactivity
and P90 Normalized Interactivity. The vertical axis is total input + output
throughput across all participating GPUs, in tok/s.

The chart uses solid Pareto lines, inline series labels and a dashed ring around
the selected point. The side panel shows its measurements and CI run, initially
for the peak-throughput point. Selecting another point updates the panel without
covering the chart. The measurement table also displays CI run numbers and links.

## CI data

The dashboard consumes the existing publication path:

```text
process_result.py → atomesh-benchmark-action.json → benchmark-dashboard/data.js
```

The explorer reads each encoded `perf_point` directly, retaining the latest
publication per recipe, model, hardware, serving mode, workload, precision and concurrency.
Distinct configurations survive until the two-dimensional Pareto calculation;
lower throughput alone does not eliminate a point with higher interactivity.
Frontiers stay within each source, hardware, model variant, serving mode,
workload and precision. **ATOM** denotes the single-node aggregated engine and
uses blue square markers; **ATOMesh** denotes PD disaggregation and uses pink
circles. Serving mode is independent of physical node count: PD can run on
one node. Topology and node placement are shown in point details.

| Axis | CI field | Definition |
| --- | --- | --- |
| P90 Interactivity | `interactivity_p90_itl` | `1 / p90(ITL)` |
| P90 Normalized Interactivity | `interactivity` with `interactivity_method=p90_e2e_normalized` | `1 / p90(ITL + TTFT / output_tokens)` |
| Total throughput | `total_tput` | Total input + output tok/s across the configuration |

Latencies in the formulas are in seconds. Older rows without the selected P90
definition remain available in the measurement table and are omitted from the
chart. Median TPOT is never substituted for P90.

## Local preview

```sh
python tools/preview_benchmark_dashboard.py --refresh
```

Open the printed local URL (port 8890 by default). The script downloads the
current published data and existing dashboard assets into a temporary
directory. Use `--refresh` to update cached assets or `--no-serve` to only
prepare files. It does not publish changes or alter the source data.

Optional local measurements can be supplied with `--points path/to/results.json`
(repeatable). Each file contains a `points` array using the CI `perf_point`
schema. Add `source_label` to distinguish local results and `run_url` to retain
their original run links. No sample data files are bundled with the dashboard.

## Import a sweep

Import the `agg_bmk.json` array produced by InferenceX's `results_bmk` artifact
directly, without rewriting it into the dashboard's point schema:

```sh
python tools/preview_benchmark_dashboard.py --sweep path/to/agg_bmk.json --run-url https://github.com/OWNER/REPO/actions/runs/RUN_ID
```

Replace the path and URL with the downloaded artifact and its original run.
If every record already contains `run_url`, the flag is optional. The importer
preserves `run_attempt` in each point's link and rejects conflicting run IDs or attempts.
Open the printed URL to select the imported model and sweep automatically.
The Source dropdown switches between that sweep and published ATOM CI results;
results from different runs stay in separate sources. **Compare sources**
overlays their curves while calculating each Pareto frontier independently.
When an ATOM bundle declares `sweep_run_attempt`, successful points retained
from earlier attempts of the same run belong to that assembly's source. Their
links still open the original point attempts; the source link opens the assembly.
Without this field, imported attempts remain separate sources.

This importer supports ATOM AgentX records (`atom`, `atom-disagg`, `atomesh`)
with `request_metrics`. Other engines in mixed sweeps are skipped. It retains
recipe fingerprints, model, hardware, precision, concurrency, GPU counts and
the following measured fields:

| Dashboard value | Sweep field |
| --- | --- |
| P90 Interactivity | `request_metrics.latency.intvty.p90` |
| P90 Normalized Interactivity | `request_metrics.latency.e2e_norm_intvty.p90` |
| Total throughput | `request_metrics.throughput.total.tokens_per_second` |

Total throughput is not the `per_gpu` value. Missing P90 measurements remain
unavailable; rounded latency or mean values are never substituted. Imported
results stay in the temporary preview directory and are not published to the
hosted dashboard.

## Produce a curve in ATOM CI

Add or modify one `.github/benchmark/sweeps/NAME.yaml` in a ready PR, or dispatch
**ATOM PR Benchmark Sweep** manually with that configuration's path. Each run
benchmarks one model, hardware, workload and serving mode using a pinned image
and source commit. See [sweep configuration and retries](../benchmark/sweeps/README.md).

Each concurrency point is a job. **Re-run failed jobs** retries failures and
retains successful measurements only from the same run, commit and recipe.
The collector validates job outcomes and both P90 metrics, then uploads
`results_bmk-attempt-N` with `agg_bmk.json` and a completeness manifest. Missing
points leave the sweep partial and the completeness check failed. Import the
downloaded JSON with `--sweep`; ATOM bundles already contain exact run links.

## Validation

```sh
python tests/test_dashboard_explorer.py
python tests/test_atomesh_pr_sweep.py
```

Checks cover CI payload parsing, latest-publication selection, preserving
distinct recipes, P90 definitions, Pareto ties and dominance, JavaScript syntax,
CI-only preview, sweep metric mapping, serving mode, retry provenance and safe local-data
embedding. Browser verification also checks
source values, Pareto filtering, CI run links, model and axis switching, URL
state, exports, themes, narrow screens and the existing tabs.
