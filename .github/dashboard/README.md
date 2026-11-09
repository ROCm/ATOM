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
publication per recipe, model, hardware, workload, precision and concurrency.
Distinct configurations survive until the two-dimensional Pareto calculation;
lower throughput alone does not eliminate a point with higher interactivity.
Frontiers stay within the selected source, hardware, model variant, workload
and precision.

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
preserves `run_attempt` in each link and rejects conflicting run IDs or attempts.
Open the printed URL to select the imported model and sweep automatically.
The Source dropdown switches between that sweep and published ATOM CI results;
results from different runs/attempts stay in separate sources.

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
results stay in the temporary preview directory. This does not configure PR
triggers or publish sweep artifacts to the hosted dashboard.

## Validation

```sh
python tests/test_dashboard_explorer.py
```

Checks cover CI payload parsing, latest-publication selection, preserving
distinct recipes, P90 definitions, Pareto ties and dominance, JavaScript syntax,
CI-only preview, sweep metric mapping, run provenance and safe local-data
embedding. Browser verification also checks
source values, Pareto filtering, CI run links, model and axis switching, URL
state, exports, themes, narrow screens and the existing tabs.
