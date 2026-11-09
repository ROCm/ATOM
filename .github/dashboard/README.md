# Agentic performance page

Open **Agentic performance** from the benchmark dashboard, or visit
`benchmark-dashboard/agentic.html`. This standalone page reads the existing
`data.js` published by ATOM CI. It selects Agentic / AgentX performance records
from the encoded `perf_point` payloads without modifying the dataset.

Select a model, hardware and workload/precision. The chart defaults to the
Pareto frontier and switches between two horizontal axes:

| Display | Published CI field | Definition |
| --- | --- | --- |
| P90 Interactivity | `interactivity_p90_itl` | `1 / p90(ITL)` |
| P90 Normalized Interactivity | `interactivity` with the P90 E2E normalized method | `1 / p90(ITL + TTFT / output_tokens)` |
| Total throughput | `total_tput` | Total input + output tok/s across all participating GPUs |

Latencies in the formulas are in seconds. Missing P90 values remain unavailable;
median TPOT is never substituted. The page retains the latest publication per
recipe, model, hardware, serving mode, workload, precision and concurrency.
Distinct recipes remain available until the two-dimensional Pareto calculation.

**ATOMesh** means PD disaggregation and uses circular markers; **ATOM** means
single-node aggregated serving and uses square markers. Serving mode and node
placement are separate: PD can also run on one physical node. Their frontiers
are computed independently, as are different model variants. Topology and the
original Actions run link are available in point details and the measurement
table. The page supports dark/light themes, shareable filters and CSV/PNG exports.

The original dashboard's JavaScript, data parsing, aggregation and publication
behavior are unchanged. Existing deployment jobs additionally copy the new
HTML page alongside `index.html`.

## Local preview

```sh
python tools/preview_benchmark_dashboard.py --refresh
```

Open the printed `agentic.html` URL (port 8890 by default). The preview downloads
published CI data and assets into a temporary directory and serves both pages.
Use `--no-serve` to prepare files without starting a server.

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

Checks cover the Agentic-only projection, latest-publication selection,
independent Pareto frontiers, P90 definitions, JavaScript syntax, CI preview,
sweep metrics and retry provenance. Local sweep imports affect only the Agentic
page; the original dashboard continues to read published CI data.
