# Agentic performance page

Choose **Scenario: Agentic** in the benchmark dashboard filter bar, or visit
`benchmark-dashboard/agentic.html`. This standalone page reads the existing
`data.js` published by ATOM CI. It keeps the existing dashboard header,
colors, fonts, control styling and shared light/dark theme. Scenario navigation
preserves the selected hardware and restores the original filters on return. It selects Agentic / AgentX performance records
from the encoded `perf_point` payloads without modifying the dataset.

Select a model, hardware and workload/precision. The chart defaults to the
Pareto frontier and switches between two horizontal axes:

| Display | Published CI field | Definition |
| --- | --- | --- |
| P90 Interactivity | `interactivity_p90_itl` | `1 / p90(ITL)` |
| P90 Normalized Interactivity | `interactivity` with the P90 E2E normalized method | `1 / p90(ITL + TTFT / output_tokens)` |
| Total throughput per GPU | `tput_per_gpu` | Total input + output tok/s divided by all participating GPUs |

The vertical axis, Pareto frontier, peak selection and exports use tok/s/GPU.
The page prefers the published `tput_per_gpu` value. When it is unavailable,
it divides `total_tput` by a known positive `total_gpu` count; otherwise the
point has no vertical-axis value. PD counts both prefill and decode GPUs.

Latencies in the formulas are in seconds. Missing P90 values remain unavailable;
median TPOT is never substituted. The page retains the latest publication per
recipe, model, hardware, serving mode, workload, precision and concurrency.
Distinct recipes remain available until the two-dimensional Pareto calculation.

**ATOMesh** means PD disaggregation and uses circular markers; **ATOM** means
single-node aggregated serving and uses square markers. Serving mode and node
placement are separate: PD can also run on one physical node. Their frontiers
are computed independently, as are different model variants. Every plotted point
is labeled `c<concurrency>`; the tooltip, inspector, table and CSV also include
concurrency. Topology and the
original Actions run link are available in point details and the measurement
table. The page supports dark/light themes, shareable filters and CSV/PNG exports.

The original dashboard's data parsing, filtering, aggregation and publication
behavior are unchanged. Existing deployment jobs additionally copy the new
HTML page alongside `index.html`.

## Local preview

```sh
python tools/preview_benchmark_dashboard.py --refresh
```

Open the printed `agentic.html` URL (port 8890 by default). The preview downloads
published CI data and assets into a temporary directory and serves both pages.
Use `--no-serve` to prepare files without starting a server.

## Validation

```sh
python tests/test_dashboard_explorer.py
```

Checks cover the Agentic-only projection, latest-publication selection,
independent Pareto frontiers, P90 definitions, JavaScript syntax and CI preview.
