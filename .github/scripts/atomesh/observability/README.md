# Agentic PD inference reports

Agentic PD benchmarks (`benchmark.kind: aiperf_agentic`) automatically collect
metrics and generate an offline HTML report for each concurrency setting. No
additional workflow input or long-running monitoring server is required.

Each matrix job uploads its own `atomesh-latency-<matrix-id>-<attempt>` artifact.
The job summary contains **Download HTML reports and data**. Download and extract
that artifact, then open a `report.html` file. GitHub Actions summaries cannot
execute the report's JavaScript; the report runs locally without a server.

The report shows two charts per row:

- Mesh overall TTFT: ingress to first generated streaming output.
- Decode ITL: output intervals normalized and weighted by new token count.
- Prefill local TTFT: request arrival to first internal token delivery.
- Decode local TTFT: request arrival to first generated streaming output.
- Prefill and Decode request queues: running, waiting, and external KV waits.
- Prefill and Decode queue time: engine receipt to first forward dispatch,
  including input queue residence, scheduling, and KV loading waits.
- Actual decode batch size: real decode request rows in each forward.
- PD KV transfer wait: Decode-side remote load wait until all workers finish.
- Prefill and Decode KV block utilization: used, evictable cached, and vacant.

Mean, P50, P90, P95, and P99 can be toggled globally or per chart. The report also
supports hiding charts, time-range selection, a data table, and CSV export.
All / Prefill / Decode buttons filter both charts and metric selection buttons;
CSV exports follow that selection. In All view, shared metrics align Prefill on
the left and Decode on the right, followed by the remaining metrics. Individual
chart and series selections are retained when switching roles. Small screens
stack charts in the same order.
The global Statistics controls contain only Mean, P50, P90, P95, and P99.
Queue and KV state controls stay inside their own panels. Units are milliseconds,
requests, or percent as indicated on each chart and in the CSV.
The existing KV utilization panels also display summed Used / Total block
counts for the latest point in the selected range, or the hovered point.
Their data tables and CSV include the raw counts with unit `blocks`.
See [metric definitions](../../../../docs/agentic_metrics.md) for timing boundaries
and the distinction between PD transfer wait and pure RDMA time.

## Collection lifecycle

The workflow explicitly enables `ATOMESH_BUILD_MESH=true` for agentic cells.
Before launching the rank-0 service container, `../setup_mesh.sh` runs a separate
build container using `../build_mesh.sh`. It builds the reviewed source with
`ATOMESH_MESH_BUILD_PROFILE=release` (or an explicitly selected `ci` profile).
A writable source copy supports read-only checkouts. The binary, build log,
resolved lockfile and `mesh-build.json` (commit, profile, Cargo version, source
dirty status and checksums) are retained under the job's `mesh-build/` directory.
Benchmark and eval phases reuse that artifact through `ATOMESH_MESH_BINARY`;
router startup only consumes the path. A supplied `ATOMESH_MESH_BINARY` skips
building. Direct server launches default to the image's release binary.

For each AIPerf invocation, `collect_metrics.py` starts its own Prometheus process
on a dynamically assigned loopback port, scraping all resolved Prefill/Decode
addresses and Mesh's configured metrics port. Addresses and ports come from the
same arrays used to launch the services, including multiple hosts and workers.

Prometheus 3.5.0 is downloaded and verified against its release checksums when
no executable is already available. `ATOMESH_PROMETHEUS_BIN` can point to an
installed executable. `ATOMESH_MESH_TARGET_DIR` selects the host's persistent
Cargo build cache (default: `/tmp/atomesh-mesh-cache-<uid>`), mounted into the
setup container. If the image's Rust installation is inaccessible to the CI
user, setup installs Rust 1.94.0 into that cache; `ATOMESH_MESH_RUST_TOOLCHAIN`
can select another fallback version. These settings pass through the existing
CI environment handling and are independent of metrics collection.

The TSDB uses temporary node-local storage. Scraping and engine/API snapshots run
every second (subject to engine progress). Histogram observations accumulate at
each event, including events between scrapes. Plots use five-second steps;
histogram points summarize the preceding 60 seconds, while queues and KV panels
show sampled state. Brief queue peaks between snapshots may be missed.
Each invocation uses a
fresh TSDB and counter baselines, so previous benchmark traffic is excluded.
Collection covers the complete AIPerf invocation, including its warmup and drain.
API TTFT exposes zero-valued `streaming=true` and `streaming=false` series at
startup, so the collector can scrape a baseline before the first request.
Percentiles are histogram estimates; Prefill and Decode percentiles cannot be
added to obtain Mesh percentiles.

After the command finishes, the wrapper waits for a successful scrape from each
target with a scrape timestamp after the benchmark end, then waits until the
next five-second query step includes those scrapes. This wait is bounded to 30
seconds and stops early on interruption; failures are reported while retaining
available data. `status.json` keeps `end` and `benchmark_end` as the command's end
and records the export cutoff separately as `collection_end`. Report metadata
also includes `benchmark_end` and `collection_end`; its `end` covers the export
range. Report notes identify the extra collection interval, which is excluded
from the benchmark duration and may include other traffic during that interval.

The wrapper then collects report data, terminates Prometheus, finalizes
diagnostics, and renders HTML once. Benchmark failures
retain the original exit code and any available metrics. Publication failures
are recorded separately in `status.json` under `publication_errors`, and do not
replace the benchmark exit code. If the status file itself cannot be written,
the error is printed in the job log. Collection failures produce an explicitly
incomplete report and a warning in Actions instead of fabricated data.
Hard termination before export can leave only status and logs.
A panel is marked missing only when all its statistics lack valid samples.
Failed statistic queries still mark the report partial when other data is
available; a missing quantile alone does not make the report unavailable.

Reports live under
`slurm_job-<job-id>/benchmark_results/aiperf-<model>-<topology>-c<concurrency>/metrics/`.
Artifact staging selects only the current matrix ID and recorded Slurm job ID.
It never searches previous runs for a report when the current job has no report.

## Reusing the report interface

Export from a running Prometheus instance:

```bash
python .github/scripts/atomesh/observability/export_report.py \
  --prometheus-url http://127.0.0.1:9090 \
  --start 2026-09-08T10:46:23Z --end 2026-09-08T10:56:26Z \
  --deployment pd --model 'GLM-5.2 · CPP4 + DCP4' \
  --output report.html --save-data report-data.json
```

Rebuild HTML from archived data:

```bash
python .github/scripts/atomesh/observability/export_report.py \
  --input-json report-data.json --output report.html
```

Python callers can use `collect_report(...)` to fetch data without writing files,
`write_report(data, output)` to render data, or `generate_report(...)` as a
convenience API that does both. The JSON interface uses
Unix timestamps in seconds, values in the panel's `unit` (default `ms` for older
reports), and `null` for missing points. Gauge panels set `kind` to `queues` or
`blocks` and use state names as series keys. See `report-data.example.json` for
a small synthetic latency example.
KV panels additionally carry `block_counts.used` and `block_counts.total`
time series; older JSON without these fields displays unavailable counts as a dash.
