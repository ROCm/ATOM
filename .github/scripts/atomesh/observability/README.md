# Agentic PD latency reports

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

Mean, P50, P90, P95, and P99 can be toggled globally or per chart. The report also
supports hiding charts, time-range selection, a data table, and CSV export.

## Collection lifecycle

The rank-0 benchmark container builds Mesh from the current checkout before
starting the router. A writable local source copy allows Cargo to create its
lockfile even though the checkout is mounted read-only. Build logs and the
resolved lockfile are included in the existing Slurm logs.

For each AIPerf invocation, `collect_metrics.py` starts its own Prometheus process
on a dynamically assigned loopback port, scraping all resolved Prefill/Decode
addresses and Mesh's configured metrics port. Addresses and ports come from the
same arrays used to launch the services, including multiple hosts and workers.

Prometheus 3.5.0 is downloaded and verified against its release checksums when
no executable is already available. `ATOMESH_PROMETHEUS_BIN` can point to an
installed executable. `ATOMESH_MESH_TARGET_DIR` can provide a writable Cargo
build cache. Both are passed through the existing CI environment handling.

The TSDB uses temporary node-local storage. Scraping runs every five seconds;
each plotted point summarizes the preceding 60 seconds. Each invocation uses a
fresh TSDB and counter baselines, so previous benchmark traffic is excluded.
Collection covers the complete AIPerf invocation, including its warmup and drain.
Percentiles are histogram estimates; Prefill and Decode percentiles cannot be
added to obtain Mesh percentiles.

After the command finishes, the wrapper exports HTML and JSON before terminating
Prometheus and deleting its temporary storage. Benchmark failures retain the
original exit code and any available metrics. Collection failures produce an
explicitly incomplete report and a warning in Actions instead of fabricated data.
Hard termination before export can leave only status and logs.

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

Python callers can use `generate_report(...)` to fetch Prometheus data or
`write_report(data, output)` to render their own data. The JSON interface uses
Unix timestamps in seconds, latency values in milliseconds, and `null` for
missing points. See `report-data.example.json` for a small synthetic example.
