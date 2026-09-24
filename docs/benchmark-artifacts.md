# Benchmark artifacts

Benchmark cells retain request evidence, configuration, telemetry and derived
metrics as files. The producer uses Python's standard library; it needs no
database or web service. A separate local viewer reads these files.

## Responsibilities

- Existing clients own execution and their legacy dashboard JSON. Single-node
  and ATOMesh AIPerf share `.github/scripts/aiperf_dashboard.py`; dataset,
  topology and installation policies remain in their own runners.
- `records.py` captures random requests and normalizes AIPerf records.
  `metadata.py` captures actual arguments and identities.
- `aggregate.py` computes request metrics and derived data, including SA mappings
  and export. Unsupported SA mappings do not invalidate ATOM bundles.
- `telemetry.py` collects GPU/server samples and derives optional power,
  Prometheus and cache observations without accessing GitHub.
- `bundle.py` owns packaging, indexing, rebuild, identity and integrity checks.
  Verification does not load configuration capture, aggregation or collectors.
- `io.py` provides bounded JSONL writes, strict JSON reads, hashes and safe paths.
- `__main__.py` exposes the producer CLI and loads each operation on demand.
- `.github/scripts/benchmark_bundle.sh` owns collector lifetime, client cleanup
  and final packaging. Drain failure and cancellation share the same cleanup.
- CI uploads each cell. Local consumers discover/index downloaded files; there
  is no extra CI matrix, run-index job or dashboard dependency on bundles.
- [AgenticViewer](https://github.com/valarLip/AgenticViewer) owns file indexing,
  GitHub downloads, legacy summary adaptation and HTTP/UI in a separate repository.

The producer has eight Python files under `atom/benchmarks/results/`:

```text
__init__.py     schema and aggregation versions
__main__.py     CLI
metadata.py     execution configuration and identities
records.py      request recording and normalization
aggregate.py    request metrics, derived data and compatible export
telemetry.py    GPU and server observations
bundle.py       packaging, verification, indexing and rebuild
io.py           file primitives
schemas/        five JSON schemas defining the bundle contract
```

The existing ATOMesh
observability report remains independent; its Prometheus process and TSDB are
not dependencies of this lightweight file producer.

## CI and configuration

The existing benchmark workflow collects bundles for random cells. Catalog
models/variants can set `bench_kind: aiperf_agentic`; the reusable template
passes the choice to `atom_test.sh`. The current catalog does not enable an
agentic cell. Configure it explicitly on the validation branch before an agentic
CI run. There are no new global dispatch inputs, and the experimental TP2
configuration is not added to nightly defaults.

Use the existing catalog `env_vars` or reusable workflow environment for:

| Variable | Purpose |
| --- | --- |
| `AIPERF_BENCHMARK_DURATION` | Profiling seconds; default 3600. Below 900 is an unsafe smoke run. |
| `ATOM_BUNDLE_REQUIRE_FULL=1` | Fail the cell when required evidence/identities are incomplete. |
| `BENCHMARK_MODEL_REVISION` | Checkpoint revision; an existing `.hf-revision` marker takes precedence. |
| `BENCHMARK_PRECISION` | Weight precision, separate from KV cache dtype. |
| `BENCHMARK_MODEL_KEY` / `BENCHMARK_HARDWARE` | Optional canonical SA export keys. |
| `ATOM_BENCHMARK_GPU_PCI_IDS` | Explicit allocated PCI IDs when HIP discovery cannot resolve allocation. |
| `AIPERF_DATASET_REVISION` | Dataset version; retained `inputs.json` also supplies a content hash. |

For manual shell runs, set `ATOM_BENCHMARK_CAPTURE=1` at server launch to capture
resolved arguments. CI sets this only on benchmark containers. A standalone
random client records requests when `ATOM_BENCHMARK_REQUESTS` names its JSONL
output. With recording disabled, it calls the original request function directly
and does not import the producer.

Workflow SHA is separate from actual ATOM/AITER/harness identities and the
running image ID. Unknown values stay unknown. GPU allocation uses TP/PP/DP and
HIP-to-PCI discovery; ambiguous subsets are not guessed from DRM numbering.
Physical GPU IDs are evidence, not curve-group dimensions.

## Files and publication

```text
benchmark-bundles/<RESULT_FILENAME>/
  manifest.json                 # version, identities, capabilities, hashes
  config.json                   # resolved execution settings
  summary.json                  # request metrics and timing window
  validation.json               # completeness, validity, export diagnostics
  requests/requests.jsonl.gz     # normalized records, including warmup/failures
  raw/                          # unchanged harness records/summary and exports
  logs/                         # compressed client/server/collector logs
  telemetry/                    # GPU samples and Prometheus responses
  exports/inferencex-v3.json     # present when SA dimensions are supported
  views/                        # overview, distributions, timeline and series
```

Missing observations have capabilities and reasons, never invented zeroes.
Synthetic/unsafe measurements retain evidence but are invalid for comparison.
The producer does not replace the legacy summary or change its statistical
window. Normalized data and the original summary remain separate.

Each cell uploads small `atom-benchmark-summary-v1-<attempt>-<cell>` and complete
`atom-benchmark-bundle-v1-<attempt>-<cell>` artifacts. The summary includes the
complete manifest for later detail downloads. Creating it verifies only summary
files; full verification explicitly scans the complete bundle. The existing
`benchmark-*` artifact holds the client's own JSON. Unpackaged failure evidence
is uploaded before container cleanup. Original benchmark failures remain failures.
Bundles retain for 30 days, diagnostics 14.

## Local operations

```bash
python3 -m atom.benchmarks.results verify /path/to/bundle --require-full
python3 -m atom.benchmarks.results rebuild /path/to/bundle --output /path/to/rebuilt
python3 -m atom.benchmarks.results index /path/to/bundles --output /path/to/index.json
```

`build --config ... --records ... --output ...` packages existing evidence.
For AIPerf, pass `--record-format aiperf --raw-dir ...`. `--require-full` rejects
incomplete evidence while preserving a diagnostic bundle. Index generation is
optional; `--plan` accepts a supplied cell list to report missing cells. Existing
outputs cannot be overwritten. Rebuild verifies the source, then regenerates
metrics into a new directory from retained originals.
Rebuild preserves point identity within an aggregation version; floating-point
summaries may differ at machine precision across Python versions. Version 1.0.1
adds captured client performance settings to recipe identity, so rebuilding a
1.0.0 bundle can produce a new point ID. The original bundle remains intact.

## Metrics and resource use

- Latencies use seconds; absolute nanoseconds are decimal strings. Only eligible
  successful profiling requests contribute to metrics. Warmup, failures and drain
  remain in evidence and diagnostic timelines.
- Full-response ITL uses an explicit metric, then full decode duration, then
  `(end - start - TTFT)/(OSL - 1)`. Native chunk ITL is retained separately.
- P90 interactivity is `1/P90(full-response ITL)`. Normalized interactivity is
  `1/P90(each request's E2EL/actual OSL)`, not a ratio of aggregate statistics.
- Throughput uses the first eligible start through the last eligible end;
  per-GPU throughput divides by allocated GPUs. Quantiles interpolate at
  `(n - 1) * p`; standard deviation is population-based.
- Exact quantiles retain scalar arrays and request IDs, so memory still grows
  with request count. Statistics are computed once per metric and reused across
  summaries/views. Timelines stream in chunks of at most 1000 requests.
- JSONL uses a 64 KiB buffer, flushed on close and on writes at least one second
  after the preceding flush. Abrupt termination can lose the buffered tail;
  the original nonzero exit identifies the failed measurement.
- Power integration streams per device, clips to the request window, and rejects
  missing boundaries, non-increasing timestamps and gaps over three seconds.
  Server/GPU views keep bounded min/max/mean buckets. Raw evidence is retained;
  server counter resets invalidate affected deltas.

## Visualization in AgenticViewer

The independent [AgenticViewer](https://github.com/valarLip/AgenticViewer)
repository owns the HTTP server, browser UI, chart styling and interactions,
GitHub downloads, SemiAnalysis adapters and all viewer tests. ATOM produces
versioned files; it does not import, install or run AgenticViewer in CI.

From an AgenticViewer checkout, run:

```bash
python3 -m agentic_viewer --data-dir /path/to/benchmark-data --port 8765
```

Existing extracted bundles, summary packages and GitHub artifact names remain
compatible. See AgenticViewer's README and `docs/viewer-guide.md` for viewer setup.
The `views/` JSON files in bundles are derived numeric evidence, not HTML or
chart code. Their generation remains part of the reproducible producer output.

## Verification

```bash
python3 -m pytest -q --confcutdir=tests/benchmarks/results \
  tests/benchmarks/results tests/test_benchmark_catalog.py tests/test_benchmark_random_dataset.py
python3 tests/benchmarks/results/check_inferencex_compat.py \
  --source /path/to/pinned/InferenceX-app --core-source /path/to/pinned/InferenceX \
  --report /tmp/compatibility-report.json
```

Compatibility tests execute pinned SA code; hashes live in the test fixture.
Node is a development dependency only. SA's v3 flattener does not consume nested
normalized interactivity; a local viewer reads it from the ATOM summary without
SA database IDs. Schemas under `results/schemas/` describe the file contract.

CPU tests do not establish GPU overhead or real-run completeness. Acceptance
requires a random full bundle, an agentic run with at least two comparable
concurrency points, and download/rebuild after runner cleanup.

For CI acceptance, use an explicit ref containing the producer changes. Confirm
that summary/full artifacts and failure diagnostics upload before cleanup, then
download and verify/rebuild them on an independent CPU environment. A local ZIP
round-trip does not establish this GitHub upload/download path.
