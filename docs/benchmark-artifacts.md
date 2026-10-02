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

The existing ATOM Benchmark workflow collects bundles for random cells. The
independent **ATOM Agentic Benchmark** workflow
(`.github/workflows/atom-agentic-benchmark.yaml`) runs daily at 08:17 UTC and can
also be dispatched manually. It loads a profile-specific catalog and reuses
`benchmark-tmpl.yml` for execution, collection and artifact uploads.
Visualization and long-term data browsing remain in AgenticViewer.

The default manual **test** profile (`models_agentic.json`) runs DeepSeek V4.1 Flash, FP8 weights, DSpark5, TP4,
BF16 KV and FP8 index cache, using `FULL` CUDA graphs at concurrency 2 and 4.
Each cell profiles for 900 seconds, uses a 262144-token context limit and 10
additional warmup requests per lane. The replay dataset is
`semianalysis_cc_traces_weka_062126` with 393 entries. Strict full-bundle validation
is enabled. The initial FULL-graph CI run completed both replays; its bundles
exposed missing checkout and dataset identities, now covered by capture regressions.

The scheduled **nightly** profile (`models_agentic_nightly.json`) is based on
[InferenceX PR #3387](https://github.com/SemiAnalysisAI/InferenceX/pull/3387) at
`89384690e5ebafd5407e965ec95b648f869524bb`, with a custom concurrency grid,
memory utilization and unified graph capture sizes:

| Profile | TP | Concurrency | Seconds per point | Warmup per lane |
| --- | --- | --- | --- | --- |
| test (manual default) | 4 | 2, 4 | 900 | 10 |
| nightly | 2 | 1, 2, 4, 8, 16, 32, 64, 128 | 3600 | 5 |
| nightly | 4 | 1, 4, 8 | 3600 | 5 |

Nightly uses FP4 weight metadata, BF16 KV, FP8 index cache, five-token DSpark
with fixed AL 3.51, 16K batching/prefill chunks, prefix caching
with block size 16, 8K state checkpoints, level 3, FULL graphs and `dsml_v41`.
Both test and nightly set `--gpu-memory-utilization 0.95`, leave `--max-num-seqs`
unset (the checked-out engine supplies its default), and capture sizes 1–32 plus
48/64/96/128/160/192/224/256 for every point. There is no EP or KV offload.
Fixed acceptance is marked synthetic in the bundle. Complete, successful fixed-AL
measurements form their own performance curves, with an explicit Synthetic/AL
label; they are not accuracy evidence or eligible for official submissions.
The producer derives forced acceptance from the executed server arguments,
including manual overrides, and records disagreements with the catalog declaration.

Agentic catalogs declare a `concurrency` list on each model variant. They do not
use random `scenarios`, ISL/OSL, length ratios or concurrency bands. The builder
rejects unsupported fields and duplicate artifact prefixes before GPU allocation.
`AIPERF_SCENARIO` is distinct: it selects the actual AIPerf trace-replay scenario.
Agentic artifact names use `<prefix><suffix>-c<concurrency>`; random artifact
names retain their input/output-length and ratio dimensions. Old bundles remain
readable through their manifests.

Both profiles use `rocm/atom-dev:latest` unless manually overridden. The image
supplies dependencies; ATOM executes from the checked-out repository. By default,
the checkout is the triggering branch's event SHA, rather than a later branch
head or the image's baked-in ATOM source. The matrix and GPU jobs use the same
resolved SHA, and a preflight verifies both the SHA and the `/workspace/atom`
Python import path. An explicit `atom_commit` remains available for controlled
replays. The CPU matrix job resolves the requested image to one immutable digest
for the entire run, including dispatch replay inputs. A same-digest nightly tag
is used for display when available; GPU cells verify and record the running image
ID. Each new nightly run resolves the then-current latest image.

`enable_profiler` retains graph-capture traces and starts a bounded replay sample
after AIPerf reports its profiling phase (after warmup). The replay sample defaults
to 30 seconds; `ATOM_AGENTIC_PROFILE_SECONDS` permits a window up to 120 seconds.
The wrapper stops the profiler on completion, error or cancellation and retains
`raw/profiler-window.json`. Profiler failures fail the client step, and trace
upload is attempted even on failed runs. Instrumented measurements are labeled
and excluded from submission eligibility.

Manual inputs select a profile and model prefixes, override profiling duration
(900–3600 seconds), and optionally select image, runner or code refs. Empty
model/concurrency/duration inputs use the profile. Nightly concurrency overrides
select a subset of each TP variant's existing points.
Use a ref containing this workflow and producer.
The scheduled workflow becomes active when it is on the default branch; it does
not run from a PR. Each run retains its own artifacts without updating gh-pages.

Choose **ATOM Agentic Benchmark** in the Actions sidebar for agentic runs.
The existing **ATOM Benchmark** retains its random-workload model checkboxes
and dashboard settings. These are separate forms: GitHub dispatch inputs do
not conditionally hide fields based on a selected profile. The temporary
`agentic_profile` bridge used for the initial branch validation has been removed.
The independent workflow must reach the default branch before its manual and
scheduled entry is available; adding a file only to a PR branch does not
register that entry. Completed validation runs remain available in their history.

For the standard manual test, select the branch under **Use workflow from**,
leave **Preset** at `test` and click **Run workflow**. Model, concurrency and
duration overrides can stay empty. Both presets use DSpark5 and FULL graphs.
Use `nightly` to run the configured 11-point grid, or enable **Preview
configuration only** to inspect it without allocating GPUs. Image, runner,
code overrides and profiler settings are labeled **Advanced**; they do not
need changing for the default test. Jobs remain grouped by model configuration
with concurrency points underneath.

Actions run titles follow the existing benchmark convention: `manual (<actor>)`
or `nightly`, with GitHub's native run number shown alongside. The separate
workflow has its own run-number sequence. Scheduled and manual runs have separate
concurrency groups; neither cancels an executing run.

Select **dry_run** to preview the resolved matrix using only a CPU runner. The
Actions summary shows the point count, concurrency, duration, checkout SHA and
expandable server/environment details. Every run saves an
`atom-agentic-run-config-<attempt>` artifact containing `run-config.json`,
`dispatch-inputs.json` and a README with a `gh workflow run` command. The dispatch
file pins the ATOM checkout and retains the preview flag; set `dry_run` to `false`
to execute. Image/AITER refs still need pinning for version comparisons. Each
GPU job also links its uploaded summary, full bundle and failure diagnostics;
an uploaded full bundle can still be incomplete after a failed run. Missing
requirements appear in the CLI output and per-point Actions summary. Configuration
and data artifacts are retained for 15 days (the repository limit), diagnostics for 14 days; download
them before expiry for long-term storage.

Model revision is read from the downloaded checkpoint's `.hf-revision` marker;
the executing image, AITER and AIPerf identities are captured at runtime. Dataset
identity comes from retained `inputs.json` when available. For Weka (which skips
that export), the replay uses a fresh `HF_DATASETS_CACHE` with AIPerf mmap reuse
disabled. After replay, `capture-hf-dataset` hashes the actual Arrow files and
`dataset_info.json`, and retains `raw/aiperf/dataset-identity.json.gz` with their
sizes, SHA-256 hashes and exported dataset provenance. The workload content hash
is the canonical hash of this ledger. Dataset bytes stay in temporary storage
and are removed after capture; they are not duplicated in uploaded artifacts.
No current remote revision or historical cache is substituted for loaded data.
The zero ISL/OSL and ratio in artifact names are shared-template placeholders;
agentic workload metadata leaves these dimensions null and uses actual token
counts. The random nightly/weekly catalog remains independent.

Use the existing catalog `env_vars` or reusable workflow environment for:

| Variable | Purpose |
| --- | --- |
| `AIPERF_BENCHMARK_DURATION` | Profiling seconds; default 3600. Below 900 is an unsafe smoke run. |
| `ATOM_BUNDLE_REQUIRE_FULL=1` | Fail the cell when required evidence/identities are incomplete. |
| `BENCHMARK_MODEL_REVISION` | Checkpoint revision; an existing `.hf-revision` marker takes precedence. |
| `BENCHMARK_PRECISION` | Weight precision, separate from KV cache dtype. |
| `BENCHMARK_MODEL_KEY` / `BENCHMARK_HARDWARE` | Optional canonical SA export keys. |
| `ATOM_BENCHMARK_GPU_PCI_IDS` | Explicit allocated PCI IDs when HIP discovery cannot resolve allocation. |
| `AIPERF_DATASET_REVISION` | Optional known dataset version; automatic capture uses the actual input content hash. |

For manual shell runs, set `ATOM_BENCHMARK_CAPTURE=1` at server launch to capture
resolved arguments. CI sets this only on benchmark containers. A standalone
random client records requests when `ATOM_BENCHMARK_REQUESTS` names its JSONL
output. With recording disabled, it calls the original request function directly
and does not import the producer.

Workflow SHA is separate from actual ATOM/AITER/harness identities and the
running image ID. Unknown values stay unknown. GPU allocation uses TP/PP/DP and
HIP-to-PCI discovery; ambiguous subsets are not guessed from DRM numbering.
Physical GPU IDs are evidence, not curve-group dimensions.

AITER identity is read from the installed `amd-aiter` distribution. A Git
installation records its commit from `direct_url.json` or its local source
directory. A wheel with a clean `+g<commit>` version suffix (for example,
`0.1.1.dev1+ga75ba53de`) records that abbreviation as `software.aiter_sha` with
`aiter_sha_source=package_version`; it is not expanded to a guessed full SHA.
`aiter_version` retains the exact package version, and `aiter_wheel_sha256`
retains pip's wheel archive SHA-256 when available. Index installs may not retain
an archive hash. A separate checkout or requested wheel pin is never substituted
for the installed package. Unknown commit identities still fail strict validation.

A successful replay with missing bundle metadata has `status=partial` and its
original client exit code remains zero in `validation.json`. With
`ATOM_BUNDLE_REQUIRE_FULL=1`, packaging returns exit code 2 and fails the Actions
step, while retaining the diagnostic bundle. Nonzero client exits remain failed
measurements even when all package identities are available.

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
`measurement_valid` requires complete metadata/evidence, valid records, successful
requests, a zero client exit and no unsafe override or AIPerf submission rejection.
`submission_eligible` additionally excludes synthetic and instrumented experiments.
Invalid measurements retain their evidence for diagnosis.
The producer does not replace the legacy summary or change its statistical
window. Normalized data and the original summary remain separate.

Each cell uploads small `atom-benchmark-summary-v1-<attempt>-<cell>` and complete
`atom-benchmark-bundle-v1-<attempt>-<cell>` artifacts. The summary includes the
complete manifest for later detail downloads. Creating it verifies only summary
files; full verification explicitly scans the complete bundle. The existing
`benchmark-*` artifact holds the client's own JSON. Unpackaged failure evidence
is uploaded before container cleanup. Original benchmark failures remain failures.
Bundles retain for 15 days, diagnostics 14.

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
1.0.0 bundle can produce a new point ID. Version 1.0.2 separates measurement
validity from submission eligibility and derives synthetic acceptance from the
executed arguments. Older bundles remain readable; rebuild with the new producer
to apply the corrected classification, which can change their point identity.
The original bundle remains intact.

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

Random dataset generation and client tests do not load server-side chat encoders
or their model-weight dependencies. The benchmark CLI loads encoders when it runs;
importing the dataset helpers does not require the engine or PyTorch.
The shared non-GPU suite also collects these tests and installs `jsonschema` for
the bundle contract assertions. The results directory has no local `conftest.py`,
so it cannot shadow the shared fixtures imported by existing engine tests.

CPU tests do not establish GPU overhead or real-run completeness. Acceptance
requires a random full bundle, an agentic run with at least two comparable
concurrency points, and download/rebuild after runner cleanup.

For CI acceptance, use an explicit ref containing the producer changes. Confirm
that summary/full artifacts and failure diagnostics upload before cleanup, then
download and verify/rebuild them on an independent CPU environment. A local ZIP
round-trip does not establish this GitHub upload/download path.
