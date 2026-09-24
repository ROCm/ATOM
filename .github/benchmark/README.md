# ATOM Benchmark CI

Nightly + on-demand performance benchmarking for the models in
[`models.json`](./models.json), driven by
[`.github/workflows/atom-benchmark.yaml`](../workflows/atom-benchmark.yaml).

Agentic trace replay uses the separate
[`ATOM Agentic Benchmark`](../workflows/atom-agentic-benchmark.yaml) workflow and
two catalogs: [`models_agentic.json`](./models_agentic.json) for the default
manual test, and [`models_agentic_nightly.json`](./models_agentic_nightly.json)
for the daily 08:17 UTC run. Both use `rocm/atom-dev:latest` and checked-out branch
code. The manual default remains DeepSeek V4.1 Flash + DSpark5, TP4 with FULL
graphs, concurrency 2/4 and 900 seconds per point. Nightly is based on InferenceX
PR #3387 with a custom grid: TP2 c=1/2/4/8/16/32/64/128 and TP4 c=1/4/8,
3600 seconds per point, 5 warmup requests per lane and fixed AL 3.51.
Both profiles set GPU memory utilization to 0.95, omit the max-num-seqs override,
and use capture sizes 1–32 plus 48/64/96/128/160/192/224/256 for every point.
Agentic variants declare `concurrency` directly; random `scenarios`, ISL/OSL,
length ratios and concurrency bands are not part of these catalogs.
Select **ATOM Agentic Benchmark**
for these runs; **ATOM Benchmark** keeps the random-workload model checkboxes
and dashboard settings in its own form. The temporary mixed entry has been
removed. The independent workflow must reach the default branch before GitHub
exposes its manual/scheduled entry. For a manual test, select the branch, leave
the preset at `test` and keep optional overrides empty. Advanced execution and
profiling settings are labeled separately. Manual `dry_run` previews
the matrix without GPU jobs; run summaries include configuration, replay commands
and per-point artifact links. Titles use `manual (<actor>)` / `nightly` with
GitHub's native run number. Both entries group jobs as model configuration →
concurrency points and directly reuse
`benchmark-tmpl.yml`; the agentic workflow publishes data artifacts for external
consumers, including AgenticViewer. See
[`benchmark-artifacts.md`](../../docs/benchmark-artifacts.md#ci-and-configuration)
for capture and verification details.

## Flow

```
build-matrix            (ubuntu) validate catalog ⟷ dispatch inputs;
  │                              expand catalog → configs_json (one config =
  │                              variant × scenario, carrying a concurrency list)
  ▼
benchmark               (caller, matrix: config) one entry per variant×scenario;
  │                              each calls benchmark-tmpl.yml (secrets: inherit)
  ▼
  └ benchmark-tmpl.yml   (GPU, matrix: conc) composite container setup →
  │                              atom_test.sh launch + benchmark → benchmark-<rf>.json
  │                              Two-level fan-out (config × conc) keeps each
  │                              matrix < GitHub's 256-jobs-per-matrix limit while
  │                              every cell still runs as its own parallel job.
  ▼
summarize-benchmark-result (ubuntu) gather results + previous-nightly baseline →
  │                              summarize.py → regression_report.json;
  │                              push data + dashboard to gh-pages
  ▼ (only if regressions)
generate-regression-matrix (ubuntu) regression_rerun.py → rerun cells
  ▼
regression-rerun        (GPU, matrix: cell) same composite setup → profiled reruns
  ▼
collect-regression-traces (ubuntu) merge trace artifacts
```

## Single source of truth: `models.json`

Structured catalog. One object per **base model**; each serving **variant**
(base / MTP / DP-attention / …) is a dimension of that model, not a duplicated
entry.

```jsonc
{
  "default_scenarios": [                 // workload grid applied to every variant
    {"isl": 1024, "osl": 1024,
     "concurrency": [4, 8, 16, 32, 64, 128, 256, 512, 1024],
     "random_range_ratio": 0.8},
    {"isl": 8192, "osl": 1024, "concurrency": [...], "random_range_ratio": 0.8}
  ],
  "models": [
    {
      "display": "DeepSeek-V4-Pro",      // dashboard / log name (base)
      "path": "deepseek-ai/DeepSeek-V4-Pro",
      "prefix": "deepseek-v4-pro",        // workflow_dispatch checkbox + result file prefix
      "runner": "atom-mi355-8gpu.predownload",
      "env_vars": "AITER_BF16_FP8_MOE_BOUND=0\nATOM_MOE_GU_ITLV=1",  // container env
      "config": {"tp": 8, "kv_cache_dtype": "fp8",
                 "extra_args": "--hf-overrides '...'"},  // shared across ALL variants
      "variants": [
        {"label": "", "suffix": "", "conc_max": 256},
        {"label": "MTP3", "suffix": "-mtp3",
         "extra_args": "--method mtp --num-speculative-tokens 3",
         "bench_args": "--use-chat-template", "conc_min": 4, "conc_max": 256},
        {"label": "DPA", "suffix": "-dpa",
         "extra_args": "--enable-dp-attention",
         "conc_min": 64, "conc_max": 1024},
        {"label": "DPA TBO", "suffix": "-dpa-tbo",
         "extra_args": "--enable-dp-attention --enable-tbo",
         "env_vars": "GPU_MAX_HW_QUEUES=5",
         "conc_min": 256, "conc_max": 1024},
        {"label": "DPA MTP3", "suffix": "-dpa-mtp3",
         "extra_args": "--method mtp --num-speculative-tokens 3 --enable-dp-attention",
         "bench_args": "--use-chat-template", "conc_min": 64, "conc_max": 1024}
      ]
    }
  ]
}
```

### Config / variant fields

`config` (shared) and per-`variant` fields are composed into the server CLI by
`catalog.build_args` in a fixed order:

Only the common basics are structured fields; anything model- or
variant-specific (MTP, DP-attention, sparse-attention overrides, memory
utilization, …) is passed verbatim through `extra_args`:

| field | where | emits |
|-------|-------|-------|
| `kv_cache_dtype` | config | `--kv_cache_dtype <v>` (default `fp8`) |
| `tp` | config | `-tp <n>` (omitted if absent, e.g. gpt-oss) |
| `trust_remote_code` | config | `--trust-remote-code` |
| `extra_args` | config and/or variant | appended verbatim (server flags) |
| `env_vars` | model and/or variant | newline-joined container env vars |
| `bench_args` | variant | passed to the benchmark client (not the server) |
| `conc_min` / `conc_max` | variant | concurrency band (filters scenarios) |
| `scenarios` | variant or model | overrides `default_scenarios` |

Examples of `extra_args` content: `--method mtp --num-speculative-tokens 3`
(MTP), `--enable-dp-attention` (DP-attention),
`--hf-overrides '{...}'` (V4 sparse-attention index cache, set at `config`
level so all variants share it).

Concurrency bands replace the old hard-coded matrix `exclude` block: out-of-band
`(variant, concurrency)` combos are never emitted, so **no GPU runner is
allocated for them**.

## Scripts

| script | role |
|--------|------|
| `catalog.py` | catalog loader: `load_variants`, `build_cells`, `build_cell_configs`, `scenario_tag`, `validate_dispatch_inputs`, `build_args` |
| `build_benchmark_matrix.py` | turns the GitHub event + dispatch inputs into the `configs_json` matrix output (variant×scenario configs, each with a concurrency list) |
| `build_agentic_benchmark_matrix.py` | expands the agentic catalog for scheduled runs or validated manual overrides; emits the reusable-template matrix and saves run configuration, replay inputs and the Actions summary |
| `dashboard_models_map.py` | prefix→display map JS for the dashboard |
| `regression_rerun.py` | regression report → rerun matrix |
| `atom_test.sh` | in-container driver: `launch` / `benchmark` / `accuracy` / `stop` |
| `summarize.py`, `plugin_benchmark_to_dashboard.py` | post-processing / dashboard input |
| `validate_catalog.py` | schema + semantic gate for the accuracy catalogs (see below) |

The GPU container lifecycle (start container + download model) is the composite
action [`.github/actions/atom-bench-container`](../actions/atom-bench-container/action.yml),
shared by the `benchmark-tmpl.yml` reusable workflow and the `regression-rerun` job.

## Accuracy catalog schema

The flat accuracy catalogs — `models_accuracy.json`, `oot_models_accuracy.json`,
`sglang_models_accuracy.json` — are validated against
[`schema/accuracy_catalog.schema.json`](schema/accuracy_catalog.schema.json) by
[`../scripts/validate_catalog.py`](../scripts/validate_catalog.py). The
`validate-catalog` job in `pre-checks.yaml` runs it on every PR (no GPU).

- **Required fields**: `model_name`, `model_path`, `env_vars`, `runner`,
  `test_level` (`pr` | `nightly` | `main`).
- **`additionalProperties: false`** — an unknown/misspelled key fails CI. Add the
  field to the schema first if it is intentional.
- **Pass bar (semantic rule)**: each entry must have exactly one of
  `accuracy_threshold` / `accuracy_test_threshold`.
- **Accuracy timeout**: set optional `accuracy_timeout_minutes` on a
  `models_accuracy.json` entry to override the native ATOM/atomesh accuracy
  step timeout. If omitted, the timeout is 30 minutes.
- **Known drift (tolerated for now)**: `extraArgs` vs `extra_args` and
  `accuracy_threshold` vs `accuracy_test_threshold` are both accepted; the schema
  documents the current reality. Normalizing these (and their consumers) is a
  separate change.

Run locally before pushing a catalog edit:

```bash
pip install jsonschema
python .github/scripts/validate_catalog.py
```

## Data contracts

The following random-workload contracts keep the existing dashboard compatible:

- **Result file**: `benchmark_serving` writes `<result_filename>.json` where
  `result_filename = "{prefix}{suffix}-{isl}-{osl}-{conc}-{ratio}"`; uploaded as
  artifact `benchmark-<result_filename>`. The dashboard + baseline diff key off
  this — do not change the format without updating the dashboard.
- **Cell**: `build_cells` emits
  `{display, prefix, suffix, model_path, server_args, bench_args, env_vars,
  runner, isl, osl, conc, ratio, result_filename}` — one fully-resolved run.
- **Config** (matrix entry): `build_cell_configs` regroups cells by
  (variant × scenario) into `{display, prefix, suffix, model_path, server_args,
  bench_args, env_vars, runner, isl, osl, ratio, ratio_str, scenario,
  concurrency}` where `concurrency` is a JSON list. The `benchmark` caller
  matrixes over configs; `benchmark-tmpl.yml` matrixes over each config's
  `concurrency`. Both stay < GitHub's 256-jobs-per-matrix limit. Adding a model
  or scenario needs no workflow edit — the caller matrix is fully dynamic.

Agentic uses `build_agentic_benchmark_matrix.py` to build one config per variant
with a direct `concurrency` list. It shares the server-argument/environment
composition helpers and execution template, but does not pass random dimensions
or unused `bench_args`. Agentic result names are `<prefix><suffix>-c<concurrency>`;
full and summary bundles continue to be discovered through their manifests.

## How to …

**Add a model** — add one object to `models.json#/models` and one boolean to
the workflow's `workflow_dispatch.inputs` whose key == the model `prefix`. The
`test_workflow_dispatch_inputs_match_catalog` test fails if they drift, and
`build-matrix` fails the run on dispatch drift.

**Add a variant** (e.g. a new MTP setting) — append to that model's `variants`
with a unique `suffix` and the structured fields above.

**Change the default workload grid** — edit `default_scenarios`. Give a single
variant a different grid via its own `scenarios`, or just tighten its
`conc_min`/`conc_max`.

**Benchmark an AITER change** — manually dispatch `ATOM Benchmark` with
`aiter_commit` set to a ROCm/aiter commit SHA, tag, or branch. The benchmark
container reinstalls `amd-aiter` from that ref before launching ATOM. Leave it
empty to keep the version already baked into the selected Docker image.

**Validate locally**
```bash
python -m pytest tests/test_benchmark_catalog.py
python .github/scripts/catalog.py --cells .github/benchmark/models.json   # preview cells
```

## V4 AgentX P/D without offload

The `DeepSeek-V4-Pro-0813` **weekly** suite runs on the existing TW schedule:
Friday 16:00 UTC (Saturday 00:00 Beijing time). It covers seven independent
concurrencies, **1, 2, 16, 32, 128, 192, and 256**, each with fresh services on
two eight-GPU nodes (1P1D, TP8 with DPA).

To dispatch manually, select **Atomesh Benchmark** (`atomesh-benchmark.yaml`),
`suite=weekly`, `model_names=DeepSeek-V4-Pro-0813`, and
`run_model_benchmark=true`. Leave `benchmark_concurrency` empty to preserve the
per-case service sizing and routing settings. To run one concurrency, set
`case_names=ds-v4-0813-1p1d-dpa-tp8-dspark3-agentic-no-offload-c256`, replacing
`256` with the desired concurrency. Choose the corresponding Slurm submit
runner for the target cluster. The existing MI350X daily schedule is unchanged.

The checkpoint is expected at
`${ATOMESH_MODEL_ROOT}/deepseek-ai/DeepSeek-V4-Pro-0813/`. The Crusoe runner
`atomesh-cicd-mi355-crusoe` uses `model_path_by_runner` to select
`/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro-0813`.
The workflow's image selection and `atomesh_image` override apply.

Serving follows the no-offload DPA settings in
[`DeepSeek-V4-Agentic-PD-Max.md`](../../recipes/DeepSeek-V4-Agentic-PD-Max.md):
FP8 KV, FP4 index cache, DSpark K3, and TBO on prefill only. The weekly suite
reuses the GLM AgentX AIPerf workload with **synthetic acceptance length 3.01**
and evals disabled; its results measure performance, not accuracy.

| Concurrency | P/D max-num-seqs | Decode graph sizes per DP rank | Cache-aware absolute / relative thresholds |
| --- | --- | --- | --- |
| 1, 2, 16 | 32 | 1..4 | 20 / 2.0 |
| 32 | 64 | 1..8 | 20 / 2.0 |
| 128 | 256 | 1..32 | 20 / 2.0 |
| 192 | 384 | 1..48 | 20 / 2.0 |
| 256 | 512 | 1..64 | 40 / 2.0 |

All seven cases use DP-aware `cache_aware` on both P and D, cache threshold 0.8,
and eviction interval 300 seconds. TW uses P/D rank mapping `none`.
Both KV connectors are plain Mooncake, without LMCache CPU/NVMe offload.
The global slot floor of 32 retains four decode slots per DP rank for the
lowest concurrency cases. GPU runs are needed to establish performance for
new concurrency settings.

For a separate accuracy run, select `suite=accuracy`, the same model, and
`run_model_benchmark=true`, or use
`case_names=ds-v4-0813-1p1d-tp8-dspark3-gsm8k-only-c16`. This manual case uses
pure TP8 (DPA disabled), DSpark K3, GSM8K 5-shot with the chat template,
concurrency 16, and an 8K model context. `eval_only=true` starts fresh services
and runs evaluation without the performance workload or synthetic acceptance
length. Per-sample evaluation logs are retained in the result artifacts.


### Optional AITER wheel for manual agentic benchmarks

`ATOM Agentic Benchmark` accepts an optional **aiter_wheel** input:

| Value | Behavior |
| --- | --- |
| Empty (default) | Keep the image's installed AITER; no wheel download or install. Scheduled runs also keep the image version. |
| `latest` | Resolve the latest main Python 3.12 wheel with the existing S3-manifest/GitHub-artifact fallback. |
| HTTPS URL ending in `amd_aiter-…whl` | Download that specific wheel version. Use a wheel compatible with the selected image's Python/PyTorch/ROCm. |
| `artifact:<ID>` | Download that immutable wheel artifact from ROCm/aiter (Python 3.12). |

Do not combine `aiter_wheel` with `aiter_commit`: matrix construction rejects
the conflict before allocating GPU jobs. A dry run validates and records the
requested selector but does not download a wheel or allocate GPUs.

For actual runs, the CPU matrix job downloads once and reads the package
metadata without importing it. It shares one artifact, including the wheel,
`selection.json` and `SHA256SUMS`, with every concurrency job. Each job verifies
the checksum, installs without replacing image dependencies, and verifies that
`aiter` imports from the installed distribution instead of an old source checkout.
Download, checksum, installation or import errors stop the job before benchmarking.

The Actions summary and `atom-agentic-run-config-<attempt>/run-config.json`
record the requested selector, pinned URL/artifact ID, package version, filename
and SHA-256. `dispatch-inputs.json` pins the resolved wheel for replay; a fresh
manual dispatch with `aiter_wheel=latest` resolves a new wheel. These links/artifacts
can expire according to upstream retention. The runtime AITER identity remains
recorded in each benchmark bundle.

CPU regression checks: `python -m pytest tests/test_benchmark_catalog.py
tests/test_benchmark_aiter_wheel.py`.
