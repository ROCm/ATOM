# ATOM Benchmark CI

Nightly + on-demand performance benchmarking for the models in
[`models.json`](./models.json), driven by
[`.github/workflows/atom-benchmark.yaml`](../workflows/atom-benchmark.yaml).

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
| `dashboard_models_map.py` | prefix→display map JS for the dashboard |
| `regression_rerun.py` | regression report → rerun matrix |
| `atom_test.sh` | in-container driver: `launch` / `benchmark` / `accuracy` / `stop` |
| `summarize.py`, `plugin_benchmark_to_dashboard.py` | post-processing / dashboard input |

The GPU container lifecycle (start container + download model) is the composite
action [`.github/actions/atom-bench-container`](../actions/atom-bench-container/action.yml),
shared by the `benchmark-tmpl.yml` reusable workflow and the `regression-rerun` job.

## Data contracts (keep stable)

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

**Validate locally**
```bash
python -m pytest tests/test_benchmark_catalog.py
python .github/scripts/catalog.py --cells .github/benchmark/models.json   # preview cells
```
