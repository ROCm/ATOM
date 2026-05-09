# ATOM vLLM Benchmark Trigger

Local browser-based trigger tool for `atom-vllm-benchmark.yaml`.

## What it does

- Reads model family and TP metadata from `.github/benchmark/oot_benchmark_models.json`
- Provides richer local interaction than GitHub's native `workflow_dispatch` UI
- Dispatches the existing benchmark workflow through local `gh` CLI auth

## Start

From the repo root:

```bash
python .github/scripts/atom_vllm_benchmark_trigger_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

## Requirements

- `gh` must be installed
- `gh auth login` must already be completed on the local machine
- The current git remote must point at the GitHub repository you want to trigger

## Notes

- The tool is additive: it does not replace the existing GitHub Actions manual form
- The backend still dispatches `.github/workflows/atom-vllm-benchmark.yaml`
- The workflow's current slot limit still applies: at most 8 model selections per run
