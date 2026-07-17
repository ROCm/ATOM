# ATOM MiniMax-M3 Draft Tracking

- Scope: ATOM + MiniMax-M3
- Entry: `.github/benchmark/models_atomesh.yaml`
- Existing model: `MiniMax-M3-MXFP8`
- Existing backend: `atom`
- Existing cases:
  - `minimax-m3-fp8-1p1d-tp4`
  - `minimax-m3-fp8-2p1d-dpa-tp4`
  - `minimax-m3-fp8-1p1d-tp4-eagle3`
  - `minimax-m3-fp8-2p1d-dpa-tp4-eagle3`

## Triggering ATOMesh Benchmark

Use the `Atomesh Benchmark` workflow with `workflow_dispatch`.

To run all MiniMax-M3-MXFP8 ATOM cases:

- `suite`: `nightly`
- `run_all_models`: `false`
- `model_names`: `MiniMax-M3-MXFP8`
- `case_names`: empty

To run specific MiniMax-M3-MXFP8 ATOM cases, set `case_names` to a comma-separated list. `case_names` overrides `model_names`.

Example:

```text
minimax-m3-fp8-1p1d-tp4,minimax-m3-fp8-2p1d-dpa-tp4,minimax-m3-fp8-1p1d-tp4-eagle3,minimax-m3-fp8-2p1d-dpa-tp4-eagle3
```

Note: draft tracking PR; no runtime behavior change unless config changes are needed.
