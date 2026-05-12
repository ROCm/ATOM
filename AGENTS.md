# AGENTS.md

## Cursor Cloud specific instructions

ATOM is an AMD ROCm GPU inference engine. The Cloud Agent VM has **no GPU**, so only CPU-mockable development workflows are available.

### Quick reference

| Task | Command | Notes |
|------|---------|-------|
| Install | `pip install -e .` | Editable install; see `pyproject.toml` for deps |
| Lint (format) | `black --check .` | CI-enforced |
| Lint (rules) | `ruff check .` | CI-enforced |
| Test | `python3 -m pytest tests/` | See GPU caveat below |
| Dev tools | `pip install pytest black ruff` | Not in project deps |

### GPU-dependent test failures are expected

The test suite stubs GPU deps via `tests/conftest.py`, but some tests directly import `torch` or `aiter` (AMD GPU kernels). Expect ~60-70 test failures from collection errors on `aiter`-dependent files. These are **not** bugs — they require AMD ROCm hardware. The ~190+ tests that pass cover the scheduler, block manager, prefix caching, sampling params, and other CPU-testable logic.

Test files that require `aiter` and will fail without it:
- `tests/test_envs.py`, `tests/test_mxfp4_moe_has_bias.py`, `tests/test_arg_utils_spec.py`
- `tests/test_kimi_k25.py`, `tests/test_per_req_cache_decoupling.py`, `tests/test_profiler_regression.py`
- `tests/test_request.py`, `tests/test_sampling_params.py`, `tests/test_scheduler.py`
- `tests/test_sequence.py`, `tests/test_utils.py`
- `tests/plugin/` (all files)

### CPU-only PyTorch

The update script installs CPU-only PyTorch (`torch` from `https://download.pytorch.org/whl/cpu`). This enables more tests to pass (those that import torch but not aiter). Do **not** install the full CUDA/ROCm PyTorch — it's ~2 GB larger and provides no benefit without GPU hardware.

### PATH note

Scripts installed by pip (`pytest`, `black`, `ruff`, etc.) go to `~/.local/bin`. The update script adds this to `PATH`. If running commands manually, ensure `export PATH="$HOME/.local/bin:$PATH"`.

### Running the server or inference (requires GPU)

The OpenAI-compatible server and offline inference entry points require AMD ROCm GPU + AITER:
```bash
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8
python -m atom.examples.simple_inference --model <model> --kv_cache_dtype fp8
```
These cannot run in the Cloud Agent VM.

### Pre-existing lint issues

`ruff check .` reports 3 pre-existing unused-import warnings (F401) in `atom/plugin/` files. These are in the repo, not introduced by setup.
