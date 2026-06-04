# CLAUDE.md — ATOM

ATOM is a lightweight LLM inference engine built on AITER GPU kernels for the AMD ROCm platform.

## Build & Test & Lint

```bash
pip install -e .                          # editable install
python -m pytest tests/                   # all tests (no GPU needed — mocks AITER and torch.cuda)
black . && ruff check .                   # format + lint (CI enforced)
```

## Running

```bash
# OpenAI-compatible serving
python -m atom.entrypoints.openai_server --model <model> --kv_cache_dtype fp8 -tp 8

# Offline inference
python -m atom.examples.simple_inference --model <model> --kv_cache_dtype fp8
```

- Accuracy validation: see `/ci-pr-guide` for `lm_eval` setup and CI thresholds
- Performance benchmark: see `/benchmark-guide` for full parameters

## Architecture (Quick Index)

```
openai_server.py → LLMEngine → CoreManager → EngineCore → Scheduler → ModelRunner
                   (tokenizer)   (ZMQ IPC)    (inference    (batching)  (forward pass,
                                               loop)                    KV cache, CUDAGraphs)
```

Key entry points:
- Server: `atom/entrypoints/openai_server.py`
- Engine: `atom/model_engine/llm_engine.py` → `engine_core.py` → `scheduler.py` → `model_runner.py`
- Models: `atom/models/` — registered in `model_runner.py:support_model_arch_dict`
- Ops: `atom/model_ops/` — AITER kernel wrappers (linear, attention, fused_moe)
- Config: `atom/config.py` (Config, KVCacheConfig, CompilationConfig)
- Env vars: `atom/utils/envs.py` (all `ATOM_*` variable definitions)

## Critical Rules

- **NEVER modify `@support_torch_compile` decorated model files** — breaks Dynamo tracing even with `--enforce-eager`. Instrument at call sites instead (e.g., `ModelRunner.run_model()`, `EagleProposer.propose()`)
- **Multiprocessing must use `spawn`** — `fork` causes CUDA re-initialization crashes
- **Set `AITER_LOG_LEVEL=WARNING` before starting server** — suppresses aiter kernel log flooding
- **Clear compile cache before server restart:** `rm -rf /root/.cache/atom/*` — stale cache causes silent failures after code changes
- **Verify server with GPU, not just HTTP:** `curl /health` can return OK even when model is not loaded. Always check `rocm-smi --showmemuse` (VRAM% > 0) to confirm
- **On any server/GPU error, run `/debug-guide` first** — do not blindly retry
- **Fix-then-sweep**: after fixing a bug, immediately grep for the same pattern across the codebase and fix all occurrences in one pass
- **Name-matches-function**: variable, function, and file names must accurately describe what they do. When behavior changes, rename immediately — stale names mislead future readers

## Key Development Patterns

- **Adding a model**: see `/add-model` for full guide
- **Model reuse**: DeepSeek V3/V3.2/GLM-5 share `deepseek_v2.py`; MTP models in `deepseek_mtp.py`, `qwen3_next_mtp.py`, and `qwen3_5_mtp.py`
- **Compilation levels**: `--level` 0=eager, 1=torch.compile, 2=dynamo once, 3=piecewise+CUDAGraph (default)

## Dependencies

- **AITER** (`from aiter import ...`) — GPU compute kernels
- **MORI** — MoE expert-parallel all-to-all communication
- **RCCL** — collective communication primitives

## Detailed Guides

| Topic | Source |
|-------|--------|
| Architecture | `docs/architecture_guide.md` |
| Environment variables | `docs/environment_variables.md` |
| Compilation & CUDAGraph | `docs/compilation_cudagraph_guide.md` |
| Model support | `docs/model_support_guide.md` |
| Model ops (AITER kernels) | `docs/model_ops_guide.md` |
| Scheduling & KV cache | `docs/scheduling_kv_cache_guide.md` |
| Serving & benchmarking | `docs/serving_benchmarking_guide.md` |
| Configuration | `docs/configuration_guide.md` |
| Distributed | `docs/distributed_guide.md` |
| CI/PR workflow | `/ci-pr-guide` |
| Performance benchmark | `/benchmark-guide` |
| Debugging | `/debug-guide` |
| Adding a model | `/add-model` |
