---
name: atomrl-coding
description: >-
  Coding standards and architecture rules for the AtomRL RL framework.
  Use when writing, reviewing, or modifying code in the AtomRL repo,
  or when adding new workers, algorithms, quantization features, or
  MoE support to AtomRL.
---

# AtomRL Coding Guide

## Core Principle

AtomRL is a standalone RL framework. It does NOT use VERL, NeMo-RL, or any
other third-party RL framework. All controller, worker, algorithm, and trainer
code is written from scratch in `atomrl/`. The only external dependencies are:

- **Lumen** (`Lumen`) — FP8 quantized training kernels and ops
- **ATOM** (repository root/current checkout) — optimized inference engine
- **AITER** (official ROCm package/repository) — low-level GPU kernels (ASM/CK/Triton)
- **MORI** (official ROCm package/repository) — RDMA + GPU communication
- **Ray** — distributed orchestration (used as a library, not a framework)

Never copy patterns, classes, or utilities from VERL or NeMo-RL into AtomRL.

## Architecture Rules

- Controller is single-process; all GPU work happens in Ray workers
- Workers communicate only via `DataProto`; no shared global state across processes
- Quantization goes through `atomrl/quantization/`; never call `atomrl.lumen.quantize` directly from worker code
- MoE R3 hooks are installed/removed by `R3Manager`; never manually register forward hooks on MoE layers
- Weight sync uses the trainer-owned production paths: Ray rollout goes through
  `RLTrainer._sync_weights_ipc()` (IPC, RDMA, or safetensors fallback), while
  `AsyncRLTrainer` uses `FilesystemWeightSync`; never carry model weights in `DataProto`
- ATOM engine lifecycle (init, sleep, wake, shutdown) is managed by `AtomRolloutWorker`; no raw `LLMEngine` usage elsewhere

## Import Boundaries

Workers are process-isolated. Enforce strict import separation:

| Module | Allowed imports from Lumen | Allowed imports from ATOM |
|--------|---------------------------|--------------------------|
| `atomrl/workers/actor_worker.py` | `atomrl.lumen.quantize`, `atomrl.lumen.models`, `atomrl.lumen.ops` | None |
| `atomrl/workers/rollout_worker.py` | None | `atom.model_engine`, `atom.model_ops`, `atom.config` |
| `atomrl/engine/training/` | `atomrl.lumen.quantize`, `atomrl.lumen.models` | None |
| `atomrl/engine/inference/` | None | `atom.model_engine`, `atom.model_ops` |
| `atomrl/quantization/` | `atomrl.lumen.quantize` (training path only) | `atom.quant_spec` (rollout path only) |
| `atomrl/moe/` | `atomrl.lumen.ops.moe` (replayer) | `atom.model_ops.moe` (recorder) |
| `atomrl/core/`, `atomrl/algorithms/`, `atomrl/trainer/` | None | None |

Never import training-side modules in rollout workers or vice versa.

## Config Rules

- All defaults live in YAML files under `configs/`, not in Python `config.get(..., default)` calls
- Use `TypedDict` for config section validation; `@dataclass` for top-level `AtomRLConfig`
- CLI overrides via OmegaConf dot notation: `policy.model_name=Qwen/Qwen3-8B`
- Required keys use `Required[]`; optional keys use `NotRequired[]` with explicit presence checks

## DataProto Contract

`DataProto` is the only data object that crosses worker boundaries.

Required fields for RL:
- `input_ids`, `attention_mask`, `position_ids`
- `old_log_probs`, `ref_log_probs` (after log-prob computation)
- `advantages`, `rewards` (after advantage computation)

Optional fields for MoE R3:
- `router_distributions`: `Dict[int, Tensor]` keyed by layer index

Optional fields for FP8 correction:
- `fp8_log_probs`: log-probs from FP8 rollout (for TIS/MIS)

Never add GPU tensors to DataProto for cross-node transfer; move to CPU first.

## FP8 Coding Patterns

When adding FP8 support to a new path:

1. Gate behind `config.quantization.rollout.precision == "fp8"` or `config.quantization.training.fp8`
2. Use `atomrl/quantization/fp8_config.py` to build `FP8Config`, not raw strings
3. Numerical tests required: compare FP8 output vs BF16 reference within calibrated tolerance
4. Always recalibrate quantization scales when policy weights change (every RL step)

## R3 Coding Patterns

When modifying MoE router paths:

1. Recording hooks go on ATOM `FusedMoE` modules only (inference side)
2. Replay hooks go on Lumen MoE router modules only (training side)
3. R3Manager owns the hook lifecycle; hooks are `contextmanager`-style (install on enter, remove on exit)
4. Router distributions are stored as `float32` tensors regardless of training precision
5. Verify bit-identical replay in unit tests (no tolerance; recorded logits must equal replayed logits)

## Testing Rules

- Every new module needs unit tests in `tests/unit/`
- FP8 numerical tests: `max_abs_error(fp8_out, bf16_out) < tolerance`
- R3 tests: `recorded_logits == replayed_logits` (exact match)
- GPU tests use `@pytest.mark.gpu`; multi-GPU tests use `@pytest.mark.multigpu`
- Integration tests go in `tests/integration/`; E2E convergence tests in `tests/e2e/`
- Use `conftest.py` fixtures for model builders; prefer small models (0.6B) for speed

## Code Style

- Type hints on all public functions
- Docstrings on all public classes and functions (Google style)
- No inline imports; all imports at module top
- Use `logging.getLogger(__name__)` for logging; never `print()`
- Constants in UPPER_SNAKE_CASE at module level

## Pairing

Pair with `atomrl-training` when debugging RL training runs, reward collapse, or FP8/R3 stability issues.
Pair with `atomrl-debug` when fixing bugs that originate in Lumen, ATOM, or AITER.
