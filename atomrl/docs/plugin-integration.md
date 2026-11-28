# AtomRL Plugin Integration Guide

This document explains the registration principles and usage of AtomRL's
plugin system for injecting ATOM rollout and Lumen FP8 training into external
RL frameworks.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────┐
│                    Host RL Framework                      │
│    (verl / vime / miles / slime)                          │
│                                                           │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  Training    │    │  Rollout     │    │  Weight     │  │
│  │  Engine      │    │  Engine      │    │  Sync       │  │
│  └──────┬──────┘    └──────┬───────┘    └─────────────┘  │
│         │                  │            (host-native,    │
│         │                  │             NOT injected)    │
└─────────┼──────────────────┼─────────────────────────────┘
          │                  │
          ▼                  ▼
┌─────────────────┐  ┌──────────────────┐
│ Lumen FP8       │  │ ATOM Rollout     │
│ (FP8Training-   │  │ (ATOMReplica-    │
│  Manager)       │  │  Manager)        │
│                 │  │                  │
│ atomrl/        │  │ atomrl/engine/  │
│ quantization/   │  │ inference/       │
└─────────────────┘  └──────────────────┘
```

The plugin layer (`atomrl/plugin/`) sits between the host framework and
AtomRL's core modules. Framework files preserve host-defined signatures while
private helpers centralize lifecycle, checkpoint reload, and output conversion.

---

## Registration principles

### 1. Thin wiring only

Plugin files must not implement RL algorithms or duplicate core AtomRL logic.
Framework-neutral adapter behavior belongs in private shared plugin helpers.

### 2. Weight sync uses public ATOM reload APIs

Host frameworks still decide when and how to export weights. Once an
HF-format checkpoint is available, plugin hooks call
`ATOMReplicaManager.reload_weights_from_path()` on rank zero. verl's tensor
stream is converted in a scoped temporary safetensors directory.

### 3. Function signatures are frozen by the host framework

The hook signatures are dictated by each framework's loader and must match
exactly. Extending them requires a change in the host framework.

### 4. Entry-points vs CLI paths

| Framework | Mechanism | When loaded |
|---|---|---|
| verl | Python packaging entry-points (`importlib.metadata`) | On `import verl` |
| vime | `importlib.import_module` triggered by CLI args | At training startup |
| miles | `importlib.import_module` triggered by CLI args | At training startup |
| slime | `importlib.import_module` triggered by CLI args | At training startup |

---

## verl integration

### Registration mechanism

verl discovers plugins via `importlib.metadata.entry_points(group="verl.plugins")`
in `verl/__init__.py`. AtomRL declares:

```toml
# pyproject.toml
[project.entry-points."verl.plugins"]
atomrl = "atomrl.plugin.verl.register:register"
```

The `register()` function (in `atomrl/plugin/verl/register.py`) runs once on
`import verl` and mutates two registries:

**1. Rollout registry** — `_ROLLOUT_REGISTRY` is a plain `dict` in
`verl/workers/rollout/base.py`:

```python
_ROLLOUT_REGISTRY[("atom", "async")] = "atomrl.plugin.verl.register.ATOMRolloutAdapter"
```

verl resolves the FQDN string lazily via `importlib.import_module` when the
trainer requests the `"atom"` rollout backend.

**2. Engine registry** — uses `@EngineRegistry.register()` decorator in
`verl/workers/engine/base.py`:

```python
# FSDP2 engine (with Lumen FP8 support)
EngineRegistry.register(
    model_type="language_model",
    backend=["atomrl_fsdp", "atomrl_fsdp2"],
    device=["cuda"],
)(AtomRLFSDP2Engine)

# Megatron engine
EngineRegistry.register(
    model_type="language_model",
    backend="atomrl_megatron",
    device=["cuda"],
)(AtomRLMegatronEngine)
```

### Classes

| Class | Role |
|---|---|
| `ATOMRolloutAdapter` | Implements verl's `BaseRollout` ABC; forwards lifecycle calls and reloads streamed weights through a scoped checkpoint |
| `AtomRLFSDP2Engine` | Wraps `atomrl.engine.training.fsdp_engine.FSDP2EngineWithLMHead`; supports FP8 via `quant_config` |
| `AtomRLMegatronEngine` | Wraps `atomrl.engine.training.megatron_engine.MegatronEngineWithLMHead`; delegates via `__getattr__` |

### Usage

```yaml
# verl hydra config — FSDP2 training + ATOM rollout
actor_rollout_ref:
  rollout:
    name: atom
    mode: async
trainer:
  engine:
    backend: atomrl_fsdp2      # or atomrl_fsdp
```

```yaml
# verl hydra config — Megatron training + ATOM rollout
actor_rollout_ref:
  rollout:
    name: atom
    mode: async
trainer:
  engine:
    backend: atomrl_megatron
```

### Control

```bash
# Load only atomrl (ignore other plugins):
export VERL_USE_EXTERNAL_PLUGINS=atomrl

# Disable all external plugins:
export VERL_USE_EXTERNAL_PLUGINS=none
```

---

## vime integration

### Registration mechanism

vime resolves hook functions at runtime using
`vime.utils.misc.load_function(path)`, which calls
`importlib.import_module(module_path)` and `getattr(module, attr)`. Each hook
must be a top-level function in its module.

### Hook reference

| CLI argument | Dotted path | Signature |
|---|---|---|
| `--custom-megatron-init-path` | `atomrl.plugin.vime.hooks.custom_megatron_init` | `(args: Namespace) -> None` |
| `--custom-model-provider-path` | `atomrl.plugin.vime.hooks.custom_model_provider` | `(pre_process: bool, post_process: bool, vp_stage: int \| None = None) -> GPTModel` |
| `--rollout-function-path` | `atomrl.plugin.vime.hooks.generate_rollout` | `(args, rollout_id: int, data_source, evaluation: bool = False) -> RolloutFnTrainOutput \| RolloutFnEvalOutput` |
| `--custom-update-weight-post-write-path` | `atomrl.plugin.vime.hooks.custom_update_weight_post_write` | `(args, version_dir: str, rollout_engines: list) -> None` |

### Hook sequencing

```
vime startup
  │
  ├─ Megatron distributed init
  │    └─ custom_megatron_init(args)           ← creates FP8TrainingManager
  │         stores on args._atomrl_fp8_manager
  │
  ├─ Model construction
  │    └─ custom_model_provider(...)           ← builds GPTModel, applies FP8
  │         reads args._atomrl_fp8_manager
  │
  └─ Training loop
       └─ generate_rollout(args, ...)          ← drives ATOM rollout
            reads args._atomrl_atom_manager
```

### Prerequisites

Before training starts, use the lifecycle helper:

```python
from atomrl.plugin.vime.lifecycle import setup_atom_manager, teardown_atom_manager

setup_atom_manager(args, my_worker_group)
try:
    train()
finally:
    teardown_atom_manager(args)
```

For FP8 training, set `args.atomrl_fp8_config` before Megatron init:

```python
from atomrl.core.config import QuantizationConfig, TrainingQuantConfig

args.atomrl_fp8_config = QuantizationConfig(
    training=TrainingQuantConfig(fp8_recipe="blockwise2d"),
)
```

---

## miles and slime integration

miles and slime expose the same Megatron, FP8 reset, lifecycle, and checkpoint
reload hooks as vime. Their rollout wrappers import host-native `Sample` and
`RolloutFn*Output` classes only when called.

The miles smoke assets live in `examples/plugin/miles/`. No slime example was
present in the synced Lumen-RL revision, so only the adapter and unit coverage
are provided.

---

## Injection scope summary

| Component | verl | vime | miles | slime |
|---|---|---|---|---|
| **Lumen FSDP2 training** | `AtomRLFSDP2Engine` | — | — | — |
| **Lumen Megatron training** | `AtomRLMegatronEngine` | hooks | hooks | hooks |
| **ATOM rollout** | adapter | native output | native output | native output |
| **Weight reload** | tensor stream → scoped checkpoint | rank-zero checkpoint hook | rank-zero checkpoint hook | rank-zero checkpoint hook |
| **Lifecycle** | adapter helper | args helper | args helper | args helper |

---

## Adding a new framework

To support an additional framework:

1. Create `atomrl/plugin/<framework>/__init__.py` and a hooks module.

2. Identify the framework's hook discovery mechanism:
   - Entry-points? → Add to `pyproject.toml` and write a `register()` function.
   - CLI path? → Write functions matching the exact expected signatures.
   - Hardcoded import? → Create the expected package/module name.

3. Write top-level host-signature wrappers and delegate shared behavior to the
   private helpers in `atomrl.plugin`.

4. Add usage instructions to this document and `atomrl/plugin/README.md`.

---

## Source code reference

| File | What |
|---|---|
| `atomrl/plugin/verl/register.py` | verl entry-point, `ATOMRolloutAdapter`, `AtomRLMegatronEngine` |
| `atomrl/plugin/{vime,miles,slime}/hooks.py` | host-signature wrappers and native rollout types |
| `atomrl/plugin/_lifecycle.py` | public `ATOMReplicaManager` setup/teardown |
| `atomrl/plugin/_megatron_hooks.py` | shared Megatron, FP8 reset, and checkpoint reload logic |
| `atomrl/quantization/fp8_training.py` | `FP8TrainingManager` (core FP8 logic) |
| `atomrl/engine/inference/atom_ray_server.py` | `ATOMReplicaManager`, `ATOMRayServer` (core ATOM logic) |
| `atomrl/engine/training/megatron_engine.py` | `MegatronEngine`, `MegatronEngineWithLMHead` |

## Isolated compatibility validation

Plugin compatibility is not part of the public AtomRL unit suite and does not
install host frameworks into public images. `examples/plugin/Dockerfile.plugin`
creates four isolated targets on top of the public base and pins:

- verl `0.9.1`
- vime `8707a5a7156223ce1d92c65279e2eed748addb82`
- miles `fe7037f561910f26ac4eccf9082bc5f4410488f8`
- slime `79aa41a884ab0bf65222c94418f6357d3c31192b`

Run all targets with `bash atomrl/examples/plugin/test.sh`.
