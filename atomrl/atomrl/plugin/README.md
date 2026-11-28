# AtomRL Plugin Adapters

Adapters that inject AtomRL capabilities (ATOM rollout, Lumen FP8 training)
into external RL frameworks. Host-signature wrappers stay framework-specific;
shared lifecycle, checkpoint reload, and rollout conversion live in private
helpers under this package.

## Directory layout

```
atomrl/plugin/
├── _lifecycle.py     # shared public ATOM manager lifecycle
├── _megatron_hooks.py
├── _sglang_host_rollout.py
├── verl/
│   ├── register.py   # entry-point auto-loaded by verl on import
│   ├── atom_replica.py
│   └── lifecycle.py
├── vime/
│   ├── hooks.py
│   └── lifecycle.py
├── miles/
│   ├── hooks.py
│   └── lifecycle.py
└── slime/
    ├── hooks.py
    └── lifecycle.py
```

---

## verl

### How it works

verl auto-discovers packages that declare an entry-point in the `verl.plugins`
group. Installing ATOM from the repository root registers the optional AtomRL
plugin automatically. The entry point imports AtomRL lazily, so installing ATOM
does not eagerly import optional training stacks:

```toml
# pyproject.toml (already configured)
[project.entry-points."verl.plugins"]
atomrl = "atomrl.plugin.verl.register:register"
```

On `import verl`, verl calls `register()` which injects:

| Target | verl key | What |
|---|---|---|
| `_ROLLOUT_REGISTRY` | `("atom", "async")` | ATOM rollout adapter |
| `EngineRegistry` | `backend="atomrl_fsdp2"` | AtomRL FSDP2 engine (+ FP8) |
| `EngineRegistry` | `backend="atomrl_megatron"` | AtomRL Megatron engine |

### Configuration

```yaml
# verl trainer config — FSDP2 variant
actor_rollout_ref:
  rollout:
    name: atom
trainer:
  engine:
    backend: atomrl_fsdp2    # or atomrl_fsdp (alias)
```

```yaml
# verl trainer config — Megatron variant
actor_rollout_ref:
  rollout:
    name: atom
trainer:
  engine:
    backend: atomrl_megatron
```

### Weight sync

`ATOMRolloutAdapter.update_weights()` writes the incoming named tensors to a
scoped temporary safetensors checkpoint and calls the public
`ATOMReplicaManager.reload_weights_from_path()`. The temporary directory is
removed after all ATOM replicas finish loading.

### Lifecycle

`atomrl.plugin.verl.lifecycle` creates and injects the manager into the rollout
adapter. The legacy async replica manages its own servers.

### Disable

```bash
export VERL_USE_EXTERNAL_PLUGINS=none
```

---

## vime

Pass hook functions by their dotted import path via CLI arguments:

```bash
python train.py \
  --custom-megatron-init-path atomrl.plugin.vime.hooks.custom_megatron_init \
  --custom-model-provider-path atomrl.plugin.vime.hooks.custom_model_provider \
  --rollout-function-path atomrl.plugin.vime.hooks.generate_rollout \
  --custom-update-weight-post-write-path \
      atomrl.plugin.vime.hooks.custom_update_weight_post_write
```

### Hook reference

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager`, stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel, applies FP8 if configured |
| `--rollout-function-path` | `generate_rollout` | Replaces vLLM rollout with ATOM |
| `--custom-megatron-before-train-step-hook-path` | `custom_megatron_before_train_step_hook` | Resets FP8 state |
| `--custom-megatron-before-log-prob-hook-path` | `custom_megatron_before_log_prob_hook` | Resets FP8 state |
| `--custom-update-weight-post-write-path` | `custom_update_weight_post_write` | Reloads ATOM from the new HF checkpoint on rank zero |

### FP8 setup

Set `args.atomrl_fp8_config` before training starts:

```python
args.atomrl_fp8_config = {"training": {"fp8_recipe": "delayed"}}
```

### ATOM lifecycle

Create and tear down the manager around the host training loop:

```python
from atomrl.plugin.vime.lifecycle import setup_atom_manager, teardown_atom_manager

setup_atom_manager(args, worker_group)
try:
    train()
finally:
    teardown_atom_manager(args)
```

---

## miles and slime

Both adapters expose the same Megatron/FP8 lifecycle hooks as vime while
returning their framework-native rollout output types. Their optional
dependencies are imported only when a rollout runs.

```python
from atomrl.plugin.miles.lifecycle import setup_atom_manager as setup_miles
from atomrl.plugin.slime.lifecycle import setup_atom_manager as setup_slime
```

The miles example is under `examples/plugin/miles/`. Lumen-RL main does not
provide a slime training example, so AtomRL provides only a pinned
install/import contract under `examples/plugin/slime/`.

Pinned framework compatibility tests live under `examples/plugin/` rather
than the public unit suite. They install each host stack in an isolated
`Dockerfile.plugin` target after the public base image is selected.

---

## Design principles

1. **Shared core, frozen wrappers** — framework-required top-level signatures
   stay in each adapter; common lifecycle and Megatron logic stays in private
   `atomrl.plugin` helpers.

2. **Public ATOM APIs only** — adapters use `servers`, `wake_all`, `sleep_all`,
   `reload_weights_from_path`, and `shutdown`.

3. **Lazy optional dependencies** — importing AtomRL does not import host
   frameworks.

4. **Frozen signatures** — hook function signatures are dictated by the host
   framework and must not change.

For the full design rationale, see [docs/plugin-integration.md](../docs/plugin-integration.md).
