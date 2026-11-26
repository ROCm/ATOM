# AtomRL Plugin Adapters

Thin wiring code that injects AtomRL capabilities (ATOM rollout, Lumen FP8
training) into external RL frameworks. All business logic lives in the main
`atomrl/` package — each file here is glue only.

## Directory layout

```
atomrl/plugin/
├── verl/
│   └── register.py   # entry-point auto-loaded by verl on import
└── vime/
│   └── hooks.py      # CLI hook functions for --custom-*-path
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

Not injected. ATOM manages its own weights via `sleep`/`wake_up`.
`ATOMRolloutAdapter.update_weights()` is a no-op.

### Disable

```bash
export VERL_USE_EXTERNAL_PLUGINS=none
```

---

## vime

Pass hook functions by their dotted import path via CLI arguments:

```bash
python train.py \
  --custom-megatron-init-path  atomrl.plugin.vime.hooks.custom_megatron_init \
  --custom-model-provider-path atomrl.plugin.vime.hooks.custom_model_provider \
  --rollout-function-path      atomrl.plugin.vime.hooks.generate_rollout
```

### Hook reference

| CLI argument | Function | What it does |
|---|---|---|
| `--custom-megatron-init-path` | `custom_megatron_init` | Creates `FP8TrainingManager`, stores on `args` |
| `--custom-model-provider-path` | `custom_model_provider` | Builds GPTModel, applies FP8 if configured |
| `--rollout-function-path` | `generate_rollout` | Replaces vLLM rollout with ATOM |

### FP8 setup

Set `args.atomrl_fp8_config` before training starts:

```python
args.atomrl_fp8_config = {"training": {"fp8_recipe": "delayed"}}
```

### ATOM rollout prerequisite

Set `args._atomrl_atom_manager` before the first rollout call:

```python
from atomrl.engine.inference.atom_ray_server import ATOMReplicaManager

args._atomrl_atom_manager = ATOMReplicaManager(
    actor_wg=worker_group,
    model_name="/path/to/model",
    engine_kwargs={"tensor_parallel_size": 8},
)
args._atomrl_atom_manager.create()
```

---

## Design principles

1. **No adapter logic** — plugin files only import and forward; all real
   computation stays in `atomrl/`.

2. **Weight sync is host-native** — each framework uses its own mechanism
   (NCCL, P2P, disk). ATOM handles its own weight lifecycle.

3. **Frozen signatures** — hook function signatures are dictated by the host
   framework and must not change.

For the full design rationale, see [docs/plugin-integration.md](../docs/plugin-integration.md).
