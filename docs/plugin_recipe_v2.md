# Plugin recipes (`atom.plugin_recipe/2`)

ATOM's plugin surface is fixed: a plugin package serves ATOM through the plugin
host (`atom/plugin_host.py`, HOOK_API 3, see `docs/plugin_host_v3.md`) and a
recipe (`atom/plugin_recipe.py`). Nothing in a plugin imports ATOM model code,
pins ATOM file hashes, or patches ATOM source; a new fusion needs no ATOM change.

A recipe is fixed for the process: `--plugin-recipe <file or directory>` at
start, `--plugin-arm` to pick an arm. To switch recipes, restart;
`ATOM_PRELOAD=1` keeps the loaded weights on the GPUs across the restart.
`python -m atom.plugin_recipe PATH` validates and dumps a recipe on CPU.

## What builds what

| Process state | The target builds | The drafter builds |
| --- | --- | --- |
| no `--plugin-recipe`, or `ATOM_PLUGIN_HOST=0` | ATOM's own model | ATOM's own drafter |
| a recipe without model classes | ATOM's own model | ATOM's own drafter |
| `model_options.target_model_cls` / `draft_model_cls` | that class (a plugin package's model, with its seams) | that class |

## A recipe

```json
{
  "schema": "atom.plugin_recipe/2",
  "arch": "DeepseekV41ForCausalLM",
  "strategy": "tp4-c16",
  "graph": {"mode": "any"},
  "defaults": {"on_decline": "native", "allow_partition": true},
  "model_options": {"target_model_cls": "pp_models.dsv41.runtime.DeepseekV41PluginRuntimeModel"},
  "seams": {"v41.block.ffn": ["tp4-ar-add-provider"], "v41.attn.out": []},
  "modules": {
    "layers.*.attn": ["my-whole-attention"],
    "@pp_models.dsv41.moe.MoE:_topk": ["router-top6-provider"]
  },
  "aiter": {"quantize_fp8_group32": ["quant16k-provider"]},
  "plugin_options": {"dsv41": {"lean_head": true}},
  "env": {"ATOM_DSV41_MONO_HI": "auto"},
  "requires": {"atom_hook_api": ">=3,<4", "plugins": {"tp4-ar-add-provider": "sha256:..."}},
  "runtime": {"expect": {"tensor_parallel_size": 4}}
}
```

The owner tables may also sit under `plan` (`plan.seams`, `plan.modules`); a
top-level `seams` / `modules` wins.

- `seams`: ordered owners per seam (several for disjoint cells, `allow_partition`);
  `[]` plans a seam native by decision (it counts as planned for a taken-over
  module's inner seams).
- `modules`: a module path (`*` one segment; the drafter's modules under `draft.`)
  or `@<qualified.Class>` (every instance of the class), optionally `:method`.
- `aiter`: AITER providers, a list or grouped by op (one global order).
- `model_options`: only the model classes. A model package's own options go to
  `plugin_options[<package>]`, which ATOM passes through unread.
- `env`: the native switches the deployment needs (`ATOM_*`, `AITER_*`,
  `FLYDSL_*`, `PYTORCH_*`, `HIP_*`, `HSA_*`, `GPU_*`, `NCCL_*`, `RCCL_*`,
  `TORCH_*`; never `ATOM_PLUGIN_*`), set once by the server's main process
  before any worker starts. A name the environment already sets to another value
  refuses the recipe (`recipe-env-conflict`).
- `runtime.expect`: what the server must be (`level`, `cudagraph_mode`,
  `tensor_parallel_size`, `model_arch`, `mono`); a mismatch refuses the start.
- `requires`: `atom_hook_api`, the seam contracts the recipe was written for
  (checked against the built model), and optional plugin code digests. No ATOM or
  AITER version is pinned.
- `arms`: named subsets of the plan's providers (`native`, `plugin`, `identity`
  are always there); an arm changes routes only, never the model.
- `plugins` is optional: every owner named under `seams` / `modules` is a layer
  plugin, every name under `aiter` an AITER provider.
- A wrong type refuses the recipe. Top-level fields ATOM does not read are the
  document's own metadata (a tool's `plan_id`, `purpose`, `baseline`, notes) and
  are ignored; the removed sections `engine` and `dspark_options` are refused (a
  model's options go to `plugin_options`). One strategy's recipe set (one file per
  capture bucket set) must agree on everything but its tables.

## Whole-module and method takeover

```python
def register(host):
    def factory(module, capability):          # once per module, after the load
        return lambda *args, **kwargs: ...     # called exactly as the taken-over method
    return host.register_module(
        slot="@pp_models.dsv41.moe.MoE", method="_topk",
        owner="router-top6-provider", factory=factory,
        select=lambda cap, *args, **kwargs: covers(*args, **kwargs),  # pure: shapes / dtypes only
        graph="capture_safe",
    )
```

- Parameters, buffers and their names do not change: only the method is routed,
  after the load and before warmup and capture (`plugin_host.install_modules`).
- Slots install deepest first. An outer owner that declines a call runs the
  module's own forward, which reaches the inner owners: overlapping plugins
  compose by construction (`shadow_policy` defaults to `"nested"`).
- Under `torch.compile` a tensor-only call is one opaque op (`atom_plugin::span_n`).
- A method is taken over on the instance: a `@staticmethod` called through its
  class (`FusedMoE.select_experts(...)`) is never reached; take over the caller (as
  above, the model's routing method).