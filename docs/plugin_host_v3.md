# Plugin host: the frozen surface (HOOK_API 3)

ATOM keeps only model-agnostic host mechanics. Everything model-specific -- the
model variant with its call sites and their seam contracts, the model's options,
the owners and their kernels -- lives in a plugin package (for example
plugin_poc, for DeepSeek-V4.1 and Kimi-K3). A new fusion changes the plugin
package and the recipe, not ATOM.

A plan is fixed for the process. **To switch recipes, restart the server**;
`ATOM_PRELOAD=1` keeps the loaded weights on the GPUs across the restart (the
weight cache's key includes the recipe's model classes, and the weight-cache
daemon builds the recipe's model classes itself, so a daemon's weights are never
mapped into another model class). There is no in-process hot-plug.

## What ATOM provides

| Surface | API | Notes |
| --- | --- | --- |
| Seam contracts | `plugin_host.seam(SeamSpec(...))`, `bind(name, native, spec=...)`, `call(name, native, ...)`, `site(name, spec=...)` | **No registry**: a model writes each seam's contract at its own call site. One name has one contract per process (`SeamConflict`). `SeamSpec`: inputs / kwargs / returns / key / graph / form (`bind` or `call`), `parent` (strict nesting), `covers` (alternative spans it makes unreachable), `excludes` (seams that must stay native while it has an owner), `scope` (module path of the call site). The host itself declares only `model.install`. |
| Seam owners | `register(seam=, contract=, impl=, select=, graph=, owns_collectives=)` | Owners register before the build; checked against the spec when the model declares it. Per-call `select`; captured buckets fix the route; a decline runs native. |
| Module / method takeover | `register_module(slot=, factory=, select=, method=)`; recipe `modules: {slot: [...]}` | A slot is a module path (`layers.*.ffn`, `*` one segment) or a class, `@<qualified.Class>` (every instance, wherever it sits: one plugin serves the class in any model), optionally `:method` (that method alone). No seam needed. Deepest first; one opaque op under compile (default fake: the module's own forward on fake parameters). The method is replaced on the instance: a `@staticmethod` the code calls through the class (`FusedMoE.select_experts(...)`) is never reached that way; take over its caller instead. |
| Boot hook | seam `model.install` (`model, atom_config, drafter, metadata_builder=`) | After load, before KV sizing and capture: an owner wraps or replaces the model (mono decode). Native runs the model's own installers. |
| Model variant | recipe `model_options.target_model_cls` / `draft_model_cls` | The plugin package's model. Without them ATOM builds its own model. |
| Plugin options | recipe `plugin_options: {name: {...}}`, `plugin_host.plugin_options(name)` | Unread by ATOM, part of `plan_id`. |
| Execution context | `plugin_host.context(rows=None) -> Context(captured, bucket, valid_rows)` | For code a seam capability does not reach. |
| AITER providers | recipe `aiter: [...]` or `{op: [providers]}` | Enabled with the plan, follow the arm (`native`: none; AITER has no `identity`), `prepare(device)` with the plan, listed in `status()`. |
| Plugin pins | recipe `requires.plugins: {provider: "sha256:..."}`, `plugin_host.provider_digest(name)` | A provider's code (its package plus `PLUGIN_CODE_ROOTS`, `plugin_code_digest` layout). No ATOM or AITER version is pinned. |
| Arms | `--plugin-arm plugin \| native \| identity \| <recipe arm>` | Chosen at start; an A/B is two starts, each a clean process. |
| Environment | recipe `env` | Native ATOM / AITER switches the deployment needs, applied once in the main process. |

## Lifecycle (ModelRunner)

| When | Host step |
| --- | --- |
| before the model is built | `configure(plan, arm, config)`: entry points loaded, owners registered, AITER providers enabled; the model classes come from the plan (`model_class_for` / `draft_model_class_for`, which a process without a plan, the weight-cache daemon, reads from the config's recipe) |
| after the target and the drafter are built | `install_built(model, drafter)`: `install_modules` (target, then the drafter under `draft.`), `check_plan()` |
| after the load, before KV sizing | `model.install` (the boot hook; native: the model's own installers), then the verify head is bound |
| before warmup | `prepare_built(device, model, drafter)`: `prepare_plan(device)` (owners' and AITER providers' one-time setup), `freeze_weights` |
| after capture | `log_route_report("capture")`: routes, planned owners no call reached, weights changed |

A plan is resolved identically on every rank (`plan_id`); captured decisions
among owners that run collectives are voted (below).

## Checks against the built model and at run time

- **`check_plan()`** (after the model and drafter are built): a planned seam the
  model never declared (`recipe-seam-unbound`), a required contract the model
  does not have (`recipe-contract-mismatch`), an owned seam whose `excludes` are
  owned (`recipe-excludes`), a taken-over module (by path, or where a class
  slot's instances sit) whose inner seams (`scope`) the recipe does not plan
  (`recipe-inner-unplanned`; a method takeover has none) refuse the start.
  Owners an outer owner may make unreachable (`parent`, `covers`) are logged;
  refused under `defaults.shadow_policy: "error"`.
- **`route_report()`** (logged after capture, part of `status()`): per seam /
  module slot, the owner each captured bucket and the eager calls routed to, the
  planned owners no call reached, and any owner outside the plan.
- **One weight preparation.** Every captured graph, prefill and decode of a
  process (and every arm) shares the weights prepared at boot. `freeze_weights`
  records every parameter and buffer then (storage, layout, sampled contents);
  the capture report names any that changed (`weights_changed`; an error under
  strict).
- **Cross-rank agreement.** A captured decision among owners that run
  collectives (`owns_collectives`) is agreed by the TP ranks; a disagreement runs
  native on every rank for that call site and bucket.

## EPLB

EPLB keeps the expert weights and their placement in ATOM: it collects each MoE
layer's expert tensors at its start (the quant method's `get_eplb_weight_views`,
else the layer's standard parameters) and migrates experts by copying in place;
routing is the logical top-k (`select_experts`, where a method takeover of the
model's routing applies) and then `place_experts` (the logical-to-physical map and
the load record). An owner that takes over a MoE block under EPLB must route
through `place_experts` and compute on those same expert tensors, or decline; a
private copy of the expert weights is not migrated.

## Still in ATOM's shared code

One V4.1 call site inside ATOM's shared ops writes its contract in its own file
and stays native until a plan owns it: `v41.op.quant_fp8_g32`
(`model_ops/blockscale.py`). Anything else is reached through a plugin
package's model variant or a module / class slot; new models add call sites in
their own plugin-package variant, never in shared code.