# Plugin host: the frozen surface (HOOK_API 3)

ATOM keeps only model-agnostic host mechanisms. Everything model-specific — the
model variant with its call sites, the seam specs, the model's options, the
owners and their kernels — lives in a plugin package (plugin_poc for
DeepSeek-V4.1). A new fusion changes the plugin package and the recipe, not
ATOM.

## What ATOM provides

| Surface | API | Notes |
| --- | --- | --- |
| Seam registry | `plugin_host.declare_seams(specs, source=)`, entry point group `atom.seam_specs` | A package declares its seams (pure import; the main process reads them to check a recipe). One name, one contract per process (`SeamConflict`). `SeamSpec.scope` is the module path of the call site, `module_slot` the slot a seam is exactly the span of. ATOM declares only `model.install`. |
| Seam owners | `register(seam=, impl=, select=, graph=, owns_collectives=)` | Per-call `select`; captured buckets fix the route; a decline runs native. |
| Module / method takeover | `register_module(slot=, factory=, select=, method=)`; recipe `modules: {"path": [...], "path:method": [...]}` | Any `nn.Module` forward or method, no seam needed. Deepest first; opaque op under compile (default fake: the module's own forward on fake parameters). |
| Boot hook | seam `model.install` (`model, atom_config, drafter, metadata_builder=`) | After load, before KV sizing and capture: an owner wraps or replaces the model to serve (mono decode). Native runs the model's own installers. |
| Model variant | recipe `model_options.target_model_cls` / `draft_model_cls` | The plugin package's model (with its call sites). Without them ATOM builds its own model. |
| Plugin options | recipe `plugin_options: {name: {...}}`, `plugin_host.plugin_options(name)` | Unread by ATOM, part of `plan_id`. |
| Execution context | `plugin_host.context(rows=None) -> Context(captured, bucket, valid_rows)` | For code a seam capability does not reach (AITER providers, a module owner's inner kernels). |
| AITER providers | recipe `aiter: [...]` or `{op: [providers]}` | Enabled with the plan, follow the arm (`native`: none; AITER has no `identity`), `prepare(device)` called with the plan, listed in `status()`. |
| Plugin pins | recipe `requires.plugins: {provider: "sha256:..."}`, `plugin_host.provider_digest(name)` | A provider's code (its package plus `PLUGIN_CODE_ROOTS`). No ATOM or AITER version is pinned. |
| Engine options | recipe `engine` | ATOM's own scheduler options (#3644). |
| Env | recipe `env` | Native switches, applied once in the main process. |

## Guarantees checked at run time

- **Recipe reachability.** `plugin_host.route_report()` (logged after capture,
  part of `status()`): per seam / module slot, the owner each captured bucket
  and the eager calls routed to, the planned owners no call reached
  (`planned_but_unreached`) and any owner outside the plan.
- **Inner seams of a taken-over module.** A recipe that takes over a module
  (not a seam's own slot) must plan every seam inside it — owners, or `[]` for
  native (`recipe-inner-unplanned`).
- **One weight preparation.** Every captured graph, prefill and decode of a
  process (and every arm) shares the weights prepared at boot (load, boot
  hook, module installs, `prepare_plan`). `freeze_weights` records every
  parameter and buffer then; the capture report names any whose storage, layout
  or in-place version changed (`weights_changed`; an error under strict). An owner
  that needs another layout keeps a private copy or declines.
- **Cross-rank agreement.** A captured decision among owners that run
  collectives (`owns_collectives`) is agreed by the TP ranks; a disagreement
  runs native on every rank for that call site and bucket.

## Still in ATOM's shared code

Call sites named by V4.1 seams that are inside ATOM's shared ops stay as plain
hooks (native until a package declares the seam and a plan owns it):
`v41.op.quant_fp8_g32` (`model_ops/blockscale.py`), `v41.moe.topk`
(`model_ops/moe.py`), `v41.attn.qk_norm` (`model_ops/layernorm.py`),
`v41.attn.index_topk` (`model_ops/indexer/score.py`), `v41.attn.attend`
(`model_ops/v4_kernels/paged_2src.py`), `v41.spec.verify_sample`
(`model_engine/model_runner.py`) and the transition window
(`plugin_transition.py`). New fusion points use the plugin model, module or
method takeover instead of new sites in shared code.
