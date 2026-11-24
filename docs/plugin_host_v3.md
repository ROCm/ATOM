# Plugin host: the frozen surface (HOOK_API 3)

ATOM keeps only model-agnostic host mechanics. Everything model-specific -- the
model variant with its call sites and their seam contracts, the model's options,
the owners and their kernels -- lives in a plugin package (plugin_poc for
DeepSeek-V4.1 and Kimi-K3). A new fusion changes the plugin package and the
recipe, not ATOM. HOOK_API 2 plugins and recipes still load.

## What ATOM provides

| Surface | API | Notes |
| --- | --- | --- |
| Seam contracts | `plugin_host.seam(SeamSpec(...))`, `bind(name, native, spec=...)`, `site(name, spec=...)` | **No registry**: a model writes each seam's contract at its own call site. One name has one contract per process (`SeamConflict`). `SeamSpec`: inputs / kwargs / returns / key / graph / form, `parent` (strict nesting), `covers` (alternative spans it makes unreachable), `excludes` (seams that must stay native while it has an owner), `scope` (module path of the call site). The host itself declares only `model.install`. |
| Seam owners | `register(seam=, contract=, impl=, select=, graph=, owns_collectives=)` | Owners register before the build; checked against the spec when the model declares it. Per-call `select`; captured buckets fix the route; a decline runs native. |
| Module / method takeover | `register_module(slot=, factory=, select=, method=)`; recipe `modules: {"path": [...], "path:method": [...]}` | Any `nn.Module` forward or method, no seam needed. Deepest first; opaque op under compile (default fake: the module's own forward on fake parameters). |
| Boot hook | seam `model.install` (`model, atom_config, drafter, metadata_builder=`) | After load, before KV sizing and capture: an owner wraps or replaces the model (mono decode). Native runs the model's own installers. |
| Model variant | recipe `model_options.target_model_cls` / `draft_model_cls` | The plugin package's model. Without them ATOM builds its own model. |
| Plugin options | recipe `plugin_options: {name: {...}}`, `plugin_host.plugin_options(name)` | Unread by ATOM, part of `plan_id`. |
| Execution context | `plugin_host.context(rows=None) -> Context(captured, bucket, valid_rows)` | For code a seam capability does not reach. |
| AITER providers | recipe `aiter: [...]` or `{op: [providers]}` | Enabled with the plan, follow the arm (`native`: none; AITER has no `identity`), `prepare(device)` with the plan, listed in `status()`. |
| Plugin pins | recipe `requires.plugins: {provider: "sha256:..."}`, `plugin_host.provider_digest(name)` | A provider's code (its package plus `PLUGIN_CODE_ROOTS`). No ATOM or AITER version is pinned. |
| Engine options / env | recipe `engine` / `env` | ATOM's own scheduler options; native switches applied once in the main process. |

## Checks against the built model and at run time

- **`check_plan()`** (after the model and drafter are built): a planned seam the
  model never declared (`recipe-seam-unbound`), a required contract the model
  does not have (`recipe-contract-mismatch`), an owned seam whose `excludes` are
  owned (`recipe-excludes`), a taken-over module whose inner seams (`scope`) the
  recipe does not plan (`recipe-inner-unplanned`) refuse the start. Owners an
  outer owner may make unreachable (`parent`, `covers`) are logged; refused under
  `defaults.shadow_policy: "error"`.
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

## Still in ATOM's shared code

V4.1 call sites inside ATOM's shared ops write their contracts in their own
files and stay native until a plan owns them: `v41.op.quant_fp8_g32`
(`model_ops/blockscale.py`), `v41.moe.topk` (`model_ops/moe.py`),
`v41.attn.qk_norm` (`model_ops/layernorm.py`), `v41.attn.index_topk`
(`model_ops/indexer/score.py`), `v41.attn.attend`
(`model_ops/v4_kernels/paged_2src.py`), `v41.spec.verify_sample`
(`model_engine/model_runner.py`) and the transition window
(`plugin_transition.py`). New models add call sites in their own plugin-package
variant, never in shared code.