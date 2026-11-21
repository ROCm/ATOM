# Plugin host hot-plug: the ATOM-side API (host half)

Branch `hotplug/host`, base `b70d7a3` (ATOM PR #3236 head + plugin host v2 /
DSV4.1 seams). This note is the contract the engine half calls; it does not
describe graph capture (that stays the engine's job).

## Boot

`atom/model_engine/model_runner.py` already calls, before `get_model`:

```python
plugin_host.configure(
    config.plugin_source,
    arm=config.plugin_arm,
    config=config,
    hotplug=getattr(config, "plugin_hotplug", False),
)
```

With `hotplug=True` and `ATOM_PLUGIN_HOST=1` the host boots for a plan it does
not have yet:

- every seam whose `SEAMS[...].form == "bind"` binds now: a `Router` for
  non-opaque seams and the `atom_plugin::span` opaque op for `opaque_op` seams
  (`v41.block.ffn`, `v41.moe.shared_act`), even when the boot plan owns none.
  The bound call sites are the ones the model keeps for its lifetime.
- `call`/`site` seams dispatch through `_dispatch` from boot; with no owner in
  the boot plan they run native and record a `no-table` decision.
- the kill switch wins: `ATOM_PLUGIN_HOST=0` binds nothing,
  `hotplug_enabled()` is False and `hotplug_plan` refuses.

`plugin_host.hotplug_enabled() -> bool` is the engine's check for "this process
booted for hot-plug" (e.g. whether to expose the route).

## Installing a plan after the build

```python
plugin_host.hotplug_plan(plan, *, arm="plugin", config=None) -> dict
```

Called by the engine utility command after the model build, with generation
quiesced (paused/drained), on every TP rank, before re-capturing CUDA graphs.

`plan` may be:

- a `atom.plugin_recipe.Plan`;
- the `atom.plugin_recipe.RecipeSource` a `Config` carries (resolved here);
- a plain, picklable payload `{"documents": [recipe JSON texts or decoded
  recipe objects, ...], "enable": ["provider", ...], "plan_id": optional}`.
  `"recipe"` is accepted as an alias for `"documents"`; a mapping that itself
  has `"schema"` is one recipe document. Construct/validate one with
  `atom.plugin_recipe.from_payload(payload)`, which returns a `RecipeSource`
  (`None` = no plugin). The engine should read recipe *files* to texts before
  sending; the host never reads paths;
- a comma-separated name string or an iterable of names (the `--plugin-enable`
  shorthand; the plan owns whatever those plugins register, in that order);
- `None`: install the empty plan (every seam native; bindings and span ops are
  kept, so a later plan still works).

The installed state is swapped under the host lock; on any failure the
previous plan, arm, route tables, registrations, strict/on-decline settings
and AITER enable list are restored before the error propagates (per process —
the engine must fail all ranks together).

Return value (JSON-safe):

```python
{
  "hotplug": True,
  "plan_id": "sha256:..." | None,
  "providers": [...],          # layer plugins loaded, preference order
  "aiter": [...],              # AITER providers handed to aiter.plugin.configure
  "capture_rows": [40, 48],    # capture-bucket table keys the plan declares
                               # (padded `running_tokens`); recapture these
  "tables": ["40", "48"],      # every table key of the installed plan, sorted,
                               # as strings: "any" | "eager" | a bucket
}
```

## Errors

- `RuntimeError("... hot-plug boot ...")`: the process was not configured with
  `hotplug=True` (or the kill switch was on at boot).
- `RuntimeError("... after the model build ...")`: no seam has bound yet;
  install only after `get_model`.
- `RuntimeError("... disabled (ATOM_PLUGIN_HOST=0) ...")`: kill switch set.
- `atom.plugin_recipe.RecipeRefused` (`.reason` is the short code):
  - `recipe-plugin-drift`: a planned provider has no entry point, or its
    entry point raised while loading;
  - `recipe-coverage-miss`: a planned owner did not register the seam the
    plan gives it;
  - `recipe-schema` / `recipe-model-mismatch` / `recipe-pin-mismatch` /
    `recipe-contract-mismatch` / `recipe-set-conflict`: the payload/plan is
    invalid or resolves differently here;
  - `recipe-runtime-mismatch`: `runtime.expect` does not match `config`.
- `ValueError` from `set_arm` for an arm the plan does not declare.

## Semantics after a successful install

- Existing bound call sites and `call`/`site` call sites route to the new
  tables; nothing is re-bound or rebuilt.
- Every planned provider's entry point is invoked again under the new plan
  (registrations are rebuilt from the plan, not carried over), so a later plan
  may cover a seam an earlier one did not. Plugin `register(host)` callables
  must therefore be pure/idempotent (the plugin_poc providers are; their tests
  already call `register` twice).
- The AITER half is switched with
  `aiter.plugin.configure(source=host.AITER_SOURCE, enable=plan.aiter)`
  (`()` for the empty plan) when AITER has `aiter.plugin`.
- Captures made before the install replay their old decisions: the engine must
  re-capture the decode graphs for `capture_rows` (or the requested rows).
- `_owned` only grows: a seam once bound/planned keeps dispatching, so a seam
  the new plan drops serves native through the same call site.
- The route cache is cleared; there is no partial installation.

## Inspect / diagnostics

`plugin_host.status()` now reports `"hotplug"`, `"plan_id"`, `"bound"`
(sorted list of bound seams), `"capture_rows"`, the per-seam tables under the
current arm, `"span_ops"` and `"refusals"`. `plugin_host.set_arm(arm)` still
switches `plugin` / `native` / `identity` / a recipe arm in place.

## Validation on this machine

`python -m pytest tests/test_plugin_host.py -q`: 89 passed, 1 skipped
(win32-only spawn test). The repo imports POSIX-only names
(`multiprocessing.context.ForkContext`, `signal.SIGKILL`) and the Windows
sandbox mistranslates `mkdir(mode=0o700)`; the run above used a throwaway
`sitecustomize` shim outside the repo
(`_tmp/atom-win32-shim`, `PYTHONPATH=...`, `-p no:cacheprovider`,
`--basetemp=<workspace dir>`). Unshimmed on this box the command fails in
`tests/conftest.py` importing `atom.utils`, before any test runs; on a Linux
container the plain command has no such issue.

## Host-side risks / limitations

- Hot-plug is per process: all ranks install independently; atomicity across
  ranks is the engine's (single barrier + fail-together).
- The swap is under `_lock`, but dispatch is lock-free: call it with
  generation quiesced, as the engine utility command does.
- A provider owned by the boot plan is re-invoked on install; a plugin whose
  `register` is not re-entrant would break. Same for side effects in the
  entry point itself.
- Hot-plug mode makes every call seam dispatch through `_dispatch` even with
  no owner (one dict lookup plus the graph-state probe); that is the cost of
  keeping the call sites hot-swappable.
- The host does not recapture graphs and does not map padded rows to batch
  sizes; `capture_rows` are the plan's declared buckets.
- The empty plan is how a server goes back to native without losing hot-plug
  (the AITER list is emptied; bindings stay).
- A boot-only seam (`v41.mono.install`, the DeepSeek-V4.1 whole-step decode
  installer) runs once, at build: `hotplug_plan` refuses a plan that would
  change its owners (`RecipeRefused("recipe-boot-only")`); restart instead.
