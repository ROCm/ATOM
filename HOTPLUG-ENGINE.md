# ATOM engine side of plugin hot-plug

Status: **CPU-side implementation complete; GPU validation pending** (`gpu-runner`).
Branch `hotplug/engine` (base `b70d7a3`), worktree `_wt/atom-hotplug-engine`.

While a TP server keeps running, atomically switch the installed plugin plan/arm
and re-record the decode CUDA graphs for the requested capture rows, on every TP
rank, without a restart. This document is the engine-side contract; the plugin
host side (`plugin_host.hotplug_plan`) is implemented in parallel on
`hotplug/host` and merged by the Lead.

## Required host API (parallel branch, merged at integration)

The engine calls three host entry points. Until the host branch is merged they
do not exist in this worktree; a hot-plug request on a build without them is
refused with a clear 409, never crash:

- `plugin_host.configure(..., hotplug=bool)` — boot with every seam bound so a
  later plan can be installed without rebuilding the model (the call site was
  already in `model_runner.py`; this branch adds the `Config.plugin_hotplug`
  knob and CLI/env wiring).
- `plugin_host.hotplug_enabled() -> bool` — whether this runner booted
  hot-plug-capable.
- `plugin_host.hotplug_plan(plan, *, arm="plugin", config=None) -> dict` —
  install the plan/arm in this process; the returned dict may carry `plan_id`
  and `rows` (used if the request named none). It must swap the route tables
  and the active arm **and drop every cached routing decision**
  (`_routes`/`_dispatch` memo), or the recapture below would record the old
  plan's decisions again. Boot with `configure(..., hotplug=True)` must bind
  every seam even when no plan is installed, so a later plan needs no rebuild.

## Enablement

```
--plugin-hotplug            # or ATOM_PLUGIN_HOTPLUG=1
```

This enables nothing by itself: a hot-plug request still names a recipe. The
flag (a) binds every seam at model build and (b) permits the internal endpoint.
`Config.plugin_hotplug` also keys the compile-cache hash (`compute_hash`), so a
hot-plug-capable boot never reuses a native artifact.

## Endpoint

`POST /plugin/hotplug` — internal, `include_in_schema=False`, on the engine-owning
process (`_engine_route`, relayed by front-end workers like the other management
routes). It is **registered always** and answers 409 when the server was not
started with hot-plug enabled.

Request JSON:

```json
{
  "recipe": ["<recipe JSON text>" | "<path to recipe/dir>" | {recipe document}],
  "enable": ["plugin-name", "..."],
  "arm": "plugin",
  "rows": [16, 32],
  "capture": true,
  "timeout": 300.0
}
```

| field | required | meaning |
|---|---|---|
| `recipe` | unless `enable` given | recipe documents as JSON text, objects, or filesystem paths (a directory = one strategy's recipe set). Resolved once here; a bad recipe is a 400 before any rank changes state. |
| `enable` | no | `--plugin-enable`-equivalent, may only narrow the recipe's providers. |
| `arm` | no (default `plugin`) | one of the resolved plan's arms (`plugin`, `native`, `identity`, or a recipe arm). |
| `rows` | no | padded `running_tokens` of decode request graphs to recapture, i.e. `bs * (mtp_k + 1)`. Omitted: the plan's capture-table rows that this server has buckets for; if it declares none, the whole current ladder. |
| `capture` | no (default `true`) | `false` switches the plan only (safe under load; recorded decode graphs then keep the previous routing). |
| `timeout` | no (default 300 s) | whole operation budget; a rank missing it answers 504. |

Response JSON (200):

```json
{
  "status": "ok",
  "plan_id": "sha256:...",
  "rows": [16, 32],
  "sizes": [4, 8],
  "capture_seconds": 2.7,
  "ranks": [{"rank": 0, "tp_world_size": 4, "status": "ok", "error": null,
             "plan_id": "sha256:...", "arm": "plugin", "rows": [16, 32],
             "sizes": [4, 8], "capture_seconds": 2.7, "captured": true,
             "elapsed_seconds": 2.9}]
}
```

Errors are a JSON envelope `{"status": "error", "error": {"reason", "message"}}`:

| HTTP | reason | when |
|---|---|---|
| 400 | `no-recipe`, `recipe-unreadable`, `recipe-schema`, `recipe-model-mismatch`, `unknown-arm`, `payload-schema`, `rows-invalid` | the caller can fix the request |
| 409 | `hotplug-disabled`, `topology-unsupported`, `hotplug-refused`, `plan-id-mismatch`, `hotplug-failed` | engine state refuses: not enabled, unsupported topology, a plan the host declines, a capture that failed, ranks disagreeing |
| 504 | `hotplug-timeout` | a rank did not answer within `timeout` |

The server never answers 500 for an expected refusal and stays up.

## Engine flow

1. **API** (`atom/entrypoints/openai/api_server.py`) validates the body into a
   `RecipeSource` + rows + arm (`atom/model_engine/plugin_hotplug.py`), checks
   enablement/topology, and calls `engine.core_mgr.hotplug_plugin(payload,
   timeout)` in a thread (a capture is seconds; the loop also serves SSE).
2. **Manager** (`engine_core_mgr.hotplug_plugin`) sends the synchronous
   utility command to every engine (`broadcast_utility_command_sync`) and
   merges the rank reports. It refuses a manager with more than one engine
   (DP/PP: the capture records the group's collectives and this cut does not
   orchestrate several engines pausing/entering together).
3. **EngineCore handler** (`engine_utility.EngineUtilityHandler._handle_hotplug_plugin`):
   - refuses (409) when disabled or in an unsupported topology;
   - pauses generation (preserving a caller's own pause), waits out any
     deferred forward work, and requires an **idle engine** for a capture (no
     running request; see Limitations);
   - `runner_mgr.call_func("hotplug_plugin", payload, wait_out=True)` — a
     synchronous broadcast, so every TP rank installs/captures before the
     answer;
   - resumes generation in `finally`, replies with the rank report or the
     refusal reason. A worker exception would end that worker on this RPC
     path, so the runner converts every failure into a TP-agreed error report
     instead of raising through the transport.
4. **ModelRunner** (`hotplug_plugin`), on every TP rank in the same order:
   1. validate: host API present, hot-plug enabled, payload/recipe resolves,
      arm/payload sane (deterministic — a refusal is collected, not raised);
   2. TP-agree on "continue vs abort" before touching the plan;
   3. `plugin_host.hotplug_plan(plan, arm=..., config=self.config)` installs
      the new route tables/arm in this process;
   4. TP-agree again (a rank must not start a collective capture its peers
      skip);
   5. `recapture_cudagraph(rows)` when `capture` and not `enforce_eager` /
      `disagg_is_decode`; else report why it was skipped;
   6. return `{"rank", "tp_world_size", "status", "plan_id", "rows", "sizes",
      "capture_seconds", "captured", ...}`.

## Capture semantics

- **Reuses the startup path.** `capture_cudagraph()`'s per-bucket body is now
  `ModelRunner._capture_request_bucket()`, and the bound builder
  `_request_capture_builder()` wraps `build_for_cudagraph_capture` with the same
  query-metadata staging, position buffer rewrite and H2D window the startup
  capture uses. A bucket cannot be recorded differently at boot and later.
- **Scope:** whole-forward decode request graphs only (`self.req_graph`, keyed by
  `request_bucket(bs, max_q_len)`). DSpark confidence scheduling recaptures every
  query-length bucket of each requested `bs`; TBO-split buckets are recorded
  split exactly as at startup.
- **Release-then-recapture (MVP).** Every recorded decode graph decided with the
  previous plan's tables, so `req_graph.release_graphs()` drops them all and only
  the requested buckets are recorded again. `self.capture_sizes` /
  `capture_sizes_np` are then narrowed to the buckets that really exist, so
  `ForwardMode.decide` runs every other decode batch **eagerly** (never replays
  a graph no rank holds).
- **TP coupling.** A CPU-group barrier before the capture and after it; a
  `MAX` all-reduce agrees on "failed" and on the surviving ladder, so all ranks
  narrow identically and a partial capture degrades every rank to eager instead
  of leaving mixed shapes. On failure all ranks raise the same refusal.
- **No live-cache exposure during the recording forward:** the capture staging
  blanks the KV slot map (same `blank_cache_write_targets()` as startup) and the
  warmup forward runs on placeholder cache rows. The synthetic batch still names
  per-request **state** slots `0..bs-1`, which is why a capture requires an idle
  engine (below).
- Memory: the recapture records into a fresh graph pool (the released graphs'
  pool is retired), so RSS grows by the recaptured graphs only.

## Known limitations (documented MVP cut)

1. **Unrequested decode buckets run eagerly** after a switching hot-plug —
   release-all is deliberate; replacing only the target buckets would keep the
   shared pool and is the follow-up.
2. **A capture requires an idle engine**: no request may be running or in
   flight (`409 hotplug-refused`). The recording forward executes the model for
   real on the serving caches; the synthetic batch names state slots `0..bs-1`
   and startup capture only ever ran on empty pools. Drain traffic, hot-plug,
   resume. `capture=false` switches the plan under load (recorded graphs keep
   their old routing).
3. **Only decode request graphs are recaptured.** Graphs recorded at startup for
   other passes keep their captured routing: prefill TR token graphs, prefill
   request/decoder graphs, and the drafter's draft graphs. Under the plugin
   host's contract plugin and native are equivalent, so this is a
   routing/performance gap, not a numerics one — restart to switch those too.
4. **Topologies:** `dp > 1`, DP attention, `pp > 1` and intra-GPU P/D
   disaggregation are refused (`409 topology-unsupported`). They need a
   pause-all-then-capture orchestration across engines.
5. **TR_HYBRID token-graph captures** cannot be recaptured per bucket; refused
   rather than patched.
6. **No rollback on failure:** a failure after `hotplug_plan` leaves the new
   plan installed (the host call is not transactional) and decode runs only on
   graphs that survived — typically eager. Retry the request or restart.
7. DSpark SPS calibration and draft warmup are startup-only; they are not
   re-run after a plan switch.
8. The endpoint is always registered and gated with 409 rather than being
   absent when disabled (a front-end worker cannot know the engine's config at
   import).
9. `docs/environment_variables.md` has no `ATOM_PLUGIN_HOTPLUG` row yet (the
   file is outside this task's write scope).

## Files

| path | change |
|---|---|
| `atom/config.py` | `Config.plugin_hotplug: bool = False` |
| `atom/model_engine/arg_utils.py` | `--plugin-hotplug`, `ATOM_PLUGIN_HOTPLUG` env fallback in `_get_engine_kwargs` |
| `atom/utils/envs.py` | `ATOM_PLUGIN_HOTPLUG` |
| `atom/model_engine/model_runner.py` | capture loop lifted into `_capture_request_bucket` / `_request_capture_builder`; `hotplug_plugin`; `recapture_cudagraph`; TP barriers/agreement |
| `atom/model_engine/engine_utility.py` | synchronous `hotplug_plugin` utility handler (pause → idle gate → runner → resume), `_require_idle_for_capture` |
| `atom/model_engine/engine_core_mgr.py` | `CoreManager.hotplug_plugin(payload, timeout)`; hot-plug ordered head-first under PP |
| `atom/entrypoints/openai/api_server.py` | internal `POST /plugin/hotplug` route + refusal envelope |
| `atom/model_engine/plugin_hotplug.py` | new, CPU-light payload/rows/report/refusal rules |
| `tests/test_plugin_hotplug.py` | new, 25 CPU-only tests |
| `HOTPLUG-ENGINE.md` | this file |

Not touched (host-owned): `atom/plugin_host.py`, `atom/plugin_recipe.py`,
`tests/test_plugin_host.py`.

## CPU verification (this worktree, Windows, Python 3.14, torch 2.15 CPU, no AITER)

```console
$ python -m py_compile atom/config.py atom/model_engine/arg_utils.py \
    atom/model_engine/model_runner.py atom/model_engine/engine_utility.py \
    atom/model_engine/engine_core_mgr.py atom/model_engine/plugin_hotplug.py \
    atom/entrypoints/openai/api_server.py atom/utils/envs.py
# (exit 0)

$ python -m pytest tests/test_plugin_hotplug.py -q
# cannot collect in-repo on this box: tests/conftest.py imports atom.config,
# which imports atom.utils; Python 3.14 removed multiprocessing.context.
# ForkContext/SpawnContext, so the repo's own conftest fails before any test
# (pre-existing, unrelated to this branch). The test file is self-contained;
# it was copied next to the checkout and run with the worktree on PYTHONPATH:
$ python -m pytest <copy>/test_plugin_hotplug.py -q -p no:cacheprovider
25 passed in 6.60s

$ black --check (via API, pinned local 26.10.1)
all changed files clean
```

GPU validation (separate `gpu-runner` task) should cover: boot with
`--plugin-hotplug` and an initial recipe; idle-server `POST /plugin/hotplug`
with a new recipe + `rows` from `GET /plugin/hotplug`-equivalent plan info
(or explicit `bs*(mtp_k+1)` rows); assert 200, `plan_id` changes, decode runs at
the recaptured buckets, unrequested buckets eager; refusals: disabled (409),
busy engine (409), bad recipe (400), unsupported topology (409); then traffic
with the new plan and compare plugin-vs-native equivalence.
