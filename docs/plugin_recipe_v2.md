# Plugin recipes v2, whole-module takeover and the DeepSeek-V4.1 plugin model

ATOM's plugin surface is fixed: a plugin package (for example `coderfeli/plugin_poc`)
serves ATOM through the plugin host (`atom/plugin_host.py`, HOOK_API 2) and a recipe
(`atom/plugin_recipe.py`). Nothing in a plugin imports ATOM model code, pins ATOM
file hashes, or patches ATOM source; a new fusion needs no ATOM change.

## What builds what

| Process state | `DeepseekV41ForCausalLM` builds | DSpark draft builds |
| --- | --- | --- |
| no `--plugin-recipe` / `--plugin-enable` / `--plugin-hotplug` | `atom.models.deepseek_v41` (ATOM's own, no seams) | `atom.models.deepseek_v41.dspark` |
| a plan, or a hot-plug boot | `atom.models.deepseek_v41_plugin.runtime.DeepseekV41PluginRuntimeModel` | `atom.models.deepseek_v41_plugin.dspark.DeepseekV41DSpark` |
| `model_options.target_model_cls` / `draft_model_cls` set | that class | that class |

`atom/models/deepseek_v41_plugin/` is the V4.1 model with every plugin seam and every
recipe option in it (ROCm/ATOM #3529 seams; #3639 / #3645 kernels). It shares ATOM's
weights layout, loader, runtime (graphs, bounded replay, guarded ops) and mono decode;
`ATOM_PLUGIN_HOST=0` builds ATOM's own model.

## Recipe v2 (`atom.plugin_recipe/2`)

```json
{
  "schema": "atom.plugin_recipe/2",
  "arch": "DeepseekV41ForCausalLM",
  "strategy": "dep8-c256",
  "graph": {"mode": "any"},
  "engine": {
    "prefill_short_first_rows": 96,
    "prefill_short_first_long_rows": 4096,
    "prefill_short_first_max_skips": 32,
    "prefill_cost_ceil": true,
    "prefill_decode_interval_busy": 2,
    "prefill_decode_interval_busy_seqs": 48,
    "prefill_long_cap_busy": 4096,
    "prefill_long_cap_busy_seqs": 48,
    "dp_session_prefix_owner_tokens": 512,
    "dp_session_prefix_owner_max_extra_reqs": 8
  },
  "model_options": {
    "ffn_mhc_fuse_quant": true,
    "shared_act_quant": true,
    "fused_attn": true,
    "fused_attn_min_rows": 0,
    "maxima_grid": 10240,
    "expert_placement": "placement.json",
    "block_transition": false,
    "mono_mode": "auto"
  },
  "dspark_options": {
    "fused_prologue": true,
    "two_buffer_attn": true,
    "lean_head": true,
    "verify_argmax_once": true
  },
  "seams": {"v41.attn.out": ["bmm8k-provider"]},
  "modules": {
    "layers.*.ffn": ["mega-moe-provider"],
    "layers.*.attn": ["my-whole-attention"]
  },
  "aiter": ["route-quant-provider"]
}
```

- `env` (v2 only): the native switches the deployment needs (`ATOM_*`, `AITER_*`, `FLYDSL_*`,
  `PYTORCH_*`, `HIP_*`, `HSA_*`, `GPU_*`, `NCCL_*`, `RCCL_*`, `TORCH_*`; never `ATOM_PLUGIN_*`),
  set once by the server's main process before any worker starts. A name the environment
  already sets to another value refuses the recipe (`recipe-env-conflict`). Part of the plan id;
  fixed for hot-plug.
- `plugins` is optional in v2: every owner named under `seams` / `modules` is a layer
  plugin, every name under `aiter` an AITER provider. A v1 recipe stays valid as is.
- Every option is off by default; an unknown option, or a wrong type, refuses the recipe.
  One strategy's recipe set (one file per capture bucket set) must agree on all options.
- Options are build-time: a hot-plug may change owners and `dspark_options`, never
  `model_options` or `engine` (`recipe-boot-only`). An A/B arm (`native`, `identity`,
  a named arm) changes routes only; the model is the same.

### Options that replace environment variables

| Recipe option | Replaces (PR) |
| --- | --- |
| `engine.prefill_short_first_rows` / `_long_rows` / `_max_skips` | `ATOM_PREFILL_SHORT_FIRST_*` (#3644) |
| `engine.prefill_cost_ceil` | `ATOM_PREFILL_COST_CEIL` (#3644) |
| `engine.prefill_decode_interval_busy` / `_seqs` | `ATOM_PREFILL_DECODE_INTERVAL_BUSY*` (#3644) |
| `engine.prefill_decode_interval_slope` / `_slope_from` | `ATOM_PREFILL_DECODE_INTERVAL_SLOPE` / `_SLOPE_FROM` |
| `engine.prefill_long_cap_busy` / `_seqs` | `ATOM_PREFILL_LONG_CAP_BUSY*` (#3644) |
| `engine.dp_session_prefix_owner_tokens` / `_max_extra_reqs` | `ATOM_DP_SESSION_PREFIX_OWNER_*` (#3644) |
| `engine.prefill_admit_order` / `prefill_admit_max_age_ms` | `ATOM_PREFILL_ADMIT_ORDER` / `ATOM_PREFILL_ADMIT_MAX_AGE_MS` |
| `engine.thp_disable` | `ATOM_THP_DISABLE` |
| `engine.engram_pin_pages` | `ATOM_ENGRAM_PIN_PAGES` |
| `model_options.ffn_mhc_fuse_quant` | `ATOM_DSV41_FFN_SEAM_QUANT` (#3645) |
| `model_options.shared_act_quant` | `ATOM_DSV41_SHARED_ACT_QUANT` (#3645) |
| `model_options.fused_attn` / `fused_attn_min_rows` | `ATOM_DSV41_FUSED_ATTN*` (#3645) |
| `model_options.maxima_grid` | `ATOM_DSV41_MAXIMA_GRID` (#3645) |
| `model_options.expert_placement` (path or `{layer: perm}`) | `ATOM_MOE_EXPERT_PLACEMENT` (#3645) |
| `model_options.block_transition` | `ATOM_V41_TRANSITION_AB` (#3529, still read) |
| `model_options.mono_mode: "off"` | skipping the mono installer |
| `dspark_options.fused_prologue` | `ATOM_DSPARK_FUSED_PROLOGUE` (#3639) |
| `dspark_options.two_buffer_attn` | `ATOM_V41_DRAFT_ATTN_NO_CAT` (#3645) |
| `dspark_options.lean_head` | `ATOM_DSPARK_LEAN_HEAD` (#3645) |
| `dspark_options.verify_argmax_once` | `ATOM_SPEC_VERIFY_ARGMAX_ONCE` (#3645) |

- `engine.prefill_decode_interval_slope` (a number, 0 = off) with `_slope_from`: after an
  executed prefill that leaves `g` sequences decoding in the DP group, the decode interval is
  `min(prefill_decode_interval_busy, ATOM_PREFILL_DECODE_INTERVAL + floor(slope * max(0, g - slope_from)))`.
  `prefill_decode_interval_busy` is then the ceiling (required) and `_busy_seqs` is not read.
- `engine.prefill_admit_order`: `"fifo"` (default: prompts mid-prefill, then new prompts as they
  arrived), `"short_new"` (new prompts with the fewest tokens left first) or `"srpt"` (fewest
  left first across both). A prompt that has gone `prefill_admit_max_age_ms` without a chunk
  goes ahead of every younger one. Refused together with `prefill_short_first_rows`.

- `engine.thp_disable`: no transparent huge pages for the server's processes
  (`PR_SET_THP_DISABLE`), set in the server's main process where the recipe's `env` is applied,
  before any worker starts, and again by each worker at its own start.
- `engine.engram_pin_pages`: the page-locked copy of an Engram shard is also registered as fixed
  buffers of an `io_uring` instance that does no I/O, which pins its pages against page
  migration, and marked `MADV_DONTFORK`. A refused pin warns and the copy stays as it was.

None of these are environment variables in this branch. The host keeps its own:
`ATOM_PLUGIN_HOST` (kill switch), `ATOM_PLUGIN_HOST_STRICT`, `ATOM_PLUGIN_HOTPLUG`,
`ATOM_PLUGIN_DEV`, `ATOM_PLUGIN_DEV_ROOTS`.

## Whole-module takeover

A `modules` slot is a module path of the built model (`*` matches one path segment;
the drafter's modules are under `draft.`). A plugin registers for it from its
`atom.layer_plugins` entry point:

```python
def register(host):
    def factory(module, capability):          # once per module, after the load
        w = module.wo_a.weight                 # the live module and its weights
        return lambda *args, **kwargs: ...     # called exactly as module.forward
    return host.register_module(
        slot="layers.*.attn", owner="my-whole-attention", factory=factory,
        select=lambda cap, *args, **kwargs: cap.rows <= 64,   # optional, pure
        graph="capture_safe",
    )
```

- Parameters, buffers and their names do not change: only the module's `forward` is
  routed, after the load and before warmup and capture (`plugin_host.install_modules`).
- Slots install deepest first. An outer owner that declines a call runs the module's
  own forward, which reaches the inner owners: overlapping (non-orthogonal) plugins
  compose by construction, neither aware of the other. Recipe v2 therefore defaults to
  `shadow_policy: "nested"`; v1 keeps `"error"`.
- Under `torch.compile` a tensor-only call is one opaque op (`atom_plugin::span_n`);
  pass `fake=` for a module whose output is not shaped like its first input.
- A slot that is exactly one seam's span is that seam: `layers.*.ffn` is
  `v41.block.ffn`, `layers.*.ffn.shared_experts` is `v41.moe.shared_expert`, `mono` is
  `v41.mono.install`. Either spelling plans either kind of owner.

## Seams of the plugin model

`v41.block.ffn`, `v41.block.transition`, `v41.attn.out`, `v41.attn.project_out`,
`v41.attn.out_full`, `v41.attn.attend_out`, `v41.attn.qk_norm`, `v41.attn.attend`,
`v41.attn.index_topk`, `v41.moe.shared_act`, `v41.moe.shared_expert`, `v41.moe.topk`,
`v41.op.quant_fp8_g32`, `v41.draft.ctx`, `v41.draft.markov`, `v41.spec.verify_sample`,
`v41.mono.install`. Unowned, each binds to its native (no Python on the hot path).

Spans that contain other spans yield to a finer owner: fused attention (#3645) runs
only where no plan owns `v41.attn.qk_norm` / `v41.attn.attend`; `v41.attn.out_full`
serves only where no plan owns `v41.attn.project_out`; the FFN mHC's MXFP8 pair
(#3645) reaches the shared expert whether ATOM or a `v41.block.ffn` owner runs the FFN,
the owner still seeing one BF16 tensor.

`register(..., owns_collectives=...)` is accepted from any plugin; no seam requires it.

## Compatibility

No plugin needs an ATOM file hash: the plugin model is versioned by HOOK_API and the
seam contracts, and a plugin qualifies a deployment from the `Capability` it is given
(dtypes, layouts such as the planar FP8 KV cache, the padded TP8 shared expert) and the
installed AITER / FlyDSL versions.
