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
| `engine.mori_defer_handles` | `ATOM_MORI_DEFER_HANDLES` |
| `engine.prefill_warmup_ladder` (path or row counts) | `ATOM_PREFILL_WARMUP_LADDER` |
| `engine.decode_warmup_widths` / `decode_warmup_tails` | `ATOM_DECODE_WARMUP_WIDTHS` / `ATOM_DECODE_WARMUP_TAILS` |
| `model_options.ffn_mhc_fuse_quant` | `ATOM_DSV41_FFN_SEAM_QUANT` (#3645) |
| `model_options.shared_act_quant` | `ATOM_DSV41_SHARED_ACT_QUANT` (#3645) |
| `model_options.fused_attn` / `fused_attn_min_rows` | `ATOM_DSV41_FUSED_ATTN*` (#3645) |
| `model_options.maxima_grid` | `ATOM_DSV41_MAXIMA_GRID` (#3645) |
| `model_options.expert_placement` (path or `{layer: perm}`) | `ATOM_MOE_EXPERT_PLACEMENT` (#3645) |
| `model_options.block_transition` | `ATOM_V41_TRANSITION_AB` (#3529, still read) |
| `model_options.seg_fuse` (items `1`, `7`) | `ATOM_DSV41_SEG_FUSE` |
| `model_options.attn_out_quant_fuse` | `ATOM_DSV41_ATTN_OUT_QUANT_FUSE` |
| `model_options.qpath_fuse` (items `"kv"`, `"q"`) | `ATOM_DSV41_QPATH_FUSE` |
| `model_options.mono_mode: "off"` | skipping the mono installer |
| `dspark_options.fused_prologue` | `ATOM_DSPARK_FUSED_PROLOGUE` (#3639) |
| `dspark_options.two_buffer_attn` | `ATOM_V41_DRAFT_ATTN_NO_CAT` (#3645) |
| `dspark_options.lean_head` | `ATOM_DSPARK_LEAN_HEAD` (#3645) |
| `dspark_options.verify_argmax_once` | `ATOM_SPEC_VERIFY_ARGMAX_ONCE` (#3645) |
| `dspark_options.dp_sharded_head` / `_max_rows` | `ATOM_DP_DSPARK_SHARDED_HEAD` / `_MAX_ROWS` |
| `dspark_options.sampler_skip_greedy_pick` | `ATOM_SAMPLER_SKIP_GREEDY_PICK` |
| `dspark_options.verify_skip_unused_target_draw` | `ATOM_SPEC_SKIP_UNUSED_TARGET_DRAW` |
| `dspark_options.sampler_split_independent_noise` | none (the draw as the tree those two come from makes it) |

- `engine.prefill_decode_interval_slope` (a number, 0 = off) with `_slope_from`: after an
  executed prefill that leaves `g` sequences decoding in the DP group, the decode interval is
  `min(prefill_decode_interval_busy, ATOM_PREFILL_DECODE_INTERVAL + floor(slope * max(0, g - slope_from)))`.
  `prefill_decode_interval_busy` is then the ceiling (required) and `_busy_seqs` is not read.
- `engine.prefill_admit_order`: `"fifo"` (default: prompts mid-prefill, then new prompts as they
  arrived), `"short_new"` (new prompts with the fewest tokens left first) or `"srpt"` (fewest
  left first across both). A prompt that has gone `prefill_admit_max_age_ms` without a chunk
  goes ahead of every younger one. Refused together with `prefill_short_first_rows`.

- `model_options.seg_fuse`: a list of items, each two back-to-back launches of the non-MoE
  decode segment as one with the same bytes out. `1`: the FFN seam's Mega-MHC also writes the
  MXFP8 pair of its normed output and the shared expert's `gate_up_proj` reads it; under an
  owner of `v41.block.ffn` the pair is held on the shared expert for the length of the call and
  used only for the very rows it was written with, so it is read whether the owner calls
  `single_stream_moe_forward` or the shared expert itself. `7`: the shared expert's
  `silu_and_mul` and the `quantize_fp8` in front of `w2` become AITER's `silu_mul_mxfp8`
  (gfx950, an AITER that has the op; the same span is the `v41.moe.shared_act` seam, so a
  plugin owner of that seam serves it on any AITER and the item stays off). Item 1 asks for the Mega-MHC launch `ffn_mhc_fuse_quant` asks for and item 7
  replaces the launches `shared_act_quant` replaces, by another kernel: a recipe that sets an
  item and that option is refused.
- `model_options.attn_out_quant_fuse`: a decode step's MXFP8 sparse attention (gfx950, 64
  heads of 512 in 8 groups, FP8 KV, not packed, all rows) is called through AITER's
  `mla_sparse_prefill_v41_inv_rope_quant`, which also writes the inverse-RoPE'd,
  group-quantized pair `wo_a` reads, so the separate `inverse_rope_group_quant` call goes
  away. It is the span of the `v41.attn.attend_out` seam, served natively: a planned owner
  of that seam, of `v41.attn.attend` or of `v41.attn.out_full` keeps its calls. The fused
  tail reads fp32 RoPE tables; the model's are BF16, so fp32 twins of them, value for value,
  are built with the model. Prefill, bounded replay, A8W4 and an AITER without the op are
  unchanged; on such an AITER a plugin owner of `v41.attn.attend_out` serves the span and
  the option stays off.
- `model_options.qpath_fuse`: a list of items that move the two halves of a decode step's
  RoPE / window launch (`rope_quant_window`) onto the launches that produce their inputs; with
  both, that launch is gone. `"kv"`: the q/kv norm launch also rotates, MXFP8-quantizes and
  stores each normed KV row into the window (`dual_rmsnorm_mxfp8_kv_window`). `"q"`: `wq_b`
  runs the tuned GEMM kernel of the row count with an epilogue that also writes the rotated
  MXFP8 query, through the `v41.attn.query_rope_quant` seam: a planned plugin owner serves
  it, else AITER's `gemm_a8w8_mxscale_bpreshuffle_rope_quant` where the AITER in use has
  it, else `wq_b` keeps its launch. Decode steps of the MXFP8 KV and
  window cache only (gfx950, 64 heads of 512, the native MXFP8 q/kv norm, all rows, no owner of
  `v41.attn.qk_norm`, not a step `fused_attn` serves); `wq_b` then runs ahead of the scorer.
  A half that is not fused runs the launch it has alone. The rotation is computed with explicit
  `fma` in one of two roundings, named per step from a record of what one reference build of
  the RoPE / window kernel compiles to (`qpath_fuse.REFERENCE_FORMS`). On that build the output
  is bit for bit the two launches'. The recorded kernel is
  `atom/model_ops/deepseek_v41/rope_window_reference.py`; `rope_window`'s own kernel is another
  source, so without `rope_window_reference` (below) the fused launches still compute the
  recorded forms, and a rotated value may differ in its last bit from the RoPE / window launch
  of the same process, about once in a million values. The log says which case a process is in.
- `model_options.rope_window_reference`: every RoPE / window launch (`rope_quant_window`: a decode
  step or half of one that `qpath_fuse` does not fuse, a prefill, bounded replay's query rows, the
  draft) runs the kernel of `rope_window_reference` with its layout rule (8 query heads a program
  where that leaves two programs a CU, else a program a head) instead of `rope_window`'s. It is
  the kernel `qpath_fuse`'s record was read from, so with both options the fused and the unfused
  steps of a deployment round alike, bit for bit. `fused_attn`'s own launch
  (`norm_rope_quant_window`) is not affected.
- `model_options.wq_b_before_index`: on every step `fused_attn` does not serve, `wq_b` is launched
  ahead of the index scorer instead of after it, as on the steps `qpath_fuse` plans. The same
  launches with the same inputs; only their order on the stream changes.
- `dspark_options.dp_sharded_head`: under pure DP attention (TP 1) the draft block's rows and
  anchors are all-gathered over the DP group, each rank projects every row onto its own
  `V / dp` slice of the replicated LM head, and each position's id is picked from the
  per-shard `(max, id)`. It serves a greedy block of at most `dp_sharded_head_max_rows` rows,
  ahead of `lean_head`; every other block takes the path it took.
- `dspark_options.sampler_skip_greedy_pick`: a sampled batch the host knows to hold no greedy
  row returns its draw without the argmax and select (on independent noise: the race alone
  instead of the fused race-and-override call). `verify_skip_unused_target_draw`: under forced
  acceptance only the noise of the target's own draw is drawn. Neither applies inside an
  owner of `v41.spec.verify_sample`, nor to an all-greedy verify (`verify_argmax_once`).
  `context_taps_concat`: the draft's context projection (`write_context_kv`) gets the taps as
  their BF16 concatenation, which `main_proj` quantizes itself (a copy and a `quantize_fp8`), in
  place of the one launch that quantizes the taps where they lie (`quantize_fp8_taps`); the same
  bytes. An owner of `v41.draft.ctx` is unaffected.
  `sampler_split_independent_noise`: a draw on independent noise is the race
  (`mixed_sample_outer_exponential`), then the greedy rows' argmax and the select, as three
  calls with the ids of the one fused call (`fused_mixed_sample`) they replace.
- `engine.thp_disable`: no transparent huge pages for the server's processes
  (`PR_SET_THP_DISABLE`), set in the server's main process where the recipe's `env` is applied,
  before any worker starts, and again by each worker at its own start.
- `engine.engram_pin_pages`: the page-locked copy of an Engram shard is also registered as fixed
  buffers of an `io_uring` instance that does no I/O, which pins its pages against page
  migration, and marked `MADV_DONTFORK`. A refused pin warns and the copy stays as it was.
- `engine.mori_defer_handles`: the standard MoE backend builds no MoRI EP op handle at model
  build; each is a stand-in that raises at first use. Only for a plan whose owners serve every
  FFN (`v41.block.ffn`) and never decline to the native path.
- `engine.prefill_warmup_ladder`: after the model warm-up, one dummy prefill forward per listed
  row count (a list of ints, or a path to a file of them, as the server process opens it),
  largest first, so each row-count bucket's kernels are first launched before serving.
- `engine.decode_warmup_widths` (`"capture"` for the declared capture sizes, or a list such as
  `"1-32,48,64"`): the draft's context projection (`write_context_kv` up to its window writes,
  `warm_context_projection` of the plugin draft model) once per row count a decode of those
  request widths, or the decoder tail of a prefill, hands it. `decode_warmup_tails` gives the
  tails' request counts: unset, the same widths; `"0"`, none.

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
Two more are single launches: `v41.block.mhc` and `v41.attn.query_rope_quant`.

`v41.block.mhc` is the HIP Mega-MHC launch of a sublayer seam (the arguments and the five
results of AITER's `mega_mhc`); an owner selects the calls it serves and the rest stay
AITER's. `v41.attn.query_rope_quant` is `model_options.qpath_fuse` item `"q"`'s launch;
the fused query path decides a step before the norm has run, so it asks the seam's owner
through `explain` with the inputs that do not exist yet (`x_fp8`, `x_scale`, `q_fp8`,
`q_scale`) as None, and an owner answers from the weight, its scale, the positions and
the tables.

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
