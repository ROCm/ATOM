# Online quantization best practices

This guide helps you choose an `--online_quant_config` when you do not have a
quantization background. It starts from a conservative configuration and expands
coverage one step at a time, so that every accuracy regression can be traced to
the single change that caused it.

For the flag syntax, the list of supported target formats, the pattern
resolution order, and how to read the `online_quant_info_*.json` dump, see the
[online quantization guide](./online_quantization_guide.md). This document only
covers *which* configuration to pick and in what order to try them.

> **Scope**
>
> - **Hardware:** the examples below use the AMD Instinct MI300 and MI350/MI355
>   series as concrete cases. The methodology is not hardware-specific — on a
>   different GPU, keep the same layer-selection and step-by-step coverage
>   strategy and only adjust the *format* to what the hardware accelerates.
> - **Beyond load-time quantization:** the reasoning about which layers tolerate
>   quantization and which do not applies equally when you are searching for an
>   offline recipe with Quark, so the conservative-to-aggressive progression
>   below carries over directly. Load-time quantization is plain
>   round-to-nearest, so it is the cheaper way to find the layer assignment; once
>   you have settled on one, Quark can push the same recipe further with
>   accuracy-recovery techniques such as rotation and SmoothQuant. See
>   [Online vs. offline quantization](./online_quantization_guide.md#online-vs-offline-quantization)
>   for when to make that switch.

## Recap of the three fields

```bash
--online_quant_config '{
  "global_quant_config": "ptpc_fp8",
  "layer_quant_config": {"*expert*": "mxfp4"},
  "exclude_layer": ["lm_head", "*.gate.*"]
}'
```

- `global_quant_config` — the default target format for every Linear and fused
  MoE module.
- `layer_quant_config` — per-layer overrides, keyed by glob patterns matched
  against the fully-qualified module name from `model.named_modules()`.
- `exclude_layer` — patterns left at source precision.

Under the [vLLM plugin backend](./vllm_plugin_backend_guide.md) the same object
is passed as `--additional-config '{"online_quant_config": {...}}'`; every
recommendation below applies unchanged.

### What "excluded" means

`exclude_layer` leaves a layer at the **source checkpoint's** precision, which
is not necessarily BF16:

| Source checkpoint | An excluded layer stays in |
|---|---|
| Unquantized (BF16/FP16) | BF16/FP16 |
| Block FP8 (e.g. DeepSeek-R1-0528) | Block FP8 |

This matters when reading the recommendations below. "Keep attention out of
quantization" means "do not apply a *new* target format to it" — on an FP8
checkpoint that layer is still FP8, not floating point. If you need a layer in
BF16 you have to start from a BF16 checkpoint.

## Choosing a format

`ptpc_fp8` is the conservative choice: FP8 E4M3 with per-channel static weight
scales and dynamic activation scales. Its dynamic range is much wider than
`mxfp4`, so it is the right default whenever you are unsure.

`mxfp4` is the aggressive target: OCP MXFP4, group size 32, E8M0 block scales.
It fits MoE experts well, because that is where most of the parameters and
memory bandwidth go. It saves the most memory but is the most likely to hurt
accuracy on sensitive paths such as attention and shared experts.

`per_block_fp8` (DeepSeek-style 128×128 block scale) is a third option and sits
between the two in granularity. It is a good drop-in when the source checkpoint
already ships in block FP8 and you want to stay close to its numerics.

### Hardware support

`mxfp4` needs native MXFP4 acceleration, which is available on CDNA 4 GPUs such
as the AMD Instinct MI350/MI355. **The MI300 series (CDNA 3) does not accelerate
MXFP4.**

On MI300, skip every `mxfp4` step below — but keep the *layer selection* from
those steps. The decisions about what to exclude (`lm_head`, router/gate, shared
experts, attention, vision modules) are independent of the format. Read the
levels below, keep their `exclude_layer` and `layer_quant_config` layer choices,
and express the whole recipe in `ptpc_fp8`.

## What to exclude by default

### `lm_head`

`lm_head` maps hidden states straight to vocabulary logits, so quantization
error there can reorder tokens. That shows up as degraded generation quality,
unstable formatting, and changed reasoning paths. The parameter savings are not
worth the risk:

```json
"exclude_layer": ["lm_head"]
```

### Router and gate layers

Here "gate" means the router, gating, or control-path module — **not** the
regular MLP `gate_proj`. These modules decide where information flows. They are
small, so quantizing them saves almost nothing, but an error is amplified by
everything downstream: in an MoE model a small router error changes the top-k
expert assignment for a token, which sends it through entirely different weights.

```json
"exclude_layer": ["lm_head", "*.gate.*"]
```

Avoid a broad `*gate*` unless you have checked that it does not also catch
`gate_proj`, which is an ordinary FFN Linear and is usually safe to quantize.
Replace `*.gate.*` with the exact router prefix your model uses.

### Vision modules in VLMs

Do not apply a text-only policy to a vision-language model. The vision tower,
patch embedding, and multi-modal projector see very different input
distributions, and errors there may not move text perplexity at all while
clearly degrading image-text understanding.

```json
"exclude_layer": [
  "lm_head",
  "*.gate.*",
  "*attn*",
  "<vision_encoder_pattern>",
  "<multi_modal_projector_pattern>"
]
```

Replace the placeholders with the actual module prefixes for your model's vision
tower, patch embedding, and image/video projector. Only consider quantizing them
after the language decoder is known to be healthy, and validate them on
image-text tasks rather than text-only metrics.

## MoE models

MoE models need the most care. Almost all the parameters are in the experts,
while almost all the accuracy risk is in routing, shared experts, and attention.
Attention is a small share of an MoE model's parameters and compute, so
quantizing it buys little and can cost a lot.

### Where to start

Pick an entry point based on what you care about, rather than always starting in
the same place:

- **Accuracy-sensitive:** start at Level 0 — `ptpc_fp8` on non-attention Linear
  layers only, with router/gate, shared experts, and attention excluded. This is
  the lowest-risk configuration.
- **Latency-sensitive:** start at Level 4 — global `mxfp4`, excluding only
  `lm_head` and the router/gate. This gives the largest memory and bandwidth
  saving and needs MI350/MI355-class hardware. On MI300, use global `ptpc_fp8`
  with the same exclusions.
- **Tuning back down:** if an aggressive configuration fails evaluation, walk the
  levels backward one step at a time. Each step moves the most sensitive
  remaining path — attention first, then shared experts, then regular experts —
  to a safer format.

The levels are ordered from most conservative to most aggressive. Read top-down
if you are starting conservative, bottom-up if you are stepping down from an
aggressive configuration.

### Which MoE components are sensitive

**Router / gate.** Excluded by default, because it decides which experts see
each token. Even a small error can change expert selection, and then the token
is processed by different weights entirely. The benefit is negligible.

**Shared experts.** Exclude them in the first stage. Shared experts serve all or
most tokens, so they behave more like a global FFN or a residual path than like
regular experts, and they are correspondingly more sensitive to distribution
shift. Quantizing them early tends to degrade quality broadly. Start with the
regular experts.

**Attention.** Handle conservatively. Attention mixes information across tokens,
and long-context or reasoning models are especially sensitive to it. Keep it out
of quantization first; if you later want to include it, use `ptpc_fp8` rather
than `mxfp4`.

Note that `*expert*` matches shared experts too (`...mlp.shared_experts` as well
as `...mlp.experts`). Because `exclude_layer` is evaluated before
`layer_quant_config`, listing `*shared_expert*` in `exclude_layer` is what keeps
them out, even when `*expert*` also appears as an override.

### Level 0 — non-attention Linear layers in `ptpc_fp8`

```bash
--online_quant_config '{
  "global_quant_config": "ptpc_fp8",
  "exclude_layer": ["lm_head", "*.gate.*", "*shared_expert*", "*attn*"]
}'
```

This is the MoE accuracy baseline. Attention is deliberately left alone: small
benefit, large risk.

If even this fails evaluation, do not move on to `mxfp4`. Check the
`online_quant_info_*.json` dump first — a typo in a pattern that sends every
layer into `exclude_layer` (or fails to exclude the router) is far more likely
than a genuine `ptpc_fp8` accuracy problem. Verify the layer count and the
per-layer formats as described in the
[online quantization guide](./online_quantization_guide.md#verifying-the-result).

### Level 1 — regular experts in `mxfp4`

```bash
--online_quant_config '{
  "global_quant_config": "mxfp4",
  "exclude_layer": ["lm_head", "*.gate.*", "*shared_expert*", "*attn*"]
}'
```

The first aggressive step targets regular experts only; shared experts and
attention stay excluded. The point of this level is to establish whether the
regular experts tolerate `mxfp4` at all.

### Level 2 — all MoE experts in `mxfp4`

```bash
--online_quant_config '{
  "global_quant_config": "mxfp4",
  "exclude_layer": ["lm_head", "*.gate.*", "*attn*"]
}'
```

Once regular experts pass, bring the shared experts in. Attention is still
excluded, so any regression is isolated to the MoE blocks.

### Level 3 — attention in `ptpc_fp8`

```bash
--online_quant_config '{
  "global_quant_config": "ptpc_fp8",
  "layer_quant_config": {"*expert*": "mxfp4"},
  "exclude_layer": ["lm_head", "*.gate.*"]
}'
```

If all experts pass in `mxfp4` and you still want to reduce the unquantized
footprint, move attention in — at `ptpc_fp8`, not `mxfp4`.

### Level 4 — global `mxfp4`

```bash
--online_quant_config '{
  "global_quant_config": "mxfp4",
  "exclude_layer": ["lm_head", "*.gate.*"]
}'
```

Attention is now in `mxfp4` as well. Treat this as the highest-risk
configuration and validate it carefully before deploying.

## Dense models

Without expert routing the progression is much shorter:

1. Global `ptpc_fp8`, excluding `lm_head`.
2. If accuracy holds, try global `mxfp4`.
3. If global `mxfp4` regresses, override the model's attention pattern back to
   `ptpc_fp8`.

Safe starting point:

```bash
--online_quant_config '{
  "global_quant_config": "ptpc_fp8",
  "exclude_layer": ["lm_head"]
}'
```

More aggressive:

```bash
--online_quant_config '{
  "global_quant_config": "mxfp4",
  "exclude_layer": ["lm_head"]
}'
```

Step 3, expressed as an override:

```bash
--online_quant_config '{
  "global_quant_config": "mxfp4",
  "layer_quant_config": {"*attn*": "ptpc_fp8"},
  "exclude_layer": ["lm_head"]
}'
```

## Vision-language models

Quantize the language decoder first and keep the vision side out. Follow the
dense or MoE progression above for the language part:

```bash
--online_quant_config '{
  "global_quant_config": "ptpc_fp8",
  "exclude_layer": [
    "lm_head",
    "<vision_encoder_pattern>",
    "<multi_modal_projector_pattern>"
  ]
}'
```

Replace the placeholders with the model's actual module prefixes. Text
perplexity alone cannot tell you whether the vision modules are safe to
quantize, so if you want to include them later, validate that step separately on
image-text tasks.

## Validating each step

Every level above is a hypothesis, so confirm two things before moving to the
next one:

1. **The configuration did what you meant.** Read
   `online_quant_info_*.json` and check the layer count and the per-layer
   `quant_type` / `quant_dtype` against your intent. Most surprising accuracy
   results turn out to be a pattern that matched more or less than expected.
2. **Accuracy held on a task you trust.** Compare against the previous level
   rather than against a published number, and use enough samples that the
   difference you care about is larger than the run-to-run variation of the
   harness.
