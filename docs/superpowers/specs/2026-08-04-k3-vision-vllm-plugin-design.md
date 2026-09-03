# Design: Kimi-K3 image support in the ATOM vLLM plugin

**Date:** 2026-08-04
**Branch:** guanbao/k3_rmsnorm_quant_fusion
**Status:** Approved design — pending implementation plan

## Goal

Add vision (image) support to Kimi-K3 when running in **ATOM vLLM plugin mode**, by
mirroring how Kimi-K2.5 achieves it. K3 currently runs text-only in the plugin even
though the checkpoint is multimodal.

## Scope

- **In scope:** Image input, plugin mode only, always-multimodal (vision tower always
  loads, K2.5 pattern).
- **Out of scope:** Video (no upstream K3 reference; the K3 checkpoint has no video
  tokens). Native ATOM (non-plugin) mode stays text-only.
- **Hard constraint:** Never modify native K3 files `atom/models/kimi_k3.py` or
  `atom/models/kimi_k3_dspark.py`. They may only be imported read-only.

## Background / current state

- The K3 checkpoint (`/workspace/shared/data/amd_int/models/Kimi-K3`) is already
  multimodal: architecture `KimiK3ForConditionalGeneration`, `model_type: kimi_k3`,
  with `vision_config` + `text_config` + `image_placeholder` +
  `media_placeholder_token_id`. The **same** architecture name is used for text and
  vision checkpoints (as with K2.5).
- vLLM upstream already ships the full K3 **image** stack:
  - `vllm/models/kimi_k3/amd/model.py::KimiK3ForConditionalGeneration` — reuses K2.5's
    `MoonViT3dPretrainedModel` + `KimiK25MultiModalProjector`; combines
    `SupportsMultiModal` + `IsHybrid` + `HasInnerState` + `SupportsPP/Quant/Eagle3`.
  - `vllm/models/kimi_k3/common/mm_preprocess.py` — `KimiK3ProcessingInfo`,
    `KimiK3MultiModalProcessor`, `KimiK3DummyInputsBuilder` (image-only). Placeholder
    token `<|kimi_image_placeholder|>`.
- ATOM forces K3 down a text-only path today:
  - `kimi_k3` is absent from `_PLUGIN_SUPPORTED_MULTIMODAL_MODELS` in `atom/config.py`,
    so the plugin strips `vision_config`.
  - The plugin outer class `KimiK3ForCausalLMVllm` inherits `ATOMMoEForCausalLM,
    IsHybrid` (text-only) rather than the multimodal `ATOMForConditionalGeneration`.
- The reference implementation for the plugin wiring is
  `atom/plugin/vllm/models/kimi_k25.py` (inner `KimiK25ForConditionalGeneration_` +
  outer `KimiK25ForConditionalGeneration`).

**Novel wrinkle vs K2.5:** K3 is a *hybrid* model (KDA recurrent + MLA). Its outer
class must be multimodal **and** retain the hybrid-state plumbing (`IsHybrid` +
`get_mamba_state_*`). K2.5 is pure-attention and did not need this. vLLM's own upstream
K3 class proves the two interface sets compose.

## Architecture

Three layers, mirroring K2.5, with hybrid-state additions on the outer class:

```
register.py:  "KimiK3ForConditionalGeneration" ─► KimiK3ForCausalLMVllm            (OUTER, modified)
                                                   = ATOMForConditionalGeneration + IsHybrid
                                                     · get_mamba_state_* (kept)
                                                     · get_placeholder_str (new)
                                                     · load_weights → self.model (new)
                                                          │ .model =
model_wrapper._ATOM_MODEL_ARCH_TO_QUALNAME ───────► KimiK3ForConditionalGeneration_ (INNER, new)
                                                   = subclass of vLLM KimiK3ForConditionalGeneration
                                                     · vision_tower  = MoonViT3dPretrainedModel   (upstream)
                                                     · mm_projector  = KimiK25MultiModalProjector (upstream)
                                                     · embed_multimodal / media parse (inherited)
                                                     · @register_processor(KimiK3MultiModalProcessor…)
                                                          │ .language_model =
                                                   KimiK3ForCausalLM (ATOM PLUGIN subclass, KDA-monkeypatched)
                                                     = existing optimized text path (dual-stream / KDA / fusion)
```

Key insight: vLLM inspects the **outer** class for interfaces, so `IsHybrid` and
`SupportsMultiModal` (via `ATOMForConditionalGeneration`) both live there. The
**inner** class owns the vision tower and delegates text to ATOM's KDA language model.
This keeps ATOM's entire K3 text-path optimization intact while the image stack is
reused verbatim from upstream.

## Detailed changes

All edits are confined to three files. **No native K3 file is touched.**

### 1. `atom/config.py`
Add `"kimi_k3"` to `_PLUGIN_SUPPORTED_MULTIMODAL_MODELS` so the plugin passes the full
`hf_config` (including `vision_config`) through instead of stripping to `text_config`.

### 2. `atom/plugin/vllm/models/kimi_k3.py`

**New inner class** `KimiK3ForConditionalGeneration_`:
- Subclasses vLLM's `KimiK3ForConditionalGeneration` (from `vllm.models.kimi_k3`,
  resolves to the AMD branch on ROCm) to inherit `embed_multimodal`,
  `_parse_and_validate_media_input`, `_process_media_input`, `_mark_tower_model`,
  `_mark_language_model`.
- Decorated with
  `@MULTIMODAL_REGISTRY.register_processor(KimiK3MultiModalProcessor,
  info=KimiK3ProcessingInfo, dummy_inputs=KimiK3DummyInputsBuilder)`.
- `__init__(self, atom_config, prefix="model")`:
  - `nn.Module.__init__(self)`.
  - Read `hf_config.vision_config` / `hf_config.text_config` directly — reuse the
    checkpoint's HF `KimiK3Config`; do **not** create a new config wrapper (decided).
  - Build `vision_tower` (`MoonViT3dPretrainedModel`) + `mm_projector`
    (`KimiK25MultiModalProjector`) under `self._mark_tower_model(vllm_config, "image")`,
    handling the same quant-exclude logic as K2.5's `_maybe_ignore_quant_config`.
  - Build the ATOM **plugin** `KimiK3ForCausalLM` as `self.language_model` under
    `self._mark_language_model(vllm_config)` (this triggers the existing KDA →
    `KimiKDAAttentionVllm` self-registration in the static forward context).
  - `self.make_empty_intermediate_tensors =
    self.language_model.make_empty_intermediate_tensors`.
- `load_weights`: `load_model_in_plugin_mode(model=self, config=self.atom_config,
  prefix="model.", weights_mapper=self.hf_to_atom_mapper)` with a `WeightsMapper`
  covering vision prefixes, `mm_projector.proj.{0,2}` → `linear_{1,2}`, and
  `language_model.` remaps (patterned on K2.5's mapper, adjusted to K3's checkpoint key
  layout — verified during implementation against the on-disk weight names).
- `get_expert_mapping` delegating to `self.language_model`.

**Add `embed_input_ids`** to the existing plugin `KimiK3ForCausalLM` subclass (NOT the
native class) — returns `self.get_input_embeddings(input_ids)`. Required by vLLM's
`SupportsMultiModal.get_language_model` discovery (K2.5 needed the same).

**Modify outer** `KimiK3ForCausalLMVllm`:
- Change base from `(ATOMMoEForCausalLM, IsHybrid)` to
  `(ATOMForConditionalGeneration, IsHybrid)`.
- Keep the three existing `get_mamba_state_*` classmethods unchanged.
- Add `get_placeholder_str(cls, modality, i)` → `"<|kimi_image_placeholder|>"` for
  `image`, raise for others.
- Add `load_weights` delegating to `self.model.load_weights`.

### 3. `atom/plugin/vllm/model_wrapper.py`
Repoint `_ATOM_MODEL_ARCH_TO_QUALNAME["KimiK3ForConditionalGeneration"]` from
`atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLM` to
`atom.plugin.vllm.models.kimi_k3:KimiK3ForConditionalGeneration_`.

### Unchanged
- `atom/plugin/vllm/register.py` — already maps the arch to the outer
  `KimiK3ForCausalLMVllm`; keep as-is.
- `atom/model_engine/model_runner.py` — native mode stays text-only (out of scope).
- `atom/models/kimi_k3.py`, `atom/models/kimi_k3_dspark.py` — never touched.

## Data flow (image request)

1. `KimiK3MultiModalProcessor` (upstream) converts image + `<|kimi_image_placeholder|>`
   prompt into `pixel_values` + `grid_thws`, expanding the placeholder into
   `media_begin … media_end` pad tokens sized to the image resolution.
2. vLLM calls `embed_multimodal` (inherited) → `vision_tower` + `mm_projector` → image
   feature tensors.
3. `embed_input_ids` (vLLM `SupportsMultiModal` default merge) embeds text tokens and
   scatters image features at `media_placeholder_token_id` positions.
4. Merged embeddings enter ATOM's `KimiK3ForCausalLM.forward` (KDA + MLA + MoE,
   dual-stream) exactly as today; hybrid recurrent state via the outer class's
   `IsHybrid` / `get_mamba_state_*`.
5. Text-only request → `embed_multimodal` returns `None`; path is identical to the
   current text-only K3.

## Risks & mitigations

- **Multimodal + hybrid interface composition** (main novelty). Mitigated: vLLM's own
  upstream K3 class combines the same interface sets, and we keep the hybrid interfaces
  on the outer class (which vLLM inspects) exactly as the working text path does.
- **Inherited `HasInnerState` / `mamba_cache` methods** on the inner class reference a
  `mamba_cache` ATOM does not use. Mitigated: vLLM inspects the *outer* class (which
  does not declare `HasInnerState`), so these stay dormant. If they are ever invoked,
  fall back to a standalone inner class (no upstream subclassing) for the vision path.
- **Weight-name mapping** — vision/projector/language_model prefixes must all resolve.
  Verified during implementation against on-disk weight names and by the smoke test's
  load-completeness check (no unexpected/missing weights).
- **VRAM / startup** — the ViT now always loads (accepted). May shift memory headroom
  for text-only benchmarks on this branch; note in the PR.

## Validation

- **Smoke test:** start the server with the K3 checkpoint in plugin mode, send an
  image + text prompt, confirm a coherent description and clean weight loading
  (`rocm-smi` VRAM > 0, no unmapped weights). Use `/run-atom-workload`.
- **Text accuracy unchanged:** run GSM8K (or the usual text eval) and confirm parity
  with the current text-only K3 — proves the KDA / dual-stream path did not regress.

## Open questions

None. Config wrapper skipped (reuse HF `KimiK3Config`); native mode out of scope; video
out of scope.
