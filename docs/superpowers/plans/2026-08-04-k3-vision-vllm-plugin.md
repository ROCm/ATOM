# Kimi-K3 image support (vLLM plugin) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable image (vision) input for Kimi-K3 in ATOM vLLM plugin mode by mirroring K2.5's wiring, reusing vLLM's upstream K3 vision stack while keeping ATOM's KDA-optimized hybrid language model.

**Architecture:** Three-layer plugin structure. The **outer** class `KimiK3ForCausalLMVllm` (what vLLM's registry instantiates) becomes multimodal by switching its base to `ATOMForConditionalGeneration` while retaining `IsHybrid` + the `get_mamba_state_*` classmethods. A **new inner** class `KimiK3ForConditionalGeneration_` (built by ATOM as `.model`) subclasses vLLM's upstream `KimiK3ForConditionalGeneration` to inherit the vision path (`embed_multimodal`, media parsing, `MoonViT3dPretrainedModel`, `KimiK25MultiModalProjector`), but instantiates ATOM's plugin `KimiK3ForCausalLM` as its `language_model` and loads weights via ATOM's plugin loader. A config gate lets the plugin keep `vision_config` instead of stripping it.

**Tech Stack:** Python, PyTorch, vLLM (editable checkout at `/shared/amdgpu/home/gyu_qle/vllm`), AITER, ROCm. Tests: pytest (no GPU; real-import checks run via subprocess).

## Global Constraints

- **NEVER modify native K3 files** `atom/models/kimi_k3.py` or `atom/models/kimi_k3_dspark.py`. Import them read-only only. All new `embed_input_ids` etc. go on the **plugin** subclass in `atom/plugin/vllm/models/kimi_k3.py`.
- **Plugin mode only.** Do not touch native-mode registration `atom/model_engine/model_runner.py` (K3 stays text-only there).
- **Image only.** No video code paths.
- **Reuse the checkpoint's HF `KimiK3Config` directly.** Do NOT create `atom/model_config/kimi_k3.py`.
- **Do not modify `@support_torch_compile`-decorated model files.**
- Format/lint before every commit: `black . && ruff check .` (CI enforced; ruff only fails on PR-diff lines).
- Reference checkpoint (multimodal): `/workspace/shared/data/amd_int/models/Kimi-K3` — arch `KimiK3ForConditionalGeneration`, `model_type: kimi_k3`, `media_placeholder_token_id: 163605`, `image_placeholder: <|kimi_image_placeholder|>`. Weight prefixes: `language_model.*`, `vision_tower.*`, `mm_projector.{proj.0,proj.2,post_norm}`.
- Before restarting any server: `rm -rf /root/.cache/atom/*` and `export AITER_LOG_LEVEL=WARNING`.

---

### Task 1: Config gate — let plugin mode keep K3's vision_config

**Files:**
- Modify: `atom/config.py:610-614` (`_PLUGIN_SUPPORTED_MULTIMODAL_MODELS`)
- Test: `tests/plugin/test_vllm_kimi_k3.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `"kimi_k3"` present in `atom.config._PLUGIN_SUPPORTED_MULTIMODAL_MODELS`. This makes `get_hf_config()` skip the text-config stripping for `model_type == "kimi_k3"` under `is_vllm()`, so `hf_config` retains `.vision_config` + `.text_config`.

**Context:** In `atom/config.py::get_hf_config`, when `is_vllm()` is true, any `model_type` listed in `_PLUGIN_SUPPORTED_MULTIMODAL_MODELS` is filtered OUT of the `multimodal_model_types` stripping map (lines 630-636), so the full multimodal config is preserved. `kimi_k3` is already in `_MULTIMODAL_MODEL_TYPES` (line 602) but missing from the plugin set (lines 610-614).

- [ ] **Step 1: Write the failing test**

Add to `tests/plugin/test_vllm_kimi_k3.py`:

```python
def test_kimi_k3_is_plugin_supported_multimodal():
    from atom.config import (
        _MULTIMODAL_MODEL_TYPES,
        _PLUGIN_SUPPORTED_MULTIMODAL_MODELS,
    )

    # kimi_k3 must be a known multimodal type AND opted into plugin-mode
    # pass-through, so get_hf_config keeps vision_config instead of stripping
    # to text_config.
    assert "kimi_k3" in _MULTIMODAL_MODEL_TYPES
    assert "kimi_k3" in _PLUGIN_SUPPORTED_MULTIMODAL_MODELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py::test_kimi_k3_is_plugin_supported_multimodal -v`
Expected: FAIL — assertion error (`"kimi_k3"` not in `_PLUGIN_SUPPORTED_MULTIMODAL_MODELS`).

- [ ] **Step 3: Add kimi_k3 to the plugin-supported set**

In `atom/config.py`, change:

```python
_PLUGIN_SUPPORTED_MULTIMODAL_MODELS: set[str] = {
    "kimi_k25",
    "qwen3_5",
    "qwen3_5_moe",
}
```

to:

```python
_PLUGIN_SUPPORTED_MULTIMODAL_MODELS: set[str] = {
    "kimi_k25",
    "kimi_k3",
    "qwen3_5",
    "qwen3_5_moe",
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py::test_kimi_k3_is_plugin_supported_multimodal -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
black atom/config.py tests/plugin/test_vllm_kimi_k3.py && ruff check atom/config.py tests/plugin/test_vllm_kimi_k3.py
git add atom/config.py tests/plugin/test_vllm_kimi_k3.py
git commit -m "feat(k3): keep vision_config in plugin mode for kimi_k3"
```

---

### Task 2: Add the multimodal inner class + language-model embed hook

**Files:**
- Modify: `atom/plugin/vllm/models/kimi_k3.py` (add imports; add `embed_input_ids` to plugin `KimiK3ForCausalLM`; add new `KimiK3ForConditionalGeneration_`)
- Test: `tests/plugin/test_vllm_kimi_k3.py`

**Interfaces:**
- Consumes: `atom.config._PLUGIN_SUPPORTED_MULTIMODAL_MODELS` containing `"kimi_k3"` (Task 1); vLLM upstream `vllm.models.kimi_k3.KimiK3ForConditionalGeneration`, `vllm.model_executor.models.kimi_k25_vit.{MoonViT3dPretrainedModel,KimiK25MultiModalProjector}`, `vllm.models.kimi_k3.common.mm_preprocess.{KimiK3MultiModalProcessor,KimiK3ProcessingInfo,KimiK3DummyInputsBuilder}`.
- Produces:
  - `atom.plugin.vllm.models.kimi_k3.KimiK3ForConditionalGeneration_` — subclass of vLLM's `KimiK3ForConditionalGeneration`; `__init__(self, atom_config: Config, prefix: str = "model")`; attrs `vision_tower`, `mm_projector`, `language_model` (an ATOM plugin `KimiK3ForCausalLM`); classvar `hf_to_atom_mapper: WeightsMapper`; methods `load_weights(weights) -> set[str]`, `get_expert_mapping()`.
  - `KimiK3ForCausalLM.embed_input_ids(self, input_ids) -> torch.Tensor` (plugin subclass only).

**Context:** Mirror `atom/plugin/vllm/models/kimi_k25.py::KimiK25ForConditionalGeneration_` (lines 178-288). The vision-tower/projector construction must mirror `/shared/amdgpu/home/gyu_qle/vllm/vllm/models/kimi_k3/amd/model.py` lines 96-123 (only the `language_model` line differs — we use ATOM's, not `init_vllm_registered_model`). The weight mapper handles the double `language_model` nesting: the checkpoint stores text under `language_model.model.*`, but ATOM nests `inner.language_model` (an ATOM `KimiK3ForCausalLM`) `.language_model` (a `KimiLinearForCausalLM`) `.model.*`, so `language_model.` → `language_model.language_model.`.

- [ ] **Step 1: Write the failing test**

Add to `tests/plugin/test_vllm_kimi_k3.py`:

```python
def test_kimi_k3_inner_conditional_generation_class():
    _run_without_test_stubs("""
        from vllm.models.kimi_k3 import (
            KimiK3ForConditionalGeneration as vLLMKimiK3,
        )
        from atom.plugin.vllm.models.kimi_k3 import (
            KimiK3ForCausalLM,
            KimiK3ForConditionalGeneration_,
        )

        # Inner class subclasses the upstream multimodal model so it inherits
        # embed_multimodal / media parsing / vision-tower forward.
        assert issubclass(KimiK3ForConditionalGeneration_, vLLMKimiK3)

        # ATOM plugin loading + expert routing entry points exist.
        assert hasattr(KimiK3ForConditionalGeneration_, "load_weights")
        assert hasattr(KimiK3ForConditionalGeneration_, "get_expert_mapping")

        # Weight mapper collapses the double language_model nesting and renames
        # the projector layers.
        mapper = KimiK3ForConditionalGeneration_.hf_to_atom_mapper
        assert mapper.orig_to_new_prefix["language_model."] == (
            "language_model.language_model."
        )
        assert mapper.orig_to_new_prefix["mm_projector.proj.0"] == (
            "mm_projector.linear_1"
        )
        assert mapper.orig_to_new_prefix["mm_projector.proj.2"] == (
            "mm_projector.linear_2"
        )

        # Language-model wrapper exposes embed_input_ids for vLLM multimodal
        # discovery.
        assert hasattr(KimiK3ForCausalLM, "embed_input_ids")
        """)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py::test_kimi_k3_inner_conditional_generation_class -v`
Expected: FAIL — `ImportError: cannot import name 'KimiK3ForConditionalGeneration_'`.

- [ ] **Step 3: Add imports at the top of `atom/plugin/vllm/models/kimi_k3.py`**

Add after the existing imports (keep the existing ones):

```python
from torch import nn
from vllm.models.kimi_k3 import (
    KimiK3ForConditionalGeneration as vLLMKimiK3,
)
from vllm.models.kimi_k3.common.mm_preprocess import (
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.model_executor.models.vision import is_vit_use_data_parallel
from vllm.multimodal import MULTIMODAL_REGISTRY

from atom.config import Config, QuantizationConfig
from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.models.utils import maybe_prefix
```

(Note: `ATOMForConditionalGeneration` is imported in Task 3, where the outer class first uses it — importing it here would trip ruff's unused-import check at the Task 2 lint gate.)

- [ ] **Step 4: Add `embed_input_ids` to the plugin `KimiK3ForCausalLM`**

The class currently is (lines ~161-168):

```python
class KimiK3ForCausalLM(KimiK3ForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_kda_cls = kimi_k3_base.KimiKDAAttention
        kimi_k3_base.KimiKDAAttention = KimiKDAAttentionVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            kimi_k3_base.KimiKDAAttention = original_kda_cls
```

Add the method (delegates to the native `get_input_embeddings`; required by vLLM `SupportsMultiModal.get_language_model` discovery, matching K2.5's `KimiK25ForCausalLM.embed_input_ids`):

```python
class KimiK3ForCausalLM(KimiK3ForCausalLMBase):
    def __init__(self, *args, **kwargs):
        original_kda_cls = kimi_k3_base.KimiKDAAttention
        kimi_k3_base.KimiKDAAttention = KimiKDAAttentionVllm
        try:
            super().__init__(*args, **kwargs)
        finally:
            kimi_k3_base.KimiKDAAttention = original_kda_cls

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Required by vLLM SupportsMultiModal.get_language_model discovery.
        return self.get_input_embeddings(input_ids)
```

- [ ] **Step 5: Add the inner `KimiK3ForConditionalGeneration_` class**

Add ABOVE the existing `KimiK3ForCausalLMVllm` class (so the outer can reference it is not required — registry wiring happens in Task 3). Mirror `vllm/models/kimi_k3/amd/model.py` lines 96-123 for the vision setup:

```python
@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForConditionalGeneration_(vLLMKimiK3):
    hf_to_atom_mapper = WeightsMapper(
        orig_to_new_prefix={
            # ATOM nests language_model.language_model (ATOM KimiK3ForCausalLM
            # wraps KimiLinearForCausalLM); checkpoint stores language_model.model.
            "language_model.": "language_model.language_model.",
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    def __init__(self, atom_config: Config, prefix: str = "model"):
        # Protocols have no __init__; skip vLLMKimiK3.__init__ (it would build
        # vLLM's KimiLinear language model) and set up nn.Module directly.
        nn.Module.__init__(self)
        hf_config = getattr(atom_config, "hf_config", None)
        assert hf_config is not None, "hf_config is not found in atom_config"
        vision_config = hf_config.vision_config
        self.config = hf_config
        self.atom_config = atom_config

        vllm_config = atom_config.plugin_config.vllm_config
        quant_config = vllm_config.quant_config
        atom_quant_config = atom_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = is_vit_use_data_parallel(
            vision_config.num_attention_heads
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_tower = MoonViT3dPretrainedModel(
                vision_config,
                quant_config=self._maybe_ignore_quant_config(
                    quant_config,
                    atom_quant_config.exclude_layers or [],
                    "vision_tower",
                ),
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            self.mm_projector = KimiK25MultiModalProjector(
                config=vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=self._maybe_ignore_quant_config(
                    quant_config,
                    atom_quant_config.exclude_layers or [],
                    "mm_projector",
                ),
                prefix=maybe_prefix(prefix, "mm_projector"),
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = KimiK3ForCausalLM(
                atom_config=atom_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder = self.config.media_placeholder_token_id

    def _maybe_ignore_quant_config(
        self, quant_config, exclude_layers, layer_name: str
    ):
        for exclude_layer in exclude_layers:
            if QuantizationConfig._matches_exclude(
                layer_name, exclude_layer, check_contains=True
            ):
                return None
        return quant_config

    def load_weights(self, weights):
        # weights generator is discarded; ATOM loads from disk in plugin mode.
        # prefix="model." because ATOMModelBase constructs this as `.model`.
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="model.",
            weights_mapper=self.hf_to_atom_mapper,
        )

    def get_expert_mapping(self):
        return self.language_model.get_expert_mapping()
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py::test_kimi_k3_inner_conditional_generation_class -v`
Expected: PASS

- [ ] **Step 7: Run the full K3 plugin test module to confirm nothing else broke**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py -v`
Expected: All existing tests still pass (the registry-sync test is updated in Task 3, not yet).

- [ ] **Step 8: Commit**

```bash
black atom/plugin/vllm/models/kimi_k3.py tests/plugin/test_vllm_kimi_k3.py && ruff check atom/plugin/vllm/models/kimi_k3.py tests/plugin/test_vllm_kimi_k3.py
git add atom/plugin/vllm/models/kimi_k3.py tests/plugin/test_vllm_kimi_k3.py
git commit -m "feat(k3): add multimodal inner class reusing upstream vision stack"
```

---

### Task 3: Make the outer class multimodal + rewire the registry

**Files:**
- Modify: `atom/plugin/vllm/models/kimi_k3.py` (`KimiK3ForCausalLMVllm` base + methods)
- Modify: `atom/plugin/vllm/model_wrapper.py:158-160` (`_ATOM_MODEL_CLASSES["KimiK3ForConditionalGeneration"]`)
- Test: `tests/plugin/test_vllm_kimi_k3.py`

**Interfaces:**
- Consumes: `KimiK3ForConditionalGeneration_` (Task 2); `ATOMForConditionalGeneration` (from `atom.plugin.vllm.model_wrapper`); `IsHybrid` (already imported).
- Produces: `KimiK3ForCausalLMVllm` with base `(ATOMForConditionalGeneration, IsHybrid)`, `get_placeholder_str`, `load_weights`, and the retained `get_mamba_state_*` classmethods; `_ATOM_MODEL_CLASSES["KimiK3ForConditionalGeneration"]` pointing at `...kimi_k3:KimiK3ForConditionalGeneration_`.

**Context:** vLLM inspects the OUTER class (from `register.py`'s `_VLLM_MODEL_REGISTRY_OVERRIDES`, which already maps the arch to `KimiK3ForCausalLMVllm`) for interfaces. So `SupportsMultiModal` (via `ATOMForConditionalGeneration`) and `IsHybrid` must both live there. `ATOMForConditionalGeneration` also brings `SupportsMRoPE`, which stays dormant because K3's config has no `rope_scaling`/mrope. The existing registry-sync test must be updated to the new inner qualname.

- [ ] **Step 1: Write/adjust the failing tests**

In `tests/plugin/test_vllm_kimi_k3.py`, UPDATE the existing `test_kimi_k3_plugin_registries_are_synchronized` so the inner map points to the new class:

```python
def test_kimi_k3_plugin_registries_are_synchronized():
    from atom.plugin.vllm.model_wrapper import _ATOM_MODEL_CLASSES
    from atom.plugin.vllm.register import _VLLM_MODEL_REGISTRY_OVERRIDES

    arch = "KimiK3ForConditionalGeneration"
    assert (
        _VLLM_MODEL_REGISTRY_OVERRIDES[arch]
        == "atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLMVllm"
    )
    assert (
        _ATOM_MODEL_CLASSES[arch]
        == "atom.plugin.vllm.models.kimi_k3:KimiK3ForConditionalGeneration_"
    )
```

ADD a new structural test for the outer class:

```python
def test_kimi_k3_outer_is_multimodal_and_hybrid():
    _run_without_test_stubs("""
        from vllm.model_executor.models.interfaces import IsHybrid

        from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
        from atom.plugin.vllm.models.kimi_k3 import KimiK3ForCausalLMVllm

        mro = KimiK3ForCausalLMVllm.__mro__
        # Multimodal via ATOMForConditionalGeneration; hybrid state retained.
        assert ATOMForConditionalGeneration in mro
        assert IsHybrid in mro

        # Image placeholder matches the checkpoint token.
        assert (
            KimiK3ForCausalLMVllm.get_placeholder_str("image", 0)
            == "<|kimi_image_placeholder|>"
        )

        # Hybrid mamba-state entry points survive the base-class change.
        for name in (
            "get_mamba_state_dtype_from_config",
            "get_mamba_state_shape_from_config",
            "get_mamba_state_copy_func",
        ):
            assert hasattr(KimiK3ForCausalLMVllm, name)
        """)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py::test_kimi_k3_plugin_registries_are_synchronized tests/plugin/test_vllm_kimi_k3.py::test_kimi_k3_outer_is_multimodal_and_hybrid -v`
Expected: FAIL — registry test asserts old qualname; outer test fails (`ATOMForConditionalGeneration` not in MRO / no `get_placeholder_str`).

- [ ] **Step 3: Add the `ATOMForConditionalGeneration` import**

In `atom/plugin/vllm/models/kimi_k3.py`, add the import used by the new outer base:

```python
from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration
```

- [ ] **Step 4: Rewire the inner-class registry map**

In `atom/plugin/vllm/model_wrapper.py`, change the `_ATOM_MODEL_CLASSES` entry:

```python
    "KimiK3ForConditionalGeneration": (
        "atom.plugin.vllm.models.kimi_k3:KimiK3ForCausalLM"
    ),
```

to:

```python
    "KimiK3ForConditionalGeneration": (
        "atom.plugin.vllm.models.kimi_k3:KimiK3ForConditionalGeneration_"
    ),
```

- [ ] **Step 5: Make the outer class multimodal**

In `atom/plugin/vllm/models/kimi_k3.py`, change the `KimiK3ForCausalLMVllm` class definition. Current:

```python
class KimiK3ForCausalLMVllm(ATOMMoEForCausalLM, IsHybrid):
    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(vllm_config)
    ...
```

Change ONLY the base classes and ADD `get_placeholder_str` + `load_weights`; keep all three `get_mamba_state_*` classmethods exactly as-is. Also add the processor decorator (matches K2.5, which decorates the outer too):

```python
@MULTIMODAL_REGISTRY.register_processor(
    KimiK3MultiModalProcessor,
    info=KimiK3ProcessingInfo,
    dummy_inputs=KimiK3DummyInputsBuilder,
)
class KimiK3ForCausalLMVllm(ATOMForConditionalGeneration, IsHybrid):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|kimi_image_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def load_weights(self, weights):
        return self.model.load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return _get_k3_state_dtype(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return _get_k3_state_shape(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
```

- [ ] **Step 6: Remove the now-unused `ATOMMoEForCausalLM` import if nothing else uses it**

Run: `grep -n "ATOMMoEForCausalLM" atom/plugin/vllm/models/kimi_k3.py`
If the only remaining reference is the `from atom.plugin.vllm.model_wrapper import ATOMMoEForCausalLM` line, delete that import (ruff will flag it as unused otherwise). Keep it if still referenced.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/plugin/test_vllm_kimi_k3.py -v`
Expected: All pass (registry-sync updated, outer structural test passes, Task 2 test still passes).

- [ ] **Step 8: Commit**

```bash
black atom/plugin/vllm/models/kimi_k3.py atom/plugin/vllm/model_wrapper.py tests/plugin/test_vllm_kimi_k3.py && ruff check atom/plugin/vllm/models/kimi_k3.py atom/plugin/vllm/model_wrapper.py tests/plugin/test_vllm_kimi_k3.py
git add atom/plugin/vllm/models/kimi_k3.py atom/plugin/vllm/model_wrapper.py tests/plugin/test_vllm_kimi_k3.py
git commit -m "feat(k3): wire multimodal outer class + registry for image support"
```

---

### Task 4: GPU validation — weight-load completeness, image smoke, text-accuracy parity

**Files:**
- No source changes expected. If validation surfaces a weight-mapping mismatch, fix `hf_to_atom_mapper` in `atom/plugin/vllm/models/kimi_k3.py` (adjust prefixes to match the "unexpected/missing weights" error), re-run, and commit as a fix.

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: a validated K3 image path (recorded in the PR description / a note under `k3_traces/` or the memory file).

**Context:** This task cannot be a pytest unit test — it needs the GPU + the multimodal checkpoint. Use the `/run-atom-workload` skill for the canonical stop→start→workload→drain→stop flow. The K3 checkpoint is `/workspace/shared/data/amd_int/models/Kimi-K3`. Weight-load completeness is the real gate on the mapper: vLLM raises on unexpected/missing weights at load time.

- [ ] **Step 1: Clean caches and start the server in plugin mode**

```bash
rm -rf /root/.cache/atom/*
export AITER_LOG_LEVEL=WARNING
```

Start the server via the `/run-atom-workload` skill (plugin/vLLM mode) against `/workspace/shared/data/amd_int/models/Kimi-K3`. After startup, confirm the model is actually resident on GPU (not just HTTP 200):

Run: `rocm-smi --showmemuse`
Expected: VRAM% > 0 on the TP GPUs; server log shows the vision tower + language model loaded with **no "unexpected weights" / "missing weights" errors**. If such errors appear, capture the offending names and adjust `hf_to_atom_mapper` prefixes, then restart from Step 1.

- [ ] **Step 2: Image smoke test (describe an image)**

Send one OpenAI-compatible chat request with an image URL/base64 + "Describe this image." (use the `/run-atom-workload` client or a `curl` to `/v1/chat/completions`).
Expected: a coherent, image-relevant description (not gibberish, not a text-only hallucination). This confirms `embed_multimodal` → vision tower → projector → merge → KDA/MLA/MoE forward all compose.

- [ ] **Step 3: Text-only request still works**

Send a plain text chat request (no image) to the same server.
Expected: a normal coherent completion — confirms `embed_multimodal` returns `None` and the text path is unchanged.

- [ ] **Step 4: Text-accuracy parity (GSM8K)**

Run GSM8K via `/run-atom-workload` (lm_eval) against this server and compare to the current text-only K3 baseline on this branch.
Expected: accuracy within normal run-to-run variance of the text-only baseline — proves the KDA/dual-stream/hybrid-state path did not regress from the rewiring.

- [ ] **Step 5: Record results**

Note the smoke output, GSM8K score vs baseline, and any mapper fix in the PR description and update the memory file `k3-vision-vllm-plugin.md` with the GPU-validated status.

- [ ] **Step 6: Commit any fixes**

```bash
# only if Step 1 required a mapper fix
black atom/plugin/vllm/models/kimi_k3.py && ruff check atom/plugin/vllm/models/kimi_k3.py
git add atom/plugin/vllm/models/kimi_k3.py
git commit -m "fix(k3): correct vision/language weight prefix mapping"
```

---

## Self-Review

**Spec coverage:**
- Config gate (`_PLUGIN_SUPPORTED_MULTIMODAL_MODELS`) → Task 1. ✓
- New inner `KimiK3ForConditionalGeneration_` (subclass upstream, reuse vision, swap ATOM LM, ATOM weight loading) → Task 2. ✓
- `embed_input_ids` on plugin LM (not native) → Task 2. ✓
- Outer class → `ATOMForConditionalGeneration, IsHybrid` + `get_placeholder_str` + `load_weights` + retained `get_mamba_state_*` → Task 3. ✓
- Registry rewire (`_ATOM_MODEL_CLASSES`; `register.py` unchanged) → Task 3. ✓
- No new config wrapper; native files untouched; plugin/image only → enforced in Global Constraints and each task's file list. ✓
- Validation: image smoke + text-accuracy-unchanged → Task 4. ✓
- Risk: multimodal+hybrid composition, dormant MRoPE, weight mapping, VRAM → addressed in Tasks 3/4 and constraints. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete code; the one runtime-verified detail (weight mapper) has a concrete starting value derived from the checkpoint layout plus an explicit adjust-on-error procedure in Task 4. ✓

**Type/name consistency:** `KimiK3ForConditionalGeneration_`, `KimiK3ForCausalLMVllm`, `KimiK3ForCausalLM`, `_ATOM_MODEL_CLASSES`, `hf_to_atom_mapper`, `get_placeholder_str`, `get_mamba_state_*` used consistently across Tasks 2-3 and match the existing code in `atom/plugin/vllm/models/kimi_k3.py` and `model_wrapper.py`. ✓
