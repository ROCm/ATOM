# [Attention] Refactor ATOM-vLLM/ATOM-SGLang Attention

## 1. Background and Motivation

ATOM now supports several runtime modes:

- ATOM native runtime
- ATOM as a vLLM plugin
- ATOM as an SGLang plugin

Before this refactor, these modes shared too much attention code. The same classes were changed at import time by decorators and monkey patches. For example, native ATOM attention backends and impl classes were patched to also satisfy vLLM interfaces.

This created several problems.

1. The code was hard to understand. A class in `atom/model_ops` could behave differently depending on whether the current mode was native, vLLM, or SGLang. The real methods on a class were not always visible in the source file because of many decorators.
2. The vLLM forced strong interface constraints on ATOM-vLLM attention. In the old design, ATOM-vLLM constructed vLLM standard `Attention` or `MLAAttention`. Because of that, ATOM code had to follow vLLM attention layer details, including backend selection, metadata builder shape, KV cache spec, output buffer behavior, and forward context behavior.
3. The `atom/plugin` folder mixed many unrelated concerns. vLLM-only attention code, SGLang-only code, shared plugin code, and native attention helper code were placed together. This made future changes risky.
4. The native ATOM attention and ATOM-vLLM attention were coupled at multiple levels:
frontend `Attention`
attention layer
attention backend
metadata builder
attention impl

This made it easy for a vLLM change to affect native ATOM, or for a native change to affect vLLM.
The goal of this refactor is to make each mode own its attention path clearly.

## 2. Refactor Principle

The main principle is **isolation**.

Each runtime mode should have its own attention layer, backend, metadata builder, and implementation path.

After the refactor:

- Native ATOM attention should not know about vLLM.
- ATOM-vLLM attention should not reuse native attention classes through decorators.
- ATOM-SGLang attention should not depend on vLLM attention code.
- Model code should still use a single frontend `Attention` API.

The new rule is:

```text
Model code imports Attention.
Attention dispatches by runtime mode.
Each mode owns the real attention implementation.
```

In other words:

```text
ATOM native  -> atom.model_ops.paged_attention.Attention
ATOM-vLLM    -> atom.plugin.vllm.attention.AttentionForVllm*
ATOM-SGLang  -> atom.plugin.sglang.attention.AttentionForSGLang
```

This removes the need to mutate global `atom.model_ops.Attention` at runtime.

It also removes the need to patch native attention classes for vLLM.

For ATOM-vLLM, the new design does not construct vLLM standard `Attention` or `MLAAttention`. Instead, ATOM-vLLM defines its own attention layers that implement vLLM's `AttentionLayerBase` contract directly.

This keeps vLLM integration at the adapter layer, not inside native ATOM attention code.

## 3. Concrete Implementation

### 3.1 Frontend Attention Dispatch

`atom.model_ops.base_attention.Attention` is now the frontend constructor.

It checks the current runtime mode and dispatches to the correct attention implementation:

```python
class Attention:
    def __new__(cls, *args, **kwargs):
        from atom.plugin.prepare import is_sglang, is_vllm

        if is_vllm():
            from atom.plugin.vllm.attention.layer import AttentionForVllm
            return AttentionForVllm(*args, **kwargs)
        if is_sglang():
            from atom.plugin.sglang.attention import AttentionForSGLang
            return AttentionForSGLang(*args, **kwargs)

        from atom.model_ops.paged_attention import Attention as AttentionForAtom
        return AttentionForAtom(*args, **kwargs)
```

Model files continue to use:

```python
from atom.model_ops.base_attention import Attention
```

They do not need to know whether the runtime is native ATOM, vLLM, or SGLang.

`atom.model_ops.__init__` no longer exposes mutable mode-specific attention classes. It only exports the frontend `Attention`.

### 3.2 Native ATOM Attention

Native ATOM attention now stays in `atom/model_ops`.

Important files:

```text
atom/model_ops/paged_attention.py
atom/model_ops/attention_mha.py
atom/model_ops/attention_mla.py
atom/model_ops/attentions/aiter_attention.py
atom/model_ops/attentions/aiter_mla.py
```

The native path is now only for ATOM native runtime.

`PagedAttentionImpl` and `MLAAttention` no longer receive vLLM decorator methods.

Native backends no longer become vLLM `"CUSTOM"` backends.

For example:

- `AiterBackend.get_name()` returns native `ATOM_ATTENTION`.
- `AiterMLABackend.get_name()` returns native `ROCM_AITER_MLA`.
- Native metadata builders directly inherit `CommonAttentionBuilder`.

There is no vLLM-specific class mutation in this path.

### 3.3 ATOM-vLLM Attention

ATOM-vLLM attention is now under:

```text
atom/plugin/vllm/attention/
```

Main files:

```text
layer.py            # AttentionForVllm factory
layer_common.py     # Shared helpers for MHA and MLA layers
layer_mha.py        # AttentionForVllmMHA
layer_mla.py        # AttentionForVllmMLA, AttentionForVllmSparseMLA
backend.py          # vLLM backend and metadata builder classes
metadata.py         # vLLM metadata types and sparse metadata builder methods
ops.py              # Custom op registration for torch.compile graph splitting
mla_impl.py         # MLA kernel helpers (reorg_kvcache, GEMM wrappers)
mla_sparse_impl.py  # Sparse MLA helpers and Indexer decorators
```

`layer.py` only keeps the factory:

```text
AttentionForVllm
```

It dispatches to:

```text
AttentionForVllmMHA
AttentionForVllmMLA
AttentionForVllmSparseMLA
```

#### MHA: Full Isolation

`AttentionForVllmMHA` inherits `nn.Module` and vLLM `AttentionLayerBase`. It does **not** inherit native `PagedAttentionImpl`.

This is because MHA weight processing is trivial (no weight splitting or absorption), and the forward paths between native and vLLM diverge significantly: vLLM needs `AttentionLayerBase` contract methods, custom op graph splitting, vLLM-specific metadata format, and different KV cache lifecycle management. The overlap in actual kernel dispatch code (RoPE, paged attention triton/ASM) is structural but not deep enough to justify coupling the two paths. Keeping MHA fully separate means vLLM MHA can evolve with vLLM's v1 API without risk to native ATOM.

#### MLA: Selective Inheritance from Native `MLAAttention`

`AttentionForVllmMLA` inherits both native `MLAAttention` and vLLM `AttentionLayerBase`:

```python
class AttentionForVllmMLA(MLAAttention, AttentionLayerBase):
```

This is a deliberate design choice, not an incomplete migration. The reason is that MLA has substantial framework-independent weight processing logic that is complex and must stay in sync between native and vLLM:

- **`__init__`**: Initializes `kv_b_proj`, `q_a_proj`, `kv_a_proj_with_mqa`, RoPE embeddings, head dimension calculations, FP8/FP4 quantization parameters, and q-absorption buffers. This is ~100 lines of non-trivial setup that depends on model config, not on runtime mode.
- **`process_weights_after_loading()`**: Splits `kv_b_proj` weight into `W_UK` and `W_UV`, handles MxFP4 pre-shuffle, computes `w_kc`/`w_vc` for q-absorption. This logic is tied to the model architecture (DeepSeek MLA) and must produce identical results regardless of runtime mode.
- **`_v_up_proj()`**: Value up-projection with FP8/FP4/BF16 dispatch, used identically by both native and vLLM forward paths.

Duplicating this (~400 lines of weight setup + kernel dispatch) would create a real maintenance burden: any change to MLA weight handling (new quant format, new model variant) would need to be applied in two places.

By contrast, `AttentionForVllmMLA` provides its own:
- `forward_impl()` / `forward_impl_sparse()` — vLLM-specific forward with vLLM metadata
- `_forward_prefill_plugin_mode()` / `_forward_decode_plugin_mode()` — vLLM prefill/decode dispatch
- `get_attn_backend()` / `get_kv_cache_spec()` — vLLM contract methods
- `_mla_plugin_mode_init()` logic inlined in `__init__` — vLLM-specific state setup

This gives a clean separation: **weights and kernel wrappers are shared via inheritance; layer lifecycle and forward dispatch are fully owned by the vLLM layer.**

#### Why Not the Same Approach for MHA?

| Aspect | MHA | MLA |
|--------|-----|-----|
| Weight processing complexity | Trivial (no splitting, no absorption) | Complex (~400 lines: kv_b_proj split, q-absorption, FP4 pre-shuffle) |
| Risk of divergence if duplicated | Low | High — any new quant format or model variant must update both copies |
| Forward path overlap | Structural only (same kernels, different metadata) | Same kernels AND shared `_v_up_proj` helper |
| Inheritance cost | Would couple vLLM MHA to native MHA lifecycle | Only couples weight setup, not forward dispatch |

The decision rule is: **inherit when the shared code is complex, framework-independent, and frequently updated; isolate when the shared code is simple or tightly coupled to runtime-specific behavior.**

These classes provide:

- `get_attn_backend()`
- `get_kv_cache_spec()`
- `layer_name`
- `kv_cache`
- `process_weights_after_loading()`
- `forward()`
- `forward_impl()`

The custom ops are registered in `ops.py`:

```text
torch.ops.aiter.atom_vllm_mha_attention
torch.ops.aiter.atom_vllm_mla_attention
```

These ops are also marked as ATOM split ops for torch compile graph splitting.

This is required because ATOM-vLLM uses ATOM's compile and graph split strategy, not vLLM's standard attention op split path.

### 3.4 ATOM-vLLM Backends and Metadata Builders

ATOM-vLLM backend code is in:

```text
atom/plugin/vllm/attention/backend.py
atom/plugin/vllm/attention_backend/mla_sparse.py
```

The vLLM backends are explicit classes:

```text
AiterMhaBackendForVllm
AiterMlaBackendForVllm
AiterSparseMlaBackendForVllm
AiterSparseMlaIndexerBackendForVllm
```

All vLLM backend contract methods are written directly in the backend classes, including:

- `get_name()`
- `get_builder_cls()`
- `get_impl_cls()`
- `get_kv_cache_shape()`
- `get_supported_head_sizes()`
- `get_kv_cache_stride_order()`
- `is_mla()`
- `is_sparse()`
- `full_cls_name()`

The metadata builders for MHA and dense MLA are also explicit classes with real member methods:

```text
AiterMhaMetadataBuilderForVllm    (in backend.py)
AiterMlaMetadataBuilderForVllm    (in backend.py)
```

They no longer use metadata builder decorators. Their `__init__`, `build`, and helper methods are real member methods.

The sparse MLA metadata builders are in `attention_backend/mla_sparse.py`:

```text
AiterMLASparseMetadataBuilder
AiterMLASparseIndexerMetadataBuilder
```

These still use method-container mixins (`AiterMlaSparseMetadataBuilderMethodsForVllm`, `AiterMlaSparseIndexerMetadataBuilderMethodsForVllm`) from `metadata.py` for their `build()` methods, because the sparse metadata build logic is shared between the metadata builder and the backend. This is a known remaining pattern that can be inlined in a follow-up.

### 3.5 ATOM-SGLang Attention

ATOM-SGLang now has its own frontend attention wrapper:

```text
atom/plugin/sglang/attention.py
```

It exposes:

```text
AttentionForSGLang
```

This keeps SGLang attention selection out of `atom.model_ops.__init__` and out of native ATOM attention code.

### 3.6 Plugin Folder Cleanup

vLLM-only attention files were moved out of `atom/plugin` top level.

Deleted top-level files:

```text
atom/plugin/attention.py
atom/plugin/attention_mha.py
atom/plugin/attention_mla.py
atom/plugin/attention_mla_sparse.py
```

vLLM-only MoE adaptation was moved from:

```text
atom/plugin/moe.py
```

to:

```text
atom/plugin/vllm/moe.py
```

This makes the top-level plugin folder contain only shared plugin infrastructure.

### 3.7 Removed Fallback Flag

The old flag `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION` has been removed.

The reason is that ATOM-vLLM no longer constructs vLLM standard `Attention` or `MLAAttention`. Therefore, there is no clean way to disable only ATOM attention while keeping ATOM model implementations active.

To run pure vLLM, use:

```bash
ATOM_DISABLE_VLLM_PLUGIN=1
```

### 3.8 Resulting Directory Shape

The new attention structure is:

```text
atom/model_ops/
  base_attention.py
  paged_attention.py
  attention_mha.py
  attention_mla.py
  attentions/
    aiter_attention.py
    aiter_mla.py

atom/plugin/vllm/
  attention/
    layer.py
    layer_common.py
    layer_mha.py
    layer_mla.py
    backend.py
    metadata.py
    ops.py
    mla_impl.py
    mla_sparse_impl.py
  attention_backend/
    mla_sparse.py
    gdn_attn.py
    attention_gdn.py
  moe.py

atom/plugin/sglang/
  attention.py
  attention_backend/
    radix_attention.py
    sgl_attn_backend.py
    sgl_attention_mla.py
    attention_gdn.py
```

#### Native ATOM (`atom/model_ops/`)

| File | Description |
|------|-------------|
| `base_attention.py` | Frontend `Attention` class whose `__new__` dispatches to native/vLLM/SGLang. Also contains shared triton kernels for context-parallel KV gather (`cp_mha_gather_cache`) and native custom op registration (`unified_attention_with_output_base`). Defines abstract `BaseAttention` and `LinearAttention` for GDN/hybrid models. |
| `paged_attention.py` | Native ATOM `Attention` layer. Selects the native backend (`AiterBackend` / `AiterMLABackend`), instantiates the impl, registers itself in `static_forward_context`, and routes `forward()` through the native custom op. |
| `attention_mha.py` | `PagedAttentionImpl` — the native MHA attention implementation. Contains RoPE cache application, flash-attention prefill path, paged-attention decode path (triton and ASM backends), prefix-cache gather, and sliding-window extension. |
| `attention_mla.py` | `MLAAttention` — the native MLA attention implementation. Handles MLA weight processing (`kv_b_proj` W_UK/W_UV splitting, q-absorption buffer setup, FP4 pre-shuffle), `_v_up_proj`, and native MLA prefill/decode forward paths. Also defines `MLAModules` dataclass used by model code to pass MLA-specific sub-modules. |
| `attentions/aiter_attention.py` | `AiterBackend` — native MHA attention backend. Returns `"ATOM_ATTENTION"` as backend name. Contains `AiterAttentionMetadataBuilder` which builds `AttentionMetaData` from `ScheduledBatch`, manages KV cache allocation, block table conversion, and TBO per-ubatch buffer setup. |
| `attentions/aiter_mla.py` | `AiterMLABackend` — native MLA attention backend. Returns `"ROCM_AITER_MLA"`. Contains `AiterMLAMetadataBuilder` with MLA-specific metadata management: sparse index buffers, MTP (multi-token prediction) support, persistent decode worker buffers, and context-parallel metadata construction. |

#### ATOM-vLLM (`atom/plugin/vllm/attention/`)

| File | Description |
|------|-------------|
| `layer.py` | `AttentionForVllm` factory. Its `__new__` dispatches to `AttentionForVllmMHA`, `AttentionForVllmMLA`, or `AttentionForVllmSparseMLA` based on `use_mla` and whether an Indexer is present. Also triggers custom op registration by importing `ops`. |
| `layer_common.py` | Shared helpers used by both MHA and MLA layers: `_init_vllm_layer_state()` (sets up KV cache dtype, quant scales, layer name), `_register_vllm_static_forward_context()` (registers layer in vLLM's `static_forward_context`), and `_set_default_scales()` (initializes default quantization scales). |
| `layer_mha.py` | `AttentionForVllmMHA` — vLLM MHA attention layer. Inherits `nn.Module` + `AttentionLayerBase`. Self-contained implementation: RoPE cache, paged attention triton/ASM, flash-attention prefill, sliding-window, `get_kv_cache_spec()`, `process_weights_after_loading()`. Does not inherit native `PagedAttentionImpl`. |
| `layer_mla.py` | `AttentionForVllmMLA` — vLLM MLA attention layer. Inherits native `MLAAttention` (for weight processing) + `AttentionLayerBase`. Contains vLLM-specific forward paths: `_forward_prefill_plugin_mode()`, `_forward_decode_plugin_mode()`, DCP (data-context parallelism) support, chunked prefill context computation. `AttentionForVllmSparseMLA` is a subclass that only overrides `attn_backend_cls`. |
| `backend.py` | vLLM backend and metadata builder classes. `AiterMhaBackendForVllm` / `AiterMlaBackendForVllm` implement the vLLM backend contract (`get_name() = "CUSTOM"`, `get_kv_cache_shape`, `full_cls_name`, etc.). `AiterMhaMetadataBuilderForVllm` builds `AiterMhaMetadataForVllm` for prefill/decode/extend. `AiterMlaMetadataBuilderForVllm` inherits vLLM's `MLACommonMetadataBuilder` and builds `AiterMlaMetadataForVllm` with persistent worker buffer management. |
| `metadata.py` | vLLM-specific metadata dataclasses: `AiterMhaMetadataForVllm` / `AiterMhaPhaseMetadata` for MHA, `AiterMlaMetadataForVllm` / `AiterMlaDecodeMetadataForVllm` / `AiterMlaPrefillMetadataForVllm` / `AiterMlaPersistentMetadataForVllm` for dense MLA, `AiterMlaSparseMetadataForVllm` for sparse MLA, and `VllmDeepseekV32IndexerMetadata` for sparse indexer metadata. Also contains sparse metadata builder method-container mixins used by `mla_sparse.py`. |
| `ops.py` | Registers two custom ops: `atom_vllm_mha_attention` and `atom_vllm_mla_attention`. Each is marked as `spliting_op` for ATOM's torch.compile graph splitting strategy. The ops retrieve the layer and metadata from vLLM's `forward_context` and delegate to `layer.forward_impl()`. |
| `mla_impl.py` | MLA kernel-level helpers shared across dense and sparse MLA paths: `reorg_kvcache()` for context-parallel KV cache reorganization after cross-rank gather, and conditional imports for fused GEMM kernels (FP8 BMM, FP4 BMM, fused split-cat). |
| `mla_sparse_impl.py` | Sparse MLA helpers: `fetch_id_to_ragged_triton()` triton kernel for topk index format conversion, `sparse_attn_indexer_plugin_mode()` custom op for sparse indexer KV cache write + topk index computation. Also contains `IndexerDecoratorForPluginMode` and `DeepseekV32IndexerCacheDecoratorForPluginMode` for Indexer/IndexerCache vLLM adaptation. |

#### ATOM-vLLM Attention Backends (`atom/plugin/vllm/attention_backend/`)

| File | Description |
|------|-------------|
| `mla_sparse.py` | Sparse MLA backends (`AiterMLASparseBackend`, `AiterMLASparseIndexerBackend`) and their metadata builders (`AiterMLASparseMetadataBuilder`, `AiterMLASparseIndexerMetadataBuilder`). Inherits from `AiterMlaBackendForVllm`. Metadata builders use method-container mixins from `metadata.py` for `build()` logic. |
| `gdn_attn.py` | `GDNAttentionBackend` — vLLM attention backend for GatedDeltaNet (linear attention) layers in hybrid models (Qwen3.5, Qwen3-Next). |
| `attention_gdn.py` | `ChunkGatedDeltaRule` and `GatedDeltaNet` — vLLM GDN attention layer implementations for hybrid architectures. |

#### ATOM-SGLang (`atom/plugin/sglang/`)

| File | Description |
|------|-------------|
| `attention.py` | `AttentionForSGLang` — thin wrapper that inherits `RadixAttention`. Used by the frontend `Attention` dispatcher when running in SGLang mode. |
| `attention_backend/radix_attention.py` | `RadixAttention` — SGLang radix-tree based attention with AITER kernel integration. |
| `attention_backend/sgl_attn_backend.py` | `ATOMAttnBackendForSgl` — SGLang attention backend registered under the `"aiter"` name for SGLang's backend selection. |
| `attention_backend/sgl_attention_mla.py` | SGLang MLA attention implementation with SGLang-specific forward paths, decode/prefill dispatch, and BMM kernel integration. |
| `attention_backend/attention_gdn.py` | SGLang GDN attention implementation for hybrid models. |

## 4. Expected Benefits

This refactor gives several benefits:

- Clear ownership for each runtime mode.
- No vLLM decorator pollution in native ATOM attention.
- No global `ops.Attention` mutation.
- Easier debugging because class methods are visible on the class itself.
- Safer future changes because native, vLLM, and SGLang attention can evolve separately.
- Better alignment with vLLM runtime: ATOM-vLLM layers implement `AttentionLayerBase` directly.
- Cleaner plugin folder layout.

## 5. Known Remaining Items

The following items are intentionally kept for follow-up:

1. **Indexer decorators**: `IndexerDecoratorForPluginMode` and `DeepseekV32IndexerCacheDecoratorForPluginMode` in `mla_sparse_impl.py` still use the decorator pattern. They affect Indexer/IndexerCache classes (not attention layers), so they are out of scope for this attention refactor.
2. **Sparse metadata builder mixins**: `AiterMLASparseMetadataBuilder` and `AiterMLASparseIndexerMetadataBuilder` use method-container mixins from `metadata.py`. These can be inlined into the builder classes in a follow-up.
3. **MHA kernel wrapper duplication**: `AttentionForVllmMHA` and native `PagedAttentionImpl` have structurally similar kernel dispatch code (`paged_attention_triton`, `paged_attention_asm`, `rope_cache`). This is the expected cost of isolation. If kernel interfaces change frequently, shared kernel wrappers can be extracted later.
