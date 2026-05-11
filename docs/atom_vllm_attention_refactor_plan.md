# atom-vllm Attention Refactor Plan

本文档是在当前架构分析基础上，结合新的重构要求制定的方案。目标是让 ATOM native attention、atom-vllm attention、atom-sglang attention 在建模入口层面分离，避免继续通过大量 decorator/patch 把同一套类改造成多种 runtime 形态。

## 核心结论

新的重构方向应该从“让 ATOM backend 适配 vLLM 标准 `Attention`”转为“ATOM 提供自己的 vLLM attention layer，并实现 vLLM runtime 需要的 layer contract”。

也就是说：

- 保留 `atom/model_ops/base_attention.py` 暴露给模型建模层的 `Attention`。
- `Attention.__new__()` 根据当前 mode 分派到不同实现。
- ATOM native、vLLM、SGLang 的 attention class 和 backend class 分离。
- atom-vllm 不再构造 vLLM 标准 `Attention` / `MLAAttention`。
- atom-vllm 的 attention layer 直接实现 vLLM `AttentionLayerBase` 所需接口。
- vLLM adapter 只保留 vLLM runtime contract，不再反向约束 ATOM native attention。

## 为什么可以不构造 vLLM 标准 Attention

从 vLLM 代码看，vLLM runtime 并不要求 attention layer 的实例必须是 `vllm.model_executor.layers.attention.Attention`。它真正依赖的是以下 contract：

- layer 是 `AttentionLayerBase` 的实例，或者至少能被 `get_layers_from_vllm_config(..., AttentionLayerBase)` 发现。
- layer 注册到 `vllm_config.compilation_config.static_forward_context`。
- layer 实现 `get_kv_cache_spec(vllm_config)`。
- layer 实现 `get_attn_backend()`。
- layer 有 `kv_cache` 属性，后续由 vLLM `bind_kv_cache()` 写入。
- backend 提供 vLLM `AttentionBackend` contract，例如 `get_kv_cache_shape()`、`get_builder_cls()`、`full_cls_name()`。
- metadata builder 能在 vLLM `build_attn_metadata()` 中通过 `build(common_prefix_len, common_attn_metadata)` 生成 per-layer metadata。
- forward 阶段能从 `vllm.forward_context.get_forward_context()` 取到 `attn_metadata`、`slot_mapping`、`no_compile_layers` 和额外上下文。

vLLM 标准 `Attention` 主要做了这些事情：

- 选择 backend。
- 初始化量化和 KV scale 相关属性。
- 构造 impl。
- 注册 `static_forward_context`。
- 暴露 `get_kv_cache_spec()` / `get_attn_backend()`。
- 在 `forward()` 中调用 vLLM 自己的 `torch.ops.vllm.unified_attention*` custom op。

这些职责 atom-vllm 可以自己实现。因此不构造 vLLM 标准 `Attention` 是可行的，而且能消除目前大量 patch vLLM `MLAAttention.forward_impl()`、`get_kv_cache_spec()`、`process_weights_after_loading()` 的必要性。

## 目标架构

### 建模层入口

保留 `atom/model_ops/base_attention.py` 中对模型暴露的 `Attention`，但让它成为真正的 mode dispatcher。

目标语义：

```python
class Attention:
    def __new__(cls, *args, **kwargs):
        if is_atom():
            return AttentionForAtom(*args, **kwargs)
        if is_vllm():
            return AttentionForVllm(*args, **kwargs)
        if is_sglang():
            return AttentionForSGLang(*args, **kwargs)
        raise RuntimeError(...)
```

注意这里的 `AttentionForAtom`、`AttentionForVllm`、`AttentionForSGLang` 是三个互相独立的 attention layer，而不是同一个类通过 decorator 在 import time 改造成不同形态。

### 建议目录结构

```text
atom/model_ops/
  base_attention.py
  attention_atom.py

atom/plugin/vllm/attention/
  __init__.py
  layer.py
  layer_mha.py
  layer_mla.py
  backend_mha.py
  backend_mla.py
  backend_sparse_mla.py
  metadata_mha.py
  metadata_mla.py
  metadata_sparse_mla.py
  metadata_sparse_indexer.py
  ops.py
  kv_cache.py
  quant.py

atom/plugin/sglang/attention/
  layer.py
```

现有 `atom/model_ops/paged_attention.py` 可以逐步收敛成 ATOM native `AttentionForAtom`，或者保留为 native 实现细节，但不再承担 vLLM layer wrapper 职责。

## 新的类边界

### `AttentionForAtom`

职责：

- 只服务 ATOM native/server mode。
- 使用 ATOM `CommonAttentionBuilder` / `ScheduledBatch` pipeline。
- 使用 native `AiterBackend`、`AiterMLABackend`、`TritonMHABackend`、`TritonMLABackend` 等 backend。
- 保留当前 `torch.ops.aiter.unified_attention_with_output_base` native dispatch。
- 不 import vLLM。
- 不包含 `is_vllm()` 分支。

### `AttentionForVllm`

职责：

- 只服务 vLLM plugin mode。
- 继承 `torch.nn.Module` 和 vLLM `AttentionLayerBase`。
- 不构造 vLLM 标准 `Attention` / `MLAAttention`。
- 自己注册到 `vllm_config.compilation_config.static_forward_context`。
- 自己暴露 `get_attn_backend()` 和 `get_kv_cache_spec()`。
- 自己持有 `kv_cache` placeholder，等待 vLLM `bind_kv_cache()` 写入。
- 自己处理 `process_weights_after_loading()`、scale 初始化、量化 scale 默认值。
- forward 中调用 ATOM-vLLM 专用 custom op 或 direct implementation。

建议内部再分成：

- `AttentionForVllmMHA`
- `AttentionForVllmMLA`
- `AttentionForVllmSparseMLA`

`AttentionForVllm` 可以是工厂，也可以在 `base_attention.Attention` 中直接根据 `use_mla`、`mla_modules.indexer` 分派。

### `AttentionForSGLang`

职责：

- 只服务 SGLang plugin mode。
- 与 Radix/SGLang runtime 的接口对齐。
- 不复用 vLLM metadata builder 或 vLLM backend。
- 不再让 `PagedAttention` 对 SGLang 做 assert 或特殊限制，而是在入口层就分到 SGLang 专属实现。

## vLLM 专属 backend 分离

当前的 `AiterBackendDecoratorForPluginMode` 会把 vLLM backend 方法注入 native `AiterBackend` / `AiterMLABackend`。新的方案里应该停止这种做法。

目标类：

- `AiterMhaBackendForVllm`
- `AiterMlaBackendForVllm`
- `AiterSparseMlaBackendForVllm`
- `AiterSparseMlaIndexerBackendForVllm`

这些类直接实现 vLLM `AttentionBackend` contract，放在 `atom/plugin/vllm/attention/backend_*.py`。

native backend 保持纯净：

- `atom/model_ops/attentions/aiter_attention.py::AiterBackend`
- `atom/model_ops/attentions/aiter_mla.py::AiterMLABackend`

不再根据 `is_vllm()` 改写自己的 class surface，也不再在 plugin mode 下返回 `"CUSTOM"` 来伪装成 vLLM backend。vLLM backend identity 由 vLLM 专属 backend class 自己负责。

## vLLM 专属 metadata builder 分离

当前 metadata builder 的问题是同一个 `AiterAttentionMetadataBuilder` / `AiterMLAMetadataBuilder` 在 native mode 和 vLLM mode 下通过 decorator 变成不同类。

新的方案：

- native metadata builder 保留在 `atom/model_ops/attentions/*`。
- vLLM metadata builder 放在 `atom/plugin/vllm/attention/metadata_*.py`。
- vLLM builder 直接以 vLLM `AttentionMetadataBuilder` / `MLACommonMetadataBuilder` contract 编写。
- 共享的数学逻辑抽成 framework-neutral helper，例如 slot mapping、KV indptr、KV indices、chunked context 组织、TBO split。

建议拆分：

- `VllmMhaMetadataBuilder`
- `VllmMlaMetadataBuilder`
- `VllmSparseMlaMetadataBuilder`
- `VllmSparseMlaIndexerMetadataBuilder`

metadata 输出也应显式命名：

- `VllmMhaMetadata`
- `VllmMlaMetadata`
- `VllmSparseMlaMetadata`
- `VllmSparseMlaIndexerMetadata`

这些类型可以继续放在 `AttentionMetaData.plugin_metadata` 中，但 producer/consumer 要明确绑定，避免松散字段约定。

## `AttentionForVllm` 的最小 vLLM contract

### MHA

`AttentionForVllmMHA` 至少需要：

- `layer_name`
- `attn_type`
- `num_heads`
- `head_size`
- `head_size_v`
- `num_kv_heads`
- `sliding_window`
- `kv_cache_dtype`
- `kv_cache_torch_dtype`
- `attn_backend = AiterMhaBackendForVllm`
- `impl = AtomVllmMhaImpl(...)`
- `kv_cache = torch.tensor([])`
- `get_attn_backend()`
- `get_kv_cache_spec()`
- `process_weights_after_loading()`
- `forward(query, key, value, positions=None, q_scale=None, qkv=None, **kwargs)`

`get_kv_cache_spec()` 返回 vLLM `FullAttentionSpec` 或 `SlidingWindowSpec`。

### MLA

`AttentionForVllmMLA` 至少需要：

- `layer_name`，建议明确区分 model-level prefix 和 vLLM KV-cache layer name。
- `q_proj` / `o_proj` 或持有 `mla_modules`，因为当前 `PagedAttention` 在 vLLM mode 下承担了 MLA q projection 和 output projection。
- `kv_b_proj`
- `qk_nope_head_dim`
- `qk_rope_head_dim`
- `v_head_dim`
- `q_lora_rank`
- `kv_lora_rank`
- `head_size = kv_lora_rank + qk_rope_head_dim`
- `num_kv_heads = 1`
- `attn_backend = AiterMlaBackendForVllm` 或 sparse backend
- `impl = AtomVllmMlaImpl(...)`
- `kv_cache = torch.tensor([])`
- `get_attn_backend()`
- `get_kv_cache_spec()`
- `process_weights_after_loading()`
- `forward(query, key, value, positions=None, q_scale=None, qkv=None, **kwargs)`

`get_kv_cache_spec()` 返回 vLLM `MLAAttentionSpec`，并由 ATOM 自己决定是否需要规避 `fp8_ds_mla` 这类 vLLM layout。

### Sparse MLA

Sparse MLA 不应该只是 `AttentionForVllmMLA` 的一个小 if 分支。它需要独立处理：

- main sparse attention layer
- indexer layer/cache
- top-k indices buffer
- sparse metadata builder
- sparse indexer metadata builder
- ragged KV page conversion

建议 `AttentionForVllmSparseMLA` 继承或组合 `AttentionForVllmMLA` 的投影和 shared state，但 backend、metadata、forward impl 独立。

## Forward 调用方案

### 不再走 vLLM `torch.ops.vllm.unified_attention`

如果不构造 vLLM 标准 `Attention`，就不应该继续依赖 vLLM `unified_attention` / `unified_mla_attention` 这组 custom op。它们内部会从 vLLM forward context 取 layer，然后调用 `self.impl.forward()` 或 `layer.forward_impl()`，这是 vLLM 标准 layer 设计。

atom-vllm 应该提供自己的 opaque custom op，例如：

- `torch.ops.aiter.atom_vllm_mha_attention`
- `torch.ops.aiter.atom_vllm_mla_attention`
- `torch.ops.aiter.atom_vllm_sparse_mla_attention`
- `torch.ops.aiter.atom_vllm_sparse_indexer`

这些 op 的 Python fallback 可以：

1. 从 `vllm.forward_context.get_forward_context()` 取当前 context。
2. 用 `layer_name` 找到 `no_compile_layers[layer_name]`，也就是 `AttentionForVllm*` 实例。
3. 从 `attn_metadata` dict 中取该 layer 的 metadata。
4. 使用 `layer.kv_cache`。
5. 调用 ATOM-vLLM impl。

这样 torch.compile 仍然能把 attention 当 opaque op 处理，但不需要 vLLM 标准 `Attention` class。

### Direct call 兼容

如果某些平台或测试环境使用 direct call，也可以在 `AttentionForVllm*` 中直接读取 forward context 并调用 impl。正式性能路径仍建议走 ATOM-vLLM custom op，以保持图切分和 CUDA graph 行为可控。

## 可以删除或收敛的旧机制

新架构稳定后，可以逐步移除：

- `AiterBackendDecoratorForPluginMode`
- `AiterAttentionMetadataBuilderDecoratorForPluginMode`
- `AiterMLAAttentionMetadataBuilderDecoratorForPluginMode`
- `AiterMLASparseAttentionMetadataBuilderDecoratorForPluginMode`
- `AiterMLASparseIndexerAttentionMetadataBuilderDecoratorForPluginMode`
- `PagedAttentionImplDecoratorForPluginMode`
- `MLAAttentionImplDecoratorForPluginMode`
- `MLASparseAttentionImplDecoratorForPluginMode`
- `PagedAttention` 中 `if is_vllm(): from vllm.attention.layer import Attention, MLAAttention`

注意：这些不应该一次性删除，而应该在新 `AttentionForVllm` 路径验证完后分阶段下线。

## 与现有代码的迁移步骤

### 阶段 1：建立入口分离，不改变行为

目标：

- 改造 `base_attention.Attention` 为 mode dispatcher。
- 引入 `AttentionForAtom`，先复用当前 `PagedAttention` native 行为。
- 引入 `AttentionForVllm` 壳类，但第一阶段可以仍代理到当前 `PagedAttention` vLLM 分支，保证模型建模层入口先稳定。
- `atom/model_ops/__init__.py` 不再作为可变 `Attention` 全局切换点，或者只 re-export `base_attention.Attention`。

验收：

- native ATOM 模型构造不变。
- vLLM plugin 模型构造不变。
- SGLang 模型构造不变。
- 所有模型代码仍然只 import/use `Attention`。

### 阶段 2：引入 vLLM 专属 backend 和 metadata builder

目标：

- 新增 `AiterMhaBackendForVllm`、`AiterMlaBackendForVllm`、`AiterSparseMlaBackendForVllm`。
- 新增 vLLM 专属 metadata builder。
- 先从旧 `atom/plugin/attention.py` 迁移代码，不改变 metadata 结构。
- vLLM backend 不再通过 decorator 注入 native backend。

验收：

- vLLM `init_attn_backend()` 能按新 backend `full_cls_name()` 分组。
- vLLM `build_attn_metadata()` 能拿到新 builder 的 metadata。
- MHA、MLA、sparse MLA metadata key 和当前行为一致。

### 阶段 3：实现真正的 `AttentionForVllm`

目标：

- `AttentionForVllmMHA` 不再构造 vLLM `Attention`。
- `AttentionForVllmMLA` 不再构造 vLLM `MLAAttention`。
- layer 自己注册 vLLM `static_forward_context`。
- layer 自己实现 `get_kv_cache_spec()` / `get_attn_backend()`。
- layer 自己维护 `kv_cache`、scale、quant 相关属性。

验收：

- vLLM `get_kv_cache_spec()` 能发现所有 ATOM attention layer。
- vLLM `bind_kv_cache()` 能正确写入 `layer.kv_cache`。
- `ATOM_DISABLE_VLLM_PLUGIN=1` pure-vLLM 路径仍清晰可控。

### 阶段 4：替换 forward custom op

目标：

- 新增 ATOM-vLLM 专属 attention custom op。
- MHA forward 从 `torch.ops.vllm.unified_attention*` 切换到 `torch.ops.aiter.atom_vllm_mha_attention`。
- MLA forward 从 patched vLLM `MLAAttention.forward_impl()` 切换到 `torch.ops.aiter.atom_vllm_mla_attention`。
- Sparse MLA indexer 和 main attention 分别走专属 op。

验收：

- torch.compile 图中 attention 仍然是明确的 split/opaque op。
- CUDA graph capture 下 output buffer 和 KV cache update 顺序正确。
- 不再需要 patch vLLM `MLAAttention.forward_impl()`。

### 阶段 5：清理旧 decorator 和大文件

目标：

- `atom/plugin/attention.py` 拆分为 vLLM attention 子包内的 backend/metadata/helper。
- native `aiter_attention.py` / `aiter_mla.py` 删除 plugin decorator。
- plugin impl decorator 迁移为显式 impl class 或 helper function。
- sparse MLA 形成独立 vLLM-only 子系统。

验收：

- native attention 文件不再 import `atom.plugin.attention`。
- vLLM attention 文件不再修改 native backend class。
- 搜索 `DecoratorForPluginMode` 时 attention 主路径不再依赖它。

## 关键设计细节

### layer name

当前 MHA 和 MLA 的 layer name 有差异：

- MHA vLLM `Attention` 使用 `prefix`。
- MLA vLLM `MLAAttention` 当前使用 `f"{prefix}.attn"`。
- ATOM wrapper 自己又使用 `prefix` 注册到 ATOM `static_forward_context`。

新方案需要显式区分：

- `model_layer_name`: ATOM model 内部层名，例如 `model.layers.0.self_attn`。
- `vllm_attn_layer_name`: vLLM KV cache/metadata layer 名。

建议在 `AttentionForVllm*` 中显式保存这两个字段，避免隐式拼接分散在代码里。

### 量化和 scale 初始化

不构造 vLLM `Attention` 后，需要自己承担这些职责：

- `_q_scale`
- `_k_scale`
- `_v_scale`
- `_prob_scale`
- `_q_scale_float`
- `_k_scale_float`
- `_v_scale_float`
- `_prob_scale_float`
- `q_range`
- `k_range`
- `v_range`
- `calculate_kv_scales`
- `process_weights_after_loading()`

建议不要直接复制 vLLM helper 到多个文件，而是在 `atom/plugin/vllm/attention/quant.py` 中形成 ATOM-vLLM 专属 helper。

### memory plan

vLLM 专属 metadata builder 和 layer 建立后，可以同时引入 `VllmAttentionMemoryPlan`：

- MHA extend workspace
- MHA persistent PA buffers
- MLA persistent buffers
- MLA chunked prefill workspace
- Sparse MLA top-k buffers
- Sparse indexer buffers
- TBO microbatch buffers

第一阶段只做枚举和日志，后续再考虑和 vLLM memory accounting 对接。

## 风险与验证重点

高风险点：

- vLLM `get_layers_from_vllm_config(..., AttentionLayerBase)` 是否能完整发现新 layer。
- vLLM KV cache spec 和 backend grouping 是否与当前一致。
- `bind_kv_cache()` 是否能写入新 layer 的 `kv_cache`。
- torch.compile opaque op 和 graph split 是否保持当前行为。
- CUDA graph capture 时 output allocation、KV cache update、metadata buffer 生命周期是否正确。
- MLA q_proj/o_proj 和 layer name 从 wrapper 迁移到 `AttentionForVllmMLA` 时不要改变权重加载路径。
- Sparse MLA indexer cache 注册仍然需要 vLLM 发现。

最小验证矩阵：

- MHA dense model decode/prefill。
- MLA dense model decode/prefill。
- Sparse MLA model decode/prefill/indexer。
- prefix cache。
- chunked prefill。
- CUDA graph capture。
- torch.compile piecewise。
- fp8 KV cache。
- `ATOM_DISABLE_VLLM_PLUGIN=1` pure-vLLM path。
- SGLang plugin smoke test，确认入口分离没有破坏 SGLang。
- ATOM native/server smoke test，确认 native backend 不再受 vLLM adapter 影响。

## 推荐最终形态

最终希望代码呈现为：

- 模型代码只看到 `Attention`。
- `Attention` 在入口层按 mode 分派。
- ATOM native attention、vLLM attention、SGLang attention 是三个独立 class family。
- native backend 不知道 vLLM。
- vLLM backend 不通过 decorator 修改 native backend。
- atom-vllm 不构造 vLLM 标准 `Attention` / `MLAAttention`，只实现 vLLM runtime 必需 contract。
- vLLM 版本变化主要影响 `atom/plugin/vllm/attention/*`，不会扩散到 `atom/model_ops/attentions/*`。

