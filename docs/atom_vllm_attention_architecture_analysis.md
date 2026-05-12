# atom-vllm Attention Architecture Refactor Notes

本文档记录 atom-vllm attention 重构前的架构、主要痛点、本次重构方式，以及重构后的代码结构和执行链路。

## 目标

atom-vllm 的目标是：vLLM 负责 runtime、scheduler、KV cache 管理、CUDA graph 等服务层能力，ATOM 负责模型实现、attention layer、attention backend、metadata builder 和 AITER kernel 调用。

本次重构的核心目标是把 ATOM native attention、atom-vllm attention、atom-sglang attention 在入口、layer、backend、metadata builder、impl 上全部分离，避免同一批 class 通过 decorator 或 monkey patch 在不同 mode 下变成不同形态。

## 重构前架构

### 入口层

重构前，模型代码通过 `atom.model_ops.base_attention.Attention` 构造 attention。这个 frontend class 会间接读取 `atom.model_ops.Attention`：

```python
class Attention:
    def __new__(cls, *args, **kwargs):
        from atom.model_ops import Attention
        return Attention(*args, **kwargs)
```

`atom.model_ops.__init__` 中维护一个可变全局：

```python
Attention = PagedAttention
```

plugin 初始化时再通过 `set_attn_cls()` 改写它：

- vLLM mode: `ops.Attention = ops.PagedAttention`
- SGLang mode: `ops.Attention = ops.RadixAttention`

这让建模入口依赖全局 mutable state，也让 backend 需要读取 `ops.Attention` 来判断实现类型。

### atom-vllm layer

重构前，`atom.model_ops.paged_attention.PagedAttention` 同时承担 ATOM native 和 vLLM plugin 两种职责：

- native/server mode 下构造 ATOM 自己的 impl 和 backend。
- vLLM mode 下构造 vLLM 标准 `Attention` 或 `MLAAttention`。

这意味着 atom-vllm attention 实际上被包在 vLLM 标准 attention object 里运行。ATOM 需要通过 vLLM 的 `Attention` / `MLAAttention` 构造参数、KV cache spec、backend selector、forward custom op、output buffer 规则来接回自己的 kernel。

### backend 和 metadata builder

重构前，native backend class 在 plugin mode 下会被 decorator 动态改造：

- `AiterBackend` 通过 `AiterBackendDecoratorForPluginMode` 注入 vLLM backend API。
- `AiterMLABackend` 同样在 vLLM mode 下被注入 vLLM backend API。
- `AiterAttentionMetadataBuilder` 通过 `AiterAttentionMetadataBuilderDecoratorForPluginMode` 动态改 base class、`__init__` 和 `build`。
- `AiterMLAMetadataBuilder` 通过 `AiterMLAAttentionMetadataBuilderDecoratorForPluginMode` 动态改造成 vLLM metadata builder。

同一个类名在 ATOM native、vLLM、SGLang 下可能代表不同 base class、不同方法集合和不同运行语义。

### impl

重构前，native impl class 也会被 vLLM decorator 修改：

- `PagedAttentionImpl` 被 `PagedAttentionImplDecoratorForPluginMode` 注入 vLLM MHA forward/cache 方法。
- `MLAAttention` 被 `MLAAttentionImplDecoratorForPluginMode` 注入 dense MLA plugin 方法。
- `MLAAttention` 又被 `MLASparseAttentionImplDecoratorForPluginMode` 注入 sparse MLA plugin 方法。

因此 native impl 里同时承载 native server path 和 vLLM plugin path，`forward()` 需要根据 `is_plugin_mode()` 分支。

### plugin 目录

重构前，`atom/plugin` 顶层包含多种不同职责：

- `prepare.py`: mode 判断和 framework backbone 状态。
- `config.py`: plugin config 翻译。
- `attention.py`: vLLM MHA/MLA/sparse metadata、backend method、decorator、大量 helper。
- `attention_mha.py`: MHA plugin impl methods。
- `attention_mla.py`: MLA plugin impl methods。
- `attention_mla_sparse.py`: sparse MLA plugin impl、indexer custom op、indexer cache decorator。
- `moe.py`: vLLM-only MoE name patch。
- `vllm/` 和 `sglang/` 子包。

这让 vLLM-only 代码、SGLang-only 代码、公共代码和 native 代码边界不清晰。

## 重构前主要痛点

### 装饰器和 monkey patch 过多

重构前 decorator 不只是语法糖，而是在运行时改变 class identity、base class、method set、`__init__` 和 backend surface。典型问题：

- IDE 和静态分析无法准确看到真实类形态。
- 真实 MRO 依赖 import 顺序和 `is_vllm()` / `is_sglang()` 状态。
- 新增功能时难判断应该改 native class、plugin method container，还是 decorator。
- bug 定位需要同时理解 native path 和 plugin path。

### vLLM 标准 Attention 带来接口硬约束

因为 atom-vllm 构造 vLLM 标准 `Attention` / `MLAAttention`，ATOM attention 必须贴合 vLLM 的内部 contract：

- backend 必须提供 vLLM `AttentionBackend` 方法。
- metadata builder 必须接收 vLLM `common_attn_metadata`。
- MLA 需要 patch vLLM `MLAAttention.forward_impl()`。
- KV cache spec 和 fp8 layout 需要绕过 vLLM 某些默认行为。
- positions 依赖 vLLM forward context 的 `additional_kwargs["atom_positions"]`。

这让 ATOM 的 attention 设计被 vLLM 标准 layer 强牵引。

### native 和 atom-vllm 实现耦合

重构前 native `PagedAttentionImpl`、`MLAAttention` 同时服务 native 和 vLLM。vLLM 方法通过 decorator 注入 native class，导致：

- native `forward()` 需要判断 plugin mode。
- vLLM 修改可能影响 native。
- native backend 名称在 plugin mode 下变成 `"CUSTOM"`。
- native metadata builder 在 plugin mode 下变成另一个 class。

### `plugin` 目录边界混乱

大量 vLLM-only attention 文件放在 `atom/plugin` 顶层，容易误认为是公共 plugin 逻辑。`attention.py` 曾经超过两千行，混合 dataclass、backend surface、metadata builder、sparse indexer 和 unified wrapper。

### fallback 语义不成立

旧代码支持 `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION=1`，意图是“保留 ATOM model，但 attention 回退到 vLLM ROCm backend”。在新方向下，atom-vllm 不再构造 vLLM 标准 `Attention` / `MLAAttention`，因此没有合理路径只关闭 ATOM attention。保留该环境变量会制造错误预期。

## 重构原则

本次重构遵循以下原则：

- 模型文件只看到一个 frontend `Attention`。
- frontend `Attention` 在入口层根据 mode 分派。
- ATOM native、vLLM、SGLang 使用各自独立 attention layer。
- native backend/metadata builder/impl 不知道 vLLM。
- atom-vllm 不构造 vLLM 标准 `Attention` / `MLAAttention`。
- atom-vllm 自己实现 vLLM `AttentionLayerBase` 所需接口。
- vLLM-only 代码放到 `atom/plugin/vllm/` 下。
- SGLang-only 代码放到 `atom/plugin/sglang/` 下。
- 不再用 decorator/method-container 把方法 patch 到 vLLM attention class；vLLM 需要的方法直接写在 vLLM 专属 class 中。

## 重构后的入口结构

### frontend Attention

`atom.model_ops.base_attention.Attention` 现在是唯一建模入口：

```python
class Attention:
    def __new__(cls, *args, **kwargs):
        if is_vllm():
            return AttentionForVllm(*args, **kwargs)
        if is_sglang():
            return AttentionForSGLang(*args, **kwargs)
        return atom.model_ops.paged_attention.Attention(*args, **kwargs)
```

三个 mode 对应三个 attention family：

- ATOM native: `atom.model_ops.paged_attention.Attention`
- atom-vllm: `atom.plugin.vllm.attention.layer.AttentionForVllm*`
- atom-sglang: `atom.plugin.sglang.attention.AttentionForSGLang`

`atom.model_ops.__init__` 只 re-export frontend `Attention`，不再暴露 `PagedAttention` / `RadixAttention`，也不再维护可变全局 `ops.Attention`。

## 重构后的 native ATOM attention

native attention 只保留 native/server 语义。

### `atom/model_ops/paged_attention.py`

`PagedAttention` 已改名为 native `Attention`，保留 `PagedAttention = Attention` alias 兼容旧引用。它只在 ATOM native/server mode 下使用，并断言不支持 plugin mode。

### `atom/model_ops/attention_mha.py`

`PagedAttentionImpl` 是 native MHA impl。它不再接受 vLLM plugin decorator，不再包含 plugin-mode forward 分支。原来的 `forward_impl_server_mode()` 已收敛为普通 `forward_impl()`。

### `atom/model_ops/attention_mla.py`

`MLAAttention` 是 native MLA impl。它不再接受 vLLM dense/sparse decorator，不再包含 plugin-mode forward 分支。原来的 `forward_impl_server_mode()` 已收敛为普通 `forward_impl()`。

### `atom/model_ops/attentions/aiter_attention.py`

`AiterBackend` 是 native MHA backend：

- `get_name()` 返回 `ATOM_ATTENTION`
- `get_impl_cls()` 返回 native `PagedAttentionImpl`
- `AiterAttentionMetadataBuilder` 直接继承 `CommonAttentionBuilder`

它不再通过 decorator 注入 vLLM backend API。

### `atom/model_ops/attentions/aiter_mla.py`

`AiterMLABackend` 是 native MLA backend：

- `get_name()` 返回 `ROCM_AITER_MLA`
- `get_impl_cls()` 返回 native `MLAAttention`
- `AiterMLAMetadataBuilder` 直接继承 `CommonAttentionBuilder`

它不再通过 decorator 注入 vLLM backend API。

## 重构后的 atom-vllm attention

所有 atom-vllm attention 逻辑集中在：

```text
atom/plugin/vllm/attention/
  __init__.py
  layer.py
  backend.py
  metadata.py
  ops.py
  mla_impl.py
  mla_sparse_impl.py

atom/plugin/vllm/attention_backend/
  mla_sparse.py
  gdn_attn.py
  attention_gdn.py
```

### `layer.py`

`layer.py` 是 atom-vllm attention layer 入口。

它提供：

- `AttentionForVllm`: factory，根据 `use_mla` 和 `mla_modules.indexer` 分派。
- `AttentionForVllmMHA`: vLLM MHA attention layer。
- `AttentionForVllmMLA`: vLLM dense MLA attention layer。
- `AttentionForVllmSparseMLA`: sparse MLA attention layer，继承 `AttentionForVllmMLA` 并切换 backend。

`AttentionForVllmMHA` 直接继承 `nn.Module, AttentionLayerBase`，不再继承 native `PagedAttentionImpl`，也不再继承 method-container。MHA vLLM 需要的 RoPE/cache、prefill、extend、decode、forward impl 等方法全部是 `AttentionForVllmMHA` 自身的 member methods。

`AttentionForVllmMLA` 直接继承 `MLAAttention, AttentionLayerBase`。dense MLA 和 sparse MLA 的 vLLM forward/helper 方法已经直接成为 `AttentionForVllmMLA` 的 member methods，不再通过 `MLAAttentionImplPluginModeMethods` 或 `MLASparseAttentionImplPluginModeMethods` 继承获得。

每个 atom-vllm layer 直接实现 vLLM runtime 需要的接口：

- `get_attn_backend()`
- `get_kv_cache_spec()`
- `kv_cache` placeholder
- `layer_name`
- `process_weights_after_loading()`
- `AttentionLayerBase` 继承关系

### `backend.py`

`backend.py` 是 atom-vllm dense MHA/MLA backend 和 metadata builder 入口。

它提供：

- `AiterMhaBackendForVllm`
- `AiterMlaBackendForVllm`
- `AiterSparseMlaBackendForVllm`
- `AiterSparseMlaIndexerBackendForVllm`
- `AiterMhaMetadataBuilderForVllm`
- `AiterMlaMetadataBuilderForVllm`

这些 backend class 不再继承 `vllmAiter*BackendMethods`。vLLM backend contract 方法直接写在 class 中，例如：

- `accept_output_buffer`
- `supported_dtypes`
- `forward_includes_kv_cache_update`
- `get_supported_kernel_block_sizes()`
- `get_preferred_block_size()`
- `get_kv_cache_shape()`
- `get_kv_cache_stride_order()`
- `is_mla()`
- `is_sparse()`
- `full_cls_name()`

这些 metadata builder class 也不再继承 method-container。`build()`、`build_for_cudagraph_capture()`、persistent metadata helper 等方法直接写在 builder class 中。

### `metadata.py`

`metadata.py` 只保留 vLLM attention metadata dataclass 和共享 helper，不再包含动态 decorator 或 backend method-container。

它保留的核心内容包括：

- MHA plugin metadata dataclass。
- MLA plugin metadata dataclass。
- sparse MLA metadata dataclass。
- sparse indexer metadata dataclass。
- sparse/indexer metadata build helper。

已移除的旧结构包括：

- `AiterBackendDecoratorForPluginMode`
- `AiterAttentionMetadataBuilderDecoratorForPluginMode`
- `AiterMLAAttentionMetadataBuilderDecoratorForPluginMode`
- `AiterMLASparseAttentionMetadataBuilderDecoratorForPluginMode`
- `AiterMLASparseIndexerAttentionMetadataBuilderDecoratorForPluginMode`
- `vllmAiterAttentionBackendMethods`
- `vllmAiterMLABackendMethods`
- `vllmAiterMLASparseBackendMethods`
- `vllmAttentionMetadataBuilderMethods`
- `vllmMLAAttentionMetadataBuilderMethods`

### `ops.py`

`ops.py` 注册 atom-vllm 专属 attention custom op：

- `torch.ops.aiter.atom_vllm_mha_attention`
- `torch.ops.aiter.atom_vllm_mla_attention`

这两个 op 都显式设置了 `spliting_op=True`，并且加入 `CompilationConfig.set_splitting_ops_for_v1()` 默认 split op 列表，保证 ATOM 的 torch.compile graph split 能识别 atom-vllm attention 边界。

### `mla_impl.py`

`mla_impl.py` 现在保留 dense MLA vLLM 所需的模块级 helper，例如：

- `reorg_kvcache`
- FP8/FP4 GEMM helper import
- `_mla_plugin_mode_init`

dense MLA forward/helper 方法已经内联进 `AttentionForVllmMLA`。

### `mla_sparse_impl.py`

`mla_sparse_impl.py` 保留 sparse MLA 所需的模块级 helper 和 custom op：

- `fetch_id_to_ragged_triton`
- `sparse_attn_indexer_plugin_mode`
- `IndexerDecoratorForPluginMode`
- `DeepseekV32IndexerCacheDecoratorForPluginMode`
- `_mla_sparse_plugin_mode_init`

sparse MLA attention forward/helper 方法已经内联进 `AttentionForVllmMLA`。

### `attention_backend/mla_sparse.py`

该文件是 vLLM-only sparse MLA backend/indexer backend 的补充模块。它不再使用 decorator。主要提供：

- `AiterMLASparseBackend`
- `AiterMLASparseMetadataBuilder`
- `AiterMLASparseIndexerBackend`
- `AiterMLASparseIndexerMetadataBuilder`

这些类直接继承 vLLM-facing base/backend 或 `AttentionMetadataBuilder`，并显式定义 `__init__` / `build` 所需逻辑。

## 重构后的 atom-vllm 执行链路

### MHA

1. vLLM plugin entry point 注册 ATOM model wrapper。
2. ATOM model 构造时调用 frontend `Attention`。
3. `Attention` 发现当前是 vLLM mode，返回 `AttentionForVllmMHA`。
4. `AttentionForVllmMHA` 注册到 vLLM `static_forward_context`，实现 `AttentionLayerBase`。
5. vLLM 根据 layer 的 `get_attn_backend()` 获取 `AiterMhaBackendForVllm`。
6. metadata builder 使用 vLLM `common_attn_metadata` 直接构造 `AiterMhaMetadataForVllm`。
7. forward 中调用 `torch.ops.aiter.atom_vllm_mha_attention`。
8. custom op 从 vLLM forward context 中取 layer、metadata、kv_cache。
9. 直接调用 `AttentionForVllmMHA.forward_impl()` 执行 RoPE/cache、prefill/extend/decode。

### Dense MLA

1. frontend `Attention` 返回 `AttentionForVllmMLA`。
2. `AttentionForVllmMLA` 注册 vLLM static forward context 和 ATOM static forward context。
3. backend 为 `AiterMlaBackendForVllm`。
4. metadata builder 构造 `AiterMlaMetadataForVllm`。
5. forward 中先执行 q projection，再调用 `torch.ops.aiter.atom_vllm_mla_attention`。
6. custom op 直接调用 `AttentionForVllmMLA.forward_impl()`。
7. `forward_impl()` 处理 dense MLA prefill、chunked context、DCP、decode、V up projection。

### Sparse MLA

1. frontend `Attention` 发现 `mla_modules.indexer` 存在，返回 `AttentionForVllmSparseMLA`。
2. main attention backend 为 `AiterSparseMlaBackendForVllm`。
3. indexer cache 使用 `AiterSparseMlaIndexerBackendForVllm` / `AiterMLASparseIndexerBackend`。
4. indexer custom op `sparse_attn_indexer_plugin_mode` 更新 `topk_indices_buffer`。
5. `AttentionForVllmMLA.forward_impl()` 根据 `_is_sparse_mla` 分派到 `forward_impl_sparse()`。
6. sparse path 执行 q absorption、RoPE/cache、top-k page index conversion、sparse decode 和 V up projection。

## 删除和废弃的机制

本次重构删除或废弃了以下机制：

- `atom.model_ops.Attention = PagedAttention` 全局可变入口。
- `set_attn_cls()` 对 `ops.Attention` 的 mutation。
- `ATOM_DISABLE_VLLM_PLUGIN_ATTENTION`。
- `PluginConfig.vllm_use_atom_attention`。
- `AttentionForVllmFallback`。
- `atom/plugin/attention.py`
- `atom/plugin/attention_mha.py`
- `atom/plugin/attention_mla.py`
- `atom/plugin/attention_mla_sparse.py`
- `atom/plugin/moe.py` 顶层文件。
- vLLM `MLAAttention` patch 文件 `atom/plugin/vllm/mla_patch.py`。
- native backend/metadata builder/impl 上的 vLLM decorator。
- vLLM backend method-container 继承。
- vLLM metadata builder method-container 继承。
- vLLM MHA/MLA impl method-container 继承。

## 重构后的目录结构

```text
atom/model_ops/
  __init__.py                  # only exports frontend Attention
  base_attention.py            # frontend Attention dispatcher + native custom ops
  paged_attention.py           # ATOM native Attention
  attention_mha.py             # ATOM native MHA impl
  attention_mla.py             # ATOM native MLA impl
  attentions/
    aiter_attention.py         # ATOM native MHA backend/builder
    aiter_mla.py               # ATOM native MLA backend/builder

atom/plugin/
  prepare.py                   # framework mode state
  config.py                    # plugin config translation
  vllm/
    register.py                # vLLM entry points and model registry
    platform.py                # ATOMPlatform wrapper, no attention backend override
    model_wrapper.py           # ATOM model wrapper for vLLM
    moe.py                     # vLLM-only MoE naming adaptation
    attention/
      layer.py                 # AttentionForVllm, MHA/MLA/Sparse layer impl
      backend.py               # vLLM MHA/MLA backend + dense metadata builders
      metadata.py              # vLLM metadata dataclasses and shared helpers
      ops.py                   # atom_vllm attention custom ops + split markers
      mla_impl.py              # MLA helper functions
      mla_sparse_impl.py       # sparse MLA helper/indexer custom op
    attention_backend/
      mla_sparse.py            # sparse MLA backend/indexer backend
      gdn_attn.py              # GDN vLLM backend
      attention_gdn.py         # GDN vLLM impl
  sglang/
    attention.py               # AttentionForSGLang
    attention_backend/
      radix_attention.py
      sgl_attn_backend.py
      sgl_attention_mla.py
```

## 当前仍需注意的问题

- vLLM-facing metadata、backend、custom op 仍然依赖 vLLM 内部 API，如 `vllm.v1.attention.*`、`vllm.forward_context`、DCP utilities。这是 vLLM adapter 层的职责，但后续升级 vLLM 时仍需重点验证。
- `metadata.py` 和 `layer.py` 仍然较大，后续可以继续按 MHA、dense MLA、sparse MLA、indexer 进一步拆分。
- sparse MLA 是独立复杂子系统，后续应继续收敛 `attention_backend/mla_sparse.py` 和 `attention/metadata.py` 中 sparse 相关代码的边界。
- 需要单独 smoke test MHA、dense MLA、sparse MLA、GDN/hybrid、prefix cache、chunked prefill、CUDA graph 和 torch.compile piecewise。

### 2. vLLM 标准 Attention 对象产生强接口约束

`PagedAttention` 在 vLLM plugin mode 下直接构造 vLLM `Attention` 或 `MLAAttention`，这让 ATOM 必须贴合 vLLM attention layer 的构造和运行接口：

- backend 必须提供 vLLM 期望的方法，如 `get_kv_cache_shape()`、`get_supported_head_sizes()`、`full_cls_name()`、`forward_includes_kv_cache_update`、`accept_output_buffer`。
- metadata builder 必须接受 vLLM `common_attn_metadata` 形态并返回 vLLM 可消费的 `AttentionMetaData`。
- MLA 需要 patch vLLM `MLAAttention.forward_impl()` 才能把 RoPE/cache fusion 和 ATOM impl 接回来。
- KV cache spec 需要 patch，避免 vLLM 某些 MLA fp8 layout 与 ATOM layout 不兼容。
- MHA `accept_output_buffer` 必须是 `False`，因为 ATOM 需要自己控制 q/k/v/output buffer。
- positions 不能稳定通过普通 forward 参数传递，只能从 vLLM forward context 的 `additional_kwargs["atom_positions"]` 或 ATOM `static_forward_context["positions"]` 取。

这类约束说明当前集成仍然被 vLLM Attention API 强牵引。ATOM 的 attention 内部模型不是一个独立、稳定的领域接口，而是围绕 vLLM 当前版本接口补齐各种 shim。

### 3. `plugin` 文件夹边界混乱

当前 `atom/plugin` 顶层同时放了：

- framework mode 管理：`prepare.py`
- config translation：`config.py`
- vLLM/SGLang 共同 plugin glue
- MHA/MLA/sparse MLA attention impl patch
- vLLM metadata builder 和 backend method
- sparse indexer custom op
- MoE plugin patch
- `vllm/` 子包和 `sglang/` 子包

其中 `atom/plugin/attention.py` 尤其混合了太多层次：dataclass、builder、backend surface、decorator、sparse indexer、workspace allocation、unified attention wrapper 都在一个 2000+ 行文件内。

这会导致：

- 新人无法快速判断某段代码属于 vLLM adapter、ATOM common scheduling，还是 AITER kernel metadata。
- MHA、MLA、sparse MLA、sparse indexer 都穿插在一个文件内，修改某一路径时容易影响其他路径。
- framework-specific 代码和 framework-agnostic 逻辑混在一起，不利于后续扩展 SGLang 或 native path。

### 4. 两套 metadata pipeline 被揉在一起

ATOM native/server path 使用 `ScheduledBatch` + `CommonAttentionBuilder`。vLLM path 使用 vLLM common attention metadata + vLLM metadata builder interface。

当前的处理方式是：

- 在同一个 `AiterAttentionMetadataBuilder` / `AiterMLAMetadataBuilder` 类名上，通过 decorator 切换 base class 和 method set。
- `AttentionMetaData` 既包含 native fields，又通过 `plugin_metadata` 挂载 vLLM plugin mode 专用结构。
- MHA、MLA、sparse MLA 的 plugin metadata 类型不同，但消费端依赖约定访问具体字段。

后果：

- 同一个 builder 名字在不同模式下不是同一套语义。
- `plugin_metadata` 的类型边界弱，producer/consumer 之间缺少稳定 contract。
- decode slot mapping、KV indptr/indices、TBO microbatch 等逻辑在 MHA 和 MLA 中重复出现。
- `triton_mha.py`、`triton_mla.py`、`gdn_attn.py` 继承 Aiter builder，进一步把 scheduling、AITER persistent buffer、Triton path 和特殊模型 path 交织在一起。

### 5. 全局 framework flag 和 import 顺序敏感

`atom/plugin/prepare.py` 使用 `_CURRENT_FRAMEWORK` 判断 `is_vllm()`、`is_sglang()`、`is_plugin_mode()`。很多 decorator 在 class 定义/import 时读取这些 flag，并决定是否替换 class。

风险：

- 如果 import 发生在 framework 设置之前，decorator 分支可能和预期不一致。
- `register_platform()` 当前特意不设置 framework，但 `register_model()` 和 `ATOMModelBase.__init__()` 都会设置 framework，说明顺序已经是敏感点。
- 一些路径为了兼容 SGLang 扫描 vLLM entry point 做了特殊处理，增加了理解成本。

### 6. Backend 身份和注册语义不清晰

plugin mode 下多个 backend 的 `get_name()` 都返回 `"CUSTOM"`：

- `AiterBackend`
- `AiterMLABackend`
- `AiterMLASparseBackend`
- `AiterMLASparseIndexerBackend`

代码里甚至需要在 sparse MLA 时特殊 re-wrap `classmethod`，确保 `full_cls_name()` 绑定到目标 class，而不是 methods mixin class。这说明 backend identity 并不自然，需要通过 class identity 和 full class name 兜底。

### 7. 内存归属不清晰

`create_attn_metadata_builder_init_method()` 中明确记录 `extend_workspace` 是在 vLLM/SGLang memory accounting 之外分配的 GPU buffer，高 `gpu_mem_utilization` 下可能增加 OOM 风险。

类似地，MHA/MLA persistent metadata、sparse indexer buffer、chunked prefill workspace 等都散落在 builder init 或 plugin builder init 中分配。当前缺少统一的 memory plan 或 accounting facade，导致：

- vLLM 认为自己管理了 KV cache 和图捕获，但 ATOM 额外分配了大量长期 buffer。
- profile run、CUDA graph capture、dummy run 时的临时 allocation 行为分散在多个 impl 中。
- 很难从配置层估算某个模型/attention path 的完整显存占用。

### 8. vLLM 内部 API 耦合面较大

代码多处直接依赖 vLLM 内部路径或版本差异：

- `vllm.attention.layer` 和 `vllm.model_executor.layers.attention` 双 import fallback。
- `vllm.v1.attention.*` 下的 metadata、ops、common utils。
- `vllm.forward_context`。
- vLLM `MLAAttention` 私有实现形态。
- sparse path 中的 `pack_seq_triton`、`unpack_seq_triton`、top-k op 和 DCP helpers。

这会让 vLLM 升级成本很高。每次 vLLM attention 内部重构，ATOM plugin attention 都可能需要同步改动。

### 9. 部分文档和代码已经漂移

`docs/vllm_plugin_backend_guide.md` 的 lifecycle 图里仍写着 `register_platform()` 调用 `_set_framework_backbone("vllm")`，但当前 `register.py` 已经明确不在 `register_platform()` 中设置 framework。这个差异说明 attention/plugin 结构演进较快，文档容易落后。

### 10. Sparse MLA 是独立复杂系统，但目前边界不够独立

Sparse MLA 包含 main attention、indexer attention、indexer cache、custom op、top-k buffer、ragged sparse decode metadata。它不是 dense MLA 的一个小分支，而是一套额外 pipeline。

当前 sparse MLA 分散在：

- `atom/plugin/attention.py`
- `atom/plugin/attention_mla_sparse.py`
- `atom/plugin/vllm/attention_backend/mla_sparse.py`
- `atom/plugin/vllm/model_wrapper.py`
- `atom/model_ops/attentions/aiter_mla.py`
- DeepSeek model/indexer cache 相关代码

同时 sparse metadata builder 对 SGLang 直接 `NotImplementedError`。这说明 sparse MLA 应该被视为 vLLM-only 子系统，而不是继续夹在通用 plugin attention 文件里。

## 对用户已有痛点的确认

### 大量装饰器

确认。并且当前问题不只是“装饰器数量多”，而是装饰器承担了 class fabrication、base class 替换、method injection、vLLM API shim、framework 分支选择等多种职责。它已经从语法层面的装饰变成架构层面的隐式装配系统。

### vLLM 标准 attention 带来的接口硬约束

确认。`PagedAttention` 构造 vLLM `Attention` / `MLAAttention` 后，ATOM attention 必须遵守 vLLM 的 backend、metadata、KV cache、forward context、output buffer 和 graph capture contract。很多 patch 都是在弥补 ATOM internal attention 模型与 vLLM attention object 之间的差异。

### `plugin` 文件夹格局混乱

确认。当前 `plugin` 顶层没有清晰区分：

- framework lifecycle
- vLLM adapter
- SGLang adapter
- attention metadata builder
- attention impl patch
- sparse MLA indexer
- AITER kernel metadata helper

其中 `attention.py` 是最需要优先拆分的文件。

## 初步重构方向

这里先给方向，不展开为可执行代码计划。

### 方向一：拆出 vLLM adapter layer

建立清晰的 vLLM attention adapter 包，例如：

```text
atom/plugin/vllm/attention/
  backend_surface.py
  metadata_mha.py
  metadata_mla.py
  metadata_sparse_mla.py
  impl_mha.py
  impl_mla.py
  impl_sparse_mla.py
  patches.py
```

目标是让顶层 `atom/plugin/attention.py` 不再同时承载所有 framework 和 attention 类型。

### 方向二：把 metadata contract 显式化

把 `plugin_metadata` 的具体类型和 producer/consumer 关系显式化：

- MHA plugin metadata
- MLA plugin metadata
- Sparse MLA metadata
- Sparse indexer metadata

每个 metadata 类型应该有明确的构造入口和消费入口，避免在 impl 中通过约定直接访问松散字段。

### 方向三：减少动态 class fabrication

优先把 `type(...)` 动态构造 class 的地方替换为更显式的 adapter/subclass/factory。

如果短期内必须保留 decorator，可以先把 decorator 限制为薄装配层：

- 不做复杂业务逻辑。
- 不直接分配大 buffer。
- 不复制一堆 method。
- 只选择已经显式定义好的 adapter class。

### 方向四：定义 ATOM 自己的 attention domain interface

目前 ATOM attention 被 vLLM `Attention`/`MLAAttention` 接口牵着走。长期更稳的方向是定义 ATOM 内部稳定接口：

- attention layer 输入输出
- KV cache view
- metadata view
- positions/source context
- buffer ownership
- graph capture requirements

vLLM adapter 只负责把 vLLM 对象翻译成这个内部接口，而不是让所有 impl 直接理解 vLLM 内部结构。

### 方向五：把 Sparse MLA 作为 vLLM-only 子系统独立边界

Sparse MLA 当前复杂度足够高，应独立成包，至少拆分：

- main sparse attention backend
- indexer backend
- indexer cache vLLM integration
- top-k buffer and ragged metadata
- sparse kernel invocation

不要继续把 sparse indexer metadata 和 dense MLA metadata 混在同一个大文件中。

### 方向六：统一 memory plan

为 plugin attention 引入显式 memory plan 或 buffer registry：

- persistent PA metadata
- persistent MLA metadata
- extend workspace
- chunked prefill workspace
- sparse indexer buffers
- TBO microbatch buffers

让 vLLM plugin mode 下的额外显存分配可枚举、可日志化、可测试，后续再考虑与 vLLM memory accounting 对齐。

## 建议的重构优先级

1. 先做文档和边界命名：把现有链路、backend、metadata、impl patch 的职责写清楚。
2. 拆 `atom/plugin/attention.py`，保持行为不变，只按 MHA/MLA/sparse/indexer/backend surface 分文件。
3. 将 vLLM 专属 attention 入口集中到 `atom/plugin/vllm/attention/`，避免散落在 register 和 native impl 中。
4. 抽象 `plugin_metadata` 类型和 builder 输出 contract。
5. 抽 shared scheduling/helper：decode slot mapping、KV indptr/indices、TBO splitting、persistent buffer setup。
6. 再考虑替换动态 decorator 为显式 adapter class。
7. 最后再做更深的内部 attention domain interface，降低 vLLM API 对 ATOM impl 的直接牵引。

## 风险提醒

attention 重构风险较高，建议分阶段推进：

- 第一阶段只做文件拆分和 import 迁移，避免行为变化。
- 每一阶段都需要覆盖 MHA、dense MLA、sparse MLA 三条路径。
- Sparse MLA 和 DeepSeek-V3.2 indexer cache 需要单独验证。
- `ATOM_DISABLE_VLLM_PLUGIN=1` pure-vLLM 路径需要保留验证。
- vLLM 版本兼容需要有专门 smoke test，因为当前代码依赖多个 vLLM 内部 import path。

