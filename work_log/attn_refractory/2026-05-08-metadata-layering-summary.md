# 2026-05-08 Metadata Layering Summary

## 1. 目的

这份笔记简要总结 `ATOM native`、`vllm plugin`、`sglang plugin` 三边 metadata 的关系，重点回答：

- `ATOM/atom/utils/forward_context.py` 中的 `AttentionMetaData` 是什么
- `ATOM/atom/plugin/attention.py` 里的 metadata dataclass 在做什么
- `ATOM/atom/plugin/sglang/attention_backend/sgl_attn_backend.py` 里的 `ForwardMetadata` 属于哪一层
- 哪些 metadata 值得共享，哪些不值得强行统一

## 2. 三套 metadata 的定位

### 2.1 `AttentionMetaData`

文件：`ATOM/atom/utils/forward_context.py`

它是 `ATOM` attention 执行链路里的**通用外层容器**。  
里面既有：

- 通用 attention 字段
  - `slot_mapping`
  - `block_tables`
  - `kv_indptr`
  - `kv_indices`
  - `work_meta_data`
- 也预留了 plugin 扩展入口
  - `plugin_metadata`

所以它更像一个大容器，而不是某个 host 专属的数据模型。

### 2.2 `vllm plugin metadata`

文件：`ATOM/atom/plugin/attention.py`

`vllm plugin` 不是重写一套完全独立的 metadata，而是在 `AttentionMetaData` 外壳上，额外挂了一层 `plugin_metadata` payload。

代表类型包括：

- `AiterFlashAttentionMetadataForPluginMode`
- `AiterMLACommonMetadataForPluginMode`
- `AiterMLADecodeMetadataForPluginMode`
- `AiterMLASparseMetadataForPluginMode`

这些 dataclass 主要承载：

- `vllm` 的 batch/runtime 语义
- decode/prefill/extend phase 拆分
- chunked prefill / DCP / sparse MLA 等 feature-specific 信息

一句话：**`vllm plugin` 是“外层复用 `AttentionMetaData`，内层自带一套 host-specific metadata family”。**

### 2.3 `sglang plugin` 的 `ForwardMetadata`

文件：`ATOM/atom/plugin/sglang/attention_backend/sgl_attn_backend.py`

`ForwardMetadata` 不是 `AttentionMetaData.plugin_metadata` 那种 payload，而是 `sglang backend` 内部的**lowering 结果缓存对象**。

它更接近：

- `ForwardBatch`
- `req_to_token`
- `token_to_kv_pool`
- graph/speculative path

向 kernel metadata 的过渡层。

一句话：**`ForwardMetadata` 更像 backend-local lowering state，而不是 host-agnostic 的最终执行 metadata。**

## 3. 三类字段

### 3.1 host/runtime-bound

这类字段描述宿主框架如何组织 batch/runtime，而不是 kernel 如何计算。

例子：

- `vllm plugin`
  - `query_start_loc`
  - `num_decodes`
  - `num_prefills`
  - `chunked_context`
- `sglang plugin`
  - `run_graph`
  - 各种 `ForwardBatch.forward_mode` 派生分支语义

### 3.2 execution/kernel-bound

这类字段最接近 kernel 真正需要的执行信息。

例子：

- `slot_mapping`
- `block_tables`
- `context_lens`
- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`
- `work_indptr`
- `reduce_indptr`

### 3.3 hardware-sensitive

这类字段和 page size、dtype、kernel 选择、并行配置相关。

例子：

- `max_q_len`
- `max_kv_len`
- `num_kv_splits`
- `attn_out_dtype`
- `head_dim`

## 4. 哪些值得共享

### 不建议强行共享的

- `AttentionMetaData` 整体
- `vllm plugin` 整套 `plugin_metadata`
- `sglang plugin` 整体的 `ForwardMetadata`

原因不是它们没价值，而是它们都带着很重的 host/runtime 语义。

### 值得重点共享的

应该共享的是更小的 **execution metadata view**，例如当前 draft 已经开始尝试的：

- `MLADecodeKernelMetadata`
- `MHAAsmKernelMetadata`
- `MHAPersistentKernelMetadata`

这些结构只保留某个 kernel call 真正需要的字段，更适合在：

- `ATOM core`
- `vllm plugin`
- `sglang plugin`

之间共享。

## 5. 一句话结论

现在不是三边共用“一套 metadata”，而是三边各自都有一层 host-owned metadata / lowering 体系。  
真正值得共享的，不是这些大 metadata 本身，而是从它们里面切出来的、更小的、kernel-facing execution metadata。
