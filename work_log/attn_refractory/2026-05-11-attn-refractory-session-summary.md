# 2026-05-11 ATOM Attention Refactory Session Summary

> 主题：从 `sglang plugin` 设计者视角，梳理 `attention backend` 重构中的 runtime / metadata / KV cache / kernel reuse 边界，并记录几种 kernel 共用 draft 的结论。

## 1. 本次会话的主线

本次讨论的核心不是继续从 `vllm plugin` 出发看问题，而是明确切换到：

- **`sglang plugin attn backend` 设计者视角**
- 关注 `sglang plugin` 自己的 runtime 语义
- 判断哪些层应该继续由 `sglang plugin` 自有
- 判断哪些层值得尽量与 `ATOM core` 共用

围绕这个目标，本次会话主要探索了 6 个问题：

1. `RadixAttention` / `radix cache` 与 `PagedAttention` 的关系
2. public `sglang` 是否也会把 radix runtime lowering 到 paged / index metadata
3. metadata 为什么会越长越多，以及哪些字段本质上属于 host/runtime
4. `kv_indptr` / `kv_indices` 为什么既绑定存储模型，又仍然能跨 host 共用
5. `sglang plugin` 和 public `sglang` 在 KV cache 管理上的异同
6. 在不大改文件结构、不做过度软件工程的前提下，怎么试探性地共用 kernel call

## 2. 对 `sglang plugin` 重构最关键的几个判断

### 2.1 不是所有问题都该叫 “attention backend 复用”

当前问题至少分成三层：

1. **host/runtime 层**
   - `ForwardBatch`
   - `RadixAttention`
   - speculative / graph / verify / extend 调度
2. **execution metadata lowering 层**
   - `page_table`
   - `kv_indptr`
   - `kv_indices`
   - `qo_indptr`
   - `kv_last_page_len`
3. **kernel / execution core 层**
   - `flash_attn_varlen_func`
   - `mla_decode_fwd`
   - `mla_prefill_fwd`
   - `pa_fwd_asm`
   - `pa_persistent_fwd`

后续任何“共用更多 code”的讨论，都应该先说明是在第几层说话。

### 2.2 `sglang plugin` 不应该为了复用去对齐到 `vllm` 的 host runtime

`vllm plugin` 和 `sglang plugin` 上层接入模型不同：

- `vllm plugin` 更像 layer-owned / metadata-builder-owned 路径
- `sglang plugin` 更像 `ForwardBatch` 驱动的 backend runtime 路径

所以：

- 不该共享 `vllm plugin` 那层 runtime facade
- 也不该强行统一大 metadata 对象
- 真正值得共享的，是更靠近 kernel 的 execution metadata 和 kernel call

### 2.3 `sglang plugin` 想共用 `ATOM` code，最现实的边界在 kernel call 一层

如果先不做大规模结构改造，最适合先共用的是：

- `mla_decode_fwd`
- `pa_persistent_fwd`
- `pa_fwd_asm`
- `flash_attn_varlen_func`

也就是：

- 保留各自的 runtime lowering
- 先把“掉 kernel 前最后一跳”抽出来

## 3. `RadixAttention` / `radix cache` 和 `PagedAttention` 的结论

### 3.1 `RadixAttention` 不是最终 kernel

`RadixAttention` 更像：

- `sglang` 模型层统一使用的 attention layer 壳
- 它知道自己跑在 `ForwardBatch` 语义里
- 最终还是委托给 `forward_batch.attn_backend`

所以它不是 “另一套底层注意力数学”，而是 **host-facing runtime abstraction**。

### 3.2 `PagedAttention` 和 `radix` 解决的问题不在同一层

- `radix cache` 更偏 prefix 命中 / 复用 / eviction
- `paged attention` 更偏 block/page 形式的物理组织和读取

可以理解成：

- `radix` 管“历史怎么共享和命中”
- `paged` 管“命中的历史最后如何变成 page/block 索引给 kernel”

### 3.3 public `sglang` 自己也在做类似 lowering

这不是 `ATOM plugin` 才有的行为。

public `sglang` 的：

- `aiter_backend`
- `triton_backend`
- `trtllm_mha_backend`
- 部分 `flashinfer` 路径

本身也会把：

- `ForwardBatch`
- `req_to_token`
- `seq_lens`

lower 成：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `page_table` / `block_tables`

所以，`radix runtime -> paged/index metadata` 不是 plugin 特例，而是公共设计模式的一部分。

## 4. Metadata 分层结论

### 4.1 现在至少有三套 metadata family

1. **`ATOM native`**
   - `AttentionMetaData`
2. **`vllm plugin`**
   - `AttentionMetaData` 外壳
   - `plugin_metadata` 内层 payload
3. **`sglang plugin`**
   - `ForwardMetadata`

### 4.2 `AttentionMetaData` 和 `plugin/attention.py` 的关系

`AttentionMetaData` 是外层通用容器；  
`plugin/attention.py` 里的很多 dataclass，是 `vllm plugin` 为表达宿主 batch/runtime 语义而塞进 `plugin_metadata` 的 payload。

所以它们不是并列关系，而是：

```text
AttentionMetaData
  └─ plugin_metadata
       ├─ flash-style plugin metadata
       ├─ MLA plugin metadata
       └─ sparse MLA plugin metadata
```

### 4.3 `ForwardMetadata` 的定位

`ForwardMetadata` 更像：

- `sglang backend` 内部的 lowering state
- 它不是最终统一 metadata
- 更不是天然可共享的 host-agnostic object

它里面混着：

- runtime control 信息
- execution metadata
- persistent kernel 预构造结果

## 5. 三类字段

为了判断能不能共享，本次把字段分成三类：

### 5.1 host/runtime-bound

典型例子：

- `vllm` 的 `query_start_loc`
- `num_decodes`
- `num_prefills`
- `chunked_context`
- `sglang` 的 `run_graph`

这些字段描述的是宿主框架如何组织 batch/runtime，而不是 kernel 怎么算。

### 5.2 execution/kernel-bound

典型例子：

- `slot_mapping`
- `block_tables`
- `page_table`
- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`
- `work_indptr`
- `reduce_indptr`

这些字段才是 kernel 真正需要消费的执行信息。

### 5.3 hardware-sensitive

典型例子：

- `max_q_len`
- `max_kv_len`
- `num_kv_splits`
- `head_dim`
- `attn_out_dtype`

这些字段不是 host 语义，但会影响：

- page/block 解释
- kernel 选路
- workspace/split 策略

## 6. 关于 `kv_indptr` / `kv_indices` 的关键结论

这个问题本次单独做了较多背景解释，结论是：

### 6.1 它们不是完全 storage-agnostic

因为它们绑定的是某种 **可索引的 KV 地址空间**：

- token slot index
- page/block index
- compacted KV index

所以它们不能脱离底层存储模型独立存在。

### 6.2 但它们可以是 host/runtime-agnostic

因为它们不表达：

- `ForwardBatch.forward_mode`
- `query_start_loc`
- prefix 命中策略
- graph/speculative 的宿主控制语义

它们只表达：

- 这次 kernel 要去哪些 KV 地址读数据
- 每个 request 的边界在哪里

### 6.3 两个例子

#### 例子 A：`sglang`

- `req_to_token[req_id, token_pos] = slot_id`
- lowering 后得到：
  - `kv_indptr = [0, 10]`
  - `kv_indices = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5]`

#### 例子 B：`vllm/ATOM`

- 上层是 `block_table`
- 展开后同样可得到：
  - `kv_indptr = [0, 10]`
  - `kv_indices = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5]`

这说明：

- 上层来源不同
- 但 lower 后的 execution metadata 可以收敛

## 7. public `sglang` 与 `sglang plugin` 在 KV cache 管理上的异同

### 7.1 相同点

- owner 都还是 public `sglang`
  - `ReqToTokenPool`
  - `TokenToKVPool`
  - `KVCache`
  - `radix_cache`
- request 到 slot 的映射机制没变
- prefix / radix 复用体系没变

### 7.2 不同点

#### A. MHA 写 cache 路径不同

public `sglang` 更多沿用 pool 提供的标准写入接口：

- `token_to_kv_pool.set_kv_buffer(...)`

而 `sglang plugin` 在 MHA 路径里会绕过标准写法，自己做一次：

- `set_kv_buffer_with_layout_shuffle(...)`
- 把同一块 public pool buffer 按更偏 `ATOM` kernel 的 SHUFFLE layout 写进去

#### B. MLA 路径更接近 public `sglang`

MLA 上，plugin 更多沿用 public pool 的 contract：

- `token_to_kv_pool.set_kv_buffer(...)`
- `get_key_buffer(...)`
- `get_value_buffer(...)`

#### C. Lowering 目标不同

- public `sglang` 的 metadata 更通用
- plugin 的 `ForwardMetadata` 更偏 `ATOM` kernel
  - `page_table`
  - `pa_metadata_*`

### 7.3 一个更准确的说法

`sglang plugin` **没有改写 public `sglang` 的 pool owner / allocator / radix tree**，  
但它在 MHA 路径上：

- 改变了同一块 public pool buffer 的布局解释与写入约定
- 并把 lowering 结果继续朝 `ATOM` kernel 所需的 metadata 形态推进

## 8. Kernel 共用的两种 draft

本次会话里，实际做了两种方向的 draft。

### 8.1 方案 A：共享 kernel wrapper

思路：

- 新增共享小 metadata
  - `MLADecodeKernelMetadata`
  - `MHAAsmKernelMetadata`
  - `MHAPersistentKernelMetadata`
- 新增共享 wrapper
  - `run_mla_decode_kernel`
  - `run_mha_asm_kernel`
  - `run_mha_persistent_kernel`

优点：

- 边界清晰
- 更像长期可演进的 execution API
- 不需要统一大 metadata

缺点：

- 还是引入了一层新的 shared contract

### 8.2 方案 B：`sglang plugin` 主动适配 `ATOM core`

思路：

- 不改 `ATOM core`
- 在 `sglang plugin` 里把 `ForwardMetadata` 转成 `AttentionMetaData`
- 再伪造最小 `shim/self`
- 直接调用：
  - `PagedAttentionImpl.paged_attention_asm`
  - `PagedAttentionImpl.paged_attention_persistent_asm`
  - `MLAAttention._forward_decode`

优点：

- 不改 `ATOM core`
- 更直观地证明 “plugin 主动复用 core” 是可行的

缺点：

- 需要 shim
- 尤其 MLA 路径更脆弱
- 更适合验证可行性，不适合长期保留

## 9. 当前最值得保留的判断

### 9.1 不应该强行统一大 metadata

不建议直接统一：

- `AttentionMetaData`
- `plugin_metadata`
- `ForwardMetadata`

因为它们都带有较重的宿主 runtime 语义。

### 9.2 最值得共享的是 kernel-facing execution metadata

真正最有价值的共享边界是：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `page_table`
- `context_lens`
- `work_*`
- `reduce_*`

以及基于它们的 kernel call / runner。

### 9.3 重构顺序建议

如果继续推进 `sglang plugin` 重构，更合理的顺序是：

1. 保持 `sglang` runtime 自有
2. 显式化 metadata lowering 边界
3. 尽量共享 kernel call / execution helper
4. 最后再考虑是否继续上推到 runner 层

## 10. 一句话总结

本次会话最终收敛出的关键认识是：

**`sglang plugin attention` 重构的关键，不是把 `sglang` runtime 改得更像 `vllm`，而是把 runtime、metadata lowering、kernel execution 这三层拆清楚；只要 lowering 之后能在 execution metadata 上对齐，就能在不统一宿主语义的前提下，共用更多 `ATOM` 的底层 attention 代码。**
