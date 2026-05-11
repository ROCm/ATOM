# 2026-05-08 KV Indices and Address Space Notes

> 预估阅读时间：8-10 分钟  
> 主题：解释 `kv_indptr` / `kv_indices` 为什么不是纯 host metadata，也不是完全 storage-agnostic 的抽象，并用 `sglang` / `ATOM` / `vllm plugin` 的实际 KV cache 存储举例说明。

## 1. 想回答的问题

最近有一个很容易卡住重构讨论的问题：

> `kv_indptr` / `kv_indices` 看起来和 KV cache 的存储方式深度绑定。  
> 既然如此，为什么还能把它们看成 `sglang` / `ATOM` / `vllm plugin` 之间可以共享的 metadata？

这个问题的关键在于区分两件事：

1. **host/runtime 是否相同**
2. **kernel 最终消费的地址空间模型是否相同**

一句话先说结论：

**`kv_indptr` / `kv_indices` 不是完全 storage-agnostic 的抽象，它们绑定的是某种“可索引的 KV 地址空间”；  
但它们可以是 host/runtime-agnostic 的，只要不同宿主最终都能把自己的 runtime state lowering 到同一种地址空间模型。**

## 2. 三层视角

理解这个问题，最好先把 attention 相关代码拆成三层：

### 2.1 host/runtime 层

这一层描述的是宿主框架如何组织请求和调度，例如：

- `ATOM native` 的 `ScheduledBatch`
- `vllm plugin` 的 `query_start_loc` / `num_prefills`
- `sglang plugin` 的 `ForwardBatch` / `forward_mode` / `spec_info`

这一层表达的是：

- 哪些请求在 batch 中
- 哪些是 decode，哪些是 prefill
- prefix / speculative / graph 的语义是什么

### 2.2 execution metadata lowering 层

这一层做的事情是：

- 把 host/runtime 里的 batch 状态
- 翻译成 kernel 真正能消费的索引结构

典型字段：

- `block_tables` / `page_table`
- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`

### 2.3 kernel / execution core 层

这一层只关心：

- query tensor
- KV cache tensor
- index / indptr / page table
- persistent kernel workspace

例如：

- `mla_decode_fwd`
- `mla_prefill_fwd`
- `pa_fwd_asm`
- `pa_persistent_fwd`

## 3. `sglang` 的实际 KV cache 存储

在 `sglang` 里，KV cache 本身不是存在 radix tree 里。  
radix tree 负责的是 prefix 复用与命中；真正的 KV 还是存在 memory pool 里。

`sglang` 自己的注释说得很清楚：

- `ReqToTokenPool` 负责 request -> token location 映射
- `TokenToKVPoolAllocator` 负责 KV cache index 管理
- `KVCache` 真正持有物理 K/V tensor

也就是说，`sglang` 的 runtime 世界里至少有两层：

1. **逻辑视角**：某个 request 的第 `i` 个 token 属于哪里
2. **物理视角**：这个 token 对应的 K/V 存在物理 pool 的哪个 slot

### 3.1 `ReqToTokenPool`

`ReqToTokenPool` 本质上是一张二维表：

```text
req_to_token[req_id, token_pos] = physical_slot_id
```

它回答的是：

- 某个 request 的第 `j` 个逻辑 token
- 对应到物理 KV pool 里的哪个 slot

### 3.2 物理 KV pool

物理 pool 里，K 和 V 都是按 slot 存的。  
写入 KV 时，最终就是按 `loc/indices` 写入：

```text
k_cache[indices] = k
v_cache[indices] = v
```

所以从 kernel 角度看，最终它看到的是：

- 一个可以索引的 KV 地址空间
- 一组 slot index

而不是看到“radix tree”本身。

## 4. `ATOM` / `vllm plugin` 的实际 KV cache 存储

`ATOM` / `vllm plugin` 更偏 paged/block 风格。

典型思路是：

- 每个 request 有一张 `block_table`
- 每个 block 对应一个固定大小的 page
- kernel 按 `block_table` 或其展开后的 index 去读 KV

所以它的上层直觉更像：

```text
request -> block table -> page/block address -> physical KV
```

和 `sglang` 相比，最大的不同不是“有没有物理地址空间”，而是：

- `sglang` 上层先经过 `req_to_token`
- `vllm/ATOM` 上层先经过 `block_table`

但两边最终都能 lower 到一组可供 kernel 读取的地址索引。

## 5. 例子一：MLA decode 中的 `kv_indptr` / `kv_indices`

这个例子里，可以把 `kv_indices` 理解为 **token-slot index**。

### 5.1 在 `sglang` 里

假设一个 request 当前长度是 10，  
它在物理 KV pool 里的 slot 顺序不是连续的，而是：

```text
[8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
```

那 `req_to_token[req_id, :10]` 就大致表示：

```text
logical token position: 0  1   2   3   4  5  6  7  8  9
physical slot id:      8  9  10  11   0  1  2  3  4  5
```

如果这个 batch 只有这一个 request，那么 lowering 之后可以得到：

```text
kv_indptr = [0, 10]
kv_indices = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
```

这里：

- `kv_indptr` 表示“第 0 个 request 的 KV 范围是 `[0, 10)`”
- `kv_indices` 表示“真正去物理 KV pool 里读这些 slot”

### 5.2 在 `vllm/ATOM` 里

如果 page size 是 4，同样 10 个 token 对应的 block layout 可以写成：

```text
block 2 -> slots [8, 9, 10, 11]
block 0 -> slots [0, 1, 2, 3]
block 1 -> slots [4, 5, 6, 7]
```

block table 大致就是：

```text
[2, 0, 1]
```

展开前 10 个 token 后，对应的 slot 顺序仍然是：

```text
[8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
```

于是 lower 之后，给 MLA decode kernel 的：

```text
kv_indptr = [0, 10]
kv_indices = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
```

和 `sglang` 这边是等价的。

### 5.3 这个例子说明什么

说明在 MLA decode 这个路径里：

- `kv_indices` 绑定的是 **token-slot 地址空间**
- 它不是“完全与存储无关”
- 但它已经不关心 slot 列表最初是由 `req_to_token` 还是 `block_table` 推出来的

也就是说，**它和具体宿主无关，但和被选定的地址空间模型有关。**

## 6. 例子二：MHA persistent decode 中的 `page_table` / `kv_indices`

这个例子里，`kv_indices` 更接近 **page/block index**，而不是 token-slot index。

### 6.1 在 `sglang plugin` 里

仍然假设 page size = 4。  
如果某个 request 的 token-slot 顺序是：

```text
[8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
```

那么每页的起始 slot 是：

```text
[8, 0, 4]
```

除以 page size 后，对应 page id：

```text
[2, 0, 1]
```

在 `sglang plugin` 里，这就是 `page_table` 的来源：  
从 `req_to_token` 按页采样，再除以 `page_size`。

然后 `_build_pa_metadata_for_decode()` 再把它变成 `pa_persistent_fwd` 所需的 page-level `kv_indices`。

### 6.2 在 `ATOM` / `vllm` 里

这一层本来就是 paged/block 视角，所以 `block_table` 原生就已经是：

```text
[2, 0, 1]
```

也就是说：

- `sglang` 是 `req_to_token -> page_table`
- `vllm/ATOM` 是直接 `block_table -> page_table`

但到了 persistent kernel 眼里，最后看到的是同一种 page/block 地址空间。

### 6.3 这个例子说明什么

说明在 MHA persistent decode 这个路径里：

- `kv_indices` 绑定的是 **page/block 地址空间**
- 它仍然不是“完全 storage-agnostic”
- 但也不需要知道上层 host 是 `sglang` 还是 `vllm`

只要两边都能 lower 到同一个 page/block 地址空间，kernel 就能共享。

## 7. 这两个例子真正说明的事

上面两个例子其实在说明同一个结论：

### 7.1 `kv_indptr` / `kv_indices` 不是 host metadata

它们不描述：

- `ForwardBatch.forward_mode`
- `query_start_loc`
- `num_prefills`
- `run_graph`
- prefix hit 的高层语义

它们只描述：

- 当前这次 kernel 调用
- 要读哪些 KV 地址
- 每个 request 的边界在哪里

所以它们更接近：

- execution metadata
- kernel-facing metadata

而不是 host/runtime metadata。

### 7.2 但它们也不是完全 storage-agnostic

它们依赖：

- 当前 KV cache 的地址空间模型
- kernel 对该地址空间的解释方式

如果底层 cache 不是：

- 可线性索引的 token slot
- 或可 flatten 的 page/block index

那 `kv_indptr` / `kv_indices` 这套抽象就未必成立。

所以它们不是“无限通用”的。

## 8. 对重构的直接启示

这个背景知识对当前 `sglang attention` 重构最有价值的结论是：

### 8.1 不要去统一 host metadata

比如不要强行统一：

- `ForwardBatch`
- `query_start_loc`
- `num_prefills`
- `run_graph`

这些都属于宿主框架自己的 runtime 语言。

### 8.2 应该争取统一 execution metadata

例如：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `page_table`
- `kv_last_page_len`
- `work_indptr`
- `reduce_indptr`

以及围绕这些字段构造的更小 dataclass，例如当前 draft 里已经开始尝试的：

- `MLADecodeKernelMetadata`
- `MHAAsmKernelMetadata`
- `MHAPersistentKernelMetadata`

### 8.3 正确的共享边界

所以真正可共享的不是：

- `sglang` 的 `ForwardMetadata`
- `vllm plugin` 那一整套 metadata family

而是：

- 它们再往下切出来的
- kernel-facing execution metadata

## 9. 一句话总结

**`kv_indptr` / `kv_indices` 不是“和存储彻底无关”的抽象，它们绑定的是某种被 kernel 接受的 KV 地址空间；但正因为它们已经脱离了宿主 batch/runtime 语义，`sglang` 和 `vllm/ATOM` 即使上层完全不同，也仍然可以在这层 metadata 上收敛并共享 kernel 调用逻辑。**
