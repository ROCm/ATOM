# 2026-05-07 Radix Cache vs Paged Attention Notes

> 预估阅读时间：15 分钟  
> 主题：梳理 `SGLang` 中 `radix cache` / `RadixAttention` 与 `ATOM` / `vLLM` 常见的 `paged attention` 之间的关系，解释为什么它们最终可以落到相同的 kernel，以及这件事对 `sglang plugin` attention backend 重构意味着什么。

## 1. 这篇笔记想回答什么

围绕 `sglang plugin attn backend` 的重构，最近反复出现了几个问题：

1. `SGLang` 用的是 `RadixAttention`，`ATOM`/`vLLM plugin` 里常见的是 `PagedAttention`，两者是不是两套完全不同的注意力体系？
2. 如果它们真的不同，为什么在 `DeepSeek MLA` 这类路径里，最终又会调用到相同的底层 kernel？
3. 既然 `SGLang` 还多了一层 `radix cache` 前缀管理，为什么没有一个非常直观的结论说 "`SGLang` 一定比 `vLLM` 更快 / 更省显存"？
4. 对当前重构来说，`radix` / `paged` 的差异到底应该被归类到：
   - host runtime 差异
   - KV cache 管理差异
   - metadata lowering 差异
   - kernel 差异
   的哪一层？

本文尝试把这几个问题放进同一个分析框架里。

## 2. 先给结论

### 2.1 最核心的一句话

`RadixAttention` 和 `PagedAttention` 主要不是在底层注意力数学上不同，而是在 **runtime 如何管理、共享、命中和定位历史 KV** 上不同。  
一旦 runtime 把历史 KV lowering 成底层 kernel 能吃的 `page_table`、`kv_indptr`、`kv_indices`、`qo_indptr` 之类的 metadata，kernel 就不再关心这些索引最初是来自 radix tree 还是来自 paged block table。

### 2.2 另一句更工程化的结论

对重构来说，不应该把 “`RadixAttention` vs `PagedAttention`” 当作 “能不能共享底层执行实现” 的直接判断依据。  
更应该把系统拆成三层：

1. **host/runtime 层**  
   例如 `ForwardBatch`、`RadixAttention`、`ModelRunner`、prefix cache、speculative 调度
2. **execution metadata lowering 层**  
   例如 `page_table`、`block_tables`、`kv_indptr`、`kv_indices`、`qo_indptr`
3. **kernel / execution core 层**  
   例如 `pa_fwd_asm`、`pa_persistent_fwd`、`flash_attn_varlen_func`、`mla_decode_fwd`、`mla_prefill_fwd`

`radix` 和 `paged` 的差异，主要在第 1、2 层；  
兼容性和共享机会，主要出现在第 2、3 层。

## 3. 为什么名字容易让人误解

当前最容易造成误解的点是：

- `PagedAttention` 这个名字听起来像“最终执行算法”
- `RadixAttention` 这个名字也听起来像“最终执行算法”

但在工程上，它们都更接近 **host-facing attention runtime abstraction**，而不是最终 kernel 名字。

更具体一点：

- `PagedAttention` 更强调 **按固定 page/block 管理 KV cache**
- `RadixAttention` 更强调 **按 prefix/radix tree 管理请求历史与前缀复用**

这两者都不是在说：

- `QK^T` 怎么算
- softmax 怎么做
- decode kernel 用哪套 asm/triton/flashinfer

这些底层执行问题，是更下一层的 backend / kernel 决定的。

## 4. `RadixAttention` 到底是什么

### 4.1 `RadixAttention` 不是 kernel

在 public `SGLang` 里，`RadixAttention` 本身并不直接定义底层 kernel 选择。  
它更像是：

- SGLang 模型层里统一使用的 attention layer 壳
- 它知道自己在 `ForwardBatch` 语义里运行
- 它把真正的执行委托给 `forward_batch.attn_backend`

因此，`RadixAttention` 更像一个 **layer/runtime 入口点**。

### 4.2 `radix` 的重点是 prefix 管理

`radix cache` 的本质，是一棵 prefix tree，用来做：

- 前缀命中
- 前缀复用
- unfinished / finished request 的缓存插入
- 基于 prefix 的 eviction / lifecycle 管理

它回答的问题更像：

- 这个请求的历史前缀已经缓存到哪里了？
- 哪一部分可以直接复用，不必重新 prefill？
- 哪些 KV slot/page 仍然属于共享前缀？
- 哪些历史片段可以回收？

所以 `radix` 优化的主要是 **prefix reuse**，而不是单次 attention kernel 的算力效率。

### 4.3 `radix cache` 在 SGLang 里不是完全“反 page”的

这一点很重要。public `SGLang` 的 `radix_cache` 本身就已经带有 page-aware 的逻辑。

例如它有 page 对齐：

- `page_align_keys()`

也有按 page 粒度匹配 prefix：

- `_key_match_paged()`

这意味着：

- `radix cache` 解决的是前缀共享与命中问题
- 但它并不排斥最终按 page/block 粒度来表达缓存

也就是说，**radix tree 的逻辑管理** 与 **paged/block 的物理表达** 并不冲突。

## 5. `PagedAttention` 到底是什么

### 5.1 `PagedAttention` 的重点不是 prefix tree

`PagedAttention` 更强调：

- KV cache 被组织成固定大小的 page/block
- 每个请求通过 `page_table` / `block_table` 找到自己历史对应的 page
- kernel 依照这些 page/block 索引读取历史 KV

它回答的问题更像：

- 当前请求历史有哪些 block/page？
- 它们在物理内存中的位置是什么？
- 这些 page 如何映射到 kernel 所需的 block table？

所以 `paged attention` 更偏：

- **物理存储布局**
- **kernel 读取 KV 的地址组织方式**

### 5.2 `PagedAttention` 天生不等于 prefix reuse

`paged attention` 当然也可以配合 prefix cache 使用，但 prefix reuse 并不是它名字里最核心的概念。  
它的核心价值更在于：

- 支持非连续物理布局
- 让 decode kernel 可以按 page/block 高效读取 KV

## 6. 为什么 `radix` 和 `paged` 最后可以兼容

### 6.1 因为它们主要解决的是不同层的问题

可以把它们看成：

- `radix`: 逻辑历史管理器
- `paged`: 物理页表表达方式

两者关系有点像：

- `radix` 负责决定“哪些历史是共享的、哪些是命中的、请求当前逻辑历史是什么”
- `paged` 负责决定“这些历史最终以什么 page/block 索引形式交给 kernel”

只要 runtime 最终能把逻辑历史翻译成 page/block 或 indptr/index 形式，底层 kernel 并不需要知道前缀最初是如何组织的。

### 6.2 兼容发生在 lowering 之后

这一点非常关键：

上层看起来是：

- `ForwardBatch`
- `RadixAttention`
- `req_to_token`
- prefix cache

但 lower 到 backend 之后，会变成执行态 metadata，例如：

- `page_table`
- `block_tables`
- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`

到了这个层次，kernel 只看：

- query 的 shape
- KV cache 的 shape
- 历史长度
- 索引表 / indptr

它不看：

- 这套索引是不是来自 radix tree
- 是不是来自某个 prefix cache 命中
- 是不是通过 `PagedAttention` 对象构造出来的

所以兼容性不是“名字兼容”，而是 **execution metadata 兼容**。

## 7. public `SGLang` 自己就在做类似 lowering

这不是 `ATOM plugin` 独有的现象。public `SGLang` 本身就在这么做，只是不同 backend lower 到的 metadata 形态略有不同。

### 7.1 `aiter_backend`

public `SGLang` 的 `aiter_backend` 会从：

- `forward_batch.seq_lens`
- `forward_batch.req_pool_indices`
- `req_to_token`

构造：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`

也就是说，它会把 host runtime 的历史表示 lower 成 AITER kernel 所需的执行态索引。

### 7.2 `triton_backend`

public `SGLang` 的 `triton_backend` 也做类似事情。  
它同样维护：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`

然后再调用 Triton 的 decode/extend kernel。

### 7.3 `trtllm_mha_backend` / 部分 `flashinfer` 路径

这类 backend 更偏向：

- 直接构造 `page_table`
- 或 `block_tables`

再喂给 flashinfer / TRTLLM 风格的 kernel。

所以更准确地说，public `SGLang` 的共同模式不是：

> “所有 backend 都把 radix runtime 翻成 paged attention”

而是：

> “所有 backend 都会把 radix/runtime 层的历史表示，lower 成各自 kernel 能吃的索引/页表 metadata”

其中有的偏 `kv_indptr/kv_indices`，有的偏 `page_table/block_tables`。

## 8. 从 kernel 角度看，为什么可以兼容

### 8.1 kernel 真正关心什么

绝大多数 attention kernel 真正关心的是：

1. 当前 query 的排列方式
2. 每个请求可见的历史 KV 长度
3. 如何从某个索引结构里拿到历史 KV 的物理地址
4. page/block 大小、stride、layout
5. 是否需要 prefix/extend/speculative 的特殊 metadata

它通常不关心：

1. 这份索引最初是不是从 prefix tree 查出来的
2. 命中前缀的逻辑是谁做的
3. runtime 是 `RadixAttention` 还是 `PagedAttention`

### 8.2 对 MHA kernel 的影响

MHA decode 常见地会落到：

- `pa_fwd_asm`
- `pa_persistent_fwd`
- `flash_attn_varlen_func`

这些 kernel 更偏 page/block 或 varlen index 风格。  
只要 runtime 最终给出：

- `page_table`
- `context_lens`
- `block_tables`

或者：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`

它们都能算。

### 8.3 对 MLA kernel 的影响

`DeepSeek MLA` 更容易看出这种兼容性，因为 MLA 的底层算子路径更明确。

典型 kernel 包括：

- `mla_decode_fwd`
- `mla_prefill_fwd`
- `concat_and_cache_mla`
- `fused_qk_rope_concat_and_cache_mla`

这些 kernel 只要求：

- 压缩后的 latent KV 表示
- 以及相应的 `qo_indptr` / `kv_indptr` / `kv_indices`

所以无论上层 host 是：

- `ATOM native`
- `ATOM vLLM plugin`
- `ATOM SGLang plugin`

只要最终 lower 成相同的 MLA execution metadata，就自然会落到同一套 MLA kernel。

## 9. `DeepSeek MLA` 为什么尤其容易“看起来收敛”

### 9.1 因为 MLA 的 execution core 更专用

`DeepSeek MLA` 的真正执行对象不是：

- “`RadixAttention` 版本的 MLA”
- “`PagedAttention` 版本的 MLA”

而是：

- 一组固定的 MLA latent/cache 表示
- 一套固定的 rope + kv cache 写入方式
- 一套固定的 prefill/decode kernel

换句话说，MLA 的底层 core 更容易抽成“共享执行层”。

### 9.2 因为 host 差异主要体现在 metadata 准备与调度

对于 `DeepSeek MLA` 来说，上层 host 差异更多体现在：

- `positions` 从哪里来
- `forward_batch` 怎么组织
- decode / extend / speculative / target_verify 怎么分流
- `req_to_token` 怎么转换成 `kv_indices`

一旦进入 `mla_decode_fwd` 或 `mla_prefill_fwd` 前，这些差异已经被消化掉了。

所以从外部观察，很容易得到一种印象：

> “怎么上面完全不一样，下面却是同一套 kernel？”

其实并不奇怪，因为两边只是上层 runtime 不同，而 MLA execution core 本来就在下层收敛。

## 10. 既然兼容，为什么 `SGLang` 没有天然更快或更省显存

这是另外一个很容易误判的问题。

### 10.1 `radix cache` 省的是前缀重复，不是所有 KV

`radix cache` 能节省的，是重复前缀带来的重复 prefill 和重复 KV 占用。

它主要帮助的是：

- 大量请求共享长前缀
- 同 prompt 多分支采样
- prefix-heavy workload
- 某些 speculative / chunked prefill 场景

如果 workload 不满足这些条件，它就不会天然表现为显著优势。

### 10.2 它优化的是 prefix reuse，不是单 token decode kernel

radix 管理层并不会让 `mla_decode_fwd`、`pa_fwd_asm` 这种 kernel 自己更便宜。  
kernel 的 raw 速度主要还是受：

- 头数
- head dim
- KV layout
- quant
- page size
- GPU kernel 实现

这些因素决定。

所以常见现象是：

- prefix-heavy workload 下，`SGLang` 的收益更明显
- decode-heavy workload 下，收益没那么明显
- 如果 `vLLM` 也有自己的 prefix caching/block caching，差距会进一步缩小

### 10.3 radix 自己也有管理成本

`radix cache` 不是免费层。它也有：

- prefix match
- tree split / insert / eviction
- request -> KV slot 映射维护
- speculative / unfinished request 的额外 bookkeeping

如果命中率不高，这部分成本并不会自动转化成收益。

## 11. 对当前重构最有价值的判断

### 11.1 不要把 `RadixAttention` vs `PagedAttention` 当成是否共享 execution core 的边界

因为从 public `SGLang` 和当前 `ATOM plugin` 的代码路径看，二者最终都在做类似事情：

- 从 host runtime 的历史表示出发
- 构造底层 kernel 所需的 execution metadata

所以：

- runtime 不同，不代表 execution core 必须不同
- 名字不同，不代表 kernel 层必须分叉

### 11.2 真正的边界更适合这样划

#### A. `sglang plugin runtime` 应拥有的部分

- `ForwardBatch`
- `RadixAttention`
- prefix cache / radix cache
- `ModelRunner` / `attn_backend_wrapper` 相关调度
- speculative / graph / verify / chunked prefill 等宿主语义

#### B. execution metadata lowering 层

这层是最值得重新定义的边界。它负责：

- 把 `req_to_token`、prefix 命中结果、cache state
- 转成 `page_table` / `kv_indptr` / `kv_indices` / `qo_indptr`

这层既带有 host 语义，又已经开始接近共享执行核心，是未来最值得梳理清楚的一层。

#### C. shared execution core

例如：

- `mla_decode_fwd`
- `mla_prefill_fwd`
- `pa_fwd_asm`
- `pa_persistent_fwd`
- `flash_attn_varlen_func`

以及与这些 kernel 紧耦合的那部分 ATOM core helper。

### 11.3 这对 `sglang plugin` 的重构意味着什么

当前 `sglang plugin` 最不该做的是：

- 因为底层 kernel 能共享，就把 `sglang` 的 runtime seam 也强行对齐到 `vllM`
- 或者因为上层叫 `RadixAttention`，就认为它和 `PagedAttention` 完全不能共享底层执行实现

更合理的重构方向是：

1. 保持 `sglang` 的 host/runtime 语义完整
2. 识别 runtime lowering 层的清晰边界
3. 尽量把 execution core 继续下沉回共享 `ATOM core`

## 12. 一种对外更容易讲清楚的表述

如果后续要向别人解释，可以用下面这几句话：

### 12.1 关于 `radix` vs `paged`

`radix cache` 主要解决的是 prefix 命中与复用，`paged attention` 主要解决的是 page/block 形式的 KV 组织与读取。  
前者偏逻辑管理，后者偏物理表达；两者不是同一层次的问题。

### 12.2 关于为什么最终会调到同一个 kernel

因为 attention kernel 只关心最终的执行态 metadata，比如 `page_table`、`kv_indptr`、`kv_indices`、`qo_indptr`。  
一旦 runtime 把历史 KV lowering 成这类形式，kernel 就不再关心这些索引最初是由 radix tree 还是 paged runtime 生成的。

### 12.3 关于为什么 `SGLang` 没有天然显著更快

因为 `radix` 带来的主要收益是 prefix reuse，而不是单 token decode kernel 的 raw speed。  
如果 workload 不是 prefix-heavy，或者 decode 占主要成本，那么 radix 带来的优势不会自然放大成“总是更快 / 更省显存”。

## 13. 最终总结

把这次探索收敛成一句话：

**`RadixAttention` 与 `PagedAttention` 的主要差异，不在底层注意力数学，也不一定在最终 KV 的物理表示本身，而在 host runtime 如何管理、共享、命中并 lowering 历史 KV；一旦 lowering 完成，底层 kernel 完全可以是共享的。**

对当前 `sglang plugin attn backend` 的重构来说，真正值得做的不是争论 “`radix` 和 `paged` 能不能统一成一个名字”，而是把系统拆清楚：

- 哪些是 `sglang` 必须自有的 runtime / prefix cache / scheduling 语义
- 哪些是 execution metadata lowering
- 哪些已经是可以共享的 `ATOM` execution core

这比直接讨论 “`radix` 和 `paged` 谁更先进、谁应该替代谁” 更接近真正的工程问题。
