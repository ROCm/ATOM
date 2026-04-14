# 2026-04-08 DeepSeek Speculative 与 Attention Metadata 关系笔记

## 文档目的

本文专门解释一个在调试 DeepSeek speculative / MTP 时非常关键、但又很容易被低估的问题：

- **speculative decoding 和 attention metadata 到底是什么关系？**

对于 DeepSeek 这类 MLA 模型来说，这个问题尤其重要。因为 speculative decoding
不是简单地“多跑一个 draft model”，它会直接改变：

- 当前 batch 有多少 query token
- 每个 query 应该看到哪些 KV
- 这些 KV 在 paged KV cache 中的索引方式
- 是否需要树状 mask / causal mask
- MLA kernel 需要的 workspace / split / persistent metadata

换句话说：

**speculative decoding 在 runtime 层的本质，就是不断重写 attention metadata。**


## 一句话理解

可以把 attention metadata 理解为：

- “这一次 attention 要怎么看 KV cache”的说明书

而 speculative decoding 做的事情，本质上就是不断改变这份说明书：

- normal decode：每个请求 1 个 query，查自己已有上下文
- draft extend：一次要处理多个 draft token，需要新的 `qo_indptr` 与 mask
- target verify：要同时验证多个候选 token，query 形状和 KV 长度都变了
- DeepSeek MLA / MTP：`max_q_len`、`kv_indptr`、`qo_indptr`、`work_metadata`
  会直接影响 kernel 如何执行


## 1. 为什么这个问题在 DeepSeek 上特别重要

DeepSeek 使用 MLA（Multi-head Latent Attention）后，attention metadata 的作用比普通
MHA 更重：

- 普通 MHA 更多是 query/key/value 张量形状和 mask 变化
- MLA 还要额外构造：
  - `kv_indptr`
  - `kv_indices`
  - `qo_indptr`
  - `kv_last_page_len`
  - `max_q_len`
  - `work_metadata`
  - `work_info_set`
  - `reduce_indptr`
  - `reduce_final_map`
  - `reduce_partial_map`

这些量直接决定：

- MLA persistent kernel 如何分块
- 每个 query 要从 paged KV cache 里取哪些 token
- multi-query（例如 verify / MTP）时 query 维度如何展开

所以在 DeepSeek speculative 路径中，真正最敏感的往往不是 model forward 本身，
而是 **attention metadata 是否按正确语义构出来**。


## 2. 先看三层 batch 抽象

核心文件：

- `sglang/python/sglang/srt/model_executor/forward_batch_info.py`
- `sglang/python/sglang/srt/managers/schedule_batch.py`

SGLang 中 batch 数据结构有三层：

- `ScheduleBatch`
- `ModelWorkerBatch`
- `ForwardBatch`

源码注释位置：

- `forward_batch_info.py` 文件开头

这三层的职责可以粗略理解为：

- `ScheduleBatch`
  - scheduler 视角
  - 关注请求、prefix、token、调度状态
- `ModelWorkerBatch`
  - worker 视角
  - 关注一次 GPU forward 所需字段
- `ForwardBatch`
  - backend / kernel 视角
  - 关注 query、KV、cache、metadata

**attention metadata 的最终落点是在 `ForwardBatch -> attn_backend.init_forward_metadata()`**
这一层。


## 3. `ForwardMode`：speculative 如何改变 metadata 初始化分支

核心文件：

- `sglang/python/sglang/srt/model_executor/forward_batch_info.py`

关键枚举：

- `ForwardMode`

关键 speculative mode：

- `TARGET_VERIFY`
- `DRAFT_EXTEND`
- `DRAFT_EXTEND_V2`

关键代码位置：

- `ForwardMode` 定义：约 `74-179`

最容易踩坑的一点：

- `ForwardMode.is_extend()` 会把 `TARGET_VERIFY` 也算进去

对应逻辑：

- `forward_batch_info.py` 约 `105-114`

这意味着如果某个 backend 只是粗暴地区分：

- decode
- extend

而没有再细分：

- target_verify
- draft_extend

那么它很容易把 verify 当普通 extend 处理，然后在 metadata 上出错。


## 4. speculative 信息是如何进入 attention 层的

核心抽象：

- `SpecInput`

文件：

- `sglang/python/sglang/srt/speculative/spec_info.py`

关键点：

- `SpecInput` 不是附带信息，而是 speculative 与 attention metadata 的桥梁
- 它负责携带：
  - speculative token 相关信息
  - 需要的 positions
  - `kv_indptr` / `kv_indices`
  - `custom_mask`
  - `accept_length`
  - `draft_token_num`
  - 其他草稿 / 验证所需状态

相关位置：

- `SpecInputType`：约 `108-113`
- `SpecInput`：约 `116-143`

这里有一个很重要的方法：

- `get_spec_adjusted_global_num_tokens()`

它说明 speculative decoding 会直接改变：

- global num tokens
- logprob token 数

这也间接影响 batch padding 和后续 metadata 构造。


## 5. `EagleVerifyInput`：speculative 到 metadata 的第一层接口

核心文件：

- `sglang/python/sglang/srt/speculative/eagle_info.py`

关键类：

- `EagleVerifyInput`

关键字段：

- `draft_token`
- `custom_mask`
- `positions`
- `draft_token_num`
- `capture_hidden_mode`
- `seq_lens_sum`
- `seq_lens_cpu`

代码位置：

- `eagle_info.py` 约 `54-78`

这些字段本身就已经说明了 speculative 和 metadata 的关系：

- `draft_token`
  - 决定 verify 阶段实际送入 target 的 query token
- `positions`
  - 决定 RoPE / position indexing
- `draft_token_num`
  - 决定一次 verify 需要几个 query
- `custom_mask`
  - 决定树状 speculative 验证时的可见性


## 6. verify 阶段：speculative 如何改写 batch

### 6.1 v1 路径

关键文件：

- `sglang/python/sglang/srt/speculative/eagle_worker.py`
- `sglang/python/sglang/srt/speculative/eagle_info.py`

关键流程：

1. `draft()` 先生成候选 token，形成 `EagleVerifyInput`
2. `verify()` 调 `spec_info.prepare_for_verify(batch, page_size)`
3. `batch.forward_mode` 被改成 `TARGET_VERIFY`
4. target worker 执行 verify forward

关键位置：

- `eagle_worker.py` 中 `verify()`：约 `699-788`
- `eagle_info.py` 中 `prepare_for_verify()`：约 `104-146`


### 6.2 `prepare_for_verify()` 改了什么

它主要会做：

- `batch.input_ids = self.draft_token`
- 分配 `batch.out_cache_loc`
- 更新 `req_to_token_pool`

也就是：

- target verify 不再看原先的“普通 decode 单 token 输入”
- 而是把所有 draft token 当作本轮 query 批次

这已经说明：

- verify 不是普通 decode
- verify 的 query 形状和 KV 形状都变了
- 所以 attention metadata 必须重新构造


## 7. `generate_attn_arg_prefill()`：draft_extend 的 metadata 生成器

文件：

- `sglang/python/sglang/srt/speculative/eagle_info.py`

关键方法：

- `generate_attn_arg_prefill()`

代码位置：

- 约 `160-216`

这个函数非常关键，因为它直接把 speculative 信息翻译成 attention metadata 里的核心索引：

- `qo_indptr`
- `cum_kv_seq_len`（本质上就是 `kv_indptr`）
- `kv_indices`
- `custom_mask`

可以理解为：

- speculative 输入先描述“我要验证/扩展多少个 draft token、树结构是什么”
- `generate_attn_arg_prefill()` 再把这种高层语义翻译成 kernel 能消费的索引格式


### 7.1 `qo_indptr` 是什么

在这里：

- `qo_indptr` 表示 query output token 在 batch 中如何分段

例如：

- 每个请求有 `draft_token_num` 个 query
- 那么 `qo_indptr` 就会按这个 query 数量分桶


### 7.2 `kv_indptr` / `cum_kv_seq_len` 是什么

它表示：

- 每个请求在当前 forward 中可见的 KV token 范围

draft / verify 会把：

- 原始 `paged_kernel_lens`

扩成：

- `paged_kernel_lens + draft_token_num`

这说明 speculative decoding 不是只“多几个 query”，而是连本轮可见 KV 长度都变了。


### 7.3 `custom_mask` 是什么

对于树状 speculative decode：

- 不是所有 draft token 都能互相看见

所以需要：

- `custom_mask`

来表示 tree-based causal structure。

这个量会在非 MLA MHA 路径里更直接地进入 attention kernel。


## 8. v2 路径：`prepare_for_v2_verify()` 如何构造 verify metadata

核心文件：

- `sglang/python/sglang/srt/speculative/eagle_info_v2.py`

关键方法：

- `prepare_for_v2_verify()`

代码位置：

- `eagle_info_v2.py` 约 `213-270`

这个方法做的事情可以理解成：

1. 先按 speculative verify 语义设置：
   - `batch.input_ids`
   - `batch.out_cache_loc`
2. 把 `batch.forward_mode` 改成 `TARGET_VERIFY`
3. 通过 `ForwardBatch.init_new(batch, target_worker.model_runner)`
   得到真正的 `ForwardBatch`
4. 然后显式调用：
   - `target_worker.model_runner.attn_backend.init_forward_metadata(verify_forward_batch)`

这说明：

- speculative verify 到 attention metadata 的连接点，不是在 model forward 里隐式发生的
- 而是在 `prepare_for_v2_verify()` 中显式发生的


## 9. attention metadata 长什么样

### 9.1 upstream SGLang `ForwardMetadata`

文件：

- `sglang/python/sglang/srt/layers/attention/aiter_backend.py`

关键 dataclass：

- `ForwardMetadata`

代码位置：

- `aiter_backend.py` 约 `76-95`

关键字段：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`
- `max_q_len`
- `max_kv_len`
- `work_metadata`
- `work_info_set`
- `reduce_indptr`
- `reduce_final_map`
- `reduce_partial_map`
- `num_kv_splits`
- `custom_mask`
- `mask_indptr`
- `max_extend_len`
- `fp8_prefill_kv_indices`


### 9.2 ATOM plugin 的 `ForwardMetadata`

文件：

- `ATOM/atom/plugin/sglang/attention_backend/sgl_attn_backend.py`

关键 dataclass：

- `ForwardMetadata`

代码位置：

- `sgl_attn_backend.py` 约 `171-198`

从字段上看，ATOM plugin 其实已经承认 speculative / MLA attention 需要这些索引和 workspace。
所以当前问题不是“不知道这些量存在”，而是：

- 没在 metadata init 分支上完全按 upstream 语义实现


## 10. upstream `AiterAttnBackend.init_forward_metadata()` 如何按 speculative 分流

核心文件：

- `sglang/python/sglang/srt/layers/attention/aiter_backend.py`

关键方法：

- `init_forward_metadata()`

代码位置：

- `aiter_backend.py` 约 `435-684`

这是整条链最重要的代码之一。

它不是简单分成：

- decode
- extend

而是分成：

1. `decode_or_idle`
2. `draft_extend`
3. `target_verify`
4. 普通 extend


### 10.1 普通 decode / idle

逻辑：

- `spec_info` 为空时，按普通 decode 构造
- `spec_info` 不为空时，直接复用 `spec_info.kv_indptr / kv_indices`

这说明 speculative 已经开始介入 decode metadata。


### 10.2 `draft_extend`

逻辑：

- 调 `spec_info.generate_attn_arg_prefill()`
- 拿到：
  - `kv_indices`
  - `kv_indptr`
  - `qo_indptr`
  - `custom_mask`
- MLA 路再进一步根据 `extend_seq_lens_cpu`
  计算 `max_seqlen_qo` 和 persistent kernel metadata

关键位置：

- `aiter_backend.py` 约 `526-606`


### 10.3 `target_verify`

这是最关键的一支。

逻辑：

- 不依赖普通 extend 的 `extend_seq_lens`
- 直接用：
  - `draft_num = spec_info.draft_token_num`
  - `kv_lens = forward_batch.seq_lens + draft_num`
- 自己构造：
  - `qo_indptr`
  - `kv_indptr`
  - `kv_indices`
- 对 MLA 路：
  - `max_q_len = draft_num`

关键位置：

- `aiter_backend.py` 约 `607-684`

这个分支完美说明：

**verify 不是普通 extend，speculative 会直接重定义 query 长度和 KV 长度。**


## 11. DeepSeek MLA：为什么 speculative 更像“metadata 问题”

对于 DeepSeek MLA 来说，attention forward 真正吃的是：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`
- `max_q_len`
- `work_metadata` / `reduce_*`

如果这些量不对：

- 哪怕 model forward、q/k/v 张量本身都没问题
- kernel 也会在错误的 KV 范围上工作

这就是为什么调试 speculative 时，attention metadata 的正确性往往比 model 本身更先决定成败。


## 12. 本次调试得到的一个关键教训

在 `ATOM plugin` 当前实现中：

- `sgl_attn_backend.py` 里的 `_forward_extend_mla()` 已经认识：
  - `TARGET_VERIFY`
  - `DRAFT_EXTEND`

代码位置：

- `sgl_attn_backend.py` 约 `1001-1022`

但 metadata 初始化层还没有完全按 upstream 分支细化：

- `init_forward_metadata()` 仍是：
  - `decode_or_idle`
  - else -> `extend`

代码位置：

- `sgl_attn_backend.py` 约 `282-288`

于是 `TARGET_VERIFY` 会被误送进普通 `_init_extend_mla()`：

- 它会错误假设 `forward_batch.extend_seq_lens` 一定存在

而在 verify 路径下：

- `extend_seq_lens` 本来就可能是 `None`

这就是为什么当前错误看起来像：

- `NoneType has no attribute max`

实际上本质是：

- **speculative 和 attention metadata 的语义没有对齐**


## 13. 从 ATOM 原生 MTP 再看一次 metadata 的重要性

如果看 ATOM 原生链路：

- `ATOM/atom/spec_decode/eagle.py`
- `ATOM/atom/model_ops/attentions/aiter_mla.py`

会发现 speculative / MTP 对 attention metadata 的耦合更直接。

### 13.1 `EagleProposer.propose()`

关键位置：

- `atom/spec_decode/eagle.py` 约 `94-190`

在多步 draft 过程中，会不断更新：

- `attn_metadata.max_seqlen_q`
- `attn_metadata.max_seqlen_k`
- `kv_indptr`
- `kv_indices`
- `cu_seqlens_q`
- `slot_mapping`
- `kv_last_page_lens`

并调用：

- `prepare_mtp_decode()`

这说明在 ATOM 原生实现里：

- speculative 不是 attention 上的一点点附加参数
- 而是会不断重写 attention metadata


### 13.2 `prepare_mtp_decode()`

文件：

- `ATOM/atom/model_ops/attentions/aiter_mla.py`

关键位置：

- `prepare_mtp_decode()`：约 `225-250`

作用：

- 为多 token 预测构造 MTP decode 需要的 KV / worker metadata

同文件里还有一个重要信号：

- `prepare_decode()` 会在有 drafter 时把
  `max_seqlen_q = drafter.mtp_k + 1`

位置：

- `aiter_mla.py` 约 `352-357`

这再次说明：

- speculative / MTP 本质上会改变 query 维度
- query 维度一变，attention metadata 就必须重建


## 14. 调试 speculative + metadata 时的实用检查表

如果后续继续调试 DeepSeek speculative / MTP，建议优先检查下面几项：

### 1. 当前 `ForwardMode` 是什么

看：

- `decode`
- `target_verify`
- `draft_extend`
- `draft_extend_v2`

如果 mode 判断错了，metadata 分支通常也会错。


### 2. 当前 `spec_info` 是不是空

如果 `spec_info` 不为空，就不应该再走普通 extend 的 metadata 逻辑。


### 3. `qo_indptr` 是否和 speculative token 数一致

例如 verify 路径里：

- `max_q_len` 应该接近 `draft_token_num`

而不是普通 decode 的 `1`。


### 4. `kv_indptr / kv_indices` 是否按 speculative 后的新 KV 长度构造

verify 阶段一般应当看到：

- `kv_lens = seq_lens + draft_token_num`

而不是原始 `seq_lens`。


### 5. 是否错误依赖了 `extend_seq_lens`

普通 extend 可以依赖：

- `extend_seq_lens`

但 `target_verify` 不应简单照搬这套假设。


### 6. 是否需要 `custom_mask`

树状 speculative / topk 路径下：

- `custom_mask`

常常是必须的；它缺失时可能不会立刻报错，但结果会错。


## 15. 推荐阅读顺序

如果以后要重新从头搞清楚 “DeepSeek speculative 与 attention metadata 的关系”，
推荐按下面顺序阅读：

1. `sglang/python/sglang/srt/model_executor/forward_batch_info.py`
   - 看 `ForwardMode`
2. `sglang/python/sglang/srt/speculative/spec_info.py`
   - 看 `SpecInput`
3. `sglang/python/sglang/srt/speculative/eagle_info.py`
   - 看 `EagleVerifyInput`
4. `sglang/python/sglang/srt/speculative/eagle_info_v2.py`
   - 看 `prepare_for_v2_verify()`
5. `sglang/python/sglang/srt/layers/attention/aiter_backend.py`
   - 看 `init_forward_metadata()` 的四种 speculative 分支
6. `ATOM/atom/plugin/sglang/attention_backend/sgl_attn_backend.py`
   - 对照 plugin 当前实现和 upstream 差异
7. `ATOM/atom/spec_decode/eagle.py`
   - 看 ATOM 原生 speculative 是怎么驱动 attn metadata 更新的
8. `ATOM/atom/model_ops/attentions/aiter_mla.py`
   - 看 MTP decode 的 metadata 准备逻辑


## 16. 最终总结

对于 DeepSeek 而言：

- speculative decoding 的重点不只是 draft model
- attention metadata 才是把 speculative 语义真正落到 kernel 的关键层

可以用下面一句话概括：

**draft / verify 负责决定“要处理哪些 token”，attention metadata 负责把这个决定变成 kernel 可执行的 KV / Q 索引和 workspace 说明。**

因此，后续如果要在 `ATOM + SGLang plugin` 路径真正接通 DeepSeek MTP，
核心工作并不是只把 draft model 换成 `ATOM DeepSeekMTP`，还包括：

- 让 plugin 的 attention metadata 初始化完整理解
  - `TARGET_VERIFY`
  - `DRAFT_EXTEND`
  - `MTP 多 query`
  - `custom_mask`
  - `qo_indptr / kv_indptr / kv_indices`

只有这层语义也打通，DeepSeek speculative 才算真正可用。
