# 2026-04-08 vLLM Continuous Batching 原理与源码位置笔记

## 文档目的

这份文档用于系统梳理 `vLLM` 中与 `continuous batching` 相关的核心机制，
重点回答下面几个问题：

- `continuous batching` 到底是什么，和静态 batch 有什么本质差异
- `scheduler` 是如何把多个 request 组装成一次 step 的执行 batch 的
- 一次 step 之后，模型输出是如何再放回各个 request 的
- 为什么 vLLM 的 batch 不是传统训练里那种规则的 `B x L`
- `prefill / decode / chunked prefill / speculative decode` 在这个框架下是怎样统一的
- 如果要顺着源码看，应该优先看哪些代码位置

本文尽量从三个维度同时展开：

- 系统设计
- 张量 shape
- 源码入口

方便后续复盘、调试和与其他推理框架做对比。


## 一句话总结

`vLLM continuous batching` 的核心不是“把很多请求 pad 成一个固定 `B x L` 大矩阵”，
而是：

- 每个 step 动态决定“每个 request 这一步前进多少 token”
- 把这些 token 展平成一个按 token 计数的 flat batch
- 用 `query_start_loc / seq_lens / block_tables / slot_mapping` 等元数据告诉 GPU：
  - 每个 token 属于哪个 request
  - 它在该 request 中的位置是多少
  - 它应该读写哪一段 KV cache
- step 结束后，再通过 `req_id_to_index` 把输出准确拆回每个 request

可以把它概括成：

```text
requests
  -> scheduler 决定本步每个 request 的 n_i
  -> 组装成 flat token batch, 总 token 数 T = sum(n_i)
  -> GPU forward / sample
  -> 用 req_id_to_index 把输出拆回 request
```


## 版本说明

本文主要覆盖两条线：

- `v1 / current main` 风格实现
- `v0` 旧版 `LLMEngine` / `SequenceGroup` 风格实现

需要注意：

- `v1` 是当前更值得优先看的主线
- `v0` 仍然有很强的参考价值，因为很多文章、issue、历史讨论都仍然沿用
  `SequenceGroup`、`SchedulerOutputs` 这套命名
- 本文提到的源码位置与行号，基于 `2026-04-08` 抓取的 upstream 快照，
  后续可能会轻微漂移


## 1. 为什么 continuous batching 不是静态 batching

### 1.1 静态 batching 的直觉

训练或普通离线推理中，大家更熟悉的是：

- 给定一批序列
- pad 到同一个长度
- 形成一个规则张量

例如：

```text
input_ids.shape = [B, L]
attention_mask.shape = [B, L]
```

这种做法的假设是：

- 这一批样本一起开始
- 一起执行
- 一起结束


### 1.2 在线 serving 的问题

在线服务时，请求并不是同时到达，也不会同时结束。

典型情况是：

- 某些 request 还在做长 prompt 的 prefill
- 某些 request 已经进入 decode，每步只需要前进 1 个 token
- 某些 request 刚结束
- 某些新 request 又刚进入系统

如果还坚持用静态 batch，就会遇到：

- 等待新 request 凑满 batch，TTFT 变差
- 某个 request 提前结束后，batch 中留下空洞
- prompt 很长的 request 会拖慢所有其他 request


### 1.3 continuous batching 的本质

所以 vLLM 的选择不是“固定一批 request 一起跑到结束”，而是：

- 每个调度 step 都重新看当前系统中的 request
- 决定本步哪些 request 参与
- 决定每个 request 本步前进多少 token
- step 结束后，再立刻重组下一轮 batch

因此，batch 是“连续流动”的。

这也是 `continuous batching` 这个名字的真正含义。


## 2. vLLM 视角下一个 request 的核心状态

在 vLLM 中，理解 request 的关键不是先区分“prefill 还是 decode”，
而是先看下面几个状态量。

### 2.1 最重要的两个量

- `all_token_ids`
  - 当前这个 request 已知的完整 token 序列
  - 包括 prompt token，也包括已经生成出来但可能尚未被下一轮 compute 的 token
- `num_computed_tokens`
  - `all_token_ids` 中已经真正做过 forward、对应 KV 已经落到 cache 的前缀长度

于是：

- 如果 `num_computed_tokens = 0`，说明 prompt 还没 prefill
- 如果 `num_computed_tokens < len(all_token_ids)`，说明还有 backlog 没算
- decode 阶段常见情况是：
  - 上一轮 sample 出了 1 个新 token
  - 下一轮需要把这 1 个 token 真正送进模型计算
  - 所以通常 backlog 是 1


### 2.2 统一 prefill / decode 的关键观察

从 scheduler 角度，并不存在一个特别刚性的：

- “prefill phase”
- “decode phase”

更接近的真实逻辑是：

- 对每个 request，看它还有多少 token 没被 compute
- 本轮决定从这些 backlog 中取多少 token 来执行

因此：

- 新 request 的 backlog 通常很大，对应 prefill
- 老 request 的 backlog 常常只有 1，对应 decode
- chunked prefill 只是“长 request 的 backlog 一次不要全吃完”

也就是说，`prefill / decode` 更像是同一调度框架下的两种常见形态。


### 2.3 KV cache 也是 request 状态的一部分

除了 token 序列本身，每个 request 还绑定：

- KV cache block
- block table
- 对应的 slot mapping

这决定了：

- decode 时虽然本轮可能只新输入 1 个 token
- 但模型仍然能通过 KV cache 读取全部历史上下文

所以一个 request 的有效状态并不只是 token ids，而是：

```text
request state
  = token sequence
  + num_computed_tokens
  + sampling / stop state
  + KV cache mapping
  + （可选）LoRA / multimodal / structured output 状态
```


## 3. 一个 batch 的“实质性内容”到底是什么

这是最容易误解的地方。

### 3.1 从 tokenizer 语义看

一个 token 最原始确实就是一个 vocab id，也就是一个整数。

例如：

```text
"hello" -> 15496
```

因此在输入层面，`input_ids` 的每个元素确实就是“一个数字”。


### 3.2 从 GPU 执行看

但真正送进 GPU 跑一次 step，远远不只有 `input_ids`。

至少还需要：

- `input_ids`
- `positions`
- `query_start_loc`
- `seq_lens`
- `block_tables`
- `slot_mapping`
- `logits_indices`

特殊情况下还会有：

- `inputs_embeds`
- multimodal encoder 相关输入
- LoRA metadata
- speculative decode 的 draft token 相关索引
- structured output grammar bitmask

所以如果问：

> 一个 batch 的实质性 token，是不是仅仅是简单的 input_id？

答案是：

- 对“token 身份”来说，最原始确实是 `input_id`
- 对“一次 forward 的完整执行语义”来说，绝对不够

因为模型还必须知道：

- 这个 token 属于哪个 request
- 它在该 request 里的绝对位置是多少
- 它应该从 KV cache 的哪一段读取历史上下文


### 3.3 这个 `input_id` 在 GPU 上吗

在真正执行时，是的。

更准确地说：

- request 和 scheduler 主要在 CPU 侧维护高层状态
- 但进入 worker / model runner 后，本步需要用到的
  `input_ids / positions / query_start_loc / block_tables / slot_mapping`
  会被放入 GPU buffer
- 然后做 embedding lookup，进入 transformer forward

所以：

- 逻辑上的 token id 一开始常出现在 CPU 侧
- 本步执行用到的 `input_ids` 会进入 GPU
- 进入模型后，它很快会被 embedding 成一个向量

例如：

```text
token_id: scalar
  -> embedding lookup
  -> hidden vector: [hidden_size]
```

如果本轮总共执行 `T` 个 token，那么 embedding 后大致就是：

```text
[T, hidden_size]
```


## 4. scheduler 到底在做什么

### 4.1 核心目标

scheduler 的工作不是“把所有 request pad 成一个矩阵”，而是：

- 从 `waiting / running` 队列里挑 request
- 决定每个 request 本步前进多少 token
- 保证不超出资源预算
- 必要时做 preemption
- 产出 worker 能执行的调度结果


### 4.2 主要约束

在 `v1` 中，最重要的两个约束是：

- `max_num_seqs`
  - 本步最多同时挂多少个 request
- `max_num_batched_tokens` 或 `max_num_scheduled_tokens`
  - 本步总共最多前进多少 token

此外还会考虑：

- model max length
- encoder 计算预算（多模态）
- LoRA 同批数量限制
- KV cache block 是否足够
- prefix cache / remote KV / async loading 状态


### 4.3 调度的高层流程

`v1` 的 `Scheduler.schedule()` 大致可以概括成：

1. 先尝试调度 `running` request
2. 再尝试从 `waiting` 里吸入新 request
3. 对每个 request 决定本步的 `n_i`
4. 维护 `token_budget`
5. 如果 block 不够或约束冲突，必要时 preempt 某些 request
6. 输出 `SchedulerOutput`

一个很关键的设计点是：

> scheduler 关心的是 “本步每个 request 前进多少 token”

而不是：

> “这个 request 属于 prefill 还是 decode 类别”


### 4.4 chunked prefill 是怎样融进去的

长 prompt 的 request，如果一次全吃完会把 token budget 吃光，
拖累其他 request。

所以 vLLM 会在需要时把它拆成多步：

- 本轮只 prefill prompt 的一部分
- 剩下的下轮再继续

因此一个 request 可以出现：

- 还在 prefill chunk 中
- 但同时其他 request 已经在 decode

这正是 continuous batching 最典型的混合场景。


## 5. scheduler 输出的关键数据结构

在 `v1` 中，scheduler 输出的核心抽象是：

- `NewRequestData`
- `CachedRequestData`
- `SchedulerOutput`

可以把这三者理解成：

- `NewRequestData`
  - 首次进入 worker 的 request，要发送完整初始化数据
- `CachedRequestData`
  - worker 已经缓存过的 request，只发送增量信息
- `SchedulerOutput`
  - 这一步所有调度决策的总封装


### 5.1 `NewRequestData`

它通常包含：

- `req_id`
- `prompt_token_ids`
- `sampling_params`
- `pooling_params`
- `block_ids`
- `num_computed_tokens`
- `lora_request`
- `prefill_token_ids`（v2 model runner 相关）

也就是说，新 request 第一次进入 worker 时，需要把足够多的静态信息发过去，
让 worker 端建立自己的 request cache。


### 5.2 `CachedRequestData`

这个结构是 continuous batching 很重要的一环，因为它体现了：

> worker 对 request 状态是“长期缓存”的，而不是每 step 重建。

典型字段有：

- `req_ids`
- `resumed_req_ids`
- `new_token_ids`
- `all_token_ids`
- `new_block_ids`
- `num_computed_tokens`
- `num_output_tokens`

其中最关键的思想是：

- 对已经在 worker 里的 request，不重复发送整条 request
- 只发送变化的部分

这能显著减少调度端和 worker 之间的通信成本。


### 5.3 `SchedulerOutput`

最重要的字段有：

- `scheduled_new_reqs`
- `scheduled_cached_reqs`
- `num_scheduled_tokens: dict[req_id, int]`
- `total_num_scheduled_tokens`
- `scheduled_spec_decode_tokens`
- `scheduled_encoder_inputs`
- `finished_req_ids`

其中：

- `num_scheduled_tokens` 是整轮 step 的核心
- 它表达的是：
  - 这个 request 这一步要前进几个 token

如果把本轮调度到了 `B` 个 request，则：

```text
num_scheduled_tokens: {req_id_1: n_1, ..., req_id_B: n_B}
T = n_1 + ... + n_B
```

这里：

- `B` 是 request 数
- `T` 是 token 数

vLLM 后续执行更偏向围绕 `T` 展开，而不是围绕规则的 `B x L`。


## 6. 真正执行时，batch 的 shape 长什么样

这是理解 vLLM 最关键的一节。

### 6.1 不是 `[B, L]`，而是 token-flat `[T]`

假设本轮有 `B` 个 request，第 `i` 个 request 本轮前进 `n_i` 个 token。

则：

```text
T = sum_i n_i
```

执行时，最核心的输入通常是：

- `input_ids`: `[T]`
- `positions`: `[T]`
- `query_start_loc`: `[B + 1]`
- `seq_lens`: `[B]`

为了 CUDA graph 或执行约束，vLLM 里还常会有 padding 后版本：

- `T_pad`
- `B_pad`

于是实际 buffer 常是：

- `input_ids`: `[T_pad]`
- `positions`: `[T_pad]`
- `seq_lens`: `[B_pad]`


### 6.2 `query_start_loc` 是什么

`query_start_loc` 是每个 request 在扁平 token buffer 中的边界。

如果：

```text
n = [1, 1, 2, 6]
```

则：

```text
query_start_loc = [0, 1, 2, 4, 10]
```

含义是：

- 第 0 个 request 用 `input_ids[0:1]`
- 第 1 个 request 用 `input_ids[1:2]`
- 第 2 个 request 用 `input_ids[2:4]`
- 第 3 个 request 用 `input_ids[4:10]`

这就是：

- 一个大 flat token buffer
- 加一个分段索引数组

共同表达 ragged batch 的典型做法。


### 6.3 decode 为何也能放进这个框架

decode request 在本轮通常只前进 1 个 token，所以常见：

```text
n_i = 1
```

那它在扁平 batch 里也就只占一个元素。

例如：

```text
input_ids = [r1_new, r2_new, p0, p1, p2, p3]
```

这里前两个是 decode token，后四个是某个 prefill request 的 prompt chunk。

看起来 decode token 很“短”，但它并不缺上下文，因为上下文来自：

- `seq_lens`
- `block_tables`
- `slot_mapping`
- KV cache


### 6.4 还有哪些重要 shape

除了上面几个，attention 执行时还非常依赖：

- `block_tables`
  - 近似可以看成：每个 KV cache group 一份
  - 形状常见近似为 `[B_pad, max_num_blocks]`
- `slot_mapping`
  - 近似为每个 token 映射到哪个 KV slot
  - 常见近似为 `[T_pad]`

因此，从 GPU 视角看，一个 batch 更接近：

```text
flat token payload
  + per-request segmentation metadata
  + KV cache address metadata
```

而不是简单的 `input_ids` 矩阵。


## 7. 一次 step 的完整生命周期

在 `v1` 中，可以把一次 step 概括为：

1. `schedule()`
2. `execute_model(...)`
3. `update_from_output(...)`
4. `OutputProcessor.process_outputs(...)`

下面按顺序拆开。


### 7.1 `schedule()`

scheduler 产生：

- 哪些 request 参与本轮
- 每个 request 本轮前进多少 token
- 新 request / cached request 的增量更新数据

同时，vLLM 还有一个很值得注意的设计：

- request 被 schedule 到后，会先把 `num_computed_tokens` 往前推进
- 这样它可以在下一轮继续被及时调度
- 如果后面 speculative token 有拒绝，再在 `update_from_output()` 里回调修正

这说明：

- 调度状态和最终 sample 结果之间不是完全同步的
- 某些统计量会“先乐观推进，再按输出修正”


### 7.2 `execute_model(...)`

worker 侧收到 `SchedulerOutput` 后，会做：

- `add_requests()`
  - 初始化首次进入 worker 的 request
- `update_requests()`
  - 更新已有 request 的 block / token 等状态
- `prepare_inputs()`
  - 组装本轮 flat token batch
- `prepare_attn()`
  - 生成 attention metadata
- 执行模型 forward
- sample token / 或做 pooling

执行完成后返回 `ModelRunnerOutput`。


### 7.3 `ModelRunnerOutput`

这是“从 GPU 结果回到 scheduler”的关键桥梁。

核心字段可以理解成：

- `req_ids`: `[B]`
- `req_id_to_index: {req_id -> batch_idx}`
- `sampled_token_ids`: `list[list[int]]`
- `logprobs`
- `prompt_logprobs_dict`
- `pooler_output`

其中最关键的是：

- `req_id_to_index`
- `sampled_token_ids`

因为 worker 为了执行效率可能重排 request 顺序，所以 scheduler 回填时不能假设：

- “第 0 个输出一定属于第 0 个 request”

而必须显式做：

```text
idx = req_id_to_index[req_id]
generated = sampled_token_ids[idx]
```


### 7.4 `update_from_output(...)`

这一步负责：

- 根据 `req_id_to_index` 找到每个 request 对应的输出
- 把 `sampled_token_ids[idx]` 回填到 request 状态
- 检查 stop / eos / length
- 处理 speculative decode 的接受 / 拒绝
- 必要时释放 request 的 KV cache
- 产出 `EngineCoreOutput`

这里有一个非常重要的概念区分：

- `n_i`
  - 本轮这个 request 被安排去“计算”的 token 数
- `g_i`
  - 本轮真正“生成出来并回给请求”的 token 数

这两个量不一定相等。

典型例子：

- chunked prefill 时，`n_i > 0`，但 `g_i = 0`
- 普通 decode 时，通常 `n_i = 1`，`g_i = 1`
- speculative decode 时，可能 `n_i = 1 + k`，而 `g_i` 可以大于 1


### 7.5 `OutputProcessor.process_outputs(...)`

这一步负责从 engine 内部输出变成用户能看到的 `RequestOutput`。

主要工作有：

- detokenize
- stop string 检查
- logprobs 处理
- 组装 `RequestOutput`

因此完整链路是：

```text
SchedulerOutput
  -> ModelRunnerOutput
  -> EngineCoreOutput
  -> RequestOutput
```


## 8. 例子一：3 个 request，4 个 step

下面给一个不带 speculative decode 的完整 toy example。

假设配置：

- `max_num_seqs = 3`
- `max_num_batched_tokens = 6`
- 开启 `chunked prefill`

4 个请求依次到达：

- `R1 = [11,12,13,14,15]`
- `R2 = [21,22]`
- `R3 = [31,32,33,34]`
- `R4 = [41,42,43]`

初始状态：

```text
R1: all=[11,12,13,14,15], comp=0
R2: all=[21,22],          comp=0
```


### Step 0

scheduler 选择：

- `R1` 前进 4 个 prompt token
- `R2` 前进 2 个 prompt token

于是：

```text
num_scheduled_tokens = {R1: 4, R2: 2}
B = 2
T = 6
```

worker 侧可能重排为：

```text
req_ids = [R2, R1]                         # shape [2]
num_scheduled_tokens = [2, 4]             # shape [2]
query_start_loc = [0, 2, 6]               # shape [3]

input_ids = [21,22, 11,12,13,14]          # shape [6]
positions = [0,1,  0,1,2,3]               # shape [6]
seq_lens  = [2,4]                         # shape [2]
```

假设这一轮输出：

- `R2` prompt 已结束，sample 到首个生成 token `23`
- `R1` 还没结束 prefill，没有生成 token

则：

```text
req_id_to_index = {R2: 0, R1: 1}
sampled_token_ids = [[23], []]
```

回填后：

```text
R1: all=[11,12,13,14,15],    comp=4
R2: all=[21,22,23],          comp=2
```

注意：

- `R2` 的 `23` 已追加到 `all_token_ids`
- 但 `comp=2`，因为本轮真正算进 KV 的还是原 prompt 的 2 个 token
- `23` 会在下一轮真正参与 decode compute


### Step 1

此时 `R3` 到达。

现在系统中：

- `R1` 还差 1 个 prompt token
- `R2` 要 decode 它的 `23`
- `R3` 是新 request，需要 prefill

于是 scheduler 可以在同一步混合调度：

```text
num_scheduled_tokens = {R1: 1, R2: 1, R3: 4}
B = 3
T = 6
```

执行 batch：

```text
req_ids = [R1, R2, R3]                    # shape [3]
query_start_loc = [0, 1, 2, 6]            # shape [4]

input_ids = [15, 23, 31,32,33,34]         # shape [6]
positions = [4,  2,  0,1,2,3]             # shape [6]
seq_lens  = [5,  3,  4]                   # shape [3]
```

假设输出：

```text
sampled_token_ids = [[101], [24], [35]]
```

回填后：

```text
R1: all=[11,12,13,14,15,101],     comp=5
R2: all=[21,22,23,24],            comp=3
R3: all=[31,32,33,34,35],         comp=4
```

这一步非常重要，因为它体现了：

- `R1` 还在补最后一段 prefill
- `R2` 已在 decode
- `R3` 是新 request 的整段 prefill

三者可以在同一个 step 里并存。


### Step 2

现在三者都进入正常 decode：

```text
num_scheduled_tokens = {R1:1, R2:1, R3:1}
B = 3
T = 3

input_ids = [101, 24, 35]                 # shape [3]
query_start_loc = [0, 1, 2, 3]            # shape [4]
```

假设输出：

```text
sampled_token_ids = [[102], [2], [36]]
```

若 `2` 是 EOS，则：

- `R2` 完成
- `R2` 的 KV block 可被释放
- `R1`、`R3` 继续保留


### Step 3

此时 `R4` 到达，于是可以立刻填补空位：

```text
num_scheduled_tokens = {R1:1, R3:1, R4:3}
B = 3
T = 5

input_ids = [102, 36, 41,42,43]           # shape [5]
query_start_loc = [0, 1, 2, 5]            # shape [4]
```

这就是 continuous batching 的最直观体现：

- 老 request 结束后立即移出
- 新 request 马上补进来
- 系统不是“整批结束再换下一批”，而是每一步都在流动


## 9. 例子二：为什么 `n_i` 不等于 `g_i`

很多人第一次看时会默认：

- scheduler 安排这个 request 算 4 个 token
- 那它就应该返回 4 个 token

这在 vLLM 中并不成立。

### 9.1 chunked prefill 场景

假设一个长 prompt request：

```text
prompt = [p0, p1, p2, p3, p4, p5, p6, p7]
num_computed_tokens = 0
```

本轮只给它 4 个 token budget：

```text
n_i = 4
input_ids = [p0, p1, p2, p3]
```

如果 prompt 还没 prefill 完，则这轮：

```text
g_i = 0
```

也就是：

- 本轮确实算了 4 个 token
- 但没有新生成 token 回给用户


### 9.2 普通 decode 场景

如果一个 request 已经完成 prefill，只差 decode：

```text
n_i = 1
g_i = 1
```

这是最常见、也最容易理解的情况。


### 9.3 speculative decode 场景

如果开启 speculative decode，情况会变成：

- 本轮可能先有若干 draft token
- target verify 后可能一次接受多个 token

于是可能出现：

```text
n_i = 1 + k
g_i = m
```

其中：

- `k` 是 draft 相关的额外计算
- `m` 是最终接受并回填的 token 数
- `m` 可以大于 1


## 10. 例子三：speculative decode 的 shape 直觉

假设某个 request 本轮有 3 个 draft token：

```text
scheduled_spec_decode_tokens = {
  R1: [501, 502, 503]
}
```

worker 执行后，假设 target verify：

- 接受了前 2 个 draft token
- 然后再给出 1 个新的 target token

则返回到 scheduler 的 `generated_token_ids` 可能近似为：

```text
sampled_token_ids[idx_of_R1] = [501, 502, 900]
```

此时：

- `num_draft_tokens = 3`
- `num_accepted = len(generated_token_ids) - 1 = 2`
- `num_rejected = 3 - 2 = 1`

也就是说：

- 被接受的 draft token 会直接作为 output 回填
- 被拒绝的 draft token 要把之前乐观推进的 `num_computed_tokens`
  再修正回来

这也是为什么 scheduler 与 output update 之间会有一个“先推进、后修正”的配合。


## 11. v0 和 v1 的关系

如果你看旧版文章，经常会看到：

- `Sequence`
- `SequenceGroup`
- `ScheduledSequenceGroup`
- `RequestOutput.from_seq_group(...)`

这主要是 `v0` 风格。

### 11.1 v0 的回填方式

旧版 `LLMEngine.step()` 的高层流程大致是：

```text
scheduler.schedule()
  -> model_executor.execute_model(...)
  -> _process_model_outputs(...)
  -> RequestOutput.from_seq_group(...)
```

这里的回填更偏向：

- 先把 sampler output 按 sequence group 拆好
- 再依次更新每个 `SequenceGroup`


### 11.2 v1 的回填方式

`v1` 更偏向 request-centric：

- scheduler 输出 `num_scheduled_tokens`
- worker 返回 `req_id_to_index`
- scheduler 用 `req_id_to_index` 查表回填

所以：

- `v0` 更像“按 sequence group 顺序回填”
- `v1` 更像“按 req_id 显式映射回填”

但本质是一样的：

> 执行 batch 在 GPU 侧可以重排、压平、做优化；
> 但 step 结束后必须有一套稳定映射，把输出放回正确的 request。


## 12. 推荐源码入口

下面给出一份更适合顺着看的源码索引。

### 12.1 v1 主线

#### 1. 调度输出结构

- `vllm/v1/core/sched/output.py`
  - `NewRequestData`
  - `CachedRequestData`
  - `SchedulerOutput`

建议先看它，因为它定义了 scheduler 究竟在给 worker 发送什么。


#### 2. scheduler 主入口

- `vllm/v1/core/sched/scheduler.py`
  - `Scheduler.schedule()`，约 `348`
  - `_make_cached_request_data()`，约 `1055`
  - `update_from_output()`，约 `1302`

这是最核心的一组函数。

尤其推荐按下面顺序看：

1. `schedule()`
2. `_make_cached_request_data()`
3. `update_from_output()`


#### 3. engine step

- `vllm/v1/engine/core.py`
  - `EngineCore.step()`，约 `380`

这能把高层链路串起来：

```text
schedule
  -> execute_model
  -> update_from_output
```


#### 4. worker 侧 batch 组装

- `vllm/v1/worker/gpu/model_runner.py`
  - `add_requests()`，约 `612`
  - `update_requests()`，约 `657`
  - `prepare_inputs()`，约 `667`

如果你最关心 shape，`prepare_inputs()` 是必须看的。

它直接体现：

- `num_scheduled_tokens -> query_start_loc`
- `flat input_ids / positions`
- `seq_lens`
- `cu_num_logits`
- speculative decode 相关展开


#### 5. 输出结构

- `vllm/v1/outputs.py`
  - `SamplerOutput`
  - `ModelRunnerOutput`

这决定了从 GPU 回 scheduler 时到底带了哪些数据。


#### 6. engine 内部输出与最终用户输出

- `vllm/v1/engine/__init__.py`
  - `EngineCoreOutput`
  - `EngineCoreOutputs`
- `vllm/v1/engine/output_processor.py`
  - `RequestState.make_request_output()`，约 `269`
  - `OutputProcessor.process_outputs()`，约 `572`

这部分更偏“回填后的用户接口层”。


### 12.2 旧版 v0 参考线

- `vllm/engine/llm_engine.py`
  - `_process_model_outputs(...)`，约 `510`
  - `step()`，约 `557`

适合在下面两种情况下参考：

- 你看到旧文档 / issue 还在讲 `SequenceGroup`
- 你想对照理解 vLLM 是怎样从旧结构演化到 `v1` 的


## 13. 看源码时建议抓住的 5 个问题

如果你在调试 continuous batching，建议始终围绕下面几个问题读代码。

### 13.1 本轮到底调度了哪些 request

看：

- `num_scheduled_tokens`
- `scheduled_new_reqs`
- `scheduled_cached_reqs`


### 13.2 每个 request 本轮前进了多少 token

看：

- `n_i = num_scheduled_tokens[req_id]`


### 13.3 扁平 batch 的边界在哪里

看：

- `query_start_loc`
- `seq_lens`


### 13.4 输出怎么知道属于哪个 request

看：

- `req_id_to_index`
- `sampled_token_ids[idx]`


### 13.5 request 状态什么时候推进，什么时候修正

看：

- schedule 后 `num_computed_tokens` 的推进
- speculative decode 拒绝后在 `update_from_output()` 中的修正


## 14. 常见误区

### 14.1 “一个 batch 就是一个 `input_ids` 矩阵”

不对。

vLLM 更接近：

```text
input_ids[T]
  + positions[T]
  + query_start_loc[B+1]
  + seq_lens[B]
  + block_tables
  + slot_mapping
```


### 14.2 “decode 只输入 1 个 token，所以计算很简单”

不对。

decode 本轮虽然只新输入 1 个 token id，但它会读取整条历史序列对应的 KV cache，
真正的上下文并没有消失。


### 14.3 “本轮调度了几个 token，就一定返回几个 token”

不对。

要始终区分：

- 计算的 token 数 `n_i`
- 生成并回填的 token 数 `g_i`


### 14.4 “prefill 和 decode 是两套完全不同的调度器”

不对。

在 vLLM 的设计里，它们更像是同一个“按 backlog 前进”的调度框架下的不同常见情形。


## 15. 最终总结

从实现上看，`vLLM continuous batching` 的关键可以归纳成下面几句话：

- scheduler 的核心决策单位不是“固定长度序列”，而是“本步每个 request 前进多少 token”
- worker 的执行核心不是规则的 `B x L`，而是 flat token batch `[T]`
- `query_start_loc / seq_lens / block_tables / slot_mapping` 决定了这些 token 如何映射回各自 request 与 KV cache
- step 结束后，`req_id_to_index` 负责把输出准确拆回 request
- `n_i` 与 `g_i` 不一定相等，这一点对理解 chunked prefill 与 speculative decode 非常重要

所以，continuous batching 真正连续流动的不是“一个静态矩阵”，而是：

- request 集合在流动
- 每步前进 token 数在流动
- GPU token batch 的形状在流动
- 完成与新加入的 request 在每个 step 都会重新重组

这正是它能在在线 serving 中同时兼顾：

- 吞吐
- 低延迟
- 动态请求混合

的根本原因。
