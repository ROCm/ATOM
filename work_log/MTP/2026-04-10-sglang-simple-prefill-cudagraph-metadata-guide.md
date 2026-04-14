# 最简单 Prefill、CUDAGraph 与 Metadata 速查

## 文档目的

这篇文档只回答一个收窄后的问题：

- **不考虑不同 kernel path**
- **不考虑 prefix cache**
- **不考虑 speculative target_verify / draft_extend**
- **只考虑最普通、最简单的 prefill / extend**

在这个前提下，说明：

1. 为什么这种最简单的 prefill 仍然难做 `CUDAGraph`
2. attention metadata 的核心字段在这种场景下分别表示什么
3. 给几个可以手算的小例子，方便以后速查


## 一句话结论

即使只看最简单 prefill，`CUDAGraph` 的挑战仍然存在。根本原因不是“metadata 会变”本身，而是：

- `total_tokens` 会变
- `max_q_len / max_kv_len` 会变
- ragged metadata 会跟着 batch 几何结构一起变
- 很多中间 tensor / workspace 的 shape 也会变

所以问题不是：

- kernel 能不能读取不同的 metadata 值

而是：

- **同一个 prefill batch family，能不能稳定成一张固定 shape 的图**


## 1. 本文说的“最简单 prefill”是什么

这里约定的“最简单 prefill”是：

- 没有 prefix cache
- 没有 speculative 分支
- 不讨论不同 kernel path 的切换
- 假定已经选中某一条固定的 prefill kernel 路径
- 一个 batch 里有若干 request
- 每个 request 本轮需要处理若干 query token
- attention 以 ragged / varlen 形式运行

可以把它理解成：

- `ForwardMode.EXTEND`
- `spec_info = None`
- `extend_prefix_lens = 0`
- attention backend 已经决定“就走这条 prefill kernel”

本文不讨论：

- prefix/no-prefix 的 kernel 分流
- absorbed / decompress 等 MLA 专有分流
- draft_extend / target_verify
- decode


## 2. 最简单 prefill 的数据形状

prefill 和 decode 最大的不同在于：

- decode 常常是每个 request 本轮只算 1 个 token
- prefill 常常是每个 request 本轮要算多个 token

因此，很多 layer 真正看到的不是：

- `[bs, hidden_size]`

而是：

- `[total_tokens, hidden_size]`

其中：

- `bs` = request 数
- `total_tokens` = 本轮所有 request 的 query token 总数

在最简单 prefill 下，常见关系是：

```text
total_tokens = sum(extend_seq_lens)
```

这意味着：

- 即使 `bs` 不变
- 只要每个 request 的长度分布变了
- `total_tokens` 就会变


## 3. 为什么最简单 prefill 仍然难做 CUDAGraph

下面只看“最简单 prefill”，不引入分支复杂度。

### 3.1 `q/k/v/o` 的 token 维会变

prefill 下最直观的问题就是：

- `q.shape[0] = total_tokens`
- `k.shape[0] = total_tokens`
- `v.shape[0] = total_tokens`
- `o.shape[0] = total_tokens`

只要：

- request 数不同
- 或每个 request 的 query 长度分布不同

那么：

- `total_tokens` 就不同
- 上面这些张量 shape 就不同

而 `CUDAGraph` 更喜欢的是：

- tensor shape 固定
- graph 中 kernel launch 形态固定

这已经是第一层挑战。

### 3.2 `max_q_len / max_kv_len` 会变

即使不看 `q.shape[0]`，varlen attention 往往还会显式传：

- `max_q_len`
- `max_kv_len`
- `cu_seqlens_q` 或 `qo_indptr`

这些量不是 decoration，而是 kernel 的核心输入。

例如对于一个 batch：

- request A: 3 tokens
- request B: 2 tokens

则：

- `max_q_len = 3`

如果下一个 batch 是：

- request A: 4 tokens
- request B: 1 token

则：

- `max_q_len = 4`

虽然：

- 两个 batch 的 `bs = 2`
- 两个 batch 的 `total_tokens = 5`

但：

- `qo_indptr` 不同
- `max_q_len` 不同

这意味着：

- 内部 tile / launch 策略可能不同
- workspace 需求也可能不同

### 3.3 ragged metadata 在描述“问题几何结构”

在 decode 里，很多 metadata 更像：

- 固定图上的输入索引数据

而在 prefill 里，metadata 往往在表达：

- 一共有多少 query token
- 这些 query token 怎样按 request 分段
- KV token 怎样按 request 分段
- 当前 batch 的最大 query / KV 长度是多少

所以它不只是“值会变”，而是在描述：

- **这轮 attention 问题本身长什么样**

这就让同一张图更难复用。

### 3.4 中间 tensor / workspace 的 shape 也会变

哪怕我们强行假设：

- kernel path 不变

很多中间结构也仍然可能随 batch 变化。

例如：

- 某些临时索引张量长度跟 `total_tokens` 走
- 某些 workspace 大小跟 `max_q_len / max_kv_len` 走
- 某些 reduce buffer 大小跟分段结构走

所以问题不止在输入张量，而是：

- graph 内部很多“中间物体”的 shape 也不稳定

### 3.5 同一个分支里也可能无法稳定 replay

这点最容易误解。

即使你已经保证：

- 一定走同一个 prefill kernel path

也不代表可以 graph。

因为同一条路径里仍然可能有：

- `total_tokens` 变化
- `max_q_len` 变化
- `max_kv_len` 变化
- workspace shape 变化

所以：

- “分支固定”

并不等于：

- “图固定”


## 4. Metadata 速查表

下面只保留最常用于“最简单 prefill”理解的字段。

### 4.1 高层 batch 字段

| 字段 | 常见 shape | 物理意义 | 简单 prefill 下的作用 |
|------|------------|----------|-----------------------|
| `bs` | Python `int` | request 数 | 当前 batch 有几个 request |
| `extend_seq_lens` | `[bs]` | 每个 request 本轮 query token 数 | 决定 `total_tokens` 和 `qo_indptr` |
| `seq_lens` | `[bs]` | 每个 request 当前可见 KV 长度 | 决定 `kv_indptr` 和 `max_kv_len` |
| `seq_lens_sum` | Python `int` | 所有 request KV 长度总和 | 常用于辅助构造 KV metadata |
| `req_pool_indices` | `[bs]` | request 在 `req_to_token` 里的行号 | 用来从映射表里取物理 KV slot |

### 4.2 Query 侧 metadata

| 字段 | 常见 shape | 物理意义 | 简单 prefill 下的作用 |
|------|------------|----------|-----------------------|
| `qo_indptr` | `[bs + 1]` | flatten 后每个 request 的 query 段边界 | 告诉 kernel 哪些 query 属于哪个 request |
| `max_q_len` | Python `int` | batch 内单 request 最大 query 长度 | kernel 的长度上限参数 |
| `total_tokens` | Python `int` | flatten 后 query token 总数 | 决定很多输入/输出第一维 |

### 4.3 KV 侧 metadata

| 字段 | 常见 shape | 物理意义 | 简单 prefill 下的作用 |
|------|------------|----------|-----------------------|
| `kv_indptr` | `[bs + 1]` | flatten 后每个 request 的 KV 段边界 | 告诉 kernel 每段 KV 从哪里到哪里 |
| `kv_indices` | `[sum(seq_lens)]` 或相近长度 | flatten 后每个 KV token 对应的物理 slot | 真正告诉 kernel 去读哪些物理 KV |
| `max_kv_len` | Python `int` | batch 内单 request 最大 KV 长度 | kernel 的 KV 长度上限参数 |
| `kv_last_page_len` | `[bs]` | 每个 request 最后一页有效 token 数 | paged MLA kernel 常用 |
| `kv_lens` | `[bs]` | 每个 request 当前 KV 长度 | 在 page-table 表达里常用 |
| `page_table` | `[bs, max_pages]` | request 到 page id 的二维映射 | 非 MLA / page-table 风格 backend 常用 |


## 5. 这些字段的物理意义，最简单地怎么记

### 5.1 `qo_indptr`

记法：

- 它是 query 侧的 CSR 前缀和边界表

典型 shape：

- `[bs + 1]`

dtype：

- 通常是 `int32`

含义：

- 第 `i` 个 request 的 query 在 flatten Q 中的范围是：
  - `[qo_indptr[i], qo_indptr[i+1])`

和哪些量对应：

- `qo_indptr[0]` 固定是 `0`
- `qo_indptr[-1]` 通常等于：
  - `total_tokens`
- `qo_indptr[i + 1] - qo_indptr[i]` 等于：
  - 第 `i` 个 request 的 query 长度

### 5.2 `kv_indptr`

记法：

- 它是 KV 侧的 CSR 前缀和边界表

典型 shape：

- `[bs + 1]`

dtype：

- 通常是 `int32`

含义：

- 第 `i` 个 request 的 KV 在 flatten `kv_indices` 中的范围是：
  - `[kv_indptr[i], kv_indptr[i+1])`

和哪些量对应：

- `kv_indptr[0]` 固定是 `0`
- `kv_indptr[-1]` 通常等于：
  - `len(kv_indices)`
- `kv_indptr[i + 1] - kv_indptr[i]` 等于：
  - 第 `i` 个 request 当前参与 attention 的 KV 长度

### 5.3 `kv_indices`

记法：

- 它是“这次 attention 真正要访问的物理 KV slot 列表”

典型 shape：

- `[sum(seq_lens)]`
- 更严格一点说：
  - `[kv_indptr[-1]]`

dtype：

- 通常是 `int32`

含义：

- 每个元素都是一个 physical KV slot id

更具体一点：

- `kv_indices` 不是“第几个 token”
- 也不是“第几个 request”
- 它是：
  - **flatten 后，每个 KV token 在物理 KV cache 里的实际位置**

它和下面几个量要一起看：

- `req_pool_indices`
  - shape 通常是 `[bs]`
  - 告诉你“当前 batch 里每个 request 对应 `req_to_token` 的哪一行”
- `req_to_token`
  - shape 通常是 `[req_pool_size, max_context_len]`
  - 告诉你“这个 request 的逻辑第 `j` 个 token，物理上写在 KV cache 的哪个 slot”
- `seq_lens`
  - shape 通常是 `[bs]`
  - 告诉你“这个 request 当前有多少个 KV token 参与 attention”
- `kv_indptr`
  - shape 通常是 `[bs + 1]`
  - 告诉你“这个 request 对应的 KV 段，在 flatten 后 `kv_indices` 里的哪一段”

所以可以把 `kv_indices` 理解成：

- 先按 `req_pool_indices` 找到每个 request 在 `req_to_token` 中的那一行
- 再按 `seq_lens[i]` 取出这行前面的有效 token 映射
- 最后把所有 request 的映射段拼接起来

也就是说：

- `kv_indptr` 负责“分段边界”
- `kv_indices` 负责“段内具体有哪些 physical slot”

### 5.3.1 它为什么重要

attention kernel 真正关心的不是：

- “这是第几个逻辑 token”

而是：

- “要去 KV cache 的哪个物理位置读 K/V”

`kv_indices` 正是在回答这个问题。

如果没有 `kv_indices`，kernel 只知道：

- batch 里有几个 request
- 每个 request 长度是多少

但它仍然不知道：

- 这些 request 的历史 token 到底落在 KV cache 里的哪些 physical slot 上

### 5.3.2 它为什么通常是 flatten 的

`kv_indices` 做成一维 flatten 形式，而不是二维 `[bs, max_kv_len]`，是因为：

- 不同 request 的 KV 长度不一样
- ragged attention 更自然的表示法就是：
  - 一条长数组
  - 再配一个 `kv_indptr`

这和 CSR 稀疏矩阵的表达方式很像：

- `kv_indices` = 数据主体
- `kv_indptr` = 每段边界

### 5.3.3 它和 `total_tokens` / `max_kv_len` 的区别

这几个量很容易混：

- `total_tokens`
  - shape 是标量 / Python `int`
  - query 侧总 token 数
- `max_kv_len`
  - shape 是标量 / Python `int`
  - 单 request 最大 KV 长度
- `kv_indices`
  - shape 是一维张量 `[sum(seq_lens)]`
  - 这轮 attention 真正要访问的所有 physical KV slot 列表

它们不是一回事。

例如：

- `bs = 2`
- `seq_lens = [3, 2]`

那么：

- `max_kv_len = 3`
- `len(kv_indices) = 5`

前者是“最大段长度”，后者是“所有段拼起来后的总长度”。

### 5.3.4 一个更完整的手算例子

假设：

- `req_pool_indices = [7, 9]`
  - shape: `[2]`
- `seq_lens = [5, 3]`
  - shape: `[2]`
- `req_to_token[7, 0:5] = [100, 101, 102, 103, 120]`
- `req_to_token[9, 0:3] = [200, 201, 220]`

那么先算边界：

```text
kv_indptr = [0, 5, 8]
```

它的 shape 是：

- `[3]`，也就是 `[bs + 1]`

再按每个 request 的有效长度取映射：

- request 0 取：
  - `[100, 101, 102, 103, 120]`
- request 1 取：
  - `[200, 201, 220]`

最后拼接得到：

```text
kv_indices = [100, 101, 102, 103, 120, 200, 201, 220]
```

它的 shape 是：

- `[8]`
- 也就是：
  - `[sum(seq_lens)] = [5 + 3]`

于是：

- request 0 的 KV 段是：
  - `kv_indices[kv_indptr[0]:kv_indptr[1]]`
  - 也就是 `kv_indices[0:5]`
- request 1 的 KV 段是：
  - `kv_indices[kv_indptr[1]:kv_indptr[2]]`
  - 也就是 `kv_indices[5:8]`

### 5.3.5 debug 时怎么看 `kv_indices`

如果你在 debug attention metadata，`kv_indices` 最值得看两件事：

1. 长度对不对

- 在最简单 prefill 里，通常应该有：
  - `len(kv_indices) == sum(seq_lens)`
- 也可以写成：
  - `kv_indices.shape == (int(seq_lens.sum()),)`

如果这个关系都不对，说明：

- `kv_indptr`
- `seq_lens`
- 或 `req_to_token` 的使用

有地方没对齐。

2. 分段内容对不对

给定：

- `kv_indptr`
  - shape: `[bs + 1]`
- `kv_indices`
  - shape: `[sum(seq_lens)]`

你应该能把每个 request 对应的 physical slot 段切出来，并和：

- `req_to_token[row, :seq_len]`

一一对应上。

如果切出来的段和 `req_to_token` 不对应，常见意味着：

- `req_pool_indices` 行号不对
- `seq_lens` 不是这轮应看的 KV 长度
- 或者 graph replay 时把“整块静态 buffer”错当成了“当前 bucket 视图”

### 5.4 `max_q_len`

记法：

- 这轮 batch 中，单 request 最长的 query 长度

shape：

- 标量 / Python `int`

它不是：

- 所有 query 总数

### 5.5 `max_kv_len`

记法：

- 这轮 batch 中，单 request 最长的 KV 长度

shape：

- 标量 / Python `int`


## 6. 最简单 prefill 的三个小例子

### 例子 1：单 request prefill

假设：

- `bs = 1`
- `extend_seq_lens = [5]`
  - shape: `[1]`
- `seq_lens = [5]`
  - shape: `[1]`

那么：

- `total_tokens = 5`
- `qo_indptr = [0, 5]`
  - shape: `[2]`
- `kv_indptr = [0, 5]`
  - shape: `[2]`
- `max_q_len = 5`
- `max_kv_len = 5`

如果 `req_to_token[row]` 对应的是：

- `[100, 101, 102, 103, 104]`

那么：

- `kv_indices = [100, 101, 102, 103, 104]`
  - shape: `[5]`

这个例子很简单，但也正好说明：

- 这轮 graph 里主干很多 tensor 第一维都是 `5`

如果下一个 batch 变成 `7` 个 token：

- 图里的很多 shape 就都要变

### 例子 2：两个 request，长度不同

假设：

- `bs = 2`
- `extend_seq_lens = [3, 2]`
  - shape: `[2]`
- `seq_lens = [3, 2]`
  - shape: `[2]`

那么：

- `total_tokens = 5`
- `qo_indptr = [0, 3, 5]`
  - shape: `[3]`
- `kv_indptr = [0, 3, 5]`
  - shape: `[3]`
- `max_q_len = 3`
- `max_kv_len = 3`

如果：

- request 0 的物理 slot 是 `[10, 11, 12]`
- request 1 的物理 slot 是 `[20, 21]`

那么：

- `kv_indices = [10, 11, 12, 20, 21]`
  - shape: `[5]`

这里最值得注意的是：

- `total_tokens = 5`
- 但 request 分段结构已经不是均匀的

### 例子 2.1：把 `qo_indptr + kv_indptr + kv_indices` 放在一起看

继续沿用上面的 batch：

- `bs = 2`
- `extend_seq_lens = [3, 2]`
- `seq_lens = [3, 2]`
- `qo_indptr = [0, 3, 5]`
- `kv_indptr = [0, 3, 5]`
- `kv_indices = [10, 11, 12, 20, 21]`

如果把 flatten 后的 Q token 记成：

```text
Q_flat = [q0, q1, q2, q3, q4]
```

那么 query 侧分段是：

- request 0:
  - `Q_flat[0:3]`
  - 也就是 `q0, q1, q2`
- request 1:
  - `Q_flat[3:5]`
  - 也就是 `q3, q4`

因为：

```text
qo_indptr = [0, 3, 5]
```

同样，KV 侧分段是：

- request 0:
  - `kv_indices[0:3]`
  - 也就是 `[10, 11, 12]`
- request 1:
  - `kv_indices[3:5]`
  - 也就是 `[20, 21]`

因为：

```text
kv_indptr = [0, 3, 5]
kv_indices = [10, 11, 12, 20, 21]
```

把它们并排看，就是：

```text
request 0:
  Q range  = [qo_indptr[0], qo_indptr[1]) = [0, 3)
  Q tokens = [q0, q1, q2]
  KV range = [kv_indptr[0], kv_indptr[1]) = [0, 3)
  KV slots = [10, 11, 12]

request 1:
  Q range  = [qo_indptr[1], qo_indptr[2]) = [3, 5)
  Q tokens = [q3, q4]
  KV range = [kv_indptr[1], kv_indptr[2]) = [3, 5)
  KV slots = [20, 21]
```

这就是 ragged attention metadata 最核心的意思：

- `qo_indptr`
  - 告诉 kernel：flatten 后哪些 query 属于哪个 request
- `kv_indptr`
  - 告诉 kernel：flatten 后哪些 KV 段属于哪个 request
- `kv_indices`
  - 告诉 kernel：这个 request 的 KV 段具体对应哪些 physical KV slot

如果再把 `req_to_token` 写出来：

```text
req_to_token[row_of_req0, 0:3] = [10, 11, 12]
req_to_token[row_of_req1, 0:2] = [20, 21]
```

那就能看到：

- `kv_indices`
  本质上就是把每个 request 在 `req_to_token` 里的有效前缀切出来，再按 request 顺序拼起来。

### 例子 3：`total_tokens` 一样，但 graph 仍然难复用

看两个 batch：

#### batch A

- `bs = 2`
- `extend_seq_lens = [3, 2]`
  - shape: `[2]`

得到：

- `total_tokens = 5`
- `qo_indptr = [0, 3, 5]`
  - shape: `[3]`
- `max_q_len = 3`

#### batch B

- `bs = 2`
- `extend_seq_lens = [4, 1]`
  - shape: `[2]`

得到：

- `total_tokens = 5`
- `qo_indptr = [0, 4, 5]`
  - shape: `[3]`
- `max_q_len = 4`

这两个 batch：

- `bs` 相同
- `total_tokens` 相同

但：

- `qo_indptr` 不同
- `max_q_len` 不同

这说明：

- 即使总 token 数没变
- prefill 的“问题几何结构”仍然变了

这就是 graph 复用困难的关键例子。

### 例子 4：为什么 decode 更容易 graph

假设 decode：

- `bs = 2`
- 每个 request 本轮只解 1 个 token

那么：

- `total_tokens = 2`
- `qo_indptr = [0, 1, 2]`
  - shape: `[3]`
- `max_q_len = 1`

下一个 batch 只要 bucket 还是这个 `bs`，即使：

- `kv_indices`
- `seq_lens`
- `kv_indptr`

的内容变了，graph 里主干 shape 往往还是稳定得多。

所以：

- decode 中 metadata 更像“数据表”
- prefill 中 metadata 更像“几何结构描述”


## 7. 如果硬要对最简单 prefill 做 graph，需要什么条件

最少需要做下面几件事中的一些：

### 7.1 固定 `bs`

最基础的 bucket 化：

- 只允许某几个 `bs` 值

但仅固定 `bs` 还不够。

### 7.2 固定 `total_tokens`

因为很多输入/输出 tensor 的第一维是：

- `total_tokens`

若它不固定，graph 很难复用。

### 7.3 固定 `max_q_len / max_kv_len`

因为它们常常影响：

- kernel launch 形态
- workspace 大小

### 7.4 固定 workspace 形状

也就是说：

- 需要让中间临时张量有固定上限
- 或者直接预分配到某个 bucket 上限

### 7.5 允许 padding / pack / unpack

最现实的手段通常是：

- graph 外把 ragged batch 归一化
- graph 内只处理固定形状张量
- graph 后再 unpad

但代价是：

- 额外数据搬运
- padding 带来的无效计算


## 8. 为什么这比 decode 难很多

可以用一句最简单的话来对比：

- `decode` 的不确定性主要是“数据值不同”
- `prefill` 的不确定性主要是“问题结构不同”

decode 常常可以做到：

- 固定 `num_tokens_per_bs = 1`
- 固定 `max_q_len = 1`
- 只靠 `bs bucket` 就稳定大部分形状

而最简单 prefill 仍然会遇到：

- `total_tokens` 变化
- `qo_indptr` 变化
- `max_q_len` 变化
- `max_kv_len` 变化
- 中间 workspace 变化


## 9. 最后总结

只记下面六句话就够了：

1. 最简单 prefill 也不是固定 shape 问题，而是 ragged / varlen 问题。
2. `total_tokens = sum(extend_seq_lens)`，它决定了很多主干张量的第一维。
3. `qo_indptr` 和 `kv_indptr` 不是装饰字段，而是在描述这轮 attention 的分段几何结构。
4. `max_q_len / max_kv_len` 会随着 batch 分布变化，常常进一步影响 kernel 和 workspace。
5. 即使不考虑不同 kernel path，prefill 仍然可能因为 shape 和 workspace 不稳定而难以复用同一张 graph。
6. 如果真的想 graph 化最简单 prefill，通常还需要 bucket 化、padding 或 pack/unpack 来先把 ragged 问题归一化。
