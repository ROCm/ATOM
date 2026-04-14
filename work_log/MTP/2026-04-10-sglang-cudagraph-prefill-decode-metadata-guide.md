# SGLang CUDAGraph、Prefill/Decode 与 Attention Metadata 说明

## 文档目的

这篇文档专门回答下面几个问题：

- `CUDAGraph` 在 `SGLang` 里到底固定了什么
- 为什么 `decode` 更适合做 `CUDAGraph`
- 为什么普通 `prefill / extend` 的 attention 很难复用同一张 graph
- `SGLang` 在 `decode` 阶段是怎样做 `capture / replay` 的
- `attention backend` 的 `ForwardMetadata` 在 graph capture / replay 中扮演什么角色
- 在 `ATOM plugin` 模式下，哪些 graph-only bug 容易出现，为什么


## 一句话结论

最重要的结论先说：

- `CUDAGraph` 固定的不是“某个 Python 函数调用”，而是一整段已经展开好的 CUDA 执行脚本
- `decode` 更适合 graph，不是因为它没有 metadata，而是因为它的 **query 结构、token 数、kernel 形状、workspace 结构** 更稳定
- 普通 `prefill / extend` 难 graph，不是因为 kernel 不能读不同的 metadata，而是因为 metadata 往往不只是“输入数据”，而是会影响：
  - 走哪条代码路径
  - 中间 tensor 分配多大
  - gather 后张量有多长
  - workspace 形状是什么
  - 最终 launch 的 kernel 形态是什么
- 换句话说：
  - `decode` 下，metadata 更像 **kernel 参数**
  - `prefill` 下，metadata 更像 **图结构控制器**


## 1. CUDAGraph 真正固定的是什么

很多人第一次接触 `CUDAGraph` 时，会误以为它只是“把一次 forward 缓存起来”。

更准确地说，`CUDAGraph` capture 固定的是：

- 这次 forward 里实际 launch 了哪些 CUDA kernel
- kernel 的调用顺序
- 每个 kernel 看到的 tensor shape / stride
- 这些 tensor 和 workspace 的内存地址
- Python 层已经展开后的控制流分支

因此：

- **tensor 的值**可以变
- 但 **shape / 地址 / 分支 / launch 计划** 最好不要变

可以把它想成：

- eager 模式像“每次现写一遍执行计划”
- cuda graph 像“录下这次执行计划，以后按原样回放”


## 2. 三个最容易混淆的量：`raw_bs`、`bs`、`num_tokens`

理解 graph 之前，必须先区分三个量：

- `raw_bs`
  - 当前真实 batch 里有多少个 request
- `bs`
  - 当前 replay 选中的 graph bucket 大小
- `num_tokens`
  - 这次真正传给很多 layer 的 token 数

它们经常不相等。

### 2.1 `raw_bs`

这是 scheduler 当前真实调度出来的 request 数。

例如：

- 真实只有 3 个 request 要做 decode
- 那么 `raw_bs = 3`

### 2.2 `bs`

这是 graph 系统为了复用固定 shape，选中的 capture bucket。

例如 capture 过这些 bucket：

- `[1, 2, 4, 8, 16, 32, 48]`

如果这次真实请求数是 3，系统可能会选择：

- `bs = 4`

然后：

- 前 3 个位置放真实请求
- 第 4 个位置放 padding / fill value

### 2.3 `num_tokens`

这不是永远等于 `bs`。

它取决于当前模式下“每个 request 本轮贡献多少 query token”。

几个典型场景：

| 场景 | `num_tokens` |
|------|--------------|
| 普通 decode | `bs * 1` |
| target verify | `bs * num_draft_tokens` |
| draft decode | `bs * topk` |
| draft extend | `bs * (speculative_num_steps + 1)` |
| 普通 prefill / extend | 通常是 `sum(extend_seq_lens)`，不一定等于 `bs * 常数` |

这也是为什么：

- 很多 layer 看到的输入 shape 是 `[num_tokens, hidden_size]`
- 而 graph bucket 却还是按 `bs` 来管理


## 3. SGLang 为什么默认把 graph 重点放在 decode

`SGLang` 的通用 `CudaGraphRunner` 默认 capture 的 forward mode 是 `DECODE`：

- 初始化时先设：
  - `capture_forward_mode = ForwardMode.DECODE`
  - `num_tokens_per_bs = 1`
- 若是 speculative target verify，再切成：
  - `ForwardMode.TARGET_VERIFY`
  - `num_tokens_per_bs = speculative_num_draft_tokens`
- 若是 `DLLM_EXTEND`，再切成 block-size 固定模式

关键点是：

- 这些模式都满足“每个 request 贡献固定个数的 query token”

而普通 `prefill / extend` 不满足这一点。

从 `sglang/python/sglang/srt/model_executor/cuda_graph_runner.py` 可以直接看到这件事：

- graph runner 默认按 `DECODE` 组织
- `num_tokens_per_bs` 是固定常数
- 再用它去算：
  - `max_bs`
  - `max_num_token`
  - 静态输入 buffer 的大小


## 4. 为什么 decode 比 prefill 更适合 graph

### 4.1 decode 的 query 结构稳定

decode 下，一个 request 往往只算一个 query token。

因此：

- `max_q_len` 通常固定为 `1`
- `num_tokens = bs`
- `qo_indptr` 结构非常规则
- kernel 形状更容易随 `bs bucket` 固定下来

即使 metadata 中像：

- `kv_indptr`
- `kv_indices`
- `kv_last_page_len`

这些值每轮都变，它们大多数时候也只是：

- 作为固定 kernel 的输入索引参数

而不是决定“这次图长什么样”。

### 4.2 prefill / extend 的 query 结构是 ragged 的

prefill / extend 下，每个 request 这一轮要处理多少 query token，通常不一样。

例如：

- request A 新增 3 个 token
- request B 新增 17 个 token
- request C 新增 1 个 token

这时：

- `num_tokens = 3 + 17 + 1`
- `qo_indptr` 随分布变化
- `max_q_len` 随分布变化
- `max_kv_len` 也随上下文长度变化

这不是简单的“值不同”，而是 batch 的 **几何结构** 不同。


## 5. 为什么“metadata 改变”会阻碍 prefill graph 复用

这个问题最容易被误解。

### 5.1 先说清楚：metadata 变，不一定阻碍 capture

如果你拿某一个固定的 prefill batch 去做 capture，这次 capture 可能是成功的。

因为那一刻：

- `q.shape`
- `kv_indices.shape`
- `qo_indptr`
- `max_q_len`
- `max_kv_len`

都是确定的。

所以问题不在“这次能不能录下来”，而在：

- **下一次不同的 prefill batch 还能不能 replay 这张图**

### 5.2 decode 中 metadata 更像“数据”

decode 中，metadata 变化通常只是：

- 不同 request 对应不同 KV 索引
- 不同 request 当前上下文长度不同

但最终仍然是在执行同一类 decode kernel。

所以它们更像：

- 同一张图里的输入数据

### 5.3 prefill 中 metadata 更像“图结构控制器”

在普通 MLA extend / prefill 里，metadata 会直接影响：

1. 走哪条 Python 分支
2. 中间张量 shape
3. workspace 大小
4. gather 结果长度
5. kernel 的 `max_q_len / max_kv_len`

这就是根本区别。


## 6. 用 ATOM plugin 的 MLA extend 代码看这个问题

`ATOM/atom/plugin/sglang/attention_backend/sgl_attn_backend.py` 里的 `MLA extend` 很能说明问题。

### 6.1 metadata 先决定本轮的 ragged 结构

普通 MLA extend 初始化时会根据当前 batch 更新：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `max_q_len`
- `max_kv_len`

它们都不是常量，而是来自当前 batch 的 `extend_seq_lens / seq_lens`。

### 6.2 metadata 决定走哪条代码路径

在 `_forward_extend_mla_normal()` 里，代码会根据 prefix 情况和 cache 形态走不同分支：

- 无 prefix
- 有 prefix 且要 decompress
- 有 prefix 且走 absorbed MLA

这意味着：

- 不同 batch 可能走完全不同的子函数
- graph capture 录下来的并不是“抽象 extend”，而是“某一条具体 extend 分支”

如果第一次 capture 时：

- `extend_no_prefix = True`

那录下来的是 `_extend_mla_no_prefix()` 这条图。

下一次如果：

- `extend_no_prefix = False`

且要走 `_extend_mla_absorbed_prefix()`，那已经不是同一张图。

### 6.3 metadata 决定中间 tensor 的 shape

在 no-prefix prefill 路径里，会根据当前 query token 总数构造：

- `temp_kv_indices`
- `output`

它们的 shape 直接依赖：

- `q.shape[0]`
- `total_s`

而 `q.shape[0]` 本身就是当前 ragged batch 展平后的 token 数。

换一个 prefill batch：

- `total_s` 变了
- 中间 tensor shape 跟着变

那 graph 也就不再可复用。

### 6.4 metadata 决定 workspace 的 shape

FP8 prefill 路径里，还会根据：

- `reduce_partial_map.size(0)`
- `total_s`

分配：

- `logits`
- `attn_lse`
- `final_lse`
- `output`

而 `reduce_partial_map` 正是从当前 batch 的分段结构推出来的。

所以这不是“kernel 读不同 metadata”这么简单，而是：

- metadata 直接控制要分配多大的临时缓冲区

### 6.5 metadata 决定 gather 后张量长度

在 absorbed prefix 路径里，会先：

- `k_selected = torch.index_select(K_Buffer, 0, kv_indices)`

这里 `k_selected.shape[0]` 就等于：

- `len(kv_indices)`

而 `kv_indices` 的长度也是当前 batch 的结构量。

因此：

- prefix KV gather 后的张量 shape 也会跟 batch 变化

这会继续向下游 kernel 传播。


## 7. 为什么不能简单靠 padding 解决普通 prefill

有人会自然想到：

- 既然 decode 能靠 bucket + padding 做 graph
- 那 prefill 也可以 pad 到固定 `bs / max_q_len / max_kv_len`

理论上不是完全不行，但工程上代价很大。

### 7.1 decode 的 padding 成本小

decode 一般每个 request 只处理一个 token。

所以即使：

- `raw_bs = 3`
- `bs = 4`

多 pad 一个 request 的成本也比较低。

### 7.2 prefill 的 padding 成本会放大 attention 计算

prefill attention 的成本接近：

- query token 数
- context 长度
- ragged 结构

的组合增长。

如果为了 graph，把所有 request 都 pad 成：

- 大 `max_q_len`
- 大 `max_kv_len`

那么：

- 无效 token 也要参与很多 attention 计算
- mask / metadata 也会跟着变大
- workspace 和显存开销也会膨胀

最后可能：

- graph 省下来的 launch 开销
- 远远抵不过 padding 带来的额外 attention FLOPs


## 8. 为什么 `TARGET_VERIFY` / `DRAFT_EXTEND` 又能 graph

因为它们虽然也不是普通 decode，但仍然满足：

- 每个 request 的 query token 数是固定常数

例如：

- `TARGET_VERIFY`
  - 每个 request 验证 `num_draft_tokens` 个 token
- `DRAFT_EXTEND`
  - 每个 request 固定处理 `speculative_num_steps + 1` 个 token

所以它们仍然可以用：

- `bs bucket`
- `num_tokens_per_bs`

来组织 graph。

换句话说：

- 它们不是“完全自由的 ragged prefill”
- 而是“固定 token-per-request 的特殊 extend”

因此 graph 化难度明显低于普通 prefill。


## 9. SGLang 在 decode 阶段怎样做 CUDAGraph capture

下面按实际代码链路讲。

### 9.1 第一步：决定 capture 模式和 bucket

`CudaGraphRunner.__init__()` 中会：

1. 设定 `capture_forward_mode`
2. 设定 `num_tokens_per_bs`
3. 通过 `get_batch_sizes_to_capture()` 得到 `capture_bs`
4. 算出：
   - `max_bs`
   - `max_num_token = max_bs * num_tokens_per_bs`

这一步的意义是：

- graph 系统先把“这类 forward 的形状规则”固定下来
- 然后再一次性分配足够大的静态 buffer

### 9.2 第二步：attention backend 先分配 graph 专用静态状态

接着会调用：

- `attn_backend.init_cuda_graph_state(max_bs, max_num_token)`

在 `ATOMAttnBackendForSgl` 里，这一步会分配 graph 期间复用的持久 buffer，例如：

- `cuda_graph_kv_last_page_len`
- `cuda_graph_kv_indices`
- `page_table`
- `seq_lens`
- MLA decode 的 `work_metadata / work_indptr / work_info_set / reduce_*`

这里的关键思想是：

- graph replay 期间，不再频繁新建这些结构
- 而是在固定 buffer 上反复更新其内容

### 9.3 第三步：为某个具体 bucket 构造静态输入视图

在 `capture_one_batch_size(bs)` 中，会从大 buffer 上切出本 bucket 对应的视图，例如：

- `input_ids = buffers.input_ids[:num_tokens]`
- `req_pool_indices = buffers.req_pool_indices[:bs]`
- `seq_lens = buffers.seq_lens[:bs]`
- `positions = buffers.positions[:num_tokens]`

然后构造一个 `ForwardBatch`：

- `forward_mode = capture_forward_mode`
- `batch_size = bs`
- 大部分字段都直接绑定到这些静态 buffer 视图上

### 9.4 第四步：capture 前先初始化 attention metadata

在真正 `graph capture` 之前，先调用：

- `attn_backend.init_forward_metadata_capture_cuda_graph(...)`

对 decode 来说，这一步本质上是：

- 根据当前 `req_pool_indices / seq_lens`
- 把 `ForwardMetadata` 组装到 graph 专用的静态 buffer 视图上

对于 MLA decode，它会构造：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`
- `work_metadata / work_indptr / work_info_set / reduce_*`

而这些对象大多来自：

- graph state 中预先分配好的持久 buffer

### 9.5 第五步：跑几次 warmup，再进入真正 graph capture

`capture_one_batch_size()` 里会：

1. 先调用几次 `run_once()`
2. 再进入 `torch.cuda.graph(...)`
3. 把这次 forward 录下来

在这个过程中：

- 输入 buffer 地址固定
- metadata buffer 地址固定
- forward mode 固定
- kernel 形状固定

于是得到一张与 bucket `bs` 绑定的 graph。


## 10. SGLang 在 decode 阶段怎样做 replay

### 10.1 先从真实 batch 选一个 bucket

在 `replay_prepare()` 中：

1. 读取真实 batch 的：
   - `raw_bs`
   - `raw_num_token`
2. 从 `capture_bs` 中找一个：
   - `bs >= raw_bs`

这一步就是把真实 batch 映射到 graph bucket。

### 10.2 把真实数据 copy 到静态 buffer 的前缀

调用：

- `buffers.populate_from_forward_batch(...)`

会把真实 batch 的内容写入静态 buffer 的前缀区域，例如：

- `input_ids[:raw_num_token]`
- `req_pool_indices[:raw_bs]`
- `seq_lens[:raw_bs]`
- `positions[:raw_num_token]`

如果 `bs != raw_bs`，还会：

- 用 fill value / zero 对后面的 padding 段做补齐

### 10.3 replay 前重建本轮 metadata

随后调用：

- `attn_backend.init_forward_metadata_replay_cuda_graph(...)`

注意这一步非常关键：

- graph replay 不是复用 capture 当时的 metadata 值
- 而是复用 **metadata 的静态 buffer 与构造方式**
- 然后把本轮真实 batch 的索引内容重新写进去

也就是说：

- 地址固定
- 内容可变

对 decode 来说，这正是 graph 友好的做法。

### 10.4 最后 `graph.replay()`

当静态 buffer 和 metadata 都准备好后，就直接：

- `self.graphs[graph_key].replay()`

执行那张已经 capture 好的图。

输出拿到后，再按照：

- `raw_bs`
- `raw_num_token`

把 padding 的尾部裁掉。


## 11. Attention Metadata 在 graph capture / replay 中的角色

可以把 `ForwardMetadata` 在 graph 中的角色概括成一句话：

- 它不是 graph 外的额外说明书
- 它是 graph 里 attention kernel 的直接输入

但它在不同模式下的“地位”不同。

### 11.1 decode 中：metadata 更像固定地址上的输入参数

decode graph 下，像：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `kv_last_page_len`

更多是在表达：

- 当前 batch 的 KV 可见范围
- 当前 batch 的 query 分段

这些值会变，但：

- 它们所在的 buffer 地址固定
- 它们的 shape 规则受 bucket 控制
- 下游还是同一类 decode kernel

所以 graph 可以复用。

### 11.2 prefill 中：metadata 会升级成“图结构的一部分”

prefill / extend 下，metadata 往往不仅仅被 kernel 读取，还会影响：

- 选择哪条路径
- 构造哪些中间 tensor
- 中间 tensor 有多大
- workspace 有多大
- kernel 看到的 `max_q_len / max_kv_len`

因此：

- metadata 变化会把“图长什么样”一起改掉

这就是它阻碍 graph 复用的根本原因。


## 12. `ATOMAttnBackendForSgl` 中 graph metadata 的几个关键点

### 12.1 `init_cuda_graph_state()`：先分配 graph 专用持久 buffer

plugin backend 里专门分配了：

- `cuda_graph_kv_last_page_len`
- `cuda_graph_kv_indices`
- `page_table`
- `seq_lens`
- MLA decode 的 persistent workspace

这样 replay 时就能复用这些地址。

### 12.2 `init_forward_metadata_capture_cuda_graph()`：把 bucket 数据写成 metadata

这一步会根据当前 mode 做不同初始化：

- `decode_or_idle`
- `target_verify`
- `draft_extend`

每种模式都把：

- bucket 对应的 `bs`
- 固定的 `num_tokens_per_bs`
- 当前 request 索引和 seq_lens

转成 kernel 需要的 metadata。

### 12.3 `init_forward_metadata_replay_cuda_graph()`：重建本轮 metadata

replay 时，plugin backend 不会继续沿用 capture 时那一轮的 metadata 值，而是：

- 在固定 graph buffer 上
- 根据本轮真实 batch 重建一次 metadata

这一步必须非常小心“当前 bucket 视图”和“整块静态 buffer”的区别。

最近在 debug 中出现的一个典型 graph-only bug 正是：

- 上游 replay 某条 speculative 路径把整块静态 buffer 传下来
- plugin backend 按“已经是当前 `bs` 视图”去理解
- 于是出现：
  - `bs = 1`
  - `seq_lens.shape[0] = 48`

后来在 plugin backend 里做了统一切片规整：

- `req_pool_indices = req_pool_indices[:bs]`
- `seq_lens = seq_lens[:bs]`
- `seq_lens_cpu = seq_lens_cpu[:bs]`

本质上就是把 replay 的输入重新对齐到“当前 bucket 视图”。


## 13. 这次 debug 暴露出的两个 graph-only 经验

### 13.1 backend 选型必须真的落到 plugin backend

之前 `kv_last_page_len` 掉到 CPU 的问题，最后定位到：

- `AiterMultiStepDraftBackend` 内部直接实例化 `AiterAttnBackend`
- 绕过了 plugin 通过 registry 注册的 `"aiter" -> ATOMAttnBackendForSgl`

这说明：

- graph-only 路径里，某些 backend 可能不是从常规 registry 路径拿到的
- 如果 direct construction 没 patch 到，graph state 就可能偷偷回落到 upstream 实现

### 13.2 replay 必须明确区分“静态大 buffer”和“当前 bucket 视图”

graph replay 中，静态 buffer 通常按：

- `max_bs`
- `max_num_token`

一次性分配。

但 backend 在构 metadata 时，真正应该看到的是：

- 当前 bucket 的前 `bs`
- 当前 token 的前 `num_tokens`

一旦把整块静态 buffer 当成当前视图使用，就很容易出现：

- shape mismatch
- CPU / CUDA tensor 混用
- metadata 与实际 batch 不一致


## 14. 用一句工程化的话总结

如果只用一句最工程化的话来总结这篇文档：

- `decode` 图里，metadata 大多是 **固定形状 graph 的输入数据**
- `prefill` 图里，metadata 往往会变成 **决定图形状和执行路径的结构量**

因此：

- `decode` 适合用 bucket + padding + 静态 buffer 做 graph
- 普通 `prefill / extend` 则很难在收益合理的前提下复用同一张 graph


## 15. 最后总结

记住下面五句话就够了：

1. `CUDAGraph` 固定的是一整段具体 CUDA 执行计划，不只是 Python 函数入口。
2. `raw_bs` 是真实请求数，`bs` 是 graph bucket，`num_tokens` 是真正传给很多 layer 的 token 数。
3. `decode` 更适合 graph，因为每个 request 的 query 结构更稳定，metadata 更像输入参数。
4. 普通 `prefill / extend` 难 graph，因为 metadata 会影响分支、shape、workspace 和 kernel 形态，升级成图结构的一部分。
5. 在 `SGLang + ATOM plugin` 里，graph replay 的关键不是“重复使用旧 metadata 值”，而是“在固定 buffer/地址上重建本轮 metadata 内容”。
