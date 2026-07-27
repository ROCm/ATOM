# DeepSeek-V4 Paged Decode：ATOM 的独特设计

本文介绍 DeepSeek-V4 在 ATOM 中的 sparse decode attention 实现 ——
`sparse_attn_v4_paged_decode` 这个统一 paged kernel（page_size = 1）的设计动机、
KV cache 布局、索引构造以及 CUDAGraph 友好性的关键约束，目标读者是希望理解或扩
展 V4 attention 后端的工程师。

> 阅读建议：§1-§5 建立基础心智模型（设计目标、层结构、KV layout、索引翻译、
> kernel 接口）；§6 介绍 per-forward 索引 buffer 构造；§7 是 V4-Pro MTP-1 下
> 索引展开的具体行为；§8 起是 dispatch 与 CUDAGraph 工程细节。

---

## 0. 与 upstream V4 reference 的差异（ATOM 特有）

DeepSeek 官方 reference（`/data/DeepSeek-V4-Pro/inference/model.py`）按"每种 ratio
一条 code path"的方式实现 attention，SWA ring buffer 和 compressor paged KV 是
独立张量，每 forward 现场把"被 indexer 挑中的 K"materialize 成一个临时 dense
tensor 再喂给 dense attention kernel。这个 reference impl 不能进 CUDAGraph
capture，因为临时 tensor 的 shape 依赖 device-side 数据。

ATOM 的实现做了 4 个根本性的重构，目标是**让 V4 decode 全程零 host-sync、零
per-forward 动态 shape 分配，能完整进入 CUDAGraph capture**：

1. **Unified KV pool（§3）**：每层把 SWA 环形缓冲和 compressor paged KV
   **物理合并**成一个 BF16 tensor。kernel 只持一个 base pointer，所有索引（不
   论来自 SWA、CSA Main、HCA Main）都是该 pool 上的 row offset。upstream 是两
   个独立张量、kernel 内部分支判断。
2. **统一 paged kernel（§5）**：自写一个 V4 专属的 triton kernel
   `sparse_attn_v4_paged_decode`，page_size = 1，接口与 ATOM V3.2
   `mla_decode_fwd` 对齐（`kv_indices` + `kv_indptr` packed cumsum + `attn_sink`），
   Dense / CSA / HCA 三种层用**同一个** kernel，只是给不同 indptr。upstream 的
   per-ratio dispatch 被这层抽象消除。
3. **CG-friendly 索引构造（§6）**：所有 per-forward 索引张量（SWA uniform stride
   buffer、CSA / HCA packed-cumsum kv_indptr 和 kv_indices）在 metadata builder
   阶段一次性算好，**全部 size 上界都是 metadata-time 已知常量**（`max_num_reqs *
   (1 + max_spec_steps)` 等），无 `.item()`，无 device-data-dependent 分配。CSA
   per-layer 翻译用一个固定 grid 的 `csa_packed_write` triton kernel
   （§6.4）替代 upstream 的 fancy scatter（fancy scatter 是动态 shape，CG 不可用）。
4. **MTP-1 via packed indptr 物理重复（§7）**：MTP 把每 request 的 token 数从 1
   翻成 `1 + max_spec_steps`。ATOM 选择在 packed-indptr 里**物理重复存储** base
   和 draft token 各自的 ring slot/topk 段，不做共享。换 6.5% 的 buffer 内存
   省去了"共享 KV slot 时 indptr 的复杂簿记"，kernel 完全无需感知 MTP，所有 per-token
   维度只看 `total_tokens = num_reqs * (1 + max_spec_steps)`。

### 为什么不直接复用 V3.2 的 `mla_decode_fwd`

V3.2 sparse decode 用的是 aiter `mla_decode_fwd`，看起来很像，但有 3 个不可调和
的差异：

| 项目              | V3.2 mla_decode_fwd      | V4 需要                                   |
|-------------------|---------------------------|-------------------------------------------|
| attn_sink         | 无                        | 有（per-head 可学习 softmax-denom bias）  |
| page_size         | 64                        | 1（SWA / CSA / HCA 三种 stride 各异）     |
| KV 来源           | 单一 paged KV cache       | SWA ring + 多种 compressor paged KV       |

把 V4 的多 stride 索引压回 page_size=64 的 paged 接口需要为每层重排 KV，成本超
过自写 kernel。所以 ATOM 选择**保留 V3.2 命名约定**（`kv_indices` /
`kv_indptr` / `softmax_scale` / `nhead`）让两套实现共用阅读心智，但 kernel 是
独立的 V4-only 实现，新概念用新名（`unified_kv`、`swa_pages`、
`csa_block_capacity`、`kv_idx_local`）。

---

## 1. 设计目标

decode 路径上每个 V4 attention layer 需要做 sparse attention：query 与一组从
SWA 环形缓冲 / 压缩 KV 池中选出的 K 做 softmax-weighted gather。这组 K 在不同
层（Dense / CSA / HCA）来源不同、数量不同，但都可以被抽象成"在一个 BF16 池上按
索引取行"的操作。

`sparse_attn_v4_paged_decode` 把这个抽象固化下来：

- 每层只暴露**一个** `unified_kv` BF16 池（page_size = 1，即一个 row 对应一个
  KV slot）
- 每个 token 的 K 集合用一段 int32 索引描述（packed cumsum 风格的
  `kv_indptr` + `kv_indices`）
- kernel runtime 决定每 token 实际读多长，不需要任何 per-forward-动态-shape 张
  量分配

这套设计直接消除了"用临时 `kv_flat_sa` materialize 选中的 K"这种需要 per-forward
分配可变 shape 张量的中间步骤 —— 这种 materialize 是 CUDAGraph capture 失败的
主因之一。

---

## 2. V4 attention 层结构

V4 是 hybrid attention，每层根据 `compress_ratios[layer_id]`（model config 中的
长度等于 `num_layers` 的整数列表）决定行为：

| compress_ratio | 名称  | 组件                                | V4-Pro 占比（共 62 层）|
|----------------|-------|-------------------------------------|------------------------|
| 0              | Dense | 仅 SWA                              | 1                      |
| 4              | CSA   | SWA + CSA Main Compressor + Indexer | 30                     |
| 128            | HCA   | SWA + HCA Main Compressor           | 31                     |

**所有层都有 SWA**（sliding-window attention，固定 `window_size = 128`，BF16，
单 KV head）。差异在于 compress 部分：

- **Dense**：没有 compressor，只 attend 最近 `window_size` 个 token
- **CSA**（compressed selective attention）：每 4 个原始 token overlap-compress
  成 1 个 K，再由 Indexer 按 score 选 top-`index_topk` 个
- **HCA**（hierarchical coarse attention）：每 128 个原始 token compress 成 1
  个 K（无 overlap），全部参与 attention（不挑选）

物理 paged-KV 块大小 `block_size = lcm(4, 128) = 128` 原始 token，所以每块容纳：

- CSA Main：`csa_block_capacity = block_size / 4 = 32` 个压缩 K
- HCA Main：`hca_block_capacity = block_size / 128 = 1` 个压缩 K
- CSA Indexer：32 个 FP8 score K（与 CSA Main 同步写入，但 dtype 独立）

**`attn_sink`** 是 V4 特有的 per-head 可学习 bias，加在 softmax 分母（不参与
numerator gather），等价于"虚拟全局 K"占一份概率质量。

V4-Pro 还启用 **MTP-1**（Multi-Token Prediction，
`num_nextn_predict_layers = 1`）—— 每个 forward 每个 request 同时算 1 个 base
token + 1 个 draft token（共 `1 + max_spec_steps = 2` 个 token）。这影响所有
"per-token" 张量的实际形状（详见 §7）。

---

## 3. KV Cache Layout

### 3.1 unified_kv：每层独立的 BF16 池

每层一个 contiguous 2D tensor，由 SWA 前缀 + 可选 compress 尾部拼接：

```
Dense layer  : unified_kv = [num_slots*window_size,                                  head_dim] BF16
CSA layer    : unified_kv = [num_slots*window_size + num_blocks*csa_block_capacity, head_dim] BF16
HCA layer    : unified_kv = [num_slots*window_size + num_blocks*hca_block_capacity, head_dim] BF16
                            └──────── SWA ────────┘ └─────────── compress ───────────┘
                                  swa_pages
```

把 SWA 和 compress 物理合并的好处：sparse attn kernel 只需要一个 base pointer，
所有索引可以表达成 `unified_kv` 上的 row offset，不需要在 kernel 里区分 KV 来源。

#### 关键量

- **`num_slots`** = `model_runner.max_per_req_cache_slots`
  - per-request state cache 的张量槽数，每个并发 request 占 `slots_per_req` 个槽
  - V4 `slots_per_req = 1`，所以 `num_slots = max_num_seqs`
  - 物理意义：每个槽存一个 request 的 SWA 环形缓冲 + Compressor 的 `kv_state` /
    `score_state`，request 进入推理时分配，结束归还
  - 来源：`atom/model_engine/model_runner.py:1157`

- **`num_blocks`** = `model_runner.num_physical_kvcache_blocks`
  - 物理 paged-KV block 总数；每块覆盖 V4 `block_size = 128` 原始 token
  - 由 GPU memory budget 和 `gpu_memory_utilization` 自动算出
  - 物理意义：所有并发 request **共享**的全局 KV 池容量；BlockManager 按需分配
    block 给 request 的 paged kv table
  - 来源：`atom/model_engine/model_runner.py:1239`

- **`window_size`** = 128（V4 SWA 窗口大小）
- **`csa_block_capacity`** = `block_size / 4 = 32`
- **`hca_block_capacity`** = `block_size / 128 = 1`
- **`swa_pages`** = `num_slots * window_size`（unified_kv 中 SWA 与 compress 分
  界点，也是后面所有 compress 偏移的基址）
- **`head_dim`** = 512（V4-Pro，全局共享 MQA 单 KV head）

`num_slots` 决定**并发**容量（同时活跃 request 数），`num_blocks` 决定**总上下文**
容量（所有 request 累计 KV 总量），两者按各自维度独立扩缩。

### 3.2 模块 view 绑定

`build_kv_cache_tensor`（在 `atom/model_ops/attentions/deepseek_v4_attn.py` 的
`DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor`）把每层 `unified_kv`
切成模块需要的 view：

```python
attn.unified_kv     = unified_kv                                              # 整池，给 paged decode kernel
attn.swa_kv         = unified_kv[:swa_pages].view(num_slots, window_size, head_dim)
compressor.kv_cache = unified_kv[swa_pages:].view(num_blocks, block_capacity, head_dim)
```

`swa_write` / `Compressor.scatter` 写路径**不需要任何改动** —— 它们写的是同一物
理存储，只是逻辑视图不同。

### 3.3 独立的 CSA Indexer FP8 池

`v4_csa_idx_kv: [num_csa_layers, num_blocks, csa_block_capacity, aligned_index_dim]`
**不**并入 `unified_kv`，原因：

- dtype 是 FP8 + 4-byte fp32 scale 交错（独立 layout，不能与 BF16 共池）
- 由 `indexer_k_quant_and_cache` 写、`cp_gather_indexer_k_quant_cache` 读，整条
  路径与 sparse attn kernel 解耦
- `aligned_index_dim = ((index_head_dim + 4 + 15) // 16) * 16`，16-byte align 是
  inductor unaligned-access 优化要求

CSA Indexer 池只用来算 `topk_local`（哪些压缩 K 应被选中），不参与最终 attention
gather。挑出来的索引转成 `unified_kv` 偏移后再交给 paged decode kernel。

---

## 4. 三种 KV 的索引方式

每个 token 的 K 集合可能来自 SWA、CSA compress、HCA compress 三类。本节定义
"逻辑位置 → unified_kv 物理偏移"的翻译规则。

### 4.1 SWA（每层都有）

**写**：环形缓冲，每个 sequence 占一个 state slot：

```
swa_kv[state_slot, ring_offset, :] = kv_value
ring_offset = position % window_size
```

**读**：position = `current_position` 的 token 应注意
`[max(0, current_position - window_size + 1), current_position]` 区间的最近
`window_size` 个 token。

`_build_window_topk_batched`（`atom/model_ops/attentions/deepseek_v4_attn.py:79`）
根据该 sequence 的 `start_pos`（该 sequence 进入当前 forward 时的起始 position）
产出 `[total_tokens, window_size]` int32。**三种模式输出语义不同**：

| 模式             | 触发条件                                | 输出语义                | 公式                                                                                     |
|------------------|-----------------------------------------|-------------------------|------------------------------------------------------------------------------------------|
| 新 prefill       | `start_pos == 0`                        | **绝对位置**（非 ring） | `[max(0, current_position - window_size + 1), current_position]`，越界用 `-1` 填         |
| prefix 模式      | `0 < start_pos < window_size - 1`       | `ring_offset`           | `[0, 1, ..., start_pos, -1, ..., -1]`（前 `start_pos+1` 个 ring 已写过）                 |
| 稳态 cyclic      | `start_pos >= window_size - 1`          | `ring_offset`           | `[(start_pos + 1 + j) % window_size for j in 0..window_size)`，最旧 → 最新               |

decode 总走 prefix 或 cyclic 模式。`current_position % window_size` 在 cyclic
模式恰好是当前 token 最新写入的 ring slot（数组末尾那个值）。

`-1` 表示 ring slot 未被写过（早期 decode 不足 `window_size` 个 token），sparse
attn kernel 会跳过。

**SWA → unified_kv 物理偏移**（**只对 decode 输出有效** —— prefill 模式输出的
是绝对位置不是 ring_offset，走另一条路径）：

```
window_paged_offset[token_idx, window_idx] =
    state_slot[token_idx] * window_size + window_topk_batched[token_idx, window_idx]
        if window_topk_batched[token_idx, window_idx] >= 0
    -1
        otherwise
```

prefill 路径直接拿 case A 的绝对位置去 gather 当前 forward 刚算出的 KV 切片
（layout `[total_prefill_tokens, head_dim]`），不读 SWA 环形缓冲。这也是
paged_decode kernel 只覆盖 decode 的原因。

### 4.2 CSA Main compress（CSA 层）

**写**：每 4 个原始 token 通过 `Compressor.forward` overlap-compress 成 1 个 K，
按 V4 paged-KV 块组织。每物理块 `csa_block_capacity = 32` 个压缩 K（128 原始
token），写入 `Compressor.kv_cache[physical_block_id, slot_in_block, :]`。

**读**：CSA Indexer 算 score，挑 top-`index_topk` 个 sequence-local 压缩位置
`compress_idx_logical ∈ [0, num_committed_csa)`，其中
`num_committed_csa = ctx_len // 4`。

**CSA → unified_kv 物理偏移**：

```
block_idx_in_seq = compress_idx_logical // csa_block_capacity   # 第几个 paged block
slot_in_block    = compress_idx_logical %  csa_block_capacity   # 块内偏移
physical_block   = block_table[sequence_id, block_idx_in_seq]
paged_offset     = swa_pages
                 + physical_block * csa_block_capacity
                 + slot_in_block
```

`-1` sentinel（indexer 输出无效位置时）保留为 `-1`。

### 4.3 HCA Main compress（HCA 层）

**写**：每 128 原始 token compress 成 1 个 K（无 overlap）。每物理块
`hca_block_capacity = 1` 个压缩 K，写入 `kv_cache[physical_block_id, 0, :]`。

**读**：HCA 不用 indexer，每个 token attend 全部 `num_committed_hca = ctx_len // 128`
个压缩 K（其中 `ctx_len = batch.context_lens[seq]`，sequence 总 token 数，
post-extend）。

**HCA → unified_kv 物理偏移**（`hca_block_capacity = 1` 时简化）：

```
physical_block = block_table[sequence_id, compress_idx_logical]   # compress_idx_logical ∈ [0, num_committed_hca)
paged_offset   = swa_pages + physical_block * 1 + 0
               = swa_pages + physical_block
```

### 4.4 翻译规则总览

| KV 类型           | unified_kv 物理偏移                                                               | 索引来源                              |
|-------------------|-----------------------------------------------------------------------------------|---------------------------------------|
| SWA               | `state_slot * window_size + ring_offset`                                          | `window_topk_batched`                 |
| CSA compress      | `swa_pages + physical_block * csa_block_capacity + slot_in_block`                 | indexer raw `topk_local`              |
| HCA compress      | `swa_pages + physical_block`                                                      | `[0, num_committed_hca)` 全取         |

---

## 5. Sparse Attention Kernel

### 5.1 接口

位置：`atom/model_ops/v4_kernels/paged_decode.py`

```python
def sparse_attn_v4_paged_decode(
    q:                 torch.Tensor,  # [total_tokens, num_heads, head_dim]   BF16
    unified_kv:        torch.Tensor,  # [total_pages, head_dim]               BF16  ← 单 base ptr
    kv_indices:  torch.Tensor,  # [total_indices_in_batch]              int32 ← 变长拼接
    kv_indptr:   torch.Tensor,  # [total_tokens + 1]                    int32 ← 真前缀和
    attn_sink:         torch.Tensor,  # [num_heads]
    softmax_scale:     float,
) -> torch.Tensor:                    # [total_tokens, num_heads, head_dim]
```

API 风格与 `aiter.mla.mla_decode_fwd` 对齐，page_size = 1（即 `unified_kv` 的一
行就是一个 KV slot，无需块内索引）。

### 5.2 语义约定

- **每 token 的有效 K 范围** =
  `kv_indices[kv_indptr[token_idx] : kv_indptr[token_idx + 1]]`
- **`-1` entry 自动跳过**：kernel 内 `valid = in_range & (slot >= 0)`，无效槽
  对 softmax 不贡献（score 设 -inf，概率为 0）
- **runtime trip count**：kernel 用 `tl.range(0, kv_len, BLOCK_K)` 决定每 token
  循环次数，短 sequence 不为长 sequence 付出 worst-case 工作量
- 没有 `K_MAX` constexpr，kernel binary 在不同 batch shape 间共用

### 5.3 数值实现

online-softmax 累加（fp32），最后 attn_sink 合并到分母：

```
max_logit_final = max(max_logit_acc, attn_sink)
sum_exp_final   = sum_exp_acc * exp(max_logit_acc - max_logit_final)
                + exp(attn_sink - max_logit_final)
output          = output_acc / sum_exp_final
```

kernel constants：`BLOCK_H = 16`（AMD MFMA 最小 tile），
`BLOCK_D = next_pow2(head_dim)`，`BLOCK_K = 16` 当 `head_dim ≥ 256` else 32。

---

## 6. Per-Forward 索引构造

每次 forward 都需要构造好 `kv_indices` 和 `kv_indptr` 才能调
sparse attn kernel。本节描述索引 buffer 布局、indptr 设计、构造分工，以及关键
的 packed write 步骤。

### 6.1 Buffer 布局

每种 ratio（SWA / CSA / HCA）一对独立 buffer，预分配在 `forward_vars`，
CUDAGraph-stable address：

```
v4_kv_indices_swa : [max_num_reqs * (1 + max_spec_steps) * window_size]                            int32
v4_kv_indptr_swa  : [max_num_reqs * (1 + max_spec_steps) + 1]                                      int32

v4_kv_indices_csa : [max_num_reqs * (1 + max_spec_steps) * (window_size + index_topk)]             int32
v4_kv_indptr_csa  : [max_num_reqs * (1 + max_spec_steps) + 1]                                      int32

v4_kv_indices_hca : [max_num_reqs * (1 + max_spec_steps) * (window_size + max_num_committed_hca)]  int32
v4_kv_indptr_hca  : [max_num_reqs * (1 + max_spec_steps) + 1]                                      int32

v4_n_committed_csa_per_seq : [max_num_reqs]                                                        int32
v4_batch_id_per_token      : [max_num_reqs * (1 + max_spec_steps)]                                 int32
```

`v4_batch_id_per_token` 是**唯一**的 per-token 映射 buffer，下游 kernel 用
`per_seq_data[batch_id_per_token[t]]` 查任何 sequence-level 数据
（state_slot / valid_count / block_tables 行），避免持久化 [T] 形状的 per-seq
数据别名。`v4_n_committed_csa_per_seq` 即是这种 per-seq buffer 的代表
（`min(n_committed_csa, index_topk)`，sequence-level，MTP base/draft token 共享）。

`max_num_committed_hca = max_model_len // 128`（V4-Pro 1M context = 8192）。
buffer 容量按 worst case 估算（packed-indptr 要求每 token 各占完整一份索引段，
MTP draft token 间不复用物理 slot，即使有重叠也物理重复存储），每 forward 实
际只用前缀。

CUDAGraph capture 时实际用到的 token 数 = `max_capture_size`（通常由
`--cudagraph-capture-sizes` 决定），远小于 buffer 上限；captured graph 读到的
是固定地址。

#### 为什么 SWA 单独一个 buffer 而不是复用 CSA/HCA 的 window 前缀

SWA / CSA / HCA 三个 buffer 的 indptr stride 不同（`window_size` vs
`window_size + index_topk` vs 变长真前缀和），同一份 window 内容必须分别落在 3
个不同物理地址，indptr 才能正确索引到本 buffer 内的连续 span。Dense 层用
`v4_kv_indices_swa`，CSA/HCA 层用各自的 buffer（其 window 前缀内容与 SWA
buffer 完全相同，只是物理位置不同）。

window 部分内容在所有 3 个 buffer 里**完全相同**（`state_slot * window_size +
ring_offset`，layer-invariant）。builder 一次性把同一份 window 数据分别写到 3
个 buffer 的头部。

### 6.2 构造分工

| 数据                           | 位置                            | per-layer? | 依赖                                           |
|--------------------------------|---------------------------------|------------|------------------------------------------------|
| window paged indices           | metadata builder                | no         | `state_slot`、`window_topk_batched`            |
| HCA compress paged indices     | metadata builder                | no         | `block_tables`、`num_committed_hca`            |
| SWA / CSA / HCA kv_indptr| metadata builder                | no         | 各 ratio 的 per-token kv_len                   |
| **CSA compress paged indices** | **V4Attention.forward CSA 分支**| **yes**    | indexer raw `topk_local`                      |

CSA 是唯一 per-layer 工作的部分，因为 indexer 输出每层不同。其他几项 layer-
invariant，builder 一次构造完毕，所有层共用。

### 6.3 indptr 三种形态

#### CSA：变长 packed cumsum

per-token 实际 kv_len = `window_size + min(n_committed_csa, index_topk)`。早期
解码（`ctx_len < 4 * index_topk = 4096`）时 `n_committed_csa < index_topk`，
indptr 自动让该 token 占用更小的 buffer 区间。

```
kv_indptr_csa[token_idx + 1] =
    kv_indptr_csa[token_idx]
    + window_size + min(n_committed_csa[batch_id_per_token[token_idx]], index_topk)
```

`n_committed_csa = ctx_len // 4`，是该 sequence 已 commit 的 CSA compressed K 数。

**例**（V4-Pro，3 个 req 各 1 token，`ctx_len` = 200 / 400 / 5000）：

```
n_committed_csa_per_seq      = [50, 100, 1250]
valid_count_per_seq          = min(n_committed_csa, 1024) = [50, 100, 1024]   # [bs] int32
batch_id_per_token           = [0, 1, 2]                                       # [T] int32
per-token kv_len             = 128 + valid_count_per_seq[batch_id_per_token]
                             = [178, 228, 1152]

indptr  = [0, 178, 406, 1558]                            # 4 entries (T+1 = 4)
indices = [w0_0..w0_127,  c0_0..c0_49,                   # req0: 128 win + 50 compress
           w1_0..w1_127,  c1_0..c1_99,                   # req1: 128 win + 100 compress
           w2_0..w2_127,  c2_0..c2_1023]                 # req2: 128 win + 1024 compress (capped)
                                                          # 共 178+228+1152 = 1558 entries
```

注意 `valid_count_per_seq` 是 sequence-level 张量（[bs] int32，存在
`v4_n_committed_csa_per_seq` buffer 里）；MTP-1 下同 req 的 base/draft token
共享同一个值（因为 ctx_len 是 seq 级别的）。kernel 通过
`valid_count_per_seq[batch_id_per_token[t]]` 查表，避免持久化 [T] 形 per-token
数组。

#### HCA：变长 packed cumsum

per-token 实际 kv_len = `window_size + num_committed_hca`，全部 committed
compressed K 都参与（无 sentinel 浪费）。

```
kv_indptr_hca[token_idx + 1] =
    kv_indptr_hca[token_idx]
    + (window_size + num_committed_hca[batch_id_per_token[token_idx]])
```

decode 1 token/sequence 时 `batch_id_per_token = arange(batch_size)`。

**例**（V4-Pro，2 个 req 各 1 token，`ctx_len` = 400 / 5000）：

```
num_committed_hca = [400//128, 5000//128] = [3, 39]
per-token kv_len  = 128 + num_committed_hca = [131, 167]

indptr  = [0, 131, 298]                                  # 3 entries (T+1 = 3)
indices = [w0_0..w0_127,  h0_0, h0_1, h0_2,              # req0: 128 win + 3 compress
           w1_0..w1_127,  h1_0..h1_38]                   # req1: 128 win + 39 compress
                                                          # 共 131+167 = 298 entries
```

#### SWA：uniform stride + sentinel

per-token buffer 容量恒等于 `window_size`，indptr 退化为等差数列：

```
kv_indptr_swa[token_idx] = token_idx * window_size
```

早期解码（`current_position < window_size`）时只有 `current_position + 1` 个
ring slot 已写入，剩下用 `-1` sentinel（由 `_build_window_topk_batched` 根据
`start_pos` 模式生成，详见 §4.1）。kernel 自动跳过 `-1`，实际 attended kv_len =
`min(current_position + 1, window_size)`。

**例**（V4-Pro，3 个 req 各 1 token，`current_position` = 50 / 5000 / 100000）：

```
indptr  = [0, 128, 256, 384]                             # 4 entries (T+1 = 4)
indices = [w0_0..w0_50,  -1 × 77,                        # req0 早期: 51 valid + 77 sentinel
           w1_0..w1_127,                                  # req1 稳态 cyclic: 128 全 valid
           w2_0..w2_127]                                  # req2 稳态 cyclic: 128 全 valid
                                                          # 共 3 * 128 = 384 entries
```

SWA 用 uniform stride 而非 packed 是有意为之：`window_size = 128` 上限本身就小，
packed 的 cumsum 工作量（在 builder 算 indptr 前缀和、在 forward packed write）
与节省的几个 sentinel slot 不划算。设计原则对称（如有需要也可改 packed），但
工程上保持 uniform 简化实现。

### 6.4 CSA per-layer 翻译 + packed write

CSA 是唯一需要 per-layer forward 工作的索引：每层 indexer 输出的 `topk_local`
都不同，必须在该层的 forward 内做翻译并写入 `v4_kv_indices_csa`。

#### 翻译公式

`block_tables` 是已有的 [bs, max_num_blocks_per_seq] int32 buffer（attn_metadata
携带），翻译时直接用 `batch_id_per_token` 做 fancy index，不需要持久化的
`block_tables_per_token` 别名（早期方案曾有，按"per-token 张量除非必要否则用
batch_id_per_token 查表"原则删除）：

```python
# topk_local: [total_tokens, index_topk] int32
#   indexer raw output, sequence-local in [0, n_committed_csa)
#   前 valid_count_per_seq[bid] 个有效，剩余 -1
# block_tables: [bs, max_blocks_per_seq] int32  (attn_metadata.block_tables)
# batch_id_per_token: [T] int32  (v4_batch_id_per_token)

block_idx_in_seq = topk_local // self.csa_block_capacity   # [T, index_topk] int32
slot_in_block    = topk_local %  self.csa_block_capacity
# fancy-index: physical_block[t, k] = block_tables[bid[t], block_idx[t, k]]
batch_id_expanded = batch_id_per_token.long().unsqueeze(1).expand(-1, index_topk)
physical_block    = block_tables[batch_id_expanded, block_idx_in_seq.long()]
paged_compress    = swa_pages + physical_block * self.csa_block_capacity + slot_in_block
# 保留 topk_local 的 -1 sentinel（packed write 也会按 valid_count 截断，下面）
paged_compress    = torch.where(topk_local >= 0, paged_compress, -1)
```

`paged_compress[t, k]` 当 `k >= valid_count_per_seq[bid[t]]` 时实际 topk_local
那位也是 -1，翻译路径会把它写回 -1；下一步 packed write 又只读
`[0, valid_count_per_seq[bid[t]])` 范围，垃圾被忽略。

#### packed write 的必要性

因为 CSA indptr 是 packed cumsum，token t 的 buffer 占用是
`[indptr[t], indptr[t+1])`，长度 per-token 不同。简单
`kv_indices[:, win:win+index_topk] = paged_compress` 切片会越界覆盖下个 token
的起始偏移，必须用 packed write。

`torch.scatter_` + bool mask 也不可用 —— `tensor[bool_mask]` 是动态 shape
selection，CUDAGraph capture 会失败。

CG-friendly 的 packed write 是**固定 grid 的 triton kernel**：每 `(token, k)`
thread 通过 `batch_id_per_token[t]` 查 `valid_count_per_seq`，再决定是否写。
`valid_count` 走 per-seq 而不是 per-token 是为了贯彻"per-token 张量只留
batch_id_per_token 一个"的设计原则（同 req 的 MTP base/draft token 共享
ctx_len，valid_count 自然 sequence-level）：

```python
@triton.jit
def csa_packed_write_kernel(paged_compress_ptr, kv_indices_ptr,
                             kv_indptr_ptr,
                             valid_count_per_seq_ptr,   # [bs] int32
                             batch_id_per_token_ptr,    # [T]  int32
                             window_size: tl.constexpr,
                             index_topk: tl.constexpr,
                             BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_kb = tl.program_id(1)
    bid = tl.load(batch_id_per_token_ptr + pid_t)
    valid_k = tl.load(valid_count_per_seq_ptr + bid)

    k_offs = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    in_range = (k_offs < valid_k) & (k_offs < index_topk)
    src = tl.load(paged_compress_ptr + pid_t * index_topk + k_offs,
                  mask=in_range, other=0)
    write_base = tl.load(kv_indptr_ptr + pid_t) + window_size
    tl.store(kv_indices_ptr + write_base + k_offs, src, mask=in_range)
    # 越界 thread no-op（buffer 该位置可能含上次 forward 残留，但 indptr[t+1]
    # 已界定 attention kernel 读不到这里）
```

实际 grid 是 `(T, ceil(index_topk / BLOCK_K))`（BLOCK_K = 64 vectorize 一行 K
load/store），不是文档草案的 `(T, index_topk)` 单线程版。

builder 在 metadata 构造阶段算好 `valid_count_per_seq = min(n_committed_csa,
index_topk)` 并 stage 到 `v4_n_committed_csa_per_seq`，所有 CSA layer 共用。

整条路径无 H2D 拷贝、无动态 shape、固定 kernel grid，CUDAGraph-friendly。

### 6.5 indexer 输出语义

`indexer_score_topk` 返回 `[total_tokens, index_topk] int32` 的 raw `topk_local`，
即 sequence-local 的压缩 K 位置（在 `[0, n_committed_csa)` 范围内），无效位置用
`-1` 填充。所有"翻译到 paged offset"的逻辑都在 caller（V4Attention.forward）做，
不污染 indexer 模块抽象。

### 6.6 CG padding 协议（actual_bs < graph_bs）

CUDAGraph 同一 graph 可在多个 actual batch size 下 replay：scheduler 把
`actual_bs` padding 到下一档 `graph_bs`（model_runner 在
`cu_seqlens_q[scheduled_bs+1:bs+1] = cu_seqlens_q[scheduled_bs]` 处填 0-token
seqs）。**captured kernel 的 grid 在 capture 时已 baked 为 `graph_bs * max_q_len`，
replay 时仍按这个 grid 跑** —— builder 必须把所有 per-fwd metadata sentinel-pad
到 `padded_total_tokens = graph_bs * max_q_len`，否则 padded slot 读到 stale
buffer 数据 → kv_indptr garbage → kv_indices OOB → `Memory access fault by GPU`。

**统一 sentinel 协议**：

| 张量类型 | sentinel | kernel 行为 |
|---|---|---|
| per-token 索引 (`v4_batch_id_per_token`, `swa_write_indices`) | `-1` | `if bid < 0: return` / `if src_id < 0: return` |
| indptr cumsum (`v4_kv_indptr_{swa,csa,hca}`) | 最后 cumsum 值重复 | `kv_len = indptr[t+1] - indptr[t] = 0` → `tl.range(0, 0, BLOCK_K)` 不进入循环 |
| kv_indices 数组 | 不需 pad | indptr 已强制 `kv_len=0`，kernel 不会读尾部 |
| `unified_kv` slot 值 | 由 kv_indices sentinel 控制 | kernel 内 `valid = in_range & (slot >= 0)` mask |

**caller 责任**（`_attach_v4_per_fwd_meta` / `_attach_v4_paged_decode_meta`）：
- 接收 `padded_bs` + `max_q_len` kwargs；eager / prefill 路径默认 `None` 退化为
  `padded_total_tokens = total_tokens`（无 padding）
- per-token 张量 `[0:total_tokens]` 真值，`[total_tokens:padded_total_tokens]` = -1
- indptr `[0:T+1]` 真 cumsum，`[T+1:padded_total_tokens+1]` 重复 `cumsum[T]`

**Python-侧 fancy indexing 防 OOB**（`_fill_csa_paged_compress`）：
即使 padded slot 不写出去（被 csa_packed_write `bid<0` 跳过），中间步骤的
`block_tables[batch_id_expanded, block_idx_in_seq]` 仍会执行；padded slot 的
batch_id (-1) + 不可控的 block_idx 可能让 GPU fancy index 越界。必须在 indexing
前 `clamp(min=0)` batch_id、`clamp(0, mnbps-1)` block_idx，让 lookup 永远落在
合法地址（值是 garbage 但下游 sentinel skip）。

### 6.7 H2D dedup：单一来源 per-fwd metadata

每个 V4 forward 的 host→device staging 数量直接乘以 layer 数（V4-Pro 62 层），
所以 builder 的去重原则是：**同一份 per-seq / per-token 数据只 H2D 一次**，下游
所有 kernel 通过 `attn_metadata` 上的统一字段读取。

需要满足两个约束：

1. **dtype 选择**：PyTorch fancy index 的 INDEX 张量必须 int64（gather SOURCE
   任意 dtype）。我们自写的 triton kernel（swa_write / csa_packed_write）读
   int32 / int64 都行（`tl.load` 从 pointer 类型推断）。**所以共享 INDEX 类
   张量统一用 int64**：一份 buffer 同时满足 PyTorch fancy-index 和 triton
   kernel 消费方，无需 int32+int64 镜像（节省一半显存 + 0 GPU dtype-cast）。
   **不要用 `.long()` 派生**：host 端 `.long()` 分配 fresh int64 tensor，
   data_ptr 跨 fwd 不稳定，captured graph replay 读旧地址 → GPU memory
   access fault。源头 stage 成 int64 才是 CG-safe 唯一正确方案。
2. **生命期**：共享数据的 producer 必须在 consumer 之前运行。
   `_attach_v4_per_fwd_meta` 产出 `batch_id_per_token` +
   `n_committed_csa_per_seq`，被 `_attach_sparse_layout_metadata` 内的
   `_build_v4_indexer_meta` 消费 —— 所以前者必须先调用（已在
   `prepare_decode` / `prepare_prefill` / `build_for_cudagraph_capture` 三处
   统一顺序）。

| 共享字段 | dtype | 生产者 | 消费者 |
|---|---|---|---|
| `state_slot_mapping` | int32 [bs] | `_populate_state_slot_mapping` | `swa_write` + Compressor + paged-decode |
| `batch_id_per_token` | **int64** [mnbt] | `_attach_v4_per_fwd_meta` | `swa_write` + `csa_packed_write` + `_fill_csa_paged_compress` + indexer |
| `n_committed_csa_per_seq` | int32 [bs] | `_attach_v4_per_fwd_meta` | `csa_packed_write`（kernel mask `< index_topk`）+ indexer |

---

## 7. MTP（Multi-Token Prediction）

V4-Pro config 中 `num_nextn_predict_layers = 1`，启用 MTP 后每个 forward 每个
request 同时计算 `(1 + max_spec_steps)` 个 token：

- **base token**：当前真实位置 `start_pos`，输出确认下一 token 的 logits
- **draft tokens**：投机推理出的 `max_spec_steps` 个 token，位置
  `[start_pos + 1, start_pos + max_spec_steps]`，由 EagleProposer 接受/拒绝

V4-Pro 默认 `max_spec_steps = 1`（MTP-1）—— 每 forward 每 request 2 个 token。

### 7.1 Per-forward token 布局

```
positions  = [s0, s0+1, s1, s1+1, ..., s_{R-1}, s_{R-1}+1]   # R = num_reqs, 2R 个 token
cu_seqlens_q = [0, 2, 4, ..., 2R]                             # 每 req 2 token 连续
token_num_per_seq = [2, 2, ..., 2]                            # 长度 R
total_tokens = R * (1 + max_spec_steps) = 2R                  # MTP-1
batch_id_per_token = [0, 0, 1, 1, ..., R-1, R-1]              # 每 req 重复 (1+spec_steps) 次
```

`prepare_decode`（`atom/model_ops/attentions/deepseek_v4_attn.py:1180`）已经按
`max_seqlen_q = batch.num_spec_step + 1` 和 `np.tile + np.repeat` 自动展开了这个
布局，paged 索引构造代码不需要特殊处理 MTP，只要按 `total_tokens` 维度算就行。

### 7.2 各 ratio 在 MTP 下的索引行为

#### SWA：相邻 draft token 的 window 高度重叠

base 在位置 p、draft 在位置 p+1，两者 attended ring slot 仅尾部偏移 1：

```
base  attended ring offsets: (p+1)%win, (p+2)%win, ..., p%win              # win 个
draft attended ring offsets: (p+2)%win, (p+3)%win, ..., (p+1)%win           # win 个
                                                                            # 重叠 win-1 个
```

但 packed-indptr 要求每 token indices 段首尾相接、不能物理重叠，所以 buffer 仍
然分别为 base 和 draft 各占 `window_size` 个 slot，逻辑上重复但物理上独立写
（详见 §6.1）。

#### CSA：每个 draft token 各自跑 indexer

base 和 draft 的 Q 不同（不同位置），各自调用 indexer 算 topk_local，结果几乎
不重叠。`csa_packed_write` kernel 按 `(total_tokens, index_topk)` 的 grid 跑，
每个 token 独立翻译 + 写。

`n_committed_csa[batch_id_per_token[token_idx]]` 对同 req 的 base / draft 给的是
同一个值（因为 ctx_len 是 sequence 级别的，不区分 base/draft），所以两者的
`valid_count` 也相同。

#### HCA：base 和 draft 的 num_committed_hca 几乎总是相同

`num_committed_hca = ctx_len // 128`，仅在 ctx_len 跨过 128 倍数边界时才会
+1。MTP-1 的 base 和 draft 位置只差 1，所以 99%+ 情况下两者
`num_committed_hca` 相同。

### 7.3 MTP-1 indptr 例子

V4-Pro MTP-1，2 个 req，`ctx_len` 分别 5000 / 5001（注意 5001 // 128 = 39 ==
5000 // 128，所以两个 req 的 `num_committed_hca` 都是 39）：

**HCA**：

```
total_tokens = 2 * 2 = 4
num_committed_hca per req = [39, 39]
per-token kv_len = 128 + 39 = 167（4 个 token 全相同）

indptr  = [0, 167, 334, 501, 668]                            # T+1 = 5
indices = [w0_*,  h0_0..h0_38,                               # req0 base:  128 win + 39 compress
           w1_*,  h1_0..h1_38,                               # req0 draft: 128 win + 39 compress
           w2_*,  h2_0..h2_38,                               # req1 base:  128 win + 39 compress
           w3_*,  h3_0..h3_38]                               # req1 draft: 128 win + 39 compress
                                                              # 共 4 * 167 = 668 entries
```

req0 的 base (`w0/h0`) 和 draft (`w1/h1`) 的 compress 部分指向**完全相同**的
unified_kv slot（因为 `num_committed_hca` 一样、block_table 一样）；window 部
分指向同一 ring，只是 ring_offset 的循环排列错开 1 位。这些重复在 packed-indptr
方案下都是物理重复存储。

**CSA**（同样 2 req × 2 token，`ctx_len = 5000/5001`，`n_committed_csa = ctx_len //
4 = 1250`，全部 ≥ `index_topk = 1024` 所以 capped）：

```
valid_count = [1024, 1024, 1024, 1024]
per-token kv_len = 128 + 1024 = 1152

indptr  = [0, 1152, 2304, 3456, 4608]                        # T+1 = 5
indices = [w0_*, c0_0..c0_1023,                              # req0 base
           w1_*, c1_0..c1_1023,                              # req0 draft (与 base 不同 topk!)
           w2_*, c2_0..c2_1023,                              # req1 base
           w3_*, c3_0..c3_1023]                              # req1 draft
                                                              # 共 4 * 1152 = 4608 entries
```

注意 CSA 不像 HCA —— req0 base 和 draft 的 compress slot 由各自 indexer 选出，
互不相同。

**SWA**（Dense 层，uniform stride，2 req × 2 token，`current_position = 5000 /
5001 / 7000 / 7001`）：

```
indptr  = [0, 128, 256, 384, 512]                            # T+1 = 5
indices = [w0_0..w0_127,                                      # req0 base  (位置 5000)
           w1_0..w1_127,                                      # req0 draft (位置 5001, ring 错 1 位)
           w2_0..w2_127,                                      # req1 base  (位置 7000)
           w3_0..w3_127]                                      # req1 draft (位置 7001)
                                                              # 共 4 * 128 = 512 entries
```

### 7.4 dispatch 检测：is_pure_decode

§7 的 `is_decode_only` 检测条件（`(token_num_per_seq == 1).all()`）只对单 token
decode 成立，MTP 模式下 `token_num_per_seq = (1 + max_spec_steps) > 1`。

为支持 MTP，dispatch 检测改为更广义的"全 sequence 都在 decode 模式"：

```
is_pure_decode =
    (cu_seqlens_q[1:] - cu_seqlens_q[:-1] == 1 + max_spec_steps).all()
    and (start_pos_per_seq > 0).all()
```

或等价地：`scheduled_batch_size * (1 + max_spec_steps) == total_tokens` 且无
fresh prefill (`start_pos == 0`) 的 sequence 混在 batch 里。

---

## 8. Decode Dispatch

`V4Attention.forward` 末尾按 `is_pure_decode` 分流：

```python
if compress_ratio == 4 and is_pure_decode:
    # CSA 层：用 indexer raw topk_local 做翻译并 packed write
    # （indexer 在前面已经跑过；topk_local 从 v4_indexer_meta 取，详见 §6.5）
    self._fill_csa_paged_compress(attn_md, total_tokens)

if is_pure_decode:
    # builder 已提前构建：window paged indices + HCA compress paged indices
    # + 三个 kv_indptr (SWA / CSA / HCA)（CSA compress 段由上面 Phase C 写入）
    if compress_ratio == 0:
        kv_indices, kv_indptr = attn_md.kv_indices_swa, attn_md.kv_indptr_swa
    elif compress_ratio == 4:
        kv_indices, kv_indptr = attn_md.kv_indices_csa, attn_md.kv_indptr_csa
    else:  # compress_ratio == 128
        kv_indices, kv_indptr = attn_md.kv_indices_hca, attn_md.kv_indptr_hca

    output = sparse_attn_v4_paged_decode(
        q, self.unified_kv, kv_indices, kv_indptr,
        self.attn_sink, self.softmax_scale,
    )
else:
    # prefill 或 mixed batch：保留 ragged_varlen kernel
    kv_sa, topk_flat = _v4_build_sparse_inputs_batched(...)
    output = sparse_attn_ragged_varlen(q, kv_sa, ..., topk_flat, ...)
```

`is_pure_decode` 由 metadata builder 在 `_attach_v4_per_fwd_meta` 检测（详见
§7.4 公式），结果挂在 `attn_metadata`，所有 layers 直接读。

---

## 9. CUDAGraph 友好性

CUDAGraph capture 对 op 的要求很严格 —— 任何 per-forward 动态分配的张量、设备
间同步、或可变 shape 都会让 capture/replay 失败。本设计的关键约束：

| 风险点                                            | 应对                                                       |
|---------------------------------------------------|------------------------------------------------------------|
| `torch.empty(varying_size)` 在 capture 内          | 所有 indices/indptr buffer 全预分配在 `forward_vars`       |
| `torch.tensor(scalar)` H2D 触发 hipError          | indptr 用 cumsum + `fill_` 改写，无标量 H2D                |
| kernel binary 依赖 `K_MAX` constexpr               | 无 `K_MAX`，runtime `tl.range(0, kv_len, BLOCK_K)`         |
| `kv_flat_sa` shape per-forward 变化                | 整个变量删除，kernel 直接 paged 读 `unified_kv`            |
| `topk_flat` 长度 per-forward 变化                  | 替换为 indptr+indices，indices buffer 容量预分配           |
| per-layer 不同 indexer topk → 多 buffer 需求       | layers 顺序执行，per-ratio 复用同一 buffer 即可             |
| CSA packed write 用 `scatter_` + bool mask         | 改用固定 grid 的 triton kernel（§6.4）                      |
| **`actual_bs < graph_bs` 时 captured kernel 越界** | **builder 必须 sentinel-pad 到 `graph_bs * max_q_len`：per-token 张量填 -1（kernel 内 `bid<0` skip）；indptr 重复最后 cumsum 值（`kv_len=0` 跳过 inner loop）。详见 §6.6** |
| `kv[indices].contiguous()` fancy index 在 capture region 内 fresh alloc | swa_write 重写：kernel 直接吃 forward_vars 全量 `kv` + `write_indices` + `positions` + `batch_id_per_token` + `state_slot_per_seq`，零 captured-region alloc |

---

## 10. 不在本设计范围

- **Prefill 路径**：保留 `sparse_attn_ragged_varlen`。prefill 不需要 CUDAGraph，
  且 N 与 KV 都大，packed materialization 反而高效。
- **CSA Indexer FP8 cache layout**：仍走 `cp_gather_indexer_k_quant_cache`，与
  sparse attn kernel 解耦，不并入 `unified_kv`（dtype 不同）。

---

## 11. 文件路径速查

| 内容                             | 路径                                                                       |
|----------------------------------|----------------------------------------------------------------------------|
| 本设计文档（中文）               | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_DECODE_DESIGN.zh.md`         |
| 本设计文档（English）            | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_DECODE_DESIGN.en.md`         |
| paged_decode kernel + 包装       | `atom/model_ops/v4_kernels/paged_decode.py`                                |
| csa_packed_write kernel + 包装   | `atom/model_ops/v4_kernels/csa_packed_write.py`                            |
| KV cache 分配 + 模块 view 绑定   | `atom/model_ops/attentions/deepseek_v4_attn.py` (`allocate_per_req_cache`, `build_kv_cache_tensor`) |
| paged 索引构造 + indptr cumsum   | `atom/model_ops/attentions/deepseek_v4_attn.py` (`_attach_v4_per_fwd_meta`) |
| V4 forward dispatch              | `atom/models/deepseek_v4.py` (`DeepseekV4Attention.forward`)               |
| indexer 输出 (`topk_local`)      | `atom/models/deepseek_v4.py` (`Indexer.indexer_score_topk`)                |
