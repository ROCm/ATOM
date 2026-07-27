# DeepSeek-V4 Paged Prefill：ATOM 的双源 sparse attention 设计

本文介绍 DeepSeek-V4 在 ATOM 中的 sparse prefill attention 实现 ——
`sparse_attn_v4_paged_prefill` 这个**双 KV 源** triton kernel 的设计动机、KV 来
源、索引构造以及与 decode 路径（`sparse_attn_v4_paged_decode`，单 KV 源）的对
偶关系。目标读者是希望理解或扩展 V4 attention 后端 prefill 路径的工程师，建议
先读 `ATOM_V4_PAGED_DECODE_DESIGN.zh.md` 建立 decode 路径心智模型再读本文。

> 阅读建议：§0-§2 建立 prefill 路径的设计前提（与 decode 的差异、双源动机）；
> §3-§4 是 KV 源拆分（prefix vs extend）和索引翻译规则；§5 是 kernel 接口与数
> 值实现；§6 是 builder 责任与 chunked prefill 下的索引构造细节；§7 起是
> dispatch 与边界条件。

---

## 0. 与 decode 路径的根本差异

decode 路径（`sparse_attn_v4_paged_decode`）只有**一个** KV 源 `unified_kv`：每
个 decode token 的 K 集合在前面的 fwd 已经全部物理写入了 `unified_kv`（SWA 段
经 `swa_write` 写入循环 ring，CSA / HCA 段经 Compressor 写入 paged 区），所以
单个 base pointer + 一组 (kv_indices, kv_indptr) 就能描述。

prefill 路径**必须有两个 KV 源**，原因是**本 fwd 的输入 token K 还没有进入 SWA
ring**：

| KV 来源 | 物理位置 | 包含内容 |
|---|---|---|
| **prefix**（前序历史） | `unified_kv`（SWA ring + compress paged） | 上一轮 chunk 写入的 SWA + 全历史压缩 K |
| **extend**（本 fwd 输入） | `kv`（本 fwd 计算出的 K，layout `[total_tokens, head_dim]` BF16） | 本 chunk 内所有 prefill token 的原始 K |

为什么不能像 decode 一样把 extend 也"先写 SWA ring，再 attend"？答案是**长
prefill 会被 ring 的 `window_size` 上限截断**。例如本 chunk
`token_num = 8192`、`window_size = 128`，`swa_write` 用循环写入只能保留最近 128
个 token；早期 prefill token 的 SWA window 在 attention 跑之前就已被本 chunk 的
后续 token 覆盖。如果坚持复用 decode 路径，要么 SWA buffer 显存膨胀 64 倍
（`window_size` → `max_num_batched_tokens`），要么逐 token 做 swa_write +
attention（牺牲 batch 并行）。

ATOM 的解法：让 attention kernel 直接吃本 fwd 的 `kv` 张量（per-fwd flat，无任
何循环覆盖），与 `unified_kv` 平行作为第二个 KV 源。代价是 kernel 多一段 loop，
收益是显存零增、长 prefill 天然支持。

### 与 upstream V4 reference 的差异

DeepSeek 官方 reference 把 prefill / decode 都用"materialize 一个临时 dense
KV tensor 喂给 dense attention kernel"实现 —— 把 SWA ring 内容、compress K
内容、本 fwd K 内容**全部 copy 到一个临时连续 buffer** 再做 attention。这条路在
ATOM 里就是老的 `sparse_attn_ragged_varlen` + `_v4_build_sparse_inputs_batched`
路径，劣势：

1. **per-fwd 动态分配**临时 KV buffer，shape 依赖 `total_committed` /
   `total_tokens`，不能进 CUDAGraph capture（虽然 prefill 不强求 CG，但仍是冗
   余分配 + 拷贝开销）。
2. **indexer 输出语义被污染**：indexer 不能简单返回 `topk_in_seq`，必须自己做
   `subtract seq_base + offset + masked_fill_(-1)` 去配合 ragged buffer 的语义
   （详见 §6.2），代码冗余且与 decode 的 indexer 输出语义不统一。
3. **per-layer 拷贝代价**：每个 CSA layer 都要重新 materialize 一份临时 buffer
   （indexer 输出每层不同），N 层 × per-fwd alloc 是显著开销。

ATOM 新设计去掉所有 materialize，**两个 KV 源直接被 kernel 索引**：
- prefix 源 `unified_kv` 是持久化 buffer（builder 之前算好 `kv_indices_prefix`
  已经覆盖 SWA history + CSA topk + HCA all-committed）
- extend 源 `kv` 是本 fwd 的张量（builder 算好 `kv_indices_extend` 描述每 token
  在本 chunk 内 attend 哪些 row）

---

## 1. 设计目标

prefill 路径上每个 V4 attention layer 需要做 sparse attention：query 与一组
"上轮 SWA 末尾 + 本 chunk SWA 末尾 + 全历史 compress topk" 的 K 做
softmax-weighted gather。这组 K 横跨持久 paged buffer（`unified_kv`）和 per-fwd
input buffer（`kv`），**必须用两段独立的 (indices, indptr) 描述**。

`sparse_attn_v4_paged_prefill` 的关键设计点：

- **双 base pointer**（`unified_kv` + `kv`），两段 (kv_indices, kv_indptr) 各自
  描述自己源里的 K range
- **online softmax 跨两段累加**：先 loop prefix region，再 loop extend region，
  共享 (m, l, o)，最后融合 `attn_sink`
- **复用 `csa_translate_pack`**：CSA 的 topk → paged offset 翻译跟 decode 完全
  同款 kernel（差别只在传不同的 dst buffer 和 offset 参数）
- **prefill 不要求 CG capture**（chunked prefill 路径走 eager），但 builder 仍
  按"per-fwd staged + 固定地址"原则构造，方便未来切换 CG 时低成本支持

这套设计把 indexer 模块的输出语义彻底简化成**统一的 `topk_in_seq`**（详见
§6.2），下游所有翻译逻辑都在 caller 端完成 —— 与 decode 路径的 indexer 输出语
义对齐，模块抽象终于一致。

---

## 2. V4 attention 层结构（沿用 decode 文档定义）

V4-Pro 共 62 层，按 `compress_ratios[layer_id]` 分三类（详见 decode 文档 §2，
本文不再展开）：

| compress_ratio | 名称  | 组件                                | V4-Pro 占比 |
|----------------|-------|-------------------------------------|-------------|
| 0              | Dense | 仅 SWA                              | 1           |
| 4              | CSA   | SWA + CSA Main Compressor + Indexer | 30          |
| 128            | HCA   | SWA + HCA Main Compressor           | 31          |

prefill 路径对所有 3 种层都生效，但**索引构造每种 ratio 不同**（详见 §6）。

prefill 默认**不启用 MTP**（MTP 只在 decode 阶段做投机推理）；
`token_num_per_seq` 在 prefill batch 里就是该 sequence 的本 chunk 真实 token 数，
没有 base/draft 分裂。所以 prefill 的 `total_tokens = sum(token_num_per_seq)`，
没有 `(1 + max_spec_steps)` 倍乘。

---

## 3. KV Cache Layout：双源拆分

prefill kernel 看到的 KV 来自两个独立张量：

### 3.1 Prefix 源：`unified_kv`（持久化 paged）

跟 decode 路径**完全同一份** `unified_kv`（见 decode 文档 §3.1）：

```
CSA layer    : unified_kv = [num_slots*window_size + num_blocks*csa_block_capacity, head_dim] BF16
                            └──────── SWA ────────┘ └─────────── compress ───────────┘
                                  swa_pages
```

prefix 索引 `kv_indices_prefix[t][k]` 是 `unified_kv` 上的 row offset：
- `< swa_pages`：SWA 段，指向某 sequence 的 ring slot（前序 chunk 写入的 K）
- `>= swa_pages`：compress 段，指向 CSA 或 HCA 的某个压缩 K page

### 3.2 Extend 源：`kv`（per-fwd flat input）

本 fwd 在 V4Attention.forward 里刚算出的 K，layout `[total_tokens, head_dim]`
BF16，对应本 chunk 内所有 prefill token 的原始 K（未压缩、未写入 ring）。

extend 索引 `kv_indices_extend[t][k]` 是 `kv` 上的 row offset：
- `kv_indices_extend[t][k] = cu_seqlens_q[batch_id[t]] + (本 chunk 内 K 的 token_pos_in_chunk)`

每 token 的 extend window 仅装本 chunk 内的 SWA 末尾段，长度
`extend_count[t] = min(token_pos_in_chunk + 1, win)`，是 `kv` 张量内的连续行。

`kv` 在本 fwd 的生命周期：从 `kv_a_proj_with_mqa` 输出开始 → swa_write 写入
ring（这是为了下一 fwd 的 decode 能读到，本 fwd 自己不读 ring）→ 本 prefill
attention 读 → fwd 结束后释放。

### 3.3 为什么不把 extend 也合并进 unified_kv

可以做，但**显存代价巨大**。`unified_kv` 的 SWA 段大小是 `num_slots *
window_size`，要装下本 fwd 所有 prefill token 需要扩到 `num_slots *
max_num_batched_tokens`，V4-Pro 默认 mnbt=8192、win=128 → SWA 段膨胀 64 倍 ≈
560MB / 层 × 62 层 = 35GB。完全不可接受。

保持 `unified_kv` SWA 段的 `window_size` 上限定义，让 extend 走第二个 base，是
**唯一显存可控的方案**。

### 3.4 独立的 CSA Indexer FP8 池（沿用 decode 文档 §3.3）

`v4_csa_idx_kv` 不变，仍走 `cp_gather_indexer_k_quant_cache` → indexer →
`topk_in_seq`。indexer 输出语义统一为 `topk_in_seq`（详见 §6.2）。

---

## 4. 三种 KV 的索引方式

prefill 每 token 的 K 集合可分为三段：
- **本 chunk SWA 段**（在 `kv` 里，由 extend 索引描述）
- **前序 chunk SWA 段**（在 `unified_kv` SWA 区，由 prefix 索引的 SWA 子段描述）
- **compress 段**（在 `unified_kv` compress 区，由 prefix 索引的 compress 子段描述）

### 4.1 SWA：跨 chunk 边界的拆分

设 token `t` 全局 sequence position 为 `p_global`，所属 chunk 起点为
`chunk_start = p_global - token_pos_in_chunk`，其中 `token_pos_in_chunk` 是 token
在本 chunk 内的偏移（从 `cu_seqlens_q` 推出）。

token `t` 的 **SWA window 范围** = `[max(0, p_global - win + 1), p_global]`，长度
`min(p_global + 1, win)`。这个范围被 `chunk_start` 分成两段：

| 段 | 范围 | 物理位置 | 索引方式 |
|---|---|---|---|
| **prefix SWA** | `[max(0, p_global - win + 1), chunk_start - 1]` | `unified_kv` SWA 区（state_slot ring） | prefix 索引子段 |
| **extend SWA** | `[chunk_start, p_global]` | `kv`（本 fwd input，row idx = `cu_seqlens_q[batch] + (pos - chunk_start)`） | extend 索引 |

**长度公式**：

```
prefix_swa_count[t] = max(0, chunk_start - max(0, p_global - win + 1))
extend_count[t]     = min(token_pos_in_chunk + 1, win)
prefix_swa_count[t] + extend_count[t] = min(p_global + 1, win)   # 总 SWA window 长度
```

**特殊情况（pure prefill，无前序 chunk，chunk_start = 0）**：
- `prefix_swa_count[t] = 0`（SWA 全在本 chunk 内）
- `extend_count[t] = min(p_global + 1, win)`
- prefix 索引的 SWA 子段长度为 0，prefix 只剩 compress 段

**prefix SWA → unified_kv 物理偏移**：

```
prefix_swa_paged[t][k] = state_slot[batch[t]] * window_size
                       + ((p_global - prefix_swa_count[t] - extend_count[t] + 1 + k) % window_size)
                                                        # k ∈ [0, prefix_swa_count[t])
```

**extend SWA → kv 物理偏移**：

```
extend_paged[t][k] = cu_seqlens_q[batch[t]] + (token_pos_in_chunk - extend_count[t] + 1 + k)
                                                        # k ∈ [0, extend_count[t])
```

### 4.2 CSA Main compress（CSA 层）

prefill 的 CSA 翻译规则**与 decode 完全一致**（参见 decode 文档 §4.2）：

```
block_idx_in_seq = topk_in_seq // csa_block_capacity
slot_in_block    = topk_in_seq %  csa_block_capacity
physical_block   = block_table[batch[t], block_idx_in_seq]
prefix_csa_paged = swa_pages + physical_block * csa_block_capacity + slot_in_block
```

`-1` sentinel 保留。

**关键复用**：`csa_translate_pack` triton kernel（`atom/model_ops/v4_kernels/csa_translate_pack.py`）
做这件事，prefill 调用方式与 decode 完全相同 —— 都传 `swa_pages` 和
`window_size` 参数（指向 `unified_kv` 的 compress 段，并跳过 prefix 索引内的
SWA 子段写入 CSA 子段）。

`csa_translate_pack` 的 dst 参数是 `(kv_indptr_csa, kv_indices_csa,
window_size)`，写入位置 = `kv_indptr_csa[t] + window_size + col`。在 prefill 路
径下，dst 改为 `(kv_indptr_prefix_csa, kv_indices_prefix_csa,
prefix_swa_count[t])`，写入位置 = `kv_indptr_prefix_csa[t] + prefix_swa_count[t]
+ col` —— **`window_size` 参数被 `prefix_swa_count[t]` 替代**。这要求
`csa_translate_pack` 接受 per-token 的 "skip prefix len" 数组而不是 scalar
`window_size`，需要做小修改（详见 §6.4）。

### 4.3 HCA Main compress（HCA 层）

prefill 的 HCA 翻译规则与 decode 完全一致（参见 decode 文档 §4.3）：

```
physical_block = block_table[batch[t], compress_idx_logical]
                                                        # compress_idx_logical ∈ [0, num_committed_hca)
prefix_hca_paged = swa_pages + physical_block             # hca_block_capacity = 1
```

HCA 不用 indexer，每 token attend 全部 `num_committed_hca = ctx_len // 128` 个压
缩 K，layer-invariant，builder 一次构造。

### 4.4 翻译规则总览

| 段 | 源 | per-token 物理偏移公式 | 索引来源 |
|---|---|---|---|
| extend SWA | `kv` | `cu_seqlens_q[batch] + (pos_in_chunk - extend_count + 1 + k)` | builder CPU 算 |
| prefix SWA | `unified_kv` | `state_slot * window_size + ((p_global - prefix_swa_count - extend_count + 1 + k) % window_size)` | builder CPU 算 |
| prefix CSA | `unified_kv` | `swa_pages + physical_block * csa_block_capacity + slot` | indexer raw `topk_in_seq` + `csa_translate_pack` |
| prefix HCA | `unified_kv` | `swa_pages + physical_block` | builder 算 |

---

## 5. Sparse Attention Kernel

### 5.1 接口

位置：`atom/model_ops/v4_kernels/paged_prefill.py`（新增）

```python
def sparse_attn_v4_paged_prefill(
    q:                  torch.Tensor,  # [total_tokens, num_heads, head_dim] BF16
    
    # Source 1: prefix (unified_kv, paged)
    unified_kv:         torch.Tensor,  # [total_pages, head_dim] BF16  ← decode 同款
    kv_indices_prefix:  torch.Tensor,  # [total_prefix_indices] int32
    kv_indptr_prefix:   torch.Tensor,  # [total_tokens + 1] int32, packed cumsum
    
    # Source 2: extend (本 fwd 的 input K, flat)
    kv:                 torch.Tensor,  # [total_tokens, head_dim] BF16
    kv_indices_extend:  torch.Tensor,  # [total_extend_indices] int32
    kv_indptr_extend:   torch.Tensor,  # [total_tokens + 1] int32, packed cumsum
    
    attn_sink:          torch.Tensor,  # [num_heads]
    softmax_scale:      float,
) -> torch.Tensor:                     # [total_tokens, num_heads, head_dim]
```

API 风格延续 decode kernel `sparse_attn_v4_paged_decode`，扩展点是多一组
`(kv, kv_indices_extend, kv_indptr_extend)`。

### 5.2 语义约定

每 token 的有效 K 范围 = prefix 段 ∪ extend 段：

```
prefix K = unified_kv[kv_indices_prefix[kv_indptr_prefix[t]:kv_indptr_prefix[t+1]]]
extend K =        kv[kv_indices_extend[kv_indptr_extend[t]:kv_indptr_extend[t+1]]]
```

- **段间无去重**：builder 保证 prefix 与 extend 描述的 K 范围**物理上不重叠**
  （SWA window 拆分时的 chunk_start 边界严格分割，参见 §4.1），kernel 不做去重
- **`-1` entry 自动跳过**：两段都用 `valid = in_range & (slot >= 0)` mask，与
  decode 同款
- **runtime trip count**：两段各自走 `tl.range(0, kv_len, BLOCK_K)`，per-token 长
  度独立可变

### 5.3 数值实现：跨两段 online softmax

online softmax 在两段 K 上联合累加（fp32 精度），最后融合 `attn_sink`：

```python
@triton.jit
def sparse_attn_v4_paged_prefill_kernel(...):
    # 初始化累加器
    m_acc = tl.full([BLOCK_H], -inf, dtype=tl.float32)
    l_acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    o_acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    
    q_block = tl.load(q + token_idx * stride_q + ...)
    
    # ===== Region 1: prefix from unified_kv =====
    prefix_start = tl.load(kv_indptr_prefix_ptr + token_idx)
    prefix_end   = tl.load(kv_indptr_prefix_ptr + token_idx + 1)
    for k_off in tl.range(prefix_start, prefix_end, BLOCK_K):
        idx = tl.load(kv_indices_prefix_ptr + k_off + tl.arange(0, BLOCK_K),
                      mask=in_range, other=-1)
        valid = in_range & (idx >= 0)
        k_block = tl.load(unified_kv + idx[:, None] * head_dim + d_off[None, :],
                          mask=valid[:, None])
        # online_softmax_step(q_block, k_block, m_acc, l_acc, o_acc)
        ...
    
    # ===== Region 2: extend from kv =====
    extend_start = tl.load(kv_indptr_extend_ptr + token_idx)
    extend_end   = tl.load(kv_indptr_extend_ptr + token_idx + 1)
    for k_off in tl.range(extend_start, extend_end, BLOCK_K):
        idx = tl.load(kv_indices_extend_ptr + k_off + tl.arange(0, BLOCK_K),
                      mask=in_range, other=-1)
        valid = in_range & (idx >= 0)
        k_block = tl.load(kv + idx[:, None] * head_dim + d_off[None, :],
                          mask=valid[:, None])
        # online_softmax_step(q_block, k_block, m_acc, l_acc, o_acc)
        ...
    
    # ===== 融合 attn_sink，输出 =====
    sink = tl.load(attn_sink_ptr + head_offsets)
    m_final = tl.maximum(m_acc, sink)
    l_final = l_acc * tl.exp(m_acc - m_final) + tl.exp(sink - m_final)
    output = o_acc * tl.exp(m_acc - m_final) / l_final
    tl.store(o_ptr + ..., output)
```

`online_softmax_step` 函数与 decode kernel 内同款（Flash-Attention v2 风格的
m/l/o update），在两段间共享累加器即可。Constants：`BLOCK_H = 16`、
`BLOCK_D = next_pow2(head_dim)`、`BLOCK_K = 16` 当 `head_dim ≥ 256` else 32 ——
完全沿用 decode kernel 的 tile 选择。

### 5.4 与 decode kernel 的共享

90%+ 代码可共享。建议工程实现：
- 抽 `_paged_attn_inner` JIT helper（接收 `(q_block, kv_base, indices_ptr,
  indptr_ptr, m_acc, l_acc, o_acc)`，做单段 online softmax 累加）
- decode kernel 调用一次（prefix only）
- prefill kernel 调用两次（prefix + extend）
- attn_sink 融合在 outer kernel 做（不在 inner helper 里），方便共享

否则也可直接 fork `paged_decode.py` 写 `paged_prefill.py`，物理拷贝 + 加 region
2 loop，工程上更简单（少抽象）。优先方案二（fork），后续重构再抽 helper。

---

## 6. Per-Forward 索引构造

每次 prefill forward 都需要构造好 `kv_indices_prefix`、`kv_indptr_prefix`、
`kv_indices_extend`、`kv_indptr_extend` 才能调 sparse attn kernel。本节描述
buffer 布局、构造分工、indptr 设计、以及 chunked prefill 下的索引拼接。

### 6.1 Buffer 布局

每种 ratio（SWA / CSA / HCA）一对独立 prefix buffer，extend buffer 全 ratio 共
享（layer-invariant）：

```
# Prefix buffers（per-ratio，跟 decode 同款 layout，包含 SWA + compress 两段）
v4_kv_indices_prefix_swa : [max_num_batched_tokens * window_size]                              int32
v4_kv_indptr_prefix_swa  : [max_num_batched_tokens + 1]                                        int32

v4_kv_indices_prefix_csa : [max_num_batched_tokens * (window_size + index_topk)]               int32
v4_kv_indptr_prefix_csa  : [max_num_batched_tokens + 1]                                        int32

v4_kv_indices_prefix_hca : [max_num_batched_tokens * (window_size + max_num_committed_hca)]    int32
v4_kv_indptr_prefix_hca  : [max_num_batched_tokens + 1]                                        int32

# Extend buffer（共享，只装本 chunk SWA）
v4_kv_indices_extend     : [max_num_batched_tokens * window_size]                              int32
v4_kv_indptr_extend      : [max_num_batched_tokens + 1]                                        int32
```

prefix buffer 容量公式跟 decode 同款（per-token 上限 = `window_size` + 该 ratio
的 compress 上限）。注意 prefix SWA 段在长 chunk 下可能 < `window_size`（因为
本 chunk 还没占满 ring），但 buffer 上限按 worst case 取。

extend buffer per-token 上限 = `window_size`（任何 token 在本 chunk 内最多
attend `window_size` 个），与 chunk 长度无关。

prefill 不要求 CG capture，所以这些 buffer 可以走 per-fwd `torch.empty(...)` 而
不是 `forward_vars` 预分配；但为统一索引语义和未来 CG 兼容，**仍建议预分配在
`forward_vars`**，eager 路径只用前缀。

#### 为什么 prefix 也要 SWA / CSA / HCA 三个 buffer

跟 decode 一样：三种 ratio 的 indptr stride 不同（`window_size`、`window_size +
index_topk`、`window_size + max_num_committed_hca`），SWA 段内容虽然相同但物理
位置必须各自独立。Dense 层用 `kv_indices_prefix_swa`，CSA 用
`kv_indices_prefix_csa`，HCA 用 `kv_indices_prefix_hca`。

### 6.2 Indexer 输出语义统一

**关键设计**：indexer (`Indexer.indexer_score_topk`) 在 prefill 路径返回的
`topk_indices` 含义改为 **`topk_in_seq`** —— sequence-local 的压缩 K 位置（在
`[0, n_committed_csa)` 范围内），无效位置用 `-1` 填充。与 decode 路径**完全一
致**。

旧的 `_post_process_topk` 方法 —— 在 indexer 内做 `topk_global - seq_base +
offset_per_token + masked_fill_(invalid, -1)` 把输出强行翻译成 ragged
`kv_sa-layout` 索引 —— **删除**。所有翻译逻辑下沉到 caller（V4Attention.forward）
通过 `csa_translate_pack` 完成，indexer 模块只负责"算 score 选 topk"。

具体改动：
- `_score_topk_decode`：原本就返回 raw `topk_in_seq`（kernel 直出，无 post），不改
- `_score_topk_prefill`：删 `_post_process_topk` 调用；kernel 输出是 GLOBAL（`+
  rowStart` 后），需在 indexer 内做一步轻量 in-graph subtract 转 seq-local：
  ```python
  topk_local = torch.where(topk_global == -1, -1,
                            topk_global - seq_base_per_token.unsqueeze(1))
  ```
  其中 `seq_base_per_token` 仍是现有 `_build_v4_indexer_meta` 算的 per-token
  seq base（`= cu_committed[batch_id]`）。`-1` sentinel 用 `where` 保护。
  
  **注**：prefill kernel 的 `cu_ends[t]` 已经把 `min((p+1)//ratio,
  n_committed_per_seq[batch[t]])` 作为 per-token causal 上界塞进去了
  （`_build_v4_indexer_meta`），所以 valid 区间内的 topk_local 自动落在
  `[0, visible_end[t]) ⊆ [0, n_committed_per_seq[batch[t]])`，不需要额外的
  future-mask（详见 indexer_meta 实现）。

### 6.3 构造分工

| 数据 | 位置 | per-layer? | 依赖 |
|---|---|---|---|
| `kv_indices_extend` | metadata builder | no | `cu_seqlens_q`、`positions`（全 ratio 共享） |
| `kv_indptr_extend` | metadata builder | no | `extend_count`（全 ratio 共享） |
| `kv_indices_prefix_swa` (Dense) | metadata builder | no | `state_slot`、`positions`、chunk_start |
| `kv_indices_prefix_csa` SWA 子段 | metadata builder | no | 同上，layer-invariant |
| `kv_indices_prefix_csa` compress 子段 | **V4Attention.forward CSA 分支** | **yes** | indexer raw `topk_in_seq` |
| `kv_indices_prefix_hca` SWA 子段 | metadata builder | no | 同上 |
| `kv_indices_prefix_hca` compress 子段 | metadata builder | no | `block_table`、`num_committed_hca` |
| `kv_indptr_prefix_*`（三个） | metadata builder | no | 各 ratio 的 per-token kv_len |

CSA 仍是唯一 per-layer 工作（indexer 输出每层不同），其他全部 layer-invariant，
builder 一次构造完毕。

### 6.4 `csa_translate_pack` 的 prefill 适配

decode 路径的 `csa_translate_pack` 接受 scalar `window_size` 作为 dst skip 长度
（每 token 都用同一个 `window_size`）。prefill 路径下 dst skip 长度是 per-token
变量 `prefix_swa_count[t]`，所以需要小修改：

```python
# 旧 signature (decode)
csa_translate_pack(topk_local, block_tables, n_committed_csa_per_seq,
                   kv_indptr_csa, batch_id_per_token, kv_indices_csa,
                   swa_pages, window_size, csa_block_capacity)

# 新 signature (兼容 decode + prefill)
csa_translate_pack(topk_local, block_tables, n_committed_csa_per_seq,
                   kv_indptr, batch_id_per_token, kv_indices,
                   swa_pages,
                   skip_prefix_len_per_token,    # NEW: [T] int32
                                                  # decode: 全填 window_size
                                                  # prefill: prefix_swa_count
                   csa_block_capacity)
```

kernel 内部：

```python
# OLD
write_base = tl.load(kv_indptr_ptr + pid_t) + window_size  # window_size 是 constexpr scalar

# NEW
skip_len = tl.load(skip_prefix_len_per_token_ptr + pid_t)
write_base = tl.load(kv_indptr_ptr + pid_t) + skip_len
```

decode caller 构造 `skip_prefix_len_per_token = torch.full([T], window_size,
dtype=int32)` 即可保持兼容（也可让 builder 静态构造一次，永久复用）。prefill
caller 用真实的 `prefix_swa_count` per-token 数组。

`swa_pages` 仍是 scalar（两路径都用同一个 `unified_kv` base，offset 一致）。

### 6.5 indptr 三种形态

prefill 的 indptr 全部用**变长 packed cumsum**（不像 decode SWA 用 uniform
stride），原因是 prefill 一个 chunk 内 token 数远大于 decode（mnbt=8192），
sentinel padding 到 `T * window_size` 显存浪费太大。

#### Extend：变长 packed cumsum

per-token kv_len = `extend_count[t] = min(token_pos_in_chunk + 1, win)`：

```
kv_indptr_extend[t+1] = kv_indptr_extend[t] + extend_count[t]
```

**例**（pure prefill，单 seq，`token_num = 4`，`win = 2`，`chunk_start = 0`）：

```
positions          = [0, 1, 2, 3]
token_pos_in_chunk = [0, 1, 2, 3]
extend_count       = [min(1, 2), min(2, 2), min(3, 2), min(4, 2)] = [1, 2, 2, 2]
kv_indptr_extend   = [0, 1, 3, 5, 7]
kv_indices_extend  = [
    cu_seqlens_q[0] + 0,                              # t=0: pos 0 attends pos 0
    cu_seqlens_q[0] + 0, cu_seqlens_q[0] + 1,         # t=1: attends pos 0, 1
    cu_seqlens_q[0] + 1, cu_seqlens_q[0] + 2,         # t=2: attends pos 1, 2
    cu_seqlens_q[0] + 2, cu_seqlens_q[0] + 3,         # t=3: attends pos 2, 3
]                                                     # 总 7 entries
```

所有 `cu_seqlens_q[0] = 0`（单 seq batch），所以 `kv_indices_extend = [0, 0, 1,
1, 2, 2, 3]`。

#### Prefix SWA + CSA：变长 packed cumsum

per-token kv_len = `prefix_swa_count[t] + min(n_committed_csa[batch[t]],
index_topk)`：

```
kv_indptr_prefix_csa[t+1] = kv_indptr_prefix_csa[t]
                          + prefix_swa_count[t]
                          + min(n_committed_csa_per_seq[batch[t]], index_topk)
```

**例**（chunked prefill，单 seq，`token_num = 4` per chunk，`win = 2`，
`chunk_start = 100`，`n_committed_csa_per_seq[0] = 25` < `index_topk = 4`）：

```
positions           = [100, 101, 102, 103]
prefix_swa_count    = [min(2, 100-(99))=1, min(2, 100-(100))=0, ...wait]
```

让我重算 `prefix_swa_count[t]`。SWA window 为 `[max(0, p-1), p]`：
- t=0 (p=100): SWA = [99, 100]，prefix 部分 = [99]，extend 部分 = [100]，
  `prefix_swa_count = 1, extend_count = 1`
- t=1 (p=101): SWA = [100, 101]，prefix 部分 = [100]（在前序 chunk），
  extend 部分 = [101]，`prefix_swa_count = 1, extend_count = 1`
  - 注：position 100 在前序 chunk 内（chunk_start=100 是本 chunk 起点，所以 99
    属于前序），上一个 chunk 应该已经把 position 100 的 K 写入 swa_kv ring 了
  - 实际上 t=1 (p=101) 的 SWA = [100, 101]，pos 100 在本 chunk 内 (因为
    chunk_start=100)，所以 prefix=0, extend=2
  
让我重新审 chunk 边界定义。`chunk_start = positions[0]` 是本 chunk 第一个 token
的 position，所以 position `chunk_start` 自身在**本 chunk 内**（属于 extend）。
prefix 段是 `< chunk_start` 的部分，extend 段是 `>= chunk_start` 的部分。

修正：
- t=0 (p=100, chunk_start=100): SWA = [99, 100]
  - prefix = [99]（pos 99 < 100），count=1
  - extend = [100]（pos 100 = chunk_start），count=1
- t=1 (p=101): SWA = [100, 101]
  - prefix = [] （都 ≥ 100），count=0
  - extend = [100, 101]，count=2
- t=2 (p=102): SWA = [101, 102]
  - prefix = []，extend = [101, 102]，count=2
- t=3 (p=103): SWA = [102, 103]
  - prefix = []，extend = [102, 103]，count=2

```
prefix_swa_count       = [1, 0, 0, 0]
extend_count           = [1, 2, 2, 2]
prefix_csa_count       = min(25, 4) = 4 per token (假设 index_topk=4)
prefix_csa_count_arr   = [4, 4, 4, 4]

kv_indptr_extend       = [0, 1, 3, 5, 7]
kv_indices_extend      = [0,  0,1,  1,2,  2,3]    # 都是 cu_seqlens_q[0] + offset

kv_indptr_prefix_csa   = [0, 1+4, 5+0+4, 9+0+4, 13+0+4]
                       = [0, 5, 9, 13, 17]
kv_indices_prefix_csa  = [
    swa_paged_99,       c0_0, c0_1, c0_2, c0_3,        # t=0: 1 swa + 4 csa
                        c1_0, c1_1, c1_2, c1_3,        # t=1: 0 swa + 4 csa
                        c2_0, c2_1, c2_2, c2_3,        # t=2: 0 swa + 4 csa
                        c3_0, c3_1, c3_2, c3_3,        # t=3: 0 swa + 4 csa
]                                                       # 共 17 entries
```

其中 `swa_paged_99 = state_slot[0] * window_size + (99 % window_size)`，
`cN_k` 是 indexer 选出的 topk_in_seq[N][k] 经 `csa_translate_pack` 翻译后的
unified_kv 物理偏移。

#### Prefix HCA：变长 packed cumsum

per-token kv_len = `prefix_swa_count[t] + num_committed_hca[batch[t]]`，HCA 不挑
选，全部 committed compressed K 都参与：

```
kv_indptr_prefix_hca[t+1] = kv_indptr_prefix_hca[t]
                          + prefix_swa_count[t]
                          + num_committed_hca[batch[t]]
```

#### Prefix Dense（仅 SWA）

Dense 层只有 SWA 段：

```
kv_indptr_prefix_swa[t+1] = kv_indptr_prefix_swa[t] + prefix_swa_count[t]
```

注意 Dense 的 prefix buffer **不带 compress 段**，所以 stride 与 CSA / HCA 不同
（per-token 上限 = `window_size`），符合"三个 ratio 各自独立 buffer" 的设计。

### 6.6 Builder 实现要点

builder 阶段（`prepare_prefill` → `_attach_v4_per_fwd_meta` →
`_attach_sparse_layout_metadata` 链）需要新增以下 CPU 计算：

```python
# 已有: positions_np, cu_seqlens_q_np, start_pos_per_seq_np, batch_id_per_token_np

# 1. token_pos_in_chunk: 本 chunk 内 token 的偏移（从 cu_seqlens_q 推）
seq_id_per_token = batch_id_per_token_np
chunk_start_per_seq = start_pos_per_seq_np   # 本 chunk 第一个 token 的全局 position
token_pos_in_chunk = positions_np - chunk_start_per_seq[seq_id_per_token]

# 2. extend_count: per-token 本 chunk SWA 长度
extend_count_np = np.minimum(token_pos_in_chunk + 1, window_size).astype(np.int32)

# 3. prefix_swa_count: per-token 前序 chunk SWA 长度
swa_window_low_global = np.maximum(0, positions_np - window_size + 1)
prefix_swa_count_np = np.maximum(0, chunk_start_per_seq[seq_id_per_token] - swa_window_low_global).astype(np.int32)

# 4. extend kv_indptr + kv_indices
extend_kv_indptr = np.concatenate([[0], np.cumsum(extend_count_np)]).astype(np.int32)
# kv_indices 用 _segment_indices 展开:
tok_idx_e, k_idx_e = _segment_indices(np.arange(total_tokens), extend_count_np)
extend_kv_indices = (
    cu_seqlens_q_np[seq_id_per_token[tok_idx_e]]
    + (token_pos_in_chunk[tok_idx_e] - extend_count_np[tok_idx_e] + 1 + k_idx_e)
).astype(np.int32)

# 5. prefix SWA section (per-ratio buffer 都需要写入)
tok_idx_s, k_idx_s = _segment_indices(np.arange(total_tokens), prefix_swa_count_np)
swa_window_low_per_token = np.maximum(0, positions_np - window_size + 1)
prefix_swa_global_pos = swa_window_low_per_token[tok_idx_s] + k_idx_s
prefix_swa_paged = (
    state_slot_np[seq_id_per_token[tok_idx_s]] * window_size
    + (prefix_swa_global_pos % window_size)
).astype(np.int32)
# 写入三个 prefix buffer 的 SWA 段（位置由各自的 indptr 决定）

# 6. prefix indptr (CSA / HCA / Dense 各自)
# CSA:
csa_count_per_seq = np.minimum(n_committed_csa_per_seq_np, index_topk)
csa_count_per_token = csa_count_per_seq[seq_id_per_token]
prefix_csa_kv_indptr = np.concatenate([[0],
    np.cumsum(prefix_swa_count_np + csa_count_per_token)]).astype(np.int32)
# HCA:
hca_count_per_seq = num_committed_hca_per_seq_np
hca_count_per_token = hca_count_per_seq[seq_id_per_token]
prefix_hca_kv_indptr = np.concatenate([[0],
    np.cumsum(prefix_swa_count_np + hca_count_per_token)]).astype(np.int32)
# Dense:
prefix_swa_kv_indptr = np.concatenate([[0],
    np.cumsum(prefix_swa_count_np)]).astype(np.int32)

# 7. Stage 到 forward_vars + copy_to_gpu
attn_metadata.kv_indices_extend     = stage("v4_kv_indices_extend",     extend_kv_indices)
attn_metadata.kv_indptr_extend      = stage("v4_kv_indptr_extend",      extend_kv_indptr)
attn_metadata.kv_indices_prefix_swa = stage_swa_filled("v4_kv_indices_prefix_swa", prefix_swa_paged, prefix_swa_kv_indptr)
attn_metadata.kv_indptr_prefix_swa  = stage("v4_kv_indptr_prefix_swa",  prefix_swa_kv_indptr)
attn_metadata.kv_indices_prefix_csa = stage_swa_filled("v4_kv_indices_prefix_csa", prefix_swa_paged, prefix_csa_kv_indptr)
attn_metadata.kv_indptr_prefix_csa  = stage("v4_kv_indptr_prefix_csa",  prefix_csa_kv_indptr)
attn_metadata.kv_indices_prefix_hca = stage_with_hca("v4_kv_indices_prefix_hca", prefix_swa_paged, hca_paged, prefix_hca_kv_indptr)
attn_metadata.kv_indptr_prefix_hca  = stage("v4_kv_indptr_prefix_hca",  prefix_hca_kv_indptr)
```

**关键不变量**：对每 token，
`prefix_swa_count[t] + extend_count[t] = min(p_global + 1, window_size)` —— SWA
总长度。builder 在装配时可以加一个 assert 验证。

#### Pure prefill (chunk_start_per_seq[s] == 0) 自动退化

所有 `prefix_swa_count[t]` 为 0，prefix buffer 的 SWA 子段长度退化到 0；CSA /
HCA 子段照常装。无需 special-case 代码。

### 6.7 H2D dedup（沿用 decode 文档 §6.7 规范）

prefill 路径下新增的共享字段：

| 共享字段 | dtype | 生产者 | 消费者 |
|---|---|---|---|
| `kv_indices_extend` | int32 [mnbt * win] | `_attach_v4_per_fwd_meta` | `sparse_attn_v4_paged_prefill` |
| `kv_indptr_extend` | int32 [mnbt + 1] | 同上 | 同上 |
| `kv_indices_prefix_*` (3 个) | int32 | `_attach_v4_per_fwd_meta` + `_fill_csa_paged_compress` (per-layer CSA) | `sparse_attn_v4_paged_prefill` |
| `kv_indptr_prefix_*` (3 个) | int32 [mnbt + 1] | `_attach_v4_per_fwd_meta` | 同上 |
| `prefix_swa_count_per_token` | int32 [mnbt] | `_attach_v4_per_fwd_meta` | `csa_translate_pack` (作为 skip_prefix_len) |

`batch_id_per_token`、`n_committed_csa_per_seq`、`state_slot_mapping` 等已有字段
直接复用。

---

## 7. MTP 与 prefill

MTP（Multi-Token Prediction）只在 decode 阶段做投机推理
（draft tokens 在 base token 之后），prefill 阶段**不启用 MTP**：

- prefill batch 的 `token_num_per_seq` 是该 sequence 本 chunk 真实 token 数，没
  有 base/draft 分裂
- `total_tokens = sum(token_num_per_seq)`，没有 `(1 + max_spec_steps)` 倍乘
- `cu_seqlens_q` 是 prefill 真实的 cumsum

所以本文 §6 所有公式直接按 `total_tokens` 维度算就行，不需要 §7 of decode doc
那种 MTP-1 物理重复存储考虑。

---

## 8. Prefill Dispatch

`V4Attention.forward` 末尾按 `is_pure_decode` 分流到两条 attention 路径：

```python
# Common: 计算 K、跑 indexer、写 SWA ring（给下一 fwd 用）
kv = ...  # 本 fwd 计算出的 K, [total_tokens, head_dim] BF16
self.indexer.compressor(...)
topk_in_seq = self.indexer.forward_batched(...)   # raw [T, index_topk] int32
swa_write(kv, ..., self.swa_kv, win)              # 写 ring（下一 fwd 用）

if attn_md.is_pure_decode:
    # decode 路径不变（详见 decode doc §8）
    if compress_ratio == 4:
        self._fill_csa_paged_compress(attn_md, total_tokens)
    if compress_ratio == 0:
        kv_indices, kv_indptr = attn_md.kv_indices_swa, attn_md.kv_indptr_swa
    elif compress_ratio == 4:
        kv_indices, kv_indptr = attn_md.kv_indices_csa, attn_md.kv_indptr_csa
    else:
        kv_indices, kv_indptr = attn_md.kv_indices_hca, attn_md.kv_indptr_hca
    output = sparse_attn_v4_paged_decode(
        q, self.unified_kv, kv_indices, kv_indptr,
        self.attn_sink, self.softmax_scale,
    )
else:
    # prefill 路径：双源 paged kernel
    if compress_ratio == 4:
        # CSA per-layer：用 indexer raw topk_in_seq 翻译并写入 prefix CSA 子段
        csa_translate_pack(
            topk_in_seq=topk_in_seq,
            block_tables=attn_md.block_tables,
            n_committed_csa_per_seq=attn_md.n_committed_csa_per_seq,
            kv_indptr=attn_md.kv_indptr_prefix_csa,
            batch_id_per_token=attn_md.batch_id_per_token,
            kv_indices=attn_md.kv_indices_prefix_csa,
            swa_pages=attn_md.swa_pages,
            skip_prefix_len_per_token=attn_md.prefix_swa_count_per_token,
            csa_block_capacity=self.compressor.kv_cache.size(1),
        )
    if compress_ratio == 0:
        kv_indices_prefix, kv_indptr_prefix = attn_md.kv_indices_prefix_swa, attn_md.kv_indptr_prefix_swa
    elif compress_ratio == 4:
        kv_indices_prefix, kv_indptr_prefix = attn_md.kv_indices_prefix_csa, attn_md.kv_indptr_prefix_csa
    else:
        kv_indices_prefix, kv_indptr_prefix = attn_md.kv_indices_prefix_hca, attn_md.kv_indptr_prefix_hca
    
    output = sparse_attn_v4_paged_prefill(
        q, 
        self.unified_kv, kv_indices_prefix, kv_indptr_prefix,
        kv, attn_md.kv_indices_extend, attn_md.kv_indptr_extend,
        self.attn_sink, self.softmax_scale,
    )
```

`is_pure_decode` 的检测沿用 decode doc §7.4：所有 sequence 的 `token_num_per_seq
== 1 + max_spec_steps` 且 `start_pos > 0`。否则走 prefill 分支。

#### Mixed batch（prefill seq + decode seq 混合）

如果 scheduler 发出 mixed batch（部分 sequence start_pos=0 走 prefill，部分
start_pos>0 走 decode），新设计有两个选项：

**选项 A：mixed 走 prefill 路径（推荐）**

prefill 路径的 extend 段对 decode token 退化成 `extend_count[t] = 0`（decode
token 的 K 已经在 swa ring 里），prefix 段照常构造。只要 builder 正确算
`prefix_swa_count + extend_count = min(p+1, win)`：
- decode token (chunk_start = positions[seq_start_in_batch] = current_position):
  prefix_swa_count = min(p, win), extend_count = 1（此 token 自己的 K 在本 fwd
  `kv` 里）
- 等价于：让 decode token 也走 prefix-from-history + extend-current 的拆分

需要确认：本 fwd 的 `kv` 在写入 swa_ring 之前是不是已经包含了 decode token 的
新 K（应该是的，否则 decode 也算不了）。

**选项 B：mixed 退化到老 ragged 路径**

保留 `sparse_attn_ragged_varlen` 调用，作为 fallback。代码复杂度 ↑。

设计文档默认选 A，实现时验证 GSM8K 等精度后决定是否接受。

---

## 9. CUDAGraph 友好性

prefill 路径**不强制**进 CUDAGraph capture（chunked prefill 的 chunk shape 多
变，CG capture 收益小）。但本设计的 buffer 仍按 forward_vars 预分配 + 固定地址
组织，方便未来切换 CG。

主要的 CG 兼容性要点（沿用 decode doc §9）：

| 风险点 | 应对 |
|---|---|
| `torch.empty(varying_size)` 在 capture 内 | 所有 indices/indptr buffer 预分配在 forward_vars |
| `torch.tensor(scalar)` H2D | 全部 indptr 走 cumsum + fill_，无标量 H2D |
| kernel binary 依赖 K_MAX constexpr | 无，runtime `tl.range(0, kv_len, BLOCK_K)` |
| `kv` 张量是 capture 区内的 fresh alloc | prefill 不走 CG，prefill `kv` per-fwd 分配可接受 |
| `actual_bs < graph_bs` padding | builder sentinel-pad indptr（重复最后 cumsum 值 → kv_len=0）+ batch_id_per_token (-1 跳过) |

prefill 的双源 kernel 自身 CG 兼容性与 decode 单源 kernel 等价 —— 两个 base
pointer 都是 forward_vars 预分配地址。

---

## 10. 不在本设计范围

- **decode 路径**：保留单源 `sparse_attn_v4_paged_decode`（详见 decode 设计文档）
- **CSA Indexer FP8 cache layout**：仍走 `cp_gather_indexer_k_quant_cache`，与
  sparse attn kernel 解耦
- **`sparse_attn_ragged_varlen` 何时彻底删除**：本设计上线、accuracy +
  performance 双验证通过后，作为单独 PR 清理。过渡期保留作为 fallback。
- **chunked prefill 调度策略**：依赖现有 ATOM scheduler 的 chunk 切分逻辑，本
  设计只关心如何在 chunk 已切好的情况下做 attention。

---

## 11. 工程实现 Phasing

按依赖顺序拆分：

| Phase | 改动 | 验证 |
|---|---|---|
| **P1** | 改 indexer：删 `_post_process_topk`，prefill 路径返回 raw `topk_in_seq` | grep 死代码、indexer 单测 |
| **P2** | 改 `csa_translate_pack`：`window_size` scalar → `skip_prefix_len_per_token` 数组（decode 路径 backwards-compat） | csa_translate_pack 单测 + decode GSM8K 回归 |
| **P3** | builder 端：算 `prefix_swa_count`、`extend_count`、5 个 indptr、staging 4 个 buffer | builder 单测 |
| **P4** | 写 `sparse_attn_v4_paged_prefill` triton kernel + reference impl + 单测 | 单测覆盖 pure prefill / chunked prefill / sentinel / boundary |
| **P5** | V4Attention.forward dispatch：`is_pure_decode==False` 分支换成新 kernel | eager smoke test |
| **P6** | 删旧路径：`sparse_attn_ragged_varlen` 调用、`_v4_build_sparse_inputs_batched`、`_post_process_topk`、indexer_meta dead fields | grep 0 引用 + 编译过 |
| **P7** | Mixed batch 验证（如适用） | mixed batch GSM8K |
| **P8** | GSM8K-100 nshot=3 ≥ 0.95 | accuracy regression |
| **P9** | commit + bench | perf 不回退 |

---

## 12. 文件路径速查

| 内容 | 路径 |
|---|---|
| 本设计文档（中文） | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_PREFILL_DESIGN.zh.md` |
| 本设计文档（English，待补） | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_PREFILL_DESIGN.en.md` |
| decode 设计文档（前置阅读） | `atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_DECODE_DESIGN.zh.md` |
| paged_prefill kernel + 包装（新增） | `atom/model_ops/v4_kernels/paged_prefill.py` |
| paged_decode kernel + 包装（沿用） | `atom/model_ops/v4_kernels/paged_decode.py` |
| csa_translate_pack（修改：scalar → per-token skip） | `atom/model_ops/v4_kernels/csa_translate_pack.py` |
| KV cache 分配 + 模块 view 绑定（沿用） | `atom/model_ops/attentions/deepseek_v4_attn.py` (`build_kv_cache_tensor`) |
| Per-fwd 索引构造（新增 prefill 路径） | `atom/model_ops/attentions/deepseek_v4_attn.py` (`prepare_prefill`、`_attach_v4_per_fwd_meta`) |
| V4 forward dispatch（修改：prefill 分支） | `atom/models/deepseek_v4.py` (`DeepseekV4Attention.forward`) |
| Indexer 输出语义（修改：prefill 也直出 `topk_in_seq`） | `atom/models/deepseek_v4.py` (`Indexer.indexer_score_topk`) |
