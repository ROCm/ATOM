# ATOM Mixed Prefill-Decode Batch 设计文档（Phase 2）

本文档是 chunked prefill 的第二阶段设计，基于 Phase 1（`docs/chunked_prefill_design.md`）已落地的前提。目标是在**同一次 forward** 中同时包含 prefill chunk 序列和 decode 序列，减少 prefill 对 decode 延迟的影响。

**前置依赖**：Phase 1 的 `num_kv_computed`、`ScheduledBatch` 切片、B 方案（中间 chunk 不算 logits）已全部实现。

---

## 1. 背景与目标

### 1.1 Phase 1 现状

Phase 1 完成后，scheduler 仍为「有 prefill 就整步仅 prefill」：

```
step 0: [prefill chunk 1 of seq A]           ← decode seq B/C 被阻塞
step 1: [prefill chunk 2 of seq A]           ← decode seq B/C 继续被阻塞
step 2: [prefill chunk 3 of seq A (final)]   ← decode seq B/C 继续被阻塞
step 3: [decode seq A, B, C]                 ← B/C 终于可以 decode
```

长 prompt 的多步 prefill 会**饿死** decode 序列，拉高所有在线请求的 inter-token latency。

### 1.2 目标

同一步中混合 prefill chunk 和 decode：

```
step 0: [prefill chunk 1 of seq A] + [decode seq B, C]
step 1: [prefill chunk 2 of seq A] + [decode seq B, C]
step 2: [prefill chunk 3 of seq A (final)] + [decode seq B, C]  ← A 产生首 token
step 3: [decode seq A, B, C]
```

### 1.3 不在本文范围

- Prefill/decode 分离部署（P/D disaggregation）
- 多条 prefill 请求与 decode 混批（M2+ 可扩展）
- CUDA graph 捕获混合 batch 形状

---

## 2. Phase 1 兼容性评估

### 2.1 可直接复用的设计

| Phase 1 设计 | Mixed batch 适用性 |
|-------------|-------------------|
| `num_kv_computed` 单字段 | 完全适用，prefill 和 decode 序列各自独立使用 |
| `ScheduledBatch` 已有 `total_seqs_num_prefill` / `total_seqs_num_decode` | 完全适用，mixed batch 下两者同时非零 |
| `context_lens[i]` 语义 | 适用：prefill seq 用 `num_kv_computed + chunk_size`，decode seq 用 `num_tokens` |
| `num_kv_computed` 更新时机 | 适用：postprocess 中按 seq type 分别处理 |
| BlockManager 全量 allocate | 适用：prefill 首次 allocate，decode 走 `may_append` |
| B 方案中间 chunk 不算 logits | 适用：mixed batch 中 prefill chunk 仍然跳过 |

### 2.2 需要修改的模块

| 模块 | 当前限制 | 所需改动 |
|------|---------|---------|
| **Scheduler** | 有 prefill → 整步 prefill，提前 return | 移除提前 return，Phase 2+3 合并 |
| **`prepare_prefill` / `prepare_decode`** | 二选一，metadata 结构不同 | 新增 `prepare_mixed` 或统一 `build` |
| **Attention dispatch** | `is_prefill` 布尔值控制全局 kernel 选择 | 拆分 batch 或使用 unified kernel |
| **`run_model`** | `is_prefill` 控制 eager vs CUDAGraph | mixed batch 走 eager |
| **`compute_logits` / `postprocess`** | prefill 全量 logits 或 B 方案跳过 | 按 seq 索引选取需要 logits 的行 |
| **`prepare_input_ids`** | prefill 和 decode 分别处理 | 合并输入布局 |
| **CUDA Graph** | decode 走 graph，prefill 走 eager | mixed batch 走 eager |

---

## 3. Scheduler 修改

### 3.1 统一调度

取消「有 prefill 就提前 return」的分支，将 Phase 1 的三阶段合并为一个连续流程：

```python
def schedule(self):
    scheduled_seqs = {}
    num_batched_tokens = 0
    num_scheduled_tokens = []
    num_seqs_prefill = 0
    num_seqs_decode = 0

    # ---- Phase 1: 续算 running 中未完成 prefill 的 seq ----
    for seq in self.running:
        if seq.num_kv_computed >= seq.num_prompt_tokens:
            continue  # 已完成 prefill，Phase 3 处理
        remaining = seq.num_prompt_tokens - seq.num_kv_computed
        chunk = min(remaining, self.max_num_batched_tokens - num_batched_tokens)
        if chunk <= 0:
            break
        num_batched_tokens += chunk
        scheduled_seqs[seq.id] = seq
        num_scheduled_tokens.append(chunk)
        seq.type = SequenceType.PREFILL
        num_seqs_prefill += 1

    # ---- Phase 2: 从 waiting 拉新 prefill 请求 ----
    while (self.waiting
           and len(scheduled_seqs) < self.max_num_seqs
           and num_batched_tokens < self.max_num_batched_tokens
           and (self.delay_factor <= 0 or self._passed_delay(time.time()))):
        seq = self.waiting[0]
        if not self.block_manager.can_allocate(seq):
            break
        self.block_manager.allocate(seq)
        remaining = seq.num_prompt_tokens - seq.num_kv_computed
        chunk = min(remaining, self.max_num_batched_tokens - num_batched_tokens)
        if chunk <= 0:
            break
        num_batched_tokens += chunk
        seq.status = SequenceStatus.RUNNING
        seq.type = SequenceType.PREFILL
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs[seq.id] = seq
        num_scheduled_tokens.append(chunk)
        num_seqs_prefill += 1

    # ---- Phase 3: decode（不再 exclusive）----
    #      与 prefill 共享 token 预算和 seq 槽位
    for seq in self.running:
        if seq.id in scheduled_seqs:
            continue  # 已在 prefill 中
        if seq.num_kv_computed < seq.num_prompt_tokens:
            continue  # partial prefill 但本步预算不足
        if len(scheduled_seqs) >= self.max_num_seqs:
            break
        if not self.block_manager.can_append(seq, self.mtp_k + 1):
            # preempt 逻辑（与 Phase 1 一致）
            ...
            continue
        num_new_tokens = self.mtp_k + 1
        if num_batched_tokens + num_new_tokens > self.max_num_batched_tokens:
            break
        num_batched_tokens += num_new_tokens
        self.block_manager.may_append(seq, num_new_tokens)
        scheduled_seqs[seq.id] = seq
        seq.type = SequenceType.DECODE
        num_scheduled_tokens.append(num_new_tokens)
        num_seqs_decode += 1

    if not scheduled_seqs:
        return None

    total_tokens_prefill = sum(
        n for seq, n in zip(scheduled_seqs.values(), num_scheduled_tokens)
        if seq.type == SequenceType.PREFILL
    )
    total_tokens_decode = sum(
        n for seq, n in zip(scheduled_seqs.values(), num_scheduled_tokens)
        if seq.type == SequenceType.DECODE
    )

    return ScheduledBatch(
        seqs=scheduled_seqs,
        num_scheduled_tokens=num_scheduled_tokens,
        total_tokens_num=total_tokens_prefill + total_tokens_decode,
        total_tokens_num_prefill=total_tokens_prefill,
        total_tokens_num_decode=total_tokens_decode,
        total_seqs_num=num_seqs_prefill + num_seqs_decode,
        total_seqs_num_prefill=num_seqs_prefill,
        total_seqs_num_decode=num_seqs_decode,
        num_spec_step=self.mtp_k,
        ...
    ), scheduled_seqs
```

**Prefix cache（与 Phase 1 一致）**：`waiting` 路径里若 prompt 已全量命中缓存、`num_new_tokens == 0`，须仍将 `num_kv_computed` 置为 `num_prompt_tokens - 1` 并调度 **1 个 token** 以计算 logits；mixed 合并后必须保留该分支，否则全缓存请求在 mixed 下调度会错。

**`running` deque 与本步结束后顺序**：Phase 1 decode 使用 `popleft` + `extendleft(reversed(...))` 维持顺序与抢占语义。Phase 2 在单步内可先按序收集 `scheduled_seqs`（prefill 先、decode 后），**postprocess 之后**对 `running` 的更新应与现网 decode 规则一致（谁在队首、preempt 弹谁），避免 FCFS 与 KV 释放顺序漂移；实现时建议对照 `scheduler.schedule` decode 分支写一小段「本步结束 reconcile running」的说明代码或单测。

### 3.2 Token 预算策略

Mixed batch 下 prefill 和 decode 共享 `max_num_batched_tokens` 预算。需要引入**可配置的 prefill 预算上限**，防止 prefill chunk 占满整个 budget：

```python
# 新增配置项
max_prefill_tokens_per_step: int  # 默认 = max_num_batched_tokens（向后兼容）
```

Phase 1 中的 chunk 计算变为：
```python
chunk = min(remaining,
            self.max_prefill_tokens_per_step - num_prefill_tokens,
            self.max_num_batched_tokens - num_batched_tokens)
```

`max_prefill_tokens_per_step` 与 `max_num_seqs` 同时生效：一步内 prefill 条数仍受 `max_num_seqs` 限制，decode 条数占用剩余槽位；ME1（1 prefill + max-1 decode）应用上述两维约束共同验收。

### 3.3 排序约定

**ScheduledBatch 中 seq 的排列顺序**：prefill 序列在前，decode 序列在后。

这是因为：
- `prepare_prefill` 读取 `batch.context_lens[:num_prefill]`
- `prepare_decode` 读取 `batch.context_lens[num_prefill:]`
- `scheduled_tokens` 中 prefill token 在前，decode token 在后

当前 `ScheduledBatch.__init__` 中 `seqs` 传入的是 `dict`（Python 3.7+ 保序），scheduler 按 Phase 1→2→3 顺序插入即可满足。

---

## 4. ScheduledBatch 修改

### 4.1 `scheduled_tokens` 布局

Mixed batch 下 token 内存布局：

```
[--- prefill tokens (变长) ---][--- decode tokens (定长 per seq) ---]
                                |
                           offset = total_tokens_num_prefill
```

当前 `ScheduledBatch.__init__` 已按此布局拼接（L224-229），无需改动。

### 4.2 `context_lens` 语义

```python
# ScheduledBatch.__init__ 中：
for i, seq in enumerate(seqs.values()):
    if seq.type == SequenceType.PREFILL:
        context_lens[i] = seq.num_kv_computed + num_scheduled_tokens[i]
    else:
        context_lens[i] = seq.num_tokens  # decode: 包含已生成 token
```

Phase 1 已做此区分，mixed batch 下自然适用。

### 4.3 `is_partial_prefill` 与 spec 元数据顺序

- **`is_partial_prefill`**：仅当本 batch 中**每一个** prefill 行均为中间 chunk（本步之后仍有未算完的 prompt）、且语义上对应 Phase 1「整步不算 logits」时为 `True`。**Mixed batch** 若含任一 decode 行，或任一 prefill 为本请求最后一段 chunk，则应为 `False`，以便 `run_model` / deferred 走「需 logits 的子集」逻辑。
- **`scheduled_spec_decode_tokens`**：`ScheduledBatch.__init__` 若仍用 `dict.values()` 展平为 ndarray，须保证 **dict 插入顺序与 decode 子序列在 `req_ids` 中的顺序一致**；更稳妥做法为按 `req_ids[total_seqs_num_prefill:]` 显式取值，避免与 MTP 路径错位。

---

## 5. Attention 方案 — Split Dispatch（核心设计）

### 5.1 设计思路

Mixed batch 的 attention 采用 **Split Dispatch** 方案：

- **Attention 算子内部做 if/else 分支**：prefill tokens 走现有 prefill kernel，decode tokens 走现有 decode kernel
- **Attention 前后的 gemm、norm 等算子**：在所有 tokens 上统一运行，不拆分

这样做的好处：
1. **复用现有 kernel**：prefill 继续用 `flash_attn_varlen`（MHA）或 `mla_prefill_fwd`（MLA），decode 继续用 `paged_attention`（MHA）或 `mla_decode_fwd`（MLA），无需引入新 kernel
2. **gemm/norm 无拆分开销**：`qkv_proj`、`o_proj`、FFN 的 gemm 和 layernorm 在完整 batch 上运行，避免 tensor 拆分/合并的 overhead
3. **kernel 性能不退化**：每种 kernel 在其最优场景下运行，不存在 unified kernel 在特定 head_dim 下的性能风险

### 5.2 整体数据流

```
                    所有 tokens (prefill + decode)
                           │
                     ┌─────▼─────┐
                     │  LayerNorm │  ← 统一运行
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │  QKV Proj  │  ← 统一 gemm
                     └─────┬─────┘
                           │
                  ┌────────▼────────┐
                  │  Attention 算子  │
                  │  ┌─────────────┐│
                  │  │ if prefill: ││
                  │  │  flash_attn ││  ← prefill tokens
                  │  │ else:       ││
                  │  │  paged_attn ││  ← decode tokens
                  │  └─────────────┘│
                  └────────┬────────┘
                           │
                     ┌─────▼─────┐
                     │  O Proj    │  ← 统一 gemm
                     └─────┬─────┘
                           │
                     ┌─────▼─────┐
                     │  FFN / MoE │  ← 统一运行
                     └─────┴─────┘
```

### 5.3 Attention 内部拆分逻辑（MHA 模型）

> MHA 模型（Llama、Qwen 等）的 Split Dispatch 实现。MLA 模型见 §6。

在 `attention_mha.py` 的 attention forward 中，根据 `context.is_mixed` 做 split：

```python
def forward(self, q, k, v, ...):
    if context.is_mixed:
        # 拆分 Q tensor
        num_prefill_tokens = batch.total_tokens_num_prefill
        q_prefill = q[:num_prefill_tokens]
        q_decode = q[num_prefill_tokens:]
        k_prefill = k[:num_prefill_tokens]
        k_decode = k[num_prefill_tokens:]
        v_prefill = v[:num_prefill_tokens]
        v_decode = v[num_prefill_tokens:]

        # Prefill 部分 → flash_attn_varlen（现有 prefill kernel）
        out_prefill = flash_attn_varlen_func(
            q_prefill, k_prefill, v_prefill,
            cu_seqlens_q=attn_metadata.prefill_cu_seqlens_q,
            cu_seqlens_k=attn_metadata.prefill_cu_seqlens_k,
            max_seqlen_q=attn_metadata.prefill_max_seqlen_q,
            max_seqlen_k=attn_metadata.prefill_max_seqlen_k,
        )

        # Decode 部分 → paged_attention（现有 decode kernel）
        out_decode = paged_attention(
            q_decode,
            kv_cache,
            block_tables=attn_metadata.decode_block_tables,
            context_lens=attn_metadata.decode_context_lens,
        )

        # 合并输出
        output = torch.cat([out_prefill, out_decode], dim=0)
        return output
    elif context.is_prefill:
        return flash_attn_varlen_func(...)   # 现有 prefill 路径
    else:
        return paged_attention(...)           # 现有 decode 路径
```

### 5.4 Metadata 构建

`prepare_mixed` 同时构建 prefill 和 decode 两套 metadata：

- **Prefill metadata**：`prefill_cu_seqlens_q`、`prefill_cu_seqlens_k`、`prefill_max_seqlen_q/k`（来自 `prepare_prefill` 逻辑）
- **Decode metadata**：`decode_block_tables`、`decode_context_lens`（来自 `prepare_decode` 逻辑）
- **共享 metadata**：`positions`、`slot_mapping`（所有 tokens 的位置和 KV cache 写入槽位）

```python
def prepare_mixed(self, batch: ScheduledBatch, bs: int):
    """构建 mixed batch 的双套 attention metadata。"""
    num_prefill = batch.total_seqs_num_prefill
    num_decode = batch.total_seqs_num_decode

    # ---- 共享：positions + slot_mapping ----
    # ... 与现有 prepare_prefill/prepare_decode 逻辑合并 ...

    # ---- Prefill 部分 metadata（seq 0..num_prefill-1）----
    prefill_cu_seqlens_q, prefill_cu_seqlens_k = ...  # 来自 prepare_prefill 逻辑

    # ---- Decode 部分 metadata（seq num_prefill..bs-1）----
    decode_block_tables = batch.block_tables[num_prefill:]
    decode_context_lens = batch.context_lens[num_prefill:]

    attn_metadata = AttentionMetaData(
        # prefill 部分
        prefill_cu_seqlens_q=...,
        prefill_cu_seqlens_k=...,
        prefill_max_seqlen_q=...,
        prefill_max_seqlen_k=...,
        # decode 部分
        decode_block_tables=...,
        decode_context_lens=...,
        # 共享
        slot_mapping=...,
        positions=...,
    )
    return attn_metadata, positions
```

### 5.5 KV Cache 写入

KV cache 写入通过 `slot_mapping` 统一处理，在 attention 之前的 `reshape_and_cache` 中完成，不区分 prefill/decode。Split Dispatch 不影响 KV cache 写入逻辑。

---

## 6. Attention 方案 — MLA 模型的 Split Dispatch

DeepSeek V2/V3/V3.2 使用 MLA（Multi-Latent Attention），prefill 和 decode 走**不同的 attention 算子和数据路径**。同样采用 Split Dispatch 方案，但拆分位置与 MHA 不同。

### 6.1 MLA 当前架构分析

`MLAAttention.forward_impl_server_mode`（`attention_mla.py:609-748`）的分支逻辑：

```
if is_prefill and not use_prefill_mla:
    # 路径 A: Prefill MHA
    #   Q projection → full Q (num_heads, qk_head_dim=192)
    #   kv_b_proj(kv_c_normed) → full K, V (num_heads, qk_nope+v_head_dim)
    #   flash_attn_varlen_func(Q, K, V)  ← 标准 MHA，head_dim=192
    #   o_proj(output)
else:
    # 路径 B: Decode MLA（或 DSA 下的 prefill_mla）
    #   Q absorption: q_proj → bmm(W_K) → ql_nope (latent space)
    #   fused_qk_rope_concat_and_cache_mla → q_out (kv_lora_rank + rope_dim=576)
    #   if is_prefill:
    #       mla_prefill_fwd / mla_decode_fwd(q_out, kv_cache)  ← paged attention on latent cache
    #   else:
    #       mla_decode_fwd(q_out, kv_cache)                    ← paged attention on latent cache
    #   _v_up_proj_and_o_proj(o)  ← bmm(W_V) + o_proj
```

两条路径的关键差异：

| | 路径 A: Prefill MHA | 路径 B: MLA |
|---|---|---|
| **Q 维度** | `(B, num_heads, qk_head_dim)` = 192 | `(B, num_heads, kv_lora_rank + rope_dim)` = 576 |
| **KV 来源** | `kv_b_proj` 展开为 full K, V | 直接用压缩 latent cache |
| **Attention kernel** | `flash_attn_varlen_func` | `mla_decode_fwd` / `mla_prefill_fwd` |
| **Output** | `(B, N, v_head_dim)` → `o_proj` | `(B, N, kv_lora_rank)` → `bmm(W_V)` → `o_proj` |
| **Metadata** | `cu_seqlens_q`, `cu_seqlens_k` | `cu_seqlens_q`, `kv_indptr`, `kv_indices`, `kv_last_page_lens` |

### 6.2 MLA 的 Split Dispatch 特殊性

**Prefill MHA 和 Decode MLA 的 Q 投影、output 投影不同**，因此拆分点需要更靠前：

1. **Q 维度不同**：MHA 的 Q 是 `qk_head_dim = 192`，MLA 的 Q 是 `kv_lora_rank + qk_rope = 576`
2. **Output 处理不同**：MHA 输出 `v_head_dim` 后直接 `o_proj`，MLA 输出 `kv_lora_rank` 后需要额外的 `bmm(W_V)` + `o_proj`

因此 MLA 模型的 Split Dispatch **在 `forward_impl_server_mode` 内部拆分**（即在 Q/KV 投影之前），而非像 MHA 模型那样在 attention forward 中间拆分。但外层 transformer block 的 gemm（不含 attention 内的 Q/KV/O proj）和 norm 仍然在所有 tokens 上统一运行。

```
                    所有 tokens (prefill + decode)
                           │
                     ┌─────▼─────┐
                     │  LayerNorm │  ← 统一运行
                     └─────┬─────┘
                           │
          ┌────────────────▼────────────────┐
          │  MLAAttention.forward_impl...   │
          │  ┌────────────────────────────┐ │
          │  │ if prefill tokens:         │ │
          │  │   Q proj (MHA dim=192)     │ │
          │  │   kv_b_proj → full K,V     │ │
          │  │   flash_attn_varlen        │ │
          │  │   o_proj                   │ │
          │  │ if decode tokens:          │ │
          │  │   Q absorption (MLA dim=576)│ │
          │  │   mla_decode_fwd           │ │
          │  │   bmm(W_V) + o_proj        │ │
          │  └────────────────────────────┘ │
          └────────────────┬────────────────┘
                           │
                     ┌─────▼─────┐
                     │  FFN / MoE │  ← 统一运行
                     └─────┴─────┘
```

### 6.3 `forward_impl_server_mode` 修改

```python
def forward_impl_server_mode(self, q, k_nope, k_rope, positions, q_scale):
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    context = forward_context.context

    kv_cache = kv_cache_data[f"layer_{self.layer_num}"].k_cache

    if context.is_mixed:
        # ---- Mixed Batch: Split Dispatch ----
        num_prefill_tokens = batch.total_tokens_num_prefill

        # 拆分输入
        q_prefill = q[:num_prefill_tokens]
        q_decode = q[num_prefill_tokens:]
        k_nope_prefill = k_nope[:num_prefill_tokens]
        k_nope_decode = k_nope[num_prefill_tokens:]
        k_rope_prefill = k_rope[:num_prefill_tokens]
        k_rope_decode = k_rope[num_prefill_tokens:]

        # Prefill 部分 → 走 Prefill MHA 路径（路径 A）
        #   Q proj → full Q, kv_b_proj → full K/V
        #   flash_attn_varlen_func
        #   o_proj
        out_prefill = self._forward_prefill_mha(
            q_prefill, k_nope_prefill, k_rope_prefill,
            kv_cache, attn_metadata.prefill_metadata,
        )

        # Decode 部分 → 走 Decode MLA 路径（路径 B）
        #   Q absorption → mla_decode_fwd
        #   bmm(W_V) + o_proj
        out_decode = self._forward_decode_mla(
            q_decode, k_nope_decode, k_rope_decode,
            kv_cache, attn_metadata.decode_metadata,
        )

        # 合并输出（两者 output dim 相同，都是 o_proj 输出后的 hidden_dim）
        return torch.cat([out_prefill, out_decode], dim=0)

    elif context.is_prefill and not use_prefill_mla:
        # 路径 A: 纯 Prefill MHA（与现有一致）
        ...
    else:
        # 路径 B: 纯 Decode MLA 或 prefill_mla（与现有一致）
        ...
```

**关键点**：prefill 和 decode 各自走完整的独立路径（包括 Q/KV 投影和 o_proj），最终输出 dim 相同（`hidden_dim`），可以直接 `cat`。

### 6.4 替代方案：统一走 MLA 路径

另一种可选方案是 mixed batch 下所有 seq（prefill + decode）统一走路径 B（MLA），避免拆分。依据：

1. `_forward_prefill_mla` 已支持 paged attention + variable seqlen_q
2. `mla_decode_fwd` 已支持 variable `max_seqlen_qo`
3. KV cache 格式统一

代价是 prefill chunk 多一次 Q absorption bmm 和 V up-projection bmm，但省去 `kv_b_proj` 展开。如果 Split Dispatch 实现复杂度过高，可退化为此方案。

### 6.5 MHA vs MLA Split Dispatch 对照

| | MHA 模型 | MLA 模型 |
|---|---|---|
| **外层 gemm/norm** | 统一运行 | 统一运行 |
| **拆分位置** | Attention forward 内部（Q/K/V 已计算） | `forward_impl_server_mode` 入口（Q/KV 投影之前） |
| **Prefill kernel** | `flash_attn_varlen` | `flash_attn_varlen`（Prefill MHA 路径） |
| **Decode kernel** | `paged_attention` | `mla_decode_fwd` |
| **合并点** | Attention output（same dim） | o_proj output（same hidden_dim） |

### 6.6 Metadata 构建

MLA 的 mixed batch 需要构建两套独立的 attention metadata：

- **Prefill metadata**：`cu_seqlens_q`、`cu_seqlens_k`、slot_mapping（来自 `prepare_prefill` 逻辑）
- **Decode metadata**：`kv_indptr`、`kv_indices`、`kv_last_page_lens`、block_tables（来自 `prepare_decode` 逻辑）

`AiterMLAMetadataBuilder.build` 在 `is_mixed` 时同时调用两套 prepare 逻辑，将结果打包到 `AttentionMetaData` 的 `prefill_metadata` / `decode_metadata` 子结构中。

### 6.7 需要验证的风险点

| 编号 | 风险 | 验证方式 |
|---|---|---|
| R1 | Split Dispatch 的 tensor 拆分/合并开销（每层 attention 都要做） | 与纯 decode 的 latency 对比，overhead 应 < 5% |
| R2 | KV cache 写入顺序：prefill 和 decode 的 `slot_mapping` 不重叠 | 检查 slot_mapping 无重复 |
| R3 | `flash_attn_varlen` 和 `mla_decode_fwd` 的 KV cache 格式兼容 | 两者共享同一 kv_cache tensor，通过 slot_mapping 分别写入 |
| R4 | DSA (DeepSeek V3.2 sparse attention) 的 split dispatch | sparse metadata 需要按 prefill/decode 分别构建 |

---

## 7. ModelRunner 修改

### 7.1 `prepare_input_ids`

当前 `tokenIDProcessor.prepare_input_ids` 以 `total_reqs_prefill > 0` 为条件提前 return prefill 的 input_ids。mixed batch 下需要合并：

```python
def prepare_input_ids(self, batch: ScheduledBatch) -> torch.Tensor:
    total_tokens_prefill = batch.total_tokens_num_prefill
    total_tokens_decode = batch.total_tokens_num_decode

    # Prefill tokens: 从 scheduled_tokens 取前 total_tokens_prefill 个
    self.input_ids.np[:total_tokens_prefill] = batch.scheduled_tokens[:total_tokens_prefill]
    self.input_ids.copy_to_gpu(total_tokens_prefill)

    if total_tokens_decode == 0:
        return self.input_ids.gpu[:total_tokens_prefill]

    # Decode tokens: deferred output 逻辑（与 Phase 1 相同）
    # ... 放在 prefill tokens 之后 ...
    # 最终布局: [prefill_tokens | decode_tokens]
    return self.input_ids.gpu[:total_tokens_prefill + total_tokens_decode]
```

### 7.2 `prepare_inputs`

```python
def prepare_inputs(self, batch, input_ids=None):
    is_mixed = batch.total_tokens_num_prefill > 0 and batch.total_tokens_num_decode > 0
    is_prefill_only = batch.total_tokens_num_prefill > 0 and batch.total_tokens_num_decode == 0

    if is_mixed or is_prefill_only:
        # Mixed batch 和纯 prefill 都走 eager
        attn_metadata, positions = self.attn_metadata_builder.build(batch=batch, bs=bs)
        graph_bs = batch.total_tokens_num  # eager 模式，graph_bs = total tokens
    else:
        # 纯 decode：走 CUDAGraph（与现有逻辑一致）
        ...
```

### 7.3 `run_model`

```python
def run_model(self, input_ids, batch=None):
    forward_context = get_forward_context()
    context = forward_context.context
    is_prefill = context.is_prefill
    is_mixed = context.is_mixed  # 新增字段

    if is_prefill or is_mixed or self.enforce_eager or bs > self.graph_bs[-1]:
        # eager 路径
        hidden_states = self.model(input_ids, positions)

        if is_mixed:
            # Mixed batch: 仅对需要 logits 的行计算
            logits_indices = self._get_logits_indices(batch, context)
            if logits_indices is not None:
                logits = self.model.compute_logits(hidden_states[logits_indices])
            else:
                logits = None  # 全是中间 chunk + decode 被 defer
        elif context.is_partial_prefill:
            logits = None  # B 方案：中间 chunk
        else:
            logits = self.model.compute_logits(hidden_states)
    else:
        # CUDAGraph 路径（纯 decode）
        ...
```

### 7.4 `_get_logits_indices` — 选取需要 logits 的行

Mixed batch 中需要 logits 的行：
- **最后 prefill chunk** 的序列：仅最后一个 token 位置
- **Decode 序列**：每个 seq 的（最后一个或被接受的）token 位置
- **中间 prefill chunk** 的序列：**不需要 logits**

**实现注意**：`ScheduledBatch` 当前不持有 `seqs` 映射时，须用 `seq_id` 从 scheduler 侧 `running`/`scheduled_seqs` 解析 `Sequence`，或在 batch 上增加与 `req_ids` 对齐的 `num_prompt_tokens`（等）数组；下面伪代码用 `get_seq(seq_id)` 表示该解析。

```python
def _get_logits_indices(self, batch, context):
    """Return indices into hidden_states that need compute_logits."""
    indices = []
    token_offset = 0

    for i, (seq_id, num_tokens) in enumerate(zip(batch.req_ids, batch.num_scheduled_tokens)):
        num_tokens = int(num_tokens)
        if i < batch.total_seqs_num_prefill:
            seq = get_seq(seq_id)
            kv_computed = batch.num_kv_computed[i]
            is_final = kv_computed + num_tokens >= seq.num_prompt_tokens
            if is_final:
                indices.append(token_offset + num_tokens - 1)
        else:
            indices.append(token_offset + num_tokens - 1)
        token_offset += num_tokens

    if not indices:
        return None
    return torch.tensor(indices, dtype=torch.long, device=self.device)
```

### 7.5 `postprocess`

Mixed batch 中 `postprocess` 需要同时处理：
1. **中间 prefill chunk** — 不采样，更新 `num_kv_computed`
2. **最后 prefill chunk** — 采样第一个 token
3. **Decode seq** — 采样下一个 token

```python
def postprocess(self, batch, logits, ...):
    is_mixed = forward_context.context.is_mixed
    is_partial = forward_context.context.is_partial_prefill

    if is_partial and not is_mixed:
        # 纯 partial prefill batch（Phase 1 逻辑）
        ...
        return

    if logits is not None:
        # logits 可能是 sparse 的（仅包含需要的行）
        # 需要按 logits_indices 映射回 batch 中的 seq
        sampled_tokens = self.sampler(logits, temperatures, ...)

    # 构建输出：
    # - 中间 prefill chunk → 空输出
    # - 最后 prefill chunk + decode → 有 token 输出
    ...
```

---

## 8. Attention Dispatch 修改

### 8.1 MHA 模型的 Split Dispatch

在 attention forward 中新增 `is_mixed` 分支（见 §5.3）。`dispatch_backend` 不再返回单个 kernel，而是在 forward 内部根据 token 类型分别调用：

- Prefill tokens → `flash_attn_varlen`（现有 prefill kernel）
- Decode tokens → `paged_attention`（现有 decode kernel）

`dispatch_backend` 仅在纯 prefill 或纯 decode 时使用，mixed batch 跳过 dispatch 直接走 split 逻辑。

### 8.2 MLA 模型的 Split Dispatch

MLA 模型在 `MLAAttention.forward_impl_server_mode` 中新增 `context.is_mixed` 分支（见 §6.3）。拆分在 Q/KV 投影之前，prefill 走 Prefill MHA 路径，decode 走 Decode MLA 路径。

### 8.3 KV Cache 写入

Mixed batch 中 prefill 和 decode seq 的 KV cache 写入通过 `slot_mapping` 统一处理：
- `rope_cache` 中的 `reshape_and_cache` 使用 `slot_mapping` 写入，不区分 prefill/decode
- `slot_mapping` 已包含所有 seq 的新写入位置

无需改动。

### 8.4 `has_cached` 和 gather 路径

Mixed batch 中，Split Dispatch 下 prefill 部分仍使用 `flash_attn_varlen`，因此 chunked prefill 的 `has_cached` gather 路径仍然适用。Decode 部分走 paged attention，不需要 gather。两部分独立处理，互不影响。

---

## 9. Deferred Output 与 MTP

### 9.1 Mixed batch 下的 deferred output

Mixed batch 中 `prev_batch` 可能包含 prefill 和 decode 的混合。`prepare_sampled_ids` 中的 `get_token_locations` 需要正确处理：

- 上一步是 mixed batch → 本步也是 mixed batch
- 上一步的 prefill seq 中间 chunk **无 sampled token**，本步的 deferred output 不应包含它们
- 解决方式：中间 chunk 的 seq 不出现在 `req_ids_out` 中

### 9.2 MTP / Spec Decode

Mixed batch 中 MTP 仅对 **decode seq** 生效：
- `spec_decode_metadata` 仅包含 decode seq 的索引
- `propose_draft_token_ids` 仅在 decode seq 的 hidden_states 上调用
- Prefill seq（无论中间还是最后 chunk）**不参与** spec decode

```python
if not is_prefill_only and hasattr(self, 'drafter') and not batch.is_dummy_run:
    # 仅提取 decode seq 的 metadata
    decode_start = batch.total_seqs_num_prefill
    decode_num_scheduled = num_scheduled_tokens[decode_start:]
    decode_cu_seqlens = cu_seqlens_q[decode_start:]
    # ... 计算 spec_decode_metadata
```

---

## 10. CUDA Graph

Mixed batch **不走 CUDA Graph**，与纯 prefill 一致（eager mode）。

原因：
- Mixed batch 的 token 总数和 seq 组合是动态的
- CUDAGraph 需要固定的 input shape
- 相对固定形状的纯 decode 步，mixed 更不易 capture graph；长 prompt 负载下 mixed 可能**很频繁**，不走 graph 的代价需用 §13.2 压测验证

纯 decode batch（无 prefill seq）**继续走 CUDAGraph**，与 Phase 1 一致。

---

## 11. DP 同步

### 11.1 `get_next_batch_info` 适配

Phase 1 已将 partial prefill 识别为 prefill。Mixed batch 下，一个 rank 可能是 mixed（prefill + decode），另一个 rank 只有 decode。

**关键约束**：DP 要求所有 rank 走相同的 forward 路径（prefill 或 decode），因为 MORI 的 MoE expert-parallel all-to-all 需要对齐。

Mixed batch 对 DP 的影响：
- Mixed batch 整体视为 **prefill**（因为包含 prefill token，走 eager）
- 其他 rank 如果只有 decode，需要 `dummy_prefill_execution` 对齐
- `get_next_batch_info` 返回 `(True, total_tokens)` 当有任何 prefill seq 时

这与 Phase 1 的行为一致，无需额外改动。

### 11.2 状态同步修正

**收紧条件**：应用 `(True, token_count)` 的时机应对齐「本步是否必须走 eager / 与含 prefill 的 rank 对齐」，而非仅凭 `waiting` 非空——否则在「仅 waiting、本步仍可纯 decode running」的场景会误触发全 rank dummy prefill。推荐与 **现网 Phase 1 `get_next_batch_info`** 行为对齐后再改；若当前实现已用 `has_waiting`，mixed 合并时需复查是否应改为「`running` 中存在未完成 prefill」或「调度结果中含 prefill token」等更可证条件。

```python
def get_next_batch_info(self) -> tuple[bool, int]:
    has_partial_prefill = any(
        seq.num_kv_computed < seq.num_prompt_tokens for seq in self.running
    )
    # 是否追加 has_waiting 以 Phase 1 实装为准；mixed 下优先保证
    # 「任 rank 本步含 prefill」时全体走 prefill/eager 路径且 token 数一致。
    if has_partial_prefill:  # 按需与现有逻辑 OR 其它条件
        prefill_tokens = ...
        decode_tokens = ...
        return (True, prefill_tokens + decode_tokens)
    elif self.running:
        return (False, len(self.running))
    else:
        return (False, 0)
```

---

## 12. Context 扩展

`Context` dataclass 新增字段：

```python
@dataclass
class Context:
    positions: torch.Tensor
    is_prefill: bool = False         # 有 prefill seq（含 mixed）
    is_mixed: bool = False           # 同时有 prefill 和 decode seq
    is_partial_prefill: bool = False # True 仅当全是中间 chunk且无 decode；mixed 一般为 False
    is_dummy_run: bool = False
    batch_size: int = 0
    graph_bs: int = 0
    is_draft: bool = False
    num_prefill_seqs: int = 0        # prefill seq 数量
    num_decode_seqs: int = 0         # decode seq 数量
    logits_indices: Optional[torch.Tensor] = None  # 需要 compute_logits 的行索引
```

---

## 13. 测试与验收

### 13.1 正确性

| 编号 | 场景 | 期望 |
|------|------|------|
| M1 | 1 条长 prefill chunk + 3 条 decode，greedy | 所有 seq 输出与串行基线一致 |
| M2 | 多条短 prefill（一步完成）+ decode | 输出一致 |
| M3 | 中间 chunk + decode mixed | decode seq 正常产出 token，prefill seq 无输出 |
| M4 | 最后 chunk + decode mixed | prefill seq 产出首 token，decode seq 正常 |
| M5 | 纯 prefill batch（回归） | 与 Phase 1 一致 |
| M6 | 纯 decode batch（回归） | 与 Phase 1 一致，仍走 CUDAGraph |

### 13.2 性能

| 编号 | 验证 | 方法 |
|------|------|------|
| MP1 | mixed batch 下 decode latency 对比纯 decode | profiler：mixed 不应超过纯 decode 的 1.5x |
| MP2 | Split Dispatch 每层 tensor split/cat 开销 | mixed batch vs 纯 decode latency 对比 |
| MP3 | Mixed batch 整体吞吐 vs Phase 1 分离调度 | end-to-end benchmark |

### 13.3 边界

| 编号 | 场景 | 验证 |
|------|------|------|
| ME1 | max_num_seqs 打满：1 prefill + (max-1) decode | 正常运行 |
| ME2 | prefill 占满 budget，decode 被跳过 | 下一步 decode 正常 |
| ME3 | TP > 1 mixed batch | 各 rank 一致，无死锁 |
| ME4 | DP mixed batch | dummy prefill 对齐正确 |
| ME5 | MTP + mixed batch | spec decode 仅在 decode seq 上生效 |

### 13.4 MLA 模型专项

| 编号 | 场景 | 验证 |
|------|------|------|
| MLA1 | DeepSeek V3 mixed batch: 1 prefill chunk + 3 decode，greedy | 输出与纯 MLA 串行基线一致 |
| MLA2 | DeepSeek V3 mixed batch: Split Dispatch prefill MHA + decode MLA 路径 | 输出与串行基线一致 |
| MLA3 | Split Dispatch 每层 tensor 拆分/合并开销 | overhead < 5% vs 纯 decode |
| MLA4 | Prefill 和 Decode 的 KV cache slot_mapping 不重叠 | 无 CUDA error |
| MLA5 | DeepSeek V3.2 (DSA sparse) mixed batch | sparse_kv_indptr 正确构建 |
| MLA6 | MLA mixed batch + MTP | spec decode 仅 decode seq，MLA V up-proj 正确 |

---

## 14. 建议开发顺序

| 阶段 | 内容 | 验收标准 |
|------|------|----------|
| P2-M1 | Scheduler 统一调度 + `ScheduledBatch` mixed 布局 + `prepare_input_ids` 合并 | 可产出 mixed batch 的 ScheduledBatch |
| P2-M2a | MHA 模型: `prepare_mixed` 双套 metadata + Split Dispatch（flash_attn_varlen + paged_attention） | MP2 通过 |
| P2-M2b | MLA 模型: Split Dispatch（Prefill MHA + Decode MLA） + `prepare_mixed` metadata | MLA1-MLA4 通过 |
| P2-M3 | `run_model` + `_get_logits_indices` + `postprocess` mixed 分支 | M1-M4 正确性通过 |
| P2-M4 | Deferred output + MTP 适配 | ME5, MLA6 通过 |
| P2-M5 | DP 同步 + 性能调优 + `max_prefill_tokens_per_step` 配置 | MP1, MP3, ME3-ME4 通过 |
| P2-M6 | DSA sparse mixed batch（DeepSeek V3.2 专项） | MLA5 通过 |

---

## 15. Phase 1 设计需要的前置改动

在 Phase 1 实现时，以下设计应**提前考虑** mixed batch 兼容性，避免 Phase 2 大规模重写：

1. **`ScheduledBatch.__init__` 中的 `context_lens` 计算**：已按 seq type 区分，无需改动。
2. **`scheduled_tokens` 布局**：保持 prefill 在前、decode 在后。Phase 1 只有 prefill 或只有 decode，自然满足。
3. **`num_kv_computed` 列表传入 `ScheduledBatch`**：Phase 1 即应实现，Phase 2 直接使用。
4. **`Context` 的 `is_partial_prefill` 字段**：Phase 1 实现，Phase 2 新增 `is_mixed`。
5. **`postprocess` 中按 seq type 分别处理**：Phase 1 中间 chunk 跳过，Phase 2 自然扩展为 mixed 处理。

**结论：Phase 1 的设计与 mixed batch Phase 2 完全兼容，无需预留 hook 或抽象层。Phase 2 的改动集中在 scheduler 取消 early return、attention Split Dispatch（算子内部 if/else 分支 prefill/decode kernel，前后 gemm/norm 统一运行）、`run_model` 新增 mixed 分支三处。**

---

## 16. 参考文献

- `docs/chunked_prefill_design.md` — Phase 1 设计
- `docs/scheduling_kv_cache_guide.md` — 调度与 KV 基础
- `atom/model_ops/attention_mha.py` — MHA: `flash_attn_varlen`（prefill）/ `paged_attention`（decode）/ Split Dispatch 实现
- `atom/model_ops/attention_mla.py` — MLA: `MLAAttention`、`_forward_prefill_mha`、`_forward_prefill_mla`、`_forward_decode` 实现
- `atom/model_ops/attentions/aiter_attention.py` — MHA metadata builder (`AiterAttentionMetadataBuilder`)
- `atom/model_ops/attentions/aiter_mla.py` — MLA metadata builder (`AiterMLAMetadataBuilder`)、persistent buffer 管理

---

*文档版本：Phase 2 mixed batch 设计，基于 Phase 1 chunked prefill 已落地。*
