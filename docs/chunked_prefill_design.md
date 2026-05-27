# ATOM Chunked Prefill 设计与开发清单（中间 Chunk 采用 B 方案）

本文档描述在 ATOM 中实现 **chunked prefill** 的目标架构、与现有模块的衔接，以及按 **B 方案**（中间 chunk 仅写 KV、不计算 logits / 不采样）落地时的开发任务拆分。不考虑 **prompt logprobs**；常规生成场景下仅在 **prefill 完成边界**做一次 lm_head + 采样。

**相关代码路径（现状）**

| 模块 | 路径 |
|------|------|
| 调度 | `atom/model_engine/scheduler.py` |
| 序列 | `atom/model_engine/sequence.py` |
| KV / prefix | `atom/model_engine/block_manager.py` |
| Batch 与 attention 元数据 | `atom/model_ops/attentions/backends.py`（`CommonAttentionBuilder.prepare_prefill`） |
| 执行 | `atom/model_engine/model_runner.py`（`forward` / `run_model` / `postprocess`） |
| MHA | `atom/model_ops/attention_mha.py` |

---

## 1. 背景与目标

### 1.1 现状问题

- **调度**：`Scheduler.schedule()` 在 prefill 阶段按 `max_num_batched_tokens` 尝试从 `waiting` 连续拉请求；单条请求的 **未缓存 token 数**若超过剩余预算则 **整批 break**，且 **无 chunked prefill**（整段 prompt 须一次通过预算门）。
- **Token 预算与 prefix**：首次调度时 `Sequence.num_cached_tokens` 仍为 0，预算比较的是 **整段 `num_tokens`**；`BlockManager.allocate()` 之后才更新 prefix 命中带来的 `num_cached_tokens`。跨请求 prefix 与 **本请求多步 prefill** 的语义需分离（见 §3）。
- **执行**：prefill 路径下 `run_model` 对 **全部** `hidden_states` 调用 `compute_logits`，`postprocess` 对 **全部** logits 采样；长 prompt 下 lm_head 与采样成本高。

### 1.2 目标

1. 按配置 **`max_num_batched_tokens`**（及 `max_num_seqs`、KV 可用性）将 **单条长 prompt** 拆成多步 prefill，每步仅推进一段 token，**与 decode 是否混批**可分期实现（当前 scheduler 为「有 prefill 则整步仅 prefill」）。
2. **B 方案**：除 **prefill 完成的最后一步**（或等价边界）外，中间 chunk **不调用 `compute_logits`**、**不调用 `sampler`**，仅运行 backbone + attention 写 KV。
3. 与 **prefix caching**、**deferred output**、**MTP/spec**、**CUDA graph** 的边界写清楚，避免静默错误。

---

## 2. B 方案语义（中间 Chunk）

| 步骤类型 | `model(...)` | `compute_logits` | `sampler` | 对外 token / 状态 |
|----------|----------------|-------------------|-----------|-------------------|
| 中间 prefill chunk | 是（本 chunk 各 token） | **否** | **否** | 无新生成 token；KV 递增 |
| 最后 prefill chunk（prompt 已全部进 KV） | 是 | **是**（仅需要的行，见 §6） | **是** | 产生第一个 decode token（或进入 decode 批） |

**说明**

- 每个中间 chunk 仍需 **完整 transformer + 各层 KV 写入**；不能「只写 KV 不算 attention」。
- 不考虑 logprobs 时，**唯一必须在 prefill 侧用 logits 的时刻**是 **prompt 结束后采第一个生成 token**（或你们定义的「prefill 结束步」）。
- 与 **vLLM A 方案**（每步 1 行 logits + discard）相比，B 在中间步 **零 lm_head / 零采样**，长序列多 chunk 时更省。

---

## 3. 数据模型与命名

### 3.1 KV 进度字段（`Sequence`）— 单字段设计

**决策：使用单一字段 `num_kv_computed` 作为 KV 进度的唯一真值来源**，不引入第三个字段名。废弃 `num_cached_tokens` 在 attention 路径中的直接使用。

- **`num_kv_computed`**：本请求在 **Paged KV** 中已连续有效的前缀长度（**不含**本步尚未执行 forward 的 token）。
  - 初始值：`0`
  - **更新时机 1** — `BlockManager.allocate()` 完成后：将 prefix cache 命中的 token 数写入 `num_kv_computed`（原来写入 `num_cached_tokens` 的值）。
  - **更新时机 2** — 每步 forward 完成后（`scheduler.postprocess` 或等价位置）：`num_kv_computed += num_scheduled_tokens[i]`。
  - **deallocate / preempt 时**：重置为 `0`。
- **`num_cached_tokens`**：**仅保留用于 `CacheStats` 统计**，不再参与 attention metadata 计算或 `ScheduledBatch` 切片。

**全链路使用规则**：

| 使用场景 | 使用字段 |
|----------|----------|
| Scheduler 预算计算：本步需 forward 的 token 数 | `seq.num_prompt_tokens - seq.num_kv_computed` |
| `ScheduledBatch` 切片 offset | `seq.num_kv_computed` |
| `prepare_prefill` 中 `cached_seqlen` / `kv_prefix_len` | `seq.num_kv_computed` |
| `prepare_prefill` 中 positions 起始 | `range(num_kv_computed, num_kv_computed + chunk_size)` |
| `CacheStats` 统计（仅日志） | `seq.num_cached_tokens`（只在 allocate 后记录一次） |

**为什么不保留双字段**：当前代码中 `num_cached_tokens` 在 `scheduler.py:350`、`backends.py:161`、`ScheduledBatch.__init__:242` 等多处使用。如果双字段并存，任何一处遗漏替换或更新顺序错误都会导致 **gather 长度不匹配**或 **KV 写入位置错误**，且这类 bug 表现为**静默的 logits 错误**，极难排查。

### 3.2 与 `BlockManager` 的衔接

**决策：M1 阶段采用「首次 allocate 全部块」策略；增量分配作为后续优化。**

- **M1 — 全量 allocate**：
  - 首次从 waiting 进入 running 时，`allocate()` 一次性分配全部 prompt 块（与现有行为一致）。
  - 后续 chunk 续算时 **不再调用 `allocate`**，直接使用已分配的 `block_table` 写入 KV。
  - **优点**：不改 `BlockManager` API，`allocate()` 的 `assert not seq.block_table` 约束不受影响。
  - **缺点**：长 prompt 一次占满块，可能阻塞后续短请求的 `can_allocate`。对于 prompt 远超 `max_num_batched_tokens` 的场景（如 128K context），前几步 chunk 时大量块已分配但 KV 尚未写入，属于「预占」。
  - **可接受原因**：当前 ATOM 的非 chunked 路径本就需要一次分配全部块，chunked 并未增加额外块消耗，只是把写入时间拉长了。

- **后续优化 — 增量 allocate**（M5+ 可选）：
  - 需移除 `allocate()` 中的 `assert not seq.block_table`。
  - 需处理 prefix caching hash 链：增量 allocate 时前缀 hash 依赖前面块的 token 已确定，chunked 写入场景下中间块的 hash 计算时机需要仔细对齐。
  - 好处是释放了「预占」的块，提升并发请求数。

- **`deallocate` / preempt**：
  - 重置 `num_kv_computed = 0`，清空 `block_table`，与现有 `num_cached_tokens = 0` 一致。
  - **注意**：partial prefill 中途被 preempt 的 seq，其已写入 KV 的块虽然内容有效，但因缺少完整 hash 链无法被后续 prefix cache 复用（M1 阶段可接受，后续可优化）。

### 3.3 `ScheduledBatch` 切片语义

**决策：`context_lens[i]` 在 prefill 阶段固定为 `seq.num_prompt_tokens`（不变），offset 直接使用 `num_kv_computed`。**

- **`num_scheduled_tokens[i]`**：本步对第 `i` 条序列实际 forward 的 **token 数**（chunk 长度）。
- **`num_kv_computed[i]`**：新增字段，传入 batch，表示本步 forward **之前**该序列 KV 已就绪的长度。

- **`scheduled_tokens` 切片**：当前代码（`scheduler.py:223`）的偏移推导为 `offs = context_lens - num_rejected - num_scheduled_tokens`，这依赖于 "prefill 一次完成" 的假设。chunked prefill 下需改为：

  ```python
  # 对 prefill 序列：offset = num_kv_computed[i]
  # 对 decode 序列：保持原有逻辑不变
  for seq, num, kv_computed in zip(seqs.values(), num_scheduled_tokens, num_kv_computed_list):
      if seq.type == SequenceType.PREFILL:
          offset = kv_computed  # 从 prompt 的第 kv_computed 个 token 开始取
      else:
          offset = seq.num_tokens - num  # 现有 decode 逻辑
      self.scheduled_tokens[pos : pos + num] = seq.token_ids[offset : offset + num]
      pos += num
  ```

- **`context_lens[i]`**：
  - Prefill 阶段：固定为 `seq.num_prompt_tokens`。这保证 `prepare_prefill` 中 `seqlen_k = context_lens[i]` 正确反映因果 attention 的 K 总长（本步结束后 KV 覆盖到 prompt 末尾或 `num_kv_computed + chunk_size`，取决于是否最后 chunk）。
  - **修正**：实际上对中间 chunk，`seqlen_k` 应为 `num_kv_computed + chunk_size`（而非全部 prompt），否则 attention 会试图访问尚未写入的 KV 位置。因此 `context_lens[i]` 在 prefill 阶段应设为 `num_kv_computed[i] + num_scheduled_tokens[i]`。

  ```python
  # ScheduledBatch.__init__ 中：
  if seq.type == SequenceType.PREFILL:
      context_lens[i] = seq.num_kv_computed + num_scheduled_tokens[i]
  else:
      context_lens[i] = seq.num_tokens  # 现有 decode 逻辑
  ```

---

## 4. 调度器（`scheduler.py`）

### 4.1 Partial prefill 状态判定

**决策：不引入新的 `SequenceType`，通过已有字段判定 partial prefill 状态。**

判定逻辑：

```python
def is_partial_prefill(seq: Sequence) -> bool:
    """seq 已在 running 中，但 prompt 尚未全部写入 KV。"""
    return seq.num_kv_computed < seq.num_prompt_tokens
```

- Prefill 完成判定：`num_kv_computed + num_scheduled_tokens == num_prompt_tokens`
- 此时 `seq.type` 保持 `PREFILL` 直到最后一个 chunk 完成，再在 `postprocess` 中转为 `DECODE`。

### 4.2 `schedule()` 修改伪代码

```python
def schedule(self):
    scheduled_seqs = {}
    num_batched_tokens = 0
    num_scheduled_tokens = []

    # ---- Phase 1: 续算 running 中未完成 prefill 的 seq ----
    for seq in self.running:
        if seq.num_kv_computed >= seq.num_prompt_tokens:
            continue  # 已完成 prefill，等 decode 阶段处理
        remaining = seq.num_prompt_tokens - seq.num_kv_computed
        chunk = min(remaining, self.max_num_batched_tokens - num_batched_tokens)
        if chunk <= 0:
            break
        num_batched_tokens += chunk
        scheduled_seqs[seq.id] = seq
        num_scheduled_tokens.append(chunk)
        seq.type = SequenceType.PREFILL

    # ---- Phase 2: 从 waiting 拉新请求 ----
    while (self.waiting
           and len(scheduled_seqs) < self.max_num_seqs
           and (self.delay_factor <= 0 or self._passed_delay(time.time()))):
        seq = self.waiting[0]
        # 预算门：用 allocate 后的真实 token 数
        if not self.block_manager.can_allocate(seq):
            break
        self.block_manager.allocate(seq)
        # allocate 后 num_kv_computed 已含 prefix cache 命中数
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

    if scheduled_seqs:
        # 有 prefill 就整步 prefill（与现有行为一致）
        return ScheduledBatch(...)

    # ---- Phase 3: decode（现有逻辑不变）----
    ...
```

### 4.3 `delay_factor` 交互

- **partial prefill 中途不受 `delay_factor` 影响**：`_passed_delay` 仅控制新请求从 waiting 进入 running。已在 running 中的 partial prefill seq 必须优先续算，否则其已占用的 KV 块白白浪费。
- 实现方式：Phase 1（续算 running 中的 partial prefill）在 `delay_factor` 检查**之前**执行。

### 4.4 Preempt 处理

- 当 decode 阶段 `can_append` 失败需要 preempt 时，如果 preempt 的 seq 处于 partial prefill：
  - `deallocate(seq)` → 清空 `block_table`、重置 `num_kv_computed = 0`
  - seq 回到 `waiting` 队首，下次重新 allocate + 从头 prefill
  - 已写入的 KV 数据随 block 释放自然失效

### 4.5 `get_next_batch_info` 适配（DP 同步）

当前实现（`scheduler.py:587-599`）仅检查 `self.waiting`。chunked prefill 后，running 中可能有未完成 prefill 的 seq，需要适配：

```python
def get_next_batch_info(self) -> tuple[bool, int]:
    # 检查 running 中是否有 partial prefill
    for seq in self.running:
        if seq.num_kv_computed < seq.num_prompt_tokens:
            remaining = seq.num_prompt_tokens - seq.num_kv_computed
            chunk = min(remaining, self.max_num_batched_tokens)
            return (True, chunk)  # 仍是 prefill
    # 原有逻辑
    if self.waiting:
        seq = self.waiting[0]
        num_tokens = seq.num_prompt_tokens - seq.num_kv_computed
        return (True, num_tokens)
    elif self.running:
        return (False, len(self.running))
    else:
        return (False, 0)
```

这保证 DP 模式下，有 partial prefill 的 rank 正确广播 prefill 状态，其他 rank 执行 `dummy_prefill_execution` 对齐。

### 4.6 `CacheStats` 去重

同一请求的多次 chunk 不应重复记录 cache 统计。解决方式：**仅在 allocate 后记录一次**（即从 waiting 进入 running 时），后续 chunk 续算不触发 `cache_stats.update()`。当前代码已在 allocate 后调用，无需改动，但需 **确认 Phase 1（续算 partial prefill）路径不走 allocate**。

### 4.7 交付物

- 调度单测（可在 mock block_manager 下测）：
  - 单请求超长 prompt（> `max_num_batched_tokens`）多步完成
  - 多请求交错：1 条长 + N 条短
  - budget 边界：prompt = budget 整数倍、budget+1、budget-1
  - prefix cache 命中 + chunk 组合
  - partial prefill 中途 preempt 后恢复
  - DP 下 `get_next_batch_info` 返回值正确性

---

## 5. Attention 元数据（`CommonAttentionBuilder.prepare_prefill`）

### 5.1 数学关系（与现有 prefix 路径一致）

对每条序列（本步）：

- `seqlen_q` = 本步新算 token 数（chunk 长度）。
- `kv_prefix_len` = 本步 forward **之前** KV 已就绪长度（prefix cache + 已完成 chunk）。
- `seqlen_k` = `kv_prefix_len + seqlen_q`（因果 attention 下 K 总长）。
- **positions**：`range(kv_prefix_len, kv_prefix_len + seqlen_q)`。
- **slot_mapping**：仅映射 **本步新写** 的 KV slot（与现有 prefix 逻辑相同，把 `cached_seqlen` 替换为 `kv_prefix_len`）。

### 5.2 `has_cached` / gather

当 `kv_prefix_len > 0` 时，走现有 **`has_cached` + `_gather_prefix_and_concat_kv`**（或等价路径），从 paged cache 拉前缀 K/V，再与当前步新写的 K/V 拼接，供 Flash / 下游使用。

### 5.3 特殊模型

- **Sliding window / Mamba / MLA**：在 `docs/prefill_cache_mla_todo.md` 与各自 backend 上增加 **「partial prefill 进度」** 与 window 边界的测试说明。
- **Plugin / vLLM 路径**：若启用，需单独对齐 metadata（不在本文展开）。

---

## 6. ModelRunner：`forward` / `run_model` / `postprocess`（B 方案）

### 6.1 `run_model`

- 输入：仍为 **本 chunk** 的 `input_ids` / `positions`（由 `prepare_input_ids` + `prepare_inputs` 提供）。
- 分支：

  ```text
  if is_prefill and is_partial_prefill_chunk:
      hidden_states = self.model(input_ids, positions)
      logits = None  # 或占位，但下游不得使用
  elif is_prefill and is_final_prefill_chunk:
      hidden_states = self.model(...)
      logits = self.model.compute_logits(select_rows(hidden_states))  # 建议仅最后位置，见下
  else:
      # 现有 decode / graph 路径
  ```

- **最后 prefill chunk**：若无 logprobs，通常只需 **每个序列 1 行** hidden 进入 `compute_logits`（最后一个 prompt 位置的表示），与 vLLM `logits_indices = query_start_loc[1:] - 1` 思想一致，减少 lm_head 体量。

### 6.2 `postprocess` — 中间 chunk 处理

当 `logits is None`（中间 chunk）：**不调用** `sampler`；但需正确维护 deferred output 流水线。

**具体实现**：

```python
def postprocess(self, batch, logits, temperatures, top_ks, top_ps, all_greedy, hidden_states):
    forward_context = get_forward_context()
    is_partial = forward_context.context.is_partial_prefill

    if is_partial:
        # 中间 chunk：不采样，但需维护 deferred 状态
        # 传入空的 sampled_tokens 让 prepare_sampled_ids 正常推进 prev_batch
        dummy_sampled = torch.empty(0, dtype=torch.int32, device=self.device)
        self.forward_done_event.record()
        req_ids_out, token_ids_out = self.tokenID_processor.prepare_sampled_ids(
            batch, dummy_sampled, self.forward_done_event
        )
        return ScheduledBatchOutput(
            req_ids=[],          # 中间 chunk 无输出
            token_ids=[],
            draft_token_ids=None,
            is_deferred_out=self.tokenID_processor.is_deferred_out,
            num_rejected=np.zeros(0, dtype=np.int32),
            num_bonus=np.zeros(0, dtype=np.int32),
        )

    # 最后 chunk / decode：走现有逻辑
    ...
```

**关键**：中间 chunk 仍然经过 `prepare_sampled_ids()` 更新 `prev_batch`，保证下一步（可能是最后 chunk 或 decode）的 deferred 输出逻辑不断裂。

### 6.3 TP broadcast 死锁防护

当前代码（`model_runner.py:1635-1636`）在 `postprocess` 中对 `sampled_tokens` 做 TP broadcast。中间 chunk 跳过采样时，各 rank 必须走一致的分支。

**解决方案**：在 `prepare_inputs` 中设置 `is_partial_prefill` flag 到 forward context：

```python
# prepare_inputs 中：
is_partial = is_prefill and any(
    seq.num_kv_computed + batch.num_scheduled_tokens[i] < seq.num_prompt_tokens
    for i, seq in enumerate(seqs)
)
context = Context(
    ...,
    is_partial_prefill=is_partial,  # 新增字段
)
```

所有 rank 基于 `ScheduledBatch`（各 rank 一致）计算此 flag，保证一致进入/跳过 broadcast 分支。

### 6.4 `prepare_sample` / `prepare_model`

- 中间 chunk 可跳过 `prepare_sample` 中不必要的 GPU 拷贝（temperature/top-k/p），或保留默认以简化分支（性能次要）。
- **建议 M1-M3 阶段保留调用**以减少分支复杂度，M4 性能优化时再按需跳过。

### 6.5 Deferred output 与 MTP

- **无 spec 的 prefill**：现状 `spec_decode_metadata is None`；中间 chunk 不采样与 **deferred**「本步不对外吐字」一致。
- **存在 `drafter` + deferred**：prefill 阶段若仍不跑 spec 元数据，**不在中间 chunk 调用** `propose_draft_token_ids`；在 **第一次 decode 步**再按现有逻辑 proposal。若产品要求 prefill 结束当步即 proposal，则仅在 **最后 prefill chunk** 在已有 `next_token_ids` 后调用（与 B 不冲突）。

### 6.6 Warmup 与 Dummy Execution 适配

`warmup_model()`、`dummy_execution()`、`dummy_prefill_execution()` 需适配 `is_partial_prefill` flag：

- **`warmup_model`**：走完整 prefill 路径（非 partial），无需改动。
- **`dummy_execution`**：decode 路径，无需改动。
- **`dummy_prefill_execution`**：DP 同步用，需在 forward context 中设 `is_partial_prefill = False`（dummy 走完整路径），无需改动。
- **新增考虑**：当 DP 中某 rank 在做 partial prefill 而另一 rank 做 dummy prefill 时，两边 `is_partial_prefill` 值不同。由于 dummy 不产生真实输出、不走 TP broadcast，不会死锁。但需**回归测试确认**。

---

## 7. CUDA Graph（`capture_cudagraph`）

- **Prefill**：当前为 **eager**；chunked prefill 的中间步形状为 **变长 token**，与 **decode graph**（按 `graph_bs` × `max_q_len`）分离，一般 **不要求**为每个 chunk 长度单独 capture。
- **注意**：若未来为固定 chunk 大小 capture prefill 子图，需单独设计 **padding 与有效 token 掩码**，本文默认 **prefill 继续 eager**。

---

## 8. 引擎层与 API

### 8.1 Scheduler → ModelRunner 接口

- `ScheduledBatch` 新增 `num_kv_computed` 列表（每个 seq 的当前 KV 进度）。
- `is_partial_prefill` 可由 `num_kv_computed[i] + num_scheduled_tokens[i] < seq.num_prompt_tokens` **推导**，无需额外字段。ModelRunner 在 `prepare_inputs` 中计算并写入 forward context。

### 8.2 `scheduler.postprocess` 适配

`scheduler.postprocess` 消费 `ScheduledBatchOutput` 更新 `Sequence` 时，需识别中间 chunk：

```python
for seq in self.running:
    idx = fwd_output.get_idx(seq.id)
    if idx is None:
        continue

    # 更新 KV 进度
    seq.num_kv_computed += batch.num_scheduled_tokens[idx]

    if seq.num_kv_computed < seq.num_prompt_tokens:
        # 中间 chunk：无新 token，不做 append_token / 不检查 EOS / 不结束请求
        continue

    # 最后 chunk 或 decode：走现有逻辑
    token_ids = fwd_output.token_ids[idx]
    ...
```

**关键**：中间 chunk 时 `fwd_output.token_ids` 为空列表，`req_ids_out` 也为空。`postprocess` 必须跳过这些 seq 的 token 追加和终止检查。

### 8.3 DP 同步（`DPEngineCoreProc`）

`_sync_dp_state` 已通过 `get_next_batch_info` 获取本 rank 的 prefill/decode 状态。§4.5 的适配保证了 partial prefill 被正确识别为 prefill 状态。

额外注意点：
- 当一个 rank 在做 partial prefill 续算（Phase 1），另一个 rank 的 waiting 为空且 running 中无 partial prefill（只有 decode seq），则后者会执行 `dummy_prefill_execution` 对齐。
- `dummy_prefill_execution` 的 `num_tokens` 参数来自 `global_max_tokens`（各 rank 取 max），这与 chunked prefill 的 chunk 大小一致，不需要额外适配。

### 8.4 `_process_engine_step` 无需改动

`_process_engine_step` 调用 `scheduler.schedule()` → `runner_mgr.call_func("forward")` → `scheduler.postprocess()`，这个流程不变。scheduler 内部区分 partial/final prefill，engine_core 无需感知。

---

## 9. 测试与验收

### 9.1 正确性（greedy 比特级一致）

| 编号 | 场景 | 期望 |
|------|------|------|
| C1 | 长 prompt（= `budget * 3 + 1`）greedy 生成 | 输出与非 chunked 基线完全一致 |
| C2 | 短 prompt（< `budget`）greedy 生成 | 输出不变（回归） |
| C3 | prompt = `budget` 整数倍 | 最后 chunk 恰好 0 余量，正确触发 final chunk |
| C4 | prompt = `budget + 1` | 拆为 2 个 chunk（budget + 1），输出一致 |
| C5 | 多请求：1 条长（3x budget）+ 2 条短（< budget） | 所有请求输出与串行基线一致 |

### 9.2 KV 一致性

| 编号 | 场景 | 验证 |
|------|------|------|
| K1 | 多步 chunk 后 `block_table` 连续 | 对比全量 prefill 的 block_table |
| K2 | 最终 hidden_states（最后 chunk 最后位置） | 与全量 prefill 完全一致 |
| K3 | prefix cache 全命中 + chunk | `num_kv_computed` 初始 = 命中数，后续 chunk 正确续算 |
| K4 | prefix cache 部分命中 + chunk | 前 N 块命中后剩余 token 正确分 chunk |

### 9.3 性能验证（B 方案）

| 编号 | 验证 | 方法 |
|------|------|------|
| P1 | 中间步无 `compute_logits` / sampler 调用 | torch profiler trace 检查 |
| P2 | 最后步仅 1 行 hidden 进入 `compute_logits`（M4） | profiler + shape 断言 |

### 9.4 边界与异常

| 编号 | 场景 | 验证 |
|------|------|------|
| E1 | `max_num_seqs` 打满 + 新长请求到达 | 新请求在 waiting 等待，不影响 running |
| E2 | partial prefill 中途 preempt | seq 回到 waiting，重新 allocate + 从头 prefill，输出一致 |
| E3 | partial prefill 中途新短请求到达 | 短请求在 waiting 排队，不打断当前 chunk |
| E4 | TP > 1，中间 chunk | 各 rank 一致跳过 broadcast，无死锁 |
| E5 | deferred output on + 中间 chunk | prev_batch 正确推进，最终输出不丢失 |
| E6 | deferred output off + 中间 chunk | 直接返回空输出，不触发采样 |
| E7 | DP 模式：一个 rank partial prefill，另一 rank idle | dummy_prefill 正确对齐 |
| E8 | prompt 长度 = 1（极短） | 不走 chunk，直接完成 |
| E9 | prompt 长度 = `max_model_len` | 正确分 chunk 到最大上下文 |

### 9.5 回归

- 现有全部单测通过
- serving 路径（`openai_server`）短 prompt 端到端无行为变化
- MTP/spec decode 路径不受影响（prefill 阶段不走 spec）

---

## 10. 建议开发顺序（里程碑）

| 阶段 | 内容 | 验收标准 |
|------|------|----------|
| **M0** | **Prototype 验证**：仅改 scheduler + `ScheduledBatch` 的切片逻辑（不改 ModelRunner），保留全量 `compute_logits` + 全量采样。目的是最早验证 `offset`、`context_lens`、`num_kv_computed` 语义正确性。 | 长 prompt（> budget）greedy 输出与基线一致（C1-C4） |
| M1 | `Sequence.num_kv_computed` 字段 + `ScheduledBatch` 正确切片 + scheduler Phase 1/2 chunk 调度 + `postprocess` 中间 chunk 跳过（仍保留全量 logits 以对比正确性） | C1-C5, K1-K2 通过 |
| M2 | `prepare_prefill` 使用 `num_kv_computed` 作为 `kv_prefix_len`；打通 MHA gather + 多步 KV；prefix cache + chunk 组合 | K3-K4 通过 |
| M3 | `run_model` / `postprocess` 实施 **B 分支**（中间 chunk 不算 logits）；TP broadcast 一致性；deferred output 流水线维护 | P1, E4-E6 通过 |
| M4 | 最后 chunk **单行 `compute_logits`** 优化（`logits_indices = cu_seqlens_q[1:] - 1`）；性能测试 | P2 通过，profiler 确认 |
| M5 |（可选）prefill/decode **mixed batch**；`delay_factor` 联调；增量 block allocate | E1-E3 回归 |

---

## 11. 风险与未决项

### 已解决的设计决策

| 原风险 | 决策 | 见 |
|--------|------|-----|
| 双字段语义混乱 | 单字段 `num_kv_computed`，`num_cached_tokens` 仅保留用于统计 | §3.1 |
| BlockManager allocate 策略 | M1 全量 allocate，后续可选增量 | §3.2 |
| `context_lens` 语义不明 | Prefill 阶段 = `num_kv_computed + chunk_size` | §3.3 |
| Scheduler 状态判定 | 不引入新 SequenceType，用 `num_kv_computed < num_prompt_tokens` | §4.1 |

### 仍需关注的风险

- **Preempt 后 KV 浪费**：partial prefill 被 preempt 时，已写入的 KV 块内容有效但无法被 prefix cache 复用（缺少完整 hash 链）。M1 阶段可接受，后续需评估是否支持「部分 KV 保留」。
- **首 token 时间指标**：TTFT 应从 **首包可见输出** 定义，中间 chunk 不应对外报 completion。chunked prefill 会拉长 TTFT，需在监控中区分 "prefill 总时间" 和 "单 chunk 时间"。
- **DP dummy prefill 精度**：`dummy_prefill_execution` 使用全零 input_ids，其 activation pattern 可能与真实 prefill 不同。当前已有此问题（非 chunked 也用 dummy），chunked 不引入新风险。
- **Sliding window attention**：当 chunk 边界落在 sliding window 内时，`kv_prefix_len` 计算需考虑 window 截断。需在 MLA / sliding window backend 上单独测试。
- **Encoder-decoder / 多模态**：占位符与 encoder budget 需单独设计（若 ATOM 后续支持）。
- **文档同步**：落地后应更新 `docs/scheduling_kv_cache_guide.md` 中的 prefill 描述。

---

## 12. 参考文献（仓库内）

- `docs/scheduling_kv_cache_guide.md` — 调度与 KV 基础
- `docs/compilation_cudagraph_guide.md` — CUDA Graph 与 eager 边界
- `docs/prefill_cache_mla_todo.md` — MLA / cache 相关待办

---

*文档版本：与「中间 chunk B 方案」对齐；实现过程中请以代码审查与单测为准。*
