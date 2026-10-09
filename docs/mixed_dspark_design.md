# Mixed prefill+decode × DSpark 设计方案

状态：设计稿（2026-09-30）· 基线：`fpz/mixed_mla_dispatch_v4` @ `efa9d3887`（已 rebase 到 main `e59d42233`）

## 1. 目标与范围

让 `--enable-mixed-prefill-decode` 与 `--method dspark` 同时开启：一个 mixed step 里，prefill 行正常跑 chunk，decode 行做 DSpark 的 1+K verify，step 结束后对**所有**行照常 propose 下一轮 draft。

**Phase 1 范围（本方案）**

| 维度 | 支持 | 不支持（仍在启动时拒绝） |
|---|---|---|
| 模型族 | V4（`Family.V4`，DeepSeek-V4-Pro-DSpark） | dense MLA（R1）+ 任何 spec；Kimi-K3（本就不在 mixed 白名单） |
| spec 方法 | `dspark` | `mtp` / `eagle`（数据流相同，Phase 2 顺带放开） |
| TBO | `--enable-tbo` + `ATOM_TBO_MIXED=1`（见 §4.8） | DCP + decode TBO + spec（main 已拒绝，不动） |
| DSpark ragged / q-bucket | 纯 decode step 照常生效 | mixed step 上自动退化为全宽 K+1（现有代码已如此，见 §4.6） |
| PP | — | PP>1（reserve 口径不一致，见 §4.1） |

dense MLA 不做的原因：`aiter_mla.py` 的 `prepare_mixed` 在 decode 段整体写死 1 token/行（`max_q_d=1`、`slot_d`、`positions_d = ctx-1`、无 `num_rejected` 回滚、persistent worker buffer 按 q=1 建），要重写成与 `prepare_decode` 等价的 spec 形状，工作量独立，放 Phase 2。V4 的 decode 段是直接调用未修改的 `prepare_decode`（经 `_MixedDecodeView`），本身已是宽度无关的，这是选 V4 先做的理由。

## 2. 现状：三道拒绝

1. `config.py` `_validate_mixed_prefill_decode`：`speculative_config is not None` 直接 `ValueError`。给出的理由之一——"reserve 按 `mtp_k` 算，decode 循环按 `spec_width` 花"——**只在 PP>1 时成立**（`spec_width = mtp_k if spec_decode_local else 0`，本地 verify 时二者相等）。
2. `model_runner.py` `prepare_input_ids` mixed 分支：`if self.use_spec: raise`。
3. `model_runner.py` `prepare_inputs`：`if is_mixed and hasattr(self, "drafter"): raise`。

## 3. 核心观察：logit 行空间已经是 verify 布局

LM head 的 mixed gather（`embed_head.py`）已经产出：

```
logits 行 = [ 每个 prefill seq 的最后一个 token (n_p_seqs 行) | 每个 decode token (n_d_tokens 行) ]
```

这正是"把每个 prefill 行当成 **0 个 draft、采 1 个样本** 的 spec 行"时的 verify 布局。而 spec 的现有机器全部以"每行采样数"为单位工作，0-draft 行是合法输入：

- `prepare_spec_decode_indices`：`num_draft = clip(lens-1, 0, mtp_k)`，`bonus = cu[1:]-1`，`target` 取每段前 `num_draft` 行——0-draft 行只贡献一个 bonus index。
- `rejection_sample`：输出 `[bs, K+1]`，0-draft 行 `num_bonus = 0`，第 0 列就是 bonus 采样结果。
- `postprocess` 的 `gather(sampled.view(bs,-1), next_token_locs)`：0-draft 行取第 0 列，正确。

所以方案的主线是：**在 logit 行空间里定义一个统一的"采样行"长度向量 `sampled_lens = [1]*n_p_seqs ++ decode_lens`，所有 spec 索引都从它导出**；再补齐 drafter 读取的两个 carrier 字段。不新造第二条 verify 路径。

## 4. 设计

### 4.1 配置与调度

- **Validator**：把 spec 那一条拆成细粒度拒绝——
  - 允许：`family == V4 and method == "dspark" and pp_size == 1`（TBO 开关不限，见 §4.8）
  - 拒绝：dense MLA + 任何 spec；`mtp`/`eagle`（Phase 1）；PP>1。理由文案逐条写明。
- **Decode reserve**（`scheduler.py` decode-first 预留）：`n_decode_inflight * (self.mtp_k + 1)` → `* (self.spec_width + 1)`，与 decode 循环真实花费同口径。PP 被 validator 挡住，这里只是消除 validator 里那条理由的根源。
- **Reserve 饱和风险**：K=7 时每个 decode 行预留 8 token。`max_num_seqs=256` 满载时 reserve = 2048，若 `--max-num-batched-tokens 2048` 则 `prefill_budget = 0`，mixed 永远不形成、prefill 饿死。Phase 1 行为：**不改调度策略**，但在 validator 里当 `max_num_batched_tokens <= max_num_seqs * (K+1)` 时打 warning 并说明原因；是否加 prefill 保底预算作为 §7 的开放问题。
- 其余调度逻辑无需改：decode 行已按 `spec_width+1` 排进 `num_scheduled_tokens`，`scheduled_spec_decode_tokens` 已是 `[bs, K]`（prefill 行为 0 行），`_settle_prefill_chunks` 只作用于 prefill 前缀，DSpark 的 `drafter_needs_next_token=False` 所以 `next_token_ids=None`。

### 4.2 Token 输入（`prepare_input_ids` mixed 分支）

去掉 `use_spec` 的 raise，把 decode 区的处理对齐纯 decode 的 deferred spec 路径：

1. 整段先从 `scheduled_tokens` staging（现状）。
2. **rejected/bonus remap**：`self.num_rejected` / `self.num_bonus` 生成**全批长度**数组，prefill 行填 0，decode 行按 `prev_batch.req_ids` 把 `prev_rejected_num` / `prev_bonus_num` remap 过来。映射只在 decode 行上建（沿用现有 `prev_id_to_idx` 的做法），**不能**用 `get_token_locations`：chunked prefill 的中间 chunk 也在 prev_batch 里，会被错当成 deferred 行。
3. 新入 decode 的行（不在 prev_batch）：按纯 decode 路径那样把 `scheduled_spec_decode_tokens[n_p + i, :len-1]` stage 到 draft 列。
4. `fill_deferred_decode_ids` 调用：`draft_token_ids` 从 `None` 改为 `self.draft_token_ids if self.pre_num_decode_token_per_seq > 1 else None`，`max_tokens_per_seq = int(decode_lens.max())`。`cu` 仍是 `publish_cu_seqlens_q` 发布的 decode 局部 spans（已是宽度无关）。
5. `decode_src` 与 `input_ids` 仍在同一次 group publish 里，保持"每 buffer 每 epoch 一次"。

### 4.3 Spec 元数据（`prepare_inputs` / `prepare_model`）

去掉 mixed + drafter 的 raise，新增 mixed 分支：

```python
n_p, n_p_tok = batch.total_seqs_num_prefill, batch.total_tokens_num_prefill
d_lens = batch.num_scheduled_tokens[n_p:]
sampled_lens = np.concatenate([np.ones(n_p, np.int32), d_lens])   # logit 行空间
cu_sampled = np.cumsum(sampled_lens)                                # 即 cu_num_sampled_tokens
shift = n_p_tok - n_p                                               # logit 行 -> token 行
spec_decode_metadata = drafter.calc_spec_decode_metadata(
    sampled_lens, cu_sampled, input_ids[shift:], prepared_indices=...)
```

- `draft_token_ids = input_ids[1:][target_logits_indices]` 在纯 decode 里靠"logit 行 = token 行"成立。mixed 下 decode token j 的 logit 行是 `n_p + j`、token 行是 `n_p_tok + j`，差一个常数 `shift`，所以只需把 `input_ids[shift:]` 传进去，`calc_spec_decode_metadata` 本身不用改。（`shift >= 0` 恒成立。）
- `prepare_spec_decode_indices` 的发布必须并入 token 输入那次 packed publish（`token_inputs` 组），与纯 decode 在 `prepare_model` 里的做法一致；不能在 mixed 分支里单独 publish 一次 `spec_decode` 组。
- 不引入 `decode_spans` 的 mixed 变体：`decode_spans` 语义保持"纯 decode step"，mixed 分支显式切片。

### 4.4 采样后处理（`postprocess`）

spec 路径基本原样工作（§3）。只改两处，**让 prefill 行对外表现与纯 prefill step 完全一致**：

- `num_reject_tokens = mtp_k - num_bonus` 对 prefill 行会得到 `mtp_k`，而纯 prefill step 给的是 0（`default_num_rejected_tokens`）。**这是正确性问题，不是统计口径问题**：它经 `send_mtp_status_to_cpu_async` → 下一步 `prev_rejected_num` remap，落到这个刚转为 decode 的请求上，V4 `prepare_decode` 会把它的 ctx 回滚 K 个位置——positions 整体错位，KV 写进错误的 slot。次要影响：scheduler 的接受率统计门槛是 `(num_new_token + num_rejected) > 1`，不归零会把 prefill 行记成"K 个 draft 全拒"，拉低接受率，而接受率正是 §6 的验证信号。改为：
  ```python
  num_reject_tokens = torch.where(is_prefill_row, 0, mtp_k - num_bonus)
  ```
  `is_prefill_row` = `arange(bs) < n_p`，设备上构造，无同步。
- `num_bonus_tokens` 对 prefill 行保持 0（= "取第 0 列"），`next_token_locs` 同理，无需改。

### 4.5 Drafter 读到的 carrier 字段

DSpark 在 target forward 之后读 **top-level** `attn_metadata`，而 mixed carrier 目前只有 prefill 段的 `cu_seqlens_q`、没有 `state_slot_out`：

| 读者 | 读什么 | mixed 现状 |
|---|---|---|
| `DSparkProposer.compute_draft_kv` → V4 `write_context_kv` | `cu_seqlens_q[:B+1]`（B=`context.scheduled_bs`=全批），`state_slot_out[:B]`，`positions` | cu 只有 prefill 段；`state_slot_out=None` |
| `Drafter.prepare_inputs`（anchor） | `cu_seqlens_q[:bs+1]`（bs=全批） | 同上 |
| DSpark block pass（`index_buffers.build`、bf16 window gather） | `state_slot_out[:running_bs]` | `None` |
| LM head mixed gather | `cu_seqlens_q[1:n_p+1]` | 只用前缀 |

**方案 A（推荐）**：在 V4 `prepare_mixed` 末尾，用设备端拼接把 carrier 补成全批视图，不做任何新的 H2D：

```python
merged.cu_seqlens_q = torch.cat([prefill_meta.cu_seqlens_q[: n_p + 1],
                                 decode_meta.cu_seqlens_q[1 : n_d + 1] + n_p_tokens])
merged.state_slot_out = _pad_to(running_bs, torch.cat([prefill_meta.state_slot_out[:n_p],
                                                        decode_meta.state_slot_out[:n_d]]))
```

- 全批 cu 的前 `n_p+1` 项就是 prefill cu，所以 LM head 不受影响。
- `running_bs` 来自 `ForwardMode.decide`：DP 下会被量化到 capture ladder，**可能大于 `total_seqs_num`**。pad 尾填 0，与 `prepare_decode` 的约定一致；`prepare_block` 的 `mask_pad_tail` 已负责屏蔽 pad 行的 KV 写入。
- **读者审计（已核对源码）**：
  - target forward：`deepseek_v4.py` 里对 carrier `state_slot_out` / `cu_seqlens_q` 的读取，一处在 `if not is_mixed` 分支内（整批 compressor launch），其余都在 `_segment_forward_context` 内——那里 `fc.attn_metadata` 已被换成段自己的元数据。mixed step 上 target 从不读 carrier 的这两个字段。
  - LM head：只读前缀 `cu_seqlens_q[1:n_p+1]`。
  - DSpark block pass：只读 `state_slot_out[:B]`；`prepare_block` / `mask_pad_tail` 读的是 builder 的静态 `row_ids` arange，不碰 `forward_vars`。
  - TBO mixed 重建：`split_attn_metadata` / `_build_ubatch_mixed_metadata` 读 `prefill_attn_metadata` / `ub_pref`，不读 carrier。
  - 其余命中都在 dense MLA/MHA（`attention_mla.py` / `attention_mha.py`），V4 不走。
  - 结论：方案 A 成立；方案 B 降为不需要的备选。

方案 B（备选）：保持 carrier 语义不变，新增 `cu_seqlens_q_all` / `state_slot_out_all`，让 `write_context_kv` 与 `Drafter.prepare_inputs` 优先读 `_all`。改动面更散（`deepseek_v4_dspark.py` 的 `write_context_kv` 是 eager 方法、不在 `@support_torch_compile` 的 `_DSparkInner` 里，可以改），但不改变已有读者看到的语义。读者审计（上文）已确认 A 成立，B 仅作记录。

**Anchor 位置**（随 main #2479 更新）：verify 过的 step 上 anchor 由 rejection verdict 直接给出，公式是 `bonus_logits_indices − num_draft + num_accepted`，结果是 **logit 行号**。mixed 下 `ModelRunner._settle_mixed_verdict` 把它换算到 token 行：prefill 行改为 `cu[i+1]−1`（chunk 最后一个 token），decode 行加上 `n_p_tokens − n_p`；同时把 prefill 行的 reject 归零（§4.4）。没有 verify 的 step 仍走 `Drafter.prepare_inputs`（取每段最后一行），配合 carrier 的全批 `cu` 本身就对。

**propose 必须覆盖全批，包括 prefill 行**：下一步的 `fill_deferred_decode_ids` 按**上一步的行号**（`src`）从 `prev_token_ids` / `draft_token_ids` 取 anchor 和 draft。一个在本 mixed step 走完最后一个 prefill chunk 的请求，下一步作为 decode 行时的 draft 就来自它在本步 prefill 行上的 propose。只给 decode 行 propose 会让这些请求的首个 decode step 没有 draft，并迫使 draft 缓冲按行号重映射。（这也是"mixed 时只让 decode 行开 DSpark"不更省事的原因之一。）

**prefill 中间 chunk**：会得到垃圾 draft，scheduler 丢弃——与今天纯 prefill+spec 的行为一致，不需处理。

### 4.6 V4 decode 段（`prepare_mixed` / `_MixedDecodeView` / `prepare_decode`）

- **`num_rejected` 回滚**：`prepare_decode` 直接读 `tokenID_processor.num_rejected`（全批数组）做 `ctx -= num_rejected`。经 view 调用时必须切到 decode 行：view 增加 `row_offset = n_p`，`prepare_decode` 用 `num_rejected[row_offset : row_offset + scheduled_bs]`。**不能**靠 `__getattr__` 兜底——这个数组不在 batch 上，guard 抓不到。
- `_MixedDecodeView` 补切 `scheduled_spec_decode_tokens`、`num_rejected`、`num_bonus`（虽然当前 V4 decode 路径未必读，但 view 的原则是"每行字段要么切要么 raise"）；`scheduled_tokens`（长度 = 总 token 数）目前会**静默**穿透，guard 只查长度 = 总 seq 数的字段——改成显式切 `[n_p_tokens:]` 或显式 raise。
- `prepare_mixed` 里 `cu[n_d] == n_d * decode_max_q` 的检查：mixed step 上 ragged/q-bucket 都不生效，decode 宽度恒为 K+1，检查仍成立。依据是 `_dspark_apply_q_bucket` 开头的 `if batch.total_tokens_num_prefill > 0: return None`——ragged 分支在它**之后**分派，所以一个守卫同时关掉两者。（"ragged 要求 prev/cur req_ids 相同、mixed 组成必变"**不成立**：一个跨多步的 chunked prefill 配同一组 decode 行，连续两个 mixed step 的 req_ids 完全相同。）检查旁注明依赖的是这个守卫，并加单测锁住"组成不变的连续 mixed step 也不 shrink"。
- 注释/docstring 里 "1 token per decode seq, no MTP in mixed" 同步更新（name-matches-function）。

### 4.7 不需要改的部分

- LM head gather、rejection sampler、`prepare_sampled_ids` 的逐行 zip、`_attn_mixed` 的 token 切分（全部按 token 数，与宽度无关）。
- `prepare_cu_seqlens_q` 的 mixed 分支（decode 局部 spans，已宽度无关；只更新 docstring 里 "1 token per row"）。
- `@support_torch_compile` 的 `_DSparkInner`：本方案不触碰。

### 4.8 TBO（`--enable-tbo` + `ATOM_TBO_MIXED=1`）

结论：**TBO 不是难点**——它只作用于 target forward 内部，verify 和 draft 都在 TBO 之外。

**已经就绪的部分（已核对源码）**
- `UBatchWrapper` 在 worker 线程里跑各 ubatch，结束后恢复父 forward context；`compute_logits` / `postprocess` / `propose` / `compute_draft_kv` 都在父 context 上执行，看到的是全批元数据。ubatch context 里的 `spec_decode_metadata=None` 因此无影响。
- drafter 的 aux hidden 捕获已按 `ubatch_token_offset` 写入各 ubatch 的 token 区间（`drafter.py` `_make_aux_hook`）。
- DP 同步已把 TBO 字段和 DSpark 的 `max_seqlen_q` 打进同一次 packed all_gather（`sync_dp_metadata`）。
- V4 decode TBO 的 per-ubatch 元数据（`_prepare_ubatch_decode`）本身就按 spec 形状建：`max_seqlen_q>1`、逐请求精确 `cu`、`compress_plans(max_q_len=...)`。
- mixed 的 TBO 切分把切点强制放在 prefill 区内，最后一个 ubatch = prefill 尾 + 全部 decode 行，其 decode 段**按引用复用父的 `decode_attn_metadata`**——所以 §4.6 修好的 spec 形状 decode 元数据会自动带过去，不需要额外改。

**需要处理 / 验证的点**
1. **前置：纯 TBO + DSpark（不开 mixed）从未在 CI 验证过**。main 只在 DCP 下拒绝 decode TBO + spec，CI 里没有 TBO+MTP/DSpark 的配置。必须先把它跑通作为对照组，否则 mixed 出问题无法归因。
2. **切分门槛会更难满足（性能，不影响正确性）**：`split_mixed_token_midpoint` 要求 `total_tokens // 2 < num_prefill_tokens`，即 prefill token 过半才切。decode 行从 1 变 K+1=8 个 token，decode 占比上升 8 倍，mixed step 被 `gate:refused` 的比例会明显上升，TBO 在 mixed 上的收益变小。强制暴露配置（小 `mnbt`）下几乎不会切。若要在 decode 占多数时也切，需要支持"纯 decode ubatch 挂在 is_prefill 父 context 下"，这是 TBO 目前表达不了的，工作量独立——先测 `gate:split` 比例再决定。
3. **aux 捕获与 ubatch pad 行（风险已降低）**。hook 写 `tensor.shape[0]` 行到 `off` 起始处；若某 ubatch 的层输出带 DP pad 行，ub0 的 pad 尾会写进 ub1 的头部。本方案只用 prefill TBO：`pad_for_all_gather` 在 MoE 内部补齐、gather 后去 pad，层输出是真实 token 数，且 B' 对照（prefill TBO + DSpark + DP）接受率与无 TBO 在同一水平。decode TBO 才可能带 pad 行，已不在范围内。
4. DSpark 的 block pass 本身不做 TBO（在父 context 上跑全批），纯性能项，不在本期。

**前置对照结果（2026-09-30，HIP 4-7，tp4 + dp-attention）**

本机没有 V4-Pro-DSpark checkpoint（`/data/DeepSeek-V4-Pro` 不含 draft 权重），改用同架构的 `/mnt/DeepSeek-V4-Flash-DSpark`（`dspark_block_size=5`，故 `--num-speculative-tokens 5`），其余参数同 CI DSpark 配置；GSM8K 3-shot，conc 1000，`ATOM_TBO_PREFILL_MIN_TOKENS=1024`。分数与 CI 的 V4-Pro 基线不可比，只做组间对比。

| 组 | TBO | GSM8K | 接受率（4 rank） | 结果 |
|---|---|---|---|---|
| A | 关 | 0.9166 ± 0.0076 | 48.2–54.6%（均 50.6%） | 通过 |
| B | `--enable-tbo all` | — | — | **首批请求即崩溃** |
| B' | `--enable-tbo`（仅 prefill） | 0.9136 ± 0.0077 | 47.1–48.7%（均 48.0%） | 通过；py-spy 确认 4 个 rank 均起了 `tbo-ub-0/1` 线程 |

- **B 的崩溃（decode TBO + DP + PIECEWISE）**：`pad_for_all_gather` 断言 `MoE was handed 588 rows on a decode step expecting 666`。decode ubatch 的 `running_tokens` 按"padding 后的 ubatch 请求数 × max_q"算（666 = 111×6），但 PIECEWISE 下 decode TBO 走 eager `_run_ubatches`，每个 ubatch 的输入只有真实 token（588 = 98×6），没被补齐。未验证是否与 spec 相关——`max_q=1` 时只要 ubatch pad 行数 > 0 同样会不等，推测是 decode TBO + 非 FULL cudagraph + DP 的通用问题。**不在本方案范围**，但若生产要开 decode TBO 必须先修。
- B' 与 A 分数在 1σ 内；接受率低约 2.6 个点，单次运行、未做重复，暂不下结论。
- mixed 的 TBO 切分走的是 prefill TBO 门控（`enable_tbo` + `ATOM_TBO_MIXED`），B' 通过即满足本方案的前置条件。

### 4.9 实现落点（2026-09-30）

| 设计 | 实现 |
|---|---|
| §4.1 reserve 同口径 | `Scheduler.decode_spec_width`（decode 循环、reserve、`num_spec_step`、placeholder 宽度共用） |
| §4.3 logit 行空间索引 | `ModelRunner._verify_spans(batch) -> (sampled_lens, cu_end, shift)`，packed / direct 两处共用 |
| §4.4/§4.5 prefill 行 reject 归零 + verdict anchor 换算到 token 行 | `ModelRunner._settle_mixed_verdict` |
| §4.5 carrier 全批字段 | `deepseek_v4_attn._mixed_carrier_spans` |
| §4.6 view | `_MixedDecodeView.row_offset` 与 spec 字段切片；`prepare_decode` 按 `row_offset` 切 `num_rejected` |
| §6 单测 | `tests/test_mixed_spec_decode.py`、`tests/test_mixed_capability_gate.py`、`tests/test_scheduler.py::TestMixedSpecDecodeReserve` |

### 4.10 接受率排查结论（2026-10-08）

V4-Flash-DSpark tp4 + dp-attention，K=5，`--max-num-batched-tokens 2048`（强制 mixed 暴露），GSM8K 3-shot conc 1000。临时探针按"draft 在哪种 step 上 propose、是否被 TBO 切分"统计每个 decode 行的接受数。

| 配置 | 接受率 | GSM8K |
|---|---|---|
| 无 mixed 对照（B''） | ~60% | 0.947 |
| mixed + TBO，修复前 | ~7% | 0.943 |
| mixed，无 TBO | 65% | 0.953 |
| mixed + TBO，`--level 0` | 65.5% | 0.946 |
| **mixed + TBO，两处修复后** | **64.9–66.1%** | **0.956** |
| 两处修复但去掉 slot 拷贝（消融） | ~7% | 0.944 |

结论：mixed 上的 verify 一直是对的（pure-decode 提出的 draft 在 mixed step 上验证，3.0–3.5/行）。坏的是**在 mixed step 上 propose 的 draft**，有两个相互独立的原因：

1. **draft block 回放读静态 `state_slot_out` buffer**（本方案引入）：mixed step 上这块 buffer 只有 decode 半批的 slot，回放的 block 从错误请求的窗口取 KV。修复：V4 builder 的 `commit_speculative_state` 在 propose 前把 carrier 的全批 slot 设备拷贝进去。去掉它所有 mixed-proposed draft 跌到 0.11/行。
2. **aux hidden 捕获的 offset 被 Dynamo 烤成常量**（main 上已有的 bug，与 mixed 无关）：hook 在 `@support_torch_compile` 的模型内被 trace，ATOM 无 guard 调用编译代码，`ubatch_token_offset` 固定为 0；TBO 切分时两个 ubatch 都从第 0 行写，后半批的 aux 留着上一步的内容，draft 窗口被旧 hidden 填充。修复：写入改走不透明 op `drafter_aux_capture`（`torch_compile_guard`，运行时读线程局部 context）。也影响**不开 mixed 的 prefill TBO + DSpark**（每个请求第一个 draft），这解释了 §4.8 中 B' 比 A 低 2.6 点。回归测试：`tests/models/deepseek_v41/test_compilation.py` 在 level 3 下跨调用改变 offset，旧写法在此失败。

未处理：`eagle_proposer._aux_strip_hook` 在 TBO 下也总从第 0 行写（同类问题，EAGLE3 本机无法验证）。

## 5. 改动清单

| 文件 | 改动 |
|---|---|
| `atom/config.py` | validator 细化拒绝表；reserve 饱和 warning |
| `atom/model_engine/scheduler.py` | decode reserve 改用 `spec_width` |
| `atom/model_engine/model_runner.py` | `prepare_input_ids` mixed spec 分支（§4.2）；mixed spec 元数据（§4.3）；`postprocess` prefill 行 reject 归零（§4.4） |
| `atom/spec_decode/drafter.py` | `drafter_aux_capture` 不透明 op（§4.10）；anchor 改由 verdict 换算（§4.5） |
| `atom/model_ops/attentions/deepseek_v4_attn.py` | carrier 全批 `cu_seqlens_q` / `state_slot_out`（方案 A）；view 补切 + `row_offset`；`prepare_decode` 切 `num_rejected` |
| `atom/models/deepseek_v4_dspark.py` | 不改（方案 A 成立，§4.5 读者审计） |
| `atom/model_ops/attentions/backends.py` | docstring |

## 6. 验证计划

**CPU 单测（CI 可跑，无需 aiter）**
1. mixed 采样行索引：给定 `n_p, prefill 段长, decode lens`，断言 `bonus/target/cu_num_draft` 与手算一致，`input_ids[shift:]` gather 出的 draft id 正确（含 0-draft 的 prefill 行、新入 decode 行）。
2. verdict 换算：prefill 行（含中间 chunk）anchor 落在段尾、reject 为 0；decode 行 anchor 等于 `cu[i] + num_accepted`；无 verify 时每段最后一行。
3. **（必须）** `postprocess` prefill 行 `num_reject == 0`、`next_token_locs == 0`；并端到端断言：一个在 mixed step 走完 prefill 的请求，下一步 decode 时 V4 `prepare_decode` 算出的 positions 与"纯 prefill → decode"路径完全一致（§4.4 的错位就在这里显形）。
3b. 组成不变的连续两个 mixed step：`_dspark_apply_q_bucket` 返回 None，`num_scheduled_tokens` 未被改写（§4.6）。
3c. 刚走完 prefill 的请求的首个 decode step：`decode_src` 指向它上一步的 prefill 行，draft 取自该行（§4.5 全批 propose）。
4. deferred remap：chunked prefill 中间 chunk 在 prev_batch 里时不被当成 deferred 行。
5. view：`num_rejected` 按 `row_offset` 切；`scheduled_tokens` 不再静默穿透。
6. validator：V4+DSpark 通过；dense+spec、mtp、PP>1、`ATOM_TBO_MIXED=1`+spec 各自拒绝且文案正确。
7. carrier：全批 cu 前缀等于 prefill cu；`state_slot_out` pad 到 `running_bs` 且尾部为 0。

**GPU（tp4 + dp-attention；本机无 V4-Pro-DSpark，用 `/mnt/DeepSeek-V4-Flash-DSpark`、`--num-speculative-tokens 5`，对照基线见 §4.8 的 A / B'）**
- 三组，同 DSpark 配置（`--method dspark --num-speculative-tokens 7 --dspark-config '{"confidence_schedule": true, "ragged": true, ...}' --enable-dp-attention`）：
  - arm0：`--enable-tbo`，不开 mixed——纯 TBO+DSpark 的前置对照（§4.8-1）
  - arm1：arm0 + `--enable-mixed-prefill-decode`，`ATOM_TBO_MIXED=0`
  - arm2：arm1 + `ATOM_TBO_MIXED=1 ATOM_PROBE_TBO_MIXED=1`，并记录 `gate:split` / `gate:refused` 比例（§4.8-2）
- 强制暴露：`--max-num-batched-tokens` 要够大以免 reserve 饱和（先用 4096，按 §4.1 公式核算），`ATOM_PROBE_STEP_BUDGET=1`；step-budget 探针 500 步一窗口，短跑可能不出（上次就没出），需要用 mixed batch 计数兜底。
- 通过标准：GSM8K 3-shot ≥ CI 门限 0.93（基线 0.96）；**接受率**与 arm1 在噪声内（CI `mtp_accept_threshold` 口径）——接受率比分数更能暴露 draft/anchor 错位；0 fault、0 `PublicationError`。
- 精度分数只能证明"没有大面积损坏"（见 memory：逐 token 缺陷在分数上不可见），所以 §6 单测 1-5 是正确性的主证据。

## 7. 开放问题

1. **Reserve 饱和**：是否给 prefill 保底预算（如 `prefill_budget >= max(chunk_min, mnbt * α)`）？会改变 decode-first 语义，需要单独的吞吐/尾延迟 A/B。
2. **mixed step 放弃 ragged**：mixed step 每个 decode 行都 verify 满 K+1，浪费一部分算力；mixed 本来就算力受限（见 CUDAGraph/mixed 测量结论）。值不值得在 mixed 上支持 ragged，取决于 mixed step 占比，先测再说。
3. **单次运行的噪声**：B' 比 A 接受率低 2.6 个点，只跑了一次。mixed 组的接受率要和 B' 比，需要各组重复 2-3 次才能区分实现问题和噪声。
4. **decode 占多数时的 mixed TBO 切分**（§4.8-2）：是否支持切点落在 decode 区，取决于 arm2 测得的 `gate:split` 比例。
5. **Phase 2**：dense MLA `prepare_mixed` 的 spec 化（R1 + MTP）、`mtp`/`eagle` 放开。
