# Prefill = CPP(PP4×TP1) + Decode = DPA 的可行性调研

目标拓扑：GPU 0-3 跑 chunked pipeline parallelism 的 prefill 节点（`-pp 4 -tp 1`，`kv_producer`），
GPU 4-7 跑 DP-attention 的 decode 节点（`-tp 4 --enable-dp-attention`，`kv_consumer`），
中间仍走 mooncake push RDMA + atomesh PD 代理。

参照现有的 `scripts/start_glm52_pp4pd.sh`（decode 是普通 TP4）。

调研基线：分支 `rebase-temp`，tip `dc51dd6c fix(attention): restore stable sparse MLA guards`。

---

## 0. 结论摘要

**新的 DPA 方案用的就是 prefill 那套 legacy 576 布局，所以 KV 传输不需要任何布局改造。**

`dc51dd6c` 把早期的 DS32 三张量 KV 格式从 `attention_mla.py` 里整个撤掉了
（"Use the validated unslimmed sparse MLA path as the branch baseline so GLM-5.2 DP-attention
avoids the HIP illegal-access regression from the slimmed variant"），
GLM-5.2 DPA decode 现在走的是 legacy 576 宽 fp8 单张量 + persistent 模式的 GQA64 kernel
（`requires_persistent_mode()`，`attention_mla.py:224`）。两端布局完全一致。

| 项 | 状态 |
|---|---|
| 拓扑本身（PP prefill + DPA decode 混搭） | **可行**，`assert pp_size == 1` 只约束 DPA 节点自己，prefill 是独立进程不受影响 |
| KV cache 布局 | **一致**，两端都是 `[L, B, P, 576] fp8` + `index_cache`，bpb 相同 |
| KV 传输 / region 映射 / rank 映射 | **零改动** |
| mesh / 路由 / handshake / 端口 | **零改动**，DPA decode 对外仍是一个 HTTP 端口 |
| RDMA 流量 | **降到 1/4**：TP4 decode 时每个 prefill stage 要写 4 份副本，DPA 只写 1 份 |
| **`aiter_mla.py` 里 DS32 的残留代码** | **唯一阻塞**，见 §1，是清理问题不是设计问题 |
| MTP | DPA + PD 下先别开（未验证） |

---

## 1. 唯一阻塞：`aiter_mla.py` 里 DS32 只撤了一半

`dc51dd6c` 撤掉了 module 侧（`attention_mla.py`，现在 `grep DS32` 是 0 个命中），
但 **builder 侧（`atom/model_ops/attentions/aiter_mla.py`）没跟着撤**，还留着 6 处：

| 行 | 残留内容 |
|---|---|
| 33 | `from atom.model_ops.attention_mla import _DS32_CACHE_BYTES, ...` |
| 171-178 | `self.use_ds32 = config.enable_dp_attention and ...` |
| 179 | `min_mla_heads = 128 if self.use_ds32 else _MLA_MIN_HEADS` |
| 833 | `kv_entry_bytes = _DS32_CACHE_BYTES if self.use_ds32 else 576 * kv_dtype_size` |
| 860-886 | `if self.use_ds32:` 分配 `kv_cache(512 fp8)` + `kv_scale_cache(16 u8)` + `kv_rope_cache(64 bf16)` |
| 925-931 | `if self.use_ds32:` 按三张量 view |
| 977-990 | `if self.use_ds32:` 往 `block_regions` 追加 scale/rope 两组 RDMA region |

两个后果：

1. **`_DS32_CACHE_BYTES` 在整个 `atom/` 里已经没有定义了**（`grep -rn "_DS32_CACHE_BYTES\s*=" atom/` 无命中），
   所以 `import atom.model_ops.attentions.aiter_mla` 现在直接 `ImportError`。这不是 DPA 特有的，
   整个 aiter MLA backend 都起不来。

2. 就算只把 import 补上，`use_ds32` 在 DPA 节点上仍然为真 →
   builder 会分配三张量 cache 并注册 **3 组 region/层**，而 module 侧写的是 576 单张量。
   这时 prefill（1 组/层，bpb 9216）和 decode（3 组/层，bpb 8192/256/2048）就真的对不上了，
   而 `_execute_block_transfer` 用的是 **producer 自己的 bpb**、且**没有任何跨端 region 数 / bpb 校验**，
   会静默地每个 block 越界写 1024 字节。

所以正确的修法是把 builder 侧的 DS32 一起删干净，让 `use_ds32` 这个概念消失：

```
aiter_mla.py:33      删掉 _DS32_CACHE_BYTES 这一行 import
aiter_mla.py:169-179 删掉 self.use_ds32 整块，min_mla_heads 直接用 _MLA_MIN_HEADS
aiter_mla.py:833     kv_entry_bytes = 576 * kv_dtype_size
aiter_mla.py:860-886 只保留 else 分支（576 单张量）
aiter_mla.py:925-931 只保留 else 分支
aiter_mla.py:977-990 整块删掉
```

删完之后 DPA 节点的 region 布局 = `[kv_cache 576 × L] + [index_cache × L]`，
和 PP prefill 每个 stage 的 `[kv_cache × L_local] + [index_cache × L_local]` 严格同构，
`port_offset.consumer_region_indices()` 的 group-major 映射（groups=2）直接就对。

> 顺带建议：无论如何都在 `_execute_block_transfer` / `_execute_block_slot_transfer` 之前
> 加一条跨端校验（producer region 数 vs `len(consumer_block_bpb)`，以及每个 `cmap[i]` 对应的
> bpb 相等），不等就 `logger.error` + 拒传。上面这种布局错配现在是静默踩内存，
> 加了断言就是启动即失败。

---

## 2. 为什么传输侧其余部分不用改

mooncake 是 **consumer 发起、producer RDMA WRITE** 的 push 模型：

1. decode 侧 `start_load_kv()` 把自己的 `consumer_block_base_addrs` / `consumer_block_bpb`
   打包成 `write_request`，按 `remote_pp_size` 发给 prefill 的每一个 PP stage
   （`mooncake_connector.py:948-978`），端口 = `handshake_port + side_channel_port_offset(dp,tp,pp)`。
2. 每个 stage 按 `consumer_region_indices()` 把自己的局部层 region 映射到 consumer 的全局层 region，
   只写自己那段层窗口。

代入本拓扑：

- **TP 映射平凡**：prefill `tp_size=1`，DPA decode 展开后也是 `tp_size=1`，
  于是 `remote_tp_rank = self.tp_rank % remote_tp_size = 0`，
  `consumers_per_rank = max(1, 1 // 1) = 1`。比现在的 PP4→TP4（每 stage 服务 4 个 consumer）还简单。
- **PP 映射复用现成逻辑**：consumer 照样对 4 个 stage 各发一份 `write_request`，
  `remote_pp_size=4` 从 prefill 的 `kv_transfer_params` 带过来。
- **DP rank 的选择不需要外部协调**：请求落到哪个 dp rank，就是那个 `EngineCore` 的 connector
  拿自己的地址去发 `write_request`，天然自洽。`remote_dp_rank` 描述的是 **producer**（prefill）
  的 dp rank，我们这边 prefill `dp_size=1` → 恒为 0。

---

## 3. DPA 在当前代码里到底是什么

`--enable-dp-attention` 在 `atom/model_engine/engine_core_mgr.py:104` 被展开：

```python
if config.enable_dp_attention:
    assert pp_size == 1, "Pipeline parallel + DP-attention is not supported yet"
    local_engine_count = tp_size * dp_size          # -tp 4 → 4
    config.parallel_config.data_parallel_size = local_engine_count
    config.tensor_parallel_size = 1
```

所以 `-tp 4 --enable-dp-attention` 实际是 **dp=4 / tp=1**：4 个独立 `EngineCore` 进程，
各自有自己的 scheduler、block manager 和一份完整层 KV cache（64 个 attention head 全在本 rank）；
MoE 侧通过 `FusedMoEParallelConfig.flatten_tp_across_dp` 把 DP×TP 拉平成 4 卡再切
（`atom/model_ops/moe.py:141`；加 `--enable-expert-parallel` 才走 EP+mori all2all，recipe 默认不加）。

一个请求只落在**一个** dp rank 上，它的 KV 也只在那一张卡——这跟 MLA+TP4 时"KV 在 4 张卡上冗余复制"
是本质不同，也正是 RDMA 流量降 4 倍的来源。

decode kernel 侧：`requires_persistent_mode()` 对 GLM-5.2 DPA（fp8 KV + sparse + `glm_moe_dsa`
+ `num_heads==64` + `num_kv_heads==1`）返回 True，强制 persistent 模式，因为
"AITER's FP8 Q/FP8 KV GQA64 kernel is available only in persistent mode"。
`is_persistent_mode()` 里有一条 `if page_size > 1: use_persistent = False`，
所以 **`ATOM_MLA_PAGE_SIZE` 必须保持默认 1**，否则 GLM DPA decode 会掉出 persistent 路径。

---

## 4. 其余注意点（都不是阻塞项）

1. **`assert pp_size == 1` 不影响本拓扑**。它在 DPA 节点自己的 `CoreManager` 里，
   prefill 是另一个进程、另一份 config。

2. **PCP 与 DPA 互斥**（`llm_engine.py:63` 直接 `raise`）。recipe 里那句
   "DPA cannot currently be combined with PCP" 说的是 PCP，**不是 PP**，别搞混。我们用 PP，不受影响。

3. **`max-num-seqs` / cudagraph capture size 是每个 DP rank 的**。
   4 个 EngineCore 各自独立调度，`--max-num-seqs 128` 意味着整机 512 路在飞。
   显存和 capture 时间都要按每 rank 重算。

4. **DP 之间是 lockstep 的**。`engine_core.py` 每步对 `has_unfinished` 做 all_reduce，
   空闲 rank 要跑 `_execute_dummy_batch()`。等 KV 到达不会卡住别的 rank（recv 异步），
   但负载越偏斜，dummy batch 浪费越多 → 保留默认的 `--dp-load-balance least_requests`。

5. **端口**。consumer 不 bind `_side_channel_port`（只有 producer 的 `_write_listener` bind，
   `mooncake_connector.py:1048`），consumer 用 `get_open_port()` 拿 notify/rpc 口，
   所以单机上 prefill 和 decode 共用 `handshake_port=6301` 不冲突。
   别的分支上有 `e72ba295 Fix ATOM DPA port conflicts for p/d disaggregated mode on single node`，
   起不来先怀疑这里。

6. **prefix cache 按 rank 分裂**。decode 侧每个 DP rank 有独立 block pool 和前缀树，
   同一 session 落到不同 rank 就命中不了，`project_pd_incremental_kv` 的增量传输收益也会同步下降。
   脚本里已经开了 `--dp-aware --decode-policy dp_sticky` 来解决这个：

   - `--dp-aware`：mesh 探每个 worker 的 `dp_size`（先 `/server_info`，回退到 ATOM 的
     `/kv_transfer_info`，`api_server.py:1754`），把一个 URL 展开成每个 DP rank 一个逻辑 worker
     （`<url>@<rank>`），转发时往 body 注入 `data_parallel_rank`，ATOM 侧
     `api_server.py:450` 据此硬 pin 到那个 EngineCore。
     decode 报 `dp_size=4` → `@0..@3`；prefill 报 `dp_size=1` → 单个 `@0`，
     注入的 rank 在 PP 路径上被无视（`pp_size > 1` 时 `add_request` 直接进 stage 0），无害。
   - `--decode-policy dp_sticky`：按 `x-session-id`（`header_utils.rs:67`）把 session 粘到一个
     decode rank，worker 不健康或空闲超 1 小时才重分配。prefill 保持默认策略。
   - **前提：压测客户端必须发 `X-Session-ID`**。不发的话 dp_sticky 退化成 mesh 自己的
     lowest-load 计数，同时又绕过了 ATOM 的 `least_requests`，比两个都不开更差。
   - 固定 isl/osl 的吞吐 sweep 没有前缀可复用，这套只会有开销没有收益。
   - 不要开 `AtomPdRankMappingPolicy::Idx2Idx`（默认 `None`）：它把 prefill rank N 映射到
     decode rank N，我们 prefill `dp_size=1`，decode rank 1-3 每个请求只会刷一条 skip 警告
     （`atom/mesh/src/routers/http_pd_router.rs:236`）。

7. **开了 dp-aware 之后，decode 的 rank 由 mesh 决定**。每个请求都带 hint，ATOM 的
   `_select_dp_rank_locked`（`least_requests`）被绕过（但仍然计费）。均衡好坏完全取决于
   session 分布，一个长会话会把负载钉在同一个 rank 上最长 1 小时。压测时盯 decode 日志里的
   `in-flight reqs=[...]` 看偏斜。

8. **MTP 先别开**。DPA + MTP + PD 三者叠加没验证过。

---

## 5. 落地步骤

1. 清掉 `aiter_mla.py` 的 DS32 残留（§1 的 6 处），确认 `python -c "import
   atom.model_ops.attentions.aiter_mla"` 通过。
2. 加 producer/consumer region 数 + bpb 一致性断言（防回归）。
3. `scripts/start_glm52_pp4pd_dpa.sh` 起服务；从 prefill/decode 日志里对一下
   `Registering %d RDMA chunks (%d block regions, ...)`，两端 block region 组数应该是
   prefill 每 stage `2 × L_local`、decode `2 × L`。
4. `simple_inference` 单条验证不是乱码 → GSM8K 1319 全量（对标现有 PP4PD 的 0.9348）。
5. isl=8192 / osl=1024，conc 128/256 对比 `start_glm52_pp4pd.sh` 头部那张基线表。
6. 再上长上下文（注意 `project_topk_grid_dim_hang` 那个 aiter `calc_grid_dim` 的坑）
   和 agentic trace replay，观察 4 个 dp rank 的负载偏斜与 dummy batch 占比。

---

## 6. 关键代码位置速查

| 内容 | 位置 |
|---|---|
| DPA 展开成 dp=N/tp=1、`assert pp_size == 1` | `atom/model_engine/engine_core_mgr.py:104` |
| DS32 残留（待清理） | `atom/model_ops/attentions/aiter_mla.py:33,171,179,833,860,925,977` |
| GLM DPA 强制 persistent（legacy 576 + GQA64） | `atom/model_ops/attention_mla.py:224` `requires_persistent_mode` |
| persistent 的 page_size 约束 | `atom/model_ops/attention_mla.py:250` `is_persistent_mode` |
| consumer 发 write_request、算 producer 端口 | `mooncake_connector.py:948-978` |
| producer RDMA 写、region 映射（缺一致性校验） | `mooncake_connector.py` `_execute_block_transfer` / `_execute_block_slot_transfer` |
| PP region → consumer region 的 group-major 映射 | `atom/kv_transfer/disaggregation/port_offset.py` |
| PCP 与 DPA 互斥 | `atom/model_engine/llm_engine.py:63` |
| mesh idx2idx rank 映射 | `atom/mesh/src/routers/http_pd_router.rs:236` |
| MoE 在 DPA 下拉平 DP×TP | `atom/model_ops/moe.py:141` |
