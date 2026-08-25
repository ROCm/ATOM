# 统一 superblock 池 — 实施计划

## 目标

把 K3 的两个独立 malloc

```
kv_cache      = (24, num_blocks, block_size, 576)   # layer-major
mamba_k_cache = (69, num_slots, ...)                # layer-major，独立分配
mamba_v_cache = (69, num_slots, ...)
```

合并成一个

```
pool = (num_supers, super_bytes)    # uint8
```

其中

```
super_bytes = ceil(state_bytes / block_bytes) * block_bytes
block_bytes = num_mla_layers * block_size * entry * dtype_size    # 层已在内
state_bytes = num_kda_layers * per_layer_bytes                    # 层已在内
```

一个 superblock 按 `kind` 有两种解读，layer 在两种解读下**都在 superblock 内部**，不进索引：

- `kind=KV`    → `blocks_per_super` 个 page，每个含全部 24 层
- `kind=STATE` → 1 个 slot，含全部 69 层

K3 TP8, block_size=128, fp8 的实测值：

```
block_bytes      =  1,769,472 B =  1.69 MiB
state_bytes      = 56,171,520 B = 53.57 MiB
blocks_per_super = 32
super_bytes      = 56,623,104 B = 54.00 MiB
STATE 侧浪费     = 0.80%
KV 分配粒度      = 4096 token
```

`32` 不是常数——bs=64/fp8 是 64，bs=128/bf16 是 16。必须按公式算，不得硬编码。
`super_bytes` 几乎恒为 54 MiB（它就是 state 大小向上取整），浪费永远 ≤0.80%。

## 为什么这条路可行而 DSV4 那条不行

DSV4 的 `UnifiedPoolGeometry` 要求"一行在两个 plane 里含义相同"
（`v4_pool_geometry.py:11-19`，因为 `mla_decode_fwd_v4_nm` 收一份共享 index buffer）。

K3 满足不了：`_state_dtypes` 给 kimi_linear 一个 **fp32 的 v 侧**，而 block 路径是
bf16（`kimi_mla_gdn_attn.py:86-88`）。没有共同的行语义。

superblock 按**字节**切，两种解读互不相干，绕开了这条约束。这是它比 DSV4 那条路
更适合 K3 的地方。

## 前一版错在哪

`SuperblockMap` + `BlockPool` 约束那部分是对的（32 个测试在）。
**Phase 6 的接线是不连贯的**：它保留了 `mamba_k_cache` 这个独立 malloc，于是
`claim_superblock` 扣走的 54MB **没有任何 kernel 会去读**——KDA 还是去私有张量取数。
那 54MB 纯粹白扣，这正是 GPU run 4 `hipErrorLaunchFailure` 的根。

本计划保留前者，重做后者。

## 关键发现：不需要改 kernel 接口

上一轮的假设是"要把 pool 指针 + superblock index 传进 kernel"。**这个假设是错的。**

`as_strided` 能从统一池里造出一个 **slot-major 的 per-layer view**，形状和
`mamba_v_cache[row]` 完全一致：

```python
v = torch.as_strided(
    pool.view(-1).view(dtype_v),
    size=(num_supers,) + shape_v,
    stride=(super_bytes // itemsize_v, shape_v[1], 1),
    storage_offset=layer_offset_bytes // itemsize_v,
)
```

已验证（CPU）：
- `v.shape == (num_supers, *shape_v)`，dim0 就是 slot
- `v[state_indices]` gather 正确
- `ssm_state[state_indices] = last_state` scatter 正确，写进了对应 superblock

也就是说 `kimi_k3.py:1070-1071` 的

```python
conv_state = cache.k_cache
ssm_state  = cache.v_cache
```

**一行都不用改**。变的只是 `build_kv_cache_tensor` 交给它的是什么张量。

`kimi_k3.py` 带 `@support_torch_compile`，CLAUDE.md 明令不得修改——这个发现让我们
不必碰它。

### 唯一的代价：非连续

`kda_state.py:56-64` 的 fused gather 已经在做 `.contiguous()`，所以 gather 侧安全。
**scatter 侧（`ssm_state[state_indices] = last_state`）和 decode 的原地更新需要实测**，
见 Phase 0。

## 阶段

### Phase 0 — 先证伪（GPU，半天）

在动任何生产代码之前，用一个独立脚本回答两个问题。任一失败则整个计划作废。

1. **非连续 state view 的正确性与性能**
   造一个 `as_strided` 的 view，跑真实 KDA 形状的 gather / scatter / decode 原地更新，
   对比连续张量的结果（要求 bit-exact）和耗时（要求退化 < 5%）。
   如果 aiter 的 decode kernel 假设了连续，这里会暴露。

2. **4096-token KV 粒度的真实代价**
   这是**最大的未验证风险**。之前模拟报的 drain rate 94–98% 是模拟，不是硬件。
   用 agentic trace 在真实 BlockPool 上量：以 32 block 为单位分配/淘汰，
   有效利用率是多少，命中率掉几个点。

产出：`docs/superblock_phase0_findings.md`，两个问题各一个数。

### Phase 1 — 几何（纯 CPU，可测）

新增 `atom/model_engine/superblock_geometry.py`，唯一拥有这套算术：

```python
class SuperblockGeometry:
    blocks_per_super: int
    super_bytes: int
    def block_span(self, block_id) -> tuple[int, int]      # 字节区间
    def slot_span(self, slot) -> tuple[int, int]           # 字节区间，= 一整个 super
    def kv_layer_offset(self, layer_idx) -> int            # super 内偏移
    def state_layer_offset(self, layer_idx) -> tuple[int, int]  # (k_off, v_off)
```

对标 DSV4 的 `UnifiedPoolGeometry` docstring：*"Sole owner of the arithmetic"*。
所有偏移只在这里算一次，`build_kv_cache_tensor` 和 checkpoint 路径都从这里取。

注意 fp32 v 侧 / bf16 k 侧的对齐——参考 DSV4 的 `slot_align_rows` 处理方式。

### Phase 2 — 分配（复用现有 SuperblockMap）

`SuperblockMap` 和 `BlockPool` 的约束逻辑原样保留，含已修的四个 bug：
- 分配优先级 vacant → mint → checkpointed（`ab929ebcf`）
- `_next_claimable_superblock` 按 hash 而非 kind（`f22ba0c1f`）
- `blocks_per_superblock` 走 `block_info` 跨进程（`d08065f05`）
- 整除向下取整（`29f8a15dc`）

`max_slots` 那个上限（`407b0c40e`）**删掉**——它是为"独立 state 张量"打的补丁，
统一池之后 slot 数只受 superblock 数约束，那才是本来该有的行为。

### Phase 3 — 分配与绑定

**3a.** `allocate_kv_cache_tensors` 改为分配单个 `(num_supers, super_bytes)` uint8 池。
`allocate_per_req_cache` 删掉——不再有 `mamba_k/v_cache`。

**3b.** `build_kv_cache_tensor` 用 `as_strided` 造 view：
- KDA 层 → slot-major state view（如上）
- MLA 层 → block-major KV view，`super*blocks_per_super + within`

**3c.** 兼容性：`model_runner.py:998`、`rollout/memory_manager.py:150` 按名字引用
`mamba_k_cache`/`mamba_v_cache`。保留同名属性指向新 view，或一并改掉——按
CLAUDE.md 的 fix-then-sweep，一次改干净。

### Phase 4 — checkpoint 变成一次拷贝

现在一个请求的 state 是 69 段跨 stride 的碎片；统一之后是**一段连续的 54MB**。
`StateTransfer` 的 COPY 路径从 69 次 strided copy 变成一次 `memcpy`。
FORK 路径本来就是指针交换，不受影响。

（对标 DSV4 `slot_span` 的注释：*"a slot is the unit a checkpoint copies and a PD
transfer registers"*。）

### Phase 5 — 验证

1. CPU：`tests/test_superblock.py` 全绿 + 新增 geometry 测试
2. GPU 单层 parity：统一池 vs 独立张量，bit-exact
3. GPU e2e：K3 agentic bench，命中率从 aiperf `profile_export.jsonl` 取
4. 对照组：同一 build 关掉 superblock（`superblocks=None`）跑一遍

## 边界：这件事解决的不是命中率

**必须记清楚。** state 容量（1.67GB / 35GB）从来不是瓶颈，瓶颈是 checkpoint 数量
（96 个 slot 里 92 个装 checkpoint，只有 4 个装在跑的请求）。

这套改造的价值是**结构性的**：
- 一个池，不再有两个互不相让的固定 malloc
- 分界线变成一个 host 侧整数，不重新分配、不重编译、不重新 capture
- checkpoint 从 69 次拷贝变成 1 次

命中率的账要靠 `--state-checkpoint-slots 64` 那条线**单独**验。两件事不要混着记功。

## 风险登记

| 风险 | 严重度 | 何时暴露 |
|---|---|---|
| aiter decode kernel 假设 state 连续 | **致命** | Phase 0.1 |
| 4096-token KV 粒度掉命中率 | **高** | Phase 0.2 |
| fp32/bf16 混合对齐 | 中 | Phase 1 |
| `mamba_*_cache` 命名依赖散落各处 | 低 | Phase 3c |

Phase 0 的两个数没出来之前，不写生产代码。
