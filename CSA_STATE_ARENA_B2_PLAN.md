# CSA-state 纳入 unified arena —— B2 字节竞技场 实施计划

**目标**:把 unified-KV arena 从"每组 row-tensor"升级为"每层**字节竞技场**",让
**SWA / compress / CSA-boundary-state 三方共享同一层的物理字节、按需互借**;CSA-state
**保 fp32**(同一 chunk 字节由 single-owner 按自己 dtype reinterpret);并给 CSA-state
加 **SWA 同构的 interval / full_retain / sparse-checkpoint retention**。目标一口气推到 GPU 验证。

## 已确认的决策(用户）
1. CSA-state 作为 arena **第三类弹性组**(不是独立池)。
2. **真三方共享物理互借**(B),且走 **B2 字节竞技场**(保 fp32,重构底座,无精度损失)。
3. 像 SWA 一样有 **interval + full_retain**。
4. **一口气全推到 GPU 验证**(内部分阶段有序,但不中途停在半成品)。

## 关键约束(代码硬数据)
- CSA-main 快照 `[4, 2*head_dim]` fp32、CSA-idx `[4, 2*index_head_dim]` fp32
  (`deepseek_v4_attn.py:332` "fp32 required for softmax-pool stability")。
- SWA/compress `unified_kv` `[rows, head_dim]` fp8/bf16。
- dtype+row 宽不同 → 只能"同一 chunk 字节、owner 各自 dtype view"(B2),不能 row 共享。
- CSA-state 只在 CSA(ratio-4)层存在 → 每类层的字节池 owner 集合不同:
  - c4 层:SWA + c4-compress + CSA-main + CSA-idx
  - c128 层:SWA + c128-compress
  - dense 层:SWA
- 复用已验证的 #8/#9/#11:每组物理表 + 双 free 列表借还 + 跨进程 propagation
  —— 把它们从 **row 单位推广到字节单位**。

## 阶段（一口气推，内部有序，每阶段 CPU 可验、flag-off 字节等价）

### 阶段 0 — 合流 snapshot 分支  ✅ DONE(merge commit f3871271)
实际结果:6 文件 15 冲突块全解(4 keep-both + block_manager StatePool 融合 +
deepseek_v4_attn keep-both);740 CPU passed(flag-off 字节等价)+ 33 skipped;两
flag 独立可开(arena 10 绿 / csa 10 绿)。StatePool.__init__ 补了 full_retain/
retention/checkpoint 转发;block_manager `self.swa = self.state_pool.swa` 别名让
arena #11 全部 self.swa.* 与 attach_arena/evict 不变。
- merge `feat/csa-snapshot-evictable` → `feat/unified-kv-arena`。
- 冲突面(两线都大改):`block_manager.py` / `scheduler.py` / `deepseek_v4_attn.py`
  / `deepseek_v4.py` / `swa_pool.py` / `envs.py` / `config.py`。
- 无冲突新增:`csa_state_pool.py` / `state_pool.py` / `v4_kernels/state_writes.py`
  / `v4_kernels/compress_plan.py` / `tests/test_csa_boundary_snapshot.py`。
- 出口:两 flag 各自 off 时 CPU 全绿 + 字节等价;两 flag 可独立开。

### 阶段 1 — arena 底座字节化  ✅ DONE(commit 2349ccd0)
实际:chunk_arena rows→bytes;unified_kv_arena 按 owner 泛化(swa/compress/csa_main/
csa_idx)+ dict specs + 保留 swa/compress wrapper(block_manager/scheduler 不变)+ CSA
alloc/free/resolve/physical-table + max_csa_pages;compute_arena_group_specs 出 dict
specs(per-owner page_bytes);model_runner/deepseek_v4_attn 读新字段。+3 CSA 单测,30
arena 测试 + 743 CPU 绿,flag-off 字节等价。normal(head_dim=512)下 bytes_per_chunk
== swa_block_bytes(CSA fp32 页整除),csa off 时 == 旧 row arena 行为。

- `chunk_arena.py`:`rows_per_chunk → bytes_per_chunk`;`page_rows → page_bytes`;
  `pages_per_chunk = bytes_per_chunk // page_bytes`(整除断言)。page id 语义不变。
- `unified_kv_arena.py`:group/owner 的 stride 用字节;加 CSA-main / CSA-idx owner
  (仅 c4 层组);`bytes_per_chunk` = 该层所有 owner page_bytes 的 LCM 对齐。
- 单测:字节化后分配/借还/id 守恒(推广现有 23 测试 + 三方互借用例)。

### 阶段 2 设计定案(option-b,GPU-in-the-loop 实现)—— 关键架构
追踪现有布局发现:每层 KV = **3 个独立张量** —— `unified_kv`(nope,dtype)、
`unified_kv_rope`(rope,bf16,仅 fp8 路径,与 nope **并行**同 page id)、CSA boundary
(独立 fp32 池)。op4/op5 decode 读 nope+rope 两张量;capture/restore 已按
`boundary[page]` 索引。

**定案(option-b,最小 kernel 改动)**:CSA boundary 变成 **`unified_kv`(nope)字节
buffer 的 fp32 view** —— arena chunk 用 **nope-only 字节**,`rope` 保持并行池(op4/op5
**不改**),只有 CSA boundary 绑定 + capture/restore 绑定 + CsaStatePool page 来源变。
- ✅ 已做(stage-2 sizing,commit 见下):`compute_arena_group_specs` 改成
  nope-only chunk(`block_size*head_dim*classical_elem`),CSA page = **kv+score 合并**
  (`2*TAIL*2*dim*fp32`),num_chunks 仍按 full swa_block_bytes(nope+rope)。head_dim=512
  下 CSA 页整除 nope_chunk(bf16 131072%32768==0 / %8192==0)。**csa-off 行为不变**
  (swa/compress 比例保留)。
- ⛔ 剩余(GPU-in-the-loop,GSM8K 验):
  - `allocate_per_req_cache`:arena+csa 时不再单独分配 boundary 池;boundary_kv/score
    = `unified_kv.view(fp32)` reshape 到 `[pages, TAIL, 2*dim]`(main/idx 各自,kv 在页
    前半、score 后半,as_strided);csa-on-arena-off 保留旧独立池。
  - `build_kv_cache_tensor`:c4 Compressor 绑定上述 view(替代 runner.v4_csa_*_boundary_*)。
  - 阶段 3:CsaStatePool page id = arena csa page(alloc_csa);capture/restore 传 view +
    arena csa 物理表;scheduler ship `physical_csa_{main,idx}_table`。
  - 只 GPU 可验(bit-exact capture/restore + GSM8K);nope/rope/op4/op5 不变降低风险。

### 阶段 2 — per-layer 字节 buffer + 多 dtype view
- `allocate_per_req_cache` / `build_kv_cache_tensor`:每层分配一块 **uint8 字节 buffer**
  (`num_chunks × bytes_per_chunk`);为每 owner 造 **dtype view over 同一 storage**
  (SWA/compress 现有 dtype;CSA-main/idx fp32),`torch.view(dtype)` / `frombuffer`。
- sizing:budget 按字节;CSA-state 不再单独 fp32 charge,并入 arena chunk 预算
  (回收 snapshot 分支 over-reserve 的 HBM)。
- **CUDAGraph**:dtype-view aliasing 在 capture/replay 地址稳定性要验(view 是持久
  storage 的固定 view,地址稳定;不稳则退每 owner 独立持久 view 同 storage)。

### 阶段 3 — CsaStatePool → arena-backed + retention  ✅ DONE(commit b1907113,CPU 验证)
实际:CsaStatePool 字节 arena 化(#11 双 free 列表:backed/unbacked)、`_claim`
**事务化**(先 arena 回填,ArenaEmpty 归还 id 再抛,不漏进 used)、`ensure_for_tokens`
优雅降级(ArenaEmpty→留 -1→recompute 兜底)、`evict_cold_for_arena` 借出。BlockManager
`_evict_cold_for_borrow` 泛化为**三方**(compress/SWA/CSA),三池 arena-alloc retry 共用;
CsaStatePool 尺寸 = `arena.max_csa_pages()`(逻辑上界,准入按真实 csa_available 门)。
StatePool 加 csa_arena_max_pages + attach_csa_arena + has_free_csa/evict_cold_csa。
+3 arena 测试(arena-backed+守恒 / 借出不漏 / 三方借用)= 13 arena 测试,746 CPU 绿,
flag-off 字节等价。基线:GPU 上 arena-on/csa-off 合流后出 token 正确(2+2=4 等)。
retention(interval/full_retain)复用 SWA 的 sparse-checkpoint —— CSA 侧的 pin/interval
留在阶段 3-后续(先把三方借用+GPU 打通)。

### 阶段 3 — CsaStatePool → arena-backed + retention(原始条目)
- CsaStatePool 的 page 改成从 arena 借 chunk(c4 层字节池)= fp32 view 字节段;
  纳入三方双 free 列表借还。
- 加 **interval / full_retain / sparse-checkpoint**(和 SWA `_pin_checkpoint` /
  `checkpoint_lru` / `retention_blocks` / `checkpoint_frac` 同构);新 env
  `ATOM_V4_CSA_RETENTION_INTERVAL` / `ATOM_V4_CSA_FULL_RETAIN` /
  `ATOM_V4_CSA_CHECKPOINT_FRAC`(或复用 SWA 的)。
- BlockManager 三方借还:`_evict_cold_*` 扩展到 CSA-state 参与(某方 ArenaEmpty →
  按 LRU 淘汰其他方最冷 ref-0,复用 #11 的双列表 + 无泄漏 + 认账复用)。

### 阶段 2-binding + 4 + 5 — GPU 打通  ✅ DONE(commit 11fa334e,GPU 验证)
实际(option-b 全落地):
- **2-binding**:`allocate_per_req_cache` 用 `_csa_boundary_arena_views` 把 4 个
  CSA boundary(main/idx × kv/score)做成 `unified_kv` 字节的 fp32 as_strided view
  (main stride 8192/off 0,4096;idx stride 2048/off 0,1024),按 pos 一层一 list;
  build 绑 view + `module._csa_owner`。view 数学先独立 GPU 单测 bit-exact
  (`logs_claude/gpu_csa_view_math.py`)。**释放原 21.21 GiB 独立 boundary 池**
  (log:0 GiB separate pool,30 CSA 层共享 unified_kv 字节)。
- **4**:`state_writes` capture/restore 的 `is_contiguous` assert 放宽成
  dim-contiguous(kernel 本就 stride-driven);scheduler **保 -1** 地把逻辑
  csa_page_table + restore source 翻成 csa_main / csa_idx 两套物理(main/idx 是
  不同 owner,一个逻辑页 → 两个不同物理页);CompressPlan 加 idx 变体字段;prepare
  建 main+idx 两 page table + 两 restore plan;`Compressor.forward` 按 `_csa_owner`
  选(off / csa-only 回落 main = 逻辑,恒等)。
- **5 GPU 验证**(DeepSeek-V4-Pro TP4 fp8,GPU4-7):
  - restore 路径(prefix-hit GSM8K):**95.5%**(382/400),restore 用物理 source
    页触发,在基线带内(snapshot 94% / recompute 96.7%)。
  - capture + 三方逼淘汰(forced-eviction GSM8K,util 0.80 紧池,unique filler):
    **95.75%**(383/400),池按鸽巢回收(1382 main 页/层=691 chunk/层 vs 400×~20
    块),restore=0(无 prefix 命中,capture 每 prefill 认领 csa 页 → 三方竞争),
    无 livelock/corruption/error。
  - CPU:36 arena/snapshot/view 测试绿;全量 delta 0(pre-existing 排序失败不变)。
- ⛔ 剩:CSA retention —— **尝试后发现设计冲突,已 REVERT**(见下);
  decode 路径的 csa 物理翻译目前每 decode step 都算(worker 只在 prefill 消费,
  仅 CPU 浪费,不影响正确)→ 可加 prefill-gate 优化。

### 阶段 6 — CSA retention  ✅ DONE(commit 0322c6d4,GPU 验证)—— 更正下方旧结论
**更正**:之前判"设计冲突已 REVERT"是**测试配置误判** —— 用了 `RETENTION_INTERVAL=512`
(每 4 块 pin 一个),比参考 32768(每 256 块)**密 64×**,病态过 pin 把共享 arena 塞爆,
才逼出崩溃 + 撬 pin 损坏。按**真实稀疏 interval=32768** 重测:
- 紧池(0.80)逼淘汰(capture+pin+三方借):**94.75%**(379/400),稳定不崩(对比 512 时 38.75% 崩)。
- prefix-hit(restore+retention):**96.75%**(387/400),restore 触发 1264 次。均在基线带。
- CPU:5 retention 单测 + 全量 delta 0。
实现:sparse-checkpoint pin(段尾+prompt 边界)+ **headroom gate**(arena 有余量才 pin,
不饿死 sibling)+ **pin 软让位**(evict_cold_for_arena 把 LRU pin 让给必需分配,不崩;
CSA 退化 recompute)。env 默认全关。
**latent 注意**:CSA page 是 sub-chunk(比 chunk 借/还粒度细),稀疏 interval 下撬 pin 基本不
fire;真要无隐患才需 chunk-原子 pin(1 block=1 chunk)。默认关 + 推荐 32768 即可安全用。

### 阶段 6 — CSA retention(旧结论:设计冲突,已 REVERT —— 见上更正,实为 interval 误判)
实现了 SWA 同构的 sparse-checkpoint pinning(`ATOM_V4_CSA_FULL_RETAIN`/
`RETENTION_INTERVAL`/`CHECKPOINT_FRAC`)+ 5 CPU 单测绿。GPU 定位:
- loose 池(0.85):prefix-hit 96% / 逼淘汰 95.25% —— **pin 机制正确**。
- 紧池(0.80)+ 逼淘汰:先 SWA 借用 ArenaEmpty 未捕获崩(加 pin-让位 fallback
  修崩),修完精度单调崩 92%→38%(headroom-gate 只减到 72%,未根治)。
- **根因(有数据)= 设计冲突**:共享字节 arena 下 retention pin 要么占 arena
  chunk(饿死 SWA/压缩弹性借用 → 触发 pin-让位淘汰 → mis-account arena chunk →
  KV 别名 → 崩)、要么不占(boundary 被 sibling 借走覆盖 → pin 无意义)。
  **arena-backed retention pin 与三方弹性借用本质冲突**,非快速 bug。
- 决定:REVERT(4 文件 + 单测,均未 commit;核心 B2 不受影响)。
- 重做方向:给 retention 独立不可借的小 fp32 池(copy-out pin 的 boundary,脱离
  arena 账目);或只在 arena-off 时启用 retention(退化 SWA-analog,无冲突)。

### 阶段 4 — kernel + scheduler + 跨进程（原始条目）
- `state_writes.py`(capture/restore)+ compress kernel:base tensor 换 owner 的 view;
  page id 走每组物理表(`arena_csa_page_tables`);`swa_pages=0`。
- `scheduler.py`:ship CSA-state 每组物理表(`physical_csa_table`,和 arena_block_tables 一起)。
- propagation:specs 带 CSA-state owner;`get_num_blocks` → `engine_core`。
- 每组物理表 staging(CG-safe 持久 buffer)扩展到 CSA-state。

### 阶段 5 — GPU 验证（一口气到底）
- flag 全开(arena + CSA-state-in-arena + retention)。
- **bit-exact**:capture→restore round-trip fp32(字节 view 保 fp32 → 应 bit-exact)。
- **GSM8K forced-prefix-hit == baseline**(snapshot 94% / recompute 96.7%,`arena_evict_gsm.py`
  强制前缀命中变体);逼淘汰三方借还不 livelock(py-spy 兜底)。
- 量三方弹性收益(arena on/off:num_kvcache_blocks / CSA-state 覆盖率 / 并发)。
- **029 agentic workload util 0.9 + full_retain/interval**:量真实收益(接上收益量化需求)。

## 风险 / 回退
- **字节化底座触及所有 KV kernel base-ptr** —— 最高风险区;逐 kernel GPU shape-trace。
- **dtype-view aliasing 在 CUDAGraph replay** —— 不稳则退每 owner 独立持久 view(同 storage）。
- **fp32 view over uint8 storage 的 stride/对齐** —— bytes_per_chunk 必须 LCM 对齐所有 owner。
- 合流冲突大 —— 阶段 0 先 CPU 全绿。
- flag 默认全关 → 不影响 shipping 路径;flag-off 全程字节等价。

## 规模提示（诚实）
这是数千行、跨分支、触及 arena 底座 + 所有 KV kernel 的重构,需多轮 GPU debug。
"一口气到 GPU"指不停在半成品,但实际会跨很多步 + 可能 background GPU 运行。
每阶段 CPU 门槛通过再进下一阶段;阶段 5 才上 GPU。

---

## 阶段 0 合流 COOKBOOK(已预演,merge --abort 保护成果;下次按此执行)

`git merge feat/csa-snapshot-evictable`(merge-base `2be397cb`;arena 领先 104
commit,snapshot 领先 3)→ **6 文件 15 冲突块**。自动合并成功:config.py /
sequence.py / **swa_pool.py**(关键:两边对 SlidingWindowPool 的改动不重叠 →
自动合并 = 同时含 arena 的 #11 双 free 列表 + snapshot 改动)/ compress_plan.py /
envs.py。

### A. 简单 keep-both(删 3 种冲突标记行即可,已验证)
`sed -i '/^<<<<<<< HEAD$/d; /^=======$/d; /^>>>>>>> feat\/csa-snapshot-evictable$/d' <file>`
- `v4_kernels/__init__.py`(2 块):HEAD 已从 state_writes import
  update_compressor_states/swa_write/swa_write_2buff_prepacked → 合成一个 import
  再加 capture/restore(+reference);__all__ 两侧都留。**注意**:不能盲 sed(会留
  重复 import),要手动合成单个 import 块(本 session 已做法记录在此)。
- `state_writes.py`(1 块):HEAD=DSpark gather kernel,snapshot=CSA capture/restore
  kernel,文件尾两段不重叠 → 纯 sed 删标记 keep-both。
- `deepseek_v4.py`(2 块):块1 Compressor.__init__ 两组成员(kv_cache_rope/write_mode
  + boundary_kv/boundary_score);块2 forward else 内 scatter_kv_cache_rope(12 空格)
  + else 外 restore_compressor_boundary(8 空格)——缩进天然分开 → 纯 sed 删标记。
- `scheduler.py`(2 块):ScheduledBatch 两个新 kwarg(arena=None +
  v4_csa_boundary_source_ids);__init__ body 两段表构建(arena_*_block_tables +
  csa_page_tables)→ 纯 sed 删标记。

### B. 真融合(高危,勿 sed;两大重构交织,需谨慎 + CPU 验证)
- **block_manager.py**(4 块:import / __init__ / can_allocate / allocate):
  - 架构分歧:snapshot 用 `self.state_pool = StatePool(...)`(封装 _swa+_csa,
    暴露 `.swa` property → 内部 SlidingWindowPool);arena 用 `self.swa =
    SlidingWindowPool`。**融合方案**:采用 StatePool 封装;把 arena 的
    `self.swa.*` 全改成 `self.state_pool.swa.*`(attach_arena / evict_cold_for_arena
    / has_free / num_evictable);`StatePool.__init__` 补 arena 参数(full_retain/
    retention_interval/checkpoint_frac 已在 SlidingWindowPool + `num_swa =
    max(num_swa, arena.max_swa_blocks())`)并在 BlockManager 里 `state_pool.swa.
    attach_arena(self.arena, self._evict_cold_compressed)`。import:去掉直接
    SlidingWindowPool,留 StatePool + UnifiedKvArena + envs。
  - can_allocate:交织 arena 的 `_has_free_compressed`/`swa.has_free`(#11 backed-free
    认账)+ snapshot 的 CSA `page_for_hash` 命中门(prefix-hit 收窄)。
  - allocate:交织 arena 的 cache-hit-claim/`_pop`(#11 _unbacked discard)+ snapshot
    的 `append_csa_placeholder`/`ensure_csa_for_tokens`。
  - deallocate/hash_blocks(自动合并区,复核):snapshot 的 CSA release/publish
    要和 arena 的 `_deallocate_block`(#11 backed-free 回收)共存。
- **deepseek_v4_attn.py**(4 块):
  - compute_block_bytes(17 行):budget 融合 —— snapshot 的 CSA boundary per-block
    charge 要改成 B2 的"并入 arena chunk 字节预算"(阶段 2/4 再彻底改;阶段 0 先
    keep-both 保 flag 独立)。
  - prepare_prefill(6 行):arena `_stage_arena_group_tables` + snapshot csa staging
    两个都调(keep-both)。
  - `_populate_swa_block_tables` 区(88 行,最大):arena 每组物理表 staging +
    snapshot `_populate_csa_page_tables` staging,逐段合并(多为 keep-both 新方法,
    但同区域插入点要对齐)。
  - _alloc_v4_metadata_buffers(18 行):keep-both —— arena `v4_arena_*` buffer +
    snapshot `csa_page_tables` buffer。

### C. 阶段 0 出口门槛
- `ATOM_V4_UNIFIED_KV_ARENA` off + `enable_v4_csa_prefix_state_cache` off:CPU 全绿
  + 字节等价(两特性各自 dormant)。
- 两 flag 可独立开(先不追求同开,那是阶段 1-4)。
- 本 session 已把 A 类 4 文件解完(工作树),但因 B 类高危融合需充足 context 而
  `merge --abort` 保护;A 类重解仅几分钟(上面记法)。
