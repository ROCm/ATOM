# CSA state 融入 SWA block —— in-chunk 融合(option A)计划

**授权**:用户选 A(真塞进同一个 chunk,弹性,接受动 SWA 热核)。分支
`feat/csa-swa-fusion` ← `d096a0fb`(兜底:已验证 B2 option-b + retention@32k
94.75%/96.75%)。目标:CSA state 跟 SWA block 同 chunk / 同 block table / 同
retention pin,**删 CsaStatePool**,retention 复用 SWA 的 pin(已验证 dense/sparse 稳)。

## 为什么走这条(和之前失败的区别)
- 之前(feat/csa-swa-fused-pools)做的是"CSA 独立 chunk-原子 owner + 仍用 CsaStatePool
  pin",GPU 证伪:没修好 retention 还更费内存。**根因嫌疑在 CsaStatePool 的 pin/arena
  交互。** 本方案**彻底删 CsaStatePool**,CSA 完全靠 SWA 的 block table + SWA pin →
  绕开那套嫌疑代码。
- 40KB CSA state = kv_state 尾4 + score_state 尾4(main+idx)。score_state 必需
  (softmax-pool 的 logit,独立投影,不可从 kv 重算;`fused_compress.py:198-219` 实证)。

## 设计:c4 chunk 变 112KB,CSA 长在 SWA chunk 里

fp8 路径:
```
c4 chunk = 112KB = [ SWA nope KV 64KB | CSA state 40KB | pad 8KB ]
  以行看(head_dim=512, fp8 → 512 B/行):112KB / 512 = 224 行/chunk
    行 [0:128)   = SWA nope KV(128 token)
    字节[64KB:104KB] = CSA state(fp32 view:main kv/score + idx kv/score)
    字节[104KB:112KB] = pad
  rope 仍在并行 unified_kv_rope 池(不动,op4/op5 decode 不碰)
c128 / dense chunk = 64KB(不变,128 行,无 CSA)
```
chunk 大小 **per-group**:c4=112KB(224 行),c128/dense=64KB(128 行)。

## 核心机制 ★:SWA block 的"行跨距"从 128 → chunk_rows(仅 c4)

SWA KV 现在寻址 = `swa_phys * block_size + slot`(块间隔 128 行,连续)。
融合后 c4 的 SWA 块间隔变 224 行(中间夹 CSA+pad),所以:

- **物理行跨距**:`swa_phys * block_size` → `swa_phys * chunk_rows`(c4 chunk_rows=224;
  c128/dense=128 不变)。
- **token 窗口数学不变**:`pos // block_size`、`pos % block_size`、`slot = global_pos -
  swa_blk*block_size` 里的 block_size 仍是 128(一个 SWA 块 = 128 token)。
- **必须区分两个 block_size 用途**:①"每块多少 token"(=128,不变)②"每块物理行跨距"
  (c4=224)。混淆 = 静默 decode 崩。→ 引入独立参数 `swa_chunk_rows`(或
  `swa_row_stride`)传给每个 SWA 寻址核。

## 要改的 SWA 寻址点(热核,逐个 GPU shape-trace)
- `qk_norm_rope_maybe_quant.py`:`swa_block_size` 参数路径(SWA write / ring 读)—— 行
  跨距用 chunk_rows,窗口用 block_size。
- `paged_prefill_indices.py:146`:`paged = swa_phys * block_size + (...)` → `swa_phys *
  chunk_rows + (...)`;`swa_pages` 边界随之改。
- `paged_decode` 索引(SWA 段):同样 `swa_phys * block_size` → chunk_rows。
- decode attention 对 swa_kv 的读(asm/aiter op)—— 确认是否经上述索引;若直接读 swa_kv
  行,也要 chunk_rows。
- `state_writes.py` swa_write:SWA 写核行跨距。
- deepseek_v4.py:`module.swa_kv` 绑定 + `swa_block_size` 传参处。

## 步骤(每步 flag-off 字节等价 + CPU 绿;flag-on 里程碑 GPU)
0. 分支 + 本文档。
1. **arena 控制面**:`compute_arena_group_specs` c4 chunk=112KB(含 CSA 40KB;凑 compress
   16KB 整除 → 112);num_chunks 重算。CSA 不再是独立 owner(它是 SWA chunk 的一段)。
   `unified_kv_arena`:c4 SWA owner page=112KB(chunk-原子,本就 1/chunk);去掉 csa owner。
   CPU:arena 单测。
2. **张量分配**:`allocate_per_req_cache` c4 unified_kv = num_chunks × 224 × head_dim;
   `swa_kv` = 每 chunk 前 128 行(chunk_rows 跨距 view);CSA state view = 每 chunk 字节
   [64KB:104KB] fp32(main/idx 4 段)。c128/dense 不变。CPU import + 形状。
3. **SWA 寻址核**:上面所有点 block_size(行跨距)→ swa_chunk_rows。**GPU:先 flag-on 出
   token 正确(纯 SWA/无 CSA 命中)**,证明 SWA 层布局改对。← 最高风险门,先过这个。
4. **capture/restore**:CSA boundary 索引 = swa_block_table 物理页 → chunk 字节
   [64KB:104KB] 的 view;restore source = 终端块 swa 页。删 csa_page_table 通道,用
   swa_block_table。GPU:bit-exact capture→restore + prefix-hit。
5. **删旧**:CsaStatePool、csa_page_table(sequence/scheduler/block_manager)、
   StatePool._csa、scheduler csa 翻译、`ATOM_V4_CSA_*` retention env。CPU 全绿。
6. **retention = SWA pin**:复用 `ATOM_SWA_FULL_RETAIN/RETENTION_INTERVAL`(pin SWA 块 →
   CSA 同 chunk 跟着 pin)。
7. **GPU 收官**:prefix-hit restore / 逼淘汰 / **retention dense(512)+sparse(32768)都试**
   —— 用 SWA pin,dense 这次应也稳。== 或优于 d096a0fb 基线。量内存(CSA 无独立池)。

## 验证门 & 兜底
- 每步 CPU:`pytest tests/test_chunk_arena test_unified_kv_arena test_block_manager_arena
  test_csa_boundary_snapshot`(+ 改后的)+ flag-off 字节等价 + import。
- GPU 里程碑:step 3(SWA 布局)、step 4(capture/restore)、step 7(retention)。
  harness:logs_claude/{gsm_prefix_hit,arena_evict_gsm}.py + serve_arena_csa.sh(GPU4-7,
  V4-Pro fp8)。
- **不测过不 commit**;GPU 暴露静默 bug 当夜难解 → 停最后 CPU 绿 commit + 报告,working
  态仍在 feat/unified-kv-arena @ d096a0fb。

## 风险(诚实)
- **step 3 SWA 热核行跨距**:最高危 —— 漏一个寻址点、或 block_size(窗口)与 chunk_rows
  (行跨距)混用,decode 静默崩。缓解:step 3 单独 GPU 门(先只验 SWA 出 token,不掺 CSA)。
- rope 并行池保持 block_size 跨距(rope 不融)——确认 op4/op5 不受影响。
- bf16 路径 chunk 数不同(SWA 128KB + CSA 40 = 168 → pad 192KB,compress 32KB×6);serve
  用 fp8,以 fp8 为准,bf16 顺带对齐。
- CUDAGraph:swa_kv 是持久 view,chunk_rows 跨距固定,地址稳定;capture/replay 应 OK,
  step 3 GPU 验时确认。

## 进度日志
- 2026-07-26 夜:计划成文(本文档)。等用户确认后从 step 1 开干。
- 2026-07-26:开始执行。
  - step 1(arena 控制面)done:`compute_arena_group_specs` 改 per-group chunk
    (c4=112KB/224 行含 CSA;c128/dense=64KB/128 行),去掉 csa_main/csa_idx owner
    (CSA 不再是 arena owner,是 SWA chunk 内固定字节段),num_chunks 按混合 chunk
    大小重算(仍全组统一)。新增 `_c4_chunk_bytes()` 单一几何来源。spec 增
    `chunk_rows`/`csa_state_off`。
  - step 2(张量分配)done:`allocate_per_req_cache` 按 per-group `chunk_rows` 分配
    unified_kv/rope(c4 高、c128/dense 矮),不再全层同高。加 c4 identity 断言
    (c4_rows == num_blocks*k1_csa)防几何漂移。
  - step 4 views done:`_csa_boundary_arena_views` 改按 chunk(== SWA 物理页)寻址
    fused 段 [64KB:104KB]。
  - **关键机制核实(纠正计划隐含风险)**:CSA boundary 原来是 CsaStatePool 内容寻址、
    跨请求共享(prefix hit 靠 hash 找别的请求发布的页)。曾担心融进"每序列、窗口化"的
    SWA block 会破坏跨请求 prefix restore。核实 `block_manager.hash_blocks` →
    **ATOM 的 paged-SWA 本身就内容发布(`publish_swa_block`)+ prefix hit 复用
    (`claim_swa_cached`)**,checkpoint 靠 SWA pin 抗窗口淘汰。所以 fused CSA 段随
    "内容寻址 + pin 的 SWA block"一起被复用:capture 目的地 = 该 block 的 c4 物理 SWA 页;
    restore source = 终端命中块的 c4 物理 SWA 页。CsaStatePool 的内容寻址与 SWA 池重复 →
    可删。融合成立(前提:checkpoint SWA 块被 pin,已在 d096a0fb 验证 retention 可行)。
  - 待做:step 3(SWA 热核 128→chunk_rows)、scheduler/kernel 把 csa 页表换成 c4
    物理 SWA 页表、step 5 删 CsaStatePool 等、step 6 retention=SWA pin、step 7 GPU。
- 2026-07-25(续,用户决策):选**字面 in-chunk 弹性**(非平行池)+ **un-fuse c4 层
  decode SWA 写**(不动 aiter,c128/dense 保留融合)。见 memory
  `project_csa_swa_fusion_decision`。关键实现事实:decode 融合 SWA 写在 aiter
  (flydsl/group_quant)只收一个 swa_block_size;reads 全在 ATOM 索引构建器
  (paged_{prefill,decode}_indices),按 kv_indices gather,不在 aiter。
- 2026-07-25 CPU-绿 checkpoint(step 3+4+核心 5 完成,待 GPU):
  - step 3:swa_write / paged_prefill_indices / paged_decode_indices 加独立
    `row_stride`(kernel+wrapper+reference),默认=block_size(字节等价)。
    module.swa_row_stride 按层组设(c4=224/其余=128);_arena_group_rows 存 self,
    decode/prefill 逐组传 stride。
  - decode 写:`fuse_swa = is_decode and swa_row_stride==swa_block_size`;c4 走独立
    swa_write(row_stride=224),c128/dense/非arena 仍融合。prefill swa_write 传
    row_stride=swa_row_stride。
  - step 4:scheduler csa_main/idx page table = arena_swa_block_tables["c4"];
    restore source = 终端命中块 c4 物理 swa 页(v4_csa_boundary_source_ids 现存终端块
    LOGICAL swa id,BlockManager 设)。prepare_prefill 读 batch 属性即自动对齐。
    boundary views 已按 chunk(swa 页)寻址 fused 段。
  - step 5(部分/自动):bound_hit 简化为 SWA-only(fused CSA 随 SWA gate);
    max_csa_pages()=0(owners 已删)→ CsaStatePool 自动禁用(vestigial,待正式删)。
    2 个测 CsaStatePool 旧 gate 的用例 skip(标注待 step5 重写为 SWA-gated)。
  - CPU:749 passed / 35 skipped(排除既有 sglang env 失败)。commit 作为 GPU 前还原点。
  - 待做:GPU step3 门(纯 SWA 出 token,c4 224 stride)→ step4(capture/restore
    bit-exact + prefix-hit)→ step7(逼淘汰/retention dense+sparse+量内存);正式删
    CsaStatePool + ATOM_V4_CSA_* env(step5 收尾)。
- 2026-07-25 GPU 启动调试(fusion serve tp4 fp8 GPU4-7 port8100):
  - bug1:张量 sizing 用 per-rank spec num_chunks(TP 下各 rank 不同,463 vs 507),
    但 num_physical_kvcache_blocks 是 rank0 广播值 → mismatch,断言炸。修:
    allocate_per_req_cache 按**权威 num_blocks** 算 num_chunks_auth =
    num_blocks//c4_pages_per_chunk,per-group rows = auth*chunk_rows(c4=224/其余=128)。
    非 fused 时退化回原 num_blocks*k1_csa。
  - bug2:HCA(c128)compressor.kv_cache 绑定用 hca_p=num_blocks*k1_csa(旧统一尺寸),
    c128 现在行数变少(auth*128)→ view 尺寸不符炸。修:arena 下 hca_p 由实际
    tensor 行数推(unified rows // k2_hca)。c4 绑定 num_blocks*k1_csa 仍对(=c4 rows,
    num_blocks 必是 7 倍数)。
  - **GPU 全validated(commit a96f5c80)**:
    - **root-cause bug**:CSA capture 目的表用 physical_swa_table 的 `phys.get(b,0)`
      回退 → unbacked SWA 块(window-only 释放/-1 占位)映射到物理页 0 = chunk0 的
      **活 compressed KV**。融合下 boundary 与 unified_kv 共享,capture 覆盖 chunk0 →
      forward 崩(旧独立 boundary 池里页 0 无害,所以关联路径从没踩到)。修(scheduler):
      CSA capture/restore 表对 unbacked id 显式给 -1(arena.is_swa_backed 门),capture
      kernel 本就 skip phys<0。
    - bisect(GSM8K forced-prefix V4-Pro TP4 fp8 n=80):csaoff 0.975 / SWA-224-only
      (capture off) 0.9625 / capture→scratch 0.975 / 修前 0.293 / **修后 0.9875**
      (OVERLAP_chunks 1→0)。
    - **restore 验证**(ATOM_SWA_FULL_RETAIN=1):70/81 命中 cached:[1024],restore 真 fire,
      accuracy 0.9875。→ SWA-224-stride ✓ / capture ✓ / restore ✓ / retention=SWA pin ✓。
    - 注意:融合 prefix 命中**依赖 SWA retention**(bound_hit 走 SWA 门);window-only
      无 retention → [0] 命中(退化为 recompute,仍正确)。
    - 待:正式删 CsaStatePool(step5,已 vestigial 自动禁用)+ sparse RETENTION_INTERVAL 确认。
  - harness 重建:logs_claude/serve_fusion.sh(env ATOM_V4_UNIFIED_KV_ARENA=1 +
    ATOM_V4_CSA_PREFIX_STATE_CACHE=1 + ATOM_SWA_RETENTION_INTERVAL 可调;--block-size 16
    --no-enable-prefix-caching 隔离用)、logs_claude/gsm_prefix_hit.py(共享 8-shot 前缀
    强制跨请求命中,基线 snapshot 0.9533/evict 0.9467/assoc 0.9400)。
