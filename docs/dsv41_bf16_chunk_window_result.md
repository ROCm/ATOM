# 紧凑 BF16、分块 ready 与设备 RS：完整链结果

2026-10-29。**本版停止，不接入服务。** 去除 route/partial 的双重 FP8 编解码、把逐列全 rank 等待改为分块等待后，三个目标形状仍出现明确的完整链退化。同步改动有收益，但不足以支付这一套 G2 分列、BF16 物化、发布与消费的总成本。

部署为 TP4/EP1、FP8 KV、真实 DSpark5 对齐的配置，使用真实 layer8 专家/shared 权重和 layer9 mHC/norm/QKV。局部实验没有运行投机服务请求，不产生新的接受长度、执行行数或整网成绩。M1536 对照包含已经验证的 owner 一次量化供料；prefill 对照保持原生 owned 路径。

## 完整链和同步对照

每个形状四 rank、三组改变输入，每组两个 ABBA bracket，每段 128 次图回放。第一组比较原生与候选完整链，第二组比较同算术候选的逐列等齐与分块等待。

| 全局 M | 对照完整链 ms | BF16 分块候选 ms | 候选时间增加 | 相对同算术逐列等齐回收的时间 |
|---:|---:|---:|---:|---:|
| 1536 | 0.728004 | 0.925934 | **27.19%** | 35.927 µs |
| 7550 | 2.033622 | 2.652151 | **30.42%** | 88.932 µs |
| 16291 | 3.883588 | 5.566313 | **43.33%** | 180.401 µs |

有效的 72 个原生/候选配对全部退化；72 个逐列等齐/分块等待配对全部改善。三个形状的完整链最大重复漂移约 0.194%、0.121%、0.330%，远小于退化幅度。

同步 bracket 的逐列等齐→分块等待分别为 0.961309→0.925382、2.739599→2.650667、5.745175→5.564774 ms。它使用相同计算和缓冲，仅增加十次整列 ready 等待，因此可以支持“删除这些等待及其调度边界确实回收时间”的结论。它不等于独立 barrier kernel 的纯启动成本。

M1536 来自修复写回后的 `owner_bf16_ready_r5`；两档 prefill 保留 `owner_bf16_ready_r4`，未重跑。整个最终结果涉及两次真实 layer8/9 权重加载，不能写成一次三形状成功运行。R4 原 decode 的 routed 输出错误，其 0.901881 ms 计时排除，不进入上表或预算。

按此前 3724×38 个 decode 层调用、1199×18 / 647×18 个 prefill 层调用，把 layer8 差额外推，仅作条件代理，会增加约 60.96 秒，即旧 374.837 秒服务时间的 16.26%。这不是整网实测，也不是各层真实积分。它已经足够否定把本版直接整合进服务。

## 最终实现和实际数值边界

- 原生供料、gate/top-k、G1 callable/参数和 G1 FP8 输入值保持相同。G2 保持真实权重、完整 M 和 K640，按十个 N512 列段组织。
- 两个紧凑 BF16 route slot，十个 BF16 partial。六个 route 用 FP32 相加后落 BF16，再与本 rank 的 TP shared 相加落 BF16；RS 包含全部四 rank 的 shared 贡献。
- 每 16 行发布一个 ready epoch，RS 仅等待覆盖本 owner 行的块，直接写完整 H5120 owner 输出。不使用独立 RCCL、compact 或 stitch，也没有中间 FP8 编解码。
- 每次调用一个入口 epoch 汇合，随后十个含 G2 的阶段和两个 drain，总计 13 次 launch。候选没有逐列全 rank 等待；同步对照额外增加十次整列等待。
- 本地 route 双缓冲依靠同流阶段顺序；入口汇合保证所有 peer 的上次调用结束后才开始新一轮复用。图销毁、同步、关闭三个 peer IPC handle、释放 workspace view 和原生通信退役均完成。

四 rank 的三形状 shared-only 检查覆盖三个变化输入、延迟一个 rank 的图回放、符号/列段/奇偶列区别和 padding。最终完整链检查保留十个输出、输入未改写及独立的最后 1024 列参考：本地 route+shared 的 BF16 partial 精确一致，设备 RS 与独立旋转 rank 顺序的 FP32 求和→BF16 精确一致，所有 ready flags 属于当前 epoch。

**这些检查不代表原生数值等价或模型精度验收。** 修复后的 decode owner 相对 L2 为 0.004505–0.004694，原生自重复为 0.000722–0.000940；候选自重复最大约 0.001333。两档 prefill owner 相对 L2 约 0.002885–0.002895，原生自重复为 0；prefill 的候选自重复、逐列/分块及 graph/eager 十输出均为 0。

最后 1024 列的 routed 本地结果：decode 相对原生 L2 为 0.003559–0.003575；prefill 为 0。decode 的原生 atomic G2 与候选 weighted BF16 route 的求和/舍入合同不同；prefill 的后续跨 rank 归约合同也不同。不能用 BF16 dtype 或无 FP8 压缩推导逐字节等价。没有运行 GSM，也没有继承历史精度结论。

实际 gfx950 产物已导出并核对各 rank hash。最终 decode G2 阶段为 96 VGPR、16 KiB LDS；prefill 为 154 VGPR、32 KiB LDS。全部 private segment、SGPR spill、VGPR spill 为 0。因此，本轮仍不支持寄存器 spill 是退化原因。

## 本轮修正的两个实现问题

1. **多形状特化不足。** R1/R2 在同一进程先运行 M7550 后，M16291 于 owner 行 1888 附近出现错误；1888 是前一个配置的 owner 行数。单独 M16291 的 eager/graph/独立 partial/RS 诊断精确通过。增加入口汇合仍失败，排除了“入口汇合即可修好”的假设。显式把 actual M 加入每个 launcher 的 constexpr key，并把设备 helper 的布局改成包含全部维度的 constexpr tuple 后，连续三形状通过。证据支持旧形状布局被复用；未进一步拆分 FlyDSL 内部的具体缓存层。
2. **BN128 的 BF16x8 写回循环为零。** BF16 分支默认 32 个 epilogue lane，每 lane 写 8 个值；`128 // (32*8) == 0`，decode routed slot 根本没有写入。shared-only/零 G2 预检查捕捉不到它，真实 routed 参考的 L2=1 暴露了问题。R5 把 BN128 写回组改为 16 lane，保留 x8 store，并加上“正循环次数、整除、每个元素唯一写者”的检查。真实 routed 检查置于性能测量前。该修复仅补跑 decode，已完成 prefill 保留。

R3 曾在模型初始化前遇到 `Constexpr[Config]` 不被 FlyDSL 接受，随后使用受支持的 primitive tuple。所有失败、诊断与成功版本各自保存源码、原始输出和一次评审；没有改写旧结果。R4 的完成评审只表示记录完整，其 decode 后续明确判为无效，见独立 ASSESSMENT。

## 根因判断与停止条件

现有证据可以得出：**十次逐列全 rank 同步不是唯一、也不是足够解释总退化的障碍。** 它被替换后，同算术完整链回收 36–180 µs，但候选相对原生仍新增 198–1683 µs。两个 FP8 编解码也已经删除，仍没有获得净收益。

不能把剩余差额全部归给 G2、BF16 流量或轮询中的一项。当前实现仍支付分列 G2 的组织成本、紧凑 route 的写读、BF16 partial 的写读、每块发布/读取 ready、计算与 RS service CTA 的资源竞争，以及填充/排空。每个消费 CTA 的 256 个线程重复 acquire 相同 ready 条件，是源码中明确存在的额外工作；它的实际耗时占比尚未测定。这里不把逻辑访问次数当作实测瓶颈。

本版停止，不扫描 N 宽度、worker 数或 FP8 参数，不新增整网/GSM 测试。后续应保留原生大矩阵 G2 的有效调度，先核算能否把本地 route/shared 合并直接接入 owner 归约，删除一次完整 partial 物化；若仍要用 ready，必须按 CTA/块聚合通知并给出可见性证明，避免让每个消费线程承担重复等待。它是新的候选合同，尚未实现，不能预先承诺净收益或整网 2×。已停止的旧 ready/预取/owner-stage 实验不重跑。

原 owner 一次量化供料的局部 1.03969× 证据保留。最后服务样本仍为 6281.294 tok/s、1.191939×，及其单臂、时间间隔、冷特化和精度限制；不与历史 1.3244× 相乘。没有达到硬件理论极限的证据。

## 证据与资源

- [最终聚合结果](../../validation/dashboard_whole_20261029/OWNER_BF16_WINDOW_RESULT.json)
- [R4 记录评审](../../validation/dashboard_whole_20261029/owner_bf16_ready_r4/REVIEW.json.gz)、[decode 无效性说明](../../validation/dashboard_whole_20261029/owner_bf16_ready_r4/ASSESSMENT.json)
- [R5 修复后 decode 评审](../../validation/dashboard_whole_20261029/owner_bf16_ready_r5/REVIEW.json.gz)
- [最终节点空闲证据](../../validation/dashboard_whole_20261029/owner_bf16_ready_r5/FINAL_NODE_IDLE.json)

生产 `atom/` 和 `tools/` 未改。248783 / crsuse2-m2m-v3-085 保留，全部八卡空闲，无 GPU 容器/KFD 进程。输出先在容器退役后逐文件复制、校验 hash，再删除节点上的同内容副本；共享记录保留。后续实验重新做资源与存储准入。
