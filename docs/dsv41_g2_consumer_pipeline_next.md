# 大M G2生产者发布与owner/Mega消费者接续

## 当前结论：停止当前排序下的分段ready，转到输入通信的独立工作重叠

2026-10-29 02:39 UTC，`consumer_g2_completion_gpu_r2`完成四卡真实layer8/9、M16315、
三次变化输入与三种完整FFN图，44.037秒退出。1946项源码绑定、12份无损writer记录、
实际spart4×2映射、每次97890个token/slot在EP4中恰好出现一次，以及graph→观察缓冲→
IPC context→native pool释放均已核对。详情见[执行计划9.37](dsv41_prefill_pipeline_plan.md#937-真实g2完成曲线不支持当前排序下的大幅分段重叠)。

完整16行owner范围的中位完成位置为G2有效时钟区间的97.451%–98.693%；到90%时，
只有2.157%–7.549%的范围已完成。再用同一批记录检查16行×256列，中位仍为
95.921%–97.363%。这里都是同rank本地贡献的完成位置，尚未计入其他rank到达、控制
计数与可见性成本；不能把这些比例当作整网收益或硬件时钟频率。

据此停止下面历史方案中的“两级完成计数→分段ready→等待消费者”实现。当前expert
排序使同一owner范围依赖许多晚完成的M tile，单纯细分发布粒度没有足够大的提前量。
观察器保留为诊断工具；不修改G2排序来制造重叠，不拆M/K，不晋级当前IPC候选。

下一项有明确数据独立性的候选是输入调度：原代码先执行本地router/top-k，再量化、
Q AllGather、scale AllGather、route AllGather，然后启动完整M的shared/G1/G2。
`prefill_router.owned_router_logits`的已启用本地路径只读BF16 norm和完整gate权重，
不需要Q/scale AllGather结果，也不发起collective，因此可以与输入通信并行。

候选执行合同如下，尚未实现或提交GPU任务：

1. 主流保留前置Mega和原量化；量化完成后，独立通信流等待这个生产事件，再依次执行
   原生Q、scale AllGather。它使用同一communicator，保持各rank一致的collective顺序。
2. 主流执行完全相同的本地router和top-k。主流等待Q/scale通信完成后，仍按原顺序执行
   route AllGather，再启动完整M/K的shared/G1/G2与现有输出链。
3. 任一时刻只允许一个流使用该communicator的暂存池/collective；不与Mega或其他
   collective并行。所有跨流张量必须有明确的持有与record_stream/事件释放合同，
   graph从主流分叉并在主流汇合，释放图后才能释放通信流持有状态。
4. 只对已确认的local-router、route-packet、TP4/EP4、BF16/H5120部署启用；原生回退在
   状态写入前确定。调用者持有一条通信流，18层串行复用，不为每层建立communicator。

首个测试只比较调度，量化、数据布局、router/top-k、三个collective以及所有GEMM算术
和输出buffer都沿用原接口。一次真实单层加载测完整FFN的两对次序、三次输入与L2，
另用一段诊断trace确认router确实与Q/scale通信重叠；不能以两个流存在推断发生重叠。
只有完整链两对均受益且观察到正确依赖，才与注册量化生产者结合或进入整网筛选。

当前服务样本router约0.05ms，因而这一候选本身也不支持51%额外整网吞吐的承诺。
它用于验证可维护的通信/计算调度边界；下一次扩大实现之前，要复用已有整网记录
重建prefill/decode及各完整阶段的剩余时间预算，目标仍是2×，不能用局部小收益替代。

---

以下保留原调查过程；其中分段发布实施顺序已由上述实测结论终止。

本方案接续2026-10-29已完成的联合768筛选与真实服务分段。生产版本仍为
`yifehuan/dsv41_fusedmoe`的`f2d3ee47f`，主线截点`ebd5f84f3`；历史1.324404976×
保持，联合候选只有768筛选1.006385740×，不晋级，2×目标尚未达到。
新的服务退出修复已经验证。下面的分段发布与生产者重叠均未实现，不宣称已有收益。

2026-10-29 02:18 UTC更新：被动G2完成观察器已实现，并完成原生/无插桩/时间戳三种
gfx950产物的静态审查；GPU曲线尚未测量。见[执行计划9.36](dsv41_prefill_pipeline_plan.md#936-g2完成观察器已编译并审查gpu完成曲线尚未测量)
及仓外`consumer_g2_completion_r5/REVIEW.json`。实际末尾顺序为BF16 atomic→wait(0)→
CTA barrier→结束时钟；观察器新增9 SGPR和8字节LDS，无新增spill。无插桩装配也必须
测量，因为其指令调度与原生不同。下一步一个单层实例比较三种完整链，取得实际writer
到owner局部范围的完成曲线，再决定分段ready；不提前启动新的整网筛选。

## 已知成本与优化范围

约M16K的服务诊断中，单层专家段约1.08–1.11ms，Q AllGather约0.41–0.43ms，
输出交接约0.90–0.98ms，独立量化约0.022ms。事件有测量开销，native与候选端点
不同，形状也不是逐请求完全一致，数值只用于确定调查顺序。
详情和证据边界在[执行计划9.34](dsv41_prefill_pipeline_plan.md#934-真实服务分段下一处应看g2到消费者的交接)。

当前producer→consumer依赖为：

```mermaid
flowchart LR
  I[完整Q及scale ready] --> S[完整M shared W2写C]
  S --> G[完整M/K G1及G2累加C]
  G --> R[每rank整段G2结束后发布ready]
  R --> P[等待四rank ready]
  P --> M[owner读取四份C并执行Mega]
  M --> D[所有reader done才可复用C]
```

已经删除独立combine和发布复制，但发布仍在整个G2之后。下一个问题是能否让已完成
的输出范围先被消费者读取，同时保持原生GEMM的权重复用。本方案保持BF16有效payload
和归约顺序，以依赖交接为中心。不能靠增加更多整网请求、packet大小扫描或调整GEMM tile
来替代这项证明。

## 先区分三类等待，再选实现

首个实施交付是一次真实单层、同M、同权重的输出链诊断，复用已有缓存与单层loader：
观察shared完成、实际G2首尾、每rank发布、Mega首尾与最后reader，包含前置Mega和
完整FFN作为总窗口。不再启动新的整网ABBA来回答这个局部问题。

1. **到达等待**：如果部分rank仍在G2，消费者等的是生产者。先证明等待区间与实际
   G2完成有关；不能仅凭专家数量或各rank区间长度推断。不得相加各rank的最大时间。
2. **读取/计算依赖**：若各rank均已ready，检查实际ISA和GPU时间线中remote-load与
   本地Mega算术的依赖，评估是否能提前发出连续读取。保持FP32求和次序和最终BF16
   舍入；额外寄存器/LDS不能造成spill或挤掉原生GEMM的资源。
3. **链路传输**：若ready后已由BF16流量占满链路，单纯少一个launch或多一个buffer
   不能消除此项。此时停止“再次合并小kernel”的候选，先建立通信与计算重叠的
   进度合同，再进入新的实际实现。

这三类需要实际kernel边界证据。最初0.9ms只有host包围的GPU事件；本轮已补齐下面
的实际kernel时间线，优先级转向数据ready后的读取调度。

## 2026-10-29：实际G2时间线与读取调度决策

`validation/whole_join_20261028/consumer_g2_timeline/REVIEW.json`已完成1877项源码
绑定、四rank原始trace、七项输出与退出核验。真实layer8 EP4/shared TP4和layer9 Mega
只加载一次；同M分别记录原生RCCL完整链、direct-C IPC完整链，以及G2已经产出C的
单独消费者。第三种图复用第二种图的有效输入/输出地址，先释放消费者图再释放生产者图。

| 实际M | rank0 G2（direct链） | 数据已ready的IPC/Mega（四rank范围） |
|---:|---:|---:|
| 7835 | 254.843µs | 396.284–417.764µs |
| 16315 | 427.605µs | 789.529–832.050µs |

ready消费者与完整direct消费者的七项输出逐字节一致。原生链实际使用RCCL，区别于
更早以IPC-combine为基线的组件对照。这里是profiled单次replay，不能用于验收吞吐；
不同缓存状态下两种时长相减也不是精确的生产者等待。Profiler设备ID为2/3/4/5，保留
原始编号；相同baseTimeNanoseconds并不证明跨rank硬件时钟已经校准。

这说明大部分消费时间在数据已经ready后仍然存在，尚不能认定链路带宽已饱和。
先提高消费者读取的并发度，比立即给原子G2加入复杂的tile完成发布更有依据。

进一步核对`consumer_mega_uniform_peer/output/isa.txt`发现：编译器已将一个行片段的
四路peer读取提前发出，之后才等待和转换。因此，单纯把源码改成四个临时变量的方案
取消，未启动重复GPU测试。

新候选`consumer_mega_rowpair_prefetch`只修改large IPC输入准备：每个线程同时取自己
负责的两个行片段，共8个packed BF16向量，再各自按原旋转rank顺序FP32求和并转回BF16。
它不拆GEMM、不提前发布、不增加通信缓冲，也不改Mega后续算术。编译结果已确认：

- 原来的两批各4条load，变为8条load后一个wait；两个片段的地址映射不变。
- large仍为256 VGPR、72 SGPR、162816B LDS、12B private；原有2条入口scratch store
  和1条尾部scratch load不变，未新增循环内spill。
- mid IPC与两个local消费者的指令流逐条一致，ABI仍为152B。

真实FFN检查已完成独立的full与ready两组ABBA，覆盖4097/7835/16315/16384/4097及
三次变化输入。full窗口包含前置Mega、完整FFN和下一层Mega；ready两版本共用同一份
生产者输入和输出地址，七项结果逐字节一致。然而该候选没有性能收益：

| 候选 | M | 原完整链 ms | 新完整链 ms | 原ready消费者 ms | 新ready消费者 ms |
|---|---:|---:|---:|---:|---:|
| 8路读取共同wait(0) | 16315 | 2.823633 | 2.906613 | 0.854124 | 0.958328 |
| 8路读取共同wait(0) | 16384 | 2.846424 | 2.926126 | 0.862043 | 0.962798 |
| 8路读取独立等待 | 16315 | 2.826716 | 2.907486 | 0.853251 | 0.950195 |
| 8路读取独立等待 | 16384 | 2.834948 | 2.904836 | 0.854903 | 0.955036 |

第二版`consumer_mega_rowpair_stream`取消跨行opaque依赖，实际指令恢复vmcnt7/5/3/1
逐步使用，8条load仍提前发出，资源和scratch仍与原版相同。新真实FFN只测改变了的
两个large形状，未重复mid；完整链两对仍均退化。两版都停止晋级，不做整网重跑。
这排除了“共同wait(0)是唯一退化原因”的解释；没有硬件counter证据，不能进一步
断言是链路饱和、cache冲突或某个rank拖慢。两次fixture共用各自一次权重加载，分别
55.804s与42.138s完成并正常释放graph/context/pool。

下一个结构对照是`consumer_mega_owner_stage`：保持相同私有C和完整G2，在publish
之后由已有低资源`consume`完成归约，写一次本地owner，再调用原生local Mega，最后
执行原done协议。编译产物的完整GPU反汇编与原版逐字节一致，仅增加host装配入口。
`consume`使用48 VGPR、24 SGPR、0 LDS、0 private；它与256 VGPR、约159KiB LDS的
Mega有明显不同的驻留条件。这是执行资源分离的对照，不能先称为已优化。

该对照最多增加40MiB/rank固定owner空间，每次M16315额外本地写读约83.54MB/rank，
远端有效读取字节不变。目标是测出：省掉这一轮本地物化，是否值得让远端传输承担
高资源Mega的执行条件。若低资源分离明显更快，下一步才讨论同一owner分段上通信
与消费的有界接续；若无收益，停止该结构，不继续调packet大小。任何后续流水都要
保留本方案的完成、可见性、最后reader及前进保证。

对照现已完成，`consumer_owner_stage_ffn/REVIEW.json`核对1937项源码绑定，四rank三次
变化输入的七项ready输出逐字节一致，新增owner空间在所有图释放后、communicator前
释放。M16315完整链2.829025→2.860352ms，M16384为2.842065→2.859744ms；两对均退化。
ready消费者只分别改善0.9%和1.4%，无法抵消完整链中的代价。该结构停止晋级。
原生RCCL时间线在M16315/rank0为849.210µs，后续Mega为102.520µs；不要误把M7835
的415.164µs RCCL时间拿来与M16315的IPC消费者比较。当前证据说明单纯扩大预取或
分离执行资源不足以打开空间，还不能据此宣称物理链路已饱和。

## 下一实施：先观察G2可发布范围，再决定分段接续

前三个候选均不改生产。下一项应先提供**完成范围观察器**，而不是直接让Mega提前spin。
实际运行G2为`bm64_bn256_bk128_atomic_a8_sbm128_bhoist_apf_spart4x2_bf16lds`。
当前源码给出如下约束：

- `mxmoe_dispatcher.py`的非persistent spatial分支通过`_spart_output_tile_index`
  把CTA映射为`(m_block_idx, n_block_idx)`，不能用原始blockIdx除20代替。
- H=5120、BN=256，完整输出共有20个N tile；BM=64，而expert sort块SBM=128。
  expert id按`m_row // 128`取，两个M tile可以属于同一sort块。
- 输出token来自`sorted_token_ids & 0x00ffffff`；高8位是route slot。有效条件沿用
  实际`write_limit`，padding、无效token和空expert都必须排除。
- BF16 LDS路径的epilogue部分与MFMA交错。`_epilog`构造、某个store thunk发出、
  一个wave完成，均不能代替整个CTA最后一次输出atomic完成。

第一步只在隔离的新G2变体增加尾部观察：保留原tile、K循环、shared初值和所有BF16
atomic，在真正退出epilogue之后，所有wave完成输出存储并经过CTA汇合，再由一个线程
记录该`(M tile,N tile)`的完成序号/时间。输出槽按实际M-block上界和20个N tile分配，
由调用方或graph持有，不加入无限缓存。clock指令需要确认跨CU的时基；跨rank时钟
未校准时只报告各rank内部的完成曲线，不拼成伪精确的全局关键路径。

同次保存实际sorted ids、有效计数与映射，离线构建“owner的16行消费范围→全部可能
writer CTA”关系。一个消费范围的ready时间是所有writer完成的最大值；四rank均需
满足此条件。先回答有多少有效范围能在整段G2结束前完成、能提前多久，再讨论同步。
这里的范围按owner本地行编号定义：`count=ceil(M/4)`，`owner=token_id//count`，
`tile=(token_id-owner*count)//16`，全局起点为`owner*count+16*tile`。M16315的
count=4079并不对齐16，不能直接使用`token_id//16`作为所有owner的消费块编号。
如果大多数范围直到最后一批G2才ready，就停止该边界的流水化，不能为了造出早ready
而拆小M/K或重排整套GEMM。

观察器本身可能改变G2调度，必须同时测原始和插桩G2/完整FFN成本，记录时钟、store
完成及额外barrier的实际ISA。其时间线用于选择结构，不计作优化收益。

只有完成曲线显示可用窗口，才实现两级完成计数：先汇合一个M tile的20个N writer，
由最后完成者计入相应owner范围；再在该范围全部贡献完成时发布system-release ready。
不能对每个BF16元素发控制atomic，也不能只给计数加relaxed原子。预期贡献数从原生
sorted ids与有效条件产生，零贡献范围仍需等shared初值完成。计数归零/epoch、最后
reader、最后层flush及不支持形状的写前回退一起设计。

最初的消费者继续等完整G2；观察器通过后再单独验证有界并行消费者。其驻留资源必须
给G2留下可证明的前进空间，不能让512线程、约159KiB LDS的完整Mega在全部CU上
占位等待。整个过程保持完整GEMM权重复用，不把当前三个局部实验的倍率乘历史整网收益。

## 输出范围的生产与读取所有权

首版继续一个communicator拥有的固定C，规模、调用前准入与18层串行复用不变。
发布范围必须同时对齐Mega的16行消费tile和G2实际存储覆盖，不能先指定一个packet
大小再让GEMM迎合它。G2按expert排序的行号与owner的连续token行号不同，绝不能
把“一个expert tile结束”当作“一段owner输出已完成”。

每个可发布范围必须证明下列合同：

| 责任 | 必须提供的证据 |
|---|---|
| 计划 | 从本次原生sorted ids和实际G2调度得到所有可能写入该范围的producer；不重算top-k，不复制完整activation |
| 完成 | 每个producer最后一次BF16 atomic写入之后，才能计入完成；shared初值和尾行清零也先完成 |
| 发布 | 最后一个producer以system release发布本epoch的ready，不能只用CTA barrier或主机launch顺序 |
| 读取 | 所有参与rank的对应范围ready后，消费者system acquire再读；保留同一FP32归约次序和BF16转换位置 |
| 回收 | 最后reader发布done；下一层/下一次调用不能提前复用任何仍被读取的C或计数区 |
| 拒绝 | 不支持的GEMM版本、动态调度、shape、流或计数溢出在写C之前选择原路径 |

原生G2是融合/原子累加kernel，当前没有已经验证的“最终tile写出”回调。
必须先定位实际加载的G2二进制和可维护的生成源，验证一个窄接口可以放在最终写入
之后。不能通过对未修改G2另发一个等待kernel来制造不存在的tile完成信号。
如果取得完成信号需要逐token额外原子、重复sort或拆成多个小GEMM，先核算完整链成本；
成本超过可隐藏等待则停止该实现，不削弱完成合同。

## 进度与资源是前置条件

当前单stream的整段ready协议天然保证生产者先执行。分段读者若提前运行，可能占满
CU/LDS，反而使G2无法完成。首版必须给出可运行的资源界限：消费者的驻留数量、每CTA
资源、G2需要的剩余资源，以及通信依赖图无环。流优先级或普通spin循环不能当作证明。

不把整个Mega改为常驻等待kernel。优先把ready接入已有Mega输入loader，保持原生算术
主体与工作区布局；但只有硬件调度能保证G2和读者同时前进才允许并发。若无法证明，
保留当前“完整G2后发布”的安全路径，先处理已经ready后的加载依赖。

所有权代码仍放在已有lease/plan/binding边界：唯一Arguments ABI、caller持有七项输出、
私有C不逃出opaque FFN→下一层Mega。收尾使用已验证的
graphs→IPC context→communicator/pool顺序，完整key与sealed miss guard保留。

## 接入与验收顺序

先完成真实G2及输出等待诊断。选定一个有可消除时间依据的候选后，独立GPU fixture
验证不同输入、反复epoch、4096原生回退、4097/7835/16315/16384、尾行和最后reader。
核对实际ISA、VGPR/LDS/private/spill与原生G1/G2名称；完整窗口含前置Mega，不能把
搬到producer的工作排除。分段数据和无观测完整窗口分开保存。

只有真实单层完整链两对均获益，才接入已预建的native/candidate服务变体。
一个实例可预先声明768筛选→条件2560→GSM8K/L2的完整序列，复用权重与已覆盖图键；
不在线改traced producer。新数值/同步组合使用自己的GSM8K和L2结果，不能继承旧shared
或IPC组件的质量结论。已完成的0.6386%筛选不重跑来挑选较好倍率。

大prefill与C256 decode分别列覆盖域。当前候选只服务16条大prefill边；即使这里明显
加速，也不能据此宣称整个模型达到2×。目标是否可达，由完整服务时间和质量结果决定。

输出预算按最新实际2.632MB而不是旧1.8MB估算，并加审查与退出日志余量；每个新任务
启动前重新核对所有旧账及单实例tmp/shm。原始结果保持冻结，已退役观察文件仅作
可还原SHA的无损归档。当前所有GPU任务已退役，借用244395/node155整节点继续保留。
