# TP4/EP1 owner FFN → RS → 下一层消费者：完整链成本

2026-10-29。三种行数的真实权重原生局部诊断已完成。普通 owner-M gate 替代被数值证据否定；该结论不适用于显式固定 full-M 调度的 owner gate。

**13:11 UTC 更新：** [新的固定调度供料候选](dsv41_fp8_owner_supply_result.md)已经实现并完成A/B，M1536完整链约1.03969×，节省28.84µs/次；gate/logits/top-6逐字节兼容。下文原生阶段成本仍保留，供料的“未测”状态以新结果为准。Window-RS尚未实现；5%为后续结构集成筛选目标，不阻止用有限候选测清未知成本。

生产计算源码仍为 `96405d860`。本次没有新整网性能成绩；最后服务样本仍为6281.294 tok/s、相对保留同节点主线1.191939×，保留时间间隔、正式阶段四个冷特化/每rank、接受长度及执行行数差异等原有限制。

## 实际执行范围

使用实际 dashboard `EngineArgs → Config`：TP4/EP1/DP1、无DEP/DPA、FP8 KV、FP4 index、DSpark5、无合成接受率。完整加载layer8原生TP4专家、shared/gate、FFN mHC，以及layer9 attention mHC/norm/QKV权重。全部384专家，G2有效K576/padded K640；原生ExpertStagingPool的checkpoint切片与padding经独立CPU reference逐字节核对。

窗口：attention owner输出 → `Block.prepare_ffn` → 完整原生FFN → owner RS → `Block.prepare_attention` → 实际`Attention.project_qkv`。M1536/owner384走现有decode BF16 AG路径；M7550/16291走现有一次量化prefill路径。三者实际shared/routed双流判定均为关闭。

每种形状使用三次改变的独立hidden/residual输入，保留owner有效尾行合同。它们不是服务hidden-state快照。本局部窗口不访问KV、不执行投机接受/回滚，不能提供整网精度、KV状态或DSpark接受率验收。

## 测量结果

单位ms，四rank均值范围。完整窗口为无插桩图回放；每个观察图仅在一个阶段前后增加两个设备时钟标记，以完整链ABBA校准。没有减去估计观察开销。

| 全局M | 无插桩完整窗口 | 完整FFN | G1 | G2及本地归约 | RS | 下一层mHC/norm/量化/QKV |
|---|---:|---:|---:|---:|---:|---:|
| 1536 | 0.7634–0.7636 | 0.7070–0.7083 | 0.2413–0.2581 | 0.1539–0.1584 | 0.1008–0.1158 | 0.0351–0.0357 |
| 7550 | 2.0372–2.0374 | 1.9172–1.9217 | 0.5690–0.5747 | 0.3987–0.4150 | 0.4301–0.4405 | 0.0812–0.0833 |
| 16291 | 3.8991–3.8995 | 3.6255–3.6375 | 1.0319–1.0433 | 0.7702–0.7736 | 0.9022–0.9172 | 0.1431–0.1484 |

表列阶段均通过四rank观察扰动绝对值≤1%的门槛。M1536 producer mHC/norm单独观察为+1.021%～+1.035%，拒绝其阶段预算，不重跑。不同阶段来自不同观察图及改变后的输入，不能相加成精确trace，也不能累加跨rank最大值。

producer norm结束到G1开始约0.180/0.432/0.809ms，但包含shared GEMM、gate、top-k、排序和通信。不能把它全部当可删除供料；本轮原生观察没有测BF16 AG与新增Q/scale/logits通信的净差额，该问题随后由上方新候选解决。

## gate与GEMM的具体发现

| M / owner M | full-M原生配置 | 普通owner-M配置 | 三组输入top-6集合改变行数 |
|---|---|---|---:|
| 1536 / 384 | FlyDSL1765，split-K1 | FlyDSL6175，split-K5 | 193/4608，4.19% |
| 7550 / 1888 | FlyDSL307，split-K1 | FlyDSL2973，split-K2 | 862/22650，3.81% |
| 16291 / 4073 | torch native | FlyDSL1952，split-K1但不同wave/K配置 | 1539/48873，3.15% |

普通owner gate的速度不能记为兼容替代收益。M1536 full gate约24.9µs；保持full-M的zero/copy/gate/extract约40.5–41.1µs，新增约16µs/层。该fallback在全部三个形状有效logits与路由权重上均与full-M一致。

现有prefill强制one-K-wave调度：M7550 logits与full-M一致；M16291三组输入全部top-6集合不变，但logits不逐字节一致。rank0最大相对L2为4.72e-5，按专家对齐的路由权重L2为1.93e-5。“top-6相同”不等于完整数值等价；这是本轮保留的已有实现。

M1536实际G1为`t64x128x256_w3_bnt0_gui`、`fuse_quant=""`：输出BF16，随后单独MXFP8量化；G2为`t64x128x128_atomic`。原方案“G1已经融合FP8输出”的表述仅适用于这里的大prefill。

M7550/16291的G1为`fuse_quant="fp8"`，G2为`layout_...t64x256x128_reduce_sbm128`。M16291名义G1 N256配置被实际wrapper因K640不满足256对齐而改为N128。预算必须使用实际分派，不能只看配置名。

两个prefill形状的三次改变输入，十项完整链输出在观察图与原生图间全部L2=0。decode atomic路径存在native-self波动；观察误差与native-self为同量级。这不是模型精度验收。

## 整网预算与停止判断

复用已完成服务计数：M1536 owner FFN3724次；两档prefill1199/647次。以下将layer8延迟作为其他层、同档其他行数的代理，**不是实测整网覆盖率、理论上界或收益承诺**。

| 代理范围 | FFN占当前wall代理值 | 若FFN快2×，当前wall减少 | 理想十个N512窗口G2/RS流水毛收益 |
|---|---:|---:|---:|
| decode与prefill均仅2..19共18层 | 34.97%–35.03% | 17.49%–17.52% | 5.85%–6.14% |
| decode扩到38个非hash层，prefill仍18层 | 49.02%–49.10% | 24.51%–24.55% | 7.65%–8.18% |

38层也是外推；各层路由、辅助消费者及后20层合同未逐一验证。当前达到同主线2×仍需减少40.40%时间。上述范围即使完整FFN快2×仍填不平；2×主线还需要未覆盖层/attention等主要阶段。

十窗口模型为`0.9 × min(G2, RS)`，只支付理想填充/排空，尚未支付GEMM退化、额外collective启动、route/partial转换、缓冲和就绪不均。采用较严格的“当前wall减少5%”目标，38层代理仅余9.93–11.92秒新增成本，即平均约57–68µs/层调用；18层范围只余32–43µs。成本尚未测清，不能写成“已经证明无收益”。先以范围有限的候选检验净成本，服务晋级另判。

下一候选须先满足以下合同：

1. 在owner384上显式绑定full-M1536原生gate调度，验证数值与成本；不能使用普通owner-M lookup或扩prefill guard。保留logits传输与原生top-k，补齐完整BF16 AG方案与一次量化Q/scale/logits供料的净差额。
2. 分开适配decode atomic与prefill layout-reduce，保持完整M/K及原生BF16输出合同。若使用现成Window的route/partial量化，独立列为数值改变。后续消费者只在完整owner输出合同成立后消费。
3. 原生已经是RS，不能计“删除最后AG”。分列RCCL若需要compact输出再拼接，拼接成本也计入；保留辅助BF16读者和完整mHC状态。
4. 用一个具体联合候选检验预算；5%用于决定更大结构集成与服务筛选，不把未测成本当作拒绝有限A/B的理由。旧ready/prefetch/owner-stage/输入重叠不重跑。

## 证据与退役

仓外目录`validation/dashboard_whole_20261029/`：`owner_window_cost_r3/REVIEW.json.gz`保留完成的M1536（63绑定）；`owner_window_cost_r4/REVIEW.json.gz`仅补两种prefill（64绑定），四rank完整图/上下文/通信退出完成；`OWNER_WINDOW_COST_RESULT.json`及`owner_window_cost_budget.py`绑定新review与旧服务计数，未重跑旧review。

原定一次权重加载被监控故障打断，实际两次：R3保留decode，R4只补prefill。更早R1/R2为无权重图内事件API失败。R3的`du`原始stderr未保存，原子替换竞态是候选解释，CPU小复现未重现；R4改用容忍文件消失的计数，为临时/导出双份预留预算。没有将工具问题声称为模型故障或修复。

任务已退役，248783/v3-085借用节点保留。没有重跑完成的M1536、旧实验、GSM或旧review，没有修改生产运行代码，没有push/merge/rebase。
