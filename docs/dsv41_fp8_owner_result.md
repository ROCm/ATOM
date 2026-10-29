# FP8 owner 迁移：C256 完整整网结果

2026-10-29 10:53 UTC。旧 owner 与 retained-prefill 路径已迁移到 **FP8 KV、完整 M1536**，真实模型512条诊断与正式2560条请求均完成。正式测得6281.294tok/s，相对保留的同节点主线5269.811tok/s为 **1.191939×，吞吐增加19.19%**。这是新迁移版本单臂与保留基线的比较，仍有时间间隔及冷编译限制，尚未作为稳定性能／任务精度验收或生产默认配置。

## 配置与对照

所有行均为用户节点 `248783 / crsuse2-m2m-v3-085`、4张MI355X运行、整节点8卡独占。配置为attention/routed/shared TP4、EP1、DP1、**无DEP/DPA**，FP8 KV、FP4 index、真实DSpark5、max-seqs512、max-batched16384、level3/FULL、原生prefix/bounded replay。工作负载C256、random8K/1K、seed0、593条独立预热后2560条正式请求。

| 版本 | 吞吐 tok/s | TPOT ms | 正式接受长度 | decode执行行数／rank |
|---|---:|---:|---:|---:|
| 锁定主线，保留A1/A2合并 | 5269.811 | 46.1870 | 2.488826 | 5,804,691（每臂均值） |
| 当前分支，旧owner未启用，保留B1/B2合并 | 5298.381 | 45.9014 | 2.472874 | 5,834,769（每臂均值） |
| 本次FP8 owner＋retained-prefill迁移，full | **6281.294** | **38.5883** | **2.471299** | **5,835,366** |

迁移版相对主线1.191939×，相对原生分支1.185512×；正式耗时374.837秒，相对主线每臂均值446.783秒减少16.10%。三组输入指纹一致，每臂18,926,022输入tokens、2,354,462输出tokens。新运行启动较最后基线退役晚3887.139秒，且随后还有初始化和预热；不是连续交错的配对实验。未重跑旧arm或旧review。

接受长度比主线低0.704%，decode执行行数多0.528%；相对原生分支，decode行数仅多0.0102%。因此不能把本次收益解释为接受长度升高或裁掉验证行。请求轨迹仍有变化，不据此给出单kernel因果倍率。

历史 **1.324404976×** 仍是有效的旧实验结果：Spec0/BF16 → DSpark5＋owner packed＋EP4＋prefill。它与本次两侧DSpark5、FP8、EP1、full的分母及工作量不同，不与本次倍率相乘。EP4部署和packed前缀策略仍需分别比较。

## 实际迁移与覆盖

六文件私有overlay负责FP8 Q/scale、owner attention、prefill缓存行与draft context的528B行传输。decode暂用四个原生H16调用组成H64，包含分片拷贝与输出拼接；保持完整原生TP4/EP1专家GEMM的M/K及权重。C256每owner384行、全局M1536，没有扩大小M Mono guard，也未调用packed的192行裁剪。

正式区间四rank一致：

- 全部40层实际为TP4/EP1，FP8激活、真实DSpark5。
- owner target 4067步，其中原生FFN M1536为3724步。
- retained-prefill 1846次、18,623,230行。
- compact draft context 4067次；原生选择性状态回接和客户端输出均保留。
- 七种owner容量都已有40层FP8 attention图；正式阶段新增owner图0、seal后缺失0。

外层`ModelRunner`的M1536计数3712与owner FFN M1536计数3724描述不同布局／padding合同，不能混用。计时与行数均保留原始来源。

组件验证包括四rank原生H16逐位对照、独立FP32 attention参考relative L2约0.002064–0.002100、完整1536和ragged883行的FP8 values/scales传输、graph payload更新及56个STATE/PAGE区域。真实模型运行证明完整调用链及请求完成；未运行本轮GSM或完整模型数值误差验收。

## 成本证据与限制

普通GPU事件稀疏采样显示，rank0 M1536 target均值51.7007→47.5696ms（新版本四rank范围47.5178–47.6089ms）。prefill target估计总时每臂221.095→185.682秒，decode target估计198.663→186.747秒，after-target约15.76→15.85秒。

这些是分桶每64次采样的估计。新版本估计enclosing总时／实际wall为1.0354–1.0364，不能把估计分项相加当作整网时间的精确分解，也不能把剩余差额归给某个未测算子。现有证据仍不支持“已经达到理论极限”。

正式阶段**每rank新增4个Triton特化**，四rank的完整key摘要一致；零冷编译检查FAIL。未记录独立编译耗时，原始374.837秒全部保留，没有推算扣除。旧基线也有冷事件。后续准备须覆盖这四条已观察签名，不能把本次单臂直接提升为稳定验收。

新启动入口仅在server/spawn/显式entry安装服务逻辑，避免ROCm辅助Python重复导入引擎。本轮模型预热约104秒，上一诊断约301秒；这是启动时间，不计入吞吐收益。退出时四rank各释放48个注册图，再执行原生退出，全部返回；容器rc0、退役，8卡利用率和显存归零、无Docker/KFD进程。没有推断未记录的内部allocation free。

## 下一步

1. 在新的准备版本中补齐四个已观察冷签名，并保留当前完整记录；不重复当前冻结run/reviewer。
2. 以此次实际成本为基线，核算FP8 owner→完整FFN→下一消费者与大prefill窗口。整网前先完整链筛选，预算必须包括新增工作和状态回接。
3. FP8原生ABI支持独立Q/scale/output stride，分片copy/concat可作为明确删除工作的小候选；[静态预算](../../validation/dashboard_whole_20261029/FP8_OWNER_STRIDE_CANDIDATE.json)尚无GPU验证、实现或排队，逻辑字节不等于节省时间。
4. EP4部署、packed前缀政策各自独立迁移／对照；不混入本次EP1/full收益。更大范围研究目标仍未达成，不承诺2×。

## 代码归并与运行入口

2026-10-29：六份已测迁移源码已归入`atom/`的可选owner路径。与冻结R1/R3版本相比，六文件仅调整导入顺序和空行，计算、调用参数、状态写入及graph释放方法的AST一致。完整工作负载证据仍来自上述冻结R3，并非归并后新跑的整网成绩。

`tools/dsv41/run_dashboard.py`默认`--execution native`；显式添加`--execution fp8-owner`启用TP4/EP1、FP8 KV、真实DSpark5、full容量及retained-prefill。入口创建`owner/capacity.txt=full`，不启用旧EP4或packed前缀策略。新增bootstrap只在server和spawn worker安装owner及退出处理，配置preflight、benchmark客户端和ROCm辅助进程不安装。退出顺序复用R3：同步→释放注册图→GC→TP CPU barrier→原生exit；去除实验文件IO并增加重复安装保护。

例如，在已有兼容ROCm/AITER环境中先生成可审阅配置：

```bash
python tools/dsv41/run_dashboard.py plan --execution fp8-owner \
  --model /models/DeepSeek-V4.1-Flash \
  --aiter-source /path/to/compatible/aiter --output /path/to/new-plan
```

将`plan`改为`serve`并使用独立新输出目录即可启动；默认profile的benchmark是公开512次预热协议，不是本次受控R3的593条独立预热、分段AL/行数统计和零冷审核流程。入口不是受控性能验收器，也不保证此配置下零冷编译。AITER必须提供V4.1 FP8 MLA；本次运行的实际依赖见[FP8依赖记录](../../validation/dashboard_whole_20261029/FP8_DEPENDENCY.json)，不能仅用旧镜像内置AITER替代。

归并验证：16项CPU入口、部署拒绝、graph幂等释放和退出顺序测试通过；Black/Ruff通过。此主机没有torch/pytest，未在本轮执行依赖torch的测试或重新编译GPU代码。复用原组件、真实模型和完整服务证据，不重复冻结run/reviewer；新入口与去诊断IO后的生命周期包装器当前为CPU接入验证。

原生产文件可由Git提交`f2d3ee47f7f865b85b085899d750306895b969b6`的blob恢复，旧入口副本另存仓外。[源码退役映射](../../validation/dashboard_whole_20261029/FP8_OWNER_INTEGRATION_SOURCE.json)及[六文件等价记录](../../validation/dashboard_whole_20261029/FP8_OWNER_INTEGRATION_EQUIVALENCE.json)保留路径与SHA；历史SOURCE/REVIEW不修改，旧reviewer不在新HEAD重跑。

证据：[正式review](../../validation/dashboard_whole_20261029/fp8_owner_service_r3/REVIEW.json.gz)、[成本与工作量](../../validation/dashboard_whole_20261029/FP8_MIGRATION_COST_RESULT.json)、[真实模型诊断](../../validation/dashboard_whole_20261029/fp8_owner_service_r2/REVIEW.json.gz)、[组件验证](../../validation/dashboard_whole_20261029/fp8_owner_r1/REVIEW.json)、[退役及节点空闲](../../validation/dashboard_whole_20261029/fp8_owner_service_r3/FINAL_NODE_IDLE.json)。
