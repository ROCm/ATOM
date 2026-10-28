# DeepSeek-V4.1 fusedmoe：R13 完整服务配置

唯一交付分支为 `yifehuan/dsv41_fusedmoe`。模型实现位于 `atom/models/deepseek_v41/`；固定运行profile位于 `tools/dsv41/r13_profile.json`，运行入口为 `tools/dsv41/run.py`。不依赖个人目录中的历史experiment入口。

## 测量范围

当前分支已 rebase 到本轮开始时锁定的 `origin/main = ebd5f84f3`。rebase 后的 **C5** 已完成 ABBA（保留同一新主线的 C4 A1，修复后运行 B1/B2/A2）：主线 **5741.094198**、候选 **7603.533722** 输出 tok/s，**1.324404976×**；平均 TPOT **42.541209→31.636498 ms**。两对 **1.315218× / 1.333702×**，候选 B2/B1 为 **1.012029851**，主线 A2/A1 为 **0.998004086**。受测提交 `fc4da1513`；四臂各 593 warm / 2560 formal，输入 18,859,319、输出 2,357,150 token，正式新增 JIT/图捕获为零，完整图键保留。主线 Spec0、候选 DSpark5，这是完整服务组合收益；本轮没有新的 GSM8K 整网精度验收。

测试期间远端 main 又前进到 `773eb3625`（新增缓存卸载/公共调度接口和 gfx1250 两项提交）。本轮以开始时锁定的 `ebd5f84f3` 收尾，新增提交已记录在 `NEW_MAIN_DURING_C5.json`，尚未纳入本轮分支或性能验收；不能将 C5 成绩标为 `773eb3625` 的结果。

历史 C3 基于旧主线 `713eb0614`，1.322416951×，受测提交 `d52127533`；其证据保留，不代替新主线验收。

R13原始完整ABBA相对主线`713eb0614`为1.324572385×：5755.113309→7623.064164输出tok/s，平均TPOT42.427437→31.556413ms。该结果属于owner/packed＋retained prefill＋grouped attention＋预量化FFN＋EP4＋local router/packet＋固定专家分配的完整组合。原始候选2053个仓库文件在收敛时逐字节核对；整理后的独立性能复核记录在`validation/consolidation_20261028/`。

这不是单个comm-fused kernel的消融结果，也没有新的R13 GSM8K整网精度结论。固定专家分配用于已测工作负载，不承诺其他路由分布获得同样倍率。

## 环境

- 4张MI355X/gfx950；本次使用8卡独占节点中的0–3卡。不要与其他GPU工作负载重叠。
- 历史镜像：`sha256:32c81e56e4680065b6ea3e0ca8363a00cd3ccb932d8990e83d799b8bbb58a4b3`。
- AITER：`729feefb7210ebc3a87ebd00a44e0c43f30bf3f2`，使用该依赖原生算术。grouped attention的私有native包装代码在本分支，保留原生device模板与编译参数。
- checkpoint在容器内挂载为`/models/DeepSeek-V4.1-Flash`。Python环境需具备对应torch、Triton、FlyDSL、AITER和benchmark依赖。
- attention TP4、routed专家EP4、shared TP4，BF16 KV、FP4 index、DSpark5、level3/FULL、max-num-seqs256、memory utilization0.85。

## 运行

下列命令在已配置GPU/Python环境的仓库根执行。`/path/to/aiter`必须是上述依赖；输出路径须是本次独立目录，不覆盖旧结果。

```bash
python tools/dsv41/run.py preflight \
  --aiter-source /path/to/aiter --output /path/to/run/entry
python tools/dsv41/run.py client-preflight \
  --aiter-source /path/to/aiter --output /path/to/run/client
python tools/dsv41/run.py benchmark \
  --aiter-source /path/to/aiter --output /path/to/run/B1 \
  --cache-dir /path/to/private-cache
```

`benchmark`会启动并正常退出一个服务器，执行593个预热请求、2560个正式请求。输入random8192、输出1024、range0.8、正式seed2026102518；ignore-eos，实际总token数以原始请求结果为准。候选预热顺序为4/512/64/12/1，覆盖owner小容量及完整图键；主线为512/64/16/1。

当前对照主线使用独立检出的`ebd5f84f3`：增加`--mode native --model-source /path/to/main-ebd5f84f3`。按A1/B1/B2/A2串行执行，合并吞吐按总输出token/总耗时计算。两侧使用同一环境、缓存准备和预热协议；正式新增JIT或图捕获使该次测量无效，不能扣除时间后报分。

仅启动服务可使用`serve`动作，地址为127.0.0.1:19125；该动作本身不代表完成了593次预热或性能验收。需要外部访问时单独建立已授权的代理，节点043本次无需反向代理。

新入口仅在`DSV41_CONSOLIDATED_ENTRY=1`时启用，子进程启动与worker spawn继承同一个bootstrap。主线和候选分别设置完整profile；benchmark client清除owner/EP服务标志，防止客户端触发模型部署检查。模型flags保持默认关闭，使用本recipe完整启用。

## 代码与证据

`tools/dsv41/acceptance/`只保留正式服务、预热与验证需要的代码。历史多级`sitecustomize`已收成一个显式安装顺序；旧GSM驱动、独立probe和trace driver的无调用入口已移除。文件中部分历史名字暂时保留，避免同时改变已测函数绑定；它们是固定验收依赖，不是多个候选方案。

不要删除`owner_head_attention.py`：它是grouped attention的不支持形状回退。`FFN_COST_FIXED_POLICY.json`是实际owner策略输入，`owner_attention_observation.py`是具名诊断接口，均沿调用链保留。原始用户的3份旧未提交改动单独保留，不混入R13性能基线。22 个冗余 worktree 和 10 个本地分支已移除，恢复方法见项目验证目录 `consolidation_20261028/RECOVERY.md`。

- [大并发完整链路架构方案](../docs/dsv41_fusedmoe_architecture.md)
- 原始性能：项目验证目录`STEP12_PERFORMANCE_R13_RESULT.json`与`FINAL_RECHECK.json`。
- 新主线性能与最终复核：`rebase_main_20261028/STEP12_PERFORMANCE_C5_RESULT.json`、`FINAL_DELIVERY_RECHECK.json`；清理恢复：`consolidation_20261028/RECOVERY.md`。
- 精度：GSM8K与相对L2；保持各自适用范围，不把复制一致或旧任务分数当新整网验收。
