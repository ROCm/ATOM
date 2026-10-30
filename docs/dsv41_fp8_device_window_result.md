# 设备内 G2／本地归约／IPC RS Window：完整链结果

2026-10-29 14:14 UTC。**新的设备阶段融合候选已经实现并完成三种行数的真实权重 A/B；三档均退化，停止此版本，不接入生产或服务。** TP shared 漏归约问题已修正并通过 GPU 合同检查。两次新增 FP8 压缩的误差已单独记录，不能继承原有精度结论。

## 实际配置和执行范围

保持当前 dashboard 配置：TP4/EP1/DP1、FP8 KV、FP4 index、真实 DSpark5 的配置，无 DEP/DPA、无合成接受率。这里运行的是 layer8 完整 FFN → owner 输出 → layer9 实际 mHC/norm/QKV，使用真实全部 384 专家、shared、gate 和 mHC/QKV 权重；不运行服务 KV、接受/拒绝与回滚。

M1536 对照使用已经验证的固定 gate、owner 一次量化供料；两档 prefill 对照使用当前 native owned prefill。候选的实际 G1 callable、参数、标量、输入形状和 FP8 值均与对照一致。G1 仍执行完整 M/K，G2 使用完整 K640 和真实 TP 权重切片；G2 的实现、输出格式与执行顺序属于本次变更。

每档三组改变输入，每组执行完整链 control/device/device/control，每 arm 128 次 graph replay。四 rank 共 72 个配对，无观察计时标记。R4 全流程 100.405 秒，只有这一次加载完整专家权重；三个更早的初始化/编译失败均发生在权重加载前。

## 本次确实删除和新增了什么

固定十个 N512 阶段。两个紧凑 route slot 交替承接 G2；第 j 阶段计算新的列窗口、归约前一窗口、通过 IPC 读取更早窗口的各 rank partial。归约结果直接写入完整 owner H5120 的最终列位置，随后运行下一层消费者。

删除了上一版的十次独立 RCCL RS、十次独立 join/compact/pad，以及最终 stitch；不再分配完整 M×6×H 的 BF16 route。没有把删除逻辑字节数计为性能收益。

保留的边界仍有 **10 个带 G2 的 dispatch、2 个 drain 和 10 个跨 rank epoch barrier**。这是设备阶段装配，并非整个 FFN 的单次 launch。G2 与 local 服务共享编译资源，另有 RS service CTA。新增计算包含加权 route 的 FP8/group8 编码、解码及 top6 求和，以及 local partial 的 FP8/group32 编码和跨卡解码。

TP shared 的修正是：每个 rank 的 full-M shared 先参与本 rank partial，再归约全部 rank；RS 后不再额外加 owner shared。保留 routed 求和后 BF16 舍入、与 shared 相加后的 BF16 舍入，然后才做新增 partial 压缩。普通 Window-RS 的“归约后加 owner shared”不再适用。

两档 ragged M 的通信容量为 7552/16292；G2 的 token 上界仍为真实 7550/16291，G1、排序 sentinel 和输入布局不变。route 尾部保持零，shared 读取以真实 M 为界，最后 owner 的补齐行保持零。

## 同次完整链性能

| 全局 M | 当前对照 ms | 设备 Window ms | 对照/候选 | 候选时间增加 | 净新增时间 |
|---|---:|---:|---:|---:|---:|
| 1536 | 0.729784 | 0.766178 | 0.952499× | 4.987% | 36.395 µs |
| 7550 | 2.044954 | 2.259060 | 0.905223× | 10.470% | 214.106 µs |
| 16291 | 3.894777 | 4.241045 | 0.918353× | 8.891% | 346.268 µs |

所有 72 个配对均为负。三档最大的同组重复时间漂移为 0.362%、0.195%、0.170%，明显小于上述退化；不是收益落在噪声内。

上轮 BF16/RCCL 版本与本轮的执行格式、归约算术和测量时间均不同，不能直接用两轮绝对时间之差声称“删除 RCCL 赚了多少”。同次 native 对照才是本候选净收益的依据。

## 数值合同

权重加载前，三种形状、四个 rank、三组各 rank 不同的 shared-only 输入全部通过：route=0 时完整 owner 输出逐字节等于各 rank shared 之和，补齐行正确，graph 更新后无旧值。

真实权重下记录完整十项输出、输入不变、native-self、device-self 和 graph/eager。候选三档的 **device-self 与 graph/eager 十项相对 L2 全为 0**。对最后 1024 列，独立解码每个 rank 的 partial，按实际轮转 rank 顺序以 FP32 求和再转 BF16，与设备 RS 输出逐字节一致。这是本次 IPC/归约合同的局部证据，不是模型精度验收。

| M | 完整 owner 对原生相对 L2 | 原生自比较 owner L2 | route 压缩及 G2 改序误差¹ | 单独 partial 压缩误差² |
|---|---:|---:|---:|---:|
| 1536 | 0.033744–0.033889 | 0.000745–0.000953 | 0.026551–0.026650 | 0.026691–0.026749 |
| 7550 | 0.033686–0.033737 | 0 | 0.026379–0.026423 | 0.026691–0.026723 |
| 16291 | 0.033691–0.033732 | 0 | 0.026394–0.026418 | 0.026705–0.026721 |

¹ 最后 1024 列解码后的 route/top6 与同一 G1 输出驱动的原生 G2 比较，包含新 G2 编排和 route 压缩，不能全部归因于量化。

² 同一批已解码 route，加 TP-local shared、按 BF16 合同舍入后，与量化并解码的 partial 比较，隔离了第二次 FP8 handoff。误差约 2.67%；与第一列误差不能相加。

后续实际 query/kv 的相对 L2 约 0.01054–0.01072。确定性不代表与原生等价，不能把 owner 约 3.37% 的局部误差当作已通过 GSM8K；本次没有运行模型质量评估。

## 编译资源和缓冲

三形状各 13 个产物记录（12 个 Window 阶段及同一个 IPC epoch barrier），四 rank SHA 一致，gfx950 ELF/metadata 无损保留。实际 launcher 的 `_run_compiled` 复用预编译 `_cf`，真实权重阶段使用相同的工厂 config 和通用 uint8 指针 ABI。

| 阶段 | VGPR | SGPR | LDS | private/spill |
|---|---:|---:|---:|---:|
| decode G2 compute/cycle | 98–102 | 54–64 | 16 KiB | 0 |
| prefill G2 compute/cycle | 152–156 | 64–74 | 32 KiB | 0 |
| local+RS drain | 87 | 34 | 0 | 0 |
| RS drain | 78 | 24 | 0 | 0 |
| epoch barrier | 6 | 16 | 8 B | 0 |

没有寄存器 spill 解释这次退化的证据。静态资源也不能单独证明运行时 occupancy 或带宽已满。

| M | 两个 route slot | IPC partial workspace | 最终 owner 输出 |
|---|---:|---:|---:|
| 1536 | 10,616,832 B | 8,388,608 B | 3,932,160 B |
| 7550 | 52,199,424 B | 41,943,040 B | 19,333,120 B |
| 16291 | 112,610,304 B | 88,080,384 B | 41,707,520 B |

另有每档 320 B 的十列四 peer 地址表和 32 B 的 peer 基址表；表中不包含原生 G1、shared、图池与权重。显式 scratch 减少很大，但完整链性能仍退化。

## 结论、停止条件和下一候选约束

本轮已经否定了一个具体假设：**删除逐列 RCCL 和打包/拼接边界，再采用现成 FP8 Window 阶段装配，就足以获得净收益。** 它在三个主要行数上均不成立。此前 TP shared 语义问题也已经解决，不能再作为本轮未获收益的解释。

尚未分离的运行时成本包括：G2 改序/分列效率、两次量化转换、本地服务与计算竞争、IPC 读取效率、十个全 rank barrier 的等待。没有细分阶段或消融对照，不能把全部退化指定给其中一项。初始化故障只解释实现过程中的时间消耗，不解释 R4 测量中的模型速度。

按旧分布条件外推（3724×38 decode、1199×18 和 647×18 prefill，layer8 代表对应各层），本候选约增加 **13.804 秒，即当前 374.837 秒的 3.683%**；这是停止全范围集成的代理预算，不是新增整网成绩。

因此停止当前 FP8 Window 版本，不扫描 worker 数、列宽或继续沿用双重量化。若继续这个方向，下一个新候选必须同时针对两点：**紧凑 BF16 route/partial 保持明确数值边界；在生产者/消费者阶段内减少全 rank phase barrier 和重复转换。** 需要先明确如何保留大 M GEMM 的完整计算效率和前进保证，再进行新的有限完整链 A/B。仅把当前 payload 位宽开关改成 BF16，或只重新测量同一版本，不构成新候选。

owner 一次量化供料的已验证局部收益保留。最后服务样本仍为 6281.294 tok/s、1.191939×及原有 cold/time-gap/AL/执行行数/精度限制；本次无新增整网成绩，未修改生产运行代码。

## 记录与退役

验证目录：`validation/dashboard_whole_20261029/owner_device_window_r4/`。`SOURCE.json.gz` 81 个绑定，89 个导出 SHA 核对；`review.py` 仅执行一次，结果为 `REVIEW.json.gz`，停止决定为 `DECISION.json`。

R1：PyTorch symmetric-memory 8 MiB 分配返回 HIP OOM；R2：MORI CCO 初始化在 `cco_init.cpp:708` 返回 OOM，对应公开源位置为 flat VA reserve，尚未证明底层原因；R3：IPC 分配成功，新增 int64 peer 表指针缺少 align8，在首个 epoch barrier 编译时失败。R4 修正对齐后完成全部检查。各失败 source/raw/review 保留，不重跑、不改写。

R4 所有 graph reset 后关闭 peer IPC 映射，释放 workspace 及其视图并检查 weakref，再退役原生通信组。14:12:58 UTC fresh 检查无容器/KFD，八卡利用率和显存占用均为 0，借用作业 248783／085 保留。

为了本次准入，仅无损压缩了八个已完成的 source/preload/结果 envelope，回收 1,995,776 B；原始解码 SHA 与映射保存在 `OWNER_DEVICE_WINDOW_ARCHIVE.json`，旧 reviewer 未重跑。R4 启动保守总上界 9,999,494,304 B，单串行 tmpfs 976 MiB。
