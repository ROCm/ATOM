# MiniMax-M3：Gluon + EAGLE3-GQA + Mooncake Store

2026-10-20。分支 `mengqing/minimax-m3-gluon-gqa-mcstore`，基于 P/D Gluon 提交 `f878a144b30a33beec4de7fd691c49ab80bceb65`。两项实验使用同一 ATOM 提交及固定镜像 `rocm/atom-dev:nightly_202610141644`。

| 配置 | C32 | C72 |
| --- | --- | --- |
| P | TP4，4 GPU | TP8，8 GPU |
| D | TP4，4 GPU | TP8，8 GPU |
| 总资源 | 8 × MI355X，单节点 | 16 × MI355X，两节点 |
| PP | 1 | 1 |
| Target / draft CUDA Graph | 开启 | 开启 |
| Draft | Inferact/MiniMax-M3-EAGLE3-GQA | 相同 |
| Speculative tokens / mean AL | 3 / 2.78 | 相同 |
| CPU KV offload | Mooncake Store，1536 GiB | 相同总预算 |
| Store 分池 | 共享池，2 × 768 GiB owner | 每 NIC 一池，8 × 192 GiB owner |
| Indexer CP | 开启 | 关闭，当前实现只支持 TP=4 |
| Case | minimax-m3-gluon-gqa-mcstore-tp4-c32 | minimax-m3-gluon-gqa-mcstore-tp8-c72 |

C72 按最终确认使用两侧 TP8。当前 PP 需要 eager，两侧 TP8 可以保留 CUDA Graph，也避免了 TP4→TP8 的异构 KV head 路由。每侧 TP8 中 rank 0/1、2/3、4/5、6/7 分别复制同一个 KV head；P/D 按相同 rank 对接，head 归属一致。Draft 在所有 TP ranks 上分片，没有 PP stage。

## Kernel 与 KV 格式

沿用 AITER `4fe2800a5c252adff8ee28d199bea66f8b18efac`：官方 v0.1.26 Gluon unified attention + context split，以及现有 BF16 Q / 原生分页 FP8 KV 的最小 scale 兼容。官方 split、tile、双缓冲、attention 数学与 reduction 通过 source fingerprint 校验，没有更改 AITER。

ATOM dense cached-prefill 和 dense decode（target / draft）直接读取原生分页 KV；MiniMax sparse attention 保持现有专用实现。没有 gather KV workspace 的预留或整池格式转换。TP4 的 target / GQA draft 每 rank 为 Q16/KV1；TP8 为 Q8/KV1。Target 使用 per-token scales，draft 继续使用运行时 FP8 KV 和 scalar scale；BF16 draft 权重不代表 KV 改成 BF16。

两项保留 FP8 KV、FP8 index、block128、SLRU、GPU memory utilization 0.95、max_num_seqs 80、max_num_batched_tokens 32768。Target graph ladder 共用 `[1..18,20,22,24,26,28,30,32,34,36,40,48,56,64,72,80]`，draft graph 开启。P→D 使用 Mooncake RDMA，所有参与方 `MC_NUM_QP_PER_EP=1`。

## Store 与启动修复

将先前分支的 M3 Store PAGE codec 支持移植到此分支：按完整 PAGE 区域存取 K/V、scales、NSA index 和 draft，沿用现有 GPU 打包/解包与 Store 传输层。拒绝缺少 PAGE 元数据或混入 SLOT 状态。

C32 的 P 使用 GPU0–3，Store owner 使用 rdma4–7。C72 的 P 使用全部 8 个 GPU/NIC，采用现有 per-NIC 分池模式，每个 rdma0–7 分配 192 GiB，确保 1536 GiB 总预算都能被 P 使用。C72 的 owner 放在 P 节点，D 不启动额外 owner。各次任务启动独立 masters/owners，退出时清理，缓存不跨实验共享。

移除旧 Store 启动流程的 page-cache 清理、NUMA 内存准备、THP compaction 和后台清理进程。保留 owner NUMA/NIC 绑定、内存注册、ready capacity、进程存活监控与退出清理。

## Draft 准备与验证

目录固定为 `/share_nfs/models/Inferact/MiniMax-M3-EAGLE3-GQA`，HF revision 固定为 `96692486b5fd38ebf8fd2a5f6bb53427d30819a8`。该版本含 `config.json` 和约 6.15 GB 的 `model.safetensors`；tokenizer 来自 target。

提交节点在申请 Slurm GPU 前运行 `prepare_m3_gqa_draft.py`：

1. 文件锁串行化同目录的两个准备任务。
2. 已有文件通过模型结构、文件大小、HF Git/LFS 内容 hash 校验后复用。
3. 缺失或损坏时调用 Python SDK `huggingface_hub.snapshot_download`，不用 HF CLI。
4. 完成后原子写入固定版本的 receipt；GPU 节点启动时再次验证目录和内容。

昂贵模型加载前还执行官方 AITER 来源检查、GQA TP4/TP8 writer 与 Gluon 数值检查、变化 batch/context/page 的 CUDA Graph replay，以及 M3 Store GPU 往返校验。数值 gate 不使用 synthetic acceptance。跳过 GPU Store 测试会使门禁失败。

本地验证包括 223 项 ATOMesh/适配层 CPU 测试（修复新增模型校验测试发现的问题后全部通过）、186 项 Store CPU 测试、两项真实矩阵和 Slurm dry-run，以及 12 组实际 adapter/dispatcher 参数的 gfx950 离线编译。离线编译容器无 GPU，不能替代 CI 上的数值执行。

## 性能统计与比较边界

AgentX 使用 1M context、393 entries、seed42、10 warmup/lane 和 3600 秒正式测试。`--spec-decode-acceptance-length 2.78` 表示含 target token 的平均输出长度，K=3 的 per-position acceptance 为 `[1.0,0.78,0.0]`，每轮整数输出的平均值趋近 2.78。它是 synthetic 性能实验，不评估生成文本质量。

C32 与 C72 同时改变并发、卡数、TP、节点通信、indexer CP 和 Store 分池方式，不能用于单独归因 Gluon、扩卡或 TP 的收益。与旧 gather/旧 draft 基线对比，还包含 GQA draft、固定 AL 和 Store 等变化。

TP8 超过全局 4 个 KV heads，KV head 会复制，因此每 rank 的逻辑 KV block 字节数与 TP4 相同；扩卡收益主要来自权重等开销降低，不能把逻辑 KV 容量直接按 2 倍估算。Store 的 1536 GiB 是物理总预算：TP8 会存储更多复制的 rank 数据，同样的总字节预算也不代表相同的逻辑 token 容量。

## 提交方式

手动触发 `atomesh-benchmark.yaml`，`suite=gluon_gqa_store`，分别选择表中的两个 case，`atom_source=mounted`、固定上述镜像、`atomesh_slurm_submit_runner=atomesh-cicd`、`atomesh_slurm_partition=amd-spur`、`publish_dashboard=false`。两次触发应使用同一提交。
