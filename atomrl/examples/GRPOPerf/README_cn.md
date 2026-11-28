# GRPOPerf：在 MI355X 上跑 NeMo-RL v0.8 性能表的 GRPO 场景

用 AtomRL（Megatron native 训练 + ATOM 生成）在 2 x 8 MI355X 上跑
[NeMo-RL v0.8 性能表](https://docs.nvidia.com/nemo/rl/latest/about/performance-summary.html)的 8 个场景，
指标按 NeMo-RL 的公式重算。[English](README.md)

## 例子

| 别名 | `run.sh` id | 模型 | 模式 | 训练 | 生成 | 机器 |
|---|---|---|---|---|---|---|
| B1 | `grpoperf-qwen3moe-onpolicy-2node` | Qwen3-30B-A3B | on-policy | Megatron EP8 x DP2 | ATOM BF16 TP1 x 16，共置 | 2x8 |
| B2 | `grpoperf-qwen3moe-1stepoff-2node` | Qwen3-30B-A3B | 1-step off | Megatron EP8，第 1 台 | ATOM BF16 TP1 x 8，第 2 台 | 2x8 |
| B3 | `grpoperf-qwen3-8b-1stepoff` | Qwen3-8B | 1-step off | Megatron TP1 x DP4，4 卡 | ATOM BF16 TP1 x 4，4 卡 | 1x8 |
| B4 | `grpoperf-qwen3-8b-onpolicy` | Qwen3-8B | on-policy | Megatron TP1 x DP8 | ATOM BF16 TP1 x 8，共置 | 1x8 |
| F1 | `grpoperf-qwen3moe-onpolicy-fp8gen-2node` | Qwen3-30B-A3B | on-policy | 同 B1 | ATOM `ptpc_fp8` | 2x8 |
| F2 | `grpoperf-qwen3moe-1stepoff-fp8gen-2node` | Qwen3-30B-A3B | 1-step off | 同 B2 | ATOM `ptpc_fp8` | 2x8 |
| F3 | `grpoperf-qwen3-8b-1stepoff-fp8gen` | Qwen3-8B | 1-step off | 同 B3 | ATOM `per_block_fp8` | 1x8 |
| F4 | `grpoperf-qwen3-8b-onpolicy-fp8gen` | Qwen3-8B | on-policy | 同 B4 | ATOM `per_block_fp8` | 1x8 |

每个场景按 NeMo-RL 的工作量和 GRPO 算法跑 10 步：每步 64 个 prompt x 32 条 = 2048 条，prompt + 回复 ≤ 4096，数据 OpenMathInstruct-2，
每批一次优化器步，Adam lr 3e-7、warmup 50 步、从 3e-8 起，对初始模型的 KL 0.01（k3），留一法优势，Math-Verify 奖励；MoE 的 router 冻结、用 fp64，专家重排用融合 kernel（都是 NeMo-RL 的默认设置）。
训练 BF16；FP8 行只量化生成（注意力、路由、`lm_head` 保持 BF16）。F1、F2 依赖 ATOM 在权重同步时重新量化 FP8 专家。
检查项要求第 8 步的 `perf/throughput` 在该场景默认配置结果（`examples.yaml`）的 10% 以内。

有两项设置会改变每步的工作量，每个场景都照表里的 NeMo-RL 配置：old log-prob 是否取自训练前向（`force_on_policy_ratio`），loss 是否带对 rollout 的重要性采样（IS）修正。
NeMo-RL 没有 Qwen3-8B 的行，B3、B4、F3、F4 照它的 Qwen3-32B 配置。

| # | NeMo-RL 配置 | old log-prob | IS 修正 | 训练之外的前向 |
|---|---|---|---|---|
| B1 | `grpo-qwen3-30ba3b-4n4g` | 取自训练前向 | - | 参考模型 |
| B2 | `grpo-qwen3-30ba3b-4n4g-async-1off` | 取自训练前向 | token | 参考模型 |
| F1 | `grpo-qwen3-30ba3b-4n4g-mxfp8-rollout` | 单独前向 | token | old、参考模型 |
| F2 | `grpo-qwen3-30ba3b-4n4g-async-1off-mxfp8-rollout` | 单独前向 | token | old、参考模型 |
| B3 | `grpo-qwen3-32b-8n4g-async-1off` | 取自训练前向 | token | 参考模型 |
| B4 | `grpo-qwen3-32b-4n4g` | 取自训练前向 | - | 参考模型 |
| F3 | `grpo-qwen3-32b-8n4g-async-1off-mxfp8-rollout` | 取自训练前向 | token | 参考模型 |
| F4 | `grpo-qwen3-32b-4n4g-mxfp8-rollout` | 取自训练前向 | token | 参考模型 |

参考模型前向用 actor 自己的模型、临时换入初始权重（`controller.ray.ref_in_actor`），计算量和一次策略前向相同。

## 结果

`rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125` 上第 3–8 步，每个场景用最快的并行配置（用 `atomrl/.claude/skills/` 里的 `atomrl-strategy-search` skill 搜出来），各计入一次运行；计入的运行都通过 NeMo-RL 的正确性门槛。
运行见 [measured/runs.csv](measured/runs.csv)。每行每步的工作量和 NeMo-RL 相同：生成、上表里的前向、训练。

| # | 卡数 | 配置 | 生成 [TP,PP] | 训练 [TP,CP,EP,PP] | 平均生成长度 | Tokens/s/GPU | 步时 (s) | NeMo-RL GB200 | 比值 |
|---|---|---|---|---|---|---|---|---|---|
| B1 | 16 | 共置 | [1,1] | [1,1,8,1] | 3,164 | 2,543 | 165.0 | 2,656、159 s | 0.96 |
| B2 | 12+4 | `B2-12P4`：EP4 x DP3、4 个 ATOM 副本 | [1,1] | [1,1,4,1] | 3,169 | 2,800 | 150.1 | 2,031、208 s | 1.38 |
| B3 | 4+4 | 同「例子」 | [1,1] | [1,1,1,1] | 3,251 | 6,189 | 139.2 | - | |
| B4 | 8 | 同「例子」 | [1,1] | [1,1,1,1] | 3,251 | 5,412 | 159.3 | - | |
| F1 | 16 | 共置 | [1,1] | [1,1,8,1] | 3,163 | 2,335 | 179.6 | - | |
| F2 | 12+4 | `F2-12P4`：EP4 x DP3、4 个 ATOM 副本 | [1,1] | [1,1,4,1] | 3,165 | 2,296 | 182.8 | 1,819、232 s（MXFP8） | 1.26 |
| F3 | 4+4 | 同「例子」 | [1,1] | [1,1,1,1] | 3,248 | 6,155 | 139.9 | - | |
| F4 | 8 | 同「例子」 | [1,1] | [1,1,1,1] | 3,247 | 5,452 | 157.9 | - | |

- B1 每步：生成 71.1 s、参考模型前向 22.9 s、训练 64.9 s、权重同步 5.8 s；F1：生成 61.0 s、old log-prob 前向 23.3 s、参考模型前向 23.0 s、训练 65.6 s、权重同步 6.5 s。F2 的关键路径在训练端：old log-prob 前向 32.7 s、参考模型前向 32.2 s、训练 110.8 s、权重同步 6.9 s，生成（114.1 s）和它们重叠。B2 也是：参考模型前向 32.4 s、训练 111.2 s、权重同步 6.3 s，生成 128.4 s。
- B3 两端差不多：生成 133.9 s，对参考模型前向 31.4 s、训练 101.2 s、权重同步 2.4 s。计入的运行取自同一配置 200 步运行的第 1–10 步（warmup 之后学习率不变，也不做验证、不存 checkpoint），这 10 步和单独跑 10 步相同。F3（B3 加 FP8 生成）两端同样差不多：生成 134.6 s，对参考模型前向 31.5 s、训练 100.8 s、权重同步 2.5 s，所以 FP8 生成缩短不了它的步时。
- F4 每步：生成 86.1 s、参考模型前向 17.3 s、训练 51.0 s、权重同步 3.3 s。FP8 生成对 Qwen3-8B 只比 B4（87.4 s）快约 1 秒。
- NeMo-RL 没有 Qwen3-8B 的数据（它的对应行是 Qwen3-32B）。
- NeMo-RL 的 fp64 router 在这里让 Qwen3-30B-A3B 的训练多花约 11%：B1 的布局放到一台机器上（`B1-1N`，第 2–4 步），用它每步训练 106.4 s，换成 fp32 router 是 95.2 s，参考模型前向 40 s 对 36 s。
  KL 项约多 2%（去掉后 104 s）；冻结 router 对耗时没有可测出的影响。两者都按 NeMo-RL 的设置保留。

## 长跑

用这版代码把 B4 和 B3 各跑 200 步（`B4-200`、`B3-200`；B3 在结果表里的那一行就是它的第 1–10 步）。每 20 步的汇总在 [measured/longrun.csv](measured/longrun.csv)。

| 200 步 | B4（on-policy，9.1 h） | B3（1-step off，7.9 h） |
|---|---|---|
| `token_mult_prob_error`，每 20 步的中位数 | 1.013 | 1.013 |
| 训推 KL（`rollout_corr/kl`） | 一直 7.7e-4 | 一直 7.6e-4 |
| `ref_kl`，第 1–20 / 181–200 步 | 7.3e-4 / 1.21e-3 | 7.3e-4 / 1.26e-3 |
| entropy | 0.25 | 0.25 |
| 回复长度，第 1–20 / 181–200 步（没有截断） | 3,311 / 3,207 | 3,312 / 3,203 |
| 显存峰值 | 222 GB，不增长 | 237 GB，不增长 |
| 超过 1.1 的步（单个 token）/ 最后一步 | 200 步里 8 步 / 1.013 | 200 步里 9 步 / 1.013 |

- 训推 KL 一直不变，说明 200 次权重同步都对。
- `ref_kl` 到第 80 步前一直在 7.5e-4 左右，之后平滑上升。这个底是数值上的：权重只差几个最低位时，参考前向和训练前向就不再逐位一致，BF16 的舍入差异逐层累积到约 7.5e-4（学习率接近 0 的对照一直低于 1e-4）。只有超过这个底的部分才是策略离开初始模型的量。
- 在 NeMo-RL 的学习率（3e-7）下，200 步里准确率还没有变化（0.48–0.51）；B3 的重要性采样权重均值是 1.0000。这两个长跑说明的是不发散、不漂移、权重同步不出错。

## 运行

在 `atomrl/` 下，配一个站点文件（site file：写明在哪些机器上跑、怎么连过去、数据放在哪，见 [site.example.yaml](../site.example.yaml)；2x8 的场景要两台主机和 `rdma` 一节）：

```bash
bash examples/run.sh prepare B4 --site site.yaml   # 一次：Qwen3-8B、Qwen3-30B-A3B 和数据
bash examples/run.sh B4 --site site.yaml --atom-src ..
python3 examples/GRPOPerf/metrics.py <训练日志> --gpus 8   # B1、B2、F1、F2 用 16

# 8 个场景依次跑完，约 5 小时
bash examples/run.sh suite --tier long --only B1,B2,B3,B4,F1,F2,F3,F4 --site site.yaml --atom-src ..
```

B2、F2 两行用条目 `B2-12P4`、`F2-12P4` 跑：12 张训练卡（8 + 4），第二台剩下的卡上跑 4 个 ATOM 副本；其余各行就是场景本身。
`B1-1N` 在一台机器上跑 B1 的共置布局，用来做训练端的实验。

```bash
bash examples/run.sh B2-12P4 --site site.yaml --atom-src .. --no-check
```

`prepare` 按 DAPO 例子固定的 revision 下载两个模型（就是测量用的版本），再用 [prepare_data.py](prepare_data.py)
生成 NeMo-RL 的数据：OpenMathInstruct-2 `train_1M`（固定 revision），95/5 切分（seed 42）和 `cot.txt` 提示，95 万 / 5 万条。
每个场景约 30–55 分钟。每次运行和其他例子一样判定（退出码、跑到最后一步、权重同步、吞吐检查），结果在 `atomrl/output/regression/<时间>-<label>/`（`summary.md` 和日志）；`metrics.py` 从训练日志算出表里各列和 NeMo-RL 的门槛。
要用 NeMo-RL 的布局，例如给 B1 加 `--override policy.training.megatron_cfg.expert_model_parallel_size=16`。

数据是用 `rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125`（`4ede265ad0f1`）、用 `--atom-src ..` 挂上这个 checkout 测的：F1、F2 要用它的 FP8 专家重新量化，镜像里的 ATOM 还没有。
共置的场景（B1、F1、B4、F4）把 `PYTORCH_CUDA_ALLOC_CONF` 设为空：这个镜像的 torch 上 `expandable_segments` 会生效，sleep 的共置 ATOM 就不再归还显存。

## FP8 训练

4 个 FP8 场景把训练也换成 FP8：TransformerEngine tensorwise current scaling、E4M3、首尾各一层保持 BF16，rollout 不变，配置同「例子」；
261125 上每个场景一次运行（[measured/runs.csv](measured/runs.csv)）。NeMo-RL 的 FP8 行训练是 BF16，它的门槛是针对 BF16 训练定的，这里不适用，改为列出偏差数值。
这组数据是在上面的 NeMo-RL GRPO 对齐之前测的（KL 为 0、组内均值 / std 优势、字符串匹配奖励），还没有用对齐后的代码重测。

| # | Tokens/s/GPU | 步时 (s) | 训练 (s) | 对 BF16 训练 | `k3_kl` | `token_mult_prob_error` 中位数 / 最后一步 |
|---|---|---|---|---|---|---|
| F1 | 2,370 | 176.8 | 78.8 | -12.9% | 0.0069 | 1.20 / 2.74 |
| F2 | 2,076 | 202.0 | 138.8 | -22.8% | 0.0069 | 1.70 / 1.06 |
| F3 | 6,315 | 136.1 | 75.7 | +0.6% | 0.0074 | 1.95 / 1.96 |
| F4 | 5,968 | 144.3 | 39.2 | +6.0% | 0.0074 | 3.15 / 30.2 |

- “对 BF16 训练”指同一配置、训练用 BF16（2,719、2,688、6,278、5,631 tokens/s/GPU）。F2 是 NeMo-RL F2（1,819，FP8 生成 + BF16 训练）的 1.14 倍。
- Qwen3-8B 训练快 18–19%；Qwen3-30B-A3B 的分组专家 GEMM 用 FP8 反而比 BF16 慢（hipBLASLt 报告没有调优过的 kernel）。F3 的步时受生成限制。
- BF16 训练时 `k3_kl` 为 0.0028–0.0034：FP8 训练按张量量化，rollout 按 block（8B）或按 token 和通道（30B）量化。
- 这个镜像上 TE 不支持 blockwise（`Float8BlockScaling`）和 MXFP8。DAPO 例子 3、4 的 FP8 训练用的是 FSDP2 上 Lumen 的 blockwise kernel，Megatron native 用不到。

```bash
bash examples/run.sh F4 --site site.yaml --atom-src .. --no-check \
  --override quantization.training.fp8=e4m3 \
  --override quantization.training.fp8_recipe=tensorwise \
  --override quantization.training.num_first_layers_in_bf16=1 \
  --override quantization.training.num_last_layers_in_bf16=1
# F1 另加：--override policy.generation.atom_cfg.gpu_memory_utilization=0.52
#（FP8 训练多占显存，0.60 时 ATOM 唤醒会 OOM）
```

## 指标

- Tokens/s/GPU = 一步的 token 数（prompt + 回复，2048 条全算）÷ 步时 ÷ 全部 GPU（含生成卡），AtomRL 日志里就是 `perf/throughput`。
  平均生成长度 = 平均回复 token 数。
- 门槛：`token_mult_prob_error`（exp|生成 logprob − 训练 logprob| 的平均）10 步的中位数 < 1.1，且最后一步 < 1.1。

## 和 NeMo-RL 的差异

算法相同，每步的工作量也相同；不同的只是东西放在哪里、怎么跑。

- 参考模型的权重一直放在 GPU 上，参考前向时原地交换进去；NeMo-RL 把它放在主机内存里，每步拷进拷出。优化器也不 offload：MI355X 的显存放得下。
- 1-step off 在更新权重前等正在生成的那一批结束；NeMo-RL 在生成中途更新权重（`in_flight_weight_updates`），它有一部分轨迹混合了两个策略。
- FP8 生成用 ATOM 的 `ptpc_fp8`（30B）和 `per_block_fp8`（8B），NeMo-RL 用 MXFP8，保持 BF16 的层相同；ATOM 的权重同步还不能把专家重新量化成 MXFP8。
- 梯度规约用 FP32（NeMo-RL 用 BF16）。
- 每行用最快的并行配置（写在表里），不是 NeMo-RL 的布局。
- NeMo-RL 每 10 步验证一次；这里不验证，不影响训练。
- prompt 来自同一份数据、同样的切分和 seed，打乱的顺序不同。
