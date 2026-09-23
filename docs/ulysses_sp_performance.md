# Ulysses SP：128K prefill 实现与验证

记录日期：2026-09-22；recipe 精度验收更新于 2026-09-23。基于 `hexwang/atom_sp`，基线提交
`9d81bb91`，包含本次工作区修改。本文件只记录当前源码与本轮实际运行结果，
不依赖已删除的 `ulysses_sp_handoff.md`。

## 目标与精度约束

优先优化原生 ATOM 的 131072-token prefill。先测试 Qwen3-30B-A3B-Instruct-2507
的 GQA + MoE，再测试原始 BF16 MiniMax-M3 的 sparse GQA + MoE。

SP 通信不得引入量化，不修改 attention mask、RoPE、KV cache dtype、稀疏索引
规则或 top-k。移除了原有 SP attention FP8 wire codec，以及 MoE combine 的
量化 QuickAllReduce 分支。保留输入 dtype 的 reduce-scatter 仍是浮点归约，
与 TP 的归约顺序可以不同；不能把“通信不压缩”表述成“整模型 logits 逐位相同”。

## 执行布局

SP 使用 PCP 的 rank 维度建立通信组，但不使用 PCP 的序列交错和 query 分片算法。
当前配置为 `TP=1, SP=W, DP=1`。本轮新增对 DP+SP 的启动检查，因为现有 DP MoE
forward 分支会跳过 SP gather/scatter，直接放行会产生错误结果。

每层的非 attention 部分处理连续的 `ceil(S/W)` 个 token，dense 权重复制；
MoE 权重仍分片。attention 前交换 token/head 维，之后各卡对**完整序列**上的
部分 query heads 执行原有 attention kernel，并持有对应 heads 的 KV cache。
attention 后反向交换，恢复本卡的连续 token 分片。因此不会出现 causal attention
按前后序列分片导致的计算量失衡。

```mermaid
flowchart LR
    A[本卡 token 分片<br/>完整 Q/K/V heads] --> B[Triton 打包字段]
    B --> C[All-to-all]
    C --> D[完整序列<br/>本卡 heads 的原生 attention]
    D --> E[反向 All-to-all]
    E --> F[本卡 token 分片<br/>完整输出 heads]
    F --> G[Dense / MoE]
```

### attention 通信

- 大消息使用 `sp_kernels.pack_fields`：单个 Triton kernel 将融合投影中的
  Q/K/V 和 M3 indexer 字段直接写入 `[destination, local_tokens, local_width]`。
  支持带 padding 的 row stride，避免 PyTorch gather/stack 的重复搬运。
- `all_to_all_into` 使用当前流上的 PyNccl grouped send/recv。每个 peer 使用原始
  buffer pointer 的偏移，避免每层构造两组 tensor views，也避免 ProcessGroupNCCL
  内部通信流与计算流之间的调度开销。通信顺序由当前流保证。
- 小消息继续使用 AITER 已注册 buffer 的 custom all-gather。所有选择条件都必须
  在各 rank 一致；shape、dtype、固定通信能力和 2 MiB 阈值满足这一要求。
- attention 输出反向交换后将 head 分片拼回本卡 token。非整除序列补零，进入
  attention 前去掉 padding，计算结束后恢复通信所需的 padding。

### GQA 和 M3 indexer

Qwen3 与 M3 都只有 4 个 KV heads。SP8 不得把一个 KV head 切成半个 head：
相邻 query-head owners 复制同一个完整 KV head。query heads 仍均匀分片。
启动时验证 query head 可整除，以及 KV head 可整除或可整倍数复制。

M3 的 index query heads 使用对应的复制规则；单个 index key head 复制到所有
rank。`Attention` 的 `index_q_size` 下限修正为一个完整 `index_head_dim`，
避免 SP 大于 index head 数时产生错误的 packed offsets。稀疏 block 选择继续由
原始模型配置控制，没有开启跨层 index 复用，也没有扩大 index 更新间隔。

### MoE 通信

未启用 routed EP 时，SP 先 gather hidden states 和 router logits，执行本卡的
专家权重分片，再 reduce-scatter 回本卡 token。大消息直接调用 PyNccl，
reduce-scatter 直接写最终输出，省去 AITER 通用维度转换封装的一次额外分配与 copy。

启用 EP 不等于启用 routed dispatch。`--enable-expert-parallel --all2all-backend none`
采用完整专家分片，但仍 gather 所有 token；它与按中间维分片的默认 MoE 有不同的
GEMM 形状。性能记录明确区分这两种设置。

routed EP 的 buffer 上限按 SP 的本卡 token 数计算：例如全局预算 131072、
SP8 时为 16384，而不是每卡预留 131072。非 routed 路径则必须为补齐后的全局
token 数预留 top-k metadata，以覆盖不能被 SP 整除的调度预算。

BF16 routed shared-expert fusion 当前没有实现，采用模型原有的独立 shared MLP。
模型构造和 checkpoint loader 共用这一判断，避免 shared 权重被错误重定向到
不存在的融合槽位。编译缓存键也加入 EP 开关、MoE transport 和 MoE backend：
这些设置会改变 shared MLP 的子图及参数列表，不能复用另一种布局的缓存。

## 测量方法

硬件为单机 8 张 gfx950 GPU，每卡约 288 GiB；软件为 PyTorch 2.10.0+ROCm 7.2.4，
AITER 位于 `/app/aiter-test`。4 卡试验使用 GPU 0–3，8 卡使用 GPU 0–7。
AITER 提交为 `22d2c7c918e96066f67122c0b827536efc598469`，其 tracked diff 为空；
运行目录有 untracked 文件。完整环境与实际测量源码 hashes 保存在结果 JSON。

`atom.benchmarks.benchmark_prefill_parallel` 直接提交恰好 131072 个 token IDs，
关闭 prefix cache，先用同形状请求预热，再测 3 次，报告中位数。
指标是离线 `generate()` wall time，含调度和一个输出 token；它不是纯 GPU attention
时间。独立的 profiler iteration 不计入三次计时。JSON 保存参数、输入 token hash、
逐次时间和输出；新版脚本还记录引擎 TTFT、源码 hash 与运行环境。

测试统一使用原始 BF16 权重、BF16 KV cache、无 online quantization、
`AITER_QUICK_REDUCE_QUANTIZATION=NONE`。M3 使用
`ATOM_FORCE_ATTN_TRITON=1 --block-size 128`：当前安装的 ASM paged-attention
没有对应 BF16 GQA16/block128 kernel，TP/SP 两侧使用相同 fallback。

`--enforce-eager` 关闭 HIP graph，但这里的编译等级仍为原生默认值；不能把它
描述为关闭了 torch.compile。单请求调度上限 131072 与上限 8192 的试验分别列出。

### 重复段落输入：完整 128K 提交

| 模型 | GPU 数 | TP 中位数 / s | SP 中位数 / s | SP 耗时降低 |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 4 | 2.317970 | 2.226362 | 4.0% |
| Qwen3-30B-A3B，双方 EP/none | 8 | 1.309495 | 1.209919 | 7.6% |
| MiniMax-M3 | 4 | 4.864619 | 4.141124 | 14.9% |
| MiniMax-M3 | 8 | 3.222072 | 2.722656 | 15.5% |

M3 还分别比较了 8 卡的 MoE 选项，以下同样使用完整 128K 重复段落输入。
当前测到的最快 TP 是 EP/none，最快 SP 是 BF16 MoRI high-throughput。

| M3 8 卡布局 | MoE 传输 | 中位数 / s |
|---|---|---:|
| TP8 + EP | none（原生 TP 归约） | 2.967777 |
| SP8 + EP | none（gather / reduce-scatter） | 2.464214 |
| SP8 + EP | RCCL routed，独立 shared MLP | 2.616303 |
| SP8 + EP | MoRI high-throughput，独立 shared MLP | 2.413745 |

最佳 SP 相对最佳 TP 的耗时降低约 18.7%。MoRI 行使用 BF16 dispatch/combine，
没有启用 FP4 dispatch 或 FP8 combine。四种布局的归约顺序与融合边界不同，
其精度证据需要与相应配置绑定，不能用一个配置的精度结果替代另一个。

Qwen3 的普通 TP8 会使 BF16 CK MoE 中间维变成 96，当前 kernel 不支持该形状。
因此 8 卡对比双方都显式启用 EP，并关闭 routed all-to-all；没有将失败配置当作
性能基线。Qwen3 SP4 的旧无损实现实测为 2.247622 s，本次最终实现为 2.226362 s，
该设置下实现本身的改善约 0.95%，不能把全部 TP/SP 差距归因于新打包 kernel。

### 重复段落输入：按 8K 调度分块

总输入仍为 131072 tokens，仅将 `max_num_batched_tokens` 设为 8192。

| 模型 | GPU 数 | TP 中位数 / s | SP 中位数 / s | SP 耗时降低 |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 4 | 2.947052 | 2.863143 | 2.8% |
| MiniMax-M3 | 4 | 5.399819 | 4.571141 | 15.3% |

### 不同题目与解答拼接的 128K 输入

将 GSM8K train 前 1000 个样本的题目与解答按顺序拼接，分别使用两个模型的
tokenizer 截取恰好 131072 tokens；各模型内 TP/SP 的 token hash 一致。
此处只测性能，不进行作答评分。

| 模型 | GPU 数 | TP 中位数 / s | SP 中位数 / s | SP 耗时降低 |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B，双方 EP/none | 8 | 1.271786 | 1.176789 | 7.5% |
| MiniMax-M3 | 8 | 3.237065 | 2.751394 | 15.0% |

使用 `--prompt-file` 可复现该输入，构造方式如下：

```python
from pathlib import Path
from datasets import load_dataset

data = load_dataset("openai/gsm8k", "main", split="train")
Path("/tmp/gsm8k-context.txt").write_text("\n\n".join(
    "Question: " + row["question"] + "\nSolution: " + row["answer"]
    for row in data.select(range(1000))
))
```

### 已定位的耗时

M3 4 卡完整 prefill 的单 rank trace 中，RCCL kernel 累计时间从 TP 的
1.762 s 降至 SP 的 1.221 s。两阶段 MoE GEMM 累计仍约 1.54 s；新打包 kernel
合计约 23 ms。这些是单次 profile 的 kernel 累计值，不是额外的端到端样本。
后续优化应优先考察 MoE 计算与通信，而不是把全部精力放在已经较小的打包耗时上。

测试过 symmetric-memory peer pull/push 和 NCCL channels=32；本机未取得收益，
未纳入生产实现。raw logs/profiles 保存在 `/tmp/atom-sp-study`，该目录不随仓库分发。

## 精度证据与边界

通信验证包括独立 PyTorch 字段打包/交换参考、QKV 分离与融合输入、M3 index 字段、
反向 head exchange、padding、128K payload 和 HIP graph replay。
另有通过实际 `ulysses_attention` 包装器的 float32 dense causal GQA 参考计算，
用于检查 KV 复制、head owner、token 顺序和 padding 的组合是否正确。
最终在 4 卡、8 卡以及关闭 custom all-gather 的 4 卡配置下全部通过。
128K 输入交换微基准中，新路径相对 PyTorch 参考降低约 16%–23% 耗时；
单独打包耗时降低约 60%–65%。这些比例不代表整个模型的加速比例。

`validate_prefill_parallel` 捕获完整词表 logits，默认覆盖
17、1025、4097、131072 tokens，每种长度重复两次。`--reference-exchange`
将新打包与传输替换为独立 PyTorch 实现。`compare_prefill_parallel` 同时报告
同配置 A/A 和跨配置差异；不会根据观测结果自动放宽“通过”阈值。

M3 的原生 TP4 A/A 在本轮 128K 探针上出现 max-abs 4.0、KL(TP-a || TP-b)
约 1.032，且 argmax 从 token 3817 变为 32。最终 SP4 重复在该探针上
max-abs 3.5625、KL 约 0.312，argmax 从 32 变为 3817。
这些比较的输入 token hashes 一致。该事实说明两侧重复运行均存在显著波动，
不能要求跨并行配置 logits 逐位相同，也不能据此宣布 SP 已通过任务精度验收。

进一步对相同输入、相同权重连续调用原生 BF16 `fused_moe`，两次调用之间没有
SP 通信：抽样的 6 个调用均未逐位一致，最大绝对差值介于 0.00390625 与 0.25。
这是一个已隔离的原生数值不确定性来源，不代表已经穷尽整模型差异的原因。
诊断工具为 `tests/distributed/check_ulysses_moe_repeat.py`，原始逐调用数据也收入
结果 JSON。

### ATOM recipe 精度评测协议

按用户指定采用原生 ATOM recipe 的评测协议。Qwen3-30B BF16 没有单独的
recipe，沿用同属 GQA + MoE 的 [Qwen3-235B recipe](../recipes/Qwen3-235b.md)
中的 GSM8K 5-shot **completions** 协议；M3 沿用
[MiniMax-M3 recipe](../recipes/MiniMax-M3.md) 的 GSM8K 5-shot **chat** 协议。
两者均使用官方 `lm_eval`，完整 test split 共 1319 题，不设置 `--limit`。

| 设置 | Qwen3-30B-A3B | MiniMax-M3 |
|---|---|---|
| API | `local-completions` | `local-chat-completions` |
| Few-shot | 5，lm-eval 默认 seeded sampler | 5，lm-eval 默认 seeded sampler |
| Chat template / multiturn | 不启用 | 两者均启用 |
| 客户端并发 | 100 | 32 |
| `max_gen_toks` | 未覆盖，lm-eval 0.4.13 默认 256 | 16384 |
| CLI batch size | 默认 1 | recipe 指定 65；chat 客户端实际强制为 1 |
| Temperature | GSM8K task 默认 0 | GSM8K task 默认 0 |
| 指标 | strict-match / flexible-extract 的 exact_match | 同左 |

固定 lm-eval 0.4.13、GSM8K task version 3.0；默认随机种子为
Python 0、NumPy/Torch/few-shot 1234。stop strings 使用 task 原值：
`Question:`、`</s>`、`<|im_end|>`，Qwen completions 客户端还按默认规则
处理 tokenizer EOS。两个客户端都发送文本，不在客户端截断上下文。
M3 的 reasoning/content 分离由原生 ATOM chat server 按模型模板处理，
没有添加自行编写的答案提取器或禁用 reasoning。

recipe 中的服务器示例使用量化 checkpoint、FP8 KV、量化通信或稀疏索引复用。
本轮沿用其**评测协议**，服务器仍保持性能试验中的原始 BF16 模型、BF16 KV /
index cache、无量化通信、无稀疏索引复用，并使用相同的已测 MoE 配置。
API 评测采用 `max_model_len=32768`、`max_num_batched_tokens=32768`、
`max_num_seqs=128`、eager，与单请求 128K 性能试验的调度形状不同。

[原生 CI catalog](../.github/benchmark/models_accuracy.json) 对 Qwen3-235B FP8
给出 0.87，对 M3 MXFP4 给出 0.93；这些阈值都有具体的模型与量化配置，
不能直接作为 Qwen3-30B BF16 或 M3 BF16 的专属阈值。recipe 中 M3 MXFP4
0.9363 / MXFP8 0.9484 是已验证的 flexible-extract 参考分数，不能改写为
本轮 BF16 SP 的通过线。本轮用相同 BF16 checkpoint 成对比较 TP/SP，报告
完整官方分数与逐题差异，不从观测值反推放宽阈值。

`atom.benchmarks.compare_recipe_accuracy` 检查完整样本数、官方聚合分数与
逐题分数一致性、任务配置、随机种子、prompt/doc/target hashes 和生成参数。
分别输出两个 filter 下双方正确、双方错误、仅 TP 正确、仅 SP 正确的数量。
原有 64 题自定义冒烟检查仅保留为辅助证据。

### recipe 全量结果与当前验收状态

共完成 8 次全量运行：两个模型的 TP/SP 各独立启动两次，每次 1319 题。
Qwen 使用 TP8+EP/none 与 SP8+EP/none；M3 使用 TP8+EP/none 与
SP8+EP/MoRI high-throughput，对应前文测到的最快配置。所有轮次均保留，
没有根据分数选择轮次或修改生成上限、few-shot、答案提取规则。

| 模型 / 轮次 | TP flexible | SP flexible | TP strict | SP strict |
|---|---:|---:|---:|---:|
| Qwen3-30B，第 1 轮 | 1168/1319 (88.55%) | 1165/1319 (88.32%) | 1158/1319 (87.79%) | 1149/1319 (87.11%) |
| Qwen3-30B，第 2 轮 | 1171/1319 (88.78%) | 1167/1319 (88.48%) | 1155/1319 (87.57%) | 1156/1319 (87.64%) |
| MiniMax-M3，第 1 轮 | 1259/1319 (95.45%) | 1261/1319 (95.60%) | 1260/1319 (95.53%) | 1261/1319 (95.60%) |
| MiniMax-M3，第 2 轮 | 1258/1319 (95.38%) | 1261/1319 (95.60%) | 1259/1319 (95.45%) | 1262/1319 (95.68%) |

**M3 两轮均未观察到总体任务分数下降；Qwen3 还不能认定“精度无回退”。**
Qwen flexible-extract 两轮分别少答对 3 题和 4 题，下降约 0.23 与 0.30
个百分点。strict-match 第一轮少 9 题，第二轮多 1 题。通信没有新增量化与
任务正确率通过验收是不同的结论，不能用前者替代后者；也不能用不同模型的
CI 阈值将 Qwen3-30B BF16 直接判定为通过。

同配置重复也存在逐题波动。Qwen TP 两轮的 flexible 判分有 31 题不同，
SP 有 42 题不同；M3 TP 有 23 题不同，SP 有 24 题不同。上述事实说明仅凭
少量题目或单次输出不能判断数值等价，但不构成放宽精度门槛的依据。
M3 成对比较中，第 1 轮有 11 题仅 TP 的 flexible 判分正确、13 题仅 SP 正确；
第 2 轮分别为 16 与 19 题，因此也不能声称逐题结果相同。

Qwen 差异题包括输出达到 256-token 上限时尚未写完最终答案、`48` 与 `41`
等实际作答差异，以及 `26` 与 `26.00` 这类数学等值但官方 exact-match
判分不同的格式差异。没有修改官方提取器以消除这些失分。
第一轮输出重新 tokenize 后长度至少 256 的题数为 TP 116、SP 122；第二轮
为 TP 116、SP 107。这只是长度诊断，不是 API `finish_reason`，也说明
不能把两轮分差全部归因于截断。原生 BF16 数值不确定性与具体差异题的因果
关系仍不能仅凭上述统计全部确定。

可复核的记录位于
[recipe 精度结果](benchmarks/ulysses_sp_recipe_accuracy.json)：包含全部官方
聚合结果、每题提取答案和判分、prompt/doc/target/response hashes、两轮
成对比较及同配置重复比较、实际命令和源码/数据集 hashes。完整生成文本与
提示保存在 `/tmp/atom-sp-study/recipe-accuracy` 的官方 JSONL 中，结果文件
记录其路径和 SHA256。对比工具还通过了数据缺失、种子不一致、实际提示文本
不一致的拒绝检查。

这些 GSM8K 结果完成了 recipe 协议下的全量任务评测；它们没有建立全部
128K 长上下文任务的精度等价性。目前不能将整个 SP 方案标记为已经满足
“所有测试模型精度无回退”的硬性要求。

### GSM8K 小样本冒烟检查

使用 test split 的前 64 题，train split 前 5 题作为真实多轮 few-shot，
模型原生 chat template、greedy、最多 1024 输出 tokens、批量 16、关闭 prefix cache。
各模型的 TP4/SP4 输入 hash 逐题一致。下表是末尾数值提取器的结果，
**不是官方 lm-eval 分数，也不是完整数据集验收**。

| 模型 | TP4 正确 | SP4（默认 MoE）正确 | TP/SP 截断数 |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 62/64 | 63/64 | 1 / 0 |
| MiniMax-M3 | 59/64 | 60/64 | 0 / 0 |

M3 有 2 题仅 TP 答对、3 题仅 SP 答对，不能只看总分增加就声称逐题等价。
该小样本未观察到总体正确率下降；这一结论不能外推到 recipe 全量评测，
也不能替代 128K 长上下文任务验收。所有小样本题目的输出和 prompt hashes 保存在
`docs/benchmarks/ulysses_sp.json`。

对最快的 M3 8 卡组合另做同样的 64 题检查：TP8+EP/none 为 **58/64**，
SP8+EP/MoRI high-throughput 为 **59/64**，双方均无截断，输入 hashes 一致。
这组同样有 2 题仅 TP 答对、3 题仅 SP 答对；其结论仍限于小样本总体无回退。

整模型 HIP graph 冒烟检查覆盖 Qwen3 SP8+EP/none 和 M3 SP8 默认 MoE：
分别在 17 与 131072-token prefill 后生成 16 tokens，各重复两次。实际捕获了
batch size 1 的 decode graph；每个模型检查了 68 份有限的完整 logits dump。
两个模型都在两次 128K 运行中找回序列开头的 `731942`。这是单 needle 检查，
没有把它扩展表述为全面的长上下文正确率。

相关单测合计 85 项通过：SP/配置缓存与 indexer gate 34 项、shared-expert
dispatch 27 项、token capacity 11 项、shared layout 与 RCCL prepare/finalize
13 项。新增文件的 Ruff 检查和 `git diff --check` 通过；`topK.py` 原有的
11 条 lint 报告在修改前后相同，本轮未扩展成全仓库格式清理。

## 复现

从 ATOM 仓库根目录运行。每个并行配置使用独立进程，按顺序测量以避免 GPU 竞争。
`ATOM_LOADER_NUM_THREADS=1` 避免本机 8 卡 M3 加载时大量并发 pinned staging
导致的启动阻塞；加载时间不进入 prefill 计时。

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export ATOM_LOADER_NUM_THREADS=1
export AITER_QUICK_REDUCE_QUANTIZATION=NONE
export AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

python -m atom.benchmarks.benchmark_prefill_parallel \
  --model /home/models/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 1 --sequence-parallel-size 8 \
  --enable-expert-parallel --all2all-backend none \
  --enforce-eager --no-enable_prefix_caching \
  --kv-cache-dtype bf16 --index-cache-dtype bf16 \
  --max-model-len 132096 --max-num-seqs 1 \
  --max-num-batched-tokens 131072 --input-length 131072 \
  --gpu-memory-utilization 0.8 --repeats 3 \
  --result-file /tmp/qwen-sp8.json
```

TP8 将并行参数改为 `--tensor-parallel-size 8 --sequence-parallel-size 1`，保留
相同 EP 参数。4 卡试验设置 `HIP_VISIBLE_DEVICES=0,1,2,3`，分别使用 TP4/SP4，
去掉 EP 参数。M3 改用 `/home/models/MiniMax-M3`、
`--trust-remote-code --block-size 128 --gpu-memory-utilization 0.94`，设置
`ATOM_FORCE_ATTN_TRITON=1`，默认 MoE 对比去掉 EP 参数。
分块试验仅将 `--max-num-batched-tokens` 改为 8192。
M3 最快的已测 SP 配置增加
`--enable-expert-parallel --all2all-backend high-throughput`，并明确设置
`ATOM_MORI_FP4_DISPATCH=0 ATOM_MORI_COMBINE_QUANT=none`；保留 `--enforce-eager`。

```bash
torchrun --master-addr=127.0.0.1 --nproc-per-node=8 \
  tests/distributed/check_ulysses_exchange.py \
  --result-file /tmp/exchange-sp8.json

python -m atom.benchmarks.validate_prefill_parallel \
  --model /home/models/Qwen3-30B-A3B-Instruct-2507 \
  --tensor-parallel-size 1 --sequence-parallel-size 4 \
  --enforce-eager --no-enable_prefix_caching \
  --max-model-len 132096 --max-num-seqs 1 \
  --max-num-batched-tokens 131072 --gpu-memory-utilization 0.8 \
  --dump-dir /tmp/qwen-sp4-logits --repeats 2

python -m atom.benchmarks.compare_prefill_parallel \
  /tmp/qwen-tp4-logits /tmp/qwen-sp4-logits \
  --result-file /tmp/qwen-tp-sp-logits.json
```

recipe 全量精度评测使用前面的无量化环境变量，先启动对应的原生 API server。
以下为 Qwen SP8；TP8 将并行参数改为 TP8/SP1。M3 改模型路径、设置
`ATOM_FORCE_ATTN_TRITON=1`、增加 `--trust-remote-code --block-size 128`；
M3 SP8 采用 `--all2all-backend high-throughput`，TP8 保持 `none`。
M3 的环境明确设置 `ATOM_MORI_FP4_DISPATCH=0`、
`ATOM_MORI_COMBINE_QUANT=none`。

```bash
python -m atom.entrypoints.openai_server \
  --model /home/models/Qwen3-30B-A3B-Instruct-2507 \
  --host 127.0.0.1 --server-port 18080 \
  --tensor-parallel-size 1 --sequence-parallel-size 8 \
  --enable-expert-parallel --all2all-backend none \
  --enforce-eager --no-enable_prefix_caching \
  --kv-cache-dtype bf16 --index-cache-dtype bf16 \
  --max-model-len 32768 --max-num-seqs 128 \
  --max-num-batched-tokens 32768 --gpu-memory-utilization 0.8
```

Qwen 客户端：

```bash
python -m lm_eval \
  --model local-completions \
  --model_args model=/home/models/Qwen3-30B-A3B-Instruct-2507,base_url=http://127.0.0.1:18080/v1/completions,num_concurrent=100,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5 \
  --log_samples --output_path /tmp/qwen-sp8-recipe
```

M3 客户端：

```bash
python -m lm_eval \
  --model local-chat-completions \
  --model_args model=/home/models/MiniMax-M3,base_url=http://127.0.0.1:18080/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,max_gen_toks=16384 \
  --tasks gsm8k --num_fewshot 5 --batch_size 65 \
  --apply_chat_template --fewshot_as_multiturn \
  --log_samples --output_path /tmp/m3-sp8-recipe

python -m atom.benchmarks.compare_recipe_accuracy \
  /tmp/m3-tp8-recipe /tmp/m3-sp8-recipe \
  --result-file /tmp/m3-recipe-comparison.json
```

GSM8K 64 题冒烟检查使用上述相同模型、并行、dtype 与环境设置，运行
`python tests/distributed/check_ulysses_gsm8k.py`，并指定
`--max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 32768`
和 `--result-file /tmp/gsm8k-sp4.json`。默认 `--limit 64 --batch-size 16`。
MoE 重复诊断使用 `MOE_REPEAT_RESULT=/tmp/moe-repeat.jsonl`，运行
`python tests/distributed/check_ulysses_moe_repeat.py` 并传入 validation 参数和
`--lengths 17,1025`；该工具明确强制编译等级为 0，使诊断 wrapper 每次都执行。

## PCP 比较范围

当前原生 PCP 的受支持模型仅为 DeepSeek-V4，见
[PCP 模型支持说明](context_parallel_guide.md#prefill-context-parallel-pcp-guide)。
Qwen3 和 M3 的 GQA 路径没有完整的 PCP token 切分、KV 通信与对应 MoE 流程，
直接打开 PCP 参数并不是有效的正确性基线。Ulysses SP 虽然复用 PCP 的进程组
维度，但执行的是独立的 token/head all-to-all 路径；SP 可用不代表原生 PCP 已适配。
本轮证据支持上述配置下的 TP/SP 比较，**不支持“已胜过 PCP”或
“所有模型、输入、批量下最优”这一结论**。
