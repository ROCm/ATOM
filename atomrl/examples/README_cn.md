# AtomRL 示例

> English version: [README.md](README.md)

在 AMD Instinct GPU 上做 RL 后训练和投机解码草稿模型蒸馏的已验证示例。每个示例一条命令运行，
并按记录的参考指标判定结果；所有示例合起来就是 AtomRL 的回归测试套件。

## 快速开始

```bash
git clone https://github.com/ROCm/ATOM.git && cd ATOM/atomrl
docker pull rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.4-261125   # release 镜像
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1 6     # 模型与数据，只需一次
bash examples/run.sh list            # 实时打印下面的清单
bash examples/run.sh 1               # 运行一个示例并判定结果
```

release 镜像固定了 ROCm、PyTorch、AITER、ATOM、Megatron 和 vLLM 的版本，并带有预编译 kernel；
`run.sh` 会把当前 checkout 挂进容器，改完代码下次运行就生效，不需要重建镜像。

## 示例清单

截至 2026-11-26，在上面的 release 镜像上。*validated*：在所列硬件上按参考值通过。*长跑*：全尺寸运行，
按需执行，尚无经 `run.sh` 记录的通过。

<table>
  <thead>
    <tr>
      <th>类别</th>
      <th>示例</th>
      <th><code>run.sh</code> ID（别名）</th>
      <th>硬件</th>
      <th>状态</th>
      <th>smoke 耗时</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="11"><a href="DAPO/README.md">DAPO</a></td>
      <td>Qwen3-8B，FSDP2 BF16，vLLM BF16</td>
      <td><code>dapo-qwen3-8b-vllm-bf16</code>（1）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>3.1 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-8B，FSDP2 BF16，vLLM FP8 rollout</td>
      <td><code>dapo-qwen3-8b-vllm-fp8-rollout</code>（2）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>2.8 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-8B，FSDP2 FP8，vLLM FP8 rollout</td>
      <td><code>dapo-qwen3-8b-vllm-fp8-e2e</code>（3）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>3.5 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-8B，FSDP2 FP8，ATOM FP8 rollout</td>
      <td><code>dapo-qwen3-8b-atom-fp8</code>（4）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>10.9 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-8B，FSDP2 BF16，ATOM BF16 rollout</td>
      <td><code>dapo-qwen3-8b-atom-bf16</code>（5）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>8.0 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B，FSDP2 BF16，vLLM BF16</td>
      <td><code>dapo-qwen3moe-vllm-bf16</code>（6）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>8.0 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B，Megatron EP=8，vLLM BF16</td>
      <td><code>dapo-qwen3moe-megatron-ep8</code>（7）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>8.2 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B，FSDP2 BF16，ATOM BF16 rollout</td>
      <td><code>dapo-qwen3moe-atom-bf16</code>（9）</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>15.8 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B，fully async，4 + 4 卡</td>
      <td><code>dapo-qwen3moe-vllm-fully-async</code></td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>12.8 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B，两机 fully async，vLLM</td>
      <td><code>dapo-qwen3moe-fully-async-vllm-2node</code></td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>15.0 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B，两机 fully async，ATOM</td>
      <td><code>dapo-qwen3moe-fully-async-atom-2node</code></td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>22.4 分钟</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="GRPO/README.md">GRPO</a></td>
      <td>Qwen3-30B-A3B，训练与 vLLM rollout 分在两台机器</td>
      <td><code>grpo-qwen3moe-vllm-2node</code>（8）</td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>20.5 分钟</td>
    </tr>
    <tr>
      <td>同上，ATOM rollout</td>
      <td><code>grpo-qwen3moe-atom-2node</code></td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>28.4 分钟</td>
    </tr>
    <tr>
      <td>DeepSeek-V4-Flash，Megatron TP4/PP4/EP4，vLLM FP8</td>
      <td><code>grpo-dsv4-flash-3node</code></td>
      <td>3×8</td>
      <td>长跑</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="8"><a href="GRPOPerf/README_cn.md">GRPOPerf</a></td>
      <td>B1：Qwen3-30B-A3B on-policy，Megatron EP8 + 共置 ATOM</td>
      <td><code>grpoperf-qwen3moe-onpolicy-2node</code> (B1)</td>
      <td>2×8 MI355X</td>
      <td>长跑</td>
      <td>55 min</td>
    </tr>
    <tr>
      <td>B2：Qwen3-30B-A3B 1-step off，训练和 ATOM 分在两台</td>
      <td><code>grpoperf-qwen3moe-1stepoff-2node</code> (B2)</td>
      <td>2×8 MI355X</td>
      <td>长跑</td>
      <td>36 min</td>
    </tr>
    <tr>
      <td>B3：Qwen3-8B 1-step off，4 + 4 卡</td>
      <td><code>grpoperf-qwen3-8b-1stepoff</code> (B3)</td>
      <td>1×8 MI355X</td>
      <td>长跑</td>
      <td>32 min</td>
    </tr>
    <tr>
      <td>B4：Qwen3-8B on-policy，共置</td>
      <td><code>grpoperf-qwen3-8b-onpolicy</code> (B4)</td>
      <td>1×8 MI355X</td>
      <td>长跑</td>
      <td>34 min</td>
    </tr>
    <tr>
      <td>F1：B1 改 FP8 生成</td>
      <td><code>grpoperf-qwen3moe-onpolicy-fp8gen-2node</code> (F1)</td>
      <td>2×8 MI355X</td>
      <td>长跑</td>
      <td>41 min</td>
    </tr>
    <tr>
      <td>F2：B2 改 FP8 生成</td>
      <td><code>grpoperf-qwen3moe-1stepoff-fp8gen-2node</code> (F2)</td>
      <td>2×8 MI355X</td>
      <td>长跑</td>
      <td>36 min</td>
    </tr>
    <tr>
      <td>F3：B3 改 FP8 生成</td>
      <td><code>grpoperf-qwen3-8b-1stepoff-fp8gen</code> (F3)</td>
      <td>1×8 MI355X</td>
      <td>长跑</td>
      <td>31 min</td>
    </tr>
    <tr>
      <td>F4：B4 改 FP8 生成</td>
      <td><code>grpoperf-qwen3-8b-onpolicy-fp8gen</code> (F4)</td>
      <td>1×8 MI355X</td>
      <td>长跑</td>
      <td>34 min</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="FlowGRPO/README_cn.md">FlowGRPO</a></td>
      <td>verl-omni 上的 Qwen-Image FlowGRPO</td>
      <td><code>flowgrpo-qwen-image-baseline</code></td>
      <td>1×8 MI350X</td>
      <td>长跑²</td>
      <td></td>
    </tr>
    <tr>
      <td>同上，加 Lumen actor 补丁</td>
      <td><code>flowgrpo-qwen-image-lumen</code></td>
      <td>1×8 MI350X</td>
      <td>长跑²</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="SDDD/README.md">SDDD</a></td>
      <td>Eagle3 蒸馏 smoke，Qwen3-8B teacher</td>
      <td><code>sddd-eagle3-smoke-hf</code></td>
      <td>1 卡</td>
      <td>validated</td>
      <td>0.8 分钟</td>
    </tr>
    <tr>
      <td>Kimi K2.5 Eagle3 第一、二阶段，ATOM MXFP4 teacher</td>
      <td><code>sddd-kimi-k25-eagle3-phase1</code>、<code>-phase2</code></td>
      <td>1×8 MI350X</td>
      <td>长跑</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="plugin/README.md">插件</a></td>
      <td>verl GRPO + AtomRL 训练引擎</td>
      <td><code>verl-plugin-qwen3-8b-vllm</code></td>
      <td>1×8</td>
      <td>尚未运行³</td>
      <td></td>
    </tr>
    <tr>
      <td>vime RL + AtomRL Megatron bridge</td>
      <td><code>vime-plugin-qwen3-8b</code></td>
      <td>1×8</td>
      <td>尚未运行³</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2">仓库</td>
      <td>release 镜像里的 AtomRL 单测</td>
      <td><code>unit-tests</code></td>
      <td>1×8</td>
      <td>validated</td>
      <td>5.0 分钟</td>
    </tr>
    <tr>
      <td>Qwen3-0.6B GRPO，单卡（<a href="../configs/"><code>../configs/</code></a>）</td>
      <td><code>quickstart-grpo-qwen3-0.6b</code></td>
      <td>1 卡</td>
      <td>validated</td>
      <td>13.8 分钟</td>
    </tr>
  </tbody>
</table>

² [FlowGRPO 结果](FlowGRPO/README_cn.md)背后的 300 步运行，是在 `run.sh` 之外用同一套脚本完成的。
³ verl 示例已在它自己的镜像里验证过（[结果](plugin/verl/README.md)）；两个插件都还没有经 `run.sh` 运行。

参考指标、波动范围和每次运行的结果见 [release/VALIDATION.md](../release/VALIDATION.md)。

## 回归测试

```bash
bash examples/run.sh suite                       # t0 + t1；配置了两台机器时再加 t2
bash examples/run.sh suite --tier t1 --only 1,4,9
bash examples/run.sh suite --tier long --only flowgrpo-qwen-image-lumen
bash examples/run.sh suite --tier long --only B1,B2,B3,B4,F1,F2,F3,F4   # GRPOPerf，需要两台主机
bash examples/run.sh compare <结果 A> <结果 B>
```

| 层级 | 内容 | 耗时 | 默认运行 |
|---|---|---|---|
| t0 | 单测 | 5–30 分钟 | 是 |
| t1 | 单机 smoke | 1.5–2 小时 | 是 |
| t2 | 两机 smoke | 1 小时 | 站点文件写了两台机器时 |
| long | 全尺寸运行，或任意示例加 `--full` | 数小时到数天 | 按需 |

一个示例判为通过，需要正常退出、跑到最后一步、日志里没有运行时错误、权重全部同步，并且每个被检查的
指标都在参考值的容差内。每次运行在 `atomrl/output/regression/` 下写出 `summary.md`、
`summary.json` 和 `junit.xml`。多机站点、`examples.yaml` 格式和判定规则见
[harness/README.md](harness/README.md)。

## 文档

| 章节 | 内容 |
|---|---|
| [1. 环境搭建](docs/01-env-setup_cn.md) | 从源码搭建环境 |
| [2. 装依赖](docs/02-dependencies_cn.md) | 安装、补丁、kernel 预编译 |
| [3. 模型与数据](docs/03-data_cn.md) | 下载与 prompt 过滤 |
| [4. 启动](docs/04-launching_cn.md) | 配置、smoke 与长跑、健康判据 |
| [5. 多节点 RDMA](docs/05-multinode-rdma_cn.md) | 两机运行的 RDMA 检查 |
| [6. 排障](docs/06-troubleshooting_cn.md) | 已知故障与处理 |
| [7. 训推分离双节点 RDMA](docs/07-disaggregated-rdma_cn.md) | 手工部署例子 8 |
| [8. release 镜像](docs/08-release_cn.md) | 镜像、DAPO 表、参考值与判据 |

延伸阅读：[异步训练](../atomrl-docs/docs/source/advance/async_training.md)、
[rollout 校正](../docs/rollout_corr.md)、[插件集成](../docs/plugin-integration.md)。
