# AtomRL Examples

> 中文版：[README_cn.md](README_cn.md)

Validated recipes for RL post-training and speculative-decoding draft distillation on
AMD Instinct GPUs. Each example runs with one command and is judged against recorded
reference metrics; together they form AtomRL's regression suite.

## Quick start

```bash
git clone https://github.com/ROCm/ATOM.git && cd ATOM/atomrl
docker pull rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119   # release image
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1 6     # models and data, once
bash examples/run.sh list            # the catalog below, live
bash examples/run.sh 1               # run one example and judge it
```

The release image pins ROCm, PyTorch, AITER, ATOM, Megatron and vLLM and ships
precompiled kernels; `run.sh` mounts this checkout into it, so code changes take effect
on the next run without a rebuild.

## Catalog

Status as of 2026-11-19, on the release image above. *Validated*: passes against its
references on the hardware shown. *Long run*: full-size, run on request, no recorded
pass yet.

<table>
  <thead>
    <tr>
      <th>Family</th>
      <th>Example</th>
      <th><code>run.sh</code> id (alias)</th>
      <th>Hardware</th>
      <th>Status</th>
      <th>Smoke time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="11"><a href="DAPO/README.md">DAPO</a></td>
      <td>Qwen3-8B, FSDP2 BF16, vLLM BF16</td>
      <td><code>dapo-qwen3-8b-vllm-bf16</code> (1)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>3.8 min</td>
    </tr>
    <tr>
      <td>Qwen3-8B, FSDP2 BF16, vLLM FP8 rollout</td>
      <td><code>dapo-qwen3-8b-vllm-fp8-rollout</code> (2)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>2.6 min</td>
    </tr>
    <tr>
      <td>Qwen3-8B, FSDP2 FP8, vLLM FP8 rollout</td>
      <td><code>dapo-qwen3-8b-vllm-fp8-e2e</code> (3)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>4.0 min</td>
    </tr>
    <tr>
      <td>Qwen3-8B, FSDP2 FP8, ATOM FP8 rollout</td>
      <td><code>dapo-qwen3-8b-atom-fp8</code> (4)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>12.3 min</td>
    </tr>
    <tr>
      <td>Qwen3-8B, FSDP2 BF16, ATOM BF16 rollout</td>
      <td><code>dapo-qwen3-8b-atom-bf16</code> (5)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>7.4 min</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B, FSDP2 BF16, vLLM BF16</td>
      <td><code>dapo-qwen3moe-vllm-bf16</code> (6)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>10.1 min</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B, Megatron EP=8, vLLM BF16</td>
      <td><code>dapo-qwen3moe-megatron-ep8</code> (7)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>8.7 min</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B, FSDP2 BF16, ATOM BF16 rollout</td>
      <td><code>dapo-qwen3moe-atom-bf16</code> (9)</td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>16.3 min</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B, fully async, 4 + 4 GPUs</td>
      <td><code>dapo-qwen3moe-vllm-fully-async</code></td>
      <td>1×8 MI355X</td>
      <td>validated</td>
      <td>13.4 min</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B, fully async, two nodes, vLLM</td>
      <td><code>dapo-qwen3moe-fully-async-vllm-2node</code></td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>17.0 min</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B, fully async, two nodes, ATOM</td>
      <td><code>dapo-qwen3moe-fully-async-atom-2node</code></td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>26.9 min</td>
    </tr>
    <tr>
      <td rowspan="3"><a href="GRPO/README.md">GRPO</a></td>
      <td>Qwen3-30B-A3B, trainer and vLLM rollout on separate nodes</td>
      <td><code>grpo-qwen3moe-vllm-2node</code> (8)</td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>25.0 min</td>
    </tr>
    <tr>
      <td>the same with ATOM rollout</td>
      <td><code>grpo-qwen3moe-atom-2node</code></td>
      <td>2×8 MI355X</td>
      <td>validated</td>
      <td>25.2 min</td>
    </tr>
    <tr>
      <td>DeepSeek-V4-Flash, Megatron TP4/PP4/EP4, vLLM FP8</td>
      <td><code>grpo-dsv4-flash-3node</code></td>
      <td>3×8</td>
      <td>long run</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="FlowGRPO/README.md">FlowGRPO</a></td>
      <td>Qwen-Image FlowGRPO on verl-omni</td>
      <td><code>flowgrpo-qwen-image-baseline</code></td>
      <td>1×8 MI350X</td>
      <td>long run²</td>
      <td></td>
    </tr>
    <tr>
      <td>the same with Lumen actor patches</td>
      <td><code>flowgrpo-qwen-image-lumen</code></td>
      <td>1×8 MI350X</td>
      <td>long run²</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="SDDD/README.md">SDDD</a></td>
      <td>Eagle3 distillation smoke, Qwen3-8B teacher</td>
      <td><code>sddd-eagle3-smoke-hf</code></td>
      <td>1 GPU</td>
      <td>validated</td>
      <td>0.9 min</td>
    </tr>
    <tr>
      <td>Kimi K2.5 Eagle3, phases 1 and 2, ATOM MXFP4 teacher</td>
      <td><code>sddd-kimi-k25-eagle3-phase1</code>, <code>-phase2</code></td>
      <td>1×8 MI350X</td>
      <td>long run</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="plugin/README.md">Plugins</a></td>
      <td>verl GRPO with the AtomRL engine</td>
      <td><code>verl-plugin-qwen3-8b-vllm</code></td>
      <td>1×8</td>
      <td>not yet run³</td>
      <td></td>
    </tr>
    <tr>
      <td>vime RL with the AtomRL Megatron bridge</td>
      <td><code>vime-plugin-qwen3-8b</code></td>
      <td>1×8</td>
      <td>not yet run³</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2">Repository</td>
      <td>AtomRL unit tests in the release image</td>
      <td><code>unit-tests</code></td>
      <td>1×8</td>
      <td>validated</td>
      <td>6.2 min</td>
    </tr>
    <tr>
      <td>GRPO on Qwen3-0.6B, one GPU (<a href="../configs/"><code>../configs/</code></a>)</td>
      <td><code>quickstart-grpo-qwen3-0.6b</code></td>
      <td>1 GPU</td>
      <td>validated</td>
      <td>14.2 min</td>
    </tr>
  </tbody>
</table>

² The 300-step runs behind the [FlowGRPO results](FlowGRPO/README.md) were made with the
same scripts outside `run.sh`.
³ The verl example was validated in its own image ([results](plugin/verl/README.md)); neither
plugin has been run through `run.sh` yet.

Reference metrics, spreads and per-run results: [release/VALIDATION.md](../release/VALIDATION.md).

## Regression suite

```bash
bash examples/run.sh suite                       # t0 + t1, plus t2 when two hosts are configured
bash examples/run.sh suite --tier t1 --only 1,4,9
bash examples/run.sh suite --tier long --only flowgrpo-qwen-image-lumen
bash examples/run.sh compare <results-A> <results-B>
```

| Tier | Contents | Time | Default |
|---|---|---|---|
| t0 | unit tests | 5–30 min | yes |
| t1 | one-node smokes | 1.5–2 h | yes |
| t2 | two-node smokes | 1 h | when the site file names two hosts |
| long | full-size runs, or any example with `--full` | hours to days | on request |

An example passes when it exits cleanly, reaches its final step, logs no runtime errors,
syncs every weight, and keeps each checked metric within tolerance of its reference.
Each run writes `summary.md`, `summary.json` and `junit.xml` under
`atomrl/output/regression/`. Multi-host sites, the `examples.yaml` format and the
verdict rules are described in [harness/README.md](harness/README.md).

## Documentation

| Chapter | Covers |
|---|---|
| [1. Environment setup](docs/01-env-setup.md) | building the environment from source |
| [2. Dependencies](docs/02-dependencies.md) | installs, patches, kernel precompilation |
| [3. Models and data](docs/03-data.md) | downloads and prompt filtering |
| [4. Launching](docs/04-launching.md) | configs, smoke and long runs, health criteria |
| [5. Multi-node RDMA](docs/05-multinode-rdma.md) | RDMA checks for two-node runs |
| [6. Troubleshooting](docs/06-troubleshooting.md) | known failure modes and fixes |
| [7. Disaggregated two-node RDMA](docs/07-disaggregated-rdma.md) | example 8 deployed by hand |
| [8. Release image](docs/08-release.md) | the image, the DAPO table, references and criteria |

Further reading: [asynchronous training](../atomrl-docs/docs/source/advance/async_training.md),
[rollout correction](../docs/rollout_corr.md), [plugin integration](../docs/plugin-integration.md).
