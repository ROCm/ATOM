> [Examples README](../README_cn.md) > 用发布镜像跑

# 8. 用发布镜像跑八个例子

> English version: [08-release.md](08-release.md)

这一章用 **release 容器镜像**跑 §8.2 的八个例子：软件栈已固定、aiter kernel 已预编译，
每个例子一条命令，不需要安装任何依赖。
第 [1](01-env-setup_cn.md)–[4](04-launching_cn.md) 章是另一条路——从源码搭一套环境，
要换模型或跑双节点（例子 8，见 [第 7 章](07-disaggregated-rdma_cn.md)）时用那一条。

> ⚠️ **本镜像仅支持 AMD gfx950 架构**（Instinct MI350X / MI355X），需要 8 张卡。
> 详见 §8.3.1。

```bash
git clone https://github.com/ROCm/ATOM.git
cd ATOM/atomrl
docker pull rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1        # 模型与数据，只需一次（§8.4.3）
bash examples/run.sh 1                # 跑例子 1，并按 §8.5.1 自动判定
```

这个镜像由本仓库构建（§8.4.2 是在本地构建同一个镜像的步骤）。换例子只改最后那个数字；`bash examples/run.sh suite --tier t1` 依次跑完全部八个。
`examples/run.sh` 是本仓库所有示例共用的启动器，完整说明见
[harness/README.md](../harness/README.md)。

镜像提供环境，刚 clone 的这份 checkout 提供代码。启动器会把它挂进容器，所以改完
`atomrl/` 再跑同一条命令，跑的就是改动后的代码——不重建镜像，不换 tag。
`--atomrl-src /other/checkout` 可以跑别处的代码。

---

## 8.1 这个镜像包含什么

| | |
|---|---|
| 任务 | DAPO 数学 RL（GRPO 风格，按 uid 分组归一化） |
| 模型 | Qwen3-8B-Base（dense）、Qwen3-30B-A3B-Base（MoE，128 专家） |
| 训练后端 | Lumen FSDP2（BF16 / FP8 blockwise2d）、Megatron-Native（EP=8） |
| Rollout 引擎 | vLLM 0.31.1.dev0+gdb9527a46.d20261119.rocm724（BF16 / `fp8_per_block`）、ATOM（BF16 / `per_block_fp8`） |
| 拓扑 | 单个 Ray driver 进程内 8 个训练 actor + 8 个同卡 colocated rollout replica（TP=1） |
| 权重同步 | ZMQ CUDA-IPC，同卡直传，带覆盖率断言 |
| 硬件 | **仅 AMD gfx950（MI350X / MI355X），8 卡** |

算法侧：clip-higher + dual-clip + token-mean 策略损失、动态采样（`filter_groups`）、
overlong 奖励缓冲、TIS rollout 修正。

镜像里已预编译构建时 warmup 覆盖到的 kernel；其余 kernel 在首次使用时编译，也就是新容器第一次运行的
前几分钟：

```bash
docker run --rm --entrypoint /bin/bash rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119 \
  -lc 'echo $ATOM_COMMIT; ls /opt/atomrl/aiter-jit/*.so | wc -l'
```

`ATOM_COMMIT` 是构建该镜像时本仓库的提交。ATOM、AtomRL、Lumen 在 `/opt/atomrl/ATOM` 下，aiter 在
`/opt/atomrl/aiter`；`run.sh` 把你的 checkout 挂到 `/opt/atomrl/ATOM/atomrl`。

§8.5.1 的参考值最初是在旧版 Lumen-RL release 镜像上测得。该私有镜像仅作为历史来源记录，
不属于公开复现流程。

### 8.1.1 版本固定

release 构建从当前 checkout 复制 ATOM、AtomRL 与 Lumen。AITER 和 MORI 则使用
`release/versions.env` 中固定的官方 ROCm 仓库 commit。因此一个结果由镜像 digest
和当前 checkout commit 共同确定；每次运行都会打印两者。

| 组件 | 仓库 | 分支 | Commit |
|---|---|---|---|
| ATOM + AtomRL | 当前仓库 | 当前分支 | 你的 checkout |
| Lumen | `atomrl/lumen` | 内置 | 当前 checkout |
| AITER | `ROCm/aiter` | detached | `c813348adc36` |
| MORI | `rocm/atom-dev:vllm-latest` | 已安装包 | `amd_mori_nightly 1.2.6.dev20261119` |

两个外部依赖都会先按 SHA checkout，再初始化其 submodule。Docker build 参数可以显式
覆盖仓库或 SHA；默认值不会跟随可变的 `main` 或 `HEAD`。

⚠️ **自行替换 ATOM 需要两项由 AtomRL 主动给出、而非从 ATOM 继承的设置**：

- `compilation_config.cudagraph_mode=FULL`（引擎参数，`_pin_cudagraph_mode` 提供）。
  缺了它，no-eager 的 ATOM rollout 会崩在第一次 CUDA graph replay。
- `ATOM_FORCE_ATTN_TRITON=1`（环境变量，`run_dapo.sh` 对所有 `atom*` 的 `MODE` 默认置 1，
  可传 `0` 覆盖）。ATOM 的汇编 paged-decode kernel 在一条序列的上下文恰好占满 16 页且
  最后一页不满时返回**有限但错误**的结果，于是每次运行都有极少数 token 拿到任意 logprob。
  它**躲得过 `abs_diff`**——分歧 token 的数量与平均幅度都不变——只在二次型的
  `chi2_token` 上现形：例子 4 第 3 步实测 5761.69，而上一步是 0.0138，
  同期 `abs_diff` 只从 0.0370 走到 0.0473。置上该变量后，同一条配置三遍九步的
  `chi2_token` 最大值是 5.39。
  ⚠️ **不要指望它改善 `k3_kl`。** 4k 响应下这个缺陷触发得很稀疏（九步一次），
  而参考值读的是**第 1 步**，例子 4 那次事件在第 3 步——本表的指标在结构上就抓不到它。
  判断这个变量是否生效要看 `chi2_token`。ATOM 自身的修复尚未进入所钉的提交，进了即可去掉。

两项对早于相应字段的 ATOM 构建都是空操作。
具体的报错形态与原因见 [`release/versions.env`](../../release/versions.env) 的注释。

sleep **不再钉**，跟随 ATOM 自己的默认，即释放 rollout 的权重、图和 KV 池。
AtomRL 原先在这里强制 `sleep_keeps_memory_resident=true`，该 pin 已于 2026-09-16
移除——它所防的那个故障已经不再复现。释放的代价是每步多一次图重捕获，约 1 秒（2–3%），
不影响已发布的参考值。

底座镜像是已发布的 `rocm/atom-dev:vllm-latest`，其中 PyTorch 为
`2.10.0+rocm7.2.4`，vLLM 为
`0.31.1.dev0+gdb9527a46.d20261119.rocm724`。验证过的附加版本为
Megatron commit `8dd45f5a`、ROCm Apex `daed8525` 和 ROCm
TransformerEngine `6e541a10`。
完整清单见 [`release/versions.env`](../../release/versions.env)。

容器启动时会打印这四棵源码树各自的 HEAD。验证软件栈：

```bash
docker exec atomrl-ex-release bash -lc 'python3 -c "
import aiter, atomrl, atomrl.lumen, vllm, flydsl, transformers
print(vllm.__version__, flydsl.__version__, transformers.__version__)
print(aiter.__file__)"'
```

期望 vLLM 为 `0.31.1.dev0+gdb9527a46.d20261119.rocm724`，且 `aiter`
解析到 `/app/aiter-test/` 下。

### 8.1.2 MORI-EP 需要 Megatron ROCm fork

镜像里的 `megatron-core` 是 NVIDIA 上游版本，不带 MORI 后端。八个示例用不到它——
它们都走默认的 `alltoall` dispatcher。要用 `moe_token_dispatcher_type=flex` + `mori`，
改为固定 fork：

```bash
git clone --depth 1 -b core_r0.18.0_rocm \
  https://github.com/ROCm/Megatron-LM.git "$DATA_ROOT/megatron-rocm"

bash examples/run.sh 7 --env MEGATRON_PATH=$DATA_ROOT/megatron-rocm
```

`MEGATRON_PATH` 会被前置到 `PYTHONPATH`，因此该路径必须在容器内可见。

---

## 8.2 八个例子

八个例子的训练与推理**都跑在同一组 8 张卡上**。

### 8.2.1 概览

| # | 例子 | 训练 | Rollout | 命令 |
|---|------|------|---------|------|
| 1 | 8B BF16 基线 | FSDP2 BF16 | vLLM BF16 | `bash examples/run.sh 1` |
| 2 | 8B FP8 rollout | FSDP2 BF16 | vLLM `fp8_per_block` | `bash examples/run.sh 2` |
| 3 | 8B FP8 端到端 | FSDP2 **FP8 blockwise2d** | vLLM `fp8_per_block` | `bash examples/run.sh 3` |
| 4 | 8B ATOM FP8 | FSDP2 **FP8 blockwise2d** | **ATOM** `per_block_fp8` | `bash examples/run.sh 4` |
| 5 | 8B ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash examples/run.sh 5` |
| 6 | MoE FSDP2 | FSDP2 BF16 | vLLM BF16 | `bash examples/run.sh 6` |
| 7 | MoE Megatron EP=8 | **Megatron** TP=PP=CP=1，EP=8，DP=8 | vLLM BF16 | `bash examples/run.sh 7` |
| 9 | MoE ATOM BF16 | FSDP2 BF16 | **ATOM** BF16 | `bash examples/run.sh 9` |

**例子 8 不在本章。** 它是 [第 7 章](07-disaggregated-rdma_cn.md) 的双节点训推分离
RDMA 部署；站点文件写了两台机器时，`bash examples/run.sh 8` 就会跑它（见
[harness/README.md](../harness/README.md#sites)）。编号在整个 examples 体系里是通用的，
所以这一章跑的是 1–7 和 9。

- 例子 2 / 3 共用一条 config，只差 `TRAIN_FP8`：`0` 只量化 rollout，`1` 训练前向也走 FP8。
- 例子 5 是例子 4 的 BF16 对照组：同一个 ATOM 引擎，只关掉 rollout 在线量化与训练侧 FP8。
- 例子 7 是例子 6 的 Megatron 孪生：两条 config 除 `training_backend` 与 `megatron_cfg`
  外逐字段相同，拓扑 EP=8 使 DP=8 与 FSDP2 一致，所以两者指标可直接相减，
  差值就是训练后端本身的差异。
- 例子 9 是例子 6 的 ATOM 孪生，这一对回答的是「换掉 rollout 引擎要付什么代价」：
  同模型、同训练配置，只把 `generation_backend` 从 vllm 改成 atom 并加一个 `atom_cfg`
  块，其余不动。它也是唯一一个让 MoE 专家权重走 ATOM 权重同步这条路的例子——
  为什么把 `skipped` 列为健康判据，见 §8.5.2。

### 8.2.2 每个例子的完整参数

`examples/DAPO/examples.yaml` 给每个例子传的就是这张表。用手工命令（§8.4.5）时这一行的每一列都要给全。

| # | `MODE` | `TRAIN_FP8` | `CONFIG_OVERRIDE`（都在 `examples/DAPO/configs/` 下） | `STEPS` | `max_response_length` | 模型 | 额外 env |
|---|---|---|---|---|---|---|---|
| 1 | `bf16` | `0` | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 2 | `fp8` | `0` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 3 | `fp8` | `1` | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | Qwen3-8B-Base | — |
| 4 | `atomfp8` | `1` | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | Qwen3-8B-Base | `ATOM_FORCE_ATTN_TRITON=1` |
| 5 | `atombf16` | `0` | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | Qwen3-8B-Base | `ATOM_FORCE_ATTN_TRITON=1` |
| 6 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `ATOMRL_FP32_MOE_ROUTER=0` |
| 7 | `bf16` | `0` | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `ATOMRL_FP32_MOE_ROUTER=0` |
| 9 | `atombf16` | `0` | `dapo_qwen3moe_a3b_ray_atom_bf16_4k_smoke.yaml` | 3 | 4096 | Qwen3-30B-A3B-Base | `ATOMRL_FP32_MOE_ROUTER=0` `ATOM_FORCE_ATTN_TRITON=1` |

> ⚠️ **`MODE` 与 `CONFIG_OVERRIDE` 必须成对给出。** `MODE` 除了选择环境变量，还会**追加一批
> Hydra override**，`CONFIG_OVERRIDE` 只替换 config 文件而不会取消这些追加项。
> 两者不匹配的典型后果：`MODE=atomfp8` 会无条件追加 `compilation_config.level=3`，
> 配一条 vLLM 的 config 就报
> `RuntimeError: aot_compile is not supported by the current configuration`。
> 使用启动器时这一层已经配好，无需关心。

这七条 config 都是 `logger.wandb_enabled: false`，**不需要 wandb 账号**（见 §8.4.6）。
`STEPS` 是命令行 `num_training_steps` 的覆盖值。七条 smoke config 都不写 checkpoint，
所以例子之间可以任意顺序连续运行。

---

## 8.3 环境要求

### 8.3.1 硬件与驱动

- **8 张 AMD gfx950**（Instinct MI350X 或 MI355X），全部处于空闲
- 宿主机 ROCm 7.2，`/dev/kfd` 与 `/dev/dri` 可访问
- Docker（若当前用户不在 docker 组，在站点文件里写 `docker: sudo docker`，见 §8.4.4）

> ⚠️ **本镜像仅能在 gfx950 上运行。** TransformerEngine 与 Apex 是以
> `NVTE_ROCM_ARCH=gfx950` / `PYTORCH_ROCM_ARCH=gfx950` 编译的，预编译进镜像的 25 个
> aiter kernel 也是在 gfx950 上构建的。这些 JIT 产物的文件名不含架构标识，因此在
> gfx942（MI300X / MI308X / MI325X）上会被直接加载并在运行期出错，而不会重新编译。
> 需要 gfx942 请单独构建：`PYTORCH_ROCM_ARCH=gfx942 bash release/build_image.sh`。

### 8.3.2 磁盘

| 项目 | 实测值 |
|---|---|
| 镜像下载量（压缩层，registry manifest 求和） | **13.4 GB** |
| 镜像解包后占盘 | **50.8 GB** |
| Lumen-RL 镜像 `260921b`：下载量 / 解包后 | 11.8 GB / 47.3 GB |

**建议预留**：镜像约 65 GB（解包后的大小加上留在 content store 里的压缩层）+ 模型与数据 74 GB
≈ **140 GB**；本地构建另需构建缓存。八个例子都是 smoke，不写 checkpoint。
若要长跑（`--full`）另需 checkpoint 空间——30B-A3B 的一份 FSDP2 checkpoint
（fp32 权重 + optimizer）约 342 GB，`save_total_limit` 决定同时保留几份。

### 8.3.3 模型与数据

`$DATA_ROOT` 下必须存在下列内容，`examples/run.sh` 预检查的就是这张清单（各例子的 `requires`）：

| 路径（相对 `$DATA_ROOT`） | 体积 | 谁需要 |
|---|---|---|
| `models/Qwen3-8B-Base/` | 16 GB | 例子 1–5，以及所有例子的 tokenizer |
| `models/Qwen3-30B-A3B-Base/` | 57 GB | 例子 6、7、9 |
| `data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet` | 1.02 GB | 全部（train） |
| `data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet` | 892 KB | 全部（val） |
| `logs/` | — | `examples/run.sh` 自动创建 |

从零准备见 §8.4.3。

---

## 8.4 使用

### 8.4.1 开跑前确认卡是空闲的

```bash
docker ps -a                                    # 有没有别人的容器还在占卡
rocm-smi --showmeminfo vram | grep -i used      # 宿主机上直接可用
```

八张卡都应处于**空闲基线约 298 MB**（MI355X 实测 297766912–297832448 B）。
高于此值说明有同租户，或上一次运行留下了孤儿进程。
`examples/run.sh` 把这一步做成了硬门槛：任何一张卡超过 2 GB 就拒绝启动并说明原因，
`--force` 可跳过。

### 8.4.2 获取镜像

拉取：

```bash
docker pull rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119
```

`run.sh` 默认用的就是它。也可以从本仓库自行构建同一个镜像，下面就是全部步骤：

```bash
git clone -b main https://github.com/ROCm/ATOM.git
cd ATOM/atomrl
TAG=atomrl:release-base bash release/build_image.sh       # 45–60 分钟，大头是 TransformerEngine
TAG=atomrl:release-base OUT_TAG=atomrl:release bash release/precompile_kernels.sh
```

然后给 `run.sh` 加 `--image release=atomrl:release`（或在站点文件里写
`images: {release: atomrl:release}`）。

`precompile_kernels.sh` 需要 GPU：aiter kernel 只在首次使用时编译，而 `docker build`
没有设备可用，所以要用带卡的容器把它们编译好再提交进镜像。该脚本的合成 warmup 覆盖
25 个 kernel 中的 5 个，覆盖全部 25 个的做法见脚本头部说明。

### 8.4.3 准备数据

```bash
export DATA_ROOT=/path/to/data
bash examples/run.sh prepare 1 6       # 8B 与 30B-A3B 模型、两个数据集，并完成过滤
```

`prepare` 在镜像里运行 `examples/DAPO/tools/prepare_data.py`。只有
`.download_complete` 与脚本中配置的 immutable revision 一致时才会复用模型；
旧目录或版本不匹配的目录会重新准备。`run.sh <N> --prepare` 会在 §8.3.3
有缺项时先做同样的准备。它在容器里做的事，分步如下：

```bash
# 1) 模型与原始数据集
docker exec -e DATA_ROOT="$DATA_ROOT" atomrl-ex-release bash -lc '
python3 - <<PY
from huggingface_hub import snapshot_download
from pathlib import Path
import os; D = os.environ["DATA_ROOT"]
snapshot_download("Qwen/Qwen3-8B-Base", local_dir=f"{D}/models/Qwen3-8B-Base",
                  revision="49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                  allow_patterns=["*.json","*.txt","*.safetensors","*.model","tokenizer*"])
Path(f"{D}/models/Qwen3-8B-Base/.download_complete").write_text(
    "hf@49e3418fbbbca6ecbdf9608b4d22e5a407081db4")
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset",
                  local_dir=f"{D}/raw/DAPO-Math-17k")
snapshot_download("BytedTsinghua-SIA/AIME-2024", repo_type="dataset",
                  local_dir=f"{D}/raw/AIME-2024")
PY'

# 例子 6、7、9 追加（约 57 GB）
docker exec -e DATA_ROOT="$DATA_ROOT" atomrl-ex-release bash -lc '
hf download Qwen/Qwen3-30B-A3B-Base \
  --revision 1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 \
  --local-dir "$DATA_ROOT/models/Qwen3-30B-A3B-Base" --max-workers 8
printf %s hf@1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9 \
  > "$DATA_ROOT/models/Qwen3-30B-A3B-Base/.download_complete"'
```

```bash
# 2) 过滤掉 prompt > 1024 token 的样本，产出 §8.3.3 的两个 parquet
docker exec -e DATA_ROOT="$DATA_ROOT" atomrl-ex-release bash -lc '
python3 - <<PY
import os, glob, datasets
from transformers import AutoTokenizer
D = os.environ["DATA_ROOT"]; MAXLEN = 1024
OUT = f"{D}/data_cached/qwen3-8b-maxprompt1024"
tok = AutoTokenizer.from_pretrained(f"{D}/models/Qwen3-8B-Base")
def first(g): return sorted(glob.glob(g, recursive=True))[0]
jobs = [(first(f"{D}/raw/DAPO-Math-17k/**/*.parquet"), f"{OUT}/dapo-math-17k.filtered.parquet"),
        (first(f"{D}/raw/AIME-2024/**/*.parquet"),     f"{OUT}/aime-2024.filtered.parquet")]
os.makedirs(OUT, exist_ok=True)
nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
for src, dst in jobs:
    ds = datasets.Dataset.from_parquet(src); n0 = len(ds)
    ds = ds.filter(lambda d: len(tok.apply_chat_template(d["prompt"], add_generation_prompt=True,
                                                        tokenize=True)) <= MAXLEN, num_proc=nproc)
    ds.to_parquet(dst); print(src, "->", dst, n0, "->", len(ds))
PY'
```

> 数据只需过滤一次，八个例子共用：两个模型的 `tokenizer.json` / `vocab.json` /
> `merges.txt` md5 相同（vocab 151936），按 8B tokenizer 过滤的结果对 MoE 同样成立。
>
> **MoE 必须用 Base 版。** instruct / thinking 版的 Qwen3-30B-A3B 在
> `max_response_length` 内不会闭合 `</think>`，导致每条样本被截断、reward 恒为 -1、
> `filter_groups` 连续 10 轮为空，最终抛
> `RuntimeError: filter_groups collected no valid groups`。
>
> 国内网络可换 ModelScope，repo ID 与本地路径不变，见
> [`03-data_cn.md`](03-data_cn.md)。

### 8.4.4 启动器

`examples/run.sh` 在 GPU 宿主机上运行（也可以在任何能连到站点文件所列机器的机器上运行）。
它检查镜像是否在、卡是否空闲，管理容器，按 `examples/DAPO/examples.yaml` 拼好全部环境变量，
跟踪日志，并按 §8.5.1 判定结果。

```bash
bash examples/run.sh list                 # 所有例子、层级与硬件
bash examples/run.sh <id|1..7|9> [选项]
bash examples/run.sh --help
```

容器名为 `atomrl-ex-release`，只创建一次；每次运行前启动器会先重启它（一次运行结束后
每张卡可能仍占约 90.9 GB，见 §8.5.2），运行结束后把它停掉。挂载或镜像对不上的容器会被重建。
运行 `run.sh` 的这份 checkout 会挂到镜像里那份 AtomRL 之上；不挂（§8.4.5）的话容器跑的是
镜像里固化的那份，一旦你改了代码，两者就不是同一个提交。

日志写在数据根目录下，并复制到结果目录：

```
$DATA_ROOT/logs/atomrl-<id>-<时间戳>.log            # 训练日志
$DATA_ROOT/logs/atomrl-<id>-<时间戳>.launcher.log   # 包装层输出与退出码
atomrl/output/regression/<时间戳>-<id>/             # summary.json、summary.md、junit.xml、日志
```

| 选项 | 作用 |
|---|---|
| `--steps N` | 覆盖训练步数 |
| `--full` | 换成该例子的长跑 config（见 §8.4.6） |
| `--no-check` | 只按退出码判定 |
| `--dry-run` | 只打印将要执行的命令，不运行（`show <id>` 还会附带例子说明） |
| `--prepare` | 先准备缺失的模型与数据 |
| `--force` | 卡忙时照样运行，并删除同一镜像的其他容器 |
| `--keep-cache` | ATOM 例子之间不清编译缓存（见 §8.6） |
| `--keep-container` | 运行结束后不停容器 |
| `--verbose` | 前台打印完整日志而不是摘要 |
| `--override K=V` | 追加 config 覆盖项 |
| `--env K=V` | 追加容器环境变量，如 `MEGATRON_PATH`（§8.1.2） |
| `--image release=REF` | 用 REF 代替已发布的 release 镜像（`@key` 换成另一份镜像定义） |
| `--atomrl-src PATH` | 挂载另一份 AtomRL checkout |
| `--atom-src PATH` | 把一份 ATOM checkout 挂到镜像里的 ATOM 之上 |
| `--site FILE` | 机器、数据根目录、docker 命令（`examples/site.example.yaml`） |

站点文件没有设置 `data_root` 时，取环境变量 `DATA_ROOT`。

镜像里的其他源码树也是 editable 安装。`--atom-src` 会把一份 ATOM 挂到镜像里那份之上；
但与 AtomRL 不同，ATOM、Lumen、aiter 正是参考值所固定的东西，换了之后的结果就不能再与
§8.5.1 比较。

### 8.4.5 不用启动器的手工命令

`bash examples/run.sh show <N>` 会打印任意例子的完整命令序列：创建容器的 `docker run`
和运行它的 `docker exec`。例子 1 的要点如下：

```bash
export DATA_ROOT=/path/to/data

docker run -d --name atomrl-ex-release \
  --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/dri --group-add=video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64G \
  -v "$DATA_ROOT":"$DATA_ROOT" -e DATA_ROOT="$DATA_ROOT" \
  -v "$PWD":/opt/atomrl/ATOM/atomrl \
  rocm/atom-dev:atomrl-dapo-gfx950-rocm7.2.3-261119 sleep infinity

docker exec \
  -e RL_ROOT=/opt/atomrl \
  -e DATA_ROOT=$DATA_ROOT \
  -e SCRATCH_ROOT=$DATA_ROOT \
  -e PYTORCH_CUDA_ALLOC_CONF= \
  -e MODE=bf16 \
  -e TRAIN_FP8=0 \
  -e STEPS=3 \
  -e CONFIG_OVERRIDE=examples/DAPO/configs/dapo_qwen3_8b_ray_vllm_smoke.yaml \
  -e MODEL_PATH=$DATA_ROOT/models/Qwen3-8B-Base \
  -e LOG=$DATA_ROOT/logs/example-1.log \
  atomrl-ex-release bash -lc 'bash /opt/atomrl/ATOM/atomrl/examples/DAPO/run_dapo.sh'
```

`$PWD` 是这份 checkout 的根目录（`atomrl/`）。其他例子按 §8.2.2 的表替换 `MODE`、
`TRAIN_FP8`、`CONFIG_OVERRIDE`、`STEPS` 和 `MODEL_PATH`，再加上该表最后一列的环境变量。

日志不走 stdout，`run_dapo.sh` 直接写 `$LOG`。用 `tail -f "$LOG"` 跟踪，用
`bash examples/run.sh check <N> --log "$LOG"` 判定；每段之间需 `docker restart atomrl-ex-release`，
精度或模型不同的 ATOM 例子之间还需清编译缓存（§8.6）。

容易漏的四点：

- `CONFIG_OVERRIDE` **相对于 `$RL_ROOT/ATOM/atomrl`**，写绝对路径会找不到。
- 不给 `CONFIG_OVERRIDE` 时，`MODE` 选中的是 **longrun** config（`wandb_enabled: true`、
  `max_response_length: 20480`），不是 smoke。
- `MODEL_PATH` 默认是 8B，例子 6、7、9 不给就会悄悄跑错模型。
- `PYTORCH_CUDA_ALLOC_CONF=` 后面的空值不是笔误：只有显式空串才会关掉 `expandable_segments`。

### 8.4.6 wandb

| | smoke config（§8.2.2 那七条） | longrun config（`--full`） |
|---|---|---|
| `logger.wandb_enabled` | `false` | `true` |
| 需要账号 | **不需要** | 需要 `WANDB_API_KEY` |
| `max_response_length` | 512 / 4096 | 20480（例子 7 为 4096） |

所以 §8.2 的八个例子不需要 wandb 账号。仅 `--full` 会用到：

```bash
WANDB_API_KEY=xxxx bash examples/run.sh 1 --full

# 没有 key 时启动器会自动追加 logger.wandb_enabled=false
bash examples/run.sh 1 --full
```

长跑在前台要跑几小时到几天，请放在 `tmux` 或 `nohup` 里。

> Hydra 键名是 `logger.wandb_enabled`，不是顶层 `wandb_enabled`；写错会得到
> `ConfigKeyError: Key 'wandb_enabled' not in 'AtomRLConfig'`。
> 缺 key 的失败发生在 `RLTrainer.setup ... complete` **之后**，前几分钟看起来一切正常。

---

## 8.5 判断结果是否正确

`examples/run.sh` 每次运行后都会自动完成本节的判定：抠出第 1 步的指标与
`examples/DAPO/examples.yaml` 里的参考值比对，统计 `Traceback` / `OutOfMemory` /
`CUDA error` / `HSA_STATUS` 的出现次数，统计跳过了张量的权重同步 bucket 数，确认最后一步
已经记下，输出 PASS / FAIL。人工判读见下。

```bash
bash examples/run.sh 1                                              # 运行并判定
bash examples/run.sh check 1 --log $DATA_ROOT/logs/atomrl-dapo-qwen3-8b-vllm-bf16-xxx.log
```

### 8.5.1 参考值表

**测量条件**：8x MI355X（gfx950），镜像 `dapo-gfx950-rocm7.2.3-260921b`
（digest `sha256:eede1d8fcdf5…`）+ AtomRL `22502fa`——两者都要记，原因见 §8.1.1；
命令即当时的启动器 `bash release/run_example.sh <N>`（`examples/run.sh` 已用同一张表、同样的判据取代了它；等价于 §8.2.2 的整行参数），
**`seed=10086`**（`run_dapo.sh` 内固定），取**第 1 步**（`step=1`）的指标。

| # | config（`examples/DAPO/configs/`） | steps | resp | 日志时间跨度 | `rollout_corr/k3_kl` | `entropy` | `rollout_corr/kl`（有符号） | 实测次数 |
|---|---|---|---|---|---|---|---|---|
| 1 | `dapo_qwen3_8b_ray_vllm_smoke.yaml` | 3 | 512 | 158 s | **0.00110** ±30% | **0.628** ±25% | 0.00102 | 3 |
| 2 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml` | 3 | 512 | 127 s | **0.00481** ±30% | **0.791** ±25% | 0.00467 | 3 |
| 3 | `dapo_qwen3_8b_ray_vllm_fp8_smoke.yaml`（`TRAIN_FP8=1`） | 3 | 512 | 133 s | **0.00410** ±30% | **0.790** ±25% | 0.00408 | 3 |
| 4 | `dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml` | 3 | 4096 | 467 s | **0.00399** ±50% | **0.599** ±50% | 0.00374 | 5 |
| 5 | `dapo_qwen3_8b_ray_atom_bf16_4k_smoke.yaml` | 1 | 4096 | 364 s | **0.000954** ±50% | **0.667** ±60% | 0.000899 | 3 |
| 6 | `dapo_qwen3moe_a3b_ray_vllm_verlref_4k_smoke.yaml` | 3 | 4096 | 524 s | **0.00144** ±50% | **0.620** ±60% | 0.00140 | 3 |
| 7 | `dapo_qwen3moe_a3b_ray_megatron_verlref_4k_smoke.yaml` | 3 | 4096 | 501 s | **0.00161** ±50% | **0.631** ±60% | 0.00161 | 3 |
| 9 | `dapo_qwen3moe_a3b_ray_atom_bf16_4k_smoke.yaml` | 3 | 4096 | 542 s | **0.00153** ±50% | **0.683** ±60% | 0.00149 | 3 |

粗体两列带容差的即 `examples/run.sh` 判定 PASS / FAIL 的两项，参考值是「实测次数」列那么多遍的均值。

⚠️ **整张表于 2026-09-21 在本镜像上重标**，每例三遍（例 4 五遍），此前多数行只有一个样本。
`k3_kl` 相对旧参考值的位移全部在 **−8.9% ~ +10.5%** 之间——也就是说换 ATOM、换 aiter 与
本轮的改动**没有可测量地改变 train / rollout 对齐程度**，重标是为了给出散布，不是为了追数值。

**散布才是重标的产物。** 同一例子各遍相对自己均值的最大偏差：例 2 是 2.0%，例 4 是 32.1%。
容差未改动，26 次运行按本表全部 PASS。

⚠️ **例 4 是其中最宽的一个，不要按单次实测收紧它的容差**：五遍分别是
0.00271 / 0.00339 / 0.00426 / 0.00460 / 0.00499，跨度 1.8 倍。本轮之前它的参考值是 0.00287，
中途一度有三遍偏高、看着像被 `ATOM_FORCE_ATTN_TRITON`（§8.1.1）系统性抬高，
补到五遍后最低的一遍回到 0.00271，说明那是散布不是位移。

**时间跨度那一列不设容差、不参与判定**——它受缓存冷热影响，
同一机器上偏差可达 ±15%。启动器报的端到端墙钟比它多约 20–35 s。

**本镜像上的结果：八个例子共 26 次运行，退出码全部为 0**，四类错误计数**全部为 0**，
每个权重同步 bucket 都是 `skipped=0`，判定 **26/26 PASS**，且每一遍的 `k3_kl` 与其
所属例子的参考值偏差都在 **±32.1%** 之内（最宽的是例 4，见上）。
逐次原始记录见 [`VALIDATION.md`](../../release/VALIDATION.md)。

**例子 9 对例子 6——换 rollout 引擎的代价。** 同模型、同训练配置，把 vLLM 换成 ATOM：
`k3_kl` 是 0.00138 对 0.00158，远在容差之内，
**换 rollout 引擎并没有可测量地改变 train / rollout 对齐程度。**
真正变的是时间：每步 56.7 s 对 106.7 s，快约 1.9 倍；代价是 setup 409 s 对 203 s。
所以 3 步的 smoke 看端到端会觉得 ATOM 更慢（579 s 对 523 s），而真正长跑该看的是每步耗时。
例子 7 是同一个 vLLM rollout 配 Megatron actor，每步 103.0 s。

### 8.5.2 判据

- **`rollout_corr/k3_kl` 是主判据**——容差 512 组（例子 1/2/3）±30%、
  4096 组（例子 4–7 和 9）±50%。它非负，且比 `entropy` 稳得多，是判断复现是否成功的首选指标。
- **`entropy` 是第二判据**——容差 512 组 ±25%、4096 组 ±50%、
  MoE 三个例子（6、7、9）±60%。它是 `filter_groups` 筛选后那一批序列上的均值，
  方差大，MoE 上尤其不稳。**判 MoE 复现请以 `k3_kl` 为准。**
- **`rollout_corr/kl` 只作数量级判据**：它是有符号均值，所以只检查实测绝对值是否落在
  参考值的 1/10–10 倍之间。**高出一个数量级才算异常**，
  最常见原因是某一侧未启用 model-sensitive RMSNorm。
- **`rollout_corr/ppl_ratio` 仅供参考**，不参与判定。
- **每个权重同步 bucket 都必须 `skipped=0`。** 这是上面四个错误计数替代不了的一条：
  rollout 引擎如果悄悄漏更新了一部分权重，进程照样 `exit=0`、每一步照样打日志、
  什么都不抛，表现出来像是精度慢慢变差，而不像故障。

  ```bash
  grep -oE 'bucket done - updated=[0-9]+, skipped=[0-9]+' <log> | sort | uniq -c
  ```

  健康的运行每一行都是 `skipped=0`。例子 1、2、3、6、7 走 vLLM 路径，
  根本不打这种行，同样算干净。

> 指标对不上时**先确认跑的是否为同一条 config**：
> `grep -m1 'CONFIG=' $DATA_ROOT/logs/atomrl-<id>-*.launcher.log`
> 会打印本次实际使用的 config、`MODE`、`TRAIN_FP8`、`STEPS`。
> 配置错位与数值回归在指标上表现相同，但前者常见得多。

---

## 8.6 问题处理

**1. 运行结束后显存不自动释放。** smoke 正常结束后每张卡可能仍占着约 90.9 GB：
Ray worker 已退出，但显存没有归还给驱动。
两次运行之间重启容器，否则下一次运行的 KV cache 预算会被压低。启动器每次启动前都会做这件事。

```bash
docker restart atomrl-ex-release
```

**2. 切换 ATOM 精度或模型都需清编译缓存。** torch inductor 缓存不按运行隔离，
例子 4 之后直接跑例子 5（或反之）会在 AOTAutograd 处失败。
启动器会记录上一次的 ATOM 精度与模型（`examples.yaml` 里的 `compile_cache_key`），任一改变时才清理（`--keep-cache` 可关闭）。

```bash
docker exec atomrl-ex-release bash -lc \
  'rm -rf /tmp/aiter_configs /tmp/atom_torch_compile_cache /tmp/torchinductor_root'
```

⚠️ **会撞车的不只是精度，模型也会。** ATOM 的缓存键只有 `mode/actor_id/replica/rank`，
所以例 5 和例 9——同为 `atombf16`、模型不同——共用同一个缓存目录。在例 5 之后跑例 9，
会把 8B 的图拿去套 30B MoE，在引擎初始化阶段挂在
`assert_size_stride ... stride 2048==4096`（正是两个模型的 hidden size）。
这个报错看着像引擎坏了而不像缓存过期。**自己驱动例子时，精度或模型只要有一项不同就清缓存。**

**3. 判断长任务是否存活要看日志，不要用 `pgrep`。** `docker exec` 启动的进程与你的 shell
不共享进程树，`pgrep` 跨会话恒返回 0。**看日志文件是否仍在增长**：

```bash
watch -n 30 'ls -l $DATA_ROOT/logs/atomrl-dapo-qwen3-8b-atom-fp8-xxx.log'
```

**4. `docker restart` 会终止正在跑的任务。** 启动器每次运行前都会重启容器，所以同一台机器上
一个 `examples/run.sh` 还没跑完时，不要再起第二个；确需共用机器，给第二个单独的站点文件并换一个
`container` 前缀。

**5. 日志中出现 `waiting for baton release` 不是卡死。** 8 个训练 actor 在排队等其中一个
完成 JIT 编译。发布镜像已预编译全部 kernel，正常不会出现；若挂载了自己的 aiter 源码则可能出现。

**6. `filter_groups round N` 不是每个例子都有。** 该日志行只在启用动态采样的 config 上出现，
**例子 2、3 关掉了它**——`max_response_length: 512` 下 base 模型很少做完一道题，
开启动态采样会筛掉所有 group。两者仍会跑完 3 步并通过判定。

**7. 覆盖 `aiter` 源码时必须同时更换 `AITER_JIT_DIR`。** 已编译的 kernel 与产生它的 aiter
revision 绑定，复用旧目录会在 import 阶段失败，且报错既不提 aiter 也不提分支：

```
AttributeError: module 'aiter.jit.module_aiter_core' has no attribute 'MlaVersion'
```

加上 `-e AITER_JIT_DIR=/tmp/aiter-jit-<你的分支>` 即可。

**8. `flydsl` 必须与 `aiter` 同步升级。** 底座镜像自带 0.1.4.2，而固定版本的 AITER
要求 `>= 0.2.4`（本镜像固定 0.3.2）。版本不匹配的表现是下面这条 import 期错误，
它由 ATOM 的 `model_ops/moe.py` 抛出，信息中完全不提 aiter：

```
ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
```

**9. FP8 训练发散**（entropy 极低 / `grad_norm` 与 `rollout_corr/kl` 都是 1e4 量级）：
见 [6. 排障](06-troubleshooting_cn.md)。

**10. 权重同步后 wake 时 KV 池算出负数**，说明 ATOM rollout 在 sleep 时释放了显存、
而不是保持常驻。只有 MoE 那个例子会中，而且是在第 0 步已经成功之后：

```
ATOM/atom/rollout/memory_manager.py:137  resume_memory
AssertionError: Not enough memory for KV cache with block size(16). At least 1 block
  (1.50MB) is required, but available_for_kv=-30805.98MB (budget=86.40GB,
   peak_torch=57.68GB, non_torch=52.88GB, safety=5.76GB, free=177.39GB)
```

`non_torch` 是节点上的其余部分，主要是同卡上的训练侧；`free=177.39GB` 是关键线索：
显存是有的，只是没算在 rollout 引擎账上。调大 `gpu_memory_utilization` 不是解法，
释放 actor 的分配器缓存也不是——原因见上面第 1 条。

⚠️ **这个现象自 2026-09-16 起已无法复现**，为它而设的 pin 也已移除，sleep 默认释放。
这里仍记着它，是因为余量一直很薄——池子当时是 −1435 MB 对 86 GB 预算——换个更大的模型
就可能把它带回来。真遇上了，`atom_cfg.engine_kwargs.sleep_keeps_memory_resident=true`
依然能让池子常驻。

---

## 8.7 延伸阅读

| 需求 | 去处 |
|---|---|
| 要改源码 / 换模型 / 不用镜像 | [1. 环境搭建](01-env-setup_cn.md) → [2. 装依赖](02-dependencies_cn.md) → [4. 启动](04-launching_cn.md) |
| 重做数据（§8.4.3 是精简版） | [3. 模型与数据](03-data_cn.md) |
| 本章 §8.6 未覆盖的故障 | [6. 排障](06-troubleshooting_cn.md) |
| 双节点训推分离（例子 8，两台机器时 `bash examples/run.sh 8`） | [5. 多节点 RDMA](05-multinode-rdma_cn.md)、[7. 训推分离双节点 RDMA](07-disaggregated-rdma_cn.md) |

镜像的构建方式与版本固定见仓库根目录的 [`release/`](../../release/)。
