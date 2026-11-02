# MiniMax-M3 prefill token graph（prefill 入图）交接指南

给接手的 agent：本文件和同目录的 `scripts/` 足够独立完成剩下的验证，不依赖原机器（e15-05）上的任何文件。

## 0. 当前状态

- 方案第 0–2 步已完成：代码基线、attention builder 改动、单测。单测在 gfx1250 上全部通过。
- 第 3–6 步没做：起服务、smoke、GSM8K、TTFT / AgentX 性能对比。原因是当时卡被占用。
- 开关是 `ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH=1`，默认关闭。关闭时走的代码路径和基线完全一样，所以随时可以用 `TG=0` 做同机对照。
- 剩下的验证在两种机器上都能做：4 卡 MI455X（gfx1250），或 8 卡 MI355X（gfx950）的其中 4 张卡。MI355X 的不同点见第 1 节最后一小节。单测目前只在 gfx1250 上跑过。

## 1. 代码

| 仓库 | 分支 | 内容 |
|---|---|---|
| `git@github.com:lirui927/ATOM.git` | `lirui/m3-prefill-token-graph-v2` | 基于 `588e9553b`，即 ROCm/ATOM 的 `yuhua/kimi-k3-prefill-token-graph`（Kimi-K3 prefill 入图，在 main `43eed2fde` 之上）。上面两个提交：功能 + 单测、本交接目录 |
| `git@github.com:lirui927/aiter.git` | `m3-stack-1030-tg` | `fd51011f7` + 10-30 性能用的 M3 栈（#7077/#7079 PA decode、FlyDSL PA planner 三个提交、M3 TP4 调参）+ #7085 + #7222 |

aiter 的两个修复是必需的，不能省：pad 行在 MoE 里被路由到 expert -1，没有这两个修复会越界。
- ROCm/aiter#7085：MX quant-sort 和多阶段 MoE sort 在 `num_local_tokens` 下不越界，已合入 main。
- ROCm/aiter#7222：fused MoE routing / remap 跳过被丢弃的 expert id，还没合入。

```bash
mkdir -p ~/code/m3_tg && cd ~/code/m3_tg
git clone -b lirui/m3-prefill-token-graph-v2 git@github.com:lirui927/ATOM.git ATOM
git clone -b m3-stack-1030-tg git@github.com:lirui927/aiter.git aiter
cd aiter && git submodule update --init 3rdparty/composable_kernel && cd ..
```

新 clone 的 aiter 没有预编译的 JIT 模块。第一次起服务会编很多模块，时间较长；建议先手动起一次服务等它编完，再跑 ladder。

### 容器

e15-05 用的是本地镜像 `ruili/m3stack-env:1030`，它是 `rocm/fw-bringup-mlperf:gfx1250-atom-20261016` 加上 `scripts/setup_container.sh` 装的 aiperf / datasets。F01 上没有这个镜像，用任意 gfx1250 ATOM 镜像即可，挂上面两棵树：

```bash
R=~/code/m3_tg; IMG=<gfx1250 ATOM 镜像>; C=m3_tg
mkdir -p $R/logs
docker run -d --name $C --network host --ipc host --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --device /dev/kfd --device /dev/dri --shm-size 64g \
  -e PYTHONPATH=/app/aiter:/app/ATOM \
  -v $R/logs:/logs -v /data/models:/models \
  -v $R/ATOM/docs/handoff/minimax_m3_prefill_token_graph/scripts:/scripts/agentic \
  -v $R/ATOM:/app/ATOM -v $R/aiter:/app/aiter -v $R:$R \
  -w /app/ATOM $IMG sleep infinity
docker exec $C python3 -c "import aiter, atom; print(aiter.__file__, atom.__file__)"   # 必须是 /app/aiter 和 /app/ATOM
docker exec -e SKIP_LMCACHE=1 $C bash /scripts/agentic/setup_container.sh              # aiperf 等；c2/c10 不需要 LMCache
```

模型需要在 `/models/MiniMax-M3-MXFP4`，EAGLE3 draft 在 `/models/MiniMax-M3-EAGLE3-GQA`。

### MI355X（gfx950）

步骤和上面一样，不同点如下：
- **镜像：** 用 gfx950 的 ATOM 镜像，比如 `rocm/atom-dev:latest` 或某个 `rocm/atom-dev:nightly_*`。`server.sh` 和 `setup_container.sh` 会用 rocminfo 自动识别架构；gfx1250 专用的环境变量在 gfx950 上不会设。
- **权重：** MI355X 集群上，模型和 draft 不在同一个父目录下。把上面的 `-v /data/models:/models` 换成分别挂载：
  ```bash
  -v /shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4:/models/MiniMax-M3-MXFP4 \
  -v /shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3-GQA:/models/MiniMax-M3-EAGLE3-GQA \
  ```
- **选卡：** 8 卡机器上，后面所有起服务和跑脚本的命令都要带 `GPUS=0,1,4,5`。
  - `server.sh` 会把它设成 `ROCR_VISIBLE_DEVICES`；`run_ladder.sh` / `acc_only.sh` 把它透传给 server，并且只等这 4 张卡空闲。
  - 0–3 号卡在 NUMA 0，4–7 号卡在 NUMA 1。recipe 要求每个 NUMA 节点用两张。`OFFLOAD=1`（CPU 卸载）时不这样分，server 起不来。
  - 0,1,4,5 被别人占着时，可以用 `GPUS=2,3,6,7`。
- **端口：** 同一台机器上别人也在跑 ATOM 时，8890 / 8896 可能被占。
  - 用 `API_PORT` / `ENGINE_PORT` 换掉。8896 是 ATOM 的 `--port`，也就是 TP 组的 `MASTER_PORT`。
  - `prefill_graph_smoke.py` 和 `replay_matched.py` 用 `--url` 指定新端口。
- **mono：**
  - gfx950 上 decode 默认走 mono megakernel（`ATOM_MONO_ENABLE` 默认是 1）。它只管 decode，和 prefill 入图互不影响。
  - A/B 对照时两边都不设 `MONO`。

## 2. 方案

### 2.1 机制（来自 Kimi 分支，模型无关）

- 模型类上的 `@token_graph(per_phase=True)` 把 forward 录成若干 CUDA graph 段，在 eager region 处切开。`per_phase` 让 prefill 有自己的一套录制，和 decode 的 graph 互不影响。
- attention op `unified_attention_with_output_base` 是 eager region。在 pad 过的 prefill 里，它只跑真正调度的那些行（`region_graph.on_scheduled_rows` / `padded_prefill_rows`）。
- 其余部分只由 token 数决定 shape，所以可以按 bucket 高度 replay。包括 embedding、norm、QKV/O 投影、MoE、all-reduce、lm_head 之前的所有计算。
- 录制：`ModelRunner._record_prefill_graphs` 在 `attn_metadata_builder.supports_token_graph` 为真时调用 `_record_tr_graphs`。
  - bucket 取 `ATOM_TOKEN_GRAPH_SIZES`，默认 `256,512,1024,2048,3072`，只录不超过 `max_num_batched_tokens` 的。
  - 每个 bucket 在 builder 的 `build_for_token_graph_capture(n)` dummy prefill 上录：`is_dummy_run=True`，attention 只返回输出 shape，不写 KV。
- 运行：
  - `ForwardMode._pad_to_token_graph` 在 `graphable` 时选能装下本步的最小 bucket。`graphable = not (enforce_eager or step_needs_eager or tbo_eager)`。
  - `_padded_inputs` 把 pad 行的 ids / positions 置 0。
  - MoE 用 `scheduled_tokens_gpu` 和 `topk_gating(num_valid_rows)` 把 pad 行路由到 expert -1。
  - 超过最大 bucket 的 prefill 照旧走 eager。
- 输出：有 drafter（EAGLE3）时保留 pad 布局，由 draft 侧按 `next_token_locs` 重新取行；没有 drafter 时直接截到调度行。

### 2.2 M3 为什么可以直接套

- M3 的 dense 层和 sparse 层，attention 都走 `unified_attention_with_output_base`。KV 写入、indexer 的 index-key 写入、index score / top-k、sparse table 构建、PA planning 都在这个 eager region 里，用的是 live metadata。
- M3 的模型 forward 不读 forward context。MoE 只在 PCP 下读 `is_dummy_run`，而 PCP 会被拒绝。
- 只有 `is_dummy_run` 的 attention 会直接返回 `torch.empty_like(q)`（`PagedAttentionImpl.forward_impl`），所以录制不写任何 cache。

### 2.3 开关和拒绝条件

- builder 只在同时满足这三条时开启，见 `aiter_attention._m3_prefill_token_graph_enabled`：
  1. 模型是 M3 且带 sparse attention 配置（`_has_sparse_attention`）；
  2. 环境变量 `ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH=1`；
  3. `atom/config.py:minimax_m3_prefill_token_graph_refusal` 返回 `None`。
- 拒绝函数会拒掉以下情况：非 M3；没有 sparse 配置；plugin 模式；DP≠1；PP≠1（`IntermediateTensors` 和 `weak_ref_tensors` 不兼容）；PCP≠1；开了 EP；开了 TBO / TBO decode。
- 被拒时打 warning `MiniMax-M3 prefill token graphs refused: <原因>`，然后继续走 eager，不会让启动失败。
- 带图片的 prefill（`batch.multimodal_data` 非空）通过 `step_needs_eager` 走 eager。原因是 `inputs_embeds` 每步都是新 tensor，图里读不到。decode 的行为不变。
- `supports_token_graph` 是实例属性。同一个 builder 类和它的子类也服务普通 MHA、GDN、Qwen4Exp 等模型，这些模型保持 False。

## 3. 改动清单（提交 1）

| 文件 | 行数 | 内容 |
|---|---:|---|
| `atom/model_ops/attentions/aiter_attention.py` | +63/−1 | `_m3_prefill_token_graph_enabled`；`__init__` 里设 `self._m3_prefill_token_graph`；`supports_token_graph` 属性；`step_needs_eager`（带图片的 prefill 走 eager）；`build_for_token_graph_capture`（和 Kimi 的同形） |
| `atom/config.py` | +43 | `minimax_m3_prefill_token_graph_refusal` |
| `atom/models/minimax_m3/model.py` | 1 | `@token_graph` 改为 `@token_graph(per_phase=True)` |
| `atom/utils/envs.py` | +6 | `ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH`，`_flag_env`，默认关 |
| `docs/environment_variables.md` | +1 | 环境变量说明 |
| `tests/test_minimax_m3_prefill_token_graph.py` | +116 | 新增，15 个用例：开关矩阵、拒绝日志、dummy capture 的字段和 positions 缓冲、越界 bucket、带图片的 prefill 走 eager |
| `tests/test_indexer_cp_gate.py` | +67/−1 | 拒绝函数的能力矩阵 |
| `tests/test_envs.py` | +8 | 默认关、可打开 |

代码 113 行，测试 191 行。

## 4. 已做的测试

在 gfx1250 容器里跑，只用了 1 张卡：

```bash
docker exec -e HIP_VISIBLE_DEVICES=0 $C bash -c 'cd /app/ATOM && python3 -m pytest -q -p no:cacheprovider \
  tests/test_minimax_m3_prefill_token_graph.py tests/test_envs.py tests/test_indexer_cp_gate.py tests/test_region_graph.py'
# 结果：170 passed。test_region_graph.py 是 Kimi 分支自带的 CUDA graph 测试，14 个在 gfx1250 上通过
```

更大一批回归：attention builder、forward mode、scheduler 和 `tests/test_minimax_m3_*`。949 passed，另有 21 failed、3 error，这些在干净的 `588e9553b` 上一模一样，是已有问题：
- `test_minimax_m3_mono_build_key.py` / `test_minimax_m3_mono_contract.py`：gfx1250 不支持 `s_waitcnt`。
- `test_minimax_m31_shared_fusion.py`：`'FusedMoE' object has no attribute 'quant_method'`。
- `test_model_runner_decode_padding.py::test_pa_preserves_complete_padded_layout`：直接调 asm PA，在 gfx1250 上 abort，要 `--deselect`。

lint：`black --check`、`ruff check`、`git diff --check` 都干净。

MI355X 上还没跑过单测，要在 gfx950 容器里用同样的命令重跑。
- 已有失败的清单会和上面不同，比如 mono 和 asm PA 的测试在 gfx950 上应该能跑。
- 判断有没有引入新失败，以同一台机器上干净的 `588e9553b` 为准。

## 5. 待做的验证（按顺序）

`scripts/` 是原机器 `~/code/m3_fp8kv/scripts/agentic` 的拷贝，改了这几处：
- `server.sh`：`TG=1` 打开本功能，`TG_SIZES` 覆盖 bucket 列表；`API_PORT` / `ENGINE_PORT` 可以改端口。`ARCH`、`GPUS`、`MONO` 原来就支持。
- `run_ladder.sh` / `acc_only.sh`：
  - 把 `TG`、`TG_SIZES`、`GPUS`、`MONO` 和两个端口透传给 `server.sh`；
  - 只等本次要用的卡（`GPUS`）空闲。判断方法是读 KFD sysfs 里每张卡的显存占用，超过 1 GiB 算被占，所以 8 卡机器另一半卡上的进程不影响。
- `aiperf.sh`：端口用 `API_PORT`。

容器内路径是 `/scripts/agentic/`。MI355X 上，下面每条起服务的命令都加 `-e GPUS=0,1,4,5`，每条 `run_ladder.sh` / `acc_only.sh` 前面都加 `GPUS=0,1,4,5`。

### 5.1 起服务（第 3 步）

```bash
docker exec -e CONC=10 -e TG=1 $C bash /scripts/agentic/server.sh /logs/m3tg/server_c10_tg.log   # 先 mkdir -p $R/logs/m3tg
curl -s localhost:8890/health -o /dev/null -w '%{http_code}\n'                                   # 等到 200
grep -E "prefill token graphs (enabled|refused)|Token graphs recorded for prefill buckets" $R/logs/m3tg/server_c10_tg.log
```

- **通过标准：**
  - 日志里有 `MiniMax-M3 prefill token graphs enabled for buckets [256, 512, 1024, 2048, 3072]`；
  - 日志里有 `Token graphs recorded for prefill buckets [...]`；
  - 服务能正常 health。
- **对照：** 再用 `TG=0` 起一次，对比启动时间和 KV cache 容量（日志里的 KV block 数 / 可用显存）。预期每卡多用 1–2 GB。基线大约有 4.5 GB 余量，所以 `GPU_MEM=0.95` 应该放得下。
- **OOM 时：** 先减 bucket（`TG_SIZES=256,512,1024,2048`），或者降 `GPU_MEM`。

### 5.2 smoke（第 4 步）

```bash
docker exec $C python3 /scripts/agentic/prefill_graph_smoke.py --out /logs/m3tg/smoke_tg.json     # TG=1 的服务
# 用 TG=0 重起服务后
docker exec $C python3 /scripts/agentic/prefill_graph_smoke.py --out /logs/m3tg/smoke_eager.json
```

- 三条 prompt 分别是：
  - `base`：约 300 token；
  - `base_extend`：命中 prefix cache 后再追加；
  - `other`：约 100 token。
- greedy 输出 8 个 token。预期两边一致。如果只在后几个 token 上有差别，可能是不同 M 下 GEMM 分块带来的数值差。如果出现乱码或重复，说明 pad 行泄漏了，见第 7 节。
- 还要补测 bucket 边界：未缓存长度取 200、256、257、1000、2048、3072、3073、5000，确认都正常。3073 和 5000 应该走 eager。

### 5.3 精度（第 5 步）

```bash
cd $R/ATOM/docs/handoff/minimax_m3_prefill_token_graph/scripts
C=$C TAG=m3tg_acc_tg ROOT=$R TG=1 bash acc_only.sh      # 真实 EAGLE3 接受率，GSM8K 5-shot 全量
C=$C TAG=m3tg_acc_eager ROOT=$R TG=0 bash acc_only.sh
```

- **通过标准：**
  - flexible-extract ≥ 0.92（`ACC_MIN`）；
  - flexible 和 strict 都和同栈 `TG=0` 的结果在噪声范围内（±0.5 pt）。之前栈的参考值是 flexible 0.9424、strict 0.9249。
- 对比服务日志里的 EAGLE3 接受率。TG 下 aux hidden 是在图里写的，接受率不应该下降。
- `acc_only.sh` 会等 aiter JIT 预编译完、等其他人的 GPU 进程退出；看一下它的 `BUILD_LOG` 参数。

### 5.4 性能（第 6 步）

1. **单步 TTFT：**
   - `replay_matched.py` 复现 102,784 cached + 2,517 new 的 prefill 形状。基线是 101.5 ms，profiler 关。
   - 另外对 256 / 512 / 1024 / 2048 / 3072 个未缓存 token 的 prefill，各测 TG=0 和 TG=1 的 TTFT。
2. **AgentX c2 / c10，每点 3600 s，同机同日对照：**
   ```bash
   cd $R/ATOM/docs/handoff/minimax_m3_prefill_token_graph/scripts
   C=$C TAG=m3tg_tg ROOT=$R ATOM_DIR=$R/ATOM AITER_DIR=$R/aiter OFFLOAD=0 ACC_FIRST=0 TG=1 bash run_ladder.sh 2 10
   C=$C TAG=m3tg_eager ROOT=$R ATOM_DIR=$R/ATOM AITER_DIR=$R/aiter OFFLOAD=0 ACC_FIRST=0 TG=0 bash run_ladder.sh 2 10
   ```
   `run_ladder.sh` 会等到本次要用的卡上没有别人的进程才开始。结果在 `$R/logs/agentic/<TAG>/ladder.log` 末尾的 summary 表里。

## 6. 预期收益和基线

基线来自 e15-05（MI455X ×4，TP4 + EAGLE3，10-30，aiter = 本分支去掉 #7085 / #7222）：

| 点 | total tok/s/chip | TTFT p50 / p90 ms |
|---|---:|---:|
| c10 | 18,146 | 266 / 704 |
| c2 | 6,667 | 417 / 735 |

依据：
- **单步时间：** 10-29 的 warm 测量中，2,517 new + 102,784 cached 的请求是 101.5 ms，GPU 步长 81.6 ms，其中 18.1 ms 是 idle。8K prefill 是 84.2 ms，其中 6.5 ms idle。
- **host 下限：** eager 时 host 端每步约 70 ms，约 1670 次 kernel launch。
- **能消掉的部分：** 每层之间的 gap，大约 60% 在 attention 之外，入图后能消掉；40% 在 attention region 内，会保留。
- **AgentX 分布：**
  - 按步算，≤3072 token 的步占 67%（c10）和 71%（c2）；≤8192 的占 85–86%。
  - 按请求算，68–74% 的请求所有 prefill 步都 ≤3072。
  - 每个请求的未缓存 token：中位数约 1.9K，p75 约 3.3K，p90 约 8K；约 30% ≤1K。

预期（估算，待实测）：
- 每步节省：2–3K token 省 5–15 ms，1–2K 省 15–30 ms，≤1K 省 30–40 ms。
- TTFT p50：c10 从 266 降到约 240–256 ms；c2 从 417 降到约 392–407 ms。
- TTFT p90 基本不变，因为长 prefill 仍走 eager。
- 吞吐 +0.5–1%。
- 显存每卡 +1–2 GB。

MI355X：
- **基线：** 还没有同栈的基线，要在同一台机器上自己跑 `TG=0`。
- **recipe 的参考值：** 可以用 `recipes/MiniMax-M3-Agentic-InferenceX.md` 的 Measured 表（MI355X，3600 s，AgentX）判断基线是否正常：
  - c2：5,485 tok/s/chip，interactivity p90 274.9；
  - c10：16,458 tok/s/chip，interactivity p90 206.6。
  - 这张表没有 TTFT，用的 ATOM / aiter 也不是本分支，只能做量级参考。
- **收益：** 上面的估算来自 MI455X。省下的时间来自 host 开销和层间 gap，这个来源在两种机器上一样；但 MI355X 的 kernel 时间和 host 开销比例不同，具体能省多少要实测。

## 7. 风险和排查

- **EAGLE3：**
  - 有 drafter 时 hidden 保留 pad 布局，draft 按 `next_token_locs` 取行，这是 Kimi 分支的通用路径，M3 + EAGLE3 还没实测。
  - 用 5.3 的接受率和 GSM8K 判断。
  - 如果有问题，先看 aux hidden 的写入是否按调度行取。
- **启动卡住：** 录制时有 TP 集合通信（custom all-reduce）。如果 `Token graphs recorded` 一直不出现，用 `py-spy dump --pid <TP 进程>` 看卡在哪一个 bucket 的哪一段。
- **乱码或重复输出：**
  - 先确认 aiter 分支里有 #7085 / #7222（`git log --oneline -3`），并且 JIT 是从本分支源码编的。`aiter/jit/module_quant.so` 必须是新编的。
  - 再查 MoE 是否拿到了 `scheduled_tokens_gpu`。
- **没测过的组合：**
  - Indexer CP：`CONC>=15` 时 `server.sh` 会设 `ATOM_M3_INDEXER_CP=1`，c24 / c40 会用到。按 index head 切分，理论上兼容。
  - LMCache offload：`CONC>=40`。
  - `ATOM_USE_UNIFIED_ATTN=1`：会走 TritonMHA 子类 builder，同样会开启本功能。
  - 有问题就在 `minimax_m3_prefill_token_graph_refusal` 里加拒绝条件，并在 `tests/test_indexer_cp_gate.py` 的矩阵里补一行。
- **gfx950 特有的：**
  - MoE 的 pad 行：#7085 / #7222 改的是 sort 和 routing，这部分代码不分架构。但 gfx950 上 MXFP4 MoE 用的 kernel 和 gfx1250 不同，expert -1 在 gfx950 上没测过，主要看 smoke 和 GSM8K。
  - aiter 分支：M3 TP4 调参那个提交只针对 gfx1250，gfx950 用 aiter 自带的配置。FlyDSL PA planner 的三个提交两种机器都会用到，因为都设了 `ATOM_PA_FLYDSL=1`。在 gfx950 上遇到 aiter 的问题，先在 `TG=0` 下复现，确认和本功能无关。
  - mono：decode 走 mono megakernel。`TG=1` 下 decode 出问题时，用 `MONO=0` 排除它。
- **长 prefill 收益：** 超过 3072 的 prefill 仍走 eager。可以试 `TG_SIZES=256,512,1024,2048,3072,4096,8192`，覆盖约 85% 的步，代价是显存。录制时每个 bucket 的激活要单独估。
- **回滚：** `TG=0`（环境变量不设）就是基线行为。

## 8. 提交规范

- 功能和测试在提交 1；本目录在提交 2。往上游提 PR 前先去掉提交 2。
- 每次改动后跑第 4 节的单测和 `black --check` / `ruff check`。
