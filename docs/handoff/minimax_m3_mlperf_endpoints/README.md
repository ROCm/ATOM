# MiniMax-M3 MLPerf Endpoints（agentic）c10 测试指南

用 MLPerf Endpoints 的官方 client 和 agentic 数据集，测 MiniMax-M3 的 serving 性能。配置是 TP4 + EAGLE3，默认测并发 10。两种机器都能跑：
- 4 卡 MI455X（gfx1250）：用全部 4 张卡；
- 8 卡 MI355X（gfx950）：用其中 4 张，默认 0,1,4,5，见第 7 节。

脚本会自己识别架构，两种机器用同一套脚本，只是 `config.env` 里镜像、权重路径和选卡不同。

client、数据集、seed、每个点的流程和判定，都沿用 Kimi-K3 那套（gyu 的 `/data/gyu/k3-mlperf-repro/README.md`）。区别在 server 换成了 M3 的 agentic 配置，而且只用一台机器。

## 0. 和 Kimi-K3 版本的区别

| | Kimi-K3（gyu） | MiniMax-M3（本目录） |
|---|---|---|
| 机器 | 2 × 4 卡 MI455X，一个 server 跨两台，TP8 + DCP8 | 1 台，TP4：4 卡 MI455X 的全部，或 8 卡 MI355X 的 4 张 |
| spec decode | DSpark | EAGLE3（`MiniMax-M3-EAGLE3-GQA`），3 个草稿 token，真实接受率 |
| server | 镜像自带的 launcher，加 #2760 overlay | `server.sh`：M3 agentic recipe（本来就是在 MI355X 上测的），gfx1250 上再加 MI455 的环境变量 |
| gate | 镜像里的 `/opt/k3/gate.py` | `gate.py`：输出不能为空、不能全是 `!`、不能重复同一个 token，`17*23` 要答出 391，还要能正确解析一次 tool call |
| API | `http://<node0>:8000` | `http://127.0.0.1:8890` |
| tok/s/GPU | `system_tps` ÷ 8 | `system_tps` ÷ 4 |
| OSL 425–520、inline ≥ 58.32% | K3 的有效性规则 | 只记录不判定，这两条是 K3 的标准 |

不变的部分：
- client 是 `mlcommons/endpoints@4b385eb`，不改；
- 数据集 `mlperf_agentic_inference_dataset.jsonl` 按 sha256 校验；
- 正式测试 613 条轨迹，seed 绑到 Endpoints seed set（cohort 2026-10-C1，set A），这个规则集只固定 seed，没有按模型定的规则；
- 只发 30 分钟请求，第一个 user 跑完就停止发请求，所以只能用来比较配置，不能提交；
- 热身换一组 seed，C 条轨迹，C 小于 16 时用 2C 条。

## 1. 机器前提

- **GPU：**
  - amdgpu 已加载。MI455X 是 4 张卡，MI355X 是 8 张卡（只用其中 4 张）。
  - `kernel.numa_balancing=0`。K3 那边遇到过 numa balancing 打开时 GPU 队列反复被驱逐、卡死的问题。关闭命令：`sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'`。
- **权重：**
  - 在 `config.env` 里设 `MODEL_PATH` 和 `DRAFT_PATH`，都是宿主机上的完整路径。容器里分别挂在 `/models/<目录名>`。
  - e15-05 上是 `/data/models/...`；MI355X 集群的路径见第 7 节。
- **代码：**
  - ATOM 用 `lirui927/ATOM` 的 `lirui/m3-prefill-token-graph-v2`；
  - aiter 用 `lirui927/aiter` 的 `m3-stack-1030-tg`。
  - clone 方法见 `../minimax_m3_prefill_token_graph/README.md` 第 1 节。`TG=0` 时就是没有 prefill 入图的基线。
- **镜像：** 架构要和机器一致，而且镜像里的 python 要能 import 挂进去的 ATOM 和 aiter。
  - MI455X 用 `ruili/m3stack-env:1030`，e15-05 上已经有。
  - MI355X 用 `rocm/atom-dev:latest`，或者某个 `rocm/atom-dev:nightly_*`。
- **共享机器：**
  - 只在本次要用的卡空闲时才起 server，`run_point.sh` 会检查。
  - 判断方法是读 KFD 的 sysfs，看有没有进程在这几张卡上占了超过 1 GiB 显存。所以在 8 卡机器上，另一半卡有人在用也不影响。
  - e15-05 同时是 K3 MLPerf 复现的 node0（`yuhzhu_k3tp8` / `gyu_k3tp8`），用的是同样的 4 张卡。双方都会等对方的卡空出来，跑之前先打招呼。

## 2. 步骤（都在宿主机上执行）

```bash
cd <ATOM>/docs/handoff/minimax_m3_mlperf_endpoints
vi config.env                     # IMAGE、ATOM_DIR、AITER_DIR、MODEL_PATH、DRAFT_PATH；8 卡机器看一下 GPUS；ATOM 是 git worktree 时设 EXTRA_DOCKER_ARGS
./setup_client.sh                 # 只装一次：uv、client@4b385eb、数据集（核对 sha256），最后打印 CLIENT READY
./prep_container.sh               # 建或启动 ${USER}_m3ep 容器，确认 import 的是 /app/aiter 和 /app/ATOM；可以重复执行
./preflight.sh                    # 只读检查，全部 OK 才打印 PREFLIGHT PASS
setsid nohup ./run_point.sh 10 >> ~/mlperf_ep/results/run.log 2>&1 < /dev/null &
tail -f ~/mlperf_ep/results/run.log
```

`run_point.sh` 后面可以跟多个并发，按给的顺序跑，默认只跑 10。每个点依次做下面这些事：
1. 等本次要用的卡（`GPUS`）空闲：连续两次检查、间隔一分钟都空闲才继续，最多等 `IDLE_WAIT` 秒。
2. 生成热身和正式测试的 YAML，用 client 的 `validate-yaml` 校验。
3. 在容器里用 `server.sh` 起新的 server，日志在 `$WORK/logs/m3_c10-<tag>.log`。等 `/health`，最多 45 分钟，因为新的 aiter 树第一次要 JIT 编译。日志里出现 Traceback、GPU Hang 之类的错误行，或者 server 进程没了，就提前放弃。
4. 跑 `gate.py`。
5. 热身：20 条轨迹，用另一组 seed，第一个 user 跑完就停。
6. 正式测试：613 条轨迹，seed set A，30 分钟后停止发请求。
7. 打印汇总、检查有没有失败的 turn，然后停 server。

任何一步失败，都会停掉 server、把卡空出来、然后退出。正常一个点大约 1 小时多一点：起 server、热身、30 分钟正式测试、收尾。

可用的环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `CAP` | `30m` | 正式测试发请求的时长，单位只能是 ms、s、m；`none` 表示整遍跑完 |
| `TG` | `0` | `TG=1` 打开 M3 prefill 入图（`ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH`）；`TG_SIZES` 改 bucket |
| `MAX_SEQS` | 2 × C | 传给 server |
| `GPU_MEM` | 0.95 | 传给 server |
| `KV` | `fp8` | 传给 server |
| `IDLE_WAIT` | 43200 | 每个点开始前，最多等多少秒让卡空出来 |
| `GPUS` | 4 卡机器：空，即全部；8 卡机器：`0,1,4,5` | server 用哪几张卡（`ROCR_VISIBLE_DEVICES`），也是空闲检查的范围 |
| `MONO` | 空 | `ATOM_MONO_ENABLE`，gfx950 上 decode 的 mono megakernel；空表示用 ATOM 默认值，即打开 |
| `API_PORT` / `ENGINE_PORT` / `CONTAINER` | 8890 / 8896 / `${USER}_m3ep` | 同一台机器上要跑第二个 server 时，这三个都要换 |

做 A/B 对照时，用同一个容器连续跑：

```bash
setsid nohup bash -c './run_point.sh 10; TG=1 ./run_point.sh 10' >> ~/mlperf_ep/results/run.log 2>&1 < /dev/null &
```

## 3. server 配置（c10）

`server.sh` 和 AgentX 用的 server（`../minimax_m3_prefill_token_graph/scripts/server.sh`）参数一样，只有两处不同：
- 不设 `--spec-decode-acceptance-rate`，EAGLE3 用真实接受率。固定接受率会改变输出，MLPerf 不能用。
- 不开 profiler，也不开 LMCache。

主要参数：
- 并行：TP4；`--max-num-seqs 20`（2 × C）；`--max-num-batched-tokens 32768`；`--attn-prefill-chunk-size 16384`。
- 缓存：开 prefix cache；FP8 KV；`--block-size 128`；`--gpu-memory-utilization 0.95`。
- 量化：MoE 是 MXFP4，其余线性层在线量化成 `ptpc_fp8`。
- 解码：EAGLE3 3 个草稿 token；decode 的 CUDA graph 尺寸是 1…64。
- 模板：chat template 默认 `thinking_mode: enabled`。
- 并发到 15 及以上时，加 `ATOM_M3_INDEXER_CP=1`（来自 recipe），c10 不会用到。
- 两种机器都设的环境变量（来自 recipe）：
  - `ATOM_FORCE_ATTN_TRITON=1`，recipe 要求必须设；
  - `ATOM_PA_FLYDSL=1`、`AITER_QUICK_REDUCE_QUANTIZATION=INT4`。
- 只在 gfx1250 上加的：`ATOM_MOE_GU_ITLV=1`、`AITER_ROPE_TRITON_BACKEND=1`、`AITER_USE_SYSTEM_TRITON=1`、`HSA_OVERRIDE_GFX_VERSION=12.5.0`。
- gfx950 上不加别的，就是 recipe 原样；decode 默认走 mono megakernel。
- tool call 和 reasoning 由 ATOM 根据 M3 的 chat template 自动识别（`MiniMaxParser`，M3 的 thinking 格式），不用加参数。gate 会验证 tool call 能被解析成 `tool_calls`。

MLPerf client 端：`temperature 1.0`、`top_p 1.0`、`max_new_tokens 8192`、流式输出、`enable_salt`、`inject_tool_delay`。

## 4. 结果

- 每个点一行，写在 `~/mlperf_ep/results/m3-sweep-<tag>.txt`，`TG=1` 时 tag 带 `-tg`。
- 完整结果在 `~/mlperf_ep/results/m3-c10-cap30m-<tag>/`（`report.txt`、`performance/result_summary.json`、`accuracy/`）。
- client 日志在同目录的 `.log` 文件里，server 日志在 `$WORK/logs/`。
- 单独汇总某些结果：`NUM_GPUS=4 ./summarize.py <结果目录>…`。

汇总行的口径：
- tok/s/GPU = `system_tps` ÷ 4；
- tok/s/user = `e2e_avg_interactivity`；
- TTFT p50 / p90、TPOT p50；
- 完成的 turn 数和失败数、正式窗口长度；
- OSL 是整遍的均值；inline 是 inline 精度。

`--check` 要求结果完整、没有失败的 turn。M3 的 OSL 和 inline 没有标准，只用来对比；thinking 打开时，OSL 会比 K3 长。

## 5. 中途停

```bash
ps -eo pid,pgid,args | grep '[r]un_point.sh'    # 查出进程组号
kill -- -<PGID>                                  # 停掉整个 sweep
./down.sh                                        # 只杀本容器里的 server
```

注意：
- 不要在 `--pid=host` 的容器里用 `pkill`，会杀到别人的进程。本目录的容器不是 `--pid=host`，`down.sh` 只会影响自己。
- client 中断过的话，下一个点一定要重起 server，因为 ATOM 不会取消孤儿请求。

## 6. 已知问题和注意事项

- **client 的 TPOT：** turn 以 tool call 开头时，client 会丢掉带 `function.name` 的第一个片段，退回按空格切分（K3 那边发现的）。只影响这类 turn 的 TPOT；OSL、E2E、TTFT 用的是完整消息，不受影响。client 不能改。
- **thinking：** `thinking_mode` 默认打开，和 AgentX 的配置一致。要比较关闭 thinking 的情况，可以在 `mkpoint_yaml.sh` 的 `model_params` 里加 `chat_template_kwargs: {thinking_mode: disabled}`。这是 client 支持的字段，会按请求传给 server。
- **已经验证过的（e15-05，11-02）：**
  - 所有脚本 `bash -n` 通过；
  - `setup_client.sh` 能完整跑完，数据集 sha256 校验通过；
  - c10 的热身和正式测试 YAML 都通过了 client 的 `validate-yaml`，退出码 0；
  - M3 的 tokenizer 能在 client 环境里加载；
  - 用 e15-05 的路径跑 `preflight.sh`，结果是 PREFLIGHT PASS；`GPUS` 只给 3 张卡时正确报 FAIL；
  - `gate.py` 对着 mock server 测过，正常回答 PASS，乱码或没解析出 tool call 的 FAIL；
  - 按卡判断空闲：在 2 号卡上占 2.77 GB 显存，`GPUS=2` 判为忙，`GPUS=0,1` 判为空。
- **还没跑过：**
  - 端到端的 c10，也就是起 server、gate、热身、正式测试这一整套。第一次跑时注意看 gate 的结果和 server 日志。
  - MI355X 上一次都没跑过，包括 preflight。

## 7. 在 MI355X（gfx950）上跑

和 MI455X 的步骤一样（第 2 节），先把 `config.env` 改成下面这样：

```bash
IMAGE=rocm/atom-dev:latest        # 或某个 rocm/atom-dev:nightly_*
MODEL_PATH=/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4        # 或 /mnt/m2m_nobackup/models/MiniMax-M3-MXFP4
DRAFT_PATH=/shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3-GQA
# GPUS 不用设：机器超过 4 张卡时，config.env 默认取 0,1,4,5
```

注意事项：
- **选卡：**
  - 8 卡机器上，0–3 号卡在 NUMA 0，4–7 号卡在 NUMA 1。recipe 规定每个 NUMA 节点各用两张卡，即 0,1,4,5。
  - 开 CPU 卸载时，这样选卡是必须的；不开时，用 0,1,2,3 测下来在噪声范围内。为了和 recipe 一致，还是用 0,1,4,5。
  - 如果别人占了 0,1,4,5，可以用 `GPUS=2,3,6,7`，同样是每个 NUMA 节点两张。
  - 同一台机器同时跑两个 server 时，第二个要换 `CONTAINER`、`API_PORT`、`ENGINE_PORT`。preflight 会检查这两个端口有没有被占。
- **第一次起 server 很慢：** 挂进去的 aiter 树要为 gfx950 做 JIT 编译，可能要 30 分钟以上。`run_point.sh` 等 `/health` 最多 45 分钟。
- **mono：**
  - gfx950 上 decode 默认走 mono megakernel（`ATOM_MONO_ENABLE` 默认是 1），它只管 decode，每步最多 16 个 token，不影响 prefill 入图。
  - 要对比关掉 mono 的情况，加 `MONO=0`。
- **基线：**
  - 两种机器上，M3 都还没有 MLPerf 基线。
  - recipe 里的 MI355X 数字是 AgentX 负载，不能直接拿来比。
  - 做 A/B 对照时，就在同一台机器上连续跑 `TG=0` 和 `TG=1`。
- **汇总：** tok/s/GPU 还是除以 4，也就是 server 实际用的卡数（`NUM_GPUS`），不是整机的 8。
