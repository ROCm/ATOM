# 455-CI 测试指南

455（gfx1250）CI 的执行手册：测哪些模型、用哪个镜像、怎么跑、记录哪些数字、怎么汇总。各模型的 server / bench 命令放在同目录的 [`455-CI-recipe.md`](455-CI-recipe.md)，本文件只规定流程和口径。

每次测试按顺序做五件事：

1. 按 §1 确定场景和模型：周末跑 MLPerf Endpoint，周中跑 8K/1K。
2. 按 §2 在固定的 base 镜像里重装 ATOM main 和 aiter main，推成 `rocm/fw-bringup-mlperf:gfx1250-atom-<YYYYMMDD>`。
3. 按 §3 用 recipe 里的命令逐个模型跑。跑挂了就记下核心 error log，接着跑下一个。
4. 按 §4 记录指标，追加到 `455-CI/history.csv`。
5. 按 §5 写本次汇总、画对比图，提交到 `455-CI` 分支。

**前置条件**：docker 已登录，且对 `rocm/fw-bringup-mlperf` 有 push 权限；测试机能访问 GitHub 和 PyPI；对 `ROCm/ATOM` 的 `455-CI` 分支有 push 权限。

## 1. 测试矩阵

| 时间 | 场景 | 模型 | slug | 规模 | GPU 数 | 并发 | 时长 |
|---|---|---|---|---|---|---|---|
| 周末 | MLPerf Endpoint | DeepSeek R1 | `deepseek-r1` | 1 tray x 1P4G | 4 | 64、4 | 10 min |
| 周末 | MLPerf Endpoint | DeepSeek-V4.1-Flash | `deepseek-v4.1-flash` | 1 tray x 1P4G | 4 | 64、4 | 10 min |
| 周末 | MLPerf Endpoint | Kimi-K3 (2.8T) | `kimi-k3` | 2 node, 2 tray x 1P4G | 8 | 64、4 | 10 min |
| 周末 | MLPerf Endpoint | Qwen-35B-A3B | `qwen-35b-a3b` | 1 card | 1 | 64、4 | 10 min |
| 周中 | 8K/1K | DeepSeek-V4-Pro | `deepseek-v4-pro` | 1 tray x 1P4G | 4 | 64、4 | — |
| 周中 | 8K/1K | GLM-5.2 | `glm-5.2` | 1 tray x 1P4G | 4 | 64、4 | — |
| 周中 | 8K/1K | MiniMax-M3 (428B) | `minimax-m3` | 1 tray x 1P4G | 4 | 64、4 | — |
| 周中 | 8K/1K | Qwen-3.8 Flash | `qwen-3.8-flash` | 1 card | 1 | 64、4 | — |

- **GPU 数**是每卡吞吐的分母：1 tray x 1P4G 为 4 张，2 tray 为 8 张，1 card 为 1 张。
- **slug** 用作目录名和 `history.csv` 的 `model` 列。定下来就不要改，否则新旧数据对不上。
- 8K/1K 的输入输出长度、请求数等参数以 recipe 为准。

## 2. 准备测试镜像

### 2.1 规则

- **base 镜像固定**为 `rocm/fw-bringup-mlperf:gfx1250-atom-20261110-dsv41`。要换 base 先和负责人确认，并在汇总里写明：换 base 前后的数字不能直接比。
- **只重装 ATOM 和 aiter，且只用 `ROCm/ATOM` 和 `ROCm/aiter` 的 `main`。** 不 checkout 私有分支或 PR，不 cherry-pick，不打 patch。镜像里 `/app/endpoints-recipe-*/patches` 这类补丁也不用。
- **其他组件保持 base 原样**（torch、triton、mori 等）。例外是 ATOM / aiter main 自己锁定版本的依赖，会随安装一起更新。例如 aiter main 锁定 `flydsl==0.5.0`，会替换镜像原来的 FlyDSL。这类变化记进汇总的"依赖变化"，不要手动换回。
- **构建失败**（clone、安装或 import 不通过）时，本次所有模型都记 N/A，error 写构建的核心报错。不要改用私有分支凑合。

### 2.2 命名和同名检查

新镜像命名为 `rocm/fw-bringup-mlperf:gfx1250-atom-<YYYYMMDD>`，日期取构建当天（北京时间）。开始构建前查一次远端有没有同名镜像，push 前再查一次：

```bash
BASE_IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-20261110-dsv41
DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
NEW_IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-${DATE}

# 输出 present 或 absent；查询本身失败（未登录、网络问题）时返回非 0
remote_tag_state() {
  local out
  if out=$(docker manifest inspect "$1" 2>&1); then echo present; return 0; fi
  if grep -q "no such manifest" <<<"$out"; then echo absent; return 0; fi
  echo "查询失败：$out" >&2; return 1
}
remote_tag_state "$NEW_IMAGE"
```

只有输出 `absent` 才算远端没有。`unauthorized` 之类的查询失败不能当成"不存在"，否则可能覆盖别人的镜像。

| 结果 | 处理 |
|---|---|
| `absent` | 继续 |
| `present`，带 `com.amd.ci455.date` label | 455-CI 当天已经推过：不重建、不重推，直接用它测；commit 以它的 label 为准 |
| `present`，不带这个 label | 别人的镜像：停下来通知负责人，不覆盖，也不自行改名 |
| 查询失败 | 先解决登录或网络问题，再查 |

查远端镜像的 label：

```bash
docker buildx imagetools inspect "$NEW_IMAGE" --format '{{json .Image.Config.Labels}}'
```

### 2.3 重装 ATOM 和 aiter

在 455 测试机上起构建容器（镜像入口是 `sleep infinity`）：

```bash
docker pull "$BASE_IMAGE"
docker run -d --name ci455-build-${DATE} --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined "$BASE_IMAGE"
docker exec -it ci455-build-${DATE} bash
```

在容器里逐步执行。任何一步报错就停下，按 2.1 的"构建失败"处理：

```bash
# 镜像用的是 Ubuntu 24.04 的系统 Python（PEP 668），不设这个变量 pip 会拒绝安装和卸载。
# 要用环境变量而不是 --break-system-packages 参数：aiter 安装时内部还会再调一次 pip。
export PIP_BREAK_SYSTEM_PACKAGES=1
PKGS='^(atom|amd-aiter|flydsl|triton|torch|transformers) '
python3 -m pip list 2>/dev/null | grep -E "$PKGS" > /tmp/pkgs.before

# base 里的 /app/ATOM 和 /app/aiter 在私有分支上，还有未提交的改动：整个删掉，重新 clone
python3 -m pip uninstall -y atom amd-aiter
rm -rf /app/ATOM /app/aiter
git clone -b main https://github.com/ROCm/ATOM.git /app/ATOM
git clone -b main --recursive https://github.com/ROCm/aiter.git /app/aiter

# aiter 按它的 README 安装：沿用镜像里的 Triton；GPU_ARCHS=gfx1250、ENABLE_CK=0 来自镜像 ENV
cd /app/aiter && AITER_USE_SYSTEM_TRITON=1 python3 -m pip install -e .
cd /app/ATOM && python3 -m pip install -e .

# 核对：两个仓库都在 main，没有本地改动，import 正常
for d in /app/ATOM /app/aiter; do
  git -C $d log -1 --date=short --format="$d %H %cd %s"
  git -C $d status --porcelain --untracked-files=no | wc -l    # 必须是 0
done
python3 -m pip show atom amd-aiter | grep -E '^(Name|Version|Editable project location):'
python3 -c 'import atom, aiter; print("import ok")'

python3 -m pip list 2>/dev/null | grep -E "$PKGS" > /tmp/pkgs.after
diff /tmp/pkgs.before /tmp/pkgs.after    # 依赖变化，抄进汇总
rm -rf /root/.cache/atom/* /root/.cache/pip
exit
```

### 2.4 commit 镜像并 push

```bash
ATOM_SHA=$(docker exec ci455-build-${DATE} git -C /app/ATOM rev-parse HEAD)
AITER_SHA=$(docker exec ci455-build-${DATE} git -C /app/aiter rev-parse HEAD)
docker commit \
  --change "LABEL com.amd.ci455.date=${DATE} com.amd.ci455.base_image=${BASE_IMAGE}" \
  --change "LABEL com.amd.ci455.atom=main@${ATOM_SHA} com.amd.ci455.aiter=main@${AITER_SHA}" \
  --change "LABEL com.amd.atom.commit=${ATOM_SHA} com.amd.aiter.commit=${AITER_SHA} image.atom_branch=main image.aiter_branch=main org.opencontainers.image.version=gfx1250-atom-${DATE}" \
  ci455-build-${DATE} "$NEW_IMAGE"

remote_tag_state "$NEW_IMAGE"          # push 前再查一次，必须是 absent
docker push "$NEW_IMAGE"
docker buildx imagetools inspect "$NEW_IMAGE" | grep -m1 Digest    # 记下 digest，写进汇总
docker rm -f ci455-build-${DATE}
```

- 第三条 LABEL 覆盖从 base 继承来的同名 label。base 的这些 label 记的是私有分支，不覆盖会误导后来的人。
- push 只上传新增的那一层，base 的层远端已经有了。
- Kimi-K3 的两个节点都要 `docker pull` 同一个 `$NEW_IMAGE`。

## 3. 跑测试

### 3.1 命令从哪来

- server 和 bench 命令一律取自 `455-CI-recipe.md`。只替换和机器相关的部分（模型路径、端口、节点地址、结果目录），不改参数。要调参数，先改 recipe 再测，并在汇总里注明。
- 实际执行的命令（变量展开后）存成 `server_cmd.sh` 和 `bench_cmd.sh`，跟结果一起提交。提交前把内网地址和 token 换成占位符：455-CI 分支在公开仓库里。
- recipe 里没有某个模型时，该模型记 N/A，error 写"recipe 缺失"。

recipe 建议每个模型一节，方便按模型名找到命令：

````markdown
## Kimi-K3 (2.8T)            ← 和 §1 表格里的模型名一致
- 场景：MLPerf Endpoint；GPU：8（2 node, 2 tray x 1P4G）
- interactivity 口径：e2e_avg_interactivity    ← 只有 MLPerf Endpoint 需要

### server
```bash
# 两个节点各写一段
```

### bench
```bash
# 用 ${CONC} 表示并发（4 / 64），${RESULT_DIR} 表示结果目录
```
````

### 3.2 每个模型的流程

`RUN_DIR` 是测试机上本次的结果目录（挂进容器），跑完按 §4.3 的结构拷进 455-CI 分支。`PORT` 是 recipe 里 server 的端口。

```bash
export AITER_LOG_LEVEL=WARNING                      # 压住 aiter 的刷屏日志
test "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0    # GPU 必须空闲
rm -rf /root/.cache/atom/*                          # 清编译缓存：旧缓存会导致静默错误

# 1) 起 server（recipe 的 server 命令），最多等 60 分钟：首次启动要 JIT 编译
mkdir -p "$RUN_DIR/$MODEL"
<server 命令> > "$RUN_DIR/$MODEL/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 180); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null && { ready=1; break; }
  kill -0 $SERVER_PID 2>/dev/null || break         # server 已经退出
  sleep 20
done
[ $ready = 1 ] || echo "server 没起来，按 3.3 处理"
rocm-smi --showmemuse        # 显存必须已被占用：/health 正常不代表模型加载好了

# 2) 按 recipe 做 warmup（不计入结果），然后依次跑并发 4 和 64
for CONC in 4 64; do
  RESULT_DIR="$RUN_DIR/$MODEL/c${CONC}"; mkdir -p "$RESULT_DIR"
  <bench 命令> > "$RESULT_DIR/bench.log" 2>&1
done

# 3) 停 server：ATOM 不理 SIGTERM，要 kill -9 主进程和 EngineCore 进程
kill -9 $(pgrep -f "[o]penai_server"); sleep 2; pkill -9 -f "ATOM[:][:]"
test "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0
```

- 两个并发用同一个 server，中间不重启。
- MLPerf Endpoint 的 client 不要中途打断。ATOM 不会取消已断开连接的请求，打断之后必须重启 server、重新 warmup 再测。
- Kimi-K3 跨两个节点：镜像 ENV 里 `NCCL_SOCKET_IFNAME=lo`、`GLOO_SOCKET_IFNAME=lo` 只适合单机，要按 recipe 换成节点之间的真实网卡。
- 1 card 的模型：镜像 ENV 默认 `HIP_VISIBLE_DEVICES=0,1,2,3`，要按 recipe 限制成 1 张卡。

### 3.3 跑挂了

下面任一情况都算失败：server 没起来或中途退出；bench 报错退出；结果文件缺失或不完整。

1. **不盲目重试。** 纯环境问题（GPU 没释放、端口被占、下载失败）修好后可以重跑一次，并在汇总里注明"重跑"。
2. **摘核心 error log**，存到 `$RUN_DIR/$MODEL/error.log`，不超过 100 行：

   ```bash
   LOG="$RUN_DIR/$MODEL/server.log"      # client 侧失败就换成对应的 bench.log
   n=$(grep -n -m1 -E "Traceback|Memory access fault|HIP error|hipError|out of memory|Segmentation fault|core dumped|AssertionError|RuntimeError|Error:" "$LOG" | cut -d: -f1)
   { [ -n "$n" ] && sed -n "$(( n > 15 ? n - 15 : 1 )),$(( n + 40 ))p" "$LOG"
     echo "----- 最后 30 行 -----"; tail -n 30 "$LOG"; } | head -n 100 > "$RUN_DIR/$MODEL/error.log"
   ```

3. **一句话报错**：取最后那行异常（例如 `RuntimeError: ...`），截到 150 个字符以内，写进 `history.csv` 的 `error` 列和汇总。
4. 按 3.2 第 3 步把 GPU 清干净，接着跑下一个模型。

CI 只负责记录。要查根因就另开任务，从 ATOM 的 `/debug-guide` 开始。

## 4. 记录

### 4.1 指标口径

**MLPerf Endpoint**：读 client 输出的 `report_dir/performance/result_summary.json`。

| 指标 | 计算 |
|---|---|
| 每卡吞吐 (tok/s/GPU) | `tps ÷ GPU 数`。`tps` 即 `system_tps`：client 数出的输出 token ÷ 测量窗口 |
| interactivity (tok/s/user) | agentic benchmark 用 `e2e_avg_interactivity`；single-turn 用 `1e9 / tpot.percentiles["90.0"]`（TPOT 单位是 ns，即 1000 / TPOT p90 [ms]） |

- Kimi-K3 和 DeepSeek-V4.1-Flash 是 agentic，用 `e2e_avg_interactivity`。这也是 MLPerf Endpoints 官网 agentic 图的横轴，和我们已有的 MI455 图口径一致。
- DeepSeek R1 和 Qwen-35B-A3B 按各自的 benchmark 类型选口径，写进 recipe 后不再改，否则历史数据不可比。
- `n_samples_failed` 不为 0 时照常记数字，在 `note` 列写明失败数。

**8K/1K**：读 ATOM `benchmark_serving` 的结果 JSON。bench 命令要带 `--save-result`；默认的 `--percentile-metrics` 已经包含 itl。

| 指标 | 计算 |
|---|---|
| 每卡吞吐 (tok/s/GPU) | `total_token_throughput ÷ GPU 数`（输入加输出 token），和 ATOM benchmark dashboard 的 `tput_per_gpu`、InferenceX 口径一致 |
| ITL P50 (ms) | `median_itl_ms` |

- `completed` 应该等于发出的请求数。`errors` 里有非空项时，把个数写进 `note`。

### 4.2 history.csv

每个模型的每个并发，每次测试写一行。失败也写，数字留空。只追加不修改；历史行要更正时单独提一个 commit，并在 message 里写明原因。

| 列 | 内容 |
|---|---|
| `run_id` | `<YYYYMMDD>-<scenario>`，同一天重跑加 `-r2` |
| `date` | 测试开始日期（北京时间），`YYYY-MM-DD` |
| `scenario` | `endpoint` 或 `8k1k` |
| `model` | §1 的 slug |
| `gpus` | 每卡吞吐的分母 |
| `concurrency` | `4` 或 `64` |
| `status` | `ok` 或 `fail` |
| `tok_s_per_gpu` | 每卡吞吐，保留两位小数 |
| `interactivity` | tok/s/user，只有 endpoint 填 |
| `itl_p50_ms` | 只有 8k1k 填 |
| `image`、`base_image` | 测试镜像 tag、base 镜像 tag |
| `atom_branch`、`atom_commit` | `main`、40 位 sha |
| `aiter_branch`、`aiter_commit` | `main`、40 位 sha |
| `error` | 失败时的一句话核心报错 |
| `note` | 其他说明，如重跑、失败请求数 |

UTF-8，逗号分隔，含逗号的字段加双引号。表头按上表顺序。

### 4.3 455-CI 分支的目录结构

`recipes/455-CI/` 在第一次提交结果时创建。

```text
recipes/
├── 455-CI-guide.md        本文件
├── 455-CI-recipe.md       各模型的 server / bench 命令
└── 455-CI/
    ├── README.md          历次测试的索引，最新的在最上面
    ├── history.csv        全部历史数据，对比图从这里画
    └── runs/<run_id>/
        ├── summary.md     本次汇总
        ├── charts/<slug>.png
        └── <slug>/
            ├── server_cmd.sh  bench_cmd.sh
            ├── c4/  c64/      结果文件
            └── error.log      只在失败时有
```

结果文件提交这些：

- **endpoint**：`performance/result_summary.json`、`report.txt`、`accuracy/accuracy_results.json`。不提交 `events.jsonl`（K3 每跑一轮数据集约 6 GB）。
- **8k1k**：结果 JSON 只留标量字段。原文件带每个请求的 itl 和生成文本，一个点就可能有几 MB：

  ```bash
  python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); json.dump({k: v for k, v in d.items() if not isinstance(v, list)}, open(sys.argv[2], "w"), indent=1)' raw.json result.json
  ```

- 不提交完整 server log、模型文件、token 或密码。单个文件不超过 1 MB。

## 5. 汇总

### 5.1 summary.md

每次测试写一份 `runs/<run_id>/summary.md`，内容和顺序固定：

````markdown
# 455-CI · <YYYY-MM-DD> · MLPerf Endpoint

| 项目 | 内容 |
|---|---|
| 测试日期 | <YYYY-MM-DD>（北京时间） |
| 测试 image | `rocm/fw-bringup-mlperf:gfx1250-atom-<YYYYMMDD>`（digest `sha256:<...>`） |
| base image | `rocm/fw-bringup-mlperf:gfx1250-atom-20261110-dsv41` |
| ATOM | `main` @ [`<sha 前 9 位>`](https://github.com/ROCm/ATOM/commit/<sha>)，<commit 日期> <标题> |
| aiter | `main` @ [`<sha 前 9 位>`](https://github.com/ROCm/aiter/commit/<sha>)，<commit 日期> <标题> |
| 依赖变化 | 2.3 里 diff 的结果，例如 flydsl 版本变化；没有就写"无" |

## 结果

| 模型 | GPU | 并发 | tok/s/GPU | vs 上次 | Interactivity (tok/s/user) | vs 上次 |
|---|---|---|---|---|---|---|
| DeepSeek R1 | 4 | 4 | <数值> | <±x.x%> | <数值> | <±x.x%> |
| | | 64 | <数值> | <±x.x%> | <数值> | <±x.x%> |
| Kimi-K3 (2.8T) | 8 | 4 | N/A | | N/A | |
| | | 64 | N/A | | N/A | |

## 失败

- **Kimi-K3 (2.8T)**：`<一句话核心报错>`。摘录见 [error.log](kimi-k3/error.log)

## 本次 vs 历史

### DeepSeek R1

![DeepSeek R1](charts/deepseek-r1.png)
````

- 8K/1K 的表把 Interactivity 两列换成 ITL P50 (ms)。
- **vs 上次**：和同模型、同并发最近一次成功的结果比，写成 `+2.1%` 或 `−6.3%` 这样。第一次有数据的写"首次"。
- **回退标记**沿用 ATOM benchmark dashboard 的阈值：变差 5% 及以上标 ⚠️，10% 及以上标 🔴。tok/s/GPU 和 interactivity 下降算变差，ITL P50 上升算变差。
- 失败的写 N/A，并在"失败"一节给出一句话核心报错和 error.log 链接。

### 5.2 对比图

每个模型一张 PNG，存为 `charts/<slug>.png`。数据全部取自 `history.csv`，每次连同本次结果一起重画，不手工改图。

- **形式**：2×2 小多图（small multiples）。两列是并发 4 和 64；两行是每卡吞吐和第二个指标（endpoint 为 interactivity，8k1k 为 ITL P50）。每一格用自己的 y 轴，从 0 开始，不用双 y 轴。
- **x 轴**：同场景最近 20 次测试的日期（`MM-DD`）。
- **历史**：灰色 `#8A8F98` 折线加小圆点。失败的那次在 x 轴上画灰色 `×`，折线在那里断开。
- **本次**：深红 `#9B1C1C` 菱形，比历史点大，旁边直接标数值和"vs 上次"的百分比。本次失败时，在格子里写 `this run: N/A`。
- **历史最佳**：灰色细虚线，不含本次。ITL 取最小值。
- **标题**以模型名开头，例如 `Kimi-K3 · MLPerf Endpoint: this run vs history`。图注两行：一行写 GPU 数和镜像 tag，一行写指标口径。
- 图里的文字用英文，和现有的 MLPerf Endpoint 图一致，也省掉中文字体的麻烦。单张 PNG 不超过 300 KB。

为什么用这种图：要回答的是"这次比以前好还是差"，也就是随时间的变化，所以用折线，把本次高亮、历史压成灰色背景。两个指标、两个并发的量级差得很远，所以拆成小多图，而不是挤在一张图里或用双 y 轴。吞吐对 interactivity 的帕累托图适合跟竞品比，但每次只有两个点，日期又只能靠颜色区分，看回退不直观，所以 CI 不用它。

### 5.3 提交到 455-CI

```bash
git clone --branch 455-CI --single-branch git@github.com:ROCm/ATOM.git atom-455-ci
cd atom-455-ci
# 放入 recipes/455-CI/runs/<run_id>/，追加 history.csv，在 README.md 的索引表最上面加一行
git add recipes/455-CI
git commit -m "[455-CI] <run_id> results"
git push origin 455-CI
```

- 一次测试一个 commit。commit message 不加 `Co-Authored-By` trailer，这是 ROCm/ATOM 维护者的要求。
- 不 force push，不改写历史。push 被拒就先 `git pull --rebase`，再推。

`recipes/455-CI/README.md` 的索引表格式：

```markdown
| 日期 | 场景 | 测试 image | 结果 | 汇总 |
|---|---|---|---|---|
| <YYYY-MM-DD> | MLPerf Endpoint | `gfx1250-atom-<YYYYMMDD>` | 例：3/4 通过，Kimi-K3 失败；⚠️ 1 项 | [summary](runs/<run_id>/summary.md) |
```

## 6. 待补充

- `455-CI-recipe.md`：各模型的 server 和 bench 命令，按 §3.1 的结构写。
- DeepSeek R1 和 Qwen-35B-A3B 的 interactivity 口径（agentic 还是 single-turn）。
- 对比图的渲染脚本：按 §5.2 从 `history.csv` 出图。放进 `455-CI/` 后，在 §5.2 写明用法。
