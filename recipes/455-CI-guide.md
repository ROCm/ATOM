# 455-CI 测试指南

455（gfx1250）CI 的执行手册：测哪些模型、用哪个镜像、怎么跑、记录哪些数字、怎么出报告。各模型的命令放在同目录的 [`455-CI-recipe.md`](455-CI-recipe.md)，本文件只规定流程和口径。

**目的**：暴露热点模型在 ATOM main + aiter main 上的支持度和覆盖度，也就是能不能跑起来、精度对不对、性能是多少。所以只用 main，依赖也跟着 main 走。跑挂本身就是结果，要如实记录。

**每周两场测试，共用一个镜像**。一场只测 4 个模型，每场出一份报告：

| 场次 | 时间 | 机器 | 测什么 | 镜像 |
|---|---|---|---|---|
| 周末验证 | 每周六 10:30–22:30，12 小时 | 2 个节点：e15-06、e15-07 | MLPerf Endpoint 的 4 个模型 | 开测前按 §2 构建本周镜像：取 `gfx1250-atom-<YYYYMMDD>` 系列里最新的一个作 base，装上当时的 ATOM main 和 aiter main，推成 `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-<周六日期>` |
| 周中验证 | 每周三 17:30–22:30，5 小时 | 1 个节点：e15-09 | 8K/1K 的 4 个模型 | 不构建，复用上一个周六的镜像（§2.5） |

- 每场按 §3 逐个模型测，每个模型先测 GSM8K 精度、再测性能。跑挂了就记下核心 error log，接着测下一个模型。机器挂了不算跑挂：告诉 user，等 user 重启恢复机器后接着测没测完的（§3.6）。
- **每测完一个模型，就把它的精度和性能数据推到 `455-CI` 分支**（§3.5），不攒到最后。数据按 §4 记录，历史数据都留在 `recipes/455-CI/` 里。
- 全部测完后按 §5 出本场的报告 `455-CI-report-<场次日期>.md`：一个 md 文件，前面是简短 summary（Highlights 和总表，给老板看），后面是详细 summary。推到 `455-CI` 后，最后给出这个 md 文件的路径（§5.6）。
- 只有周六构建镜像。周六构建失败时，这一周的两场都记 N/A（§2.1）。

**前置条件**：

- docker 已登录，且对 `rocm/fw-bringup-mlperf` 有 push 权限。账号和 token 找负责人要。token 不要出现在命令行上（会留在 shell history 和进程列表里），也不要写进脚本或这个仓库（455-CI 分支是公开的）：

  ```bash
  read -rs DOCKERHUB_TOKEN && echo "$DOCKERHUB_TOKEN" | docker login -u <账号> --password-stdin
  ```

- 测试机能访问 GitHub 和 PyPI。
- 对 `ROCm/ATOM` 的 `455-CI` 分支有 push 权限。开测前在每个测试节点上 clone 这个分支，镜像信息（§2.4）、每个模型的数据（§3.5）和报告（§5.6）都提交到这里：

  ```bash
  git clone --branch 455-CI --single-branch git@github.com:ROCm/ATOM.git atom-455-ci
  CI_DIR=$PWD/atom-455-ci/recipes/455-CI
  ```

## 1. 测试矩阵

| 场次 | 场景 | 模型 | slug | 规模 | GPU 数 | 并发 | 时长 |
|---|---|---|---|---|---|---|---|
| 周六 | MLPerf Endpoint | DeepSeek R1 | `deepseek-r1` | 1 tray x 1P4G | 4 | 64、4 | 10 min |
| 周六 | MLPerf Endpoint | DeepSeek-V4.1-Flash | `deepseek-v4.1-flash` | 1 tray x 1P4G | 4 | 64、4 | 10 min |
| 周六 | MLPerf Endpoint | Kimi-K3 (2.8T) | `kimi-k3` | 2 node, 2 tray x 1P4G | 8 | 64、4 | 10 min |
| 周六 | MLPerf Endpoint | Qwen-35B-A3B | `qwen-35b-a3b` | 1 card | 1 | 64、4 | 10 min |
| 周三 | 8K/1K | DeepSeek-V4-Pro | `deepseek-v4-pro` | 1 tray x 1P4G | 4 | 64、4 | — |
| 周三 | 8K/1K | GLM-5.2 | `glm-5.2` | 1 tray x 1P4G | 4 | 64、4 | — |
| 周三 | 8K/1K | MiniMax-M3 (428B) | `minimax-m3` | 1 tray x 1P4G | 4 | 64、4 | — |
| 周三 | 8K/1K | Qwen-3.8 Flash | `qwen-3.8-flash` | 1 card | 1 | 64、4 | — |

- **DeepSeek R1 的权重固定用 [`amd/Deepseek-S3_sq_a05_v2_mlperf6_1`](https://huggingface.co/amd/Deepseek-S3_sq_a05_v2_mlperf6_1)**，即 MLPerf v6.1 closed division 的 Quark MXFP4 权重。不要换成其他 DeepSeek R1 checkpoint，换了数字就和历史对不上。
- **GPU 数**是每卡吞吐的分母：1 tray x 1P4G 为 4 张，2 tray 为 8 张，1 card 为 1 张。
- **slug** 用作目录名，以及 `perf.csv`、`accuracy.csv` 的 `model` 列。定下来就不要改，否则新旧数据对不上。
- 每个模型都先测 GSM8K 精度（§3.3），再测表里的性能。
- 8K/1K 的输入输出长度、请求数等参数以 recipe 为准。

## 2. 准备本周镜像（每周六一次）

### 2.1 规则

- **每周只建一次**：周六开测前建，周三那场直接用它（§2.5），不构建。
- **base 取 `gfx1250-atom-<YYYYMMDD>` 系列里最新的一个**：只看不带后缀的纯日期 tag，也就是大家每天出的 bring-up 镜像，例如 `gfx1250-atom-20261111`。最新的那个 pull 不下来或明显有问题时，往前取一个，并在报告里写明。base 每周都会变，周与周之间的差异也可能来自 base（ROCm、torch、triton、mori 等），所以报告里要写出 base。
- **只重装 ATOM 和 aiter，且只用 `ROCm/ATOM` 和 `ROCm/aiter` 的 `main`。** 不 checkout 私有分支或 PR，不 cherry-pick，不打 patch。base 里自带的补丁（如 `/app/endpoints-recipe-*/patches`）也不用。
- **其他组件保持 base 原样**（torch、triton、mori 等），只有 ATOM / aiter main 自己锁定版本的依赖跟着 main 走。例如 aiter main 锁定 `flydsl==0.5.0`，base 里的版本不同时，安装会把它换掉。这正是本 CI 要暴露的状态：不要手动换回，把变化记进 `image.json` 的 `dependency_changes`（§2.4）。
- **构建失败**（clone、安装或 import 不通过）时，这一周两场的所有模型都记 N/A，error 写构建的核心报错。周三不另建镜像，要补建先和负责人确认。不要改用私有分支凑合。
- **构建时机器挂了不算构建失败**：按 §3.6 告诉 user。恢复后先 `docker rm -f ci455-build-${DATE}`，再从 §2.2 重新来；§2.2 的同名检查会说明本周镜像是不是已经推上去了。

### 2.2 选 base、命名和同名检查

先定义查远端 tag 的函数，后面选 base 和查同名都用它：

```bash
# 输出 present 或 absent；查询本身失败（未登录、网络问题）时返回非 0
remote_tag_state() {
  local out
  if out=$(docker manifest inspect "$1" 2>&1); then echo present; return 0; fi
  if grep -q "no such manifest" <<<"$out"; then echo absent; return 0; fi
  echo "查询失败：$out" >&2; return 1
}
```

只有输出 `absent` 才算远端没有。`unauthorized` 之类的查询失败不能当成"不存在"，否则可能覆盖别人的镜像。

**选 base**：从今天往前找 7 天，取第一个存在的纯日期 tag。

```bash
BASE_IMAGE=""
for i in $(seq 0 7); do
  d=$(TZ=Asia/Shanghai date -d "-$i day" +%Y%m%d)
  state=$(remote_tag_state "rocm/fw-bringup-mlperf:gfx1250-atom-$d") || break    # 查询失败：先解决登录或网络
  if [ "$state" = present ]; then BASE_IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-$d; break; fi
done
echo "BASE_IMAGE=${BASE_IMAGE:-没找到，停下来通知负责人}"
```

**命名**：新镜像叫 `rocm/fw-bringup-mlperf:gfx1250-atom-ci455-<YYYYMMDD>`，日期取周六构建当天（北京时间），例如 `gfx1250-atom-ci455-20261114`。

- **不用 `gfx1250-atom-<YYYYMMDD>`**：那是大家每天出镜像的格式（还有 `-dsv41`、`-dsv4pro` 等后缀），周六很容易撞名。`ci455` 只给本 CI 用，现有的 tag 里都没有。
- **同一周重建**要先经负责人同意，tag 末尾加 `-r2`、`-r3`。
- **不另开新仓库**：往 org 里推一个不存在的仓库会自动创建它，可见性取决于 org 的默认设置，可能是公开的。

**同名检查**：开始构建前查一次，push 前再查一次。

```bash
DATE=$(TZ=Asia/Shanghai date +%Y%m%d)
NEW_IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-ci455-${DATE}
remote_tag_state "$NEW_IMAGE"
```

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

# base 里的 /app/ATOM 和 /app/aiter 在私有分支上（例如 jiaolyu/duty1111-main-rebase），
# 可能还有未提交的改动：整个删掉，重新 clone
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
diff /tmp/pkgs.before /tmp/pkgs.after    # 依赖变化，记进 image.json（§2.4）
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
  --change "LABEL com.amd.atom.commit=${ATOM_SHA} com.amd.aiter.commit=${AITER_SHA} image.atom_branch=main image.aiter_branch=main org.opencontainers.image.version=gfx1250-atom-ci455-${DATE}" \
  ci455-build-${DATE} "$NEW_IMAGE"

remote_tag_state "$NEW_IMAGE"          # push 前再查一次，必须是 absent
docker push "$NEW_IMAGE"
docker buildx imagetools inspect "$NEW_IMAGE" | grep -m1 Digest    # 记下 digest，写进报告

# 记下本周镜像的信息，报告的 Environment 从这里读。CI_DIR 见前置条件
mkdir -p "$CI_DIR/runs/$DATE"
NEW_IMAGE="$NEW_IMAGE" BASE_IMAGE="$BASE_IMAGE" ATOM_SHA="$ATOM_SHA" AITER_SHA="$AITER_SHA" \
DIGEST=$(docker buildx imagetools inspect "$NEW_IMAGE" | awk '/^Digest:/ {print $2; exit}') \
ATOM_LOG=$(docker exec ci455-build-${DATE} git -C /app/ATOM log -1 --date=short --format='%cd%x09%s') \
AITER_LOG=$(docker exec ci455-build-${DATE} git -C /app/aiter log -1 --date=short --format='%cd%x09%s') \
DEPS="<2.3 里 diff 出的依赖变化，逗号分隔；没有就留空>" \
python3 - "$CI_DIR/runs/$DATE/image.json" <<'EOF'
import json, os, sys
e = os.environ
def repo(sha, log):
    date, title = (e[log].split("\t", 1) + [""])[:2]
    return {"commit": e[sha], "date": date, "title": title}
json.dump({"image": e["NEW_IMAGE"], "digest": e["DIGEST"], "base_image": e["BASE_IMAGE"],
           "atom": repo("ATOM_SHA", "ATOM_LOG"), "aiter": repo("AITER_SHA", "AITER_LOG"),
           "dependency_changes": [d.strip() for d in e["DEPS"].split(",") if d.strip()]},
          open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
EOF
docker rm -f ci455-build-${DATE}
```

- 第三条 LABEL 覆盖从 base 继承来的同名 label。base 的这些 label 记的是私有分支，不覆盖会误导后来的人。其他继承来的 label（如 `dsv41.atom`）描述的是 base，不是本镜像，以 `com.amd.ci455.*` 为准。
- push 只上传新增的那一层，base 的层远端已经有了。
- `image.json` 不用单独提交，随第一个模型的数据一起推到 `455-CI`（§3.5）。

### 2.5 测试机使用本周镜像

周六的两个节点（e15-06、e15-07，Kimi-K3 两个都用）和周三的 e15-09 都 pull 同一个本周镜像。周三不构建、不推送，直接用上一个周六推的那个，开测前先核对：

```bash
NEW_IMAGE=rocm/fw-bringup-mlperf:gfx1250-atom-ci455-<本周六日期>    # 上一个周六推的镜像
docker buildx imagetools inspect "$NEW_IMAGE" --format '{{json .Image.Config.Labels}}' | tr ',' '\n' | grep ci455
docker pull "$NEW_IMAGE"
```

能看到 `com.amd.ci455.*` 这几个 label，而且 commit 和周六那场报告里写的一致，才开测。周六构建失败、没有本周镜像时，周三这场也记 N/A，原因同周六的构建失败。

## 3. 跑测试

### 3.1 命令从哪来

- server 和 bench 命令一律取自 `455-CI-recipe.md`。GSM8K 用 §3.3 的统一命令；某个模型需要不同参数时，写进 recipe，以 recipe 为准。
- 只替换和机器相关的部分（模型路径、端口、节点地址、结果目录），不改参数。要调参数，先改 recipe 再测，并在报告里注明。
- 实际执行的命令（变量展开后）存成 `server_cmd.sh`、`gsm8k_cmd.sh`、`bench_cmd.sh`，跟结果一起提交。提交前把内网地址和 token 换成占位符：455-CI 分支在公开仓库里。
- recipe 里没有某个模型时，该模型记 N/A，error 写"recipe 缺失"。

recipe 建议每个模型一节，方便按模型名找到命令：

````markdown
## Kimi-K3 (2.8T)            ← 和 §1 表格里的模型名一致
- 场景：MLPerf Endpoint；GPU：8（2 node, 2 tray x 1P4G）
- 并行：TP8 + DCP8（`perf.csv` 里记作 `tp8dcp8`）
- MODEL_PATH：<模型路径>；PORT：<端口>

### server
```bash
# 两个节点各写一段
```

### gsm8k（可选，只在和 §3.3 不同时写）
```bash
# 例如需要不同的 few-shot 数
```

### bench
```bash
# 用 ${CONC} 表示并发（4 / 64），${RESULT_DIR} 表示结果目录
```
````

### 3.2 每个模型的流程

`MODEL` 是 §1 的 slug。`RUN_DIR` 是测试机上本次的结果目录（挂进容器），每测完一个模型就按 §3.5 拷进 455-CI 分支并推送。`MODEL_PATH`、`PORT` 取自 recipe。

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
[ $ready = 1 ] || echo "server 没起来，按 3.4 处理"
rocm-smi --showmemuse        # 显存必须已被占用：/health 正常不代表模型加载好了

# 2) 先按 §3.3 测 GSM8K 精度

# 3) 按 recipe 做 warmup（不计入结果），然后依次跑并发 4 和 64
for CONC in 4 64; do
  RESULT_DIR="$RUN_DIR/$MODEL/c${CONC}"; mkdir -p "$RESULT_DIR"
  <bench 命令> > "$RESULT_DIR/bench.log" 2>&1
done

# 4) 停 server：ATOM 不理 SIGTERM，要 kill -9 主进程和 EngineCore 进程
kill -9 $(pgrep -f "[o]penai_server"); sleep 2; pkill -9 -f "ATOM[:][:]"
test "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0

# 5) 按 §3.5 把这个模型的数据推到 455-CI，再测下一个模型
```

- 精度和两个并发的性能用同一个 server，中间不重启。
- GSM8K 分数低但 server 还活着时，照常测性能，在报告里标出精度回退。
- MLPerf Endpoint 的 client 不要中途打断。ATOM 不会取消已断开连接的请求，打断之后必须重启 server、重新 warmup 再测。
- Kimi-K3 跨两个节点：镜像 ENV 里 `NCCL_SOCKET_IFNAME=lo`、`GLOO_SOCKET_IFNAME=lo` 只适合单机，要按 recipe 换成节点之间的真实网卡。
- 1 card 的模型：镜像 ENV 默认 `HIP_VISIBLE_DEVICES=0,1,2,3`，要按 recipe 限制成 1 张卡。

### 3.3 GSM8K 精度

口径和 ATOM CI（`.github/scripts/atom_test.sh accuracy`）一致：`lm_eval` 的 `local-completions`，3-shot，看 `exact_match,flexible-extract`。用 ATOM 自带的 `scripts/wait_infer_drain.sh` 盯着 server：engine 卡住或出 GPU 故障时尽快停下，不让 lm_eval 对着挂掉的 server 一直重试。

```bash
command -v lm_eval >/dev/null || PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install "lm-eval[api]"
ACC_DIR="$RUN_DIR/$MODEL/gsm8k"; mkdir -p "$ACC_DIR"
set -m                                   # lm_eval 放进单独的进程组，出问题时整组停掉
( lm_eval --model local-completions \
    --model_args "model=${MODEL_PATH},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=65,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k --num_fewshot 3 --output_path "$ACC_DIR" > "$ACC_DIR/lm_eval.log" 2>&1 ) &
LM_PID=$!
set +m
bash /app/ATOM/scripts/wait_infer_drain.sh "$PORT" 30 10 "$ACC_DIR/lm_eval.log" 18
DRAIN_RC=$?                              # 0 跑完；1 卡住；2 GPU 故障；4 超过 30 分钟
if [ $DRAIN_RC != 0 ]; then
  kill -TERM -- -"$LM_PID" 2>/dev/null; echo "GSM8K 没跑完（rc=$DRAIN_RC），按 3.4 处理"
fi
wait "$LM_PID"

# 读分数：flexible-extract 是报告的分数，strict-match 也记下（精度图上画成虚线）
python3 - "$ACC_DIR" <<'EOF'
import json, pathlib, sys
f = max(pathlib.Path(sys.argv[1]).rglob("results*.json"), key=lambda p: p.stat().st_mtime)
r = json.load(open(f))["results"]["gsm8k"]
print("flexible-extract:", r["exact_match,flexible-extract"], " strict-match:", r["exact_match,strict-match"])
EOF
```

- `MODEL_PATH` 要和 server 的 `--model` 一致：它也是请求里的模型名。
- 一个模型的 GSM8K 一般要 5 到 15 分钟。
- 3-shot 和本地常用的 5-shot 不能直接比，历史数据只和 3-shot 比。

### 3.4 跑挂了

下面任一情况都算失败：server 没起来或中途退出；GSM8K 或 bench 报错退出、卡住；结果文件缺失或不完整。机器挂了不算，按 §3.6 处理。

1. **不盲目重试。** 纯环境问题（GPU 没释放、端口被占、下载失败）修好后可以重跑一次，并在报告里注明"重跑"。
2. **摘核心 error log**，存到 `$RUN_DIR/$MODEL/error.log`，不超过 100 行。报告详细 summary 的 Failures 一节会贴出它的前 60 行：

   ```bash
   LOG="$RUN_DIR/$MODEL/server.log"      # client 侧失败就换成对应的 lm_eval.log 或 bench.log
   n=$(grep -n -m1 -E "Traceback|Memory access fault|HIP error|hipError|out of memory|Segmentation fault|core dumped|AssertionError|RuntimeError|Error:" "$LOG" | cut -d: -f1)
   { [ -n "$n" ] && sed -n "$(( n > 15 ? n - 15 : 1 )),$(( n + 40 ))p" "$LOG"
     echo "----- last 30 lines -----"; tail -n 30 "$LOG"; } | head -n 100 > "$RUN_DIR/$MODEL/error.log"
   ```

3. **一句话报错**：取最后那行异常（例如 `RuntimeError: ...`），截到 150 个字符以内，写进 `accuracy.csv` 或 `perf.csv` 的 `error` 列和报告。
4. server 挂了的话，这个模型还没测的项（精度或性能）都记 N/A，原因相同。按 3.2 第 4 步把 GPU 清干净，按 §3.5 上传（跑挂的结果也要上传），再测下一个模型。

CI 只负责记录。要查根因就另开任务，从 ATOM 的 `/debug-guide` 开始。

### 3.5 每测完一个模型就上传

一个模型测完（包括跑挂）后，马上把它的精度和性能数据推到 `455-CI`，再测下一个模型。这样中途出问题也不会丢掉已经测完的数据，大家随时能在分支上看到进度。

1. 按 §4.4 挑出要提交的结果文件，从 `$RUN_DIR/$MODEL` 拷到 `$CI_DIR/runs/<YYYYMMDD>/$MODEL/`。跑挂的话要包括 `error.log`。
2. 往 `$CI_DIR/perf.csv` 追加这个模型的 c4、c64 两行，往 `accuracy.csv` 追加一行（§4.2、§4.3）。跑挂的也写，数字留空。
3. 提交并推送：

   ```bash
   D=<YYYYMMDD>                            # 场次日期
   cd "$CI_DIR"
   git add perf.csv accuracy.csv "runs/$D"
   git commit -q -m "[455-CI] $D $MODEL"
   for i in 1 2 3; do                      # 另一个节点可能刚推过：先 rebase 到远端最新，再推
     git pull -q --rebase --autostash origin 455-CI && git push -q origin 455-CI && break
     sleep 10
   done
   git status -sb | head -1                # 出现 [ahead N] 就是还没推上去
   ```

- 周六第一个模型的这次提交会把 `runs/<YYYYMMDD>/image.json`（§2.4）一起带上去。
- 两个节点同时追加 CSV 不会冲突：`recipes/455-CI/.gitattributes` 给 `perf.csv`、`accuracy.csv` 设了 `merge=union`，rebase 时两边新加的行都保留。其他结果文件按模型分目录，互不影响。
- 推不上去（例如网络问题）时本地 commit 还在，可以先接着测，下一个模型上传时会一起推。不要 force push。
- commit message 不加 `Co-Authored-By` trailer，这是 ROCm/ATOM 维护者的要求。§5.6 的提交也一样。

### 3.6 机器挂了

测的过程中机器本身挂了，不是模型或软件的问题，**不记成错误**：不写 `perf.csv`、`accuracy.csv` 的行，也不写 `error.log`。停下来告诉 user，请 user 重启、恢复机器，恢复后接着测没测完的模型。机器由 user 来重启和修复，agent 不要自己重启。

**先分清是机器挂了还是跑挂了。** 测试停下来后，在这个模型用到的每个节点上查（Kimi-K3 是两个）：

```bash
uptime -s                                                         # 开机时间晚于开测时间：中途重启过
rocm-smi --showid | grep -o '^GPU\[[0-9]*\]' | sort -u | wc -l    # GPU 数，应该和开测前一样（1P4G 是 4）
# 按 3.2 第 4 步清掉进程，等 2 分钟再看
ls /sys/class/kfd/kfd/proc | wc -l                                # 应该是 0
```

| 现象 | 算什么 |
|---|---|
| 节点登不上（ssh 连不上、命令一直不返回）；中途重启过；GPU 少了或 `rocm-smi` 报错；进程杀不掉、GPU 一直释放不了；docker 没响应 | 机器挂了，按本节处理 |
| 机器正常，只是 server、lm_eval 或 bench 报错退出或卡住；清掉进程后 GPU 都在、都空闲 | 跑挂了，按 §3.4 记录 |

拿不准就先不记，问 user。周六只挂了一个节点时，另一个节点上的测试照常进行。

**告诉 user**：哪个节点、什么时候、什么现象，当时在测什么，哪些已经测完上传、哪些还没测。例如：

> e15-07 在 14:20 挂了：ssh 连不上。当时在测 Kimi-K3 的 c64 性能。已经测完并上传的：DeepSeek R1、DeepSeek-V4.1-Flash；还没测完的：Kimi-K3、Qwen-35B-A3B。请重启恢复 e15-07，好了告诉我，我从 Kimi-K3 接着测。

agent 自己就跑在挂掉的机器上、会话也断了时，user 恢复机器后新开一个会话，从下面的"恢复后接着测"开始。

**恢复后接着测**：

```bash
# 1) 机器好了：GPU 都在、都空闲，本周镜像还在（NEW_IMAGE 见 §2.5）
rocm-smi --showid | grep -o '^GPU\[[0-9]*\]' | sort -u | wc -l    # 和开测前一样
test "$(ls /sys/class/kfd/kfd/proc | wc -l)" = 0
docker image inspect "$NEW_IMAGE" >/dev/null 2>&1 || docker pull "$NEW_IMAGE"

# 2) 本场哪些模型已经上传了，以 455-CI 分支为准（checkout 不在了就按前置条件重新 clone）
cd "$CI_DIR"
git status -sb | head -1        # 有 [ahead N]：挂之前 commit 了还没推，先按 §3.5 的循环推上去
git pull -q --rebase --autostash origin 455-CI
awk -F, -v d=<YYYY-MM-DD> '$2 == d {print $4}' perf.csv accuracy.csv | sort -u
```

- 本场的模型里，列表上没有的都要测。挂掉时正在测的那个从头重测：重起 server，GSM8K 和两个并发都重来，不用挂之前的半截结果。测完照常按 §3.5 上传，`note` 注明是机器恢复后重测的。
- 同一个模型重测时机器又挂了，可能是这个模型把机器弄挂的：停下来把两次的现象告诉 user，由 user 决定怎么记。不要自己记成跑挂，也不要测第三次。
- 到场次结束时间还没恢复或没测完，问 user 是延后接着测，还是这场到此为止。没测成的模型不写任何行：报告里显示 —（没有结果），Highlights 里也不会出现，不算跑挂。

## 4. 记录

### 4.1 指标口径

**GSM8K 精度**：lm_eval 结果 JSON 里的 `results.gsm8k["exact_match,flexible-extract"]`，3-shot，这是报告的分数。`exact_match,strict-match` 也记下，精度图上和 dashboard 一样画成虚线作参考。

**MLPerf Endpoint**：读 client 输出的 `report_dir/performance/result_summary.json`。

| 指标 | 计算 |
|---|---|
| 每卡吞吐 (tok/s/GPU) | `tps ÷ GPU 数`。`tps` 即 `system_tps`：client 数出的输出 token ÷ 测量窗口 |
| E2E average interactivity (tok/s/user) | `e2e_avg_interactivity`。不能用整机 tok/s ÷ 并发代替 |

- 四个模型都用这两个口径，和 MLPerf Endpoints 官网 agentic 图、我们已有的 MI455 图一致。
- `n_samples_failed` 不为 0 时照常记数字，在 `note` 列写明失败数。

**8K/1K**：读 ATOM `benchmark_serving` 的结果 JSON，口径和 ATOM benchmark dashboard 一致。bench 命令要带 `--save-result`；默认的 `--percentile-metrics` 已经包含 itl。

| 指标 | 计算 |
|---|---|
| 每卡吞吐 (tok/s/GPU) | `total_token_throughput ÷ GPU 数`（输入加输出 token），即 dashboard 的 `tput_per_gpu` |
| ITL P50 (ms) | `median_itl_ms` |
| Interactive (tok/s/user)，只用于性能图 | `1000 ÷ mean_tpot_ms`，即 dashboard 上 Throughput vs Latency 图的横轴 |

- `completed` 应该等于发出的请求数。`errors` 里有非空项时，把个数写进 `note`。

### 4.2 perf.csv

每个模型的每个并发，每次测试写一行。失败也写，数字留空。只追加不修改；历史行要更正时单独提一个 commit，并在 message 里写明原因。同一场（同一天）重跑同一个模型时追加新行，`note` 写"重跑"，报告和画图取这一场最后一次的结果。

| 列 | 内容 |
|---|---|
| `image_date` | 所用镜像的日期，也就是构建它的那个周六，`YYYY-MM-DD`。只用来追溯镜像 |
| `test_date` | 场次日期（周六或周三，北京时间），`YYYY-MM-DD`。报告、对比和画图都以场次为单位 |
| `scenario` | `endpoint` 或 `8k1k` |
| `model` | §1 的 slug |
| `gpus` | 每卡吞吐的分母 |
| `parallel` | 并行策略，照 recipe 里该模型"并行"那一行写，小写、不加空格，例如 `tp8dcp8`、`tp4`。报告的详细 summary 里显示的是这一列 |
| `concurrency` | `4` 或 `64` |
| `status` | `ok` 或 `fail` |
| `tok_s_per_gpu` | 每卡吞吐，按 §4.1 计算，保留两位小数 |
| `interactivity` | tok/s/user，只有 endpoint 填 |
| `itl_p50_ms` | 只有 8k1k 填 |
| `tpot_mean_ms` | 只有 8k1k 填，性能图的横轴由它换算 |
| `image`、`base_image` | 测试镜像 tag、base 镜像 tag |
| `atom_branch`、`atom_commit` | `main`、40 位 sha |
| `aiter_branch`、`aiter_commit` | `main`、40 位 sha |
| `error` | 失败时的一句话核心报错 |
| `note` | 其他说明，如重跑、失败请求数 |

### 4.3 accuracy.csv

每个模型每次测试写一行，规则同 `perf.csv`。

| 列 | 内容 |
|---|---|
| `image_date`、`test_date` | 同 `perf.csv` |
| `scenario` | `endpoint` 或 `8k1k` |
| `model` | §1 的 slug |
| `status` | `ok` 或 `fail` |
| `gsm8k` | `exact_match,flexible-extract`，保留四位小数 |
| `gsm8k_strict` | `exact_match,strict-match`，保留四位小数 |
| `num_fewshot` | `3`；recipe 改过就写实际值 |
| `image`、`base_image` | 测试镜像 tag、base 镜像 tag |
| `atom_branch`、`atom_commit` | `main`、40 位 sha |
| `aiter_branch`、`aiter_commit` | `main`、40 位 sha |
| `error` | 失败时的一句话核心报错 |
| `note` | 其他说明，如重跑 |

两个 CSV 都用 UTF-8、逗号分隔，含逗号的字段加双引号，表头按上表顺序。

精度图上的 baseline 参考线来自可选的 `accuracy_baseline.csv`，两列：`model`（slug）和 `baseline`（分数，例如模型卡或 ATOM CI 的 baseline）。没有这个文件就不画。

### 4.4 455-CI 分支的目录结构

`perf.csv`、`accuracy.csv` 已经建好，只有表头；`README.md` 在第一次提交报告时创建。

```text
recipes/
├── 455-CI-guide.md        本文件
├── 455-CI-recipe.md       各模型的命令
└── 455-CI/
    ├── README.md          报告索引，一场一行，最新的在最上面
    ├── perf.csv           全部历史性能
    ├── accuracy.csv       全部历史 GSM8K 精度
    ├── accuracy_baseline.csv   可选：精度图上的 baseline 参考线
    ├── owners.csv         每个模型的 owner，Highlights 里 @ 他们（§5.1）
    ├── .gitattributes     两个 CSV 设了 merge=union，两个节点同时上传不冲突（§3.5）
    ├── tools/             生成图和报告的脚本、样例（§5.5）
    ├── reports/           一场一份，YYYYMMDD 是场次日期
    │   └── 455-CI-report-<YYYYMMDD>.md   简短 summary（Highlights 和总表）+ 详细 summary
    └── runs/<YYYYMMDD>/   一场一个目录：本场的原始结果和图
        ├── image.json     只在周六的目录里：本周镜像的信息（§2.4）
        ├── charts/        perf-<scenario>.png、accuracy-<scenario>.png
        └── <slug>/        本场每个模型一个目录
            ├── server_cmd.sh  gsm8k_cmd.sh  bench_cmd.sh
            ├── gsm8k/         lm_eval 的 results_*.json
            ├── c4/  c64/      性能结果文件
            └── error.log      只在失败时有
```

结果文件提交这些：

- **GSM8K**：lm_eval 输出目录里的 `results_*.json`。不加 `--log_samples`，文件很小。
- **endpoint**：`performance/result_summary.json`、`report.txt`、`accuracy/accuracy_results.json`。不提交 `events.jsonl`（K3 每跑一轮数据集约 6 GB）。
- **8k1k**：结果 JSON 只留标量字段。原文件带每个请求的 itl 和生成文本，一个点就可能有几 MB：

  ```bash
  python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); json.dump({k: v for k, v in d.items() if not isinstance(v, list)}, open(sys.argv[2], "w"), indent=1)' raw.json result.json
  ```

- 不提交完整的 server log 和 lm_eval.log、模型文件、token 或密码。单个文件不超过 1 MB。

## 5. 报告

全部模型测完、数据都按 §3.5 推上去之后，生成本场的报告。**一场一个 md 文件**：`reports/455-CI-report-<YYYYMMDD>.md`，日期是场次日期，例如周六 `455-CI-report-20261114.md`、周三 `455-CI-report-20261118.md`。**报告用英文写**，分两部分：

| 部分 | 给谁看 | 内容 |
|---|---|---|
| 简短 summary（`## Summary`） | 老板 | Highlights 和总表（§5.1、§5.2） |
| 详细 summary（`## Details`） | 要看细节的人 | 测试环境、全部数字、趋势图、跑挂 case 的核心 error log（§5.3、§5.4） |

报告和图都用 §5.5 的脚本从 `perf.csv`、`accuracy.csv` 生成，不手填数字。样例见 [`455-CI/tools/examples/reports/`](455-CI/tools/examples/reports/)，那是 EXAMPLE（示例）：数字都是编的，不是真实测试数据。

### 5.1 Highlights

大老板只看 Highlights，所以只写本次出了大事或跑挂的模型，每个一行：`<标记> <模型>: <发生了什么>`。不写场景，只给一个关键数字，例如 `⚠️ GLM-5.2: throughput down 4.2%.`、`❌ Qwen-35B-A3B: c64 failed (RuntimeError: …).`。持平、变化不大或第一次有数据的模型都不提；一个都没有时只写 `No failures or major changes.`。

每条后面 @ 这个模型的 owner，让负责的人看到，例如 `❌ Qwen-35B-A3B: c64 failed (…). @owner`。owner 名单在 `recipes/455-CI/owners.csv`（两列：`model` 是 §1 的 slug，`owner` 是要 @ 的人，多人用空格隔开），由测试负责人自己填写和维护，没填就不 @。用 GitHub 账号还是 IM 里的名字，取决于 Highlights 贴到哪里：仓库里的 md 文件写 @ 不会触发 GitHub 通知，真正提醒到人的是把 Highlights 贴进 issue 或群聊那一下。有消息的模型没登记 owner 时，`make_report.py` 会打警告，提交前补上。

`make_report.py` 按下表比较本次和历史结果，生成草稿；测试 agent 读一遍再提交，可以改措辞，但不能删掉 ❌ 和 🔴 的条目，也不能改数字。

所有比较都对该模型最近一次成功的那一场（"上一场"），不和更早的最好成绩比。

| 标记 | 什么时候算 | 写法 |
|---|---|---|
| ❌ | 任一并发或 GSM8K 跑挂 | `c64 failed (<一句话报错>)`、`all runs failed (…)` |
| 🔴 | 吞吐比上一场低 10% 及以上，或 GSM8K 掉超过 0.06 | `throughput down 11.0%`、`GSM8K down 7.0 points to 88.0%` |
| ⚠️ | 吞吐比上一场低 3% 及以上，或 GSM8K 掉超过 0.04 | `throughput down 4.2%`、`GSM8K down 4.6 points to 88.1%` |
| 🟢 | 吞吐比上一场高 3% 及以上 | `throughput up 6.9%` |
| ✅ | 上一场跑挂，这一场跑通 | `recovered from the previous failure` |

- throughput 指并发 64 的每卡吞吐，和总表一致。并发 4 按同样的门槛判断，但只在 c64 没变化、或并发 4 更严重时才写，例如 `c4 throughput down 5.0%`。
- GSM8K 看 flexible-extract，差值按点算（0.01 = 1 point）。
- 门槛的来由：吞吐跑到跑之间有 1% 到 2% 的波动；GSM8K 在这套环境上 temp 0 也不完全确定，±0.02 以内通常是噪声。

### 5.2 简短 summary：Highlights 和总表

报告开头的格式如下（周六那场，数字是编的，owner 是占位名）：

```markdown
# 455-CI report · <YYYY-MM-DD> · MLPerf Endpoint

Image `gfx1250-atom-ci455-<YYYYMMDD>` · ATOM `main@<sha 前 9 位>` · aiter `main@<sha 前 9 位>`

## Summary

### Highlights

- ❌ Qwen-35B-A3B: c64 failed (RuntimeError: HIP error: out of memory). @qwen35b_owner
- 🟢 DeepSeek-V4.1-Flash: throughput up 6.9%. @v41_flash_owner
- ✅ Kimi-K3: recovered from the previous failure. @k3_owner

### Results

| Model | Scenario | 2026-11-07 (tok/s/GPU) | 2026-11-14 (tok/s/GPU) | Delta | GSM8K delta |
|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | MLPerf Endpoint | 2,655 | 2,838 | 🟢 +6.9% | −0.2 pts (now 93.9%) |
| Kimi-K3 (2.8T) | MLPerf Endpoint | 336 (2026-10-31) | 340 | +1.0% | −0.1 pts (now 96.1%) (vs 2026-10-31) |
| Qwen-35B-A3B | MLPerf Endpoint | 6,345 | N/A | N/A | −0.1 pts (now 90.1%) |

## Details

……（详细 summary，§5.3）
```

- 一个模型一行，依次是 Model、Scenario、上一场、本场、Delta、GSM8K delta。表格下面不加说明文字。
- 第三、四列是并发 64 的每卡吞吐（口径见 §4.1：endpoint 是输出 token，8K/1K 是输入加输出 token），表头写上一场和本场的日期。上一场取该模型最近一次成功的那一场：例子里 Kimi-K3 上一场（11-07）跑挂了，就和 10-31 那场比，并在格子里注明日期。
- Delta 只写百分比，例如 `+6.9%`。GSM8K delta 写掉了多少、现在多少，例如 `−4.6 pts (now 88.1%)`。
- N/A 表示跑挂、没有值；— 表示没有结果（例如机器挂了没测成，§3.6）；first 表示第一次有数据，没有可比的。

### 5.3 详细 summary

紧接在简短 summary 后面，`## Details` 下依次是以下几节，全部英文：

1. **Environment**：测试 image 和 digest、ATOM 和 aiter 的 branch 与 commit（带 commit 日期和标题）、测试日期。镜像信息取自构建它的那个周六目录里的 `image.json`（§2.4），缺的项从 CSV 里补。
2. **Performance**：每个模型每个并发一行，列出并行策略（例如 `tp8dcp8`，不写 GPU 数）、tok/s/GPU 和第二指标（endpoint 是 E2E avg interactivity，8K/1K 是 ITL P50），以及各自和上一场比的变化。变化列的表头写上一场的日期，例如 `vs 2026-11-07`；某个模型上一场跑挂了，就和它最近一次成功的那场比，并在格子里注明，例如 `+1.0% (vs 2026-10-31)`。
3. **Accuracy**：flexible-extract、strict-match、上一场的分数（表头写日期）和变化，写法同上。
4. **Trends**：两张趋势图，见 §5.4。
5. **Failures**：每个跑挂的 case 一小节，写一句话报错、`error.log` 的链接，以及核心 error log 段落（`error.log` 的前 60 行，§3.4 摘出来的那份）。

### 5.4 趋势图

只跑两个并发，画成帕累托图只是一条线段或一个直角，看不出变化，所以按场次画趋势。每场两张，存在 `runs/<YYYYMMDD>/charts/`：

- **`perf-<scenario>.png`**：每卡吞吐随场次的变化，一个模型一个并发一格，左 c4、右 c64，最近 12 场。每格右上角写最新值和对比上次的百分比；跑挂的那场在 x 轴上画灰色 `×` 并标 N/A。
- **`accuracy-<scenario>.png`**：照 ATOM benchmark dashboard 的精度趋势图，一个模型一格，最近 12 场。flexible-extract 实线，strict-match 虚线；baseline 是灰色虚线，在 `accuracy_baseline.csv` 里设了才画，数值写在格子标题行右侧；最新分数标在最后一个点右边。
- 两张图都用 dashboard 的浅色样式和深红 `#9B1C1C`，字体 Instrument Sans，2 倍分辨率 PNG，2640 px 宽。图里文字用英文，脚注两行写数据来源和口径。

### 5.5 生成图和报告

脚本在 `recipes/455-CI/tools/`，需要 Python 3.10 以上和 matplotlib，说明见那里的 [`README.md`](455-CI/tools/README.md)：

```bash
cd recipes/455-CI/tools
python3 fetch_fonts.py                              # 第一次用时下载 Instrument Sans；不下也能画
python3 make_charts.py --date <YYYY-MM-DD>          # 场次日期（周六或周三），写 runs/<YYYYMMDD>/charts/
python3 make_report.py --date <YYYY-MM-DD>          # 写本场的报告（一个 md），打印 Highlights 草稿，最后一行是报告路径
```

- 每场测完跑一次，连同历史数据重新生成。
- 生成后按 §5.1 审一遍 Highlights，再提交。
- 改了脚本先跑测试：`python3 -m pytest -q recipes/455-CI/tools`。

### 5.6 提交报告，给出路径

各模型的数据已经在 §3.5 推上去了，这里只提交本场的图、报告和 README 索引：

```bash
cd "$CI_DIR"
git pull -q --rebase --autostash origin 455-CI    # 先拉下另一个节点推的数据，报告才完整
# 按 §5.5 生成图和报告，审过 Highlights，在 README.md 的索引最上面加一行（见下）
git add README.md reports runs/<YYYYMMDD>/charts
git commit -q -m "[455-CI] <YYYY-MM-DD> MLPerf Endpoint report"    # 周三写 8K/1K report
git pull -q --rebase --autostash origin 455-CI && git push -q origin 455-CI
git status -sb | head -1                # 不能有 [ahead N]
```

- 不 force push，不改写历史；commit message 不加 `Co-Authored-By` trailer。

`recipes/455-CI/README.md` 的索引表一场一行，最新的在最上面：

```markdown
| 报告 | 场景 | 镜像 | ATOM / aiter | 结果 |
|---|---|---|---|---|
| [<YYYY-MM-DD>](reports/455-CI-report-<YYYYMMDD>.md) | MLPerf Endpoint | `gfx1250-atom-ci455-<YYYYMMDD>` | `<sha 前 9 位>` / `<sha 前 9 位>` | 例：3/4 跑通，❌ 1 项，⚠️ 1 项 |
```

**最后给出报告 md 文件的路径。** push 成功后，在本场的收尾消息里写出报告的仓库路径和 GitHub 链接。`make_report.py` 输出的最后一行就是它的本地路径。例如：

- `recipes/455-CI/reports/455-CI-report-20261114.md`
- https://github.com/ROCm/ATOM/blob/455-CI/recipes/455-CI/reports/455-CI-report-20261114.md

## 6. 待补充

- `455-CI-recipe.md`：各模型的 server 和 bench 命令（GSM8K 有特殊参数时也写上），按 §3.1 的结构写。
