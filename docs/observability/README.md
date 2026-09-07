# 用 CI 验证 ATOM 可观测性

在 `dpp-dcp-ci` 分支的 **Atomesh Benchmark** workflow 中手动运行：

| 输入 | 值 |
|---|---|
| `suite` | `nightly` |
| `case_names` | `glm-52-mxfp4-1p1d-cpp4-dcp4-agentic-lmcache-1m-c48` |
| `observability` | `on`（该 CPP4/DCP4 系列在 `auto` 下也启用） |
| `run_model_benchmark` | `true` |
| `publish_dashboard` | `true`：发布可直接打开的 HTML；`false`：仅下载 artifact |
| `atomesh_1p1d_nodes` | 可运行该模型的一台 8 GPU 节点 |

此用例：单节点，prefill 为 PP4×TP1，decode 为 TP4+DCP4，并发 48。
模型参数、AIPerf 结果 JSON、`server_metrics_export.*` 和现有跨 run 看板格式不变。
引擎使用 workflow checkout 的 Python 代码；MESH 使用所选镜像里的二进制。
CI 需要能下载固定版本 VictoriaMetrics/vmagent v1.111.0；也可提前将对应
`victoria-metrics-prod` / `vmagent-prod` 放进容器 PATH。

## 两档采集与埋点

| 层 | 粒度 | 内容 |
|---|---|---|
| 引擎输出事件 | 每次 callback | request ID、choice、输出时间、批次 token 数、累计 token 数 |
| 请求完成 | 每个完成 sequence | TTFT、TPOT、引擎完成耗时、输入/输出/缓存 token 数 |
| 调度埋点 | 每次调度/真实执行 | 调度 CPU 耗时、首次入队到执行等待、batch/query/context 统计 |
| 快档 `/metrics/scheduling` | **100ms** | 独立轻量快照，调度与 batch Histogram、最近 batch、队列状态 |
| 全量 `/metrics`、MESH、GPU | **1s** | 保留全部已有指标，加请求 Histogram 和事件写入状态 |

快档不执行 KV/cache 扫描，不做引擎 RPC，不读取 GPU tensor。引擎在现有
循环中推送快照，因此耗时 forward 或调度阻塞期间不能保证每 100ms 更新；
`atom:scheduling_snapshot_timestamp_seconds` 和页面的 snapshot age 曲线显示真实新鲜度。
完整快照刷新在 CI 中设为 1s，非 CI 默认仍为 5s。快档推送默认关闭，CI 显式启用。

两路采集不会重复入库：全量抓取仅在采集配置中排除新加的 scheduler/scheduling/
executed 指标，这些由快档独立抓取。**原 `/metrics` 端点仍包含这些新增指标**，
原有名称、类型、标签和累计语义保持不变。

## 数据口径

- Prefill、decode 独立端点、独立 `role`；快照保留 `rank`，batch 指标另带 `stage`。
- `atom:ttft_seconds` / `tpot_seconds` / `e2e_latency_seconds`：引擎输出回调时间，
  只统计完成 sequence；流式/非流式、文本/多模态、fanout 均覆盖。
  TTFT 包含 API 发起预处理到第一个引擎输出的时间；不包含 HTTP 入站前的时间。
- `atom:output_chunk_interval_seconds`：相邻非空输出批次的时间差。MTP 一批可有多个
  token，不能当作真实逐 token ITL。单 token prefill 没有 TPOT/输出间隔样本。
  客户端 SSE 到达间隔以原 AIPerf 指标为准；HTML 不推断不存在的客户端时间戳。
- `atom:scheduler_duration_seconds`：CPU 选择 batch 的耗时，含空调度轮次，
  不包含模型 forward、API 排队、日志统计收尾。
- `atom:scheduler_queue_seconds`：第一次进入调度器至首次提交模型执行的时间，
  包含等待 KV 就绪，**不包含后续抢占再排队**。请求被拒绝/取消且未执行时没有样本。
- `atom:executed_batch_size`：本轮真实请求数；在模型提交入口计数，不含 dummy、
  connector-only batch 和 CUDA Graph padding；DCP4 不乘 4，PP 仅 head 计数。
- `atom:executed_query_tokens`：每条 sequence 本轮新增 token 数，chunked prefill 为当前
  chunk，spec decode 为本轮调度的 query token 数。
- `atom:executed_context_tokens`：CPU `ScheduledBatch.context_lens`，prefill 为缓存长度加
  当前 chunk，decode 为本轮调度时上下文；不是原始 prompt 全长，也不是 GPU 实际访存量。
- `*_last`：最近一次真实 batch 的有效 batch size、query 总数及 query/context 的
  min/max/mean。空闲期间保留最后值，需结合 `last_execution_timestamp_seconds` 看。
- Histogram 为累计值，抓取不清零。请求延迟分位数用 30s 窗口，快档分布用 1s 窗口；
  分布可覆盖窗口内每一轮执行，100ms 的 last 快照则不保证保留每轮 batch 顺序。
- GPU 数据来自 AMD DRM/hwmon sysfs：card ID、利用率、显存、温度、功率；不占 GPU。
  某些驱动未提供温度/功率时对应曲线为空。card ID 是 DRM 编号，不保证等于 HIP 编号。

## CI 产物与验收

下载 `atomesh-model-benchmark-…` artifact，进入：

```text
…/slurm_job-<job_id>/observability/<combined 或 benchmark>/
  index.html                 # 单文件离线时间轴；100ms 数据可缩放至 10s 窗口
  validation.json            # 校验结果、完整曲线、请求事件汇总
  run.json                   # run/case/phase、端点、起止时间
  scrape.json                # 1s + 100ms 实际采集配置（JSON 是合法 YAML）
  timeseries.jsonl.gz         # VictoriaMetrics 原生 JSONL，回放/二次分析
  vmdata/                    # 本轮本地时序存储，随 artifact 保留
  events/prefill-*.jsonl      # prefill 事件
  events/decode-*.jsonl       # decode 事件
  before-*.prom / after-*.prom
  preflight.json / *.log
```

开启 `publish_dashboard=true` 后，`Publish ATOMesh Observability Reports` job 会将报告
发布到 GitHub Pages，并在 CI Summary 中提供 **Open HTML reports** 链接。
地址为 `https://rocm.github.io/ATOM/observability/<GitHub run ID>/<attempt>/`。
仓库需启用 GitHub Pages 并允许该 workflow 在 `github-pages` environment 部署；
workflow 显式上传/部署完整 gh-pages 站点，保留已有看板，不依赖 GITHUB_TOKEN push 自动触发 Pages build。
HTML 发布前压缩，通过英文加载页解压显示，避免高频曲线的单个 HTML 超过 Git 文件大小限制。
原始事件和 VM 数据保留在 artifact；在线页面可以用文件选择器打开下载的事件文件。
发布失败可继续下载原 HTML 离线查看。`ATOMESH_PAGES_BASE_URL` 可覆盖自定义站点的链接基址。

`index.html` 无 CDN 依赖，页面指标和控件均为英文：

- `Quantiles` 可单选或多选 P50/P90/P95/P99；`Overlay` 叠加显示，
  `Separate charts` 分开展示。Gauge 不做分位数处理，也不重复绘制。
- 每张图右上角 `×` 隐藏图表，`Charts` 下拉列表可搜索并恢复。
- 选择 `10 seconds` 后，`Move time window` 滑块移动固定宽度的 10 秒窗口，
  拖动时更新图表；运行不足 10 秒时显示整个运行。`All` 显示全程并禁用滑块。
- 鼠标悬停显示各曲线的最近样本值及其实际采样时间，同组图表联动时间游标。
  总览使用 UTC，单请求图使用从请求开始计算的 elapsed seconds。
- 总览中的 TPOT 是完成请求 choice 的 TPOT 分位数。每个请求 choice 的 TPOT
  为 `(last_output_time - first_output_time) / (output_tokens - 1)`，并非逐 token ITL。

通过文件选择器打开 `events/*.jsonl`，选择 request ID/choice，可以看该请求的
TPOT、TTFT、完成耗时、输入/输出/缓存 token 数，以及流式输出间隔和每批 token 数。
未完成或输出少于两个 token 时 TPOT 显示 `N/A`。MTP 仍只展示真实输出批次间隔。
事件文件采用分段读取，不将整份日志一次读入浏览器内存。事件时间为 UTC wall clock，
持续时间使用 monotonic clock。未完成的请求可能只有 chunk 事件，没有完成统计。

CI 检查 P/D 原有指标和新增 Histogram、GPU 数据、每个目标抓取成功率 ≥95%、
P/D 均有完成样本和 TTFT 曲线、事件文件有完成记录、事件丢弃/写入错误为零。
错误使 job 失败，仍保留报告。MESH HTTP 镜像未上报某种 Histogram 时报告为缺失；
不会用 ATOM 或客户端的数字代替。HTML 的 PASS 代表采集链路检查通过，
不代表已验证吞吐开销 <1%、桶近似误差 <15% 等性能标准。

每次回调会产生事件，开启 MTP 后是一批一个事件；长时间/高吞吐运行会比“每请求一行”
大得多。队列有界且异步写盘；无法跟上时计数并让 CI 验收失败。只看聚合指标时，
可在提交环境设置 `ATOMESH_OBSERVABILITY_EVENTS=0`。

## Grafana 实时查看与回放

启动提供的中心配置（VM 和 Grafana 数据使用持久化 Docker volume）：

```bash
docker compose -f docs/observability/compose.yaml up -d
```

Grafana 默认 `http://localhost:3000`，数据源和 dashboard 已自动配置。
可通过环境变量 `GRAFANA_ADMIN_PASSWORD` 设置登录密码；`OBS_BIND_HOST` 控制监听地址。
实时采集：在 GitHub repository variable `ATOMESH_VM_REMOTE_WRITE_URL` 中配置
`http://<中心机>:8428/api/v1/write`。vmagent 同时写入本地和中心存储，中心不可达时
有有界磁盘队列；本地 artifact 仍可查看。中心访问/持久性需在实际部署环境验证。

离线回放（无需恢复整份 vmdata）：

```bash
gzip -dc /path/to/timeseries.jsonl.gz | curl --fail \
  --data-binary @- http://localhost:8428/api/v1/import
```

Grafana 选择 artifact 对应的绝对起止时间、run ID/case/phase。
已有跨 run benchmark dashboard 的数据 JSON 保持原样；可观测性看板独立运行。

## 开销对照与本地验证

同一 case 分别运行 `observability=off` 和 `on`，保持模型/节点/并发/数据集一致，
对比 AIPerf 吞吐、TTFT/TPOT 和 GPU 利用率。页面展示采集耗时，100ms 档的 scrape
timeout 为 90ms；若失败，先检查事件丢弃、快照 age 和节点负载，不把失败当成正常曲线。

CPU 测试：

```bash
python -m pytest tests/entrypoints/test_request_metrics.py \
  tests/test_scheduling_metrics.py tests/test_atomesh_observability.py
```

提供 `ATOM_OBSERVABILITY_TEST_BIN`（含上述两个 v1.111.0 二进制）可运行实际
vmagent→VictoriaMetrics→导出/HTML 的集成测试；服务输入为模拟数据，**不代表模型性能**。
