# ATOM / MESH 压测可观测性方案

> 目标：把压测过程中引擎的全部运行数据按**统一时间轴**采集、存储、可视化，
> 支持 P50 / P90 / P95 / P99 分位数筛选与跨 run 对比。
>
> 状态：设计稿 · 2026-09-07 · CI 实现见 [observability/README.md](observability/README.md)
>
> 实现校正：已接入独立的 100ms 调度/batch 采集与 1s 全量采集，
> 覆盖流式/非流式和 fanout；P/D 独立指标与事件文件，保留原有 metrics 格式。
> MESH 使用独立 Prometheus 端口；其 HTTP backend 是否上报某个 Histogram
> 以运行时数据为准。原文“三处非流式调用即可覆盖”的假设不适用于 agentic streaming。
> 每次流式输出批次也会记录事件，因此事件体积不能沿用本文“每请求一行”的估算。
> 本地链路测试使用模拟输入；实际 c48 GPU CI 和性能开销对照仍需执行。

---

## 0. TL;DR

```
ATOM :8010/metrics ┐
ATOM :8020/metrics ├─→ vmagent ──remote_write──→ VictoriaMetrics ←── Grafana
MESH :8000/metrics ┤   (1s 抓取,             (存储 + PromQL)      (时间轴 +
amd_smi  :2021     ┘    run_id 打标)                               分位数下拉)
                          │
每请求事件 jsonl ──────────┴──→ /it-share/.../events/  →  离线精确分位数
```

四件事，只有第一件需要改引擎代码：

| # | 事项 | 工作量 | 阻塞关系 |
|---|---|---|---|
| 1 | ATOM 补 `Histogram`（TTFT / TPOT / E2E） | ~60 行 | 出 ATOM 分位数的前提 |
| 2 | 起 VictoriaMetrics + Grafana | 1 个 compose | 无，可立即做 |
| 3 | vmagent 接入 slurm 脚本 | ~15 行 shell | 依赖 2 |
| 4 | 每请求事件流（细粒度精确分位数） | ~30 行 | 独立 |

**MESH 侧已经有 Histogram，不改代码就能出真图** —— 建议先做 2+3 打通链路，再回头做 1。

---

## 1. 目标与非目标

### 目标

- **G1** 压测过程中，引擎的延迟、吞吐、KV cache、排队、抢占、GPU 状态在同一时间轴上可见
- **G2** TTFT / TPOT 支持 P50 / P90 / P95 / P99 切换，粒度 ≤ 1s
- **G3** 每轮压测自动带 `run_id`，可跨 run 对比（对齐现有 `atom-mesh-dashboard` 的 run 概念）
- **G4** 采集链路不显著影响压测结果（开销 < 1% CPU，不占 GPU）
- **G5** SLURM 节点回收后数据仍在

### 非目标

- 不做告警（压测是人盯着的，Alertmanager 后续再说）
- 不做分布式 tracing（单请求跨 P/D 的链路追踪是另一个课题，本方案不覆盖）
- 不替代 `atom-mesh-dashboard`：那是**跨 run 的 Pareto 汇报看板**，本方案是**单 run 内的时间轴调试看板**，两者互补

---

## 2. 现状盘点

| 组件 | 打点现状 | 位置 | 分位数 |
|---|---|---|---|
| **ATOM 引擎** | `/metrics` 端点已有，前缀 `atom:`，40+ 指标 | `atom/entrypoints/openai/metrics.py`（423 行）<br>端点注册 `api_server.py:2353` | ❌ **全是 Gauge / Counter，无 Histogram** |
| **MESH router** | `metrics-exporter-prometheus` 已内置 | `mesh/src/observability/metrics.rs` | ✅ `mesh_router_ttft_seconds` / `tpot_seconds` / `request_duration_seconds` 均为 Histogram，20 档 bucket（`metrics.rs:322-340`，可通过 `duration_buckets` 覆盖，`metrics.rs:135`） |
| **GPU** | 无 | — | — |
| **压测客户端** | `bench_serving.py` 输出汇总 JSON | slurm 脚本内 `scripts/benchmark.sh` | ✅ 精确（有全量样本），但只有每轮一个汇总值 |

### 关键发现

**TTFT / TPOT 在 ATOM 里已经逐请求算好了**，只是没进 metric：

```
atom/entrypoints/openai/api_server.py:955    ttft = (first_token_at - started_at) ...
atom/entrypoints/openai/api_server.py:956    tpot = (last_token_at - first_token_at) / (num_tokens_output - 1)
```

同样的收尾逻辑出现在 **三处**（三条生成路径）：

| 行号 | 函数 | 场景 |
|---|---|---|
| `api_server.py:955` | `generate_async` | 普通流式 |
| `api_server.py:1056` | `generate_async_multimodal` | 多模态 |
| `api_server.py:1183` | fanout 路径 | `n > 1` 并行采样 |

三处都把 `ttft` / `tpot` 塞进了返回 dict。**改动 = 在这三处各加一行 `observe()`**，不需要新建计时逻辑。

---

## 3. 总体架构

### 四层职责

| 层 | 组件 | 职责 | 为什么不能省 |
|---|---|---|---|
| **打点** | ATOM / MESH 进程内 | 把观测值写进内存计数器 | 分布信息只有在这一层才完整 |
| **采集** | vmagent（独立进程） | 定期抓 `/metrics`，打标签，推送 | 解耦；引擎崩了已抓的数据还在 |
| **存储** | VictoriaMetrics | 时序存储 + PromQL 查询 | `/metrics` 只有"此刻"，没有历史 |
| **展示** | Grafana | 画图、分位数下拉、跨 run 对比 | — |

### 数据流

```
              ┌─ 进程内 ─┐   ┌── 节点内 ──┐   ┌──── 中心 ────┐
每个请求完成
    │
    ├─ Histogram.observe(ttft) ──→ /metrics ──→ vmagent ──→ VictoriaMetrics ──→ Grafana
    │      (纳秒级，分桶累计)         (文本)      (1s 抓取)      (压缩存储)        (PromQL)
    │
    └─ events.write(json line) ──→ events/<run_id>.jsonl ──→ pandas / DuckDB
           (每请求一行，不聚合)        (共享盘)                 (任意粒度精确分位数)
```

**两条线的分工**（重要，不要合并）：

| | metric 线 | event 线 |
|---|---|---|
| 粒度 | 1s（受 scrape 间隔限制） | 任意（每请求一条） |
| 分位数 | 近似（桶内插值） | 精确 |
| 实时性 | 秒级可见 | 压测后离线 |
| 存储 | ~2 MB / run | ~5 MB / run（10 万请求） |
| 用途 | 盯盘、看拐点、定位根因 | 出报告、算准确的 P99 |

---

## 4. 关键设计决策

### D1 · 分位数必须用 Histogram，不能上报平均值

**约束**：Prometheus 数据模型中，`(metric, labels) @ timestamp` 只能存**一个 float**。
无法把一批观测值的 list 传出去。所以在 `/metrics` 出口上只有两条路：

| 做法 | 保留 | 丢失 |
|---|---|---|
| 求平均 → 1 个数 | 中心趋势 | **整个分布，P50/P90/P95 永远拿不回来** |
| 分桶 → N 个计数 | 分布形状 | 桶内精确位置（可线性插值） |

**为什么这很致命**：一秒内 100 个请求，99 个 50 ms，1 个 5000 ms。

- 平均 = **99.5 ms** → 看板上一切正常
- P99 = **5000 ms** → 实际有用户等了 5 秒

压测要抓的恰恰是这条长尾，而平均值是专门用来抹掉长尾的。

> `Histogram.observe(v)` 的实现就是"定位到桶，该桶 +1"，开销在纳秒量级，
> 与"遍历 list 计数"是同一件事，只是按值域分了 N 个桶且**累计不清零**。

### D2 · 严禁 clear-on-scrape

**反模式**：在 `collect()` 里读完 list 就清空。

ATOM 当前用的是自定义 Collector（`_AtomMetricsCollector.collect()`），
它在**每次 scrape 时被调用**。若在其中清空状态：

- 手动 `curl /metrics` 调试一次 → 这段数据永久丢失
- vmagent 超时重试 → 窗口错乱、重复计数
- 起第二个采集器 → 两边各拿一半，都不对
- 采集端宕机 30 秒 → 这 30 秒永久丢失

**正确语义**：指标**单调累计**，差分交给查询时的 `rate()` 做。
这样漏几次 scrape 也只是精度下降，不丢数据。

> 附带结论：**"1 秒内的平均" 这个窗口不由引擎决定，由 `scrape_interval` 决定**。
> 采集端改成 15s，你的"1 秒平均"就变成了"15 秒平均"，而引擎毫不知情。

### D3 · push（vmagent）而非 pull（Prometheus 直采）

SLURM 节点是临时分配的：`SLURM_JOB_ID` 每轮变，IP 从 `scontrol show hostnames` 现取
（见 `weekly_mesh_benchmark/*.sh:61,96-98`）。

中心 Prometheus **无法预知下一个 job 落在哪个节点的哪个端口**，pull 模型天然不适用。
vmagent 由作业自己拉起、自己知道 target、主动推送，是唯一干净的做法。

附带收益：`external_labels.run_id = $SLURM_JOB_ID` 让**每轮压测自动带标签**。

### D4 · VictoriaMetrics 而非 Prometheus

| | Prometheus | VictoriaMetrics |
|---|---|---|
| 形态 | 采集+存储+查询三合一 | 拆开：`vmagent` 采集 / `victoria-metrics` 存储 |
| 1s 高频采集 | 内存占用明显上升 | 压缩率与内存表现更好 |
| 推送接入 | 需 agent mode 或 Pushgateway | 原生 `remote_write` |
| PromQL | ✅ | ✅ 完全兼容（MetricsQL 是超集） |

若未来环境固定为几台常驻机器，用 Prometheus 也完全可以，少一个组件。**协议兼容，可随时切换。**

### D5 · 高基数 label 绝不进 Histogram

一个 Histogram 的 series 数 = `(桶数 + 2) × label 组合数`。

- ✅ 允许：`model`（1~3）、`role`（prefill / decode）
- ❌ 禁止：`request_id`、`prompt` 摘要、`user`、精确的 `isl` / `osl` 数值
- ⚠️ 谨慎：`dp_rank`（8~16）—— **放 Gauge 可以，放 Histogram 会把 series 数乘 8~16 倍**

高基数 label 的正确去处是 **event 线**（每请求一行 jsonl），那里想放多少维度都行。

---

## 5. 指标规范

### 5.1 命名

沿用现有前缀：ATOM 用 `atom:`，MESH 用 `mesh_`。
> 注：`:` 在 Prometheus 惯例中保留给 recording rule，ATOM 现有指标已这样用，
> **保持一致比纠正惯例更重要**，不改。

### 5.2 待补的 Histogram（ATOM）

| 指标 | 类型 | 单位 | labels |
|---|---|---|---|
| `atom:ttft_seconds` | Histogram | 秒 | `model`, `role` |
| `atom:tpot_seconds` | Histogram | 秒 | `model`, `role` |
| `atom:e2e_latency_seconds` | Histogram | 秒 | `model`, `role` |
| `atom:output_tokens_per_request` | Histogram | 个 | `model` |

### 5.3 Bucket 边界

对数间隔，覆盖压测实际量程两端各留一档余量：

```python
TTFT_BUCKETS = (0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.5, 0.8,
                1.25, 2.0, 3.2, 5.0, 8.0, 15.0, 30.0, float("inf"))   # 16 档

TPOT_BUCKETS = (0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035,
                0.05, 0.075, 0.11, 0.16, 0.25, 0.5, float("inf"))     # 14 档

E2E_BUCKETS  = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0,
                30.0, 60.0, 120.0, float("inf"))                       # 12 档
```

**桶边界自检**（压测跑通后必做一次，否则分位数会失真）：

```promql
# 1) P95 不能落在首桶或 +Inf 桶里
histogram_quantile(0.95, sum by (le) (rate(atom:ttft_seconds_bucket[1m])))

# 2) 单桶占比不应超过 ~40%，否则该区间需要加密
sum by (le) (increase(atom:ttft_seconds_bucket[10m]))
```

若 P95 贴着某个桶边界不动，说明边界太粗，往那一段加档。
MESH 侧同理，改 `duration_buckets` 配置项（`mesh/src/observability/metrics.rs:135`），
不必改代码。

### 5.4 已有指标（无需改动，直接用）

- ATOM Gauge：`atom:requests_running` / `requests_waiting` / `kv_cache_usage_ratio` /
  `preemptions` / `prefix_cache_hit_ratio` / `dp_inflight_requests` …
- MESH Histogram：`mesh_router_ttft_seconds` / `mesh_router_tpot_seconds` /
  `mesh_router_request_duration_seconds` / `mesh_router_stage_duration_seconds`
- MESH 其他：`mesh_worker_health` / `mesh_worker_cb_state` / `mesh_router_tokens_total` …

---

## 6. 分阶段实施

### Phase 0 — 链路打通（不改任何引擎代码）

**目标**：用 MESH 现成的 Histogram 出第一张真图，验证整条链路。

1. 在一台常驻机起 VictoriaMetrics + Grafana（§7.1）
2. 手动跑一次 vmagent 指向已有的 MESH 实例
3. Grafana 建一个 panel：`histogram_quantile(0.95, sum by (le) (rate(mesh_router_ttft_seconds_bucket[30s])))`

**产出**：MESH 侧 TTFT 分位数真实曲线。
**验收**：曲线有数据、`run_id` 标签正确、时间轴对得上压测起止时间。

### Phase 1 — 接入 slurm 作业

1. 把 §7.2 的片段插进 `weekly_mesh_benchmark/*.sh`（引擎起来之后、压测开始之前）
2. `external_labels` 打上 `run_id` / `topo` / `model`
3. `trap` 保证作业退出时 vmagent 被清理

**验收**：跑一轮 slurm 作业，Grafana 里能按 `run_id` 下拉切换。

### Phase 2 — ATOM 补 Histogram

按 §8 改三处 + metrics.py。

**验收**：`atom:ttft_seconds_bucket` 出现在 `/metrics`；
Grafana 上 ATOM 与 MESH 两侧的 P95 曲线形状一致（MESH 值应略高，含路由开销）。

### Phase 3 — 每请求事件流

在同一个回调里多写一行 jsonl（§8.3），落到 `${LOG_ROOT}/events/`。

**验收**：`pandas.read_json(lines=True)` 能加载；
`df.resample("100ms")["ttft"].quantile(0.95)` 与 Grafana 的 P95 曲线趋势吻合
（数值不会完全相等 —— 见 §11 R4）。

### Phase 4 — 与现有看板打通

压测收尾时用 `query_range` 把时序压成聚合点，写进
`atom-mesh-dashboard/data/<run_id>.json`（§9.4），两个看板共用一份采集。

---

## 7. 配置清单

### 7.1 中心侧 · docker-compose.yml

```yaml
services:
  victoriametrics:
    image: victoriametrics/victoria-metrics:v1.111.0
    ports: ["8428:8428"]
    command:
      - "-storageDataPath=/vmdata"
      - "-retentionPeriod=6"          # 保留 6 个月
      - "-search.maxQueryDuration=60s"
    volumes: ["/it-share/yajizhan/obs/vmdata:/vmdata"]
    restart: unless-stopped

  grafana:
    image: grafana/grafana:11.3.0
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "change-me"
      GF_USERS_DEFAULT_THEME: "dark"
    volumes: ["/it-share/yajizhan/obs/grafana:/var/lib/grafana"]
    restart: unless-stopped
```

数据源：Grafana → Add data source → **Prometheus** 类型 → URL `http://victoriametrics:8428`

### 7.2 作业侧 · 插进 slurm 脚本

变量名沿用 `weekly_mesh_benchmark/*.sh` 里已有的定义。

```bash
# ---- 可观测性采集（引擎就绪后、压测开始前）----
VM_REMOTE="${VM_REMOTE:-http://<中心机>:8428/api/v1/write}"

cat > "${LOG_ROOT}/vmagent.yml" <<EOF
global:
  scrape_interval: 1s
  scrape_timeout: 900ms
  external_labels:
    run_id:  "${SLURM_JOB_ID}"
    model:   "dsv4"
    topo:    "1p8_1d8_tp"
    cluster: "mi355x"
scrape_configs:
  - job_name: atom
    static_configs:
      - targets: ["${PREFILL_IP}:${PREFILL_PORT}"]
        labels: {role: prefill, node: "${PREFILL_NODE}"}
      - targets: ["${DECODE_IP}:${DECODE_PORT}"]
        labels: {role: decode,  node: "${DECODE_NODE}"}
  - job_name: mesh
    static_configs:
      - targets: ["${PREFILL_IP}:${ROUTER_PORT}"]
        labels: {role: router}
  - job_name: gpu
    static_configs:
      - targets: ["${PREFILL_IP}:2021", "${DECODE_IP}:2021"]
EOF

vmagent \
  -promscrape.config="${LOG_ROOT}/vmagent.yml" \
  -remoteWrite.url="${VM_REMOTE}" \
  -remoteWrite.tmpDataPath="${LOG_ROOT}/vmagent-buf" \
  -remoteWrite.maxDiskUsagePerURL=2GB \
  > "${LOG_ROOT}/vmagent.log" 2>&1 &
VMAGENT_PID=$!
trap 'kill $VMAGENT_PID 2>/dev/null' EXIT

echo "[obs] vmagent pid=${VMAGENT_PID} run_id=${SLURM_JOB_ID} → ${VM_REMOTE}"
```

**离线兜底**（中心机不可达时）：把 `-remoteWrite.url` 指向本机临时起的
`victoria-metrics -storageDataPath=${LOG_ROOT}/vmdata`，数据落共享盘，事后再查。

### 7.3 GPU exporter

AMD 官方 `ROCm/amd_smi_exporter`，暴露 `amd_gpu_power` / `temp` / `busy_percent` /
`vram_used` 等。比自行轮询 `amd-smi` 稳定，且不占 GPU。
在容器里与引擎同节点起，监听 `:2021`。

---

## 8. 代码改动点

### 8.1 `atom/entrypoints/openai/metrics.py` — 新增 Histogram

现有文件用的是自定义 Collector。Histogram 直接用 `prometheus_client.Histogram`
注册到同一个 registry 即可，两者可共存。

```python
from prometheus_client import Histogram

TTFT_BUCKETS = (0.02, 0.04, 0.07, 0.12, 0.2, 0.32, 0.5, 0.8,
                1.25, 2.0, 3.2, 5.0, 8.0, 15.0, 30.0, float("inf"))
TPOT_BUCKETS = (0.003, 0.005, 0.008, 0.012, 0.018, 0.025, 0.035,
                0.05, 0.075, 0.11, 0.16, 0.25, 0.5, float("inf"))
E2E_BUCKETS  = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0,
                30.0, 60.0, 120.0, float("inf"))

_TTFT = Histogram("atom:ttft_seconds", "Time to first token per request.",
                  ("model", "role"), buckets=TTFT_BUCKETS, registry=REGISTRY)
_TPOT = Histogram("atom:tpot_seconds", "Mean time per output token per request.",
                  ("model", "role"), buckets=TPOT_BUCKETS, registry=REGISTRY)
_E2E  = Histogram("atom:e2e_latency_seconds", "End-to-end request latency.",
                  ("model", "role"), buckets=E2E_BUCKETS, registry=REGISTRY)


def record_request(*, ttft: float, tpot: float, latency: float,
                   model: str, role: str) -> None:
    """三条生成路径共用的收尾埋点。零分配、纳秒级，可安全放在热路径。"""
    if ttft > 0.0:
        _TTFT.labels(model, role).observe(ttft)
    if tpot > 0.0:
        _TPOT.labels(model, role).observe(tpot)
    if latency > 0.0:
        _E2E.labels(model, role).observe(latency)
```

`role` 在进程启动时求一次值，**不要每请求现算**。ATOM 里已有现成的解析函数
（`api_server.py:2390` `_resolve_kv_transfer_role`）和模型名全局（`api_server.py:2570` `model_name`）：

```python
# api_server.py，引擎初始化完成之后求值一次
_kv_role, _ = _resolve_kv_transfer_role(engine.config.kv_transfer_config or {})
SERVE_ROLE = {"kv_producer": "prefill", "kv_consumer": "decode"}.get(_kv_role, "hybrid")
```

> 非 PD 分离部署时 `kv_role` 为 `None`，落到 `hybrid`，不要默认成 `decode` —— 否则
> 单机部署的数据会和 PD 部署的 decode 侧混在同一条曲线里。

### 8.2 `atom/entrypoints/openai/api_server.py` — 三处调用

在每处 `ttft` / `tpot` 计算完、构造 `response` dict **之前**插入一行：

| 插入位置 | 所在函数 |
|---|---|
| `:963` 附近（`ttft`/`tpot` 算完后） | `generate_async` |
| `:1064` 附近 | `generate_async_multimodal` |
| `:1195` 附近（在 `for i in ...` 循环内，逐路记录） | fanout / `n > 1` |

```python
    metrics.record_request(ttft=ttft, tpot=tpot, latency=latency,
                           model=model_name, role=SERVE_ROLE)
```

**注意**：三处都要加。只加一处会导致多模态和 `n>1` 的请求在分位数里凭空消失，
而这种"少了一部分样本"的错误在图上看不出来。

### 8.3 事件流（Phase 3，同一位置多加一行）

```python
    _events.write(ts=finished_at, ttft=ttft, tpot=tpot, latency=latency,
                  isl=num_tokens_input, osl=num_tokens_output,
                  dp_rank=data_parallel_rank, role=SERVE_ROLE,
                  cached=num_cached_tokens_seen, finish=finish_reason)
```

`_events` 用 **带缓冲的后台写线程**（队列 + 批量 flush），
绝不在请求路径上做同步磁盘 IO。

---

## 9. PromQL 速查

### 9.1 分位数（Grafana 变量版）

建一个 Custom 变量 `quantile = 0.5,0.9,0.95,0.99`，面板顶部即得下拉框：

```promql
histogram_quantile($quantile,
  sum by (le, role) (rate(atom:ttft_seconds_bucket{run_id="$run_id"}[30s])))
```

> `rate()` 窗口取 **≥ 4 × scrape_interval**。1s 采集下 `[5s]` 是下限，
> `[30s]` 更稳；窗口过小会出现空洞和毛刺。
> `sum by (le)` 是跨实例聚合后再算分位数的**唯一正确写法**。

### 9.2 常用面板

```promql
# 吞吐（输出 token/s）
sum(rate(atom:generation_tokens{run_id="$run_id"}[30s]))

# KV cache 使用率（按 role）
max by (role) (atom:kv_cache_usage_ratio{run_id="$run_id"})

# 抢占速率
sum(rate(atom:preemptions{run_id="$run_id"}[10s]))

# 排队深度
sum(atom:requests_waiting{run_id="$run_id"})

# 前缀缓存命中率
avg(atom:prefix_cache_hit_ratio{run_id="$run_id"})

# MESH 侧端到端（含路由开销），与 ATOM 侧对比可分离出路由耗时
histogram_quantile($quantile,
  sum by (le) (rate(mesh_router_request_duration_seconds_bucket{run_id="$run_id"}[30s])))
```

### 9.3 跨 run 对比

去掉 `run_id` 过滤、改用 `by (run_id)`，配合 Grafana 的 multi-value 变量：

```promql
histogram_quantile(0.95,
  sum by (le, run_id) (rate(atom:ttft_seconds_bucket{run_id=~"$run_id"}[30s])))
```

### 9.4 导出给 `atom-mesh-dashboard`

```python
import requests
r = requests.get("http://<中心机>:8428/api/v1/query_range", params={
    "query": f'histogram_quantile(0.95, sum by (le,role) '
             f'(rate(atom:ttft_seconds_bucket{{run_id="{run_id}"}}[30s])))',
    "start": t0, "end": t1, "step": "1s",
}).json()
```

---

## 10. 容量与开销

| 项 | 估算 | 依据 |
|---|---|---|
| Series 总数 | ~2,500 | ATOM×2 节点 ≈ 900，MESH ≈ 800，GPU ≈ 200，Histogram 新增 ≈ 150 |
| 采样量 / run | ~1.5 M | 2,500 series × 600 点（10 min @ 1s） |
| 存储 / run | **~1–2 MB** | VM 压缩后约 0.5–1.5 B/样本 |
| 存储 / 年 | ~7 GB | 按每天 10 轮压测 |
| vmagent 内存 | < 200 MB | 官方基准，2.5k series 量级 |
| 网络 | < 100 KB/s | remote_write 走 snappy 压缩 |

**结论：存储不是瓶颈，可以放心用 1s 采集。**

真正需要实测的是**引擎侧 `/metrics` 的生成开销** —— ATOM 的 Collector 是
pull 时现算的，1s 一次会把这个成本放大 15 倍：

```bash
# 压测进行中执行，看 p50 耗时
for i in $(seq 20); do
  curl -s -o /dev/null -w "%{time_total}\n" http://localhost:8010/metrics
done | sort -n | awk '{a[NR]=$1} END{print "p50:", a[int(NR/2)]}'
```

> **判据**：若 > 50 ms，把 `scrape_interval` 放宽到 2s，或把 Collector 里的
> 重计算改为后台定时刷新 + 读快照。

---

## 11. 风险与红线

| # | 风险 | 后果 | 对策 |
|---|---|---|---|
| **R1** | label 基数爆炸（`request_id` 之类进了 label） | VM 内存暴涨、查询超时 | §5.4 白名单；上线前 `curl /metrics \| wc -l` 对比预期 |
| **R2** | `collect()` 里做重活 | 1s 采集把开销放大 15×，**污染压测结果** | §10 实测；超标就改后台刷新 |
| **R3** | clear-on-scrape | 静默丢数据，且**图上看不出来** | §D2，代码评审红线 |
| **R4** | bucket 边界选错 | 分位数失真最高可达 30%+ | §5.3 自检；**以客户端 `bench_serving` 的精确分位数为准**，引擎侧只用于看趋势 |
| **R5** | 多节点时钟不同步 | P/D 两侧曲线错位，因果关系读错 | 部署前确认 chrony/ntp；`clockdiff` 抽查 |
| **R6** | 只改了三处埋点中的一处 | 部分请求从分位数里消失，**无告警、无异常** | §8.2 三处清单；补一个断言测试 |
| **R7** | vmagent 缓冲盘写满 | 采集静默停止 | `-remoteWrite.maxDiskUsagePerURL=2GB` + 作业结束检查 `vmagent.log` |

> **R4 补充**：引擎侧 Histogram 的分位数与客户端报的**必然对不上**（一个是桶近似，
> 一个是全量样本精确值）。这不是 bug。引擎侧的价值在于**能看到时间轴上何时开始劣化**。

---

## 12. 验收标准

- [ ] **A1** 压测进行中，Grafana 时间轴延迟 < 5s 可见
- [ ] **A2** TTFT / TPOT 的 P50 / P90 / P95 / P99 可通过下拉框切换，无需改 panel
- [ ] **A3** `run_id` 自动等于 `SLURM_JOB_ID`，可多选对比
- [ ] **A4** KV cache 打满 → 抢占 → TTFT P99 抬升，三个面板拐点在时间轴上对齐
- [ ] **A5** 开采集与不开采集，压测吞吐差异 **< 1%**（跑两轮对照）
- [ ] **A6** 作业结束、节点回收后，数据仍可在 Grafana 查到
- [ ] **A7** 引擎侧 P95 与 `bench_serving` 报告的 P95 偏差 **< 15%**（超出说明桶太粗）
- [ ] **A8** `/metrics` 生成耗时 p50 < 50 ms

---

## 附录 A · 看板形态预览

`grafana_preview/atom_dashboard_demo.html` —— 用仿真数据（并发阶梯 32→256，
600 点 @ 1s）做的静态预览，单文件无依赖，`file://` 直接打开。

包含：分位数切换、对数/线性纵轴、跨面板同步十字线、事件时间线、
分位数起始 vs 结束对比表。**用于确认看板形态，数据非真实测量。**

## 附录 B · 术语

| 术语 | 说明 |
|---|---|
| **TTFT** | Time To First Token，请求发出到收到第一个 token |
| **TPOT** | Time Per Output Token，首 token 之后的平均出字间隔 |
| **Histogram** | Prometheus 指标类型，按值域分桶累计计数，查询时插值出分位数 |
| **scrape** | 采集端拉取 `/metrics` 的动作 |
| **remote_write** | Prometheus 生态的推送协议 |
| **external_labels** | 采集端统一附加到所有指标上的标签 |
| **series** | 一条时间序列 = 指标名 + 一组唯一的 label 值 |
