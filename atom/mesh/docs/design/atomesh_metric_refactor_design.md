# Atomesh Metric Refactor Design

## 1. 背景与范围

Atomesh 是 Rust inference gateway，metrics 贯穿请求入口、router、worker、retry、circuit breaker、health check 和 worker engine 指标聚合。后续 smart router、worker pool 统一和动态 worker 注册都会继续扩大 metrics 的使用范围，因此本次重构不能只是移动 `metrics.rs`，而要先明确 metrics 的模块边界、调用方式和兼容性契约。

本文描述的是从当前实现迁移到目标结构的方案，不是已经落地的代码结构。当前仍是 `src/observability/metrics.rs` 单文件承载 `PrometheusConfig`、`init_metrics`、`start_prometheus`、`Metrics` facade、label/interner 和全部 `mesh_*` 打点 API；`server.rs` 直接注册 health 和 `/engine_metrics` 路由；worker engine metrics 聚合仍在 `core/metrics_aggregator.rs` 和 `WorkerManager::get_engine_metrics` 中完成。本文只讨论 `atom/mesh/src/observability` 下 metrics 相关能力的设计重构，参考 Dynamo、vLLM、SGLang 等推理框架的命名习惯，保留 `observability` 作为根观测目录，并将 Prometheus metrics 收敛到 `src/observability/metrics` 子系统。

概念边界：

```text
observability = logging + events + tracing + inflight tracking + metrics
metrics       = Prometheus 指标定义、记录、暴露、worker engine 指标聚合
```

## 重构总览图

本次重构的核心不是简单把 `metrics.rs` 换目录，而是把“指标记录、指标暴露、engine metrics 聚合、健康检查路由”从 `server.rs` 和 `observability/metrics.rs` 中拆出来，形成独立的 metrics 子系统。

### 核心变化

现状：

- `server.rs`、`observability/metrics.rs`、`core/worker_manager.rs`、`core/metrics_aggregator.rs` 共同承担 metrics 相关职责，模块耦合严重。
- `observability/metrics.rs` 过大，指标定义、label 处理、打点 API、Mesh self metrics 暴露初始化混在一起，不便于维护和扩展。
- Mesh self metrics 与 worker engine metrics 的暴露链路混在一起，容易误解 `/engine_metrics` 与 Mesh 自身 `/metrics` 的关系。
- router、middleware、worker、circuit breaker 等业务路径直接依赖 `Metrics::*`，缺少稳定的 metrics 子系统边界。

重构后：

- `server.rs` 只负责组合 route factory，metrics/health 相关 public routes 下沉到 `observability/metrics/routes.rs`。
- `observability/metrics/` 成为独立 metrics 子系统，按配置、schema、record、export、route、engine metrics 聚合拆分职责。
- 业务代码统一通过 `MeshMetrics` facade 打点，降低对内部实现细节的依赖。
- Mesh self metrics 与 worker engine metrics 分成两条清晰链路：前者由 `mesh_metrics.rs` 暴露，后者由 `/engine_metrics` 拉取 worker 指标并聚合。
- `/v1/models`、`/get_model_info`、`/get_server_info` 不归入 metrics factory，后续放入 model/server info route factory。

```text
Before
┌──────────────────────────────────────────────────────────────────┐
│ server.rs                                                        │
│ - start_prometheus                                               │
│ - public_routes = Router::new().route(...)                       │
│ - liveness/readiness/health handlers                             │
│ - engine_metrics handler                                         │
└───────────────┬───────────────────────────┬──────────────────────┘
                │                           │
                ▼                           ▼
┌──────────────────────────────┐   ┌───────────────────────────────┐
│ observability/metrics.rs     │   │ core/worker_manager.rs        │
│ - metric names / HELP        │   │ - get_engine_metrics          │
│ - labels / interner          │   │ - fan out worker /metrics     │
│ - Metrics facade             │   └───────────────┬───────────────┘
│ - mesh metrics exporter      │                   ▼
│ - record_* APIs              │   ┌───────────────────────────────┐
└───────────────┬──────────────┘   │ core/metrics_aggregator.rs    │
                │                  │ - parse Prometheus text       │
                │                  │ - inject worker labels        │
                │                  │ - merge samples               │
                │                  └───────────────────────────────┘
                ▼
┌──────────────────────────────┐
│ routers / middleware / core  │
│ - direct Metrics::* calls    │
└──────────────────────────────┘

After
┌──────────────────────────────────────────────────────────────────┐
│ server.rs                                                        │
│ - build app                                                      │
│ - merge(metrics_factory.get(...))                                │
│ - merge(model_info_routes_factory.get(...))                      │
│ - merge(protected/admin routes)                                  │
└───────────────────────────────┬──────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────┐
│ observability/metrics/                                           │
├──────────────────────────────┬───────────────────────────────────┤
│ routes.rs                    │ recorder.rs                       │
│ - liveness/readiness/health  │ - MeshMetrics facade              │
│ - health_generate            │ - record HTTP/router/worker       │
│ - engine_metrics route       │ - record retry/circuit breaker    │
│ - optional /metrics          │                                   │
├──────────────────────────────┼───────────────────────────────────┤
│ mesh_metrics.rs              │ engine_metrics.rs                 │
│ - expose Mesh mesh_* metrics │ - fan out worker /metrics         │
│ - duration buckets           │ - parse and merge Prom text       │
│ - scrape handler             │ - partial failure semantics       │
├──────────────────────────────┼───────────────────────────────────┤
│ schema.rs                    │ config.rs                         │
│ - metric names / HELP        │ - Prometheus config               │
│ - labels / path normalize    │ - route config                    │
│ - interner / cardinality     │ - engine metrics config           │
└──────────────────────────────┴───────────────────────────────────┘
```

### 代码结构前后对比

```text
当前结构                                      目标结构
----------------------------------------      ----------------------------------------
src/                                          src/
├── server.rs                         853L    ├── server.rs                    ≈780-820L
│   ├── liveness/readiness/health handlers    │   ├── Router::new()
│   ├── engine_metrics handler                │   │   ├── .merge(metrics_factory.get(...))
│   ├── public_routes = Router::new()...      │   │   ├── .merge(model_info_routes_factory.get(...))
│   └── ...                                   │   │   ├── .merge(protected_routes_factory.get(...))
├── observability/                            │   │   └── .merge(admin_routes_factory.get(...))
│   ├── mod.rs                          8L    │   └── ...
│   ├── metrics.rs                   1411L    ├── observability/
│   ├── logging.rs                    159L    │   ├── mod.rs                   ≈10-15L
│   ├── events.rs                     118L    │   ├── logging.rs                  159L
│   ├── inflight_tracker.rs           196L    │   ├── events.rs                   118L
│   └── gauge_histogram.rs            584L    │   ├── inflight_tracker.rs         196L
├── core/                                     │   ├── gauge_histogram.rs          584L
│   ├── metrics_aggregator.rs          92L    │   └── metrics/
│   ├── worker_manager.rs             395L    │       ├── mod.rs                ≈30-50L
│   └── ...                                   │       ├── config.rs            ≈80-120L
├── routers/                                  │       ├── schema.rs           ≈250-350L
│   └── ...                                   │       ├── recorder.rs         ≈550-700L
└── ...                                       │       ├── mesh_metrics.rs     ≈120-180L
                                              │       ├── routes.rs           ≈160-240L
                                              │       └── engine_metrics.rs   ≈220-320L
                                              ├── core/
                                              │   └── ...
                                              ├── routers/
                                              │   └── ...
                                              └── ...
```

说明：

- `server.rs`：从“逐个注册 public endpoint”改为“组合 route factory”。它仍然负责构建 Axum app，但不再直接维护 metrics/health/model info 的细粒度路由清单。
- `observability/metrics.rs`：从单个大文件拆成 `observability/metrics/` 子模块。指标命名、记录、暴露、路由和 engine metrics 聚合不再混在一个文件里。
- `core/metrics_aggregator.rs`：从 `core` 中移出，合并到 `observability/metrics/engine_metrics.rs`。原因是它处理的是 worker engine metrics 聚合，属于 metrics 子系统，而不是 worker core runtime。
- `observability/`：仍然是唯一根观测目录，完整保留 `mod.rs`、`logging.rs`、`events.rs`、`inflight_tracker.rs`、`gauge_histogram.rs`，新增 `metrics/` 子目录。
- 行数说明：当前结构中的行数来自现有源码；目标结构中的 `≈` 是按当前 `metrics.rs`、`metrics_aggregator.rs` 和 `worker_manager.rs` 中相关逻辑拆分后的设计估算，实际行数会随实现细节和注释量变化。

### Public Routes 前后对比

当前 `server.rs` 中直接组装：

```rust
let public_routes = Router::new()
    .route("/liveness", get(liveness))
    .route("/readiness", get(readiness))
    .route("/health", get(health))
    .route("/health_generate", get(health_generate))
    .route("/engine_metrics", get(engine_metrics))
    .route("/v1/models", get(v1_models))
    .route("/get_model_info", get(get_model_info))
    .route("/get_server_info", get(get_server_info));
```

目标改为 route factory 组合：

```rust
let public_routes = Router::new()
    .merge(metrics_factory.get(app_state.clone()))
    .merge(model_info_routes_factory.get(app_state.clone()));
```

其中 metrics factory 只负责观测类 endpoint：

```rust
impl MetricsRouteFactory {
    pub fn get(&self, state: Arc<AppState>) -> Router {
        Router::new()
            .route("/liveness", get(liveness))
            .route("/readiness", get(readiness))
            .route("/health", get(health))
            .route("/health_generate", get(health_generate))
            .route("/engine_metrics", get(engine_metrics))
            .merge(self_metrics_route_if_enabled(&self.config))
            .with_state(state)
    }
}
```

model/server info routes 单独管理：

```rust
impl ModelInfoRouteFactory {
    pub fn get(&self, state: Arc<AppState>) -> Router {
        Router::new()
            .route("/v1/models", get(v1_models))
            .route("/get_model_info", get(get_model_info))
            .route("/get_server_info", get(get_server_info))
            .with_state(state)
    }
}
```

### `observability/metrics/` 文件功能详解

`mod.rs` 是 metrics 子系统的模块入口。它只负责声明子模块和对外 re-export，避免业务代码感知内部实现细节。

```rust
pub mod config;
pub mod schema;
pub mod recorder;
pub mod mesh_metrics;
pub mod routes;
pub mod engine_metrics;

pub use config::{MetricsConfig, PrometheusConfig};
pub use recorder::MeshMetrics;
pub use routes::MetricsRouteFactory;
```

`config.rs` 负责 metrics 相关配置模型，不做打点、不启动 server。它把 Mesh self metrics 暴露配置、主业务端口是否暴露 `/metrics`、`/engine_metrics` 是否启用、duration buckets 等配置集中管理。

```rust
pub struct MetricsConfig {
    pub prometheus: PrometheusConfig,
    pub routes: MetricsRouteConfig,
    pub engine: EngineMetricsConfig,
}

pub struct MetricsRouteConfig {
    pub expose_self_metrics_on_main_port: bool,
    pub self_metrics_path: String,
    pub expose_engine_metrics: bool,
    pub engine_metrics_path: String,
}
```

`schema.rs` 只定义 metrics 的统一规范，不负责真正打点。它集中管理指标名、HELP 文案、label 名称、HTTP method/status 转换、endpoint 规范化、动态 label 缓存和 cardinality 规则。`recorder.rs` 在调用 `counter!`、`histogram!`、`gauge!` 时引用这些规范，避免指标名或 label 在不同业务模块里写出多个变体。

第一阶段必须保持现有 `middleware.rs::normalize_path_for_metrics` 的 label 语义，不把未知路径统一改成 `unknown`，也不把已有 `{id}` 占位符改成其他名字。后续如需收敛为更严格的静态 endpoint 枚举，应作为独立 breaking change 迁移，并同步 dashboard/alert。

```rust
pub mod names {
    pub const HTTP_REQUESTS_TOTAL: &str = "mesh_http_requests_total";
    pub const ROUTER_REQUEST_DURATION_SECONDS: &str =
        "mesh_router_request_duration_seconds";
}

pub fn normalize_path(path: &str) -> Cow<'_, str> {
    // Preserve the current normalize_path_for_metrics behavior:
    // known static paths pass through, dynamic segments become {id},
    // and unknown-but-stable paths are not collapsed to "unknown".
    normalize_dynamic_segments(path)
}
```

`recorder.rs` 是统一的指标记录入口。HTTP middleware、router、worker、retry、circuit breaker 等业务路径只调用 `MeshMetrics`，并传入“发生了什么”的语义化参数；具体使用哪个指标名、哪些 label、调用 `counter!` / `histogram!` / `gauge!`，都由 `recorder.rs` 结合 `schema.rs` 内部完成。这样业务代码不需要直接拼指标名和 label，也不会到处散落 metrics 宏。

```rust
pub struct MeshMetrics;

impl MeshMetrics {
    pub fn record_http_request(method: &str, path: &str) {
        counter!(
            schema::names::HTTP_REQUESTS_TOTAL,
            "method" => schema::method(method),
            "path" => schema::normalize_path(path),
        )
        .increment(1);
    }

    pub fn record_router_duration(ctx: RouterMetricContext, duration: Duration) {
        histogram!(
            schema::names::ROUTER_REQUEST_DURATION_SECONDS,
            "router_type" => ctx.router_type,
            "model" => schema::intern_label(&ctx.model),
            "endpoint" => ctx.endpoint.as_str(),
        )
        .record(duration.as_secs_f64());
    }
}
```

`mesh_metrics.rs` 负责把 Mesh 自身产生的 `mesh_*` 指标暴露给 Prometheus。它做的是 Mesh self metrics 的暴露初始化，例如声明指标 HELP、配置监听地址、设置 histogram bucket、安装 Prometheus exporter。它不参与业务路径打点，业务路径仍然通过 `recorder.rs`；它也不处理 worker 的 `/metrics` 聚合，后者属于 `engine_metrics.rs`。

迁移时必须保留当前 exporter 行为：默认独立监听 `0.0.0.0:29000`，duration histogram buckets 与现有默认值一致，保留 `upkeep_timeout(Duration::from_secs(5 * 60))`，且只安装一次全局 recorder。第一阶段不默认在主业务端口暴露 `/metrics`；若新增主端口 `/metrics`，应通过 exporter handle 显式实现，并用配置关闭，避免改变现有部署的 scrape 拓扑。

```rust
pub fn start_prometheus(config: &PrometheusConfig) -> anyhow::Result<()> {
    describe_all_metrics();

    PrometheusBuilder::new()
        .with_http_listener(config.socket_addr()?)
        .set_buckets_for_metric(Matcher::Suffix("duration_seconds".into()), config.buckets())
        .install()?;

    Ok(())
}
```

`routes.rs` 负责把观测相关的 public endpoints 组装成一个 Axum `Router`，供 `server.rs` 通过 `.merge(metrics_factory.get(...))` 接入。它承接原来散落在 `server.rs` 中的 `/liveness`、`/readiness`、`/health`、`/health_generate`、`/engine_metrics`，并根据配置决定是否在主业务端口额外挂载 `/metrics`。它只负责路由组织和 handler 入口，不负责具体指标记录逻辑。

需要注意依赖方向：这些 handler 当前依赖 `server.rs::AppState`。迁移时可以短期让 `routes.rs` 接受 `Arc<AppState>`，但不应把 metrics 子系统与 `server.rs` 深度耦合。更稳妥的后续形态是抽出一个最小 state trait 或 route state struct，只暴露 `router`、`worker_registry`、`client`、`router_config` 等 handler 真正需要的字段。

```rust
impl MetricsRouteFactory {
    pub fn get(&self, state: Arc<AppState>) -> Router {
        let routes = Router::new()
            .route("/liveness", get(liveness))
            .route("/readiness", get(readiness))
            .route("/health", get(health))
            .route("/health_generate", get(health_generate))
            .route("/engine_metrics", get(engine_metrics));

        if self.config.expose_self_metrics_on_main_port {
            routes.route("/metrics", get(self_metrics))
        } else {
            routes
        }
        .with_state(state)
    }
}
```

`engine_metrics.rs` 负责 `/engine_metrics` 背后的 worker 指标聚合流程。它从 worker registry 获取 worker 列表，并发请求每个 worker 的 `/metrics`，再解析 Prometheus 文本、注入 worker label、合并结果，并处理部分 worker 失败、全部失败、解析失败、label 不一致等情况。它只处理下游 engine 原始指标，不记录 Mesh 自身 `mesh_*` 指标。

第一阶段应保持现有 HTTP 语义：无 worker 返回 500 `No available workers`；全部 worker 请求失败或无成功文本返回 500 `All backend requests failed`；部分 worker 请求失败时跳过失败 worker，成功聚合剩余 metrics 并返回 200；单个 worker Prometheus 文本 parse 失败时 warn 并跳过；同名 metric family label names 不一致时聚合失败并返回 500。后续如果要把 parse failure 或 label mismatch 改成 partial failure，需要单独设计，因为这会改变 `/engine_metrics` 的可观测语义。

```rust
pub async fn collect_engine_metrics(
    registry: &WorkerRegistry,
    client: &reqwest::Client,
) -> EngineMetricsResult {
    let workers = registry.get_all();
    let scrape_results = scrape_workers_metrics(workers, client).await;
    let snapshot = aggregate_prometheus_texts(scrape_results)?;

    if snapshot.all_failed() {
        return EngineMetricsResult::Err("all backend metrics requests failed".into());
    }

    EngineMetricsResult::Ok(snapshot.text)
}
```

### Metric 数据流

Mesh self metrics 链路：

```text
client request
    │
    ▼
middleware ─────┐
    │           │ record
    ▼           ▼
router ───► MeshMetrics facade ───► mesh_metrics.rs ───► Prometheus
    │
    ▼
worker/backend
```

Engine metrics 链路：

```text
GET /engine_metrics
    │
    ▼
observability/metrics/routes.rs
    │
    ▼
observability/metrics/engine_metrics.rs
    │ fan out
    ▼
worker-1 /metrics ┐
worker-2 /metrics ├──► merge and label injection ───► merged Prometheus text
worker-n /metrics ┘
```

两条链路的区别：

```text
Mesh self metrics:
  业务请求过程中打点，Prometheus scrape mesh_metrics.rs 暴露的 mesh_*。

Engine metrics:
  访问 /engine_metrics 时临时请求 workers /metrics，再聚合 worker 原始指标。
```

### Metrics 兼容性契约

本次重构的第一阶段是结构重构，不是指标语义重构。除非在独立 PR 中明确声明 breaking change，否则必须保持以下契约：

- 指标名不变：现有 `mesh_*` metric name 不改名、不拆分、不合并。
- label key 不变：`method`、`path`、`status_code`、`error_code`、`router_type`、`backend_type`、`connection_mode`、`model`、`endpoint`、`streaming`、`worker_type`、`worker`、`worker_addr` 等既有 label key 不在本轮改名。
- label value 不漂移：HTTP `path` 第一阶段继续沿用当前动态段归一规则，动态 ID 使用 `{id}`，未知稳定路径保留原路径；router `endpoint` 继续沿用 `routers/shared/metrics_utils.rs` 的映射语义。
- worker 维度不直接切换：Mesh self metrics 中现有 worker URL label 第一阶段保持不变；engine metrics 聚合继续注入 `worker_addr`。收敛到稳定 `worker_id` 需要单独迁移计划，必要时双写新旧 label，并给 PromQL/dashboard/alert 留出迁移窗口。
- exporter 拓扑不变：默认仍由独立 Prometheus exporter 端口暴露 Mesh self metrics，不默认把 `/metrics` 挂到主业务端口。
- failure semantics 不变：`/engine_metrics` 的无 worker、全失败、部分失败、parse failure、label mismatch 行为必须有测试锁定。

如果后续确实要做指标语义收敛，建议拆成单独 PR，并在 PR 描述中列出受影响的 metric、label、PromQL 和 dashboard。

### 指标库存与清理策略

迁移到 `schema.rs` 时需要把所有 metric 纳入库存表，至少标记为 `active`、`planned`、`deprecated` 或 `missing_describe`。当前代码中已观察到以下需要特别处理的项：


| 类别                                            | 当前状态                                   | 处理建议                                                         |
| --------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| HTTP/router/worker/retry/circuit breaker 核心指标 | 已 describe 且有调用                        | 第一阶段原样迁移，新增契约测试防止 name/label 漂移。                             |
| discovery 指标                                  | 已 describe，但当前无业务调用                    | 标记 `planned` 或 `deprecated`，不要在重构中假装已接线。                     |
| DB 指标                                         | 已 describe，但当前无业务调用                    | 标记 `planned` 或 `deprecated`，等实际 storage path 接线后再转 `active`。 |
| `mesh_manual_policy_cache_entries`            | 已 describe，但当前未看到有效调用                  | 标记 `planned` 或补齐调用；若 manual policy 不再使用则 deprecated。         |
| `mesh_manual_policy_branch_total`             | 有 record API，但缺 describe 且当前未接线        | 若保留 API，补 describe；否则标记 deprecated 并删除死 API。                 |
| `mesh_prefix_hash_policy_branch_total`        | 有调用，但缺 describe                        | 补 describe 和测试，避免 HELP/TYPE 缺失。                              |
| `mesh_worker_routing_keys_active`             | 有调用，但缺 describe                        | 补 describe 和测试，明确 label 中 worker URL 的迁移策略。                  |
| `record_router_stage_duration`                | 已 describe/API，但当前 gRPC pipeline 未充分接线 | 明确是否接线；未接线前标记 `planned`。                                     |
| `set_worker_connections_active`               | 有 API，但当前未接线                           | 标记 `planned` 或删除。                                            |


库存表应成为 `schema.rs` 的迁移输入：每个 metric 都有唯一 name、HELP、TYPE、label list、状态、调用入口和测试覆盖。禁止在 `recorder.rs` 中新增未登记的 metric name。

## 2. 改造范围

1. 新建 `src/observability/metrics/`，拆出 `mod.rs`、`config.rs`、`schema.rs`、`recorder.rs`、`mesh_metrics.rs`、`routes.rs`、`engine_metrics.rs`。第一阶段保持现有 `mesh_*` 指标名、label、HTTP path normalize、worker label 和 Prometheus exporter 行为不变。
2. 将业务路径中的直接 `Metrics::*` 调用迁移到 `recorder.rs` 的 `MeshMetrics`，由调用方传入语义化参数，指标名、label、path normalize 和 interner 统一从 `schema.rs` 获取。
3. 将 Mesh self metrics 暴露逻辑迁入 `mesh_metrics.rs`，包括指标 describe、Prometheus exporter 安装、监听地址、duration buckets 和 `upkeep_timeout`；它只负责暴露 `mesh_*`，不处理 worker `/metrics`。
4. 将 `core/metrics_aggregator.rs` 和 `WorkerManager::get_engine_metrics` 中的 worker `/metrics` 拉取、解析、label 注入和合并逻辑迁入 `engine_metrics.rs`，用测试锁定 partial failure、parse failure、label mismatch 等现有语义。
5. 将 `/liveness`、`/readiness`、`/health`、`/health_generate`、`/engine_metrics` 从 `server.rs` 抽到 `routes.rs`，`server.rs` 改为 `.merge(metrics_factory.get(...))`；`/v1/models`、`/get_model_info`、`/get_server_info` 保持在 model/server info route factory。迁移时先接受 `Arc<AppState>`，后续再收敛成最小 route state。
6. 清理未接线指标：discovery、DB、manual policy 等要么接线，要么标记 planned/deprecated；已调用但缺 describe 的 policy/routing keys 指标必须补齐 HELP/TYPE。Mesh self metrics 的 worker 维度从 URL 收敛到稳定 `worker_id` 不放在第一阶段完成，应单独设计双写和迁移窗口；`worker_addr` 第一阶段继续保留在 engine metrics 聚合结果中。

## 3. 实施计划

按 5 个工作日排布，优先保证“不改指标语义、不影响请求路径”，并为高风险的 exporter、engine metrics 聚合和路由迁移预留充足验证时间。


| 阶段    | 时间  | 主要工作                                                                                                                                  | 交付物                                                                                      |
| ----- | --- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Day 1 | 1 天 | 建立 `observability/metrics/`，迁移 `mod.rs/config.rs/schema.rs`，整理指标库存、HELP、label、path normalize、interner、cardinality 规则。                 | 新目录可编译；`schema.rs` 成为统一规范入口；原指标名、label、path value 不变。                                    |
| Day 2 | 1 天 | 迁移 `recorder.rs`，将主要 `Metrics::*` 调用切到 `MeshMetrics`。优先完成 HTTP middleware、HTTP router、gRPC router、worker/reliability 的调用入口切换。         | 核心业务路径通过 `MeshMetrics` 打点；指标名和 label 语义不变。                                               |
| Day 3 | 1 天 | 迁移 `mesh_metrics.rs`，保持 Prometheus exporter、scrape 端口、duration buckets、`upkeep_timeout`、describe 行为兼容；补齐 Mesh self metrics scrape 验证。 | Mesh self metrics 暴露链路独立；Prometheus scrape 行为不变；不默认新增主端口 `/metrics`。                     |
| Day 4 | 1 天 | 迁移 `engine_metrics.rs` 和 `routes.rs`：迁入 `/engine_metrics` 聚合，抽出 health/metrics routes，`server.rs` 改为 route factory 组合。                | Mesh self metrics 与 engine metrics 分离；`server.rs` 职责收敛；路由和失败语义不变。                        |
| Day 5 | 1 天 | 清理未接线指标状态、补齐 missing describe、补齐单元/集成测试、跑完整回归、更新 README/metrics contract。                                                             | schema、recorder、mesh_metrics、engine_metrics、routes 关键测试通过；文档与代码一致；worker label 收敛方案单独记录。 |


如果需要降低风险，Day 1 的调用点迁移可以拆成 4 个小 PR：

- PR 1：HTTP middleware。只改 `middleware.rs` 中的 HTTP request/response/duration/rate limit 打点，将直接 `Metrics::*` 调用切到 `MeshMetrics`；验证 API 和 rate limit 测试。
- PR 2：HTTP router。只改 `routers/http/router.rs`、`routers/http/pd_router.rs` 中的 router request、upstream response、retry/backoff、worker selection 打点；验证 routing、PD routing、retry 测试。
- PR 3：gRPC router。只改 `routers/grpc/*` pipeline、stage、streaming 中的 router/inference/stage 打点；验证 gRPC 相关单测和至少一组 smoke test。
- PR 4：worker/reliability。只改 worker、worker registry、circuit breaker、retry、policy 中的 worker health/load、circuit breaker、retry、policy branch 打点；验证 reliability、worker management、policy 测试。

每个 PR 都应保持指标名和 label 不变，只改变调用入口，避免“拆模块”和“改指标语义”混在一起。

主要风险与控制原则：

- 指标契约漂移是贯穿全程的最大风险。所有阶段都必须保持 metric name、label key、HTTP path label value、worker label 和 `/engine_metrics` 失败语义不变；确需改变时拆独立 PR。
- 高风险改动集中在 exporter 和 engine metrics 聚合。`mesh_metrics.rs` 迁移要锁定 scrape 端口、bucket、`upkeep_timeout` 和全局 recorder 安装；`engine_metrics.rs` 迁移要锁定 partial failure、parse failure、label 注入和 label mismatch 行为。
- Day 5 只做收尾验证、missing describe 修复和文档同步，不承接新的结构迁移。如果 Day 4 未完成，应顺延或继续拆 PR。

## 4. 测试与验收

第 3 部分负责说明迁移动作和交付物；本节只回答如何证明重构没有破坏现有行为。测试按两层组织：单元/局部测试锁定 metrics contract，`TestHarness` 覆盖真实请求链路和运行模式矩阵。

| 验收对象 | 测试方式 | 通过标准 |
| --- | --- | --- |
| `schema.rs` | 单元测试 | metric name、HELP/TYPE、label 常量唯一且完整；path normalize 保持现有 `/health`、`/v1/models`、动态 `{id}` 行为；interner/cardinality 规则可验证。 |
| `recorder.rs` | contract 测试 | `MeshMetrics` 关键 record API 产出正确 metric name 和 label；业务代码不再直接依赖旧单文件 `Metrics::*`。 |
| `mesh_metrics.rs` | exporter 测试 | 独立 Prometheus exporter 可 scrape；scrape 端口、duration buckets、`upkeep_timeout`、describe 和全局 recorder 安装语义保持兼容。 |
| `engine_metrics.rs` | 聚合单测 + harness 扩展 | 保留现有 aggregator 用例，新增 partial failure、all failure、parse failure、label mismatch、SGLang colon metric name；`VirtualWorker` 支持 `/metrics` replay 后，用 `TestHarness` 验证 `worker_addr` 注入和 200/500 状态。 |
| `routes.rs` | `TestHarness` / app 集成测试 | `/liveness`、`/readiness`、`/health`、`/health_generate`、`/engine_metrics` 迁入 route factory 后路径、状态码和 state 访问语义不漂移。 |
| 业务打点链路 | `TestHarness` metrics guardrail | 复用 HTTP regular、HTTP PD、gRPC regular、gRPC PD fixture，发送真实请求后断言关键 `mesh_*` 指标已产生，且 label 与 metrics contract 一致。 |
| 指标库存 | 静态检查或单元测试 | 未接线指标有明确 `active`、`planned`、`deprecated` 或 `missing_describe` 状态；已调用但缺 describe 的指标补齐 HELP/TYPE。 |

最终验收标准：

- `observability/metrics.rs` 被拆为 `observability/metrics/*`，`server.rs` 不再直接维护 metrics/health 细粒度 public routes。
- 业务路径通过 `MeshMetrics` 打点，metric name、label key、HTTP path label value、worker label 和 `/engine_metrics` 失败语义保持兼容。
- 单元/局部测试、exporter 测试、聚合测试和 `TestHarness` 端到端 guardrail 都通过。
- 文档、README/metrics contract 与代码一致。

