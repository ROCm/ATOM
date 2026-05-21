# Atomesh Metric Refactor Design

## 1. 背景与范围

Atomesh 是 Rust inference gateway，metrics 贯穿请求入口、router、worker、retry、circuit breaker、health check 和 worker engine 指标聚合。后续 smart router、worker pool 统一和动态 worker 注册都会继续扩大 metrics 的使用范围，因此本次重构不能只是移动 `metrics.rs`，而要先明确 metrics 的模块边界和调用方式。

本文只讨论 `atom/mesh/src/observability` 下 metrics 相关能力的设计重构。参考 Dynamo、vLLM、SGLang、TensorRT-LLM 等推理框架的命名习惯，保留 `observability` 作为根观测目录，并将 Prometheus metrics 收敛到 `src/observability/metrics` 子系统。

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

```rust
pub mod names {
    pub const HTTP_REQUESTS_TOTAL: &str = "mesh_http_requests_total";
    pub const ROUTER_REQUEST_DURATION_SECONDS: &str =
        "mesh_router_request_duration_seconds";
}

pub fn normalize_path(path: &str) -> &'static str {
    match path {
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/generate" => "/generate",
        _ if path.starts_with("/v1/responses/") => "/v1/responses/{response_id}",
        _ => "unknown",
    }
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

## 2. 改造范围

1. 新建 `src/observability/metrics/`，拆出 `mod.rs`、`config.rs`、`schema.rs`、`recorder.rs`、`mesh_metrics.rs`、`routes.rs`、`engine_metrics.rs`。第一阶段保持现有 `mesh_*` 指标名、label 和 Prometheus exporter 行为不变。
2. 将业务路径中的直接 `Metrics::*` 调用迁移到 `recorder.rs` 的 `MeshMetrics`，由调用方传入语义化参数，指标名、label、path normalize 和 interner 统一从 `schema.rs` 获取。
3. 将 Mesh self metrics 暴露逻辑迁入 `mesh_metrics.rs`，包括指标 describe、Prometheus exporter 安装、监听地址和 duration buckets；它只负责暴露 `mesh_*`，不处理 worker `/metrics`。
4. 将 `core/metrics_aggregator.rs` 和 `WorkerManager::get_engine_metrics` 中的 worker `/metrics` 拉取、解析、label 注入和合并逻辑迁入 `engine_metrics.rs`，补齐 partial failure、parse failure、label mismatch 语义。
5. 将 `/liveness`、`/readiness`、`/health`、`/health_generate`、`/engine_metrics` 从 `server.rs` 抽到 `routes.rs`，`server.rs` 改为 `.merge(metrics_factory.get(...))`；`/v1/models`、`/get_model_info`、`/get_server_info` 保持在 model/server info route factory。
6. 清理未接线指标：discovery、DB、manual policy 等要么接线，要么标记 planned/deprecated；逐步将 Mesh self metrics 的 worker 维度从 URL 收敛到稳定 `worker_id`，`worker_addr` 仅保留在 engine metrics 聚合结果中。

## 3. 实施计划

按 5 个工作日排布，优先保证“不改指标语义、不影响请求路径”，并为高风险的 exporter、engine metrics 聚合和路由迁移预留充足验证时间。


| 阶段    | 时间  | 主要工作                                                                                                                          | 交付物                                                                |
| ----- | --- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| Day 1 | 1 天 | 建立 `observability/metrics/`，迁移 `mod.rs/config.rs/schema.rs`，整理指标名、HELP、label、path normalize、interner、cardinality 规则。          | 新目录可编译；`schema.rs` 成为统一规范入口；原指标名和 label 不变。                        |
| Day 2 | 1 天 | 迁移 `recorder.rs`，将主要 `Metrics::*` 调用切到 `MeshMetrics`。优先完成 HTTP middleware、HTTP router、gRPC router、worker/reliability 的调用入口切换。 | 核心业务路径通过 `MeshMetrics` 打点；指标名和 label 语义不变。                         |
| Day 3 | 1 天 | 迁移 `mesh_metrics.rs`，保持 Prometheus exporter、scrape 端口、duration buckets、describe 行为兼容；补齐 Mesh self metrics scrape 验证。          | Mesh self metrics 暴露链路独立；Prometheus scrape 行为不变。                   |
| Day 4 | 1 天 | 迁移 `engine_metrics.rs` 和 `routes.rs`：迁入 `/engine_metrics` 聚合，抽出 health/metrics routes，`server.rs` 改为 route factory 组合。        | Mesh self metrics 与 engine metrics 分离；`server.rs` 职责收敛；路由语义不变。     |
| Day 5 | 1 天 | 清理未接线指标状态、收敛 worker label 规则、补齐单元/集成测试、跑完整回归、更新 README/metrics contract。                                                      | schema、recorder、mesh_metrics、engine_metrics、routes 关键测试通过；文档与代码一致。 |


如果需要降低风险，Day 1 的调用点迁移可以拆成 4 个小 PR：

- PR 1：HTTP middleware。只改 `middleware.rs` 中的 HTTP request/response/duration/rate limit 打点，将直接 `Metrics::*` 调用切到 `MeshMetrics`；验证 API 和 rate limit 测试。
- PR 2：HTTP router。只改 `routers/http/router.rs`、`routers/http/pd_router.rs` 中的 router request、upstream response、retry/backoff、worker selection 打点；验证 routing、PD routing、retry 测试。
- PR 3：gRPC router。只改 `routers/grpc/*` pipeline、stage、streaming 中的 router/inference/stage 打点；验证 gRPC 相关单测和至少一组 smoke test。
- PR 4：worker/reliability。只改 worker、worker registry、circuit breaker、retry、policy 中的 worker health/load、circuit breaker、retry、policy branch 打点；验证 reliability、worker management、policy 测试。

每个 PR 都应保持指标名和 label 不变，只改变调用入口，避免“拆模块”和“改指标语义”混在一起。

各阶段主要风险：

- Day 1 风险较低，但要避免在整理 `schema.rs` 时顺手改指标名或 label。
- Day 2 风险中等，调用点多，适合拆小 PR；每个 PR 都应保持指标名和 label 不变，只改变调用入口。
- Day 3 风险较高，`mesh_metrics.rs` 涉及 Prometheus exporter 初始化，容易影响 scrape 端口、bucket 或全局 recorder 安装。
- Day 4 风险最高，`engine_metrics.rs` 涉及 worker `/metrics` 聚合，容易改变部分失败、解析失败、label 注入行为；`routes.rs` 迁移也可能影响 health/readiness 语义。
- Day 5 不应承担大功能，只用于测试、修复迁移遗漏和文档更新。如果 Day 4 未完成，不建议挤占 Day 5 硬合并，应顺延或拆 PR。

## 4. 测试与验收

- Day 1 验收：`schema.rs` 覆盖指标名唯一性、HELP/label 常量、path normalize、label interner、cardinality 规则；新 `observability/metrics/` 模块可编译，原 `mesh_*` 指标名和 label 不变。
- Day 2 验收：`recorder.rs` 的关键打点入口使用正确指标名和 label；HTTP middleware、HTTP router、gRPC router、worker/reliability 调用点不再直接依赖旧单文件 `Metrics::*`；相关 API、routing、reliability 测试通过。
- Day 3 验收：`mesh_metrics.rs` 保持 Prometheus exporter、scrape 端口、duration buckets、describe 行为兼容；如启用主业务 `/metrics`，核心 `mesh_*` 可被 scrape。
- Day 4 验收：`engine_metrics.rs` 覆盖正常聚合、部分失败、全部失败、parse failure、label mismatch；`routes.rs` 验证 `/liveness`、`/readiness`、`/health`、`/health_generate`、`/engine_metrics` 路由语义不漂移。
- Day 5 验收：保留并迁移现有 `metrics_aggregator_test.rs`、`inflight_tracker_test.rs`；跑 API、routing、reliability 回归；未接线指标有明确状态；Mesh self metrics 与 worker engine metrics 两条链路边界清晰。
- 最终验收标准：`observability/metrics.rs` 被拆为 `observability/metrics/*`；`server.rs` 不再直接维护 metrics/health 细粒度 public routes；业务路径通过 `MeshMetrics` 打点；文档、README/metrics contract 与代码一致。

