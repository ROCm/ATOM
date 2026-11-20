//! `/v1/rl/engine/{*path}?selector=` through `build_app`, over workers in the
//! real registry: DP-aware workers registered with `POST /workers`, and
//! workers it cannot add put in directly.

use std::{
    collections::BTreeSet,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::{
    body::Body,
    extract::Request,
    http::{header::AUTHORIZATION, HeaderValue, StatusCode, Uri},
    Json, Router,
};
use mesh::{
    config::{BackendType, RouterConfig},
    core::{BasicWorkerBuilder, ConnectionMode, Worker},
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
    AppTestContext, TestRouterConfig,
};

const RL_KEY: &str = "rl-fanout-key";

/// Sends `request` with the `/v1/rl` bearer.
async fn send(app: &Router, mut request: Request<Body>) -> (StatusCode, Value) {
    request.headers_mut().insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {RL_KEY}")).unwrap(),
    );
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap())
}

/// `POST /v1/rl/engine/{path}` with `selector` and an empty body.
async fn fan_out(app: &Router, path: &str, selector: &str) -> (StatusCode, Value) {
    let uri = format!("/v1/rl/engine/{path}?selector={}", encode(selector));
    send(app, Request::post(uri).body(Body::empty()).unwrap()).await
}

/// Percent-encodes everything but URL-safe characters.
fn encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

async fn mock_worker(dp_size: usize) -> (MockWorker, String) {
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url = worker.start().await.unwrap();
    worker.set_dp_size(dp_size).await;
    (worker, url)
}

fn rl_config(fanout_concurrency: usize) -> RouterConfig {
    let mut config = TestRouterConfig::random(3995);
    config.backend = BackendType::Atom;
    config.dp_aware = true;
    config.rl.enabled = true;
    config.rl.api_key = Some(RL_KEY.to_string());
    config.rl.control_timeout_secs = 5;
    config.rl.fanout_concurrency = fanout_concurrency;
    config
}

/// Registers `worker` through `POST /workers` and waits until the registry
/// holds `registered` workers.
async fn register(ctx: &AppTestContext, app: &Router, worker: Value, registered: usize) {
    let request = Request::post("/workers")
        .header("content-type", "application/json")
        .body(Body::from(worker.to_string()))
        .unwrap();
    let (status, _) = send(app, request).await;
    assert!(status.is_success(), "{status}");
    for _ in 0..1500 {
        if ctx.app_context.worker_registry.len() == registered {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("{worker} was not registered");
}

/// Puts `worker` straight into the registry, for workers `POST /workers`
/// cannot add (a gRPC worker, a dead address, many workers on one engine).
fn insert(ctx: &AppTestContext, worker: BasicWorkerBuilder) -> String {
    let worker: Arc<dyn Worker> = Arc::new(worker.build());
    ctx.app_context
        .worker_registry
        .register(worker)
        .as_str()
        .to_string()
}

fn id_of(ctx: &AppTestContext, url: &str) -> String {
    ctx.app_context
        .worker_registry
        .get_all_with_ids()
        .into_iter()
        .find(|(_, w)| w.url() == url)
        .unwrap_or_else(|| panic!("no worker at {url}"))
        .0
        .as_str()
        .to_string()
}

fn result_ids(report: &Value) -> BTreeSet<String> {
    report["results"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

#[tokio::test]
async fn rl_fanout_all_ok_is_200_and_reaches_each_engine_once() {
    let (mut a, a_url) = mock_worker(2).await;
    let (mut b, b_url) = mock_worker(1).await;
    let ctx = AppTestContext::new_with_config(rl_config(32), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, json!({"url": a_url}), 2).await;
    // The `engine` label is shadowed by the worker's real engine.
    register(
        &ctx,
        &app,
        json!({"url": b_url, "api_key": "b-secret", "labels": {"engine": "sglang"}}),
        3,
    )
    .await;
    let (a0, b0) = (
        id_of(&ctx, &format!("{a_url}@0")),
        id_of(&ctx, &format!("{b_url}@0")),
    );

    let sent = "{\"mode\":  \"keep\"}";
    let request = Request::post("/v1/rl/engine/rl_echo?mode=keep&selector=engine%3Datom")
        .header("content-type", "application/json")
        .body(Body::from(sent))
        .unwrap();
    let (status, report) = send(&app, request).await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["total"], 2);
    assert_eq!(report["succeeded"], 2);
    assert_eq!(report["failed"], json!([]));
    assert_eq!(
        result_ids(&report),
        BTreeSet::from([a0.clone(), b0.clone()])
    );
    assert_eq!((a.rl_echo_calls(), b.rl_echo_calls()), (1, 1));
    let echo = &report["results"][&b0]["body"];
    assert_eq!(echo["body"], sent, "the body reaches each worker verbatim");
    assert_eq!(echo["query"], "mode=keep", "the selector is not forwarded");
    assert_eq!(echo["headers"]["authorization"], "Bearer b-secret");
    assert!(report["results"][&a0]["body"]["headers"]
        .get("authorization")
        .is_none());

    let (status, report) = fan_out(&app, "rl_echo", &format!("url={b_url}@0")).await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["total"], 1);
    assert_eq!(result_ids(&report), BTreeSet::from([b0]));
    assert_eq!((a.rl_echo_calls(), b.rl_echo_calls()), (1, 2));

    ctx.shutdown().await;
    a.stop().await;
    b.stop().await;
}

#[tokio::test]
async fn rl_fanout_calls_a_dp_group_once_on_its_lowest_rank() {
    let (mut worker, url) = mock_worker(4).await;
    let ctx = AppTestContext::new_with_config(rl_config(32), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, json!({"url": url}), 4).await;

    let (status, report) = fan_out(&app, "rl_echo", "engine=atom").await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["total"], 1);
    let rank0 = id_of(&ctx, &format!("{url}@0"));
    assert_eq!(result_ids(&report), BTreeSet::from([rank0.clone()]));
    assert_eq!(report["results"][&rank0]["url"], format!("{url}@0"));
    assert_eq!(worker.rl_echo_calls(), 1, "four ranks, one engine call");

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_fanout_with_a_failing_worker_is_207_naming_it() {
    let (mut worker, url) = mock_worker(1).await;
    let ctx = AppTestContext::new_with_config(rl_config(32), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, json!({"url": url}), 1).await;
    let good = id_of(&ctx, &format!("{url}@0"));
    let dead = insert(&ctx, BasicWorkerBuilder::new("http://127.0.0.1:9"));
    let grpc = insert(
        &ctx,
        BasicWorkerBuilder::new("grpc://127.0.0.1:50051")
            .connection_mode(ConnectionMode::Grpc { port: None }),
    );

    let (status, report) = fan_out(&app, "rl_echo", "model!=none").await;
    assert_eq!(status, StatusCode::MULTI_STATUS, "{report}");
    assert_eq!(report["total"], 3);
    assert_eq!(report["succeeded"], 1);
    assert_eq!(result_ids(&report), BTreeSet::from([good.clone()]));
    let mut failed: Vec<(String, String)> = report["failed"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let id = f["worker_id"].as_str().unwrap().to_string();
            (id, f["error"].as_str().unwrap().to_string())
        })
        .collect();
    failed.sort();
    let mut want = vec![
        (dead.clone(), "upstream_unreachable".to_string()),
        (grpc.clone(), "no_control_endpoint".to_string()),
    ];
    want.sort();
    assert_eq!(failed, want);
    let by_id = |id: &str| {
        report["failed"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["worker_id"] == id)
            .unwrap()
            .clone()
    };
    assert_eq!(by_id(&dead)["url"], "http://127.0.0.1:9");
    assert_eq!(by_id(&grpc)["connection_mode"], "grpc");

    // An engine's own error status is a failure too, its outcome kept.
    let uri = format!("/v1/rl/engine/rl_echo?status=500&selector=id%3D{good}");
    let (status, report) = send(&app, Request::post(uri).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::MULTI_STATUS, "{report}");
    assert_eq!(report["succeeded"], 0);
    assert_eq!(report["failed"][0]["worker_id"], good);
    assert_eq!(report["failed"][0]["error"], "upstream_error");
    assert_eq!(report["failed"][0]["status"], 500);
    assert_eq!(report["results"][&good]["status"], 500);

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_fanout_no_match_is_400_and_sends_nothing() {
    let (mut worker, url) = mock_worker(2).await;
    let ctx = AppTestContext::new_with_config(rl_config(32), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, json!({"url": url}), 2).await;

    let rank1 = format!("url={url}@1");
    for selector in ["engine=sglang", rank1.as_str()] {
        let (status, body) = fan_out(&app, "rl_echo", selector).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{selector}");
        assert_eq!(body["error"], "no_workers_match");
        assert_eq!(body["selector"], selector);
    }
    assert_eq!(worker.rl_echo_calls(), 0);

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_fanout_selector_errors_are_400() {
    let ctx = AppTestContext::new_with_config(rl_config(32), vec![]).await;
    let app = ctx.create_app().await;

    for uri in [
        "/v1/rl/engine/flush_cache",
        "/v1/rl/engine/flush_cache?selector=",
        "/v1/rl/engine/flush_cache?selector=%20&mode=keep",
    ] {
        let (status, body) = send(&app, Request::post(uri).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        assert_eq!(body["error"], "selector_required", "{uri}");
    }
    let (status, body) = fan_out(&app, "flush_cache", "engine==atom").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "invalid_selector");
    assert_eq!(body["offset"], 7);
    let (status, body) = fan_out(&app, "a/../b", "engine=atom").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"], "invalid_engine_path");

    ctx.shutdown().await;
}

/// `pause_generation` calls answered, and the most in flight at once.
#[derive(Default)]
struct Load {
    calls: AtomicUsize,
    in_flight: AtomicUsize,
    peak: AtomicUsize,
}

/// An engine answering any path with `{}`, a `pause_generation` call after
/// `hold`; only those count, not the router's health checks.
async fn slow_engine(hold: Duration) -> (String, Arc<Load>) {
    let load = Arc::new(Load::default());
    let seen = load.clone();
    let app = Router::new().fallback(move |uri: Uri| {
        let load = seen.clone();
        async move {
            if !uri.path().ends_with("/pause_generation") {
                return Json(json!({}));
            }
            let now = load.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            load.peak.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(hold).await;
            load.in_flight.fetch_sub(1, Ordering::SeqCst);
            load.calls.fetch_add(1, Ordering::SeqCst);
            Json(json!({}))
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (url, load)
}

#[tokio::test]
async fn rl_fanout_never_exceeds_the_concurrency_flag() {
    let (url, load) = slow_engine(Duration::from_millis(150)).await;
    let ctx = AppTestContext::new_with_config(rl_config(2), vec![]).await;
    let app = ctx.create_app().await;
    // Six base URLs on one engine: six targets.
    for i in 0..6 {
        insert(&ctx, BasicWorkerBuilder::new(format!("{url}/{i}")));
    }

    let (status, report) = fan_out(&app, "pause_generation", "engine=atom").await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["succeeded"], 6);
    assert_eq!(load.calls.load(Ordering::SeqCst), 6);
    assert_eq!(load.peak.load(Ordering::SeqCst), 2, "at most two at once");

    ctx.shutdown().await;
}

#[tokio::test]
async fn rl_fanout_keeps_the_status_when_a_body_read_fails() {
    let url = super::proxy_test::cut_engine().await;
    let ctx = AppTestContext::new_with_config(rl_config(4), vec![]).await;
    let app = ctx.create_app().await;
    let id = insert(&ctx, BasicWorkerBuilder::new(&url));

    let (status, report) = fan_out(&app, "flush_cache", "engine=atom").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the engine applied the call: {report}"
    );
    assert_eq!(report["succeeded"], 1);
    assert_eq!(report["failed"], json!([]));
    let outcome = &report["results"][&id];
    assert_eq!(outcome["status"], 202);
    assert!(outcome["body_error"]
        .as_str()
        .unwrap()
        .starts_with("reading response body"));

    ctx.shutdown().await;
}
