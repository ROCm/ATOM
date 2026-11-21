//! `/v1/rl/workers/{id}/engine/{*path}` through `build_app`, over workers in
//! the real registry.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::AUTHORIZATION, HeaderValue, StatusCode},
    Router,
};
use mesh::{
    config::{BackendType, RouterConfig},
    core::{BasicWorkerBuilder, ConnectionMode, Worker},
    rl::{
        path::{passthrough_query, validate_engine_path},
        proxy::BODY_CAP,
    },
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
    AppTestContext, TestRouterConfig,
};

const RL_KEY: &str = "rl-proxy-key";

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

async fn post(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(app, Request::post(uri).body(Body::empty()).unwrap()).await
}

fn mock_worker() -> MockWorker {
    MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    })
}

fn rl_config(control_timeout_secs: u64) -> RouterConfig {
    let mut config = TestRouterConfig::random(3994);
    config.backend = BackendType::Atom;
    config.rl.enabled = true;
    config.rl.api_key = Some(RL_KEY.to_string());
    config.rl.control_timeout_secs = control_timeout_secs;
    config
}

/// Registers `worker` through `POST /workers` and returns its registry id.
async fn register(ctx: &AppTestContext, app: &Router, worker: Value) -> String {
    let url = worker["url"].as_str().unwrap().to_string();
    let request = Request::post("/workers")
        .header("content-type", "application/json")
        .body(Body::from(worker.to_string()))
        .unwrap();
    let (status, _) = send(app, request).await;
    assert!(status.is_success(), "{status}");
    for _ in 0..1500 {
        let found = ctx
            .app_context
            .worker_registry
            .get_all_with_ids()
            .into_iter()
            .find(|(_, w)| w.url() == url);
        if let Some((id, _)) = found {
            return id.as_str().to_string();
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("{url} was not registered");
}

/// Puts `worker` straight into the registry, for workers `POST /workers`
/// cannot add (a gRPC worker, a dead address).
fn insert(ctx: &AppTestContext, worker: BasicWorkerBuilder) -> String {
    let worker: Arc<dyn Worker> = Arc::new(worker.build());
    ctx.app_context
        .worker_registry
        .register(worker)
        .as_str()
        .to_string()
}

#[tokio::test]
async fn rl_proxy_forwards_body_bytes_and_headers_verbatim() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    let ctx = AppTestContext::new_with_config(rl_config(5), vec![]).await;
    let app = ctx.create_app().await;
    let id = register(&ctx, &app, json!({"url": url, "api_key": "engine-secret"})).await;

    let sent = "{\"mode\":  \"keep\" ,\"n\":1}";
    let request = Request::post(format!(
        "/v1/rl/workers/{id}/engine/rl_echo?mode=keep&selector=engine%3Datom&level=2"
    ))
    .header("content-type", "application/json; charset=utf-8")
    .header("x-request-id", "req-1")
    .header(
        "traceparent",
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    )
    .header("tracestate", "vendor=1")
    .header("x-other", "not-forwarded")
    .body(Body::from(sent))
    .unwrap();
    let (status, outcome) = send(&app, request).await;
    assert_eq!(status, StatusCode::OK, "{outcome}");
    assert_eq!(outcome["worker_id"], id);
    assert_eq!(outcome["url"], url);
    assert_eq!(outcome["status"], 200);
    assert!(outcome["latency_ms"].is_u64());
    assert!(outcome.get("body_truncated").is_none());

    let echo = &outcome["body"];
    assert_eq!(echo["method"], "POST");
    assert_eq!(
        echo["body"], sent,
        "the body reaches the worker byte for byte"
    );
    assert_eq!(echo["query"], "mode=keep&level=2");
    let headers = &echo["headers"];
    assert_eq!(headers["content-type"], "application/json; charset=utf-8");
    assert_eq!(headers["authorization"], "Bearer engine-secret");
    assert_eq!(headers["x-request-id"], "req-1");
    assert_eq!(
        headers["traceparent"],
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    );
    assert_eq!(headers["tracestate"], "vendor=1");
    assert!(headers.get("x-other").is_none(), "{headers}");

    // A body without a content type is sent as JSON; a GET is sent as a GET.
    let request = Request::post(format!("/v1/rl/workers/{id}/engine/rl_echo"))
        .body(Body::from("{}"))
        .unwrap();
    let (_, outcome) = send(&app, request).await;
    assert_eq!(
        outcome["body"]["headers"]["content-type"],
        "application/json"
    );
    let request = Request::get(format!("/v1/rl/workers/{id}/engine/rl_echo"))
        .body(Body::empty())
        .unwrap();
    let (status, outcome) = send(&app, request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(outcome["body"]["method"], "GET");
    assert_eq!(outcome["body"]["query"], Value::Null);

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_proxy_relays_an_engine_500_as_an_outcome() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    let ctx = AppTestContext::new_with_config(rl_config(5), vec![]).await;
    let app = ctx.create_app().await;
    let id = register(&ctx, &app, json!({"url": url})).await;

    let (status, outcome) = post(
        &app,
        &format!("/v1/rl/workers/{id}/engine/rl_echo?status=500"),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(outcome["status"], 500);
    assert_eq!(outcome["worker_id"], id);
    assert_eq!(outcome["body"]["method"], "POST");
    // A worker without a key gets no `authorization`: the caller's `/v1/rl`
    // bearer never reaches the engine.
    let headers = &outcome["body"]["headers"];
    assert!(headers.get("authorization").is_none(), "{headers}");

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_proxy_times_out_with_504_upstream_timeout() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    let ctx = AppTestContext::new_with_config(rl_config(1), vec![]).await;
    let app = ctx.create_app().await;
    let id = register(&ctx, &app, json!({"url": url})).await;

    let (status, body) = post(&app, &format!("/v1/rl/workers/{id}/engine/rl_slow")).await;
    assert_eq!(status, StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(body["error"], "upstream_timeout");
    assert_eq!(body["worker_id"], id);
    assert_eq!(body["url"], url);
    assert!(body["message"]
        .as_str()
        .unwrap()
        .ends_with("timed out after 1s"));

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_proxy_truncates_a_body_past_the_cap() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    let ctx = AppTestContext::new_with_config(rl_config(5), vec![]).await;
    let app = ctx.create_app().await;
    let id = register(&ctx, &app, json!({"url": url})).await;

    let request = Request::post(format!("/v1/rl/workers/{id}/engine/rl_echo"))
        .header("content-type", "text/plain")
        .body(Body::from("x".repeat(BODY_CAP + 1024)))
        .unwrap();
    let (status, outcome) = send(&app, request).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(outcome["body_truncated"], true);
    assert_eq!(outcome["body"].as_str().unwrap().len(), BODY_CAP);

    ctx.shutdown().await;
    worker.stop().await;
}

/// An engine that answers any request `202` with a JSON body promised at 64
/// bytes, sends 11 of them and closes the connection.
pub(super) async fn cut_engine() -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut buf).await {
                        Ok(0) | Err(_) => return,
                        Ok(n) => request.extend_from_slice(&buf[..n]),
                    }
                }
                let head = "HTTP/1.1 202 Accepted\r\ncontent-type: application/json\r\n\
                            content-length: 64\r\n\r\n{\"applied\":";
                let _ = socket.write_all(head.as_bytes()).await;
                let _ = socket.shutdown().await;
            });
        }
    });
    url
}

#[tokio::test]
async fn rl_proxy_keeps_the_status_when_the_body_read_fails() {
    let url = cut_engine().await;
    let ctx = AppTestContext::new_with_config(rl_config(5), vec![]).await;
    let app = ctx.create_app().await;
    let id = insert(&ctx, BasicWorkerBuilder::new(&url));

    let (status, outcome) = post(&app, &format!("/v1/rl/workers/{id}/engine/flush_cache")).await;
    assert_eq!(status, StatusCode::ACCEPTED, "{outcome}");
    assert_eq!(outcome["status"], 202);
    assert_eq!(outcome["worker_id"], id);
    assert_eq!(outcome["body"], "{\"applied\":");
    assert!(outcome.get("body_truncated").is_none());
    assert!(outcome["body_error"]
        .as_str()
        .unwrap()
        .starts_with("reading response body"));

    ctx.shutdown().await;
}

#[tokio::test]
async fn rl_proxy_errors_carry_smg_statuses() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    let ctx = AppTestContext::new_with_config(rl_config(5), vec![]).await;
    let app = ctx.create_app().await;
    let id = register(&ctx, &app, json!({"url": url})).await;
    let grpc = insert(
        &ctx,
        BasicWorkerBuilder::new("grpc://127.0.0.1:50051")
            .connection_mode(ConnectionMode::Grpc { port: None }),
    );
    let dead = insert(&ctx, BasicWorkerBuilder::new("http://127.0.0.1:9"));

    for path in ["a/b/c/d/e", "a/../b", "pause%20generation", "pause/"] {
        let (status, body) = post(&app, &format!("/v1/rl/workers/{id}/engine/{path}")).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
        assert_eq!(body["error"], "invalid_engine_path", "{path}");
    }

    let (status, body) = post(&app, "/v1/rl/workers/nope/engine/flush_cache").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "worker_not_found");

    let (status, body) = post(&app, &format!("/v1/rl/workers/{grpc}/engine/flush_cache")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(body["error"], "no_control_endpoint");
    assert_eq!(body["worker_id"], grpc);
    assert_eq!(body["connection_mode"], "grpc");

    let (status, body) = post(&app, &format!("/v1/rl/workers/{dead}/engine/flush_cache")).await;
    assert_eq!(status, StatusCode::BAD_GATEWAY);
    assert_eq!(body["error"], "upstream_unreachable");
    assert_eq!(body["url"], "http://127.0.0.1:9");

    ctx.shutdown().await;
    worker.stop().await;
}

#[test]
fn rl_engine_path_and_query_checks() {
    for (raw, want) in [
        ("pause_generation", "pause_generation"),
        ("/pause_generation", "pause_generation"),
        ("inference/v1/generate", "inference/v1/generate"),
        ("v1/chat/completions/render", "v1/chat/completions/render"),
    ] {
        assert_eq!(validate_engine_path(raw).unwrap(), want, "{raw}");
    }
    for raw in [
        "",
        "/",
        "//pause",
        "../workers",
        "a/../b",
        "./a",
        "pause?mode=keep",
        "pause#frag",
        "a/b/c/d/e",
        "pause generation",
        "pause%20generation",
        "pause/",
    ] {
        let error = validate_engine_path(raw).unwrap_err();
        assert_eq!(error.code(), "invalid_engine_path", "{raw:?}");
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    }

    assert_eq!(passthrough_query(None), None);
    assert_eq!(passthrough_query(Some("")), None);
    assert_eq!(passthrough_query(Some("selector=engine%3Datom")), None);
    for (raw, want) in [
        (
            "mode=keep&selector=engine%3Datom&level=2",
            "mode=keep&level=2",
        ),
        ("tags=weights&tags=kv_cache", "tags=weights&tags=kv_cache"),
        ("selectorx=1", "selectorx=1"),
        ("flag", "flag"),
        ("a=1&&b=2", "a=1&b=2"),
    ] {
        assert_eq!(passthrough_query(Some(raw)).as_deref(), Some(want), "{raw}");
    }
}
