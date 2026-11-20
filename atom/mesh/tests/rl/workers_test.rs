//! `/v1/rl/workers` through `build_app`, over workers registered with
//! `POST /workers` in the real registry: SMG's rows with `--enable-rl`, nothing
//! without it.

use std::collections::HashMap;

use axum::{
    body::Body,
    extract::Request,
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        StatusCode,
    },
    Router,
};
use clap::Parser;
use mesh::{
    cliargs::CliArgs,
    config::{BackendType, RouterConfig},
    rl::RlConfig,
};
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
    AppTestContext, TestRouterConfig,
};

const RL_KEY: &str = "rl-test-key";

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    call_as(app, method, uri, body, Some(&format!("Bearer {RL_KEY}"))).await
}

/// `call` with `Authorization: <authorization>`, or no such header.
async fn call_as(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
    authorization: Option<&str>,
) -> (StatusCode, Value) {
    let body = body.map_or_else(Body::empty, |body| Body::from(body.to_string()));
    let mut request = Request::builder()
        .method(method)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json");
    if let Some(authorization) = authorization {
        request = request.header(AUTHORIZATION, authorization);
    }
    let request = request.body(body).unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
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

fn rl_config(backend: BackendType, dp_aware: bool) -> RouterConfig {
    let mut config = TestRouterConfig::random(3993);
    config.backend = backend;
    config.dp_aware = dp_aware;
    config.rl.enabled = true;
    config.rl.api_key = Some(RL_KEY.to_string());
    config
}

/// Registers `url` with `labels` and waits until the registry holds `registered` workers.
async fn register(ctx: &AppTestContext, app: &Router, url: &str, labels: Value, registered: usize) {
    let (status, _) = call(
        app,
        "POST",
        "/workers",
        Some(json!({"url": url, "labels": labels})),
    )
    .await;
    assert!(status.is_success(), "{status}");
    for _ in 0..1500 {
        if ctx.app_context.worker_registry.len() == registered {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    panic!("{url} was not registered");
}

async fn rl_workers(app: &Router) -> Value {
    let (status, body) = call(app, "GET", "/v1/rl/workers", None).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body
}

#[tokio::test]
async fn rl_workers_lists_atom_workers_with_smg_fields() {
    let mut workers = [mock_worker(), mock_worker()];
    let mut urls = Vec::new();
    for worker in &mut workers {
        urls.push(worker.start().await.unwrap());
        worker.set_tp_size(4).await;
        worker.set_dp_size(2).await;
    }
    urls.sort();
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Atom, false), vec![]).await;
    let app = ctx.create_app().await;
    register(
        &ctx,
        &app,
        &urls[0],
        json!({"weight_version": "v0", "role": "actor"}),
        1,
    )
    .await;
    register(&ctx, &app, &urls[1], json!({}), 2).await;

    let body = rl_workers(&app).await;
    assert_eq!(body["protocol_version"], 1);
    assert_eq!(body["total"], 2);
    let rows = body["workers"].as_array().unwrap();
    for (row, url) in rows.iter().zip(&urls) {
        assert_eq!(&row["url"], url, "rows are sorted by base_url");
        assert_eq!(&row["base_url"], url);
        assert_eq!(&row["control_url"], url);
        assert_eq!(row["engine"], "atom");
        assert_eq!(row["worker_type"], "regular");
        assert_eq!(row["connection_mode"], "http");
        assert_eq!(row["dp_ranks"], 1);
        assert!(["ready", "not_ready"].contains(&row["health"].as_str().unwrap()));
        assert_eq!(
            row["capabilities"],
            json!({"source": "static", "pause_modes": [], "update_from": [],
                "abort": false, "flush_cache": false, "sleep_wake": false,
                "reports_weight_version": false}),
            "atom advertises no RL control yet"
        );
        assert_eq!(row["tp_size"], 4, "read from the worker's /server_info");
        assert_eq!(row["dp_size"], 2, "read from the worker's /server_info");
        assert_eq!(row["pp_size"], Value::Null);
        assert_eq!(row["engine_version"], Value::Null);
    }
    assert_eq!(rows[0]["weight_version"], "v0");
    assert_eq!(rows[0]["role"], "actor");
    assert_eq!(rows[0]["labels"]["weight_version"], "v0");
    assert_eq!(rows[1]["weight_version"], Value::Null);

    let id = rows[0]["id"].as_str().unwrap();
    let (status, one) = call(&app, "GET", &format!("/v1/rl/workers/{id}"), None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(one, rows[0]);

    ctx.shutdown().await;
    for worker in &mut workers {
        worker.stop().await;
    }
}

#[tokio::test]
async fn rl_workers_reports_sglang_with_its_capability_row() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    worker.set_tp_size(2).await;
    worker.set_dp_size(3).await;
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Sglang, false), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, &url, json!({}), 1).await;

    let row = &rl_workers(&app).await["workers"][0];
    assert_eq!(row["engine"], "sglang");
    assert_eq!(row["tp_size"], 2, "read from the worker's /server_info");
    assert_eq!(row["dp_size"], 3, "read from the worker's /server_info");
    assert_eq!(
        row["capabilities"]["pause_modes"],
        json!(["abort", "retract", "in_place"])
    );
    assert_eq!(
        row["capabilities"]["update_from"],
        json!(["disk", "tensor", "distributed"])
    );

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_workers_reports_the_sglang_engine_version() {
    // No sizes set: the mock answers /get_server_info, which carries `version`.
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Sglang, false), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, &url, json!({}), 1).await;

    let row = &rl_workers(&app).await["workers"][0];
    assert_eq!(row["engine_version"], "0.3.0");

    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn rl_workers_collapses_dp_ranks_into_one_row() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();
    worker.set_dp_size(2).await;
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Atom, true), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, &url, json!({}), 2).await;

    let body = rl_workers(&app).await;
    assert_eq!(body["total"], 1);
    let row = &body["workers"][0];
    assert_eq!(row["url"], format!("{url}@0"), "the lowest rank is kept");
    assert_eq!(row["base_url"], url);
    assert_eq!(row["control_url"], url);
    assert_eq!(row["dp_ranks"], 2);

    let rank1 = ctx
        .app_context
        .worker_registry
        .get_all_with_ids()
        .into_iter()
        .find(|(_, w)| w.url() == format!("{url}@1"))
        .unwrap()
        .0;
    let (status, one) = call(
        &app,
        "GET",
        &format!("/v1/rl/workers/{}", rank1.as_str()),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(one["url"], format!("{url}@1"));
    assert_eq!(one["dp_ranks"], 2);

    ctx.shutdown().await;
    worker.stop().await;
}

/// Registers `url` on a router with `--backend <backend>` and RL off, and
/// returns the labels discovery gave it.
async fn rl_off_labels(backend: BackendType, url: &str) -> HashMap<String, String> {
    let mut config = rl_config(backend, false);
    config.rl.enabled = false;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, url, json!({}), 1).await;
    let labels = ctx.app_context.worker_registry.get_all()[0]
        .metadata()
        .labels
        .clone();
    ctx.shutdown().await;
    labels
}

#[tokio::test]
async fn rl_off_discovery_adds_no_rl_labels() {
    let mut worker = mock_worker();
    let url = worker.start().await.unwrap();

    // No sizes set: the mock answers /get_server_info, which carries `version`.
    let labels = rl_off_labels(BackendType::Sglang, &url).await;
    assert_eq!(labels["model_path"], "mock-model-path", "{labels:?}");
    assert!(!labels.contains_key("version"), "{labels:?}");

    worker.set_tp_size(4).await;
    worker.set_dp_size(2).await;
    let labels = rl_off_labels(BackendType::Atom, &url).await;
    assert!(!labels.contains_key("tp_size"), "{labels:?}");
    assert!(!labels.contains_key("dp_size"), "{labels:?}");

    worker.stop().await;
}

#[tokio::test]
async fn rl_worker_unknown_id_is_404_worker_not_found() {
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Atom, false), vec![]).await;
    let app = ctx.create_app().await;
    let (status, body) = call(&app, "GET", "/v1/rl/workers/nope", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"], "worker_not_found");
    assert_eq!(body["id"], "nope");
    ctx.shutdown().await;
}

#[tokio::test]
async fn rl_off_mounts_no_route() {
    let mut config = rl_config(BackendType::Atom, false);
    config.rl.enabled = false;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let app = ctx.create_app().await;
    let (status, _) = call(&app, "GET", "/v1/rl/workers", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    ctx.shutdown().await;
}

#[tokio::test]
async fn rl_routes_require_the_bearer_token() {
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Atom, false), vec![]).await;
    let app = ctx.create_app().await;
    let refused = [
        None,
        Some("Bearer "),
        Some("Bearer wrong"),
        Some("Bearer rl-test-ke"),
        Some("Bearer rl-test-kez"),
        Some(RL_KEY),
        Some("Basic rl-test-key"),
    ];
    for uri in ["/v1/rl/workers", "/v1/rl/workers/nope"] {
        for authorization in refused {
            let (status, body) = call_as(&app, "GET", uri, None, authorization).await;
            assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri} {authorization:?}");
            assert_eq!(body["error"], "unauthorized", "{uri} {authorization:?}");
        }
    }
    for scheme in ["Bearer", "bearer", "BEARER"] {
        let authorization = format!("{scheme} {RL_KEY}");
        let (status, _) = call_as(&app, "GET", "/v1/rl/workers", None, Some(&authorization)).await;
        assert_eq!(status, StatusCode::OK, "{authorization}");
    }
    ctx.shutdown().await;
}

#[tokio::test]
async fn rl_routes_without_a_configured_key_refuse_every_request() {
    let mut config = rl_config(BackendType::Atom, false);
    config.rl.api_key = None;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let app = ctx.create_app().await;
    for authorization in [None, Some("Bearer ")] {
        let (status, _) = call_as(&app, "GET", "/v1/rl/workers", None, authorization).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{authorization:?}");
    }
    ctx.shutdown().await;
}

#[test]
fn rl_flags_parse_to_defaults_and_require_the_key() {
    let config = CliArgs::try_parse_from(["atomesh"])
        .unwrap()
        .to_router_config(Vec::new())
        .unwrap();
    assert_eq!(config.rl, RlConfig::default());

    let args = |extra: &[&str]| {
        let argv = ["atomesh", "--enable-rl"].iter().chain(extra);
        CliArgs::try_parse_from(argv)
            .unwrap()
            .to_router_config(Vec::new())
    };
    let config = args(&["--rl-api-key", "k"]).unwrap();
    assert_eq!(
        config.rl,
        RlConfig {
            enabled: true,
            api_key: Some("k".to_string()),
        }
    );
    for missing in [&[][..], &["--rl-api-key", ""], &["--rl-api-key", " "]] {
        let error = args(missing).unwrap_err().to_string();
        assert!(error.contains("--rl-api-key"), "{missing:?}: {error}");
    }
    for key in ["secret-key ", " secret-key", "secret-key\t"] {
        let error = args(&["--rl-api-key", key]).unwrap_err().to_string();
        assert!(error.contains("--rl-api-key"), "{key:?}: {error}");
        assert!(
            !error.contains("secret-key"),
            "the key is not echoed: {error}"
        );
    }
}
