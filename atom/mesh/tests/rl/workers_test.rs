//! `/v1/rl/workers` through `build_app`, over workers registered with
//! `POST /workers` in the real registry: SMG's rows with `--enable-rl`, nothing
//! without it.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
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

async fn call(app: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let body = body.map_or_else(Body::empty, |body| Body::from(body.to_string()));
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(body)
        .unwrap();
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
            json!({"source": "static", "pause_modes": ["keep"],
                "update_from": ["disk", "distributed"], "abort": false, "flush_cache": true,
                "sleep_wake": false, "reports_weight_version": true})
        );
        for key in [
            "engine_version",
            "model_id",
            "tp_size",
            "dp_size",
            "pp_size",
        ] {
            assert!(row.get(key).is_some(), "{key} missing from {row}");
        }
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
    let ctx = AppTestContext::new_with_config(rl_config(BackendType::Sglang, false), vec![]).await;
    let app = ctx.create_app().await;
    register(&ctx, &app, &url, json!({}), 1).await;

    let row = &rl_workers(&app).await["workers"][0];
    assert_eq!(row["engine"], "sglang");
    assert_eq!(row["tp_size"], 1, "read from the worker's /get_server_info");
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

#[test]
fn rl_flags_parse_to_defaults_and_reject_zero() {
    let config = CliArgs::try_parse_from(["atomesh"])
        .unwrap()
        .to_router_config(Vec::new())
        .unwrap();
    assert_eq!(
        config.rl,
        RlConfig {
            enabled: false,
            control_timeout_secs: 600,
            fanout_concurrency: 32,
        }
    );

    let args = |extra: &[&str]| {
        let argv = ["atomesh", "--enable-rl"].iter().chain(extra);
        CliArgs::try_parse_from(argv)
            .unwrap()
            .to_router_config(Vec::new())
    };
    let config = args(&[
        "--rl-control-timeout-secs",
        "7",
        "--rl-fanout-concurrency",
        "3",
    ])
    .unwrap();
    assert_eq!(
        config.rl,
        RlConfig {
            enabled: true,
            control_timeout_secs: 7,
            fanout_concurrency: 3,
        }
    );
    for flag in ["--rl-control-timeout-secs", "--rl-fanout-concurrency"] {
        let error = args(&[flag, "0"]).unwrap_err().to_string();
        assert!(error.contains("rl."), "{flag}: {error}");
    }
}
