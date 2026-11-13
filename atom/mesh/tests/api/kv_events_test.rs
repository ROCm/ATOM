//! With `--kv-events`, a worker registered through `/workers` is followed at the
//! endpoint it reports at `/kv_event_sources` until it is removed, as
//! `/kv_events/sources` shows. The batches are ATOM's own, from the decoder's samples.

use std::time::Duration;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    Router,
};
use bytes::Bytes;
use serde_json::{json, Value};
use tower::ServiceExt;
use zeromq::{PubSocket, Socket, SocketSend, ZmqMessage};

use crate::common::{
    mock_worker::{HealthStatus, MockWorker, MockWorkerConfig, WorkerType},
    AppTestContext, TestRouterConfig,
};

const SAMPLES: &str = include_str!("../fixtures/kv_events/samples.jsonl");

fn sample(name: &str) -> Bytes {
    let sample = SAMPLES
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .find(|sample| sample["name"] == name)
        .unwrap();
    let hex = sample["payload"].as_str().unwrap();
    let byte = |i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap();
    (0..hex.len()).step_by(2).map(byte).collect()
}

async fn publish(publisher: &mut PubSocket, seq: u64, payload: Bytes) {
    let mut message = ZmqMessage::from(Bytes::new());
    message.push_back(Bytes::copy_from_slice(&seq.to_be_bytes()));
    message.push_back(payload);
    publisher.send(message).await.unwrap();
}

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

/// `/kv_events/sources` once `done` holds for it. With a publisher, the startup clear is
/// published meanwhile: a subscription misses what is published before it arrives.
async fn sources_once(
    app: &Router,
    mut publisher: Option<&mut PubSocket>,
    done: impl Fn(&Value) -> bool,
) -> Value {
    let mut sources = Value::Null;
    for _ in 0..1500 {
        if let Some(publisher) = publisher.as_deref_mut() {
            publish(publisher, 0, sample("startup_clear")).await;
        }
        let (status, body) = call(app, "GET", "/kv_events/sources", None).await;
        assert_eq!(status, StatusCode::OK);
        sources = body;
        if done(&sources) {
            return sources;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let (_, workers) = call(app, "GET", "/workers", None).await;
    panic!("timed out at {sources}; workers {workers}");
}

// The worker's tokenizer job may block a thread while it tries to download.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_registered_worker_is_followed_until_removed() {
    let mut publisher = PubSocket::new();
    let bound = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
    let port = bound.to_string().rsplit_once(':').unwrap().1.to_string();
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url = worker.start().await.unwrap();
    let source = json!({"endpoint": format!("tcp://*:{port}"), "topic": "", "block_size": 4});
    worker.set_kv_event_sources(json!({"0": source})).await;
    let mut config = TestRouterConfig::random(3990);
    config.kv_events = true;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let app = ctx.create_app().await;

    let (status, _) = call(&app, "POST", "/workers", Some(json!({"url": url}))).await;
    assert!(status.is_success(), "{status}");
    let joined = |sources: &Value| sources[0]["batches"].as_u64() > Some(0);
    let sources = sources_once(&app, Some(&mut publisher), joined).await;
    let endpoint = format!("tcp://127.0.0.1:{port}");
    let want = json!({"worker": url, "endpoint": endpoint, "dp_rank": 0, "block_size": 4,
        "connected": true, "trust": "exact"});
    for (key, value) in want.as_object().unwrap() {
        assert_eq!(&sources[0][key], value, "{key}");
    }

    publish(&mut publisher, 1, sample("stored_root")).await;
    let sources = sources_once(&app, None, |sources| sources[0]["last_seq"] == 1).await;
    assert_eq!(sources[0]["stored_blocks"], 2);
    assert_eq!(sources[0]["trust"], "exact");

    let (_, workers) = call(&app, "GET", "/workers", None).await;
    let id = workers["workers"][0]["id"].as_str().unwrap();
    let (status, _) = call(&app, "DELETE", &format!("/workers/{id}"), None).await;
    assert!(status.is_success(), "{status}");
    sources_once(&app, None, |sources| sources == &json!([])).await;
    ctx.shutdown().await;
    worker.stop().await;
}

#[tokio::test]
async fn kv_event_sources_are_not_found_without_kv_events() {
    let ctx = AppTestContext::new(vec![]).await;
    let app = ctx.create_app().await;
    let (status, _) = call(&app, "GET", "/kv_events/sources", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    ctx.shutdown().await;
}
