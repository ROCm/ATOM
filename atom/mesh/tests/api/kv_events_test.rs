//! With `--kv-events`, a worker registered through `/workers`, or each rank of a DP-aware
//! one, is followed at the endpoint it reports at `/kv_event_sources` until it is
//! removed, as `/kv_events/sources` shows, and `cache_aware` places requests by the blocks
//! it holds. The batches are ATOM's own, from the decoder's samples.

use std::{path::Path, sync::Arc, time::Duration};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
    Router,
};
use bytes::Bytes;
use mesh::tokenizer::{traits::Tokenizer, HuggingFaceTokenizer};
use serde_json::{json, Map, Value};
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn each_rank_of_a_dp_aware_worker_is_followed_until_the_worker_is_removed() {
    let mut publishers = Vec::new();
    let mut ports = Vec::new();
    for _ in 0..2 {
        let mut publisher = PubSocket::new();
        let bound = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
        ports.push(bound.to_string().rsplit_once(':').unwrap().1.to_string());
        publishers.push(publisher);
    }
    let mut worker = MockWorker::new(MockWorkerConfig {
        port: 0,
        worker_type: WorkerType::Regular,
        health_status: HealthStatus::Healthy,
        response_delay_ms: 0,
        fail_rate: 0.0,
    });
    let url = worker.start().await.unwrap();
    worker.set_dp_size(2).await;
    let source =
        |port: &str| json!({"endpoint": format!("tcp://*:{port}"), "topic": "", "block_size": 4});
    worker
        .set_kv_event_sources(json!({"0": source(&ports[0]), "1": source(&ports[1])}))
        .await;
    let mut config = TestRouterConfig::random(3992);
    config.kv_events = true;
    config.dp_aware = true;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let app = ctx.create_app().await;

    let (status, added) = call(&app, "POST", "/workers", Some(json!({"url": url}))).await;
    assert!(status.is_success(), "{status}");
    let connected = |sources: &Value| {
        let sources = sources.as_array().unwrap();
        sources.len() == 2 && sources.iter().all(|source| source["connected"] == true)
    };
    let sources = sources_once(&app, None, connected).await;
    for (rank, port) in ports.iter().enumerate() {
        let want = json!({"worker": format!("{url}@{rank}"), "dp_rank": rank,
            "endpoint": format!("tcp://127.0.0.1:{port}")});
        for (key, value) in want.as_object().unwrap() {
            assert_eq!(&sources[rank][key], value, "rank {rank} {key}");
        }
    }

    let id = added["worker_id"].as_str().unwrap();
    let (status, _) = call(&app, "DELETE", &format!("/workers/{id}"), None).await;
    assert!(status.is_success(), "{status}");
    sources_once(&app, None, |sources| sources == &json!([])).await;
    ctx.shutdown().await;
    worker.stop().await;
}

/// The worker that served a request.
async fn served_by(app: &Router, uri: &str, body: &Value) -> String {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    response.headers()["x-worker-id"].to_str().unwrap().into()
}

/// Registers a worker and waits until it is listed among `listed` workers.
async fn register(app: &Router, url: &str, listed: usize) {
    let (status, _) = call(app, "POST", "/workers", Some(json!({"url": url}))).await;
    assert!(status.is_success(), "{status}");
    for _ in 0..1500 {
        let (_, workers) = call(app, "GET", "/workers", None).await;
        if workers["workers"].as_array().map(Vec::len) == Some(listed) {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("{url} was not listed");
}

/// A tokenizer that, as Llama's, puts `<s>` before the words `a` to `i`, and renders a
/// chat as its messages' text.
fn bos_tokenizer(dir: &Path) -> Arc<dyn Tokenizer> {
    let words = ["<s>", "<unk>", "a", "b", "c", "d", "e", "f", "g", "h", "i"];
    let vocab: Map<String, Value> = (words.iter().enumerate())
        .map(|(id, word)| (word.to_string(), json!(id)))
        .collect();
    let special = |id: usize| {
        json!({"id": id, "content": words[id], "single_word": false, "lstrip": false,
            "rstrip": false, "normalized": false, "special": true})
    };
    let tokenizer = json!({"version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [special(0), special(1)], "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"}});
    let config = json!({"add_bos_token": true, "bos_token": "<s>",
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"});
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap())
}

/// ATOM's batch storing blocks of 4 `tokens` from the root, at rank 0.
fn stored(hashes: &[u64], tokens: &[u32]) -> Bytes {
    let event = json!({"type": "BlockStored", "block_hashes": hashes, "token_ids": tokens,
        "block_size": 4, "medium": "GPU"});
    let batch = rmpv::ext::to_value(json!([1.0, [event], 0])).unwrap();
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &batch).unwrap();
    payload.into()
}

/// With `cache_aware`, a request whose tokens a worker's device cache holds goes to that
/// worker, whatever the tree learned: its `input_ids`, or its prompt or chat encoded with
/// the special tokens, as ATOM stores the blocks. One with media is placed by its text.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_aware_places_requests_by_the_kv_event_index() {
    let mut publisher = PubSocket::new();
    let bound = publisher.bind("tcp://127.0.0.1:0").await.unwrap();
    let port = bound.to_string().rsplit_once(':').unwrap().1.to_string();
    let source = json!({"endpoint": format!("tcp://*:{port}"), "topic": "", "block_size": 4});
    let mut workers = Vec::new();
    for sources in [json!({"0": source}), json!({})] {
        let mut worker = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Regular,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let url = worker.start().await.unwrap();
        worker.set_kv_event_sources(sources).await;
        let id = format!("worker-{}", url.rsplit_once(':').unwrap().1);
        workers.push((worker, url, id));
    }
    let mut config = TestRouterConfig::cache_aware(3991);
    config.kv_events = true;
    let ctx = AppTestContext::new_with_config(config, vec![]).await;
    let dir = tempfile::tempdir().unwrap();
    let tokenizer = bos_tokenizer(dir.path());
    (ctx.app_context.tokenizer_registry)
        .load("bos", "test-model", "test", || async { Ok(tokenizer) })
        .await
        .unwrap();
    let app = ctx.create_app().await;
    let request = json!({"input_ids": (100..112).collect::<Vec<u32>>(),
        "sampling_params": {"max_new_tokens": 1}});
    let text = "a b c d e f g h i";
    let requests = [
        ("/generate", request.clone()),
        (
            "/v1/completions",
            json!({"model": "test-model", "prompt": text, "max_tokens": 1}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "test-model", "messages": [{"role": "user", "content": text}]}),
        ),
    ];

    // The tree learns each request on the worker without KV events, alone yet.
    register(&app, &workers[1].1, 1).await;
    for (uri, body) in &requests {
        assert_eq!(served_by(&app, uri, body).await, workers[1].2);
    }
    register(&app, &workers[0].1, 2).await;
    let joined = |sources: &Value| sources[0]["batches"].as_u64() > Some(0);
    sources_once(&app, Some(&mut publisher), joined).await;
    publish(&mut publisher, 1, sample("stored_root")).await;
    sources_once(&app, None, |sources| sources[0]["indexed_blocks"] == 2).await;

    // With media, the tree places it: an engine keys a media prompt's blocks by the media.
    let mut media = request.clone();
    media["image_data"] = json!("https://example.com/cat.png");
    assert_eq!(served_by(&app, "/generate", &media).await, workers[1].2);
    // stored_root holds tokens 100..=107, two of the request's three blocks.
    for _ in 0..4 {
        assert_eq!(served_by(&app, "/generate", &request).await, workers[0].2);
    }
    // `<s>` and a to g: two of the text's three blocks, as ATOM encodes it.
    publish(
        &mut publisher,
        2,
        stored(&[21, 22], &[0, 2, 3, 4, 5, 6, 7, 8]),
    )
    .await;
    sources_once(&app, None, |sources| sources[0]["indexed_blocks"] == 4).await;
    for (uri, body) in &requests[1..] {
        assert_eq!(served_by(&app, uri, body).await, workers[0].2, "{uri}");
    }
    ctx.shutdown().await;
    for (mut worker, _, _) in workers {
        worker.stop().await;
    }
}

#[tokio::test]
async fn kv_event_sources_are_not_found_without_kv_events() {
    let ctx = AppTestContext::new(vec![]).await;
    let app = ctx.create_app().await;
    let (status, _) = call(&app, "GET", "/kv_events/sources", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    ctx.shutdown().await;
}
