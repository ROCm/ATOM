use super::*;
use crate::core::{
    prepare_pool::{PoolConfig, PreparePoolRuntime},
    BasicWorkerBuilder,
};
use serde_json::json;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

fn default_pool() -> PoolConfig {
    crate::config::RouterConfig::default()
        .resolved_prepare_pool()
        .unwrap()
}

#[test]
fn response_state_never_fails_over_to_a_different_owner() {
    let parsed = ParsedInference::parse(
        "/v1/responses",
        br#"{"model":"m","input":"next","previous_response_id":"resp-1"}"#,
    )
    .unwrap();
    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:1")
            .model_id("m")
            .build(),
    );
    let second: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:2")
            .model_id("m")
            .build(),
    );
    assert!(parsed.requires_state_domain());
    first.set_healthy(false);
    assert_eq!(
        IngressRouting::validate_state_domain(&[first.clone(), second])
            .unwrap_err()
            .code,
        "stateful_routing_unsupported"
    );
    assert!(IngressRouting::validate_state_domain(&[first]).is_ok());
    let shared = [1, 2].map(|port| {
        Arc::new(
            BasicWorkerBuilder::new(format!("http://127.0.0.1:{port}"))
                .labels(std::collections::HashMap::from([(
                    "mesh.responses_state".into(),
                    "shared".into(),
                )]))
                .build(),
        ) as Arc<dyn Worker>
    });
    assert!(IngressRouting::validate_state_domain(&shared).is_ok());
    let background = ParsedInference::parse(
        "/v1/responses",
        br#"{"model":"m","input":"new","background":true}"#,
    )
    .unwrap();
    assert!(!background.requires_state_domain());
}

#[tokio::test]
async fn capabilities_and_exact_token_routing_are_explicit() {
    let worker = BasicWorkerBuilder::new("http://127.0.0.1:1")
        .labels(std::collections::HashMap::from([(
            "mesh.apis".into(),
            "/v1/messages, /v1/chat/completions".into(),
        )]))
        .build();
    assert!(EndpointSpec::find("/v1/messages")
        .unwrap()
        .supports(&worker));
    assert!(!EndpointSpec::find("/v1/responses")
        .unwrap()
        .supports(&worker));
    let config = crate::config::RouterConfig {
        policy: crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
        ..Default::default()
    };
    let _pool = PreparePoolRuntime::new(default_pool()).unwrap();
    let app = AppContext::from_config(config, 5, _pool.handle())
        .await
        .unwrap();
    for path in ["/v1/messages", "/v1/responses"] {
        let body = if path.ends_with("messages") {
            json!({"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1})
        } else {
            json!({"model":"m","input":"hi"})
        };
        let parsed = ParsedInference::parse(path, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(
            EndpointSpec::find(path)
                .unwrap()
                .validate_topology(true)
                .unwrap_err()
                .code,
            "unsupported_api_topology"
        );
        let routing = IngressRouting::new(&app);
        let result = app
            .prepare_pool
            .submit(app.prepare_pool.prepare_deadline(), move |ctx| {
                routing.prepare(&parsed, ctx)
            })
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(result.unwrap_err().code, "token_routing_unsupported");
    }
}
fn json_headers() -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "content-type",
        http::HeaderValue::from_static("application/json"),
    );
    headers
}

async fn preparation_test_router(
    pd: bool,
    policy: crate::config::PolicyConfig,
) -> (
    PreparePoolRuntime,
    Arc<AppContext>,
    Arc<dyn crate::routers::RouterTrait>,
) {
    let pool = PreparePoolRuntime::new(default_pool()).unwrap();
    let mut config = crate::config::RouterConfig {
        policy,
        backend: crate::config::BackendType::Sglang,
        disable_retries: true,
        ..Default::default()
    };
    if pd {
        config.mode = crate::config::RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
    }
    let app = Arc::new(
        AppContext::from_config(config, 5, pool.handle())
            .await
            .unwrap(),
    );
    let router: Arc<dyn crate::routers::RouterTrait> = if pd {
        Arc::new(
            crate::routers::http_pd_router::PDRouter::new(&app)
                .await
                .unwrap(),
        )
    } else {
        Arc::new(
            crate::routers::http_router::Router::new(&app)
                .await
                .unwrap(),
        )
    };
    (pool, app, router)
}

#[tokio::test]
async fn lightweight_http_routing_needs_no_second_pool_submission() {
    for pd in [false, true] {
        let (pool, app, router) =
            preparation_test_router(pd, crate::config::PolicyConfig::RoundRobin).await;
        let mut requests = Vec::new();
        for (path, body) in [
            (
                "/generate",
                json!({"model":"m","text":"test","vendor":{"keep":[1,2]}}),
            ),
            (
                "/v1/chat/completions",
                json!({"model":"m","messages":[{"role":"user","content":"test"}]}),
            ),
            ("/v1/completions", json!({"model":"m","prompt":"test"})),
            (
                "/v1/messages",
                json!({"model":"m","messages":[{"role":"user","content":"test"}],"max_tokens":1}),
            ),
            ("/v1/responses", json!({"model":"m","input":"test"})),
        ] {
            let request = InferenceEnvelope::parse(
                path.parse().unwrap(),
                json_headers(),
                Bytes::from(serde_json::to_vec(&body).unwrap()),
                &app,
            )
            .await
            .unwrap();
            assert!(request.metadata.text.is_empty());
            if pd {
                assert_eq!(request.pd_body.as_ref(), Some(&body));
            }
            requests.push(request);
        }
        let model_metadata = requests[0].metadata.clone();
        let mut expired = InferenceEnvelope::parse(
            "/generate".parse().unwrap(),
            json_headers(),
            Bytes::from_static(br#"{"model":"m","text":"test"}"#),
            &app,
        )
        .await
        .unwrap();
        expired.prepare_deadline = Instant::now() - Duration::from_millis(1);
        let invalid = InferenceEnvelope::parse(
            "/generate".parse().unwrap(),
            json_headers(),
            Bytes::from_static(br#"{"model":" ","text":"test"}"#),
            &app,
        )
        .await
        .err()
        .unwrap();
        assert_eq!(invalid.code, "invalid_request");

        // A closed pool rejects every submission, so reaching placement proves
        // routing did not acquire another CPU slot after the initial parse.
        pool.close();
        for request in requests {
            let unsupported =
                pd && matches!(request.metadata.route, "/v1/messages" | "/v1/responses");
            let response = router.route_inference(request, &app).await;
            assert_eq!(
                response.headers()["x-mesh-error-code"],
                if unsupported {
                    "unsupported_api_topology"
                } else {
                    "model_not_found"
                }
            );
        }
        // Preparation finished before the deadline passed, so the request is
        // still forwarded rather than discarded.
        let response = router.route_inference(expired, &app).await;
        assert_eq!(response.headers()["x-mesh-error-code"], "model_not_found");

        // Lightweight routing still keeps candidates within the model boundary.
        if !pd {
            for (port, model) in [(1, "m"), (2, "other")] {
                app.worker_registry.register(Arc::new(
                    BasicWorkerBuilder::new(format!("http://127.0.0.1:{port}"))
                        .model_id(model)
                        .build(),
                ));
            }
            let candidates = IngressRouting::new(&app)
                .candidates(&model_metadata, false)
                .unwrap();
            assert_eq!(candidates.len(), 1);
            assert_eq!(candidates[0].model_id(), "m");
        }
    }
}

#[tokio::test]
async fn sequential_prepare_preserves_deadline_and_body() {
    let raw = Bytes::from_static(br#"{ "model": "m", "text": "test", "extension": [1, 2] }"#);
    let pool = PreparePoolRuntime::new(PoolConfig {
        workers: 1,
        ..default_pool()
    })
    .unwrap();
    let handle = pool.handle();
    let app = AppContext::from_config(crate::config::RouterConfig::default(), 5, handle.clone())
        .await
        .unwrap();
    let request = InferenceEnvelope::parse(
        "/generate?trace=1".parse().unwrap(),
        json_headers(),
        raw.clone(),
        &app,
    )
    .await
    .unwrap();
    assert_eq!(
        request.body.as_ptr(),
        raw.as_ptr(),
        "input must not be deep-cloned"
    );
    let deadline = request.prepare_deadline;
    let (request, observed_deadline) = request
        .prepare(&handle, |request, ctx| {
            ctx.check()?;
            assert!(std::thread::current()
                .name()
                .unwrap_or("")
                .starts_with("mesh-prepare-"));
            Ok(request.prepare_deadline)
        })
        .await
        .unwrap();
    assert_eq!(observed_deadline, deadline);
    assert_eq!(request.prepare_deadline, deadline);
    assert_eq!(request.uri.to_string(), "/generate?trace=1");
    assert_eq!(request.body, raw);
    let forwarding = request.into_http_request();
    assert_eq!(forwarding.body.as_ptr(), raw.as_ptr());
    drop(forwarding);

    let mut request =
        InferenceEnvelope::parse("/generate".parse().unwrap(), json_headers(), raw, &app)
            .await
            .unwrap();
    request.prepare_deadline = Instant::now() - Duration::from_millis(1);
    let ran = Arc::new(AtomicBool::new(false));
    let observed = ran.clone();
    let error = request
        .prepare(&handle, move |_, _| {
            observed.store(true, Ordering::Release);
            Ok(())
        })
        .await
        .err()
        .unwrap();
    assert_eq!(error.code, "prepare_timeout");
    assert!(!ran.load(Ordering::Acquire));
}

struct TrackedRawInput {
    bytes: Vec<u8>,
    dropped: Arc<AtomicBool>,
}

impl AsRef<[u8]> for TrackedRawInput {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl Drop for TrackedRawInput {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Release);
    }
}

#[tokio::test]
async fn pd_handoff_releases_duplicate_raw_input() {
    let (_pool, app, _) =
        preparation_test_router(true, crate::config::PolicyConfig::RoundRobin).await;
    let body = json!({"model":"m","text":"test","vendor":{"keep":[1,2]}});
    let bytes = serde_json::to_vec(&body).unwrap();
    let dropped = Arc::new(AtomicBool::new(false));
    let raw = Bytes::from_owner(TrackedRawInput {
        bytes,
        dropped: dropped.clone(),
    });
    let request = InferenceEnvelope::parse(
        "/generate?trace=1".parse().unwrap(),
        json_headers(),
        raw,
        &app,
    )
    .await
    .unwrap();
    let (request, _) = request
        .prepare_routing(&app.prepare_pool, &IngressRouting::new(&app))
        .await
        .unwrap();
    assert!(!dropped.load(Ordering::Acquire));
    let (uri, _, forwarding) = request.into_pd_parts().unwrap();
    assert!(dropped.load(Ordering::Acquire));
    assert_eq!(uri.to_string(), "/generate?trace=1");
    assert_eq!(forwarding, body);
}

#[tokio::test]
async fn running_preparation_keeps_raw_input_after_cancel_or_timeout() {
    for cancel in [true, false] {
        let original = br#"{"model":"m","text":"test"}"#;
        let dropped = Arc::new(AtomicBool::new(false));
        let raw = Bytes::from_owner(TrackedRawInput {
            bytes: original.to_vec(),
            dropped: dropped.clone(),
        });
        let pool = PreparePoolRuntime::new(PoolConfig {
            workers: 1,
            ..default_pool()
        })
        .unwrap();
        let handle = pool.handle();
        let config = crate::config::RouterConfig {
            policy: crate::config::PolicyConfig::PrefixHash {
                prefix_token_count: 4,
                load_factor: 1.25,
            },
            ..Default::default()
        };
        let app = AppContext::from_config(config, 5, handle.clone())
            .await
            .unwrap();
        let (started, running) = tokio::sync::oneshot::channel();
        let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let release = ReleaseTokenizer(gate.clone());
        let tokenizer: Arc<dyn crate::tokenizer::traits::Tokenizer> = Arc::new(GatedTokenizer {
            inner: crate::tokenizer::MockTokenizer::new(),
            started: std::sync::Mutex::new(Some(started)),
            gate,
        });
        app.tokenizer_registry
            .load("gated-cancel", "m", "gated", || async { Ok(tokenizer) })
            .await
            .unwrap();
        let mut request =
            InferenceEnvelope::parse("/generate".parse().unwrap(), json_headers(), raw, &app)
                .await
                .unwrap();
        if !cancel {
            request.prepare_deadline = Instant::now() + Duration::from_millis(200);
        }
        let routing = IngressRouting::new(&app);
        let worker_handle = handle.clone();
        let task =
            tokio::spawn(async move { request.prepare_routing(&worker_handle, &routing).await });
        tokio::time::timeout(Duration::from_secs(2), running)
            .await
            .unwrap()
            .unwrap();
        if cancel {
            task.abort();
            assert!(matches!(task.await, Err(error) if error.is_cancelled()));
        } else {
            let result = task.await.unwrap();
            assert_eq!(result.err().unwrap().code, "prepare_timeout");
        }
        assert_eq!(handle.stats().busy_workers, 1);
        assert!(!dropped.load(Ordering::Acquire));
        drop(release);
        tokio::time::timeout(Duration::from_secs(2), async {
            while !dropped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

#[derive(Debug)]
struct NativeIngressRouter {
    raw_dropped: Option<Arc<AtomicBool>>,
}

#[async_trait::async_trait]
impl crate::routers::RouterTrait for NativeIngressRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn router_type(&self) -> &'static str {
        "native-test"
    }
    async fn route_chat(
        &self,
        _: Option<&HeaderMap>,
        _: &crate::protocols::chat::ChatCompletionRequest,
        _: Option<&str>,
    ) -> Response {
        Response::new(axum::body::Body::empty())
    }
    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &crate::protocols::generate::GenerateRequest,
        model: Option<&str>,
    ) -> Response {
        assert_eq!(model, Some("m"));
        assert_eq!(body.metadata().text, "test");
        assert_eq!(headers.unwrap().get("x-native").unwrap(), "yes");
        if let Some(dropped) = &self.raw_dropped {
            assert!(dropped.load(Ordering::Acquire));
        }
        tokio::task::yield_now().await;
        Response::new(axum::body::Body::from("native"))
    }
}

#[tokio::test]
async fn native_handoff_keeps_router_contract_and_releases_raw_input() {
    use crate::routers::RouterTrait;
    use http_body_util::BodyExt;

    let pool = PreparePoolRuntime::new(default_pool()).unwrap();
    let config = crate::config::RouterConfig {
        policy: crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
        connection_mode: ConnectionMode::Grpc { port: None },
        ..Default::default()
    };
    // HTTP preparation would fail without a tokenizer. The trait default must
    // preserve the native router's handling after the universal JSON parse.
    let context = Arc::new(
        AppContext::from_config(config, 5, pool.handle())
            .await
            .unwrap(),
    );
    let state = Arc::new(AppState {
        router: Arc::new(NativeIngressRouter { raw_dropped: None }),
        context,
        router_manager: None,
    });
    let request = Request::builder()
        .uri("/generate")
        .method("POST")
        .header("content-type", "application/json")
        .header("x-native", "yes")
        .body(axum::body::Body::from(r#"{"model":"m","text":"test"}"#))
        .unwrap();
    let response = EndpointSpec::inference(State(state), request).await;
    assert_eq!(response.status(), http::StatusCode::OK);
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        "native"
    );

    // The raw input is released before the native router runs, even when the
    // preparation deadline has passed after parsing.
    for expired in [false, true] {
        let pool = PreparePoolRuntime::new(default_pool()).unwrap();
        let app = Arc::new(
            AppContext::from_config(crate::config::RouterConfig::default(), 5, pool.handle())
                .await
                .unwrap(),
        );
        let dropped = Arc::new(AtomicBool::new(false));
        let raw = Bytes::from_owner(TrackedRawInput {
            bytes: br#"{"model":"m","text":"test"}"#.to_vec(),
            dropped: dropped.clone(),
        });
        let mut headers = json_headers();
        headers.insert("x-native", http::HeaderValue::from_static("yes"));
        let mut request =
            InferenceEnvelope::parse("/generate".parse().unwrap(), headers, raw, &app)
                .await
                .unwrap();
        assert!(!dropped.load(Ordering::Acquire));
        if expired {
            request.prepare_deadline = Instant::now() - Duration::from_millis(1);
        }
        let router = NativeIngressRouter {
            raw_dropped: Some(dropped.clone()),
        };
        // A deadline passing after preparation must not discard the request.
        let response = router.route_inference(request, &app).await;
        assert_eq!(response.status(), http::StatusCode::OK);
        assert!(dropped.load(Ordering::Acquire));
    }
}

struct ReleaseTokenizer(Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>);

impl Drop for ReleaseTokenizer {
    fn drop(&mut self) {
        let (lock, wake) = &*self.0;
        *lock.lock().unwrap() = true;
        wake.notify_all();
    }
}

struct GatedTokenizer {
    inner: crate::tokenizer::MockTokenizer,
    started: std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    gate: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
}

impl crate::tokenizer::traits::Encoder for GatedTokenizer {
    fn encode(
        &self,
        input: &str,
        special: bool,
    ) -> anyhow::Result<crate::tokenizer::traits::Encoding> {
        if let Some(started) = self.started.lock().unwrap().take() {
            let _ = started.send(());
        }
        let (lock, wake) = &*self.gate;
        let mut released = lock.lock().unwrap();
        while !*released {
            released = wake.wait(released).unwrap();
        }
        self.inner.encode(input, special)
    }
    fn encode_batch(
        &self,
        _: &[&str],
        _: bool,
    ) -> anyhow::Result<Vec<crate::tokenizer::traits::Encoding>> {
        panic!("HTTP preparation must preserve sequential encode")
    }
}

impl crate::tokenizer::traits::Decoder for GatedTokenizer {
    fn decode(&self, ids: &[u32], special: bool) -> anyhow::Result<String> {
        self.inner.decode(ids, special)
    }
}

impl crate::tokenizer::traits::Tokenizer for GatedTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn get_special_tokens(&self) -> &crate::tokenizer::traits::SpecialTokens {
        self.inner.get_special_tokens()
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

async fn blocked_encode_keeps_tokio_responsive(pd: bool) {
    use crate::{
        config::{BackendType, PolicyConfig, RouterConfig, RoutingMode},
        routers::RouterTrait,
    };
    let pool = PreparePoolRuntime::new(PoolConfig {
        workers: 1,
        prepare_timeout: Duration::from_secs(20),
        ..default_pool()
    })
    .unwrap();
    let mut config = RouterConfig {
        policy: PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
        backend: BackendType::Sglang,
        ..Default::default()
    };
    // Only the OS watchdog can release encode if Tokio itself is blocked.
    if pd {
        config.mode = RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        };
    }
    let context = Arc::new(
        AppContext::from_config(config, 5, pool.handle())
            .await
            .unwrap(),
    );
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let tokenizer: Arc<dyn crate::tokenizer::traits::Tokenizer> = Arc::new(GatedTokenizer {
        inner: crate::tokenizer::MockTokenizer::new(),
        started: std::sync::Mutex::new(Some(started_tx)),
        gate: gate.clone(),
    });
    context
        .tokenizer_registry
        .load("gated-id", "m", "gated", || async { Ok(tokenizer) })
        .await
        .unwrap();
    let router: Arc<dyn RouterTrait> = if pd {
        Arc::new(
            crate::routers::http_pd_router::PDRouter::new(&context)
                .await
                .unwrap(),
        )
    } else {
        Arc::new(
            crate::routers::http_router::Router::new(&context)
                .await
                .unwrap(),
        )
    };
    let state = Arc::new(AppState {
        router,
        context,
        router_manager: None,
    });
    let (progress_tx, progress_rx) = std::sync::mpsc::channel();
    let watchdog = std::thread::spawn(move || {
        let advanced = progress_rx.recv_timeout(Duration::from_secs(5)).is_ok();
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
        advanced
    });
    let request = Request::builder()
        .uri("/generate")
        .method("POST")
        .header("content-type", "application/json")
        .body(axum::body::Body::from(r#"{"model":"m","text":"test"}"#))
        .unwrap();
    let heavy = tokio::spawn(EndpointSpec::inference(State(state), request));
    tokio::time::timeout(Duration::from_secs(10), started_rx)
        .await
        .unwrap()
        .unwrap();
    // This heartbeat has to run while synchronous encode is still gated.
    tokio::task::yield_now().await;
    let _ = progress_tx.send(());
    let response = heavy.await.unwrap();
    assert!(
        watchdog.join().unwrap(),
        "Tokio heartbeat stalled until the OS watchdog released encode"
    );
    // No workers were registered: preparation succeeds, then placement fails.
    assert_eq!(response.status(), http::StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(flavor = "current_thread")]
async fn regular_http_encode_does_not_block_the_tokio_worker() {
    blocked_encode_keeps_tokio_responsive(false).await;
}

#[tokio::test(flavor = "current_thread")]
async fn pd_http_encode_does_not_block_the_tokio_worker() {
    blocked_encode_keeps_tokio_responsive(true).await;
}

#[tokio::test(flavor = "current_thread")]
async fn tokenize_endpoint_encodes_on_the_prepare_pool() {
    let pool = PreparePoolRuntime::new(PoolConfig {
        workers: 1,
        ..default_pool()
    })
    .unwrap();
    let context = Arc::new(
        AppContext::from_config(Default::default(), 5, pool.handle())
            .await
            .unwrap(),
    );
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let tokenizer: Arc<dyn crate::tokenizer::traits::Tokenizer> = Arc::new(GatedTokenizer {
        inner: crate::tokenizer::MockTokenizer::new(),
        started: std::sync::Mutex::new(Some(started_tx)),
        gate: gate.clone(),
    });
    context
        .tokenizer_registry
        .load("gated-tokenize", "m", "gated", || async { Ok(tokenizer) })
        .await
        .unwrap();
    // On a single-threaded runtime, inline encode would stall Tokio until the
    // watchdog releases the gate.
    let (progress_tx, progress_rx) = std::sync::mpsc::channel();
    let watchdog = std::thread::spawn(move || {
        let advanced = progress_rx.recv_timeout(Duration::from_secs(5)).is_ok();
        let (lock, wake) = &*gate;
        *lock.lock().unwrap() = true;
        wake.notify_all();
        advanced
    });
    let request: crate::protocols::tokenize::TokenizeRequest =
        serde_json::from_value(json!({"model": "m", "prompt": "hello world"})).unwrap();
    let task_context = context.clone();
    let task = tokio::spawn(async move {
        crate::routers::comm::tokenize::tokenize(&task_context, request).await
    });
    started_rx.await.unwrap();
    tokio::task::yield_now().await;
    let _ = progress_tx.send(());
    assert!(watchdog.join().unwrap(), "tokenize encode blocked Tokio");
    let response = task.await.unwrap();
    assert_eq!(response.status(), http::StatusCode::OK);
}

#[test]
fn dp_rank_is_forwarded_to_messages_and_responses_only_when_listed() {
    let unlabeled = crate::core::DPAwareWorkerBuilder::new("http://127.0.0.1:1", 0, 2).build();
    let labeled = crate::core::DPAwareWorkerBuilder::new("http://127.0.0.1:1", 1, 2)
        .label("mesh.apis", "/v1/chat/completions, /v1/messages")
        .build();
    let endpoint = |path| EndpointSpec::find(path).unwrap();
    for path in ["/v1/chat/completions", "/v1/completions", "/generate"] {
        assert!(endpoint(path).forwards_dp_rank(&unlabeled), "{path}");
    }
    for path in ["/v1/messages", "/v1/responses"] {
        assert!(endpoint(path).supports(&unlabeled), "{path}");
        assert!(!endpoint(path).forwards_dp_rank(&unlabeled), "{path}");
    }
    assert!(endpoint("/v1/messages").forwards_dp_rank(&labeled));
    assert!(!endpoint("/v1/responses").supports(&labeled));
}

fn single_rank(body: &[u8]) -> serde_json::Value {
    let value: serde_json::Value = serde_json::from_slice(body).unwrap();
    let text = std::str::from_utf8(body).unwrap();
    assert_eq!(text.matches("data_parallel_rank").count(), 1, "{text}");
    value
}

#[test]
fn dp_rank_body_replaces_the_client_rank_and_keeps_other_fields() {
    let empty = DpRankBody::prepare(br#"{"data_parallel_rank":9}"#).unwrap();
    assert_eq!(&empty.with_rank(3)[..], br#"{"data_parallel_rank":3}"#);
    assert_eq!(
        &DpRankBody::prepare(b"{}").unwrap().with_rank(0)[..],
        br#"{"data_parallel_rank":0}"#
    );
    let body = DpRankBody::prepare(
        br#"{"model":"m","data_parallel_rank":7,"vendor":{"keep":[1,{"x":null}]},"text":"t"}"#,
    )
    .unwrap();
    for rank in [0, 12] {
        let value = single_rank(&body.with_rank(rank));
        assert_eq!(
            value,
            json!({"model":"m","vendor":{"keep":[1,{"x":null}]},"text":"t","data_parallel_rank":rank})
        );
    }
    let error = DpRankBody::prepare(b"[1]").err().unwrap();
    assert_eq!(error.status, http::StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(error.code, "worker_request_preparation_failed");
}

#[tokio::test]
async fn dp_rank_body_is_prepared_in_the_pool_and_reused_for_each_rank() {
    let (_pool, app, _router) =
        preparation_test_router(false, crate::config::PolicyConfig::RoundRobin).await;
    let raw = Bytes::from_static(br#"{"model":"m","text":"t","data_parallel_rank":7}"#);
    let parse = || {
        InferenceEnvelope::parse_for_routing(
            "/generate".parse().unwrap(),
            json_headers(),
            raw.clone(),
            &app,
            Some(IngressRouting::new(&app)),
        )
    };
    let regular = parse().await.unwrap().into_http_request();
    assert!(regular.dp_rank_body.is_none());

    let workers: Vec<Arc<dyn Worker>> = (0..2)
        .map(|rank| {
            Arc::new(
                crate::core::DPAwareWorkerBuilder::new("http://127.0.0.1:1", rank, 2)
                    .model_id("m")
                    .build(),
            ) as Arc<dyn Worker>
        })
        .collect();
    for worker in &workers {
        app.worker_registry.register(worker.clone());
    }
    let request = parse().await.unwrap().into_http_request();
    let prepared = request.dp_rank_body.as_ref();
    assert!(prepared.is_some());
    for (rank, worker) in workers.iter().enumerate() {
        // Every attempt uses the prepared bytes, never the raw body.
        let body = dp_ranked_body(worker.as_ref(), "/generate", prepared, b"not json")
            .unwrap()
            .unwrap();
        assert_eq!(single_rank(&body)["data_parallel_rank"], rank);
        // A worker registered after preparation builds the body on forwarding.
        let late = dp_ranked_body(worker.as_ref(), "/generate", None, &raw)
            .unwrap()
            .unwrap();
        assert_eq!(late, body);
    }
    let basic = BasicWorkerBuilder::new("http://127.0.0.1:2").build();
    assert!(dp_ranked_body(&basic, "/generate", prepared, &raw)
        .unwrap()
        .is_none());
    assert!(
        dp_ranked_body(workers[0].as_ref(), "/v1/messages", prepared, &raw)
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn token_routing_is_prepared_in_the_parse_job_or_needs_the_pool_again() {
    // Without parse-job preparation, token routing needs another pool job.
    for pd in [false, true] {
        let (pool, app, router) = preparation_test_router(
            pd,
            crate::config::PolicyConfig::PrefixHash {
                prefix_token_count: 4,
                load_factor: 1.25,
            },
        )
        .await;
        let request = InferenceEnvelope::parse(
            "/generate".parse().unwrap(),
            json_headers(),
            Bytes::from_static(br#"{"model":"m","text":"test"}"#),
            &app,
        )
        .await
        .unwrap();
        pool.close();
        let response = router.route_inference(request, &app).await;
        assert_eq!(
            response.headers()["x-mesh-error-code"],
            "prepare_pool_closed"
        );
    }

    let pool = PreparePoolRuntime::new(PoolConfig {
        workers: 1,
        ..default_pool()
    })
    .unwrap();
    let config = crate::config::RouterConfig {
        policy: crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
        ..Default::default()
    };
    let app = AppContext::from_config(config, 5, pool.handle())
        .await
        .unwrap();
    let tokenizer: Arc<dyn crate::tokenizer::traits::Tokenizer> =
        Arc::new(crate::tokenizer::MockTokenizer::new());
    app.tokenizer_registry
        .load("mock-id", "m", "mock", || async { Ok(tokenizer) })
        .await
        .unwrap();
    let routing = IngressRouting::for_app(&app).unwrap();
    let request = InferenceEnvelope::parse_for_routing(
        "/generate".parse().unwrap(),
        json_headers(),
        Bytes::from_static(br#"{"model":"m","text":"hello world"}"#),
        &app,
        Some(routing.clone()),
    )
    .await
    .unwrap();
    assert!(request.routed.is_some());
    // A closed pool proves routing needs no second preparation job.
    let closed = PrepareHandle::closed();
    let (_, (metadata, tokens)) = request.prepare_routing(&closed, &routing).await.unwrap();
    assert_eq!(metadata.text, "hello world");
    assert!(tokens.is_some_and(|tokens| !tokens.is_empty()));
}

#[test]
fn untyped_item_references_are_accepted_as_state_references() {
    for (input, state) in [
        (json!([{"id": "msg_1"}]), true),
        (json!([{"type": "item_reference", "id": "msg_1"}]), true),
        (json!([{"role": "user", "content": "hi"}]), false),
        (
            json!([{"type": "message", "role": "user", "content": "hi"}]),
            false,
        ),
    ] {
        let body = serde_json::to_vec(&json!({"model": "m", "input": input})).unwrap();
        let parsed = ParsedInference::parse("/v1/responses", &body).unwrap();
        assert_eq!(parsed.requires_state_domain(), state, "{input}");
    }
    // An untyped item with neither an id nor a message shape is still invalid.
    let body = serde_json::to_vec(&json!({"model": "m", "input": [{"content": "hi"}]})).unwrap();
    assert!(ParsedInference::parse("/v1/responses", &body).is_err());
}

#[tokio::test]
async fn ingress_prepares_routing_only_for_http_routers() {
    use crate::config::{RouterConfig, RoutingMode};
    let pool = PreparePoolRuntime::new(default_pool()).unwrap();
    let pd = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    let grpc = ConnectionMode::Grpc { port: None };
    // The production AppState routes through RouterManager, so the decision
    // follows the configuration that RouterFactory uses to pick a router.
    for (mode, connection_mode, atom_standalone, prepares) in [
        (
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            ConnectionMode::Http,
            false,
            true,
        ),
        (pd.clone(), ConnectionMode::Http, false, true),
        (
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            grpc.clone(),
            false,
            false,
        ),
        (pd, grpc, false, false),
        (
            RoutingMode::Regular {
                worker_urls: vec![],
            },
            ConnectionMode::Http,
            true,
            false,
        ),
    ] {
        let config = RouterConfig {
            mode,
            connection_mode,
            atom_standalone,
            ..Default::default()
        };
        let app = AppContext::from_config(config, 5, pool.handle())
            .await
            .unwrap();
        assert_eq!(IngressRouting::for_app(&app).is_some(), prepares);
    }
}
