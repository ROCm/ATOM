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

#[test]
fn new_api_routing_includes_system_tools_and_extension_content() {
    for path in ["/v1/messages", "/v1/responses"] {
        let mut value = if path.ends_with("messages") {
            json!({"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"system":"first"})
        } else {
            json!({"model":"m","input":"hi","instructions":"first"})
        };
        let first = ParsedInference::parse(path, &serde_json::to_vec(&value).unwrap())
            .unwrap()
            .metadata()
            .text;
        value[if path.ends_with("messages") {
            "system"
        } else {
            "instructions"
        }] = json!("different");
        let second = ParsedInference::parse(path, &serde_json::to_vec(&value).unwrap())
            .unwrap()
            .metadata()
            .text;
        assert_ne!(first, second);
        value["tools"] = json!([{"type":"function","name":"tool","vendor":{"x":1}}]);
        let third = ParsedInference::parse(path, &serde_json::to_vec(&value).unwrap())
            .unwrap()
            .metadata()
            .text;
        assert_ne!(second, third);
    }
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
    let _pool = PreparePoolRuntime::new(PoolConfig::default()).unwrap();
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
            .try_submit(app.prepare_pool.prepare_deadline(), move |ctx| {
                routing.prepare(&parsed, ctx)
            })
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(result.unwrap_err().code, "token_routing_unsupported");
    }
}
#[tokio::test]
async fn round_robin_skips_routing_text_and_candidates_keep_model_boundaries() {
    let _pool = PreparePoolRuntime::new(PoolConfig::default()).unwrap();
    let app = AppContext::from_config(
        crate::config::RouterConfig {
            policy: crate::config::PolicyConfig::RoundRobin,
            ..Default::default()
        },
        5,
        _pool.handle(),
    )
    .await
    .unwrap();
    let target: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:1")
            .model_id("m")
            .build(),
    );
    let other: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:2")
            .model_id("other")
            .build(),
    );
    app.worker_registry.register(target);
    app.worker_registry.register(other);
    let routing = IngressRouting::new(&app);
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model":"m","messages":[{"role":"user","content":"text"}]}),
        ),
        (
            "/v1/messages",
            json!({"model":"m","messages":[{"role":"user","content":"text"}],"max_tokens":1}),
        ),
        ("/v1/responses", json!({"model":"m","input":"text"})),
    ] {
        let parsed = ParsedInference::parse(path, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert!(!parsed.metadata().text.is_empty());
        let resources = routing.clone();
        let (metadata, tokens) = app
            .prepare_pool
            .try_submit(app.prepare_pool.prepare_deadline(), move |ctx| {
                resources.prepare(&parsed, ctx)
            })
            .unwrap()
            .wait()
            .await
            .unwrap()
            .unwrap();
        assert!(metadata.text.is_empty());
        assert!(tokens.is_none());
        let candidates = routing.candidates(&metadata, false).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].model_id(), "m");
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

#[tokio::test]
async fn sequential_prepare_preserves_deadline_body_and_input_lease() {
    let raw = Bytes::from_static(br#"{ "model": "m", "text": "test", "extension": [1, 2] }"#);
    let pool = PreparePoolRuntime::new(PoolConfig {
        workers: 1,
        max_retained_input_bytes: raw.len(),
        ..PoolConfig::default()
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
    assert_eq!(handle.stats().retained_input_bytes, raw.len());
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
    assert_eq!(handle.stats().retained_input_bytes, raw.len());
    let downstream = request.body.clone();
    drop(request);
    assert_eq!(
        handle.stats().retained_input_bytes,
        raw.len(),
        "downstream Bytes retains the input lease"
    );
    drop(downstream);
    assert_eq!(handle.stats().retained_input_bytes, 0);

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
    assert_eq!(handle.stats().retained_input_bytes, 0);
}

#[derive(Debug)]
struct NativeIngressRouter;

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
        _: &crate::protocols::generate::GenerateRequest,
        model: Option<&str>,
    ) -> Response {
        assert_eq!(model, Some("m"));
        assert_eq!(headers.unwrap().get("x-native").unwrap(), "yes");
        Response::new(axum::body::Body::from("native"))
    }
}

#[tokio::test]
async fn universal_parse_keeps_native_router_preparation_contract() {
    use http_body_util::BodyExt;
    let pool = PreparePoolRuntime::new(PoolConfig::default()).unwrap();
    let config = crate::config::RouterConfig {
        policy: crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
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
        router: Arc::new(NativeIngressRouter),
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
        ..PoolConfig::default()
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
