//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use mesh::config::RouterConfig;
use serde_json::json;
use tower::ServiceExt;

use crate::common::{
    mock_worker::{HealthStatus, MockWorkerConfig, WorkerType},
    AppTestContext, TestWorkerConfig,
};

#[cfg(test)]
mod pd_routing_tests {
    use super::*;

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use mesh::config::RetryConfig;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }
}

/// E/P/D routing with ATOM stubs: E runs before P for image chat requests,
/// and P encodes itself whenever E cannot serve.
mod epd_routing_tests {
    use std::sync::Arc;

    use http_body_util::BodyExt;
    use mesh::{
        config::BackendType,
        core::{prepare_pool::PreparePoolRuntime, Job},
        routers::RouterFactory,
    };
    use serde_json::Value;

    use super::*;
    use crate::common::{
        create_test_context,
        mock_worker::{
            start_atom_role, stub_ec_transfer_params, AtomRole, AtomRoleStub, EncodeReply,
            STUB_PREFILL_CACHE_SEED,
        },
        test_app::create_test_app_with_context,
    };

    struct Epd {
        app: axum::Router,
        app_context: Arc<mesh::app_context::AppContext>,
        prefill: AtomRoleStub,
        decode: AtomRoleStub,
        encode: AtomRoleStub,
        _prepare: PreparePoolRuntime,
    }

    /// One P, one D and one E, with a one-second request timeout for E.
    async fn start_epd(reply: EncodeReply) -> Epd {
        let prefill = start_atom_role(AtomRole::Prefill).await;
        let decode = start_atom_role(AtomRole::Decode).await;
        let encode = start_atom_role(AtomRole::Encode(reply)).await;
        let mut config = RouterConfig::builder()
            .prefill_decode_mode(vec![(prefill.url.clone(), None)], vec![decode.url.clone()])
            .backend(BackendType::Atom)
            .round_robin_policy()
            .host("127.0.0.1")
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(1)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();
        config.encode_urls = vec![encode.url.clone()];

        let (app_context, prepare) = create_test_context(config.clone()).await;
        app_context
            .worker_job_queue
            .get()
            .unwrap()
            .submit(Job::InitializeWorkersFromConfig {
                router_config: Box::new(config),
            })
            .await
            .unwrap();
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(10);
        while app_context
            .worker_registry
            .get_all()
            .iter()
            .filter(|w| w.is_healthy())
            .count()
            < 3
        {
            assert!(
                tokio::time::Instant::now() < deadline,
                "E, P and D never all registered"
            );
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let app = create_test_app_with_context(Arc::from(router), app_context.clone());
        Epd {
            app,
            app_context,
            prefill,
            decode,
            encode,
            _prepare: prepare,
        }
    }

    fn chat_payload(content: Value, stream: bool) -> Value {
        json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        })
    }

    fn chat(content: Value, stream: bool) -> Request<Body> {
        post_chat(chat_payload(content, stream))
    }

    fn post_chat(payload: Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    }

    fn image_parts() -> Value {
        json!([
            {"type": "text", "text": "Which color is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/red.png"}},
        ])
    }

    fn bodies(stub: &AtomRoleStub) -> Vec<Value> {
        stub.chat_bodies.lock().unwrap().clone()
    }

    /// Sends one request, expects 200, and returns P's and D's bodies.
    async fn serve(epd: &Epd, request: Request<Body>) -> (Value, Value, axum::response::Response) {
        let response = epd.app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let [prefill] = bodies(&epd.prefill).try_into().unwrap();
        let [decode] = bodies(&epd.decode).try_into().unwrap();
        (prefill, decode, response)
    }

    #[tokio::test]
    async fn image_chat_reaches_e_first_and_p_carries_its_params() {
        let epd = start_epd(EncodeReply::Params).await;
        let (prefill, decode, response) = serve(&epd, chat(image_parts(), false)).await;

        let [encode] = bodies(&epd.encode).try_into().unwrap();
        assert_eq!(encode["stream"], false);
        assert_eq!(encode["messages"][0]["content"], image_parts());
        assert_eq!(prefill["ec_transfer_params"], stub_ec_transfer_params());
        assert!(decode.get("ec_transfer_params").is_none(), "{decode}");
        assert_eq!(
            decode["kv_transfer_params"]["cache_seed"],
            STUB_PREFILL_CACHE_SEED
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let answer: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(answer["choices"][0]["message"]["content"], "red");
    }

    #[tokio::test]
    async fn image_chat_falls_back_to_pd_when_e_errors_or_times_out() {
        for (reply, posts_to_e) in [
            (EncodeReply::Status(503), 1),
            (EncodeReply::DelayMs(3000), 1),
        ] {
            let epd = start_epd(reply).await;
            let (prefill, decode, _) = serve(&epd, chat(image_parts(), false)).await;
            assert_eq!(bodies(&epd.encode).len(), posts_to_e, "{reply:?}");
            assert!(prefill.get("ec_transfer_params").is_none(), "{reply:?}");
            assert_eq!(prefill["messages"][0]["content"], image_parts());
            assert_eq!(
                decode["kv_transfer_params"]["cache_seed"],
                STUB_PREFILL_CACHE_SEED
            );
        }
    }

    #[tokio::test]
    async fn client_ec_transfer_params_never_reach_e_or_p() {
        let epd = start_epd(EncodeReply::Status(503)).await;
        let forged = json!({"session": "client:1", "addr": 0, "len": 1});
        for content in [json!("Which color is red?"), image_parts()] {
            epd.prefill.chat_bodies.lock().unwrap().clear();
            epd.decode.chat_bodies.lock().unwrap().clear();
            let mut payload = chat_payload(content, false);
            payload["ec_transfer_params"] = forged.clone();
            let (prefill, decode, _) = serve(&epd, post_chat(payload)).await;
            assert!(prefill.get("ec_transfer_params").is_none(), "{prefill}");
            assert!(decode.get("ec_transfer_params").is_none(), "{decode}");
        }
        let [encode] = bodies(&epd.encode).try_into().unwrap();
        assert!(encode.get("ec_transfer_params").is_none(), "{encode}");
    }

    #[tokio::test]
    async fn image_chat_without_a_healthy_encode_worker_skips_e() {
        let epd = start_epd(EncodeReply::Params).await;
        epd.app_context
            .worker_registry
            .get_by_url(&epd.encode.url)
            .expect("encode worker")
            .set_healthy(false);
        let (prefill, _, _) = serve(&epd, chat(image_parts(), false)).await;
        assert!(bodies(&epd.encode).is_empty());
        assert!(prefill.get("ec_transfer_params").is_none(), "{prefill}");
    }

    #[tokio::test]
    async fn text_chat_never_contacts_e() {
        let epd = start_epd(EncodeReply::Params).await;
        let text = json!([{"type": "text", "text": "Which color is red?"}]);
        for content in [text, json!("Which color is red?")] {
            epd.prefill.chat_bodies.lock().unwrap().clear();
            epd.decode.chat_bodies.lock().unwrap().clear();
            let (prefill, _, _) = serve(&epd, chat(content, false)).await;
            assert!(prefill.get("ec_transfer_params").is_none(), "{prefill}");
        }
        assert!(bodies(&epd.encode).is_empty());
    }

    #[tokio::test]
    async fn streaming_image_chat_streams_from_d() {
        let epd = start_epd(EncodeReply::Params).await;
        let (prefill, decode, response) = serve(&epd, chat(image_parts(), true)).await;

        let [encode] = bodies(&epd.encode).try_into().unwrap();
        assert_eq!(encode["stream"], false);
        assert_eq!(prefill["stream"], false);
        assert_eq!(prefill["ec_transfer_params"], stub_ec_transfer_params());
        assert_eq!(decode["stream"], true);
        assert_eq!(
            response.headers()[CONTENT_TYPE.as_str()],
            "text/event-stream"
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&body).ends_with("data: [DONE]\n\n"));
    }
}
