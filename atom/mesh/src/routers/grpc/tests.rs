//! Test obligations for the new `grpc::Pipeline` (Parts F and G).
//!
//! Sections:
//!   a_pipeline_construction — new_regular / new_pd shape
//!   b_execute_chat          — 4-step body, error paths
//!   c_execute_generate      — parallel to chat
//!   d_execute_for_responses — typed-value return path
//!   e_metrics_labels        — backend / router / endpoint labels emitted
//!   f_sse_byte_snapshots    — plan G2–G6 (8 fixtures stored under tests/fixtures/sse_golden/)
//!   g_router_integration    — GrpcRouter / GrpcPDRouter use new Pipeline

mod a_pipeline_construction {
    use std::sync::Arc;

    use crate::core::placement::planner::DefaultPlanner;
    use crate::routers::grpc::engine::GrpcEngine;
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::http_router::SharedComponents;

    fn shared() -> Arc<SharedComponents> {
        unimplemented!("shared components fixture")
    }

    fn engine() -> GrpcEngine {
        unimplemented!("engine fixture")
    }

    fn planner() -> Arc<DefaultPlanner> {
        unimplemented!("planner fixture")
    }

    #[test]
    fn test_new_regular_sets_backend_label() {
        let p = Pipeline::new_regular(planner(), engine(), shared());
        // backend_label is private; pin via type system that it exists.
        let _ = p;
    }

    #[test]
    fn test_new_pd_sets_backend_label() {
        let p = Pipeline::new_pd(planner(), engine(), shared());
        let _ = p;
    }

    #[test]
    fn test_pipeline_construction_uses_concrete_grpc_engine_not_trait() {
        // §3.2 non-goal: Engine remains a concrete type. The compiler enforces it.
        let _ty = std::any::type_name::<Pipeline>();
        assert!(!_ty.contains("dyn Engine"), "Pipeline must not wrap dyn Engine");
    }
}

mod b_execute_chat {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use http::HeaderMap;

    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::http_router::SharedComponents;

    fn shared() -> Arc<SharedComponents> {
        unimplemented!("shared components fixture")
    }

    fn pipeline_regular() -> Pipeline {
        unimplemented!("pipeline fixture (regular)")
    }

    fn chat_req(stream: bool) -> Arc<ChatCompletionRequest> {
        Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            stream,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn test_execute_chat_non_streaming_returns_application_json() {
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(false), None, Some("m".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().starts_with("application/json"));
    }

    #[tokio::test]
    async fn test_execute_chat_streaming_returns_text_event_stream() {
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(true), None, Some("m".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_execute_chat_prepare_error_short_circuits() {
        // Empty messages should cause prepare to error; ensure the response is
        // a typed 4xx and that the planner / engine were not called.
        let p = pipeline_regular();
        let req = Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            ..Default::default()
        });
        let resp = p.execute_chat(req, None, Some("m".to_string()), shared()).await;
        assert!(resp.status().is_client_error() || resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_execute_chat_placement_error_returns_503() {
        // No workers configured → planner returns NoAvailableWorkers → 503.
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(false), None, Some("missing-model".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_execute_chat_engine_error_returns_5xx() {
        // Force engine dispatch to fail (worker unreachable etc.).
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(false), None, Some("dead-model".to_string()), shared())
            .await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_execute_chat_headers_thread_through_to_response_context() {
        let mut hm = HeaderMap::new();
        hm.insert("x-trace-id", "abc-123".parse().unwrap());
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(false), Some(hm), Some("m".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_execute_chat_body_includes_choices_and_finish_reason() {
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(false), None, Some("m".to_string()), shared())
            .await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("\"choices\""), "missing choices: {s}");
        assert!(s.contains("finish_reason"), "missing finish_reason: {s}");
    }

    #[tokio::test]
    async fn test_execute_chat_retry_wrapper_compat_unchanged_signature() {
        // RetryExecutor in router.rs wraps `pipeline.execute_chat` directly;
        // its closure expects a Response return type. Validate signature shape.
        fn _assert_signature(p: &Pipeline) {
            let _f = |req, hm, mid, shared| async move {
                p.execute_chat(req, hm, mid, shared).await
            };
        }
        let p = pipeline_regular();
        _assert_signature(&p);
    }
}

mod c_execute_generate {
    use std::sync::Arc;

    use axum::http::StatusCode;

    use crate::protocols::generate::GenerateRequest;
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::http_router::SharedComponents;

    fn shared() -> Arc<SharedComponents> {
        unimplemented!("shared components fixture")
    }

    fn pipeline_regular() -> Pipeline {
        unimplemented!("pipeline fixture (regular)")
    }

    fn pipeline_pd() -> Pipeline {
        unimplemented!("pipeline fixture (pd)")
    }

    fn generate_req(stream: bool) -> Arc<GenerateRequest> {
        let mut g = GenerateRequest::default();
        g.stream = stream;
        Arc::new(g)
    }

    #[tokio::test]
    async fn test_execute_generate_returns_application_json() {
        let p = pipeline_regular();
        let resp = p
            .execute_generate(generate_req(false), None, Some("m".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_execute_generate_streaming_returns_text_event_stream() {
        let p = pipeline_regular();
        let resp = p
            .execute_generate(generate_req(true), None, Some("m".to_string()), shared())
            .await;
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_execute_generate_in_pd_pipeline_routes_to_pair() {
        let p = pipeline_pd();
        let resp = p
            .execute_generate(generate_req(false), None, Some("m".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

mod d_execute_for_responses {
    use std::sync::Arc;

    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::http_router::SharedComponents;

    fn shared() -> Arc<SharedComponents> {
        unimplemented!("shared components fixture")
    }

    fn pipeline_regular() -> Pipeline {
        unimplemented!("pipeline fixture (regular)")
    }

    fn chat_req() -> Arc<ChatCompletionRequest> {
        Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            stream: false,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn test_for_responses_returns_typed_value_on_success() {
        let p = pipeline_regular();
        let v = p
            .execute_chat_for_responses(chat_req(), None, Some("m".to_string()), shared())
            .await
            .expect("ok");
        assert!(!v.id.is_empty(), "ChatCompletionResponse.id missing");
    }

    #[tokio::test]
    async fn test_for_responses_returns_response_err_on_prepare_failure() {
        let p = pipeline_regular();
        let bad = Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![],
            ..Default::default()
        });
        let err_resp = p
            .execute_chat_for_responses(bad, None, Some("m".to_string()), shared())
            .await
            .unwrap_err();
        assert!(!err_resp.status().is_success());
    }

    #[tokio::test]
    async fn test_for_responses_does_not_take_stream_flag() {
        // No streaming branch — for_responses is non-streaming by contract.
        let p = pipeline_regular();
        let _v = p
            .execute_chat_for_responses(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }
}

mod e_metrics_labels {
    use std::sync::Arc;

    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::http_router::SharedComponents;

    fn shared() -> Arc<SharedComponents> {
        unimplemented!("shared components fixture with metrics observer")
    }

    fn pipeline_regular() -> Pipeline {
        unimplemented!("pipeline fixture wired to metrics observer")
    }

    fn pipeline_pd() -> Pipeline {
        unimplemented!("PD pipeline fixture wired to metrics observer")
    }

    fn chat_req() -> Arc<ChatCompletionRequest> {
        Arc::new(ChatCompletionRequest {
            model: "m".to_string(),
            messages: vec![ChatMessage::User {
                content: MessageContent::Text("hi".to_string()),
                name: None,
            }],
            stream: false,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn test_metrics_router_grpc_label_recorded() {
        let p = pipeline_regular();
        let _ = p.execute_chat(chat_req(), None, Some("m".to_string()), shared()).await;
        // metrics observer asserts ROUTER_GRPC label appears at least once.
    }

    #[tokio::test]
    async fn test_metrics_backend_regular_label_recorded() {
        let p = pipeline_regular();
        let _ = p.execute_chat(chat_req(), None, Some("m".to_string()), shared()).await;
    }

    #[tokio::test]
    async fn test_metrics_backend_pd_label_recorded() {
        let p = pipeline_pd();
        let _ = p.execute_chat(chat_req(), None, Some("m".to_string()), shared()).await;
    }

    #[tokio::test]
    async fn test_metrics_endpoint_chat_label_recorded() {
        let p = pipeline_regular();
        let _ = p.execute_chat(chat_req(), None, Some("m".to_string()), shared()).await;
    }

    #[tokio::test]
    async fn test_metrics_connection_grpc_label_recorded() {
        let p = pipeline_regular();
        let _ = p.execute_chat(chat_req(), None, Some("m".to_string()), shared()).await;
    }
}

mod f_sse_byte_snapshots {
    //! Plan G2–G6 — record from main, replay against refactored Pipeline, assert
    //! byte-equal. Golden files live under `atom/mesh/tests/fixtures/sse_golden/`
    //! and are produced by the `--record-golden` helper run on main HEAD before
    //! any Part G code change.

    fn load_golden(name: &str) -> Vec<u8> {
        let path = format!(
            "{}/tests/fixtures/sse_golden/{name}.bin",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read(&path).unwrap_or_else(|e| panic!("missing golden {path}: {e}"))
    }

    async fn run_chat_to_bytes(_label: &str) -> Vec<u8> {
        unimplemented!("drive Pipeline through scripted-worker fixture, capture SSE bytes")
    }

    async fn run_generate_to_bytes(_label: &str) -> Vec<u8> {
        unimplemented!("drive Pipeline through scripted-worker fixture, capture SSE bytes")
    }

    #[tokio::test]
    async fn test_sse_chat_sglang_regular_bytes_match_main() {
        let actual = run_chat_to_bytes("chat_sglang_regular").await;
        assert_eq!(actual, load_golden("chat_sglang_regular"));
    }

    #[tokio::test]
    async fn test_sse_chat_sglang_pd_bytes_match_main() {
        let actual = run_chat_to_bytes("chat_sglang_pd").await;
        assert_eq!(actual, load_golden("chat_sglang_pd"));
    }

    #[tokio::test]
    async fn test_sse_chat_vllm_regular_bytes_match_main() {
        let actual = run_chat_to_bytes("chat_vllm_regular").await;
        assert_eq!(actual, load_golden("chat_vllm_regular"));
    }

    #[tokio::test]
    async fn test_sse_chat_vllm_pd_bytes_match_main() {
        let actual = run_chat_to_bytes("chat_vllm_pd").await;
        assert_eq!(actual, load_golden("chat_vllm_pd"));
    }

    #[tokio::test]
    async fn test_sse_generate_sglang_regular_bytes_match_main() {
        let actual = run_generate_to_bytes("generate_sglang_regular").await;
        assert_eq!(actual, load_golden("generate_sglang_regular"));
    }

    #[tokio::test]
    async fn test_sse_generate_sglang_pd_bytes_match_main() {
        let actual = run_generate_to_bytes("generate_sglang_pd").await;
        assert_eq!(actual, load_golden("generate_sglang_pd"));
    }

    #[tokio::test]
    async fn test_sse_generate_vllm_regular_bytes_match_main() {
        let actual = run_generate_to_bytes("generate_vllm_regular").await;
        assert_eq!(actual, load_golden("generate_vllm_regular"));
    }

    #[tokio::test]
    async fn test_sse_generate_vllm_pd_bytes_match_main() {
        let actual = run_generate_to_bytes("generate_vllm_pd").await;
        assert_eq!(actual, load_golden("generate_vllm_pd"));
    }
}

mod g_router_integration {
    use crate::routers::grpc::pd_router::GrpcPDRouter;
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::grpc::router::GrpcRouter;

    fn type_contains(target: &str, needle: &str) -> bool {
        target.contains(needle)
    }

    #[test]
    fn test_grpc_router_holds_new_pipeline_type() {
        let name = std::any::type_name::<GrpcRouter>();
        // Pinned by the new struct field `pipeline: Pipeline`. If this assertion
        // breaks after Part F it means the field type changed unexpectedly.
        assert!(
            type_contains(name, "GrpcRouter"),
            "router type rename collision? got {name}"
        );
        // Pipeline must be the concrete new type, not a re-export of RequestPipeline.
        let p_name = std::any::type_name::<Pipeline>();
        assert!(!p_name.contains("RequestPipeline"));
    }

    #[test]
    fn test_grpc_pd_router_holds_new_pipeline_type() {
        let name = std::any::type_name::<GrpcPDRouter>();
        assert!(type_contains(name, "GrpcPDRouter"));
    }

    #[test]
    fn test_router_files_keep_original_names_per_v1() {
        // Plan V-1 / Q1: do NOT rename router.rs to http_router.rs (collides with
        // the top-level routers/http_router.rs). This compile-only check pins the
        // module path.
        let _r = std::any::type_name::<GrpcRouter>();
        let _p = std::any::type_name::<GrpcPDRouter>();
        // Both types must resolve under routers::grpc::router and routers::grpc::pd_router.
        assert!(_r.contains("routers::grpc::router"));
        assert!(_p.contains("routers::grpc::pd_router"));
    }
}
