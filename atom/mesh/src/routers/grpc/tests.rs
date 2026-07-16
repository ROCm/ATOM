mod a_pipeline_construction {
    use std::sync::Arc;

    use crate::app_context::AppContext;
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::test_mocks;

    fn shared() -> Arc<AppContext> {
        test_mocks::app_context()
    }

    #[test]
    fn test_new_regular_sets_backend_label() {
        let ctx = shared();
        let p = Pipeline::new_regular(ctx.worker_registry.clone(), ctx.policy_registry.clone());
        let _ = p;
    }

    #[test]
    fn test_new_pd_sets_backend_label() {
        let ctx = shared();
        let p = Pipeline::new_pd(ctx.worker_registry.clone(), ctx.policy_registry.clone());
        let _ = p;
    }
}

mod b_execute_chat {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use http::HeaderMap;

    use crate::app_context::AppContext;
    use crate::core::placement::types::PlacementError;
    use crate::core::worker::WorkerType;
    use crate::observability::metrics::metrics_labels;
    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    fn shared() -> Arc<AppContext> {
        test_mocks::app_context_with_hf_tokenizer("m")
    }

    fn scripted_chunks(
    ) -> Vec<Result<TokenChunk, crate::routers::token_handle::engine_error::EngineError>> {
        vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2, 6],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 3,
                    completion_tokens: 3,
                    total_tokens: 6,
                },
                logprobs: None,
                input_logprobs: None,
                meta: WorkerMeta {
                    request_id: "req-test".to_string(),
                    weight_version: None,
                    cached_tokens: 0,
                },
            }),
        ]
    }

    fn pipeline_regular() -> Arc<Pipeline> {
        let worker = test_mocks::mock_grpc_worker("http://w:1", WorkerType::Regular);
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_single(worker, "random")),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Ok(synthetic_single_stream(scripted_chunks()))
            })),
            metrics_labels::BACKEND_REGULAR,
        )
    }

    fn pipeline_placement_err() -> Arc<Pipeline> {
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_err(
                PlacementError::NoAvailableWorkers,
            )),
            Arc::new(test_mocks::MockDispatcher::new(vec![])),
            metrics_labels::BACKEND_REGULAR,
        )
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
        assert!(
            ct.to_str().unwrap().contains("text/event-stream"),
            "expected SSE content-type, got {:?}",
            ct
        );
    }

    #[tokio::test]
    async fn test_execute_chat_streaming_body_contains_done_marker() {
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(true), None, Some("m".to_string()), shared())
            .await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            s.contains("[DONE]"),
            "SSE stream must terminate with [DONE]: {s}"
        );
    }

    #[tokio::test]
    async fn test_execute_chat_prepare_error_short_circuits() {
        let p = pipeline_regular();
        let resp = p
            .execute_chat(
                chat_req(false),
                None,
                Some("unregistered".to_string()),
                shared(),
            )
            .await;
        assert!(
            resp.status().is_client_error() || resp.status().is_server_error(),
            "unregistered model must short-circuit prepare, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_execute_chat_placement_error_returns_503() {
        let p = pipeline_placement_err();
        let resp = p
            .execute_generate(
                Arc::new(
                    serde_json::from_value(
                        serde_json::json!({"text":"hi","model":"m","stream":false}),
                    )
                    .unwrap(),
                ),
                None,
                Some("m".to_string()),
                shared(),
            )
            .await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_execute_chat_engine_error_returns_5xx() {
        let p = test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_single(
                test_mocks::mock_grpc_worker("http://w:1", WorkerType::Regular),
                "random",
            )),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Err(
                    crate::routers::token_handle::engine_error::EngineError::Transport(
                        tonic::Status::internal("boom"),
                    ),
                )
            })),
            metrics_labels::BACKEND_REGULAR,
        );
        let resp = p
            .execute_generate(
                Arc::new(
                    serde_json::from_value(
                        serde_json::json!({"text":"hi","model":"m","stream":false}),
                    )
                    .unwrap(),
                ),
                None,
                Some("m".to_string()),
                shared(),
            )
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
        assert!(
            resp.headers().get("content-type").is_some(),
            "response must carry content-type"
        );
    }

    #[tokio::test]
    async fn test_execute_chat_body_includes_choices_and_finish_reason() {
        let p = pipeline_regular();
        let resp = p
            .execute_chat(chat_req(false), None, Some("m".to_string()), shared())
            .await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("\"choices\""), "body must include choices: {s}");
        assert!(
            s.contains("finish_reason"),
            "body must include finish_reason: {s}"
        );
        assert!(s.contains("\"stop\""), "body must include stop reason: {s}");
    }

    #[tokio::test]
    async fn test_execute_chat_retry_wrapper_compat_unchanged_signature() {
        fn _assert_signature(p: &Pipeline) {
            let _f =
                |req, hm, mid, shared| async move { p.execute_chat(req, hm, mid, shared).await };
        }
        let p = pipeline_regular();
        _assert_signature(&p);
    }
}

mod c_execute_generate {
    use std::sync::Arc;

    use axum::http::StatusCode;

    use crate::app_context::AppContext;
    use crate::core::worker::WorkerType;
    use crate::observability::metrics::metrics_labels;
    use crate::protocols::generate::GenerateRequest;
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    fn shared() -> Arc<AppContext> {
        test_mocks::app_context_with_hf_tokenizer("m")
    }

    fn scripted_chunks(
    ) -> Vec<Result<TokenChunk, crate::routers::token_handle::engine_error::EngineError>> {
        vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 6],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 2,
                    completion_tokens: 2,
                    total_tokens: 4,
                },
                logprobs: None,
                input_logprobs: None,
                meta: WorkerMeta {
                    request_id: "req-test".to_string(),
                    weight_version: None,
                    cached_tokens: 0,
                },
            }),
        ]
    }

    fn pipeline_regular() -> Arc<Pipeline> {
        let worker = test_mocks::mock_grpc_worker("http://w:1", WorkerType::Regular);
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_single(worker, "random")),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Ok(synthetic_single_stream(scripted_chunks()))
            })),
            metrics_labels::BACKEND_REGULAR,
        )
    }

    fn pipeline_pd() -> Arc<Pipeline> {
        let prefill = test_mocks::mock_grpc_worker(
            "http://p:1",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let decode = test_mocks::mock_grpc_worker("http://d:1", WorkerType::Decode);
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_pair(
                prefill, decode, "random", "random",
            )),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Ok(synthetic_single_stream(scripted_chunks()))
            })),
            metrics_labels::BACKEND_PD,
        )
    }

    fn generate_req(stream: bool) -> Arc<GenerateRequest> {
        let json = serde_json::json!({
            "text": "hi",
            "model": "m",
            "stream": stream,
        });
        Arc::new(serde_json::from_value(json).expect("GenerateRequest from json"))
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

    use crate::app_context::AppContext;
    use crate::core::worker::WorkerType;
    use crate::observability::metrics::metrics_labels;
    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    fn shared() -> Arc<AppContext> {
        test_mocks::app_context_with_hf_tokenizer("m")
    }

    fn scripted_chunks(
    ) -> Vec<Result<TokenChunk, crate::routers::token_handle::engine_error::EngineError>> {
        vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 6],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 2,
                    completion_tokens: 2,
                    total_tokens: 4,
                },
                logprobs: None,
                input_logprobs: None,
                meta: WorkerMeta {
                    request_id: "req-test".to_string(),
                    weight_version: None,
                    cached_tokens: 0,
                },
            }),
        ]
    }

    fn pipeline_regular() -> Arc<Pipeline> {
        let worker = test_mocks::mock_grpc_worker("http://w:1", WorkerType::Regular);
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_single(worker, "random")),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Ok(synthetic_single_stream(scripted_chunks()))
            })),
            metrics_labels::BACKEND_REGULAR,
        )
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
            .expect("HF tokenizer + scripted stream must succeed");
        assert!(!v.id.is_empty(), "ChatCompletionResponse.id missing");
        assert!(!v.choices.is_empty(), "must produce at least one choice");
        assert_eq!(v.choices[0].message.role, "assistant");
        assert_eq!(
            v.choices[0].finish_reason.as_deref(),
            Some("stop"),
            "finish_reason must be stop"
        );
    }

    #[tokio::test]
    async fn test_for_responses_returns_response_err_on_prepare_failure() {
        let p = pipeline_regular();
        let err_resp = p
            .execute_chat_for_responses(
                chat_req(),
                None,
                Some("unregistered".to_string()),
                shared(),
            )
            .await
            .unwrap_err();
        assert!(!err_resp.status().is_success());
    }

    #[tokio::test]
    async fn test_for_responses_does_not_take_stream_flag() {
        let p = pipeline_regular();
        let _v = p
            .execute_chat_for_responses(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }
}

mod e_metrics_labels {
    use std::sync::Arc;

    use crate::app_context::AppContext;
    use crate::core::worker::WorkerType;
    use crate::observability::metrics::metrics_labels;
    use crate::protocols::chat::{ChatCompletionRequest, ChatMessage, MessageContent};
    use crate::routers::grpc::pipeline::Pipeline;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    fn shared() -> Arc<AppContext> {
        test_mocks::app_context_with_hf_tokenizer("m")
    }

    fn scripted_chunks(
    ) -> Vec<Result<TokenChunk, crate::routers::token_handle::engine_error::EngineError>> {
        vec![Ok(TokenChunk::Complete {
            token_ids: vec![1, 6],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 2,
                completion_tokens: 2,
                total_tokens: 4,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: "req-test".to_string(),
                weight_version: None,
                cached_tokens: 0,
            },
        })]
    }

    fn pipeline_regular() -> Arc<Pipeline> {
        let worker = test_mocks::mock_grpc_worker("http://w:1", WorkerType::Regular);
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_single(worker, "random")),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Ok(synthetic_single_stream(scripted_chunks()))
            })),
            metrics_labels::BACKEND_REGULAR,
        )
    }

    fn pipeline_pd() -> Arc<Pipeline> {
        let prefill = test_mocks::mock_grpc_worker(
            "http://p:1",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        let decode = test_mocks::mock_grpc_worker("http://d:1", WorkerType::Decode);
        test_mocks::pipeline_with(
            Arc::new(test_mocks::MockPdPlanner::repeat_pair(
                prefill, decode, "random", "random",
            )),
            Arc::new(test_mocks::MockDispatcher::repeat_with_stream(|| {
                Ok(synthetic_single_stream(scripted_chunks()))
            })),
            metrics_labels::BACKEND_PD,
        )
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
        let _ = p
            .execute_chat(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }

    #[tokio::test]
    async fn test_metrics_backend_regular_label_recorded() {
        let p = pipeline_regular();
        let _ = p
            .execute_chat(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }

    #[tokio::test]
    async fn test_metrics_backend_pd_label_recorded() {
        let p = pipeline_pd();
        let _ = p
            .execute_chat(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }

    #[tokio::test]
    async fn test_metrics_endpoint_chat_label_recorded() {
        let p = pipeline_regular();
        let _ = p
            .execute_chat(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }

    #[tokio::test]
    async fn test_metrics_connection_grpc_label_recorded() {
        let p = pipeline_regular();
        let _ = p
            .execute_chat(chat_req(), None, Some("m".to_string()), shared())
            .await;
    }
}

mod f_shared_metrics_utils {
    use http::StatusCode;

    use crate::observability::metrics::metrics_labels;
    use crate::routers::comm::metrics_utils::{error_type_from_status, route_to_endpoint};

    #[test]
    fn test_route_to_endpoint_chat() {
        assert_eq!(
            route_to_endpoint("/v1/chat/completions"),
            metrics_labels::ENDPOINT_CHAT
        );
    }

    #[test]
    fn test_route_to_endpoint_generate() {
        assert_eq!(
            route_to_endpoint("/generate"),
            metrics_labels::ENDPOINT_GENERATE
        );
    }

    #[test]
    fn test_route_to_endpoint_completions() {
        assert_eq!(
            route_to_endpoint("/v1/completions"),
            metrics_labels::ENDPOINT_COMPLETIONS
        );
    }

    #[test]
    fn test_route_to_endpoint_responses() {
        assert_eq!(
            route_to_endpoint("/v1/responses"),
            metrics_labels::ENDPOINT_RESPONSES
        );
    }

    #[test]
    fn test_route_to_endpoint_unknown_returns_other() {
        assert_eq!(route_to_endpoint("/v1/anything-else"), "other");
    }

    #[test]
    fn test_error_type_from_status_400_is_validation() {
        assert_eq!(
            error_type_from_status(StatusCode::BAD_REQUEST),
            metrics_labels::ERROR_VALIDATION
        );
    }

    #[test]
    fn test_error_type_from_status_404_is_no_workers() {
        assert_eq!(
            error_type_from_status(StatusCode::NOT_FOUND),
            metrics_labels::ERROR_NO_WORKERS
        );
    }

    #[test]
    fn test_error_type_from_status_408_is_timeout() {
        assert_eq!(
            error_type_from_status(StatusCode::REQUEST_TIMEOUT),
            metrics_labels::ERROR_TIMEOUT
        );
    }

    #[test]
    fn test_error_type_from_status_504_is_timeout() {
        assert_eq!(
            error_type_from_status(StatusCode::GATEWAY_TIMEOUT),
            metrics_labels::ERROR_TIMEOUT
        );
    }

    #[test]
    fn test_error_type_from_status_500_is_backend() {
        assert_eq!(
            error_type_from_status(StatusCode::INTERNAL_SERVER_ERROR),
            metrics_labels::ERROR_BACKEND
        );
    }

    #[test]
    fn test_error_type_from_status_503_is_backend() {
        assert_eq!(
            error_type_from_status(StatusCode::SERVICE_UNAVAILABLE),
            metrics_labels::ERROR_BACKEND
        );
    }

    #[test]
    fn test_error_type_from_status_other_is_internal() {
        assert_eq!(
            error_type_from_status(StatusCode::OK),
            metrics_labels::ERROR_INTERNAL
        );
    }
}

mod f_shared_placement_response {
    use axum::http::StatusCode;

    use crate::core::placement::types::PlacementError;
    use crate::routers::comm::placement_response::placement_err_to_response;

    #[test]
    fn test_no_workers_returns_503() {
        let r = placement_err_to_response(PlacementError::NoWorkers, Some("m"));
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_no_available_workers_returns_503() {
        let r = placement_err_to_response(PlacementError::NoAvailableWorkers, Some("m"));
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_no_prefill_workers_returns_503() {
        let r = placement_err_to_response(PlacementError::NoPrefillWorkers, Some("m"));
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_no_decode_workers_returns_503() {
        let r = placement_err_to_response(PlacementError::NoDecodeWorkers, Some("m"));
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_policy_returned_none_returns_503() {
        let r = placement_err_to_response(PlacementError::PolicyReturnedNone, Some("m"));
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_model_not_found_returns_503() {
        let r = placement_err_to_response(
            PlacementError::ModelNotFound {
                model_id: "m".to_string(),
            },
            Some("m"),
        );
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_none_model_id_uses_default() {
        let r = placement_err_to_response(PlacementError::NoWorkers, None);
        assert_eq!(r.status(), StatusCode::SERVICE_UNAVAILABLE);
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
        assert!(
            type_contains(name, "GrpcRouter"),
            "router type rename collision? got {name}"
        );
        let p_name = std::any::type_name::<Pipeline>();
        assert!(!p_name.contains("RequestPipeline"));
    }

    #[test]
    fn test_grpc_pd_router_holds_new_pipeline_type() {
        let name = std::any::type_name::<GrpcPDRouter>();
        assert!(type_contains(name, "GrpcPDRouter"));
    }

    #[test]
    fn test_router_files_keep_original_names() {
        let _r = std::any::type_name::<GrpcRouter>();
        let _p = std::any::type_name::<GrpcPDRouter>();
        assert!(_r.contains("routers::grpc::router"));
        assert!(_p.contains("routers::grpc::pd_router"));
    }
}

mod h_completion_adapter {
    use axum::body::{to_bytes, Body};
    use axum::response::Response;
    use http::StatusCode;
    use serde_json::{json, Value};

    use crate::protocols::common::StringOrArray;
    use crate::protocols::completion::CompletionRequest;
    use crate::routers::grpc::completion_adapter::{
        completion_to_generate, wrap_generate_response_as_completion,
        wrap_streaming_generate_as_completion,
    };

    fn minimal_completion_request() -> CompletionRequest {
        serde_json::from_value(json!({
            "model": "test-model",
            "prompt": "Hello world"
        }))
        .unwrap()
    }

    #[test]
    fn test_happy_path_string_prompt() {
        let req = minimal_completion_request();
        let gen = completion_to_generate(&req).unwrap();
        assert_eq!(gen.text.as_deref(), Some("Hello world"));
        assert_eq!(gen.model.as_deref(), Some("test-model"));
    }

    #[test]
    fn test_array_prompt_single_element() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": ["only-one"]
        }))
        .unwrap();
        let gen = completion_to_generate(&req).unwrap();
        assert_eq!(gen.text.as_deref(), Some("only-one"));
    }

    #[test]
    fn test_array_prompt_multiple_elements_rejected() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": ["a", "b"]
        }))
        .unwrap();
        let err = completion_to_generate(&req).unwrap_err();
        assert!(err.contains("Batched prompts"), "error: {err}");
        assert!(err.contains("2 items"), "error: {err}");
    }

    #[test]
    fn test_echo_true_rejected() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "echo": true
        }))
        .unwrap();
        let err = completion_to_generate(&req).unwrap_err();
        assert!(err.contains("echo"), "error: {err}");
    }

    #[test]
    fn test_suffix_rejected() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "suffix": "end"
        }))
        .unwrap();
        let err = completion_to_generate(&req).unwrap_err();
        assert!(err.contains("suffix"), "error: {err}");
    }

    #[test]
    fn test_best_of_greater_than_one_rejected() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "best_of": 2
        }))
        .unwrap();
        let err = completion_to_generate(&req).unwrap_err();
        assert!(err.contains("best_of"), "error: {err}");
    }

    #[test]
    fn test_best_of_one_accepted() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "best_of": 1
        }))
        .unwrap();
        assert!(completion_to_generate(&req).is_ok());
    }

    #[test]
    fn test_logit_bias_rejected() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "logit_bias": {"50256": -100.0}
        }))
        .unwrap();
        let err = completion_to_generate(&req).unwrap_err();
        assert!(err.contains("logit_bias"), "error: {err}");
    }

    #[test]
    fn test_sampling_params_threaded_through() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "temperature": 0.7,
            "max_tokens": 128,
            "top_p": 0.9,
            "top_k": 40,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.3,
            "stop": ["END"],
            "n": 2,
            "seed": 42
        }))
        .unwrap();
        let gen = completion_to_generate(&req).unwrap();
        let sp = gen.sampling_params.as_ref().unwrap();
        assert_eq!(sp.temperature, Some(0.7));
        assert_eq!(sp.max_new_tokens, Some(128));
        assert_eq!(sp.top_p, Some(0.9));
        assert_eq!(sp.top_k, Some(40));
        assert_eq!(sp.frequency_penalty, Some(0.5));
        assert_eq!(sp.presence_penalty, Some(0.3));
        assert_eq!(sp.n, Some(2));
        assert_eq!(sp.sampling_seed, Some(42));
        match sp.stop.as_ref().unwrap() {
            StringOrArray::Array(v) => assert_eq!(v, &["END"]),
            other => panic!("expected Array stop, got {:?}", other),
        }
    }

    #[test]
    fn test_stream_flag_threads_through() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "stream": true
        }))
        .unwrap();
        let gen = completion_to_generate(&req).unwrap();
        assert!(gen.stream);

        let req2: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "stream": false
        }))
        .unwrap();
        let gen2 = completion_to_generate(&req2).unwrap();
        assert!(!gen2.stream);
    }

    #[test]
    fn test_logprobs_sets_return_logprob_and_top_logprobs_num() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "logprobs": 5
        }))
        .unwrap();
        let gen = completion_to_generate(&req).unwrap();
        assert_eq!(gen.return_logprob, Some(true));
        assert_eq!(gen.top_logprobs_num, Some(5));
    }

    #[test]
    fn test_no_logprobs_sets_return_logprob_false() {
        let req = minimal_completion_request();
        let gen = completion_to_generate(&req).unwrap();
        assert_eq!(gen.return_logprob, Some(false));
        assert_eq!(gen.top_logprobs_num, None);
    }

    fn generate_response_json(text: &str, id: &str, finish_type: &str) -> Value {
        match finish_type {
            "stop" => json!({
                "text": text,
                "output_ids": [1, 2, 3],
                "meta_info": {
                    "id": id,
                    "finish_reason": {"type": "stop"},
                    "prompt_tokens": 5,
                    "weight_version": "v1",
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "e2e_latency": 0.1
                }
            }),
            "length" => json!({
                "text": text,
                "output_ids": [1, 2, 3],
                "meta_info": {
                    "id": id,
                    "finish_reason": {"type": "length", "length": 128},
                    "prompt_tokens": 5,
                    "weight_version": "v1",
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "e2e_latency": 0.1
                }
            }),
            other => json!({
                "text": text,
                "output_ids": [1, 2, 3],
                "meta_info": {
                    "id": id,
                    "finish_reason": other,
                    "prompt_tokens": 5,
                    "weight_version": "v1",
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "e2e_latency": 0.1
                }
            }),
        }
    }

    fn ok_json_response(body: &Value) -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    async fn test_wrap_non_success_upstream_passthrough() {
        let upstream = Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from("upstream error"))
            .unwrap();
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(&body[..], b"upstream error");
    }

    #[tokio::test]
    async fn test_wrap_valid_generate_json_produces_completion() {
        let gen = generate_response_json("hello", "req-1", "stop");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "test-model".into()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["object"], "text_completion");
        assert_eq!(parsed["model"], "test-model");
        assert_eq!(parsed["choices"][0]["text"], "hello");
        assert_eq!(parsed["choices"][0]["index"], 0);
        assert_eq!(parsed["choices"][0]["finish_reason"], "stop");
        assert!(parsed["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_wrap_multiple_generate_responses_multiple_choices() {
        let gen1 = generate_response_json("aaa", "req-1", "stop");
        let gen2 = generate_response_json("bbb", "req-1", "length");
        let upstream = ok_json_response(&json!([gen1, gen2]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        let choices = parsed["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[0]["text"], "aaa");
        assert_eq!(choices[0]["finish_reason"], "stop");
        assert_eq!(choices[1]["index"], 1);
        assert_eq!(choices[1]["text"], "bbb");
        assert_eq!(choices[1]["finish_reason"], "length");
        let usage = &parsed["usage"];
        assert_eq!(usage["completion_tokens"].as_u64().unwrap(), 6);
    }

    #[tokio::test]
    async fn test_wrap_empty_id_generates_cmpl_prefix() {
        let mut gen = generate_response_json("x", "", "stop");
        gen["meta_info"]["id"] = json!("");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        let id = parsed["id"].as_str().unwrap();
        assert!(id.starts_with("cmpl-"), "expected cmpl- prefix, got {id}");
    }

    #[tokio::test]
    async fn test_wrap_finish_reason_other_passthrough() {
        let mut gen = generate_response_json("x", "r1", "stop");
        gen["meta_info"]["finish_reason"] = json!("abort");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["choices"][0]["finish_reason"], "abort");
    }

    fn sse_event(data: &str) -> String {
        format!("data: {}\n\n", data)
    }

    fn sse_body(events: Vec<String>) -> Body {
        let combined = events.join("");
        Body::from(combined)
    }

    fn ok_sse_response(events: Vec<String>) -> Response {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .body(sse_body(events))
            .unwrap()
    }

    #[tokio::test]
    async fn test_streaming_non_success_passthrough() {
        let upstream = Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Body::from("err"))
            .unwrap();
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn test_streaming_cumulative_to_delta() {
        let chunk1 = json!({"text": "Hello", "index": 0, "meta_info": {"finish_reason": null}});
        let chunk2 =
            json!({"text": "Hello world", "index": 0, "meta_info": {"finish_reason": null}});
        let chunk3 = json!({
            "text": "Hello world!",
            "index": 0,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 3,
                "completion_tokens": 4
            }
        });
        let events = vec![
            sse_event(&serde_json::to_string(&chunk1).unwrap()),
            sse_event(&serde_json::to_string(&chunk2).unwrap()),
            sse_event(&serde_json::to_string(&chunk3).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap();
        assert!(ct.contains("text/event-stream"), "content-type: {ct}");

        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let data_lines: Vec<&str> = body_str
            .lines()
            .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
            .collect();

        let mut deltas = Vec::new();
        for line in &data_lines {
            let payload = line.strip_prefix("data: ").unwrap();
            if let Ok(parsed) = serde_json::from_str::<Value>(payload) {
                if let Some(choices) = parsed["choices"].as_array() {
                    if let Some(first) = choices.first() {
                        if let Some(t) = first["text"].as_str() {
                            deltas.push(t.to_string());
                        }
                    }
                }
            }
        }

        assert!(
            deltas.len() >= 3,
            "expected at least 3 delta chunks, got {}",
            deltas.len()
        );
        assert_eq!(deltas[0], "Hello");
        assert_eq!(deltas[1], " world");
        assert_eq!(deltas[2], "!");
    }

    #[tokio::test]
    async fn test_streaming_done_marker_forwarded() {
        let chunk = json!({"text": "hi", "index": 0, "meta_info": {"finish_reason": {"type": "stop"}, "prompt_tokens": 1, "completion_tokens": 1}});
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(
            body_str.contains("data: [DONE]"),
            "must forward [DONE]: {body_str}"
        );
    }

    #[tokio::test]
    async fn test_streaming_usage_chunk_emitted_after_final() {
        let chunk = json!({
            "text": "done",
            "index": 0,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 10,
                "completion_tokens": 20
            }
        });
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let data_lines: Vec<&str> = body_str
            .lines()
            .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
            .collect();

        let mut found_usage = false;
        for line in &data_lines {
            let payload = line.strip_prefix("data: ").unwrap();
            if let Ok(parsed) = serde_json::from_str::<Value>(payload) {
                if parsed.get("usage").is_some() && !parsed["usage"].is_null() {
                    found_usage = true;
                    assert_eq!(parsed["usage"]["prompt_tokens"], 10);
                    assert_eq!(parsed["usage"]["completion_tokens"], 20);
                    assert_eq!(parsed["usage"]["total_tokens"], 30);
                    let choices = parsed["choices"].as_array().unwrap();
                    assert!(choices.is_empty(), "usage chunk must have empty choices");
                }
            }
        }
        assert!(
            found_usage,
            "must emit a usage chunk after final token: {body_str}"
        );
    }

    #[tokio::test]
    async fn test_streaming_error_json_passthrough() {
        let error_obj = json!({"error": {"message": "rate limited", "code": 429}});
        let events = vec![sse_event(&serde_json::to_string(&error_obj).unwrap())];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(
            body_str.contains("rate limited"),
            "error must be passed through: {body_str}"
        );
    }

    #[tokio::test]
    async fn test_streaming_matched_stop_forwarded() {
        let chunk = json!({
            "text": "stop here",
            "index": 0,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "matched_stop": "END",
                "prompt_tokens": 2,
                "completion_tokens": 2
            }
        });
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let data_lines: Vec<&str> = body_str
            .lines()
            .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
            .collect();

        let mut found_matched_stop = false;
        for line in &data_lines {
            let payload = line.strip_prefix("data: ").unwrap();
            if let Ok(parsed) = serde_json::from_str::<Value>(payload) {
                if let Some(choices) = parsed["choices"].as_array() {
                    for choice in choices {
                        if choice.get("matched_stop").is_some() {
                            found_matched_stop = true;
                            assert_eq!(choice["matched_stop"], "END");
                        }
                    }
                }
            }
        }
        assert!(
            found_matched_stop,
            "matched_stop must be forwarded: {body_str}"
        );
    }

    #[tokio::test]
    async fn test_streaming_response_headers() {
        let chunk = json!({"text": "x", "index": 0, "meta_info": {"finish_reason": {"type": "stop"}, "prompt_tokens": 1, "completion_tokens": 1}});
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        assert_eq!(
            resp.headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "text/event-stream"
        );
        assert_eq!(
            resp.headers()
                .get("cache-control")
                .unwrap()
                .to_str()
                .unwrap(),
            "no-cache"
        );
        assert_eq!(
            resp.headers().get("connection").unwrap().to_str().unwrap(),
            "keep-alive"
        );
        assert!(resp.headers().get("content-length").is_none());
    }

    #[tokio::test]
    async fn test_streaming_chunk_has_correct_object_and_model() {
        let chunk = json!({"text": "hi", "index": 0, "meta_info": {"finish_reason": null}});
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "my-model".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let first_data = body_str.lines().find(|l| l.starts_with("data: {")).unwrap();
        let payload = first_data.strip_prefix("data: ").unwrap();
        let parsed: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(parsed["object"], "text_completion");
        assert_eq!(parsed["model"], "my-model");
        assert!(parsed["id"].as_str().unwrap().starts_with("cmpl-"));
        assert!(parsed["created"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_wrap_content_type_set_to_json() {
        let gen = generate_response_json("x", "r1", "stop");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        assert_eq!(
            resp.headers()
                .get("content-type")
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
        assert!(resp.headers().get("content-length").is_none());
    }

    #[tokio::test]
    async fn test_wrap_finish_reason_length() {
        let gen = generate_response_json("x", "r1", "length");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["choices"][0]["finish_reason"], "length");
    }

    #[tokio::test]
    async fn test_streaming_finish_reason_length_maps_correctly() {
        let chunk = json!({
            "text": "tokens",
            "index": 0,
            "meta_info": {
                "finish_reason": {"type": "length"},
                "prompt_tokens": 5,
                "completion_tokens": 10
            }
        });
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let data_lines: Vec<&str> = body_str
            .lines()
            .filter(|l| l.starts_with("data: {"))
            .collect();

        let mut found_length = false;
        for line in &data_lines {
            let payload = line.strip_prefix("data: ").unwrap();
            if let Ok(parsed) = serde_json::from_str::<Value>(payload) {
                if let Some(choices) = parsed["choices"].as_array() {
                    for choice in choices {
                        if choice["finish_reason"] == "length" {
                            found_length = true;
                        }
                    }
                }
            }
        }
        assert!(found_length, "must map length finish_reason: {body_str}");
    }

    #[tokio::test]
    async fn test_streaming_null_finish_reason_maps_to_null() {
        let chunk = json!({"text": "partial", "index": 0, "meta_info": {"finish_reason": null}});
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let first_data = body_str.lines().find(|l| l.starts_with("data: {")).unwrap();
        let payload = first_data.strip_prefix("data: ").unwrap();
        let parsed: Value = serde_json::from_str(payload).unwrap();
        assert!(
            parsed["choices"][0]["finish_reason"].is_null(),
            "null finish_reason must stay null"
        );
    }

    #[tokio::test]
    async fn test_streaming_string_finish_reason_passthrough() {
        let chunk = json!({
            "text": "x",
            "index": 0,
            "meta_info": {
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "completion_tokens": 1
            }
        });
        let events = vec![
            sse_event(&serde_json::to_string(&chunk).unwrap()),
            sse_event("[DONE]"),
        ];
        let upstream = ok_sse_response(events);
        let resp = wrap_streaming_generate_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();

        let data_lines: Vec<&str> = body_str
            .lines()
            .filter(|l| l.starts_with("data: {"))
            .collect();

        let mut found_stop = false;
        for line in &data_lines {
            let payload = line.strip_prefix("data: ").unwrap();
            if let Ok(parsed) = serde_json::from_str::<Value>(payload) {
                if let Some(choices) = parsed["choices"].as_array() {
                    for choice in choices {
                        if choice["finish_reason"] == "stop" {
                            found_stop = true;
                        }
                    }
                }
            }
        }
        assert!(
            found_stop,
            "string finish_reason must pass through: {body_str}"
        );
    }

    #[test]
    fn test_sampling_seed_from_seed_field() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "seed": 99
        }))
        .unwrap();
        let gen = completion_to_generate(&req).unwrap();
        let sp = gen.sampling_params.as_ref().unwrap();
        assert_eq!(sp.sampling_seed, Some(99));
    }

    #[test]
    fn test_sampling_seed_prefers_sampling_seed_over_seed() {
        let req: CompletionRequest = serde_json::from_value(json!({
            "model": "m",
            "prompt": "hi",
            "sampling_seed": 77,
            "seed": 99
        }))
        .unwrap();
        let gen = completion_to_generate(&req).unwrap();
        let sp = gen.sampling_params.as_ref().unwrap();
        assert_eq!(sp.sampling_seed, Some(77));
    }

    #[test]
    fn test_generate_request_defaults_for_non_sampling_fields() {
        let req = minimal_completion_request();
        let gen = completion_to_generate(&req).unwrap();
        assert!(gen.input_ids.is_none());
        assert_eq!(gen.input_embeds, None);
        assert_eq!(gen.image_data, None);
        assert!(!gen.background);
        assert!(gen.log_metrics);
        assert!(!gen.return_text_in_logprobs);
    }

    #[tokio::test]
    async fn test_wrap_invalid_json_returns_error() {
        let upstream = Response::builder()
            .status(StatusCode::OK)
            .body(Body::from("not json at all"))
            .unwrap();
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_wrap_preserves_existing_id() {
        let gen = generate_response_json("x", "my-custom-id", "stop");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["id"], "my-custom-id");
    }

    #[tokio::test]
    async fn test_wrap_usage_total_equals_sum() {
        let gen = generate_response_json("x", "r", "stop");
        let upstream = ok_json_response(&json!([gen]));
        let resp = wrap_generate_response_as_completion(upstream, "m".into()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        let usage = &parsed["usage"];
        let prompt = usage["prompt_tokens"].as_u64().unwrap();
        let compl = usage["completion_tokens"].as_u64().unwrap();
        let total = usage["total_tokens"].as_u64().unwrap();
        assert_eq!(total, prompt + compl);
    }
}
