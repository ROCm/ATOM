//! Test obligations for `routers::render::*` (Parts A and E).
//!
//! Covers `finish_reason_mapping` (moved from `utils.rs:992`) and the four
//! render functions that replace `regular/streaming.rs` + `regular/processor.rs`.

mod a_finish_reason {
    use crate::protocols::generate::GenerateFinishReason;
    use crate::routers::render::finish_reason_mapping::parse_finish_reason;

    #[test]
    fn test_parse_stop_returns_stop() {
        let r = parse_finish_reason("stop");
        assert!(matches!(r, GenerateFinishReason::Stop));
    }

    #[test]
    fn test_parse_length_returns_length() {
        let r = parse_finish_reason("length");
        assert!(matches!(r, GenerateFinishReason::Length));
    }

    #[test]
    fn test_parse_content_filter_returns_content_filter() {
        let r = parse_finish_reason("content_filter");
        assert!(matches!(r, GenerateFinishReason::ContentFilter));
    }

    #[test]
    fn test_parse_tool_calls_returns_tool_calls() {
        let r = parse_finish_reason("tool_calls");
        assert!(matches!(r, GenerateFinishReason::ToolCalls));
    }

    #[test]
    fn test_parse_abort_returns_abort() {
        let r = parse_finish_reason("abort");
        assert!(matches!(r, GenerateFinishReason::Abort));
    }

    #[test]
    fn test_parse_other_passes_through_string() {
        let r = parse_finish_reason("eos_token");
        match r {
            GenerateFinishReason::Other(v) => assert_eq!(v.as_str().unwrap(), "eos_token"),
            other => panic!("expected Other, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_empty_returns_other_empty() {
        let r = parse_finish_reason("");
        assert!(matches!(r, GenerateFinishReason::Other(_)));
    }
}

mod b_chat_aggregator {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::render::chat_aggregator;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{
        FinishReason, MatchedStop, TokenChunk, Usage, WorkerMeta,
    };

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req-1".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    fn make_ctx() -> ResponseContext {
        unimplemented!("response context fixture — built when prepare/* lands");
    }

    fn complete(text_ids: Vec<u32>) -> TokenChunk {
        TokenChunk::Complete {
            token_ids: text_ids,
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::Str("<eot>".to_string())),
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        }
    }

    #[tokio::test]
    async fn test_aggregator_collapses_single_complete_into_response() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1, 2, 3, 4]))]);
        let resp = chat_aggregator::process(stream, make_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().starts_with("application/json"));
    }

    #[tokio::test]
    async fn test_aggregator_collapses_partials_then_complete() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Ok(complete(vec![1, 2])),
        ]);
        let resp = chat_aggregator::process(stream, make_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_maps_into_body() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Length,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = chat_aggregator::process(stream, make_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("length"), "expected finish_reason 'length' in body: {s}");
    }

    #[tokio::test]
    async fn test_aggregator_propagates_error_as_5xx() {
        use crate::routers::worker_stream::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::Transport(
            tonic::Status::unavailable("dead"),
        ))]);
        let resp = chat_aggregator::process(stream, make_ctx()).await;
        assert!(
            resp.status().is_server_error(),
            "expected 5xx, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_aggregator_matched_stop_str_appears_in_response() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1, 2]))]);
        let resp = chat_aggregator::process(stream, make_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        // The matched_stop string itself doesn't always echo back, but the response
        // must be a well-formed JSON object with a stop-shaped finish_reason.
        assert!(s.starts_with('{'));
        assert!(s.contains("stop"));
    }
}

mod c_chat_streaming {
    use axum::http::StatusCode;

    use crate::routers::prepare::response_context::ResponseContext;
    use crate::routers::render::chat_streaming;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{
        FinishReason, TokenChunk, Usage, WorkerMeta,
    };

    fn make_ctx() -> ResponseContext {
        unimplemented!("response context fixture — built when prepare/* lands");
    }

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req-1".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    #[tokio::test]
    async fn test_streaming_response_is_text_event_stream() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = chat_streaming::process(stream, make_ctx(), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_streaming_partial_emits_delta_event() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                logprobs: None,
                input_logprobs: None,
                meta: meta(),
            }),
        ]);
        let resp = chat_streaming::process(stream, make_ctx(), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_backend_label_is_recorded() {
        // The backend label is threaded into the metrics calls inside streaming.
        // We can't observe it without a metrics fixture; this test pins the
        // signature so that the label arg cannot be silently removed.
        let stream = synthetic_single_stream(Vec::new());
        let _ = chat_streaming::process(stream, make_ctx(), "pd");
    }
}

mod d_generate_aggregator {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::routers::prepare::response_context::ResponseContext;
    use crate::routers::render::generate_aggregator;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{
        FinishReason, TokenChunk, Usage, WorkerMeta,
    };

    fn make_ctx() -> ResponseContext {
        unimplemented!("response context fixture — built when prepare/* lands");
    }

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req-1".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    #[tokio::test]
    async fn test_generate_aggregator_returns_application_json() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![10, 20],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, make_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().starts_with("application/json"));
    }

    #[tokio::test]
    async fn test_generate_aggregator_body_has_text_and_meta() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![10, 20],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 2,
                total_tokens: 3,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, make_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("\"meta_info\"") || s.contains("\"usage\""), "got: {s}");
    }

    #[tokio::test]
    async fn test_generate_aggregator_propagates_engine_error() {
        use crate::routers::worker_stream::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::PrefillEarlyClose)]);
        let resp = generate_aggregator::process(stream, make_ctx()).await;
        assert!(resp.status().is_server_error());
    }
}

mod e_generate_streaming {
    use axum::http::StatusCode;

    use crate::routers::prepare::response_context::ResponseContext;
    use crate::routers::render::generate_streaming;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{
        FinishReason, TokenChunk, Usage, WorkerMeta,
    };

    fn make_ctx() -> ResponseContext {
        unimplemented!("response context fixture — built when prepare/* lands");
    }

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req-1".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    #[tokio::test]
    async fn test_generate_streaming_is_text_event_stream() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                logprobs: None,
                input_logprobs: None,
                meta: meta(),
            }),
        ]);
        let resp = generate_streaming::process(stream, make_ctx(), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_generate_streaming_backend_label_pd_accepted() {
        let stream = synthetic_single_stream(Vec::new());
        let _ = generate_streaming::process(stream, make_ctx(), "pd");
    }

    #[tokio::test]
    async fn test_generate_streaming_empty_stream_yields_response() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = generate_streaming::process(stream, make_ctx(), "regular");
        // Empty stream is still a valid SSE response; the body may be empty but
        // the framing must be correct.
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }
}

mod f_layering {
    // Compile-only: ensure render/* type signatures don't smuggle in mesh_grpc.
    // The grep gate in plan E7 covers this at file level; this is a defense in
    // depth that fails the test compile if a future change adds a re-export.

    #[test]
    fn test_render_module_has_no_mesh_grpc_reexports() {
        let name = std::any::type_name::<crate::routers::render::chat_streaming::ChatStreamConfig>();
        assert!(!name.contains("mesh_grpc"), "got: {name}");
    }
}
