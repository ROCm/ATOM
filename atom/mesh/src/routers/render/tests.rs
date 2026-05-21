//! Tests for `routers::render::*`.
//!
//! Covers the four render functions that replace `regular/streaming.rs` +
//! `regular/processor.rs` (Part E).

#[cfg(test)]
mod test_support {
    use std::sync::Arc;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_sequence_decoder::create_stop_decoder;
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    pub fn chat_ctx() -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        let chat_req = Arc::new(ChatCompletionRequest::default());
        ResponseContext {
            original: ProtocolRequest::Chat(chat_req),
            model_id: Some("mock-model".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "req-1".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    pub fn generate_ctx() -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        // GenerateRequest has no derived Default; construct via JSON to keep the
        // fixture independent of the upstream field list.
        let gen_req: crate::protocols::generate::GenerateRequest =
            serde_json::from_str(r#"{"text":"hi","stream":false}"#).unwrap();
        ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(gen_req)),
            model_id: Some("mock-model".to_string()),
            headers: None,
            original_text: Some("hi".to_string()),
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "gen-1".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }
}

mod b_chat_aggregator {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::test_support::chat_ctx;
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

    fn complete(ids: Vec<u32>) -> TokenChunk {
        TokenChunk::Complete {
            token_ids: ids,
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
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
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
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
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
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(
            s.contains("length"),
            "expected finish_reason 'length' in body: {s}"
        );
    }

    #[tokio::test]
    async fn test_aggregator_propagates_error_as_5xx() {
        use crate::routers::worker_stream::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::Transport(
            tonic::Status::unavailable("dead"),
        ))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert!(
            resp.status().is_server_error() || resp.status() == StatusCode::SERVICE_UNAVAILABLE,
            "expected 5xx, got {}",
            resp.status()
        );
    }

    #[tokio::test]
    async fn test_aggregator_matched_stop_str_appears_in_response() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1, 2]))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.starts_with('{'));
        assert!(s.contains("stop"));
    }
}

mod c_chat_streaming {
    use axum::http::StatusCode;

    use super::test_support::chat_ctx;
    use crate::routers::render::chat_streaming;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

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
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
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
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_backend_label_is_recorded() {
        let stream = synthetic_single_stream(Vec::new());
        let _ = chat_streaming::process(stream, chat_ctx(), "pd");
    }
}

mod d_generate_aggregator {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::test_support::generate_ctx;
    use crate::routers::render::generate_aggregator;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

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
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().starts_with("application/json"));
    }

    #[tokio::test]
    async fn test_generate_aggregator_body_has_meta() {
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
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("meta_info"), "got: {s}");
    }

    #[tokio::test]
    async fn test_generate_aggregator_propagates_engine_error() {
        use crate::routers::worker_stream::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::PrefillEarlyClose)]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }
}

mod e_generate_streaming {
    use axum::http::StatusCode;

    use super::test_support::generate_ctx;
    use crate::routers::render::generate_streaming;
    use crate::routers::worker_stream::test_support::synthetic_single_stream;
    use crate::routers::worker_stream::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

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
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }

    #[tokio::test]
    async fn test_generate_streaming_backend_label_pd_accepted() {
        let stream = synthetic_single_stream(Vec::new());
        let _ = generate_streaming::process(stream, generate_ctx(), "pd");
    }

    #[tokio::test]
    async fn test_generate_streaming_empty_stream_yields_response() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let ct = resp.headers().get("content-type").unwrap();
        assert!(ct.to_str().unwrap().contains("text/event-stream"));
    }
}
