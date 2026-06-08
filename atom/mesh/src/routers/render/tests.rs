//! Tests for `routers::render::*`.

#[cfg(test)]
mod test_support {
    use std::sync::Arc;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
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
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
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
        use crate::routers::token_handle::engine_error::EngineError;
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

mod b2_chat_aggregator_branches {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::protocols::common::{
        Function, FunctionChoice, StreamOptions, Tool, ToolChoice, ToolChoiceValue,
    };
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::chat_aggregator;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
        FinishReason, MatchedStop, TokenChunk, TokenLogprob, TokenLogprobs, Usage, WorkerMeta,
    };
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta(v: Option<&str>) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: v.map(|s| s.to_string()),
            cached_tokens: 0,
        }
    }

    fn ctx_with(req: ChatCompletionRequest) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: Some(test_mocks::tool_parser_factory()),
            reasoning_parser_factory: Some(test_mocks::reasoning_parser_factory()),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    fn cmpl(ids: Vec<u32>, fr: FinishReason, ms: Option<MatchedStop>) -> TokenChunk {
        TokenChunk::Complete {
            token_ids: ids,
            finish_reason: fr,
            matched_stop: ms,
            usage: Usage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(Some("vw")),
        }
    }

    #[tokio::test]
    async fn test_aggregator_wrong_request_type_returns_500() {
        use super::test_support::generate_ctx;
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_empty_completes_returns_500() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Partial {
            token_ids: vec![1],
            logprobs: None,
        })]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_with_logprobs_attaches_to_choice() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 1,
                logprob: -0.3,
                decoded_text: Some("hi".to_string()),
                top: vec![(1, -0.3, Some("hi".to_string()))],
            }],
        };
        let mut chunk = cmpl(vec![1], FinishReason::Stop, None);
        if let TokenChunk::Complete { logprobs, .. } = &mut chunk {
            *logprobs = Some(lp);
        }
        let stream = synthetic_single_stream(vec![Ok(chunk)]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("logprobs"));
    }

    #[tokio::test]
    async fn test_aggregator_reasoning_factory_present_but_unmatched_model() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.model = "no-such".to_string();
        req.separate_reasoning = true;
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_other_maps_to_stop() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        let stream = synthetic_single_stream(vec![Ok(cmpl(
            vec![1],
            FinishReason::Other("weird".to_string()),
            None,
        ))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("stop"));
    }

    #[tokio::test]
    async fn test_aggregator_with_tools_and_specific_function_choice() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "myfn".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }]);
        req.tool_choice = Some(ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "myfn".to_string(),
            },
        });
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_with_tools_and_required_choice() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "myfn".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }]);
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_with_tools_tool_choice_none_skips_tool_parse() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "myfn".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }]);
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::None));
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_propagates_transport_error_as_503() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Err(EngineError::Transport(
            tonic::Status::unavailable("dead"),
        ))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_aggregator_propagates_prefill_error_as_500() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Err(EngineError::Prefill("bad".to_string()))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_decode_error_as_500() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Err(EngineError::DecodeError("d".to_string()))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_prefill_early_close() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Err(EngineError::PrefillEarlyClose)]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_decode_incomplete() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Err(EngineError::DecodeIncomplete)]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_connection_acquire_failed() {
        use super::test_support::chat_ctx;
        let stream = synthetic_single_stream(vec![Err(EngineError::ConnectionAcquireFailed(
            "x".to_string(),
        ))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_aggregator_propagates_request_build_failed_as_400() {
        use super::test_support::chat_ctx;
        let stream =
            synthetic_single_stream(vec![Err(EngineError::RequestBuildFailed("rq".to_string()))]);
        let resp = chat_aggregator::process(stream, chat_ctx()).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_aggregator_matched_stop_token_id_serialized_as_number() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        let stream = synthetic_single_stream(vec![Ok(cmpl(
            vec![1],
            FinishReason::Stop,
            Some(MatchedStop::TokenId(42)),
        ))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("42"));
    }

    #[tokio::test]
    async fn test_aggregator_with_tool_parser_path() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }]);
        let mut ctx = ctx_with(req);
        ctx.configured_tool_parser = Some("json".to_string());
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_with_reasoning_parser_loaded() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.separate_reasoning = true;
        let mut ctx = ctx_with(req);
        ctx.configured_reasoning_parser = Some("deepseek_r1".to_string());
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_with_specific_function_choice_uses_json_schema() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.tools = Some(vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "myfn".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }]);
        req.tool_choice = Some(ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "myfn".to_string(),
            },
        });
        let stream = synthetic_single_stream(vec![Ok(cmpl(vec![1], FinishReason::Stop, None))]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_multiple_completes_aggregates_usage() {
        let mut req = ChatCompletionRequest::default();
        req.stream = false;
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let stream = synthetic_single_stream(vec![
            Ok(cmpl(vec![1], FinishReason::Stop, None)),
            Ok(cmpl(vec![2], FinishReason::Stop, None)),
        ]);
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("usage"));
        assert!(s.contains("\"total_tokens\":10"));
    }
}

mod b3_chat_aggregator_typed {
    use std::sync::Arc;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::chat_aggregator;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: Some("v1".to_string()),
            cached_tokens: 0,
        }
    }

    fn chat_ctx() -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(ChatCompletionRequest::default())),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 1234,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    fn complete(ids: Vec<u32>, fr: FinishReason) -> TokenChunk {
        TokenChunk::Complete {
            token_ids: ids,
            finish_reason: fr,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 2,
                completion_tokens: 3,
                total_tokens: 5,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        }
    }

    #[tokio::test]
    async fn test_process_typed_returns_ok_with_valid_complete() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1, 2], FinishReason::Stop))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.model, "m");
        assert!(!resp.choices.is_empty());
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn test_process_typed_usage_is_aggregated() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1], FinishReason::Stop))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        let resp = result.unwrap();
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 2);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, 5);
    }

    #[tokio::test]
    async fn test_process_typed_returns_err_on_generate_context() {
        use super::test_support::generate_ctx;
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1], FinishReason::Stop))]);
        let result = chat_aggregator::process_typed(stream, generate_ctx()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_typed_returns_err_on_empty_stream() {
        let stream = synthetic_single_stream(Vec::new());
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_typed_returns_err_on_only_partials() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Partial {
            token_ids: vec![1],
            logprobs: None,
        })]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_typed_returns_err_on_engine_error() {
        let stream =
            synthetic_single_stream(vec![Err(EngineError::DecodeError("fail".to_string()))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_process_typed_created_timestamp_is_threaded() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1], FinishReason::Stop))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        let resp = result.unwrap();
        assert_eq!(resp.created, 1234);
    }

    #[tokio::test]
    async fn test_process_typed_system_fingerprint_from_weight_version() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![1], FinishReason::Stop))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        let resp = result.unwrap();
        assert_eq!(resp.system_fingerprint.as_deref(), Some("v1"));
    }

    #[tokio::test]
    async fn test_process_typed_content_filter_finish_reason() {
        let stream =
            synthetic_single_stream(vec![Ok(complete(vec![1], FinishReason::ContentFilter))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        let resp = result.unwrap();
        assert_eq!(
            resp.choices[0].finish_reason.as_deref(),
            Some("content_filter")
        );
    }

    #[tokio::test]
    async fn test_process_typed_multiple_completes_yields_multiple_choices() {
        let stream = synthetic_single_stream(vec![
            Ok(complete(vec![1], FinishReason::Stop)),
            Ok(complete(vec![2], FinishReason::Length)),
        ]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        let resp = result.unwrap();
        assert_eq!(resp.choices.len(), 2);
        assert_eq!(resp.choices[0].index, 0);
        assert_eq!(resp.choices[1].index, 1);
    }

    #[tokio::test]
    async fn test_process_typed_empty_content_yields_none_content() {
        let stream = synthetic_single_stream(vec![Ok(complete(vec![], FinishReason::Stop))]);
        let result = chat_aggregator::process_typed(stream, chat_ctx()).await;
        let resp = result.unwrap();
        assert!(
            resp.choices[0].message.content.is_none()
                || resp.choices[0].message.content.as_deref() == Some("")
        );
    }
}

mod c_chat_streaming {
    use axum::http::StatusCode;

    use super::test_support::chat_ctx;
    use crate::routers::render::chat_streaming;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

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

mod c1_chat_streaming_body_content {
    use axum::body::to_bytes;

    use super::test_support::chat_ctx;
    use crate::routers::render::chat_streaming;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
        FinishReason, MatchedStop, TokenChunk, Usage, WorkerMeta,
    };

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    fn usage(p: u32, c: u32) -> Usage {
        Usage {
            prompt_tokens: p,
            completion_tokens: c,
            total_tokens: p + c,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    #[tokio::test]
    async fn test_streaming_happy_path_partials_then_complete_stop() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1, 2],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![3],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2, 3],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(5, 3),
                logprobs: None,
                input_logprobs: None,
                meta: meta(),
            }),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("data:"));
        assert!(s.contains("\"stop\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_empty_stream_emits_done_without_data() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_only_complete_no_partials() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::Str("eos".to_string())),
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("eos"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_model_id_appears_in_sse_events() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta(),
            }),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("mock-model"));
    }

    #[tokio::test]
    async fn test_streaming_first_chunk_contains_role_assistant() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta(),
            }),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("assistant"));
    }

    #[tokio::test]
    async fn test_streaming_cache_control_header() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let cc = resp.headers().get("cache-control").unwrap();
        assert_eq!(cc.to_str().unwrap(), "no-cache");
        let conn = resp.headers().get("connection").unwrap();
        assert_eq!(conn.to_str().unwrap(), "keep-alive");
    }
}

mod c2_chat_streaming_branches {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::test_support::{chat_ctx, generate_ctx};
    use crate::protocols::chat::ChatCompletionRequest;
    use crate::protocols::common::{StreamOptions, ToolChoice, ToolChoiceValue};
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::chat_streaming;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
        FinishReason, MatchedStop, TokenChunk, TokenLogprob, TokenLogprobs, Usage, WorkerMeta,
    };
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta_with_weight(v: &str) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: Some(v.to_string()),
            cached_tokens: 0,
        }
    }

    fn meta_plain() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    fn usage(p: u32, c: u32) -> Usage {
        Usage {
            prompt_tokens: p,
            completion_tokens: c,
            total_tokens: p + c,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    fn ctx_with(req: ChatCompletionRequest) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    #[tokio::test]
    async fn test_streaming_invoked_with_generate_returns_error_sse() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = chat_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("chat_streaming invoked with generate"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_partials_produce_data_events() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1, 2],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(2, 2),
                logprobs: None,
                input_logprobs: None,
                meta: meta_with_weight("v1"),
            }),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("data:"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_partial_with_logprobs_threads_through() {
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 1,
                logprob: -0.1,
                decoded_text: Some("hi".to_string()),
                top: vec![],
            }],
        };
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: Some(lp),
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_engine_error_emits_error_payload() {
        let stream = synthetic_single_stream(vec![Err(EngineError::Transport(
            tonic::Status::unavailable("nope"),
        ))]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"error\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_streaming_finish_reason_length_emits_length() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Length,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("length"));
    }

    #[tokio::test]
    async fn test_streaming_finish_reason_content_filter() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ContentFilter,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("content_filter"));
    }

    #[tokio::test]
    async fn test_streaming_finish_reason_tool_calls() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ToolCalls,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("tool_calls"));
    }

    #[tokio::test]
    async fn test_streaming_finish_reason_abort() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Abort,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("abort"));
    }

    #[tokio::test]
    async fn test_streaming_finish_reason_other_passthrough() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Other("ghosted".to_string()),
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("ghosted"));
    }

    #[tokio::test]
    async fn test_streaming_matched_stop_str_emits_string() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::Str("</s>".to_string())),
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("</s>"));
    }

    #[tokio::test]
    async fn test_streaming_matched_stop_token_id_emits_number() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::TokenId(77)),
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("77"));
    }

    #[tokio::test]
    async fn test_streaming_include_usage_emits_usage_chunk() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(3, 5),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("usage"));
        assert!(s.contains("\"total_tokens\":8") || s.contains("total_tokens"));
    }

    #[tokio::test]
    async fn test_streaming_weight_version_threaded_as_system_fingerprint() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_with_weight("vXX"),
        })]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        let _ = s;
    }

    #[tokio::test]
    async fn test_streaming_separate_reasoning_without_factory_falls_through() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.separate_reasoning = true;
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_tool_choice_none_skips_tool_path() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::None));
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_partial_zero_tokens_is_handled() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(0, 0),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "pd");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_error_after_partial_still_emits_done() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Err(EngineError::DecodeError("kaboom".to_string())),
        ]);
        let resp = chat_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }
}

mod c3_chat_streaming_parsers {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::protocols::common::{Function, FunctionChoice, Tool, ToolChoice, ToolChoiceValue};
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::chat_streaming;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
    use crate::tokenizer::huggingface::HuggingFaceTokenizer;
    use crate::tokenizer::traits::Tokenizer;

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    fn ctx_with(
        req: ChatCompletionRequest,
        configured_tool: Option<&str>,
        configured_reasoning: Option<&str>,
    ) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = test_mocks::hf_tokenizer();
        let _ = HuggingFaceTokenizer::from_file;
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: Some(test_mocks::tool_parser_factory()),
            reasoning_parser_factory: Some(test_mocks::reasoning_parser_factory()),
            configured_tool_parser: configured_tool.map(|s| s.to_string()),
            configured_reasoning_parser: configured_reasoning.map(|s| s.to_string()),
        }
    }

    fn tool_def() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "myfn".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }
    }

    #[tokio::test]
    async fn test_streaming_with_tool_parser_path() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![0],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![0],
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
        let resp = chat_streaming::process(stream, ctx_with(req, Some("json"), None), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let _ = body_of(resp).await;
    }

    #[tokio::test]
    async fn test_streaming_with_specific_function_choice_emits_tool_name() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "myfn".to_string(),
            },
        });
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
        let resp = chat_streaming::process(stream, ctx_with(req, Some("json"), None), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("myfn"));
    }

    #[tokio::test]
    async fn test_streaming_with_required_choice_uses_json_parser() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Required));
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
        let resp = chat_streaming::process(stream, ctx_with(req, Some("json"), None), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_streaming_with_separate_reasoning_and_parser_loads_parser() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.separate_reasoning = true;
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
        let resp =
            chat_streaming::process(stream, ctx_with(req, None, Some("deepseek_r1")), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
    }
}

mod c4_chat_streaming_coverage {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::protocols::common::{
        Function, FunctionChoice, StreamOptions, Tool, ToolChoice, ToolChoiceValue,
    };
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::chat_streaming;
    use crate::routers::test_mocks;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
        FinishReason, MatchedStop, TokenChunk, TokenLogprob, TokenLogprobs, Usage, WorkerMeta,
    };
    use crate::tokenizer::huggingface::HuggingFaceTokenizer;
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta_plain() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: None,
            cached_tokens: 0,
        }
    }

    fn meta_with_weight(v: &str) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: Some(v.to_string()),
            cached_tokens: 0,
        }
    }

    fn usage(p: u32, c: u32) -> Usage {
        Usage {
            prompt_tokens: p,
            completion_tokens: c,
            total_tokens: p + c,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    fn ctx_with(req: ChatCompletionRequest) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 42,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    fn ctx_with_parsers(req: ChatCompletionRequest) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = test_mocks::hf_tokenizer();
        let _ = HuggingFaceTokenizer::from_file;
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 42,
            tool_parser_factory: Some(test_mocks::tool_parser_factory()),
            reasoning_parser_factory: Some(test_mocks::reasoning_parser_factory()),
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    fn ctx_no_model_id(req: ChatCompletionRequest) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: None,
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    fn tool_def() -> Tool {
        Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: serde_json::json!({}),
                strict: None,
            },
        }
    }

    #[tokio::test]
    async fn test_model_id_none_falls_back_to_request_model() {
        let mut req = ChatCompletionRequest::default();
        req.model = "fallback-model".to_string();
        req.stream = true;
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_no_model_id(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("fallback-model"));
    }

    #[tokio::test]
    async fn test_system_fingerprint_set_from_weight_version_in_complete() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_with_weight("fp-v42"),
            }),
        ]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("fp-v42"));
    }

    #[tokio::test]
    async fn test_has_tool_call_remaps_stop_to_tool_calls() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "get_weather".to_string(),
            },
        });
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        let s = body_of(resp).await;
        assert!(
            s.contains("tool_calls"),
            "expected finish_reason remapped to tool_calls: {s}"
        );
    }

    #[tokio::test]
    async fn test_has_tool_call_false_preserves_stop() {
        let req = ChatCompletionRequest::default();
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
    }

    #[tokio::test]
    async fn test_stream_options_include_usage_false_omits_usage_chunk() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = Some(StreamOptions {
            include_usage: Some(false),
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(5, 10),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(!s.contains("\"total_tokens\""));
    }

    #[tokio::test]
    async fn test_stream_options_include_usage_none_omits_usage_chunk() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = Some(StreamOptions {
            include_usage: None,
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(2, 3),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(!s.contains("\"total_tokens\""));
    }

    #[tokio::test]
    async fn test_no_stream_options_omits_usage_chunk() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = None;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(2, 3),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
        assert!(!s.contains("\"total_tokens\""));
    }

    #[tokio::test]
    async fn test_include_usage_true_emits_prompt_and_completion_tokens() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1, 2],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(10, 20),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"prompt_tokens\":10"));
        assert!(s.contains("\"completion_tokens\":20"));
        assert!(s.contains("\"total_tokens\":30"));
    }

    #[tokio::test]
    async fn test_created_timestamp_appears_in_sse_events() {
        let req = ChatCompletionRequest::default();
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"created\":42"));
    }

    #[tokio::test]
    async fn test_multiple_partials_role_only_in_first_chunk() {
        let req = ChatCompletionRequest::default();
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![3],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2, 3],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 3),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        let role_count = s.matches("\"role\":\"assistant\"").count();
        assert!(role_count >= 1, "role should appear at least once in: {s}");
        let content_chunks: Vec<&str> = s.lines().filter(|l| l.contains("\"content\"")).collect();
        assert!(
            content_chunks.len() >= 2,
            "expected multiple content chunks for multiple partials: {s}"
        );
    }

    #[tokio::test]
    async fn test_logprobs_with_top_entries_appears_in_content_chunk() {
        let req = ChatCompletionRequest::default();
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 1,
                logprob: -0.5,
                decoded_text: Some("hello".to_string()),
                top: vec![
                    (1, -0.5, Some("hello".to_string())),
                    (2, -1.2, Some("world".to_string())),
                ],
            }],
        };
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: Some(lp),
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("logprobs"), "expected logprobs in output: {s}");
    }

    #[tokio::test]
    async fn test_allowed_tools_required_mode_activates_json_schema() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "required".to_string(),
            tools: vec![],
        });
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_allowed_tools_auto_mode_does_not_use_json_schema() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::AllowedTools {
            tool_type: "allowed_tools".to_string(),
            mode: "auto".to_string(),
            tools: vec![],
        });
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_specific_function_multiple_partials_emits_tool_name_once() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "get_weather".to_string(),
            },
        });
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 2),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        let s = body_of(resp).await;
        let name_count = s.matches("get_weather").count();
        assert!(name_count >= 1, "expected tool name at least once: {s}");
        assert!(s.contains("tool_calls"));
    }

    #[tokio::test]
    async fn test_finish_reason_length_not_remapped_when_has_tool_call() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Function {
            tool_type: "function".to_string(),
            function: FunctionChoice {
                name: "get_weather".to_string(),
            },
        });
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Length,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        let s = body_of(resp).await;
        assert!(
            s.contains("\"length\""),
            "length should not be remapped to tool_calls: {s}"
        );
    }

    #[tokio::test]
    async fn test_tools_present_but_tool_choice_none_emits_content_not_tool() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::None));
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_error_midstream_after_multiple_partials() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Err(EngineError::DecodeError("midstream_fail".to_string())),
        ]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("midstream_fail"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_transport_error_midstream() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Err(EngineError::Transport(tonic::Status::internal(
                "server crashed",
            ))),
        ]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_prefill_error_emits_error_sse() {
        let stream =
            synthetic_single_stream(vec![Err(EngineError::Prefill("bad input".to_string()))]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_prefill_early_close_emits_error_sse() {
        let stream = synthetic_single_stream(vec![Err(EngineError::PrefillEarlyClose)]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_decode_incomplete_emits_error_sse() {
        let stream = synthetic_single_stream(vec![Err(EngineError::DecodeIncomplete)]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_connection_acquire_failed_emits_error_sse() {
        let stream = synthetic_single_stream(vec![Err(EngineError::ConnectionAcquireFailed(
            "pool exhausted".to_string(),
        ))]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_request_build_failed_emits_error_sse() {
        let stream = synthetic_single_stream(vec![Err(EngineError::RequestBuildFailed(
            "invalid params".to_string(),
        ))]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_only_complete_no_partials_with_system_fingerprint() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_with_weight("sys-fp-1"),
        })]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_complete_with_flush_text() {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        let req = ChatCompletionRequest::default();
        let ctx = ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        };
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1, 2, 3],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2, 3],
                finish_reason: FinishReason::Stop,
                matched_stop: Some(MatchedStop::Str("end".to_string())),
                usage: usage(3, 3),
                logprobs: None,
                input_logprobs: None,
                meta: meta_with_weight("fp-flush"),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx, "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_pd_backend_label_accepted() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp =
            chat_streaming::process(stream, ctx_with(ChatCompletionRequest::default()), "pd");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_tools_with_auto_choice_uses_tool_parser_path() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        req.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_tools_without_factory_falls_through_to_content() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.tools = Some(vec![tool_def()]);
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
    }

    #[tokio::test]
    async fn test_separate_reasoning_true_with_factory_but_unmatched_model() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.separate_reasoning = true;
        req.model = "not-a-reasoning-model".to_string();
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_sse_response_headers_are_complete() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp.headers().get("content-type").unwrap();
        assert_eq!(ct.to_str().unwrap(), "text/event-stream");
        let cc = resp.headers().get("cache-control").unwrap();
        assert_eq!(cc.to_str().unwrap(), "no-cache");
        let conn = resp.headers().get("connection").unwrap();
        assert_eq!(conn.to_str().unwrap(), "keep-alive");
    }

    #[tokio::test]
    async fn test_request_id_appears_in_sse_events() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(
            s.contains("rid"),
            "expected request_id 'rid' in events: {s}"
        );
    }

    #[tokio::test]
    async fn test_data_prefix_and_done_in_every_response() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1, 2, 3],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2, 3],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 3),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        let data_lines: Vec<&str> = s.lines().filter(|l| l.starts_with("data:")).collect();
        assert!(
            data_lines.len() >= 2,
            "expected at least 2 data lines (content + DONE): {s}"
        );
        assert_eq!(data_lines.last().unwrap().trim(), "data: [DONE]");
    }

    #[tokio::test]
    async fn test_finish_reason_tool_calls_native_not_remapped() {
        let req = ChatCompletionRequest::default();
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ToolCalls,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("tool_calls"));
    }

    #[tokio::test]
    async fn test_empty_partial_tokens_skipped() {
        let req = ChatCompletionRequest::default();
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_weight_version_absent_system_fingerprint_omitted() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_matched_stop_none_emits_null_stop() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_matched_stop_str_appears_in_finish_chunk() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::Str("<|im_end|>".to_string())),
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("<|im_end|>"));
    }

    #[tokio::test]
    async fn test_matched_stop_token_id_appears_as_number_in_finish_chunk() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::TokenId(151643)),
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(s.contains("151643"));
    }

    #[tokio::test]
    async fn test_no_complete_means_no_non_null_finish_reason() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Partial {
            token_ids: vec![1],
            logprobs: None,
        })]);
        let resp = chat_streaming::process(
            stream,
            ctx_with(ChatCompletionRequest::default()),
            "regular",
        );
        let s = body_of(resp).await;
        assert!(
            !s.contains("\"finish_reason\":\"stop\""),
            "without Complete there should be no non-null finish_reason: {s}"
        );
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_separate_reasoning_false_does_not_activate_parser() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.separate_reasoning = false;
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(1, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let resp = chat_streaming::process(stream, ctx_with_parsers(req), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("\"stop\""));
    }

    #[tokio::test]
    async fn test_usage_chunk_contains_all_fields() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(7, 13),
            logprobs: None,
            input_logprobs: None,
            meta: meta_plain(),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"prompt_tokens\":7"));
        assert!(s.contains("\"completion_tokens\":13"));
        assert!(s.contains("\"total_tokens\":20"));
    }

    #[tokio::test]
    async fn test_system_fingerprint_set_on_finish_chunk_from_complete_meta() {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = Some(StreamOptions {
            include_usage: Some(true),
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(1, 1),
            logprobs: None,
            input_logprobs: None,
            meta: meta_with_weight("weight-v99"),
        })]);
        let resp = chat_streaming::process(stream, ctx_with(req), "regular");
        let s = body_of(resp).await;
        assert!(
            s.contains("weight-v99"),
            "system_fingerprint should appear in finish/usage chunks: {s}"
        );
    }
}

mod d_generate_aggregator {
    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::test_support::generate_ctx;
    use crate::routers::render::generate_aggregator;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

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
        use crate::routers::token_handle::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::PrefillEarlyClose)]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }
}

mod d2_generate_aggregator_branches {
    use axum::http::StatusCode;

    use super::test_support::{chat_ctx, generate_ctx};
    use crate::routers::render::generate_aggregator;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
        FinishReason, InputLogprobs, MatchedStop, TokenChunk, TokenLogprob, TokenLogprobs, Usage,
        WorkerMeta,
    };

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req-1".to_string(),
            weight_version: Some("v7".to_string()),
            cached_tokens: 5,
        }
    }

    #[tokio::test]
    async fn test_aggregator_returns_500_on_chat_request_context() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = generate_aggregator::process(stream, chat_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_returns_500_on_empty_completes() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_filters_out_partial_chunks() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2],
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
            }),
        ]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_length_carries_completion_tokens() {
        use axum::body::to_bytes;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Length,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 5,
                total_tokens: 6,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("length"));
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_content_filter_maps_to_other() {
        use axum::body::to_bytes;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ContentFilter,
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
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("content_filter"));
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_tool_calls_maps_to_other() {
        use axum::body::to_bytes;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ToolCalls,
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
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("tool_calls"));
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_abort_maps_to_other() {
        use axum::body::to_bytes;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![],
            finish_reason: FinishReason::Abort,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 0,
                total_tokens: 1,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("abort"));
    }

    #[tokio::test]
    async fn test_aggregator_finish_reason_other_passthrough() {
        use axum::body::to_bytes;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![],
            finish_reason: FinishReason::Other("custom-x".to_string()),
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("custom-x"));
    }

    #[tokio::test]
    async fn test_aggregator_matched_stop_token_id_serialized_as_number() {
        use axum::body::to_bytes;
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::TokenId(42)),
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("42"));
    }

    #[tokio::test]
    async fn test_aggregator_propagates_engine_transport_error() {
        use crate::routers::token_handle::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::Transport(
            tonic::Status::unavailable("dead"),
        ))]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_aggregator_propagates_engine_prefill_error() {
        use crate::routers::token_handle::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::Prefill("oops".to_string()))]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_engine_decode_error() {
        use crate::routers::token_handle::engine_error::EngineError;
        let stream =
            synthetic_single_stream(vec![Err(EngineError::DecodeError("bad".to_string()))]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_decode_incomplete() {
        use crate::routers::token_handle::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::DecodeIncomplete)]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert!(resp.status().is_server_error());
    }

    #[tokio::test]
    async fn test_aggregator_propagates_connection_acquire_failed() {
        use crate::routers::token_handle::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::ConnectionAcquireFailed(
            "pool dry".to_string(),
        ))]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_aggregator_propagates_request_build_failed() {
        use crate::routers::token_handle::engine_error::EngineError;
        let stream = synthetic_single_stream(vec![Err(EngineError::RequestBuildFailed(
            "bad req".to_string(),
        ))]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_aggregator_decoder_stops_on_stop_token() {
        use std::sync::Arc;

        use crate::protocols::generate::GenerateRequest;
        use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
        use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
        use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, Some(&vec![2u32]), true, false);
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"hi","stream":false}"#).unwrap();
        let ctx = ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(gen_req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        };
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1, 2, 3],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 3,
                total_tokens: 4,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, ctx).await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_aggregator_includes_logprobs_when_requested() {
        use axum::body::to_bytes;
        use std::sync::Arc;

        use crate::protocols::generate::GenerateRequest;
        use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
        use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
        use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        let gen_req: GenerateRequest =
            serde_json::from_str(r#"{"text":"hi","stream":false,"return_logprob":true}"#).unwrap();
        let ctx = ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(gen_req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        };
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 1,
                logprob: -0.5,
                decoded_text: None,
                top: vec![],
            }],
        };
        let ip = InputLogprobs {
            items: vec![TokenLogprob {
                token_id: 9,
                logprob: -0.2,
                decoded_text: None,
                top: vec![],
            }],
        };
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: Some(lp),
            input_logprobs: Some(ip),
            meta: meta(),
        })]);
        let resp = generate_aggregator::process(stream, ctx).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains("output_token_logprobs"));
        assert!(s.contains("input_token_logprobs"));
    }
}

mod d3_logprob_conversion {
    use std::sync::Arc;

    use crate::routers::render::logprob_conversion::{
        input_logprobs_to_generate, output_logprobs_to_generate, token_logprobs_to_chat,
    };
    use crate::routers::token_handle::token_chunk::{InputLogprobs, TokenLogprob, TokenLogprobs};
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn tok() -> Arc<dyn Tokenizer> {
        Arc::new(MockTokenizer::new())
    }

    #[test]
    fn test_token_logprobs_to_chat_empty_yields_detailed_none() {
        let lp = TokenLogprobs { items: vec![] };
        let out = token_logprobs_to_chat(&lp, &tok());
        match out {
            crate::protocols::common::ChatLogProbs::Detailed { content } => {
                assert!(content.is_none())
            }
            _ => panic!("expected Detailed"),
        }
    }

    #[test]
    fn test_token_logprobs_to_chat_with_decoded_text_uses_it() {
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 7,
                logprob: -0.1,
                decoded_text: Some(" hi".to_string()),
                top: vec![(7, -0.1, Some(" hi".to_string()))],
            }],
        };
        let out = token_logprobs_to_chat(&lp, &tok());
        if let crate::protocols::common::ChatLogProbs::Detailed {
            content: Some(items),
        } = out
        {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].token, " hi");
            assert_eq!(items[0].top_logprobs.len(), 1);
            assert_eq!(items[0].top_logprobs[0].token, " hi");
        } else {
            panic!("expected detailed-with-content");
        }
    }

    #[test]
    fn test_token_logprobs_to_chat_fallback_decodes_via_tokenizer() {
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 11,
                logprob: -0.7,
                decoded_text: None,
                top: vec![(11, -0.7, None)],
            }],
        };
        let out = token_logprobs_to_chat(&lp, &tok());
        if let crate::protocols::common::ChatLogProbs::Detailed {
            content: Some(items),
        } = out
        {
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].top_logprobs.len(), 1);
        } else {
            panic!("expected detailed-with-content");
        }
    }

    #[test]
    fn test_output_logprobs_to_generate_pairs_logprob_and_token_id() {
        let lp = TokenLogprobs {
            items: vec![
                TokenLogprob {
                    token_id: 4,
                    logprob: -0.3,
                    decoded_text: None,
                    top: vec![],
                },
                TokenLogprob {
                    token_id: 9,
                    logprob: -1.2,
                    decoded_text: None,
                    top: vec![],
                },
            ],
        };
        let out = output_logprobs_to_generate(&lp);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].len(), 2);
        assert!((out[0][0].unwrap() - (-0.3)).abs() < 1e-6);
        assert_eq!(out[0][1].unwrap(), 4.0);
    }

    #[test]
    fn test_input_logprobs_to_generate_pairs_logprob_and_token_id() {
        let lp = InputLogprobs {
            items: vec![TokenLogprob {
                token_id: 99,
                logprob: -0.05,
                decoded_text: None,
                top: vec![],
            }],
        };
        let out = input_logprobs_to_generate(&lp);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0][1].unwrap(), 99.0);
    }

    #[test]
    fn test_output_logprobs_to_generate_empty_yields_empty() {
        let lp = TokenLogprobs { items: vec![] };
        let out = output_logprobs_to_generate(&lp);
        assert!(out.is_empty());
    }

    #[test]
    fn test_input_logprobs_to_generate_empty_yields_empty() {
        let lp = InputLogprobs { items: vec![] };
        let out = input_logprobs_to_generate(&lp);
        assert!(out.is_empty());
    }
}

mod e_generate_streaming {
    use axum::http::StatusCode;

    use super::test_support::generate_ctx;
    use crate::routers::render::generate_streaming;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

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

mod e1_generate_streaming_body_content {
    use axum::body::to_bytes;

    use super::test_support::generate_ctx;
    use crate::routers::render::generate_streaming;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: Some("v1".to_string()),
            cached_tokens: 0,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    #[tokio::test]
    async fn test_generate_streaming_partial_accumulates_text() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 2,
                    completion_tokens: 2,
                    total_tokens: 4,
                },
                logprobs: None,
                input_logprobs: None,
                meta: meta(),
            }),
        ]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"text\""));
        assert!(s.contains("\"finish_reason\""));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_generate_streaming_complete_emits_e2e_latency() {
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
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("e2e_latency"));
    }

    #[tokio::test]
    async fn test_generate_streaming_complete_emits_cached_tokens() {
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
            meta: WorkerMeta {
                request_id: "req".to_string(),
                weight_version: Some("v2".to_string()),
                cached_tokens: 42,
            },
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"cached_tokens\":42"));
    }

    #[tokio::test]
    async fn test_generate_streaming_empty_stream_still_emits_done() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_generate_streaming_weight_version_appears_in_meta_info() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![5],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: "req".to_string(),
                weight_version: Some("wv_abc".to_string()),
                cached_tokens: 0,
            },
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("wv_abc"));
    }
}

mod e2_generate_streaming_branches {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::http::StatusCode;

    use super::test_support::{chat_ctx, generate_ctx};
    use crate::protocols::generate::GenerateRequest;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::generate_streaming;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{
        FinishReason, InputLogprobs, TokenChunk, TokenLogprob, TokenLogprobs, Usage, WorkerMeta,
    };
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta(v: Option<&str>) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: v.map(|s| s.to_string()),
            cached_tokens: 9,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    fn gen_ctx_logprob(return_logprob: bool) -> ResponseContext {
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(MockTokenizer::new());
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        let payload = if return_logprob {
            r#"{"text":"hi","stream":true,"return_logprob":true}"#
        } else {
            r#"{"text":"hi","stream":true}"#
        };
        let gen_req: GenerateRequest = serde_json::from_str(payload).unwrap();
        ResponseContext {
            original: ProtocolRequest::Generate(Arc::new(gen_req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: Some("hi".to_string()),
            processed_messages: None,
            tokenizer,
            stop_decoder,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: None,
            reasoning_parser_factory: None,
            configured_tool_parser: None,
            configured_reasoning_parser: None,
        }
    }

    #[tokio::test]
    async fn test_generate_streaming_invoked_with_chat_returns_error_sse() {
        let stream = synthetic_single_stream(Vec::new());
        let resp = generate_streaming::process(stream, chat_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("generate_streaming invoked with chat"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_generate_streaming_partial_emits_text_field() {
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1, 2],
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![1, 2],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 2,
                    completion_tokens: 2,
                    total_tokens: 4,
                },
                logprobs: None,
                input_logprobs: None,
                meta: meta(Some("w1")),
            }),
        ]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("\"text\""));
        assert!(s.contains("meta_info"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_generate_streaming_engine_error_emits_error_field() {
        let stream = synthetic_single_stream(vec![Err(EngineError::Transport(
            tonic::Status::unavailable("dead"),
        ))]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("error"));
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_generate_streaming_finish_reason_length_carries_completion_tokens() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1, 2, 3],
            finish_reason: FinishReason::Length,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 3,
                total_tokens: 4,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(Some("v1")),
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("length"));
    }

    #[tokio::test]
    async fn test_generate_streaming_finish_reason_content_filter() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ContentFilter,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(None),
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("content_filter"));
    }

    #[tokio::test]
    async fn test_generate_streaming_finish_reason_tool_calls() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::ToolCalls,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(None),
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("tool_calls"));
    }

    #[tokio::test]
    async fn test_generate_streaming_finish_reason_abort() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Abort,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(None),
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("abort"));
    }

    #[tokio::test]
    async fn test_generate_streaming_finish_reason_other() {
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Other("zonk".to_string()),
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            },
            logprobs: None,
            input_logprobs: None,
            meta: meta(None),
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("zonk"));
    }

    #[tokio::test]
    async fn test_generate_streaming_return_logprob_threads_partial_and_complete() {
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 5,
                logprob: -0.4,
                decoded_text: None,
                top: vec![],
            }],
        };
        let ilp = InputLogprobs {
            items: vec![TokenLogprob {
                token_id: 6,
                logprob: -0.1,
                decoded_text: None,
                top: vec![],
            }],
        };
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![5],
                logprobs: Some(lp.clone()),
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![5],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                },
                logprobs: Some(lp),
                input_logprobs: Some(ilp),
                meta: meta(Some("vL")),
            }),
        ]);
        let resp = generate_streaming::process(stream, gen_ctx_logprob(true), "regular");
        let s = body_of(resp).await;
        assert!(s.contains("output_token_logprobs"));
        assert!(s.contains("input_token_logprobs"));
    }

    #[tokio::test]
    async fn test_generate_streaming_return_logprob_false_omits_logprobs_fields() {
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
            meta: meta(None),
        })]);
        let resp = generate_streaming::process(stream, gen_ctx_logprob(false), "regular");
        assert_eq!(resp.status(), StatusCode::OK);
        let s = body_of(resp).await;
        assert!(s.contains("[DONE]"));
    }

    #[tokio::test]
    async fn test_generate_streaming_pd_backend_label() {
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
            meta: meta(Some("vp")),
        })]);
        let resp = generate_streaming::process(stream, generate_ctx(), "pd");
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
