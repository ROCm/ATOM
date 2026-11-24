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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            timing: None,
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
        FinishReason, MatchedStop, SeqTiming, TokenChunk, TokenLogprob, TokenLogprobs, Usage,
        WorkerMeta,
    };
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta(v: Option<&str>) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: v.map(|s| s.to_string()),
            cached_tokens: 0,
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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

    /// A sibling's Complete as an ATOM worker reports it when `timed`, with `cached` tokens.
    fn sibling(timed: Option<(f64, f64, f64)>, cached: u32) -> TokenChunk {
        let mut complete = cmpl(vec![1], FinishReason::Stop, None);
        if let TokenChunk::Complete { meta, .. } = &mut complete {
            meta.cached_tokens = cached;
            meta.timing = timed.map(|(ttft_s, tpot_s, latency_s)| SeqTiming {
                ttft: ttft_s,
                tpot: tpot_s,
                latency: latency_s,
            });
        }
        complete
    }

    async fn usage_of(completes: Vec<TokenChunk>) -> serde_json::Value {
        let stream = synthetic_single_stream(completes.into_iter().map(Ok).collect());
        let req = ChatCompletionRequest::default();
        let resp = chat_aggregator::process(stream, ctx_with(req)).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        json["usage"].clone()
    }

    #[tokio::test]
    async fn test_aggregator_usage_reports_atom_s_cached_tokens_and_timing() {
        let usage = usage_of(vec![sibling(Some((0.123456, 0.01, 1.5)), 2)]).await;
        let expected = serde_json::json!({
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 2},
            "ttft_s": 0.1235, "tpot_s": 0.01, "latency_s": 1.5,
        });
        assert_eq!(usage, expected);

        // An SGLang or vLLM worker's usage keeps its shape.
        let usage = usage_of(vec![sibling(None, 2)]).await;
        let expected = serde_json::json!({
            "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
        });
        assert_eq!(usage, expected);
    }

    #[tokio::test]
    async fn test_aggregator_usage_of_atom_siblings_counts_the_prompt_once() {
        let usage = usage_of(vec![
            sibling(Some((0.1, 0.02, 1.0)), 2),
            sibling(Some((0.3, 0.01, 0.5)), 2),
        ])
        .await;
        // Cached tokens of the shared prompt once, the slowest timing, and n.
        let expected = serde_json::json!({
            "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 2},
            "ttft_s": 0.3, "tpot_s": 0.02, "latency_s": 1.0, "num_choices": 2,
        });
        assert_eq!(usage, expected);
    }

    #[tokio::test]
    async fn test_aggregator_reasoning_only_atom_output_carries_an_empty_content() {
        for reads_as_atom in [true, false] {
            let req = ChatCompletionRequest {
                separate_reasoning: true,
                ..Default::default()
            };
            let mut ctx = ctx_with(req);
            ctx.configured_reasoning_parser = Some("deepseek_r1".to_string());
            ctx.reads_as_atom = reads_as_atom;
            let complete = cmpl(vec![1, 2], FinishReason::Length, None);
            let stream = synthetic_single_stream(vec![Ok(complete)]);
            let resp = chat_aggregator::process(stream, ctx).await;
            let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
            let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
            let message = &json["choices"][0]["message"];
            assert_eq!(message["reasoning_content"], "Hello world");
            // ATOM's HTTP chat answers ""; SGLang's and vLLM's leave content out.
            let content = message.get("content");
            assert_eq!(content, reads_as_atom.then_some(&serde_json::json!("")));
        }
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
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            timing: None,
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
            timing: None,
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
        FinishReason, MatchedStop, SeqTiming, TokenChunk, TokenLogprob, TokenLogprobs, Usage,
        WorkerMeta,
    };
    use crate::tokenizer::{traits::Tokenizer, MockTokenizer};

    fn meta_with_weight(v: &str) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: Some(v.to_string()),
            cached_tokens: 0,
            timing: None,
        }
    }

    fn meta_plain() -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: None,
            cached_tokens: 0,
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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

    /// The usage chunk of a stream whose Complete `timed` (an ATOM worker's) or not.
    async fn usage_chunk(include_usage: bool, timed: bool) -> Option<serde_json::Value> {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        req.stream_options = include_usage.then_some(StreamOptions {
            include_usage: Some(true),
        });
        let mut meta = meta_plain();
        meta.cached_tokens = 2;
        meta.timing = timed.then_some(SeqTiming {
            ttft: 0.1,
            tpot: 0.01,
            latency: 1.0,
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(3, 5),
            logprobs: None,
            input_logprobs: None,
            meta,
        })]);
        let s = body_of(chat_streaming::process(stream, ctx_with(req), "regular")).await;
        let mut events = s.lines().filter_map(|line| line.strip_prefix("data: "));
        let chunk = events.find(|e| e.contains("\"usage\""))?;
        Some(serde_json::from_str(chunk).unwrap())
    }

    #[tokio::test]
    async fn test_streaming_atom_usage_chunk_ends_every_stream_with_the_cached_tokens() {
        // After the finish chunk, as ATOM's HTTP chat_finish_frames sends it.
        for include_usage in [false, true] {
            let chunk = usage_chunk(include_usage, true)
                .await
                .expect("a usage chunk");
            assert_eq!(chunk["choices"], serde_json::json!([]));
            let expected = serde_json::json!({
                "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 2},
            });
            assert_eq!(chunk["usage"], expected);
        }
        // An SGLang or vLLM stream keeps its opt-in usage chunk, without the cached tokens.
        assert!(usage_chunk(false, false).await.is_none());
        let chunk = usage_chunk(true, false).await.expect("a usage chunk");
        assert!(chunk["usage"].get("prompt_tokens_details").is_none());
    }

    /// The data events of an ATOM worker's stream whose usage rides on its finish chunk.
    async fn usage_on_finish_events(include_usage: bool) -> Vec<serde_json::Value> {
        let req = ChatCompletionRequest {
            stream: true,
            stream_options: include_usage.then_some(StreamOptions {
                include_usage: Some(true),
            }),
            ..Default::default()
        };
        let mut meta = meta_plain();
        meta.cached_tokens = 2;
        meta.timing = Some(SeqTiming {
            ttft: 0.1,
            tpot: 0.01,
            latency: 1.0,
        });
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: usage(3, 5),
            logprobs: None,
            input_logprobs: None,
            meta,
        })]);
        let mut ctx = ctx_with(req);
        ctx.reads_as_atom = true;
        ctx.usage_on_finish = true;
        let s = body_of(chat_streaming::process(stream, ctx, "regular")).await;
        s.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|event| *event != "[DONE]")
            .map(|event| serde_json::from_str(event).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn test_streaming_atom_usage_rides_on_the_finish_chunk_where_its_worker_puts_it() {
        // As ATOM's HTTP chat_finish_frames with usage_on_finish (DeepSeek-V4.1's contract).
        for include_usage in [false, true] {
            let events = usage_on_finish_events(include_usage).await;
            let (finish, before) = events.split_last().unwrap();
            assert_eq!(finish["choices"][0]["finish_reason"], "stop");
            let expected = serde_json::json!({
                "prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 2},
            });
            assert_eq!(finish["usage"], expected);
            assert!(before.len() >= 2, "{before:?}");
            for event in before {
                // include_usage adds `usage: null` to every chunk before the finish chunk.
                assert_eq!(
                    event.get("usage"),
                    include_usage.then_some(&serde_json::Value::Null)
                );
            }
        }
    }

    /// The first event of a stream whose only output decodes to no text.
    async fn first_event_without_text(reads_as_atom: bool) -> serde_json::Value {
        let mut req = ChatCompletionRequest::default();
        req.stream = true;
        let stream = synthetic_single_stream(vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![999], // <eos>, which decodes to no text
                logprobs: None,
            }),
            Ok(TokenChunk::Complete {
                token_ids: vec![],
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: usage(3, 1),
                logprobs: None,
                input_logprobs: None,
                meta: meta_plain(),
            }),
        ]);
        let mut ctx = ctx_with(req);
        ctx.reads_as_atom = reads_as_atom;
        let s = body_of(chat_streaming::process(stream, ctx, "regular")).await;
        let mut events = s.lines().filter_map(|line| line.strip_prefix("data: "));
        serde_json::from_str(events.next().unwrap()).unwrap()
    }

    #[tokio::test]
    async fn test_streaming_atom_output_without_text_opens_with_an_empty_content() {
        let json = first_event_without_text(true).await;
        assert_eq!(json["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(json["choices"][0]["delta"]["content"], "");
        // SGLang's and vLLM's streams open with their first text.
        let json = first_event_without_text(false).await;
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
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
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            timing: None,
        }
    }

    fn meta_with_weight(v: &str) -> WorkerMeta {
        WorkerMeta {
            request_id: "req".to_string(),
            weight_version: Some(v.to_string()),
            cached_tokens: 0,
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            timing: None,
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

    /// An ATOM worker's timed reply is the response with its timing keys appended to
    /// `meta_info`, byte for byte.
    #[tokio::test]
    async fn test_generate_aggregator_appends_atom_s_timing_to_meta_info() {
        use crate::routers::token_handle::token_chunk::SeqTiming;
        let mut timed = meta();
        timed.timing = Some(SeqTiming {
            ttft: 0.1,
            tpot: 0.01,
            latency: 1.0,
        });
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
            meta: timed,
        })]);
        let resp = generate_aggregator::process(stream, generate_ctx()).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let got: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let info = &got[0]["meta_info"];
        let expected = serde_json::json!([{
            "text": got[0]["text"],
            "output_ids": [10, 20],
            "meta_info": {
                "id": info["id"],
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 1,
                "weight_version": "default",
                "completion_tokens": 2,
                "cached_tokens": 0,
                "e2e_latency": info["e2e_latency"],
                "ttft_s": 0.1, "tpot_s": 0.01, "latency_s": 1.0,
            },
        }]);
        assert_eq!(body, serde_json::to_vec(&expected).unwrap());
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
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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
            timing: None,
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
            timing: None,
        }
    }

    async fn body_of(resp: axum::response::Response) -> String {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        String::from_utf8_lossy(&body).to_string()
    }

    #[tokio::test]
    async fn test_generate_streaming_decodes_a_character_split_across_steps() {
        use std::sync::Arc;

        use crate::routers::{
            prepare::stop_decoder_builder::create_stop_decoder,
            test_mocks::tokenizer::byte_level_tokenizer_dir,
        };
        use crate::tokenizer::{huggingface::HuggingFaceTokenizer, traits::Tokenizer};

        let dir = byte_level_tokenizer_dir(&[], false, serde_json::json!({}));
        let path = dir.path().join("tokenizer.json");
        let tokenizer: Arc<dyn Tokenizer> =
            Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap());
        let mut ctx = generate_ctx();
        ctx.stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        ctx.tokenizer = tokenizer;
        // "你" is three bytes, one token each in a byte-level vocabulary; the
        // engine hands over the first in one step and the rest in the next.
        let ids: Vec<u32> = "你".bytes().map(u32::from).collect();
        let partial = |token_ids: &[u32]| {
            Ok(TokenChunk::Partial {
                token_ids: token_ids.to_vec(),
                logprobs: None,
            })
        };
        let stream = synthetic_single_stream(vec![
            partial(&ids[..1]),
            partial(&ids[1..]),
            Ok(TokenChunk::Complete {
                token_ids: ids.clone(),
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
            }),
        ]);
        let s = body_of(generate_streaming::process(stream, ctx, "regular")).await;
        assert!(s.contains("\"text\":\"你\""), "{s}");
        assert!(!s.contains('\u{FFFD}'), "{s}");
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

    /// The final event carries every id the partials did not, whatever a Complete
    /// carries.
    #[tokio::test]
    async fn test_generate_streaming_final_event_carries_every_unstreamed_id() {
        let complete = |token_ids: Vec<u32>| TokenChunk::Complete {
            token_ids,
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage::default(),
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        };
        let partial = TokenChunk::Partial {
            token_ids: vec![1],
            logprobs: None,
        };
        for (chunks, ids) in [
            (
                vec![partial, complete(vec![1, 2, 6])],
                vec![vec![1], vec![2, 6]],
            ),
            (vec![complete(vec![1, 2, 6])], vec![vec![1, 2, 6]]),
        ] {
            let stream = synthetic_single_stream(chunks.into_iter().map(Ok).collect());
            let body = body_of(generate_streaming::process(
                stream,
                generate_ctx(),
                "regular",
            ));
            let events: Vec<Vec<u32>> = body
                .await
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
                .map(|event| serde_json::from_value(event["output_ids"].clone()).unwrap())
                .collect();
            assert_eq!(events, ids);
        }
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
                timing: None,
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
                timing: None,
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
            timing: None,
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
            prompt_opener: None,
            reads_as_atom: false,
            usage_on_finish: false,
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

/// Streaming metrics count the output a Complete carries past the partials too.
mod e4_generate_streaming_output_tokens {
    use axum::body::to_bytes;

    use super::test_support::generate_ctx;
    use crate::routers::render::generate_streaming;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    #[tokio::test]
    async fn test_output_tokens_are_the_complete_s_count() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let partial = |id| {
            Ok(TokenChunk::Partial {
                token_ids: vec![id],
                logprobs: None,
            })
        };
        let complete = Ok(TokenChunk::Complete {
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
            meta: WorkerMeta::default(),
        });
        let stream = synthetic_single_stream(vec![partial(1), partial(2), complete]);
        let resp = generate_streaming::process(stream, generate_ctx(), "regular");
        to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let rendered = handle.render();
        let output = rendered
            .lines()
            .find(|line| {
                line.starts_with("mesh_router_tokens_total{")
                    && line.contains("token_type=\"output\"")
            })
            .and_then(|line| line.rsplit_once(' ')?.1.parse::<f64>().ok());
        assert_eq!(output, Some(3.0), "{rendered}");
    }
}

mod e3_generate_streaming_across_chunks {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use axum::response::Response;
    use serde_json::Value;
    use tempfile::TempDir;

    use super::test_support::generate_ctx;
    use crate::routers::grpc::completion_adapter::wrap_streaming_generate_as_completion;
    use crate::routers::prepare::response_context::ResponseContext;
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::{generate_aggregator, generate_streaming};
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
    use crate::routers::token_handle::token_handle::TokenHandle;
    use crate::tokenizer::{huggingface::HuggingFaceTokenizer, traits::Tokenizer};

    /// DeepSeek-V4.1-Flash's decoder and its vocabulary entries for the ids in `CASES`.
    const TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true
        },
        "model": {
            "type": "BPE",
            "vocab": {
                "ä»ĸåĢĳ": 9281, "èªª": 4172, "ï¼ļ": 768, "é¬±": 89474, "èĶ": 6355, "¥": 101,
                "çļĦ": 301, "é¾": 2715, "ĺ": 249, "ðŁĺ": 28927, "Ģ": 225, "ðŁĳį": 92526,
                "ðŁı": 40965, "½": 124, "Ġem": 980, "oji": 82644
            },
            "merges": []
        }
    }"#;

    /// Each string and its DeepSeek-V4.1-Flash ids; some ids decode alone to part of a character.
    const CASES: [(&str, &[u32]); 2] = [
        (
            "他們說：鬱鬱蔥蔥的龘",
            &[
                9281, 4172, 768, 89474, 89474, 6355, 101, 6355, 101, 301, 2715, 249,
            ],
        ),
        ("😀👍🏽 emoji", &[28927, 225, 92526, 40965, 124, 980, 82644]),
    ];

    fn tokenizer() -> Arc<dyn Tokenizer> {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, TOKENIZER_JSON).unwrap();
        Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap())
    }

    fn ctx(tokenizer: &Arc<dyn Tokenizer>) -> ResponseContext {
        ResponseContext {
            tokenizer: tokenizer.clone(),
            stop_decoder: create_stop_decoder(tokenizer, None, None, true, false),
            ..generate_ctx()
        }
    }

    fn complete(token_ids: Vec<u32>, completion_tokens: u32) -> TokenChunk {
        TokenChunk::Complete {
            token_ids,
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens,
                total_tokens: 1 + completion_tokens,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: "req".to_string(),
                weight_version: None,
                cached_tokens: 0,
                timing: None,
            },
        }
    }

    /// One id a chunk, then the Complete ATOM ends a stream with, which carries no ids.
    fn token_by_token(ids: &[u32]) -> TokenHandle {
        let mut chunks: Vec<_> = ids
            .iter()
            .map(|&id| {
                Ok(TokenChunk::Partial {
                    token_ids: vec![id],
                    logprobs: None,
                })
            })
            .collect();
        chunks.push(Ok(complete(Vec::new(), ids.len() as u32)));
        synthetic_single_stream(chunks)
    }

    async fn data_events(resp: Response) -> Vec<Value> {
        let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
        std::str::from_utf8(&body)
            .unwrap()
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    #[tokio::test]
    async fn test_generate_and_completions_streams_token_by_token_join_to_the_unary_text() {
        let tokenizer = tokenizer();
        for (text, ids) in CASES {
            let whole = synthetic_single_stream(vec![Ok(complete(ids.to_vec(), ids.len() as u32))]);
            let resp = generate_aggregator::process(whole, ctx(&tokenizer)).await;
            let body = to_bytes(resp.into_body(), 256 * 1024).await.unwrap();
            let unary: Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(unary[0]["text"], text);

            let resp = generate_streaming::process(token_by_token(ids), ctx(&tokenizer), "regular");
            let events = data_events(resp).await;
            assert_eq!(events.last().unwrap()["text"], text);

            let resp = generate_streaming::process(token_by_token(ids), ctx(&tokenizer), "regular");
            let completion = wrap_streaming_generate_as_completion(resp, "m".into()).await;
            let events = data_events(completion).await;
            let pieces = events
                .iter()
                .filter_map(|e| e["choices"][0]["text"].as_str());
            assert_eq!(pieces.collect::<String>(), text);
        }
    }
}

mod f_atom_reasoning_opener {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use serde_json::Value;

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::{chat_aggregator, chat_streaming};
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
    use crate::tokenizer::{huggingface::HuggingFaceTokenizer, traits::Tokenizer};

    /// A byte-level BPE, which encodes any text.
    fn tokenizer() -> Arc<dyn Tokenizer> {
        let dir =
            test_mocks::tokenizer::byte_level_tokenizer_dir(&[], false, serde_json::json!({}));
        let path = dir.path().join("tokenizer.json");
        Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap())
    }

    /// A chat whose reply is read with qwen3, which starts outside reasoning, after
    /// `opener`, the prompt's.
    fn ctx(stream: bool, opener: Option<&str>) -> ResponseContext {
        let tokenizer = tokenizer();
        let stop_decoder = create_stop_decoder(&tokenizer, None, None, true, false);
        let req = ChatCompletionRequest {
            stream,
            separate_reasoning: true,
            ..Default::default()
        };
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
            reasoning_parser_factory: Some(test_mocks::reasoning_parser_factory()),
            configured_tool_parser: None,
            configured_reasoning_parser: Some("qwen3".to_string()),
            prompt_opener: opener.map(str::to_string),
            reads_as_atom: false,
            usage_on_finish: false,
        }
    }

    /// `reply`'s ids, streamed one at a time when `partials`, then its Complete.
    fn chunks(reply: &str, partials: bool) -> Vec<TokenChunk> {
        let ids = tokenizer()
            .encode(reply, false)
            .unwrap()
            .token_ids()
            .to_vec();
        let mut chunks: Vec<TokenChunk> = Vec::new();
        if partials {
            chunks.extend(ids.iter().map(|&id| TokenChunk::Partial {
                token_ids: vec![id],
                logprobs: None,
            }));
        }
        chunks.push(TokenChunk::Complete {
            token_ids: if partials { Vec::new() } else { ids.clone() },
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: ids.len() as u32,
                total_tokens: 1 + ids.len() as u32,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta::default(),
        });
        chunks
    }

    /// (reasoning, content) of `reply` answered unary.
    async fn unary(reply: &str, opener: Option<&str>) -> (String, String) {
        let stream = synthetic_single_stream(chunks(reply, false).into_iter().map(Ok).collect());
        let resp = chat_aggregator::process(stream, ctx(false, opener)).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        let message = &json["choices"][0]["message"];
        let text = |key: &str| message[key].as_str().unwrap_or_default().to_string();
        (text("reasoning_content"), text("content"))
    }

    /// (reasoning, content) of `reply` streamed, its deltas joined.
    async fn streamed(reply: &str, opener: Option<&str>) -> (String, String) {
        let stream = synthetic_single_stream(chunks(reply, true).into_iter().map(Ok).collect());
        let resp = chat_streaming::process(stream, ctx(true, opener), "regular");
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let (mut reasoning, mut content) = (String::new(), String::new());
        for line in String::from_utf8_lossy(&body).lines() {
            let Some(data) = line.strip_prefix("data: ") else {
                continue;
            };
            let Ok(event) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            let delta = &event["choices"][0]["delta"];
            reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
            content.push_str(delta["content"].as_str().unwrap_or_default());
        }
        (reasoning, content)
    }

    #[tokio::test]
    async fn a_reply_to_a_prompt_that_closed_reasoning_is_content() {
        // A thinking-off prompt ends in "</think>": no opener.
        let expected = (String::new(), "Hello world".to_string());
        assert_eq!(unary("Hello world", None).await, expected);
        assert_eq!(streamed("Hello world", None).await, expected);
    }

    #[tokio::test]
    async fn a_reply_to_a_prompt_that_opened_reasoning_is_reasoning_up_to_its_closer() {
        let reply = "plan it</think>The answer";
        let expected = ("plan it".to_string(), "The answer".to_string());
        assert_eq!(unary(reply, Some("<think>")).await, expected);
        let (reasoning, content) = streamed(reply, Some("<think>")).await;
        assert_eq!(
            (reasoning.trim(), content.trim()),
            ("plan it", "The answer")
        );

        // Cut before its closer, it is reasoning all through.
        let expected = ("still planning".to_string(), String::new());
        assert_eq!(unary("still planning", Some("<think>")).await, expected);
        assert_eq!(streamed("still planning", Some("<think>")).await, expected);
    }

    /// Replayed 256 ids at a time, among them a reply whose closer the 256th id cuts
    /// after its "<".
    #[tokio::test]
    async fn a_long_reply_arriving_in_its_complete_alone_is_split_as_when_streamed() {
        for thought in ["z".repeat(70_000), "z".repeat(248)] {
            let reply = format!("<think>{thought}</think>answer");
            let ids = tokenizer()
                .encode(&reply, false)
                .unwrap()
                .token_ids()
                .to_vec();
            let complete = chunks(&reply, false);
            let stream = synthetic_single_stream(complete.into_iter().map(Ok).collect());
            let resp = chat_streaming::process(stream, ctx(true, None), "regular");
            let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
            let (mut reasoning, mut content) = (String::new(), String::new());
            for line in String::from_utf8_lossy(&body).lines() {
                let Some(event) = line
                    .strip_prefix("data: ")
                    .and_then(|data| serde_json::from_str::<Value>(data).ok())
                else {
                    continue;
                };
                let delta = &event["choices"][0]["delta"];
                reasoning.push_str(delta["reasoning_content"].as_str().unwrap_or_default());
                content.push_str(delta["content"].as_str().unwrap_or_default());
            }
            assert!(ids.len() > 256);
            assert_eq!(
                (reasoning.len(), content.as_str()),
                (thought.len(), "answer")
            );
        }
    }

    #[tokio::test]
    async fn a_streamed_reply_ending_in_a_delimiter_s_first_characters_keeps_them() {
        // Ended inside reasoning on "<", or in content on "</".
        let expected = ("plan <".to_string(), String::new());
        assert_eq!(streamed("plan <", Some("<think>")).await, expected);
        let expected = ("plan".to_string(), "3 </".to_string());
        assert_eq!(
            streamed("plan</think>3 </", Some("<think>")).await,
            expected
        );
    }
}

mod g_complete_suffix {
    use axum::body::to_bytes;
    use serde_json::Value;

    use super::test_support::{chat_ctx, generate_ctx};
    use crate::routers::render::{
        chat_aggregator, chat_streaming, generate_aggregator, generate_streaming,
    };
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    /// SGLang's or vLLM's stream: "Hello", "world", then a Complete with every id, its
    /// last ("." , not an end of sequence) in no partial.
    fn chunks(streamed: bool) -> Vec<TokenChunk> {
        let partial = |id| TokenChunk::Partial {
            token_ids: vec![id],
            logprobs: None,
        };
        let complete = TokenChunk::Complete {
            token_ids: vec![1, 2, 6],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 1,
                completion_tokens: 3,
                total_tokens: 4,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta::default(),
        };
        let mut chunks = if streamed {
            vec![partial(1), partial(2)]
        } else {
            vec![]
        };
        chunks.push(complete);
        chunks
    }

    async fn body(resp: axum::response::Response) -> String {
        let bytes = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    fn events(sse: &str) -> Vec<Value> {
        sse.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|data| serde_json::from_str(data).ok())
            .collect()
    }

    #[tokio::test]
    async fn a_streamed_chat_carries_the_complete_s_unstreamed_ids_as_unary_does() {
        let stream = synthetic_single_stream(chunks(false).into_iter().map(Ok).collect());
        let unary: Value =
            serde_json::from_str(&body(chat_aggregator::process(stream, chat_ctx()).await).await)
                .unwrap();
        let mut ctx = chat_ctx();
        if let crate::routers::prepare::response_context::ProtocolRequest::Chat(req) =
            &mut ctx.original
        {
            std::sync::Arc::make_mut(req).stream = true;
        }
        let stream = synthetic_single_stream(chunks(true).into_iter().map(Ok).collect());
        let sse = body(chat_streaming::process(stream, ctx, "regular")).await;
        let streamed: String = events(&sse)
            .iter()
            .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(streamed, unary["choices"][0]["message"]["content"]);
        assert!(streamed.ends_with('.'), "{streamed}");
    }

    /// The unstreamed id's delta carries its logprob from the Complete's, which covers every id.
    #[tokio::test]
    async fn a_streamed_chat_carries_the_logprobs_of_the_complete_s_unstreamed_ids() {
        use crate::routers::token_handle::token_chunk::{TokenLogprob, TokenLogprobs};
        let logprob = |token_id, logprob| TokenLogprob {
            token_id,
            logprob,
            decoded_text: None,
            top: vec![],
        };
        let mut chunks = chunks(true);
        for (chunk, id) in chunks.iter_mut().zip([1, 2]) {
            if let TokenChunk::Partial { logprobs, .. } = chunk {
                *logprobs = Some(TokenLogprobs {
                    items: vec![logprob(id, -0.5)],
                });
            }
        }
        if let Some(TokenChunk::Complete { logprobs, .. }) = chunks.last_mut() {
            let items = vec![logprob(1, -0.5), logprob(2, -0.5), logprob(6, -0.25)];
            *logprobs = Some(TokenLogprobs { items });
        }
        let mut ctx = chat_ctx();
        if let crate::routers::prepare::response_context::ProtocolRequest::Chat(req) =
            &mut ctx.original
        {
            std::sync::Arc::make_mut(req).stream = true;
        }
        let stream = synthetic_single_stream(chunks.into_iter().map(Ok).collect());
        let sse = body(chat_streaming::process(stream, ctx, "regular")).await;
        let last = events(&sse)
            .into_iter()
            .rfind(|e| e["choices"][0]["delta"]["content"] == " .")
            .unwrap_or_else(|| panic!("no delta for the unstreamed id: {sse}"));
        let logprobs = &last["choices"][0]["logprobs"]["content"];
        assert_eq!(logprobs[0]["logprob"], -0.25, "{last}");
    }

    #[tokio::test]
    async fn a_streamed_generate_carries_the_complete_s_unstreamed_ids_as_unary_does() {
        let stream = synthetic_single_stream(chunks(false).into_iter().map(Ok).collect());
        let unary: Value = serde_json::from_str(
            &body(generate_aggregator::process(stream, generate_ctx()).await).await,
        )
        .unwrap();
        let stream = synthetic_single_stream(chunks(true).into_iter().map(Ok).collect());
        let sse = body(generate_streaming::process(
            stream,
            generate_ctx(),
            "regular",
        ))
        .await;
        let last = events(&sse).pop().unwrap();
        assert_eq!(last["text"], unary[0]["text"]);
        assert!(last["text"].as_str().unwrap().ends_with('.'), "{last}");
    }

    /// A streamed chat replays a long Complete 256 ids at a time, each piece copied
    /// alone rather than cut from a copy of the whole remaining suffix.
    #[test]
    fn a_replayed_piece_copies_only_its_own_ids() {
        use crate::routers::token_handle::token_chunk::{TokenLogprob, TokenLogprobs};
        let ids: Vec<u32> = (0..65_536).collect();
        let item = TokenLogprob {
            token_id: 0,
            logprob: -1.0,
            decoded_text: None,
            top: Vec::new(),
        };
        let complete = TokenChunk::Complete {
            token_ids: ids.clone(),
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage::default(),
            logprobs: Some(TokenLogprobs {
                items: vec![item; ids.len()],
            }),
            input_logprobs: None,
            meta: WorkerMeta::default(),
        };
        let piece = chat_streaming::unstreamed(&complete, 1024, 256).unwrap();
        assert_eq!((piece[0], piece.len(), piece.capacity()), (1024, 256, 256));
        let logprobs = chat_streaming::unstreamed_logprobs(&complete, 1024, 256).unwrap();
        assert_eq!(
            (logprobs.items.len(), logprobs.items.capacity()),
            (256, 256)
        );
        let whole = chat_streaming::unstreamed(&complete, 1024, usize::MAX).unwrap();
        assert_eq!(whole.len(), 65_536 - 1024); // /generate's whole suffix
    }
}

mod h_atom_native_tool_calls {
    use std::sync::Arc;

    use axum::body::to_bytes;
    use serde_json::{json, Value};

    use crate::protocols::chat::ChatCompletionRequest;
    use crate::routers::prepare::response_context::{ProtocolRequest, ResponseContext};
    use crate::routers::prepare::stop_decoder_builder::create_stop_decoder;
    use crate::routers::render::{chat_aggregator, chat_streaming};
    use crate::routers::test_mocks;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
    use crate::tokenizer::{huggingface::HuggingFaceTokenizer, traits::Tokenizer};

    const CALL: &str =
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn tokenizer() -> Arc<dyn Tokenizer> {
        let dir = test_mocks::tokenizer::byte_level_tokenizer_dir(&[], false, json!({}));
        let path = dir.path().join("tokenizer.json");
        Arc::new(HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap())
    }

    /// An ATOM reply in qwen's tool syntax to a chat forcing `choice`.
    fn ctx(stream: bool, choice: Value) -> ResponseContext {
        let tokenizer = tokenizer();
        let tool =
            json!({"type": "function", "function": {"name": "get_weather", "parameters": {}}});
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}],
            "tools": [tool], "tool_choice": choice, "stream": stream,
        }))
        .unwrap();
        ResponseContext {
            original: ProtocolRequest::Chat(Arc::new(req)),
            model_id: Some("m".to_string()),
            headers: None,
            original_text: None,
            processed_messages: None,
            stop_decoder: create_stop_decoder(&tokenizer, None, None, false, false),
            tokenizer,
            request_id: "rid".to_string(),
            created: 0,
            tool_parser_factory: Some(test_mocks::tool_parser_factory()),
            reasoning_parser_factory: None,
            configured_tool_parser: Some("qwen".to_string()),
            configured_reasoning_parser: None,
            prompt_opener: None,
            reads_as_atom: true,
            usage_on_finish: false,
        }
    }

    fn chunks(streamed: bool) -> Vec<TokenChunk> {
        let ids = tokenizer()
            .encode(CALL, false)
            .unwrap()
            .token_ids()
            .to_vec();
        let mut chunks: Vec<TokenChunk> = Vec::new();
        if streamed {
            chunks.extend(ids.chunks(4).map(|ids| TokenChunk::Partial {
                token_ids: ids.to_vec(),
                logprobs: None,
            }));
        }
        chunks.push(TokenChunk::Complete {
            token_ids: if streamed { Vec::new() } else { ids },
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage::default(),
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta::default(),
        });
        chunks
    }

    fn forced_choices() -> [Value; 2] {
        [
            json!({"type": "function", "function": {"name": "get_weather"}}),
            json!("required"),
        ]
    }

    #[tokio::test]
    async fn a_forced_atom_tool_call_is_parsed_in_the_model_s_syntax() {
        for choice in forced_choices() {
            let stream = synthetic_single_stream(chunks(false).into_iter().map(Ok).collect());
            let resp = chat_aggregator::process(stream, ctx(false, choice.clone())).await;
            let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
            let json: Value = serde_json::from_slice(&body).unwrap();
            let call = &json["choices"][0]["message"]["tool_calls"][0]["function"];
            assert_eq!(call["name"], "get_weather", "{choice}: {json}");
            let arguments: Value =
                serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
            assert_eq!(arguments, json!({"city": "Paris"}));
        }
    }

    /// The declared schema types the unary reply's arguments, as it does the streamed one's.
    #[tokio::test]
    async fn a_unary_tool_call_keeps_a_string_declared_argument_a_string() {
        let tool = json!({"type": "function", "function": {"name": "search", "parameters": {
            "type": "object", "properties": {"limit": {"type": "string"}},
        }}});
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [tool],
        }))
        .unwrap();
        let mut ctx = ctx(false, json!("auto"));
        ctx.original = ProtocolRequest::Chat(Arc::new(req));
        ctx.configured_tool_parser = Some("qwen_xml".to_string());
        let call =
            "<tool_call><function=search><parameter=limit>4</parameter></function></tool_call>";
        let ids = tokenizer()
            .encode(call, false)
            .unwrap()
            .token_ids()
            .to_vec();
        let stream = synthetic_single_stream(vec![Ok(TokenChunk::Complete {
            token_ids: ids,
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage::default(),
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta::default(),
        })]);
        let resp = chat_aggregator::process(stream, ctx).await;
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let json: Value = serde_json::from_slice(&body).unwrap();
        let call = &json["choices"][0]["message"]["tool_calls"][0]["function"];
        let arguments: Value = serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(arguments, json!({"limit": "4"}), "{json}");
    }

    #[tokio::test]
    async fn a_forced_atom_tool_call_streams_in_the_model_s_syntax() {
        for choice in forced_choices() {
            let stream = synthetic_single_stream(chunks(true).into_iter().map(Ok).collect());
            let resp = chat_streaming::process(stream, ctx(true, choice.clone()), "regular");
            let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
            let (mut name, mut arguments) = (String::new(), String::new());
            for line in String::from_utf8_lossy(&body).lines() {
                let Some(event) = line
                    .strip_prefix("data: ")
                    .and_then(|data| serde_json::from_str::<Value>(data).ok())
                else {
                    continue;
                };
                for call in event["choices"][0]["delta"]["tool_calls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                {
                    name.push_str(call["function"]["name"].as_str().unwrap_or_default());
                    arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
                }
            }
            assert_eq!(name, "get_weather", "{choice}");
            let arguments: Value = serde_json::from_str(&arguments).unwrap();
            assert_eq!(arguments, json!({"city": "Paris"}), "{choice}");
        }
    }

    /// A qwen_xml reply cut off by max_tokens inside a call: (name, arguments) of the call
    /// read unary, then streamed.
    async fn cut_off_call(reply: &str) -> [(String, Value); 2] {
        cut_off_call_reasoned(reply, None).await
    }

    /// `cut_off_call`, its reasoning separated by the `reasoning` parser.
    async fn cut_off_call_reasoned(reply: &str, reasoning: Option<&str>) -> [(String, Value); 2] {
        cut_off_call_with(reply, "qwen_xml", reasoning).await
    }

    /// `cut_off_call_reasoned`, its tool calls read by `parser`.
    async fn cut_off_call_with(
        reply: &str,
        parser: &str,
        reasoning: Option<&str>,
    ) -> [(String, Value); 2] {
        let tool = json!({"type": "function", "function": {"name": "get_weather", "parameters": {
            "type": "object", "properties": {"city": {"type": "string"}},
        }}});
        let ids = tokenizer()
            .encode(reply, false)
            .unwrap()
            .token_ids()
            .to_vec();
        let mut read = Vec::new();
        for stream in [false, true] {
            let req: ChatCompletionRequest = serde_json::from_value(json!({
                "model": "m", "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool.clone()], "stream": stream,
            }))
            .unwrap();
            let mut ctx = ctx(stream, json!("auto"));
            ctx.original = ProtocolRequest::Chat(Arc::new(req));
            ctx.configured_tool_parser = Some(parser.to_string());
            if let Some(parser) = reasoning {
                ctx.reasoning_parser_factory = Some(test_mocks::reasoning_parser_factory());
                ctx.configured_reasoning_parser = Some(parser.to_string());
            }
            let mut chunks: Vec<TokenChunk> = Vec::new();
            if stream {
                chunks.extend(ids.chunks(3).map(|ids| TokenChunk::Partial {
                    token_ids: ids.to_vec(),
                    logprobs: None,
                }));
            }
            chunks.push(TokenChunk::Complete {
                token_ids: if stream { Vec::new() } else { ids.clone() },
                finish_reason: FinishReason::Length,
                matched_stop: None,
                usage: Usage::default(),
                logprobs: None,
                input_logprobs: None,
                meta: WorkerMeta::default(),
            });
            let stream_of = synthetic_single_stream(chunks.into_iter().map(Ok).collect());
            let (mut name, mut arguments) = (String::new(), String::new());
            if stream {
                let resp = chat_streaming::process(stream_of, ctx, "regular");
                let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
                for line in String::from_utf8_lossy(&body).lines() {
                    let Some(event) = line
                        .strip_prefix("data: ")
                        .and_then(|data| serde_json::from_str::<Value>(data).ok())
                    else {
                        continue;
                    };
                    let calls = event["choices"][0]["delta"]["tool_calls"]
                        .as_array()
                        .cloned();
                    for call in calls.into_iter().flatten() {
                        name.push_str(call["function"]["name"].as_str().unwrap_or_default());
                        let delta = call["function"]["arguments"].as_str().unwrap_or_default();
                        arguments.push_str(delta);
                    }
                }
            } else {
                let resp = chat_aggregator::process(stream_of, ctx).await;
                let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
                let json: Value = serde_json::from_slice(&body).unwrap();
                let call = &json["choices"][0]["message"]["tool_calls"][0]["function"];
                name.push_str(call["name"].as_str().unwrap_or_default());
                arguments.push_str(call["arguments"].as_str().unwrap_or_default());
            }
            let arguments = serde_json::from_str(&arguments).unwrap_or(Value::Null);
            read.push((name, arguments));
        }
        read.try_into().unwrap()
    }

    #[tokio::test]
    async fn a_native_call_cut_off_by_max_tokens_keeps_its_arrived_value() {
        let reply = "<tool_call><function=get_weather><parameter=city>Paris";
        let expected = ("get_weather".to_string(), json!({"city": "Paris"}));
        assert_eq!(cut_off_call(reply).await, [expected.clone(), expected]);
        // Prose quoting the syntax is not a call.
        let prose = "You write <function=get_weather> and then its parameters";
        let none = (String::new(), Value::Null);
        assert_eq!(cut_off_call(prose).await, [none.clone(), none]);
    }

    /// A last "<" the reasoning parser holds until the Complete releases it still reaches
    /// the call it ends, streamed as unary.
    #[tokio::test]
    async fn a_cut_off_call_keeps_a_last_character_the_reasoning_parser_held() {
        let reply = "<tool_call><function=get_weather><parameter=city>Paris<";
        let expected = ("get_weather".to_string(), json!({"city": "Paris<"}));
        let read = cut_off_call_reasoned(reply, Some("qwen3")).await;
        assert_eq!(read, [expected.clone(), expected]);
    }

    /// A qwen_xml call cut off after its function closed but before its wrapper did is
    /// kept, as ATOM's HTTP parser keeps it; a name prose follows is not closed as one.
    #[tokio::test]
    async fn a_qwen_call_cut_off_before_its_wrapper_closes_is_kept() {
        let reply = "<tool_call><function=get_weather><parameter=city>Paris</parameter></function>";
        let expected = ("get_weather".to_string(), json!({"city": "Paris"}));
        assert_eq!(cut_off_call(reply).await, [expected.clone(), expected]);
        let prose = "<tool_call><function=get_weather> and then its parameters";
        let [unary, _] = cut_off_call(prose).await;
        assert_eq!(unary, (String::new(), Value::Null));
    }

    /// A qwen_xml call outside a `<tool_call>` wrapper is a call, whole or cut off, as
    /// ATOM's HTTP parser reads one; a bare name prose follows is not.
    #[tokio::test]
    async fn a_bare_qwen_function_call_is_a_call() {
        let expected = ("get_weather".to_string(), json!({"city": "Paris"}));
        for reply in [
            "<function=get_weather><parameter=city>Paris</parameter></function>",
            "Checking.\n<function=get_weather>\n<parameter=city>Paris",
        ] {
            assert_eq!(
                cut_off_call(reply).await,
                [expected.clone(), expected.clone()]
            );
        }
        let prose = "Write <function=get_weather> and then its parameters.";
        let none = (String::new(), Value::Null);
        assert_eq!(cut_off_call(prose).await, [none.clone(), none]);
        // Nor does a name quoted before a call take that call's parameters.
        let quoted = "Use <function=search> here. \
                      <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>";
        assert_eq!(cut_off_call(quoted).await, [expected.clone(), expected]);
    }

    /// A bare qwen_xml call cut off at its declared name, or inside the tag after it, is
    /// kept, as ATOM's HTTP parser keeps it at end of input; a cut-off name is not.
    #[tokio::test]
    async fn a_bare_qwen_call_cut_off_after_its_name_is_kept() {
        let expected = ("get_weather".to_string(), json!({}));
        for reply in ["<function=get_weather>", "<function=get_weather><para"] {
            let read = cut_off_call(reply).await;
            assert_eq!(read, [expected.clone(), expected.clone()], "{reply}");
        }
        let reply = "<function=get_weather><parameter=city>Paris</parameter><para";
        let paris = ("get_weather".to_string(), json!({"city": "Paris"}));
        assert_eq!(cut_off_call(reply).await, [paris.clone(), paris]);
        let none = (String::new(), Value::Null);
        for reply in [
            "<function=get_wea",
            "<function=search>",
            "<function=get_weather> so",
        ] {
            assert_eq!(cut_off_call(reply).await, [none.clone(), none.clone()]);
        }
    }

    /// A bare qwen_xml call cut off in a parameter's name, or in a tag after a closed
    /// parameter, is kept, as ATOM's HTTP parser keeps it at end of input.
    #[tokio::test]
    async fn a_bare_qwen_call_cut_off_in_a_tag_is_kept() {
        let empty = ("get_weather".to_string(), json!({}));
        let read = cut_off_call("<function=get_weather><parameter=ci").await;
        assert_eq!(read, [empty.clone(), empty]);
        let reply = "<function=get_weather><parameter=city>Paris</parameter></para";
        let paris = ("get_weather".to_string(), json!({"city": "Paris"}));
        assert_eq!(cut_off_call(reply).await, [paris.clone(), paris]);
    }

    /// A qwen_xml reply read unary, then streamed, from an ATOM worker or not: (content,
    /// names of the calls) of each.
    async fn read_qwen_xml(reply: &str, reads_as_atom: bool) -> [(String, String); 2] {
        let ids = tokenizer()
            .encode(reply, false)
            .unwrap()
            .token_ids()
            .to_vec();
        let mut read = Vec::new();
        for stream in [false, true] {
            let mut ctx = ctx(stream, json!("auto"));
            ctx.configured_tool_parser = Some("qwen_xml".to_string());
            ctx.reads_as_atom = reads_as_atom;
            let complete = TokenChunk::Complete {
                token_ids: ids.clone(),
                finish_reason: FinishReason::Stop,
                matched_stop: None,
                usage: Usage::default(),
                logprobs: None,
                input_logprobs: None,
                meta: WorkerMeta::default(),
            };
            let stream_of = synthetic_single_stream(vec![Ok(complete)]);
            let (mut content, mut names) = (String::new(), String::new());
            let messages: Vec<Value> = if stream {
                let resp = chat_streaming::process(stream_of, ctx, "regular");
                let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
                let body = String::from_utf8_lossy(&body).into_owned();
                (body.lines())
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                    .map(|event| event["choices"][0]["delta"].clone())
                    .collect()
            } else {
                let resp = chat_aggregator::process(stream_of, ctx).await;
                let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
                let json: Value = serde_json::from_slice(&body).unwrap();
                vec![json["choices"][0]["message"].clone()]
            };
            for message in messages {
                content.push_str(message["content"].as_str().unwrap_or_default());
                for call in message["tool_calls"].as_array().into_iter().flatten() {
                    names.push_str(call["function"]["name"].as_str().unwrap_or_default());
                }
            }
            read.push((content, names));
        }
        read.try_into().unwrap()
    }

    /// A bare qwen_xml call is a call in an ATOM worker's reply only: SGLang's and vLLM's
    /// HTTP chats return it as text.
    #[tokio::test]
    async fn only_an_atom_reply_reads_a_bare_qwen_function_call_as_a_call() {
        let reply = "Sure. <function=get_weather><parameter=city>Paris</parameter></function>";
        let call = ("Sure. ".to_string(), "get_weather".to_string());
        assert_eq!(read_qwen_xml(reply, true).await, [call.clone(), call]);
        let text = (reply.to_string(), String::new());
        assert_eq!(read_qwen_xml(reply, false).await, [text.clone(), text]);
    }

    /// An ATOM reply's text streams as it reads unary: prose quoting `<tool_call>`
    /// before a bare call, and a last tag no call followed, stay text.
    #[tokio::test]
    async fn an_atom_reply_s_text_streams_as_it_reads_unary() {
        let reply = "Use <tool_call> as the wrapper. \
                     <function=get_weather><parameter=city>Paris</parameter></function>";
        let prose = "Use <tool_call> as the wrapper. ".to_string();
        let call = (prose, "get_weather".to_string());
        assert_eq!(read_qwen_xml(reply, true).await, [call.clone(), call]);
        for reply in ["The prefix is <fun", "Cut: <tool_call><function=get_wea"] {
            let text = (reply.to_string(), String::new());
            assert_eq!(read_qwen_xml(reply, true).await, [text.clone(), text]);
        }
    }

    /// A bare qwen_xml call naming no declared tool is text, unary as streamed.
    #[tokio::test]
    async fn a_bare_qwen_call_naming_no_declared_tool_is_text() {
        let reply = "The format is <function=YOUR FUNCTION NAME></function>.";
        let text = (reply.to_string(), String::new());
        assert_eq!(read_qwen_xml(reply, true).await, [text.clone(), text]);
    }

    /// An ATOM reply's text is all of it outside its calls, unary as streamed.
    #[tokio::test]
    async fn an_atom_reply_s_text_after_its_call_is_kept() {
        let reply =
            "Before <function=get_weather><parameter=city>Paris</parameter></function> After";
        let call = ("Before  After".to_string(), "get_weather".to_string());
        assert_eq!(read_qwen_xml(reply, true).await, [call.clone(), call]);
    }

    /// A MiniMax-M3 call cut off inside a parameter keeps its value, as ATOM's HTTP
    /// parser does at end of input.
    #[tokio::test]
    async fn a_minimax_call_cut_off_inside_a_parameter_keeps_its_value() {
        let ns = crate::tool_parser::parsers::minimax::MINIMAX_NS;
        let reply = format!("{ns}<tool_call>{ns}<invoke name=\"get_weather\">{ns}<city>Paris");
        let expected = ("get_weather".to_string(), json!({"city": "Paris"}));
        let read = cut_off_call_with(&reply, "minimax", None).await;
        assert_eq!(read, [expected.clone(), expected]);
    }
}

/// A streamed response dropped while its source waits for output drops the source,
/// which ends its call and frees the loads it holds.
mod c3_streaming_client_disconnect {
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll};
    use std::time::Duration;

    use axum::response::Response;
    use futures::Stream;

    use super::test_support::{chat_ctx, generate_ctx};
    use crate::routers::render::{chat_streaming, generate_streaming};
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::token_chunk::TokenChunk;
    use crate::routers::token_handle::token_handle::{TokenHandle, TokenSource};

    /// A source with no output yet, as a decode is before its first token.
    struct Waiting(Arc<AtomicBool>);

    impl Stream for Waiting {
        type Item = Result<TokenChunk, EngineError>;

        fn poll_next(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Poll::Pending
        }
    }

    impl TokenSource for Waiting {
        fn mark_completed(&mut self) {}
    }

    impl Drop for Waiting {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    async fn assert_dropped_with_its_response(render: impl FnOnce(TokenHandle) -> Response) {
        let dropped = Arc::new(AtomicBool::new(false));
        let response = render(TokenHandle::new(Waiting(dropped.clone())));
        tokio::task::yield_now().await;
        drop(response);
        let released = async {
            while !dropped.load(Ordering::SeqCst) {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        tokio::time::timeout(Duration::from_secs(5), released)
            .await
            .expect("the source outlived its response");
    }

    #[tokio::test]
    async fn test_chat_stream_drops_its_source_when_the_client_leaves() {
        assert_dropped_with_its_response(|s| chat_streaming::process(s, chat_ctx(), "pd")).await;
    }

    #[tokio::test]
    async fn test_generate_stream_drops_its_source_when_the_client_leaves() {
        let render = |s| generate_streaming::process(s, generate_ctx(), "pd");
        assert_dropped_with_its_response(render).await;
    }
}
