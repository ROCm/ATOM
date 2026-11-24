//! Aggregate a `TokenHandle<TokenChunk>` into a non-streaming
//! `ChatCompletionResponse`.

use std::{borrow::Cow, sync::Arc};

use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::Value;
use tracing::error;

use crate::{
    protocols::{
        chat::{ChatChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse},
        common::{FunctionCallResponse, Tool, ToolCall, ToolChoice, ToolChoiceValue, Usage},
    },
    reasoning_parser::{ParserFactory as ReasoningParserFactory, ParserResult},
    routers::{
        comm::error,
        prepare::{
            parser_factory_lookup::{
                check_reasoning_parser_availability, create_reasoning_parser, get_reasoning_parser,
            },
            response_context::{ProtocolRequest, ResponseContext},
            tool_constraints::{
                generate_tool_call_id, get_history_tool_calls_count, parse_json_schema_response,
            },
        },
        render::{
            atom_usage::{self, AtomUsage},
            logprob_conversion::token_logprobs_to_chat,
        },
        token_handle::{
            engine_error::EngineError,
            token_chunk::{FinishReason, MatchedStop, TokenChunk},
            token_handle::TokenHandle,
        },
    },
    tokenizer::{
        stop::{SequenceDecoderOutput, StopSequenceDecoder},
        traits::Tokenizer,
    },
    tool_parser::{
        registry::{check_tool_parser_availability, create_reply_tool_parser, get_tool_parser},
        ParserFactory as ToolParserFactory,
    },
};

pub async fn process(stream: TokenHandle, ctx: ResponseContext) -> Response {
    match aggregate(stream, ctx).await {
        Ok((resp, None)) => axum::Json(resp).into_response(),
        Ok((resp, Some(atom))) => match atom.extend_response(&resp, atom_usage::CHAT) {
            Ok(value) => axum::Json(value).into_response(),
            Err(e) => error::internal_error("serialization_failed", e.to_string()),
        },
        Err(resp) => resp,
    }
}

pub async fn process_typed(
    stream: TokenHandle,
    ctx: ResponseContext,
) -> Result<ChatCompletionResponse, Response> {
    Ok(aggregate(stream, ctx).await?.0)
}

/// The response, and ATOM's usage additions where an ATOM worker answered.
async fn aggregate(
    stream: TokenHandle,
    ctx: ResponseContext,
) -> Result<(ChatCompletionResponse, Option<AtomUsage>), Response> {
    let chat_request = match &ctx.original {
        ProtocolRequest::Chat(r) => Arc::clone(r),
        ProtocolRequest::Generate(_) => {
            return Err(error::internal_error(
                "wrong_render_path",
                "chat_aggregator invoked with a generate request",
            ));
        }
    };

    let completes = collect_completes(stream).await?;
    if completes.is_empty() {
        return Err(error::internal_error(
            "no_responses_from_server",
            "No responses from server",
        ));
    }

    let history_tool_calls_count = get_history_tool_calls_count(&chat_request);

    let reasoning_available = chat_request.separate_reasoning
        && ctx
            .reasoning_parser_factory
            .as_ref()
            .map(|f| {
                check_reasoning_parser_availability(
                    f,
                    ctx.configured_reasoning_parser.as_deref(),
                    &chat_request.model,
                )
            })
            .unwrap_or(false);

    let tool_choice_enabled = !matches!(
        &chat_request.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    );
    let tool_available = tool_choice_enabled
        && chat_request.tools.is_some()
        && ctx
            .tool_parser_factory
            .as_ref()
            .map(|f| {
                check_tool_parser_availability(
                    f,
                    ctx.configured_tool_parser.as_deref(),
                    &chat_request.model,
                )
            })
            .unwrap_or(false);

    let mut stop_decoder = ctx.stop_decoder;
    let mut choices = Vec::with_capacity(completes.len());
    for (idx, complete) in completes.iter().enumerate() {
        let res = process_choice(
            complete,
            idx as u32,
            &chat_request,
            &ctx.tokenizer,
            &mut stop_decoder,
            history_tool_calls_count,
            reasoning_available,
            tool_available,
            ctx.tool_parser_factory.as_ref(),
            ctx.reasoning_parser_factory.as_ref(),
            ctx.configured_tool_parser.as_deref(),
            ctx.configured_reasoning_parser.as_deref(),
            ctx.prompt_opener.as_deref(),
            ctx.reads_as_atom,
        )
        .await;
        match res {
            Ok(c) => choices.push(c),
            Err(e) => {
                return Err(error::internal_error(
                    "process_choice_failed",
                    format!("Failed to process choice {}: {}", idx, e),
                ))
            }
        }
    }

    let weight_version = completes.iter().find_map(|c| match c {
        TokenChunk::Complete { meta, .. } => meta.weight_version.clone(),
        _ => None,
    });

    let response = ChatCompletionResponse::builder(
        &ctx.request_id,
        ctx.model_id.as_deref().unwrap_or(&chat_request.model),
    )
    .created(ctx.created)
    .choices(choices)
    .usage(build_usage(&completes))
    .maybe_system_fingerprint(weight_version)
    .build();

    Ok((response, AtomUsage::of_completes(&completes)))
}

async fn collect_completes(mut stream: TokenHandle) -> Result<Vec<TokenChunk>, Response> {
    let mut completes = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(chunk @ TokenChunk::Complete { .. }) => completes.push(chunk),
            Ok(TokenChunk::Partial { .. }) => continue,
            Err(e) => return Err(engine_error_to_response(e)),
        }
    }
    stream.mark_completed();
    Ok(completes)
}

#[allow(clippy::too_many_arguments)]
async fn process_choice(
    complete: &TokenChunk,
    index: u32,
    chat_request: &ChatCompletionRequest,
    tokenizer: &Arc<dyn Tokenizer>,
    stop_decoder: &mut StopSequenceDecoder,
    history_tool_calls_count: usize,
    reasoning_available: bool,
    tool_available: bool,
    tool_parser_factory: Option<&ToolParserFactory>,
    reasoning_parser_factory: Option<&ReasoningParserFactory>,
    configured_tool_parser: Option<&str>,
    configured_reasoning_parser: Option<&str>,
    prompt_opener: Option<&str>,
    reads_as_atom: bool,
) -> Result<ChatChoice, String> {
    let (token_ids, finish_reason, matched_stop, logprobs) = match complete {
        TokenChunk::Complete {
            token_ids,
            finish_reason,
            matched_stop,
            logprobs,
            ..
        } => (token_ids, finish_reason, matched_stop, logprobs.as_ref()),
        TokenChunk::Partial { .. } => return Err("expected Complete chunk".to_string()),
    };

    stop_decoder.reset();
    let outputs = stop_decoder
        .process_tokens(token_ids)
        .map_err(|e| format!("Failed to process tokens: {}", e))?;

    // The decoded pieces, as a streamed reply would carry them.
    let mut pieces = Vec::new();
    for output in outputs {
        match output {
            SequenceDecoderOutput::Text(t) => pieces.push(t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                pieces.push(t);
                break;
            }
            SequenceDecoderOutput::Stopped => break,
            SequenceDecoderOutput::Held => {}
        }
    }
    if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
        pieces.push(t);
    }
    let final_text = pieces.concat();

    let mut reasoning_text: Option<String> = None;
    let mut processed_text = final_text;

    if reasoning_available {
        if let Some(factory) = reasoning_parser_factory {
            let result = split_reasoning(
                factory,
                configured_reasoning_parser,
                &chat_request.model,
                &pieces,
                &processed_text,
                prompt_opener,
            )
            .await
            .map_err(|e| format!("Reasoning parsing error: {}", e))?;
            if !result.reasoning_text.is_empty() {
                reasoning_text = Some(result.reasoning_text);
            }
            processed_text = result.normal_text;
        }
    }

    let mut tool_calls: Option<Vec<ToolCall>> = None;
    let tool_choice_enabled = !matches!(
        &chat_request.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    );

    if tool_choice_enabled && chat_request.tools.is_some() {
        // ATOM derives a forced choice's format itself: the model's own tool syntax.
        let used_json_schema = !reads_as_atom
            && match &chat_request.tool_choice {
                Some(ToolChoice::Function { .. }) => true,
                Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
                Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
                _ => false,
            };

        if used_json_schema {
            (tool_calls, processed_text) = parse_json_schema_response(
                &processed_text,
                &chat_request.tool_choice,
                &chat_request.model,
                history_tool_calls_count,
            );
        } else if tool_available {
            if let Some(factory) = tool_parser_factory {
                let tools = chat_request.tools.as_deref().unwrap_or_default();
                // An ATOM reply cut off inside a native call is read with the call closed.
                let closers = match reads_as_atom {
                    true => {
                        let model = &chat_request.model;
                        let (text, configured) = (&processed_text, configured_tool_parser);
                        cut_off_call_closers(factory, configured, model, text, tools).await
                    }
                    false => None,
                };
                let closed = closers.map(|closers| format!("{processed_text}{closers}"));
                let parse = |text: String| {
                    let model = &chat_request.model;
                    async move {
                        let parsed = parse_tool_calls(
                            factory,
                            configured_tool_parser,
                            &text,
                            tools,
                            model,
                            history_tool_calls_count,
                            reads_as_atom,
                        )
                        .await;
                        parsed
                    }
                };
                let mut parsed = None;
                if let Some(closed) = closed {
                    parsed = Some(parse(closed).await).filter(|(calls, _)| calls.is_some());
                }
                (tool_calls, processed_text) = match parsed {
                    Some(parsed) => parsed,
                    None => parse(std::mem::take(&mut processed_text)).await,
                };
            }
        }
    }

    let finish_reason_str = finish_reason_to_str(finish_reason);
    // A reply that called a tool and stopped ends in `tool_calls`; one cut at
    // its length keeps `length`, as the streamed reply does.
    let final_finish_reason = if tool_calls.is_some() && finish_reason_str == "stop" {
        "tool_calls"
    } else {
        finish_reason_str
    };

    let matched_stop_value = matched_stop.as_ref().map(|ms| match ms {
        MatchedStop::Str(s) => Value::String(s.clone()),
        MatchedStop::TokenId(t) => Value::Number(serde_json::Number::from(*t)),
    });

    let chat_logprobs = logprobs.map(|lp| token_logprobs_to_chat(lp, tokenizer));

    Ok(ChatChoice {
        index,
        message: ChatCompletionMessage {
            role: "assistant".to_string(),
            // ATOM's HTTP chat answers "" when empty; None leaves the key out.
            content: (reads_as_atom || !processed_text.is_empty()).then_some(processed_text),
            tool_calls,
            reasoning_content: reasoning_text,
        },
        logprobs: chat_logprobs,
        finish_reason: Some(final_finish_reason.to_string()),
        matched_stop: matched_stop_value,
        hidden_states: None,
    })
}

/// Every reasoning parser's `max_buffer_size`: `detect_and_parse_reasoning`
/// refuses a longer text outright.
const ONE_SHOT_MAX_BYTES: usize = 65536;

/// Fed last to the piece-by-piece parse: no reasoning delimiter starts with it.
pub(super) const RELEASE: &str = "\0";

/// A whole reply (`text`, decoded as `pieces`) split into content and reasoning, the
/// prompt's `opener` read first; one over the one-shot limit is parsed piece by piece.
async fn split_reasoning(
    factory: &ReasoningParserFactory,
    configured: Option<&str>,
    model: &str,
    pieces: &[String],
    text: &str,
    opener: Option<&str>,
) -> Result<ParserResult, String> {
    let opened: Cow<str> = match opener {
        Some(opener) => format!("{opener}{text}").into(),
        None => text.into(),
    };
    if opened.len() <= ONE_SHOT_MAX_BYTES {
        let pooled = get_reasoning_parser(factory, configured, model);
        let mut parser = pooled.lock().await;
        return parser
            .detect_and_parse_reasoning(&opened)
            .map_err(|e| e.to_string());
    }
    let Some(mut parser) = create_reasoning_parser(factory, configured, model) else {
        return Ok(ParserResult::normal(text.to_string()));
    };
    // A parser that starts inside reasoning (deepseek_r1) has entered it before any piece.
    let entered = parser.is_in_reasoning();
    let (mut normal, mut reasoning, mut entered) = (String::new(), String::new(), entered);
    for piece in opener.into_iter().chain(pieces.iter().map(String::as_str)) {
        let result = parser
            .parse_reasoning_streaming_incremental(piece)
            .map_err(|e| e.to_string())?;
        entered |= parser.is_in_reasoning() || !result.reasoning_text.is_empty();
        normal.push_str(&result.normal_text);
        reasoning.push_str(&result.reasoning_text);
    }
    // A last piece that could start a delimiter ("<", "</") is held back; a character
    // no delimiter starts with releases it, and is dropped again.
    let held = parser
        .parse_reasoning_streaming_incremental(RELEASE)
        .map_err(|e| e.to_string())?;
    normal.push_str(held.normal_text.trim_end_matches(RELEASE));
    reasoning.push_str(held.reasoning_text.trim_end_matches(RELEASE));
    if !entered {
        // The one-shot parse hands back a reply without reasoning untouched.
        return Ok(ParserResult::normal(text.to_string()));
    }
    // Byte for byte what the model wrote minus the delimiters, as ATOM's HTTP chat
    // splits it, and as the streamed reply carries it.
    Ok(ParserResult::new(normal, reasoning))
}

/// The markup that closes the call `text` was cut off in, as the reply's own tool parser
/// reads one at end of input (`ToolParser::end_of_input_closers`), when there is one.
async fn cut_off_call_closers(
    factory: &ToolParserFactory,
    configured: Option<&str>,
    model: &str,
    text: &str,
    tools: &[Tool],
) -> Option<String> {
    let mut parser = create_reply_tool_parser(factory, configured, model, true)?;
    parser.parse_incremental(text, tools).await.ok()?;
    parser.end_of_input_closers(tools)
}

async fn parse_tool_calls(
    factory: &ToolParserFactory,
    configured: Option<&str>,
    text: &str,
    tools: &[Tool],
    model: &str,
    history_tool_calls_count: usize,
    reads_as_atom: bool,
) -> (Option<Vec<ToolCall>>, String) {
    // An ATOM worker's reply is read by a parser of its own, set to read it as ATOM's.
    let own = reads_as_atom.then(|| create_reply_tool_parser(factory, configured, model, true));
    let pooled = match own.flatten() {
        Some(parser) => Arc::new(tokio::sync::Mutex::new(parser)),
        None => get_tool_parser(factory, configured, model),
    };
    let result = {
        let parser = pooled.lock().await;
        // The declared schemas type each argument, as the streamed reply's parser has them.
        parser.parse_complete_with_tools(text, tools).await
    };
    match result {
        Ok((normal_text, parsed)) => {
            if parsed.is_empty() {
                return (None, normal_text);
            }
            let calls = parsed
                .into_iter()
                .enumerate()
                .map(|(i, tc)| ToolCall {
                    id: generate_tool_call_id(
                        model,
                        &tc.function.name,
                        i,
                        history_tool_calls_count,
                    ),
                    tool_type: "function".to_string(),
                    function: FunctionCallResponse {
                        name: tc.function.name,
                        arguments: Some(tc.function.arguments),
                    },
                })
                .collect();
            (Some(calls), normal_text)
        }
        Err(e) => {
            error!("Tool call parsing error: {}", e);
            (None, text.to_string())
        }
    }
}

fn build_usage(completes: &[TokenChunk]) -> Usage {
    let mut prompt = 0u32;
    let mut completion = 0u32;
    for c in completes {
        if let TokenChunk::Complete { usage, .. } = c {
            prompt += usage.prompt_tokens;
            completion += usage.completion_tokens;
        }
    }
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: prompt + completion,
        completion_tokens_details: None,
    }
}

fn finish_reason_to_str(r: &FinishReason) -> &'static str {
    match r {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::Abort => "abort",
        FinishReason::Other(_) => "stop",
    }
}

fn engine_error_to_response(err: EngineError) -> Response {
    match err {
        EngineError::Transport(s) => {
            error::service_unavailable("engine_transport_error", format!("transport error: {}", s))
        }
        EngineError::Prefill(m) => {
            error::internal_error("engine_prefill_error", format!("prefill error: {}", m))
        }
        EngineError::DecodeError(m) => {
            error::internal_error("engine_decode_error", format!("decode error: {}", m))
        }
        EngineError::PrefillEarlyClose => error::internal_error(
            "engine_prefill_early_close",
            "prefill stream closed without Complete",
        ),
        EngineError::DecodeIncomplete => error::internal_error(
            "engine_decode_incomplete",
            "decode stream closed without Complete",
        ),
        EngineError::ConnectionAcquireFailed(r) => {
            error::service_unavailable("engine_connection_failed", r)
        }
        EngineError::RequestBuildFailed(r) => error::bad_request("engine_request_build_failed", r),
        EngineError::Http { status, message } => {
            error::create_error(status, "engine_error", message)
        }
    }
}

#[cfg(test)]
mod split_reasoning_tests {
    use super::*;

    /// A reply as the detokenizer hands it over: text a few bytes at a time,
    /// a special token (`</think>`) whole.
    fn pieces(parts: &[&str]) -> Vec<String> {
        parts
            .iter()
            .flat_map(|part| {
                if part.starts_with('<') {
                    vec![part.to_string()]
                } else {
                    part.as_bytes()
                        .chunks(7)
                        .map(|c| String::from_utf8(c.to_vec()).unwrap())
                        .collect()
                }
            })
            .collect()
    }

    async fn split(parts: &[&str], opener: Option<&str>) -> ParserResult {
        split_by("qwen3", parts, opener).await
    }

    async fn split_by(parser: &str, parts: &[&str], opener: Option<&str>) -> ParserResult {
        let factory = ReasoningParserFactory::new();
        let text = parts.concat();
        split_reasoning(&factory, Some(parser), "m", &pieces(parts), &text, opener)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn a_long_reply_closing_an_empty_implicit_reasoning_block_drops_the_marker() {
        // deepseek_r1 starts inside reasoning: its first piece closes an empty block.
        for len in [100, ONE_SHOT_MAX_BYTES + 1000] {
            let answer = "a".repeat(len);
            let result = split_by("deepseek_r1", &["</think>", &answer], None).await;
            assert!(result.reasoning_text.is_empty(), "len {len}");
            assert_eq!(result.normal_text, answer, "len {len}");
        }
    }

    #[tokio::test]
    async fn a_reply_over_the_one_shot_limit_splits_as_a_short_one_does() {
        for len in [100, ONE_SHOT_MAX_BYTES + 1000] {
            let thought = "x".repeat(len);
            let result = split(&["<think>", &thought, "</think>", "The answer."], None).await;
            assert_eq!(result.reasoning_text, thought, "len {len}");
            assert_eq!(result.normal_text, "The answer.", "len {len}");
        }
    }

    #[tokio::test]
    async fn a_prompt_that_opened_reasoning_has_its_opener_read_first() {
        for len in [100, ONE_SHOT_MAX_BYTES + 1000] {
            let thought = "y".repeat(len);
            let result = split(&[&thought, "</think>", "Done."], Some("<think>")).await;
            assert_eq!(result.reasoning_text, thought, "len {len}");
            assert_eq!(result.normal_text, "Done.", "len {len}");
        }
        // Without the opener the same reply has no reasoning to find.
        let result = split(&["plain", "</think>", "Done."], None).await;
        assert!(result.reasoning_text.is_empty());
    }

    #[tokio::test]
    async fn a_long_reply_keeps_its_answer_s_whitespace() {
        let thought = "z".repeat(ONE_SHOT_MAX_BYTES + 1000);
        let result = split(&["<think>", &thought, "</think>", "    print(1)\n"], None).await;
        assert_eq!(result.reasoning_text, thought);
        assert_eq!(result.normal_text, "    print(1)\n");
    }

    #[tokio::test]
    async fn a_reply_ending_in_a_delimiter_s_first_characters_keeps_them() {
        for len in [100, ONE_SHOT_MAX_BYTES + 1000] {
            let thought = "z".repeat(len);
            // Generation ended inside reasoning, then after it, on a lone "<" or "</".
            let result = split(&["<think>", &thought, "<"], None).await;
            assert_eq!(result.reasoning_text, format!("{thought}<"), "len {len}");
            assert!(result.normal_text.is_empty(), "len {len}");
            let result = split(&["<think>", &thought, "</think>", "3 ", "</"], None).await;
            assert_eq!(result.reasoning_text, thought, "len {len}");
            assert_eq!(result.normal_text, "3 </", "len {len}");
        }
    }
}
