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
        common::{FunctionCallResponse, ToolCall, ToolChoice, ToolChoiceValue, Usage},
    },
    reasoning_parser::{
        ParseError, ParserFactory as ReasoningParserFactory, ParserResult, ReasoningParser,
    },
    routers::{
        comm::error,
        prepare::{
            parser_factory_lookup::{
                check_reasoning_parser_availability, create_reasoning_parser, get_reasoning_parser,
            },
            response_context::{ProtocolRequest, ResponseContext, THINK_OPEN},
            tool_constraints::{
                generate_tool_call_id, get_history_tool_calls_count, parse_json_schema_response,
            },
        },
        render::{
            atom_usage::{self, AtomUsage},
            logprob_conversion::token_logprobs_to_chat,
        },
        token_handle::{
            token_chunk::{FinishReason, MatchedStop, TokenChunk},
            token_handle::TokenHandle,
        },
    },
    tokenizer::{
        stop::{SequenceDecoderOutput, StopSequenceDecoder},
        traits::Tokenizer,
    },
    tool_parser::{
        registry::{check_tool_parser_availability, get_tool_parser},
        ParserFactory as ToolParserFactory,
    },
};

pub async fn process(stream: TokenHandle, ctx: ResponseContext) -> Response {
    match aggregate(stream, ctx).await {
        Ok((resp, None)) => axum::Json(resp).into_response(),
        Ok((resp, Some(atom))) => match atom.extend_response(&resp, atom_usage::CHAT) {
            Ok(value) => axum::Json(value).into_response(),
            Err(e) => error::internal_error("serialize_failed", e.to_string()),
        },
        Err(resp) => resp,
    }
}

pub async fn process_typed(
    stream: TokenHandle,
    ctx: ResponseContext,
) -> Result<ChatCompletionResponse, Response> {
    aggregate(stream, ctx).await.map(|(resp, _)| resp)
}

/// The response, and ATOM's usage additions where the backend has them.
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
            ctx.prompt_opens_reasoning,
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
            Err(e) => return Err(e.into_response()),
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
    prompt_opens_reasoning: bool,
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

    let mut final_text = String::new();
    for output in outputs {
        match output {
            SequenceDecoderOutput::Text(t) => final_text.push_str(&t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                final_text.push_str(&t);
                break;
            }
            SequenceDecoderOutput::Stopped => break,
            SequenceDecoderOutput::Held => {}
        }
    }
    if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
        final_text.push_str(&t);
    }

    let mut reasoning_text: Option<String> = None;
    let mut processed_text = final_text;

    if reasoning_available {
        if let Some(factory) = reasoning_parser_factory {
            let result = split_reasoning(
                factory,
                configured_reasoning_parser,
                &chat_request.model,
                &processed_text,
                prompt_opens_reasoning,
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
        let used_json_schema = match &chat_request.tool_choice {
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
                (tool_calls, processed_text) = parse_tool_calls(
                    factory,
                    configured_tool_parser,
                    &processed_text,
                    &chat_request.model,
                    history_tool_calls_count,
                )
                .await;
            }
        }
    }

    let finish_reason_str = finish_reason_to_str(finish_reason);
    let final_finish_reason = if tool_calls.is_some() {
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
            content: if processed_text.is_empty() {
                None
            } else {
                Some(processed_text)
            },
            tool_calls,
            reasoning_content: reasoning_text,
        },
        logprobs: chat_logprobs,
        finish_reason: Some(final_finish_reason.to_string()),
        matched_stop: matched_stop_value,
        hidden_states: None,
    })
}

/// A whole reply split into content and reasoning; `prompt_opens_reasoning`
/// puts the prompt's opener before it.
///
/// `detect_and_parse_reasoning` refuses a text over the parser's
/// `max_buffer_size` (64 KiB, counting an opener the parser adds itself, as
/// MiniMax's does); such a reply is split by [`split_in_pieces`] instead.
async fn split_reasoning(
    factory: &ReasoningParserFactory,
    configured: Option<&str>,
    model: &str,
    text: &str,
    prompt_opens_reasoning: bool,
) -> Result<ParserResult, String> {
    let opener = prompt_opens_reasoning.then_some(THINK_OPEN);
    let opened: Cow<str> = match opener {
        Some(opener) => format!("{opener}{text}").into(),
        None => text.into(),
    };
    {
        let pooled = get_reasoning_parser(factory, configured, model);
        let mut parser = pooled.lock().await;
        match parser.detect_and_parse_reasoning(&opened) {
            Err(ParseError::BufferOverflow(_)) => {}
            result => return result.map_err(|e| e.to_string()),
        }
    }
    match create_reasoning_parser(factory, configured, model) {
        Some(mut parser) => split_in_pieces(parser.as_mut(), &opened),
        None => Ok(ParserResult::normal(text.to_string())),
    }
}

/// The most a piece of [`split_in_pieces`] holds, well under any parser's
/// `max_buffer_size`.
const PIECE_MAX_BYTES: usize = 32 * 1024;

/// A `<` this close to the end with no `>` after it may be a tag cut short:
/// it stays on the piece before it.
const TAG_TAIL_BYTES: usize = 16;

/// `detect_and_parse_reasoning` for a text of any length, through a fresh
/// `parser`'s streaming path a piece at a time.
///
/// Each `<` starts a piece, so the parser reads every tag whole and at the
/// head of its piece: the streaming path trims the reasoning before a closer
/// in the same piece, which here is none. A `<` in the last
/// [`TAG_TAIL_BYTES`] with no `>` after it stays on the piece before, or the
/// parser would hold it back as a tag's prefix and drop it. The result is
/// then trimmed as the one-shot parse trims: reasoning after the opener, to
/// the end if no closer follows, and the content after the closer.
fn split_in_pieces(parser: &mut dyn ReasoningParser, text: &str) -> Result<ParserResult, String> {
    let (mut normal, mut reasoning, mut entered) = (String::new(), String::new(), false);
    for piece in reasoning_pieces(text) {
        let result = parser
            .parse_reasoning_streaming_incremental(piece)
            .map_err(|e| e.to_string())?;
        entered |= parser.is_in_reasoning() || !result.reasoning_text.is_empty();
        normal.push_str(&result.normal_text);
        reasoning.push_str(&result.reasoning_text);
    }
    if !entered {
        // The one-shot parse hands back a reply without reasoning untouched.
        return Ok(ParserResult::normal(text.to_string()));
    }
    let reasoning = if parser.is_in_reasoning() {
        reasoning.trim()
    } else {
        reasoning.trim_start()
    };
    Ok(ParserResult::new(
        normal.trim().to_string(),
        reasoning.to_string(),
    ))
}

/// `text` cut before each `<` (but one in a short unclosed tail), and every
/// piece longer than [`PIECE_MAX_BYTES`] cut again at a char boundary.
fn reasoning_pieces(text: &str) -> Vec<&str> {
    let tail_start = text.len().saturating_sub(TAG_TAIL_BYTES);
    let mut cuts: Vec<usize> = text
        .match_indices('<')
        .map(|(at, _)| at)
        .filter(|&at| at > 0 && (at < tail_start || text[at..].contains('>')))
        .collect();
    cuts.push(text.len());
    let mut pieces = Vec::with_capacity(cuts.len());
    let mut start = 0;
    for cut in cuts {
        while cut - start > PIECE_MAX_BYTES {
            let mut end = start + PIECE_MAX_BYTES;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            pieces.push(&text[start..end]);
            start = end;
        }
        if cut > start {
            pieces.push(&text[start..cut]);
        }
        start = cut;
    }
    pieces
}

async fn parse_tool_calls(
    factory: &ToolParserFactory,
    configured: Option<&str>,
    text: &str,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    let pooled = get_tool_parser(factory, configured, model);
    let result = {
        let parser = pooled.lock().await;
        parser.parse_complete(text).await
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

/// The siblings share one prompt: it is counted once (each sibling reports
/// it), their outputs summed.
fn build_usage(completes: &[TokenChunk]) -> Usage {
    let mut prompt = 0u32;
    let mut completion = 0u32;
    for c in completes {
        if let TokenChunk::Complete { usage, .. } = c {
            prompt = prompt.max(usage.prompt_tokens);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parser(name: &str) -> Box<dyn ReasoningParser> {
        ReasoningParserFactory::new()
            .registry()
            .create_parser(name)
            .unwrap()
    }

    /// Under 64 KiB both parses run: the piecewise one says what the
    /// one-shot one says, tags, whitespace, a dangling `<` and pieces cut
    /// inside multi-byte text alike.
    #[test]
    fn the_piecewise_split_matches_the_one_shot_parse() {
        let long_reasoning = format!(
            "<think>{}</think>\n\n{}",
            "héllo wörld, 1 < 2 ".repeat(2000),
            "answer ".repeat(10)
        );
        let many_tags = format!("<think>{}</think>c <", "a<b> c</b ".repeat(3000));
        let texts = [
            "plain answer",
            "x < y and z > w",
            "<think>a b</think> answer",
            "<think>\n  reasoning with spaces \n</think>\n\n answer  ",
            "<think>unclosed reasoning  ",
            "<think>r</think>answer with <b>html</b> and a < b",
            "<think>r</think>ends with <",
            "<think>r</think>ends with </",
            "<think>r <maybe a tag> more</think> x",
            "<think>word</think>",
            "reasoning only, the opener in the prompt</think>answer",
            &long_reasoning,
            &many_tags,
        ];
        for name in ["qwen3", "glm45", "deepseek_r1", "minimax"] {
            for text in texts {
                assert!(text.len() < 60_000, "{}", text.len());
                let one_shot = parser(name).detect_and_parse_reasoning(text).unwrap();
                let pieces = split_in_pieces(parser(name).as_mut(), text).unwrap();
                let short = &text[..text.len().min(60)];
                assert_eq!(
                    (pieces.normal_text, pieces.reasoning_text),
                    (one_shot.normal_text, one_shot.reasoning_text),
                    "{name}: {short:?}"
                );
            }
        }
    }

    #[test]
    fn pieces_start_at_tags_and_stay_under_the_cap() {
        let text = format!("a<think>b</think>{}<x", "é".repeat(PIECE_MAX_BYTES));
        let pieces = reasoning_pieces(&text);
        assert_eq!(pieces.concat(), text);
        assert_eq!(&pieces[..2], ["a", "<think>b"]);
        assert!(pieces[2].starts_with("</think>é"));
        assert!(pieces.iter().all(|p| p.len() <= PIECE_MAX_BYTES));
        // The unclosed `<x` at the end stays on the piece before it.
        assert!(pieces.last().unwrap().ends_with("é<x"));
    }
}
