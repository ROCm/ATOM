//! Stream a `TokenHandle<TokenChunk>` as Server-Sent Events for
//! `ChatCompletionStreamResponse`.

use std::{io, sync::Arc, time::Instant};

use super::chat_aggregator::RELEASE;
use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use futures::StreamExt;
use http::header::{HeaderValue, CONTENT_TYPE};
use serde_json::{json, Value};
use tokio::sync::{mpsc, mpsc::UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, warn};

use crate::{
    observability::metrics::{metrics_labels, MeshMetrics, StreamingMetricsParams},
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionStreamResponse},
        common::{FunctionCallDelta, Tool, ToolCallDelta, ToolChoice, ToolChoiceValue, Usage},
    },
    reasoning_parser::{
        ParseError, ParserFactory as ReasoningParserFactory, ParserResult, ReasoningParser,
    },
    routers::{
        prepare::{
            parser_factory_lookup::{check_reasoning_parser_availability, create_reasoning_parser},
            response_context::{ProtocolRequest, ResponseContext},
            tool_constraints::{generate_tool_call_id, get_history_tool_calls_count},
        },
        render::{
            atom_usage::{self, AtomUsage},
            logprob_conversion::token_logprobs_to_chat,
        },
        token_handle::{
            engine_error::EngineError,
            token_chunk::{FinishReason, MatchedStop, TokenChunk, TokenLogprobs},
            token_handle::TokenHandle,
        },
    },
    tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder},
    tool_parser::{
        registry::{check_tool_parser_availability, create_reply_tool_parser},
        ParserFactory as ToolParserFactory, StreamingParseResult, ToolParser,
    },
};

pub(crate) struct ChatStreamConfig {
    pub backend_label: &'static str,
}

pub fn process(stream: TokenHandle, ctx: ResponseContext, backend_label: &'static str) -> Response {
    let chat_request = match &ctx.original {
        ProtocolRequest::Chat(r) => Arc::clone(r),
        ProtocolRequest::Generate(_) => {
            return build_error_sse("chat_streaming invoked with generate request");
        }
    };

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
    let cfg = ChatStreamConfig { backend_label };

    tokio::spawn(async move {
        let result = tokio::select! {
            result = run_chat_stream(stream, ctx, chat_request, cfg, &tx) => result,
            // The client left: dropping the stream ends its call and frees its loads.
            _ = tx.closed() => return,
        };
        if let Err(e) = result {
            let payload = json!({"error": {"message": e, "type": "internal_error"}});
            let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", payload))));
        }
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    build_sse_response(rx)
}

async fn run_chat_stream(
    mut stream: TokenHandle,
    ctx: ResponseContext,
    chat_request: Arc<ChatCompletionRequest>,
    cfg: ChatStreamConfig,
    tx: &UnboundedSender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    let clock = stream.clock();
    let start_time = clock.map_or_else(Instant::now, |c| c.started);
    let mut first_token_time: Option<Instant> = clock.and_then(|c| c.first_token);

    let separate_reasoning = chat_request.separate_reasoning;
    let tool_choice = &chat_request.tool_choice;
    let tools = &chat_request.tools;
    let history_tool_calls_count = get_history_tool_calls_count(&chat_request);
    let stream_options = &chat_request.stream_options;

    let request_id = ctx.request_id.as_str();
    let model = ctx
        .model_id
        .as_deref()
        .unwrap_or(chat_request.model.as_str());
    let created = ctx.created;
    let mut system_fingerprint: Option<String> = None;

    let reasoning_parser_available = separate_reasoning
        && ctx
            .reasoning_parser_factory
            .as_ref()
            .map(|f| {
                check_reasoning_parser_availability(
                    f,
                    ctx.configured_reasoning_parser.as_deref(),
                    model,
                )
            })
            .unwrap_or(false);

    // ATOM derives a forced choice's format itself: the model's own tool syntax.
    let used_json_schema = !ctx.reads_as_atom
        && match tool_choice {
            Some(ToolChoice::Function { .. }) => true,
            Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
            Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
            _ => false,
        };
    let is_specific_function =
        used_json_schema && matches!(tool_choice, Some(ToolChoice::Function { .. }));

    let tool_parser_available = tools.is_some()
        && ctx
            .tool_parser_factory
            .as_ref()
            .map(|f| {
                check_tool_parser_availability(f, ctx.configured_tool_parser.as_deref(), model)
            })
            .unwrap_or(false);

    if separate_reasoning && !reasoning_parser_available {
        debug!("No reasoning parser found for model '{}'", model);
    }
    if tools.is_some() && !tool_parser_available {
        debug!("No tool parser found for model '{}'", model);
    }

    const INDEX: u32 = 0;
    let mut is_first = true;
    let mut finish_reason_str: Option<String> = None;
    let mut matched_stop_value: Option<Value> = None;
    let mut prompt_tokens: u32 = 0;
    let mut completion_tokens: u32 = 0;
    let mut atom_usage: Option<AtomUsage> = None;
    let mut has_tool_call = false;
    let include_usage = stream_options
        .as_ref()
        .is_some_and(|opts| opts.include_usage.unwrap_or(false));
    // As ATOM's HTTP chat_finish_frames: on the finish chunk, or a usage-only chunk after it.
    let usage_on_finish = ctx.usage_on_finish;
    let null_usage = usage_on_finish && include_usage;

    let mut stop_decoder = ctx.stop_decoder;

    let mut reasoning_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>> = None;
    let mut tool_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>> = None;
    let mut sse_buffer = Vec::with_capacity(512);

    let mut consumed = 0;
    let mut held: Option<TokenChunk> = None;
    loop {
        let chunk = match held.take() {
            Some(chunk) => chunk,
            None => match stream.next().await {
                Some(item) => item.map_err(engine_error_to_string)?,
                None => break,
            },
        };
        // A Complete's ids past the partials' are streamed first, as one more partial.
        // In pieces, so no delta outgrows the reasoning and tool parsers' buffers.
        let chunk = match unstreamed(&chunk, consumed, SUFFIX_PIECE) {
            Some(rest) => {
                let logprobs = unstreamed_logprobs(&chunk, consumed, rest.len());
                held = Some(chunk);
                TokenChunk::Partial {
                    token_ids: rest,
                    logprobs,
                }
            }
            None => chunk,
        };
        if let TokenChunk::Partial { token_ids, .. } = &chunk {
            consumed += token_ids.len();
        }
        if is_first && ctx.reads_as_atom {
            // On the engine's first message, text or not, as ATOM's HTTP chat opens its stream.
            let first_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .add_choice_content(INDEX, "assistant", "")
                .maybe_system_fingerprint(system_fingerprint.clone())
                .build();
            format_sse_chunk_into(&mut sse_buffer, &first_chunk, null_usage);
            send_bytes(tx, &sse_buffer, "first chunk")?;
            is_first = false;
        }
        match chunk {
            TokenChunk::Partial {
                token_ids,
                logprobs,
            } => {
                if first_token_time.is_none() {
                    first_token_time = Some(Instant::now());
                }
                completion_tokens += token_ids.len() as u32;

                let (pieces, _stopped) = process_chunk_pieces(&mut stop_decoder, &token_ids);
                let chunk_text = pieces.concat();
                if chunk_text.is_empty() {
                    continue;
                }

                if is_first {
                    let first_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_role(INDEX, "assistant")
                        .maybe_system_fingerprint(system_fingerprint.clone())
                        .build();
                    format_sse_chunk_into(&mut sse_buffer, &first_chunk, null_usage);
                    send_bytes(tx, &sse_buffer, "first chunk")?;
                    is_first = false;
                }

                let mut delta = chunk_text;

                let in_reasoning = if reasoning_parser_available {
                    let (normal_text, reasoning_chunk, in_reasoning) = process_reasoning_stream(
                        &pieces,
                        &mut reasoning_parser,
                        ctx.reasoning_parser_factory.as_ref(),
                        ctx.configured_reasoning_parser.as_deref(),
                        ctx.prompt_opener.as_deref(),
                        request_id,
                        model,
                        created,
                        system_fingerprint.as_deref(),
                    )
                    .await;
                    if let Some(c) = reasoning_chunk {
                        format_sse_chunk_into(&mut sse_buffer, &c, null_usage);
                        send_bytes(tx, &sse_buffer, "reasoning chunk")?;
                    }
                    delta = normal_text;
                    in_reasoning
                } else {
                    false
                };

                let tool_choice_enabled =
                    !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));
                if !in_reasoning
                    && tool_choice_enabled
                    && tools.is_some()
                    && (tool_parser_available || used_json_schema)
                {
                    let tool_chunks = if is_specific_function {
                        process_specific_function_stream(
                            &delta,
                            &mut has_tool_call,
                            tool_choice,
                            request_id,
                            model,
                            created,
                            system_fingerprint.as_deref(),
                            history_tool_calls_count,
                        )
                    } else {
                        process_tool_calls_stream(
                            &delta,
                            &mut tool_parser,
                            &mut has_tool_call,
                            tools.as_ref().unwrap(),
                            ctx.tool_parser_factory.as_ref(),
                            ctx.configured_tool_parser.as_deref(),
                            request_id,
                            model,
                            created,
                            system_fingerprint.as_deref(),
                            history_tool_calls_count,
                            used_json_schema,
                            ctx.reads_as_atom,
                        )
                        .await
                    };

                    for c in tool_chunks {
                        format_sse_chunk_into(&mut sse_buffer, &c, null_usage);
                        send_bytes(tx, &sse_buffer, "tool call chunk")?;
                    }
                    continue;
                }

                if !delta.is_empty() {
                    let chat_logprobs = logprobs
                        .as_ref()
                        .map(|lp| token_logprobs_to_chat(lp, &ctx.tokenizer));
                    let content_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_content_with_logprobs(INDEX, "assistant", delta, chat_logprobs)
                        .maybe_system_fingerprint(system_fingerprint.clone())
                        .build();
                    format_sse_chunk_into(&mut sse_buffer, &content_chunk, null_usage);
                    send_bytes(tx, &sse_buffer, "content chunk")?;
                }
            }
            TokenChunk::Complete {
                finish_reason,
                matched_stop,
                usage,
                meta,
                ..
            } => {
                let mut text = match stop_decoder.flush() {
                    SequenceDecoderOutput::Text(text) => text,
                    _ => String::new(),
                };
                // The reasoning parser holds a last "<" or "</"; RELEASE frees it.
                if let Some(parser) = &reasoning_parser {
                    let released = parser
                        .lock()
                        .await
                        .parse_reasoning_streaming_incremental(&format!("{text}{RELEASE}"));
                    if let Ok(released) = released {
                        let reasoning = released.reasoning_text.trim_end_matches(RELEASE);
                        if !reasoning.is_empty() {
                            let chunk = ChatCompletionStreamResponse::builder(request_id, model)
                                .created(created)
                                .add_choice_reasoning(INDEX, reasoning.to_string())
                                .maybe_system_fingerprint(system_fingerprint.clone())
                                .build();
                            format_sse_chunk_into(&mut sse_buffer, &chunk, null_usage);
                            send_bytes(tx, &sse_buffer, "released reasoning")?;
                        }
                        text = released.normal_text.trim_end_matches(RELEASE).to_string();
                    }
                }
                let tool_choice_enabled =
                    !matches!(tool_choice, Some(ToolChoice::Value(ToolChoiceValue::None)));
                // Read as a partial's text is: it may end a tool call.
                if !text.is_empty()
                    && tool_choice_enabled
                    && tools.is_some()
                    && (tool_parser_available || used_json_schema)
                {
                    let tool_chunks = if is_specific_function {
                        process_specific_function_stream(
                            &text,
                            &mut has_tool_call,
                            tool_choice,
                            request_id,
                            model,
                            created,
                            system_fingerprint.as_deref(),
                            history_tool_calls_count,
                        )
                    } else {
                        process_tool_calls_stream(
                            &text,
                            &mut tool_parser,
                            &mut has_tool_call,
                            tools.as_ref().unwrap(),
                            ctx.tool_parser_factory.as_ref(),
                            ctx.configured_tool_parser.as_deref(),
                            request_id,
                            model,
                            created,
                            system_fingerprint.as_deref(),
                            history_tool_calls_count,
                            used_json_schema,
                            ctx.reads_as_atom,
                        )
                        .await
                    };
                    for c in tool_chunks {
                        format_sse_chunk_into(&mut sse_buffer, &c, null_usage);
                        send_bytes(tx, &sse_buffer, "flushed tool call chunk")?;
                    }
                } else if !text.is_empty() {
                    let chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_content(INDEX, "assistant", text)
                        .maybe_system_fingerprint(system_fingerprint.clone())
                        .build();
                    format_sse_chunk_into(&mut sse_buffer, &chunk, null_usage);
                    send_bytes(tx, &sse_buffer, "flushed content")?;
                }

                prompt_tokens = usage.prompt_tokens;
                completion_tokens = usage.completion_tokens;
                atom_usage = AtomUsage::of([&meta]);
                finish_reason_str = Some(finish_reason_to_string(&finish_reason));
                matched_stop_value = matched_stop.map(|ms| match ms {
                    MatchedStop::Str(s) => Value::String(s),
                    MatchedStop::TokenId(t) => Value::Number(serde_json::Number::from(t)),
                });
                if system_fingerprint.is_none() {
                    system_fingerprint = meta.weight_version;
                }
            }
        }
    }

    // An ATOM reply cut off inside a native call has the call closed, as its parser
    // reads one at end of input.
    if let (Some(parser), Some(tools), true) = (&tool_parser, tools.as_ref(), ctx.reads_as_atom) {
        let closers = parser.lock().await.end_of_input_closers(tools);
        if let Some(closers) = closers {
            let chunks = process_tool_calls_stream(
                &closers,
                &mut tool_parser,
                &mut has_tool_call,
                tools,
                ctx.tool_parser_factory.as_ref(),
                ctx.configured_tool_parser.as_deref(),
                request_id,
                model,
                created,
                system_fingerprint.as_deref(),
                history_tool_calls_count,
                used_json_schema,
                ctx.reads_as_atom,
            )
            .await;
            for c in chunks {
                format_sse_chunk_into(&mut sse_buffer, &c, null_usage);
                send_bytes(tx, &sse_buffer, "cut-off tool call")?;
            }
        }
        // What it held as the start of a call that never came is text.
        let held = match &tool_parser {
            Some(parser) => parser.lock().await.end_of_input_text(),
            None => String::new(),
        };
        if !held.is_empty() {
            let chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .add_choice_content(INDEX, "assistant", held)
                .maybe_system_fingerprint(system_fingerprint.clone())
                .build();
            format_sse_chunk_into(&mut sse_buffer, &chunk, null_usage);
            send_bytes(tx, &sse_buffer, "held text")?;
        }
    }
    if let Some(parser) = &tool_parser {
        let guard = parser.lock().await;
        if let Some(unstreamed) = guard.get_unstreamed_tool_args() {
            for item in unstreamed {
                let delta = ToolCallDelta {
                    index: item.tool_index as u32,
                    id: None,
                    tool_type: None,
                    function: Some(FunctionCallDelta {
                        name: None,
                        arguments: if item.parameters.is_empty() {
                            None
                        } else {
                            Some(item.parameters)
                        },
                    }),
                };
                let chunk = ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_tool_call_delta(INDEX, delta)
                    .maybe_system_fingerprint(system_fingerprint.clone())
                    .build();
                format_sse_chunk_into(&mut sse_buffer, &chunk, null_usage);
                send_bytes(tx, &sse_buffer, "unstreamed tool args")?;
            }
        }
    }

    let usage = Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        completion_tokens_details: None,
    };
    // The chunk that carries the usage, with ATOM's fields (the cached tokens) for ATOM's.
    let with_usage = |chunk: &ChatCompletionStreamResponse| match &atom_usage {
        None => serde_json::to_string(chunk),
        Some(atom) => atom
            .extend_response(chunk, atom_usage::CHAT_STREAM)
            .map(|v| v.to_string()),
    };
    if let Some(fr) = finish_reason_str {
        let final_fr = if has_tool_call && fr == "stop" {
            "tool_calls".to_string()
        } else {
            fr
        };
        let chunk = ChatCompletionStreamResponse::builder(request_id, model)
            .created(created)
            .add_choice_finish_reason(INDEX, final_fr, matched_stop_value)
            .maybe_system_fingerprint(system_fingerprint.clone())
            .maybe_usage(usage_on_finish.then(|| usage.clone()))
            .build();
        let s =
            with_usage(&chunk).map_err(|e| format!("Failed to serialize finish chunk: {}", e))?;
        tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
            .map_err(|_| "send finish".to_string())?;
    }

    // ATOM's HTTP chat ends every stream with its usage chunk unless the finish chunk
    // carries it (`chat_finish_frames`); SGLang's and vLLM's only under include_usage.
    if !usage_on_finish && (include_usage || atom_usage.is_some()) {
        let usage_chunk = ChatCompletionStreamResponse::builder(request_id, model)
            .created(created)
            .usage(usage)
            .maybe_system_fingerprint(system_fingerprint.clone())
            .build();
        let s = with_usage(&usage_chunk)
            .map_err(|e| format!("Failed to serialize usage chunk: {}", e))?;
        tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
            .map_err(|_| "send usage".to_string())?;
    }

    stream.mark_completed();

    MeshMetrics::record_streaming_metrics(StreamingMetricsParams {
        router_type: metrics_labels::ROUTER_GRPC,
        backend_type: cfg.backend_label,
        model_id: model,
        endpoint: metrics_labels::ENDPOINT_CHAT,
        ttft: first_token_time.map(|t| t.duration_since(start_time)),
        generation_duration: start_time.elapsed(),
        input_tokens: Some(prompt_tokens as u64),
        output_tokens: completion_tokens as u64,
    });

    Ok(())
}

/// The most ids of a Complete's unstreamed suffix sent as one delta.
const SUFFIX_PIECE: usize = 256;

/// The first `limit` ids of a Complete past the `consumed` its partials streamed:
/// SGLang's and vLLM's carry every output id, ATOM's streamed one none.
pub(super) fn unstreamed(chunk: &TokenChunk, consumed: usize, limit: usize) -> Option<Vec<u32>> {
    match chunk {
        TokenChunk::Complete { token_ids, .. } => token_ids
            .get(consumed..)
            .filter(|rest| !rest.is_empty())
            .map(|rest| rest[..rest.len().min(limit)].to_vec()),
        TokenChunk::Partial { .. } => None,
    }
}

/// The logprobs of those ids, where the Complete's cover every id it carries.
pub(super) fn unstreamed_logprobs(
    chunk: &TokenChunk,
    consumed: usize,
    limit: usize,
) -> Option<TokenLogprobs> {
    match chunk {
        TokenChunk::Complete {
            token_ids,
            logprobs: Some(logprobs),
            ..
        } if logprobs.items.len() == token_ids.len() => Some(TokenLogprobs {
            items: logprobs
                .items
                .get(consumed..)?
                .iter()
                .take(limit)
                .cloned()
                .collect(),
        }),
        _ => None,
    }
}

pub(super) fn process_chunk_tokens(
    stop_decoder: &mut StopSequenceDecoder,
    token_ids: &[u32],
) -> (String, bool) {
    let mut text = String::new();
    let stopped = decode_chunk(stop_decoder, token_ids, |piece| text.push_str(&piece));
    (text, stopped)
}

/// `process_chunk_tokens`, the text in the pieces the decoder wrote it in.
fn process_chunk_pieces(
    stop_decoder: &mut StopSequenceDecoder,
    token_ids: &[u32],
) -> (Vec<String>, bool) {
    let mut pieces = Vec::new();
    let stopped = decode_chunk(stop_decoder, token_ids, |piece| pieces.push(piece));
    (pieces, stopped)
}

/// Decodes `token_ids`, handing each piece of text to `write`; whether a stop was hit.
fn decode_chunk(
    stop_decoder: &mut StopSequenceDecoder,
    token_ids: &[u32],
    mut write: impl FnMut(String),
) -> bool {
    for &token_id in token_ids {
        match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
            debug!(
                "Error processing token {}: {}. Treating as Held.",
                token_id, e
            );
            SequenceDecoderOutput::Held
        }) {
            SequenceDecoderOutput::Text(t) => write(t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                write(t);
                return true;
            }
            SequenceDecoderOutput::Stopped => return true,
            SequenceDecoderOutput::Held => {}
        }
    }
    false
}

#[allow(clippy::too_many_arguments)]
async fn process_reasoning_stream(
    pieces: &[String],
    reasoning_parser: &mut Option<Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>>,
    factory: Option<&ReasoningParserFactory>,
    configured: Option<&str>,
    prompt_opener: Option<&str>,
    request_id: &str,
    model: &str,
    created: u64,
    system_fingerprint: Option<&str>,
) -> (String, Option<ChatCompletionStreamResponse>, bool) {
    if reasoning_parser.is_none() {
        let factory = match factory {
            Some(f) => f,
            None => return (pieces.concat(), None, false),
        };
        match create_reasoning_parser(factory, configured, model) {
            Some(mut p) => {
                if let Some(opener) = prompt_opener {
                    // The prompt opened the block: the parser reads its
                    // opener before the reply, which has none.
                    let _ = p.parse_reasoning_streaming_incremental(opener);
                }
                *reasoning_parser = Some(Arc::new(tokio::sync::Mutex::new(p)));
            }
            None => return (pieces.concat(), None, false),
        }
    }

    let parser_arc = match reasoning_parser {
        Some(p) => p.clone(),
        None => return (pieces.concat(), None, false),
    };
    let (parse_result, in_reasoning) = {
        let mut parser = parser_arc.lock().await;
        // Piece by piece, as the decoder wrote them: the parser holds a delimiter's start
        // only when that is all it is given, so a piece of ids may end inside one.
        let result = pieces
            .iter()
            .try_fold(ParserResult::default(), |mut read, piece| {
                let piece = parser.parse_reasoning_streaming_incremental(piece)?;
                read.normal_text.push_str(&piece.normal_text);
                read.reasoning_text.push_str(&piece.reasoning_text);
                Ok::<_, ParseError>(read)
            });
        let in_reasoning = parser.is_in_reasoning();
        (result, in_reasoning)
    };
    match parse_result {
        Ok(ParserResult {
            reasoning_text,
            normal_text,
        }) => {
            let chunk = if !reasoning_text.is_empty() {
                Some(
                    ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_reasoning(0, reasoning_text)
                        .maybe_system_fingerprint(system_fingerprint.map(|s| s.to_string()))
                        .build(),
                )
            } else {
                None
            };
            (normal_text, chunk, in_reasoning)
        }
        Err(e) => {
            warn!("Reasoning parsing error: {}", e);
            (pieces.concat(), None, false)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_specific_function_stream(
    delta: &str,
    has_tool_call: &mut bool,
    tool_choice: &Option<ToolChoice>,
    request_id: &str,
    model: &str,
    created: u64,
    system_fingerprint: Option<&str>,
    history_tool_calls_count: usize,
) -> Vec<ChatCompletionStreamResponse> {
    let mut chunks = Vec::new();
    if let Some(ToolChoice::Function { function, .. }) = tool_choice {
        if !*has_tool_call {
            *has_tool_call = true;
            let id = generate_tool_call_id(model, &function.name, 0, history_tool_calls_count);
            chunks.push(
                ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_tool_name(0, id, function.name.clone())
                    .maybe_system_fingerprint(system_fingerprint.map(|s| s.to_string()))
                    .build(),
            );
        }
        if !delta.is_empty() {
            chunks.push(
                ChatCompletionStreamResponse::builder(request_id, model)
                    .created(created)
                    .add_choice_tool_args(0, delta.to_string())
                    .maybe_system_fingerprint(system_fingerprint.map(|s| s.to_string()))
                    .build(),
            );
        }
    }
    chunks
}

#[allow(clippy::too_many_arguments)]
async fn process_tool_calls_stream(
    delta: &str,
    tool_parser: &mut Option<Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>>,
    has_tool_call: &mut bool,
    tools: &[Tool],
    factory: Option<&ToolParserFactory>,
    configured: Option<&str>,
    request_id: &str,
    model: &str,
    created: u64,
    system_fingerprint: Option<&str>,
    history_tool_calls_count: usize,
    use_json_parser: bool,
    reads_as_atom: bool,
) -> Vec<ChatCompletionStreamResponse> {
    let mut chunks = Vec::new();
    if tool_parser.is_none() {
        let factory = match factory {
            Some(f) => f,
            None => return chunks,
        };
        let configured = if use_json_parser {
            Some("json")
        } else {
            configured
        };
        match create_reply_tool_parser(factory, configured, model, reads_as_atom) {
            Some(p) => *tool_parser = Some(Arc::new(tokio::sync::Mutex::new(p))),
            None => return chunks,
        }
    }

    let parser_arc = match tool_parser {
        Some(p) => p.clone(),
        None => return chunks,
    };
    let mut parser = parser_arc.lock().await;
    match parser.parse_incremental(delta, tools).await {
        Ok(StreamingParseResult { normal_text, calls }) => {
            if !normal_text.is_empty() {
                chunks.push(
                    ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_content(0, "assistant", normal_text)
                        .maybe_system_fingerprint(system_fingerprint.map(|s| s.to_string()))
                        .build(),
                );
            }
            for item in calls {
                *has_tool_call = true;
                let id = item.name.as_ref().map(|n| {
                    generate_tool_call_id(model, n, item.tool_index, history_tool_calls_count)
                });
                let delta = ToolCallDelta {
                    index: item.tool_index as u32,
                    id,
                    tool_type: if item.name.is_some() {
                        Some("function".to_string())
                    } else {
                        None
                    },
                    function: Some(FunctionCallDelta {
                        name: item.name,
                        arguments: if item.parameters.is_empty() {
                            None
                        } else {
                            Some(item.parameters)
                        },
                    }),
                };
                chunks.push(
                    ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_tool_call_delta(0, delta)
                        .maybe_system_fingerprint(system_fingerprint.map(|s| s.to_string()))
                        .build(),
                );
            }
        }
        Err(e) => error!("Tool call parsing error: {}", e),
    }
    chunks
}

#[inline]
fn format_sse_chunk_into(
    buffer: &mut Vec<u8>,
    chunk: &ChatCompletionStreamResponse,
    null_usage: bool,
) {
    buffer.clear();
    buffer.extend_from_slice(b"data: ");
    let written = if null_usage && chunk.usage.is_none() {
        // ATOM's `usage: null` on each chunk before the one carrying the usage.
        serde_json::to_value(chunk).and_then(|mut value| {
            value["usage"] = Value::Null;
            serde_json::to_writer(&mut *buffer, &value)
        })
    } else {
        serde_json::to_writer(&mut *buffer, chunk)
    };
    if let Err(e) = written {
        error!("Failed to serialize SSE chunk: {}", e);
        buffer.clear();
        buffer.extend_from_slice(b"data: ");
        let err = json!({"error": "serialization_failed"}).to_string();
        buffer.extend_from_slice(err.as_bytes());
    }
    buffer.extend_from_slice(b"\n\n");
}

fn send_bytes(
    tx: &UnboundedSender<Result<Bytes, io::Error>>,
    buffer: &[u8],
    label: &'static str,
) -> Result<(), String> {
    tx.send(Ok(Bytes::copy_from_slice(buffer)))
        .map_err(|_| format!("Failed to send {}", label))
}

fn build_sse_response(rx: mpsc::UnboundedReceiver<Result<Bytes, io::Error>>) -> Response {
    let stream = UnboundedReceiverStream::new(rx);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
        .headers_mut()
        .insert("Cache-Control", HeaderValue::from_static("no-cache"));
    response
        .headers_mut()
        .insert("Connection", HeaderValue::from_static("keep-alive"));
    response
}

fn build_error_sse(message: &str) -> Response {
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
    let payload = json!({"error": {"message": message, "type": "internal_error"}});
    let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", payload))));
    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    drop(tx);
    build_sse_response(rx)
}

fn finish_reason_to_string(r: &FinishReason) -> String {
    match r {
        FinishReason::Stop => "stop".to_string(),
        FinishReason::Length => "length".to_string(),
        FinishReason::ContentFilter => "content_filter".to_string(),
        FinishReason::ToolCalls => "tool_calls".to_string(),
        FinishReason::Abort => "abort".to_string(),
        FinishReason::Other(s) => s.clone(),
    }
}

fn engine_error_to_string(e: EngineError) -> String {
    format!("{}", e)
}
