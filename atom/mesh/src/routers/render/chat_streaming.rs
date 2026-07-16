//! Stream a `TokenHandle<TokenChunk>` as Server-Sent Events for
//! `ChatCompletionStreamResponse`.

use std::{io, sync::Arc, time::Instant};

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
    reasoning_parser::{ParserFactory as ReasoningParserFactory, ParserResult, ReasoningParser},
    routers::{
        prepare::{
            parser_factory_lookup::{
                check_reasoning_parser_availability, check_tool_parser_availability,
                create_reasoning_parser, create_tool_parser,
            },
            response_context::{ProtocolRequest, ResponseContext},
            tool_constraints::{generate_tool_call_id, get_history_tool_calls_count},
        },
        render::logprob_conversion::token_logprobs_to_chat,
        token_handle::{
            engine_error::EngineError,
            token_chunk::{FinishReason, MatchedStop, TokenChunk},
            token_handle::TokenHandle,
        },
    },
    tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder},
    tool_parser::{ParserFactory as ToolParserFactory, StreamingParseResult, ToolParser},
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
        let result = run_chat_stream(stream, ctx, chat_request, cfg, &tx).await;
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
    let start_time = Instant::now();
    let mut first_token_time: Option<Instant> = None;

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

    let used_json_schema = match tool_choice {
        Some(ToolChoice::Function { .. }) => true,
        Some(ToolChoice::Value(ToolChoiceValue::Required)) => true,
        Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
        _ => false,
    };
    let is_specific_function = matches!(tool_choice, Some(ToolChoice::Function { .. }));

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
    let mut has_tool_call = false;

    let mut stop_decoder = ctx.stop_decoder;

    let mut reasoning_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>> = None;
    let mut tool_parser: Option<Arc<tokio::sync::Mutex<Box<dyn ToolParser>>>> = None;
    let mut sse_buffer = Vec::with_capacity(512);

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(engine_error_to_string)?;
        match chunk {
            TokenChunk::Partial {
                token_ids,
                logprobs,
            } => {
                if first_token_time.is_none() {
                    first_token_time = Some(Instant::now());
                }
                completion_tokens += token_ids.len() as u32;

                let (chunk_text, _stopped) = process_chunk_tokens(&mut stop_decoder, &token_ids);
                if chunk_text.is_empty() {
                    continue;
                }

                if is_first {
                    let first_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                        .created(created)
                        .add_choice_role(INDEX, "assistant")
                        .maybe_system_fingerprint(system_fingerprint.clone())
                        .build();
                    format_sse_chunk_into(&mut sse_buffer, &first_chunk);
                    send_bytes(tx, &sse_buffer, "first chunk")?;
                    is_first = false;
                }

                let mut delta = chunk_text;

                let in_reasoning = if reasoning_parser_available {
                    let (normal_text, reasoning_chunk, in_reasoning) = process_reasoning_stream(
                        &delta,
                        &mut reasoning_parser,
                        ctx.reasoning_parser_factory.as_ref(),
                        ctx.configured_reasoning_parser.as_deref(),
                        request_id,
                        model,
                        created,
                        system_fingerprint.as_deref(),
                    )
                    .await;
                    if let Some(c) = reasoning_chunk {
                        format_sse_chunk_into(&mut sse_buffer, &c);
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
                        )
                        .await
                    };

                    for c in tool_chunks {
                        format_sse_chunk_into(&mut sse_buffer, &c);
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
                    format_sse_chunk_into(&mut sse_buffer, &content_chunk);
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
                if let SequenceDecoderOutput::Text(text) = stop_decoder.flush() {
                    if !text.is_empty() {
                        let chunk = ChatCompletionStreamResponse::builder(request_id, model)
                            .created(created)
                            .add_choice_content(INDEX, "assistant", text)
                            .maybe_system_fingerprint(system_fingerprint.clone())
                            .build();
                        format_sse_chunk_into(&mut sse_buffer, &chunk);
                        send_bytes(tx, &sse_buffer, "flushed content")?;
                    }
                }

                prompt_tokens = usage.prompt_tokens;
                completion_tokens = usage.completion_tokens;
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
                let s = serde_json::to_string(&chunk)
                    .map_err(|e| format!("Failed to serialize tool chunk: {}", e))?;
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
                    .map_err(|_| "send unstreamed".to_string())?;
            }
        }
    }

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
            .build();
        let s = serde_json::to_string(&chunk)
            .map_err(|e| format!("Failed to serialize finish chunk: {}", e))?;
        tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
            .map_err(|_| "send finish".to_string())?;
    }

    if let Some(opts) = stream_options {
        if opts.include_usage.unwrap_or(false) {
            let usage_chunk = ChatCompletionStreamResponse::builder(request_id, model)
                .created(created)
                .usage(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    completion_tokens_details: None,
                })
                .maybe_system_fingerprint(system_fingerprint.clone())
                .build();
            let s = serde_json::to_string(&usage_chunk)
                .map_err(|e| format!("Failed to serialize usage chunk: {}", e))?;
            tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
                .map_err(|_| "send usage".to_string())?;
        }
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

fn process_chunk_tokens(
    stop_decoder: &mut StopSequenceDecoder,
    token_ids: &[u32],
) -> (String, bool) {
    let mut chunk_text = String::new();
    for &token_id in token_ids {
        match stop_decoder.process_token(token_id).unwrap_or_else(|e| {
            debug!(
                "Error processing token {}: {}. Treating as Held.",
                token_id, e
            );
            SequenceDecoderOutput::Held
        }) {
            SequenceDecoderOutput::Text(t) => chunk_text.push_str(&t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                chunk_text.push_str(&t);
                return (chunk_text, true);
            }
            SequenceDecoderOutput::Stopped => return (chunk_text, true),
            SequenceDecoderOutput::Held => {}
        }
    }
    (chunk_text, false)
}

#[allow(clippy::too_many_arguments)]
async fn process_reasoning_stream(
    delta: &str,
    reasoning_parser: &mut Option<Arc<tokio::sync::Mutex<Box<dyn ReasoningParser>>>>,
    factory: Option<&ReasoningParserFactory>,
    configured: Option<&str>,
    request_id: &str,
    model: &str,
    created: u64,
    system_fingerprint: Option<&str>,
) -> (String, Option<ChatCompletionStreamResponse>, bool) {
    if reasoning_parser.is_none() {
        let factory = match factory {
            Some(f) => f,
            None => return (delta.to_string(), None, false),
        };
        match create_reasoning_parser(factory, configured, model) {
            Some(p) => *reasoning_parser = Some(Arc::new(tokio::sync::Mutex::new(p))),
            None => return (delta.to_string(), None, false),
        }
    }

    let parser_arc = match reasoning_parser {
        Some(p) => p.clone(),
        None => return (delta.to_string(), None, false),
    };
    let (parse_result, in_reasoning) = {
        let mut parser = parser_arc.lock().await;
        let result = parser.parse_reasoning_streaming_incremental(delta);
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
            (delta.to_string(), None, false)
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
        match create_tool_parser(factory, configured, model) {
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
fn format_sse_chunk_into(buffer: &mut Vec<u8>, chunk: &ChatCompletionStreamResponse) {
    buffer.clear();
    buffer.extend_from_slice(b"data: ");
    if let Err(e) = serde_json::to_writer(&mut *buffer, chunk) {
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
