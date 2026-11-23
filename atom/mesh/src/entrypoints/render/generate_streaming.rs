//! Stream a `TokenHandle<TokenChunk>` as SGLang-style SSE for the
//! `/generate` endpoint.

use std::{io, sync::Arc, time::Instant};

use axum::{body::Body, http::StatusCode, response::Response};
use bytes::Bytes;
use futures::StreamExt;
use http::header::{HeaderValue, CONTENT_TYPE};
use serde_json::json;
use tokio::sync::{mpsc, mpsc::UnboundedSender};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::error;

use crate::{
    observability::metrics::{metrics_labels, MeshMetrics, StreamingMetricsParams},
    protocols::generate::{GenerateFinishReason, GenerateRequest},
    routers::{
        prepare::response_context::{ProtocolRequest, ResponseContext},
        render::logprob_conversion::{input_logprobs_to_generate, output_logprobs_to_generate},
        token_handle::{
            engine_error::EngineError,
            token_chunk::{FinishReason, TokenChunk},
            token_handle::TokenHandle,
        },
    },
    tokenizer::traits::Tokenizer,
};

pub(crate) struct GenerateStreamConfig {
    pub backend_label: &'static str,
}

pub fn process(stream: TokenHandle, ctx: ResponseContext, backend_label: &'static str) -> Response {
    let generate_request = match &ctx.original {
        ProtocolRequest::Generate(r) => Arc::clone(r),
        ProtocolRequest::Chat(_) => {
            return build_error_sse("generate_streaming invoked with chat request");
        }
    };

    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
    let cfg = GenerateStreamConfig { backend_label };
    let tokenizer = ctx.tokenizer.clone();
    let request_id = ctx.request_id.clone();
    let return_logprob = generate_request.return_logprob.unwrap_or(false);
    let model = ctx
        .model_id
        .clone()
        .unwrap_or_else(|| generate_request.model.clone().unwrap_or_default());

    tokio::spawn(async move {
        let result = run_generate_stream(
            stream,
            tokenizer,
            generate_request,
            request_id,
            model,
            return_logprob,
            cfg,
            &tx,
        )
        .await;
        if let Err(e) = result {
            let _ = tx.send(Ok(Bytes::from(format!(
                "data: {{\"error\": \"{}\"}}\n\n",
                e
            ))));
        }
        let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    });

    build_sse_response(rx)
}

async fn run_generate_stream(
    mut stream: TokenHandle,
    tokenizer: Arc<dyn Tokenizer>,
    _generate_request: Arc<GenerateRequest>,
    request_id: String,
    model: String,
    return_logprob: bool,
    cfg: GenerateStreamConfig,
    tx: &UnboundedSender<Result<Bytes, io::Error>>,
) -> Result<(), String> {
    let start_time = Instant::now();
    let mut first_token_time: Option<Instant> = None;

    let mut accumulated_text = String::new();
    let mut completion_tokens: u32 = 0;
    let mut weight_version: Option<String> = None;
    let mut prompt_tokens_seen: u32 = 0;
    let mut accumulated_output_logprobs: Option<Vec<Vec<Option<f64>>>> = None;

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

                let chunk_text = tokenizer.decode(&token_ids, true).unwrap_or_default();
                accumulated_text.push_str(&chunk_text);

                if return_logprob {
                    if let Some(lp) = logprobs.as_ref() {
                        accumulated_output_logprobs = Some(output_logprobs_to_generate(lp));
                    }
                }

                let index_id = format!("{}-{}", request_id, 0);
                let response = json!({
                    "text": accumulated_text.clone(),
                    "output_ids": token_ids,
                    "meta_info": {
                        "id": index_id,
                        "finish_reason": null,
                        "prompt_tokens": prompt_tokens_seen,
                        "weight_version": weight_version.as_deref().unwrap_or("default"),
                        "output_token_logprobs": accumulated_output_logprobs.as_ref(),
                        "completion_tokens": completion_tokens,
                        "cached_tokens": 0,
                    },
                    "index": 0,
                });
                let s = serde_json::to_string(&response)
                    .map_err(|e| format!("Failed to serialize chunk: {}", e))?;
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
                    .map_err(|_| "send chunk".to_string())?;
            }
            TokenChunk::Complete {
                token_ids,
                finish_reason,
                usage,
                meta,
                logprobs,
                input_logprobs,
                ..
            } => {
                prompt_tokens_seen = usage.prompt_tokens;
                if weight_version.is_none() {
                    weight_version = meta.weight_version.clone();
                }
                let input_token_logprobs = if return_logprob {
                    input_logprobs.as_ref().map(input_logprobs_to_generate)
                } else {
                    None
                };
                let final_output_logprobs = if return_logprob {
                    logprobs
                        .as_ref()
                        .map(output_logprobs_to_generate)
                        .or(accumulated_output_logprobs.clone())
                } else {
                    None
                };
                let index_id = format!("{}-{}", request_id, 0);
                let e2e_latency = start_time.elapsed().as_secs_f64();
                let fr = finish_reason_to_generate(&finish_reason, usage.completion_tokens);

                let last = token_ids
                    .last()
                    .copied()
                    .map(|t| vec![t])
                    .unwrap_or_default();

                let response = json!({
                    "text": accumulated_text.clone(),
                    "output_ids": last,
                    "meta_info": {
                        "id": index_id,
                        "finish_reason": fr,
                        "prompt_tokens": usage.prompt_tokens,
                        "weight_version": weight_version.as_deref().unwrap_or("default"),
                        "input_token_logprobs": input_token_logprobs,
                        "output_token_logprobs": final_output_logprobs,
                        "completion_tokens": usage.completion_tokens,
                        "cached_tokens": meta.cached_tokens,
                        "e2e_latency": e2e_latency,
                    },
                    "index": 0,
                });
                let s = serde_json::to_string(&response)
                    .map_err(|e| format!("Failed to serialize finish chunk: {}", e))?;
                tx.send(Ok(Bytes::from(format!("data: {}\n\n", s))))
                    .map_err(|_| "send finish".to_string())?;
            }
        }
    }

    stream.mark_completed();

    MeshMetrics::record_streaming_metrics(StreamingMetricsParams {
        router_type: metrics_labels::ROUTER_GRPC,
        backend_type: cfg.backend_label,
        model_id: &model,
        endpoint: metrics_labels::ENDPOINT_GENERATE,
        ttft: first_token_time.map(|t| t.duration_since(start_time)),
        generation_duration: start_time.elapsed(),
        input_tokens: None,
        output_tokens: completion_tokens as u64,
    });

    Ok(())
}

fn finish_reason_to_generate(r: &FinishReason, completion_tokens: u32) -> GenerateFinishReason {
    use serde_json::Value;
    match r {
        FinishReason::Stop => GenerateFinishReason::Stop,
        FinishReason::Length => GenerateFinishReason::Length {
            length: completion_tokens,
        },
        FinishReason::ContentFilter => {
            GenerateFinishReason::Other(Value::String("content_filter".to_string()))
        }
        FinishReason::ToolCalls => {
            GenerateFinishReason::Other(Value::String("tool_calls".to_string()))
        }
        FinishReason::Abort => GenerateFinishReason::Other(Value::String("abort".to_string())),
        FinishReason::Other(s) => GenerateFinishReason::Other(Value::String(s.clone())),
    }
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
    let _ = tx.send(Ok(Bytes::from(format!(
        "data: {{\"error\": \"{}\"}}\n\n",
        message
    ))));
    let _ = tx.send(Ok(Bytes::from("data: [DONE]\n\n")));
    drop(tx);
    build_sse_response(rx)
}

fn engine_error_to_string(e: EngineError) -> String {
    if let EngineError::Transport(_) = &e {
        error!("generate streaming transport error: {}", e);
    }
    format!("{}", e)
}
