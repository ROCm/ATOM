//! Aggregate a `TokenHandle<TokenChunk>` into a non-streaming
//! `Vec<GenerateResponse>` (SGLang native generate format).

use std::time::Instant;

use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::Value;

use crate::{
    protocols::generate::{GenerateFinishReason, GenerateMetaInfo, GenerateResponse},
    routers::{
        comm::error,
        prepare::response_context::{ProtocolRequest, ResponseContext},
        render::logprob_conversion::{input_logprobs_to_generate, output_logprobs_to_generate},
        token_handle::{
            engine_error::EngineError,
            token_chunk::{FinishReason, MatchedStop, TokenChunk},
            token_handle::TokenHandle,
        },
    },
    tokenizer::stop::SequenceDecoderOutput,
};

pub async fn process(stream: TokenHandle, ctx: ResponseContext) -> Response {
    let start = Instant::now();

    if !matches!(&ctx.original, ProtocolRequest::Generate(_)) {
        return error::internal_error(
            "wrong_render_path",
            "generate_aggregator invoked with a chat request",
        );
    }

    let completes = match collect_completes(stream).await {
        Ok(c) => c,
        Err(resp) => return resp,
    };
    if completes.is_empty() {
        return error::internal_error("no_responses_from_server", "No responses from server");
    }

    let request_logprobs = match &ctx.original {
        ProtocolRequest::Generate(r) => r.return_logprob.unwrap_or(false),
        ProtocolRequest::Chat(_) => false,
    };

    let mut stop_decoder = ctx.stop_decoder;
    let mut result_array = Vec::with_capacity(completes.len());

    for complete in &completes {
        let (token_ids, finish_reason, matched_stop, usage, meta, output_lp, input_lp) =
            match complete {
                TokenChunk::Complete {
                    token_ids,
                    finish_reason,
                    matched_stop,
                    usage,
                    meta,
                    logprobs,
                    input_logprobs,
                    ..
                } => (
                    token_ids,
                    finish_reason,
                    matched_stop,
                    usage,
                    meta,
                    logprobs.as_ref(),
                    input_logprobs.as_ref(),
                ),
                TokenChunk::Partial { .. } => continue,
            };

        stop_decoder.reset();
        let outputs = match stop_decoder.process_tokens(token_ids) {
            Ok(o) => o,
            Err(e) => {
                return error::internal_error(
                    "process_tokens_failed",
                    format!("Failed to process tokens: {}", e),
                );
            }
        };

        let mut decoded_text = String::new();
        for output in outputs {
            match output {
                SequenceDecoderOutput::Text(t) => decoded_text.push_str(&t),
                SequenceDecoderOutput::StoppedWithText(t) => {
                    decoded_text.push_str(&t);
                    break;
                }
                SequenceDecoderOutput::Stopped => break,
                SequenceDecoderOutput::Held => {}
            }
        }
        if let SequenceDecoderOutput::Text(t) = stop_decoder.flush() {
            decoded_text.push_str(&t);
        }

        let matched_stop_value = matched_stop.as_ref().map(|ms| match ms {
            MatchedStop::Str(s) => Value::String(s.clone()),
            MatchedStop::TokenId(t) => Value::Number(serde_json::Number::from(*t)),
        });

        let (input_token_logprobs, output_token_logprobs) = if request_logprobs {
            (
                input_lp.map(input_logprobs_to_generate),
                output_lp.map(output_logprobs_to_generate),
            )
        } else {
            (None, None)
        };

        result_array.push(GenerateResponse {
            text: decoded_text,
            output_ids: token_ids.clone(),
            meta_info: GenerateMetaInfo {
                id: ctx.request_id.clone(),
                finish_reason: finish_reason_to_generate(finish_reason, usage.completion_tokens),
                prompt_tokens: usage.prompt_tokens,
                weight_version: meta
                    .weight_version
                    .clone()
                    .unwrap_or_else(|| "default".to_string()),
                input_token_logprobs,
                output_token_logprobs,
                completion_tokens: usage.completion_tokens,
                cached_tokens: meta.cached_tokens,
                e2e_latency: start.elapsed().as_secs_f64(),
                matched_stop: matched_stop_value,
            },
        });
    }

    axum::Json(result_array).into_response()
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

fn finish_reason_to_generate(r: &FinishReason, completion_tokens: u32) -> GenerateFinishReason {
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
    }
}
