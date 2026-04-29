//! Completion endpoint adapter for gRPC routers.
//!
//! Converts between OpenAI `/v1/completions` protocol and SGLang's native
//! generate protocol so the existing generate pipeline can be reused.
//!
//! Supports both streaming and non-streaming requests. Batched prompts,
//! echo, suffix, best_of, and logit_bias return a clear 4xx error.

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::StreamExt;
use http::{header, HeaderValue};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::{
    protocols::{
        common::{StringOrArray, Usage},
        completion::{CompletionChoice, CompletionRequest, CompletionResponse},
        generate::{GenerateFinishReason, GenerateRequest, GenerateResponse},
        sampling_params::SamplingParams,
    },
    routers::error,
};

/// Build a synthetic SGLang `GenerateRequest` from an OpenAI `CompletionRequest`.
///
/// Returns `Err(message)` for unsupported features so the caller can emit
/// `400 Bad Request` with the same message.
pub(crate) fn completion_to_generate(c: &CompletionRequest) -> Result<GenerateRequest, String> {
    if c.echo {
        return Err("`echo` is not supported on the gRPC /v1/completions path".into());
    }
    if c.suffix.is_some() {
        return Err("`suffix` is not supported on the gRPC /v1/completions path".into());
    }
    if c.best_of.map(|n| n > 1).unwrap_or(false) {
        return Err("`best_of > 1` is not supported on the gRPC /v1/completions path".into());
    }
    if c.logit_bias.is_some() {
        return Err("`logit_bias` is not supported on the gRPC /v1/completions path".into());
    }

    let text = match &c.prompt {
        StringOrArray::String(s) => s.clone(),
        StringOrArray::Array(arr) => {
            if arr.len() != 1 {
                return Err(format!(
                    "Batched prompts (got {} items) are not supported on the gRPC /v1/completions path; send one prompt per request",
                    arr.len()
                ));
            }
            arr[0].clone()
        }
    };

    let sampling_params = SamplingParams {
        temperature: c.temperature,
        max_new_tokens: c.max_tokens,
        top_p: c.top_p,
        top_k: c.top_k,
        frequency_penalty: c.frequency_penalty,
        presence_penalty: c.presence_penalty,
        repetition_penalty: None,
        stop: c.stop.clone(),
        ignore_eos: Some(c.ignore_eos),
        skip_special_tokens: Some(c.skip_special_tokens),
        json_schema: c.json_schema.clone(),
        regex: c.regex.clone(),
        ebnf: c.ebnf.clone(),
        min_p: c.min_p,
        min_new_tokens: None,
        stop_token_ids: c.stop_token_ids.clone(),
        no_stop_trim: Some(c.no_stop_trim),
        n: c.n,
        sampling_seed: c.sampling_seed.or_else(|| c.seed.map(|s| s as u64)),
    };

    Ok(GenerateRequest {
        text: Some(text),
        model: Some(c.model.clone()),
        input_ids: None,
        input_embeds: None,
        image_data: None,
        video_data: None,
        audio_data: None,
        sampling_params: Some(sampling_params),
        return_logprob: Some(c.logprobs.is_some()),
        logprob_start_len: None,
        top_logprobs_num: c.logprobs.map(|n| n as i32),
        token_ids_logprob: None,
        return_text_in_logprobs: false,
        stream: c.stream,
        log_metrics: true,
        return_hidden_states: c.return_hidden_states,
        modalities: None,
        session_params: c.session_params.clone(),
        lora_path: c.lora_path.clone(),
        lora_id: None,
        custom_logit_processor: None,
        bootstrap_host: None,
        bootstrap_port: None,
        bootstrap_room: None,
        bootstrap_pair_key: None,
        data_parallel_rank: None,
        background: false,
        conversation_id: None,
        priority: None,
        extra_key: None,
        no_logs: false,
        custom_labels: None,
        return_bytes: false,
        return_entropy: false,
        rid: None,
    })
}

/// Convert a list of native `GenerateResponse`s into an OpenAI `CompletionResponse`.
fn build_completion_response(
    gens: Vec<GenerateResponse>,
    model: String,
) -> CompletionResponse {
    let id = gens
        .first()
        .map(|g| g.meta_info.id.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("cmpl-{}", Uuid::new_v4().simple()));

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let prompt_tokens = gens
        .first()
        .map(|g| g.meta_info.prompt_tokens)
        .unwrap_or(0);
    let completion_tokens: u32 = gens.iter().map(|g| g.meta_info.completion_tokens).sum();

    let choices = gens
        .into_iter()
        .enumerate()
        .map(|(i, g)| {
            let finish_reason = match &g.meta_info.finish_reason {
                GenerateFinishReason::Stop => Some("stop".to_string()),
                GenerateFinishReason::Length { .. } => Some("length".to_string()),
                GenerateFinishReason::Other(v) => v.as_str().map(|s| s.to_string()),
            };
            CompletionChoice {
                text: g.text,
                index: i as u32,
                logprobs: None,
                finish_reason,
                matched_stop: g.meta_info.matched_stop,
            }
        })
        .collect();

    CompletionResponse {
        id,
        object: "text_completion".to_string(),
        created,
        model,
        choices,
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            completion_tokens_details: None,
        }),
        system_fingerprint: None,
    }
}

/// Take the body of a successful `execute_generate` response and re-wrap it
/// as a `CompletionResponse` JSON.
///
/// On any deserialization or wrapping error, returns a 500 response with a
/// descriptive code. Non-success upstream responses are passed through unchanged.
pub(crate) async fn wrap_generate_response_as_completion(
    upstream: Response,
    model: String,
) -> Response {
    if !upstream.status().is_success() {
        return upstream;
    }

    let (parts, body) = upstream.into_parts();

    // 16 MB cap; completion responses are small.
    let bytes: Bytes = match axum::body::to_bytes(body, 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return error::internal_error(
                "completion_adapter_read_failed",
                format!("Failed to read upstream generate body: {}", e),
            );
        }
    };

    let gens: Vec<GenerateResponse> = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            return error::internal_error(
                "completion_adapter_parse_failed",
                format!("Failed to parse upstream generate JSON: {}", e),
            );
        }
    };

    let completion = build_completion_response(gens, model);
    let json = match serde_json::to_vec(&completion) {
        Ok(v) => v,
        Err(e) => {
            return error::internal_error(
                "completion_adapter_serialize_failed",
                format!("Failed to serialize completion response: {}", e),
            );
        }
    };

    let mut new = Response::from_parts(parts, Body::from(json));
    new.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    new.headers_mut().remove(header::CONTENT_LENGTH);
    new
}

/// Map a generate `finish_reason` Value (the raw JSON form, since streaming
/// chunks are parsed loosely as Value to avoid coupling to a single schema)
/// into the OpenAI completion `finish_reason` string.
fn map_finish_reason(v: &Value) -> Option<String> {
    if v.is_null() {
        return None;
    }
    if let Some(s) = v.as_str() {
        // SGLang sometimes serializes the unit variant `Stop` as the string
        // "stop"; pass it through.
        return Some(s.to_string());
    }
    if let Some(t) = v.get("type").and_then(|t| t.as_str()) {
        return Some(t.to_string());
    }
    None
}

/// Wrap an upstream `/generate` SSE stream as an OpenAI-style `/v1/completions`
/// SSE stream.
///
/// Generate emits cumulative `text` per chunk; OpenAI completion expects
/// per-chunk `text` deltas. We track the previous cumulative text and emit
/// only the new suffix.
pub(crate) async fn wrap_streaming_generate_as_completion(
    upstream: Response,
    model: String,
) -> Response {
    if !upstream.status().is_success() {
        return upstream;
    }

    let (parts, body) = upstream.into_parts();

    let cmpl_id = format!("cmpl-{}", Uuid::new_v4().simple());
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();

    tokio::spawn(async move {
        let mut data_stream = body.into_data_stream();
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        // Per-choice cumulative text seen so far, indexed by `index` field.
        let mut prev_text: Vec<String> = Vec::new();

        let send = |tx: &tokio::sync::mpsc::UnboundedSender<_>, payload: String| {
            tx.send(Ok(Bytes::from(payload))).is_ok()
        };

        'outer: while let Some(chunk_res) = data_stream.next().await {
            let chunk = match chunk_res {
                Ok(b) => b,
                Err(_) => break,
            };
            buf.extend_from_slice(&chunk);

            // Process every complete SSE event (terminated by "\n\n").
            loop {
                let Some(idx) = find_subsequence(&buf, b"\n\n") else {
                    break;
                };
                let event_bytes: Vec<u8> = buf.drain(..idx + 2).collect();

                // SSE event may have multiple lines; we only care about
                // `data: ...` lines (concatenated, per spec). For simplicity
                // assume a single-line `data: ` payload as SGLang produces.
                let event_str = match std::str::from_utf8(&event_bytes) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let mut data_payload = String::new();
                for line in event_str.split('\n') {
                    let line = line.trim_end_matches('\r');
                    if let Some(rest) = line.strip_prefix("data: ") {
                        if !data_payload.is_empty() {
                            data_payload.push('\n');
                        }
                        data_payload.push_str(rest);
                    } else if let Some(rest) = line.strip_prefix("data:") {
                        if !data_payload.is_empty() {
                            data_payload.push('\n');
                        }
                        data_payload.push_str(rest);
                    }
                }
                if data_payload.is_empty() {
                    continue;
                }

                let trimmed = data_payload.trim();
                if trimmed == "[DONE]" {
                    if !send(&tx, "data: [DONE]\n\n".to_string()) {
                        break 'outer;
                    }
                    continue;
                }

                let parsed: Value = match serde_json::from_str(trimmed) {
                    Ok(v) => v,
                    Err(_) => {
                        // Forward malformed payload as-is so client sees the
                        // original error rather than silently dropping.
                        let _ = send(
                            &tx,
                            format!("data: {}\n\n", trimmed),
                        );
                        continue;
                    }
                };

                // If upstream sent a JSON error object instead of an SSE chunk,
                // pass it through verbatim.
                if parsed.get("error").is_some() {
                    let _ = send(&tx, format!("data: {}\n\n", trimmed));
                    continue;
                }

                let index = parsed
                    .get("index")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let cumulative = parsed
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if prev_text.len() <= index {
                    prev_text.resize(index + 1, String::new());
                }
                let prev = &prev_text[index];
                let delta = if cumulative.starts_with(prev.as_str()) {
                    cumulative[prev.len()..].to_string()
                } else {
                    cumulative.clone()
                };
                prev_text[index] = cumulative;

                let meta_info = parsed.get("meta_info");

                let finish_reason = meta_info
                    .and_then(|m| m.get("finish_reason"))
                    .map(map_finish_reason)
                    .and_then(|x| x);
                let is_final = finish_reason.is_some();

                let matched_stop = meta_info
                    .and_then(|m| m.get("matched_stop"))
                    .cloned();

                let mut choice = json!({
                    "text": delta,
                    "index": index,
                    "logprobs": Value::Null,
                    "finish_reason": finish_reason,
                });
                if let Some(ms) = matched_stop {
                    if !ms.is_null() {
                        choice
                            .as_object_mut()
                            .unwrap()
                            .insert("matched_stop".to_string(), ms);
                    }
                }

                let chunk_obj = json!({
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [choice],
                });

                let serialized = match serde_json::to_string(&chunk_obj) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                if !send(&tx, format!("data: {}\n\n", serialized)) {
                    break 'outer;
                }

                // Emit a usage chunk after the final token so streaming
                // clients (e.g. bench_serving) can compute TPOT.
                if is_final {
                    let prompt_tokens = meta_info
                        .and_then(|m| m.get("prompt_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let completion_tokens = meta_info
                        .and_then(|m| m.get("completion_tokens"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);

                    let usage_chunk = json!({
                        "id": cmpl_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    });

                    if let Ok(s) = serde_json::to_string(&usage_chunk) {
                        if !send(&tx, format!("data: {}\n\n", s)) {
                            break 'outer;
                        }
                    }
                }
            }
        }
    });

    let stream = UnboundedReceiverStream::new(rx);

    let mut new = Response::from_parts(parts, Body::from_stream(stream));
    new.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    new.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    new.headers_mut()
        .insert(header::CONNECTION, HeaderValue::from_static("keep-alive"));
    new.headers_mut().remove(header::CONTENT_LENGTH);
    new
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}
