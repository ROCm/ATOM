//! Render a `TokenHandle` as an OpenAI `/v1/completions` response: SSE text
//! deltas, or one aggregated body. Decodes the tokens itself, so a
//! completion does not detour through `/generate`'s SSE.

use std::{
    io,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    http::{header::CONTENT_TYPE, HeaderValue},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use futures::StreamExt;
use serde::Serialize;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use uuid::Uuid;

use crate::{
    protocols::{
        common::Usage,
        completion::{
            CompletionChoice, CompletionRequest, CompletionResponse, CompletionStreamChoice,
            CompletionStreamResponse,
        },
    },
    routers::{
        prepare::response_context::ResponseContext,
        token_handle::{
            token_chunk::{FinishReason, TokenChunk, Usage as ChunkUsage},
            token_handle::TokenHandle,
        },
    },
    tokenizer::stop::{SequenceDecoderOutput, StopSequenceDecoder},
};

type SseSender = mpsc::UnboundedSender<Result<Bytes, io::Error>>;

pub struct CompletionRender {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
}

impl CompletionRender {
    pub fn new(request: &CompletionRequest, model: String) -> Self {
        Self {
            id: format!("cmpl-{}", Uuid::new_v4().simple()),
            model,
            created: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            include_usage: request
                .stream_options
                .as_ref()
                .and_then(|o| o.include_usage)
                .unwrap_or(false),
        }
    }

    pub fn stream(self, mut handle: TokenHandle, ctx: ResponseContext) -> Response {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let frame = DeltaFrame::new(&self.id, self.created, &self.model);
            let mut decoder = ctx.stop_decoder;
            let mut usage = None;
            let mut stopped = false;
            while let Some(item) = handle.next().await {
                match item {
                    Ok(TokenChunk::Partial { token_ids, .. }) if !stopped => {
                        let text;
                        (text, stopped) = decode(&mut decoder, &token_ids);
                        if !text.is_empty() && tx.send(Ok(frame.event(&text))).is_err() {
                            return;
                        }
                    }
                    Ok(TokenChunk::Partial { .. }) => {}
                    Ok(TokenChunk::Complete {
                        finish_reason,
                        usage: done,
                        ..
                    }) => {
                        let tail = match decoder.flush() {
                            SequenceDecoderOutput::Text(t) if !stopped => t,
                            _ => String::new(),
                        };
                        if !self.send_finish(&tx, tail, &finish_reason) {
                            return;
                        }
                        usage = Some(done);
                    }
                    Err(e) => {
                        let error =
                            json!({"error": {"message": e.to_string(), "type": "internal_error"}});
                        let _ = tx.send(Ok(sse(&error)));
                        break;
                    }
                }
            }
            handle.mark_completed();
            if let (true, Some(usage)) = (self.include_usage, usage) {
                let chunk = json!({
                    "id": self.id, "object": "text_completion", "created": self.created,
                    "model": self.model, "choices": [], "usage": openai_usage(&usage),
                });
                let _ = tx.send(Ok(sse(&chunk)));
            }
            let _ = tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
        });

        let mut response = Response::new(Body::from_stream(UnboundedReceiverStream::new(rx)));
        let headers = response.headers_mut();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
        response
    }

    /// Every sibling's `Complete`, in order, as one choice each.
    pub async fn aggregate(self, mut handle: TokenHandle, ctx: ResponseContext) -> Response {
        let mut decoder = ctx.stop_decoder;
        let mut choices = Vec::new();
        let (mut prompt_tokens, mut completion_tokens) = (0, 0);
        while let Some(item) = handle.next().await {
            match item {
                Ok(TokenChunk::Complete {
                    token_ids,
                    finish_reason,
                    usage,
                    ..
                }) => {
                    decoder.reset();
                    let (mut text, stopped) = decode(&mut decoder, &token_ids);
                    if let (false, SequenceDecoderOutput::Text(t)) = (stopped, decoder.flush()) {
                        text.push_str(&t);
                    }
                    choices.push(CompletionChoice {
                        text,
                        index: choices.len() as u32,
                        logprobs: None,
                        finish_reason: Some(openai_finish_reason(&finish_reason).into()),
                        matched_stop: None,
                    });
                    prompt_tokens += usage.prompt_tokens;
                    completion_tokens += usage.completion_tokens;
                }
                Ok(TokenChunk::Partial { .. }) => {}
                Err(e) => return e.into_response(),
            }
        }
        handle.mark_completed();
        Json(CompletionResponse {
            id: self.id,
            object: "text_completion".into(),
            created: self.created,
            model: self.model,
            choices,
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                completion_tokens_details: None,
            }),
            system_fingerprint: None,
        })
        .into_response()
    }

    /// Send the last delta and why it ended; false once the client is gone.
    fn send_finish(&self, tx: &SseSender, text: String, reason: &FinishReason) -> bool {
        let chunk = CompletionStreamResponse {
            id: self.id.clone(),
            object: "text_completion".into(),
            created: self.created,
            choices: vec![CompletionStreamChoice {
                text,
                index: 0,
                logprobs: None,
                finish_reason: Some(openai_finish_reason(reason).into()),
            }],
            model: self.model.clone(),
            system_fingerprint: None,
        };
        tx.send(Ok(sse(&chunk))).is_ok()
    }
}

/// A text delta's SSE event. Everything but the text is serialized once per
/// request, so a token costs one escape and one allocation.
struct DeltaFrame {
    head: Vec<u8>,
    tail: Vec<u8>,
}

impl DeltaFrame {
    fn new(id: &str, created: u64, model: &str) -> Self {
        let json = |value: &str| serde_json::to_string(value).expect("strings serialize");
        Self {
            head: format!(
                r#"data: {{"id":{},"object":"text_completion","created":{created},"choices":[{{"text":"#,
                json(id)
            )
            .into_bytes(),
            tail: format!(
                r#","index":0,"finish_reason":null}}],"model":{}}}"#,
                json(model)
            )
            .into_bytes(),
        }
    }

    fn event(&self, text: &str) -> Bytes {
        let mut buf = Vec::with_capacity(self.head.len() + text.len() + self.tail.len() + 16);
        buf.extend_from_slice(&self.head);
        serde_json::to_writer(&mut buf, text).expect("strings serialize");
        buf.extend_from_slice(&self.tail);
        buf.extend_from_slice(b"\n\n");
        Bytes::from(buf)
    }
}

/// Decode `tokens` through the stop decoder; true once a stop matched.
fn decode(decoder: &mut StopSequenceDecoder, tokens: &[u32]) -> (String, bool) {
    let mut text = String::new();
    for &token in tokens {
        match decoder
            .process_token(token)
            .unwrap_or(SequenceDecoderOutput::Held)
        {
            SequenceDecoderOutput::Text(t) => text.push_str(&t),
            SequenceDecoderOutput::StoppedWithText(t) => {
                text.push_str(&t);
                return (text, true);
            }
            SequenceDecoderOutput::Stopped => return (text, true),
            SequenceDecoderOutput::Held => {}
        }
    }
    (text, false)
}

/// What the OpenAI API calls the reason: anything not a length cap is a stop.
fn openai_finish_reason(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Abort => "abort",
        _ => "stop",
    }
}

fn openai_usage(usage: &ChunkUsage) -> serde_json::Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    })
}

fn sse<T: Serialize>(payload: &T) -> Bytes {
    let mut buf = b"data: ".to_vec();
    serde_json::to_writer(&mut buf, payload).expect("completion chunks serialize");
    buf.extend_from_slice(b"\n\n");
    Bytes::from(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_delta_frame_is_the_serialized_chunk() {
        let text = "say \"hi\"\n\u{1F600}";
        let chunk = CompletionStreamResponse {
            id: "cmpl-1".into(),
            object: "text_completion".into(),
            created: 7,
            choices: vec![CompletionStreamChoice {
                text: text.into(),
                index: 0,
                logprobs: None,
                finish_reason: None,
            }],
            model: "/m/\"q\"".into(),
            system_fingerprint: None,
        };
        let frame = DeltaFrame::new("cmpl-1", 7, "/m/\"q\"");
        assert_eq!(frame.event(text), sse(&chunk));
    }
}
