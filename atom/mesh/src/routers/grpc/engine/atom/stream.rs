// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! ATOM's Generate stream as `TokenChunk`s. A GenerateError keeps the HTTP
//! status ATOM's HTTP routes answer the same request with.

use std::pin::Pin;
use std::task::{ready, Context, Poll};
use std::time::Instant;

use futures::{Stream, StreamExt};
use http::StatusCode;
use tonic::Streaming;

use super::AtomClient;
use crate::atom_engine::grpc::{
    generate_response::Response, GenerateComplete, GenerateRequest, GenerateResponse,
};
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::{
    FinishReason, MatchedStop, SeqTiming, TokenChunk, Usage, WorkerMeta,
};
use crate::routers::token_handle::token_handle::{SourceClock, TokenSource};

pub(crate) struct AtomStream {
    inner: Streaming<GenerateResponse>,
    /// The first message, read by `open`.
    first: Option<TokenChunk>,
    /// Unstreamed, each sibling's Complete arrives as it ends; all are read, then
    /// handed out by index.
    streamed: bool,
    completes: Vec<(u32, TokenChunk)>,
    finished: bool,
    clock: SourceClock,
}

impl AtomStream {
    /// Starts the call and reads its first message, so a request ATOM refuses
    /// fails here, before a response has begun.
    pub(crate) async fn open(
        client: &AtomClient,
        request: GenerateRequest,
    ) -> Result<Self, EngineError> {
        let streamed = request.stream;
        let started = Instant::now();
        let inner = client
            .generate(request)
            .await
            .map_err(EngineError::Transport)?;
        let mut stream = Self {
            inner,
            first: None,
            streamed,
            completes: Vec::new(),
            finished: false,
            clock: SourceClock {
                started,
                first_token: None,
            },
        };
        let first = stream.next().await;
        let first = first.unwrap_or(Err(EngineError::DecodeIncomplete))?;
        if matches!(first, TokenChunk::Partial { .. }) {
            stream.clock.first_token = Some(Instant::now());
        }
        stream.first = Some(first);
        Ok(stream)
    }
}

impl Stream for AtomStream {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(chunk) = self.first.take() {
            return Poll::Ready(Some(Ok(chunk)));
        }
        if self.finished {
            return Poll::Ready(self.completes.pop().map(|(_, chunk)| Ok(chunk)));
        }
        loop {
            let response = match ready!(self.inner.poll_next_unpin(cx)) {
                Some(Ok(GenerateResponse { response: Some(r) })) => r,
                Some(Ok(GenerateResponse { response: None })) => continue,
                Some(Err(status)) => {
                    self.finished = true;
                    self.completes.clear();
                    return Poll::Ready(Some(Err(EngineError::Transport(status))));
                }
                None if self.completes.is_empty() => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(EngineError::DecodeIncomplete)));
                }
                None => {
                    self.finished = true;
                    // Highest index first, so `pop` hands them out from index 0.
                    self.completes
                        .sort_by_key(|(index, _)| std::cmp::Reverse(*index));
                    return Poll::Ready(self.completes.pop().map(|(_, chunk)| Ok(chunk)));
                }
            };
            let item = match response {
                Response::Chunk(chunk) => Ok(TokenChunk::Partial {
                    token_ids: chunk.token_ids,
                    logprobs: None,
                }),
                Response::Complete(complete) if !self.streamed => {
                    let index = complete.index;
                    self.completes.push((index, complete_chunk(complete)));
                    continue;
                }
                Response::Complete(complete) => Ok(complete_chunk(complete)),
                Response::Error(error) => Err(EngineError::Http {
                    status: u16::try_from(error.http_status)
                        .ok()
                        .and_then(|code| StatusCode::from_u16(code).ok())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    message: error.message,
                }),
            };
            self.finished = !matches!(item, Ok(TokenChunk::Partial { .. }));
            if item.is_err() {
                self.completes.clear();
            }
            return Poll::Ready(Some(item));
        }
    }
}

impl TokenSource for AtomStream {
    /// Nothing to suppress: a cancelled call is ATOM's abort, and a finished
    /// call has nothing left to cancel.
    fn mark_completed(&mut self) {}

    /// `open` reads the first message before any renderer starts its clock.
    fn clock(&self) -> Option<SourceClock> {
        Some(self.clock)
    }
}

fn complete_chunk(complete: GenerateComplete) -> TokenChunk {
    TokenChunk::Complete {
        finish_reason: finish_reason(&complete.finish_reason),
        // ATOM's reason for a stop token is `stop_<token id>`; a stop-string hit is
        // `stop_sequence`, naming none, so it has no matched_stop, as on ATOM's HTTP routes.
        matched_stop: complete
            .finish_reason
            .strip_prefix("stop_")
            .and_then(|id| id.parse().ok())
            .map(MatchedStop::TokenId),
        usage: Usage {
            prompt_tokens: complete.prompt_tokens,
            completion_tokens: complete.completion_tokens,
            total_tokens: complete.prompt_tokens + complete.completion_tokens,
        },
        logprobs: None,
        input_logprobs: None,
        meta: WorkerMeta {
            request_id: String::new(),
            weight_version: None,
            cached_tokens: complete.cached_tokens,
            timing: Some(SeqTiming {
                ttft: complete.ttft_s,
                tpot: complete.tpot_s,
                latency: complete.latency_s,
            }),
        },
        token_ids: complete.output_ids,
    }
}

/// The engine's leave reason as ATOM's HTTP routes report it
/// (`atom.entrypoints.openai.protocol.openai_stop_reason`).
fn finish_reason(reason: &str) -> FinishReason {
    match reason {
        "length" | "max_tokens" | "max_new_tokens" => FinishReason::Length,
        "tool_calls" => FinishReason::ToolCalls,
        _ => FinishReason::Stop,
    }
}
