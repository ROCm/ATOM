//! ATOM's Generate stream, or a Session request's stream of the same messages, as
//! `TokenChunk`s. A GenerateError keeps the HTTP status ATOM's HTTP routes answer.

use std::pin::Pin;
use std::task::{ready, Context, Poll};

use futures::{Stream, StreamExt};
use http::StatusCode;
use tonic::{Status, Streaming};

use super::AtomClient;
use crate::atom_engine::grpc::{
    generate_response::Response, GenerateComplete, GenerateError, GenerateRequest, GenerateResponse,
};
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::{
    FinishReason, MatchedStop, TokenChunk, Usage, WorkerMeta,
};
use crate::routers::token_handle::token_handle::TokenSource;

pub(crate) struct AtomStream<S = Streaming<GenerateResponse>> {
    inner: S,
    /// The first message, read by `start`.
    first: Option<TokenChunk>,
    finished: bool,
}

impl AtomStream {
    /// Starts the call and reads its first message, as `start` does.
    pub(crate) async fn open(
        client: &AtomClient,
        request: GenerateRequest,
    ) -> Result<Self, EngineError> {
        let inner = client
            .generate(request)
            .await
            .map_err(EngineError::Transport)?;
        Self::start(inner).await
    }
}

impl<S: Stream<Item = Result<GenerateResponse, Status>> + Unpin> AtomStream<S> {
    /// Reads `inner`'s first message, so a request ATOM refuses fails here,
    /// before a response has begun.
    pub(crate) async fn start(inner: S) -> Result<Self, EngineError> {
        let mut stream = Self {
            inner,
            first: None,
            finished: false,
        };
        let first = stream.next().await;
        stream.first = Some(first.unwrap_or(Err(EngineError::DecodeIncomplete))?);
        Ok(stream)
    }
}

impl<S: Stream<Item = Result<GenerateResponse, Status>> + Unpin> Stream for AtomStream<S> {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if let Some(chunk) = self.first.take() {
            return Poll::Ready(Some(Ok(chunk)));
        }
        while !self.finished {
            let response = match ready!(self.inner.poll_next_unpin(cx)) {
                Some(Ok(GenerateResponse { response: Some(r) })) => r,
                Some(Ok(GenerateResponse { response: None })) => continue,
                Some(Err(status)) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(EngineError::Transport(status))));
                }
                None => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(EngineError::DecodeIncomplete)));
                }
            };
            let item = match response {
                Response::Chunk(chunk) => Ok(TokenChunk::Partial {
                    token_ids: chunk.token_ids,
                    logprobs: None,
                }),
                Response::Complete(complete) => Ok(complete_chunk(complete)),
                Response::Error(error) => Err(http_error(error)),
            };
            self.finished = !matches!(item, Ok(TokenChunk::Partial { .. }));
            return Poll::Ready(Some(item));
        }
        Poll::Ready(None)
    }
}

impl<S: Stream<Item = Result<GenerateResponse, Status>> + Send + Unpin> TokenSource
    for AtomStream<S>
{
    /// Nothing to suppress: a cancelled call or a dropped Session request is
    /// ATOM's abort, and a finished one has nothing left to cancel.
    fn mark_completed(&mut self) {}
}

/// A GenerateError, with the status ATOM's HTTP routes answer its request with.
pub(super) fn http_error(error: GenerateError) -> EngineError {
    EngineError::Http {
        status: u16::try_from(error.http_status)
            .ok()
            .and_then(|code| StatusCode::from_u16(code).ok())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        message: error.message,
    }
}

fn complete_chunk(complete: GenerateComplete) -> TokenChunk {
    TokenChunk::Complete {
        finish_reason: finish_reason(&complete.finish_reason),
        // ATOM's reason for a stop token is `stop_<token id>`.
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
