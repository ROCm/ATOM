//! Token boundary between Entrypoints and a downstream engine.
//!
//! Prepare has already rendered the template and encoded the prompt. The
//! backend routes those token ids and returns a token stream. It does not
//! detokenize, and it does not see the HTTP server.

use std::pin::Pin;
use std::task::{Context, Poll};

use async_trait::async_trait;
use axum::{http::StatusCode, response::Response};
use futures::Stream;
use http::HeaderMap;

use super::prepare::generation_payload::GenerationPayload;
use super::prepare::response_context::ResponseContext;
use crate::routers::token_handle::{
    engine_error::EngineError,
    token_chunk::TokenChunk,
    token_handle::{TokenHandle, TokenSource},
};

/// A prepared request handed to the backend.
pub struct TokenRequest {
    pub payload: GenerationPayload,
    pub model: Option<String>,
    pub headers: Option<HeaderMap>,
}

impl TokenRequest {
    /// Moves the request headers out of `ctx`; render does not read them.
    pub fn from_prepared(payload: GenerationPayload, ctx: &mut ResponseContext) -> Self {
        Self {
            model: ctx.model_id.clone(),
            headers: ctx.headers.take(),
            payload,
        }
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.payload.token_ids
    }
}

/// Token stream plus a value dropped with the HTTP response body.
///
/// The finish value is how a backend keeps a worker load guard alive until
/// the client is done reading, including after a non-streaming render has
/// already consumed the chunks.
pub struct TokenStream {
    pub handle: TokenHandle,
    finish: Option<Box<dyn Send + 'static>>,
}

impl TokenStream {
    pub fn new(handle: TokenHandle) -> Self {
        Self {
            handle,
            finish: None,
        }
    }

    pub fn with_finish(handle: TokenHandle, finish: impl Send + 'static) -> Self {
        Self {
            handle,
            finish: Some(Box::new(finish)),
        }
    }

    pub fn from_chunks(chunks: Vec<TokenChunk>) -> Self {
        Self::new(TokenHandle::new(ChunkSource {
            chunks: chunks.into_iter().map(Ok).collect::<Vec<_>>().into_iter(),
        }))
    }

    pub fn into_parts(self) -> (TokenHandle, Option<Box<dyn Send + 'static>>) {
        (self.handle, self.finish)
    }
}

struct ChunkSource {
    chunks: std::vec::IntoIter<Result<TokenChunk, EngineError>>,
}

impl Stream for ChunkSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(self.get_mut().chunks.next())
    }
}

impl TokenSource for ChunkSource {
    fn mark_completed(&mut self) {}
}

#[derive(Debug)]
pub struct BackendError {
    pub status: StatusCode,
    pub code: String,
    pub message: String,
}

impl BackendError {
    pub fn new(status: StatusCode, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_IMPLEMENTED, "unsupported", message)
    }

    /// The same Mesh-local error response the router produced before the
    /// split, so the ingress layer still shapes its body for the request path.
    pub fn into_response(self) -> Response {
        crate::routers::comm::error::create_error(self.status, self.code, self.message)
    }
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for BackendError {}

#[cfg(test)]
pub(crate) fn test_request(token_ids: Vec<u32>) -> TokenRequest {
    use super::prepare::generation_payload::{LogprobConfig, SamplingParams, StopConfig};
    TokenRequest {
        payload: GenerationPayload {
            request_id: "req-1".to_string(),
            text: String::new(),
            token_ids,
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: -1,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                max_new_tokens: Some(8),
                min_new_tokens: 0,
                n: 1,
                ignore_eos: false,
            },
            stop: StopConfig {
                stop: None,
                stop_token_ids: None,
                skip_special_tokens: true,
                no_stop_trim: false,
            },
            logprob: LogprobConfig {
                return_logprob: false,
                top_logprobs_num: 0,
                logprob_start_len: -1,
                token_ids_logprob: Vec::new(),
                input_logprobs: false,
            },
            tool_constraints: None,
            pd_metadata: None,
            stream: false,
            return_hidden_states: false,
            log_metrics: false,
        },
        model: Some("m".to_string()),
        headers: None,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };

    use super::*;

    struct Flag(Arc<AtomicBool>);
    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Relaxed);
        }
    }

    #[test]
    fn finish_guard_drops_with_the_response_attachment() {
        let flag = Arc::new(AtomicBool::new(false));
        let stream = TokenStream::with_finish(
            TokenStream::from_chunks(Vec::new()).into_parts().0,
            Flag(flag.clone()),
        );
        let (_handle, finish) = stream.into_parts();
        assert!(!flag.load(Ordering::Relaxed));
        drop(finish);
        assert!(flag.load(Ordering::Relaxed));
    }

    #[test]
    fn backend_error_uses_the_entrypoint_envelope() {
        let response = BackendError::unsupported("http has no token entry").into_response();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        assert_eq!(
            crate::routers::comm::error::extract_error_code_from_response(&response),
            "unsupported"
        );
        assert!(response
            .extensions()
            .get::<crate::routers::comm::error::MeshLocalError>()
            .is_some());
    }
}

#[async_trait]
pub trait Backend: Send + Sync {
    /// Route `request` and return the engine's token stream.
    async fn generate(&self, request: TokenRequest) -> Result<TokenStream, BackendError>;

    /// Stop a request that `generate` already accepted.
    async fn abort(&self, request_id: &str) -> Result<(), BackendError>;
}
