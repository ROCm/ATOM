//! gRPC engine: the only subtree that imports `mesh_grpc::*`.

#[cfg(feature = "grpc-client")]
pub(crate) mod atom;
pub mod payload_to_proto;
pub mod proto_stream_wrapper;
pub mod proto_to_chunk;
pub mod worker_client_cache;

pub(crate) mod pd_stream_merge;

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use async_trait::async_trait;
use futures::Stream;

use crate::core::placement::PlacementPlan;
use crate::core::Worker;
use crate::routers::grpc::engine::proto_stream_wrapper::{
    ProtoGenerateRequest, ProtoResponseVariant, ProtoStream,
};
use crate::routers::grpc::engine::worker_client_cache::{get_grpc_client_from_worker, GrpcClient};
use crate::routers::prepare::generation_payload::{GenerationPayload, PdMetadata};
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::TokenChunk;
use crate::routers::token_handle::token_handle::{TokenHandle, TokenSource};
use crate::tokenizer::traits::Tokenizer;

pub use pd_stream_merge::merge_pd_streams;

#[async_trait]
pub(crate) trait Dispatcher: Send + Sync {
    async fn dispatch(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
    ) -> Result<TokenHandle, EngineError>;

    /// `dispatch` with the request's tokenizer, with which an ATOM worker's
    /// stop strings are encoded.
    async fn dispatch_with_tokenizer(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
        _tokenizer: &Arc<dyn Tokenizer>,
    ) -> Result<TokenHandle, EngineError> {
        self.dispatch(placement, payload).await
    }
}

#[async_trait]
impl Dispatcher for GrpcEngine {
    async fn dispatch(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
    ) -> Result<TokenHandle, EngineError> {
        GrpcEngine::dispatch(self, placement, payload).await
    }

    async fn dispatch_with_tokenizer(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> Result<TokenHandle, EngineError> {
        GrpcEngine::dispatch_with_tokenizer(self, placement, payload, tokenizer).await
    }
}

#[derive(Clone, Default)]
pub struct GrpcEngine {}

impl GrpcEngine {
    pub fn new() -> Self {
        Self {}
    }

    /// Without a tokenizer, a request with stop strings fails for an ATOM worker;
    /// see `dispatch_with_tokenizer`.
    pub async fn dispatch(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
    ) -> Result<TokenHandle, EngineError> {
        self.dispatch_inner(placement, payload, None).await
    }

    pub async fn dispatch_with_tokenizer(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> Result<TokenHandle, EngineError> {
        self.dispatch_inner(placement, payload, Some(tokenizer))
            .await
    }

    async fn dispatch_inner(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
        tokenizer: Option<&Arc<dyn Tokenizer>>,
    ) -> Result<TokenHandle, EngineError> {
        match placement {
            PlacementPlan::Single { worker, .. } => {
                self.dispatch_one(worker, payload, ProtoErrorRole::Worker, tokenizer)
                    .await
            }
            PlacementPlan::Pair {
                prefill, decode, ..
            } => {
                payload.pd_metadata = Some(PdMetadata {
                    bootstrap_host: prefill.bootstrap_host().to_string(),
                    bootstrap_port: prefill.bootstrap_port().map(|p| p as i32).unwrap_or(0),
                    bootstrap_room: (rand::random::<u32>() & (i32::MAX as u32)) as i32,
                });
                let payload_ref: &GenerationPayload = payload;
                let (prefill_result, decode_result) = tokio::join!(
                    self.dispatch_one(prefill, payload_ref, ProtoErrorRole::Prefill, tokenizer),
                    self.dispatch_one(decode, payload_ref, ProtoErrorRole::Decode, tokenizer),
                );
                let prefill_stream = prefill_result?;
                let decode_stream = decode_result?;
                Ok(merge_pd_streams(
                    prefill_stream,
                    decode_stream,
                    payload.logprob.input_logprobs,
                ))
            }
        }
    }

    #[cfg_attr(not(feature = "grpc-client"), allow(unused_variables))]
    async fn dispatch_one(
        &self,
        worker: &Arc<dyn Worker>,
        payload: &GenerationPayload,
        role: ProtoErrorRole,
        tokenizer: Option<&Arc<dyn Tokenizer>>,
    ) -> Result<TokenHandle, EngineError> {
        let mut client = get_grpc_client_from_worker(worker)
            .await
            .map_err(|_| EngineError::ConnectionAcquireFailed(worker.url().to_string()))?;

        let proto_request = match &client {
            GrpcClient::Sglang(_) => {
                ProtoGenerateRequest::Sglang(Box::new(payload_to_proto::to_sglang_proto(payload)))
            }
            GrpcClient::Vllm(_) => {
                ProtoGenerateRequest::Vllm(Box::new(payload_to_proto::to_vllm_proto(payload)))
            }
            #[cfg(feature = "grpc-client")]
            GrpcClient::Atom(atom_client) => {
                let tokenizer = tokenizer.map(|t| t.as_ref());
                let request = atom::request::generate_request(payload, worker.as_ref(), tokenizer)?;
                let stream = atom::stream::AtomStream::open(atom_client, request).await?;
                return Ok(TokenHandle::new(stream));
            }
        };

        let stream = client
            .generate(proto_request)
            .await
            .map_err(|e| EngineError::RequestBuildFailed(e.to_string()))?;

        Ok(TokenHandle::new(ProtoStreamSource::new(stream, role)))
    }
}

/// Distinguishes how a proto `Error` response on this stream is labelled:
/// single-mode worker errors flatten to `Transport`; the PD merger needs
/// `Prefill` / `DecodeError` so the spec's typed-error contract holds.
#[derive(Copy, Clone)]
pub(crate) enum ProtoErrorRole {
    Worker,
    Prefill,
    Decode,
}

pub(crate) struct ProtoStreamSource {
    stream: ProtoStream,
    role: ProtoErrorRole,
    finished: bool,
}

impl ProtoStreamSource {
    pub(crate) fn new(stream: ProtoStream, role: ProtoErrorRole) -> Self {
        Self {
            stream,
            role,
            finished: false,
        }
    }

    fn classify_proto_error(&self, message: String) -> EngineError {
        match self.role {
            ProtoErrorRole::Worker => EngineError::Transport(tonic::Status::internal(message)),
            ProtoErrorRole::Prefill => EngineError::Prefill(message),
            ProtoErrorRole::Decode => EngineError::DecodeError(message),
        }
    }
}

impl Stream for ProtoStreamSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        loop {
            // Recreate-per-poll is sound because `AbortOnDropStream::poll_next`
            // delegates to `tonic::Streaming`, which registers the waker on the
            // underlying H2 channel rather than on the returned `Next` future.
            // Dropping the future after Pending does not lose the wake signal.
            let polled = {
                let fut = self.stream.next();
                let pinned = std::pin::pin!(fut);
                std::future::Future::poll(pinned, cx)
            };
            let next = match polled {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(v) => v,
            };
            match next {
                None => {
                    self.finished = true;
                    return Poll::Ready(None);
                }
                Some(Err(status)) => {
                    self.finished = true;
                    return Poll::Ready(Some(Err(EngineError::Transport(status))));
                }
                Some(Ok(response)) => match response.into_response() {
                    ProtoResponseVariant::Chunk(chunk) => {
                        return Poll::Ready(Some(Ok(proto_to_chunk::proto_chunk_to_chunk(chunk))));
                    }
                    ProtoResponseVariant::Complete(complete) => {
                        self.finished = true;
                        return Poll::Ready(Some(Ok(proto_to_chunk::proto_complete_to_chunk(
                            complete,
                        ))));
                    }
                    ProtoResponseVariant::Error(err) => {
                        self.finished = true;
                        let engine_err = self.classify_proto_error(err.message().to_string());
                        return Poll::Ready(Some(Err(engine_err)));
                    }
                    ProtoResponseVariant::None => continue,
                },
            }
        }
    }
}

impl TokenSource for ProtoStreamSource {
    fn mark_completed(&mut self) {
        self.stream.mark_completed();
    }
}

#[cfg(test)]
mod tests;
