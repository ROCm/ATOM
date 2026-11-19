//! Token path.
//!
//! Entrypoints has already encoded the prompt. This module routes on those
//! token ids, orchestrates ATOM prefill/decode, and forwards the decode token
//! stream one event at a time. It does not render, tokenize, or detokenize.

mod engine;
mod router;
mod types;

pub use engine::{grpc_endpoint, AtomEngine, GrpcAtomEngine};
pub use router::{TokenRouter, TokenStream};
pub use types::{AbortRequests, TokenGenerateEvent, TokenGenerateRequest, TokenUsage};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use thiserror::Error;

use crate::core::placement::{AdapterError, PlacementError};
use crate::core::Worker;
use crate::observability::metrics::MeshMetrics;

#[derive(Debug, Error)]
pub enum TokenPathError {
    #[error("token_ids is empty")]
    EmptyTokens,
    #[error("model is empty")]
    MissingModel,
    #[error("token path supports n=1 only")]
    UnsupportedN,
    #[error("token path only speaks gRPC to ATOM engines")]
    UnsupportedBackend,
    #[error("session_id is not a valid HTTP header value")]
    InvalidSession,
    #[error("request_id is already in flight")]
    DuplicateRequest,
    #[error("{0}")]
    Placement(#[from] PlacementError),
    #[error("{role} failed: {message}")]
    Engine { role: &'static str, message: String },
    #[error("prefill response missing kv_transfer_params")]
    MissingKv,
    #[error("{0}")]
    Adapter(#[from] AdapterError),
}

impl TokenPathError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::EmptyTokens => "empty_token_ids",
            Self::MissingModel => "missing_model",
            Self::UnsupportedN => "unsupported_n",
            Self::UnsupportedBackend => "unsupported_backend",
            Self::InvalidSession => "invalid_session_id",
            Self::DuplicateRequest => "duplicate_request_id",
            Self::Placement(PlacementError::NoWorkers) => "no_workers",
            Self::Placement(PlacementError::NoAvailableWorkers) => "no_available_workers",
            Self::Placement(PlacementError::NoPrefillWorkers) => "no_prefill_workers",
            Self::Placement(PlacementError::NoDecodeWorkers) => "no_decode_workers",
            Self::Placement(PlacementError::PolicyReturnedNone) => "policy_returned_none",
            Self::Placement(PlacementError::ModelNotFound { .. }) => "model_not_found",
            Self::Engine { .. } => "engine_error",
            Self::MissingKv => "prefill_missing_kv_transfer_params",
            Self::Adapter(_) => "atom_adapter",
        }
    }

    pub fn status_code(&self) -> http::StatusCode {
        use http::StatusCode;
        match self {
            Self::EmptyTokens
            | Self::MissingModel
            | Self::UnsupportedN
            | Self::UnsupportedBackend
            | Self::InvalidSession => StatusCode::BAD_REQUEST,
            Self::DuplicateRequest => StatusCode::CONFLICT,
            Self::Placement(_) => StatusCode::SERVICE_UNAVAILABLE,
            Self::Engine { .. } | Self::MissingKv => StatusCode::BAD_GATEWAY,
            Self::Adapter(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// Shared cancel flag for an in-flight token request.
///
/// Dropping the decode stream aborts prefill and decode. An explicit
/// `abort_requests` uses the same flag so the engine is cancelled once.
struct AbortState {
    finished: AtomicBool,
    workers: Vec<Arc<dyn Worker>>,
    request_id: String,
    engine: Arc<dyn AtomEngine>,
    inflight: Arc<DashMap<String, Arc<AbortState>>>,
}

impl AbortState {
    fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    /// Drop the in-flight entry without aborting. Used when the engine call
    /// never started, or this task already aborted the workers itself.
    fn release(&self) {
        self.finished.store(true, Ordering::SeqCst);
        self.inflight.remove(&self.request_id);
    }

    fn cancel_finished(&self) {
        self.finished.store(true, Ordering::SeqCst);
        self.inflight.remove(&self.request_id);
    }

    fn cancel(&self, reason: &'static str) {
        if self.finished.swap(true, Ordering::SeqCst) {
            return;
        }
        self.inflight.remove(&self.request_id);
        MeshMetrics::record_token_abort(reason);
        let engine = self.engine.clone();
        let workers = self.workers.clone();
        let ids = vec![self.request_id.clone()];
        tokio::spawn(async move {
            for worker in workers {
                if let Err(error) = engine.abort(worker.as_ref(), &ids).await {
                    tracing::warn!(
                        request_id = %ids[0],
                        worker = worker.url(),
                        %error,
                        "token-path abort failed"
                    );
                }
            }
        });
    }
}
