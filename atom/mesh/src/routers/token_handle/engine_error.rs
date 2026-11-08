//! Failure modes emitted by the engines (`GrpcEngine::dispatch`,
//! `merge_pd_streams`, the ATOM engine) and the HTTP answer to each.

use std::fmt;

use axum::response::Response;
use http::StatusCode;

use crate::routers::comm::error;

#[derive(Debug)]
pub enum EngineError {
    Transport(tonic::Status),
    /// Prefill stream yielded a proto `Error` (PD mode, `WaitingPrefill`).
    Prefill(String),
    /// Decode stream yielded a proto `Error` while we were already streaming.
    DecodeError(String),
    /// Prefill stream closed without `Complete` or `Error`.
    PrefillEarlyClose,
    /// Decode stream closed without `Complete` or `Error`.
    DecodeIncomplete,
    ConnectionAcquireFailed(String),
    RequestBuildFailed(String),
    /// The engine refused or lost the request; `status` is the HTTP answer.
    Rejected {
        status: StatusCode,
        reason: String,
    },
}

impl EngineError {
    pub fn into_response(self) -> Response {
        match self {
            Self::Transport(s) => error::service_unavailable(
                "engine_transport_error",
                format!("transport error: {}", s),
            ),
            Self::Prefill(m) => {
                error::internal_error("engine_prefill_error", format!("prefill error: {}", m))
            }
            Self::DecodeError(m) => {
                error::internal_error("engine_decode_error", format!("decode error: {}", m))
            }
            Self::PrefillEarlyClose => error::internal_error(
                "engine_prefill_early_close",
                "prefill stream closed without Complete",
            ),
            Self::DecodeIncomplete => error::internal_error(
                "engine_decode_incomplete",
                "decode stream closed without Complete",
            ),
            Self::ConnectionAcquireFailed(r) => {
                error::service_unavailable("engine_connection_failed", r)
            }
            Self::RequestBuildFailed(r) => error::bad_request("engine_request_build_failed", r),
            Self::Rejected { status, reason } => {
                error::create_error(status, "engine_rejected", reason)
            }
        }
    }
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(s) => write!(f, "transport: {}", s),
            Self::Prefill(m) => write!(f, "prefill error: {}", m),
            Self::DecodeError(m) => write!(f, "decode error: {}", m),
            Self::PrefillEarlyClose => f.write_str("prefill stream closed without Complete"),
            Self::DecodeIncomplete => f.write_str("decode stream closed without Complete"),
            Self::ConnectionAcquireFailed(r) => write!(f, "connection acquire failed: {}", r),
            Self::RequestBuildFailed(r) => write!(f, "request build failed: {}", r),
            Self::Rejected { reason, .. } => write!(f, "rejected: {}", reason),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(s) => Some(s),
            _ => None,
        }
    }
}
