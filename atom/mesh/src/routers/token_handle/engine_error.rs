//! Failure modes emitted by `GrpcEngine::dispatch` and `merge_pd_streams`.

use std::fmt;

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
