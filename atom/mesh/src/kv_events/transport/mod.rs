//! Event transports. Live frames are `[topic, seq, payload]`; replay follows vLLM's
//! DEALER/ROUTER protocol and ends with a frame whose seq is all ones.

use async_trait::async_trait;
use bytes::Bytes;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    Batch {
        seq: u64,
        payload: Bytes,
        replayed: bool,
    },
    ReplayEnd,
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("not supported by this source")]
    Unsupported,
    #[error("transport closed")]
    Closed,
    #[error("transport error: {0}")]
    Io(String),
}

#[async_trait]
pub trait EventTransport: Send {
    async fn recv(&mut self) -> Result<Frame, TransportError>;
    /// Ask for every buffered batch from `from_seq` on; `Unsupported` without `caps.replay`.
    async fn replay(&mut self, from_seq: u64) -> Result<(), TransportError>;
    /// `Unsupported` without `caps.snapshot`.
    async fn snapshot(&mut self) -> Result<(), TransportError>;
}
