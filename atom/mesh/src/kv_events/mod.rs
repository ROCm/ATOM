//! KV cache events of the workers' engines (ROCm/ATOM#3275), enabled by `--kv-events`.
//!
//! Each registered worker rank reports where it publishes them (`source`); the monitor
//! subscribes to every rank, decodes its batches (`wire`) and admits them in sequence
//! (`rank_state`). `GET /kv_events/sources` shows every subscription.

pub mod model;
pub mod monitor;
pub mod rank_state;
pub mod source;
pub mod wire;

pub use monitor::{KvEventMonitor, SourceStatus};
pub use source::KvEventSource;
