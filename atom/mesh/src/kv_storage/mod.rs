//! Shared KV storage the workers offload to, with `--kv-events`.
//!
//! A worker that offloads to a Mooncake Store reports it at `/kv_event_sources`
//! (`offload`, ROCm/ATOM#3275). Each store a model's workers report is a source those
//! workers read (`sources`), its chunks keyed by ATOM's rule (`keys`) and asked of each
//! pool master's admin service (`mooncake`). `cache_aware` estimates what a worker's engine
//! will report cached with what it loads from the store it reads, looked up on the request
//! path (`lookup`). `GET /kv_storage/sources` shows every source.

pub mod keys;
mod lookup;
pub mod mooncake;
pub mod sources;

pub use sources::{OffloadReport, StorageSources};
