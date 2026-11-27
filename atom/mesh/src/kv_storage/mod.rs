//! Shared KV storage the workers offload to, with `--kv-events`.
//!
//! A worker that offloads to a Mooncake Store reports it at `/kv_event_sources`
//! (`offload`, ROCm/ATOM#3275). Each store a model's workers report alike is a source
//! those workers read (`sources`), its chunks keyed by ATOM's rule (`keys`) and asked of
//! each pool master's admin service (`mooncake`). With `--kv-storage-weight`, what a
//! worker's engine will load from the store it reads, looked up on the request path
//! (`lookup`), counts in `cache_aware`'s match rate and estimate: the policy asks
//! `StorageSources::cached_tokens` for it. `GET /kv_storage/sources` shows every source.

pub(crate) mod keys;
mod lookup;
pub(crate) mod mooncake;
pub mod sources;

pub use sources::{OffloadReport, StorageSources};
