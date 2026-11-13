//! KV event ingestion: engine events update the router's prefix index.

pub mod model;
pub mod rank_state;
pub mod source;
pub mod wire;

pub use self::{
    model::{DropReason, EngineHash, Group, GroupKind, KvEvent, Namespace, StoredBlock, Tier},
    rank_state::{Phase, Trust},
    source::{OffloadLoad, PoolRole, WireProfile},
};
