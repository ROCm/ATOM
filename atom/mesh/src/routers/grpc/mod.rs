//! gRPC router implementations

pub(crate) mod common;
pub(crate) mod completion_adapter;
pub(crate) mod context;
pub mod engine;
pub(crate) mod pd_router; // Used by routers/factory
pub(crate) mod pipeline;
pub(crate) mod regular;
pub(crate) mod router; // Used by routers/factory
pub(crate) mod utils; // Used by routers/http

#[cfg(any())]
mod tests;
