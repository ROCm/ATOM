//! gRPC router implementations

pub(crate) mod completion_adapter;
pub mod engine;
pub(crate) mod pd_router; // Used by routers/factory
pub(crate) mod pipeline;
pub(crate) mod router; // Used by routers/factory

#[cfg(any())]
mod tests;
