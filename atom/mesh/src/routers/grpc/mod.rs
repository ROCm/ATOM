//! gRPC router implementations

pub(crate) mod completion_adapter;
pub mod engine;
pub(crate) mod pd_router;
pub(crate) mod pipeline;
pub(crate) mod router;

#[cfg(test)]
mod tests;
