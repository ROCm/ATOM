//! Relocated /v1/responses implementation; depends on grpc::Pipeline (concrete type only).

pub(crate) mod context;
pub(crate) mod conversation;
pub(crate) mod conversions;
pub(crate) mod handlers;
pub(crate) mod non_streaming;
pub(crate) mod persistence;
pub(crate) mod retrieve;
pub(crate) mod streaming;

#[cfg(any())]
mod tests;
