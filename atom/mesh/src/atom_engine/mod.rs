//! ATOM's protocols: `wire` (engine_v1) and `grpc` (atom.grpc.v1). Under
//! `atom-engine`, Rust drives EngineCores: `spec` is the launch hand-over, `link` one Core.

/// ATOM's gRPC service, `AtomEngine`: its messages, client and server.
#[allow(clippy::all, unused_qualifications)]
pub mod grpc {
    include!(concat!(env!("OUT_DIR"), "/atom.grpc.v1.rs"));
}
#[cfg(feature = "atom-engine")]
pub mod link;
#[cfg(feature = "atom-engine")]
pub mod spec;
#[cfg(test)]
mod tests;
pub mod wire;

#[cfg(feature = "atom-engine")]
pub use link::{CoreLink, Interrupted};
#[cfg(feature = "atom-engine")]
pub use spec::{CoreSpec, EngineSpec};
