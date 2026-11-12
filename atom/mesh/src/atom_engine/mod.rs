//! The ATOM engine runtime: Rust as the one client of every EngineCore.
//!
//! Python spawns the Cores and hands over an `EngineSpec`; Rust then drives
//! them over the engine_v1 protocol. `spec` is that hand-over, `wire` is the
//! protocol's encoding, `link` is one Core's sockets. Nothing here knows HTTP.

pub mod link;
pub mod spec;
#[cfg(test)]
mod tests;
pub mod wire;

pub use link::{CoreLink, Interrupted};
pub use spec::{CoreSpec, EngineSpec};
