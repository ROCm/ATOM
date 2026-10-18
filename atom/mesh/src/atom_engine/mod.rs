//! The ATOM engine runtime: Rust as the one client of every EngineCore.
//!
//! Python spawns the Cores and hands over an `EngineSpec`; from then on this
//! module owns their sockets. `link` is one Core's sockets, `client` drives
//! all of them (placement in `balance`), `wire` is the protocol's encoding.
//! Nothing here knows HTTP: `routers::atom_engine` is the frontend over it.

mod balance;
mod client;
mod link;
pub mod spec;
pub mod wire;

pub use client::{EngineClient, Finished, SeqEvent, SeqFailure, SeqHandle, Submission};
pub use spec::{EngineSpec, ModelSpec};

/// A connected engine and the model it serves.
pub struct AtomEngine {
    pub model: ModelSpec,
    pub client: EngineClient,
}

impl AtomEngine {
    /// Parse `spec_json` and connect to every Core in it. Blocks until all
    /// are ready.
    pub fn start(spec_json: &str) -> Result<Self, String> {
        let spec = EngineSpec::from_json(spec_json)?;
        let client = EngineClient::connect(&spec)?;
        Ok(Self {
            model: spec.model,
            client,
        })
    }
}

impl std::fmt::Debug for AtomEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomEngine")
            .field("model", &self.model.served_name)
            .field("cores", &self.client.core_count())
            .finish()
    }
}
