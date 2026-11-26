//! The ATOM engine runtime: Rust as the one client of every EngineCore.
//!
//! Python spawns the Cores and hands over an `EngineSpec`; from then on this
//! module owns their sockets. `link` is one Core's sockets, `client` drives
//! all of them, `wire` is the protocol's encoding. Nothing here knows HTTP or
//! places requests: the router chooses each request's DP rank
//! (`routers::atom_bridge::direct`).

mod client;
mod link;
pub mod spec;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
pub mod wire;

pub use client::{EngineClient, Finished, SeqEvent, SeqFailure, SeqHandle, SeqTimes, Submission};
pub use link::Interrupted;
pub use spec::{BalanceSpec, BalanceStrategy, EngineSpec, ModelSpec, ReasoningToggle};

/// A connected engine, the model it serves and how its DP ranks are shared.
pub struct AtomEngine {
    pub model: ModelSpec,
    pub balance: BalanceSpec,
    pub client: EngineClient,
}

impl AtomEngine {
    /// Parse `spec_json` and connect to every Core in it. Blocks until all
    /// are ready, or `interrupted` says to give up.
    pub fn start(spec_json: &str, interrupted: Interrupted<'_>) -> Result<Self, String> {
        let spec = EngineSpec::from_json(spec_json)?;
        let client = EngineClient::connect(&spec, interrupted)?;
        Ok(Self {
            model: spec.model,
            balance: spec.balance,
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
