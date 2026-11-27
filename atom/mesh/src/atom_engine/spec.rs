//! `EngineSpec`: everything Python hands Rust once at launch, as JSON.
//!
//! Built by the Python launcher that spawns the Cores; after it is parsed Rust
//! never calls back into Python. Every field is required; fields the spec
//! does not know are ignored.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct EngineSpec {
    pub engines: Vec<CoreSpec>,
}

/// One EngineCore: its process and the three addresses its client binds.
#[derive(Debug, Clone, Deserialize)]
pub struct CoreSpec {
    pub dp_rank: usize,
    pub pid: u32,
    pub input: String,
    pub control: String,
    pub output: String,
}

impl EngineSpec {
    pub fn from_json(json: &str) -> Result<Self, String> {
        let spec: Self = serde_json::from_str(json).map_err(|e| format!("bad engine spec: {e}"))?;
        if spec.engines.is_empty() {
            return Err("engine spec lists no EngineCore".into());
        }
        if spec.engines.iter().enumerate().any(|(i, e)| e.dp_rank != i) {
            return Err("engine spec must list EngineCores in dp_rank order".into());
        }
        Ok(spec)
    }
}
