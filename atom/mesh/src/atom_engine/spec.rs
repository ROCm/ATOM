//! `EngineSpec`: everything Python hands Rust once at launch, as JSON.
//!
//! Built by `atom/entrypoints/atomesh/engine_spec.py`; after it is parsed Rust
//! never calls back into Python.

use serde::Deserialize;
use serde_json::{Map, Value};

#[derive(Debug, Clone, Deserialize)]
pub struct EngineSpec {
    pub model: ModelSpec,
    pub balance: BalanceSpec,
    pub engines: Vec<CoreSpec>,
}

/// What the frontend needs to prepare requests and render responses.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSpec {
    pub served_name: String,
    pub tokenizer_path: String,
    #[serde(default)]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub default_chat_template_kwargs: Map<String, Value>,
    #[serde(default)]
    pub max_model_len: Option<u64>,
    pub default_max_tokens: u32,
    /// Speculative decoding is on; the Core cannot return logprobs then.
    #[serde(default)]
    pub speculative: bool,
}

/// How requests are spread over DP ranks; mirrors CoreManager's settings.
#[derive(Debug, Clone, Deserialize)]
pub struct BalanceSpec {
    pub strategy: BalanceStrategy,
    pub req_equiv: u64,
    #[serde(default)]
    pub session_affinity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BalanceStrategy {
    RoundRobin,
    LeastRequests,
    LeastTokens,
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
