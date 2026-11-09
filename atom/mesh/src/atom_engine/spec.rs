//! `EngineSpec`: everything Python hands Rust once at launch, as JSON.
//!
//! Built by the Python launcher that spawns the Cores; after it is parsed Rust
//! never calls back into Python. Fields added later default, so an older
//! spec still parses.

use serde::Deserialize;
use serde_json::{Map, Value};

use super::wire::pb;

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
    /// `/v1/models`' deployment fields (`deployment_metadata`).
    #[serde(default)]
    pub model_card: Map<String, Value>,
    #[serde(default)]
    pub max_model_len: Option<u64>,
    pub default_max_tokens: u32,
    /// The deployment's cap on `max_tokens` (`deployment_metadata`): an
    /// explicit request over it is refused, the default is cut to it.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// Speculative decoding forces acceptance at fixed rates, so it has no real
    /// token logprobs (`SamplingParams.validate_logprob_options`).
    #[serde(default)]
    pub synthetic_acceptance: bool,
    /// What a `tool_choice="required"` request is held to, already ending
    /// on the engine's end ids; none where the tool-call format has none.
    #[serde(default)]
    pub required_channel: Option<RequiredChannel>,
    /// The template kwarg that switches reasoning, as `resolve_reasoning_toggle`
    /// probed it; none where the template has no switch.
    #[serde(default)]
    pub reasoning_toggle: Option<ReasoningToggle>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RequiredChannel {
    pub after: Vec<i32>,
    pub opening: Vec<i32>,
    pub opens_at: u32,
    pub end_ids: Vec<i32>,
}

impl From<&RequiredChannel> for pb::RequiredChannel {
    fn from(c: &RequiredChannel) -> Self {
        Self {
            after: c.after.clone(),
            opening: c.opening.clone(),
            opens_at: c.opens_at,
            end_ids: c.end_ids.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReasoningToggle {
    pub name: String,
    pub off: Value,
    pub on: Value,
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
