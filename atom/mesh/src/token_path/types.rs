//! Token request and token event exchanged with entrypoints and the ATOM engine.
//!
//! Mesh reads routing fields (`kv_transfer_params`, `data_parallel_rank`,
//! `session_id`) and forwards every other field, including `request_id`.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

fn empty_object() -> Value {
    json!({})
}

/// Already-tokenized generate request. Entrypoints fills `token_ids`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenGenerateRequest {
    pub request_id: String,
    pub model: String,
    pub token_ids: Vec<u32>,
    #[serde(default = "empty_object")]
    pub sampling_params: Value,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub kv_transfer_params: Option<Value>,
    #[serde(default)]
    pub data_parallel_rank: Option<u32>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

/// One engine step. MTP may put several new ids in a single event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenGenerateEvent {
    #[serde(default)]
    pub token_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_transfer_params: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbortRequests {
    pub request_ids: Vec<String>,
}

impl TokenGenerateRequest {
    pub fn validate(&self) -> Result<(), super::TokenPathError> {
        if self.token_ids.is_empty() {
            return Err(super::TokenPathError::EmptyTokens);
        }
        if self.model.is_empty() {
            return Err(super::TokenPathError::MissingModel);
        }
        let n = self.sampling_params.get("n").and_then(Value::as_u64);
        if n.is_some_and(|n| n != 1) {
            return Err(super::TokenPathError::UnsupportedN);
        }
        Ok(())
    }
}
