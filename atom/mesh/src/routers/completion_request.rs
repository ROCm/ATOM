//! Mesh-local completion request with a prompt widened to accept token ids.

use serde::{de, Deserialize, Serialize};
use serde_json::Value;

use crate::protocols::completion::CompletionRequest as ProtocolCompletionRequest;

// u32 to match the token-id width used elsewhere (prepare: token_ids: Vec<u32>).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)] // declaration order matters: string, then ints, then strings, then int-batches
pub enum CompletionPrompt {
    String(String),
    Tokens(Vec<u32>),
    Texts(Vec<String>),
    TokenBatches(Vec<Vec<u32>>),
}

#[derive(Clone)]
pub struct CompletionRequest {
    pub prompt: CompletionPrompt,
    // `inner.prompt` is a placeholder; always read `prompt` above instead.
    pub inner: ProtocolCompletionRequest,
}

impl CompletionRequest {
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    pub fn is_stream(&self) -> bool {
        self.inner.stream
    }

    pub fn routing_text(&self) -> String {
        match &self.prompt {
            CompletionPrompt::String(s) => s.clone(),
            CompletionPrompt::Texts(v) => v.first().cloned().unwrap_or_default(),
            CompletionPrompt::Tokens(_) | CompletionPrompt::TokenBatches(_) => String::new(),
        }
    }
}

impl<'de> Deserialize<'de> for CompletionRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;
        let obj = value
            .as_object_mut()
            .ok_or_else(|| de::Error::custom("completion request must be a JSON object"))?;

        // Swap in a placeholder so the crate struct's narrower prompt still parses.
        let prompt = match obj.get_mut("prompt") {
            Some(slot) => {
                let raw = slot.take();
                let parsed: CompletionPrompt =
                    serde_json::from_value(raw).map_err(de::Error::custom)?;
                *slot = Value::String(String::new());
                parsed
            }
            None => return Err(de::Error::missing_field("prompt")),
        };

        let inner: ProtocolCompletionRequest =
            serde_json::from_value(value).map_err(de::Error::custom)?;

        Ok(CompletionRequest { prompt, inner })
    }
}

impl Serialize for CompletionRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Emit the crate struct (keeps its serde attrs), then override the prompt.
        let mut value = serde_json::to_value(&self.inner).map_err(serde::ser::Error::custom)?;
        let prompt_value = serde_json::to_value(&self.prompt).map_err(serde::ser::Error::custom)?;
        if let Value::Object(map) = &mut value {
            map.insert("prompt".to_string(), prompt_value);
        }
        value.serialize(serializer)
    }
}
