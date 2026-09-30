//! Validated routing views; callers retain raw request bodies.
use crate::protocols::{
    chat::ChatCompletionRequest,
    common::{GenerationRequest, InputIds, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    validated::Normalizable,
};
use validator::Validate;

#[derive(Debug, Clone)]
pub struct InferenceMetadata {
    pub route: &'static str,
    pub model: Option<String>,
    pub text: String,
    pub batch_size: Option<usize>,
    pub stream: bool,
    pub return_logprob: bool,
}

impl InferenceMetadata {
    /// Execution needs only protocol facts; routing text can be released after placement.
    pub fn execution_metadata(&self) -> Self {
        Self {
            route: self.route,
            model: self.model.clone(),
            text: String::new(),
            batch_size: self.batch_size,
            stream: self.stream,
            return_logprob: self.return_logprob,
        }
    }
}

pub trait InferenceRequest {
    fn metadata(&self) -> InferenceMetadata;
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        let mut metadata = self.metadata();
        if !include_text {
            metadata.text.clear();
        }
        metadata
    }
}
impl InferenceRequest for ChatCompletionRequest {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        InferenceMetadata {
            route: "/v1/chat/completions",
            model: Some(self.model.clone()),
            text: if include_text {
                self.extract_text_for_routing()
            } else {
                String::new()
            },
            batch_size: self.n.filter(|n| *n > 1).map(|n| n as usize),
            stream: self.is_stream(),
            return_logprob: self.logprobs,
        }
    }
}
impl InferenceRequest for CompletionRequest {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        let batch_size = match &self.prompt {
            StringOrArray::Array(values) if !values.is_empty() => Some(values.len()),
            _ => None,
        };
        InferenceMetadata {
            route: "/v1/completions",
            model: Some(self.model.clone()),
            text: if include_text {
                self.extract_text_for_routing()
            } else {
                String::new()
            },
            batch_size,
            stream: self.is_stream(),
            return_logprob: self.logprobs.is_some(),
        }
    }
}
impl InferenceRequest for GenerateRequest {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        let batch_size = match &self.input_ids {
            Some(InputIds::Batch(values)) if !values.is_empty() => Some(values.len()),
            _ => None,
        };
        InferenceMetadata {
            route: "/generate",
            model: self.model.clone(),
            text: if include_text {
                self.extract_text_for_routing()
            } else {
                String::new()
            },
            batch_size,
            stream: self.is_stream(),
            return_logprob: self.return_logprob.unwrap_or(false),
        }
    }
}

pub enum ParsedInference {
    Chat(ChatCompletionRequest),
    Completion(CompletionRequest),
    Generate(GenerateRequest),
    Messages(serde_json::Value),
    Responses(serde_json::Value),
}
impl ParsedInference {
    pub fn parse(path: &str, bytes: &[u8]) -> Result<Self, String> {
        use crate::routers::ingress::{Api, EndpointSpec};
        let endpoint = EndpointSpec::find(path).ok_or("unsupported inference API")?;
        match endpoint.api {
            Api::Chat => {
                let mut request: ChatCompletionRequest =
                    serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                request.normalize();
                request.validate().map_err(|e| e.to_string())?;
                Ok(Self::Chat(request))
            }
            Api::Completions => serde_json::from_slice(bytes)
                .map(Self::Completion)
                .map_err(|e| e.to_string()),
            Api::Generate => serde_json::from_slice(bytes)
                .map(Self::Generate)
                .map_err(|e| e.to_string()),
            Api::Messages | Api::Responses => {
                let body: serde_json::Value =
                    serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                Self::validate_api_body(&body, endpoint.api == Api::Messages)?;
                Ok(if endpoint.api == Api::Messages {
                    Self::Messages(body)
                } else {
                    Self::Responses(body)
                })
            }
        }
    }
    pub fn requires_state_domain(&self) -> bool {
        let Self::Responses(body) = self else {
            return false;
        };
        ["previous_response_id", "conversation"]
            .iter()
            .any(|key| body.get(*key).is_some_and(|v| !v.is_null()))
    }

    fn validate_api_body(body: &serde_json::Value, messages: bool) -> Result<(), String> {
        use serde_json::Value;
        if !body.is_object() {
            return Err("request must be a JSON object".into());
        }
        if !body.get("model").is_some_and(Self::is_nonempty_string) {
            return Err("model must be a nonempty string".into());
        }
        Self::validate_optional_field(body, "stream", Value::is_boolean, "a boolean")?;
        Self::validate_optional_field(body, "metadata", Value::is_object, "an object")?;
        Self::validate_optional_field(
            body,
            "tools",
            |v| v.as_array().is_some_and(|a| a.iter().all(Value::is_object)),
            "an array of objects",
        )?;
        Self::validate_optional_field(
            body,
            "temperature",
            |v| {
                v.as_f64()
                    .is_some_and(|n| (0.0..=if messages { 1.0 } else { 2.0 }).contains(&n))
            },
            "within the API temperature range",
        )?;
        Self::validate_optional_field(
            body,
            "top_p",
            |v| v.as_f64().is_some_and(|n| (0.0..=1.0).contains(&n)),
            "between 0 and 1",
        )?;
        if messages {
            Self::validate_messages(body)
        } else {
            Self::validate_responses(body)
        }
    }

    fn validate_messages(body: &serde_json::Value) -> Result<(), String> {
        use serde_json::Value;
        if body
            .get("max_tokens")
            .and_then(Value::as_u64)
            .is_none_or(|n| n == 0)
        {
            return Err("max_tokens must be a positive integer".into());
        }
        let items = body
            .get("messages")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .ok_or("messages must be a nonempty array")?;
        for item in items {
            if !matches!(
                item.get("role").and_then(Value::as_str),
                Some("user" | "assistant")
            ) {
                return Err("messages[].role must be user or assistant".into());
            }
            if !item.get("content").is_some_and(|v| {
                v.is_string()
                    || v.as_array()
                        .is_some_and(|a| !a.is_empty() && a.iter().all(Self::is_message_block))
            }) {
                return Err("messages[].content must be text or an array of content blocks".into());
            }
        }
        Self::validate_optional_field(
            body,
            "system",
            |v| {
                v.is_string()
                    || v.as_array().is_some_and(|a| {
                        a.iter().all(|b| {
                            b.get("type").and_then(Value::as_str) == Some("text")
                                && b.get("text").is_some_and(Value::is_string)
                        })
                    })
            },
            "text or an array of text blocks",
        )?;
        Self::validate_optional_field(
            body,
            "top_k",
            |v| v.as_u64().is_some(),
            "a nonnegative integer",
        )?;
        Self::validate_optional_field(
            body,
            "stop_sequences",
            |v| {
                v.as_array()
                    .is_some_and(|a| a.iter().all(Self::is_nonempty_string))
            },
            "an array of nonempty strings",
        )?;
        Self::validate_optional_field(body, "tool_choice", Value::is_object, "an object")?;
        Ok(())
    }

    fn validate_responses(body: &serde_json::Value) -> Result<(), String> {
        use serde_json::Value;
        Self::validate_optional_field(
            body,
            "input",
            |v| {
                v.is_string()
                    || v.as_array()
                        .is_some_and(|items| items.iter().all(Self::is_response_item))
            },
            "text or an array of input items",
        )?;
        Self::validate_optional_field(body, "instructions", Value::is_string, "a string")?;
        Self::validate_optional_field(
            body,
            "previous_response_id",
            Self::is_nonempty_string,
            "a nonempty string",
        )?;
        Self::validate_optional_field(
            body,
            "conversation",
            |v| Self::is_nonempty_string(v) || v.get("id").is_some_and(Self::is_nonempty_string),
            "a conversation ID or an object containing id",
        )?;
        if ["previous_response_id", "conversation"]
            .iter()
            .all(|key| body.get(*key).is_some_and(|v| !v.is_null()))
        {
            return Err("conversation and previous_response_id are mutually exclusive".into());
        }
        for key in ["background", "store", "parallel_tool_calls"] {
            Self::validate_optional_field(body, key, Value::is_boolean, "a boolean")?;
        }
        for key in ["max_output_tokens", "max_tool_calls"] {
            Self::validate_optional_field(
                body,
                key,
                |v| v.as_u64().is_some_and(|n| n > 0),
                "a positive integer",
            )?;
        }
        Self::validate_optional_field(
            body,
            "tool_choice",
            |v| v.is_string() || v.is_object(),
            "a string or object",
        )?;
        Ok(())
    }

    fn validate_optional_field(
        body: &serde_json::Value,
        key: &str,
        valid: impl FnOnce(&serde_json::Value) -> bool,
        expected: &str,
    ) -> Result<(), String> {
        if let Some(value) = body.get(key).filter(|v| !v.is_null()) {
            if !valid(value) {
                return Err(format!("{key} must be {expected}"));
            }
        }
        Ok(())
    }

    fn is_nonempty_string(value: &serde_json::Value) -> bool {
        value.as_str().is_some_and(|s| !s.trim().is_empty())
    }

    fn is_response_item(value: &serde_json::Value) -> bool {
        use serde_json::Value;
        match value.get("type").and_then(Value::as_str) {
            None | Some("message") => {
                matches!(
                    value.get("role").and_then(Value::as_str),
                    Some("user" | "assistant" | "system" | "developer")
                ) && value.get("content").is_some_and(|v| {
                    v.is_string()
                        || v.as_array().is_some_and(|blocks| {
                            blocks.iter().all(|block| {
                                match block.get("type").and_then(Value::as_str) {
                                    Some("input_text" | "output_text") => {
                                        block.get("text").is_some_and(Value::is_string)
                                    }
                                    Some(kind) => !kind.trim().is_empty(),
                                    None => false,
                                }
                            })
                        })
                })
            }
            Some(kind) => !kind.trim().is_empty(),
        }
    }

    fn is_message_block(value: &serde_json::Value) -> bool {
        use serde_json::Value;
        match value.get("type").and_then(Value::as_str) {
            Some("text") => value.get("text").is_some_and(Value::is_string),
            Some("tool_use") => {
                ["id", "name"]
                    .iter()
                    .all(|k| value.get(*k).is_some_and(Self::is_nonempty_string))
                    && value.get("input").is_some_and(Value::is_object)
            }
            Some("tool_result") => {
                value
                    .get("tool_use_id")
                    .is_some_and(Self::is_nonempty_string)
                    && value.get("content").is_none_or(|v| {
                        v.is_string()
                            || v.as_array()
                                .is_some_and(|blocks| blocks.iter().all(Self::is_message_block))
                    })
            }
            // New block types and their extensions remain backend-defined.
            Some(kind) => !kind.trim().is_empty(),
            None => false,
        }
    }

    pub fn input_tokens(&self) -> Result<Option<Vec<u32>>, &'static str> {
        match self {
            Self::Generate(GenerateRequest {
                input_ids: Some(InputIds::Single(ids)),
                ..
            }) => ids
                .iter()
                .map(|&id| u32::try_from(id))
                .collect::<Result<Vec<_>, _>>()
                .map(Some)
                .map_err(|_| "token routing requires nonnegative input_ids"),
            Self::Generate(GenerateRequest {
                input_ids: Some(InputIds::Batch(_)),
                ..
            }) => Err("token routing requires a single input_ids array"),
            _ => Ok(None),
        }
    }
    pub fn chat(&self) -> Option<&ChatCompletionRequest> {
        if let Self::Chat(chat) = self {
            Some(chat)
        } else {
            None
        }
    }
}
impl InferenceRequest for ParsedInference {
    fn metadata(&self) -> InferenceMetadata {
        self.metadata_with_text(true)
    }
    fn metadata_with_text(&self, include_text: bool) -> InferenceMetadata {
        match self {
            Self::Chat(r) => r.metadata_with_text(include_text),
            Self::Completion(r) => r.metadata_with_text(include_text),
            Self::Generate(r) => r.metadata_with_text(include_text),
            Self::Messages(body) | Self::Responses(body) => {
                let messages = matches!(self, Self::Messages(_));
                let keys: &[&str] = if messages {
                    &["system", "messages", "tools", "tool_choice"]
                } else {
                    &[
                        "instructions",
                        "input",
                        "tools",
                        "tool_choice",
                        "previous_response_id",
                        "conversation",
                    ]
                };
                let text = if include_text {
                    let view: std::collections::BTreeMap<&str, &serde_json::Value> = keys
                        .iter()
                        .filter_map(|key| body.get(*key).map(|v| (*key, v)))
                        .collect();
                    serde_json::to_string(&view).expect("JSON values are serializable")
                } else {
                    String::new()
                };
                InferenceMetadata {
                    route: if messages {
                        "/v1/messages"
                    } else {
                        "/v1/responses"
                    },
                    model: body["model"].as_str().map(str::to_owned),
                    text,
                    batch_size: None,
                    stream: body["stream"].as_bool().unwrap_or(false),
                    return_logprob: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn execution_facts_follow_typed_null_and_batch_semantics_without_retaining_text() {
        for (route, body, batch, stream, logprobs) in [
            (
                "/v1/chat/completions",
                r#"{"model":"m","messages":[{"role":"user","content":"large prompt"}],"n":4,"stream":true,"logprobs":true}"#,
                Some(4),
                true,
                true,
            ),
            (
                "/v1/completions",
                r#"{"model":"m","prompt":["a","b"],"logprobs":0}"#,
                Some(2),
                false,
                true,
            ),
            (
                "/v1/completions",
                r#"{"model":"m","prompt":[],"logprobs":null}"#,
                None,
                false,
                false,
            ),
            (
                "/generate",
                r#"{"model":"m","input_ids":[[1,2],[3,4]],"return_logprob":null}"#,
                Some(2),
                false,
                false,
            ),
        ] {
            let parsed = ParsedInference::parse(route, body.as_bytes()).unwrap();
            let metadata = parsed.metadata();
            assert_eq!(
                (
                    metadata.batch_size,
                    metadata.stream,
                    metadata.return_logprob
                ),
                (batch, stream, logprobs)
            );
            let execution = metadata.execution_metadata();
            assert!(execution.text.is_empty());
            assert_eq!(execution.model.as_deref(), Some("m"));
            assert_eq!(
                (
                    execution.route,
                    execution.batch_size,
                    execution.stream,
                    execution.return_logprob
                ),
                (route, batch, stream, logprobs)
            );
        }
    }
}
