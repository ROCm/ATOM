//! Validated routing views; callers retain raw request bodies.
use crate::protocols::{
    chat::ChatCompletionRequest,
    common::{GenerationRequest, InputIds, StringOrArray},
    completion::CompletionRequest,
    generate::GenerateRequest,
    validated::Normalizable,
};
use serde::{Serialize, Serializer};
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
    /// Every ingress parses here, so an empty model is rejected for all routers.
    pub fn parse(path: &str, bytes: &[u8]) -> Result<Self, String> {
        let parsed = Self::parse_body(path, bytes)?;
        if parsed
            .metadata_with_text(false)
            .model
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err("model is required".into());
        }
        Ok(parsed)
    }

    fn parse_body(path: &str, bytes: &[u8]) -> Result<Self, String> {
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
            || body
                .get("input")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|items| items.iter().any(Self::is_item_reference))
    }

    /// `item_reference` items may omit `type`; an untyped item with an `id` and
    /// no message `role` refers to stored state.
    fn is_item_reference(item: &serde_json::Value) -> bool {
        use serde_json::Value;
        match item.get("type").and_then(Value::as_str) {
            Some(kind) => kind == "item_reference",
            None => {
                item.get("id").is_some_and(Self::is_nonempty_string) && item.get("role").is_none()
            }
        }
    }

    /// Validate only what routing reads: the body shape, `model`, `stream` and
    /// the Responses state keys. Everything else is the backend's to accept or
    /// reject, since backends extend these APIs (optional `max_tokens`,
    /// `top_k: -1`, other roles and sampling ranges).
    fn validate_api_body(body: &serde_json::Value, messages: bool) -> Result<(), String> {
        use serde_json::Value;
        if !body.is_object() {
            return Err("request must be a JSON object".into());
        }
        if !body.get("model").is_some_and(Self::is_nonempty_string) {
            return Err("model must be a nonempty string".into());
        }
        Self::validate_optional_field(body, "stream", Value::is_boolean, "a boolean")?;
        if messages {
            Ok(())
        } else {
            Self::validate_state_keys(body)
        }
    }

    /// The keys `requires_state_domain` reads to keep stateful requests on
    /// their owner.
    fn validate_state_keys(body: &serde_json::Value) -> Result<(), String> {
        use serde_json::Value;
        Self::validate_optional_field(
            body,
            "input",
            |v| {
                v.is_string()
                    || v.as_array()
                        .is_some_and(|items| items.iter().all(Value::is_object))
            },
            "text or an array of input items",
        )?;
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

// Keep stable context before input without cloning the request.
struct RoutingView<'a> {
    body: &'a serde_json::Value,
    keys: &'static [&'static str],
}

impl Serialize for RoutingView<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_map(
            self.keys
                .iter()
                .filter_map(|key| self.body.get(*key).map(|value| (*key, value))),
        )
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
                    &["system", "tools", "tool_choice", "messages"]
                } else {
                    &[
                        "instructions",
                        "tools",
                        "tool_choice",
                        "input",
                        "previous_response_id",
                        "conversation",
                    ]
                };
                let text = if include_text {
                    let view = RoutingView { body, keys };
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

    #[test]
    fn messages_and_responses_leave_backend_fields_to_the_backend() {
        for (route, body) in [
            // No max_tokens, a system role, top_k -1 and backend sampling ranges.
            (
                "/v1/messages",
                r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"tool","content":42}],"temperature":1.5,"top_k":-1,"top_p":2}"#,
            ),
            ("/v1/messages", r#"{"model":"m"}"#),
            (
                "/v1/responses",
                r#"{"model":"m","input":[{"role":"bad","content":42}],"temperature":3,"max_output_tokens":0,"background":"yes"}"#,
            ),
        ] {
            assert!(
                ParsedInference::parse(route, body.as_bytes()).is_ok(),
                "{body}"
            );
        }
        for (route, body) in [
            ("/v1/messages", r#"[]"#),
            ("/v1/messages", r#"{"model":" "}"#),
            ("/v1/messages", r#"{"model":"m","stream":"yes"}"#),
            ("/v1/responses", r#"{"model":"m","input":[42]}"#),
            ("/v1/responses", r#"{"model":"m","previous_response_id":7}"#),
            (
                "/v1/responses",
                r#"{"model":"m","previous_response_id":"r","conversation":"c"}"#,
            ),
        ] {
            assert!(
                ParsedInference::parse(route, body.as_bytes()).is_err(),
                "{body}"
            );
        }
    }
}
