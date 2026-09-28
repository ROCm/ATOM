//! Transport-independent metadata derived from the typed inference protocols.
//! Callers retain the original payload separately, including extension fields.
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
}
impl InferenceRequest for ChatCompletionRequest {
    fn metadata(&self) -> InferenceMetadata {
        InferenceMetadata {
            route: "/v1/chat/completions",
            model: Some(self.model.clone()),
            text: self.extract_text_for_routing(),
            batch_size: self.n.filter(|n| *n > 1).map(|n| n as usize),
            stream: self.is_stream(),
            return_logprob: self.logprobs,
        }
    }
}
impl InferenceRequest for CompletionRequest {
    fn metadata(&self) -> InferenceMetadata {
        let batch_size = match &self.prompt {
            StringOrArray::Array(values) if !values.is_empty() => Some(values.len()),
            _ => None,
        };
        InferenceMetadata {
            route: "/v1/completions",
            model: Some(self.model.clone()),
            text: self.extract_text_for_routing(),
            batch_size,
            stream: self.is_stream(),
            return_logprob: self.logprobs.is_some(),
        }
    }
}
impl InferenceRequest for GenerateRequest {
    fn metadata(&self) -> InferenceMetadata {
        let batch_size = match &self.input_ids {
            Some(InputIds::Batch(values)) if !values.is_empty() => Some(values.len()),
            _ => None,
        };
        InferenceMetadata {
            route: "/generate",
            model: self.model.clone(),
            text: self.extract_text_for_routing(),
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
}
impl ParsedInference {
    pub fn parse(path: &str, bytes: &[u8]) -> Result<Self, String> {
        match path {
            "/v1/chat/completions" => {
                let mut request: ChatCompletionRequest =
                    serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
                request.normalize();
                request.validate().map_err(|e| e.to_string())?;
                Ok(Self::Chat(request))
            }
            "/v1/completions" => serde_json::from_slice(bytes)
                .map(Self::Completion)
                .map_err(|e| e.to_string()),
            "/generate" => serde_json::from_slice(bytes)
                .map(Self::Generate)
                .map_err(|e| e.to_string()),
            _ => Err("unsupported inference API".into()),
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
        match self {
            Self::Chat(r) => r.metadata(),
            Self::Completion(r) => r.metadata(),
            Self::Generate(r) => r.metadata(),
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
