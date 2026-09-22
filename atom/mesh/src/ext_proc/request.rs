use std::{collections::HashSet, sync::Arc};

use tokio::sync::Semaphore;

use http::{HeaderMap, HeaderName, HeaderValue};
use prost_types::value::Kind;
use validator::Validate;

use crate::{
    app_context::AppContext,
    protocols::{
        chat::ChatCompletionRequest, common::GenerationRequest, completion::CompletionRequest,
        generate::GenerateRequest, validated::Normalizable,
    },
    routers::prepare::chat_template::process_chat_messages,
};

use super::{core, error::ProcessingError, pb};

pub(super) struct RequestEnvelope {
    pub headers: HeaderMap,
    pub path: String,
    pub id: String,
    pub raw: Vec<u8>,
    pub subset: Option<HashSet<String>>,
    buffered_bytes: usize,
}

pub(super) struct RoutingInput {
    pub model: String,
    pub text: String,
    pub tokens: Option<Vec<u32>>,
    pub stream: bool,
}

impl RequestEnvelope {
    pub fn new(input: pb::HttpHeaders) -> Result<Self, ProcessingError> {
        let mut headers = HeaderMap::new();
        let mut path = None;
        let mut method = None;
        for header in input.headers.unwrap_or_default().headers {
            let value = Self::header_bytes(&header);
            match header.key.as_str() {
                ":path" => {
                    if path
                        .replace(
                            String::from_utf8(value.to_vec())
                                .map_err(|_| ProcessingError::invalid("invalid path"))?,
                        )
                        .is_some()
                    {
                        return Err(ProcessingError::invalid("duplicate :path"));
                    }
                }
                ":method" => {
                    if method.replace(value.to_vec()).is_some() {
                        return Err(ProcessingError::invalid("duplicate :method"));
                    }
                }
                key if key.starts_with(':') => {}
                _ => {
                    let name = HeaderName::from_bytes(header.key.as_bytes())
                        .map_err(|_| ProcessingError::invalid("invalid header name"))?;
                    let value = HeaderValue::from_bytes(value)
                        .map_err(|_| ProcessingError::invalid("invalid header value"))?;
                    headers.append(name, value);
                }
            }
        }
        if method.as_deref() != Some(b"POST") {
            return Err(ProcessingError::new(
                405,
                "method_not_allowed",
                "ext-proc inference routes require POST",
            ));
        }
        let path = path.ok_or_else(|| ProcessingError::invalid("missing :path"))?;
        let route = path.split('?').next().unwrap_or("");
        if !matches!(
            route,
            "/v1/chat/completions" | "/v1/completions" | "/generate"
        ) {
            return Err(ProcessingError::new(
                404,
                "unsupported_path",
                "unsupported inference API",
            ));
        }
        if headers
            .get("content-encoding")
            .is_some_and(|v| v != "identity")
        {
            return Err(ProcessingError::new(
                415,
                "unsupported_encoding",
                "decompress requests before ext-proc",
            ));
        }
        if !headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| {
                v.split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .eq_ignore_ascii_case("application/json")
            })
        {
            return Err(ProcessingError::new(
                415,
                "unsupported_content_type",
                "application/json is required",
            ));
        }
        let id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
        headers.remove(super::mutation::Mutation::DESTINATION);
        Ok(Self {
            headers,
            path: route.to_owned(),
            id,
            raw: Vec::new(),
            subset: None,
            buffered_bytes: 0,
        })
    }

    pub fn header_bytes(header: &core::HeaderValue) -> &[u8] {
        if header.raw_value.is_empty() {
            header.value.as_bytes()
        } else {
            &header.raw_value
        }
    }

    pub fn metadata(&mut self, metadata: Option<core::Metadata>) -> Result<(), ProcessingError> {
        let Some(metadata) = metadata else {
            return Ok(());
        };
        let Some(namespace) = metadata.filter_metadata.get("envoy.lb.subset_hint") else {
            return Ok(());
        };
        let Some(value) = namespace
            .fields
            .get("x-gateway-destination-endpoint-subset")
        else {
            return Ok(());
        };
        let Some(Kind::ListValue(list)) = &value.kind else {
            return Err(ProcessingError::invalid("endpoint subset must be a list"));
        };
        self.subset = Some(
            list.values
                .iter()
                .map(|v| match &v.kind {
                    Some(Kind::StringValue(s)) => Ok(s.clone()),
                    _ => Err(ProcessingError::invalid(
                        "endpoint subset entries must be addresses",
                    )),
                })
                .collect::<Result<_, _>>()?,
        );
        Ok(())
    }

    pub fn append(&mut self, body: &[u8], limit: usize) -> Result<(), ProcessingError> {
        if body.len() > limit.saturating_sub(self.raw.len()) {
            return Err(ProcessingError::new(
                413,
                "body_too_large",
                "request body limit exceeded",
            ));
        }
        self.buffered_bytes += body.len();
        self.raw.extend_from_slice(body);
        metrics::gauge!("mesh_ext_proc_buffered_request_bytes").increment(body.len() as f64);
        Ok(())
    }

    pub fn needs_tokens(app: &AppContext, model: &str) -> bool {
        if app.router_config.mode.is_pd_mode() {
            app.policy_registry.get_prefill_policy().needs_tokens()
                || app.policy_registry.get_decode_policy().needs_tokens()
        } else {
            app.policy_registry
                .get_policy_or_default(model)
                .needs_tokens()
        }
    }

    pub fn parse(&self, app: &AppContext) -> Result<RoutingInput, ProcessingError> {
        let (model, text, stream, chat) = match self.path.as_str() {
            "/v1/chat/completions" => {
                let mut request: ChatCompletionRequest = serde_json::from_slice(&self.raw)?;
                request.normalize();
                request
                    .validate()
                    .map_err(|e| ProcessingError::invalid(e.to_string()))?;
                (
                    request.model.clone(),
                    request.extract_text_for_routing(),
                    request.is_stream(),
                    Some(request),
                )
            }
            "/v1/completions" => {
                let request: CompletionRequest = serde_json::from_slice(&self.raw)?;
                (
                    request.model.clone(),
                    request.extract_text_for_routing(),
                    request.is_stream(),
                    None,
                )
            }
            _ => {
                let request: GenerateRequest = serde_json::from_slice(&self.raw)?;
                (
                    request.model.clone().unwrap_or_default(),
                    request.extract_text_for_routing(),
                    request.is_stream(),
                    None,
                )
            }
        };
        if model.trim().is_empty() {
            return Err(ProcessingError::invalid("model is required"));
        }
        let tokens = if Self::needs_tokens(app, &model) {
            let raw: serde_json::Value = serde_json::from_slice(&self.raw)?;
            if let Some(ids) = raw.get("input_ids") {
                Some(
                    serde_json::from_value::<Vec<u32>>(ids.clone()).map_err(|_| {
                        ProcessingError::invalid(
                            "token routing requires a single nonnegative input_ids array",
                        )
                    })?,
                )
            } else {
                let tokenizer = app.tokenizer_registry.get(&model).ok_or_else(|| {
                    ProcessingError::new(
                        503,
                        "tokenizer_unavailable",
                        "token routing requires a registered model tokenizer",
                    )
                })?;
                let prompt = if let Some(chat) = chat {
                    process_chat_messages(&chat, &*tokenizer)
                        .map_err(ProcessingError::invalid)?
                        .text
                } else {
                    text.clone()
                };
                Some(
                    tokenizer
                        .encode(&prompt, false)
                        .map_err(|e| ProcessingError::invalid(e.to_string()))?
                        .token_ids()
                        .to_vec(),
                )
            }
        } else {
            None
        };
        Ok(RoutingInput {
            model,
            text,
            tokens,
            stream,
        })
    }
}

/// Bounds CPU work independently of gRPC streams. A canceled blocking job retains
/// its permit until parsing/tokenization finishes, so cancellations cannot flood it.
pub(super) struct RequestParser {
    app: Arc<AppContext>,
    slots: Arc<Semaphore>,
}

impl RequestParser {
    pub fn new(app: Arc<AppContext>) -> Self {
        let count = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(app.router_config.ext_proc.max_streams);
        Self {
            app,
            slots: Arc::new(Semaphore::new(count)),
        }
    }

    pub async fn parse(
        &self,
        request: RequestEnvelope,
    ) -> Result<(RequestEnvelope, RoutingInput), ProcessingError> {
        let permit =
            self.slots.clone().acquire_owned().await.map_err(|_| {
                ProcessingError::new(503, "parser_closed", "request parser is closed")
            })?;
        let app = self.app.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let input = request.parse(&app)?;
            Ok((request, input))
        })
        .await
        .map_err(|_| ProcessingError::new(500, "parser_failed", "request parser failed"))?
    }
}

impl Drop for RequestEnvelope {
    fn drop(&mut self) {
        metrics::gauge!("mesh_ext_proc_buffered_request_bytes")
            .decrement(self.buffered_bytes as f64);
    }
}
