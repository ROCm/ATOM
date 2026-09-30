//! Shared HTTP proxy contract. The routing view never replaces the original payload.
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use axum::{
    extract::{Path, RawQuery, Request, State},
    response::Response,
    routing::{get, post, MethodRouter},
    Router,
};
use bytes::Bytes;
use http::{HeaderMap, Uri};

use crate::{
    app_context::AppContext,
    core::{Worker, WorkerType},
    routers::{
        comm::error::{IngressError, MeshLocalError},
        prepare::inference::{InferenceMetadata, InferenceRequest, ParsedInference},
    },
    server::AppState,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Chat,
    Messages,
    Responses,
    Completions,
    Generate,
}

#[derive(Clone, Copy)]
pub struct EndpointSpec {
    pub path: &'static str,
    pub api: Api,
    subroutes: &'static [SubrouteSpec],
}

struct SubrouteSpec {
    path: &'static str,
    methods: fn() -> MethodRouter<Arc<AppState>>,
}

impl EndpointSpec {
    pub const ALL: &[EndpointSpec] = &[
        EndpointSpec {
            path: "/v1/chat/completions",
            api: Api::Chat,
            subroutes: &[],
        },
        EndpointSpec {
            path: "/v1/messages",
            api: Api::Messages,
            subroutes: &[],
        },
        EndpointSpec {
            path: "/v1/responses",
            api: Api::Responses,
            subroutes: &[
                SubrouteSpec {
                    path: "/{response_id}",
                    methods: || get(Self::get_response).delete(Self::delete_response),
                },
                SubrouteSpec {
                    path: "/{response_id}/cancel",
                    methods: || post(Self::cancel_response),
                },
                SubrouteSpec {
                    path: "/{response_id}/input_items",
                    methods: || get(Self::list_response_input_items),
                },
            ],
        },
        EndpointSpec {
            path: "/v1/completions",
            api: Api::Completions,
            subroutes: &[],
        },
        EndpointSpec {
            path: "/generate",
            api: Api::Generate,
            subroutes: &[],
        },
    ];

    pub(crate) fn register(&self, router: Router<Arc<AppState>>) -> Router<Arc<AppState>> {
        let router = router.route(
            self.path,
            post(Self::inference).fallback(Self::inference_method_not_allowed),
        );
        self.subroutes.iter().fold(router, |router, subroute| {
            router.route(
                &format!("{}{}", self.path, subroute.path),
                (subroute.methods)(),
            )
        })
    }

    pub fn find(path: &str) -> Option<&'static Self> {
        let path = path.split('?').next().unwrap_or(path);
        Self::ALL.iter().find(|spec| spec.path == path)
    }

    async fn inference_method_not_allowed(request: Request) -> Response {
        IngressError::new(
            http::StatusCode::METHOD_NOT_ALLOWED,
            "method_not_allowed",
            "inference routes require POST",
        )
        .response(request.uri().path())
    }

    async fn inference(State(state): State<Arc<AppState>>, request: Request) -> Response {
        let (parts, body) = request.into_parts();
        let path = parts.uri.path().to_owned();
        let body =
            match axum::body::to_bytes(body, state.context.router_config.max_payload_size).await {
                Ok(body) => body,
                Err(error) => {
                    let mut source: &(dyn std::error::Error + 'static) = &error;
                    loop {
                        if source.is::<http_body_util::LengthLimitError>() {
                            return IngressError::new(
                                http::StatusCode::PAYLOAD_TOO_LARGE,
                                "body_too_large",
                                "request body limit exceeded",
                            )
                            .response(&path);
                        }
                        match source.source() {
                            Some(next) => source = next,
                            None => break,
                        }
                    }
                    return IngressError::invalid("failed to read request body").response(&path);
                }
            };
        let envelope = match InferenceEnvelope::parse(parts.uri, parts.headers, body) {
            Ok(envelope) => envelope,
            Err(error) => return error.response(&path),
        };
        MeshLocalError::format_response(
            &path,
            state
                .router
                .route_inference(&envelope, &state.context)
                .await,
        )
    }

    async fn get_response(
        State(state): State<Arc<AppState>>,
        Path(response_id): Path<String>,
        headers: HeaderMap,
        RawQuery(query): RawQuery,
    ) -> Response {
        state
            .router
            .get_response(Some(&headers), &response_id, query.as_deref())
            .await
    }

    async fn cancel_response(
        State(state): State<Arc<AppState>>,
        Path(response_id): Path<String>,
        headers: HeaderMap,
    ) -> Response {
        state
            .router
            .cancel_response(Some(&headers), &response_id)
            .await
    }

    async fn delete_response(
        State(state): State<Arc<AppState>>,
        Path(response_id): Path<String>,
        headers: HeaderMap,
    ) -> Response {
        state
            .router
            .delete_response(Some(&headers), &response_id)
            .await
    }

    async fn list_response_input_items(
        State(state): State<Arc<AppState>>,
        Path(response_id): Path<String>,
        headers: HeaderMap,
        RawQuery(query): RawQuery,
    ) -> Response {
        state
            .router
            .list_response_input_items(Some(&headers), &response_id, query.as_deref())
            .await
    }

    fn validate_topology(&self, pd: bool) -> Result<(), IngressError> {
        if pd && matches!(self.api, Api::Messages | Api::Responses) {
            return Err(IngressError::new(
                http::StatusCode::NOT_IMPLEMENTED,
                "unsupported_api_topology",
                "Messages and Responses PD adapters are not supported; use a regular HTTP backend",
            ));
        }
        Ok(())
    }

    fn supports(&self, worker: &dyn Worker) -> bool {
        worker
            .metadata()
            .labels
            .get("mesh.apis")
            .is_none_or(|apis| apis.split(',').any(|api| api.trim() == self.path))
    }
}

pub struct InferenceEnvelope {
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub parsed: ParsedInference,
    pub metadata: InferenceMetadata,
}
impl InferenceEnvelope {
    pub fn parse(uri: Uri, headers: HeaderMap, body: Bytes) -> Result<Self, IngressError> {
        Self::validate_headers(&headers)?;
        let parsed = ParsedInference::parse(uri.path(), &body).map_err(IngressError::invalid)?;
        let metadata = parsed.metadata_with_text(false);
        if metadata
            .model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(IngressError::invalid("model is required"));
        }
        Ok(Self {
            uri,
            headers,
            body,
            parsed,
            metadata,
        })
    }

    pub(crate) fn validate_headers(headers: &HeaderMap) -> Result<(), IngressError> {
        if headers.get_all("content-encoding").iter().any(|v| {
            !v.to_str()
                .is_ok_and(|v| v.trim().eq_ignore_ascii_case("identity"))
        }) {
            return Err(IngressError::new(
                http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_encoding",
                "decompress requests before inference routing",
            ));
        }
        if headers.get_all("content-type").iter().count() != 1
            || !headers
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
            return Err(IngressError::new(
                http::StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_content_type",
                "application/json is required",
            ));
        }
        Ok(())
    }
}

pub(crate) struct IngressRouting<'a> {
    app: &'a AppContext,
}

impl<'a> IngressRouting<'a> {
    pub(crate) fn new(app: &'a AppContext) -> Self {
        Self { app }
    }

    fn requirements(&self, model: Option<&str>) -> (bool, bool) {
        let registry = &self.app.policy_registry;
        if self.app.router_config.mode.is_pd_mode() {
            let prefill = registry.get_prefill_policy();
            let decode = registry.get_decode_policy();
            (
                prefill.needs_request_text() || decode.needs_request_text(),
                prefill.needs_tokens() || decode.needs_tokens(),
            )
        } else {
            let policy = match model {
                Some(model) => registry.get_policy_or_default(model),
                None => registry.get_default_policy(),
            };
            (policy.needs_request_text(), policy.needs_tokens())
        }
    }

    pub(crate) fn needs_tokens(&self, model: Option<&str>) -> bool {
        self.requirements(model).1
    }

    pub(crate) fn prepare(
        &self,
        parsed: &ParsedInference,
        canceled: &AtomicBool,
    ) -> Result<(InferenceMetadata, Option<Vec<u32>>), IngressError> {
        let mut metadata = parsed.metadata_with_text(false);
        if metadata
            .model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(IngressError::invalid("model is required"));
        }
        let endpoint = EndpointSpec::find(metadata.route)
            .ok_or_else(|| IngressError::invalid("unsupported inference API"))?;
        endpoint.validate_topology(self.app.router_config.mode.is_pd_mode())?;
        let (text, tokens) = self.requirements(metadata.model.as_deref());
        if text || tokens {
            metadata = parsed.metadata();
        }
        let tokens = self.tokens(parsed, &metadata, canceled)?;
        Ok((metadata, tokens))
    }

    pub(crate) fn candidates(
        &self,
        metadata: &InferenceMetadata,
        state_reference: bool,
    ) -> Result<Vec<Arc<dyn Worker>>, IngressError> {
        let endpoint = EndpointSpec::find(metadata.route)
            .ok_or_else(|| IngressError::invalid("unsupported inference API"))?;
        let pd = self.app.router_config.mode.is_pd_mode();
        endpoint.validate_topology(pd)?;
        let mut workers = match metadata.model.as_deref() {
            Some(model) => self.app.worker_registry.get_by_model(model).to_vec(),
            None => self.app.worker_registry.get_all(),
        };
        workers.retain(|w| matches!(w.connection_mode(), crate::core::ConnectionMode::Http));
        let configured = !workers.is_empty();
        workers.retain(|w| {
            endpoint.supports(w.as_ref()) && matches!(w.worker_type(), WorkerType::Regular) != pd
        });
        if configured && workers.is_empty() {
            return Err(IngressError::new(
                http::StatusCode::NOT_IMPLEMENTED,
                "unsupported_api",
                "no configured backend supports this API and topology",
            ));
        }
        if state_reference {
            Self::validate_state_domain(&workers)?;
        }
        Ok(workers)
    }

    // Include unhealthy candidates so failover cannot silently change state ownership.
    fn validate_state_domain(workers: &[Arc<dyn Worker>]) -> Result<(), IngressError> {
        let distinct_origins = workers
            .first()
            .is_some_and(|first| workers.iter().any(|w| w.base_url() != first.base_url()));
        if distinct_origins
            && !workers.iter().all(|w| {
                w.metadata()
                    .labels
                    .get("mesh.responses_state")
                    .is_some_and(|v| v == "shared")
            })
        {
            return Err(IngressError::new(http::StatusCode::NOT_IMPLEMENTED, "stateful_routing_unsupported",
                "Responses state requires a single backend state domain or mesh.responses_state=shared on all candidates"));
        }
        Ok(())
    }

    fn tokens(
        &self,
        parsed: &ParsedInference,
        metadata: &InferenceMetadata,
        canceled: &AtomicBool,
    ) -> Result<Option<Vec<u32>>, IngressError> {
        let check = || {
            if canceled.load(Ordering::Acquire) {
                Err(IngressError::new(
                    http::StatusCode::from_u16(499).expect("valid cancellation status"),
                    "parser_canceled",
                    "request parsing canceled",
                ))
            } else {
                Ok(())
            }
        };
        check()?;
        let app = self.app;
        let needs_tokens = self.needs_tokens(metadata.model.as_deref());
        if !needs_tokens {
            return Ok(None);
        }
        if let Some(ids) = parsed.input_tokens().map_err(IngressError::invalid)? {
            return Ok(Some(ids));
        }
        if matches!(
            parsed,
            ParsedInference::Messages(_) | ParsedInference::Responses(_)
        ) {
            return Err(IngressError::new(http::StatusCode::NOT_IMPLEMENTED, "token_routing_unsupported", "this API has no registered token-routing template; use a policy that does not require exact tokens"));
        }
        let tokenizer = metadata
            .model
            .as_deref()
            .and_then(|m| app.tokenizer_registry.get(m))
            .ok_or_else(|| {
                IngressError::new(
                    http::StatusCode::SERVICE_UNAVAILABLE,
                    "tokenizer_unavailable",
                    "token routing requires input_ids or a model with a registered tokenizer",
                )
            })?;
        #[cfg(feature = "ext-proc")]
        let limit = app.router_config.ext_proc.max_tokenize_bytes;
        #[cfg(not(feature = "ext-proc"))]
        let limit = 1024 * 1024;
        if metadata.text.len() > limit {
            return Err(IngressError::new(
                http::StatusCode::PAYLOAD_TOO_LARGE,
                "tokenizer_input_too_large",
                "prompt exceeds the synchronous tokenizer byte limit",
            ));
        }
        let prompt = if let Some(chat) = parsed.chat() {
            super::prepare::chat_template::process_chat_messages(chat, &*tokenizer)
                .map_err(IngressError::invalid)?
                .text
        } else {
            metadata.text.clone()
        };
        check()?;
        if prompt.len() > limit {
            return Err(IngressError::new(
                http::StatusCode::PAYLOAD_TOO_LARGE,
                "tokenizer_input_too_large",
                "prompt exceeds the synchronous tokenizer byte limit",
            ));
        }
        let tokens = tokenizer
            .encode(&prompt, false)
            .map_err(|e| IngressError::invalid(e.to_string()))?
            .token_ids()
            .to_vec();
        check()?;
        Ok(Some(tokens))
    }
}

#[cfg(test)]
#[path = "../../tests/routing/ingress_tests.rs"]
mod tests;
