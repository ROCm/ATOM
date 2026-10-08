//! Shared HTTP proxy contract. The routing view never replaces the original payload.
use std::{sync::Arc, time::Instant};

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
    core::{
        prepare_pool::{InputLease, JobContext, PrepareError, PrepareHandle},
        Worker, WorkerRegistry, WorkerType,
    },
    policies::PolicyRegistry,
    routers::{
        comm::error::{IngressError, MeshLocalError},
        prepare::inference::{InferenceMetadata, InferenceRequest, ParsedInference},
    },
    server::AppState,
    tokenizer::TokenizerRegistry,
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
        let envelope =
            match InferenceEnvelope::parse(parts.uri, parts.headers, body, &state.context).await {
                Ok(envelope) => envelope,
                Err(error) => return error.response(&path),
            };
        MeshLocalError::format_response(
            &path,
            state.router.route_inference(envelope, &state.context).await,
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

    pub(crate) fn supports(&self, worker: &dyn Worker) -> bool {
        worker
            .metadata()
            .labels
            .get("mesh.apis")
            .is_none_or(|apis| apis.split(',').any(|api| api.trim() == self.path))
    }
}

/// The regular HTTP forwarding phase no longer owns preparation state.
pub(crate) struct HttpInferenceRequest {
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub metadata: InferenceMetadata,
}

pub struct InferenceEnvelope {
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub parsed: ParsedInference,
    pub metadata: InferenceMetadata,
    // PD needs a mutable JSON body while preserving backend-specific fields.
    pd_body: Option<serde_json::Value>,
    /// Shared by JSON parsing and every subsequent prepare submission.
    pub(crate) prepare_deadline: Instant,
    // Remains with the envelope through queued/running jobs and returned results.
    _input_lease: InputLease,
}
impl InferenceEnvelope {
    pub async fn parse(
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
        app: &AppContext,
    ) -> Result<Self, IngressError> {
        Self::validate_headers(&headers)?;
        let deadline = app.prepare_pool.prepare_deadline();
        let lease = app.prepare_pool.retain_input(body.len())?;
        let inline_limit = app.router_config.prepare_pool.parse_inline_max_bytes;
        let inline = inline_limit != 0 && body.len() <= inline_limit;
        let pd = app.router_config.mode.is_pd_mode();
        let parse = move || -> Result<Self, IngressError> {
            let started = Instant::now();
            let parsed =
                ParsedInference::parse(uri.path(), &body).map_err(IngressError::invalid)?;
            // Build the PD forwarding body during the first CPU job. Serializing
            // the typed routing view would discard backend-specific fields.
            let pd_body = if pd {
                Some(
                    serde_json::from_slice(&body).map_err(|error: serde_json::Error| {
                        IngressError::invalid(error.to_string())
                    })?,
                )
            } else {
                None
            };
            metrics::histogram!("mesh_prepare_stage_seconds", "stage" => "json_parse")
                .record(started.elapsed().as_secs_f64());
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
                pd_body,
                prepare_deadline: deadline,
                _input_lease: lease,
            })
        };
        if inline {
            let request = parse()?;
            if Instant::now() >= deadline {
                return Err(PrepareError::Timeout.into());
            }
            return Ok(request);
        }
        app.prepare_pool
            .submit(deadline, move |ctx| -> Result<Self, IngressError> {
                ctx.check()?;
                let request = parse()?;
                ctx.check()?;
                Ok(request)
            })
            .await?
            .wait()
            .await?
    }

    /// Move the envelope and its input lease through the job and result together.
    pub(crate) async fn prepare<T: Send + 'static>(
        self,
        pool: &PrepareHandle,
        work: impl FnOnce(&Self, &JobContext) -> Result<T, IngressError> + Send + 'static,
    ) -> Result<(Self, T), IngressError> {
        pool.submit(
            self.prepare_deadline,
            move |ctx| -> Result<_, IngressError> {
                ctx.check()?;
                let value = work(&self, ctx)?;
                ctx.check()?;
                Ok((self, value))
            },
        )
        .await?
        .wait()
        .await?
    }

    pub(crate) async fn prepare_routing(
        self,
        pool: &PrepareHandle,
        routing: &IngressRouting,
    ) -> Result<(Self, (InferenceMetadata, Option<Vec<u32>>)), IngressError> {
        self.check_deadline()?;
        routing.validate_metadata(&self.metadata)?;
        let (needs_text, needs_tokens) = routing.requirements(self.metadata.model.as_deref());
        if !needs_text && !needs_tokens {
            let metadata = self.metadata.execution_metadata();
            self.check_deadline()?;
            return Ok((self, (metadata, None)));
        }
        let routing = routing.clone();
        self.prepare(pool, move |request, ctx| {
            routing.prepare(&request.parsed, ctx)
        })
        .await
    }

    fn check_deadline(&self) -> Result<(), IngressError> {
        if Instant::now() >= self.prepare_deadline {
            return Err(PrepareError::Timeout.into());
        }
        Ok(())
    }

    /// Consume only after the final preparation result has been received.
    /// The parsed views and preparation lease are dropped before forwarding.
    pub(crate) fn into_http_request(self) -> Result<HttpInferenceRequest, IngressError> {
        self.check_deadline()?;
        Ok(HttpInferenceRequest {
            uri: self.uri,
            headers: self.headers,
            body: self.body,
            metadata: self.metadata,
        })
    }

    /// PD forwards its mutable JSON body, so its duplicate raw and typed bodies
    /// and preparation lease can be released before either backend executes.
    pub(crate) fn into_pd_parts(self) -> Result<(Uri, HeaderMap, serde_json::Value), IngressError> {
        self.check_deadline()?;
        let body = self.pd_body.ok_or_else(|| {
            IngressError::new(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "missing_pd_body",
                "PD request body was not prepared",
            )
        })?;
        Ok((self.uri, self.headers, body))
    }

    /// Native backends consume typed requests; release raw JSON and preparation
    /// accounting before awaiting their own request handling.
    pub(crate) fn into_native_parts(
        self,
    ) -> Result<(Uri, HeaderMap, ParsedInference, InferenceMetadata), IngressError> {
        self.check_deadline()?;
        Ok((self.uri, self.headers, self.parsed, self.metadata))
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

#[derive(Clone)]
pub(crate) struct IngressRouting {
    policy_registry: Arc<PolicyRegistry>,
    worker_registry: Arc<WorkerRegistry>,
    tokenizer_registry: Arc<TokenizerRegistry>,
    pd: bool,
    max_tokenize_bytes: usize,
}

impl IngressRouting {
    pub(crate) fn new(app: &AppContext) -> Self {
        Self {
            policy_registry: app.policy_registry.clone(),
            worker_registry: app.worker_registry.clone(),
            tokenizer_registry: app.tokenizer_registry.clone(),
            pd: app.router_config.mode.is_pd_mode(),
            max_tokenize_bytes: app.router_config.resolved_max_tokenize_bytes(),
        }
    }

    fn requirements(&self, model: Option<&str>) -> (bool, bool) {
        let registry = &self.policy_registry;
        if self.pd {
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

    #[cfg(feature = "ext-proc")]
    pub(crate) fn needs_tokens(&self, model: Option<&str>) -> bool {
        self.requirements(model).1
    }

    pub(crate) fn prepare(
        &self,
        parsed: &ParsedInference,
        ctx: &JobContext,
    ) -> Result<(InferenceMetadata, Option<Vec<u32>>), IngressError> {
        ctx.check()?;
        let mut metadata = parsed.metadata_with_text(false);
        self.validate_metadata(&metadata)?;
        let (needs_text, needs_tokens) = self.requirements(metadata.model.as_deref());
        if needs_text || needs_tokens {
            metadata = parsed.metadata();
        }
        ctx.check()?;
        let tokens = if needs_tokens {
            Some(self.tokens(parsed, &metadata, ctx)?)
        } else {
            None
        };
        ctx.check()?;
        Ok((metadata, tokens))
    }

    fn validate_metadata(&self, metadata: &InferenceMetadata) -> Result<(), IngressError> {
        if metadata
            .model
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(IngressError::invalid("model is required"));
        }
        let endpoint = EndpointSpec::find(metadata.route)
            .ok_or_else(|| IngressError::invalid("unsupported inference API"))?;
        endpoint.validate_topology(self.pd)
    }

    pub(crate) fn candidates(
        &self,
        metadata: &InferenceMetadata,
        state_reference: bool,
    ) -> Result<Vec<Arc<dyn Worker>>, IngressError> {
        let endpoint = EndpointSpec::find(metadata.route)
            .ok_or_else(|| IngressError::invalid("unsupported inference API"))?;
        let pd = self.pd;
        endpoint.validate_topology(pd)?;
        let mut workers = match metadata.model.as_deref() {
            Some(model) => self.worker_registry.get_by_model(model).to_vec(),
            None => self.worker_registry.get_all(),
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
        ctx: &JobContext,
    ) -> Result<Vec<u32>, IngressError> {
        ctx.check()?;
        if let Some(ids) = parsed.input_tokens().map_err(IngressError::invalid)? {
            return Ok(ids);
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
            .and_then(|m| self.tokenizer_registry.get(m))
            .ok_or_else(|| {
                IngressError::new(
                    http::StatusCode::SERVICE_UNAVAILABLE,
                    "tokenizer_unavailable",
                    "token routing requires input_ids or a model with a registered tokenizer",
                )
            })?;
        let limit = self.max_tokenize_bytes;
        if metadata.text.len() > limit {
            return Err(IngressError::new(
                http::StatusCode::PAYLOAD_TOO_LARGE,
                "tokenizer_input_too_large",
                "prompt exceeds the synchronous tokenizer byte limit",
            ));
        }
        ctx.check()?;
        let template_started = Instant::now();
        let prompt = if let Some(chat) = parsed.chat() {
            super::prepare::chat_template::process_chat_messages(chat, &*tokenizer)
                .map_err(IngressError::invalid)?
                .text
        } else {
            metadata.text.clone()
        };
        metrics::histogram!("mesh_prepare_stage_seconds", "stage" => "template")
            .record(template_started.elapsed().as_secs_f64());
        ctx.check()?;
        if prompt.len() > limit {
            return Err(IngressError::new(
                http::StatusCode::PAYLOAD_TOO_LARGE,
                "tokenizer_input_too_large",
                "prompt exceeds the synchronous tokenizer byte limit",
            ));
        }
        let encode_started = Instant::now();
        let tokens = tokenizer
            .encode(&prompt, false)
            .map_err(|e| IngressError::invalid(e.to_string()))?
            .token_ids()
            .to_vec();
        metrics::histogram!("mesh_prepare_stage_seconds", "stage" => "encode")
            .record(encode_started.elapsed().as_secs_f64());
        ctx.check()?;
        Ok(tokens)
    }
}

#[cfg(test)]
#[path = "../../tests/routing/ingress_tests.rs"]
mod tests;
