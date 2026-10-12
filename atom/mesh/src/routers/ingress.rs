//! Shared HTTP proxy contract. The routing view never replaces the original payload.
use std::{sync::Arc, time::Instant};

use axum::{
    extract::{FromRequestParts, Path, RawQuery, Request, State},
    response::{IntoResponse, Response},
    routing::{get, post, MethodRouter},
    Router,
};
use bytes::Bytes;
use http::{HeaderMap, Uri};

use crate::{
    app_context::AppContext,
    core::{
        placement::{
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
        },
        prepare_pool::{JobContext, PrepareHandle},
        ConnectionMode, Worker, WorkerRegistry, WorkerType,
    },
    policies::PolicyRegistry,
    routers::{
        comm::error::{IngressError, MeshLocalError},
        prepare::inference::{InferenceMetadata, InferenceRequest, ParsedInference},
    },
    server::AppState,
    tokenizer::TokenizerRegistry,
};

const MAX_RESPONSE_ID_BYTES: usize = 256;
/// Worker label listing the inference paths a backend serves.
const APIS_LABEL: &str = "mesh.apis";
/// Worker label declaring how a backend stores Responses state.
const RESPONSES_STATE_LABEL: &str = "mesh.responses_state";
/// `RESPONSES_STATE_LABEL` value for backends that share one state domain.
const SHARED_RESPONSES_STATE: &str = "shared";

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
                            return IngressError::body_too_large().response(&path);
                        }
                        match source.source() {
                            Some(next) => source = next,
                            None => break,
                        }
                    }
                    return IngressError::invalid("failed to read request body").response(&path);
                }
            };
        let envelope = match InferenceEnvelope::parse_for_routing(
            parts.uri,
            parts.headers,
            body,
            &state.context,
            IngressRouting::for_app(&state.context),
        )
        .await
        {
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
        ResponseId(response_id): ResponseId,
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
        ResponseId(response_id): ResponseId,
        headers: HeaderMap,
    ) -> Response {
        state
            .router
            .cancel_response(Some(&headers), &response_id)
            .await
    }

    async fn delete_response(
        State(state): State<Arc<AppState>>,
        ResponseId(response_id): ResponseId,
        headers: HeaderMap,
    ) -> Response {
        state
            .router
            .delete_response(Some(&headers), &response_id)
            .await
    }

    async fn list_response_input_items(
        State(state): State<Arc<AppState>>,
        ResponseId(response_id): ResponseId,
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
        self.listed_in_apis_label(worker).unwrap_or(true)
    }

    /// Whether a DP-aware worker receives `data_parallel_rank` for this API.
    /// Chat, Completions and Generate carry the rank on supported backends.
    /// Messages and Responses only do so when the worker's `mesh.apis` label
    /// lists them, which declares that the backend honors the field.
    pub(crate) fn forwards_dp_rank(&self, worker: &dyn Worker) -> bool {
        match self.api {
            Api::Chat | Api::Completions | Api::Generate => true,
            Api::Messages | Api::Responses => self.listed_in_apis_label(worker).unwrap_or(false),
        }
    }

    fn listed_in_apis_label(&self, worker: &dyn Worker) -> Option<bool> {
        worker
            .metadata()
            .labels
            .get(APIS_LABEL)
            .map(|apis| apis.split(',').any(|api| api.trim() == self.path))
    }
}

/// A Responses resource ID from the request path.
///
/// IDs arrive percent-decoded and become one backend path segment, so reject
/// anything that could change the forwarded path or query.
struct ResponseId(String);

impl<S: Send + Sync> FromRequestParts<S> for ResponseId {
    type Rejection = Response;

    async fn from_request_parts(
        parts: &mut http::request::Parts,
        state: &S,
    ) -> Result<Self, Response> {
        let Path(response_id) = Path::<String>::from_request_parts(parts, state)
            .await
            .map_err(IntoResponse::into_response)?;
        let valid = !response_id.is_empty()
            && response_id.len() <= MAX_RESPONSE_ID_BYTES
            && response_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-');
        if !valid {
            return Err(IngressError::new(
                http::StatusCode::BAD_REQUEST,
                "invalid_response_id",
                "response_id may contain only ASCII letters, digits, '_' and '-'",
            )
            .response("/v1/responses"));
        }
        Ok(Self(response_id))
    }
}

/// Dispatch-time routing preparation with a planner restricted to the
/// request's eligible workers.
pub(crate) struct PreparedPlacement {
    pub request: InferenceEnvelope,
    pub metadata: InferenceMetadata,
    pub tokens: Option<Vec<u32>>,
    pub planner: DefaultPlanner,
}

/// A request body serialized without `data_parallel_rank` and its closing
/// brace, so each DP-aware attempt appends its rank without reparsing.
#[derive(Clone, Debug)]
pub(crate) struct DpRankBody {
    head: Bytes,
    empty: bool,
}

impl DpRankBody {
    /// Parse once and drop any client rank; Mesh sets the selected worker's.
    pub(crate) fn prepare(raw: &[u8]) -> Result<Self, IngressError> {
        let mut body: serde_json::Value =
            serde_json::from_slice(raw).map_err(|e| IngressError::invalid(e.to_string()))?;
        let Some(object) = body.as_object_mut() else {
            return Err(IngressError::new(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "worker_request_preparation_failed",
                crate::core::WorkerError::InvalidConfiguration {
                    message: "Request must be a JSON object for DP-aware routing".into(),
                }
                .to_string(),
            ));
        };
        object.shift_remove("data_parallel_rank");
        let empty = object.is_empty();
        let mut head =
            serde_json::to_vec(&body).map_err(|e| IngressError::invalid(e.to_string()))?;
        head.pop();
        Ok(Self {
            head: head.into(),
            empty,
        })
    }

    pub(crate) fn with_rank(&self, rank: usize) -> Bytes {
        let rank = rank.to_string();
        let mut body = Vec::with_capacity(self.head.len() + rank.len() + 24);
        body.extend_from_slice(&self.head);
        if !self.empty {
            body.push(b',');
        }
        body.extend_from_slice(b"\"data_parallel_rank\":");
        body.extend_from_slice(rank.as_bytes());
        body.push(b'}');
        body.into()
    }
}

/// The body `worker` receives with its `data_parallel_rank`, or `None` when the
/// raw body is forwarded unchanged.
pub(crate) fn dp_ranked_body(
    worker: &dyn Worker,
    route: &str,
    prepared: Option<&DpRankBody>,
    raw: &[u8],
) -> Result<Option<Bytes>, IngressError> {
    let Some(rank) = worker.dp_rank().filter(|_| {
        worker.is_dp_aware()
            && EndpointSpec::find(route).is_some_and(|endpoint| endpoint.forwards_dp_rank(worker))
    }) else {
        return Ok(None);
    };
    match prepared {
        Some(body) => Ok(Some(body.with_rank(rank))),
        None => prepare_dp_rank_body_unpooled(raw).map(|body| Some(body.with_rank(rank))),
    }
}

/// Rare path: a DP-aware worker was registered after the request was prepared.
fn prepare_dp_rank_body_unpooled(raw: &[u8]) -> Result<DpRankBody, IngressError> {
    DpRankBody::prepare(raw)
}

/// The regular HTTP forwarding phase no longer owns preparation state.
pub(crate) struct HttpInferenceRequest {
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub metadata: InferenceMetadata,
    pub dp_rank_body: Option<DpRankBody>,
}

pub struct InferenceEnvelope {
    pub uri: Uri,
    pub headers: HeaderMap,
    pub body: Bytes,
    pub parsed: ParsedInference,
    pub metadata: InferenceMetadata,
    // PD needs a mutable JSON body while preserving backend-specific fields.
    pd_body: Option<serde_json::Value>,
    /// Prepared when a DP-aware worker may receive this request.
    dp_rank_body: Option<DpRankBody>,
    /// Shared by JSON parsing and every subsequent prepare submission.
    pub(crate) prepare_deadline: Instant,
    /// Routing text and tokens computed in the parse job, so the request does
    /// not queue for the preparation pool a second time.
    routed: Option<RoutingPreparation>,
}

/// Execution metadata and, when the policy needs them, routing tokens.
pub(crate) type RoutingPreparation = (InferenceMetadata, Option<Vec<u32>>);
impl InferenceEnvelope {
    #[cfg(test)]
    pub async fn parse(
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
        app: &AppContext,
    ) -> Result<Self, IngressError> {
        Self::parse_for_routing(uri, headers, body, app, None).await
    }

    /// Parse, and when `routing` is given and its policy needs request text or
    /// tokens, prepare them in the same pool job.
    pub(crate) async fn parse_for_routing(
        uri: Uri,
        headers: HeaderMap,
        body: Bytes,
        app: &AppContext,
        routing: Option<IngressRouting>,
    ) -> Result<Self, IngressError> {
        Self::validate_headers(&headers)?;
        let deadline = app.prepare_pool.prepare_deadline();
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
            Ok(Self {
                uri,
                headers,
                body,
                parsed,
                metadata,
                pd_body,
                dp_rank_body: None,
                prepare_deadline: deadline,
                routed: None,
            })
        };
        app.prepare_pool
            .submit(deadline, move |ctx| -> Result<Self, IngressError> {
                let mut request = parse()?;
                if let Some(routing) = routing {
                    request.routed = Some(routing.prepare(&request.parsed, ctx)?);
                    request.dp_rank_body = routing.dp_rank_body(&request.metadata, &request.body);
                }
                Ok(request)
            })
            .await?
            .wait()
            .await?
    }

    /// Move the envelope through the job and result together.
    pub(crate) async fn prepare<T: Send + 'static>(
        self,
        pool: &PrepareHandle,
        work: impl FnOnce(&Self, &JobContext) -> Result<T, IngressError> + Send + 'static,
    ) -> Result<(Self, T), IngressError> {
        pool.submit(
            self.prepare_deadline,
            move |ctx| -> Result<_, IngressError> {
                let value = work(&self, ctx)?;
                Ok((self, value))
            },
        )
        .await?
        .wait()
        .await?
    }

    pub(crate) async fn prepare_routing(
        mut self,
        pool: &PrepareHandle,
        routing: &IngressRouting,
    ) -> Result<(Self, RoutingPreparation), IngressError> {
        // The deadline bounds preparation work only. Requests whose preparation
        // already finished are forwarded instead of discarding that work.
        if let Some(routed) = self.routed.take() {
            return Ok((self, routed));
        }
        let (needs_text, needs_tokens) = routing.admit(&self.metadata)?;
        if !needs_text && !needs_tokens {
            let metadata = self.metadata.execution_metadata();
            return Ok((self, (metadata, None)));
        }
        let routing = routing.clone();
        self.prepare(pool, move |request, ctx| {
            routing.prepare_text_and_tokens(&request.parsed, needs_tokens, ctx)
        })
        .await
    }

    /// Prepare routing, then restrict a planner to the request's eligible workers.
    pub(crate) async fn prepare_placement(
        self,
        app: &AppContext,
        routing: &IngressRouting,
        policies: Arc<PolicyRegistryAdapter>,
    ) -> Result<PreparedPlacement, IngressError> {
        let (request, (metadata, tokens)) =
            self.prepare_routing(&app.prepare_pool, routing).await?;
        let candidates = routing.candidates(&metadata, request.parsed.requires_state_domain())?;
        Ok(PreparedPlacement {
            planner: routing.planner(candidates, policies),
            request,
            metadata,
            tokens,
        })
    }

    /// Consume only after the final preparation result has been received.
    /// The parsed views are dropped before forwarding.
    pub(crate) fn into_http_request(self) -> HttpInferenceRequest {
        HttpInferenceRequest {
            uri: self.uri,
            headers: self.headers,
            body: self.body,
            metadata: self.metadata,
            dp_rank_body: self.dp_rank_body,
        }
    }

    /// PD forwards its mutable JSON body, so its duplicate raw and typed bodies
    /// can be released before either backend executes.
    pub(crate) fn into_pd_parts(self) -> Result<(Uri, HeaderMap, serde_json::Value), IngressError> {
        let body = self.pd_body.ok_or_else(|| {
            IngressError::new(
                http::StatusCode::INTERNAL_SERVER_ERROR,
                "missing_pd_body",
                "PD request body was not prepared",
            )
        })?;
        Ok((self.uri, self.headers, body))
    }

    /// Native backends consume typed requests; release raw JSON before awaiting
    /// their own request handling.
    pub(crate) fn into_native_parts(self) -> (Uri, HeaderMap, ParsedInference, InferenceMetadata) {
        (self.uri, self.headers, self.parsed, self.metadata)
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
pub struct IngressRouting {
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

    /// Only the HTTP and PD routers place requests through `IngressRouting`;
    /// gRPC and ATOM standalone routers handle typed requests themselves.
    pub(crate) fn for_app(app: &AppContext) -> Option<Self> {
        let config = &app.router_config;
        (!config.atom_standalone && matches!(config.connection_mode, ConnectionMode::Http))
            .then(|| Self::new(app))
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

    /// Validate the request against this topology and prepare what its policy needs.
    pub(crate) fn prepare(
        &self,
        parsed: &ParsedInference,
        ctx: &JobContext,
    ) -> Result<RoutingPreparation, IngressError> {
        ctx.check()?;
        let metadata = parsed.metadata_with_text(false);
        let (needs_text, needs_tokens) = self.admit(&metadata)?;
        if !needs_text && !needs_tokens {
            return Ok((metadata, None));
        }
        self.prepare_text_and_tokens(parsed, needs_tokens, ctx)
    }

    /// Validate the API and topology, then return whether the policy needs
    /// request text and tokens.
    fn admit(&self, metadata: &InferenceMetadata) -> Result<(bool, bool), IngressError> {
        let endpoint = EndpointSpec::find(metadata.route)
            .ok_or_else(|| IngressError::invalid("unsupported inference API"))?;
        endpoint.validate_topology(self.pd)?;
        Ok(self.requirements(metadata.model.as_deref()))
    }

    /// Routing text, and tokens when `needs_tokens`, for an admitted request.
    fn prepare_text_and_tokens(
        &self,
        parsed: &ParsedInference,
        needs_tokens: bool,
        ctx: &JobContext,
    ) -> Result<RoutingPreparation, IngressError> {
        let metadata = parsed.metadata();
        let tokens = if needs_tokens {
            Some(self.tokens(parsed, &metadata, ctx)?)
        } else {
            None
        };
        Ok((metadata, tokens))
    }

    /// The rank-agnostic body when a DP-aware worker serving this request
    /// receives `data_parallel_rank`. A body that cannot carry a rank fails at
    /// forwarding, and only if a DP-aware worker is selected.
    pub(crate) fn dp_rank_body(
        &self,
        metadata: &InferenceMetadata,
        raw: &[u8],
    ) -> Option<DpRankBody> {
        if self.pd {
            return None;
        }
        let endpoint = EndpointSpec::find(metadata.route)?;
        let workers: Arc<[Arc<dyn Worker>]> = match metadata.model.as_deref() {
            Some(model) => self.worker_registry.get_by_model(model),
            None => self.worker_registry.get_all().into(),
        };
        workers
            .iter()
            .any(|w| w.is_dp_aware() && endpoint.forwards_dp_rank(w.as_ref()))
            .then(|| DpRankBody::prepare(raw).ok())
            .flatten()
    }

    /// A planner restricted to `candidates`.
    pub(crate) fn planner(
        &self,
        candidates: Vec<Arc<dyn Worker>>,
        policies: Arc<PolicyRegistryAdapter>,
    ) -> DefaultPlanner {
        DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::with_candidates(
                self.worker_registry.clone(),
                candidates,
            )),
            policies,
        )
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
        workers.retain(|w| matches!(w.connection_mode(), ConnectionMode::Http));
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
                    .get(RESPONSES_STATE_LABEL)
                    .is_some_and(|v| v == SHARED_RESPONSES_STATE)
            })
        {
            return Err(IngressError::new(http::StatusCode::NOT_IMPLEMENTED, "stateful_routing_unsupported",
                format!("Responses state requires a single backend state domain or {RESPONSES_STATE_LABEL}={SHARED_RESPONSES_STATE} on all candidates")));
        }
        Ok(())
    }

    fn tokenizer_input_too_large() -> IngressError {
        IngressError::new(
            http::StatusCode::PAYLOAD_TOO_LARGE,
            "tokenizer_input_too_large",
            "prompt exceeds the synchronous tokenizer byte limit",
        )
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
        let entry = metadata
            .model
            .as_deref()
            .and_then(|m| {
                self.tokenizer_registry
                    .get_by_name(m)
                    .or_else(|| self.tokenizer_registry.get_by_id(m))
            })
            .ok_or_else(|| {
                IngressError::new(
                    http::StatusCode::SERVICE_UNAVAILABLE,
                    "tokenizer_unavailable",
                    "token routing requires input_ids or a model with a registered tokenizer",
                )
            })?;
        let limit = self.max_tokenize_bytes;
        if metadata.text.len() > limit {
            return Err(Self::tokenizer_input_too_large());
        }
        ctx.check()?;
        let template_started = Instant::now();
        let prompt = if let Some(chat) = parsed.chat() {
            super::prepare::chat_template::process_chat_messages(chat, &*entry.tokenizer)
                .map_err(IngressError::invalid)?
                .text
        } else {
            metadata.text.clone()
        };
        metrics::histogram!("mesh_prepare_stage_seconds", "stage" => "template")
            .record(template_started.elapsed().as_secs_f64());
        ctx.check()?;
        if prompt.len() > limit {
            return Err(Self::tokenizer_input_too_large());
        }
        let encoder = super::prepare::thread_tokenizer::encoder(&entry);
        let encode_started = Instant::now();
        let tokens = encoder
            .encode(&prompt, false)
            .map_err(|e| IngressError::invalid(e.to_string()))?
            .token_ids()
            .to_vec();
        metrics::histogram!("mesh_prepare_stage_seconds", "stage" => "encode")
            .record(encode_started.elapsed().as_secs_f64());
        Ok(tokens)
    }
}

#[cfg(test)]
#[path = "../../tests/routing/ingress_tests.rs"]
mod tests;
