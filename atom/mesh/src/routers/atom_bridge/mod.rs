//! `AtomBridgeRouter`: ATOM standalone with the OpenAI API served by Rust.
//!
//! The gRPC router's pipeline end to end -- prepare (chat template,
//! tokenize), dispatch, render (detokenize, reasoning and tool parsers, SSE)
//! -- with the engine behind `Dispatcher` being ATOM's own: its LLMEngine,
//! reached through the token bridge (`py_bridge`), or its EngineCores, reached
//! directly (`direct`). The request-level parts here and in `events` know
//! nothing of Python.
//!
//! Python starts it from `atom/entrypoints/atomesh/server.py` with
//! `USE_ATOMESH_ENTRYPOINTS=1`; see `docs/atomesh_rust_frontend.md`.

mod direct;
pub(crate) mod events;
pub mod py_bridge;

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use pyo3::prelude::*;
use serde_json::{json, Map, Value};

use self::{
    direct::{DirectDispatcher, DpPlanner},
    events::EventRegistry,
    py_bridge::BridgeDispatcher,
};
use crate::{
    app_context::AppContext,
    atom_engine::AtomEngine,
    core::{
        placement::{
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, RequestDescriptor},
        },
        BasicWorkerBuilder, Worker,
    },
    observability::metrics::metrics_labels,
    policies::DpLoad,
    protocols::{
        chat::ChatCompletionRequest,
        common::{ResponseFormat, ToolChoice, ToolChoiceValue},
        completion::CompletionRequest,
        generate::GenerateRequest,
    },
    routers::{
        comm::error::{self, IngressError},
        grpc::{
            completion_adapter::{
                completion_to_generate, wrap_generate_response_as_completion,
                wrap_streaming_generate_as_completion,
            },
            engine::Dispatcher,
            pipeline::{AtomOutput, Pipeline},
        },
        ingress::InferenceEnvelope,
        prepare::inference::ParsedInference,
        RouterTrait,
    },
};

/// How requests reach the engine.
pub(crate) enum EngineLink {
    /// Python's `TokenBridge` to its LLMEngine, and the sink that bridge
    /// pushes engine steps into.
    Bridge {
        bridge: Arc<Py<PyAny>>,
        registry: Arc<EventRegistry>,
    },
    /// The EngineCores, Rust their one client.
    Direct(Arc<AtomEngine>),
}

/// What Python hands Rust at launch: the engine, and the name the model is
/// served as (which is also the name its tokenizer is registered under).
pub struct AtomBridgeRuntime {
    pub(crate) engine: EngineLink,
    pub served_model_name: String,
    /// `--default-chat-template-kwargs`: under every chat request's own.
    pub(crate) default_chat_template_kwargs: Map<String, Value>,
    /// The model's chat template is Python's to render (`TokenBridge.renders_chat`).
    pub(crate) renders_chat: bool,
    /// Every id that ends the model's output (`TokenBridge.end_token_ids`).
    pub(crate) end_token_ids: Vec<u32>,
}

impl AtomBridgeRuntime {
    /// `bridge.sink` must be the `StepSink` the bridge pushes into;
    /// `default_chat_template_kwargs` is a JSON object, or empty.
    pub fn from_bridge(
        bridge: Bound<'_, PyAny>,
        served_model_name: String,
        default_chat_template_kwargs: Option<&str>,
    ) -> PyResult<Self> {
        let sink = bridge.getattr("sink")?;
        let sink: PyRef<'_, py_bridge::StepSink> = sink.extract()?;
        let registry = Arc::clone(&sink.registry);
        let renders_chat = bridge.getattr("renders_chat")?.extract::<bool>()?;
        let end_token_ids = bridge.getattr("end_token_ids")?.extract::<Vec<u32>>()?;
        let default_chat_template_kwargs = match default_chat_template_kwargs {
            None | Some("") => Map::new(),
            Some(raw) => serde_json::from_str(raw).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "default_chat_template_kwargs must be a JSON object: {e}"
                ))
            })?,
        };
        Ok(Self {
            engine: EngineLink::Bridge {
                bridge: Arc::new(bridge.unbind()),
                registry,
            },
            served_model_name,
            default_chat_template_kwargs,
            renders_chat,
            end_token_ids,
        })
    }

    /// The EngineCores `engine` is connected to, serving the model its spec
    /// describes; the router renders every chat itself.
    pub fn from_engine(engine: Arc<AtomEngine>) -> Self {
        let model = &engine.model;
        Self {
            served_model_name: model.served_name.clone(),
            default_chat_template_kwargs: model.default_chat_template_kwargs.clone(),
            renders_chat: false,
            end_token_ids: model.end_token_ids.clone(),
            engine: EngineLink::Direct(engine),
        }
    }

    /// The EngineCores, when Rust is their client.
    pub fn direct_engine(&self) -> Option<&Arc<AtomEngine>> {
        match &self.engine {
            EngineLink::Direct(engine) => Some(engine),
            EngineLink::Bridge { .. } => None,
        }
    }

    fn bridge(&self) -> Option<&Arc<Py<PyAny>>> {
        match &self.engine {
            EngineLink::Bridge { bridge, .. } => Some(bridge),
            EngineLink::Direct(_) => None,
        }
    }
}

/// The keys `RequestExtras::read` looks for, as they appear in a JSON body:
/// a body naming none of them is not parsed again.
const RAW_KEYS: [&[u8]; 2] = [b"\"data_parallel_rank\"", b"\"kv_transfer_params\""];

tokio::task_local! {
    /// What a request carries besides its typed body, for the dispatcher.
    /// Set around each request's pipeline, which dispatches on the same task.
    pub(crate) static REQUEST_EXTRAS: RequestExtras;
}

/// Request fields the typed chat and completion bodies drop.
#[derive(Debug, Clone, Default)]
pub(crate) struct RequestExtras {
    /// The DP rank a router pinned the request to; CoreManager checks it.
    pub data_parallel_rank: Option<i64>,
    /// A chat's `tool_choice: "required"` with tools: the engine holds its
    /// output to the tool-call channel where the format defines one.
    pub tool_choice_required: bool,
    /// The request's DP session and its parent, from the same headers the
    /// OpenAI server reads (`_get_dp_session_affinity_ids`). CoreManager
    /// routes on them only with `ATOM_DP_SESSION_AFFINITY`, which Python
    /// checks.
    pub dp_session_id: Option<String>,
    pub dp_parent_session_id: Option<String>,
    /// The encoding `TokenBridge.render_chat` kept for this prompt, which
    /// `submit` hands to preprocessing so the prompt is hashed once.
    pub prompt_handle: Option<u64>,
    /// A chat's body as the client sent it, where Python renders the prompt:
    /// the typed request drops fields its template reads (a system message's
    /// `tools`, a tool message's `name`) and normalizes others (`tool_choice`).
    pub chat_body: Option<Arc<Map<String, Value>>>,
}

/// The session headers, in the OpenAI server's order of preference.
const SESSION_HEADERS: [&str; 3] = ["x-dynamo-session-id", "x-session-id", "x-correlation-id"];
const PARENT_SESSION_HEADER: &str = "x-dynamo-parent-session-id";

/// A request's DP session and parent session ids from its headers.
fn session_ids(headers: &HeaderMap) -> (Option<String>, Option<String>) {
    let header = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    (
        SESSION_HEADERS.iter().find_map(|name| header(name)),
        header(PARENT_SESSION_HEADER),
    )
}

impl RequestExtras {
    pub(crate) fn current() -> Self {
        REQUEST_EXTRAS.try_with(Clone::clone).unwrap_or_default()
    }

    /// The extras of a request: its session headers, and the body fields the
    /// typed request drops -- a chat's whole body with `keeps_chat_body`. A P/D
    /// request -- `kv_transfer_params`, or `/generate`'s bootstrap fields -- is
    /// refused: the transfer is not carried to the engine, and served plainly
    /// its peer waits for nothing.
    fn read(request: &InferenceEnvelope, keeps_chat_body: bool) -> Result<Self, Response> {
        let (dp_session_id, dp_parent_session_id) = session_ids(&request.headers);
        let mut extras = Self {
            dp_session_id,
            dp_parent_session_id,
            ..Self::default()
        };
        if let ParsedInference::Generate(body) = &request.parsed {
            if requests_pd_bootstrap(body) {
                return Err(pd_refused());
            }
            extras.data_parallel_rank = body.data_parallel_rank.map(i64::from);
        }
        let keeps_chat_body = keeps_chat_body && matches!(request.parsed, ParsedInference::Chat(_));
        if !keeps_chat_body
            && !RAW_KEYS
                .iter()
                .any(|key| memchr::memmem::find(&request.body, key).is_some())
        {
            return Ok(extras);
        }
        let raw = serde_json::from_slice::<Map<String, Value>>(&request.body)
            .map_err(|e| error::bad_request("invalid_request", e.to_string()))?;
        let field = |name: &str| raw.get(name).filter(|v| !v.is_null());
        if field("kv_transfer_params").is_some() {
            return Err(pd_refused());
        }
        if let Some(rank) = field("data_parallel_rank") {
            extras.data_parallel_rank =
                Some(rank.as_i64().filter(|r| *r >= 0).ok_or_else(|| {
                    error::bad_request(
                        "invalid_request",
                        format!("data_parallel_rank must be a non-negative integer, got {rank}"),
                    )
                })?);
        }
        if keeps_chat_body {
            extras.chat_body = Some(Arc::new(raw));
        }
        Ok(extras)
    }
}

/// A `/generate` body naming a P/D bootstrap peer.
fn requests_pd_bootstrap(body: &GenerateRequest) -> bool {
    body.bootstrap_host.is_some()
        || body.bootstrap_port.is_some()
        || body.bootstrap_room.is_some()
        || body.bootstrap_pair_key.is_some()
}

fn pd_refused() -> Response {
    error::bad_request(
        "pd_unsupported",
        "P/D (kv_transfer_params, bootstrap fields) is not supported by the ATOM Rust \
         frontend yet; serve P/D workers with the OpenAI server (USE_ATOMESH_ENTRYPOINTS=0)",
    )
}

/// Every request goes to the one local engine; CoreManager picks its DP rank.
struct LocalPlanner {
    worker: Arc<dyn Worker>,
}

#[async_trait]
impl PdPlanner for LocalPlanner {
    async fn plan(&self, _req: &RequestDescriptor<'_>) -> Result<PlacementPlan, PlacementError> {
        Ok(PlacementPlan::Single {
            worker: Arc::clone(&self.worker),
            policy_name: "atom_local",
            prefill_tokens: 0,
        })
    }
}

pub struct AtomBridgeRouter {
    runtime: Arc<AtomBridgeRuntime>,
    pipeline: Pipeline,
    app_context: Arc<AppContext>,
}

impl AtomBridgeRouter {
    pub fn new(ctx: &Arc<AppContext>, runtime: Arc<AtomBridgeRuntime>) -> Result<Self, String> {
        if ctx.reasoning_parser_factory.is_none() || ctx.tool_parser_factory.is_none() {
            return Err("ATOM Rust frontend requires the parser factories".to_string());
        }
        let model = runtime.served_model_name.as_str();
        let (planner, dispatcher): (Arc<dyn PdPlanner>, Arc<dyn Dispatcher>) = match &runtime.engine
        {
            EngineLink::Bridge { bridge, registry } => {
                let worker: Arc<dyn Worker> = Arc::new(
                    BasicWorkerBuilder::new("atom://local")
                        .model_id(model)
                        .build(),
                );
                (
                    Arc::new(LocalPlanner { worker }),
                    Arc::new(BridgeDispatcher::new(
                        Arc::clone(bridge),
                        Arc::clone(registry),
                    )),
                )
            }
            EngineLink::Direct(engine) => {
                let ranks = engine.client.core_count();
                let load = DpLoad::new(ranks);
                let policy = direct::policy_for(&engine.balance, &load);
                (
                    Arc::new(DpPlanner::new(direct::rank_workers(ranks, model), policy)),
                    Arc::new(DirectDispatcher::new(
                        Arc::clone(engine),
                        load,
                        Arc::clone(ctx),
                    )),
                )
            }
        };
        let pipeline =
            Pipeline::with_injected(planner, dispatcher, metrics_labels::BACKEND_REGULAR)
                .reading_output_as_atom(AtomOutput::new(runtime.end_token_ids.clone()));
        Ok(Self {
            runtime,
            pipeline,
            app_context: Arc::clone(ctx),
        })
    }

    /// The model every request is prepared for: there is one, whatever name
    /// the client sent.
    fn model_id(&self) -> Option<String> {
        Some(self.runtime.served_model_name.clone())
    }

    /// Call the `TokenBridge`'s `method`; there is none without one.
    async fn call_bridge(&self, method: &'static str) -> PyResult<bool> {
        let Some(bridge) = self.runtime.bridge() else {
            return Ok(true);
        };
        let bridge = Arc::clone(bridge);
        tokio::task::spawn_blocking(move || {
            Python::attach(|py| bridge.bind(py).call_method0(method)?.extract::<bool>())
        })
        .await
        .unwrap_or_else(|e| Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())))
    }
}

/// Constrained output is the OpenAI server's to serve for now: the shared
/// prepare step does not carry a grammar to the engine, so serving these here
/// would answer unconstrained without saying so.
fn structured_output_refused() -> Response {
    error::bad_request(
        "structured_output_unsupported",
        "response_format, json_schema, regex and ebnf are not supported by the ATOM Rust \
         frontend yet; use the OpenAI server (USE_ATOMESH_ENTRYPOINTS=0)",
    )
}

fn chat_wants_structured_output(body: &ChatCompletionRequest) -> bool {
    matches!(
        body.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    ) || body.regex.is_some()
        || body.ebnf.is_some()
}

/// The kind of the first non-text part in a chat's messages: the engine is
/// handed token ids only, so an image would be dropped without a word.
fn media_in(body: &ChatCompletionRequest) -> Option<String> {
    body.messages.iter().find_map(|message| {
        let message = serde_json::to_value(message).ok()?;
        message
            .get("content")?
            .as_array()?
            .iter()
            .filter_map(|part| part.get("type")?.as_str())
            .find(|kind| *kind != "text")
            .map(str::to_string)
    })
}

fn generate_wants_structured_output(body: &GenerateRequest) -> bool {
    body.sampling_params
        .as_ref()
        .is_some_and(|p| p.json_schema.is_some() || p.regex.is_some() || p.ebnf.is_some())
}

impl std::fmt::Debug for AtomBridgeRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomBridgeRouter")
            .field("served_model_name", &self.runtime.served_model_name)
            .finish()
    }
}

#[async_trait]
impl RouterTrait for AtomBridgeRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn shutdown(&self) {
        match &self.runtime.engine {
            EngineLink::Bridge { registry, .. } => {
                registry.fail_all("server is shutting down");
                if let Err(e) = self.call_bridge("close").await {
                    tracing::warn!(error = %e, "TokenBridge.close failed");
                }
            }
            // The launcher stops the Cores once the server is down; their
            // exit is expected from here on.
            EngineLink::Direct(engine) => engine.client.begin_shutdown(),
        }
    }

    /// The typed routes, each inside its `RequestExtras`.
    async fn route_inference(
        &self,
        request: InferenceEnvelope,
        _app: &Arc<AppContext>,
    ) -> Response {
        let extras = match RequestExtras::read(&request, self.runtime.renders_chat) {
            Ok(extras) => extras,
            Err(response) => return response,
        };
        let (uri, headers, parsed, metadata) = request.into_native_parts();
        let headers = Some(&headers);
        let model = metadata.model.as_deref();
        REQUEST_EXTRAS
            .scope(extras, async move {
                match parsed {
                    ParsedInference::Chat(body) => self.route_chat(headers, &body, model).await,
                    ParsedInference::Completion(body) => {
                        self.route_completion(headers, &body, model).await
                    }
                    ParsedInference::Generate(body) => {
                        self.route_generate(headers, &body, model).await
                    }
                    ParsedInference::Messages(_) | ParsedInference::Responses(_) => {
                        IngressError::new(
                            StatusCode::NOT_IMPLEMENTED,
                            "unsupported_api",
                            "not served by the ATOM Rust frontend yet",
                        )
                        .response(uri.path())
                    }
                }
            })
            .await
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        if let Some(engine) = self.runtime.direct_engine() {
            return match engine.client.lost() {
                None => (StatusCode::OK, "OK").into_response(),
                Some(reason) => (StatusCode::SERVICE_UNAVAILABLE, reason).into_response(),
            };
        }
        match self.call_bridge("is_healthy").await {
            Ok(true) => (StatusCode::OK, "OK").into_response(),
            Ok(false) => (StatusCode::SERVICE_UNAVAILABLE, "engine unhealthy").into_response(),
            Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e.to_string()).into_response(),
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        Json(json!({
            "router_type": self.router_type(),
            "served_model_name": self.runtime.served_model_name,
        }))
        .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        Json(json!({
            "object": "list",
            "data": [{
                "id": self.runtime.served_model_name,
                "object": "model",
                "owned_by": "atom",
            }],
        }))
        .into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        Json(json!({ "model_path": self.runtime.served_model_name })).into_response()
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        if chat_wants_structured_output(body) {
            return structured_output_refused();
        }
        if let Some(kind) = media_in(body) {
            return error::bad_request(
                "multimodal_unsupported",
                format!("{kind} content is not supported by the ATOM Rust frontend yet"),
            );
        }
        let mut body = body.clone();
        // The served model, whatever name the client sent: the parsers are
        // looked up by it, streamed or not.
        body.model = self.runtime.served_model_name.clone();
        let defaults = &self.runtime.default_chat_template_kwargs;
        if !defaults.is_empty() {
            let kwargs = body
                .chat_template_kwargs
                .get_or_insert_with(Default::default);
            for (key, value) in defaults {
                kwargs.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        // The engine does not hold a reply to a JSON schema of the tool, so
        // a tool_choice that asks for one is read as the model writes calls,
        // with the format's own parser, as the Python standalone service
        // reads it; "required" holds the output to the tool-call channel
        // instead, where the format has one.
        let mut extras = RequestExtras::current();
        // ATOM renders the prompt from the request as the client sent it,
        // tool_choice included (Kimi-K3's template reads it).
        // A chat reaching here without its body (not through `route_inference`)
        // is rendered from the typed request.
        let chat_body = extras.chat_body.take().or_else(|| {
            if !self.runtime.renders_chat {
                return None;
            }
            let typed = serde_json::to_value(&body).ok()?;
            typed.as_object().cloned().map(Arc::new)
        });
        let rendered = if let (Some(chat_body), Some(bridge)) = (chat_body, self.runtime.bridge()) {
            match py_bridge::render_chat(bridge, &chat_body, body.chat_template_kwargs.as_ref())
                .await
            {
                Ok((rendered, handle)) => {
                    extras.prompt_handle = handle;
                    Some(rendered)
                }
                Err(response) => return response,
            }
        } else {
            None
        };
        extras.tool_choice_required = matches!(
            body.tool_choice,
            Some(ToolChoice::Value(ToolChoiceValue::Required))
        ) && body.tools.as_ref().is_some_and(|t| !t.is_empty());
        if matches!(
            body.tool_choice,
            Some(
                ToolChoice::Value(ToolChoiceValue::Required)
                    | ToolChoice::Function { .. }
                    | ToolChoice::AllowedTools { .. }
            )
        ) {
            body.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
        }
        let pipeline = self.pipeline.execute_chat_with(
            Arc::new(body),
            headers.cloned(),
            self.model_id(),
            Arc::clone(&self.app_context),
            rendered,
        );
        REQUEST_EXTRAS.scope(extras, pipeline).await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        // Completions arrive here too, converted to a generate.
        if generate_wants_structured_output(body) {
            return structured_output_refused();
        }
        self.pipeline
            .execute_generate(
                Arc::new(body.clone()),
                headers.cloned(),
                self.model_id(),
                Arc::clone(&self.app_context),
            )
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let synthetic = match completion_to_generate(body) {
            Ok(g) => g,
            Err(msg) => return error::bad_request("completion_unsupported_field", msg),
        };
        let upstream = self.route_generate(headers, &synthetic, model_id).await;
        let model = self.runtime.served_model_name.clone();
        if body.stream {
            wrap_streaming_generate_as_completion(upstream, model).await
        } else {
            wrap_generate_response_as_completion(upstream, model).await
        }
    }

    fn router_type(&self) -> &'static str {
        "atom_bridge"
    }

    fn is_pd_mode(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in pairs {
            map.insert(*name, value.parse().unwrap());
        }
        map
    }

    #[tokio::test]
    async fn a_python_rendered_chat_keeps_the_fields_its_typed_body_drops() {
        let pool = crate::core::prepare_pool::PreparePoolRuntime::new(
            crate::config::RouterConfig::default()
                .resolved_prepare_pool()
                .unwrap(),
        )
        .unwrap();
        let app = Arc::new(
            AppContext::from_config(Default::default(), 5, pool.handle())
                .await
                .unwrap(),
        );
        let body = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "s", "tools": [{"type": "function",
                    "function": {"name": "f", "parameters": {}}}]},
                {"role": "tool", "content": "r", "tool_call_id": "c", "name": "f"},
            ],
            "tools": [],
        });
        let parse = |keeps: bool| {
            let body = serde_json::to_vec(&body).unwrap();
            let app = Arc::clone(&app);
            async move {
                let request = InferenceEnvelope::parse(
                    "/v1/chat/completions".parse().unwrap(),
                    headers(&[("content-type", "application/json")]),
                    body.into(),
                    &app,
                )
                .await
                .unwrap();
                RequestExtras::read(&request, keeps).unwrap()
            }
        };
        assert!(parse(false).await.chat_body.is_none());
        let kept = parse(true).await.chat_body.unwrap();
        assert_eq!(kept["messages"][0]["tools"][0]["function"]["name"], "f");
        assert_eq!(kept["messages"][1]["name"], "f");
        // As sent: an empty tool list is not read as `tool_choice: "none"`.
        assert!(kept.get("tool_choice").is_none());
    }

    #[test]
    fn a_generate_naming_a_bootstrap_peer_is_a_pd_request() {
        let body = |json: &str| serde_json::from_str::<GenerateRequest>(json).unwrap();
        assert!(!requests_pd_bootstrap(&body(r#"{"text": "hi"}"#)));
        for field in [
            r#""bootstrap_host": "10.0.0.1""#,
            r#""bootstrap_port": 8998"#,
            r#""bootstrap_room": 7"#,
            r#""bootstrap_pair_key": "k""#,
        ] {
            let json = format!(r#"{{"text": "hi", {field}}}"#);
            assert!(requests_pd_bootstrap(&body(&json)), "{field}");
        }
    }

    #[test]
    fn session_ids_come_from_the_openai_servers_headers_in_its_order() {
        assert_eq!(session_ids(&headers(&[])), (None, None));
        assert_eq!(
            session_ids(&headers(&[
                ("x-correlation-id", "c"),
                ("x-session-id", "s")
            ])),
            (Some("s".into()), None)
        );
        assert_eq!(
            session_ids(&headers(&[
                ("x-session-id", "s"),
                ("x-dynamo-session-id", "d"),
                ("x-dynamo-parent-session-id", "p"),
            ])),
            (Some("d".into()), Some("p".into()))
        );
        assert_eq!(
            session_ids(&headers(&[("x-correlation-id", "c")])),
            (Some("c".into()), None)
        );
    }
}
