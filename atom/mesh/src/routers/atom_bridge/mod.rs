//! `AtomBridgeRouter`: ATOM standalone with the OpenAI API served by Rust.
//!
//! The gRPC router's pipeline end to end -- prepare (chat template,
//! tokenize), dispatch, render (detokenize, reasoning and tool parsers, SSE)
//! -- with the engine behind `Dispatcher` being ATOM's own LLMEngine, reached
//! through the token bridge (`py_bridge`). The request-level parts here and
//! in `events` know nothing of Python; a dispatcher that speaks to the
//! EngineCores directly can take `BridgeDispatcher`'s place.
//!
//! Python starts it from `atom/entrypoints/atomesh/server.py` with
//! `USE_ATOMESH_ENTRYPOINTS=1`; see `docs/atomesh_rust_frontend.md`.
//!
//! It serves a P/D leg behind the Atomesh router as the OpenAI server does:
//! a chat's or completion's `kv_transfer_params` go to the engine, a decode
//! submits the prompt its prefill ran, and the response carries a prefill's
//! handoff back (`Returned`); `kv_transfer_info` / `server_info` answer the
//! router's registration.

pub(crate) mod events;
pub mod py_bridge;

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use pyo3::prelude::*;
use serde_json::{json, Map, Value};

use self::{
    events::{EventRegistry, Returned},
    py_bridge::BridgeDispatcher,
};
use crate::{
    app_context::AppContext,
    core::{
        placement::{
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, RequestDescriptor},
        },
        BasicWorkerBuilder, Worker,
    },
    observability::metrics::metrics_labels,
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
            pipeline::{AtomOutput, Pipeline},
        },
        ingress::InferenceEnvelope,
        prepare::inference::ParsedInference,
        RouterTrait,
    },
};

/// What Python hands Rust at launch: its `TokenBridge`, the sink that bridge
/// pushes engine steps into, and the name the model is served as (which is
/// also the name its tokenizer is registered under).
pub struct AtomBridgeRuntime {
    pub(crate) bridge: Arc<Py<PyAny>>,
    pub(crate) registry: Arc<EventRegistry>,
    pub served_model_name: String,
    /// `--default-chat-template-kwargs`: under every chat request's own.
    pub(crate) default_chat_template_kwargs: Map<String, Value>,
    /// The bridge checks a chat's tool calls against the grammar the OpenAI
    /// server would hold them to (`TokenBridge.checks_tool_call_grammar`),
    /// and so takes the chat's body.
    pub(crate) checks_tool_call_grammar: bool,
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
        let checks_tool_call_grammar = bridge
            .getattr("checks_tool_call_grammar")?
            .extract::<bool>()?;
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
            bridge: Arc::new(bridge.unbind()),
            registry,
            served_model_name,
            default_chat_template_kwargs,
            checks_tool_call_grammar,
            end_token_ids,
        })
    }

    /// `GET /kv_transfer_info`, as the OpenAI server answers it
    /// (`pd_handoff.kv_transfer_info`): the engine's P/D role, its TP and DP
    /// sizes, and a chunked prefill's endpoint.
    async fn kv_transfer_info(&self) -> Result<Value, String> {
        let bridge = Arc::clone(&self.bridge);
        let info = tokio::task::spawn_blocking(move || {
            Python::attach(|py| {
                bridge
                    .bind(py)
                    .call_method0("kv_transfer_info")?
                    .extract::<String>()
            })
        })
        .await
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;
        serde_json::from_str(&info).map_err(|e| e.to_string())
    }
}

/// `GET /kv_transfer_info`.
pub(crate) async fn kv_transfer_info(runtime: &AtomBridgeRuntime) -> Response {
    match runtime.kv_transfer_info().await {
        Ok(info) => Json(info).into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
    }
}

/// `GET /server_info`, as the OpenAI server answers it: the router's
/// dp-aware discovery reads `dp_size` here.
pub(crate) async fn server_info(runtime: &AtomBridgeRuntime) -> Response {
    match runtime.kv_transfer_info().await {
        Ok(info) => Json(json!({
            "model_id": runtime.served_model_name,
            "served_model_name": runtime.served_model_name,
            "tp_size": info["tp_size"],
            "dp_size": info["dp_size"],
        }))
        .into_response(),
        Err(e) => (StatusCode::SERVICE_UNAVAILABLE, e).into_response(),
    }
}

/// The keys `RequestExtras::read` looks for, as they appear in a JSON body:
/// a body naming none of them is not parsed again.
const RAW_KEYS: [&[u8]; 3] = [
    b"\"data_parallel_rank\"",
    b"\"kv_transfer_params\"",
    b"\"return_token_ids\"",
];

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
    /// A chat's body as the client sent it, for a model that renders with an
    /// encoder of its own (`chat_render`): the typed request drops fields the
    /// encoder reads (`thinking`, a system message's `tools`, a tool message's
    /// `name`) and normalizes others (`tool_choice`).
    pub chat_body: Option<Arc<Map<String, Value>>>,
    /// The chat's body as JSON, for `TokenBridge.submit` to check its tool
    /// calls against the grammar (`checks_tool_call_grammar`).
    pub chat_json: Option<String>,
    /// A P/D leg's `kv_transfer_params` as JSON, for `TokenBridge.submit`,
    /// without the prompt ids a prefill handed its decode (`prompt_token_ids`).
    pub kv_transfer_params: Option<String>,
    /// The prompt a decode's prefill ran, which the decode submits in place
    /// of its own render, as the OpenAI server does.
    pub prompt_token_ids: Option<Arc<Vec<u32>>>,
    /// What the response carries back besides its render.
    pub returned: Option<Arc<Returned>>,
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
    /// typed request drops -- a chat's whole body with `keeps_chat_body`, a
    /// P/D leg's `kv_transfer_params`. A `/generate` P/D request is refused:
    /// its response (a list) has no place for a prefill's handoff, and served
    /// plainly its peer waits for nothing.
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
        if let Some(sent) = field("kv_transfer_params") {
            if matches!(request.parsed, ParsedInference::Generate(_)) {
                return Err(pd_refused());
            }
            let mut params = sent.as_object().cloned().ok_or_else(|| {
                error::bad_request("invalid_request", "kv_transfer_params must be an object")
            })?;
            if let Some(ids) = params.remove("prompt_token_ids").filter(|v| !v.is_null()) {
                extras.prompt_token_ids = Some(Arc::new(prompt_token_ids(&ids)?));
            }
            // Its handoff goes in the response body, which a stream has not.
            if params.get("do_remote_decode") == Some(&Value::Bool(true))
                && field("stream") == Some(&Value::Bool(true))
            {
                return Err(error::bad_request(
                    "invalid_request",
                    "a prefill leg (do_remote_decode) is served without streaming",
                ));
            }
            extras.kv_transfer_params = Some(Value::Object(params).to_string());
        }
        let return_token_ids = field("return_token_ids") == Some(&Value::Bool(true));
        if extras.kv_transfer_params.is_some() || return_token_ids {
            extras.returned = Some(Arc::new(Returned::new(return_token_ids)));
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
        "P/D over /generate (kv_transfer_params, bootstrap fields) is not supported by the \
         ATOM Rust frontend; send P/D legs to /v1/chat/completions or /v1/completions",
    )
}

/// `kv_transfer_params.prompt_token_ids`, as `resolve_prompt_token_ids` takes it.
fn prompt_token_ids(ids: &Value) -> Result<Vec<u32>, Response> {
    ids.as_array()
        .filter(|ids| !ids.is_empty())
        .and_then(|ids| {
            ids.iter()
                .map(|id| id.as_u64().and_then(|id| u32::try_from(id).ok()))
                .collect()
        })
        .ok_or_else(|| {
            error::bad_request(
                "invalid_request",
                "kv_transfer_params['prompt_token_ids'] must be a non-empty list of token ids",
            )
        })
}

/// `response` with what `returned` holds, at the top of its JSON body, as the
/// OpenAI server writes it: a prefill's `kv_transfer_params` and, asked for,
/// the `prompt_token_ids` it ran. A stream or an error is left as it is.
async fn attach(returned: &Returned, response: Response) -> Response {
    let handoff = returned.kv_transfer_params.get();
    let ids = returned.prompt_token_ids.get();
    let is_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("application/json"));
    if !response.status().is_success() || !is_json || (handoff.is_none() && ids.is_none()) {
        return response;
    }
    let (mut parts, body) = response.into_parts();
    let body = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(body) => body,
        Err(e) => return error::internal_error("response_read_failed", e.to_string()),
    };
    let Ok(Value::Object(mut object)) = serde_json::from_slice::<Value>(&body) else {
        return Response::from_parts(parts, Body::from(body));
    };
    if let Some(handoff) = handoff {
        match serde_json::from_str(handoff) {
            Ok(handoff) => object.insert("kv_transfer_params".into(), handoff),
            Err(e) => return error::internal_error("invalid_kv_transfer_params", e.to_string()),
        };
    }
    if let Some(ids) = ids {
        object.insert("prompt_token_ids".into(), json!(ids));
    }
    parts.headers.remove(header::CONTENT_LENGTH);
    Response::from_parts(parts, Body::from(Value::Object(object).to_string()))
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
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("atom://local")
                .model_id(runtime.served_model_name.clone())
                .build(),
        );
        let pipeline = Pipeline::with_injected(
            Arc::new(LocalPlanner { worker }),
            Arc::new(BridgeDispatcher::new(
                Arc::clone(&runtime.bridge),
                Arc::clone(&runtime.registry),
            )),
            metrics_labels::BACKEND_REGULAR,
        )
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

    /// The served model renders its chats with an encoder of its own, which
    /// reads them as sent (`chat_render`).
    fn renders_with_encoder(&self) -> bool {
        self.app_context
            .chat_controls_of(&self.runtime.served_model_name)
            .is_some()
    }

    /// A chat's body as sent is read: by an encoder, or by `submit`'s tool-call check.
    fn keeps_chat_body(&self) -> bool {
        self.renders_with_encoder() || self.runtime.checks_tool_call_grammar
    }

    async fn call_bridge(&self, method: &'static str) -> PyResult<bool> {
        let bridge = Arc::clone(&self.runtime.bridge);
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
        self.runtime.registry.fail_all("server is shutting down");
        if let Err(e) = self.call_bridge("close").await {
            tracing::warn!(error = %e, "TokenBridge.close failed");
        }
    }

    /// The typed routes, each inside its `RequestExtras`.
    async fn route_inference(
        &self,
        request: InferenceEnvelope,
        _app: &Arc<AppContext>,
    ) -> Response {
        let extras = match RequestExtras::read(&request, self.keeps_chat_body()) {
            Ok(extras) => extras,
            Err(response) => return response,
        };
        let returned = extras.returned.clone();
        let (uri, headers, parsed, metadata) = request.into_native_parts();
        let headers = Some(&headers);
        let model = metadata.model.as_deref();
        let response = REQUEST_EXTRAS
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
            .await;
        match returned {
            Some(returned) => attach(&returned, response).await,
            None => response,
        }
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
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
        // The chat as the client sent it, tool_choice included (K3's template
        // reads it), before the rewrite below. A chat reaching here without
        // its body (not through `route_inference`) is read from the typed one.
        let chat_body = extras.chat_body.take().or_else(|| {
            if !self.keeps_chat_body() {
                return None;
            }
            let typed = serde_json::to_value(&body).ok()?;
            typed.as_object().cloned().map(Arc::new)
        });
        // Every chat goes to `submit`'s check, whose `request_tools` also reads
        // the tools a message declares (K3's dynamic tools).
        if self.runtime.checks_tool_call_grammar {
            extras.chat_json = chat_body
                .as_deref()
                .map(|chat| Value::Object(chat.clone()).to_string());
        }
        let sent = chat_body.filter(|_| self.renders_with_encoder());
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
            sent,
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

    /// An app, and the prepare pool it parses on (closed when dropped).
    async fn app() -> (
        crate::core::prepare_pool::PreparePoolRuntime,
        Arc<AppContext>,
    ) {
        let pool = crate::core::prepare_pool::PreparePoolRuntime::new(
            crate::config::RouterConfig::default()
                .resolved_prepare_pool()
                .unwrap(),
        )
        .unwrap();
        let app = AppContext::from_config(Default::default(), 5, pool.handle())
            .await
            .unwrap();
        (pool, Arc::new(app))
    }

    async fn extras(app: &Arc<AppContext>, path: &str, body: Value) -> Result<RequestExtras, u16> {
        let request = InferenceEnvelope::parse(
            path.parse().unwrap(),
            headers(&[("content-type", "application/json")]),
            serde_json::to_vec(&body).unwrap().into(),
            app,
        )
        .await
        .unwrap();
        RequestExtras::read(&request, false).map_err(|r| r.status().as_u16())
    }

    #[tokio::test]
    async fn a_pd_legs_handoff_is_read_and_a_decode_takes_its_prefills_prompt() {
        let (_pool, app) = app().await;
        let chat = |kv: Value, extra: Value| {
            let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
            body["kv_transfer_params"] = kv;
            body.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            body
        };
        let path = "/v1/chat/completions";
        let decode = json!({"do_remote_prefill": true, "prompt_token_ids": [5, 6]});
        let read = extras(&app, path, chat(decode, json!({"stream": true})))
            .await
            .unwrap();
        assert_eq!(read.prompt_token_ids.as_deref(), Some(&vec![5, 6]));
        assert_eq!(
            read.kv_transfer_params.as_deref(),
            Some(r#"{"do_remote_prefill":true}"#)
        );
        let prefill = json!({"do_remote_decode": true});
        let read = extras(
            &app,
            path,
            chat(prefill.clone(), json!({"return_token_ids": true})),
        )
        .await
        .unwrap();
        assert!(read.returned.unwrap().wants_prompt_token_ids());
        // Refused: a streamed prefill (its handoff has nowhere to go), prompt
        // ids that are not ids, and P/D over /generate.
        for (path, body) in [
            (path, chat(prefill, json!({"stream": true}))),
            (path, chat(json!({"prompt_token_ids": [-1]}), json!({}))),
            ("/generate", json!({"text": "hi", "kv_transfer_params": {}})),
        ] {
            assert_eq!(extras(&app, path, body).await.unwrap_err(), 400);
        }
    }

    /// A prefill's response, as the Atomesh router reads it into its decode's
    /// request (`AtomAdapter::carry_prompt_token_ids`).
    #[tokio::test]
    async fn a_prefills_response_carries_what_the_router_hands_its_decode() {
        use crate::core::placement::backend::atom::AtomAdapter;
        let returned = Returned::new(true);
        returned.prompt_token_ids.set(vec![1, 2, 3]).unwrap();
        returned
            .kv_transfer_params
            .set(r#"{"remote_host":"h","dp_rank":1}"#.into())
            .unwrap();
        let response = attach(
            &returned,
            Json(json!({"id": "c", "choices": []})).into_response(),
        )
        .await;
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["id"], "c");
        let mut decode_kv = body["kv_transfer_params"].clone();
        assert_eq!(decode_kv["remote_host"], "h");
        assert_eq!(
            AtomAdapter::carry_prompt_token_ids(&body, &mut decode_kv),
            3
        );
        assert_eq!(decode_kv["prompt_token_ids"], json!([1, 2, 3]));
        // An error, or a stream, is left as it is.
        let error = attach(&returned, error::bad_request("x", "y")).await;
        assert_eq!(error.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn an_encoder_rendered_chat_keeps_the_fields_its_typed_body_drops() {
        let (_pool, app) = app().await;
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
