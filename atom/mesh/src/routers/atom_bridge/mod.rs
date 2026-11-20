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
//! `--frontend rust`; see `docs/atomesh_rust_frontend.md`.

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

use self::{events::EventRegistry, py_bridge::BridgeDispatcher};
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
        chat::ChatCompletionRequest, common::ResponseFormat, completion::CompletionRequest,
        generate::GenerateRequest,
    },
    routers::{
        comm::error::{self, IngressError},
        grpc::{
            completion_adapter::{
                completion_to_generate, wrap_generate_response_as_completion,
                wrap_streaming_generate_as_completion,
            },
            pipeline::Pipeline,
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
        })
    }
}

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
}

impl RequestExtras {
    pub(crate) fn current() -> Self {
        REQUEST_EXTRAS.try_with(Clone::clone).unwrap_or_default()
    }

    fn read(request: &InferenceEnvelope) -> Result<Self, Response> {
        let raw = match &request.parsed {
            ParsedInference::Generate(body) => {
                return Ok(Self {
                    data_parallel_rank: body.data_parallel_rank.map(i64::from),
                });
            }
            _ if memchr::memmem::find(&request.body, b"\"data_parallel_rank\"").is_none() => {
                return Ok(Self::default());
            }
            _ => serde_json::from_slice::<Map<String, Value>>(&request.body)
                .map_err(|e| error::bad_request("invalid_request", e.to_string()))?,
        };
        let data_parallel_rank = match raw.get("data_parallel_rank") {
            None | Some(Value::Null) => None,
            Some(Value::Number(n)) if n.as_i64().is_some_and(|r| r >= 0) => n.as_i64(),
            Some(other) => {
                return Err(error::bad_request(
                    "invalid_request",
                    format!("data_parallel_rank must be a non-negative integer, got {other}"),
                ))
            }
        };
        Ok(Self { data_parallel_rank })
    }
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
        );
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
         frontend yet; use --frontend python",
    )
}

fn chat_wants_structured_output(body: &ChatCompletionRequest) -> bool {
    matches!(
        body.response_format,
        Some(ResponseFormat::JsonObject | ResponseFormat::JsonSchema { .. })
    ) || body.regex.is_some()
        || body.ebnf.is_some()
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
        let extras = match RequestExtras::read(&request) {
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
        let mut body = body.clone();
        let defaults = &self.runtime.default_chat_template_kwargs;
        if !defaults.is_empty() {
            let kwargs = body
                .chat_template_kwargs
                .get_or_insert_with(Default::default);
            for (key, value) in defaults {
                kwargs.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        self.pipeline
            .execute_chat(
                Arc::new(body),
                headers.cloned(),
                self.model_id(),
                Arc::clone(&self.app_context),
            )
            .await
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
