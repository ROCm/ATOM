use std::{
    io,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use parking_lot::Mutex;
use pyo3::{
    types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods, PyTypeMethods},
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
};
use serde::Serialize;
use serde_json::{json, Map, Number, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::{
    app_context::AppContext,
    protocols::{
        chat::ChatCompletionRequest,
        common::StringOrArray,
        completion::CompletionRequest,
        generate::GenerateRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        engine_core::{EngineCoreClient, EngineCoreTransport},
        grpc::completion_adapter::{
            completion_to_generate, wrap_generate_response_as_completion,
            wrap_streaming_generate_as_completion,
        },
        prepare::{self, generation_payload::GenerationPayload},
        render,
        token_handle::engine_error::EngineError,
        RouterTrait,
    },
};

type RouterResult<T> = Result<T, Response>;

pub struct AtomStandaloneRouter {
    pub service: Option<Py<PyAny>>,
    pub engine_core_transport: Option<Arc<Mutex<EngineCoreTransport>>>,
    engine_core_client: Option<Arc<EngineCoreClient>>,
    engine_core_block_size: Option<i32>,
    app_context: Arc<AppContext>,
    default_chat_template_kwargs: Map<String, Value>,
    session_affinity_enabled: bool,
    close_service_on_shutdown: bool,
    closed: AtomicBool,
}

pub struct AtomStandaloneRuntime {
    pub service: Option<Py<PyAny>>,
    pub engine_core_transport: Option<Arc<Mutex<EngineCoreTransport>>>,
    pub engine_core_block_size: Option<i32>,
    pub engine_core_max_model_len: Option<i64>,
    pub engine_core_max_pool_tokens: Option<i64>,
    pub engine_core_client_slot: Option<Arc<Mutex<Option<Arc<EngineCoreClient>>>>>,
    pub engine_core_shutdown_grace_period_secs: u64,
    pub engine_core_dp_load_balance: Option<String>,
    pub engine_core_dp_lb_request_equivalent: Option<u64>,
    pub engine_core_num_draft_tokens: Option<i32>,
    pub engine_core_has_per_req_cache: Option<bool>,
    pub engine_core_session_affinity_enabled: bool,
    pub engine_core_dp_attention_enabled: bool,
    pub default_chat_template_kwargs: Map<String, Value>,
    pub close_service_on_shutdown: bool,
}

impl AtomStandaloneRouter {
    pub fn from_runtime(
        runtime: &AtomStandaloneRuntime,
        app_context: Arc<AppContext>,
    ) -> Result<Self, String> {
        Python::attach(|py| {
            let engine_core_client = runtime
                .engine_core_transport
                .as_ref()
                .map(|transport| {
                    EngineCoreClient::new(
                        transport.clone(),
                        runtime
                            .engine_core_dp_load_balance
                            .as_deref()
                            .unwrap_or("round_robin"),
                        runtime.engine_core_dp_lb_request_equivalent.unwrap_or(256),
                        Duration::from_secs(app_context.router_config.request_timeout_secs),
                        runtime.engine_core_num_draft_tokens.unwrap_or(0),
                        runtime.engine_core_has_per_req_cache.unwrap_or(false),
                        runtime.engine_core_max_model_len,
                        runtime.engine_core_max_pool_tokens,
                        Duration::from_secs(runtime.engine_core_shutdown_grace_period_secs),
                        runtime.engine_core_dp_attention_enabled,
                    )
                    .map_err(|error| error.to_string())
                })
                .transpose()?;
            if let Some(slot) = runtime.engine_core_client_slot.as_ref() {
                *slot.lock() = engine_core_client.clone();
            }
            Ok(Self {
                service: runtime
                    .service
                    .as_ref()
                    .map(|service| service.clone_ref(py)),
                engine_core_transport: runtime.engine_core_transport.clone(),
                engine_core_client,
                engine_core_block_size: runtime.engine_core_block_size,
                app_context,
                default_chat_template_kwargs: runtime.default_chat_template_kwargs.clone(),
                session_affinity_enabled: runtime.engine_core_session_affinity_enabled,
                close_service_on_shutdown: runtime.close_service_on_shutdown,
                closed: AtomicBool::new(false),
            })
        })
    }

    pub async fn flush_engine_cache(&self) -> Result<(), String> {
        self.engine_core_client
            .as_ref()
            .ok_or_else(|| "Rust EngineCore transport is not active".to_string())?
            .execute_utility_all("clear_kv_cache", None, Duration::from_secs(300))
            .await
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    pub async fn execute_engine_utility(&self, command: &str) -> Result<Value, String> {
        let responses = self
            .engine_core_client
            .as_ref()
            .ok_or_else(|| "Rust EngineCore transport is not active".to_string())?
            .execute_utility_all_json(command, None, Duration::from_secs(300))
            .await
            .map_err(|error| error.to_string())?;
        Ok(json!({"command": command, "ranks": responses}))
    }

    pub fn engine_loads(&self) -> Option<Value> {
        self.engine_core_client.as_ref().map(|client| {
            Value::Array(
                client
                    .load_snapshot()
                    .into_iter()
                    .map(|(rank, (requests, prompt_tokens))| {
                        json!({
                            "dp_rank": rank,
                            "inflight_requests": requests,
                            "inflight_prompt_tokens": prompt_tokens,
                        })
                    })
                    .collect(),
            )
        })
    }

    fn engine_core_not_ready(endpoint: &'static str) -> Response {
        Self::error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "ATOM standalone {endpoint} is waiting for the Rust EngineCore \
                 request pipeline"
            ),
        )
    }

    fn routing_hints<T: Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        body: &T,
    ) -> Result<(Option<usize>, Option<String>, Option<String>), String> {
        let value = serde_json::to_value(body)
            .map_err(|error| format!("failed to inspect request routing hints: {error}"))?;
        let header_rank = headers
            .and_then(|headers| headers.get("x-data-parallel-rank"))
            .map(|value| {
                value
                    .to_str()
                    .map_err(|_| "X-Data-Parallel-Rank must be valid UTF-8".to_string())?
                    .parse::<usize>()
                    .map_err(|_| "X-Data-Parallel-Rank must be a non-negative integer".to_string())
            })
            .transpose()?;
        let body_rank = value
            .get("data_parallel_rank")
            .map(|rank| {
                rank.as_u64()
                    .and_then(|rank| usize::try_from(rank).ok())
                    .ok_or_else(|| "data_parallel_rank must be a non-negative integer".to_string())
            })
            .transpose()?;
        let preferred_rank = header_rank.or(body_rank);
        if !self.session_affinity_enabled {
            return Ok((preferred_rank, None, None));
        }
        let header_value = |name: &str| {
            headers
                .and_then(|headers| headers.get(name))
                .and_then(|value| value.to_str().ok())
                .map(str::to_string)
        };
        let session_id = header_value("x-dynamo-session-id")
            .or_else(|| header_value("x-correlation-id"))
            .or_else(|| {
                value
                    .get("session_params")
                    .and_then(Value::as_object)
                    .and_then(|params| params.get("session_id").or_else(|| params.get("id")))
                    .and_then(Value::as_str)
                    .or_else(|| value.get("conversation_id").and_then(Value::as_str))
                    .map(str::to_string)
            });
        let parent_session_id = header_value("x-dynamo-parent-session-id");
        Ok((preferred_rank, session_id, parent_session_id))
    }

    fn contains_multimodal_input<T: Serialize>(body: &T) -> bool {
        fn visit(value: &Value) -> bool {
            match value {
                Value::Object(object) => object.iter().any(|(key, value)| {
                    matches!(
                        key.as_str(),
                        "image_data"
                            | "video_data"
                            | "audio_data"
                            | "input_embeds"
                            | "image_url"
                            | "video_url"
                            | "audio_url"
                    ) && !value.is_null()
                        || visit(value)
                }),
                Value::Array(values) => values.iter().any(visit),
                _ => false,
            }
        }
        serde_json::to_value(body).is_ok_and(|value| visit(&value))
    }

    fn encode_stop_sequences(
        payload: &GenerationPayload,
        response_context: &prepare::response_context::ResponseContext,
    ) -> Result<Vec<Vec<u32>>, String> {
        let stops: Vec<&str> = match payload.stop.stop.as_ref() {
            Some(StringOrArray::String(stop)) => vec![stop.as_str()],
            Some(StringOrArray::Array(stops)) => stops.iter().map(String::as_str).collect(),
            None => Vec::new(),
        };
        stops
            .into_iter()
            .map(|stop| {
                response_context
                    .tokenizer
                    .encode(stop, false)
                    .map(|encoding| encoding.token_ids().to_vec())
                    .map_err(|error| format!("failed to tokenize stop sequence: {error}"))
            })
            .collect()
    }

    fn apply_default_chat_template_kwargs(
        &self,
        body: &ChatCompletionRequest,
    ) -> Result<ChatCompletionRequest, String> {
        if self.default_chat_template_kwargs.is_empty() {
            return Ok(body.clone());
        }
        let mut value = serde_json::to_value(body)
            .map_err(|error| format!("failed to serialize chat request: {error}"))?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| "chat request did not serialize as an object".to_string())?;
        let request_kwargs = object
            .entry("chat_template_kwargs")
            .or_insert_with(|| Value::Object(Map::new()));
        if request_kwargs.is_null() {
            *request_kwargs = Value::Object(Map::new());
        }
        let request_kwargs = request_kwargs
            .as_object_mut()
            .ok_or_else(|| "chat_template_kwargs must serialize as an object".to_string())?;
        for (key, value) in &self.default_chat_template_kwargs {
            request_kwargs
                .entry(key.clone())
                .or_insert_with(|| value.clone());
        }
        serde_json::from_value(value)
            .map_err(|error| format!("failed to apply chat template defaults: {error}"))
    }

    fn not_implemented(endpoint: &'static str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({
                "error": {
                    "type": "not_implemented",
                    "message": format!("ATOM standalone route for {endpoint} is not implemented yet"),
                }
            })),
        )
            .into_response()
    }

    fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
        (
            status,
            Json(json!({
                "error": {
                    "message": message.into(),
                    "type": if status.is_client_error() {
                        "invalid_request_error"
                    } else {
                        "internal_server_error"
                    },
                    "code": status.as_u16(),
                }
            })),
        )
            .into_response()
    }

    fn engine_dispatch_error(error: EngineError) -> Response {
        let status = if matches!(error, EngineError::RequestBuildFailed(_)) {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        Self::error_response(status, format!("EngineCore dispatch failed: {error}"))
    }

    fn run_chat_completion(&self, body: &ChatCompletionRequest) -> RouterResult<Value> {
        self.call_service("chat_completions", body, "chat completion")
    }

    fn run_chat_completion_stream(&self, body: &ChatCompletionRequest) -> Response {
        self.run_sse_service_stream(
            body,
            "start_chat_completions_stream",
            "drain_chat_completions_stream",
            "close_chat_completions_stream",
            "chat completion",
        )
    }

    fn close_python_stream(service: &Py<PyAny>, close_method: &'static str, stream_id: &str) {
        Python::attach(|py| {
            let _ = service.bind(py).call_method1(close_method, (stream_id,));
        });
    }

    fn py_error_status(error: &pyo3::PyErr) -> StatusCode {
        Python::attach(|py| {
            error
                .get_type(py)
                .name()
                .map(|name| {
                    if name == "ValueError" {
                        StatusCode::BAD_REQUEST
                    } else {
                        StatusCode::INTERNAL_SERVER_ERROR
                    }
                })
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        })
    }

    fn run_completion(&self, body: &CompletionRequest) -> RouterResult<Value> {
        self.call_service("completions", body, "completion")
    }

    fn run_completion_stream(&self, body: &CompletionRequest) -> Response {
        self.run_sse_service_stream(
            body,
            "start_completions_stream",
            "drain_completions_stream",
            "close_completions_stream",
            "completion",
        )
    }

    async fn execute_engine_core_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        if Self::contains_multimodal_input(body) {
            return Self::error_response(
                StatusCode::BAD_REQUEST,
                "Rust EngineCore transport does not yet support multimodal input",
            );
        }
        let body = match self.apply_default_chat_template_kwargs(body) {
            Ok(body) => body,
            Err(error) => {
                return Self::error_response(StatusCode::BAD_REQUEST, error);
            }
        };
        let (payload, response_context) = match prepare::prepare_chat(
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(str::to_string),
            &self.app_context,
        ) {
            Ok(prepared) => prepared,
            Err(response) => return response,
        };
        let Some(client) = self.engine_core_client.as_ref() else {
            return Self::engine_core_not_ready("chat completion");
        };
        let encoded_stops = match Self::encode_stop_sequences(&payload, &response_context) {
            Ok(stops) => stops,
            Err(error) => {
                return Self::error_response(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
        };
        let block_size = self.engine_core_block_size.unwrap_or(16);
        let (preferred_rank, session_id, parent_session_id) =
            match self.routing_hints(headers, &body) {
                Ok(hints) => hints,
                Err(error) => return Self::error_response(StatusCode::BAD_REQUEST, error),
            };
        let stream = match client.submit_routed(
            &payload,
            block_size,
            preferred_rank,
            session_id.as_deref(),
            parent_session_id.as_deref(),
            &encoded_stops,
        ) {
            Ok(stream) => stream,
            Err(error) => return Self::engine_dispatch_error(error),
        };
        if body.stream {
            render::chat_streaming::process(stream, response_context, "atom")
        } else {
            render::chat_aggregator::process(stream, response_context).await
        }
    }

    async fn execute_engine_core_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        if Self::contains_multimodal_input(body) {
            return Self::error_response(
                StatusCode::BAD_REQUEST,
                "Rust EngineCore transport does not yet support multimodal input",
            );
        }
        let (payload, response_context) = match prepare::prepare_generate(
            Arc::new(body.clone()),
            headers.cloned(),
            model_id.map(str::to_string),
            &self.app_context,
        ) {
            Ok(prepared) => prepared,
            Err(response) => return response,
        };
        let Some(client) = self.engine_core_client.as_ref() else {
            return Self::engine_core_not_ready("generate");
        };
        let encoded_stops = match Self::encode_stop_sequences(&payload, &response_context) {
            Ok(stops) => stops,
            Err(error) => {
                return Self::error_response(StatusCode::INTERNAL_SERVER_ERROR, error);
            }
        };
        let block_size = self.engine_core_block_size.unwrap_or(16);
        let (preferred_rank, session_id, parent_session_id) =
            match self.routing_hints(headers, body) {
                Ok(hints) => hints,
                Err(error) => return Self::error_response(StatusCode::BAD_REQUEST, error),
            };
        let stream = match client.submit_routed(
            &payload,
            block_size,
            preferred_rank,
            session_id.as_deref(),
            parent_session_id.as_deref(),
            &encoded_stops,
        ) {
            Ok(stream) => stream,
            Err(error) => return Self::engine_dispatch_error(error),
        };
        if body.stream {
            render::generate_streaming::process(stream, response_context, "atom")
        } else {
            render::generate_aggregator::process(stream, response_context).await
        }
    }

    fn run_sse_service_stream<T: Serialize>(
        &self,
        body: &T,
        start_method: &'static str,
        drain_method: &'static str,
        close_method: &'static str,
        endpoint: &'static str,
    ) -> Response {
        let Some(service) = self.service.as_ref() else {
            return Self::engine_core_not_ready(endpoint);
        };
        let request_value = match serde_json::to_value(body) {
            Ok(value) => value,
            Err(e) => {
                return Self::error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Failed to serialize {endpoint} request: {e}"),
                )
            }
        };

        let stream_id = match Python::attach(|py| -> PyResult<String> {
            let request = Self::json_to_py(py, &request_value)?;
            service
                .bind(py)
                .call_method1(start_method, (request,))?
                .extract::<String>()
        }) {
            Ok(stream_id) => stream_id,
            Err(e) => {
                return Self::error_response(
                    Self::py_error_status(&e),
                    format!("ATOM standalone {endpoint} stream failed: {e}"),
                )
            }
        };

        let service = Python::attach(|py| service.clone_ref(py));
        let stream_id_for_worker = stream_id.clone();
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, io::Error>>();
        let _ = tokio::task::spawn_blocking(move || loop {
            let chunks = Python::attach(|py| -> PyResult<Vec<String>> {
                service
                    .bind(py)
                    .call_method1(
                        drain_method,
                        (stream_id_for_worker.as_str(), 16usize, 0.05f64),
                    )?
                    .extract::<Vec<String>>()
            });

            match chunks {
                Ok(chunks) => {
                    if chunks.is_empty() {
                        continue;
                    }
                    for chunk in chunks {
                        let done = chunk.trim() == "data: [DONE]";
                        if tx.send(Ok(Bytes::from(chunk))).is_err() {
                            Self::close_python_stream(
                                &service,
                                close_method,
                                &stream_id_for_worker,
                            );
                            return;
                        }
                        if done {
                            Self::close_python_stream(
                                &service,
                                close_method,
                                &stream_id_for_worker,
                            );
                            return;
                        }
                    }
                }
                Err(error) => {
                    let error_chunk = json!({
                        "error": {
                            "message": error.to_string(),
                            "type": "internal_server_error",
                        }
                    });
                    let _ = tx.send(Ok(Bytes::from(format!("data: {}\n\n", error_chunk))));
                    Self::close_python_stream(&service, close_method, &stream_id_for_worker);
                    return;
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let mut response = Response::new(Body::from_stream(stream));
        *response.status_mut() = StatusCode::OK;
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        response
            .headers_mut()
            .insert("Cache-Control", HeaderValue::from_static("no-cache"));
        response
            .headers_mut()
            .insert("Connection", HeaderValue::from_static("keep-alive"));
        response
    }

    fn call_service<T: Serialize>(
        &self,
        method_name: &'static str,
        body: &T,
        endpoint: &'static str,
    ) -> RouterResult<Value> {
        let Some(service) = self.service.as_ref() else {
            return Err(Self::engine_core_not_ready(endpoint));
        };
        let request_value = serde_json::to_value(body).map_err(|e| {
            Self::error_response(
                StatusCode::BAD_REQUEST,
                format!("Failed to serialize {endpoint} request: {e}"),
            )
        })?;

        Python::attach(|py| -> PyResult<Value> {
            let request = Self::json_to_py(py, &request_value)?;
            let response = service.bind(py).call_method1(method_name, (request,))?;
            Self::py_to_json(&response)
        })
        .map_err(|e| {
            Self::error_response(
                Self::py_error_status(&e),
                format!("ATOM standalone {endpoint} failed: {e}"),
            )
        })
    }

    fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
        match value {
            Value::Null => Ok(py.None()),
            Value::Bool(value) => Ok(value.into_pyobject(py)?.to_owned().into_any().unbind()),
            Value::Number(value) => {
                if let Some(value) = value.as_i64() {
                    Ok(value.into_pyobject(py)?.into_any().unbind())
                } else if let Some(value) = value.as_u64() {
                    Ok(value.into_pyobject(py)?.into_any().unbind())
                } else if let Some(value) = value.as_f64() {
                    Ok(value.into_pyobject(py)?.into_any().unbind())
                } else {
                    Ok(py.None())
                }
            }
            Value::String(value) => Ok(value.into_pyobject(py)?.into_any().unbind()),
            Value::Array(values) => {
                let items: PyResult<Vec<_>> = values
                    .iter()
                    .map(|value| Self::json_to_py(py, value))
                    .collect();
                Ok(PyList::new(py, items?)?.into_any().unbind())
            }
            Value::Object(values) => {
                let dict = PyDict::new(py);
                for (key, value) in values {
                    dict.set_item(key, Self::json_to_py(py, value)?)?;
                }
                Ok(dict.into_any().unbind())
            }
        }
    }

    fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<Value> {
        if value.is_none() {
            return Ok(Value::Null);
        }
        if let Ok(value) = value.extract::<bool>() {
            return Ok(Value::Bool(value));
        }
        if let Ok(value) = value.extract::<i64>() {
            return Ok(Value::Number(Number::from(value)));
        }
        if let Ok(value) = value.extract::<u64>() {
            return Ok(Value::Number(Number::from(value)));
        }
        if let Ok(value) = value.extract::<f64>() {
            if let Some(number) = Number::from_f64(value) {
                return Ok(Value::Number(number));
            }
        }
        if let Ok(value) = value.extract::<String>() {
            return Ok(Value::String(value));
        }
        if let Ok(values) = value.cast::<PyList>() {
            let mut result = Vec::with_capacity(values.len());
            for item in values.iter() {
                result.push(Self::py_to_json(&item)?);
            }
            return Ok(Value::Array(result));
        }
        if let Ok(dict) = value.cast::<PyDict>() {
            let mut result = Map::new();
            for (key, item) in dict.iter() {
                result.insert(key.extract::<String>()?, Self::py_to_json(&item)?);
            }
            return Ok(Value::Object(result));
        }
        let item = value.call_method0("item")?;
        Self::py_to_json(&item)
    }

    fn python_type_name(&self) -> String {
        let Some(service) = self.service.as_ref() else {
            return "RustEngineCoreTransport".to_string();
        };
        Python::attach(|py| {
            service
                .bind(py)
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "unknown".to_string())
        })
    }

    fn close_service(&self, reason: &'static str) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        let Some(service) = self.service.as_ref() else {
            if let Some(client) = self.engine_core_client.as_ref() {
                tracing::info!("Shutting down Rust-owned EngineCore transport ({reason})");
                if let Err(error) = client.shutdown() {
                    tracing::warn!("Failed to shut down EngineCore transport: {error}");
                }
            } else if let Some(transport) = self.engine_core_transport.as_ref() {
                let _ = transport.lock().send_shutdown_all();
            }
            return;
        };
        if !self.close_service_on_shutdown {
            tracing::info!(
                "Skipping ATOM standalone Python service close because it is externally owned ({})",
                reason
            );
            return;
        }

        tracing::info!("Closing ATOM standalone Python service ({})", reason);
        Python::attach(|py| {
            if let Err(e) = service.bind(py).call_method0("close") {
                tracing::warn!("Failed to close ATOM standalone Python service: {}", e);
            }
        });
    }
}

impl std::fmt::Debug for AtomStandaloneRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomStandaloneRouter")
            .finish_non_exhaustive()
    }
}

impl Drop for AtomStandaloneRouter {
    fn drop(&mut self) {
        self.close_service("drop");
    }
}

#[async_trait]
impl RouterTrait for AtomStandaloneRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn shutdown(&self) {
        if self.service.is_none() {
            if self.closed.swap(true, Ordering::AcqRel) {
                return;
            }
            if let Some(client) = self.engine_core_client.as_ref().cloned() {
                tracing::info!("Shutting down Rust-owned EngineCore transport (server shutdown)");
                match tokio::task::spawn_blocking(move || client.shutdown()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::warn!("Failed to shut down EngineCore transport: {error}");
                    }
                    Err(error) => {
                        tracing::warn!("EngineCore shutdown task failed: {error}");
                    }
                }
            } else if let Some(transport) = self.engine_core_transport.as_ref() {
                let transport = transport.clone();
                match tokio::task::spawn_blocking(move || transport.lock().send_shutdown_all())
                    .await
                {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::warn!("Failed to shut down EngineCore transport: {error}");
                    }
                    Err(error) => {
                        tracing::warn!("EngineCore transport shutdown task failed: {error}");
                    }
                }
            }
            return;
        }
        self.close_service("server shutdown");
    }

    fn standalone_readiness(&self) -> Option<(bool, usize, usize)> {
        let client = self.engine_core_client.as_ref()?;
        let healthy = client.healthy_rank_count();
        let total = client.total_rank_count();
        let tokenizer_ready = self
            .app_context
            .router_config
            .tokenizer_path
            .as_ref()
            .or(self.app_context.router_config.model_path.as_ref())
            .is_some_and(|name| self.app_context.tokenizer_registry.contains(name));
        Some((tokenizer_ready && client.serving_ready(), healthy, total))
    }

    fn standalone_engine_metrics(&self) -> Option<(bool, Value)> {
        let client = self.engine_core_client.as_ref()?;
        let ranks = client.metrics_snapshot_json();
        let healthy_ranks = client.healthy_rank_count();
        let serving_ready = client.serving_ready();
        Some((
            serving_ready,
            json!({
                "status": if !serving_ready {
                    "not_ready"
                } else if ranks.is_empty() {
                    "waiting_for_snapshot"
                } else {
                    "ready"
                },
                "ranks": ranks,
                "healthy_ranks": healthy_ranks,
                "total_ranks": client.total_rank_count(),
            }),
        ))
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        if matches!(self.standalone_readiness(), Some((true, _, _))) {
            return StatusCode::OK.into_response();
        }
        StatusCode::SERVICE_UNAVAILABLE.into_response()
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::OK,
            Json(json!({
                "router_type": self.router_type(),
                "service_type": self.python_type_name(),
            })),
        )
            .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let model = self
            .app_context
            .router_config
            .model_path
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        (
            StatusCode::OK,
            Json(json!({
                "object": "list",
                "data": [{
                    "id": model,
                    "object": "model",
                    "owned_by": "atom"
                }]
            })),
        )
            .into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        if let Some(model) = self.app_context.router_config.model_path.as_ref() {
            return (
                StatusCode::OK,
                Json(json!({
                    "model_path": model,
                    "backend": "atom_engine_core",
                })),
            )
                .into_response();
        }
        Self::not_implemented("get_model_info")
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        if self.engine_core_client.is_some() {
            return self
                .execute_engine_core_generate(headers, body, model_id)
                .await;
        }
        Self::not_implemented("generate")
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        if self.engine_core_client.is_some() {
            return self.execute_engine_core_chat(headers, body, model_id).await;
        }
        if body.stream {
            return self.run_chat_completion_stream(body);
        }

        match self.run_chat_completion(body) {
            Ok(body) => (StatusCode::OK, Json(body)).into_response(),
            Err(response) => response,
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        if self.engine_core_client.is_some() {
            let generate = match completion_to_generate(body) {
                Ok(request) => request,
                Err(message) => {
                    return Self::error_response(StatusCode::BAD_REQUEST, message);
                }
            };
            let response = self
                .execute_engine_core_generate(headers, &generate, model_id)
                .await;
            return if body.stream {
                wrap_streaming_generate_as_completion(response, body.model.clone()).await
            } else {
                wrap_generate_response_as_completion(response, body.model.clone()).await
            };
        }
        if body.stream {
            return self.run_completion_stream(body);
        }

        match self.run_completion(body) {
            Ok(body) => (StatusCode::OK, Json(body)).into_response(),
            Err(response) => response,
        }
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        Self::not_implemented("responses")
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        Self::not_implemented("responses_get")
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        Self::not_implemented("responses_cancel")
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        Self::not_implemented("responses_delete")
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        Self::not_implemented("responses_input_items")
    }

    fn router_type(&self) -> &'static str {
        "atom_standalone"
    }

    fn is_pd_mode(&self) -> bool {
        false
    }
}
