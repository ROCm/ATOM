use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use pyo3::{
    types::{PyAnyMethods, PyBool, PyDict, PyDictMethods, PyList, PyListMethods, PyTypeMethods},
    Bound, IntoPyObject, Py, PyAny, PyResult, Python,
};
use serde::Serialize;
use serde_json::{json, Map, Number, Value};

use crate::{
    protocols::{
        chat::ChatCompletionRequest,
        completion::CompletionRequest,
        generate::GenerateRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
};

type RouterResult<T> = Result<T, Response>;

pub struct AtomStandaloneRouter {
    pub service: Py<PyAny>,
    close_service_on_shutdown: bool,
    closed: AtomicBool,
}

pub struct AtomStandaloneRuntime {
    pub service: Py<PyAny>,
    pub close_service_on_shutdown: bool,
}

impl AtomStandaloneRouter {
    pub fn from_runtime(runtime: &AtomStandaloneRuntime) -> Self {
        Python::attach(|py| Self {
            service: runtime.service.clone_ref(py),
            close_service_on_shutdown: runtime.close_service_on_shutdown,
            closed: AtomicBool::new(false),
        })
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

    fn run_chat_completion(&self, body: &ChatCompletionRequest) -> RouterResult<Value> {
        self.call_service("chat_completions", body, "chat completion")
    }

    fn run_completion(&self, body: &CompletionRequest) -> RouterResult<Value> {
        self.call_service("completions", body, "completion")
    }

    fn call_service<T: Serialize>(
        &self,
        method_name: &'static str,
        body: &T,
        endpoint: &'static str,
    ) -> RouterResult<Value> {
        let request_value = serde_json::to_value(body).map_err(|e| {
            Self::error_response(
                StatusCode::BAD_REQUEST,
                format!("Failed to serialize {endpoint} request: {e}"),
            )
        })?;

        Python::attach(|py| -> PyResult<Value> {
            let request = Self::json_to_py(py, &request_value)?;
            let response = self.service.bind(py).call_method1(method_name, (request,))?;
            Self::py_to_json(&response)
        })
        .map_err(|e| {
            Self::error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("ATOM standalone {endpoint} failed: {e}"),
            )
        })
    }

    fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
        match value {
            Value::Null => Ok(py.None()),
            Value::Bool(value) => Ok(value
                .into_pyobject(py)?
                .to_owned()
                .into_any()
                .unbind()),
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
                let items: PyResult<Vec<_>> =
                    values.iter().map(|value| Self::json_to_py(py, value)).collect();
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
        Python::attach(|py| {
            self.service
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
        if !self.close_service_on_shutdown {
            tracing::info!(
                "Skipping ATOM standalone Python service close because it is externally owned ({})",
                reason
            );
            return;
        }

        tracing::info!("Closing ATOM standalone Python service ({})", reason);
        Python::attach(|py| {
            if let Err(e) = self.service.bind(py).call_method0("close") {
                tracing::warn!("Failed to close ATOM standalone Python service: {}", e);
            }
        });
    }
}

impl std::fmt::Debug for AtomStandaloneRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomStandaloneRouter").finish_non_exhaustive()
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
        self.close_service("server shutdown");
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        Self::not_implemented("health_generate")
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
        (
            StatusCode::OK,
            Json(json!({
                "object": "list",
                "data": []
            })),
        )
            .into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        Self::not_implemented("get_model_info")
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        Self::not_implemented("generate")
    }

    async fn route_chat(
        &self,
        _headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        match self.run_chat_completion(body) {
            Ok(body) => (StatusCode::OK, Json(body)).into_response(),
            Err(response) => response,
        }
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
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
