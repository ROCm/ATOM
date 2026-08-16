use std::{
    sync::atomic::{AtomicUsize, Ordering},
    sync::Arc,
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::Deserialize;
use serde_json::json;

use crate::{
    app_context::AppContext,
    protocols::{
        chat::ChatCompletionRequest,
        completion::CompletionRequest,
        generate::GenerateRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        direct_engine::{
            into_token_handle, DirectEngineClient, DirectEngineSubmit, DirectSamplingParams,
        },
        grpc::completion_adapter::{
            completion_to_generate, wrap_generate_response_as_completion,
            wrap_streaming_generate_as_completion,
        },
        prepare::{
            generation_payload::GenerationPayload, prepare_chat, prepare_generate,
            response_context::ResponseContext,
        },
        render::{chat_aggregator, chat_streaming, generate_aggregator, generate_streaming},
        RouterTrait,
    },
};

const DEFAULT_MAX_TOKENS: i32 = 8192;

pub struct AtomStandaloneRuntime {
    pub engine_core_ipc_endpoints: Vec<EngineCoreIpcEndpoint>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EngineCoreIpcEndpoint {
    pub address: String,
    pub dp_rank: usize,
    pub pp_rank: usize,
    pub protocol_version: u32,
}

pub struct AtomStandaloneRouter {
    clients: Vec<(EngineCoreIpcEndpoint, DirectEngineClient)>,
    components: Arc<AppContext>,
    rank_cursor: AtomicUsize,
}

impl AtomStandaloneRouter {
    pub fn from_runtime(runtime: &AtomStandaloneRuntime, components: Arc<AppContext>) -> Self {
        Self {
            clients: runtime
                .engine_core_ipc_endpoints
                .iter()
                .cloned()
                .map(|endpoint| (endpoint.clone(), DirectEngineClient::new(endpoint)))
                .collect(),
            components,
            rank_cursor: AtomicUsize::new(0),
        }
    }

    fn unsupported(endpoint: &'static str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(json!({"error": {
                "type": "not_implemented",
                "message": format!("ATOM standalone direct IPC does not support {endpoint}")
            }})),
        )
            .into_response()
    }

    fn client(&self, requested_dp_rank: Option<usize>) -> Result<DirectEngineClient, Response> {
        if let Some(dp_rank) = requested_dp_rank {
            return self
                .clients
                .iter()
                .find(|(endpoint, _)| endpoint.dp_rank == dp_rank && endpoint.pp_rank == 0)
                .map(|(_, client)| client.clone())
                .ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        Json(json!({"error": {
                            "message": format!("unknown data_parallel_rank={dp_rank}")
                        }})),
                    )
                        .into_response()
                });
        }
        let count = self.clients.len();
        let start = self.rank_cursor.fetch_add(1, Ordering::Relaxed);
        (0..count)
            .map(|offset| &self.clients[(start + offset) % count])
            .find(|(endpoint, _)| endpoint.pp_rank == 0)
            .map(|(_, client)| client.clone())
            .ok_or_else(|| {
                (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({"error": {"message": "no direct EngineCore endpoint"}})),
                )
                    .into_response()
            })
    }

    async fn submit(
        &self,
        payload: GenerationPayload,
        context: ResponseContext,
        data_parallel_rank: Option<usize>,
    ) -> Result<
        (
            crate::routers::token_handle::token_handle::TokenHandle,
            ResponseContext,
        ),
        Response,
    > {
        let stop_token_sequences = payload
            .stop
            .stop_token_ids
            .as_ref()
            .map(|tokens| vec![tokens.clone()])
            .unwrap_or_default();
        let mut submit = DirectEngineSubmit::new(
            payload.request_id.clone(),
            payload.token_ids,
            DirectSamplingParams {
                temperature: payload.sampling.temperature,
                top_k: payload.sampling.top_k,
                top_p: payload.sampling.top_p,
                max_tokens: payload
                    .sampling
                    .max_new_tokens
                    .unwrap_or(DEFAULT_MAX_TOKENS)
                    .max(0) as u32,
                ignore_eos: payload.sampling.ignore_eos,
                n: payload.sampling.n.max(1) as u32,
                stop_strings: None,
            },
            stop_token_sequences,
        );
        submit.data_parallel_rank = data_parallel_rank;
        let client = self.client(submit.data_parallel_rank)?;
        let stream = client.submit(&submit).await.map_err(|error| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": {"message": error.to_string()}})),
            )
                .into_response()
        })?;
        Ok((into_token_handle(stream), context))
    }

    fn normalize_dp_rank(raw: Option<i32>) -> Result<Option<usize>, Response> {
        raw.map(|rank| {
            usize::try_from(rank).map_err(|_| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": {
                        "message": "data_parallel_rank must be a non-negative integer"
                    }})),
                )
                    .into_response()
            })
        })
        .transpose()
    }

    fn chat_dp_rank(headers: Option<&HeaderMap>) -> Result<Option<usize>, Response> {
        let Some(value) = headers.and_then(|headers| headers.get("x-atom-dp-rank")) else {
            return Ok(None);
        };
        value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(Some)
            .ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": {
                        "message": "x-atom-dp-rank must be a non-negative integer"
                    }})),
                )
                    .into_response()
            })
    }

    async fn chat(&self, body: &ChatCompletionRequest, dp_rank: Option<usize>) -> Response {
        if body.n.unwrap_or(1) != 1 {
            return Self::unsupported("chat completions with n > 1");
        }
        let (payload, context) = match prepare_chat(
            Arc::new(body.clone()),
            None,
            Some(body.model.clone()),
            self.components.as_ref(),
        ) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let (stream, context) = match self.submit(payload, context, dp_rank).await {
            Ok(value) => value,
            Err(response) => return response,
        };
        if body.stream {
            chat_streaming::process(stream, context, "atom_direct")
        } else {
            chat_aggregator::process(stream, context).await
        }
    }

    async fn generate(&self, body: &GenerateRequest) -> Response {
        if body
            .sampling_params
            .as_ref()
            .and_then(|params| params.n)
            .unwrap_or(1)
            != 1
        {
            return Self::unsupported("generate with n > 1");
        }
        let (payload, context) = match prepare_generate(
            Arc::new(body.clone()),
            None,
            body.model.clone(),
            self.components.as_ref(),
        ) {
            Ok(value) => value,
            Err(response) => return response,
        };
        let dp_rank = match Self::normalize_dp_rank(body.data_parallel_rank) {
            Ok(rank) => rank,
            Err(response) => return response,
        };
        let (stream, context) = match self.submit(payload, context, dp_rank).await {
            Ok(value) => value,
            Err(response) => return response,
        };
        if body.stream {
            generate_streaming::process(stream, context, "atom_direct")
        } else {
            generate_aggregator::process(stream, context).await
        }
    }

    async fn completion(&self, body: &CompletionRequest) -> Response {
        let generate = match completion_to_generate(body) {
            Ok(request) => request,
            Err(message) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": {"message": message}})),
                )
                    .into_response()
            }
        };
        let response = self.generate(&generate).await;
        if body.stream {
            wrap_streaming_generate_as_completion(response, body.model.clone()).await
        } else {
            wrap_generate_response_as_completion(response, body.model.clone()).await
        }
    }
}

impl std::fmt::Debug for AtomStandaloneRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomStandaloneRouter")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl RouterTrait for AtomStandaloneRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        Self::unsupported("generate")
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        (
            StatusCode::OK,
            Json(json!({
                "router_type": self.router_type(),
                "direct_engine_ipc": true,
                "dp_endpoints": self.clients.len(),
            })),
        )
            .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        (StatusCode::OK, Json(json!({"object": "list", "data": []}))).into_response()
    }

    async fn get_model_info(&self, _req: Request<Body>) -> Response {
        Self::unsupported("model_info")
    }

    async fn route_generate(
        &self,
        _headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        _model_id: Option<&str>,
    ) -> Response {
        self.generate(body).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        match Self::chat_dp_rank(headers) {
            Ok(dp_rank) => self.chat(body, dp_rank).await,
            Err(response) => response,
        }
    }

    async fn route_completion(
        &self,
        _headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        _model_id: Option<&str>,
    ) -> Response {
        self.completion(body).await
    }

    async fn route_responses(
        &self,
        _headers: Option<&HeaderMap>,
        _body: &ResponsesRequest,
        _model_id: Option<&str>,
    ) -> Response {
        Self::unsupported("responses")
    }

    async fn get_response(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        Self::unsupported("responses_get")
    }

    async fn cancel_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        Self::unsupported("responses_cancel")
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        Self::unsupported("responses_delete")
    }

    async fn list_response_input_items(
        &self,
        _headers: Option<&HeaderMap>,
        _response_id: &str,
    ) -> Response {
        Self::unsupported("responses_input_items")
    }

    fn router_type(&self) -> &'static str {
        "atom_standalone"
    }

    fn is_pd_mode(&self) -> bool {
        false
    }
}
