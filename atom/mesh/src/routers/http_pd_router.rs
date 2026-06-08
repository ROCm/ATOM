use std::{collections::HashMap, sync::Arc, time::Instant};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error, info, warn};

use crate::{
    config::types::{BackendType, RetryConfig},
    core::{
        is_retryable_status,
        placement::{
            backend::{
                atom::{AtomAdapter, AtomPrefillInfo},
                sglang::SglangAdapter,
                vllm::{VllmAdapter, VllmPrefillInfo},
                BackendAdapter, PairCtx,
            },
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
            traits::PdPlanner,
            types::{PlacementPlan, Protocol, RequestDescriptor},
        },
        RetryExecutor, Worker, WorkerLoadGuard, WorkerRegistry, UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{bool_to_static_str, metrics_labels, MeshMetrics},
    },
    policies::PolicyRegistry,
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::{InputIds, StringOrArray},
        completion::CompletionRequest,
        generate::GenerateRequest,
    },
    routers::{
        comm::{
            error, header_utils,
            metrics_utils::{error_type_from_status, route_to_endpoint},
            placement_response::placement_err_to_response,
        },
        RouterTrait,
    },
};

/// Construct a full API URL from a base URL and path.
fn api_path(url: &str, api_path: &str) -> String {
    if api_path.starts_with('/') {
        format!("{}{}", url, api_path)
    } else {
        format!("{}/{}", url, api_path)
    }
}

pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub backend: BackendType,
    pub planner: Arc<dyn PdPlanner>,
    pub adapter: Arc<dyn BackendAdapter>,
    /// Set when backend == Atom. enrich_decode_kv is ATOM-specific and not on the trait.
    atom_adapter: Option<Arc<AtomAdapter>>,
}

impl std::fmt::Debug for PDRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PDRouter")
            .field("worker_registry", &self.worker_registry)
            .field("client", &self.client)
            .field("retry_config", &self.retry_config)
            .field("backend", &self.backend)
            .finish()
    }
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<String>,
    model_id: Option<&'a str>,
    headers: Option<Arc<HeaderMap>>,
}

impl PDRouter {
    async fn proxy_to_first_prefill_worker(
        &self,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let workers = self.worker_registry.get_prefill_workers();
        let first_worker_url = workers.first().map(|w| w.url().to_string());

        if let Some(worker_url) = first_worker_url {
            self.proxy_to_worker(worker_url, endpoint, headers).await
        } else {
            error::service_unavailable("no_prefill_servers", "No prefill servers available")
        }
    }

    async fn proxy_to_worker(
        &self,
        worker_url: String,
        endpoint: &str,
        headers: Option<Vec<(String, String)>>,
    ) -> Response {
        let url = format!("{}/{}", worker_url, endpoint);
        let mut request_builder = self.client.get(&url);

        if let Some(headers) = headers {
            for (name, value) in headers {
                request_builder = request_builder.header(name, value);
            }
        }

        match request_builder.send().await {
            Ok(res) if res.status().is_success() => {
                let response_headers = header_utils::preserve_response_headers(res.headers());

                match res.bytes().await {
                    Ok(body) => {
                        let mut response = Response::new(Body::from(body));
                        *response.status_mut() = StatusCode::OK;
                        *response.headers_mut() = response_headers;
                        response
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        error::internal_error(
                            "read_response_body_failed",
                            format!("Failed to read response body: {}", e),
                        )
                    }
                }
            }
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                error::create_error(
                    status,
                    "server_error",
                    format!("Server returned status: {}", res.status()),
                )
            }
            Err(e) => {
                error!("Failed to proxy request server: {}", e);
                error::internal_error(
                    "proxy_request_failed",
                    format!("Failed to proxy request: {}", e),
                )
            }
        }
    }

    pub async fn new(ctx: &Arc<crate::app_context::AppContext>) -> Result<Self, String> {
        let backend = ctx.router_config.backend;
        let worker_registry = Arc::clone(&ctx.worker_registry);
        let policy_registry = Arc::clone(&ctx.policy_registry);
        let client = ctx.client.clone();

        let mut atom_adapter: Option<Arc<AtomAdapter>> = None;
        let adapter: Arc<dyn BackendAdapter> = match backend {
            BackendType::Sglang => Arc::new(SglangAdapter),
            BackendType::Vllm => {
                let info =
                    Arc::new(Self::fetch_vllm_prefill_info(&worker_registry, &client).await?);
                Arc::new(VllmAdapter::new(info))
            }
            BackendType::Atom => {
                let info =
                    Arc::new(Self::fetch_atom_prefill_info(&worker_registry, &client).await?);
                let a = Arc::new(AtomAdapter::new(info));
                atom_adapter = Some(a.clone());
                a
            }
        };

        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry.clone())),
            Arc::new(PolicyRegistryAdapter::new(policy_registry.clone())),
        ));

        Ok(PDRouter {
            worker_registry,
            policy_registry,
            client,
            retry_config: ctx.router_config.effective_retry_config(),
            backend,
            planner,
            adapter,
            atom_adapter,
        })
    }

    async fn fetch_vllm_prefill_info(
        worker_registry: &WorkerRegistry,
        client: &Client,
    ) -> Result<VllmPrefillInfo, String> {
        let prefill_workers = worker_registry.get_prefill_workers();
        if prefill_workers.is_empty() {
            return Err(
                "vLLM PD mode requires at least one prefill worker, but none were registered"
                    .to_string(),
            );
        }

        let mut bootstrap_addrs = HashMap::new();
        let mut engine_ids = HashMap::new();

        for worker in &prefill_workers {
            let worker_url = worker.url().to_string();
            let parsed = url::Url::parse(&worker_url)
                .map_err(|e| format!("Invalid prefill URL {}: {}", worker_url, e))?;
            let host = parsed
                .host_str()
                .ok_or_else(|| format!("No host in prefill URL {}", worker_url))?
                .to_string();
            let port = worker.bootstrap_port().unwrap_or(8998);
            let bootstrap_addr = format!("http://{}:{}", host, port);

            info!("Querying vLLM prefill bootstrap: {}/query", bootstrap_addr);

            let resp = client
                .get(format!("{}/query", bootstrap_addr))
                .send()
                .await
                .map_err(|e| {
                    format!(
                        "Failed to query vLLM bootstrap at {}/query: {}",
                        bootstrap_addr, e
                    )
                })?;

            if !resp.status().is_success() {
                return Err(format!(
                    "vLLM bootstrap {}/query returned status {}",
                    bootstrap_addr,
                    resp.status()
                ));
            }

            let data: HashMap<String, Value> = resp.json().await.map_err(|e| {
                format!(
                    "Failed to parse vLLM bootstrap response from {}: {}",
                    bootstrap_addr, e
                )
            })?;

            let mut rank_map = HashMap::new();
            for (rank_str, entry) in &data {
                let rank: usize = rank_str.parse().map_err(|e| {
                    format!(
                        "Invalid dp_rank '{}' from {}: {}",
                        rank_str, bootstrap_addr, e
                    )
                })?;
                let eid = entry
                    .get("engine_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        format!(
                            "Missing engine_id for rank {} from {}",
                            rank, bootstrap_addr
                        )
                    })?
                    .to_string();
                rank_map.insert(rank, eid);
            }

            if rank_map.is_empty() {
                return Err(format!(
                    "vLLM bootstrap {}/query returned empty engine_id map",
                    bootstrap_addr
                ));
            }

            info!(
                "vLLM prefill {} bootstrap_addr={} engine_ids={:?}",
                worker_url, bootstrap_addr, rank_map
            );

            bootstrap_addrs.insert(worker_url.clone(), bootstrap_addr);
            engine_ids.insert(worker_url, rank_map);
        }

        Ok(VllmPrefillInfo {
            bootstrap_addrs,
            engine_ids,
        })
    }

    async fn fetch_atom_prefill_info(
        worker_registry: &WorkerRegistry,
        client: &Client,
    ) -> Result<AtomPrefillInfo, String> {
        let prefill_workers = worker_registry.get_prefill_workers();
        if prefill_workers.is_empty() {
            return Err("ATOM PD mode requires at least one prefill worker".to_string());
        }

        let mut tp_sizes = HashMap::new();
        for worker in &prefill_workers {
            let worker_url = worker.url().to_string();
            let info_url = format!("{}/kv_transfer_info", worker_url);

            info!("Querying ATOM prefill kv_transfer_info: {}", info_url);
            let resp = client
                .get(&info_url)
                .send()
                .await
                .map_err(|e| format!("GET {} failed: {}", info_url, e))?;
            if !resp.status().is_success() {
                return Err(format!("{} returned {}", info_url, resp.status()));
            }
            let data: Value = resp
                .json()
                .await
                .map_err(|e| format!("Parse {} response: {}", info_url, e))?;

            let tp = data
                .get("tp_size")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| format!("Missing tp_size in {} response", info_url))?
                as usize;
            let kv_role = data.get("kv_role").and_then(|v| v.as_str());
            if kv_role != Some("kv_producer") {
                return Err(format!(
                    "{} is not a prefill (kv_role={:?}, expected kv_producer)",
                    worker_url, kv_role
                ));
            }
            info!("ATOM prefill {} tp_size={}", worker_url, tp);
            tp_sizes.insert(worker_url, tp);
        }
        Ok(AtomPrefillInfo { tp_sizes })
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    fn get_generate_batch_size(req: &GenerateRequest) -> Option<usize> {
        // GenerateRequest doesn't support batch via arrays, only via input_ids
        if let Some(InputIds::Batch(batches)) = &req.input_ids {
            if !batches.is_empty() {
                return Some(batches.len());
            }
        }
        None
    }

    fn get_chat_batch_size(req: &ChatCompletionRequest) -> Option<usize> {
        if let Some(n) = req.n {
            if n > 1 {
                return Some(n as usize);
            }
        }
        None
    }

    fn get_completion_batch_size(req: &CompletionRequest) -> Option<usize> {
        if let StringOrArray::Array(arr) = &req.prompt {
            if !arr.is_empty() {
                return Some(arr.len());
            }
        }
        None
    }

    /// Dispatch a request based on backend type. SGLang uses dual-dispatch+bootstrap;
    /// vLLM uses Mooncake fire-and-forget P + streamed D with kv_transfer_params.
    async fn dispatch_pd<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        match self.backend {
            BackendType::Sglang => {
                self.execute_dual_dispatch(headers, original_request, context)
                    .await
            }
            BackendType::Vllm => {
                self.execute_vllm_mooncake(headers, original_request, context)
                    .await
            }
            BackendType::Atom => {
                self.execute_atom_relay(headers, original_request, context)
                    .await
            }
        }
    }

    async fn plan_pd_pair(
        &self,
        context: &PDRequestContext<'_>,
    ) -> Result<(Arc<dyn Worker>, Arc<dyn Worker>, PairCtx), Response> {
        let descriptor = RequestDescriptor {
            model_id: context.model_id,
            protocol: Some(Protocol::Http),
            text: context.request_text.as_deref(),
            tokens: None,
            headers: context.headers.as_deref(),
            stream: context.is_stream,
        };

        let (prefill, decode, prefill_policy, decode_policy) =
            match self.planner.plan(&descriptor).await {
                Ok(PlacementPlan::Pair {
                    prefill,
                    decode,
                    prefill_policy,
                    decode_policy,
                    ..
                }) => (prefill, decode, prefill_policy, decode_policy),
                Ok(PlacementPlan::Single { .. }) => {
                    return Err(error::internal_error(
                        "unexpected_single_plan",
                        "Planner returned Single plan for PD router",
                    ));
                }
                Err(err) => return Err(placement_err_to_response(err, context.model_id)),
            };

        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        MeshMetrics::record_worker_selection(
            metrics_labels::WORKER_PREFILL,
            metrics_labels::CONNECTION_HTTP,
            model,
            prefill_policy,
        );
        MeshMetrics::record_worker_selection(
            metrics_labels::WORKER_DECODE,
            metrics_labels::CONNECTION_HTTP,
            model,
            decode_policy,
        );

        let ctx = self
            .adapter
            .prepare_pair(prefill.as_ref(), decode.as_ref())
            .map_err(Self::handle_serialization_error)?;
        Ok((prefill, decode, ctx))
    }

    /// vLLM Mooncake mode: fire prefill request as background task, stream decode response.
    /// Replaces the dual-dispatch+bootstrap protocol used for SGLang.
    async fn execute_vllm_mooncake<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );

        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode, ctx) = match self.plan_pd_pair(&context).await {
                            Ok(t) => t,
                            Err(resp) => return resp,
                        };

                        debug!(
                            "vLLM PD retry attempt {} prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        let mut prefill_request_json =
                            match serde_json::to_value(shared_request.as_ref()) {
                                Ok(v) => v,
                                Err(e) => return Self::handle_serialization_error(e),
                            };
                        let mut decode_request_json = prefill_request_json.clone();
                        if let Err(e) = self
                            .adapter
                            .inject_prefill_fields(&mut prefill_request_json, &ctx)
                        {
                            return Self::handle_serialization_error(e);
                        }
                        if let Err(e) = self
                            .adapter
                            .inject_decode_fields(&mut decode_request_json, &ctx)
                        {
                            return Self::handle_serialization_error(e);
                        }
                        let correlation_id = self.adapter.correlation_id(&ctx);

                        self.dispatch_vllm_mooncake_internal(
                            headers,
                            prefill_request_json,
                            decode_request_json,
                            context,
                            Arc::clone(&prefill),
                            Arc::clone(&decode),
                            start_time,
                            correlation_id,
                        )
                        .await
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    endpoint,
                );
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    endpoint,
                );
            },
        )
        .await;

        let duration = start_time.elapsed();
        if response.status().is_success() {
            MeshMetrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            MeshMetrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    /// Core vLLM Mooncake dispatch: fire P as background task, stream D response back to client.
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_vllm_mooncake_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_request_json: Value,
        decode_request_json: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        _start_time: Instant,
        correlation_id: Option<String>,
    ) -> Response {
        // Load tracking: streaming uses guards inside create_streaming_response.
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        // P request: fire-and-forget background task. Mooncake coordinates KV transfer
        // via its own out-of-band channel; we only need to ensure P starts processing.
        let prefill_post = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &prefill_request_json,
            headers,
            false,
        );
        let prefill_url_for_log = prefill.url().to_string();
        let prefill_for_outcome = prefill.clone();
        let correlation_for_log = correlation_id.unwrap_or_else(|| "unknown".to_string());
        tokio::spawn(async move {
            match prefill_post.send().await {
                Ok(res) => {
                    let status = res.status();
                    if status.is_success() {
                        debug!(
                            "vLLM prefill {} request_id={} status={}",
                            prefill_url_for_log, correlation_for_log, status
                        );
                    } else {
                        warn!(
                            "vLLM prefill {} request_id={} returned non-success status={}",
                            prefill_url_for_log, correlation_for_log, status
                        );
                    }
                    // Drain body so the connection can be reused.
                    let _ = res.bytes().await;
                    prefill_for_outcome.record_outcome(status.is_success());
                }
                Err(e) => {
                    error!(
                        "vLLM prefill {} request_id={} failed: {}",
                        prefill_url_for_log, correlation_for_log, e
                    );
                    prefill_for_outcome.record_outcome(false);
                }
            }
        });

        // D request: client sees the streamed (or buffered) response from D.
        let decode_post = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &decode_request_json,
            headers,
            false,
        );
        let decode_result = decode_post.send().await;
        events::RequestReceivedEvent {}.emit();

        let res = match decode_result {
            Ok(r) => r,
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    error_debug = ?e,
                    "vLLM decode request failed"
                );
                decode.record_outcome(false);
                return error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let not_error = status.is_success() || status.is_client_error();
        decode.record_outcome(not_error);

        if !status.is_success() {
            error!(
                "vLLM decode {} returned error status={}",
                decode.url(),
                status
            );
            MeshMetrics::record_worker_error(
                metrics_labels::WORKER_DECODE,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
            return self
                .handle_decode_error_response(res, &context, prefill, decode)
                .await;
        }

        if context.is_stream {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            self.create_streaming_response(
                res.bytes_stream(),
                status,
                None,
                false,
                None,
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            match res.bytes().await {
                Ok(decode_body) => {
                    let mut response = Response::new(Body::from(decode_body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    error!("Failed to read vLLM decode response: {}", e);
                    error::internal_error("read_response_failed", "Failed to read response")
                }
            }
        }
    }

    /// ATOM Mooncake mode: P must run first and return kv_transfer_params; mesh
    /// enriches them with remote_dp_size/remote_tp_size, then forwards to D.
    /// Decode's response is streamed (or buffered) back to the client.
    async fn execute_atom_relay<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );

        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode, ctx) = match self.plan_pd_pair(&context).await {
                            Ok(t) => t,
                            Err(resp) => return resp,
                        };

                        debug!(
                            "ATOM PD retry attempt {} prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        let mut prefill_request_json =
                            match serde_json::to_value(shared_request.as_ref()) {
                                Ok(v) => v,
                                Err(e) => return Self::handle_serialization_error(e),
                            };
                        let mut decode_request_json = prefill_request_json.clone();
                        if let Err(e) = self
                            .adapter
                            .inject_prefill_fields(&mut prefill_request_json, &ctx)
                        {
                            return Self::handle_serialization_error(e);
                        }
                        if let Err(e) = self
                            .adapter
                            .inject_decode_fields(&mut decode_request_json, &ctx)
                        {
                            return Self::handle_serialization_error(e);
                        }
                        let correlation_id = self.adapter.correlation_id(&ctx);

                        self.dispatch_atom_relay_internal(
                            headers,
                            prefill_request_json,
                            decode_request_json,
                            context,
                            Arc::clone(&prefill),
                            Arc::clone(&decode),
                            ctx,
                            start_time,
                            correlation_id,
                        )
                        .await
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    endpoint,
                );
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    endpoint,
                );
            },
        )
        .await;

        let duration = start_time.elapsed();
        if response.status().is_success() {
            MeshMetrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            MeshMetrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_atom_relay_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_request_json: Value,
        mut decode_request_json: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        ctx: PairCtx,
        _start_time: Instant,
        correlation_id: Option<String>,
    ) -> Response {
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let prefill_post = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &prefill_request_json,
            headers,
            false,
        );
        let correlation_for_log = correlation_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        let prefill_result = prefill_post.send().await;
        let prefill_resp = match prefill_result {
            Ok(r) => r,
            Err(e) => {
                error!(
                    "ATOM prefill {} request_id={} failed: {}",
                    prefill.url(),
                    correlation_for_log,
                    e
                );
                prefill.record_outcome(false);
                return error::bad_gateway(
                    "prefill_server_error",
                    format!("Prefill server error: {}", e),
                );
            }
        };

        let prefill_status = prefill_resp.status();
        prefill.record_outcome(prefill_status.is_success());
        if !prefill_status.is_success() {
            let body_text = prefill_resp
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable>".to_string());
            error!(
                "ATOM prefill {} request_id={} status={} body={}",
                prefill.url(),
                correlation_for_log,
                prefill_status,
                body_text
            );
            let code = StatusCode::from_u16(prefill_status.as_u16())
                .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
            return error::create_error(
                code,
                "prefill_error",
                format!("Prefill server error ({}): {}", prefill_status, body_text),
            );
        }

        let prefill_body: Value = match prefill_resp.json().await {
            Ok(v) => v,
            Err(e) => {
                error!(
                    "ATOM prefill {} request_id={} response parse failed: {}",
                    prefill.url(),
                    correlation_for_log,
                    e
                );
                return error::bad_gateway(
                    "prefill_parse_error",
                    format!("Prefill response parse error: {}", e),
                );
            }
        };

        let mut kv_params = match prefill_body.get("kv_transfer_params").cloned() {
            Some(v) => v,
            None => {
                error!(
                    "ATOM prefill {} request_id={} response missing kv_transfer_params",
                    prefill.url(),
                    correlation_for_log
                );
                return error::bad_gateway(
                    "prefill_missing_kv_transfer_params",
                    "Prefill response missing kv_transfer_params",
                );
            }
        };

        let atom_adapter = match self.atom_adapter.as_ref() {
            Some(a) => a,
            None => {
                error!("atom_adapter is None but backend == Atom — programming error");
                return error::internal_error(
                    "atom_adapter_missing",
                    "Internal: ATOM adapter not initialized",
                );
            }
        };
        if let Err(e) = atom_adapter.enrich_decode_kv(&mut kv_params, &ctx) {
            error!(
                "ATOM enrich_decode_kv failed for prefill {} request_id={}: {}",
                prefill.url(),
                correlation_for_log,
                e
            );
            return error::internal_error(
                "enrich_decode_kv_failed",
                format!("Failed to enrich decode kv: {}", e),
            );
        }

        let decode_obj = match decode_request_json.as_object_mut() {
            Some(o) => o,
            None => {
                return error::internal_error(
                    "decode_body_not_object",
                    "Decode request body must be a JSON object",
                );
            }
        };
        decode_obj.insert("kv_transfer_params".to_string(), kv_params);

        let decode_post = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &decode_request_json,
            headers,
            false,
        );
        let decode_result = decode_post.send().await;
        events::RequestReceivedEvent {}.emit();

        let res = match decode_result {
            Ok(r) => r,
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "ATOM decode request failed"
                );
                decode.record_outcome(false);
                return error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        let not_error = status.is_success() || status.is_client_error();
        decode.record_outcome(not_error);

        if !status.is_success() {
            error!(
                "ATOM decode {} returned error status={}",
                decode.url(),
                status
            );
            MeshMetrics::record_worker_error(
                metrics_labels::WORKER_DECODE,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
            return self
                .handle_decode_error_response(res, &context, prefill, decode)
                .await;
        }

        if context.is_stream {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            self.create_streaming_response(
                res.bytes_stream(),
                status,
                None,
                false,
                None,
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            match res.bytes().await {
                Ok(decode_body) => {
                    let mut response = Response::new(Body::from(decode_body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    error!("Failed to read ATOM decode response: {}", e);
                    error::internal_error("read_response_failed", "Failed to read response")
                }
            }
        }
    }

    async fn execute_dual_dispatch<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let start_time = Instant::now();

        let route = context.route;
        let model = context.model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        MeshMetrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(context.is_stream),
        );
        // Clone request once outside the retry loop, then use Arc to share across attempts
        // This avoids O(retries) clones by sharing the same data
        let shared_request = Arc::new(original_request.clone());
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            {
                move |attempt: u32| {
                    // Clone Arc (cheap reference count increment) instead of cloning the entire request
                    let shared_request = Arc::clone(&shared_request);
                    let context = context.clone();
                    async move {
                        let (prefill, decode, ctx) = match self.plan_pd_pair(&context).await {
                            Ok(t) => t,
                            Err(resp) => return resp,
                        };

                        debug!(
                            "PD retry attempt {} using prefill={} decode={}",
                            attempt,
                            prefill.url(),
                            decode.url()
                        );

                        let mut json_request = match serde_json::to_value(shared_request.as_ref()) {
                            Ok(v) => v,
                            Err(e) => return Self::handle_serialization_error(e),
                        };

                        let inject_result = match context.batch_size {
                            Some(n) => {
                                self.adapter
                                    .inject_batch_prefill_fields(&mut json_request, &ctx, n)
                            }
                            None => self.adapter.inject_prefill_fields(&mut json_request, &ctx),
                        };
                        if let Err(e) = inject_result {
                            return Self::handle_serialization_error(e);
                        }

                        let response = self
                            .execute_dual_dispatch_internal(
                                headers,
                                json_request,
                                context,
                                Arc::clone(&prefill),
                                Arc::clone(&decode),
                                start_time,
                            )
                            .await;

                        let status = response.status();
                        let not_error = status.is_success() || status.is_client_error();
                        prefill.record_outcome(not_error);
                        decode.record_outcome(not_error);

                        // Record worker errors for server errors (5xx)
                        if status.is_server_error() {
                            let error_type = error_type_from_status(status);
                            MeshMetrics::record_worker_error(
                                metrics_labels::WORKER_PREFILL,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                            MeshMetrics::record_worker_error(
                                metrics_labels::WORKER_DECODE,
                                metrics_labels::CONNECTION_HTTP,
                                error_type,
                            );
                        }

                        response
                    }
                }
            },
            |res, _attempt| is_retryable_status(res.status()),
            |delay, attempt| {
                // Layer 3 worker metrics (PD mode uses both prefill and decode workers)
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_PREFILL, endpoint);
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_DECODE, endpoint);
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_PREFILL,
                    endpoint,
                );
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_DECODE,
                    endpoint,
                );
            },
        )
        .await;

        // Record Layer 2 metrics
        let duration = start_time.elapsed();
        if response.status().is_success() {
            MeshMetrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            MeshMetrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_PD,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn handle_decode_error_response(
        &self,
        res: reqwest::Response,
        context: &PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        let status = res.status();

        if context.is_stream {
            // Handle streaming error response
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let error_payload = match res.bytes().await {
                Ok(error_body) => {
                    if let Ok(error_json) = serde_json::from_slice::<Value>(&error_body) {
                        json!({ "message": error_json, "status": status.as_u16() })
                    } else {
                        json!({ "message": String::from_utf8_lossy(&error_body).to_string(), "status": status.as_u16() })
                    }
                }
                Err(e) => {
                    json!({ "message": format!("Decode server error: {}", e), "status": status.as_u16() })
                }
            };

            let sse_data = format!(
                "data: {{'error': {}}}",
                serde_json::to_string(&error_payload).unwrap_or_default()
            );
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                prefill,
                decode,
            )
        } else {
            // Handle non-streaming error response
            match res.bytes().await {
                Ok(error_body) => {
                    // Try to parse error message from body, fallback to status-based error
                    let error_message = if let Ok(error_json) =
                        serde_json::from_slice::<Value>(&error_body)
                    {
                        if let Some(msg) = error_json
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else if let Some(msg) = error_json.get("message").and_then(|m| m.as_str())
                        {
                            msg.to_string()
                        } else {
                            String::from_utf8_lossy(&error_body).to_string()
                        }
                    } else {
                        String::from_utf8_lossy(&error_body).to_string()
                    };

                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    error::create_error(status_code, "decode_error", error_message)
                }
                Err(e) => {
                    let error_message = format!("Decode server error: {}", e);
                    let status_code = StatusCode::from_u16(status.as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    error::create_error(status_code, "decode_read_failed", error_message)
                }
            }
        }
    }

    // Internal method that performs the actual dual dispatch (without retry logic)
    async fn execute_dual_dispatch_internal(
        &self,
        headers: Option<&HeaderMap>,
        json_request: Value,
        context: PDRequestContext<'_>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        _start_time: Instant,
    ) -> Response {
        // For non-streaming: use guard for automatic load management
        // For streaming: load will be managed in create_streaming_response
        let _prefill_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(prefill.clone(), headers));
        let _decode_guard =
            (!context.is_stream).then(|| WorkerLoadGuard::new(decode.clone(), headers));

        // Build both requests
        let prefill_request = self.build_post_with_headers(
            &self.client,
            prefill.url(),
            context.route,
            &json_request,
            headers,
            false,
        );
        let decode_request = self.build_post_with_headers(
            &self.client,
            decode.url(),
            context.route,
            &json_request,
            headers,
            false,
        );

        // Send both requests concurrently and wait for both
        // Note: Using borrowed references avoids heap allocation
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let (prefill_result, decode_result) =
            tokio::join!(prefill_request.send(), decode_request.send());

        events::RequestReceivedEvent {}.emit();

        // Process decode response
        match decode_result {
            Ok(res) => {
                let status = StatusCode::from_u16(res.status().as_u16())
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                debug!("Decode response status: {}", status);

                if !status.is_success() {
                    error!(
                        "Decode server returned error status decode_url={} status={}",
                        decode.url(),
                        status
                    );

                    return self
                        .handle_decode_error_response(res, &context, prefill, decode)
                        .await;
                }

                // Process prefill response
                let prefill_body = if context.return_logprob {
                    match self
                        .process_prefill_response(
                            prefill_result,
                            prefill.url(),
                            context.return_logprob,
                        )
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                } else {
                    // Even if we don't need logprobs, we should check prefill status
                    match self
                        .process_prefill_response(prefill_result, prefill.url(), false)
                        .await
                    {
                        Ok((_, body)) => body,
                        Err(error_response) => return error_response,
                    }
                };

                if context.is_stream {
                    // Streaming response
                    let prefill_logprobs = if context.return_logprob {
                        prefill_body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .and_then(|json| {
                                json.pointer("/meta_info/input_token_logprobs").cloned()
                            })
                    } else {
                        None
                    };

                    let response_headers = header_utils::preserve_response_headers(res.headers());

                    self.create_streaming_response(
                        res.bytes_stream(),
                        status,
                        prefill_logprobs,
                        context.return_logprob,
                        None,
                        Some(response_headers),
                        prefill,
                        decode,
                    )
                } else {
                    // Non-streaming response
                    if context.return_logprob {
                        self.process_non_streaming_response(
                            res,
                            status,
                            context.return_logprob,
                            prefill_body,
                        )
                        .await
                    } else {
                        // Direct passthrough when no logprobs needed
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(decode_body) => {
                                let mut response = Response::new(Body::from(decode_body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => {
                                error!("Failed to read decode response: {}", e);
                                error::internal_error(
                                    "read_response_failed",
                                    "Failed to read response",
                                )
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    error_debug = ?e,
                    "Decode request failed"
                );
                error::bad_gateway("decode_server_error", format!("Decode server error: {}", e))
            }
        }
    }

    fn policies_need_request_text(&self) -> bool {
        let prefill_policy = self.policy_registry.get_prefill_policy();
        let decode_policy = self.policy_registry.get_decode_policy();
        prefill_policy.needs_request_text() || decode_policy.needs_request_text()
    }

    #[allow(clippy::too_many_arguments)]
    fn create_streaming_response(
        &self,
        stream: impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
        status: StatusCode,
        prefill_logprobs: Option<Value>,
        return_logprob: bool,
        decode_url: Option<String>,
        headers: Option<HeaderMap>,
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
    ) -> Response {
        use crate::core::AttachedBody;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            futures_util::pin_mut!(stream);
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let is_done = memmem::find(&chunk, b"data: [DONE]").is_some();

                        let result = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                .unwrap_or(chunk)
                        } else {
                            chunk
                        };

                        if tx.send(Ok(result)).is_err() {
                            break;
                        }

                        if is_done {
                            break;
                        }
                    }
                    Err(e) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, e);
                        }
                        let _ = tx.send(Err(format!("Stream error: {}", e)));
                        break;
                    }
                }
            }
        });

        let stream = UnboundedReceiverStream::new(rx);
        let body = Body::from_stream(stream);

        let guards = vec![
            WorkerLoadGuard::new(prefill, headers.as_ref()),
            WorkerLoadGuard::new(decode, headers.as_ref()),
        ];

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        AttachedBody::wrap_response(response, guards)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: reqwest::Response,
        status: StatusCode,
        return_logprob: bool,
        prefill_body: Option<bytes::Bytes>,
    ) -> Response {
        let response = res.bytes().await;
        let decode_body = match response {
            Ok(decode_body) => decode_body,
            Err(e) => {
                error!("Failed to read decode response: {}", e);
                return error::internal_error("read_response_failed", "Failed to read response");
            }
        };

        if !return_logprob {
            return (status, decode_body).into_response();
        }

        let Some(prefill_body) = prefill_body else {
            return (status, decode_body).into_response();
        };

        // Merge logprobs from prefill and decode
        let (Ok(prefill_json), Ok(mut decode_json)) = (
            serde_json::from_slice::<Value>(&prefill_body),
            serde_json::from_slice::<Value>(&decode_body),
        ) else {
            warn!("Failed to parse responses for logprob merging");
            return (status, decode_body).into_response();
        };

        Self::merge_logprobs_in_json(&prefill_json, &mut decode_json);

        // Return merged response
        match serde_json::to_vec(&decode_json) {
            Ok(body) => (status, body).into_response(),
            Err(e) => {
                error!("Failed to serialize merged response: {}", e);
                (status, decode_body).into_response()
            }
        }
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<reqwest::Response, reqwest::Error>,
        prefill_url: &str,
        return_logprob: bool,
    ) -> Result<(StatusCode, Option<bytes::Bytes>), Response> {
        // Check prefill result first - it's critical for disaggregated mode
        let prefill_response = match prefill_result {
            Ok(response) => response,
            Err(e) => {
                error!(
                    "Prefill server failed (CRITICAL) prefill_url={} error={}. Decode will timeout without prefill KV cache.",
                    prefill_url,
                    e
                );

                // Return error immediately - don't wait for decode to timeout
                return Err(error::bad_gateway(
                    "prefill_server_error",
                    format!(
                        "Prefill server error: {}. This will cause decode timeout.",
                        e
                    ),
                ));
            }
        };

        let prefill_status = StatusCode::from_u16(prefill_response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        // Check if prefill succeeded
        if !prefill_status.is_success() {
            // Get error body from prefill
            let error_msg = prefill_response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown prefill error".to_string());

            error!(
                "Prefill server returned error status prefill_url={} status={} body={}",
                prefill_url, prefill_status, error_msg
            );

            let error_response = error::create_error(
                prefill_status,
                "prefill_error",
                format!("Prefill server error ({}): {}", prefill_status, error_msg),
            );
            return Err(error_response);
        }

        // Read prefill body if needed for logprob merging
        let prefill_body = if return_logprob {
            match prefill_response.bytes().await {
                Ok(body) => Some(body),
                Err(e) => {
                    warn!("Failed to read prefill response body for logprobs: {}", e);
                    None
                }
            }
        } else {
            // For non-logprob requests, just consume the response without storing
            debug!("Consuming prefill response body (non-logprob request)");
            match prefill_response.bytes().await {
                Ok(_) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    fn build_post_with_headers(
        &self,
        client: &Client,
        url: &str,
        route: &'static str,
        json_request: &Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> reqwest::RequestBuilder {
        let mut request = client.post(api_path(url, route)).json(json_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        request
    }

    // Helper to merge logprobs from prefill and decode responses
    // Optimized to avoid double cloning by taking ownership of decode array
    fn merge_logprobs_in_json(prefill_json: &Value, decode_json: &mut Value) -> bool {
        if let (Some(prefill_meta), Some(decode_meta)) = (
            prefill_json.get("meta_info"),
            decode_json.get_mut("meta_info"),
        ) {
            if let (Some(prefill_logprobs), Some(decode_logprobs)) = (
                prefill_meta.get("input_token_logprobs"),
                decode_meta.get_mut("input_token_logprobs"),
            ) {
                if let Some(prefill_arr) = prefill_logprobs.as_array() {
                    // Take ownership of decode array to avoid cloning it
                    let decode_arr = std::mem::take(decode_logprobs);
                    if let Value::Array(decode_vec) = decode_arr {
                        // Pre-allocate merged array with exact capacity
                        let mut merged = Vec::with_capacity(prefill_arr.len() + decode_vec.len());
                        merged.extend(prefill_arr.iter().cloned());
                        merged.extend(decode_vec);
                        decode_meta["input_token_logprobs"] = Value::Array(merged);
                        return true;
                    }
                }
            }
        }
        false
    }

    // Simple helper to merge logprobs in streaming responses
    // Optimized to reduce allocations in the merge path
    fn merge_streaming_logprobs(
        prefill_logprobs: Option<Value>,
        decode_chunk: &[u8],
    ) -> Result<bytes::Bytes, ()> {
        // Skip non-data chunks
        let chunk_str = std::str::from_utf8(decode_chunk).map_err(|_| ())?;
        if !chunk_str.starts_with("data: ") || chunk_str.contains("[DONE]") {
            return Err(());
        }

        // Parse JSON from chunk
        let json_str = chunk_str.trim_start_matches("data: ").trim();
        let mut decode_json: Value = serde_json::from_str(json_str).map_err(|_| ())?;

        // Merge prefill logprobs if available
        if let Some(ref p_logprobs) = prefill_logprobs {
            if let Some(meta) = decode_json.get_mut("meta_info") {
                if let Some(d_logprobs) = meta.get_mut("input_token_logprobs") {
                    if let Some(p_arr) = p_logprobs.as_array() {
                        // Take ownership of decode array to avoid cloning it
                        let decode_arr = std::mem::take(d_logprobs);
                        if let Value::Array(d_vec) = decode_arr {
                            // Pre-allocate merged array with exact capacity
                            let mut merged = Vec::with_capacity(p_arr.len() + d_vec.len());
                            merged.extend(p_arr.iter().cloned());
                            merged.extend(d_vec);
                            *d_logprobs = Value::Array(merged);
                        }
                    }
                }
            }
        }

        // Re-serialize
        let merged_str = format!(
            "data: {}\n\n",
            serde_json::to_string(&decode_json).unwrap_or_default()
        );
        Ok(bytes::Bytes::from(merged_str))
    }
}

#[async_trait]
impl RouterTrait for PDRouter {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Note: This endpoint actually causes the model to generate tokens, so we only test one pair

        let descriptor = RequestDescriptor {
            protocol: Some(Protocol::Http),
            ..Default::default()
        };
        let (prefill, decode) = match self.planner.plan(&descriptor).await {
            Ok(PlacementPlan::Pair {
                prefill, decode, ..
            }) => (prefill, decode),
            Ok(PlacementPlan::Single { .. }) => {
                return error::internal_error(
                    "unexpected_single_plan",
                    "Planner returned Single plan for PD router",
                );
            }
            Err(err) => return placement_err_to_response(err, None),
        };

        let prefill_url = format!("{}/health_generate", prefill.url());
        let (prefill_result, decode_result) = tokio::join!(
            self.client.get(&prefill_url).send(),
            self.client
                .get(format!("{}/health_generate", decode.url()))
                .send()
        );

        // Check results
        let mut errors = Vec::new();

        match prefill_result {
            Ok(res) if res.status().is_success() => {
                debug!(
                    "Health generate passed for prefill server: {}",
                    prefill.url()
                );
            }
            Ok(res) => {
                errors.push(format!(
                    "Prefill {} returned status {}",
                    prefill.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Prefill {} error: {}", prefill.url(), e));
            }
        }

        match decode_result {
            Ok(res) if res.status().is_success() => {
                debug!("Health generate passed for decode server: {}", decode.url());
            }
            Ok(res) => {
                errors.push(format!(
                    "Decode {} returned status {}",
                    decode.url(),
                    res.status()
                ));
            }
            Err(e) => {
                errors.push(format!("Decode {} error: {}", decode.url(), e));
            }
        }

        if errors.is_empty() {
            (
                StatusCode::OK,
                format!(
                    "Health generate passed on selected pair: prefill={}, decode={}",
                    prefill.url(),
                    decode.url()
                ),
            )
                .into_response()
        } else {
            error::service_unavailable(
                "health_generate_failed",
                format!("Health generate failed: {:?}", errors),
            )
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // Get info from the first decode server to match sglang's server info format
        // Note: We use decode workers for server info to match expected format
        self.proxy_to_first_prefill_worker("get_server_info", None)
            .await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("v1/models", Some(headers))
            .await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Extract headers first to avoid Send issues
        let headers = header_utils::copy_request_headers(&req);

        // Proxy to first prefill worker
        self.proxy_to_first_prefill_worker("get_model_info", Some(headers))
            .await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.return_logprob.unwrap_or(false);

        let request_text = if self.policies_need_request_text() {
            body.text.as_deref().map(|s| s.to_string())
        } else {
            None
        };

        let batch_size = Self::get_generate_batch_size(body);

        let context = PDRequestContext {
            route: "/generate",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned().map(Arc::new),
        };

        self.dispatch_pd(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs;

        let request_text = if self.policies_need_request_text() {
            body.messages.first().and_then(|msg| match msg {
                ChatMessage::User { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::Developer { content, .. } => match content {
                    MessageContent::Text(text) => Some(text.clone()),
                    MessageContent::Parts(_) => None,
                },
                ChatMessage::System { content, .. } => Some(content.to_simple_string()),
                _ => None,
            })
        } else {
            None
        };

        // Calculate batch size
        let batch_size = Self::get_chat_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned().map(Arc::new),
        };

        self.dispatch_pd(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = body.stream;
        let return_logprob = body.logprobs.is_some();

        let request_text = if self.policies_need_request_text() {
            match &body.prompt {
                StringOrArray::String(s) => Some(s.clone()),
                StringOrArray::Array(v) => v.first().map(|s| s.to_string()),
            }
        } else {
            None
        };

        let batch_size = Self::get_completion_batch_size(body);

        let context = PDRequestContext {
            route: "/v1/completions",
            batch_size,
            is_stream,
            return_logprob,
            request_text,
            model_id,
            headers: headers.cloned().map(Arc::new),
        };

        self.dispatch_pd(headers, body, context).await
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{placement::backend::sglang::SglangAdapter, BasicWorkerBuilder, WorkerType};

    fn create_test_pd_router() -> PDRouter {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry =
            Arc::new(PolicyRegistry::new(crate::config::PolicyConfig::RoundRobin));

        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry.clone())),
            Arc::new(PolicyRegistryAdapter::new(policy_registry.clone())),
        ));
        let adapter: Arc<dyn BackendAdapter> = Arc::new(SglangAdapter);

        PDRouter {
            worker_registry,
            policy_registry,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            backend: BackendType::Sglang,
            planner,
            adapter,
            atom_adapter: None,
        }
    }

    fn create_test_worker(url: String, worker_type: WorkerType, healthy: bool) -> Box<dyn Worker> {
        let worker = BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .build();
        worker.set_healthy(healthy);
        Box::new(worker)
    }

    #[test]
    fn test_worker_load_metrics() {
        let prefill_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        ));
        let decode_worker: Arc<dyn Worker> = Arc::from(create_test_worker(
            "http://decode".to_string(),
            WorkerType::Decode,
            true,
        ));

        let _prefill_guard = WorkerLoadGuard::new(prefill_worker.clone(), None);
        let _decode_guard = WorkerLoadGuard::new(decode_worker.clone(), None);

        assert_eq!(prefill_worker.load(), 1);
        assert_eq!(decode_worker.load(), 1);

        drop(_prefill_guard);
        drop(_decode_guard);

        assert_eq!(prefill_worker.load(), 0);
        assert_eq!(decode_worker.load(), 0);
    }

    #[tokio::test]
    async fn test_streaming_load_tracking() {
        use futures_util::StreamExt;
        use tokio::time::{sleep, Duration};

        let router = create_test_pd_router();

        let prefill_worker = create_test_worker(
            "http://prefill".to_string(),
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_worker =
            create_test_worker("http://decode".to_string(), WorkerType::Decode, true);

        router.worker_registry.register(Arc::from(prefill_worker));
        router.worker_registry.register(Arc::from(decode_worker));

        let prefill_workers = router.worker_registry.get_prefill_workers();
        let decode_workers = router.worker_registry.get_decode_workers();

        let prefill_ref = prefill_workers[0].clone();
        let decode_ref = decode_workers[0].clone();

        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let stream = UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                None,
                prefill_ref.clone(),
                decode_ref.clone(),
            );

            // Guards are now attached to response body, so load should be 1
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 1);
            assert_eq!(decode_ref.load(), 1);

            drop(tx);

            // Response (and its body with guards) dropped here
            drop(response);
        }

        // Guards dropped when response dropped
        assert_eq!(prefill_ref.load(), 0);
        assert_eq!(decode_ref.load(), 0);
    }

    // --- get_chat_batch_size / get_generate_batch_size ---

    #[test]
    fn test_get_chat_batch_size_none() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}]}"#,
        )
        .unwrap();
        assert_eq!(PDRouter::get_chat_batch_size(&req), None);
    }

    #[test]
    fn test_get_chat_batch_size_n_1() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}], "n": 1}"#,
        )
        .unwrap();
        assert_eq!(PDRouter::get_chat_batch_size(&req), None);
    }

    #[test]
    fn test_get_chat_batch_size_n_4() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}], "n": 4}"#,
        )
        .unwrap();
        assert_eq!(PDRouter::get_chat_batch_size(&req), Some(4));
    }

    // --- merge_logprobs_in_json ---

    #[test]
    fn test_merge_logprobs_basic() {
        let prefill_json = json!({
            "meta_info": {
                "input_token_logprobs": [1.0, 2.0, 3.0]
            }
        });
        let mut decode_json = json!({
            "meta_info": {
                "input_token_logprobs": [4.0, 5.0]
            }
        });

        let result = PDRouter::merge_logprobs_in_json(&prefill_json, &mut decode_json);
        assert!(result);

        let merged = decode_json["meta_info"]["input_token_logprobs"]
            .as_array()
            .unwrap();
        assert_eq!(merged.len(), 5);
        assert_eq!(merged[0], 1.0);
        assert_eq!(merged[4], 5.0);
    }

    #[test]
    fn test_merge_logprobs_no_meta_info() {
        let prefill_json = json!({"text": "hello"});
        let mut decode_json = json!({"text": "world"});
        assert!(!PDRouter::merge_logprobs_in_json(
            &prefill_json,
            &mut decode_json
        ));
    }

    #[test]
    fn test_merge_logprobs_empty_prefill() {
        let prefill_json = json!({
            "meta_info": {
                "input_token_logprobs": []
            }
        });
        let mut decode_json = json!({
            "meta_info": {
                "input_token_logprobs": [1.0, 2.0]
            }
        });

        let result = PDRouter::merge_logprobs_in_json(&prefill_json, &mut decode_json);
        assert!(result);
        let merged = decode_json["meta_info"]["input_token_logprobs"]
            .as_array()
            .unwrap();
        assert_eq!(merged.len(), 2);
    }

    // --- merge_streaming_logprobs ---

    #[test]
    fn test_merge_streaming_logprobs_non_data_chunk() {
        let result = PDRouter::merge_streaming_logprobs(None, b"event: heartbeat\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_streaming_logprobs_done_chunk() {
        let result = PDRouter::merge_streaming_logprobs(None, b"data: [DONE]\n\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_merge_streaming_logprobs_no_prefill() {
        let chunk = b"data: {\"meta_info\":{\"input_token_logprobs\":[1.0]}}\n\n";
        let result = PDRouter::merge_streaming_logprobs(None, chunk);
        assert!(result.is_ok());
    }

    #[test]
    fn test_merge_streaming_logprobs_with_prefill() {
        let prefill_logprobs = json!([0.1, 0.2]);
        let chunk = b"data: {\"meta_info\":{\"input_token_logprobs\":[0.3]}}\n\n";
        let result = PDRouter::merge_streaming_logprobs(Some(prefill_logprobs), chunk);
        assert!(result.is_ok());
        let bytes = result.unwrap();
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.starts_with("data: "));
        let json_str = s.trim_start_matches("data: ").trim();
        let parsed: Value = serde_json::from_str(json_str).unwrap();
        let logprobs = parsed["meta_info"]["input_token_logprobs"]
            .as_array()
            .unwrap();
        assert_eq!(logprobs.len(), 3); // 2 prefill + 1 decode
    }

    // --- policies_need_request_text ---

    #[test]
    fn test_policies_need_request_text_default() {
        let router = create_test_pd_router();
        // Default RoundRobin doesn't need request text
        assert!(!router.policies_need_request_text());
    }

    #[test]
    fn test_policies_need_request_text_cache_aware() {
        let router = create_test_pd_router();
        router
            .policy_registry
            .set_prefill_policy(Arc::new(crate::policies::CacheAwarePolicy::new()));
        assert!(router.policies_need_request_text());
    }

    #[test]
    fn test_handle_serialization_error() {
        let response = PDRouter::handle_serialization_error("bad json");
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    }

    // --- router_type ---

    #[test]
    fn test_router_type() {
        let router = create_test_pd_router();
        assert_eq!(router.router_type(), "pd");
    }

    #[test]
    fn test_pd_request_context_headers_arc_shared_across_retries() {
        let mut headers = HeaderMap::new();
        headers.insert("x-trace", HeaderValue::from_static("abc"));
        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size: None,
            is_stream: false,
            return_logprob: false,
            request_text: None,
            model_id: Some("m"),
            headers: Some(Arc::new(headers)),
        };

        let attempt_1 = context.clone();
        let attempt_2 = context.clone();

        let original = context.headers.as_ref().expect("headers set");
        let a1 = attempt_1.headers.as_ref().expect("headers set");
        let a2 = attempt_2.headers.as_ref().expect("headers set");
        assert!(Arc::ptr_eq(original, a1));
        assert!(Arc::ptr_eq(original, a2));
        assert_eq!(Arc::strong_count(original), 3);
    }

    #[test]
    fn test_upstream_status_preserved_for_4xx_5xx() {
        for status in [
            StatusCode::UNAUTHORIZED,
            StatusCode::UNPROCESSABLE_ENTITY,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            let response = error::create_error(status, "decode_error", "upstream rejected");
            assert_eq!(
                response.status(),
                status,
                "upstream status {} must pass through unchanged",
                status
            );
        }
    }
}
