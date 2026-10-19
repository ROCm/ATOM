use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{
        header::{AUTHORIZATION, CONTENT_TYPE},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;

#[cfg(test)]
mod load_tests;

use memchr::memmem;
use reqwest::Client;
use serde::Serialize;
use serde_json::{json, Value};
use tracing::{debug, error, info, warn};

use crate::routers::prepare::inference::{InferenceMetadata, InferenceRequest};

use crate::{
    config::types::{AtomPdRankMappingPolicy, BackendType, RetryConfig},
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
        metrics::{metrics_labels, MeshMetrics},
    },
    policies::PolicyRegistry,
    protocols::{
        chat::ChatCompletionRequest, completion::CompletionRequest, generate::GenerateRequest,
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

#[derive(Clone)]
pub struct PDRouter {
    pub worker_registry: Arc<WorkerRegistry>,
    pub policy_registry: Arc<PolicyRegistry>,
    pub client: Client,
    pub retry_config: RetryConfig,
    pub backend: BackendType,
    pub atom_pd_rank_mapping_policy: AtomPdRankMappingPolicy,
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
            .field(
                "atom_pd_rank_mapping_policy",
                &self.atom_pd_rank_mapping_policy,
            )
            .finish()
    }
}

/// Reserve both workers once, then transfer each guard to its actual execution.
pub(crate) struct ReservedPair {
    pub prefill: Arc<dyn Worker>,
    pub decode: Arc<dyn Worker>,
    load: parking_lot::Mutex<Option<[WorkerLoadGuard; 2]>>,
    canceled: tokio::sync::watch::Sender<bool>,
}

impl ReservedPair {
    fn take_load(&self) -> Result<[WorkerLoadGuard; 2], Response> {
        self.load.lock().take().ok_or_else(|| {
            error::internal_error(
                "placement_unavailable",
                "Reserved placement was already executed or canceled",
            )
        })
    }

    #[cfg(feature = "ext-proc")]
    pub(crate) fn cancel(&self) {
        self.canceled.send_replace(true);
        // An unaccepted execution still owns both guards here.
        self.load.lock().take();
    }
}

/// Owns one worker attempt through response consumption, including cancellation.
struct WorkerOutcome {
    worker: Option<Arc<dyn Worker>>,
    policy: Arc<dyn crate::policies::LoadBalancingPolicy>,
    status: Option<StatusCode>,
}

impl WorkerOutcome {
    fn record_error(&self, worker: &dyn Worker) {
        let kind = match worker.worker_type() {
            crate::core::WorkerType::Prefill { .. } => metrics_labels::WORKER_PREFILL,
            _ => metrics_labels::WORKER_DECODE,
        };
        let error = self
            .status
            .filter(|s| s.is_client_error() || s.is_server_error())
            .map(error_type_from_status)
            .unwrap_or(metrics_labels::ERROR_BACKEND);
        MeshMetrics::record_worker_error(kind, metrics_labels::CONNECTION_HTTP, error);
    }

    fn finish(&mut self, body_ok: bool) {
        if let Some(worker) = self.worker.take() {
            let success = body_ok
                && self
                    .status
                    .is_some_and(|s| s.is_success() || s.is_redirection());
            if !body_ok || self.status.is_some_and(|s| s.is_server_error()) {
                worker.record_outcome(false);
            } else if success {
                worker.record_outcome(true);
            }
            if !body_ok
                || self
                    .status
                    .is_some_and(|s| s.is_client_error() || s.is_server_error())
            {
                self.record_error(worker.as_ref());
            }
            self.policy.on_request_complete(worker.url(), success);
        }
    }
}

impl Drop for WorkerOutcome {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            // Cancellation alone is not evidence of an unhealthy worker.
            if self.status.is_some_and(|s| s.is_server_error()) {
                worker.record_outcome(false);
                self.record_error(worker.as_ref());
            }
            self.policy.on_request_complete(worker.url(), false);
            debug!(
                worker_url = worker.url(),
                "worker response canceled before completion"
            );
        }
    }
}

struct WorkerResponse {
    inner: reqwest::Response,
    outcome: WorkerOutcome,
}

impl WorkerResponse {
    fn status(&self) -> StatusCode {
        self.inner.status()
    }
    fn headers(&self) -> &HeaderMap {
        self.inner.headers()
    }

    async fn bytes(self) -> Result<bytes::Bytes, reqwest::Error> {
        let Self { inner, mut outcome } = self;
        let result = inner.bytes().await;
        outcome.finish(result.is_ok());
        result
    }

    async fn drain(self) -> Result<(), reqwest::Error> {
        let Self { inner, mut outcome } = self;
        let mut stream = inner.bytes_stream();
        while let Some(chunk) = stream.next().await {
            if let Err(error) = chunk {
                outcome.finish(false);
                return Err(error);
            }
        }
        outcome.finish(true);
        Ok(())
    }

    async fn text(self) -> Result<String, reqwest::Error> {
        let Self { inner, mut outcome } = self;
        let result = inner.text().await;
        outcome.finish(result.is_ok());
        result
    }

    async fn json<T: serde::de::DeserializeOwned>(self) -> Result<T, reqwest::Error> {
        let Self { inner, mut outcome } = self;
        let result = inner.json().await;
        outcome.finish(result.is_ok());
        result
    }

    fn bytes_stream(
        self,
    ) -> impl futures_util::Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send {
        let Self { inner, outcome } = self;
        futures_util::stream::unfold(
            (Box::pin(inner.bytes_stream()), outcome),
            |(mut stream, mut outcome)| async move {
                match stream.next().await {
                    Some(result) => {
                        if result
                            .as_ref()
                            .is_ok_and(|b| memmem::find(b, b"data: [DONE]").is_some())
                        {
                            outcome.finish(true);
                        } else if result.is_err() {
                            outcome.finish(false);
                        }
                        Some((result, (stream, outcome)))
                    }
                    None => {
                        outcome.finish(true);
                        None
                    }
                }
            },
        )
    }
}

/// Cancel prefill on dispatch errors; successful streams can detach its drain.
struct PrefillTask(Option<tokio::task::JoinHandle<()>>);

impl PrefillTask {
    async fn finish(mut self) {
        match tokio::time::timeout(Duration::from_secs(5), self.0.as_mut().unwrap()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => warn!(%error, "prefill task failed"),
            Err(_) => warn!("prefill response drain timed out; canceling task"),
        }
    }
}

impl PrefillTask {
    fn detach(mut self) {
        self.0.take();
    }
}

impl Drop for PrefillTask {
    fn drop(&mut self) {
        if let Some(task) = &self.0 {
            task.abort();
        }
    }
}

#[derive(Clone)]
struct PDRequestContext<'a> {
    route: &'static str,
    batch_size: Option<usize>,
    is_stream: bool,
    return_logprob: bool,
    request_text: Option<&'a str>,
    model_id: Option<&'a str>,
    headers: Option<Arc<HeaderMap>>,
}

impl<'a> PDRequestContext<'a> {
    fn from_metadata(
        metadata: &'a InferenceMetadata,
        headers: Option<&HeaderMap>,
        model: Option<&'a str>,
    ) -> Self {
        Self {
            route: metadata.route,
            batch_size: metadata.batch_size,
            is_stream: metadata.stream,
            return_logprob: metadata.return_logprob,
            request_text: Some(&metadata.text),
            model_id: model.or_else(|| metadata.model.as_deref().filter(|model| !model.is_empty())),
            headers: headers.cloned().map(Arc::new),
        }
    }
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
            atom_pd_rank_mapping_policy: ctx.router_config.atom_pd_rank_mapping_policy,
            planner,
            adapter,
            atom_adapter,
        })
    }

    pub(crate) fn reserve_pair(
        &self,
        plan: PlacementPlan,
        headers: Option<&HeaderMap>,
    ) -> Result<Arc<ReservedPair>, Response> {
        let PlacementPlan::Pair {
            prefill,
            decode,
            prefill_policy,
            decode_policy,
        } = plan
        else {
            return Err(error::internal_error(
                "unexpected_single_plan",
                "PD execution requires a pair",
            ));
        };
        let prefill = if matches!(self.backend, BackendType::Atom) {
            self.apply_atom_pd_rank_mapping_policy(prefill, &decode)
        } else {
            prefill
        };
        for (worker, kind, policy) in [
            (&prefill, metrics_labels::WORKER_PREFILL, prefill_policy),
            (&decode, metrics_labels::WORKER_DECODE, decode_policy),
        ] {
            MeshMetrics::record_worker_selection(
                kind,
                metrics_labels::CONNECTION_HTTP,
                worker.model_id(),
                policy,
            );
        }
        Ok(Arc::new(ReservedPair {
            load: parking_lot::Mutex::new(Some([
                WorkerLoadGuard::new(prefill.clone(), headers),
                WorkerLoadGuard::new(decode.clone(), headers),
            ])),
            canceled: tokio::sync::watch::channel(false).0,
            prefill,
            decode,
        }))
    }

    /// Execute exactly this reserved pair. Selection and retries belong to the caller.
    #[cfg(feature = "ext-proc")]
    pub(crate) async fn execute_placement(
        &self,
        headers: &HeaderMap,
        body: Value,
        metadata: &InferenceMetadata,
        placement: Arc<ReservedPair>,
    ) -> Response {
        let context = PDRequestContext::from_metadata(metadata, Some(headers), None);
        self.execute_reserved(Some(headers), body, context, placement)
            .await
    }

    fn apply_atom_pd_rank_mapping_policy(
        &self,
        prefill: Arc<dyn Worker>,
        decode: &Arc<dyn Worker>,
    ) -> Arc<dyn Worker> {
        match self.atom_pd_rank_mapping_policy {
            AtomPdRankMappingPolicy::None => prefill,
            AtomPdRankMappingPolicy::Idx2Idx => self.map_atom_prefill_idx2idx(prefill, decode),
        }
    }

    fn map_atom_prefill_idx2idx(
        &self,
        prefill: Arc<dyn Worker>,
        decode: &Arc<dyn Worker>,
    ) -> Arc<dyn Worker> {
        let (Some(prefill_dp_size), Some(decode_dp_rank)) = (prefill.dp_size(), decode.dp_rank())
        else {
            debug!(
                "ATOM PD rank mapping policy=idx2idx skipped: prefill={} prefill_dp_size={:?} decode={} decode_dp_rank={:?}",
                prefill.url(),
                prefill.dp_size(),
                decode.url(),
                decode.dp_rank()
            );
            return prefill;
        };

        if decode_dp_rank >= prefill_dp_size {
            warn!(
                "ATOM PD rank mapping policy=idx2idx skipped: decode rank {} is outside prefill dp_size {} (prefill={}, decode={})",
                decode_dp_rank,
                prefill_dp_size,
                prefill.url(),
                decode.url()
            );
            return prefill;
        }

        if prefill.dp_rank() == Some(decode_dp_rank) {
            return prefill;
        }

        let mapped_url = format!("{}@{}", prefill.base_url(), decode_dp_rank);
        match self.worker_registry.get_by_url(&mapped_url) {
            Some(mapped)
                if mapped.is_available()
                    && mapped.model_id() == prefill.model_id()
                    && matches!(
                        mapped.worker_type(),
                        crate::core::WorkerType::Prefill { .. }
                    )
                    && mapped.connection_mode().matches(prefill.connection_mode()) =>
            {
                info!(
                    "ATOM PD rank mapping policy=idx2idx: prefill {} -> {}, decode={}",
                    prefill.url(),
                    mapped.url(),
                    decode.url()
                );
                mapped
            }
            Some(mapped) => {
                warn!(
                    "ATOM PD rank mapping policy=idx2idx target {} is unhealthy; keeping prefill {} (decode={})",
                    mapped.url(),
                    prefill.url(),
                    decode.url()
                );
                prefill
            }
            None => {
                warn!(
                    "ATOM PD rank mapping policy=idx2idx target {} not found; keeping prefill {} (decode={})",
                    mapped_url,
                    prefill.url(),
                    decode.url()
                );
                prefill
            }
        }
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
        let mut queried_base_urls = HashSet::new();
        for worker in &prefill_workers {
            let worker_url = worker.url().to_string();
            let base_url = worker.base_url().trim_end_matches('/').to_string();
            if !queried_base_urls.insert(base_url.clone()) {
                if let Some(tp) = tp_sizes.get(&base_url).copied() {
                    tp_sizes.insert(worker_url, tp);
                }
                continue;
            }
            let info_url = format!("{}/kv_transfer_info", base_url);

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
                    base_url, kv_role
                ));
            }
            info!("ATOM prefill {} tp_size={}", base_url, tp);
            tp_sizes.insert(base_url, tp);
            tp_sizes.insert(worker_url, tp);
        }
        Ok(AtomPrefillInfo { tp_sizes })
    }

    fn handle_serialization_error(error: impl std::fmt::Display) -> Response {
        error!("Failed to serialize request error={}", error);
        error::internal_error("serialization_failed", "Failed to serialize request")
    }

    async fn dispatch_pd<T: Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        original_request: &T,
        context: PDRequestContext<'_>,
    ) -> Response {
        let request = match serde_json::to_value(original_request) {
            Ok(value) => value,
            Err(error) => return Self::handle_serialization_error(error),
        };
        let endpoint = route_to_endpoint(context.route);
        let observation = crate::observability::request::RequestMetrics::new(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_PD,
            context.model_id.unwrap_or(UNKNOWN_MODEL_ID),
            context.route,
            context.is_stream,
        );
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_attempt| {
                let request = request.clone();
                let context = context.clone();
                async move {
                    let placement = match self.plan_pd_pair(&context).await {
                        Ok(pair) => pair,
                        Err(response) => return response,
                    };
                    self.execute_reserved(headers, request, context, placement)
                        .await
                }
            },
            |response, _| is_retryable_status(response.status()),
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
        observation.wrap_response(response)
    }

    async fn plan_pd_pair(
        &self,
        context: &PDRequestContext<'_>,
    ) -> Result<Arc<ReservedPair>, Response> {
        let descriptor = RequestDescriptor {
            model_id: context.model_id,
            protocol: Some(Protocol::Http),
            text: context.request_text.as_deref(),
            tokens: None,
            headers: context.headers.as_deref(),
            stream: context.is_stream,
        };
        let plan = self
            .planner
            .plan(&descriptor)
            .await
            .map_err(|error| placement_err_to_response(error, context.model_id))?;
        self.reserve_pair(plan, context.headers.as_deref())
    }

    async fn execute_reserved(
        &self,
        headers: Option<&HeaderMap>,
        mut body: Value,
        context: PDRequestContext<'_>,
        placement: Arc<ReservedPair>,
    ) -> Response {
        let prefill = placement.prefill.clone();
        let decode = placement.decode.clone();
        let ctx = match self.adapter.prepare_pair(prefill.as_ref(), decode.as_ref()) {
            Ok(ctx) => ctx,
            Err(error) => return Self::handle_serialization_error(error),
        };
        let started = Instant::now();
        let response = match self.backend {
            BackendType::Sglang => {
                let injected = match context.batch_size {
                    Some(n) => self.adapter.inject_batch_prefill_fields(&mut body, &ctx, n),
                    None => self.adapter.inject_prefill_fields(&mut body, &ctx),
                };
                if let Err(error) = injected {
                    return Self::handle_serialization_error(error);
                }
                self.execute_dual_dispatch_internal(
                    headers,
                    body,
                    context,
                    placement.clone(),
                    started,
                )
                .await
            }
            BackendType::Vllm | BackendType::Atom => {
                let mut decode_body = body.clone();
                if let Err(error) = self.adapter.inject_prefill_fields(&mut body, &ctx) {
                    return Self::handle_serialization_error(error);
                }
                if let Err(error) = self.adapter.inject_decode_fields(&mut decode_body, &ctx) {
                    return Self::handle_serialization_error(error);
                }
                let correlation = self.adapter.correlation_id(&ctx);
                if matches!(self.backend, BackendType::Vllm) {
                    self.dispatch_vllm_mooncake_internal(
                        headers,
                        body,
                        decode_body,
                        context,
                        placement.clone(),
                        started,
                        correlation,
                    )
                    .await
                } else {
                    self.dispatch_atom_relay_internal(
                        headers,
                        body,
                        decode_body,
                        context,
                        placement.clone(),
                        ctx,
                        started,
                        correlation,
                    )
                    .await
                }
            }
        };
        crate::core::AttachedBody::wrap_response(response, placement)
    }

    /// Core vLLM Mooncake dispatch: fire P as background task, stream D response back to client.
    #[allow(clippy::too_many_arguments)]
    async fn dispatch_vllm_mooncake_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_request_json: Value,
        decode_request_json: Value,
        context: PDRequestContext<'_>,
        placement: Arc<ReservedPair>,
        _start_time: Instant,
        correlation_id: Option<String>,
    ) -> Response {
        // Take the existing reservations before preparing or sending requests.
        // Streaming requests also count while waiting for response headers.
        let prefill = placement.prefill.clone();
        let decode = placement.decode.clone();
        let [prefill_guard, decode_guard] = match placement.take_load() {
            Ok(guards) => guards,
            Err(response) => return response,
        };

        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        // Mooncake coordinates KV transfer out of band. Keep prefill owned until
        // decode succeeds, then allow its drain to complete independently.
        let prefill_post = match self
            .build_worker_post_with_headers(
                &self.client,
                prefill.as_ref(),
                context.route,
                prefill_request_json,
                headers,
                false,
            )
            .await
        {
            Ok(req) => req,
            Err(resp) => return resp,
        };
        let prefill_url_for_log = prefill.url().to_string();
        let prefill_worker = prefill.clone();
        let prefill_router = self.clone();
        let correlation_for_log = correlation_id.unwrap_or_else(|| "unknown".to_string());
        let mut canceled = placement.canceled.subscribe();
        let work = async move {
            let _prefill_guard = prefill_guard;
            match prefill_router
                .send_worker(prefill_post, prefill_worker)
                .await
            {
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
                    if let Err(error) = res.drain().await {
                        warn!(%error, "failed to drain prefill response");
                    }
                }
                Err(e) => {
                    error!(
                        "vLLM prefill {} request_id={} failed: {}",
                        prefill_url_for_log, correlation_for_log, e
                    );
                }
            }
        };
        let prefill_task = PrefillTask(Some(tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = async {
                    if canceled.wait_for(|value| *value).await.is_err() {
                        // Dropping a reservation is not explicit cancellation:
                        // an HTTP prefill may still be draining after decode ends.
                        std::future::pending::<()>().await;
                    }
                } => {},
                _ = work => {},
            }
        })));

        // D request: client sees the streamed (or buffered) response from D.
        let decode_post = match self
            .build_worker_post_with_headers(
                &self.client,
                decode.as_ref(),
                context.route,
                decode_request_json,
                headers,
                false,
            )
            .await
        {
            Ok(req) => req,
            Err(resp) => return resp,
        };
        let decode_result = self.send_worker(decode_post, decode.clone()).await;
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

                return error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !status.is_success() {
            error!(
                "vLLM decode {} returned error status={}",
                decode.url(),
                status
            );
            return self
                .handle_decode_error_response(res, &context, decode_guard, decode)
                .await;
        }

        if context.is_stream {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let response = self.create_streaming_response(
                res.bytes_stream(),
                status,
                None,
                false,
                None,
                Some(response_headers),
                decode_guard,
            );
            // Preserve detached HTTP prefill draining. Execution leases can
            // still cancel the task through the reservation's cancellation signal.
            prefill_task.detach();
            response
        } else {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            match res.bytes().await {
                Ok(decode_body) => {
                    prefill_task.finish().await;
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

    #[allow(clippy::too_many_arguments)]
    async fn dispatch_atom_relay_internal(
        &self,
        headers: Option<&HeaderMap>,
        prefill_request_json: Value,
        mut decode_request_json: Value,
        context: PDRequestContext<'_>,
        placement: Arc<ReservedPair>,
        ctx: PairCtx,
        _start_time: Instant,
        correlation_id: Option<String>,
    ) -> Response {
        // Take the reserved pair before the first await, including streaming requests.
        // D remains reserved while P runs because this request already selected D.
        let prefill = placement.prefill.clone();
        let decode = placement.decode.clone();
        let [prefill_guard, decode_guard] = match placement.take_load() {
            Ok(guards) => guards,
            Err(response) => return response,
        };

        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        let prefill_post = match self
            .build_worker_post_with_headers(
                &self.client,
                prefill.as_ref(),
                context.route,
                prefill_request_json,
                headers,
                false,
            )
            .await
        {
            Ok(req) => req,
            Err(resp) => return resp,
        };
        let correlation_for_log = correlation_id
            .clone()
            .unwrap_or_else(|| "unknown".to_string());

        let prefill_result = self.send_worker(prefill_post, prefill.clone()).await;
        let prefill_resp = match prefill_result {
            Ok(r) => r,
            Err(e) => {
                error!(
                    "ATOM prefill {} request_id={} failed: {}",
                    prefill.url(),
                    correlation_for_log,
                    e
                );

                return error::bad_gateway(
                    "prefill_server_error",
                    format!("Prefill server error: {}", e),
                );
            }
        };

        let prefill_status = prefill_resp.status();

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

        // P has completed its work; D's output stream must not keep P loaded.
        drop(prefill_guard);

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
        let carried_ids = AtomAdapter::carry_prompt_token_ids(&prefill_body, &mut kv_params);
        debug!(
            "ATOM PD request_id={} carried {} prompt token ids to decode {}",
            correlation_for_log,
            carried_ids,
            decode.url()
        );

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

        let decode_post = match self
            .build_worker_post_with_headers(
                &self.client,
                decode.as_ref(),
                context.route,
                decode_request_json,
                headers,
                false,
            )
            .await
        {
            Ok(req) => req,
            Err(resp) => return resp,
        };
        let decode_result = self.send_worker(decode_post, decode.clone()).await;
        events::RequestReceivedEvent {}.emit();

        let res = match decode_result {
            Ok(r) => r,
            Err(e) => {
                error!(
                    decode_url = %decode.url(),
                    error = %e,
                    "ATOM decode request failed"
                );

                return error::bad_gateway(
                    "decode_server_error",
                    format!("Decode server error: {}", e),
                );
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !status.is_success() {
            error!(
                "ATOM decode {} returned error status={}",
                decode.url(),
                status
            );
            return self
                .handle_decode_error_response(res, &context, decode_guard, decode)
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
                decode_guard,
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

    async fn handle_decode_error_response(
        &self,
        res: WorkerResponse,
        context: &PDRequestContext<'_>,
        decode_guard: WorkerLoadGuard,
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

            let sse_data = format!("data: {}\n\n", json!({"error": error_payload}));
            let error_stream = tokio_stream::once(Ok(axum::body::Bytes::from(sse_data)));

            let decode_url = decode.url().to_string();
            self.create_streaming_response(
                error_stream,
                status,
                None,
                context.return_logprob,
                Some(decode_url),
                Some(response_headers),
                decode_guard,
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
        placement: Arc<ReservedPair>,
        _start_time: Instant,
    ) -> Response {
        let prefill = placement.prefill.clone();
        let decode = placement.decode.clone();
        let [prefill_guard, decode_guard] = match placement.take_load() {
            Ok(guards) => guards,
            Err(response) => return response,
        };

        let prefill_request = match self
            .build_worker_post_with_headers(
                &self.client,
                prefill.as_ref(),
                context.route,
                json_request.clone(),
                headers,
                false,
            )
            .await
        {
            Ok(request) => request,
            Err(response) => return response,
        };
        let decode_request = match self
            .build_worker_post_with_headers(
                &self.client,
                decode.as_ref(),
                context.route,
                json_request,
                headers,
                false,
            )
            .await
        {
            Ok(request) => request,
            Err(response) => return response,
        };

        // Send both requests concurrently and wait for both
        // Note: Using borrowed references avoids heap allocation
        events::RequestPDSentEvent {
            prefill_url: prefill.url(),
            decode_url: decode.url(),
        }
        .emit();

        enum DispatchError {
            Prefill(WorkerResponse, WorkerLoadGuard),
            Decode(WorkerResponse, WorkerLoadGuard),
            Response(Response),
        }

        // Drain successful P responses independently of D's headers. Return
        // HTTP errors on headers so try_join! drops the peer and its reservation
        // before we await a potentially stalled error body below.
        let results = tokio::try_join!(
            async {
                let result = self.send_worker(prefill_request, prefill.clone()).await;
                let result = match result {
                    Ok(res) if !res.status().is_success() => {
                        return Err(DispatchError::Prefill(res, prefill_guard));
                    }
                    result => result,
                };
                let result = self
                    .process_prefill_response(result, prefill.url(), context.return_logprob)
                    .await
                    .map_err(DispatchError::Response);
                drop(prefill_guard);
                result
            },
            async {
                let res = self
                    .send_worker(decode_request, decode.clone())
                    .await
                    .map_err(|e| {
                        error!(decode_url = %decode.url(), error = %e, "Decode request failed");
                        DispatchError::Response(error::bad_gateway(
                            "decode_server_error",
                            format!("Decode server error: {}", e),
                        ))
                    })?;
                if !res.status().is_success() {
                    return Err(DispatchError::Decode(res, decode_guard));
                }
                Ok((res, decode_guard))
            }
        );

        events::RequestReceivedEvent {}.emit();
        let ((_, prefill_body), (res, decode_guard)) = match results {
            Ok(results) => results,
            Err(DispatchError::Response(response)) => return response,
            Err(DispatchError::Prefill(res, _prefill_guard)) => {
                // Keep only the failing worker reserved while reading its body.
                return self.handle_prefill_error_response(res, prefill.url()).await;
            }
            Err(DispatchError::Decode(res, decode_guard)) => {
                return self
                    .handle_decode_error_response(res, &context, decode_guard, decode)
                    .await;
            }
        };
        let status = res.status();

        if context.is_stream {
            let prefill_logprobs = if context.return_logprob {
                prefill_body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .and_then(|json| json.pointer("/meta_info/input_token_logprobs").cloned())
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
                decode_guard,
            )
        } else if context.return_logprob {
            self.process_non_streaming_response(res, status, true, prefill_body)
                .await
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
                    error!("Failed to read decode response: {}", e);
                    error::internal_error("read_response_failed", "Failed to read response")
                }
            }
        }
    }

    #[cfg(test)]
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
        decode_guard: WorkerLoadGuard,
    ) -> Response {
        // Poll the upstream only when the downstream asks for data. Dropping the
        // response also drops the upstream, including while it is idle.
        let stream = futures_util::stream::unfold(
            (Box::pin(stream), false, prefill_logprobs, decode_url),
            move |(mut stream, finished, prefill_logprobs, decode_url)| async move {
                if finished {
                    return None;
                }
                let result = stream.next().await?;
                let (result, finished) = match result {
                    Ok(chunk) => {
                        let finished = memmem::find(&chunk, b"data: [DONE]").is_some();
                        let chunk = if return_logprob && prefill_logprobs.is_some() {
                            Self::merge_streaming_logprobs(prefill_logprobs.clone(), &chunk)
                                .unwrap_or(chunk)
                        } else {
                            chunk
                        };
                        (Ok(chunk), finished)
                    }
                    Err(error) => {
                        if let Some(ref url) = decode_url {
                            error!("Stream error from decode server {}: {}", url, error);
                        }
                        (Err(error), true)
                    }
                };
                Some((result, (stream, finished, prefill_logprobs, decode_url)))
            },
        );
        let body = Body::from_stream(stream);

        let mut response = Response::new(body);
        *response.status_mut() = status;

        let mut response_headers = headers.unwrap_or_default();
        response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        *response.headers_mut() = response_headers;

        // Transfer the existing reservation instead of incrementing load again.
        crate::core::AttachedBody::wrap_response(response, decode_guard)
    }

    // Helper to process non-streaming decode response with logprob merging
    async fn process_non_streaming_response(
        &self,
        res: WorkerResponse,
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

    async fn handle_prefill_error_response(
        &self,
        response: WorkerResponse,
        prefill_url: &str,
    ) -> Response {
        let status = response.status();
        let error_msg = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown prefill error".to_string());
        error!(
            "Prefill server returned error status prefill_url={} status={} body={}",
            prefill_url, status, error_msg
        );
        error::create_error(
            status,
            "prefill_error",
            format!("Prefill server error ({}): {}", status, error_msg),
        )
    }

    // Helper to process prefill response and extract body if needed for logprobs
    async fn process_prefill_response(
        &self,
        prefill_result: Result<WorkerResponse, reqwest::Error>,
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
            return Err(self
                .handle_prefill_error_response(prefill_response, prefill_url)
                .await);
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
            match prefill_response.drain().await {
                Ok(()) => debug!("Prefill response consumed successfully"),
                Err(e) => warn!("Error consuming prefill response: {}", e),
            }
            None
        };

        Ok((prefill_status, prefill_body))
    }

    async fn send_worker(
        &self,
        request: reqwest::RequestBuilder,
        worker: Arc<dyn Worker>,
    ) -> Result<WorkerResponse, reqwest::Error> {
        let policy = match worker.worker_type() {
            crate::core::WorkerType::Prefill { .. } => self.policy_registry.get_prefill_policy(),
            _ => self.policy_registry.get_decode_policy(),
        };
        let mut outcome = WorkerOutcome {
            worker: Some(worker),
            policy,
            status: None,
        };
        match request.send().await {
            Ok(inner) => {
                outcome.status = Some(inner.status());
                Ok(WorkerResponse { inner, outcome })
            }
            Err(error) => {
                outcome.finish(false);
                Err(error)
            }
        }
    }

    async fn build_worker_post_with_headers(
        &self,
        client: &Client,
        worker: &dyn Worker,
        route: &'static str,
        json_request: Value,
        headers: Option<&HeaderMap>,
        connection_close: bool,
    ) -> Result<reqwest::RequestBuilder, Response> {
        let prepared_request = worker.prepare_request(json_request).await.map_err(|e| {
            error!(
                worker_url = %worker.url(),
                error = ?e,
                "Failed to prepare DP-aware worker request"
            );
            error::internal_error(
                "worker_prepare_request_failed",
                format!(
                    "Failed to prepare worker request for {}: {:?}",
                    worker.url(),
                    e
                ),
            )
        })?;

        let mut request = client
            .post(worker.endpoint_url(route))
            .json(&prepared_request);
        if connection_close {
            request = request.header("Connection", "close");
        }
        let api_key = worker.api_key();
        if let Some(headers) = headers {
            for (name, value) in headers.iter() {
                // bearer_auth appends, so omit client credentials when the worker has its own.
                if name == AUTHORIZATION && api_key.is_some() {
                    continue;
                }
                if header_utils::should_forward_request_header(name.as_str()) {
                    if let Ok(val) = value.to_str() {
                        request = request.header(name, val);
                    }
                }
            }
        }
        if let Some(key) = api_key {
            request = request.bearer_auth(key);
        }
        Ok(request)
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
        let metadata = body.metadata();
        let context = PDRequestContext::from_metadata(&metadata, headers, model_id);
        self.dispatch_pd(headers, body, context).await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let metadata = body.metadata();
        let context = PDRequestContext::from_metadata(&metadata, headers, model_id);
        self.dispatch_pd(headers, body, context).await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let metadata = body.metadata();
        let context = PDRequestContext::from_metadata(&metadata, headers, model_id);
        self.dispatch_pd(headers, body, context).await
    }

    fn router_type(&self) -> &'static str {
        "pd"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        placement::backend::sglang::SglangAdapter, BasicWorkerBuilder, DPAwareWorkerBuilder,
        WorkerType,
    };

    #[derive(Debug, Default)]
    struct CompletionPolicy(std::sync::Mutex<Vec<bool>>);
    #[async_trait::async_trait]
    impl crate::policies::LoadBalancingPolicy for CompletionPolicy {
        async fn select_worker(
            &self,
            _: &[Arc<dyn Worker>],
            _: &crate::policies::SelectWorkerInfo<'_>,
        ) -> Option<usize> {
            Some(0)
        }
        fn on_request_complete(&self, _: &str, success: bool) {
            self.0.lock().unwrap().push(success);
        }
        fn name(&self) -> &'static str {
            "completion-test"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn worker_outcome_completes_policy_once_and_cancellation_is_neutral() {
        for (status, body, successes, failures, completed) in [
            (Some(StatusCode::OK), Some(true), 1, 0, true),
            (Some(StatusCode::OK), Some(false), 0, 1, false),
            (
                Some(StatusCode::SERVICE_UNAVAILABLE),
                Some(true),
                0,
                1,
                false,
            ),
            (Some(StatusCode::BAD_REQUEST), Some(true), 0, 0, false),
            (Some(StatusCode::OK), None, 0, 0, false),
            (None, None, 0, 0, false),
            (None, Some(false), 0, 1, false),
        ] {
            let worker: Arc<dyn Worker> =
                Arc::new(BasicWorkerBuilder::new("http://worker").build());
            let policy = Arc::new(CompletionPolicy::default());
            let mut outcome = WorkerOutcome {
                worker: Some(worker.clone()),
                policy: policy.clone(),
                status,
            };
            if let Some(ok) = body {
                outcome.finish(ok);
                outcome.finish(ok);
            }
            drop(outcome);
            assert_eq!(*policy.0.lock().unwrap(), vec![completed]);
            assert_eq!(worker.circuit_breaker().total_successes(), successes);
            assert_eq!(worker.circuit_breaker().total_failures(), failures);
        }
    }

    #[tokio::test]
    async fn truncated_worker_body_is_failure_but_unread_body_is_neutral() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        for consume in [true, false] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let server = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = [0; 4096];
                socket.read(&mut request).await.unwrap();
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{}",
                    )
                    .await
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(20)).await;
            });
            let router = create_test_pd_router();
            let worker: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(&url)
                    .worker_type(WorkerType::Decode)
                    .build(),
            );
            let response = router
                .send_worker(router.client.get(&url), worker.clone())
                .await
                .unwrap();
            assert_eq!(worker.circuit_breaker().total_successes(), 0);
            if consume {
                let mut stream = Box::pin(response.bytes_stream());
                let mut failed = false;
                while let Some(chunk) = stream.next().await {
                    failed |= chunk.is_err();
                }
                assert!(failed);
            } else {
                drop(response);
            }
            server.await.unwrap();
            assert_eq!(
                worker.circuit_breaker().total_failures(),
                u64::from(consume)
            );
            assert_eq!(worker.circuit_breaker().total_successes(), 0);
        }
    }

    #[tokio::test]
    async fn buffered_vllm_decode_drains_slow_prefill_and_decode_error_does_not_blame_prefill() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let backend = axum::Router::new()
            .route(
                "/prefill",
                axum::routing::post(|| async {
                    Body::from_stream(futures_util::stream::once(async {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        Ok::<_, std::convert::Infallible>("{}")
                    }))
                }),
            )
            .route(
                "/decode",
                axum::routing::post(|axum::Json(body): axum::Json<Value>| async move {
                    if body["fail"] == true {
                        (StatusCode::SERVICE_UNAVAILABLE, "decode failed")
                    } else {
                        (StatusCode::OK, "decode done")
                    }
                }),
            );
        let server = tokio::spawn(async move {
            axum::serve(listener, backend).await.unwrap();
        });
        for fail in [false, true] {
            let router = create_test_pd_router();
            let prefill: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("{url}/prefill"))
                    .worker_type(WorkerType::Prefill {
                        bootstrap_port: None,
                    })
                    .build(),
            );
            let decode: Arc<dyn Worker> = Arc::new(
                BasicWorkerBuilder::new(format!("{url}/decode"))
                    .worker_type(WorkerType::Decode)
                    .build(),
            );
            let context = PDRequestContext {
                route: "",
                batch_size: None,
                is_stream: false,
                return_logprob: false,
                request_text: None,
                model_id: None,
                headers: None,
            };
            let placement = router
                .reserve_pair(
                    PlacementPlan::Pair {
                        prefill: prefill.clone(),
                        decode: decode.clone(),
                        prefill_policy: "round_robin",
                        decode_policy: "round_robin",
                    },
                    None,
                )
                .unwrap();
            let response = router
                .dispatch_vllm_mooncake_internal(
                    None,
                    json!({}),
                    json!({"fail":fail}),
                    context,
                    placement,
                    Instant::now(),
                    None,
                )
                .await;
            assert_eq!(
                response.status(),
                if fail {
                    StatusCode::SERVICE_UNAVAILABLE
                } else {
                    StatusCode::OK
                }
            );
            drop(response);
            tokio::task::yield_now().await;
            assert_eq!(prefill.circuit_breaker().total_failures(), 0);
            assert_eq!(
                prefill.circuit_breaker().total_successes(),
                u64::from(!fail)
            );
            assert_eq!(decode.circuit_breaker().total_failures(), u64::from(fail));
            assert_eq!(decode.circuit_breaker().total_successes(), u64::from(!fail));
            assert_eq!(prefill.load(), 0);
            assert_eq!(decode.load(), 0);
        }
        server.abort();
    }

    #[cfg(feature = "ext-proc")]
    #[tokio::test]
    async fn http_retries_replan_but_reserved_execution_never_replans_or_counts_twice() {
        use crate::core::placement::types::PlacementError;
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct CountingPlanner {
            inner: Arc<dyn PdPlanner>,
            calls: AtomicUsize,
        }
        #[async_trait::async_trait]
        impl PdPlanner for CountingPlanner {
            async fn plan(
                &self,
                request: &RequestDescriptor<'_>,
            ) -> Result<PlacementPlan, PlacementError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.plan(request).await
            }
        }
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let calls = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let app = axum::Router::new().fallback({
            let calls = calls.clone();
            move || {
                let calls = calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    StatusCode::SERVICE_UNAVAILABLE
                }
            }
        });
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let mut router = create_test_pd_router();
        router.retry_config.max_retries = 2;
        router.retry_config.initial_backoff_ms = 1;
        let prefill: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("{url}/prefill"))
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: Some(8001),
                })
                .build(),
        );
        let decode: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("{url}/decode"))
                .worker_type(WorkerType::Decode)
                .build(),
        );
        router.worker_registry.register(prefill.clone());
        router.worker_registry.register(decode.clone());
        let planner = Arc::new(CountingPlanner {
            inner: router.planner.clone(),
            calls: AtomicUsize::new(0),
        });
        router.planner = planner.clone();
        let request: CompletionRequest =
            serde_json::from_value(json!({"model": UNKNOWN_MODEL_ID, "prompt": "test"})).unwrap();
        let response = router.route_completion(None, &request, None).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(planner.calls.load(Ordering::SeqCst), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!((prefill.load(), decode.load()), (0, 0));
        let pair = router
            .reserve_pair(
                PlacementPlan::Pair {
                    prefill: prefill.clone(),
                    decode: decode.clone(),
                    prefill_policy: "round_robin",
                    decode_policy: "round_robin",
                },
                None,
            )
            .unwrap();
        let metadata = request.metadata().execution_metadata();
        let response = router
            .execute_placement(
                &HeaderMap::new(),
                serde_json::to_value(&request).unwrap(),
                &metadata,
                pair,
            )
            .await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        assert_eq!(planner.calls.load(Ordering::SeqCst), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 6);
        assert_eq!((prefill.load(), decode.load()), (0, 0));
        let rendered = handle.render();
        let request_counts: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("mesh_router_requests_total{"))
            .collect();
        assert_eq!(request_counts.len(), 1, "{rendered}");
        assert!(request_counts[0].ends_with(" 1"), "{rendered}");
        server.abort();
    }

    pub(super) fn create_test_pd_router() -> PDRouter {
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
            atom_pd_rank_mapping_policy: AtomPdRankMappingPolicy::None,
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

    fn create_test_dp_worker(
        base_url: &str,
        dp_rank: usize,
        dp_size: usize,
        worker_type: WorkerType,
        healthy: bool,
    ) -> Arc<dyn Worker> {
        let worker = DPAwareWorkerBuilder::new_with_type(
            base_url.to_string(),
            dp_rank,
            dp_size,
            worker_type,
        )
        .build();
        worker.set_healthy(healthy);
        Arc::new(worker)
    }

    #[test]
    fn test_atom_pd_rank_mapping_none_keeps_prefill_rank() {
        let mut router = create_test_pd_router();
        router.backend = BackendType::Atom;
        router.atom_pd_rank_mapping_policy = AtomPdRankMappingPolicy::None;

        let prefill = create_test_dp_worker(
            "http://prefill",
            2,
            8,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode = create_test_dp_worker("http://decode", 5, 8, WorkerType::Decode, true);

        let mapped = router.apply_atom_pd_rank_mapping_policy(prefill.clone(), &decode);
        assert_eq!(mapped.url(), prefill.url());
        assert_eq!(mapped.dp_rank(), Some(2));
    }

    #[test]
    fn test_atom_pd_rank_mapping_idx2idx_maps_prefill_to_decode_rank() {
        let mut router = create_test_pd_router();
        router.backend = BackendType::Atom;
        router.atom_pd_rank_mapping_policy = AtomPdRankMappingPolicy::Idx2Idx;

        let prefill_rank_2 = create_test_dp_worker(
            "http://prefill",
            2,
            8,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let prefill_rank_5 = create_test_dp_worker(
            "http://prefill",
            5,
            8,
            WorkerType::Prefill {
                bootstrap_port: None,
            },
            true,
        );
        let decode_rank_5 = create_test_dp_worker("http://decode", 5, 8, WorkerType::Decode, true);

        router.worker_registry.register(prefill_rank_2.clone());
        router.worker_registry.register(prefill_rank_5.clone());

        let pair = router
            .reserve_pair(
                PlacementPlan::Pair {
                    prefill: prefill_rank_2.clone(),
                    decode: decode_rank_5.clone(),
                    prefill_policy: "round_robin",
                    decode_policy: "round_robin",
                },
                None,
            )
            .unwrap();
        assert_eq!(pair.prefill.url(), "http://prefill@5");
        assert_eq!(pair.prefill.dp_rank(), Some(5));
        assert_eq!(prefill_rank_2.load(), 0);
        assert_eq!((prefill_rank_5.load(), decode_rank_5.load()), (1, 1));
        let execution = pair.clone();
        drop(pair);
        assert_eq!((prefill_rank_5.load(), decode_rank_5.load()), (1, 1));
        drop(execution);
        assert_eq!((prefill_rank_5.load(), decode_rank_5.load()), (0, 0));
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
        let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

        {
            let response = router.create_streaming_response(
                stream.map(Ok),
                StatusCode::OK,
                None,
                false,
                None,
                None,
                WorkerLoadGuard::new(decode_ref.clone(), None),
            );

            // Only D remains loaded after P has completed.
            assert_eq!(prefill_ref.load(), 0);
            assert_eq!(decode_ref.load(), 1);

            tx.send(bytes::Bytes::from("test data")).unwrap();

            sleep(Duration::from_millis(10)).await;

            // Load still 1 while response body exists
            assert_eq!(prefill_ref.load(), 0);
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
        assert_eq!(req.metadata().batch_size, None);
    }

    #[test]
    fn test_get_chat_batch_size_n_1() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}], "n": 1}"#,
        )
        .unwrap();
        assert_eq!(req.metadata().batch_size, None);
    }

    #[test]
    fn test_get_chat_batch_size_n_4() {
        let req: ChatCompletionRequest = serde_json::from_str(
            r#"{"model": "test", "messages": [{"role": "user", "content": "hi"}], "n": 4}"#,
        )
        .unwrap();
        assert_eq!(req.metadata().batch_size, Some(4));
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
