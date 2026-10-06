use std::{collections::BTreeMap, sync::Arc};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::Response,
};
use futures_util::{stream, StreamExt};
use reqwest::Client;
use tracing::{debug, error};

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{
        is_retryable_status,
        placement::{
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
            traits::PdPlanner,
            types::{PlacementPlan, Protocol, RequestDescriptor},
        },
        AttachedBody, ConnectionMode, RetryExecutor, WorkerLoadGuard, WorkerRegistry, WorkerType,
        UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{metrics_labels, MeshMetrics},
    },
    protocols::{
        chat::ChatCompletionRequest, common::GenerationRequest, completion::CompletionRequest,
        generate::GenerateRequest, responses::ResponsesRequest,
    },
    routers::{
        comm::{
            error::{self, extract_error_code_from_response},
            header_utils,
            metrics_utils::{error_type_from_status, route_to_endpoint},
            placement_response::placement_err_to_response,
        },
        RouterTrait,
    },
};

pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    planner: Arc<dyn PdPlanner>,
    policies: Arc<PolicyRegistryAdapter>,
    client: Client,
    dp_aware: bool,
    retry_config: RetryConfig,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("client", &self.client)
            .field("dp_aware", &self.dp_aware)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        let policies = Arc::new(PolicyRegistryAdapter::new(ctx.policy_registry.clone()));
        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(ctx.worker_registry.clone())),
            policies.clone(),
        ));
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            planner,
            policies,
            client: ctx.client.clone(),
            dp_aware: ctx.router_config.dp_aware,
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    async fn send_inference_once(
        &self,
        request: &super::ingress::InferenceEnvelope,
        descriptor: &RequestDescriptor<'_>,
        planner: &dyn PdPlanner,
        policy: Arc<dyn crate::policies::LoadBalancingPolicy>,
    ) -> Response {
        use super::comm::error::IngressError;
        let metadata = &request.metadata;
        let (worker, policy_name) = match planner.plan(descriptor).await {
            Ok(PlacementPlan::Single {
                worker,
                policy_name,
                ..
            }) => (worker, policy_name),
            Ok(_) => {
                return IngressError::new(
                    StatusCode::NOT_IMPLEMENTED,
                    "unsupported_api_topology",
                    "regular proxy requires a regular HTTP worker",
                )
                .response(metadata.route)
            }
            Err(err) => return placement_err_to_response(err, metadata.model.as_deref()),
        };
        MeshMetrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            metadata.model.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
            policy_name,
        );
        let load = WorkerLoadGuard::new(worker.clone(), Some(&request.headers));
        let headers = match header_utils::inference_request_headers(
            &request.headers,
            metadata.route,
            worker.api_key().as_deref(),
        ) {
            Ok(headers) => headers,
            Err(_) => {
                return IngressError::new(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "invalid_backend_credentials",
                    "configured worker credential is not a valid HTTP header",
                )
                .response(metadata.route)
            }
        };
        let mut builder = self
            .client
            .post(
                worker.endpoint_url(
                    request
                        .uri
                        .path_and_query()
                        .map(|p| p.as_str())
                        .unwrap_or(metadata.route),
                ),
            )
            .headers(headers);
        if worker.is_dp_aware() {
            let body: serde_json::Value = match serde_json::from_slice(&request.body) {
                Ok(body) => body,
                Err(err) => return IngressError::invalid(err.to_string()).response(metadata.route),
            };
            let body = match worker.prepare_request(body).await {
                Ok(body) => body,
                Err(err) => return IngressError::invalid(err.to_string()).response(metadata.route),
            };
            builder = builder.json(&body);
        } else {
            builder = builder
                .header(
                    CONTENT_TYPE,
                    request
                        .headers
                        .get(CONTENT_TYPE)
                        .cloned()
                        .unwrap_or_else(|| HeaderValue::from_static("application/json")),
                )
                .body(request.body.clone());
        }
        events::RequestSentEvent { url: worker.url() }.emit();
        let upstream = builder.send().await;
        events::RequestReceivedEvent {}.emit();
        let response = match upstream {
            Ok(response) => {
                super::comm::proxy_body::ProxyBody::response(response, worker, policy, load)
            }
            Err(err) => {
                worker.record_outcome(false);
                policy.on_request_complete(worker.url(), false);
                convert_reqwest_error(err)
            }
        };
        if response.status().is_server_error() {
            MeshMetrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(response.status()),
            );
        }
        response
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    if header_utils::should_forward_request_header(&name) {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
    ) -> Response {
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();
        let model = model_id.unwrap_or(UNKNOWN_MODEL_ID);
        let endpoint = route_to_endpoint(route);

        let observation = crate::observability::request::RequestMetrics::new(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            model,
            route,
            is_stream,
        );

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let res = self
                    .route_typed_request_once(headers, typed_req, route, model_id, is_stream, &text)
                    .await;

                // Need to be outside `route_typed_request_once` because that function has multiple return paths
                MeshMetrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    res.status().as_u16(),
                    extract_error_code_from_response(&res),
                );

                res
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                // Layer 3 worker metrics
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            // on_exhausted hook
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    endpoint,
                );
            },
        )
        .await;

        observation.wrap_response(response)
    }

    async fn route_typed_request_once<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let descriptor = RequestDescriptor {
            model_id,
            protocol: Some(Protocol::Http),
            text: Some(text),
            tokens: None,
            headers,
            stream: is_stream,
        };

        let (worker, policy_name) = match self.planner.plan(&descriptor).await {
            Ok(PlacementPlan::Single {
                worker,
                policy_name,
                ..
            }) => (worker, policy_name),
            Ok(PlacementPlan::Pair { .. }) => {
                error!(
                    function = "Router::route_typed_request_once",
                    "Planner returned Pair plan for regular HTTP router"
                );
                return error::internal_error(
                    "unexpected_pair_plan",
                    "Planner returned Pair plan for regular router",
                );
            }
            Err(err) => {
                return placement_err_to_response(err, model_id);
            }
        };

        MeshMetrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id.unwrap_or(UNKNOWN_MODEL_ID),
            policy_name,
        );

        let load_guard = ["cache_aware", "manual", "dp_sticky"]
            .contains(&policy_name)
            .then(|| WorkerLoadGuard::new(worker.clone(), headers));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();

        let response = self
            .send_typed_request(
                headers,
                typed_req,
                route,
                worker.url(),
                is_stream,
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        worker.record_outcome(status.is_success());

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            MeshMetrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    // Resource IDs have no model/owner hint, so probe eligible backend origins.
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
        query: Option<&str>,
    ) -> Response {
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }
        let api = super::ingress::EndpointSpec::find("/v1/responses")
            .expect("Responses endpoint is registered");
        let mut origins = BTreeMap::new();
        for worker in workers.into_iter().filter(|worker| {
            matches!(worker.connection_mode(), ConnectionMode::Http)
                && matches!(worker.worker_type(), WorkerType::Regular)
                && api.supports(worker.as_ref())
        }) {
            // An unhealthy worker can still own the requested resource.
            let key = worker.api_key().clone();
            match origins.entry(worker.base_url().trim_end_matches('/').to_owned()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(key);
                }
                std::collections::btree_map::Entry::Occupied(entry) => {
                    if entry.get() != &key {
                        return error::service_unavailable(
                            "conflicting_backend_credentials",
                            "Responses workers at the same backend URL must use the same credential",
                        );
                    }
                }
            }
        }
        if origins.is_empty() {
            return error::not_implemented(
                "unsupported_api",
                "no configured HTTP backend supports Responses resource operations",
            );
        }
        let headers = headers.cloned().unwrap_or_default();
        if origins.len() > 1
            && (headers.contains_key("authorization") || headers.contains_key("x-api-key"))
            && origins.values().any(Option::is_none)
        {
            return error::service_unavailable(
                "ambiguous_response_credentials",
                "Responses resource lookup cannot forward client credentials to multiple backends; configure a worker API key for each backend or use a single backend",
            );
        }

        // Validate every credential before sending any DELETE or cancel request.
        let mut requests = Vec::with_capacity(origins.len());
        for (base, key) in origins {
            let headers = match header_utils::inference_request_headers(
                &headers,
                "/v1/responses",
                key.as_deref(),
            ) {
                Ok(headers) => headers,
                Err(_) => {
                    return error::service_unavailable(
                        "invalid_backend_credentials",
                        "configured worker credential is not a valid HTTP header",
                    );
                }
            };
            let url = match query {
                Some(query) => format!("{base}/{endpoint}?{query}"),
                None => format!("{base}/{endpoint}"),
            };
            requests.push(self.client.request(method.clone(), url).headers(headers));
        }
        let futures = requests
            .into_iter()
            .enumerate()
            .map(|(index, request)| async move {
                (index, request.send().await.map_err(convert_reqwest_error))
            });
        let mut stream = stream::iter(futures).buffer_unordered(32);
        let mut best_error: Option<((u8, usize), Response)> = None;
        while let Some((index, result)) = stream.next().await {
            let response = match result {
                Ok(upstream) => {
                    let status = upstream.status();
                    let headers = header_utils::preserve_response_headers(upstream.headers());
                    // Stream retrieval events as they arrive.
                    let mut response = Response::new(Body::from_stream(upstream.bytes_stream()));
                    *response.status_mut() = status;
                    *response.headers_mut() = headers;
                    if status.is_success() {
                        return response;
                    }
                    response
                }
                Err(response) => response,
            };
            let status = response.status();
            let priority = if status.is_server_error() {
                0
            } else if status == StatusCode::NOT_FOUND {
                2
            } else {
                1
            };
            // Prefer server failures, then non-404 errors; ties follow URL order.
            let rank = (priority, index);
            if best_error
                .as_ref()
                .is_none_or(|(current, _)| rank < *current)
            {
                best_error = Some((rank, response));
            }
        }
        best_error
            .map(|(_, response)| response)
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // TODO (rui): Better accommodate to the Worker abstraction
    fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
        let parts: Vec<&str> = worker_url.split('@').collect();
        if parts.len() != 2 {
            return Err(format!("invalid worker_url format: {}", worker_url));
        }

        // Parse the second part (dp_rank) into an integer
        match parts[1].parse::<usize>() {
            Ok(dp_rank) => Ok((parts[0], dp_rank)),
            Err(_) => Err(format!(
                "failed to parse dp_rank from worker_url: {}",
                worker_url
            )),
        }
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker_url: &str,
        is_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        // Get the worker once and reuse for API key and load tracking
        let worker = self.worker_registry.get_by_url(worker_url);
        let api_key = worker.as_ref().and_then(|w| w.api_key().clone());

        // Static key string to avoid per-request allocations
        const DP_RANK_KEY: &str = "data_parallel_rank";

        let mut request_builder = if self.dp_aware {
            let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return error::internal_error(
                        "dp_rank_extraction_failed",
                        format!("Failed to extract dp_rank: {}", e),
                    );
                }
            };

            let mut json_val = match serde_json::to_value(typed_req) {
                Ok(j) => j,
                Err(e) => {
                    return error::bad_request(
                        "serialization_failed",
                        format!("Convert into serde_json::Value failed: {}", e),
                    );
                }
            };

            if let Some(map) = json_val.as_object_mut() {
                // Use static key string to avoid allocation
                map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
                // Only serialize if debug logging is enabled to avoid CPU overhead
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!(
                        "Modified request body: {}",
                        serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
                    );
                }
            } else {
                return error::bad_request(
                    "dp_rank_insertion_failed",
                    "Failed to insert the data_parallel_rank field into the request body",
                );
            }

            self.client
                .post(format!("{}{}", worker_url_prefix, route))
                .json(&json_val)
        } else {
            self.client
                .post(format!("{}{}", worker_url, route))
                .json(typed_req) // Use json() directly with typed request
        };

        if let Some(ref key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        if let Some(headers) = headers {
            for (name, value) in headers {
                if header_utils::should_forward_request_header(name.as_str())
                    && !(api_key.is_some() && (name == "authorization" || name == "x-api-key"))
                {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        } else {
            let response_headers = header_utils::preserve_response_headers(res.headers());
            let body = Body::from_stream(res.bytes_stream());

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            if let Some(guard) = load_guard {
                response = AttachedBody::wrap_response(response, guard);
            }
            response
        }
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn route_inference(
        &self,
        request: super::ingress::InferenceEnvelope,
        app: &Arc<AppContext>,
    ) -> Response {
        use super::ingress::IngressRouting;
        use crate::core::placement::traits::PolicySource;
        let routing = IngressRouting::new(app);
        let resources = routing.clone();
        let route = request.metadata.route;
        let (request, (metadata, tokens)) = match request
            .prepare(&app.prepare_pool, move |request, ctx| {
                resources.prepare(&request.parsed, ctx)
            })
            .await
        {
            Ok(prepared) => prepared,
            Err(err) => return err.response(route),
        };
        let candidates = match routing.candidates(&metadata, request.parsed.requires_state_domain())
        {
            Ok(workers) => workers,
            Err(err) => return err.response(metadata.route),
        };
        let planner = DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::with_candidates(
                self.worker_registry.clone(),
                candidates,
            )),
            self.policies.clone(),
        );
        let descriptor = RequestDescriptor {
            model_id: metadata.model.as_deref(),
            protocol: Some(Protocol::Http),
            text: Some(&metadata.text),
            tokens: tokens.as_deref(),
            headers: Some(&request.headers),
            stream: metadata.stream,
        };
        let policy = self.policies.regular_policy(metadata.model.as_deref());
        let observation = crate::observability::request::RequestMetrics::new(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metadata.model.as_deref().unwrap_or(UNKNOWN_MODEL_ID),
            metadata.route,
            metadata.stream,
        );
        let endpoint = route_to_endpoint(metadata.route);
        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            |_| async {
                let response = self
                    .send_inference_once(&request, &descriptor, &planner, policy.clone())
                    .await;
                MeshMetrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    response.status().as_u16(),
                    extract_error_code_from_response(&response),
                );
                response
            },
            // A failed Responses submission may already have created stored work.
            |response, _| {
                metadata.route != "/v1/responses" && is_retryable_status(response.status())
            },
            |delay, attempt| {
                MeshMetrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                MeshMetrics::record_worker_retry_backoff(attempt, delay);
            },
            || {
                MeshMetrics::record_worker_retries_exhausted(
                    metrics_labels::WORKER_REGULAR,
                    endpoint,
                )
            },
        )
        .await;
        observation.wrap_response(response)
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
            .await
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/responses", model_id)
            .await
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        query: Option<&str>,
    ) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_simple_request(headers, &endpoint, Method::GET, query)
            .await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_simple_request(headers, &endpoint, Method::POST, None)
            .await
    }

    async fn delete_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_simple_request(headers, &endpoint, Method::DELETE, None)
            .await
    }

    async fn list_response_input_items(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        query: Option<&str>,
    ) -> Response {
        let endpoint = format!("v1/responses/{}/input_items", response_id);
        self.route_simple_request(headers, &endpoint, Method::GET, query)
            .await
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;
    use crate::policies::PolicyRegistry;

    fn create_test_regular_router() -> Router {
        use crate::core::WorkerType;

        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .build();
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry.clone())),
            Arc::new(PolicyRegistryAdapter::new(policy_registry.clone())),
        ));

        Router {
            worker_registry,
            planner,
            policies: Arc::new(PolicyRegistryAdapter::new(policy_registry)),
            dp_aware: false,
            client: Client::new(),
            retry_config: RetryConfig::default(),
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_healthy(false);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }

    #[test]
    fn test_select_first_worker_all_unhealthy() {
        let router = create_test_regular_router();
        for w in router.worker_registry.get_all() {
            w.set_healthy(false);
        }
        // All workers are unhealthy -> should return Err
        let result = router.select_first_worker();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No workers"));
    }

    #[test]
    fn test_select_first_worker_empty_registry() {
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));
        let planner: Arc<dyn PdPlanner> = Arc::new(DefaultPlanner::new(
            Arc::new(WorkerRegistryAdapter::new(worker_registry.clone())),
            Arc::new(PolicyRegistryAdapter::new(policy_registry.clone())),
        ));
        let router = Router {
            worker_registry,
            planner,
            policies: Arc::new(PolicyRegistryAdapter::new(policy_registry)),
            dp_aware: false,
            client: Client::new(),
            retry_config: RetryConfig::default(),
        };
        let result = router.select_first_worker();
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_dp_rank_valid() {
        let (url, rank) = Router::extract_dp_rank("http://worker:8000@2").unwrap();
        assert_eq!(url, "http://worker:8000");
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_extract_dp_rank_zero() {
        let (url, rank) = Router::extract_dp_rank("http://worker:8000@0").unwrap();
        assert_eq!(url, "http://worker:8000");
        assert_eq!(rank, 0);
    }

    #[test]
    fn test_extract_dp_rank_no_at() {
        let result = Router::extract_dp_rank("http://worker:8000");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_dp_rank_invalid_number() {
        let result = Router::extract_dp_rank("http://worker:8000@abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_dp_rank_multiple_at() {
        let result = Router::extract_dp_rank("http://worker@8000@2");
        assert!(result.is_err());
    }

    #[test]
    fn test_router_type() {
        let router = create_test_regular_router();
        assert_eq!(router.router_type(), "regular");
    }

    #[test]
    fn test_convert_reqwest_error() {
        // Build a reqwest error via an invalid URL
        let err = Client::new().get("http://[invalid]").build().unwrap_err();
        let response = convert_reqwest_error(err);
        // Should produce an error response
        let status = response.status();
        assert!(status.is_client_error() || status.is_server_error());
    }
}
