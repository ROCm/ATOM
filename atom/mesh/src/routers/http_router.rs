use std::{collections::BTreeMap, sync::Arc};

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, Method, StatusCode},
    response::Response,
};
use futures_util::{stream, StreamExt};
use reqwest::Client;

use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{
        is_retryable_status,
        placement::{
            registry_adapters::PolicyRegistryAdapter,
            traits::PdPlanner,
            types::{PlacementPlan, Protocol, RequestDescriptor},
        },
        ConnectionMode, RetryExecutor, WorkerLoadGuard, WorkerRegistry, WorkerType,
        UNKNOWN_MODEL_ID,
    },
    observability::{
        events::{self, Event},
        metrics::{metrics_labels, MeshMetrics},
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
    policies: Arc<PolicyRegistryAdapter>,
    client: Client,
    retry_config: RetryConfig,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("client", &self.client)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policies: Arc::new(PolicyRegistryAdapter::new(ctx.policy_registry.clone())),
            client: ctx.client.clone(),
            retry_config: ctx.router_config.effective_retry_config(),
        })
    }

    async fn send_inference_once(
        &self,
        request: &super::ingress::HttpInferenceRequest,
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
            Err(err) => return err.response(metadata.route),
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
        builder = match super::ingress::dp_ranked_body(
            worker.as_ref(),
            metadata.route,
            request.dp_rank_body.as_ref(),
            &request.body,
        ) {
            Ok(Some(body)) => builder
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .body(body),
            Ok(None) => builder
                .header(
                    CONTENT_TYPE,
                    request
                        .headers
                        .get(CONTENT_TYPE)
                        .cloned()
                        .unwrap_or_else(|| HeaderValue::from_static("application/json")),
                )
                .body(request.body.clone()),
            Err(err) => return err.response(metadata.route),
        };
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

        // Match inference: configured worker keys override client credentials;
        // eligible backends without a worker key receive the client credentials.
        // Validate every credential before sending any DELETE or cancel request.
        let mut requests = Vec::with_capacity(origins.len());
        for (base, key) in origins {
            let headers = match header_utils::inference_request_headers(
                &headers,
                "/v1/responses",
                key.as_deref(),
            ) {
                Ok(headers) => headers,
                Err(err) => return error::create_error(err.status, err.code, err.message),
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
                // A transport failure means this backend could not answer; keep
                // its internal URL out of the client-visible error.
                let result = request.send().await.map_err(|_| {
                    error::bad_gateway(
                        "backend_unreachable",
                        "a backend that may own this response could not be reached",
                    )
                });
                (index, result)
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
            // A backend without these routes says nothing about the resource.
            let route_unsupported =
                status == StatusCode::METHOD_NOT_ALLOWED || status == StatusCode::NOT_IMPLEMENTED;
            let priority = if route_unsupported {
                4
            } else if status.is_client_error() && status != StatusCode::NOT_FOUND {
                0
            } else if status.is_server_error() {
                1
            } else if status == StatusCode::NOT_FOUND {
                3
            } else {
                2
            };
            // Preserve actionable client errors over failures at other backends.
            // An unreachable or failing backend may own the resource, so its
            // error outranks 404; ties follow URL order.
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
}

const CONNECTION_FAILED_CODE: &str = "call_upstream_connection_failed";

/// Mesh-local errors caused by configuration or the request itself; retrying
/// another worker cannot change the outcome.
const NON_RETRYABLE_ERROR_CODES: &[&str] = &[
    "invalid_backend_credentials",
    "worker_request_preparation_failed",
    "invalid_request",
];

/// Decide whether an inference attempt should be retried on another worker.
pub(super) fn should_retry_inference(route: &str, response: &Response) -> bool {
    let code = extract_error_code_from_response(response);
    if NON_RETRYABLE_ERROR_CODES.contains(&code) {
        return false;
    }
    // A Responses submission that reached a backend may already have created
    // stored work, so only fail over when the connection was never made.
    if route == "/v1/responses" {
        return code == CONNECTION_FAILED_CODE;
    }
    is_retryable_status(response.status())
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    // Connection failures are also request errors; classify them first so
    // callers can tell that nothing reached the backend.
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_connect() {
        (StatusCode::INTERNAL_SERVER_ERROR, CONNECTION_FAILED_CODE)
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
        use crate::core::placement::traits::PolicySource;
        let route = request.metadata.route;
        let routing = super::ingress::IngressRouting::new(app);
        let placement = match request
            .prepare_placement(app, &routing, self.policies.clone())
            .await
        {
            Ok(placement) => placement,
            Err(err) => return err.response(route),
        };
        let (metadata, tokens, planner) = (placement.metadata, placement.tokens, placement.planner);
        let request = placement.request.into_http_request();
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
            |response, _| should_retry_inference(metadata.route, response),
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

    #[test]
    fn retries_skip_configuration_errors_and_responses_only_fail_over_unsent() {
        let mesh_error =
            |status, code: &str| error::create_error(status, code.to_owned(), "message");
        for &code in NON_RETRYABLE_ERROR_CODES {
            let response = mesh_error(StatusCode::SERVICE_UNAVAILABLE, code);
            assert!(
                !should_retry_inference("/v1/chat/completions", &response),
                "{code}"
            );
        }
        let unsent = mesh_error(StatusCode::INTERNAL_SERVER_ERROR, CONNECTION_FAILED_CODE);
        let backend_failure = mesh_error(StatusCode::BAD_GATEWAY, "call_upstream_status_error");
        assert!(should_retry_inference("/v1/chat/completions", &unsent));
        assert!(should_retry_inference(
            "/v1/chat/completions",
            &backend_failure
        ));
        assert!(should_retry_inference("/v1/responses", &unsent));
        assert!(!should_retry_inference("/v1/responses", &backend_failure));
    }

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

        Router {
            worker_registry,
            policies: Arc::new(PolicyRegistryAdapter::new(policy_registry)),
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
        let router = Router {
            worker_registry,
            policies: Arc::new(PolicyRegistryAdapter::new(policy_registry)),
            client: Client::new(),
            retry_config: RetryConfig::default(),
        };
        let result = router.select_first_worker();
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
