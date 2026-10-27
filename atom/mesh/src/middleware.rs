use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::Response,
};
use tower::{Layer, Service};
use tower_http::trace::{MakeSpan, OnRequest, OnResponse, TraceLayer};
use tracing::{error, field::Empty, info, info_span, warn, Span};

pub use crate::core::token_bucket::TokenBucket;
use crate::{
    observability::{
        inflight_tracker::InFlightRequestTracker,
        metrics::{method_to_static_str, metrics_labels, normalize_path_for_metrics, MeshMetrics},
    },
    routers::comm::error::extract_error_code_from_response,
    server::AppState,
};

/// Extension type for storing request ID
#[derive(Clone, Debug)]
pub struct RequestId(pub String);

/// Tower Layer for request ID middleware
#[derive(Clone)]
pub struct RequestIdLayer {
    headers: Arc<Vec<String>>,
}

impl RequestIdLayer {
    pub fn new(headers: Vec<String>) -> Self {
        Self {
            headers: Arc::new(headers),
        }
    }
}

impl<S> Layer<S> for RequestIdLayer {
    type Service = RequestIdMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        RequestIdMiddleware {
            inner,
            headers: self.headers.clone(),
        }
    }
}

/// Tower Service for request ID middleware
#[derive(Clone)]
pub struct RequestIdMiddleware<S> {
    inner: S,
    headers: Arc<Vec<String>>,
}

impl<S> Service<Request> for RequestIdMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, mut req: Request) -> Self::Future {
        let request_id =
            crate::observability::request_id::resolve(&self.headers, req.uri().path(), |name| {
                req.headers()
                    .get(name)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned)
            });

        // Insert request ID into request extensions for other middleware/handlers to use
        req.extensions_mut().insert(RequestId(request_id.clone()));

        // Call the inner service
        let future = self.inner.call(req);

        Box::pin(async move {
            let mut response = future.await?;

            // Add request ID to response headers
            response.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_str(&request_id)
                    .unwrap_or_else(|_| HeaderValue::from_static("invalid-request-id")),
            );

            Ok(response)
        })
    }
}

/// Custom span maker that includes request ID
#[derive(Clone, Debug)]
pub struct RequestSpan;

impl<B> MakeSpan<B> for RequestSpan {
    fn make_span(&mut self, request: &Request<B>) -> Span {
        // Don't try to extract request ID here - it won't be available yet
        // The RequestIdLayer runs after TraceLayer creates the span
        info_span!(
            "http_request",
            method = %request.method(),
            uri = %request.uri(),
            version = ?request.version(),
            request_id = Empty,  // Will be set later
            status_code = Empty,
            latency = Empty,
            error = Empty,
            module = "mesh"
        )
    }
}

/// Custom on_request handler
#[derive(Clone, Debug)]
pub struct RequestLogger;

impl<B> OnRequest<B> for RequestLogger {
    fn on_request(&mut self, request: &Request<B>, span: &Span) {
        let _enter = span.enter();

        // Try to get the request ID from extensions
        // This will work if RequestIdLayer has already run
        if let Some(request_id) = request.extensions().get::<RequestId>() {
            span.record("request_id", request_id.0.as_str());
        }

        let method = method_to_static_str(request.method().as_str());
        let path = normalize_path_for_metrics(request.uri().path());
        MeshMetrics::record_http_request(method, &path);

        // Log the request start
        info!(
            target: "mesh::request",
            "started processing request"
        );
    }
}

/// Custom on_response handler
#[derive(Clone, Debug, Default)]
pub struct ResponseLogger;

impl<B> OnResponse<B> for ResponseLogger {
    fn on_response(self, response: &Response<B>, latency: Duration, span: &Span) {
        let status = response.status();
        let status_code = status.as_u16();

        let error_code = extract_error_code_from_response(response);

        // Layer 1: HTTP metrics
        MeshMetrics::record_http_response(status_code, error_code);

        // Record these in the span for structured logging/observability tools
        span.record("status_code", status_code);
        // Use microseconds as integer to avoid format! string allocation
        span.record("latency", latency.as_micros() as u64);

        // Log the response completion
        let _enter = span.enter();
        if status.is_server_error() {
            error!(
                target: "mesh::response",
                "request failed with server error"
            );
        } else if status.is_client_error() {
            warn!(
                target: "mesh::response",
                "request failed with client error"
            );
        } else {
            info!(
                target: "mesh::response",
                "finished processing request"
            );
        }
    }
}

/// Create a configured TraceLayer for HTTP logging
/// Note: Actual request/response logging with request IDs is done in RequestIdService
pub fn create_logging_layer() -> TraceLayer<
    tower_http::classify::SharedClassifier<tower_http::classify::ServerErrorsAsFailures>,
    RequestSpan,
    RequestLogger,
    ResponseLogger,
> {
    TraceLayer::new_for_http()
        .make_span_with(RequestSpan)
        .on_request(RequestLogger)
        .on_response(ResponseLogger)
}

/// Admission is acquired before invoking the handler and follows the response body.
/// Dropping either the handler future or the body returns all reserved resources.
pub async fn concurrency_limit_middleware(
    State(app_state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    match app_state.context.admission.acquire("http").await {
        Ok(lease) => {
            MeshMetrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_ALLOWED);
            let response = next.run(request).await;
            crate::core::AttachedBody::wrap_response(response, lease)
        }
        Err(error) => {
            MeshMetrics::record_http_rate_limit(metrics_labels::RATE_LIMIT_REJECTED);
            crate::routers::comm::error::create_error(
                StatusCode::from_u16(error.status).unwrap(),
                error.code,
                error.message,
            )
        }
    }
}

// ============================================================================
// HTTP Metrics Layer (Layer 1: Mesh metrics)
// ============================================================================

/// Global counter for active HTTP connections (handlers currently executing)
static ACTIVE_HTTP_CONNECTIONS: AtomicU64 = AtomicU64::new(0);

struct ActiveHttpGuard;
impl ActiveHttpGuard {
    fn new() -> Self {
        let active = ACTIVE_HTTP_CONNECTIONS.fetch_add(1, Ordering::Relaxed) + 1;
        MeshMetrics::set_http_connections_active(active as usize);
        Self
    }
}
impl Drop for ActiveHttpGuard {
    fn drop(&mut self) {
        let active = ACTIVE_HTTP_CONNECTIONS.fetch_sub(1, Ordering::Relaxed) - 1;
        MeshMetrics::set_http_connections_active(active as usize);
    }
}

/// Tower Layer for HTTP metrics collection (Mesh Layer 1 metrics)
#[derive(Clone)]
pub struct HttpMetricsLayer {
    tracker: Arc<InFlightRequestTracker>,
}

impl HttpMetricsLayer {
    pub fn new(tracker: Arc<InFlightRequestTracker>) -> Self {
        Self { tracker }
    }
}

impl<S> Layer<S> for HttpMetricsLayer {
    type Service = HttpMetricsMiddleware<S>;

    fn layer(&self, inner: S) -> Self::Service {
        HttpMetricsMiddleware {
            inner,
            in_flight_request_tracker: self.tracker.clone(),
        }
    }
}

/// Tower Service for HTTP metrics collection
#[derive(Clone)]
pub struct HttpMetricsMiddleware<S> {
    inner: S,
    in_flight_request_tracker: Arc<InFlightRequestTracker>,
}

impl<S> Service<Request> for HttpMetricsMiddleware<S>
where
    S: Service<Request, Response = Response> + Send + Clone + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future =
        Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request) -> Self::Future {
        // Convert method to static string to avoid allocation
        let method = method_to_static_str(req.method().as_str());
        let path = normalize_path_for_metrics(req.uri().path());
        let start = Instant::now();

        let mut inner = self.inner.clone();
        let in_flight_request_tracker = self.in_flight_request_tracker.clone();

        Box::pin(async move {
            // Increment inside async block - ensures no leak if future is dropped before polling
            let _active = ActiveHttpGuard::new();

            let guard = in_flight_request_tracker.track();

            // Capture result before decrementing to ensure decrement happens on error too
            let result = inner.call(req).await;

            drop(guard);

            let response = result?;

            let duration = start.elapsed();
            MeshMetrics::record_http_duration(method, &path, duration);

            Ok(response)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_path_no_ids() {
        // Common API paths should pass through unchanged
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions"),
            "/v1/chat/completions"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/completions"),
            "/v1/completions"
        );
        assert_eq!(normalize_path_for_metrics("/v1/models"), "/v1/models");
        assert_eq!(normalize_path_for_metrics("/health"), "/health");
    }

    #[test]
    fn test_normalize_path_with_prefixed_id() {
        // Prefixed IDs (resp_xxx, chatcmpl_xxx) should be normalized
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/resp_abc123def456"),
            "/v1/responses/{id}"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions/chatcmpl_abc123xyz"),
            "/v1/chat/completions/{id}"
        );
    }

    #[test]
    fn test_normalize_path_with_uuid() {
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/550e8400-e29b-41d4-a716-446655440000"),
            "/v1/responses/{id}"
        );
    }

    #[test]
    fn test_normalize_path_with_numeric_id() {
        assert_eq!(
            normalize_path_for_metrics("/v1/workers/12345"),
            "/v1/workers/{id}"
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use axum::{routing::get, Router};
    use futures_util::poll;
    use tower::ServiceExt;
    #[tokio::test]
    async fn http_cancellation_before_headers_and_while_queued_refunds_shared_admission() {
        let config = crate::config::RouterConfig {
            max_concurrent_requests: 1,
            queue_size: 2,
            ..Default::default()
        };
        let bucket = Arc::new(TokenBucket::new(1, 0));
        let context = Arc::new(
            crate::app_context::AppContextBuilder::from_config(config, 5)
                .await
                .unwrap()
                .rate_limiter(Some(bucket.clone()))
                .build()
                .unwrap(),
        );
        let router = Arc::new(
            crate::routers::http_router::Router::new(&context)
                .await
                .unwrap(),
        );
        let state = Arc::new(AppState {
            context: context.clone(),
            router,
            router_manager: None,
        });
        let app = Router::new()
            .route(
                "/pending",
                get(|| async { std::future::pending::<Response>().await }),
            )
            .route("/body", get(|| async { "payload" }))
            .layer(axum::middleware::from_fn_with_state(
                state,
                concurrency_limit_middleware,
            ));
        let request = |path: &str| Request::builder().uri(path).body(Body::empty()).unwrap();
        let mut running = Box::pin(app.clone().oneshot(request("/pending")));
        assert!(poll!(&mut running).is_pending());
        assert_eq!(bucket.available_tokens().await, 0.0);
        let mut queued = Box::pin(app.clone().oneshot(request("/pending")));
        assert!(poll!(&mut queued).is_pending());
        drop(queued);
        drop(running);
        assert_eq!(bucket.available_tokens().await, 1.0);
        // The returned body must hold the same lease until disposal.
        let response = app.oneshot(request("/body")).await.unwrap();
        assert_eq!(bucket.available_tokens().await, 0.0);
        let mut other_ingress = Box::pin(context.admission.acquire("ext_proc"));
        assert!(poll!(&mut other_ingress).is_pending());
        drop(response);
        drop(other_ingress.await.unwrap());
        assert_eq!(bucket.available_tokens().await, 1.0);
    }
}
