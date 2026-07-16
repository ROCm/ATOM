//! Mesh metrics recorder facade.
//!
//! This module owns the `metrics` crate macro calls for Mesh self metrics. Call
//! sites pass semantic values such as worker type, endpoint, and model; metric
//! names stay centralized in `schema::names` to avoid drift between recorder,
//! inventory, and Prometheus `describe_*` registration.

use std::{borrow::Cow, sync::Arc, time::Duration};

use metrics::{counter, gauge, histogram};

use super::schema::{intern_string, labels as metrics_labels, names, status_code_to_cow};

/// Public metrics recording facade used by business paths.
///
/// Design principles for low overhead:
/// - Dynamic labels use string interning (single allocation per unique value).
/// - Static labels use the metrics crate's internal caching.
pub struct MeshMetrics;

/// Parameters for recording streaming metrics.
pub struct StreamingMetricsParams<'a> {
    /// Router type label (e.g., "grpc", "http")
    pub router_type: &'static str,
    /// Backend type label (e.g., "regular", "pd")
    pub backend_type: &'static str,
    /// Model identifier (will be converted to owned String for metrics)
    pub model_id: &'a str,
    /// Endpoint label (e.g., "chat", "generate")
    pub endpoint: &'static str,
    /// Time to first token (None if no tokens were generated)
    pub ttft: Option<Duration>,
    /// Total generation time
    pub generation_duration: Duration,
    /// Input token count (None for endpoints that don't track this)
    pub input_tokens: Option<u64>,
    /// Output token count
    pub output_tokens: u64,
}

impl MeshMetrics {
    /// Record an HTTP request.
    /// Here we want a metric to directly reflect user's experience ("I am sending a request")
    /// when viewing the router as a blackbox, and is bumped immediately when the request arrives.
    pub fn record_http_request(method: &'static str, path: &str) {
        let path_interned = intern_string(path);
        counter!(
            names::HTTP_REQUESTS_TOTAL,
            "method" => method,
            "path" => path_interned,
        )
        .increment(1);
    }

    /// Record HTTP request duration.
    /// For best performance, pass static strings for method.
    pub fn record_http_duration(method: &'static str, path: &str, duration: Duration) {
        let path_interned = intern_string(path);
        histogram!(
            names::HTTP_REQUEST_DURATION_SECONDS,
            "method" => method,
            "path" => path_interned
        )
        .record(duration.as_secs_f64());
    }

    /// Set active HTTP connections count.
    pub fn set_http_connections_active(count: usize) {
        gauge!(names::HTTP_CONNECTIONS_ACTIVE).set(count as f64);
    }

    /// Record HTTP response.
    pub fn record_http_response(status_code: u16, error_code: &str) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            names::HTTP_RESPONSES_TOTAL,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    /// Record rate limit decision.
    pub fn record_http_rate_limit(result: &'static str) {
        counter!(
            names::HTTP_RATE_LIMIT_TOTAL,
            "result" => result
        )
        .increment(1);
    }

    /// Record a routed request.
    ///
    /// Uses string interning for model_id to avoid repeated allocations.
    ///
    /// # Arguments
    /// * `streaming` - Use `bool_to_static_str(request.stream)` or the constants.
    pub fn record_router_request(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        streaming: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            names::ROUTER_REQUESTS_TOTAL,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "streaming" => streaming
        )
        .increment(1);
    }

    /// Record router request duration.
    /// Uses string interning for model_id.
    pub fn record_router_duration(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            names::ROUTER_REQUEST_DURATION_SECONDS,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record a router error.
    /// Uses string interning for model_id.
    pub fn record_router_error(
        router_type: &'static str,
        backend_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        endpoint: &'static str,
        error_type: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            names::ROUTER_REQUEST_ERRORS_TOTAL,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "endpoint" => endpoint,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record pipeline stage duration (gRPC only).
    /// All labels are static, so this is very fast.
    pub fn record_router_stage_duration(
        router_type: &'static str,
        stage: &'static str,
        duration: Duration,
    ) {
        histogram!(
            names::ROUTER_STAGE_DURATION_SECONDS,
            "router_type" => router_type,
            "stage" => stage
        )
        .record(duration.as_secs_f64());
    }

    /// Record upstream backend response.
    /// Uses static strings for common status codes and interning for error_code.
    pub fn record_router_upstream_response(
        router_type: &'static str,
        status_code: u16,
        error_code: &str,
    ) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            names::ROUTER_UPSTREAM_RESPONSES_TOTAL,
            "router_type" => router_type,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    /// Record time to first token.
    /// Uses string interning for model_id.
    pub fn record_router_ttft(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            names::ROUTER_TTFT_SECONDS,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record time per output token.
    pub fn record_router_tpot(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            names::ROUTER_TPOT_SECONDS,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record tokens processed.
    pub fn record_router_tokens(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        token_type: &'static str,
        count: u64,
    ) {
        let model = intern_string(model_id);
        counter!(
            names::ROUTER_TOKENS_TOTAL,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => token_type
        )
        .increment(count);
    }

    /// Record total generation duration.
    /// Uses string interning for model_id.
    pub fn record_router_generation_duration(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            names::ROUTER_GENERATION_DURATION_SECONDS,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record all streaming metrics in a single batch call.
    ///
    /// This consolidates TTFT, TPOT, generation duration, and token metrics
    /// into one function, handling TPOT calculation internally.
    pub fn record_streaming_metrics(params: StreamingMetricsParams<'_>) {
        let StreamingMetricsParams {
            router_type,
            backend_type,
            model_id,
            endpoint,
            ttft,
            generation_duration,
            input_tokens,
            output_tokens,
        } = params;

        let model = intern_string(model_id);

        if let Some(ttft_duration) = ttft {
            histogram!(
                names::ROUTER_TTFT_SECONDS,
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint
            )
            .record(ttft_duration.as_secs_f64());

            if output_tokens > 1 {
                let time_after_first = generation_duration.saturating_sub(ttft_duration);
                let tpot = time_after_first / (output_tokens as u32 - 1);
                histogram!(
                    names::ROUTER_TPOT_SECONDS,
                    "router_type" => router_type,
                    "backend_type" => backend_type,
                    "model" => Arc::clone(&model),
                    "endpoint" => endpoint
                )
                .record(tpot.as_secs_f64());
            }
        }

        histogram!(
            names::ROUTER_GENERATION_DURATION_SECONDS,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => Arc::clone(&model),
            "endpoint" => endpoint
        )
        .record(generation_duration.as_secs_f64());

        if let Some(input) = input_tokens {
            counter!(
                names::ROUTER_TOKENS_TOTAL,
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint,
                "token_type" => metrics_labels::TOKEN_INPUT
            )
            .increment(input);
        }

        counter!(
            names::ROUTER_TOKENS_TOTAL,
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => metrics_labels::TOKEN_OUTPUT
        )
        .increment(output_tokens);
    }

    /// Set worker pool size.
    pub fn set_worker_pool_size(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        size: usize,
    ) {
        let model = intern_string(model_id);
        gauge!(
            names::WORKER_POOL_SIZE,
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model
        )
        .set(size as f64);
    }

    /// Set active worker connections.
    pub fn set_worker_connections_active(
        worker_type: &'static str,
        connection_mode: &'static str,
        count: usize,
    ) {
        gauge!(
            names::WORKER_CONNECTIONS_ACTIVE,
            "worker_type" => worker_type,
            "connection_mode" => connection_mode
        )
        .set(count as f64);
    }

    /// Record health check result.
    pub fn record_worker_health_check(worker_type: &'static str, result: &'static str) {
        counter!(
            names::WORKER_HEALTH_CHECKS_TOTAL,
            "worker_type" => worker_type,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker selection.
    pub fn record_worker_selection(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        policy: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            names::WORKER_SELECTION_TOTAL,
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "policy" => policy
        )
        .increment(1);
    }

    /// Record worker error.
    pub fn record_worker_error(
        worker_type: &'static str,
        connection_mode: &'static str,
        error_type: &'static str,
    ) {
        counter!(
            names::WORKER_ERRORS_TOTAL,
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record manual policy execution branch for routing decisions.
    pub fn record_worker_manual_policy_branch(branch: &'static str) {
        counter!(
            names::MANUAL_POLICY_BRANCH_TOTAL,
            "branch" => branch
        )
        .increment(1);
    }

    /// Set manual policy cache entries count.
    pub fn set_manual_policy_cache_entries(count: usize) {
        gauge!(names::MANUAL_POLICY_CACHE_ENTRIES).set(count as f64);
    }

    /// Record prefix hash policy execution branch for routing decisions.
    pub fn record_worker_prefix_hash_policy_branch(branch: &'static str) {
        counter!(
            names::PREFIX_HASH_POLICY_BRANCH_TOTAL,
            "branch" => branch
        )
        .increment(1);
    }

    /// Set running requests per worker.
    pub fn set_worker_requests_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            names::WORKER_REQUESTS_ACTIVE,
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set active routing keys per worker.
    pub fn set_worker_routing_keys_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            names::WORKER_ROUTING_KEYS_ACTIVE,
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set worker health status.
    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        let worker_interned = intern_string(worker_url);
        gauge!(
            names::WORKER_HEALTH,
            "worker" => worker_interned
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    /// Set circuit breaker state (0=closed, 1=open, 2=half_open).
    pub fn set_worker_cb_state(worker: &str, state_code: u8) {
        let worker_interned = intern_string(worker);
        gauge!(
            names::WORKER_CB_STATE,
            "worker" => worker_interned
        )
        .set(state_code as f64);
    }

    /// Record circuit breaker state transition.
    pub fn record_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            names::WORKER_CB_TRANSITIONS_TOTAL,
            "worker" => worker_interned,
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    /// Record circuit breaker outcome.
    pub fn record_worker_cb_outcome(worker: &str, outcome: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            names::WORKER_CB_OUTCOMES_TOTAL,
            "worker" => worker_interned,
            "outcome" => outcome
        )
        .increment(1);
    }

    /// Set circuit breaker consecutive failures.
    pub fn set_worker_cb_consecutive_failures(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            names::WORKER_CB_CONSECUTIVE_FAILURES,
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set circuit breaker consecutive successes.
    pub fn set_worker_cb_consecutive_successes(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            names::WORKER_CB_CONSECUTIVE_SUCCESSES,
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Record retry attempt.
    pub fn record_worker_retry(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            names::WORKER_RETRIES_TOTAL,
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retries exhausted.
    pub fn record_worker_retries_exhausted(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            names::WORKER_RETRIES_EXHAUSTED_TOTAL,
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retry backoff duration.
    pub fn record_worker_retry_backoff(attempt: u32, duration: Duration) {
        let attempt_str: Cow<'static, str> = match attempt {
            1 => Cow::Borrowed("1"),
            2 => Cow::Borrowed("2"),
            3 => Cow::Borrowed("3"),
            4 => Cow::Borrowed("4"),
            5 => Cow::Borrowed("5"),
            _ => Cow::Owned(attempt.to_string()),
        };
        histogram!(
            names::WORKER_RETRY_BACKOFF_SECONDS,
            "attempt" => attempt_str
        )
        .record(duration.as_secs_f64());
    }

    /// Record worker registration attempt.
    pub fn record_discovery_registration(source: &'static str, result: &'static str) {
        counter!(
            names::DISCOVERY_REGISTRATIONS_TOTAL,
            "source" => source,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker deregistration.
    pub fn record_discovery_deregistration(source: &'static str, reason: &'static str) {
        counter!(
            names::DISCOVERY_DEREGISTRATIONS_TOTAL,
            "source" => source,
            "reason" => reason
        )
        .increment(1);
    }

    /// Record discovery sync duration.
    pub fn record_discovery_sync_duration(source: &'static str, duration: Duration) {
        histogram!(
            names::DISCOVERY_SYNC_DURATION_SECONDS,
            "source" => source
        )
        .record(duration.as_secs_f64());
    }

    /// Set workers discovered count.
    pub fn set_discovery_workers_discovered(source: &'static str, count: usize) {
        gauge!(
            names::DISCOVERY_WORKERS_DISCOVERED,
            "source" => source
        )
        .set(count as f64);
    }

    /// Record database operation.
    pub fn record_db_operation(
        storage_type: &'static str,
        operation: &'static str,
        result: &'static str,
    ) {
        counter!(
            names::DB_OPERATIONS_TOTAL,
            "storage_type" => storage_type,
            "operation" => operation,
            "result" => result
        )
        .increment(1);
    }

    /// Record database operation duration.
    pub fn record_db_operation_duration(
        storage_type: &'static str,
        operation: &'static str,
        duration: Duration,
    ) {
        histogram!(
            names::DB_OPERATION_DURATION_SECONDS,
            "storage_type" => storage_type,
            "operation" => operation
        )
        .record(duration.as_secs_f64());
    }

    /// Set active database connections.
    pub fn set_db_connections_active(storage_type: &'static str, count: usize) {
        gauge!(
            names::DB_CONNECTIONS_ACTIVE,
            "storage_type" => storage_type
        )
        .set(count as f64);
    }

    /// Record item stored.
    pub fn increment_db_items_stored(storage_type: &'static str) {
        counter!(
            names::DB_ITEMS_STORED,
            "storage_type" => storage_type
        )
        .increment(1);
    }

    /// Mark worker-scoped gauges as inactive when a worker leaves the registry.
    ///
    /// The metrics crate cannot delete a previously emitted label set, so the
    /// cleanup path writes neutral/sentinel values while preserving existing
    /// label names and worker URL labels.
    pub fn remove_worker_metrics(worker_url: &str) {
        let worker = intern_string(worker_url);

        gauge!(names::WORKER_CB_CONSECUTIVE_FAILURES, "worker" => Arc::clone(&worker)).set(0.0);
        gauge!(names::WORKER_CB_CONSECUTIVE_SUCCESSES, "worker" => Arc::clone(&worker)).set(0.0);
        gauge!(names::WORKER_REQUESTS_ACTIVE, "worker" => Arc::clone(&worker)).set(0.0);

        // Zero for these metrics have special valid meaning, so mark removed
        // workers with -1 until metrics-rs supports deleting label sets.
        gauge!(names::WORKER_CB_STATE, "worker" => Arc::clone(&worker)).set(-1.0);
        gauge!(names::WORKER_HEALTH, "worker" => worker).set(-1.0);
    }
}
