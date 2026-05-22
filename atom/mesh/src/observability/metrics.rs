use std::{
    borrow::Cow,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    sync::Arc,
    time::Duration,
};

use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder};

use super::{config::default_duration_buckets, schema};

pub use super::config::PrometheusConfig;
pub(crate) use super::schema::intern_string;
pub use super::schema::{labels as metrics_labels, status_code_to_cow};

pub(crate) fn init_metrics() {
    schema::describe_all_metrics();
}

pub fn start_prometheus(config: PrometheusConfig) {
    init_metrics();

    let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
    let duration_bucket: Vec<f64> = config
        .duration_buckets
        .unwrap_or_else(default_duration_buckets);

    let ip_addr: IpAddr = config
        .host
        .parse()
        .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
    let socket_addr = SocketAddr::new(ip_addr, config.port);

    PrometheusBuilder::new()
        .with_http_listener(socket_addr)
        .upkeep_timeout(Duration::from_secs(5 * 60))
        .set_buckets_for_metric(duration_matcher, &duration_bucket)
        .expect("failed to set duration bucket")
        .install()
        .expect("failed to install Prometheus metrics exporter");
}

/// Mesh Metrics helper struct for the new layered metrics architecture.
///
/// Design principles for low overhead:
/// - Dynamic labels use string interning (single allocation per unique value)
/// - Static labels use the metrics crate's internal caching
pub struct Metrics;

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

impl Metrics {
    /// Record an HTTP request.
    /// Here we want a metric to directly reflect user's experience ("I am sending a request")
    /// when viewing the router as a blackbox, and is bumped immediately when the request arrives.
    pub fn record_http_request(method: &'static str, path: &str) {
        let path_interned = intern_string(path);
        counter!(
            "mesh_http_requests_total",
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
            "mesh_http_request_duration_seconds",
            "method" => method,
            "path" => path_interned
        )
        .record(duration.as_secs_f64());
    }

    /// Set active HTTP connections count
    pub fn set_http_connections_active(count: usize) {
        gauge!("mesh_http_connections_active").set(count as f64);
    }

    /// Record HTTP response.
    pub fn record_http_response(status_code: u16, error_code: &str) {
        let status_str: Cow<'static, str> = status_code_to_cow(status_code);
        let error_interned = intern_string(error_code);
        counter!(
            "mesh_http_responses_total",
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    /// Record rate limit decision.
    pub fn record_http_rate_limit(result: &'static str) {
        counter!(
            "mesh_http_rate_limit_total",
            "result" => result
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router metrics
    // ========================================================================

    /// Record a routed request.
    ///
    /// Uses string interning for model_id to avoid repeated allocations.
    ///
    /// # Arguments
    /// * `streaming` - Use `bool_to_static_str(request.stream)` or the constants
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
            "mesh_router_requests_total",
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
            "mesh_router_request_duration_seconds",
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
            "mesh_router_request_errors_total",
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
            "mesh_router_stage_duration_seconds",
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
            "mesh_router_upstream_responses_total",
            "router_type" => router_type,
            "status_code" => status_str,
            "error_code" => error_interned
        )
        .increment(1);
    }

    // ========================================================================
    // Layer 2: Router inference metrics (gRPC only)
    // ========================================================================

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
            "mesh_router_ttft_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record time per output token
    pub fn record_router_tpot(
        router_type: &'static str,
        backend_type: &'static str,
        model_id: &str,
        endpoint: &'static str,
        duration: Duration,
    ) {
        let model = intern_string(model_id);
        histogram!(
            "mesh_router_tpot_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint
        )
        .record(duration.as_secs_f64());
    }

    /// Record tokens processed
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
            "mesh_router_tokens_total",
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
            "mesh_router_generation_duration_seconds",
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

        // Intern model string once - Arc::clone is just a ref count increment
        let model = intern_string(model_id);

        // TTFT and TPOT (only if we have a first token time)
        if let Some(ttft_duration) = ttft {
            histogram!(
                "mesh_router_ttft_seconds",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint
            )
            .record(ttft_duration.as_secs_f64());

            // TPOT - only meaningful with >1 output token
            if output_tokens > 1 {
                let time_after_first = generation_duration.saturating_sub(ttft_duration);
                let tpot = time_after_first / (output_tokens as u32 - 1);
                histogram!(
                    "mesh_router_tpot_seconds",
                    "router_type" => router_type,
                    "backend_type" => backend_type,
                    "model" => Arc::clone(&model),
                    "endpoint" => endpoint
                )
                .record(tpot.as_secs_f64());
            }
        }

        // Generation duration (always recorded)
        histogram!(
            "mesh_router_generation_duration_seconds",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => Arc::clone(&model),
            "endpoint" => endpoint
        )
        .record(generation_duration.as_secs_f64());

        // Input tokens (if available)
        if let Some(input) = input_tokens {
            counter!(
                "mesh_router_tokens_total",
                "router_type" => router_type,
                "backend_type" => backend_type,
                "model" => Arc::clone(&model),
                "endpoint" => endpoint,
                "token_type" => metrics_labels::TOKEN_INPUT
            )
            .increment(input);
        }

        // Output tokens (always recorded - move model on final use)
        counter!(
            "mesh_router_tokens_total",
            "router_type" => router_type,
            "backend_type" => backend_type,
            "model" => model,
            "endpoint" => endpoint,
            "token_type" => metrics_labels::TOKEN_OUTPUT
        )
        .increment(output_tokens);
    }

    // ========================================================================
    // Layer 3: Worker metrics
    // ========================================================================

    /// Set worker pool size
    pub fn set_worker_pool_size(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        size: usize,
    ) {
        let model = intern_string(model_id);
        gauge!(
            "mesh_worker_pool_size",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model
        )
        .set(size as f64);
    }

    /// Set active worker connections
    pub fn set_worker_connections_active(
        worker_type: &'static str,
        connection_mode: &'static str,
        count: usize,
    ) {
        gauge!(
            "mesh_worker_connections_active",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode
        )
        .set(count as f64);
    }

    /// Record health check result
    pub fn record_worker_health_check(worker_type: &'static str, result: &'static str) {
        counter!(
            "mesh_worker_health_checks_total",
            "worker_type" => worker_type,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker selection
    pub fn record_worker_selection(
        worker_type: &'static str,
        connection_mode: &'static str,
        model_id: &str,
        policy: &'static str,
    ) {
        let model = intern_string(model_id);
        counter!(
            "mesh_worker_selection_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "model" => model,
            "policy" => policy
        )
        .increment(1);
    }

    /// Record worker error
    pub fn record_worker_error(
        worker_type: &'static str,
        connection_mode: &'static str,
        error_type: &'static str,
    ) {
        counter!(
            "mesh_worker_errors_total",
            "worker_type" => worker_type,
            "connection_mode" => connection_mode,
            "error_type" => error_type
        )
        .increment(1);
    }

    /// Record manual policy execution branch for routing decisions
    pub fn record_worker_manual_policy_branch(branch: &'static str) {
        counter!(
            "mesh_manual_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Set manual policy cache entries count
    pub fn set_manual_policy_cache_entries(count: usize) {
        gauge!("mesh_manual_policy_cache_entries").set(count as f64);
    }

    /// Record prefix hash policy execution branch for routing decisions
    pub fn record_worker_prefix_hash_policy_branch(branch: &'static str) {
        counter!(
            "mesh_prefix_hash_policy_branch_total",
            "branch" => branch
        )
        .increment(1);
    }

    /// Set running requests per worker
    pub fn set_worker_requests_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            "mesh_worker_requests_active",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set active routing keys per worker
    pub fn set_worker_routing_keys_active(worker: &str, count: usize) {
        let worker_interned = intern_string(worker);
        gauge!(
            "mesh_worker_routing_keys_active",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set worker health status
    pub fn set_worker_health(worker_url: &str, healthy: bool) {
        let worker_interned = intern_string(worker_url);
        gauge!(
            "mesh_worker_health",
            "worker" => worker_interned
        )
        .set(if healthy { 1.0 } else { 0.0 });
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (circuit breaker)
    // ========================================================================

    /// Set circuit breaker state (0=closed, 1=open, 2=half_open)
    pub fn set_worker_cb_state(worker: &str, state_code: u8) {
        let worker_interned = intern_string(worker);
        gauge!(
            "mesh_worker_cb_state",
            "worker" => worker_interned
        )
        .set(state_code as f64);
    }

    /// Record circuit breaker state transition
    pub fn record_worker_cb_transition(worker: &str, from: &'static str, to: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "mesh_worker_cb_transitions_total",
            "worker" => worker_interned,
            "from" => from,
            "to" => to
        )
        .increment(1);
    }

    /// Record circuit breaker outcome
    pub fn record_worker_cb_outcome(worker: &str, outcome: &'static str) {
        let worker_interned = intern_string(worker);
        counter!(
            "mesh_worker_cb_outcomes_total",
            "worker" => worker_interned,
            "outcome" => outcome
        )
        .increment(1);
    }

    /// Set circuit breaker consecutive failures
    pub fn set_worker_cb_consecutive_failures(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            "mesh_worker_cb_consecutive_failures",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    /// Set circuit breaker consecutive successes
    pub fn set_worker_cb_consecutive_successes(worker: &str, count: u32) {
        let worker_interned = intern_string(worker);
        gauge!(
            "mesh_worker_cb_consecutive_successes",
            "worker" => worker_interned
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 3: Worker resilience metrics (retry)
    // ========================================================================

    /// Record retry attempt
    pub fn record_worker_retry(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "mesh_worker_retries_total",
            "worker_type" => worker_type,
            "endpoint" => endpoint
        )
        .increment(1);
    }

    /// Record retries exhausted
    pub fn record_worker_retries_exhausted(worker_type: &'static str, endpoint: &'static str) {
        counter!(
            "mesh_worker_retries_exhausted_total",
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
            "mesh_worker_retry_backoff_seconds",
            "attempt" => attempt_str
        )
        .record(duration.as_secs_f64());
    }

    // ========================================================================
    // Layer 4: Discovery metrics
    // ========================================================================

    /// Record worker registration attempt
    pub fn record_discovery_registration(source: &'static str, result: &'static str) {
        counter!(
            "mesh_discovery_registrations_total",
            "source" => source,
            "result" => result
        )
        .increment(1);
    }

    /// Record worker deregistration
    pub fn record_discovery_deregistration(source: &'static str, reason: &'static str) {
        counter!(
            "mesh_discovery_deregistrations_total",
            "source" => source,
            "reason" => reason
        )
        .increment(1);
    }

    /// Record discovery sync duration
    pub fn record_discovery_sync_duration(source: &'static str, duration: Duration) {
        histogram!(
            "mesh_discovery_sync_duration_seconds",
            "source" => source
        )
        .record(duration.as_secs_f64());
    }

    /// Set workers discovered count
    pub fn set_discovery_workers_discovered(source: &'static str, count: usize) {
        gauge!(
            "mesh_discovery_workers_discovered",
            "source" => source
        )
        .set(count as f64);
    }

    // ========================================================================
    // Layer 6: Database metrics
    // ========================================================================

    /// Record database operation
    pub fn record_db_operation(
        storage_type: &'static str,
        operation: &'static str,
        result: &'static str,
    ) {
        counter!(
            "mesh_db_operations_total",
            "storage_type" => storage_type,
            "operation" => operation,
            "result" => result
        )
        .increment(1);
    }

    /// Record database operation duration
    pub fn record_db_operation_duration(
        storage_type: &'static str,
        operation: &'static str,
        duration: Duration,
    ) {
        histogram!(
            "mesh_db_operation_duration_seconds",
            "storage_type" => storage_type,
            "operation" => operation
        )
        .record(duration.as_secs_f64());
    }

    /// Set active database connections
    pub fn set_db_connections_active(storage_type: &'static str, count: usize) {
        gauge!(
            "mesh_db_connections_active",
            "storage_type" => storage_type
        )
        .set(count as f64);
    }

    /// Record item stored
    pub fn increment_db_items_stored(storage_type: &'static str) {
        counter!(
            "mesh_db_items_stored",
            "storage_type" => storage_type
        )
        .increment(1);
    }

    // ========================================================================
    // Worker cleanup
    // ========================================================================

    pub fn remove_worker_metrics(worker_url: &str) {
        // Intern once, clone (cheap) for each metric
        let worker = intern_string(worker_url);

        gauge!("mesh_worker_cb_consecutive_failures", "worker" => Arc::clone(&worker)).set(0.0);
        gauge!("mesh_worker_cb_consecutive_successes", "worker" => Arc::clone(&worker)).set(0.0);
        gauge!("mesh_worker_requests_active", "worker" => Arc::clone(&worker)).set(0.0);

        // Zero for these metrics have special valid meaning, thus we set to -1 temporarily
        // (and will remove them completely after https://github.com/metrics-rs/metrics/issues/653)
        gauge!("mesh_worker_cb_state", "worker" => Arc::clone(&worker)).set(-1.0);
        gauge!("mesh_worker_health", "worker" => worker).set(-1.0);
    }
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;

    use super::super::schema::{
        bool_to_static_str, interner_size, method_to_static_str, status_code_to_static_str,
    };
    use super::*;

    #[test]
    fn test_prometheus_config_default() {
        let config = PrometheusConfig::default();
        assert_eq!(config.port, 29000);
        assert_eq!(config.host, "0.0.0.0");
    }

    #[test]
    fn test_prometheus_config_custom() {
        let config = PrometheusConfig {
            port: 8080,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };
        assert_eq!(config.port, 8080);
        assert_eq!(config.host, "127.0.0.1");
    }

    #[test]
    fn test_prometheus_config_clone() {
        let config = PrometheusConfig {
            port: 9090,
            host: "192.168.1.1".to_string(),
            duration_buckets: None,
        };
        let cloned = config.clone();
        assert_eq!(cloned.port, config.port);
        assert_eq!(cloned.host, config.host);
    }

    #[test]
    fn test_valid_ipv4_parsing() {
        let test_cases = vec!["127.0.0.1", "192.168.1.1", "0.0.0.0"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V4(_)));
        }
    }

    #[test]
    fn test_valid_ipv6_parsing() {
        let test_cases = vec!["::1", "2001:db8::1", "::"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            assert!(matches!(ip_addr, IpAddr::V6(_)));
        }
    }

    #[test]
    fn test_invalid_ip_parsing() {
        let test_cases = vec!["invalid", "256.256.256.256", "hostname"];

        for ip_str in test_cases {
            let config = PrometheusConfig {
                port: 29000,
                host: ip_str.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config
                .host
                .parse()
                .unwrap_or(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));

            assert_eq!(ip_addr, IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)));
        }
    }

    #[test]
    fn test_socket_addr_creation() {
        let test_cases = vec![("127.0.0.1", 8080), ("0.0.0.0", 29000), ("::1", 9090)];

        for (host, port) in test_cases {
            let config = PrometheusConfig {
                port,
                host: host.to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
            assert_eq!(socket_addr.ip().to_string(), host);
        }
    }

    #[test]
    fn test_socket_addr_with_different_ports() {
        let ports = vec![0, 80, 8080, 65535];

        for port in ports {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            let ip_addr: IpAddr = config.host.parse().unwrap();
            let socket_addr = SocketAddr::new(ip_addr, config.port);

            assert_eq!(socket_addr.port(), port);
        }
    }

    #[test]
    fn test_duration_bucket_coverage() {
        let test_cases: [(f64, &str); 7] = [
            (0.0005, "sub-millisecond"),
            (0.005, "5ms"),
            (0.05, "50ms"),
            (1.0, "1s"),
            (10.0, "10s"),
            (60.0, "1m"),
            (240.0, "4m"),
        ];

        let buckets: [f64; 20] = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        for (duration, label) in test_cases {
            let bucket_found = buckets
                .iter()
                .any(|&b| (b - duration).abs() < 0.0001 || b > duration);
            assert!(bucket_found, "No bucket found for {} ({})", duration, label);
        }
    }

    #[test]
    fn test_duration_suffix_matcher() {
        let matcher = Matcher::Suffix(String::from("duration_seconds"));

        let _matching_metrics = [
            "request_duration_seconds",
            "response_duration_seconds",
            "mesh_request_duration_seconds",
        ];

        let _non_matching_metrics = ["duration_total", "duration_seconds_total", "other_metric"];

        match matcher {
            Matcher::Suffix(suffix) => assert_eq!(suffix, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_prometheus_builder_configuration() {
        let _config = PrometheusConfig::default();

        let duration_matcher = Matcher::Suffix(String::from("duration_seconds"));
        let duration_bucket = [
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 45.0,
            60.0, 90.0, 120.0, 180.0, 240.0,
        ];

        assert_eq!(duration_bucket.len(), 20);

        match duration_matcher {
            Matcher::Suffix(s) => assert_eq!(s, "duration_seconds"),
            _ => panic!("Expected Suffix matcher"),
        }
    }

    #[test]
    fn test_upkeep_timeout_duration() {
        let timeout = Duration::from_secs(5 * 60);
        assert_eq!(timeout.as_secs(), 300);
    }

    #[test]
    fn test_custom_buckets_for_different_metrics() {
        let request_buckets = [0.001, 0.01, 0.1, 1.0, 10.0];
        let generate_buckets = [0.1, 0.5, 1.0, 5.0, 30.0, 60.0];

        assert_eq!(request_buckets.len(), 5);
        assert_eq!(generate_buckets.len(), 6);

        for i in 1..request_buckets.len() {
            assert!(request_buckets[i] > request_buckets[i - 1]);
        }

        for i in 1..generate_buckets.len() {
            assert!(generate_buckets[i] > generate_buckets[i - 1]);
        }
    }

    #[test]
    fn test_port_already_in_use() {
        let port = 29123;

        if let Ok(_listener) = TcpListener::bind(("127.0.0.1", port)) {
            let config = PrometheusConfig {
                port,
                host: "127.0.0.1".to_string(),
                duration_buckets: None,
            };

            assert_eq!(config.port, port);
        }
    }

    #[test]
    fn test_metrics_endpoint_accessibility() {
        let config = PrometheusConfig {
            port: 29000,
            host: "127.0.0.1".to_string(),
            duration_buckets: None,
        };

        let ip_addr: IpAddr = config.host.parse().unwrap();
        let socket_addr = SocketAddr::new(ip_addr, config.port);

        assert_eq!(socket_addr.to_string(), "127.0.0.1:29000");
    }

    // ========================================================================
    // String interning tests
    // ========================================================================

    #[test]
    fn test_intern_string_returns_same_arc() {
        let s1 = intern_string("test_model");
        let s2 = intern_string("test_model");

        // Should return the same Arc (pointer equality)
        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "test_model");
    }

    #[test]
    fn test_intern_string_different_strings() {
        let s1 = intern_string("model_a");
        let s2 = intern_string("model_b");

        // Different strings should have different Arcs
        assert!(!Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "model_a");
        assert_eq!(&*s2, "model_b");
    }

    #[test]
    fn test_intern_string_empty() {
        let s1 = intern_string("");
        let s2 = intern_string("");

        assert!(Arc::ptr_eq(&s1, &s2));
        assert_eq!(&*s1, "");
    }

    #[test]
    fn test_interner_size_grows() {
        let initial_size = interner_size();

        // Intern some unique strings
        let unique = format!("unique_test_string_{}", initial_size);
        intern_string(&unique);

        assert!(interner_size() > initial_size);
    }

    #[test]
    fn test_bool_to_static_str() {
        assert_eq!(bool_to_static_str(true), "true");
        assert_eq!(bool_to_static_str(false), "false");
    }

    #[test]
    fn test_status_code_to_static_str() {
        // Common codes should return static strings
        assert_eq!(status_code_to_static_str(200), Some("200"));
        assert_eq!(status_code_to_static_str(404), Some("404"));
        assert_eq!(status_code_to_static_str(500), Some("500"));

        // Uncommon codes should return None
        assert_eq!(status_code_to_static_str(418), None);
        assert_eq!(status_code_to_static_str(999), None);
    }

    #[test]
    fn test_status_code_to_cow() {
        // Common codes should be borrowed
        let cow_200 = status_code_to_cow(200);
        assert!(matches!(cow_200, Cow::Borrowed(_)));
        assert_eq!(cow_200, "200");

        // Uncommon codes should be owned
        let cow_418 = status_code_to_cow(418);
        assert!(matches!(cow_418, Cow::Owned(_)));
        assert_eq!(cow_418, "418");
    }

    #[test]
    fn test_method_to_static_str() {
        assert_eq!(method_to_static_str("GET"), "GET");
        assert_eq!(method_to_static_str("POST"), "POST");
        assert_eq!(method_to_static_str("UNKNOWN"), "OTHER");
    }
}
