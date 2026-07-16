use std::{borrow::Cow, sync::Arc};

use dashmap::DashMap;
use metrics::{describe_counter, describe_gauge, describe_histogram};
use once_cell::sync::Lazy;

// =============================================================================
// STRING INTERNING
// =============================================================================
//
// Dynamic strings (model_id, worker URLs, paths) are interned to avoid repeated
// heap allocations. The interner uses Arc<str> which is cheap to clone and
// allows the metrics crate to store references without repeated allocations.
//
// Performance characteristics:
// - First occurrence: One allocation + DashMap insert
// - Subsequent occurrences: DashMap lookup + Arc::clone (very cheap)
// - Memory: Strings are never freed (acceptable for bounded label cardinality)

/// Global string interner for metric labels.
/// Uses DashMap for lock-free concurrent access.
static STRING_INTERNER: Lazy<DashMap<String, Arc<str>>> = Lazy::new(DashMap::new);

/// Intern a string, returning a cheaply-cloneable Arc<str>.
///
/// This function is designed for high-throughput scenarios where the same
/// strings (model IDs, worker URLs) appear repeatedly. The first call allocates,
/// subsequent calls just clone the Arc (very cheap - just a ref count increment).
pub(crate) fn intern_string(s: &str) -> Arc<str> {
    // Fast path: check if already interned
    if let Some(entry) = STRING_INTERNER.get(s) {
        return Arc::clone(entry.value());
    }

    // Slow path: intern the string
    // Use entry API to avoid TOCTOU race
    STRING_INTERNER
        .entry(s.to_string())
        .or_insert_with(|| Arc::from(s))
        .clone()
}

#[allow(dead_code)]
pub(crate) fn interner_size() -> usize {
    STRING_INTERNER.len()
}

/// Static string constants for boolean labels to avoid allocations.
pub const STREAMING_TRUE: &str = "true";
pub const STREAMING_FALSE: &str = "false";

pub const fn bool_to_static_str(b: bool) -> &'static str {
    if b {
        STREAMING_TRUE
    } else {
        STREAMING_FALSE
    }
}

/// Static lookup table for common HTTP status codes to avoid allocations.
/// Returns a static string for known codes, or None for unknown codes.
#[inline]
pub fn status_code_to_static_str(code: u16) -> Option<&'static str> {
    // Using a match with explicit arms is faster than a lookup table for this size
    match code {
        200 => Some("200"),
        201 => Some("201"),
        204 => Some("204"),
        400 => Some("400"),
        401 => Some("401"),
        403 => Some("403"),
        404 => Some("404"),
        408 => Some("408"),
        422 => Some("422"),
        429 => Some("429"),
        500 => Some("500"),
        502 => Some("502"),
        503 => Some("503"),
        504 => Some("504"),
        _ => None,
    }
}

/// Static HTTP method strings to avoid allocations on every request.
pub(crate) mod http_methods {
    pub const GET: &str = "GET";
    pub const POST: &str = "POST";
    pub const PUT: &str = "PUT";
    pub const DELETE: &str = "DELETE";
    pub const PATCH: &str = "PATCH";
    pub const HEAD: &str = "HEAD";
    pub const OPTIONS: &str = "OPTIONS";
}

/// Convert HTTP method to static string. Returns "OTHER" for unknown methods.
#[inline]
pub fn method_to_static_str(method: &str) -> &'static str {
    match method {
        "GET" => http_methods::GET,
        "POST" => http_methods::POST,
        "PUT" => http_methods::PUT,
        "DELETE" => http_methods::DELETE,
        "PATCH" => http_methods::PATCH,
        "HEAD" => http_methods::HEAD,
        "OPTIONS" => http_methods::OPTIONS,
        _ => "OTHER",
    }
}

/// Get status code as Cow - static for common codes, allocated for rare ones.
#[inline]
pub fn status_code_to_cow(code: u16) -> Cow<'static, str> {
    match status_code_to_static_str(code) {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(code.to_string()),
    }
}

/// Normalize HTTP paths for metrics without changing the current label contract.
///
/// This preserves the pre-refactor behavior of middleware path normalization:
/// common stable paths pass through unchanged, and dynamic path segments after
/// the second segment are replaced with `{id}`.
pub fn normalize_path_for_metrics(path: &str) -> String {
    let bytes = path.as_bytes();
    let mut segment_start = 0;
    let mut segment_idx = 0;
    let mut result: Option<String> = None;

    for (pos, &b) in bytes.iter().enumerate() {
        if b == b'/' || pos == bytes.len() - 1 {
            let segment_end = if b == b'/' { pos } else { pos + 1 };
            let segment = &path[segment_start..segment_end];

            if segment_idx > 2 && !segment.is_empty() && is_dynamic_id(segment) {
                let result = result.get_or_insert_with(|| {
                    let mut s = String::with_capacity(path.len());
                    s.push_str(&path[..segment_start]);
                    s
                });
                result.push_str("{id}");
            } else if let Some(ref mut r) = result {
                r.push_str(segment);
            }

            if b == b'/' {
                if let Some(ref mut r) = result {
                    r.push('/');
                }
                segment_start = pos + 1;
                segment_idx += 1;
            }
        }
    }

    result.unwrap_or_else(|| path.to_owned())
}

/// Check if segment looks like a dynamic ID (prefixed ID, UUID, or numeric).
#[inline]
fn is_dynamic_id(s: &str) -> bool {
    // Prefixed IDs: resp_xxx, chatcmpl_xxx (len > 10 with underscore)
    if s.len() > 10 && s.contains('_') {
        return true;
    }
    // UUIDs: 32+ hex chars with dashes
    if s.len() >= 32 && s.bytes().all(|b| b.is_ascii_hexdigit() || b == b'-') {
        return true;
    }
    // Numeric IDs
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Label constants for consistent metric labeling.
pub mod labels {
    // Router types
    pub const ROUTER_HTTP: &str = "http";
    pub const ROUTER_GRPC: &str = "grpc";

    // Backend types
    pub const BACKEND_REGULAR: &str = "regular";
    pub const BACKEND_PD: &str = "pd";
    // Connection modes
    pub const CONNECTION_HTTP: &str = "http";
    pub const CONNECTION_GRPC: &str = "grpc";

    // Endpoints
    pub const ENDPOINT_CHAT: &str = "chat";
    pub const ENDPOINT_GENERATE: &str = "generate";
    pub const ENDPOINT_RESPONSES: &str = "responses";
    pub const ENDPOINT_COMPLETIONS: &str = "completions";

    // Worker types
    pub const WORKER_REGULAR: &str = "regular";
    pub const WORKER_PREFILL: &str = "prefill";
    pub const WORKER_DECODE: &str = "decode";
    pub const WORKER_HTTP: &str = "http";
    pub const WORKER_GRPC: &str = "grpc";

    // Token types
    pub const TOKEN_INPUT: &str = "input";
    pub const TOKEN_OUTPUT: &str = "output";

    // Result types
    pub const RESULT_SUCCESS: &str = "success";
    pub const RESULT_ERROR: &str = "error";
    pub const RESULT_TIMEOUT: &str = "timeout";
    pub const RESULT_NOT_FOUND: &str = "not_found";

    // Rate limit results
    pub const RATE_LIMIT_ALLOWED: &str = "allowed";
    pub const RATE_LIMIT_REJECTED: &str = "rejected";

    // Circuit breaker outcomes
    pub const CB_SUCCESS: &str = "success";
    pub const CB_FAILURE: &str = "failure";

    // Router error types
    pub const ERROR_NO_WORKERS: &str = "no_workers";
    pub const ERROR_TIMEOUT: &str = "timeout";
    pub const ERROR_BACKEND: &str = "backend_error";
    pub const ERROR_VALIDATION: &str = "validation_error";
    pub const ERROR_INTERNAL: &str = "internal_error";
}

pub mod names {
    pub const HTTP_REQUESTS_TOTAL: &str = "mesh_http_requests_total";
    pub const HTTP_REQUEST_DURATION_SECONDS: &str = "mesh_http_request_duration_seconds";
    pub const HTTP_INFLIGHT_REQUEST_AGE_COUNT: &str = "mesh_http_inflight_request_age_count";
    pub const HTTP_RESPONSES_TOTAL: &str = "mesh_http_responses_total";
    pub const HTTP_CONNECTIONS_ACTIVE: &str = "mesh_http_connections_active";
    pub const HTTP_RATE_LIMIT_TOTAL: &str = "mesh_http_rate_limit_total";

    pub const ROUTER_REQUESTS_TOTAL: &str = "mesh_router_requests_total";
    pub const ROUTER_REQUEST_DURATION_SECONDS: &str = "mesh_router_request_duration_seconds";
    pub const ROUTER_REQUEST_ERRORS_TOTAL: &str = "mesh_router_request_errors_total";
    pub const ROUTER_STAGE_DURATION_SECONDS: &str = "mesh_router_stage_duration_seconds";
    pub const ROUTER_UPSTREAM_RESPONSES_TOTAL: &str = "mesh_router_upstream_responses_total";
    pub const ROUTER_TTFT_SECONDS: &str = "mesh_router_ttft_seconds";
    pub const ROUTER_TPOT_SECONDS: &str = "mesh_router_tpot_seconds";
    pub const ROUTER_TOKENS_TOTAL: &str = "mesh_router_tokens_total";
    pub const ROUTER_GENERATION_DURATION_SECONDS: &str = "mesh_router_generation_duration_seconds";

    pub const WORKER_POOL_SIZE: &str = "mesh_worker_pool_size";
    pub const WORKER_CONNECTIONS_ACTIVE: &str = "mesh_worker_connections_active";
    pub const WORKER_REQUESTS_ACTIVE: &str = "mesh_worker_requests_active";
    pub const WORKER_HEALTH: &str = "mesh_worker_health";
    pub const WORKER_HEALTH_CHECKS_TOTAL: &str = "mesh_worker_health_checks_total";
    pub const WORKER_SELECTION_TOTAL: &str = "mesh_worker_selection_total";
    pub const WORKER_ERRORS_TOTAL: &str = "mesh_worker_errors_total";
    pub const MANUAL_POLICY_CACHE_ENTRIES: &str = "mesh_manual_policy_cache_entries";
    pub const MANUAL_POLICY_BRANCH_TOTAL: &str = "mesh_manual_policy_branch_total";
    pub const PREFIX_HASH_POLICY_BRANCH_TOTAL: &str = "mesh_prefix_hash_policy_branch_total";
    pub const WORKER_ROUTING_KEYS_ACTIVE: &str = "mesh_worker_routing_keys_active";

    pub const WORKER_CB_STATE: &str = "mesh_worker_cb_state";
    pub const WORKER_CB_TRANSITIONS_TOTAL: &str = "mesh_worker_cb_transitions_total";
    pub const WORKER_CB_OUTCOMES_TOTAL: &str = "mesh_worker_cb_outcomes_total";
    pub const WORKER_CB_CONSECUTIVE_FAILURES: &str = "mesh_worker_cb_consecutive_failures";
    pub const WORKER_CB_CONSECUTIVE_SUCCESSES: &str = "mesh_worker_cb_consecutive_successes";
    pub const WORKER_RETRIES_TOTAL: &str = "mesh_worker_retries_total";
    pub const WORKER_RETRIES_EXHAUSTED_TOTAL: &str = "mesh_worker_retries_exhausted_total";
    pub const WORKER_RETRY_BACKOFF_SECONDS: &str = "mesh_worker_retry_backoff_seconds";

    pub const DISCOVERY_REGISTRATIONS_TOTAL: &str = "mesh_discovery_registrations_total";
    pub const DISCOVERY_DEREGISTRATIONS_TOTAL: &str = "mesh_discovery_deregistrations_total";
    pub const DISCOVERY_SYNC_DURATION_SECONDS: &str = "mesh_discovery_sync_duration_seconds";
    pub const DISCOVERY_WORKERS_DISCOVERED: &str = "mesh_discovery_workers_discovered";

    pub const DB_OPERATIONS_TOTAL: &str = "mesh_db_operations_total";
    pub const DB_OPERATION_DURATION_SECONDS: &str = "mesh_db_operation_duration_seconds";
    pub const DB_CONNECTIONS_ACTIVE: &str = "mesh_db_connections_active";
    pub const DB_ITEMS_STORED: &str = "mesh_db_items_stored";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    Counter,
    Gauge,
    Histogram,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricStatus {
    Active,
    Planned,
    MissingDescribe,
}

#[derive(Debug, Clone, Copy)]
pub struct MetricSpec {
    pub name: &'static str,
    pub kind: MetricKind,
    pub help: &'static str,
    pub status: MetricStatus,
}

pub const METRIC_INVENTORY: &[MetricSpec] = &[
    MetricSpec {
        name: names::HTTP_REQUESTS_TOTAL,
        kind: MetricKind::Counter,
        help: "Total HTTP requests by method and path",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::HTTP_REQUEST_DURATION_SECONDS,
        kind: MetricKind::Histogram,
        help: "HTTP request duration by method and path",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::HTTP_INFLIGHT_REQUEST_AGE_COUNT,
        kind: MetricKind::Gauge,
        help: "In-flight HTTP requests per age bucket (gt < age <= le, non-cumulative)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::HTTP_RESPONSES_TOTAL,
        kind: MetricKind::Counter,
        help: "Total HTTP responses by status_code and error_code",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::HTTP_CONNECTIONS_ACTIVE,
        kind: MetricKind::Gauge,
        help: "Currently active HTTP connections",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::HTTP_RATE_LIMIT_TOTAL,
        kind: MetricKind::Counter,
        help: "Rate limiting decisions by result (allowed/rejected)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_REQUESTS_TOTAL,
        kind: MetricKind::Counter,
        help: "Total routed requests by router_type, backend_type, connection_mode, model, endpoint, streaming",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_REQUEST_DURATION_SECONDS,
        kind: MetricKind::Histogram,
        help: "Router request duration by router_type, backend_type, connection_mode, model, endpoint",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_REQUEST_ERRORS_TOTAL,
        kind: MetricKind::Counter,
        help: "Router errors by router_type, backend_type, connection_mode, model, endpoint, error_type",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_STAGE_DURATION_SECONDS,
        kind: MetricKind::Histogram,
        help: "Pipeline stage duration by router_type and stage (gRPC only)",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::ROUTER_UPSTREAM_RESPONSES_TOTAL,
        kind: MetricKind::Counter,
        help: "Upstream backend HTTP responses by router_type, status_code, error_code",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_TTFT_SECONDS,
        kind: MetricKind::Histogram,
        help: "Time to first token by router_type, backend_type, model, endpoint (gRPC only)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_TPOT_SECONDS,
        kind: MetricKind::Histogram,
        help: "Time per output token by router_type, backend_type, model, endpoint (gRPC only)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_TOKENS_TOTAL,
        kind: MetricKind::Counter,
        help: "Total tokens processed by router_type, backend_type, model, endpoint, token_type (gRPC only)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::ROUTER_GENERATION_DURATION_SECONDS,
        kind: MetricKind::Histogram,
        help: "Total generation time by router_type, backend_type, model, endpoint (gRPC only)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_POOL_SIZE,
        kind: MetricKind::Gauge,
        help: "Current worker pool size by worker_type, connection_mode, model",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_CONNECTIONS_ACTIVE,
        kind: MetricKind::Gauge,
        help: "Active connections to workers by worker_type, connection_mode",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::WORKER_REQUESTS_ACTIVE,
        kind: MetricKind::Gauge,
        help: "Currently running requests per worker",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_HEALTH,
        kind: MetricKind::Gauge,
        help: "Worker health status (1=healthy, 0=unhealthy)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_HEALTH_CHECKS_TOTAL,
        kind: MetricKind::Counter,
        help: "Health check results by worker_type and result",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_SELECTION_TOTAL,
        kind: MetricKind::Counter,
        help: "Worker selection events by worker_type, connection_mode, model, policy",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_ERRORS_TOTAL,
        kind: MetricKind::Counter,
        help: "Worker-level errors by worker_type, connection_mode, error_type",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::MANUAL_POLICY_CACHE_ENTRIES,
        kind: MetricKind::Gauge,
        help: "Number of routing entries in manual policy cache",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::MANUAL_POLICY_BRANCH_TOTAL,
        kind: MetricKind::Counter,
        help: "Manual policy execution branches by branch",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::PREFIX_HASH_POLICY_BRANCH_TOTAL,
        kind: MetricKind::Counter,
        help: "Prefix hash policy execution branches by branch",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_ROUTING_KEYS_ACTIVE,
        kind: MetricKind::Gauge,
        help: "Active routing keys per worker",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_CB_STATE,
        kind: MetricKind::Gauge,
        help: "Circuit breaker state per worker (0=closed, 1=open, 2=half_open)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_CB_TRANSITIONS_TOTAL,
        kind: MetricKind::Counter,
        help: "Circuit breaker state transitions by worker, from, to",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_CB_OUTCOMES_TOTAL,
        kind: MetricKind::Counter,
        help: "Circuit breaker outcomes by worker and outcome (success/failure)",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_CB_CONSECUTIVE_FAILURES,
        kind: MetricKind::Gauge,
        help: "Current consecutive failure count per worker",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_CB_CONSECUTIVE_SUCCESSES,
        kind: MetricKind::Gauge,
        help: "Current consecutive success count per worker",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_RETRIES_TOTAL,
        kind: MetricKind::Counter,
        help: "Total retry attempts by worker_type and endpoint",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_RETRIES_EXHAUSTED_TOTAL,
        kind: MetricKind::Counter,
        help: "Requests that exhausted all retries by worker_type and endpoint",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::WORKER_RETRY_BACKOFF_SECONDS,
        kind: MetricKind::Histogram,
        help: "Retry backoff duration by attempt number",
        status: MetricStatus::Active,
    },
    MetricSpec {
        name: names::DISCOVERY_REGISTRATIONS_TOTAL,
        kind: MetricKind::Counter,
        help: "Worker registration attempts by source and result",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DISCOVERY_DEREGISTRATIONS_TOTAL,
        kind: MetricKind::Counter,
        help: "Worker deregistration events by source and reason",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DISCOVERY_SYNC_DURATION_SECONDS,
        kind: MetricKind::Histogram,
        help: "Discovery sync duration by source",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DISCOVERY_WORKERS_DISCOVERED,
        kind: MetricKind::Gauge,
        help: "Workers known via discovery by source",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DB_OPERATIONS_TOTAL,
        kind: MetricKind::Counter,
        help: "Total database operations by storage_type, operation, result",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DB_OPERATION_DURATION_SECONDS,
        kind: MetricKind::Histogram,
        help: "Database operation duration by storage_type, operation",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DB_CONNECTIONS_ACTIVE,
        kind: MetricKind::Gauge,
        help: "Active database connections by storage_type",
        status: MetricStatus::Planned,
    },
    MetricSpec {
        name: names::DB_ITEMS_STORED,
        kind: MetricKind::Counter,
        help: "Total items stored by storage_type",
        status: MetricStatus::Planned,
    },
];

/// Register HELP/TYPE metadata for every metric with a complete schema entry.
///
/// Planned metrics remain in the inventory so recorder APIs and future wiring
/// have a single source of truth, but they are still described here. Only
/// `MissingDescribe` is skipped, and tests assert that no such entries remain.
pub(crate) fn describe_all_metrics() {
    for metric in METRIC_INVENTORY {
        if metric.status == MetricStatus::MissingDescribe {
            continue;
        }

        match metric.kind {
            MetricKind::Counter => describe_counter!(metric.name, metric.help),
            MetricKind::Gauge => describe_gauge!(metric.name, metric.help),
            MetricKind::Histogram => describe_histogram!(metric.name, metric.help),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn metric_inventory_names_are_unique() {
        let mut seen = HashSet::new();
        for metric in METRIC_INVENTORY {
            assert!(
                seen.insert(metric.name),
                "duplicate metric name {}",
                metric.name
            );
        }
    }

    #[test]
    fn metric_inventory_has_no_missing_describe_entries() {
        let missing_describe: Vec<_> = METRIC_INVENTORY
            .iter()
            .filter(|metric| metric.status == MetricStatus::MissingDescribe)
            .map(|metric| metric.name)
            .collect();

        assert!(
            missing_describe.is_empty(),
            "metrics still missing describe entries: {:?}",
            missing_describe
        );
    }

    #[test]
    fn test_bool_to_static_str() {
        assert_eq!(bool_to_static_str(true), "true");
        assert_eq!(bool_to_static_str(false), "false");
    }

    #[test]
    fn test_method_to_static_str() {
        assert_eq!(method_to_static_str("GET"), "GET");
        assert_eq!(method_to_static_str("POST"), "POST");
        assert_eq!(method_to_static_str("UNKNOWN"), "OTHER");
    }

    #[test]
    fn test_status_code_to_static_str() {
        assert_eq!(status_code_to_static_str(200), Some("200"));
        assert_eq!(status_code_to_static_str(404), Some("404"));
        assert_eq!(status_code_to_static_str(418), None);
    }

    #[test]
    fn test_normalize_path_preserves_stable_paths() {
        assert_eq!(normalize_path_for_metrics("/health"), "/health");
        assert_eq!(normalize_path_for_metrics("/v1/models"), "/v1/models");
        assert_eq!(
            normalize_path_for_metrics("/v1/chat/completions"),
            "/v1/chat/completions"
        );
    }

    #[test]
    fn test_normalize_path_replaces_dynamic_ids() {
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/resp_abc123def456"),
            "/v1/responses/{id}"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/responses/550e8400-e29b-41d4-a716-446655440000"),
            "/v1/responses/{id}"
        );
        assert_eq!(
            normalize_path_for_metrics("/v1/workers/12345"),
            "/v1/workers/{id}"
        );
    }

    #[test]
    fn test_interner_reuses_strings() {
        let a = intern_string("test-model");
        let b = intern_string("test-model");
        assert!(Arc::ptr_eq(&a, &b));
    }
}
