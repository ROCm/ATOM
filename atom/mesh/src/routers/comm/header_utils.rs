use axum::{body::Body, extract::Request, http::HeaderMap};
/// Copy request headers to a Vec of name-value string pairs
/// Used for forwarding headers to backend workers
pub fn copy_request_headers(req: &Request<Body>) -> Vec<(String, String)> {
    req.headers()
        .iter()
        .filter_map(|(name, value)| {
            // Convert header value to string, skipping non-UTF8 headers
            value
                .to_str()
                .ok()
                .map(|v| (name.to_string(), v.to_string()))
        })
        .collect()
}

/// Convert headers from reqwest Response to axum HeaderMap
/// Filters out hop-by-hop headers that shouldn't be forwarded
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    for (name, value) in reqwest_headers.iter() {
        // Skip hop-by-hop headers that shouldn't be forwarded
        // Use eq_ignore_ascii_case to avoid string allocation
        if should_forward_header_no_alloc(name.as_str()) {
            // The original name and value are already valid, so we can just clone them
            headers.insert(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded without allocating (case-insensitive)
fn should_forward_header_no_alloc(name: &str) -> bool {
    // List of headers that should NOT be forwarded (hop-by-hop headers)
    // Use eq_ignore_ascii_case to avoid to_lowercase() allocation
    !(name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("proxy-authenticate")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailers")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("content-encoding")
        || name.eq_ignore_ascii_case("host"))
}

#[inline]
pub fn should_forward_request_header(name: &str) -> bool {
    const REQUEST_ID_PREFIX: &str = "x-request-id-";

    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("x-request-id")
        || name.eq_ignore_ascii_case("x-correlation-id")
        || name.eq_ignore_ascii_case("traceparent")
        || name.eq_ignore_ascii_case("tracestate")
        || name
            .get(..REQUEST_ID_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(REQUEST_ID_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    #[test]
    fn test_should_forward_request_header_whitelist() {
        assert!(should_forward_request_header("authorization"));
        assert!(should_forward_request_header("Authorization"));
        assert!(should_forward_request_header("AUTHORIZATION"));
        assert!(should_forward_request_header("x-request-id"));
        assert!(should_forward_request_header("X-Request-Id"));
        assert!(should_forward_request_header("x-correlation-id"));
        assert!(should_forward_request_header("X-Correlation-ID"));
        assert!(should_forward_request_header("traceparent"));
        assert!(should_forward_request_header("Traceparent"));
        assert!(should_forward_request_header("tracestate"));
        assert!(should_forward_request_header("Tracestate"));
        assert!(should_forward_request_header("x-request-id-user"));
        assert!(should_forward_request_header("X-Request-ID-Span"));
        assert!(should_forward_request_header("x-request-id-123"));
    }

    #[test]
    fn test_should_forward_request_header_blocked() {
        assert!(!should_forward_request_header("content-type"));
        assert!(!should_forward_request_header("Content-Type"));
        assert!(!should_forward_request_header("content-length"));
        assert!(!should_forward_request_header("host"));
        assert!(!should_forward_request_header("Host"));
        assert!(!should_forward_request_header("connection"));
        assert!(!should_forward_request_header("transfer-encoding"));
        assert!(!should_forward_request_header("accept"));
        assert!(!should_forward_request_header("accept-encoding"));
        assert!(!should_forward_request_header("user-agent"));
        assert!(!should_forward_request_header("cookie"));
        assert!(!should_forward_request_header("x-custom-header"));
        assert!(!should_forward_request_header("x-api-key"));
    }

    // ===================== should_forward_header_no_alloc tests =====================

    #[test]
    fn test_hop_by_hop_headers_filtered() {
        let hop_by_hop = [
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
            "content-encoding",
            "host",
        ];
        for h in hop_by_hop {
            assert!(!should_forward_header_no_alloc(h), "{h} should be filtered");
        }
    }

    #[test]
    fn test_hop_by_hop_case_insensitive() {
        assert!(!should_forward_header_no_alloc("Connection"));
        assert!(!should_forward_header_no_alloc("CONNECTION"));
        assert!(!should_forward_header_no_alloc("Keep-Alive"));
        assert!(!should_forward_header_no_alloc("Transfer-Encoding"));
        assert!(!should_forward_header_no_alloc("Host"));
        assert!(!should_forward_header_no_alloc("HOST"));
    }

    #[test]
    fn test_regular_headers_forwarded() {
        let forward = [
            "content-type",
            "content-length",
            "authorization",
            "x-request-id",
            "accept",
            "user-agent",
            "x-custom-header",
        ];
        for h in forward {
            assert!(should_forward_header_no_alloc(h), "{h} should be forwarded");
        }
    }

    // ===================== preserve_response_headers tests =====================

    #[test]
    fn test_preserve_response_headers_filters_hop_by_hop() {
        let mut input = HeaderMap::new();
        input.insert("content-type", HeaderValue::from_static("application/json"));
        input.insert("connection", HeaderValue::from_static("keep-alive"));
        input.insert("x-request-id", HeaderValue::from_static("abc123"));
        input.insert("transfer-encoding", HeaderValue::from_static("chunked"));

        let result = preserve_response_headers(&input);
        assert!(result.contains_key("content-type"));
        assert!(result.contains_key("x-request-id"));
        assert!(!result.contains_key("connection"));
        assert!(!result.contains_key("transfer-encoding"));
    }

    #[test]
    fn test_preserve_response_headers_empty() {
        let input = HeaderMap::new();
        let result = preserve_response_headers(&input);
        assert!(result.is_empty());
    }

    #[test]
    fn test_preserve_response_headers_all_forwardable() {
        let mut input = HeaderMap::new();
        input.insert("content-type", HeaderValue::from_static("text/plain"));
        input.insert("x-custom", HeaderValue::from_static("value"));

        let result = preserve_response_headers(&input);
        assert_eq!(result.len(), 2);
    }

    // ===================== copy_request_headers tests =====================

    #[test]
    fn test_copy_request_headers_basic() {
        let mut req = Request::builder();
        req = req.header("content-type", "application/json");
        req = req.header("x-custom", "value");
        let request = req.body(Body::empty()).unwrap();

        let copied = copy_request_headers(&request);
        assert!(copied
            .iter()
            .any(|(k, v)| k == "content-type" && v == "application/json"));
        assert!(copied.iter().any(|(k, v)| k == "x-custom" && v == "value"));
    }

    #[test]
    fn test_copy_request_headers_empty() {
        let request = Request::builder().body(Body::empty()).unwrap();
        let copied = copy_request_headers(&request);
        assert!(copied.is_empty());
    }
}
