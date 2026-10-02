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

/// Preserve repeated response headers while removing hop-by-hop fields.
pub fn preserve_response_headers(reqwest_headers: &HeaderMap) -> HeaderMap {
    let mut headers = HeaderMap::new();

    let nominated: Vec<_> = reqwest_headers
        .get_all("connection")
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(','))
        .map(str::trim)
        .collect();
    for (name, value) in reqwest_headers {
        if should_forward_header_no_alloc(name.as_str())
            && !nominated
                .iter()
                .any(|v| v.eq_ignore_ascii_case(name.as_str()))
        {
            headers.append(name.clone(), value.clone());
        }
    }

    headers
}

/// Determine if a header should be forwarded without allocating (case-insensitive)
fn should_forward_header_no_alloc(name: &str) -> bool {
    !(name.eq_ignore_ascii_case("connection")
        || name.eq_ignore_ascii_case("keep-alive")
        || name.eq_ignore_ascii_case("proxy-authenticate")
        || name.eq_ignore_ascii_case("proxy-authorization")
        || name.eq_ignore_ascii_case("te")
        || name.eq_ignore_ascii_case("trailer")
        || name.eq_ignore_ascii_case("trailers")
        || name.eq_ignore_ascii_case("transfer-encoding")
        || name.eq_ignore_ascii_case("upgrade")
        || name.eq_ignore_ascii_case("host"))
}

#[inline]
pub fn should_forward_request_header(name: &str) -> bool {
    const REQUEST_ID_PREFIX: &str = "x-request-id-";

    name.eq_ignore_ascii_case("authorization")
        || name.eq_ignore_ascii_case("x-api-key")
        || name.eq_ignore_ascii_case("anthropic-version")
        || name.eq_ignore_ascii_case("anthropic-beta")
        || name.eq_ignore_ascii_case("x-request-id")
        || name.eq_ignore_ascii_case("x-correlation-id")
        || name.eq_ignore_ascii_case("x-session-id")
        || name.eq_ignore_ascii_case("traceparent")
        || name.eq_ignore_ascii_case("tracestate")
        || name
            .get(..REQUEST_ID_PREFIX.len())
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case(REQUEST_ID_PREFIX))
}

/// Return the non-empty session ID used as the data-parallel sticky routing key.
#[inline]
pub fn extract_sticky_routing_key(headers: Option<&HeaderMap>) -> Option<&str> {
    headers?
        .get("x-session-id")?
        .to_str()
        .ok()
        .filter(|value| !value.is_empty())
}

/// Resolve client headers and worker credentials once for HTTP and Envoy mutation.
pub fn inference_request_headers(
    original: &HeaderMap,
    path: &str,
    api_key: Option<&str>,
) -> Result<HeaderMap, http::header::InvalidHeaderValue> {
    let mut result = HeaderMap::new();
    for (name, value) in original {
        if should_forward_request_header(name.as_str())
            && !(api_key.is_some() && (name == "authorization" || name == "x-api-key"))
        {
            result.append(name.clone(), value.clone());
        }
    }
    if let Some(key) = api_key {
        let (name, value) = if path.split('?').next() == Some("/v1/messages") {
            ("x-api-key", key.to_owned())
        } else {
            ("authorization", format!("Bearer {key}"))
        };
        result.insert(name, http::HeaderValue::from_str(&value)?);
    }
    Ok(result)
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
        assert!(should_forward_request_header("x-session-id"));
        assert!(should_forward_request_header("X-Session-ID"));
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
        assert!(should_forward_request_header("x-api-key"));
    }

    #[test]
    fn test_extract_sticky_routing_key() {
        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", HeaderValue::from_static("session-123"));
        assert_eq!(
            extract_sticky_routing_key(Some(&headers)),
            Some("session-123")
        );
    }

    #[test]
    fn test_extract_sticky_routing_key_ignores_missing_or_empty_values() {
        assert_eq!(extract_sticky_routing_key(None), None);

        let mut headers = HeaderMap::new();
        headers.insert("x-session-id", HeaderValue::from_static(""));
        assert_eq!(extract_sticky_routing_key(Some(&headers)), None);
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
