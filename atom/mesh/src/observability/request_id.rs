//! Shared request correlation rules for HTTP and ext-proc ingress.
use rand::Rng;

/// Resolve the configured priority order, preserving an explicitly empty list.
pub fn header_names(configured: Option<&[String]>) -> Vec<String> {
    configured.map(<[String]>::to_vec).unwrap_or_else(|| {
        [
            "x-request-id",
            "x-correlation-id",
            "x-trace-id",
            "request-id",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    })
}

/// The first readable header wins, including an empty value, as in HTTP ingress.
/// Transport adapters supply the first value of each case-insensitive header.
pub fn resolve(
    names: &[String],
    path: &str,
    mut value: impl FnMut(&str) -> Option<String>,
) -> String {
    names
        .iter()
        .find_map(|name| value(name))
        .unwrap_or_else(|| generate_request_id(path))
}

/// Alphanumeric characters for request ID generation (as bytes for O(1) indexing)
const REQUEST_ID_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Generate OpenAI-compatible request ID based on endpoint.
fn generate_request_id(path: &str) -> String {
    let prefix = if path.contains("/chat/completions") {
        "chatcmpl-"
    } else if path.contains("/completions") {
        "cmpl-"
    } else if path.contains("/generate") {
        "gnt-"
    } else if path.contains("/responses") {
        "resp-"
    } else {
        "req-"
    };

    // Generate a random string similar to OpenAI's format
    // Use byte array indexing (O(1)) instead of chars().nth() (O(n))
    let mut rng = rand::rng();
    let random_part: String = (0..24)
        .map(|_| {
            let idx = rng.random_range(0..REQUEST_ID_CHARS.len());
            REQUEST_ID_CHARS[idx] as char
        })
        .collect();

    format!("{}{}", prefix, random_part)
}
