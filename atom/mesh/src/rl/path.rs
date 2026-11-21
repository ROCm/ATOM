//! The engine path and query string of a proxied call.

use super::error::RlError;

/// Most segments an engine path may have (`inference/v1/generate` has 3).
pub const MAX_SEGMENTS: usize = 4;

/// `raw` without a leading `/`, when it is a relative engine path: 1 to
/// `MAX_SEGMENTS` non-empty segments of `[A-Za-z0-9._-]`, none `.` or `..`.
pub fn validate_engine_path(raw: &str) -> Result<String, RlError> {
    let path = raw.strip_prefix('/').unwrap_or(raw);
    let invalid = |why: String| Err(RlError::InvalidEnginePath(why));
    if path.is_empty() {
        return invalid("path is empty".to_string());
    }
    if path.contains('?') || path.contains('#') {
        return invalid("path must not contain `?` or `#`".to_string());
    }
    let segments: Vec<&str> = path.split('/').collect();
    if segments.len() > MAX_SEGMENTS {
        return invalid(format!("path has more than {MAX_SEGMENTS} segments"));
    }
    for seg in segments {
        if seg.is_empty() {
            return invalid("path has an empty segment".to_string());
        }
        if seg == "." || seg == ".." {
            return invalid("path must not contain `.` or `..` segments".to_string());
        }
        if !seg
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            return invalid(format!(
                "segment `{seg}` contains characters outside [A-Za-z0-9._-]"
            ));
        }
    }
    Ok(path.to_string())
}

/// The caller's query string without its `selector` pairs, not re-encoded;
/// `None` when nothing is left.
pub fn passthrough_query(raw: Option<&str>) -> Option<String> {
    let kept: Vec<&str> = raw?
        .split('&')
        .filter(|pair| !pair.is_empty() && pair.split('=').next() != Some("selector"))
        .collect();
    (!kept.is_empty()).then(|| kept.join("&"))
}
