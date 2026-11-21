//! HTTP error responses produced by Entrypoints.
//!
//! The JSON shape matches Mesh's local error envelope so admission and payload
//! limits stay compatible with the existing clients.

use axum::{
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
};

pub const ERROR_CODE_HEADER: &str = "X-Mesh-Error-Code";

pub fn error_code<B>(response: &Response<B>) -> &str {
    response
        .headers()
        .get(ERROR_CODE_HEADER)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
}

pub fn body_too_large(path: &str) -> Response {
    local_response(
        path,
        StatusCode::PAYLOAD_TOO_LARGE,
        "body_too_large",
        "request body limit exceeded",
    )
}

pub fn local_response(path: &str, status: StatusCode, code: &str, message: &str) -> Response {
    let mut headers = HeaderMap::with_capacity(2);
    headers.insert(
        ERROR_CODE_HEADER,
        HeaderValue::from_str(code).unwrap_or_else(|_| HeaderValue::from_static("invalid_code")),
    );
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    if status == StatusCode::METHOD_NOT_ALLOWED {
        headers.insert(http::header::ALLOW, HeaderValue::from_static("POST"));
    }
    let body = serde_json::to_vec(&envelope(path, status, code, message))
        .expect("error payload is serializable");
    (status, headers, body).into_response()
}

fn envelope(path: &str, status: StatusCode, code: &str, message: &str) -> serde_json::Value {
    let messages = path.split('?').next() == Some("/v1/messages");
    let kind = match status.as_u16() {
        413 if messages => "request_too_large",
        529 if messages => "overloaded_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        _ => "api_error",
    };
    if messages {
        serde_json::json!({"type": "error", "error": {"type": kind, "message": message}})
    } else {
        serde_json::json!({"error": {"type": kind, "code": code, "message": message}})
    }
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use axum::response::Response;

    use super::{body_too_large, error_code, local_response};

    async fn body(response: Response) -> Vec<u8> {
        to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    #[tokio::test]
    async fn local_response_matches_the_mesh_envelope() {
        for (path, status, code, message) in [
            (
                "/v1/chat/completions",
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limited",
                "busy",
            ),
            (
                "/v1/messages",
                StatusCode::PAYLOAD_TOO_LARGE,
                "body_too_large",
                "request body limit exceeded",
            ),
            (
                "/generate",
                StatusCode::SERVICE_UNAVAILABLE,
                "no_workers",
                "No workers in registry (model: m)",
            ),
        ] {
            let mesh =
                crate::routers::comm::error::MeshLocalError::response(path, status, code, message);
            let ours = local_response(path, status, code, message);
            assert_eq!(mesh.status(), ours.status());
            assert_eq!(error_code(&mesh), error_code(&ours));
            assert_eq!(error_code(&ours), code);
            assert_eq!(body(mesh).await, body(ours).await);
        }
    }

    #[tokio::test]
    async fn body_too_large_matches_ingress() {
        let mesh = crate::routers::comm::error::IngressError::body_too_large()
            .response("/v1/chat/completions");
        let ours = body_too_large("/v1/chat/completions");
        assert_eq!(mesh.status(), ours.status());
        assert_eq!(body(mesh).await, body(ours).await);
    }
}
