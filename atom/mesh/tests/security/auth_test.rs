//! Integration tests for inference and health endpoints without authentication.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod auth_tests {
    use super::*;

    /// Test request without API key when auth is not required
    #[tokio::test]
    async fn test_no_auth_required() {
        let config = TestRouterConfig::round_robin(4300);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20300)]).await;

        let app = ctx.create_app().await;

        // Request without auth header should succeed when no auth required
        let payload = json!({
            "text": "Test without auth",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request without auth should succeed when no auth required"
        );

        ctx.shutdown().await;
    }

    /// Test health endpoint doesn't require authentication
    #[tokio::test]
    async fn test_health_endpoint_no_auth() {
        let config = TestRouterConfig::round_robin(4302);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(20302)]).await;

        let app = ctx.create_app().await;

        // Health endpoint should be accessible without auth
        let req = Request::builder()
            .method("GET")
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Health endpoint should not require auth"
        );

        ctx.shutdown().await;
    }
}
