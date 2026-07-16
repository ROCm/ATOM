//! PD Topology integration tests: 1P1D, 2P2D, and Regular mode end-to-end.
//!
//! These tests verify complete request flows through the router using MockWorker
//! backends — no real inference engine (ATOM/SGLang) is needed.

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use http_body_util::BodyExt;
use mesh::config::RouterConfig;
use serde_json::json;
use tower::ServiceExt;

use crate::common::{AppTestContext, TestWorkerConfig};

fn pick_port() -> u16 {
    portpicker::pick_unused_port().expect("no free port")
}

/// Build a PD mode config from pre-assigned ports.
fn pd_config(prefill_ports: &[u16], decode_ports: &[u16], policy: &str) -> RouterConfig {
    let prefill_urls: Vec<(String, Option<u16>)> = prefill_ports
        .iter()
        .map(|p| (format!("http://127.0.0.1:{}", p), None))
        .collect();
    let decode_urls: Vec<String> = decode_ports
        .iter()
        .map(|p| format!("http://127.0.0.1:{}", p))
        .collect();

    let mut builder = RouterConfig::builder()
        .prefill_decode_mode(prefill_urls, decode_urls)
        .host("127.0.0.1")
        .port(pick_port())
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(30)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60);

    builder = match policy {
        "round_robin" => builder.round_robin_policy(),
        "random" => builder.random_policy(),
        "power_of_two" => builder.power_of_two_policy(1),
        _ => builder.round_robin_policy(),
    };

    builder.build_unchecked()
}

/// Build a Regular mode config from pre-assigned ports.
fn regular_config(worker_ports: &[u16]) -> RouterConfig {
    let worker_urls: Vec<String> = worker_ports
        .iter()
        .map(|p| format!("http://127.0.0.1:{}", p))
        .collect();

    RouterConfig::builder()
        .regular_mode(worker_urls)
        .round_robin_policy()
        .host("127.0.0.1")
        .port(pick_port())
        .max_payload_size(256 * 1024 * 1024)
        .request_timeout_secs(30)
        .worker_startup_timeout_secs(5)
        .worker_startup_check_interval_secs(1)
        .max_concurrent_requests(64)
        .queue_timeout_secs(60)
        .build_unchecked()
}

fn generate_request(text: &str) -> Request<Body> {
    let payload = json!({ "text": text, "stream": false });
    Request::builder()
        .method("POST")
        .uri("/generate")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap()
}

fn chat_request(content: &str, stream: bool) -> Request<Body> {
    let payload = json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": content}],
        "stream": stream
    });
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&payload).unwrap()))
        .unwrap()
}

fn health_generate_request() -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri("/health_generate")
        .body(Body::empty())
        .unwrap()
}

fn models_request() -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap()
}

// ==========================================================================
// 1P1D Tests
// ==========================================================================

#[cfg(test)]
mod test_1p1d {
    use super::*;

    #[tokio::test]
    async fn test_1p1d_basic_generate() {
        let pp = pick_port();
        let dp = pick_port();

        let config = pd_config(&[pp], &[dp], "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(pp), TestWorkerConfig::decode(dp)],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app.oneshot(generate_request("hello world")).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_1p1d_chat_completions() {
        let pp = pick_port();
        let dp = pick_port();

        let config = pd_config(&[pp], &[dp], "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(pp), TestWorkerConfig::decode(dp)],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app
            .oneshot(chat_request("What is Rust?", false))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify response body is valid JSON
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.get("choices").is_some() || json.get("text").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_1p1d_streaming() {
        let pp = pick_port();
        let dp = pick_port();

        let config = pd_config(&[pp], &[dp], "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(pp), TestWorkerConfig::decode(dp)],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app
            .oneshot(chat_request("Stream test", true))
            .await
            .unwrap();

        // Streaming may return 200 with SSE or the status from backend
        assert!(
            resp.status() == StatusCode::OK || resp.status().is_success(),
            "Streaming request should succeed, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_1p1d_health_generate() {
        let pp = pick_port();
        let dp = pick_port();

        let config = pd_config(&[pp], &[dp], "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(pp), TestWorkerConfig::decode(dp)],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app.oneshot(health_generate_request()).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "health_generate should return 200 when all workers healthy"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_1p1d_models_endpoint() {
        let pp = pick_port();
        let dp = pick_port();

        let config = pd_config(&[pp], &[dp], "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(pp), TestWorkerConfig::decode(dp)],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app.oneshot(models_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Should contain model data
        assert!(json.get("data").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_1p1d_multiple_requests() {
        let pp = pick_port();
        let dp = pick_port();

        let config = pd_config(&[pp], &[dp], "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![TestWorkerConfig::prefill(pp), TestWorkerConfig::decode(dp)],
        )
        .await;

        let app = ctx.create_app().await;

        let mut success = 0;
        for i in 0..10 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("request {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert_eq!(success, 10, "All 10 requests should succeed in 1P1D");

        ctx.shutdown().await;
    }
}

// ==========================================================================
// 2P2D Tests
// ==========================================================================

#[cfg(test)]
mod test_2p2d {
    use super::*;

    #[tokio::test]
    async fn test_2p2d_basic_routing() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let mut success = 0;
        for i in 0..20 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("2p2d req {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert_eq!(success, 20, "All 20 requests should succeed in 2P2D");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_2p2d_chat_completions() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        for i in 0..10 {
            let resp = app
                .clone()
                .oneshot(chat_request(&format!("2p2d chat {}", i), false))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
        }

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_2p2d_streaming() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let resp = app
            .oneshot(chat_request("2p2d stream", true))
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "2P2D streaming should succeed, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_2p2d_partial_prefill_failure() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        // After init, mark second prefill as unhealthy in the registry
        let workers = ctx.app_context.worker_registry.get_all();
        let prefill_url = format!("http://127.0.0.1:{}", pp[1]);
        for w in &workers {
            if w.url() == prefill_url {
                w.set_healthy(false);
            }
        }

        let app = ctx.create_app().await;

        // Requests should still route to the healthy prefill worker
        let mut success = 0;
        for i in 0..5 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("partial fail {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert!(
            success >= 3,
            "Most requests should succeed with one healthy prefill, got {}/5",
            success
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_2p2d_with_random_policy() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "random");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let mut success = 0;
        for i in 0..10 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("random policy {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert_eq!(
            success, 10,
            "All requests should succeed with random policy"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_2p2d_with_power_of_two_policy() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "power_of_two");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let mut success = 0;
        for i in 0..10 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("p2 policy {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert_eq!(
            success, 10,
            "All requests should succeed with power_of_two policy"
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_2p2d_health_generate() {
        let pp = [pick_port(), pick_port()];
        let dp = [pick_port(), pick_port()];

        let config = pd_config(&pp, &dp, "round_robin");
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(pp[0]),
                TestWorkerConfig::prefill(pp[1]),
                TestWorkerConfig::decode(dp[0]),
                TestWorkerConfig::decode(dp[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app.oneshot(health_generate_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }
}

// ==========================================================================
// Regular Mode (Baseline Comparison)
// ==========================================================================

#[cfg(test)]
mod test_regular {
    use super::*;

    #[tokio::test]
    async fn test_regular_2w_basic() {
        let ports = [pick_port(), pick_port()];

        let config = regular_config(&ports);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(ports[0]),
                TestWorkerConfig::healthy(ports[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        let mut success = 0;
        for i in 0..10 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("regular {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert_eq!(success, 10, "All regular mode requests should succeed");

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_regular_2w_chat_completions() {
        let ports = [pick_port(), pick_port()];

        let config = regular_config(&ports);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(ports[0]),
                TestWorkerConfig::healthy(ports[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app
            .oneshot(chat_request("Regular chat test", false))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json.get("choices").is_some());

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_regular_2w_streaming() {
        let ports = [pick_port(), pick_port()];

        let config = regular_config(&ports);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(ports[0]),
                TestWorkerConfig::healthy(ports[1]),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let resp = app
            .oneshot(chat_request("Regular stream", true))
            .await
            .unwrap();
        assert!(
            resp.status().is_success(),
            "Regular streaming should succeed, got {}",
            resp.status()
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_regular_worker_failover() {
        let ports = [pick_port(), pick_port()];

        let config = regular_config(&ports);
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(ports[0]),
                TestWorkerConfig::healthy(ports[1]),
            ],
        )
        .await;

        // After init, mark second worker as unhealthy in the registry
        let workers = ctx.app_context.worker_registry.get_all();
        let unhealthy_url = format!("http://127.0.0.1:{}", ports[1]);
        for w in &workers {
            if w.url() == unhealthy_url {
                w.set_healthy(false);
            }
        }

        let app = ctx.create_app().await;

        // With one unhealthy worker, requests should route to the healthy one
        let mut success = 0;
        for i in 0..5 {
            let resp = app
                .clone()
                .oneshot(generate_request(&format!("failover {}", i)))
                .await
                .unwrap();
            if resp.status() == StatusCode::OK {
                success += 1;
            }
        }
        assert!(
            success >= 3,
            "Most requests should succeed via failover, got {}/5",
            success
        );

        ctx.shutdown().await;
    }

    #[tokio::test]
    async fn test_regular_models_endpoint() {
        let ports = [pick_port()];

        let config = regular_config(&ports);
        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(ports[0])])
                .await;

        let app = ctx.create_app().await;
        let resp = app.oneshot(models_request()).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        ctx.shutdown().await;
    }
}
