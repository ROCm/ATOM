//! Worker management integration tests
//!
//! Tests for dynamic worker add/remove operations via management API.
//! The actual worker management API uses:
//! - POST /workers - create a worker
//! - GET /workers - list workers
//! - DELETE /workers/{worker_id} - remove a worker

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

use crate::common::{mock_worker::MockWorker, AppTestContext, TestRouterConfig, TestWorkerConfig};

#[cfg(test)]
mod worker_management_tests {
    use super::*;

    /// Test listing workers via API
    #[tokio::test]
    async fn test_list_workers() {
        let config = TestRouterConfig::round_robin(3900);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19900),
                TestWorkerConfig::healthy(19901),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // List workers via GET /workers
        let req = Request::builder()
            .method("GET")
            .uri("/workers")
            .body(Body::empty())
            .unwrap();

        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "GET /workers should return OK"
        );

        ctx.shutdown().await;
    }

    /// Test that routing continues to work with multiple workers
    #[tokio::test]
    async fn test_routing_with_multiple_workers() {
        let config = TestRouterConfig::round_robin(3901);

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::healthy(19902),
                TestWorkerConfig::healthy(19903),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        // Verify routing distributes across workers
        for i in 0..20 {
            let payload = json!({
                "text": format!("Test request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed with multiple workers"
        );

        ctx.shutdown().await;
    }

    /// Test that requests continue to work during worker operations
    #[tokio::test]
    async fn test_requests_during_worker_changes() {
        let config = TestRouterConfig::round_robin(3902);

        let ctx =
            AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(19904)]).await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        let mut success_count = 0;
        for i in 0..10 {
            let payload = json!({
                "text": format!("Request during changes {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 10,
            "All requests should succeed during normal operation"
        );

        ctx.shutdown().await;
    }

    /// A directory holding a word-level tokenizer over `vocab`, ids from 1.
    fn tokenizer_dir(vocab: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let mut ids = serde_json::Map::from_iter([("<unk>".to_string(), json!(0))]);
        for (id, word) in vocab.iter().enumerate() {
            ids.insert(word.to_string(), json!(id + 1));
        }
        let tokenizer = json!({"version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": ids}});
        std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
        dir
    }

    /// Waits up to 10 s for the tokenizer of `model` and encodes `text` with it.
    async fn encode_with(ctx: &AppTestContext, model: &str, text: &str, why: &str) -> Vec<u32> {
        let registry = &ctx.app_context.tokenizer_registry;
        let start = tokio::time::Instant::now();
        while registry.get(model).is_none()
            && start.elapsed() < tokio::time::Duration::from_secs(10)
        {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
        let tokenizer = registry.get(model).expect(why);
        tokenizer.encode(text, false).unwrap().token_ids().to_vec()
    }

    /// A worker that `/v1/models` names by an alias (the mock's `mock-model`) gets its
    /// tokenizer from `--tokenizer-path`, not from the alias, under `--backend atom`.
    #[tokio::test]
    async fn test_aliased_worker_tokenizer_comes_from_the_tokenizer_path() {
        let dir = tokenizer_dir(&["a", "b"]);
        let mut config = TestRouterConfig::round_robin(3910);
        config.backend = mesh::config::types::BackendType::Atom;
        config.tokenizer_path = Some(dir.path().to_str().unwrap().to_string());
        let ctx = AppTestContext::new_with_config(config, vec![TestWorkerConfig::healthy(0)]).await;

        let why = "the aliased worker's tokenizer should load from --tokenizer-path";
        assert_eq!(encode_with(&ctx, "mock-model", "a b", why).await, [1, 2]);

        ctx.shutdown().await;
    }

    /// Registers a worker that serves its load path (`MockWorker::set_model_path`) behind
    /// a router with `--model-path` and `backend`, and encodes "a b" with the tokenizer
    /// the router registered for `model_id` (`None`: the worker's path).
    async fn encode_with_discovered_path(
        port: u16,
        backend: mesh::config::types::BackendType,
        model_id: Option<&str>,
    ) -> Vec<u32> {
        let router_dir = tokenizer_dir(&["a", "b"]);
        let worker_dir = tokenizer_dir(&["b", "a"]);
        let worker_path = worker_dir.path().to_str().unwrap();
        let mut worker = MockWorker::new(TestWorkerConfig::healthy(0));
        let url = worker.start().await.unwrap();
        worker.set_model_path(worker_path).await;

        let mut config = TestRouterConfig::round_robin(port);
        config.backend = backend;
        config.model_path = Some(router_dir.path().to_str().unwrap().to_string());
        let ctx = AppTestContext::new_with_config(config, vec![]).await;
        let req = Request::builder()
            .method("POST")
            .uri("/workers")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(json!({"url": url}).to_string()))
            .unwrap();
        let resp = ctx.create_app().await.oneshot(req).await.unwrap();
        assert!(resp.status().is_success(), "{}", resp.status());

        let why = "the worker's tokenizer should load from its discovered path";
        let ids = encode_with(&ctx, model_id.unwrap_or(worker_path), "a b", why).await;

        ctx.shutdown().await;
        worker.stop().await;
        ids
    }

    /// A worker's discovered `model_path` beats a router-wide `--model-path`, so a
    /// router in front of workers for several models loads each model's own tokenizer.
    #[tokio::test]
    async fn test_discovered_model_path_beats_the_router_model_path() {
        let backend = mesh::config::types::BackendType::Sglang;
        let ids = encode_with_discovered_path(3911, backend, None).await;
        assert_eq!(ids, [2, 1]);
    }

    /// The same for a vLLM HTTP worker, whose `/v1/models` card gives the load path as
    /// `root` and the model id (here `mock-model`) apart from it.
    #[tokio::test]
    async fn test_vllm_model_root_beats_the_router_model_path() {
        let backend = mesh::config::types::BackendType::Vllm;
        let ids = encode_with_discovered_path(3912, backend, Some("mock-model")).await;
        assert_eq!(ids, [2, 1]);
    }
}
