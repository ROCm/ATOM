use crate::common::multi_api;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::json;
use tower::ServiceExt;

#[tokio::test]
async fn http_preserves_all_api_requests_and_json_sse_errors() {
    for key in [None, Some("worker-secret")] {
        let mut backend = multi_api::Backend::start().await;
        let (context, worker) = backend.context(key).await;
        let app = multi_api::http_app(context).await;
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        backend.verify_url(&url, key).await;
        assert_eq!(worker.load(), 0);
        task.abort();
    }
}

#[tokio::test]
async fn api_validation_errors_follow_the_requested_protocol() {
    let backend = multi_api::Backend::start().await;
    let (context, _) = backend.context(None).await;
    let app = multi_api::http_app(context).await;
    for path in multi_api::APIS {
        for body in [
            "{",
            "[]",
            "{\"model\":\"test-model\",\"messages\":[],\"stream\":42}",
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::post(*path)
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST);
            let bytes = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["error"]["type"], "invalid_request_error");
            if *path == "/v1/messages" {
                assert_eq!(value["type"], "error");
            }
        }
    }
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .header("content-encoding", "gzip")
                .body(Body::from(json!({}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn messages_method_and_size_errors_are_protocol_errors() {
    let backend = multi_api::Backend::start().await;
    let (context, _) = backend.context(None).await;
    let app = multi_api::http_app(context).await;
    for (method, body, content_length, status) in [
        ("GET", String::new(), true, StatusCode::METHOD_NOT_ALLOWED),
        (
            "POST",
            "x".repeat(1024 * 1024 + 1),
            true,
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
        (
            "POST",
            "x".repeat(1024 * 1024 + 1),
            false,
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
    ] {
        let mut request = Request::builder()
            .method(method)
            .uri("/v1/messages")
            .header("content-type", "application/json");
        if content_length {
            request = request.header("content-length", body.len().to_string());
        }
        // Use an unknown-size stream to exercise the actual limit during upload.
        let chunks = futures_util::stream::iter(
            body.into_bytes()
                .chunks(8192)
                .map(|c| Ok::<_, std::convert::Infallible>(bytes::Bytes::copy_from_slice(c)))
                .collect::<Vec<_>>(),
        );
        let response = app
            .clone()
            .oneshot(request.body(Body::from_stream(chunks)).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(
            value["error"]["type"],
            if status == StatusCode::PAYLOAD_TOO_LARGE {
                "request_too_large"
            } else {
                "invalid_request_error"
            }
        );
        assert!(value["error"]["message"].is_string());
    }
}

#[tokio::test]
async fn disconnect_drops_upstream_and_refunds_load_for_every_api() {
    use http_body_util::BodyExt;
    let backend = multi_api::Backend::start().await;
    let (context, worker) = backend.context(None).await;
    let app = multi_api::http_app(context).await;
    for path in multi_api::APIS {
        let mut value = multi_api::request(path, true);
        value["test_stall"] = json!(true);
        let response = app
            .clone()
            .oneshot(
                Request::post(*path)
                    .header("content-type", "application/json")
                    .body(Body::from(value.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let mut body = response.into_body();
        let frame = tokio::time::timeout(std::time::Duration::from_secs(2), body.frame())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(frame.is_data());
        assert_eq!(worker.load(), 1);
        drop(body);
        assert_eq!(worker.load(), 0);
    }
}

#[tokio::test]
async fn responses_submissions_are_never_retried_after_backend_failure() {
    for path in multi_api::APIS {
        let mut backend = multi_api::Backend::start().await;
        let (context, worker) = backend.context(None).await;
        let mut context = (*context).clone();
        context.router_config.disable_retries = false;
        context.router_config.retry.max_retries = 3;
        context.router_config.retry.initial_backoff_ms = 1;
        let app = multi_api::http_app(std::sync::Arc::new(context)).await;
        let mut request = multi_api::request(path, true);
        request["test_error"] = json!(true);
        request["test_status"] = json!(500);
        let response = app
            .oneshot(
                Request::post(*path)
                    .header("content-type", "application/json")
                    .body(Body::from(request.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.headers()["content-type"], "application/json");
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(error["error"]["vendor"], 42);
        let mut attempts = 0;
        while backend.captured.try_recv().is_ok() {
            attempts += 1;
        }
        assert_eq!(attempts, if *path == "/v1/responses" { 1 } else { 3 });
        assert_eq!(worker.load(), 0);
    }
}

#[tokio::test]
async fn broken_upload_is_not_reported_as_oversized() {
    let backend = multi_api::Backend::start().await;
    let (context, _) = backend.context(None).await;
    let app = multi_api::http_app(context).await;
    let body = Body::from_stream(futures_util::stream::once(async {
        Err::<bytes::Bytes, _>(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "upload interrupted",
        ))
    }));
    let response = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .body(body)
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(response.headers()["x-mesh-error-code"], "invalid_request");
    let body = axum::body::to_bytes(response.into_body(), 4096)
        .await
        .unwrap();
    let error: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(error["type"], "error");
}

#[tokio::test]
async fn response_resources_preserve_methods_queries_credentials_and_streaming() {
    use axum::{extract::State, http::Method, response::Response, Router};
    use bytes::Bytes;
    use http_body_util::BodyExt;
    use mesh::{app_context::AppContext, config::RouterConfig, core::DPAwareWorkerBuilder};
    use std::{sync::Arc, time::Duration};
    use tokio::sync::{mpsc, oneshot};
    use tokio_stream::wrappers::UnboundedReceiverStream;

    type Captured = (http::request::Parts, Bytes, oneshot::Sender<Response>);
    async fn backend(
        State(tx): State<mpsc::UnboundedSender<Captured>>,
        request: Request<Body>,
    ) -> Response {
        let (parts, body) = request.into_parts();
        let body = axum::body::to_bytes(body, 4096).await.unwrap();
        let (reply, response) = oneshot::channel();
        tx.send((parts, body, reply)).unwrap();
        response
            .await
            .unwrap_or_else(|_| Response::new(Body::empty()))
    }
    let (tx, mut captured) = mpsc::unbounded_channel::<Captured>();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, Router::new().fallback(backend).with_state(tx))
            .await
            .unwrap();
    });
    let context = Arc::new(
        AppContext::from_config(
            RouterConfig {
                dp_aware: true,
                disable_retries: true,
                ..Default::default()
            },
            5,
        )
        .await
        .unwrap(),
    );
    for rank in 0..2 {
        context.worker_registry.register(Arc::new(
            DPAwareWorkerBuilder::new(&url, rank, 2)
                .api_key("worker-secret")
                .build(),
        ));
    }
    let app = multi_api::http_app(context).await;
    for (method, path, status) in [
        (
            Method::GET,
            concat!(
                "/v1/responses/resp-test?stream=false&starting_after=13",
                "&include%5B%5D=message.output_text.logprobs&vendor=a%2Fb&vendor=c",
            ),
            StatusCode::OK,
        ),
        (
            Method::POST,
            "/v1/responses/resp-test/cancel",
            StatusCode::OK,
        ),
        (Method::DELETE, "/v1/responses/resp-test", StatusCode::OK),
        (
            Method::GET,
            concat!(
                "/v1/responses/resp-test/input_items?after=item%2F1&before=item-9",
                "&limit=2&order=desc&include%5B%5D=foo",
            ),
            StatusCode::OK,
        ),
        (
            Method::GET,
            "/v1/responses/resp-missing",
            StatusCode::NOT_FOUND,
        ),
    ] {
        let task = tokio::spawn(
            app.clone().oneshot(
                Request::builder()
                    .method(method.clone())
                    .uri(path)
                    .header("authorization", "Bearer client-secret")
                    .header("x-api-key", "client-key")
                    .header("x-request-id", "resource-id")
                    .body(Body::empty())
                    .unwrap(),
            ),
        );
        let (parts, body, reply) = tokio::time::timeout(Duration::from_secs(2), captured.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(parts.method, method);
        assert_eq!(parts.uri.to_string(), path);
        assert!(body.is_empty());
        assert_eq!(parts.headers["authorization"], "Bearer worker-secret");
        assert_eq!(parts.headers.get_all("authorization").iter().count(), 1);
        assert!(!parts.headers.contains_key("x-api-key"));
        assert_eq!(parts.headers["x-request-id"], "resource-id");
        let payload = if status == StatusCode::OK {
            br#"{"result":"backend-owned"}"#.as_slice()
        } else {
            br#"{"error":{"message":"missing","vendor":42}}"#.as_slice()
        };
        reply
            .send(
                Response::builder()
                    .status(status)
                    .header("content-type", "application/json")
                    .header("x-mesh-error-code", "backend-owned")
                    .body(Body::from(payload))
                    .unwrap(),
            )
            .unwrap();
        let response = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(response.status(), status);
        assert_eq!(response.headers()["x-mesh-error-code"], "backend-owned");
        assert_eq!(
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap(),
            payload
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), captured.recv())
                .await
                .is_err(),
            "duplicate request sent to the same DP origin"
        );
    }

    let path = "/v1/responses/resp-test?stream=true&starting_after=13";
    let task = tokio::spawn(app.oneshot(Request::get(path).body(Body::empty()).unwrap()));
    let (parts, _, reply) = tokio::time::timeout(Duration::from_secs(2), captured.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(parts.uri.to_string(), path);
    let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
    let frame =
        Bytes::from_static(b"event: response.output_text.delta\ndata: {\"delta\":\"hello\"}\n\n");
    tx.send(Ok(frame.clone())).unwrap();
    reply
        .send(
            Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(UnboundedReceiverStream::new(rx)))
                .unwrap(),
        )
        .unwrap();
    let response = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    let mut body = response.into_body();
    let first = tokio::time::timeout(Duration::from_secs(2), body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(first.into_data().unwrap(), frame);
    drop(body);
    drop(tx);
    assert!(captured.try_recv().is_err());
    server.abort();
}
