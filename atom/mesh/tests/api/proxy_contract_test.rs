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
        let (context, worker, _prepare_runtime) = backend.context(key).await;
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
async fn malformed_requests_and_uploads_return_protocol_errors() {
    let backend = multi_api::Backend::start().await;
    let (context, _, _prepare_runtime) = backend.context(None).await;
    let app = multi_api::http_app(context).await;
    // Bodies that are not JSON objects fail before schema validation.
    for path in multi_api::APIS {
        for body in ["{", "[]"] {
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

    // Use an unknown-size stream to exercise the actual limit during upload.
    let oversized = || {
        Body::from_stream(futures_util::stream::iter(
            vec![b'x'; 1024 * 1024 + 1]
                .chunks(8192)
                .map(|c| Ok::<_, std::convert::Infallible>(bytes::Bytes::copy_from_slice(c)))
                .collect::<Vec<_>>(),
        ))
    };
    let broken = Body::from_stream(futures_util::stream::once(async {
        Err::<bytes::Bytes, _>(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "upload interrupted",
        ))
    }));
    let messages = |method: &str| {
        Request::builder()
            .method(method)
            .uri("/v1/messages")
            .header("content-type", "application/json")
    };
    for (request, status, error_type) in [
        (
            messages("GET").body(Body::empty()).unwrap(),
            StatusCode::METHOD_NOT_ALLOWED,
            "invalid_request_error",
        ),
        (
            messages("POST")
                .header("content-length", (1024 * 1024 + 1).to_string())
                .body(oversized())
                .unwrap(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
        ),
        (
            messages("POST").body(oversized()).unwrap(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
        ),
        // A broken upload is not reported as oversized.
        (
            messages("POST").body(broken).unwrap(),
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
        ),
        (
            messages("POST")
                .header("content-encoding", "gzip")
                .body(Body::from(json!({}).to_string()))
                .unwrap(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
        ),
    ] {
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), status);
        if status == StatusCode::BAD_REQUEST {
            assert_eq!(response.headers()["x-mesh-error-code"], "invalid_request");
        }
        let body = axum::body::to_bytes(response.into_body(), 4096)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["type"], "error", "{status}: {value}");
        assert_eq!(value["error"]["type"], error_type, "{status}: {value}");
        assert!(value["error"]["message"].is_string());
    }
}

#[tokio::test]
async fn disconnect_drops_upstream_and_refunds_load_for_every_api() {
    use http_body_util::BodyExt;
    let backend = multi_api::Backend::start().await;
    let (context, worker, _prepare_runtime) = backend.context(None).await;
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
        let (context, worker, _prepare_runtime) = backend.context(None).await;
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
    let config = RouterConfig {
        dp_aware: true,
        disable_retries: true,
        ..Default::default()
    };
    let prepare_runtime =
        mesh::core::prepare_pool::PreparePoolRuntime::new(config.resolved_prepare_pool().unwrap())
            .unwrap();
    let context = Arc::new(
        AppContext::from_config(config, 5, prepare_runtime.handle())
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

    // Decoded IDs must not be able to escape the Responses resource path.
    for (method, path) in [
        (
            Method::POST,
            "/v1/responses/..%2F..%2Fstart_profile%3F/cancel",
        ),
        (Method::GET, "/v1/responses/..%2Fhealth"),
        (Method::DELETE, "/v1/responses/resp%3Fx%3D1"),
        (Method::GET, "/v1/responses/resp%23frag/input_items"),
        (Method::GET, "/v1/responses/resp%20id"),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        assert_eq!(
            response.headers()["x-mesh-error-code"],
            "invalid_response_id"
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(20), captured.recv())
                .await
                .is_err(),
            "invalid response_id reached a backend: {path}"
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

#[tokio::test]
async fn response_resource_errors_prefer_possible_owners_over_unsupported_routes() {
    use axum::{http::Method, Router};
    use mesh::{
        app_context::AppContext, config::RouterConfig, core::prepare_pool::PreparePoolRuntime,
        core::BasicWorkerBuilder,
    };
    use std::sync::Arc;

    async fn serve(status: StatusCode) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().fallback(move || async move { status }),
            )
            .await
            .unwrap();
        });
        url
    }
    let unsupported = serve(StatusCode::METHOD_NOT_ALLOWED).await;
    let missing = serve(StatusCode::NOT_FOUND).await;
    // Bind a port without listening: connections are refused, and the port
    // cannot be reused by another test while the socket is held.
    let reserved = tokio::net::TcpSocket::new_v4().unwrap();
    reserved.bind("127.0.0.1:0".parse().unwrap()).unwrap();
    let unreachable = format!("http://{}", reserved.local_addr().unwrap());
    for (urls, status, code) in [
        // A backend without the routes says nothing about the resource.
        (vec![&unsupported, &missing], StatusCode::NOT_FOUND, None),
        // An unreachable backend may own the resource, so 404 is not certain.
        (
            vec![&missing, &unreachable],
            StatusCode::BAD_GATEWAY,
            Some("backend_unreachable"),
        ),
    ] {
        let config = RouterConfig {
            disable_retries: true,
            ..Default::default()
        };
        let prepare_runtime =
            PreparePoolRuntime::new(config.resolved_prepare_pool().unwrap()).unwrap();
        let context = Arc::new(
            AppContext::from_config(config, 5, prepare_runtime.handle())
                .await
                .unwrap(),
        );
        for url in urls {
            context
                .worker_registry
                .register(Arc::new(BasicWorkerBuilder::new(url.as_str()).build()));
        }
        let response = multi_api::http_app(context)
            .await
            .oneshot(
                Request::builder()
                    .method(Method::GET)
                    .uri("/v1/responses/resp-x")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        if let Some(code) = code {
            assert_eq!(response.headers()["x-mesh-error-code"], code);
            let body = axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
            assert!(!String::from_utf8_lossy(&body).contains(&unreachable));
        }
    }
}
