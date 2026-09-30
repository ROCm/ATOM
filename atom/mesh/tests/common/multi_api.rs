//! Contract fixtures shared by direct HTTP and real Envoy tests.
#![allow(dead_code)]
use axum::{
    body::{Body, Bytes},
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    response::Response,
    Router,
};
use futures_util::StreamExt;
use mesh::{
    app_context::AppContext,
    config::RouterConfig,
    core::{BasicWorkerBuilder, Worker},
    routers::{http_router, RouterTrait},
    server::{build_app, AppState},
};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::mpsc, task::JoinHandle};

pub const APIS: &[&str] = &["/v1/chat/completions", "/v1/messages", "/v1/responses"];
pub struct Captured {
    pub uri: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}
pub struct Backend {
    pub url: String,
    pub captured: mpsc::UnboundedReceiver<Captured>,
    task: JoinHandle<()>,
}
impl Drop for Backend {
    fn drop(&mut self) {
        self.task.abort();
    }
}

pub fn request(path: &str, stream: bool) -> Value {
    let mut body = match path {
        "/v1/messages" => json!({
            "model": "test-model",
            "system": [{
                "type": "text", "text": "system", "cache_control": {"type": "ephemeral"}
            }],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
            "max_tokens": 9
        }),
        "/v1/responses" => json!({
            "model": "test-model",
            "instructions": "system",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "你好"}]}],
            "max_output_tokens": 9
        }),
        _ => json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 9
        }),
    };
    body["stream"] = json!(stream);
    body["vendor_extension"] = json!({"nested":{"must":"survive"},"array":[1,null,{"x":true}]});
    body
}
pub fn invalid_requests(path: &str) -> Vec<Value> {
    let mut cases = Vec::new();
    let common = [
        ("model", json!(" ")),
        ("stream", json!(42)),
        ("tools", json!({})),
    ];
    for (key, value) in common {
        let mut body = request(path, false);
        body[key] = value;
        cases.push(body);
    }
    let invalid = match path {
        "/v1/messages" => vec![
            (
                "messages",
                json!([{"role":"system","content":"wrong role"}]),
            ),
            ("messages", json!([{"role":"user","content":42}])),
            (
                "messages",
                json!([{"role":"user","content":[{"type":"text","text":42}]}]),
            ),
            (
                "messages",
                json!([{"role":"assistant","content":[{"type":"tool_use","id":"tool-1","name":"lookup","input":"bad"}]}]),
            ),
            ("max_tokens", json!(0)),
            ("max_tokens", json!(1.5)),
            ("system", json!({"text":"bad shape"})),
            ("temperature", json!(1.1)),
            ("top_p", json!(-0.1)),
            ("stop_sequences", json!([42])),
        ],
        "/v1/responses" => vec![
            ("input", json!([42])),
            (
                "input",
                json!([{"type":"message","role":"bad","content":"text"}]),
            ),
            (
                "input",
                json!([{"role":"user","content":[{"type":"input_text","text":42}]}]),
            ),
            ("input", json!([{}])),
            ("previous_response_id", json!(42)),
            ("conversation", json!({})),
            ("instructions", json!(42)),
            ("max_output_tokens", json!(0)),
            ("background", json!("true")),
            ("temperature", json!(2.1)),
        ],
        _ => Vec::new(),
    };
    for (key, value) in invalid {
        let mut body = request(path, false);
        body[key] = value;
        cases.push(body);
    }
    if path == "/v1/responses" {
        let mut body = request(path, false);
        body["previous_response_id"] = json!("resp-1");
        body["conversation"] = json!("conv-1");
        cases.push(body);
    }
    cases
}

pub fn output(path: &str, streaming: bool) -> Vec<u8> {
    if !streaming {
        let body = match path {
            "/v1/messages" => json!({
                "type": "message",
                "content": [{"type": "text", "text": "你好"}],
                "usage": {"input_tokens": 7, "output_tokens": 2}
            }),
            "/v1/responses" => json!({
                "object": "response",
                "id": "resp-test",
                "output": [{
                    "type": "message", "content": [{"type": "output_text", "text": "你好"}]
                }],
                "usage": {"input_tokens": 7, "output_tokens": 2}
            }),
            _ => json!({
                "choices": [{"message": {"content": "你好"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2}
            }),
        };
        return serde_json::to_vec(&body).unwrap();
    }
    match path {
        "/v1/messages" => concat!(
            ": heartbeat\r\n\r\n",
            "event: message_start\ndata: {\"type\":\"message_start\",",
            "\"message\":{\"model\":\"test-model\",\"usage\":{\"input_tokens\":7,",
            "\"output_tokens\":0}}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",",
            "\"delta\":{\"type\":\"text_delta\",\"text\":\"你好\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",",
            "\"usage\":{\"output_tokens\":2}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        )
        .as_bytes()
        .to_vec(),
        "/v1/responses" => concat!(
            ": heartbeat\n\n",
            "event: response.created\ndata: {\"type\":\"response.created\",",
            "\"response\":{\"id\":\"resp-test\",\"model\":\"test-model\"}}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"你好\"}\n\n",
            "event: response.completed\ndata: {\"type\":\"response.completed\",",
            "\"response\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":2}}}\n\n",
        )
        .as_bytes()
        .to_vec(),
        _ => concat!(
            ": heartbeat\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"你好\"}}]}\n\n",
            "data: {\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\n",
            "data: [DONE]\n\n",
        )
        .as_bytes()
        .to_vec(),
    }
}
impl Backend {
    async fn handle(
        State(sender): State<mpsc::UnboundedSender<Captured>>,
        request: Request,
    ) -> Response {
        let (parts, body) = request.into_parts();
        let body = axum::body::to_bytes(body, 1024 * 1024).await.unwrap();
        let value: Value = serde_json::from_slice(&body).unwrap();
        let streaming = value["stream"] == true;
        let error = value["test_error"] == true;
        let response_bytes = if error {
            br#"{"error":{"type":"backend_error","message":"original error","vendor":42}}"#.to_vec()
        } else {
            output(parts.uri.path(), streaming)
        };
        sender
            .send(Captured {
                uri: parts.uri.to_string(),
                headers: parts.headers,
                body,
            })
            .unwrap();
        if value["test_stall"] == true {
            let first = futures_util::stream::once(async {
                Ok::<_, std::convert::Infallible>(Bytes::from_static(b"data: {}\n\n"))
            });
            return Response::builder()
                .header("content-type", "text/event-stream")
                .body(Body::from_stream(
                    first.chain(futures_util::stream::pending()),
                ))
                .unwrap();
        }
        let chunks = response_bytes
            .chunks(3)
            .map(Bytes::copy_from_slice)
            .collect::<Vec<_>>();
        Response::builder()
            .status(if error {
                value["test_status"]
                    .as_u64()
                    .and_then(|s| StatusCode::from_u16(s as u16).ok())
                    .unwrap_or(StatusCode::BAD_REQUEST)
            } else {
                StatusCode::OK
            })
            .header(
                "content-type",
                if streaming && !error {
                    "text/event-stream; charset=utf-8"
                } else {
                    "application/json"
                },
            )
            .header("x-backend-header", "preserved")
            .header("x-mesh-error-code", "backend-owned")
            .body(Body::from_stream(futures_util::stream::iter(
                chunks.into_iter().map(Ok::<_, std::convert::Infallible>),
            )))
            .unwrap()
    }

    pub async fn context(&self, key: Option<&str>) -> (Arc<AppContext>, Arc<dyn Worker>) {
        let config = RouterConfig {
            disable_retries: true,
            ..Default::default()
        };
        let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
        let mut builder = BasicWorkerBuilder::new(&self.url).model_id("test-model");
        if let Some(key) = key {
            builder = builder.api_key(key);
        }
        let worker: Arc<dyn Worker> = Arc::new(builder.build());
        app.worker_registry.register(worker.clone());
        (app, worker)
    }

    pub async fn verify_url(&mut self, url: &str, key: Option<&str>) {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        for path in APIS {
            for body in invalid_requests(path) {
                let response = client
                    .post(format!("{url}{path}"))
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}: {body}");
                assert_eq!(response.headers()["x-mesh-error-code"], "invalid_request");
                let error: Value = response.json().await.unwrap();
                assert_eq!(error["error"]["type"], "invalid_request_error");
                if *path == "/v1/messages" {
                    assert_eq!(error["type"], "error");
                }
                assert!(
                    self.captured.try_recv().is_err(),
                    "invalid input reached the backend"
                );
            }
            let response = client.get(format!("{url}{path}")).send().await.unwrap();
            assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
            assert_eq!(response.headers()["allow"], "POST");
            for streaming in [false, true] {
                for error in [false, true] {
                    let mut body = request(path, streaming);
                    if error {
                        body["test_error"] = json!(true);
                    }
                    // Whitespace, field order, and unknown fields must remain byte-identical.
                    let body = serde_json::to_string_pretty(&body).unwrap();
                    let uri = format!("{path}?vendor=a%2Fb&item=1&item=2");
                    let response = client
                        .post(format!("{url}{uri}"))
                        .header("content-type", "application/json")
                        .header("x-request-id", "contract-id")
                        .header("authorization", "Bearer client")
                        .header("x-api-key", "client-key")
                        .header("anthropic-version", "2023-06-01")
                        .header("anthropic-beta", "tools-2025")
                        .header("cookie", "private")
                        .body(body.clone())
                        .send()
                        .await
                        .unwrap();
                    assert_eq!(
                        response.status(),
                        if error {
                            StatusCode::BAD_REQUEST
                        } else {
                            StatusCode::OK
                        },
                        "{path}"
                    );
                    let content_type = response.headers()["content-type"].to_str().unwrap();
                    assert!(content_type.starts_with(if streaming && !error {
                        "text/event-stream"
                    } else {
                        "application/json"
                    }));
                    assert_eq!(response.headers()["x-backend-header"], "preserved");
                    let bytes = response.bytes().await.unwrap();
                    let expected = if error {
                        br#"{"error":{"type":"backend_error","message":"original error","vendor":42}}"#
                            .to_vec()
                    } else {
                        output(path, streaming)
                    };
                    assert_eq!(bytes.as_ref(), expected.as_slice());
                    let captured =
                        tokio::time::timeout(Duration::from_secs(2), self.captured.recv())
                            .await
                            .unwrap()
                            .unwrap();
                    assert_eq!(captured.uri, uri);
                    assert_eq!(captured.body.as_ref(), body.as_bytes());
                    assert_eq!(captured.headers["anthropic-version"], "2023-06-01");
                    assert_eq!(captured.headers["anthropic-beta"], "tools-2025");
                    assert!(!captured.headers.contains_key("cookie"));
                    if let Some(key) = key {
                        if *path == "/v1/messages" {
                            assert_eq!(captured.headers["x-api-key"], key);
                            assert!(!captured.headers.contains_key("authorization"));
                        } else {
                            assert_eq!(captured.headers["authorization"], format!("Bearer {key}"));
                            assert!(!captured.headers.contains_key("x-api-key"));
                        }
                    } else {
                        assert_eq!(captured.headers["authorization"], "Bearer client");
                        assert_eq!(captured.headers["x-api-key"], "client-key");
                    }
                }
            }
        }
    }

    pub async fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (sender, captured) = mpsc::unbounded_channel();
        let app = Router::new().fallback(Self::handle).with_state(sender);
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Self {
            url,
            captured,
            task,
        }
    }
}

pub async fn http_app(context: Arc<AppContext>) -> Router {
    #[cfg(feature = "ext-proc")]
    let context = {
        let mut context = (*context).clone();
        context.router_config.ext_proc.enabled = false;
        Arc::new(context)
    };
    let router: Arc<dyn RouterTrait> = Arc::new(http_router::Router::new(&context).await.unwrap());
    build_app(
        Arc::new(AppState {
            router,
            context,
            router_manager: None,
        }),
        1024 * 1024,
        vec!["x-request-id".into()],
    )
}
