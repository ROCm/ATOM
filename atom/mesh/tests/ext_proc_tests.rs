use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use mesh::{
    app_context::AppContext,
    config::RouterConfig,
    core::{BasicWorkerBuilder, Worker},
    ext_proc::{
        proto::{
            envoy::{
                config::core::v3::{HeaderMap, HeaderValue},
                service::ext_proc::v3::{
                    self as pb, processing_request::Request, processing_response::Response,
                },
            },
            grpc::health::v1::{health_client::HealthClient, HealthCheckRequest},
        },
        ExtProcConfig, ExtProcRuntime,
    },
};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

struct Fixture {
    app: Arc<AppContext>,
    runtime: ExtProcRuntime,
    worker: Arc<dyn Worker>,
}

impl Fixture {
    async fn new(mut config: RouterConfig) -> Self {
        config.ext_proc.enabled = true;
        config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
        let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://127.0.0.1:18001")
                .model_id("test-model")
                .build(),
        );
        app.worker_registry.register(worker.clone());
        let runtime = ExtProcRuntime::start(app.clone()).await.unwrap();
        Self {
            app,
            runtime,
            worker,
        }
    }

    async fn open(&self) -> Stream {
        let mut client = pb::external_processor_client::ExternalProcessorClient::connect(format!(
            "http://{}",
            self.runtime.address
        ))
        .await
        .unwrap();
        let (sender, receiver) = mpsc::channel(16);
        let response = client
            .process(ReceiverStream::new(receiver))
            .await
            .unwrap()
            .into_inner();
        Stream {
            sender,
            response,
            first_message: AtomicBool::new(true),
        }
    }

    async fn unloaded(&self) {
        tokio::time::timeout(Duration::from_secs(3), async {
            while self.worker.load() != 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
    }
}

struct Stream {
    sender: mpsc::Sender<pb::ProcessingRequest>,
    response: tonic::Streaming<pb::ProcessingResponse>,
    first_message: AtomicBool,
}

impl Stream {
    const BODY: &'static [u8] = br#"{"model":"test-model","messages":[{"role":"user","content":"hello"}],"stream":true,"vendor_extension":{"keep":42}}"#;

    fn headers(values: &[(&str, &str)], end: bool) -> pb::HttpHeaders {
        pb::HttpHeaders {
            headers: Some(HeaderMap {
                headers: values
                    .iter()
                    .map(|(key, value)| HeaderValue {
                        key: (*key).into(),
                        raw_value: value.as_bytes().to_vec(),
                        ..Default::default()
                    })
                    .collect(),
            }),
            end_of_stream: end,
            ..Default::default()
        }
    }

    async fn send(&self, request: Request) {
        self.sender
            .send(pb::ProcessingRequest {
                request: Some(request),
                protocol_config: self
                    .first_message
                    .swap(false, Ordering::SeqCst)
                    .then(Self::protocol),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    fn protocol() -> pb::ProtocolConfiguration {
        use mesh::ext_proc::proto::envoy::extensions::filters::http::ext_proc::v3::processing_mode::BodySendMode;
        pb::ProtocolConfiguration {
            request_body_mode: BodySendMode::FullDuplexStreamed as i32,
            response_body_mode: BodySendMode::FullDuplexStreamed as i32,
            ..Default::default()
        }
    }

    async fn recv(&mut self) -> Response {
        tokio::time::timeout(Duration::from_secs(3), self.response.message())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .response
            .unwrap()
    }

    async fn headers_only(&self) {
        self.request_headers("/v1/chat/completions").await;
    }

    async fn request_headers(&self, path: &str) {
        self.send(Request::RequestHeaders(Self::headers(
            &[
                (":method", "POST"),
                (":path", path),
                ("content-type", "application/json"),
                ("x-request-id", "same-id"),
                ("x-gateway-destination-endpoint", "127.0.0.1:1"),
            ],
            false,
        )))
        .await;
    }

    async fn body(&self, body: &[u8], end: bool) {
        self.send(Request::RequestBody(pb::HttpBody {
            body: body.to_vec(),
            end_of_stream: end,
            ..Default::default()
        }))
        .await;
    }

    async fn routed(&mut self) {
        self.headers_only().await;
        self.body(Self::BODY, true).await;
        assert!(matches!(self.recv().await, Response::RequestHeaders(_)));
        assert!(matches!(self.recv().await, Response::RequestBody(_)));
    }

    async fn response_headers(&mut self, end: bool) {
        self.send(Request::ResponseHeaders(Self::headers(
            &[(":status", "200"), ("content-type", "text/event-stream")],
            end,
        )))
        .await;
        assert!(matches!(self.recv().await, Response::ResponseHeaders(_)));
    }

    async fn response_body(&mut self, bytes: &[u8], end: bool) {
        self.send(Request::ResponseBody(pb::HttpBody {
            body: bytes.to_vec(),
            end_of_stream: end,
            ..Default::default()
        }))
        .await;
        let Response::ResponseBody(response) = self.recv().await else {
            panic!("expected body");
        };
        let Some(pb::body_mutation::Mutation::StreamedResponse(body)) =
            response.response.unwrap().body_mutation.unwrap().mutation
        else {
            panic!("expected streamed mutation");
        };
        assert_eq!(body.body, bytes);
        assert_eq!(body.end_of_stream, end);
    }
}

#[tokio::test]
async fn buffers_request_and_preserves_bytes_and_stream_lifetime() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.body(&Stream::BODY[..17], false).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(40), stream.response.message())
            .await
            .is_err()
    );
    assert_eq!(fixture.worker.load(), 0);
    stream.body(&Stream::BODY[17..], true).await;
    let Response::RequestHeaders(headers) = stream.recv().await else {
        panic!("expected headers");
    };
    let response = headers.response.unwrap();
    assert!(response.clear_route_cache);
    let mutation = response.header_mutation.unwrap();
    assert!(mutation
        .set_headers
        .iter()
        .filter_map(|v| v.header.as_ref())
        .all(|header| header.key != "content-length"));
    let destination = mutation
        .set_headers
        .into_iter()
        .filter_map(|v| v.header)
        .find(|h| h.key == "x-gateway-destination-endpoint")
        .unwrap();
    assert_eq!(destination.raw_value, b"127.0.0.1:18001");
    let Response::RequestBody(body) = stream.recv().await else {
        panic!("expected body");
    };
    let Some(pb::body_mutation::Mutation::StreamedResponse(body)) =
        body.response.unwrap().body_mutation.unwrap().mutation
    else {
        panic!();
    };
    assert_eq!(body.body, Stream::BODY);
    assert!(body.end_of_stream);
    assert_eq!(fixture.worker.load(), 1);
    stream.response_headers(false).await;
    stream
        .response_body(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n",
            false,
        )
        .await;
    assert_eq!(fixture.worker.load(), 1);
    stream.response_body(b"\ndata: [DONE]\n\n", true).await;
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn trailers_finish_request_and_response_without_body_eos() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.body(Stream::BODY, false).await;
    stream
        .send(Request::RequestTrailers(pb::HttpTrailers::default()))
        .await;
    assert!(matches!(stream.recv().await, Response::RequestHeaders(_)));
    assert!(matches!(stream.recv().await, Response::RequestBody(_)));
    assert!(matches!(stream.recv().await, Response::RequestTrailers(_)));
    stream.response_headers(false).await;
    stream.response_body(b"data: [DONE]\n\n", false).await;
    stream
        .send(Request::ResponseTrailers(pb::HttpTrailers::default()))
        .await;
    assert!(matches!(stream.recv().await, Response::ResponseTrailers(_)));
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn header_only_response_and_disconnect_release_load() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut first = fixture.open().await;
    let mut second = fixture.open().await;
    first.routed().await;
    second.routed().await;
    assert_eq!(fixture.worker.load(), 2);
    first.response_headers(true).await;
    drop(second);
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn malformed_and_oversized_requests_never_reserve_worker() {
    let fixture = Fixture::new(RouterConfig {
        ext_proc: ExtProcConfig {
            max_body_bytes: 256,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    for (body, status) in [
        (b"invalid".as_slice(), 400),
        (b"{\"messages\":[]}".as_slice(), 400),
        (vec![b'x'; 257].as_slice(), 413),
    ] {
        let mut stream = fixture.open().await;
        stream.headers_only().await;
        stream.body(body, true).await;
        let Response::ImmediateResponse(error) = stream.recv().await else {
            panic!("expected immediate error");
        };
        assert_eq!(error.status.unwrap().code, status);
        assert_eq!(fixture.worker.load(), 0);
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn unexpected_sequence_and_missing_body_are_rejected() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut stream = fixture.open().await;
    stream.body(Stream::BODY, true).await;
    assert!(matches!(
        stream.recv().await,
        Response::ImmediateResponse(_)
    ));
    let mut stream = fixture.open().await;
    stream
        .send(Request::RequestHeaders(Stream::headers(
            &[
                (":method", "POST"),
                (":path", "/v1/chat/completions"),
                ("content-type", "application/json"),
            ],
            true,
        )))
        .await;
    assert!(matches!(
        stream.recv().await,
        Response::ImmediateResponse(_)
    ));
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn admission_queue_cancellation_and_running_cancellation_return_capacity() {
    let fixture = Fixture::new(RouterConfig {
        max_concurrent_requests: 1,
        queue_size: 1,
        ..Default::default()
    })
    .await;
    let mut first = fixture.open().await;
    first.routed().await;
    let second = fixture.open().await;
    second.headers_only().await;
    second.body(Stream::BODY, true).await;
    tokio::time::sleep(Duration::from_millis(40)).await;
    drop(second);
    drop(first);
    fixture.unloaded().await;
    let mut third = fixture.open().await;
    third.routed().await;
    third.response_headers(true).await;
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn health_and_bounded_drain() {
    let fixture = Fixture::new(RouterConfig {
        ext_proc: ExtProcConfig {
            drain_timeout_secs: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    let mut health = HealthClient::connect(format!("http://{}", fixture.runtime.address))
        .await
        .unwrap();
    assert_eq!(
        health
            .check(HealthCheckRequest {
                service: "envoy.service.ext_proc.v3.ExternalProcessor".into()
            })
            .await
            .unwrap()
            .into_inner()
            .status,
        1
    );
    fixture.worker.set_healthy(false);
    assert_eq!(
        health
            .check(HealthCheckRequest {
                service: String::new()
            })
            .await
            .unwrap()
            .into_inner()
            .status,
        2
    );
    fixture.worker.set_healthy(true);
    let mut watch = health
        .watch(HealthCheckRequest {
            service: String::new(),
        })
        .await
        .unwrap()
        .into_inner();
    // Watch publishes the latest shared snapshot; allow one publisher interval
    // after the registry was changed back to healthy.
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if watch.message().await.unwrap().unwrap().status == 1 {
                break;
            }
        }
    })
    .await
    .unwrap();
    let mut stream = fixture.open().await;
    stream.routed().await;
    (fixture.runtime.shutdown_handle())();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), watch.message())
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .status,
        2
    );
    drop(watch);
    let worker = fixture.worker.clone();
    tokio::time::timeout(Duration::from_secs(3), fixture.runtime.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(worker.load(), 0);
}

#[test]
fn cli_flattens_protocol_configuration_and_default_stays_disabled() {
    use clap::Parser;
    let cli = mesh::cliargs::Cli::try_parse_from([
        "atomesh",
        "--ext-proc",
        "--ext-proc-listen",
        "127.0.0.1:9003",
    ])
    .unwrap();
    assert!(cli.router_args.ext_proc.enabled);
    assert_eq!(
        cli.router_args
            .to_router_config(vec![])
            .unwrap()
            .ext_proc
            .listen
            .port(),
        9003
    );
    assert!(!RouterConfig::default().ext_proc.enabled);
}

impl Stream {
    async fn subset(&self, value: prost_types::Value) {
        self.subset_body(value, Self::BODY).await;
    }

    async fn subset_body(&self, value: prost_types::Value, body: &[u8]) {
        use std::collections::BTreeMap;
        self.sender
            .send(pb::ProcessingRequest {
                request: Some(Request::RequestBody(pb::HttpBody {
                    body: body.to_vec(),
                    end_of_stream: true,
                    ..Default::default()
                })),
                metadata_context: Some(mesh::ext_proc::proto::envoy::config::core::v3::Metadata {
                    filter_metadata: std::collections::HashMap::from([(
                        "envoy.lb.subset_hint".into(),
                        prost_types::Struct {
                            fields: BTreeMap::from([(
                                "x-gateway-destination-endpoint-subset".into(),
                                value,
                            )]),
                        },
                    )]),
                    ..Default::default()
                }),
                ..Default::default()
            })
            .await
            .unwrap();
    }

    fn list(addresses: &[&str]) -> prost_types::Value {
        prost_types::Value {
            kind: Some(prost_types::value::Kind::ListValue(
                prost_types::ListValue {
                    values: addresses
                        .iter()
                        .map(|s| prost_types::Value {
                            kind: Some(prost_types::value::Kind::StringValue((*s).into())),
                        })
                        .collect(),
                },
            )),
        }
    }

    async fn error(&mut self, status: i32) {
        let Response::ImmediateResponse(error) = self.recv().await else {
            panic!("expected error");
        };
        assert_eq!(error.status.unwrap().code, status);
    }

    async fn destination(&mut self) -> String {
        let Response::RequestHeaders(headers) = self.recv().await else {
            panic!("expected headers");
        };
        let target = headers
            .response
            .unwrap()
            .header_mutation
            .unwrap()
            .set_headers
            .into_iter()
            .filter_map(|v| v.header)
            .find(|h| h.key == "x-gateway-destination-endpoint")
            .unwrap();
        String::from_utf8(target.raw_value).unwrap()
    }
}

#[tokio::test]
async fn candidate_subset_is_intersected_with_registered_healthy_model_workers() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let other: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:18002")
            .model_id("test-model")
            .build(),
    );
    fixture.app.worker_registry.register(other.clone());
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream
        .subset(Stream::list(&["127.0.0.1:18002", "127.0.0.1:1"]))
        .await;
    assert_eq!(stream.destination().await, "127.0.0.1:18002");
    assert!(matches!(stream.recv().await, Response::RequestBody(_)));
    stream.response_headers(true).await;
    let mut empty = fixture.open().await;
    empty.headers_only().await;
    empty.subset(Stream::list(&[])).await;
    assert!(matches!(empty.recv().await, Response::RequestHeaders(_)));
    assert!(matches!(empty.recv().await, Response::RequestBody(_)));
    empty.response_headers(true).await;
    for addresses in [vec!["127.0.0.1:1"]] {
        let mut stream = fixture.open().await;
        stream.headers_only().await;
        stream.subset(Stream::list(&addresses)).await;
        stream.error(503).await;
    }
    other.set_healthy(false);
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.subset(Stream::list(&["127.0.0.1:18002"])).await;
    stream.error(503).await;
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream
        .subset(prost_types::Value {
            kind: Some(prost_types::value::Kind::BoolValue(true)),
        })
        .await;
    stream.error(400).await;
    assert_eq!(fixture.worker.load(), 0);
    assert_eq!(other.load(), 0);
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn generate_optional_model_uses_default_policy_and_preserves_candidate_filters() {
    let fixture = Fixture::new(RouterConfig {
        policy: mesh::config::PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 10,
        },
        ..Default::default()
    })
    .await;
    let other: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:18002")
            .model_id("other-model")
            .build(),
    );
    fixture.app.worker_registry.register(other.clone());
    // An unspecified model must use the default policy, even if the chosen
    // worker's model has a policy that requires tokens.
    fixture
        .app
        .policy_registry
        .on_worker_added("other-model", Some("prefix_hash"));
    let busy = mesh::core::WorkerLoadGuard::new(fixture.worker.clone(), None);
    for (body, destination) in [
        (
            r#"{"text":"hello","vendor_extension":42}"#,
            "127.0.0.1:18002",
        ),
        (r#"{"model":null,"text":"hello"}"#, "127.0.0.1:18002"),
        (
            r#"{"model":"test-model","text":"hello"}"#,
            "127.0.0.1:18001",
        ),
    ] {
        let mut stream = fixture.open().await;
        stream.request_headers("/generate").await;
        stream.body(body.as_bytes(), true).await;
        assert_eq!(stream.destination().await, destination);
        let Response::RequestBody(response) = stream.recv().await else {
            panic!("expected body");
        };
        let Some(pb::body_mutation::Mutation::StreamedResponse(forwarded)) =
            response.response.unwrap().body_mutation.unwrap().mutation
        else {
            panic!("expected streamed mutation");
        };
        assert_eq!(forwarded.body, body.as_bytes());
        stream.response_headers(true).await;
    }
    drop(busy);
    for healthy in [true, false] {
        other.set_healthy(healthy);
        let mut stream = fixture.open().await;
        stream.request_headers("/generate").await;
        stream
            .subset_body(Stream::list(&["127.0.0.1:18002"]), br#"{"text":"hello"}"#)
            .await;
        if healthy {
            assert_eq!(stream.destination().await, "127.0.0.1:18002");
            assert!(matches!(stream.recv().await, Response::RequestBody(_)));
            stream.response_headers(true).await;
        } else {
            stream.error(503).await;
        }
    }
    for (path, body, status) in [
        ("/generate", r#"{"model":"missing","text":"hello"}"#, 503),
        ("/generate", r#"{"model":" ","text":"hello"}"#, 400),
        (
            "/v1/chat/completions",
            r#"{"messages":[{"role":"user","content":"hello"}]}"#,
            503, // Chat's protocol default model has no registered worker.
        ),
        (
            "/v1/chat/completions",
            r#"{"model":null,"messages":[{"role":"user","content":"hello"}]}"#,
            400,
        ),
        ("/v1/completions", r#"{"prompt":"hello"}"#, 400),
    ] {
        let mut stream = fixture.open().await;
        stream.request_headers(path).await;
        stream.body(body.as_bytes(), true).await;
        stream.error(status).await;
    }
    fixture.runtime.shutdown().await.unwrap();
    assert_eq!(fixture.worker.load(), 0);
    assert_eq!(other.load(), 0);
}

#[tokio::test]
async fn dp_rank_uses_origin_address_and_preserves_extension_fields() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    fixture.worker.set_healthy(false);
    let worker: Arc<dyn Worker> = Arc::new(
        mesh::core::DPAwareWorkerBuilder::new("http://[::1]:18003", 2, 4)
            .model_id("test-model")
            .build(),
    );
    fixture.app.worker_registry.register(worker.clone());
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.body(Stream::BODY, true).await;
    assert_eq!(stream.destination().await, "[::1]:18003");
    let Response::RequestBody(body) = stream.recv().await else {
        panic!();
    };
    let Some(pb::body_mutation::Mutation::StreamedResponse(body)) =
        body.response.unwrap().body_mutation.unwrap().mutation
    else {
        panic!();
    };
    let value: serde_json::Value = serde_json::from_slice(&body.body).unwrap();
    assert_eq!(value["data_parallel_rank"], 2);
    assert_eq!(value["vendor_extension"]["keep"], 42);
    stream.response_headers(true).await;
    fixture.runtime.shutdown().await.unwrap();
    assert_eq!(worker.load(), 0);
}

#[tokio::test]
async fn prefix_hash_requires_tokens_and_accepts_generate_input_ids() {
    let fixture = Fixture::new(RouterConfig {
        policy: mesh::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
        ..Default::default()
    })
    .await;
    let mut missing = fixture.open().await;
    missing.headers_only().await;
    missing.body(Stream::BODY, true).await;
    missing.error(503).await;
    let mut missing_model = fixture.open().await;
    missing_model.request_headers("/generate").await;
    missing_model.body(br#"{"text":"hello"}"#, true).await;
    missing_model.error(503).await;
    for body in [
        r#"{"model":"test-model","input_ids":[1,2,3,4],"stream":false}"#,
        r#"{"input_ids":[1,2,3,4],"stream":false}"#,
        r#"{"model":null,"input_ids":[1,2,3,4],"stream":false}"#,
    ] {
        let mut stream = fixture.open().await;
        stream.request_headers("/generate").await;
        stream.body(body.as_bytes(), true).await;
        assert_eq!(stream.destination().await, "127.0.0.1:18001");
        assert!(matches!(stream.recv().await, Response::RequestBody(_)));
        stream.response_headers(true).await;
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn protocol_modes_encoding_and_api_paths_are_explicitly_rejected() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    for (path, method, encoding, status) in [
        ("/v1/responses", "POST", "identity", 404),
        ("/v1/chat/completions", "GET", "identity", 405),
        ("/v1/chat/completions", "POST", "gzip", 415),
    ] {
        let mut stream = fixture.open().await;
        stream
            .send(Request::RequestHeaders(Stream::headers(
                &[
                    (":method", method),
                    (":path", path),
                    ("content-type", "application/json"),
                    ("content-encoding", encoding),
                ],
                false,
            )))
            .await;
        stream.error(status).await;
    }
    let mut stream = fixture.open().await;
    stream
        .sender
        .send(pb::ProcessingRequest {
            request: Some(Request::RequestHeaders(Stream::headers(&[], false))),
            protocol_config: Some(pb::ProtocolConfiguration::default()),
            ..Default::default()
        })
        .await
        .unwrap();
    stream.error(500).await;
    fixture.worker.set_healthy(false);
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.body(Stream::BODY, true).await;
    stream.error(503).await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn first_message_requires_compatible_protocol_before_waiting_for_body() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let valid = Stream::protocol();
    for (config, code, detail) in [
        (None, "protocol_config_missing", "first ProcessingRequest"),
        (
            Some(pb::ProtocolConfiguration {
                request_body_mode: 0,
                ..valid
            }),
            "unsupported_processing_mode",
            "request_body_mode=NONE",
        ),
        (
            Some(pb::ProtocolConfiguration {
                request_body_mode: 2,
                ..valid
            }),
            "unsupported_processing_mode",
            "request_body_mode=BUFFERED",
        ),
        (
            Some(pb::ProtocolConfiguration {
                response_body_mode: 1,
                ..valid
            }),
            "unsupported_processing_mode",
            "response_body_mode=STREAMED",
        ),
        (
            Some(pb::ProtocolConfiguration {
                response_body_mode: 999,
                ..valid
            }),
            "unsupported_processing_mode",
            "response_body_mode=UNKNOWN(999)",
        ),
    ] {
        let mut stream = fixture.open().await;
        stream
            .sender
            .send(pb::ProcessingRequest {
                request: Some(Request::RequestHeaders(Stream::headers(&[], false))),
                protocol_config: config,
                ..Default::default()
            })
            .await
            .unwrap();
        let Response::ImmediateResponse(error) =
            tokio::time::timeout(Duration::from_secs(1), stream.recv())
                .await
                .unwrap()
        else {
            panic!("expected immediate configuration error");
        };
        assert_eq!(error.status.unwrap().code, 500);
        assert_eq!(error.details, format!("mesh_ext_proc_{code}"));
        assert!(String::from_utf8(error.body).unwrap().contains(detail));
        assert_eq!(fixture.worker.load(), 0);
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn protocol_is_remembered_and_cannot_change_midstream() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    for changed in [false, true] {
        let mut stream = fixture.open().await;
        stream.headers_only().await;
        let mut protocol = Stream::protocol();
        protocol.send_body_without_waiting_for_header_response = changed;
        stream
            .sender
            .send(pb::ProcessingRequest {
                request: Some(Request::RequestBody(pb::HttpBody {
                    body: Stream::BODY.to_vec(),
                    end_of_stream: true,
                    ..Default::default()
                })),
                protocol_config: Some(protocol),
                ..Default::default()
            })
            .await
            .unwrap();
        if changed {
            let Response::ImmediateResponse(error) = stream.recv().await else {
                panic!()
            };
            assert_eq!(error.status.unwrap().code, 500);
            assert_eq!(error.details, "mesh_ext_proc_protocol_config_changed");
        } else {
            assert!(matches!(stream.recv().await, Response::RequestHeaders(_)));
            assert!(matches!(stream.recv().await, Response::RequestBody(_)));
            // No protocol_config on subsequent response messages.
            stream.response_headers(true).await;
        }
        fixture.unloaded().await;
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn observability_mode_closes_the_rpc_with_a_diagnostic() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut stream = fixture.open().await;
    stream
        .sender
        .send(pb::ProcessingRequest {
            request: Some(Request::RequestHeaders(Stream::headers(&[], false))),
            protocol_config: Some(Stream::protocol()),
            observability_mode: true,
            ..Default::default()
        })
        .await
        .unwrap();
    let error = tokio::time::timeout(Duration::from_secs(1), stream.response.message())
        .await
        .unwrap()
        .unwrap_err();
    assert_eq!(error.code(), tonic::Code::FailedPrecondition);
    assert!(error.message().contains("observability_mode must be false"));
    assert_eq!(fixture.worker.load(), 0);
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn empty_messages_and_trailers_without_headers_are_protocol_errors() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    for request in [
        None,
        Some(Request::RequestTrailers(pb::HttpTrailers::default())),
        Some(Request::ResponseTrailers(pb::HttpTrailers::default())),
    ] {
        let mut stream = fixture.open().await;
        stream
            .sender
            .send(pb::ProcessingRequest {
                request,
                protocol_config: Some(Stream::protocol()),
                ..Default::default()
            })
            .await
            .unwrap();
        let Response::ImmediateResponse(error) = stream.recv().await else {
            panic!()
        };
        assert_eq!(error.status.unwrap().code, 400);
        assert_eq!(error.details, "mesh_ext_proc_invalid_processing_sequence");
        assert_eq!(fixture.worker.load(), 0);
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn headers_then_trailers_distinguish_empty_requests_from_empty_responses() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream
        .send(Request::RequestTrailers(pb::HttpTrailers::default()))
        .await;
    let Response::ImmediateResponse(error) = stream.recv().await else {
        panic!()
    };
    assert_eq!(error.status.unwrap().code, 400);
    assert_eq!(error.details, "mesh_ext_proc_invalid_request");
    assert_eq!(fixture.worker.load(), 0);
    let mut stream = fixture.open().await;
    stream.routed().await;
    stream.response_headers(false).await;
    stream
        .send(Request::ResponseTrailers(pb::HttpTrailers::default()))
        .await;
    assert!(matches!(stream.recv().await, Response::ResponseTrailers(_)));
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn early_local_replies_preserve_response_and_discard_inflight_request_data() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    for (status, header_only, request_started) in [
        ("413", false, true),
        ("504", true, true),
        ("403", false, false),
    ] {
        let mut stream = fixture.open().await;
        if request_started {
            stream.headers_only().await;
            stream.body(&Stream::BODY[..17], false).await;
        }
        stream
            .send(Request::ResponseHeaders(Stream::headers(
                &[
                    (":status", status),
                    ("content-type", "text/plain"),
                    ("x-local-reply", "original"),
                ],
                header_only,
            )))
            .await;
        let Response::ResponseHeaders(headers) = stream.recv().await else {
            panic!("local reply replaced")
        };
        let mutation = headers.response.unwrap().header_mutation;
        if request_started {
            let mutation = mutation.unwrap();
            assert_eq!(
                selected_header(&mutation, "x-request-id").as_deref(),
                Some("same-id")
            );
            assert_eq!(mutation.set_headers.len(), 1);
            assert!(mutation.remove_headers.is_empty());
        } else {
            assert!(mutation.is_none());
        }
        assert_eq!(fixture.worker.load(), 0);
        if !header_only {
            if request_started {
                stream.body(&Stream::BODY[17..], false).await;
                stream
                    .send(Request::RequestTrailers(pb::HttpTrailers::default()))
                    .await;
            }
            stream.response_body(b"original Envoy error", false).await;
            stream
                .send(Request::ResponseTrailers(pb::HttpTrailers::default()))
                .await;
            assert!(matches!(stream.recv().await, Response::ResponseTrailers(_)));
        }
        assert!(stream.response.message().await.unwrap().is_none());
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn early_reply_cancels_pending_admission_without_a_late_dispatch() {
    let fixture = Fixture::new(RouterConfig {
        max_concurrent_requests: 1,
        queue_size: 1,
        ..Default::default()
    })
    .await;
    let mut first = fixture.open().await;
    first.routed().await;
    let mut second = fixture.open().await;
    second.headers_only().await;
    second.body(Stream::BODY, true).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(80), second.response.message())
            .await
            .is_err()
    );
    second
        .send(Request::ResponseHeaders(Stream::headers(
            &[(":status", "504")],
            false,
        )))
        .await;
    assert!(matches!(
        tokio::time::timeout(Duration::from_secs(1), second.recv())
            .await
            .unwrap(),
        Response::ResponseHeaders(_)
    ));
    // Free capacity while the early response is still streaming. Only the third
    // request may use it; the canceled second request must never dispatch.
    first.response_headers(true).await;
    fixture.unloaded().await;
    let mut third = fixture.open().await;
    third.routed().await;
    third.response_headers(true).await;
    second
        .response_body(b"original gateway timeout", true)
        .await;
    assert!(second.response.message().await.unwrap().is_none());
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn pending_admission_keeps_its_deadline_while_listening_for_envoy() {
    let fixture = Fixture::new(RouterConfig {
        max_concurrent_requests: 1,
        queue_timeout_secs: 1,
        ext_proc: ExtProcConfig {
            decision_timeout_secs: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    let mut first = fixture.open().await;
    first.routed().await;
    let mut second = fixture.open().await;
    second.headers_only().await;
    second.body(Stream::BODY, true).await;
    let Response::ImmediateResponse(error) = second.recv().await else {
        panic!()
    };
    assert_eq!(error.status.unwrap().code, 408);
    assert_eq!(error.details, "mesh_ext_proc_admission_timeout");
    first.response_headers(true).await;
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn local_reply_interrupts_request_forwarding_under_backpressure() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    let mut stream = fixture.open().await;
    let body = serde_json::to_vec(&serde_json::json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "x".repeat(6 * 1024 * 1024)}],
    }))
    .unwrap();
    stream.headers_only().await;
    for (index, chunk) in body.chunks(64 * 1024).enumerate() {
        stream
            .body(chunk, (index + 1) * 64 * 1024 >= body.len())
            .await;
    }
    assert!(matches!(stream.recv().await, Response::RequestHeaders(_)));
    // Stop reading the large upload, then let Envoy report its local reply.
    // Mesh must keep reading the other direction while its output is blocked.
    stream
        .send(Request::ResponseHeaders(Stream::headers(
            &[(":status", "413"), ("content-type", "text/plain")],
            false,
        )))
        .await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    let mut forwarded = 0;
    loop {
        match stream.recv().await {
            Response::RequestBody(body) => {
                let Some(pb::body_mutation::Mutation::StreamedResponse(body)) =
                    body.response.unwrap().body_mutation.unwrap().mutation
                else {
                    panic!("expected streamed body");
                };
                forwarded += body.body.len();
            }
            Response::ResponseHeaders(headers) => {
                let mutation = headers.response.unwrap().header_mutation.unwrap();
                assert_eq!(
                    selected_header(&mutation, "x-request-id").as_deref(),
                    Some("same-id")
                );
                assert_eq!(mutation.set_headers.len(), 1);
                assert!(mutation.remove_headers.is_empty());
                break;
            }
            other => panic!("local reply replaced by {other:?}"),
        }
    }
    assert!(
        forwarded < body.len(),
        "upload continued after the local reply"
    );
    stream
        .response_body(b"original payload limit error", true)
        .await;
    assert!(stream.response.message().await.unwrap().is_none());
    fixture.unloaded().await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn errors_after_response_headers_do_not_generate_an_immediate_response() {
    let fixture = Fixture::new(RouterConfig::default()).await;
    for missing_status in [true, false] {
        let mut stream = fixture.open().await;
        stream.headers_only().await;
        if missing_status {
            stream
                .send(Request::ResponseHeaders(Stream::headers(&[], false)))
                .await;
        } else {
            stream.response_headers(false).await;
            stream
                .sender
                .send(pb::ProcessingRequest::default())
                .await
                .unwrap();
        }
        let error = tokio::time::timeout(Duration::from_secs(1), stream.response.message())
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
        assert!(error.message().contains(if missing_status {
            "missing :status"
        } else {
            "request is missing"
        }));
        assert_eq!(fixture.worker.load(), 0);
    }
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn stream_cap_body_timeout_and_runtime_drop_release_resources() {
    let fixture = Fixture::new(RouterConfig {
        ext_proc: ExtProcConfig {
            max_streams: 1,
            body_timeout_secs: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    let mut first = fixture.open().await;
    first.headers_only().await;
    let mut client = pb::external_processor_client::ExternalProcessorClient::connect(format!(
        "http://{}",
        fixture.runtime.address
    ))
    .await
    .unwrap();
    let (_tx, rx) = mpsc::channel::<pb::ProcessingRequest>(1);
    assert_eq!(
        client
            .process(ReceiverStream::new(rx))
            .await
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    first.error(408).await;
    drop(first);
    let mut running = fixture.open().await;
    running.routed().await;
    drop(fixture.runtime);
    tokio::time::timeout(Duration::from_secs(2), async {
        while fixture.worker.load() != 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

struct TestIdentity {
    directory: tempfile::TempDir,
    cert: Vec<u8>,
    ca: Vec<u8>,
    key: Vec<u8>,
}

impl TestIdentity {
    fn openssl(directory: &std::path::Path, args: &[&str]) {
        let output = std::process::Command::new("openssl")
            .current_dir(directory)
            .args(args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path();
        std::fs::write(
            path.join("request.cnf"),
            "[req]\ndistinguished_name=dn\n[dn]\n",
        )
        .unwrap();
        Self::openssl(
            path,
            &[
                "req",
                "-config",
                "request.cnf",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=ext-proc-test-CA",
                "-addext",
                "basicConstraints=critical,CA:TRUE",
                "-keyout",
                "ca.key",
                "-out",
                "ca.pem",
            ],
        );
        Self::openssl(
            path,
            &[
                "req",
                "-config",
                "request.cnf",
                "-new",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-subj",
                "/CN=localhost",
                "-keyout",
                "key.pem",
                "-out",
                "leaf.csr",
            ],
        );
        std::fs::write(path.join("extensions"), "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth,clientAuth\nsubjectAltName=DNS:localhost\n").unwrap();
        Self::openssl(
            path,
            &[
                "x509",
                "-req",
                "-in",
                "leaf.csr",
                "-CA",
                "ca.pem",
                "-CAkey",
                "ca.key",
                "-CAcreateserial",
                "-days",
                "1",
                "-extfile",
                "extensions",
                "-out",
                "cert.pem",
            ],
        );
        Self::openssl(path, &["verify", "-CAfile", "ca.pem", "cert.pem"]);
        Self {
            cert: std::fs::read(path.join("cert.pem")).unwrap(),
            ca: std::fs::read(path.join("ca.pem")).unwrap(),
            key: std::fs::read(path.join("key.pem")).unwrap(),
            directory,
        }
    }

    async fn channel(
        &self,
        address: std::net::SocketAddr,
        identity: bool,
    ) -> Result<tonic::transport::Channel, tonic::transport::Error> {
        let mut tls = tonic::transport::ClientTlsConfig::new()
            .domain_name("localhost")
            .ca_certificate(tonic::transport::Certificate::from_pem(&self.ca));
        if identity {
            tls = tls.identity(tonic::transport::Identity::from_pem(&self.cert, &self.key));
        }
        tonic::transport::Endpoint::from_shared(format!("https://{address}"))
            .unwrap()
            .tls_config(tls)
            .unwrap()
            .connect()
            .await
    }
}

#[tokio::test]
async fn tls_and_mutual_tls_health_handshake() {
    let identity = TestIdentity::new();
    for mutual in [false, true] {
        let fixture = Fixture::new(RouterConfig {
            ext_proc: ExtProcConfig {
                tls_cert: Some(identity.directory.path().join("cert.pem")),
                tls_key: Some(identity.directory.path().join("key.pem")),
                client_ca: mutual.then(|| identity.directory.path().join("ca.pem")),
                ..Default::default()
            },
            ..Default::default()
        })
        .await;
        let channel = identity
            .channel(fixture.runtime.address, mutual)
            .await
            .unwrap();
        let result = HealthClient::new(channel)
            .check(HealthCheckRequest {
                service: String::new(),
            })
            .await
            .unwrap();
        assert_eq!(result.into_inner().status, 1);
        if mutual {
            if let Ok(channel) = identity.channel(fixture.runtime.address, false).await {
                assert!(HealthClient::new(channel)
                    .check(HealthCheckRequest {
                        service: String::new()
                    })
                    .await
                    .is_err());
            }
        }
        fixture.runtime.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn pd_execution_lease_pins_pair_rejects_replay_and_expires_on_disconnect() {
    use mesh::{config::RoutingMode, core::WorkerType};
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let mut servers = Vec::new();
    let mut workers = Vec::new();
    let mut config = RouterConfig::default();
    config.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    config.ext_proc.enabled = true;
    config.ext_proc.reservation_timeout_secs = 1;
    config.max_concurrent_requests = 1;
    config.ext_proc.listen = "127.0.0.1:0".parse().unwrap();
    config.ext_proc.executor_listen = "127.0.0.1:0".parse().unwrap();
    let app = Arc::new(AppContext::from_config(config, 5).await.unwrap());
    for kind in [
        WorkerType::Prefill {
            bootstrap_port: Some(9000),
        },
        WorkerType::Decode,
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let calls = calls.clone();
        let router = axum::Router::new().route(
            "/v1/chat/completions",
            axum::routing::post(
                move |axum::Json(body): axum::Json<serde_json::Value>| async move {
                    if body["slow"] == true {
                        tokio::time::sleep(Duration::from_millis(1200)).await;
                    }
                    calls.fetch_add(1, Ordering::SeqCst);
                    axum::Json(serde_json::json!({"choices":[{"text":"ok"}]}))
                },
            ),
        );
        servers.push(tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        }));
        let worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(format!("http://{address}"))
                .model_id("test-model")
                .worker_type(kind)
                .build(),
        );
        app.worker_registry.register(worker.clone());
        workers.push(worker);
    }
    let runtime = ExtProcRuntime::start(app.clone()).await.unwrap();
    let fixture = Fixture {
        app: app.clone(),
        runtime,
        worker: workers[0].clone(),
    };
    let client = reqwest::Client::new();
    let body = br#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#;
    for attempt in 0..5 {
        let slow_body =
            br#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"slow":true}"#;
        let body: &[u8] = if attempt == 4 { slow_body } else { body };
        let mut stream = fixture.open().await;
        stream.headers_only().await;
        stream.body(body, true).await;
        let response = stream.recv().await;
        let Response::RequestHeaders(headers) = response else {
            panic!("attempt {attempt}: {response:?}");
        };
        let values: std::collections::HashMap<_, _> = headers
            .response
            .unwrap()
            .header_mutation
            .unwrap()
            .set_headers
            .into_iter()
            .filter_map(|v| v.header)
            .map(|h| (h.key, String::from_utf8(h.raw_value).unwrap()))
            .collect();
        assert!(matches!(stream.recv().await, Response::RequestBody(_)));
        assert!(workers.iter().all(|w| w.load() == 1));
        let url = format!(
            "http://{}/v1/chat/completions",
            values["x-gateway-destination-endpoint"]
        );
        let id = &values["x-mesh-execution-id"];
        if attempt == 2 {
            drop(stream);
            fixture.unloaded().await;
            assert_eq!(
                client
                    .post(&url)
                    .header("x-mesh-execution-id", id)
                    .body(body.to_vec())
                    .send()
                    .await
                    .unwrap()
                    .status(),
                403
            );
            continue;
        }
        if attempt == 3 {
            // Keep the RPC connected: reservation expiry must release actual
            // load guards and admission, well before the 300-second idle limit.
            let Response::ImmediateResponse(error) = stream.recv().await else {
                panic!("expected lease expiry");
            };
            assert_eq!(error.status.unwrap().code, 504);
            assert_eq!(error.details, "mesh_ext_proc_execution_lease_timeout");
            fixture.unloaded().await;
            assert!(workers.iter().all(|worker| worker.load() == 0));
            assert_eq!(
                client
                    .post(&url)
                    .header("x-mesh-execution-id", id)
                    .body(body.to_vec())
                    .send()
                    .await
                    .unwrap()
                    .status(),
                403
            );
            continue;
        }
        // Registry changes after selection cannot cause a second placement.
        for worker in &workers {
            fixture.app.worker_registry.remove_by_url(worker.url());
        }
        let response = client
            .post(&url)
            .header("x-mesh-execution-id", id)
            .body(if attempt != 1 {
                body.to_vec()
            } else {
                b"{}".to_vec()
            })
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), if attempt != 1 { 200 } else { 400 });
        let _ = response.bytes().await.unwrap();
        assert_eq!(
            client
                .post(&url)
                .header("x-mesh-execution-id", id)
                .body(body.to_vec())
                .send()
                .await
                .unwrap()
                .status(),
            403
        );
        stream.response_headers(true).await;
        fixture.unloaded().await;
        for worker in &workers {
            worker.set_healthy(true);
            fixture.app.worker_registry.register(worker.clone());
        }
    }
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    fixture.runtime.shutdown().await.unwrap();
    assert!(workers.iter().all(|w| w.load() == 0));
    for server in servers {
        server.abort();
    }
}

struct MeshProcess(std::process::Child);
impl Drop for MeshProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
async fn cli_ext_proc_keeps_management_disables_http_inference_and_stops_on_sigterm() {
    verify_cli_mode(true).await;
}

#[tokio::test]
async fn cli_http_mode_keeps_inference_routes_without_ext_proc() {
    verify_cli_mode(false).await;
}

async fn verify_cli_mode(ext_proc: bool) {
    let logs = tempfile::NamedTempFile::new().unwrap();
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_atomesh"));
    command.env("RUST_LOG", "info");
    command.args([
        "launch",
        "--host",
        "127.0.0.1",
        "--port",
        "0",
        "--policy",
        "round_robin",
        "--prometheus-port",
        "0",
        "--log-level",
        "info",
        "--json-log",
        "--ext-proc-listen",
        "127.0.0.1:0",
        "--ext-proc-drain-timeout-secs",
        "1",
        "--shutdown-grace-period-secs",
        "1",
    ]);
    if ext_proc {
        command.arg("--ext-proc");
    }
    let mut process = MeshProcess(
        command
            .stdout(logs.as_file().try_clone().unwrap())
            .stderr(logs.as_file().try_clone().unwrap())
            .spawn()
            .unwrap(),
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    let addresses = || {
        let logs = std::fs::read_to_string(logs.path()).unwrap();
        let address = |message: &str| -> Option<std::net::SocketAddr> {
            logs.lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
                .find(|entry| entry["message"] == message)
                .and_then(|entry| entry["address"].as_str()?.parse().ok())
        };
        (
            address("HTTP listener bound"),
            address("ext-proc listener started"),
        )
    };
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            assert!(
                process.0.try_wait().unwrap().is_none(),
                "{}",
                std::fs::read_to_string(logs.path()).unwrap()
            );
            let (Some(http_address), grpc_address) = addresses() else {
                tokio::time::sleep(Duration::from_millis(25)).await;
                continue;
            };
            if client
                .get(format!("http://{http_address}/health"))
                .send()
                .await
                .is_ok()
            {
                if !ext_proc {
                    break;
                }
                if let Some(grpc_address) = grpc_address {
                    if let Ok(mut health) =
                        HealthClient::connect(format!("http://{grpc_address}")).await
                    {
                        assert_eq!(
                            health
                                .check(HealthCheckRequest {
                                    service: String::new()
                                })
                                .await
                                .unwrap()
                                .into_inner()
                                .status,
                            2
                        );
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|error| panic!("{error}: {}", std::fs::read_to_string(logs.path()).unwrap()));
    let (http_address, grpc_address) = addresses();
    let http_address = http_address.unwrap();
    for path in ["/health", "/liveness", "/workers", "/v1/tokenizers"] {
        let response = client
            .get(format!("http://{http_address}{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK, "{path}");
    }
    for path in [
        "/generate",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/responses/test-response",
        "/v1/responses/test-response/cancel",
        "/v1/responses/test-response/input_items",
    ] {
        // OPTIONS distinguishes an absent route (404) from a registered route (405)
        // without depending on worker availability or request validation.
        let response = client
            .request(
                reqwest::Method::OPTIONS,
                format!("http://{http_address}{path}"),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if ext_proc {
                reqwest::StatusCode::NOT_FOUND
            } else {
                reqwest::StatusCode::METHOD_NOT_ALLOWED
            },
            "{path} with ext_proc={ext_proc}"
        );
        if ext_proc {
            let response = client
                .post(format!("http://{http_address}{path}"))
                .json(&serde_json::json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "prompt": "hello", "text": "hello", "input": "hello"
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::NOT_FOUND, "{path}");
        }
    }
    if !ext_proc {
        assert!(grpc_address.is_none());
    }
    assert!(std::process::Command::new("kill")
        .args(["-TERM", &process.0.id().to_string()])
        .status()
        .unwrap()
        .success());
    tokio::time::timeout(Duration::from_secs(4), async {
        loop {
            if let Some(status) = process.0.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "{}",
                    std::fs::read_to_string(logs.path()).unwrap()
                );
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .unwrap();
    assert!(tokio::net::TcpStream::connect(http_address).await.is_err());
    if let Some(address) = grpc_address {
        assert!(tokio::net::TcpStream::connect(address).await.is_err());
    }
}

#[tokio::test]
async fn load_policy_observes_shared_guards_without_crossing_model_pools() {
    let fixture = Fixture::new(RouterConfig {
        policy: mesh::config::PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 10,
        },
        ..Default::default()
    })
    .await;
    let worker: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:18002")
            .model_id("test-model")
            .build(),
    );
    fixture.app.worker_registry.register(worker.clone());
    fixture.app.worker_registry.register(Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:18003")
            .model_id("different-model")
            .build(),
    ));
    let busy = mesh::core::WorkerLoadGuard::new(fixture.worker.clone(), None);
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.body(Stream::BODY, true).await;
    assert_eq!(stream.destination().await, "127.0.0.1:18002");
    assert!(matches!(stream.recv().await, Response::RequestBody(_)));
    stream.response_headers(true).await;
    fixture.runtime.shutdown().await.unwrap();
    assert_eq!(worker.load(), 0);
    drop(busy);
    assert_eq!(fixture.worker.load(), 0);
}

#[tokio::test]
async fn admission_rejects_at_headers_before_receiving_or_parsing_body() {
    let fixture = Fixture::new(RouterConfig {
        max_concurrent_requests: 1,
        queue_size: 0,
        ..Default::default()
    })
    .await;
    let mut first = fixture.open().await;
    first.headers_only().await;
    assert!(
        tokio::time::timeout(Duration::from_millis(40), first.response.message())
            .await
            .is_err()
    );
    let mut second = fixture.open().await;
    second.headers_only().await;
    let Response::ImmediateResponse(error) = second.recv().await else {
        panic!();
    };
    assert_eq!(error.status.unwrap().code, 429);
    assert_eq!(error.details, "mesh_ext_proc_admission_full");
    assert_eq!(fixture.worker.load(), 0);
    first.response_headers(true).await;
    let mut third = fixture.open().await;
    third.routed().await;
    third.response_headers(true).await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn queued_uploads_and_global_retained_buffers_have_independent_limits() {
    let fixture = Fixture::new(RouterConfig {
        max_concurrent_requests: 1,
        ext_proc: ExtProcConfig {
            max_message_bytes: 131_072,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    let mut active = fixture.open().await;
    active.routed().await;
    let mut queued = fixture.open().await;
    queued.headers_only().await;
    for _ in 0..3 {
        queued.body(&vec![b' '; 64 * 1024], false).await;
    }
    let Response::ImmediateResponse(error) = queued.recv().await else {
        panic!();
    };
    assert_eq!(error.status.unwrap().code, 429);
    assert_eq!(error.details, "mesh_ext_proc_admission_buffer_full");
    active.response_headers(true).await;
    fixture.runtime.shutdown().await.unwrap();

    let fixture = Fixture::new(RouterConfig {
        ext_proc: ExtProcConfig {
            max_body_bytes: 131_072,
            max_buffered_bytes: 131_072,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    let mut first = fixture.open().await;
    first.headers_only().await;
    first.body(&vec![b' '; 64 * 1024], false).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(40), first.response.message())
            .await
            .is_err()
    );
    let mut second = fixture.open().await;
    second.headers_only().await;
    second.body(&vec![b' '; 128 * 1024], false).await;
    let Response::ImmediateResponse(error) = second.recv().await else {
        panic!();
    };
    assert_eq!(error.status.unwrap().code, 503);
    assert_eq!(error.details, "mesh_ext_proc_buffer_budget_exhausted");
    first.response_headers(true).await;
    let mut recovered = fixture.open().await;
    recovered.routed().await;
    recovered.response_headers(true).await;
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn runtime_wait_is_cancel_safe_and_repeatable() {
    let mut fixture = Fixture::new(RouterConfig::default()).await;
    assert!(
        tokio::time::timeout(Duration::from_millis(10), fixture.runtime.wait())
            .await
            .is_err()
    );
    (fixture.runtime.shutdown_handle())();
    fixture.runtime.wait().await.unwrap();
    fixture.runtime.wait().await.unwrap();
    fixture.runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn health_watch_is_bounded_releases_dropped_subscribers_and_closes_unknown_services() {
    let fixture = Fixture::new(RouterConfig {
        ext_proc: ExtProcConfig {
            max_streams: 1,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
    let mut client = HealthClient::connect(format!("http://{}", fixture.runtime.address))
        .await
        .unwrap();
    let mut first = client
        .watch(HealthCheckRequest {
            service: "unknown".into(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(first.message().await.unwrap().unwrap().status, 3);
    assert_eq!(
        client
            .watch(HealthCheckRequest {
                service: String::new()
            })
            .await
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    drop(first);
    let mut watch = tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            match client
                .watch(HealthCheckRequest {
                    service: "unknown".into(),
                })
                .await
            {
                Ok(response) => break response.into_inner(),
                Err(error) => assert_eq!(error.code(), tonic::Code::ResourceExhausted),
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(watch.message().await.unwrap().unwrap().status, 3);
    // Registry changes do not generate duplicate SERVICE_UNKNOWN notifications.
    fixture.worker.set_healthy(false);
    assert!(
        tokio::time::timeout(Duration::from_millis(300), watch.message())
            .await
            .is_err()
    );
    (fixture.runtime.shutdown_handle())();
    assert!(
        tokio::time::timeout(Duration::from_secs(2), watch.message())
            .await
            .unwrap()
            .unwrap()
            .is_none()
    );
    fixture.runtime.shutdown().await.unwrap();
}

#[test]
fn listener_validation_ignores_inactive_executor_and_accepts_ephemeral_ports() {
    let mut config = RouterConfig {
        port: 0,
        ext_proc: ExtProcConfig {
            enabled: true,
            listen: "127.0.0.1:9002".parse().unwrap(),
            executor_listen: "127.0.0.1:9002".parse().unwrap(),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(config.validate().is_ok());
    config.mode = mesh::config::RoutingMode::PrefillDecode {
        prefill_urls: vec![],
        decode_urls: vec![],
        prefill_policy: None,
        decode_policy: None,
    };
    assert!(config
        .validate()
        .unwrap_err()
        .to_string()
        .contains("overlaps"));
    config.ext_proc.listen.set_port(0);
    config.ext_proc.executor_listen.set_port(0);
    assert!(config.validate().is_ok());
}

fn selected_header(mutation: &pb::HeaderMutation, name: &str) -> Option<String> {
    mutation
        .set_headers
        .iter()
        .filter_map(|h| h.header.as_ref())
        .find(|h| h.key == name)
        .map(|h| String::from_utf8(h.raw_value.clone()).unwrap())
}

#[tokio::test]
async fn correlation_ids_follow_config_and_survive_request_validation_and_early_replies() {
    for custom in [false, true] {
        let fixture = Fixture::new(RouterConfig {
            request_id_headers: custom.then(|| vec!["x-customer-id".into(), "x-trace-id".into()]),
            ..Default::default()
        })
        .await;
        let cases: Vec<(Vec<(&str, &str)>, Option<&str>)> = if custom {
            vec![
                (
                    vec![
                        ("x-request-id", "ignored"),
                        ("X-Customer-ID", "customer"),
                        ("x-trace-id", "trace"),
                    ],
                    Some("customer"),
                ),
                (
                    vec![("x-request-id", "ignored"), ("x-trace-id", "trace")],
                    Some("trace"),
                ),
                (vec![("x-request-id", "ignored")], None),
            ]
        } else {
            vec![
                (
                    vec![("x-correlation-id", "correlation")],
                    Some("correlation"),
                ),
                (vec![("x-trace-id", "trace")], Some("trace")),
                (vec![("request-id", "request")], Some("request")),
                (
                    vec![
                        ("x-correlation-id", "correlation"),
                        ("x-request-id", "first"),
                        ("x-request-id", "second"),
                    ],
                    Some("first"),
                ),
                (
                    vec![("x-request-id", ""), ("x-correlation-id", "correlation")],
                    Some(""),
                ),
                (vec![], None),
            ]
        };
        for (ids, expected) in cases {
            let mut stream = fixture.open().await;
            let mut headers = vec![
                (":method", "POST"),
                (":path", "/generate"),
                ("content-type", "application/json"),
            ];
            headers.extend(ids);
            stream
                .send(Request::RequestHeaders(Stream::headers(&headers, false)))
                .await;
            stream.body(br#"{"text":"hi"}"#, true).await;
            let Response::RequestHeaders(response) = stream.recv().await else {
                panic!("expected request headers");
            };
            let id = selected_header(
                &response.response.unwrap().header_mutation.unwrap(),
                "x-request-id",
            )
            .unwrap();
            if let Some(expected) = expected {
                assert_eq!(id, expected);
            } else {
                assert!(id.starts_with("gnt-") && id.len() == 28, "{id}");
            }
            assert!(matches!(stream.recv().await, Response::RequestBody(_)));
            stream
                .send(Request::ResponseHeaders(Stream::headers(
                    &[(":status", "200"), ("x-request-id", "backend-id")],
                    true,
                )))
                .await;
            let Response::ResponseHeaders(response) = stream.recv().await else {
                panic!("expected response headers");
            };
            assert_eq!(
                selected_header(
                    &response.response.unwrap().header_mutation.unwrap(),
                    "x-request-id"
                ),
                Some(id)
            );
        }
        for (path, content_type, body, status) in [
            ("/unsupported", "application/json", None, 404),
            ("/generate", "text/plain", None, 415),
            (
                "/generate",
                "application/json",
                Some(b"broken".as_slice()),
                400,
            ),
        ] {
            let mut stream = fixture.open().await;
            stream
                .send(Request::RequestHeaders(Stream::headers(
                    &[
                        (":method", "POST"),
                        (":path", path),
                        ("content-type", content_type),
                        ("x-trace-id", "error-id"),
                    ],
                    false,
                )))
                .await;
            if let Some(body) = body {
                stream.body(body, true).await;
            }
            let Response::ImmediateResponse(response) = stream.recv().await else {
                panic!("expected error");
            };
            assert_eq!(response.status.unwrap().code, status);
            assert_eq!(
                selected_header(&response.headers.unwrap(), "x-request-id").as_deref(),
                Some("error-id")
            );
        }
        let mut stream = fixture.open().await;
        stream
            .send(Request::RequestHeaders(Stream::headers(
                &[
                    (":method", "POST"),
                    (":path", "/generate"),
                    ("content-type", "application/json"),
                    ("x-trace-id", "early-id"),
                ],
                false,
            )))
            .await;
        stream
            .send(Request::ResponseHeaders(Stream::headers(
                &[(":status", "413")],
                true,
            )))
            .await;
        let Response::ResponseHeaders(response) = stream.recv().await else {
            panic!("expected local reply");
        };
        assert_eq!(
            selected_header(
                &response.response.unwrap().header_mutation.unwrap(),
                "x-request-id"
            )
            .as_deref(),
            Some("early-id")
        );
        fixture.runtime.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn correlation_id_is_returned_when_admission_rejects_at_headers() {
    let fixture = Fixture::new(RouterConfig {
        max_concurrent_requests: 1,
        queue_size: 0,
        ..Default::default()
    })
    .await;
    let mut active = fixture.open().await;
    active.routed().await;
    let mut rejected = fixture.open().await;
    rejected
        .send(Request::RequestHeaders(Stream::headers(
            &[
                (":method", "POST"),
                (":path", "/generate"),
                ("content-type", "application/json"),
                ("x-correlation-id", "rejected-id"),
            ],
            false,
        )))
        .await;
    let Response::ImmediateResponse(response) = rejected.recv().await else {
        panic!("expected rejection");
    };
    assert_eq!(response.status.unwrap().code, 429);
    assert_eq!(
        selected_header(&response.headers.unwrap(), "x-request-id").as_deref(),
        Some("rejected-id")
    );
    active.response_headers(true).await;
    fixture.runtime.shutdown().await.unwrap();
}
