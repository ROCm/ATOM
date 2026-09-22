use std::{sync::Arc, time::Duration};

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
        Stream { sender, response }
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
                ..Default::default()
            })
            .await
            .unwrap();
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
        self.send(Request::RequestHeaders(Self::headers(
            &[
                (":method", "POST"),
                (":path", "/v1/chat/completions"),
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
    let destination = response
        .header_mutation
        .unwrap()
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
    assert_eq!(watch.message().await.unwrap().unwrap().status, 1);
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
        use std::collections::BTreeMap;
        self.sender
            .send(pb::ProcessingRequest {
                request: Some(Request::RequestBody(pb::HttpBody {
                    body: Self::BODY.to_vec(),
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
    for addresses in [vec![], vec!["127.0.0.1:1"]] {
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
    let mut stream = fixture.open().await;
    stream
        .send(Request::RequestHeaders(Stream::headers(
            &[
                (":method", "POST"),
                (":path", "/generate"),
                ("content-type", "application/json"),
            ],
            false,
        )))
        .await;
    stream
        .body(
            br#"{"model":"test-model","input_ids":[1,2,3,4],"stream":false}"#,
            true,
        )
        .await;
    assert_eq!(stream.destination().await, "127.0.0.1:18001");
    assert!(matches!(stream.recv().await, Response::RequestBody(_)));
    stream.response_headers(true).await;
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
    stream.error(400).await;
    fixture.worker.set_healthy(false);
    let mut stream = fixture.open().await;
    stream.headers_only().await;
    stream.body(Stream::BODY, true).await;
    stream.error(503).await;
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
            axum::routing::post(move || async move {
                calls.fetch_add(1, Ordering::SeqCst);
                axum::Json(serde_json::json!({"choices":[{"text":"ok"}]}))
            }),
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
    for attempt in 0..3 {
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
            break;
        }
        // Registry changes after selection cannot cause a second placement.
        for worker in &workers {
            fixture.app.worker_registry.remove_by_url(worker.url());
        }
        let response = client
            .post(&url)
            .header("x-mesh-execution-id", id)
            .body(if attempt == 0 {
                body.to_vec()
            } else {
                b"{}".to_vec()
            })
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), if attempt == 0 { 200 } else { 400 });
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
    assert_eq!(calls.load(Ordering::SeqCst), 2);
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
    let http = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let http_address = http.local_addr().unwrap();
    let grpc = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let grpc_address = grpc.local_addr().unwrap();
    let metrics = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let metrics_port = metrics.local_addr().unwrap().port();
    drop(metrics);
    let logs = tempfile::NamedTempFile::new().unwrap();
    drop(http);
    drop(grpc);
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_atomesh"));
    command.args([
        "launch",
        "--host",
        "127.0.0.1",
        "--port",
        &http_address.port().to_string(),
        "--policy",
        "round_robin",
        "--prometheus-port",
        &metrics_port.to_string(),
        "--log-level",
        "warn",
        "--ext-proc-listen",
        &grpc_address.to_string(),
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
            .stdout(logs.reopen().unwrap())
            .stderr(logs.reopen().unwrap())
            .spawn()
            .unwrap(),
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            assert!(
                process.0.try_wait().unwrap().is_none(),
                "{}",
                std::fs::read_to_string(logs.path()).unwrap()
            );
            if client
                .get(format!("http://{http_address}/health"))
                .send()
                .await
                .is_ok()
            {
                if !ext_proc {
                    break;
                }
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
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap();
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
        assert!(tokio::net::TcpStream::connect(grpc_address).await.is_err());
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
    assert!(tokio::net::TcpStream::connect(grpc_address).await.is_err());
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
