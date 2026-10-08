//! PD load accounting and credential forwarding to prefill and decode workers.
use super::*;
use crate::core::{BasicWorkerBuilder, WorkerType};
use axum::{routing::post, Router};
use bytes::Bytes;
use http_body_util::BodyExt;
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot, Mutex, Notify},
    task::JoinHandle,
    time::{sleep, timeout, Duration},
};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Clone, Copy, Debug)]
enum DispatchKind {
    Atom,
    Vllm,
    Sglang,
}

const KINDS: [DispatchKind; 3] = [DispatchKind::Atom, DispatchKind::Vllm, DispatchKind::Sglang];

struct GatedServer {
    worker: Arc<dyn Worker>,
    entered: Arc<Notify>,
    captured: mpsc::UnboundedReceiver<HeaderMap>,
    response: Option<oneshot::Sender<Response>>,
    task: JoinHandle<()>,
}

impl GatedServer {
    async fn start(role: WorkerType) -> Self {
        Self::start_with_api_key(role, None).await
    }

    async fn start_with_api_key(role: WorkerType, api_key: Option<&str>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let entered = Arc::new(Notify::new());
        let notify = entered.clone();
        let (tx, rx) = oneshot::channel::<Response>();
        let response = Arc::new(Mutex::new(Some(rx)));
        let (capture, captured) = mpsc::unbounded_channel();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap| {
                let notify = notify.clone();
                let response = response.clone();
                let capture = capture.clone();
                async move {
                    capture.send(headers).unwrap();
                    let rx = response.lock().await.take().unwrap();
                    notify.notify_one();
                    rx.await
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                }
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut worker = BasicWorkerBuilder::new(url).worker_type(role);
        if let Some(key) = api_key {
            worker = worker.api_key(key);
        }
        Self {
            worker: Arc::new(worker.build()),
            entered,
            captured,
            response: Some(tx),
            task,
        }
    }

    async fn wait_entered(&self) {
        timeout(Duration::from_secs(5), self.entered.notified())
            .await
            .unwrap();
    }

    fn respond(&mut self, response: Response) {
        self.response.take().unwrap().send(response).unwrap();
    }
}

impl Drop for GatedServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn servers() -> (GatedServer, GatedServer) {
    (
        GatedServer::start(WorkerType::Prefill {
            bootstrap_port: None,
        })
        .await,
        GatedServer::start(WorkerType::Decode).await,
    )
}

fn prefill_response() -> Response {
    axum::Json(json!({"kv_transfer_params": {"dp_rank": 0}})).into_response()
}

fn stream_response(
    status: StatusCode,
) -> (
    Response,
    mpsc::UnboundedSender<Result<Bytes, std::io::Error>>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let body = Body::from_stream(UnboundedReceiverStream::new(rx));
    let response = Response::builder()
        .status(status)
        .header(CONTENT_TYPE, "text/event-stream")
        .body(body)
        .unwrap();
    (response, tx)
}

fn dispatch(
    kind: DispatchKind,
    p: &GatedServer,
    d: &GatedServer,
    streaming: bool,
) -> JoinHandle<Response> {
    dispatch_with_headers(kind, p, d, streaming, None)
}

fn dispatch_with_headers(
    kind: DispatchKind,
    p: &GatedServer,
    d: &GatedServer,
    streaming: bool,
    headers: Option<HeaderMap>,
) -> JoinHandle<Response> {
    let mut router = tests::create_test_pd_router();
    let prefill = p.worker.clone();
    let decode = d.worker.clone();
    let mut info = AtomPrefillInfo::default();
    info.tp_sizes.insert(prefill.url().to_string(), 1);
    let atom = Arc::new(AtomAdapter::new(Arc::new(info)));
    let ctx = atom
        .prepare_pair(prefill.as_ref(), decode.as_ref())
        .unwrap();
    router.atom_adapter = Some(atom);
    tokio::spawn(async move {
        let context = PDRequestContext {
            route: "/v1/chat/completions",
            batch_size: None,
            is_stream: streaming,
            return_logprob: false,
            request_text: None,
            tokens: None,
            planner: None,
            model_id: None,
            headers: headers.clone().map(Arc::new),
        };
        let placement = router
            .reserve_pair(
                PlacementPlan::Pair {
                    prefill,
                    decode,
                    prefill_policy: "round_robin",
                    decode_policy: "round_robin",
                },
                headers.as_ref(),
            )
            .unwrap();
        match kind {
            DispatchKind::Atom => {
                router
                    .dispatch_atom_relay_internal(
                        headers.as_ref(),
                        json!({}),
                        json!({}),
                        context,
                        placement,
                        ctx,
                        Instant::now(),
                        None,
                    )
                    .await
            }
            DispatchKind::Vllm => {
                router
                    .dispatch_vllm_mooncake_internal(
                        headers.as_ref(),
                        json!({}),
                        json!({}),
                        context,
                        placement,
                        Instant::now(),
                        None,
                    )
                    .await
            }
            DispatchKind::Sglang => {
                router
                    .execute_dual_dispatch_internal(
                        headers.as_ref(),
                        json!({}),
                        context,
                        placement,
                        Instant::now(),
                    )
                    .await
            }
        }
    })
}

async fn wait_load(worker: &Arc<dyn Worker>, expected: usize) {
    timeout(Duration::from_secs(5), async {
        while worker.load() != expected {
            sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

async fn result(task: JoinHandle<Response>) -> Response {
    timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
}

struct IngressBackend {
    worker: Arc<dyn Worker>,
    requests: mpsc::UnboundedReceiver<(Value, oneshot::Sender<Response>)>,
    task: JoinHandle<()>,
}

impl IngressBackend {
    async fn start(role: WorkerType) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (capture, requests) = mpsc::unbounded_channel();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |body: Bytes| {
                let capture = capture.clone();
                async move {
                    let (reply, response) = oneshot::channel();
                    capture
                        .send((serde_json::from_slice::<Value>(&body).unwrap(), reply))
                        .unwrap();
                    response.await.unwrap()
                }
            }),
        );
        Self {
            worker: Arc::new(
                BasicWorkerBuilder::new(url)
                    .worker_type(role)
                    .model_id("budget-test")
                    .build(),
            ),
            requests,
            task: tokio::spawn(async move { axum::serve(listener, app).await.unwrap() }),
        }
    }

    async fn next_request(&mut self) -> (Value, oneshot::Sender<Response>) {
        timeout(Duration::from_secs(5), self.requests.recv())
            .await
            .unwrap()
            .unwrap()
    }
}

impl Drop for IngressBackend {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn pd_ingress_releases_preparation_budget_before_prefill_body_finishes() {
    use crate::{
        app_context::AppContext,
        config::{PolicyConfig, PreparePoolConfig, RouterConfig, RoutingMode},
        core::prepare_pool::PreparePoolRuntime,
        routers::ingress::InferenceEnvelope,
    };

    let mut prefill = IngressBackend::start(WorkerType::Prefill {
        bootstrap_port: None,
    })
    .await;
    let mut decode = IngressBackend::start(WorkerType::Decode).await;
    let raw = Bytes::from_static(
        br#"{"model":"budget-test","messages":[{"role":"user","content":"hello"}],"vendor":{"keep":[1,2]}}"#,
    );
    let config = RouterConfig {
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![],
            decode_urls: vec![],
            prefill_policy: None,
            decode_policy: None,
        },
        policy: PolicyConfig::RoundRobin,
        backend: BackendType::Sglang,
        disable_retries: true,
        prepare_pool: PreparePoolConfig {
            workers: Some(1),
            max_retained_input_bytes: raw.len(),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut pool = PreparePoolRuntime::new(config.resolved_prepare_pool()).unwrap();
    let handle = pool.handle();
    let app = Arc::new(
        AppContext::from_config(config, 5, handle.clone())
            .await
            .unwrap(),
    );
    app.worker_registry.register(prefill.worker.clone());
    app.worker_registry.register(decode.worker.clone());
    let router = Arc::new(PDRouter::new(&app).await.unwrap());
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    let parse = || {
        InferenceEnvelope::parse(
            "/v1/chat/completions".parse().unwrap(),
            headers.clone(),
            raw.clone(),
            &app,
        )
    };
    let dispatch = |request| {
        let app = app.clone();
        let router = router.clone();
        tokio::spawn(async move { router.route_inference(request, &app).await })
    };
    let first = parse().await.unwrap();
    assert_eq!(handle.stats().retained_input_bytes, raw.len());
    let first = dispatch(first);
    let (first_prefill, prefill_reply) = prefill.next_request().await;
    let (first_decode, decode_reply) = decode.next_request().await;
    assert_eq!(first_prefill["vendor"], json!({"keep":[1,2]}));
    assert_eq!(first_decode["vendor"], first_prefill["vendor"]);

    let (body_started, started) = oneshot::channel();
    let (release_body, body_gate) = oneshot::channel();
    let gated_body = futures_util::stream::once(async move {
        body_started.send(()).unwrap();
        body_gate.await.unwrap();
        Ok::<_, std::io::Error>(Bytes::from_static(b"{}"))
    });
    prefill_reply
        .send(
            Response::builder()
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from_stream(gated_body))
                .unwrap(),
        )
        .unwrap();
    let (decode_sent, decode_done) = oneshot::channel();
    let decode_body = futures_util::stream::once(async move {
        decode_sent.send(()).unwrap();
        Ok::<_, std::io::Error>(Bytes::from_static(br#"{"choices":[]}"#))
    });
    decode_reply
        .send(
            Response::builder()
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from_stream(decode_body))
                .unwrap(),
        )
        .unwrap();
    timeout(Duration::from_secs(5), started)
        .await
        .unwrap()
        .unwrap();
    timeout(Duration::from_secs(5), decode_done)
        .await
        .unwrap()
        .unwrap();
    assert!(
        !first.is_finished(),
        "prefill body must still block the first request"
    );
    assert_eq!(
        handle.stats().retained_input_bytes,
        0,
        "backend response latency must not retain the preparation input budget"
    );

    // This identical input exhausts the entire configured budget on its own.
    // It must reach both workers while the first prefill body is still gated.
    let second = dispatch(parse().await.unwrap());
    let (second_prefill, prefill_reply) = prefill.next_request().await;
    let (second_decode, decode_reply) = decode.next_request().await;
    assert_eq!(second_prefill["vendor"], first_prefill["vendor"]);
    assert_eq!(second_decode["vendor"], first_decode["vendor"]);
    prefill_reply.send(prefill_response()).unwrap();
    decode_reply
        .send(axum::Json(json!({"choices":[]})).into_response())
        .unwrap();
    let second = result(second).await;
    assert_eq!(second.status(), StatusCode::OK);
    second.into_body().collect().await.unwrap();
    assert!(!first.is_finished());
    assert_eq!(handle.stats().retained_input_bytes, 0);

    release_body.send(()).unwrap();
    let first = result(first).await;
    assert_eq!(first.status(), StatusCode::OK);
    first.into_body().collect().await.unwrap();
    assert_eq!((prefill.worker.load(), decode.worker.load()), (0, 0));
    assert_eq!(pool.shutdown().await.remaining_workers, 0);
}

#[tokio::test]
async fn pd_dispatch_uses_each_workers_credentials_without_leaking_client_keys() {
    for kind in KINDS {
        for (prefill_key, decode_key) in [
            (Some("prefill-secret"), Some("decode-secret")),
            (Some("prefill-secret"), None),
            (None, Some("decode-secret")),
            (None, None),
        ] {
            let mut p = GatedServer::start_with_api_key(
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
                prefill_key,
            )
            .await;
            let mut d = GatedServer::start_with_api_key(WorkerType::Decode, decode_key).await;
            let mut headers = HeaderMap::new();
            headers.insert(
                "authorization",
                HeaderValue::from_static("Bearer client-secret"),
            );
            headers.append("x-api-key", HeaderValue::from_static("client-key"));
            headers.append("X-Api-Key", HeaderValue::from_static("second-client-key"));
            headers.insert("x-request-id", HeaderValue::from_static("pd-credentials"));
            headers.insert("cookie", HeaderValue::from_static("private=session"));
            let task = dispatch_with_headers(kind, &p, &d, false, Some(headers));

            p.wait_entered().await;
            p.respond(prefill_response());
            d.wait_entered().await;
            d.respond(axum::Json(json!({"choices": []})).into_response());
            let response = result(task).await;
            assert_eq!(response.status(), StatusCode::OK, "{kind:?}");
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();

            for (server, key) in [(&mut p, prefill_key), (&mut d, decode_key)] {
                let received = server.captured.try_recv().unwrap();
                assert_eq!(received.get_all("authorization").iter().count(), 1);
                if let Some(key) = key {
                    assert_eq!(received["authorization"], format!("Bearer {key}"));
                    assert!(
                        !received.contains_key("x-api-key"),
                        "{kind:?}: client API keys leaked to a worker with its own credentials"
                    );
                } else {
                    assert_eq!(received["authorization"], "Bearer client-secret");
                    assert_eq!(
                        received.get_all("x-api-key").iter().collect::<Vec<_>>(),
                        ["client-key", "second-client-key"]
                    );
                }
                assert_eq!(received["x-request-id"], "pd-credentials");
                assert!(!received.contains_key("cookie"));
            }
        }
    }
}

#[derive(Clone, Copy)]
enum StreamEnd {
    Eof,
    Done,
    Disconnect,
    Error,
}

async fn check_streaming_lifecycle(kind: DispatchKind, end: StreamEnd) {
    let (mut p, mut d) = servers().await;
    let task = dispatch(kind, &p, &d, true);
    p.wait_entered().await;
    assert_eq!(
        p.worker.load(),
        1,
        "{kind:?}: P must count before response headers"
    );
    assert_eq!(d.worker.load(), 1, "{kind:?}: selected D must be reserved");

    p.respond(prefill_response());
    d.wait_entered().await;
    wait_load(&p.worker, 0).await;
    assert_eq!(d.worker.load(), 1, "D must count while waiting for headers");

    let (response, tx) = stream_response(StatusCode::OK);
    d.respond(response);
    let response = result(task).await;
    assert_eq!(p.worker.load(), 0, "D streaming must not increment P again");
    assert_eq!(
        d.worker.load(),
        1,
        "moving the guard must not double-count D"
    );
    let mut body = response.into_body();
    tx.send(Ok(Bytes::from_static(b"data: {\"text\":\"hello\"}\n\n")))
        .unwrap();
    timeout(Duration::from_secs(5), body.frame())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(d.worker.load(), 1);

    match end {
        StreamEnd::Eof => drop(tx),
        StreamEnd::Done => {
            tx.send(Ok(Bytes::from_static(b"data: [DONE]\n\n")))
                .unwrap();
        }
        StreamEnd::Disconnect => {
            drop(body);
            assert_eq!(d.worker.load(), 0);
            return;
        }
        StreamEnd::Error => {
            tx.send(Err(std::io::Error::other("upstream failed")))
                .unwrap();
        }
    }
    let completed = timeout(Duration::from_secs(5), body.collect())
        .await
        .unwrap();
    if matches!(end, StreamEnd::Error) {
        assert!(completed.is_err());
    } else {
        assert!(completed.is_ok());
    }
    assert_eq!(p.worker.load(), 0);
    assert_eq!(d.worker.load(), 0);
}

#[tokio::test]
async fn streaming_load_covers_prefill_and_decode_waits() {
    for kind in KINDS {
        check_streaming_lifecycle(kind, StreamEnd::Eof).await;
    }
}

#[tokio::test]
async fn streaming_done_releases_load() {
    for kind in KINDS {
        check_streaming_lifecycle(kind, StreamEnd::Done).await;
    }
}

#[tokio::test]
async fn client_disconnect_releases_streaming_load() {
    for kind in KINDS {
        check_streaming_lifecycle(kind, StreamEnd::Disconnect).await;
    }
}

#[tokio::test]
async fn upstream_stream_error_releases_load() {
    for kind in KINDS {
        check_streaming_lifecycle(kind, StreamEnd::Error).await;
    }
}

#[tokio::test]
async fn atom_prefill_failure_releases_pair_without_dispatching_decode() {
    for response in [
        StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Body::from("invalid JSON").into_response(),
        axum::Json(json!({})).into_response(),
    ] {
        let (mut p, d) = servers().await;
        let task = dispatch(DispatchKind::Atom, &p, &d, true);
        p.wait_entered().await;
        assert_eq!(p.worker.load(), 1);
        p.respond(response);
        assert!(result(task).await.status().is_server_error());
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 0);
        assert!(timeout(Duration::from_millis(20), d.entered.notified())
            .await
            .is_err());
    }
}

#[tokio::test]
async fn decode_http_error_does_not_reacquire_prefill_load() {
    for kind in KINDS {
        let (mut p, mut d) = servers().await;
        let task = dispatch(kind, &p, &d, true);
        p.wait_entered().await;
        p.respond(prefill_response());
        d.wait_entered().await;
        wait_load(&p.worker, 0).await;
        d.respond((StatusCode::SERVICE_UNAVAILABLE, "decode failed").into_response());
        let response = result(task).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 1);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&bytes).contains("decode failed"));
        assert_eq!(d.worker.load(), 0);
    }
}

#[tokio::test]
async fn cancelling_dispatch_releases_pending_load() {
    for kind in [DispatchKind::Atom, DispatchKind::Sglang] {
        let (p, d) = servers().await;
        let task = dispatch(kind, &p, &d, true);
        p.wait_entered().await;
        assert_eq!(p.worker.load(), 1);
        assert_eq!(d.worker.load(), 1);
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 0);
    }
}

#[tokio::test]
async fn vllm_detached_prefill_retains_load_until_it_finishes() {
    let (mut p, mut d) = servers().await;
    let task = dispatch(DispatchKind::Vllm, &p, &d, true);
    p.wait_entered().await;
    d.wait_entered().await;
    let (response, tx) = stream_response(StatusCode::OK);
    d.respond(response);
    let response = result(task).await;
    drop(response);
    assert_eq!(d.worker.load(), 0);
    assert_eq!(p.worker.load(), 1);
    p.respond(prefill_response());
    wait_load(&p.worker, 0).await;
    drop(tx);
}

#[tokio::test]
async fn nonstreaming_dispatch_releases_prefill_before_decode_body() {
    for kind in KINDS {
        let (mut p, mut d) = servers().await;
        let task = dispatch(kind, &p, &d, false);
        p.wait_entered().await;
        assert_eq!(p.worker.load(), 1);
        p.respond(prefill_response());
        d.wait_entered().await;
        wait_load(&p.worker, 0).await;
        assert_eq!(d.worker.load(), 1);
        d.respond(axum::Json(json!({"text":"done"})).into_response());
        let response = result(task).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 0);
    }
}

#[tokio::test]
async fn dual_dispatch_error_cancels_the_pending_peer() {
    for fail_prefill in [true, false] {
        let (mut p, mut d) = servers().await;
        let task = dispatch(DispatchKind::Sglang, &p, &d, true);
        p.wait_entered().await;
        d.wait_entered().await;
        assert_eq!(p.worker.load(), 1);
        assert_eq!(d.worker.load(), 1);
        let failed = if fail_prefill { &mut p } else { &mut d };
        failed.respond(StatusCode::SERVICE_UNAVAILABLE.into_response());
        let response = result(task).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        drop(response);
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 0);
    }
}

async fn check_stalled_dual_dispatch_error(fail_prefill: bool) {
    for streaming in [true, false] {
        let (mut p, mut d) = servers().await;
        let task = dispatch(DispatchKind::Sglang, &p, &d, streaming);
        p.wait_entered().await;
        d.wait_entered().await;
        let (failed, peer) = if fail_prefill {
            (&mut p, &d)
        } else {
            (&mut d, &p)
        };
        let (response, tx) = stream_response(StatusCode::SERVICE_UNAVAILABLE);
        failed.respond(response);

        // Keep the error body open: the peer must be cancelled on headers,
        // without waiting for the failing worker's error payload to arrive.
        wait_load(&peer.worker, 0).await;
        assert_eq!(failed.worker.load(), 1);
        assert!(!task.is_finished());
        tx.send(Ok(Bytes::from_static(b"upstream unavailable")))
            .unwrap();
        drop(tx);

        let response = result(task).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        assert!(String::from_utf8_lossy(&bytes).contains("upstream unavailable"));
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 0);
    }
}

#[tokio::test]
async fn stalled_prefill_error_body_cancels_pending_decode() {
    check_stalled_dual_dispatch_error(true).await;
}

#[tokio::test]
async fn stalled_decode_error_body_cancels_pending_prefill() {
    check_stalled_dual_dispatch_error(false).await;
}
