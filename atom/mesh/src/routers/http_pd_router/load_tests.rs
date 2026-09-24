//! Exercise real HTTP dispatch while independently holding P and D responses.
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
    response: Option<oneshot::Sender<Response>>,
    task: JoinHandle<()>,
}

impl GatedServer {
    async fn start(role: WorkerType) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let entered = Arc::new(Notify::new());
        let notify = entered.clone();
        let (tx, rx) = oneshot::channel::<Response>();
        let response = Arc::new(Mutex::new(Some(rx)));
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move || {
                let notify = notify.clone();
                let response = response.clone();
                async move {
                    let rx = response.lock().await.take().unwrap();
                    notify.notify_one();
                    rx.await
                        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
                }
            }),
        );
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            worker: Arc::new(BasicWorkerBuilder::new(url).worker_type(role).build()),
            entered,
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
            model_id: None,
            headers: None,
        };
        match kind {
            DispatchKind::Atom => {
                router
                    .dispatch_atom_relay_internal(
                        None,
                        json!({}),
                        json!({}),
                        context,
                        prefill,
                        decode,
                        ctx,
                        Instant::now(),
                        None,
                    )
                    .await
            }
            DispatchKind::Vllm => {
                router
                    .dispatch_vllm_mooncake_internal(
                        None,
                        json!({}),
                        json!({}),
                        context,
                        prefill,
                        decode,
                        Instant::now(),
                        None,
                    )
                    .await
            }
            DispatchKind::Sglang => {
                router
                    .execute_dual_dispatch_internal(
                        None,
                        json!({}),
                        context,
                        prefill,
                        decode,
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
