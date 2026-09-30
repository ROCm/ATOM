//! Completion callbacks and worker accounting with real HTTP response framing.
use super::*;
use crate::{
    core::{BasicWorkerBuilder, WorkerLoadGuard},
    policies::{LoadBalancingPolicy, SelectWorkerInfo},
};
use http_body_util::BodyExt;
use std::sync::Mutex;
use tokio::{net::TcpListener, sync::mpsc, task::JoinHandle};
use tokio_stream::wrappers::UnboundedReceiverStream;

#[derive(Debug, Default)]
struct RecordingPolicy(Mutex<Vec<bool>>);
#[async_trait::async_trait]
impl LoadBalancingPolicy for RecordingPolicy {
    async fn select_worker(
        &self,
        _: &[Arc<dyn Worker>],
        _: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        Some(0)
    }
    fn on_request_complete(&self, _: &str, success: bool) {
        self.0.lock().unwrap().push(success);
    }
    fn name(&self) -> &'static str {
        "recording"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

async fn upstream(response: Response) -> (reqwest::Response, JoinHandle<()>) {
    let response = Arc::new(Mutex::new(Some(response)));
    let app = axum::Router::new().fallback(move || {
        let response = response.lock().unwrap().take().unwrap();
        async move { response }
    });
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let response = reqwest::Client::new().get(url).send().await.unwrap();
    (response, server)
}

fn wrap(response: reqwest::Response) -> (Body, Arc<dyn Worker>, Arc<RecordingPolicy>) {
    let worker: Arc<dyn Worker> = Arc::new(BasicWorkerBuilder::new("http://test-worker").build());
    let policy = Arc::new(RecordingPolicy::default());
    let load = WorkerLoadGuard::new(worker.clone(), None);
    let body = ProxyBody::response(response, worker.clone(), policy.clone(), load).into_body();
    (body, worker, policy)
}

#[tokio::test]
async fn fixed_length_completion_is_recorded_without_an_eof_poll() {
    for status in [
        StatusCode::OK,
        StatusCode::BAD_REQUEST,
        StatusCode::SERVICE_UNAVAILABLE,
    ] {
        let payload = b"{\"ok\":true}";
        let response = Response::builder()
            .status(status)
            .body(Body::from(payload.as_slice()))
            .unwrap();
        let (response, server) = upstream(response).await;
        assert_eq!(response.content_length(), Some(payload.len() as u64));
        let (mut body, worker, policy) = wrap(response);
        let mut received = Vec::new();
        while received.len() < payload.len() {
            let frame = body.frame().await.unwrap().unwrap();
            received.extend_from_slice(frame.data_ref().unwrap());
        }
        assert_eq!(received, payload);
        // A real HTTP server stops reading at Content-Length, without polling None.
        assert_eq!(*policy.0.lock().unwrap(), vec![status.is_success()]);
        drop(body);
        assert_eq!(*policy.0.lock().unwrap(), vec![status.is_success()]);
        assert_eq!(
            worker.circuit_breaker().stats().total_successes,
            u64::from(status.is_success())
        );
        assert_eq!(
            worker.circuit_breaker().stats().total_failures,
            u64::from(status.is_server_error())
        );
        assert_eq!(worker.load(), 0);
        server.abort();
    }
}

#[tokio::test]
async fn empty_response_completes_without_polling_its_body() {
    let (response, server) = upstream(Response::new(Body::empty())).await;
    assert_eq!(response.content_length(), Some(0));
    let (body, worker, policy) = wrap(response);
    drop(body);
    assert_eq!(*policy.0.lock().unwrap(), vec![true]);
    assert_eq!(worker.circuit_breaker().stats().total_successes, 1);
    assert_eq!(worker.load(), 0);
    server.abort();
}

#[tokio::test]
async fn stream_eof_error_and_early_drop_record_distinct_outcomes() {
    for (length, end) in [
        (None, "eof"),
        (None, "error"),
        (None, "cancel"),
        (Some(100), "cancel"),
        (Some(100), "error"),
    ] {
        let (tx, rx) = mpsc::unbounded_channel::<Result<Bytes, std::io::Error>>();
        tx.send(Ok(Bytes::from_static(b"data: hello\n\n"))).unwrap();
        let mut response = Response::builder().header("content-type", "text/event-stream");
        if let Some(length) = length {
            response = response.header("content-length", length);
        }
        let (response, server) = upstream(
            response
                .body(Body::from_stream(UnboundedReceiverStream::new(rx)))
                .unwrap(),
        )
        .await;
        let (mut body, worker, policy) = wrap(response);
        assert!(body.frame().await.unwrap().unwrap().is_data());
        assert!(policy.0.lock().unwrap().is_empty());
        assert_eq!(worker.load(), 1);
        if end == "cancel" {
            drop(body);
        } else {
            if end == "error" {
                tx.send(Err(std::io::Error::other("truncated upstream")))
                    .unwrap();
            }
            drop(tx);
            let result = body.collect().await;
            assert_eq!(result.is_ok(), end == "eof");
        }
        assert_eq!(*policy.0.lock().unwrap(), vec![end == "eof"]);
        assert_eq!(
            worker.circuit_breaker().stats().total_successes,
            u64::from(end == "eof")
        );
        assert_eq!(
            worker.circuit_breaker().stats().total_failures,
            u64::from(end == "error")
        );
        assert_eq!(worker.load(), 0);
        server.abort();
    }
}
