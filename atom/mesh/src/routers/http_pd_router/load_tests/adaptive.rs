use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Default)]
struct Probe {
    dispatched: AtomicUsize,
    completed: AtomicUsize,
    dropped: AtomicUsize,
    feedback: std::sync::Mutex<Option<PrefillFeedback>>,
}

#[derive(Debug)]
struct Reservation(Arc<Probe>);

impl PrefillReservation for Reservation {
    fn mark_dispatched(&mut self) {
        self.0.dispatched.fetch_add(1, Ordering::SeqCst);
    }
    fn complete(&mut self, feedback: PrefillFeedback) {
        self.0.completed.fetch_add(1, Ordering::SeqCst);
        *self.0.feedback.lock().unwrap() = Some(feedback);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.0.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

fn start(p: &GatedServer, d: &GatedServer) -> (JoinHandle<Response>, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    (
        dispatch_with_reservation(
            DispatchKind::Atom,
            p,
            d,
            true,
            Some(Box::new(Reservation(probe.clone()))),
        ),
        probe,
    )
}

#[tokio::test]
async fn adaptive_reservation_completes_after_p_body_and_before_d_stream() {
    let (mut p, mut d) = servers().await;
    let (task, probe) = start(&p, &d);
    p.wait_entered().await;
    assert_eq!(probe.dispatched.load(Ordering::SeqCst), 1);
    assert_eq!(probe.completed.load(Ordering::SeqCst), 0);
    assert_eq!(probe.dropped.load(Ordering::SeqCst), 0);
    p.respond(axum::Json(json!({"kv_transfer_params":{"dp_rank":0},"usage":{"prompt_tokens":100,"prompt_tokens_details":{"cached_tokens":80}}})).into_response());
    d.wait_entered().await;
    assert_eq!(probe.completed.load(Ordering::SeqCst), 1);
    assert_eq!(probe.dropped.load(Ordering::SeqCst), 1);
    let feedback = probe.feedback.lock().unwrap().unwrap();
    assert_eq!(
        (feedback.input_tokens, feedback.cached_tokens),
        (Some(100), Some(80))
    );
    assert_eq!(p.worker.load(), 0);
    assert_eq!(d.worker.load(), 1);
    let (stream, tx) = stream_response(StatusCode::OK);
    d.respond(stream);
    let response = result(task).await;
    assert_eq!(d.worker.load(), 1);
    drop(response);
    drop(tx);
    wait_load(&d.worker, 0).await;
    assert_eq!(probe.dropped.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn adaptive_prefill_errors_release_without_confirming_cache() {
    for response in [
        StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Body::from("bad JSON").into_response(),
        axum::Json(json!({})).into_response(),
    ] {
        let (mut p, d) = servers().await;
        let (task, probe) = start(&p, &d);
        p.wait_entered().await;
        p.respond(response);
        assert!(result(task).await.status().is_server_error());
        assert_eq!(probe.completed.load(Ordering::SeqCst), 0);
        assert_eq!(probe.dropped.load(Ordering::SeqCst), 1);
        assert_eq!(p.worker.load(), 0);
        assert_eq!(d.worker.load(), 0);
    }
}

#[tokio::test]
async fn adaptive_cancellation_while_p_is_pending_releases_reservation() {
    let (p, d) = servers().await;
    let (task, probe) = start(&p, &d);
    p.wait_entered().await;
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(probe.completed.load(Ordering::SeqCst), 0);
    assert_eq!(probe.dropped.load(Ordering::SeqCst), 1);
    assert_eq!(p.worker.load(), 0);
    assert_eq!(d.worker.load(), 0);
}
