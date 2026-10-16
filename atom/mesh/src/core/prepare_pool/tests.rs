use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
};

use futures::poll;

use super::*;

fn config() -> PoolConfig {
    PoolConfig {
        workers: 1,
        queue_capacity: 2,
        queue_timeout: Duration::from_secs(2),
        prepare_timeout: Duration::from_secs(5),
    }
}

fn start(settings: PoolConfig) -> PreparePoolRuntime {
    let mut runtime = PreparePoolRuntime::new(settings).unwrap();
    runtime.shutdown_grace = Duration::from_millis(50);
    runtime
}

async fn blocked<T: Send + 'static>(
    handle: &PrepareHandle,
    output: T,
    deadline: Instant,
) -> (PrepareTicket<T>, oneshot::Receiver<()>, mpsc::Sender<()>) {
    let (started_tx, started) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let ticket = handle
        .submit(deadline, move |_| {
            let _ = started_tx.send(());
            gate.recv_timeout(Duration::from_secs(5)).unwrap();
            output
        })
        .await
        .unwrap();
    (ticket, started, release)
}

async fn eventually(mut condition: impl FnMut() -> bool) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while !condition() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}

// Retain the worker receiver without scheduling an OS worker, so dispatch
// timing and cancellation can be exercised without scheduler races.
fn deferred_worker(settings: PoolConfig) -> (PreparePoolRuntime, Arc<Receiver<Job>>) {
    PreparePoolRuntime::without_workers(settings)
}

#[tokio::test(flavor = "current_thread")]
async fn burst_fills_execution_and_queue_capacity_without_idle_workers() {
    let mut settings = config();
    settings.workers = 10;
    settings.queue_capacity = 100;
    let (mut runtime, receiver) = deferred_worker(settings);
    let handle = runtime.handle();
    // Dispatch does not wait for an idle OS thread to receive the job.
    let mut tickets = Vec::new();
    for _ in 0..10 {
        tickets.push(
            handle
                .submit(handle.prepare_deadline(), |_| ())
                .await
                .unwrap(),
        );
    }
    assert_eq!(handle.stats().busy_workers, 0);
    assert_eq!(handle.stats().execution_reserved, 10);
    assert_eq!(handle.stats().dispatched_jobs, 10);
    assert_eq!(handle.stats().queued_jobs, 0);
    let mut waiting = Vec::new();
    for _ in 0..100 {
        let mut submit = Box::pin(handle.submit(handle.prepare_deadline(), |_| ()));
        assert!(poll!(&mut submit).is_pending());
        waiting.push(submit);
    }
    assert_eq!(handle.stats().queued_jobs, 100);
    assert!(matches!(
        handle.submit(handle.prepare_deadline(), |_| ()).await,
        Err(PrepareError::Full)
    ));
    drop(waiting);
    assert_eq!(handle.stats().queued_jobs, 0);
    for ticket in tickets {
        receiver.recv().unwrap().execute();
        ticket.wait().await.unwrap();
    }
    assert_eq!(handle.stats().execution_reserved, 0);
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn canceled_waiter_releases_queue_slot_and_input_immediately() {
    let mut runtime = start(config());
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, (), handle.prepare_deadline()).await;
    started.await.unwrap();
    let called = Arc::new(AtomicBool::new(false));
    let job_called = called.clone();
    let input = Arc::new(());
    let job_input = input.clone();
    let mut canceled = Box::pin(handle.submit(handle.prepare_deadline(), move |_| {
        job_called.store(true, Ordering::Release);
        job_input
    }));
    assert!(poll!(&mut canceled).is_pending());
    let mut waiting = Box::pin(handle.submit(handle.prepare_deadline(), |_| 42));
    assert!(poll!(&mut waiting).is_pending());
    assert_eq!(handle.stats().queued_jobs, 2);
    assert!(matches!(
        handle.submit(handle.prepare_deadline(), |_| ()).await,
        Err(PrepareError::Full)
    ));
    drop(canceled);
    assert_eq!(handle.stats().queued_jobs, 1);
    assert_eq!(Arc::strong_count(&input), 1);
    assert_eq!(handle.stats().execution_reserved, 1);
    let mut replacement = Box::pin(handle.submit(handle.prepare_deadline(), |_| ()));
    assert!(poll!(&mut replacement).is_pending());
    drop(replacement);
    release.send(()).unwrap();
    active.wait().await.unwrap();
    assert_eq!(waiting.await.unwrap().wait().await.unwrap(), 42);
    assert!(!called.load(Ordering::Acquire));
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn waiting_jobs_are_fifo_and_new_submitters_cannot_bypass_them() {
    let mut runtime = start(config());
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, (), handle.prepare_deadline()).await;
    started.await.unwrap();
    let mut first = Box::pin(handle.submit(handle.prepare_deadline(), |_| 1));
    assert!(poll!(&mut first).is_pending());
    release.send(()).unwrap();
    active.wait().await.unwrap();
    eventually(|| handle.stats().busy_workers == 0).await;
    // The released permit belongs to the first waiter even before it polls.
    let mut second = Box::pin(handle.submit(handle.prepare_deadline(), |_| 2));
    assert!(poll!(&mut second).is_pending());
    assert_eq!(first.await.unwrap().wait().await.unwrap(), 1);
    assert_eq!(second.await.unwrap().wait().await.unwrap(), 2);
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn queue_and_overall_deadlines_free_waiting_slot_and_input_without_running_job() {
    let mut settings = config();
    settings.queue_timeout = Duration::from_millis(20);
    let mut runtime = start(settings);
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, (), handle.prepare_deadline()).await;
    started.await.unwrap();
    let input = Arc::new(());
    let job_input = input.clone();
    let result = handle
        .submit(handle.prepare_deadline(), move |_| {
            drop(job_input);
            panic!("expired waiting job must not execute");
        })
        .await;
    assert!(matches!(result, Err(PrepareError::QueueTimeout)));
    assert_eq!(handle.stats().queued_jobs, 0);
    assert_eq!(Arc::strong_count(&input), 1);
    assert_eq!(handle.stats().execution_reserved, 1);
    // An overall deadline earlier than the queue deadline reports Timeout.
    let result = handle
        .submit(Instant::now() + Duration::from_millis(5), |_| ())
        .await;
    assert!(matches!(result, Err(PrepareError::Timeout)));
    assert_eq!(handle.stats().queued_jobs, 0);
    release.send(()).unwrap();
    active.wait().await.unwrap();
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn running_cancellation_and_timeout_hold_execution_and_input_until_return() {
    let mut runtime = start(config());
    let handle = runtime.handle();
    let input = Arc::new(());
    for timeout in [false, true] {
        let deadline = if timeout {
            Instant::now() + Duration::from_millis(30)
        } else {
            handle.prepare_deadline()
        };
        let (active, started, release) = blocked(&handle, input.clone(), deadline).await;
        started.await.unwrap();
        if timeout {
            assert!(matches!(active.wait().await, Err(PrepareError::Timeout)));
        } else {
            drop(active);
        }
        // The worker is neither replaced nor released while the job runs.
        assert_eq!(handle.stats().live_workers, 1);
        assert_eq!(handle.stats().execution_reserved, 1);
        assert_eq!(handle.stats().busy_workers, 1);
        assert_eq!(Arc::strong_count(&input), 2);
        let mut next = Box::pin(handle.submit(handle.prepare_deadline(), |_| 9));
        assert!(poll!(&mut next).is_pending());
        release.send(()).unwrap();
        assert_eq!(next.await.unwrap().wait().await.unwrap(), 9);
        eventually(|| Arc::strong_count(&input) == 1).await;
    }
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn queue_timer_cannot_timeout_an_already_running_job() {
    let mut settings = config();
    settings.queue_timeout = Duration::from_millis(20);
    let mut runtime = start(settings);
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, 1, handle.prepare_deadline()).await;
    started.await.unwrap();
    let release_later = async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        release.send(()).unwrap();
    };
    let (result, ()) = tokio::join!(active.wait(), release_later);
    assert_eq!(result.unwrap(), 1);
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn dispatched_timeout_holds_execution_slot_until_worker_cleanup() {
    let mut settings = config();
    settings.queue_timeout = Duration::from_millis(20);
    let (mut runtime, receiver) = deferred_worker(settings);
    let handle = runtime.handle();
    let input = Arc::new(());
    let job_input = input.clone();
    let ticket = handle
        .submit(handle.prepare_deadline(), move |_| {
            drop(job_input);
            panic!("canceled dispatched job must not execute");
        })
        .await
        .unwrap();
    assert!(matches!(
        ticket.wait().await,
        Err(PrepareError::QueueTimeout)
    ));
    assert_eq!(handle.stats().execution_reserved, 1);
    assert_eq!(handle.stats().dispatched_jobs, 1);
    assert_eq!(Arc::strong_count(&input), 2);
    receiver.recv().unwrap().execute();
    assert_eq!(handle.stats().execution_reserved, 0);
    assert_eq!(Arc::strong_count(&input), 1);
    // The worker checks the queue deadline even if the ticket was never polled.
    let ticket = handle
        .submit(handle.prepare_deadline(), |_| panic!("expired job ran"))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;
    receiver.recv().unwrap().execute();
    assert!(matches!(
        ticket.wait().await,
        Err(PrepareError::QueueTimeout)
    ));
    assert_eq!(handle.stats().execution_reserved, 0);
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn job_panic_is_reported_and_worker_serves_next_job() {
    let mut runtime = start(config());
    let handle = runtime.handle();
    let panic = handle
        .submit(handle.prepare_deadline(), |_| panic!("expected test panic"))
        .await
        .unwrap();
    assert!(matches!(panic.wait().await, Err(PrepareError::Panicked)));
    assert_eq!(
        handle
            .submit(handle.prepare_deadline(), |_| 42)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap(),
        42
    );
    assert!(!handle.stats().failed);
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn slow_output_cleanup_retains_execution_capacity_and_input() {
    struct SlowOutput {
        started: Option<oneshot::Sender<()>>,
        gate: mpsc::Receiver<()>,
        _input: Arc<()>,
    }
    impl Drop for SlowOutput {
        fn drop(&mut self) {
            let _ = self.started.take().unwrap().send(());
            self.gate.recv_timeout(Duration::from_secs(5)).unwrap();
        }
    }
    let mut runtime = start(config());
    let handle = runtime.handle();
    let input = Arc::new(());
    let (work_started_tx, work_started) = oneshot::channel();
    let (work_release, work_gate) = mpsc::channel();
    let (cleanup_started_tx, cleanup_started) = oneshot::channel();
    let (cleanup_release, cleanup_gate) = mpsc::channel();
    let output = SlowOutput {
        started: Some(cleanup_started_tx),
        gate: cleanup_gate,
        _input: input.clone(),
    };
    let ticket = handle
        .submit(handle.prepare_deadline(), move |_| {
            let _ = work_started_tx.send(());
            work_gate.recv_timeout(Duration::from_secs(5)).unwrap();
            output
        })
        .await
        .unwrap();
    work_started.await.unwrap();
    drop(ticket);
    work_release.send(()).unwrap();
    cleanup_started.await.unwrap();
    let mut next = Box::pin(handle.submit(handle.prepare_deadline(), |_| 9));
    assert!(poll!(&mut next).is_pending());
    assert_eq!(handle.stats().busy_workers, 1);
    assert_eq!(handle.stats().execution_reserved, 1);
    assert_eq!(Arc::strong_count(&input), 2);
    cleanup_release.send(()).unwrap();
    assert_eq!(next.await.unwrap().wait().await.unwrap(), 9);
    assert_eq!(Arc::strong_count(&input), 1);
    runtime.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn close_wakes_waiters_and_rejects_new_submissions() {
    let mut runtime = start(config());
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, (), handle.prepare_deadline()).await;
    started.await.unwrap();
    let input = Arc::new(());
    let job_input = input.clone();
    let mut waiting = Box::pin(handle.submit(handle.prepare_deadline(), move |_| job_input));
    assert!(poll!(&mut waiting).is_pending());
    runtime.close();
    assert!(matches!(waiting.await, Err(PrepareError::Closed)));
    assert_eq!(handle.stats().queued_jobs, 0);
    assert_eq!(Arc::strong_count(&input), 1);
    assert!(matches!(
        handle.submit(handle.prepare_deadline(), |_| ()).await,
        Err(PrepareError::Closed)
    ));
    release.send(()).unwrap();
    active.wait().await.unwrap();
    assert_eq!(runtime.shutdown().await, 0);
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_is_bounded_with_worker_still_blocked_and_wakes_waiters() {
    let mut runtime = start(config());
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, (), handle.prepare_deadline()).await;
    started.await.unwrap();
    let mut waiting = Box::pin(handle.submit(handle.prepare_deadline(), |_| ()));
    assert!(poll!(&mut waiting).is_pending());
    let remaining_workers = tokio::time::timeout(Duration::from_secs(1), runtime.shutdown())
        .await
        .unwrap();
    assert_eq!(remaining_workers, 1);
    assert!(matches!(waiting.await, Err(PrepareError::Closed)));
    assert_eq!(handle.stats().execution_reserved, 1);
    release.send(()).unwrap();
    assert!(matches!(active.wait().await, Err(PrepareError::Cancelled)));
    eventually(|| handle.stats().live_workers == 0).await;
}

#[tokio::test(flavor = "current_thread")]
async fn dropped_runtime_does_not_join_and_weak_handles_do_not_keep_it_open() {
    let runtime = start(config());
    let handle = runtime.handle();
    let (active, started, release) = blocked(&handle, (), handle.prepare_deadline()).await;
    started.await.unwrap();
    let start = Instant::now();
    drop(runtime);
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(matches!(
        handle.submit(handle.prepare_deadline(), |_| ()).await,
        Err(PrepareError::Closed)
    ));
    release.send(()).unwrap();
    assert!(matches!(active.wait().await, Err(PrepareError::Cancelled)));
}

#[tokio::test(flavor = "current_thread")]
async fn unexpected_worker_exit_wakes_pending_waiters_with_worker_failed() {
    struct PanickingOutput;
    impl Drop for PanickingOutput {
        fn drop(&mut self) {
            panic!("expected output destructor panic");
        }
    }
    let mut runtime = start(config());
    let handle = runtime.handle();
    let (started_tx, started) = oneshot::channel();
    let (release, gate) = mpsc::channel();
    let ticket = handle
        .submit(handle.prepare_deadline(), move |_| {
            started_tx.send(()).unwrap();
            gate.recv_timeout(Duration::from_secs(5)).unwrap();
            PanickingOutput
        })
        .await
        .unwrap();
    started.await.unwrap();
    let called = Arc::new(AtomicBool::new(false));
    let job_called = called.clone();
    let mut waiting = Box::pin(handle.submit(handle.prepare_deadline(), move |_| {
        job_called.store(true, Ordering::Release);
    }));
    assert!(poll!(&mut waiting).is_pending());
    drop(ticket);
    release.send(()).unwrap();
    assert!(matches!(waiting.await, Err(PrepareError::WorkerFailed)));
    assert!(!called.load(Ordering::Acquire));
    assert!(handle.stats().failed);
    assert!(matches!(
        handle.submit(handle.prepare_deadline(), |_| ()).await,
        Err(PrepareError::WorkerFailed)
    ));
    assert_eq!(runtime.shutdown().await, 0);
}

#[test]
fn each_job_outcome_is_counted_once() {
    let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
    let metrics = recorder.handle();
    let _recorder = metrics::set_default_local_recorder(&recorder);
    let tokio = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    tokio.block_on(async {
        let (mut runtime, receiver) = deferred_worker(config());
        let handle = runtime.handle();
        // A ticket timeout, then the worker observing the resulting cancellation.
        let timed_out = handle
            .submit(Instant::now() + Duration::from_millis(20), |_| ())
            .await
            .unwrap();
        assert!(matches!(timed_out.wait().await, Err(PrepareError::Timeout)));
        receiver.recv().unwrap().execute();
        // A worker error the ticket receives.
        let panicked = handle
            .submit(handle.prepare_deadline(), |_| panic!("expected test panic"))
            .await
            .unwrap();
        receiver.recv().unwrap().execute();
        assert!(matches!(panicked.wait().await, Err(PrepareError::Panicked)));
        // A dropped ticket, then the worker rejecting its job.
        drop(
            handle
                .submit(handle.prepare_deadline(), |_| ())
                .await
                .unwrap(),
        );
        receiver.recv().unwrap().execute();
        runtime.shutdown().await;
    });
    let rendered = metrics.render();
    for reason in ["prepare_timeout", "prepare_panicked", "prepare_cancelled"] {
        assert!(
            rendered.contains(&format!(
                "mesh_prepare_errors_total{{reason=\"{reason}\"}} 1\n"
            )),
            "{rendered}"
        );
    }
}

#[test]
fn queue_cancel_and_worker_claim_have_exactly_one_winner() {
    let (runtime, _receiver) = deferred_worker(config());
    for _ in 0..32 {
        let control = JobControl::new(runtime.state.control.clone());
        let start = Arc::new(std::sync::Barrier::new(2));
        let worker_start = start.clone();
        let worker_control = control.clone();
        let claim = thread::spawn(move || {
            worker_start.wait();
            worker_control.claim()
        });
        start.wait();
        let canceled = control.cancel_queued();
        assert_ne!(canceled, claim.join().unwrap());
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_submitters_cannot_submit_after_close_returns() {
    let runtime = start(config());
    let handle = runtime.handle();
    let closed = Arc::new(tokio::sync::Barrier::new(5));
    let mut callers = tokio::task::JoinSet::new();
    for _ in 0..4 {
        let handle = handle.clone();
        let closed = closed.clone();
        callers.spawn(async move {
            drop(handle.submit(handle.prepare_deadline(), |_| ()).await);
            closed.wait().await;
            for _ in 0..32 {
                assert!(matches!(
                    handle.submit(handle.prepare_deadline(), |_| ()).await,
                    Err(PrepareError::Closed)
                ));
            }
        });
    }
    runtime.close();
    closed.wait().await;
    while let Some(result) = callers.join_next().await {
        result.unwrap();
    }
}

#[test]
fn invalid_config_is_rejected() {
    for settings in [
        PoolConfig {
            workers: 0,
            ..config()
        },
        PoolConfig {
            queue_capacity: 0,
            ..config()
        },
        PoolConfig {
            workers: Semaphore::MAX_PERMITS + 1,
            ..config()
        },
        PoolConfig {
            queue_capacity: Semaphore::MAX_PERMITS + 1,
            ..config()
        },
        PoolConfig {
            prepare_timeout: Duration::MAX,
            ..config()
        },
    ] {
        assert!(PreparePoolRuntime::new(settings).is_err());
    }
}
