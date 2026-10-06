//! Bounded CPU execution for request parsing, chat templates and tokenization.
//!
//! Cancellation retains the worker/queue slot until actual cleanup. Weak handles
//! let the server close the pool even when queued jobs retain an AppContext.

use std::{
    io,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
        Arc, Weak,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::Mutex;
use tokio::sync::oneshot;

const QUEUED: u8 = 0;
const RUNNING: u8 = 1;
const CANCELLED_QUEUED: u8 = 2;
const FINISHED: u8 = 3;

pub const DEFAULT_PREPARE_WORKERS: usize = 10;

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub workers: usize,
    pub queue_capacity: usize,
    pub max_retained_input_bytes: usize,
    pub queue_timeout: Duration,
    pub prepare_timeout: Duration,
    pub shutdown_grace: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            workers: DEFAULT_PREPARE_WORKERS,
            queue_capacity: 4,
            max_retained_input_bytes: 32 * 1024 * 1024,
            queue_timeout: Duration::from_millis(250),
            prepare_timeout: Duration::from_secs(5),
            shutdown_grace: Duration::from_secs(5),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PrepareError {
    #[error("request preparation queue is full")]
    Full,
    #[error("request preparation pool is closed")]
    Closed,
    #[error("request preparation input budget is exhausted")]
    BudgetExceeded,
    #[error("request preparation queue timeout")]
    QueueTimeout,
    #[error("request preparation timeout")]
    Timeout,
    #[error("request preparation cancelled")]
    Cancelled,
    #[error("request preparation panicked")]
    Panicked,
    #[error("request preparation worker failed")]
    WorkerFailed,
}

impl PrepareError {
    pub fn status_code(self) -> u16 {
        match self {
            Self::QueueTimeout | Self::Timeout => 504,
            Self::Panicked | Self::WorkerFailed => 500,
            _ => 503,
        }
    }

    pub fn code(self) -> &'static str {
        match self {
            Self::Full => "prepare_queue_full",
            Self::Closed => "prepare_pool_closed",
            Self::BudgetExceeded => "prepare_input_budget",
            Self::QueueTimeout => "prepare_queue_timeout",
            Self::Timeout => "prepare_timeout",
            Self::Cancelled => "prepare_cancelled",
            Self::Panicked => "prepare_panicked",
            Self::WorkerFailed => "prepare_worker_failed",
        }
    }

    fn count(self) -> Self {
        metrics::counter!("mesh_prepare_errors_total", "reason" => self.code()).increment(1);
        self
    }
}

#[derive(Debug)]
struct PoolControl {
    force_cancel: AtomicBool,
    failed: AtomicBool,
    live_workers: AtomicUsize,
    busy_workers: AtomicUsize,
    cancelled_queued: AtomicUsize,
    retained_input_bytes: AtomicUsize,
    max_retained_input_bytes: usize,
}

impl PoolControl {
    fn release_cancelled_queued(&self) {
        self.cancelled_queued.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_cancelled_queued_jobs").decrement(1.0);
    }

    fn run_worker(self: &Arc<Self>, receiver: Arc<Receiver<Job>>) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            while let Ok(job) = receiver.recv() {
                let _busy = BusyGuard::new(self.clone());
                if self.force_cancel.load(Ordering::Acquire) {
                    job.reject(PrepareError::Cancelled);
                } else {
                    job.execute();
                }
            }
            // Receiver cleanup is part of the worker lifetime and panic guard.
            drop(receiver);
        }));
        if outcome.is_err() {
            self.failed.store(true, Ordering::Release);
            self.force_cancel.store(true, Ordering::Release);
            tracing::error!("request preparation worker exited unexpectedly");
            PrepareError::WorkerFailed.count();
        }
        self.live_workers.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_live_workers").decrement(1.0);
    }

    fn reap(
        &self,
        mut threads: Vec<JoinHandle<()>>,
        receiver: Weak<Receiver<Job>>,
        deadline: Instant,
    ) {
        loop {
            let mut index = 0;
            while index < threads.len() {
                if threads[index].is_finished() {
                    let _ = threads.swap_remove(index).join();
                } else {
                    index += 1;
                }
            }
            if threads.is_empty() {
                return;
            }
            if Instant::now() >= deadline || self.force_cancel.load(Ordering::Acquire) {
                self.force_cancel.store(true, Ordering::Release);
                if let Some(receiver) = receiver.upgrade() {
                    while let Ok(job) = receiver.try_recv() {
                        // A panicking destructor must not prevent later cleanup.
                        let _ =
                            catch_unwind(AssertUnwindSafe(|| job.reject(PrepareError::Cancelled)));
                    }
                }
                // Detach remaining synchronous work; live_workers still counts it.
                return;
            }
            thread::sleep(
                Duration::from_millis(5).min(deadline.saturating_duration_since(Instant::now())),
            );
        }
    }
}

struct SubmitState {
    // The only Sender lives here. Never drop jobs or the sender under this lock.
    sender: Mutex<Option<Sender<Job>>>,
    receiver: Weak<Receiver<Job>>,
    control: Arc<PoolControl>,
    config: PoolConfig,
}

#[derive(Clone, Debug)]
pub struct PrepareHandle {
    state: Weak<SubmitState>,
}

/// One shared input charge, retained through queued, running and returned data.
#[derive(Clone, Debug)]
pub struct InputLease {
    _charge: Arc<InputCharge>,
}

impl InputLease {
    pub fn bytes(&self) -> usize {
        self._charge.bytes
    }
}

#[derive(Debug)]
struct InputCharge {
    bytes: usize,
    control: Arc<PoolControl>,
}

impl Drop for InputCharge {
    fn drop(&mut self) {
        self.control
            .retained_input_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_retained_input_bytes").decrement(self.bytes as f64);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub live_workers: usize,
    pub busy_workers: usize,
    pub queued_jobs: usize,
    pub cancelled_queued_jobs: usize,
    pub retained_input_bytes: usize,
    pub failed: bool,
}

impl PrepareHandle {
    /// Explicit placeholder for components that never submit preparation work.
    pub fn closed() -> Self {
        Self { state: Weak::new() }
    }

    pub fn prepare_deadline(&self) -> Instant {
        let timeout = self
            .state
            .upgrade()
            .map(|state| state.config.prepare_timeout)
            .unwrap_or_default();
        Instant::now() + timeout
    }

    pub fn stats(&self) -> PoolStats {
        let Some(state) = self.state.upgrade() else {
            return PoolStats::default();
        };
        PoolStats {
            live_workers: state.control.live_workers.load(Ordering::Acquire),
            busy_workers: state.control.busy_workers.load(Ordering::Acquire),
            queued_jobs: state
                .receiver
                .upgrade()
                .map_or(0, |receiver| receiver.len()),
            cancelled_queued_jobs: state.control.cancelled_queued.load(Ordering::Acquire),
            retained_input_bytes: state.control.retained_input_bytes.load(Ordering::Acquire),
            failed: state.control.failed.load(Ordering::Acquire),
        }
    }

    pub fn retain_input(&self, bytes: usize) -> Result<InputLease, PrepareError> {
        let state = self
            .state
            .upgrade()
            .ok_or_else(|| PrepareError::Closed.count())?;
        let sender = state.sender.lock();
        state.check_open(&sender)?;
        state
            .control
            .retained_input_bytes
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current
                    .checked_add(bytes)
                    .filter(|&next| next <= state.control.max_retained_input_bytes)
            })
            .map_err(|_| PrepareError::BudgetExceeded.count())?;
        drop(sender);
        metrics::gauge!("mesh_prepare_retained_input_bytes").increment(bytes as f64);
        Ok(InputLease {
            _charge: Arc::new(InputCharge {
                bytes,
                control: state.control.clone(),
            }),
        })
    }

    /// Submit without waiting for capacity. All stages share one absolute deadline.
    pub fn try_submit<T, F>(
        &self,
        deadline: Instant,
        work: F,
    ) -> Result<PrepareTicket<T>, PrepareError>
    where
        T: Send + 'static,
        F: FnOnce(&JobContext) -> T + Send + 'static,
    {
        let state = self
            .state
            .upgrade()
            .ok_or_else(|| PrepareError::Closed.count())?;
        let enqueued = Instant::now();
        if enqueued >= deadline {
            return Err(PrepareError::Timeout.count());
        }
        let queue_deadline = enqueued + state.config.queue_timeout;
        let control = Arc::new(JobControl {
            state: AtomicU8::new(QUEUED),
            cancelled: AtomicBool::new(false),
            pool: state.control.clone(),
        });
        let (sender, receiver) = oneshot::channel();
        let job = Job {
            control: control.clone(),
            queued: Some(QueuedCharge::new(control.clone())),
            enqueued,
            queue_deadline,
            deadline,
            work: Box::new(move |context| match context {
                Ok(context) => {
                    let result = catch_unwind(AssertUnwindSafe(|| work(context)))
                        .map_err(|_| PrepareError::Panicked.count())
                        .and_then(|value| {
                            context.check().map(|()| value).map_err(PrepareError::count)
                        });
                    context.control.state.store(FINISHED, Ordering::Release);
                    // Failed sends dispose of returned input on this worker.
                    let _ = sender.send(result);
                }
                Err(error) => {
                    let _ = sender.send(Err(error.count()));
                    drop(work);
                }
            }),
        };
        let result = {
            let sender = state.sender.lock();
            // Keep ownership of `job` outside the lock, including errors.
            match state.check_open(&sender) {
                Ok(()) => sender
                    .as_ref()
                    .unwrap()
                    .try_send(job)
                    .map_err(|error| match error {
                        TrySendError::Full(job) => (PrepareError::Full.count(), job),
                        TrySendError::Disconnected(job) => {
                            (PrepareError::WorkerFailed.count(), job)
                        }
                    }),
                Err(error) => Err((error, job)),
            }
        };
        if let Err((error, job)) = result {
            drop(job);
            return Err(error);
        }
        Ok(PrepareTicket {
            receiver,
            control,
            deadline,
            queue_deadline,
            completed: false,
        })
    }
}

impl SubmitState {
    fn check_open(&self, sender: &Option<Sender<Job>>) -> Result<(), PrepareError> {
        if self.control.failed.load(Ordering::Acquire) {
            Err(PrepareError::WorkerFailed.count())
        } else if sender.is_none() {
            Err(PrepareError::Closed.count())
        } else {
            Ok(())
        }
    }

    fn close(&self) {
        let sender = self.sender.lock().take();
        // Releasing the final Sender wakes idle recv() calls, even for a full queue.
        drop(sender);
    }
}

struct JobControl {
    state: AtomicU8,
    cancelled: AtomicBool,
    pool: Arc<PoolControl>,
}

impl JobControl {
    fn cancel_queued(&self) -> bool {
        // Account before CAS so a worker cannot decrement before this increment.
        self.pool.cancelled_queued.fetch_add(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_cancelled_queued_jobs").increment(1.0);
        let won = self
            .state
            .compare_exchange(
                QUEUED,
                CANCELLED_QUEUED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok();
        if !won {
            self.pool.release_cancelled_queued();
        }
        won
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.cancel_queued();
    }
}

pub struct JobContext {
    control: Arc<JobControl>,
    deadline: Instant,
}

impl JobContext {
    pub fn check(&self) -> Result<(), PrepareError> {
        if self.control.pool.failed.load(Ordering::Acquire) {
            return Err(PrepareError::WorkerFailed);
        }
        if self.control.pool.force_cancel.load(Ordering::Acquire) {
            self.control.cancelled.store(true, Ordering::Release);
        }
        if self.control.cancelled.load(Ordering::Acquire) {
            Err(PrepareError::Cancelled)
        } else if Instant::now() >= self.deadline {
            Err(PrepareError::Timeout)
        } else {
            Ok(())
        }
    }
}

pub struct PrepareTicket<T> {
    receiver: oneshot::Receiver<Result<T, PrepareError>>,
    control: Arc<JobControl>,
    deadline: Instant,
    queue_deadline: Instant,
    completed: bool,
}

impl<T> PrepareTicket<T> {
    pub async fn wait(mut self) -> Result<T, PrepareError> {
        let overall = tokio::time::sleep_until(self.deadline.into());
        let queue = tokio::time::sleep_until(self.queue_deadline.into());
        tokio::pin!(overall, queue);
        let mut queue_pending = self.queue_deadline < self.deadline;
        let result = loop {
            tokio::select! {
                biased;
                result = &mut self.receiver => {
                    break result.unwrap_or(Err(PrepareError::WorkerFailed));
                }
                _ = &mut queue, if queue_pending => {
                    if self.control.cancel_queued() {
                        self.control.cancelled.store(true, Ordering::Release);
                        break Err(PrepareError::QueueTimeout.count());
                    }
                    // Once running, only the overall preparation deadline applies.
                    queue_pending = false;
                }
                _ = &mut overall => {
                    self.control.cancel();
                    break Err(PrepareError::Timeout.count());
                }
            }
        };
        self.completed = true;
        result
    }
}

impl<T> Drop for PrepareTicket<T> {
    fn drop(&mut self) {
        if !self.completed {
            self.control.cancel();
        }
    }
}

struct Job {
    control: Arc<JobControl>,
    queued: Option<QueuedCharge>,
    enqueued: Instant,
    queue_deadline: Instant,
    deadline: Instant,
    work: Box<dyn FnOnce(Result<&JobContext, PrepareError>) + Send>,
}

impl Job {
    fn reject(mut self, error: PrepareError) {
        let previous = self.control.state.swap(FINISHED, Ordering::AcqRel);
        if previous == CANCELLED_QUEUED {
            self.control.pool.release_cancelled_queued();
        }
        drop(self.queued.take());
        (self.work)(Err(error));
    }

    fn execute(mut self) {
        if self
            .control
            .state
            .compare_exchange(QUEUED, RUNNING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            self.reject(PrepareError::Cancelled);
            return;
        }
        drop(self.queued.take());
        let now = Instant::now();
        metrics::histogram!("mesh_prepare_queue_seconds")
            .record(now.duration_since(self.enqueued).as_secs_f64());
        let context = JobContext {
            control: self.control.clone(),
            deadline: self.deadline,
        };
        let error = if now >= self.queue_deadline && self.queue_deadline < self.deadline {
            Some(PrepareError::QueueTimeout)
        } else {
            context.check().err()
        };
        if let Some(error) = error {
            self.reject(error);
            return;
        }
        (self.work)(Ok(&context));
        metrics::histogram!("mesh_prepare_run_seconds").record(now.elapsed().as_secs_f64());
    }
}

struct QueuedCharge(Arc<JobControl>);

impl QueuedCharge {
    fn new(control: Arc<JobControl>) -> Self {
        metrics::gauge!("mesh_prepare_queued_jobs").increment(1.0);
        Self(control)
    }
}

impl Drop for QueuedCharge {
    fn drop(&mut self) {
        metrics::gauge!("mesh_prepare_queued_jobs").decrement(1.0);
        // Also release jobs dropped by failed submission or channel destruction.
        let mut state = self.0.state.load(Ordering::Acquire);
        while state == QUEUED || state == CANCELLED_QUEUED {
            match self.0.state.compare_exchange_weak(
                state,
                FINISHED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(previous) => {
                    if previous == CANCELLED_QUEUED {
                        self.0.pool.release_cancelled_queued();
                    }
                    break;
                }
                Err(current) => state = current,
            }
        }
    }
}

struct BusyGuard(Arc<PoolControl>);

impl BusyGuard {
    fn new(control: Arc<PoolControl>) -> Self {
        control.busy_workers.fetch_add(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_busy_workers").increment(1.0);
        Self(control)
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        self.0.busy_workers.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_busy_workers").decrement(1.0);
    }
}

/// The server owns this value; AppContext holds only `PrepareHandle`.
pub struct PreparePoolRuntime {
    state: Arc<SubmitState>,
    threads: Vec<JoinHandle<()>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShutdownReport {
    pub remaining_workers: usize,
    pub forced: bool,
}

impl PreparePoolRuntime {
    pub fn new(config: PoolConfig) -> io::Result<Self> {
        if config.workers == 0
            || config.queue_capacity == 0
            || config
                .queue_capacity
                .checked_add(1)
                .and_then(usize::checked_next_power_of_two)
                .and_then(|capacity| capacity.checked_mul(2))
                .is_none()
            || config.max_retained_input_bytes == 0
            || config.queue_timeout.is_zero()
            || config.prepare_timeout.is_zero()
            || config.shutdown_grace.is_zero()
            || Instant::now().checked_add(config.queue_timeout).is_none()
            || Instant::now().checked_add(config.prepare_timeout).is_none()
            || Instant::now().checked_add(config.shutdown_grace).is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "prepare pool limits must be positive and representable",
            ));
        }
        let (sender, receiver) = crossbeam_channel::bounded(config.queue_capacity);
        let receiver = Arc::new(receiver);
        let control = Arc::new(PoolControl {
            force_cancel: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            live_workers: AtomicUsize::new(0),
            busy_workers: AtomicUsize::new(0),
            cancelled_queued: AtomicUsize::new(0),
            retained_input_bytes: AtomicUsize::new(0),
            max_retained_input_bytes: config.max_retained_input_bytes,
        });
        let state = Arc::new(SubmitState {
            sender: Mutex::new(Some(sender)),
            receiver: Arc::downgrade(&receiver),
            control: control.clone(),
            config,
        });
        let mut runtime = Self {
            state,
            threads: Vec::new(),
        };
        for index in 0..runtime.state.config.workers {
            let receiver = receiver.clone();
            let thread_control = control.clone();
            control.live_workers.fetch_add(1, Ordering::AcqRel);
            metrics::gauge!("mesh_prepare_live_workers").increment(1.0);
            match thread::Builder::new()
                .name(format!("mesh-prepare-{index}"))
                .spawn(move || thread_control.run_worker(receiver))
            {
                Ok(handle) => runtime.threads.push(handle),
                Err(error) => {
                    control.live_workers.fetch_sub(1, Ordering::AcqRel);
                    metrics::gauge!("mesh_prepare_live_workers").decrement(1.0);
                    // Runtime Drop closes/cancels partial startup without joining.
                    return Err(error);
                }
            }
        }
        // Only real workers retain the Receiver. A dead pool cannot accept jobs.
        drop(receiver);
        Ok(runtime)
    }

    pub fn handle(&self) -> PrepareHandle {
        PrepareHandle {
            state: Arc::downgrade(&self.state),
        }
    }

    pub fn close(&self) {
        self.state.close();
    }

    /// Close after transport drain. Cleanup runs on an OS thread; waiting remains
    /// bounded even when synchronous work or an input destructor blocks.
    pub async fn shutdown(&mut self) -> ShutdownReport {
        self.close();
        let deadline = Instant::now() + self.state.config.shutdown_grace;
        let threads = std::mem::take(&mut self.threads);
        let control = self.state.control.clone();
        let receiver = self.state.receiver.clone();
        let (sender, result) = oneshot::channel();
        let reaper = thread::Builder::new()
            .name("mesh-prepare-reaper".into())
            .spawn(move || {
                control.reap(threads, receiver, deadline);
                let _ = sender.send(ShutdownReport {
                    remaining_workers: control.live_workers.load(Ordering::Acquire),
                    forced: control.force_cancel.load(Ordering::Acquire),
                });
            });
        if let Err(error) = reaper {
            tracing::error!(%error, "could not start preparation shutdown reaper");
        } else {
            tokio::select! {
                result = result => {
                    if let Ok(report) = result { return report; }
                }
                _ = tokio::time::sleep_until(deadline.into()) => {}
            }
        }
        self.state
            .control
            .force_cancel
            .store(true, Ordering::Release);
        ShutdownReport {
            remaining_workers: self.state.control.live_workers.load(Ordering::Acquire),
            forced: true,
        }
    }
}

impl Drop for PreparePoolRuntime {
    fn drop(&mut self) {
        self.close();
        self.state
            .control
            .force_cancel
            .store(true, Ordering::Release);
        if !self.threads.is_empty() {
            let threads = std::mem::take(&mut self.threads);
            let receiver = self.state.receiver.clone();
            let control = self.state.control.clone();
            // Drop cannot wait for CPU work, including at startup failure.
            let _ = thread::Builder::new()
                .name("mesh-prepare-cleanup".into())
                .spawn(move || {
                    control.reap(threads, receiver, Instant::now());
                });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn config() -> PoolConfig {
        PoolConfig {
            workers: 1,
            queue_capacity: 2,
            max_retained_input_bytes: 32,
            queue_timeout: Duration::from_secs(2),
            prepare_timeout: Duration::from_secs(5),
            shutdown_grace: Duration::from_millis(50),
        }
    }

    fn blocked(
        handle: &PrepareHandle,
        lease: InputLease,
        deadline: Instant,
    ) -> (
        PrepareTicket<InputLease>,
        oneshot::Receiver<()>,
        mpsc::Sender<()>,
    ) {
        let (started_tx, started) = oneshot::channel();
        let (release, gate) = mpsc::channel();
        let ticket = handle
            .try_submit(deadline, move |_| {
                let _ = started_tx.send(());
                // An OS timeout prevents hangs even if Tokio timers stall.
                gate.recv_timeout(Duration::from_secs(5)).unwrap();
                lease
            })
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

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_and_queued_cancellation_keep_physical_nodes_and_input() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let called = Arc::new(AtomicBool::new(false));
        let called_in_job = called.clone();
        let queued_lease = handle.retain_input(7).unwrap();
        let cancelled = handle
            .try_submit(handle.prepare_deadline(), move |_| {
                called_in_job.store(true, Ordering::Release);
                queued_lease
            })
            .unwrap();
        let queued = handle
            .try_submit(handle.prepare_deadline(), |_| 42)
            .unwrap();
        drop(cancelled);
        assert_eq!(handle.stats().busy_workers, 1);
        assert_eq!(handle.stats().queued_jobs, 2);
        assert_eq!(handle.stats().cancelled_queued_jobs, 1);
        assert_eq!(handle.stats().retained_input_bytes, 12);
        assert!(matches!(
            handle.try_submit(handle.prepare_deadline(), |_| ()),
            Err(PrepareError::Full)
        ));
        assert!(matches!(
            handle.retain_input(21),
            Err(PrepareError::BudgetExceeded)
        ));
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        assert_eq!(queued.wait().await.unwrap(), 42);
        eventually(|| handle.stats().retained_input_bytes == 0).await;
        assert!(!called.load(Ordering::Acquire));
        assert_eq!(handle.stats().cancelled_queued_jobs, 0);
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn running_cancellation_holds_real_worker_and_input_until_return() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let lease = handle.retain_input(32).unwrap();
        let (active, started, release) = blocked(&handle, lease, handle.prepare_deadline());
        started.await.unwrap();
        drop(active);
        assert_eq!(handle.stats().busy_workers, 1);
        assert_eq!(handle.stats().retained_input_bytes, 32);
        assert!(matches!(
            handle.retain_input(1),
            Err(PrepareError::BudgetExceeded)
        ));
        let next = handle.try_submit(handle.prepare_deadline(), |_| 9).unwrap();
        assert_eq!(handle.stats().queued_jobs, 1);
        release.send(()).unwrap();
        assert_eq!(next.wait().await.unwrap(), 9);
        eventually(|| handle.stats().retained_input_bytes == 0).await;
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn running_timeout_does_not_replace_worker_or_release_input() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let deadline = Instant::now() + Duration::from_millis(30);
        let (active, started, release) =
            blocked(&handle, handle.retain_input(32).unwrap(), deadline);
        started.await.unwrap();
        assert!(matches!(active.wait().await, Err(PrepareError::Timeout)));
        assert_eq!(handle.stats().live_workers, 1);
        assert_eq!(handle.stats().busy_workers, 1);
        assert_eq!(handle.stats().retained_input_bytes, 32);
        release.send(()).unwrap();
        eventually(|| handle.stats().retained_input_bytes == 0).await;
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queue_timeout_never_runs_expired_job_and_keeps_it_queued() {
        let mut settings = config();
        settings.queue_timeout = Duration::from_millis(20);
        let mut runtime = PreparePoolRuntime::new(settings).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let lease = handle.retain_input(7).unwrap();
        let queued = handle
            .try_submit(handle.prepare_deadline(), move |_| {
                drop(lease);
                panic!("expired queued job must not execute");
            })
            .unwrap();
        assert!(matches!(
            queued.wait().await,
            Err(PrepareError::QueueTimeout)
        ));
        assert_eq!(handle.stats().queued_jobs, 1);
        assert_eq!(handle.stats().retained_input_bytes, 12);
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        eventually(|| handle.stats().retained_input_bytes == 0).await;
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queue_timer_cannot_timeout_an_already_running_job() {
        let mut settings = config();
        settings.queue_timeout = Duration::from_millis(20);
        let mut runtime = PreparePoolRuntime::new(settings).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(1).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let release_later = async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            release.send(()).unwrap();
        };
        let (result, ()) = tokio::join!(active.wait(), release_later);
        assert_eq!(result.unwrap().bytes(), 1);
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn worker_checks_queue_deadline_even_when_ticket_has_never_been_polled() {
        let mut settings = config();
        settings.queue_timeout = Duration::from_millis(20);
        let mut runtime = PreparePoolRuntime::new(settings).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(1).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let queued = handle
            .try_submit(handle.prepare_deadline(), |_| panic!("expired job ran"))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        assert!(matches!(
            queued.wait().await,
            Err(PrepareError::QueueTimeout)
        ));
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn job_panic_is_reported_and_worker_serves_next_job() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let panic = handle
            .try_submit(handle.prepare_deadline(), |_| panic!("expected test panic"))
            .unwrap();
        assert!(matches!(panic.wait().await, Err(PrepareError::Panicked)));
        assert_eq!(
            handle
                .try_submit(handle.prepare_deadline(), |_| 42)
                .unwrap()
                .wait()
                .await
                .unwrap(),
            42
        );
        assert!(!handle.stats().failed);
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn result_and_lease_clones_keep_one_charge_after_worker_finishes() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let lease = handle.retain_input(32).unwrap();
        let shared = lease.clone();
        let ticket = handle
            .try_submit(handle.prepare_deadline(), move |_| lease)
            .unwrap();
        eventually(|| ticket.control.state.load(Ordering::Acquire) == FINISHED).await;
        assert_eq!(handle.stats().retained_input_bytes, 32);
        assert!(matches!(
            handle.retain_input(1),
            Err(PrepareError::BudgetExceeded)
        ));
        // The completed oneshot owns its result until it is consumed or dropped.
        drop(ticket);
        assert_eq!(handle.stats().retained_input_bytes, 32);
        drop(shared);
        eventually(|| handle.stats().retained_input_bytes == 0).await;
        let lease = handle.retain_input(32).unwrap();
        let returned = handle
            .try_submit(handle.prepare_deadline(), move |_| lease)
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(handle.stats().retained_input_bytes, 32);
        drop(returned);
        assert_eq!(handle.stats().retained_input_bytes, 0);
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sequential_jobs_share_deadline_and_do_not_nest_on_single_worker() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let deadline = Instant::now() + Duration::from_millis(50);
        let lease = handle.retain_input(32).unwrap();
        let lease = handle
            .try_submit(deadline, move |_| lease)
            .unwrap()
            .wait()
            .await
            .unwrap();
        let lease = handle
            .try_submit(deadline, move |_| lease)
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(handle.stats().retained_input_bytes, 32);
        tokio::time::sleep_until(deadline.into()).await;
        assert!(matches!(
            handle.try_submit(deadline, move |_| lease),
            Err(PrepareError::Timeout)
        ));
        assert_eq!(handle.stats().retained_input_bytes, 0);
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_rejects_submissions_and_full_queue_needs_no_shutdown_sentinel() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(1).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let first = handle.try_submit(handle.prepare_deadline(), |_| 1).unwrap();
        let second = handle.try_submit(handle.prepare_deadline(), |_| 2).unwrap();
        runtime.close();
        assert!(matches!(
            handle.try_submit(handle.prepare_deadline(), |_| 3),
            Err(PrepareError::Closed)
        ));
        assert!(matches!(handle.retain_input(1), Err(PrepareError::Closed)));
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        assert_eq!(first.wait().await.unwrap(), 1);
        assert_eq!(second.wait().await.unwrap(), 2);
        let report = runtime.shutdown().await;
        assert_eq!(
            report,
            ShutdownReport {
                remaining_workers: 0,
                forced: false
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_is_bounded_and_force_cleans_queue_with_worker_still_blocked() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let lease = handle.retain_input(7).unwrap();
        let queued = handle
            .try_submit(handle.prepare_deadline(), move |_| {
                drop(lease);
                panic!("forced queued work must not run");
            })
            .unwrap();
        let start = Instant::now();
        let report = runtime.shutdown().await;
        assert!(start.elapsed() < Duration::from_secs(1));
        assert_eq!(
            report,
            ShutdownReport {
                remaining_workers: 1,
                forced: true
            }
        );
        assert!(matches!(queued.wait().await, Err(PrepareError::Cancelled)));
        eventually(|| handle.stats().retained_input_bytes == 5).await;
        assert_eq!(handle.stats().busy_workers, 1);
        release.send(()).unwrap();
        assert!(matches!(active.wait().await, Err(PrepareError::Cancelled)));
        eventually(|| handle.stats().live_workers == 0).await;
        eventually(|| handle.stats().retained_input_bytes == 0).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_runtime_does_not_join_and_weak_handles_do_not_keep_it_open() {
        let runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        );
        started.await.unwrap();
        let start = Instant::now();
        drop(runtime);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(matches!(handle.retain_input(1), Err(PrepareError::Closed)));
        release.send(()).unwrap();
        assert!(matches!(active.wait().await, Err(PrepareError::Cancelled)));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unexpected_worker_exit_marks_pool_failed_instead_of_reducing_capacity() {
        struct PanickingOutput;
        impl Drop for PanickingOutput {
            fn drop(&mut self) {
                panic!("expected output destructor panic");
            }
        }
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (started_tx, started) = oneshot::channel();
        let (release, gate) = mpsc::channel();
        let ticket = handle
            .try_submit(handle.prepare_deadline(), move |_| {
                started_tx.send(()).unwrap();
                gate.recv_timeout(Duration::from_secs(5)).unwrap();
                PanickingOutput
            })
            .unwrap();
        started.await.unwrap();
        // Exercise the outer worker guard with a result destructor panic.
        drop(ticket);
        release.send(()).unwrap();
        eventually(|| handle.stats().failed).await;
        assert!(matches!(
            handle.try_submit(handle.prepare_deadline(), |_| ()),
            Err(PrepareError::WorkerFailed)
        ));
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[test]
    fn queue_cancel_and_worker_claim_have_exactly_one_winner() {
        let runtime = PreparePoolRuntime::new(config()).unwrap();
        for _ in 0..32 {
            let control = Arc::new(JobControl {
                state: AtomicU8::new(QUEUED),
                cancelled: AtomicBool::new(false),
                pool: runtime.state.control.clone(),
            });
            let start = Arc::new(std::sync::Barrier::new(2));
            let worker_start = start.clone();
            let worker_control = control.clone();
            let claim = thread::spawn(move || {
                worker_start.wait();
                worker_control
                    .state
                    .compare_exchange(QUEUED, RUNNING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            });
            start.wait();
            let cancelled = control.cancel_queued();
            let running = claim.join().unwrap();
            assert_ne!(cancelled, running);
            if cancelled {
                assert_eq!(control.state.load(Ordering::Acquire), CANCELLED_QUEUED);
                control.pool.release_cancelled_queued();
            } else {
                assert_eq!(control.state.load(Ordering::Acquire), RUNNING);
            }
        }
        assert_eq!(runtime.handle().stats().cancelled_queued_jobs, 0);
    }

    #[test]
    fn concurrent_submitters_cannot_submit_after_close_returns() {
        let runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let closed = Arc::new(std::sync::Barrier::new(5));
        let mut callers = Vec::new();
        for _ in 0..4 {
            let handle = handle.clone();
            let closed = closed.clone();
            callers.push(thread::spawn(move || {
                // Race the first submission, then verify close's linearization.
                drop(handle.try_submit(handle.prepare_deadline(), |_| ()));
                closed.wait();
                for _ in 0..32 {
                    assert!(matches!(
                        handle.try_submit(handle.prepare_deadline(), |_| ()),
                        Err(PrepareError::Closed)
                    ));
                }
            }));
        }
        runtime.close();
        closed.wait();
        for caller in callers {
            caller.join().unwrap();
        }
    }

    #[test]
    fn invalid_capacity_and_zero_budget_are_rejected() {
        let mut settings = config();
        settings.workers = 0;
        assert!(PreparePoolRuntime::new(settings).is_err());
        let mut settings = config();
        settings.queue_capacity = 0;
        assert!(PreparePoolRuntime::new(settings).is_err());
        let mut settings = config();
        settings.max_retained_input_bytes = 0;
        assert!(PreparePoolRuntime::new(settings).is_err());
        let mut settings = config();
        settings.queue_capacity = usize::MAX;
        assert!(PreparePoolRuntime::new(settings).is_err());
        let mut settings = config();
        settings.prepare_timeout = Duration::MAX;
        assert!(PreparePoolRuntime::new(settings).is_err());
    }
}
