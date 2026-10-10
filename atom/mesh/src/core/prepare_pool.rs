//! Bounded CPU execution for request parsing, chat templates and tokenization.
//!
//! Waiting for execution capacity is asynchronous and bounded. Dispatched work
//! retains its execution slot until actual cleanup. Weak handles let the server
//! close the pool even when jobs retain an AppContext.

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
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore, TryAcquireError};

const QUEUED: u8 = 0;
const RUNNING: u8 = 1;
const CANCELLED_QUEUED: u8 = 2;
const FINISHED: u8 = 3;

pub const DEFAULT_PREPARE_WORKERS: usize = 10;

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub workers: usize,
    pub queue_capacity: usize,
    /// Input bytes owned by preparation, including waiting work and result handoff.
    pub max_retained_input_bytes: usize,
    pub queue_timeout: Duration,
    pub prepare_timeout: Duration,
    pub shutdown_grace: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            workers: DEFAULT_PREPARE_WORKERS,
            queue_capacity: DEFAULT_PREPARE_WORKERS * 10,
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
    waiting_jobs: AtomicUsize,
    cancelled_queued: AtomicUsize,
    retained_input_bytes: AtomicUsize,
    max_retained_input_bytes: usize,
    execution_slots: Arc<Semaphore>,
    waiting_slots: Arc<Semaphore>,
}

impl PoolControl {
    fn release_cancelled_queued(&self) {
        self.cancelled_queued.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_cancelled_queued_jobs").decrement(1.0);
    }

    fn run_worker(self: &Arc<Self>, receiver: Arc<Receiver<Job>>) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            while let Ok(mut job) = receiver.recv() {
                let _busy = BusyGuard::new(self.clone());
                // Hold capacity outside job cleanup's panic guard. A destructor
                // panic closes admission before this permit can wake a waiter.
                let _execution = job._execution.take();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    if self.force_cancel.load(Ordering::Acquire) {
                        job.reject(PrepareError::Cancelled);
                    } else {
                        job.execute();
                    }
                }));
                if outcome.is_err() {
                    self.mark_failed();
                    break;
                }
            }
            // Receiver cleanup is part of the worker lifetime and panic guard.
            drop(receiver);
        }));
        if outcome.is_err() {
            self.mark_failed();
        }
        self.live_workers.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_live_workers").decrement(1.0);
    }

    fn mark_failed(&self) {
        self.failed.store(true, Ordering::Release);
        self.force_cancel.store(true, Ordering::Release);
        self.execution_slots.close();
        self.waiting_slots.close();
        tracing::error!("request preparation worker exited unexpectedly");
        PrepareError::WorkerFailed.count();
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

/// One shared preparation input charge, retained through waiting work, execution
/// and result handoff. Release it before forwarding; raw body clones must not
/// prolong its lifetime. Cancelled synchronous work keeps its charge until cleanup.
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
    /// Jobs that have reserved execution capacity, including pending dispatch.
    pub execution_reserved: usize,
    /// OS threads currently processing or cleaning up a job.
    pub busy_workers: usize,
    /// Bounded asynchronous waiters that have not reserved execution capacity.
    pub queued_jobs: usize,
    /// Jobs delivered to the OS-thread channel but not yet received by a worker.
    pub dispatched_jobs: usize,
    /// Canceled dispatched jobs that still await worker cleanup.
    pub cancelled_queued_jobs: usize,
    /// Input bytes still owned by preparation, excluding forwarding bodies.
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
            execution_reserved: state.config.workers
                - state.control.execution_slots.available_permits(),
            busy_workers: state.control.busy_workers.load(Ordering::Acquire),
            queued_jobs: state.control.waiting_jobs.load(Ordering::Acquire),
            dispatched_jobs: state
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

    /// Wait for execution capacity without blocking Tokio. Only `queue_capacity`
    /// callers may wait; additional submissions fail immediately. All stages
    /// share one absolute deadline supplied by the caller.
    pub async fn submit<T, F>(
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
        state.check_open(&state.sender.lock())?;
        let execution = state.acquire_execution(queue_deadline, deadline).await?;
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
            // Keep this field last so closure/input cleanup precedes release.
            _execution: Some(execution),
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
                        // Every sent job owns one execution permit, and the
                        // channel can hold all permits. Full is unreachable
                        // unless that invariant is broken.
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
    fn unavailable(&self) -> PrepareError {
        if self.control.failed.load(Ordering::Acquire) {
            PrepareError::WorkerFailed.count()
        } else {
            PrepareError::Closed.count()
        }
    }

    async fn acquire_execution(
        &self,
        queue_deadline: Instant,
        deadline: Instant,
    ) -> Result<ExecutionCharge, PrepareError> {
        match self.control.execution_slots.clone().try_acquire_owned() {
            Ok(permit) => return Ok(ExecutionCharge::new(permit)),
            Err(TryAcquireError::Closed) => return Err(self.unavailable()),
            Err(TryAcquireError::NoPermits) => {}
        }

        // Bound waiter allocation before awaiting the fair execution semaphore.
        // Dropping a pending submit future releases this slot and its input.
        let permit = self
            .control
            .waiting_slots
            .clone()
            .try_acquire_owned()
            .map_err(|error| match error {
                TryAcquireError::Closed => self.unavailable(),
                TryAcquireError::NoPermits => PrepareError::Full.count(),
            })?;
        let _waiting = WaitingCharge::new(permit, self.control.clone());
        let wait_deadline = queue_deadline.min(deadline);
        let result = tokio::select! {
            biased;
            result = self.control.execution_slots.clone().acquire_owned() => {
                result.map_err(|_| self.unavailable())
            }
            _ = tokio::time::sleep_until(wait_deadline.into()) => {
                Err(if deadline <= queue_deadline {
                    PrepareError::Timeout.count()
                } else {
                    PrepareError::QueueTimeout.count()
                })
            }
        }?;
        // A ready permit cannot revive work whose deadline has already elapsed.
        if Instant::now() >= deadline {
            return Err(PrepareError::Timeout.count());
        }
        if Instant::now() >= queue_deadline {
            return Err(PrepareError::QueueTimeout.count());
        }
        Ok(ExecutionCharge::new(result))
    }

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
        self.control.execution_slots.close();
        self.control.waiting_slots.close();
        // Releasing the final Sender wakes idle recv() calls, even for a full queue.
        drop(sender);
    }
}

struct WaitingCharge {
    _permit: OwnedSemaphorePermit,
    control: Arc<PoolControl>,
}

impl WaitingCharge {
    fn new(permit: OwnedSemaphorePermit, control: Arc<PoolControl>) -> Self {
        control.waiting_jobs.fetch_add(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_queued_jobs").increment(1.0);
        Self {
            _permit: permit,
            control,
        }
    }
}

impl Drop for WaitingCharge {
    fn drop(&mut self) {
        self.control.waiting_jobs.fetch_sub(1, Ordering::AcqRel);
        metrics::gauge!("mesh_prepare_queued_jobs").decrement(1.0);
    }
}

struct ExecutionCharge {
    _permit: OwnedSemaphorePermit,
}

impl ExecutionCharge {
    fn new(permit: OwnedSemaphorePermit) -> Self {
        metrics::gauge!("mesh_prepare_execution_reserved").increment(1.0);
        Self { _permit: permit }
    }
}

impl Drop for ExecutionCharge {
    fn drop(&mut self) {
        metrics::gauge!("mesh_prepare_execution_reserved").decrement(1.0);
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
    // Last: even a panicking input destructor runs before capacity is released.
    _execution: Option<ExecutionCharge>,
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
        metrics::gauge!("mesh_prepare_dispatched_jobs").increment(1.0);
        Self(control)
    }
}

impl Drop for QueuedCharge {
    fn drop(&mut self) {
        metrics::gauge!("mesh_prepare_dispatched_jobs").decrement(1.0);
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
            || config.workers > Semaphore::MAX_PERMITS
            || config.queue_capacity > Semaphore::MAX_PERMITS
            || config
                .workers
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
        // Jobs reserve execution slots before dispatch. This channel can hold
        // every reserved job, even before an idle OS thread is scheduled.
        let (sender, receiver) = crossbeam_channel::bounded(config.workers);
        let receiver = Arc::new(receiver);
        let control = Arc::new(PoolControl {
            force_cancel: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            live_workers: AtomicUsize::new(0),
            busy_workers: AtomicUsize::new(0),
            waiting_jobs: AtomicUsize::new(0),
            cancelled_queued: AtomicUsize::new(0),
            retained_input_bytes: AtomicUsize::new(0),
            max_retained_input_bytes: config.max_retained_input_bytes,
            execution_slots: Arc::new(Semaphore::new(config.workers)),
            waiting_slots: Arc::new(Semaphore::new(config.queue_capacity)),
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
    use futures::poll;
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

    async fn blocked(
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
            .submit(deadline, move |_| {
                let _ = started_tx.send(());
                gate.recv_timeout(Duration::from_secs(5)).unwrap();
                lease
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
        let (sender, receiver) = crossbeam_channel::bounded(settings.workers);
        let receiver = Arc::new(receiver);
        let control = Arc::new(PoolControl {
            force_cancel: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            live_workers: AtomicUsize::new(0),
            busy_workers: AtomicUsize::new(0),
            waiting_jobs: AtomicUsize::new(0),
            cancelled_queued: AtomicUsize::new(0),
            retained_input_bytes: AtomicUsize::new(0),
            max_retained_input_bytes: settings.max_retained_input_bytes,
            execution_slots: Arc::new(Semaphore::new(settings.workers)),
            waiting_slots: Arc::new(Semaphore::new(settings.queue_capacity)),
        });
        let runtime = PreparePoolRuntime {
            state: Arc::new(SubmitState {
                sender: Mutex::new(Some(sender)),
                receiver: Arc::downgrade(&receiver),
                control,
                config: settings,
            }),
            threads: Vec::new(),
        };
        (runtime, receiver)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ten_execution_slots_and_one_hundred_waiters_accept_a_burst() {
        let mut settings = PoolConfig::default();
        settings.queue_timeout = Duration::from_secs(10);
        settings.prepare_timeout = Duration::from_secs(20);
        let mut runtime = PreparePoolRuntime::new(settings).unwrap();
        let handle = runtime.handle();
        let (release, gate) = crossbeam_channel::bounded::<()>(10);
        let mut running = Vec::new();
        // There is no need to wait for idle OS threads to receive these jobs.
        for _ in 0..10 {
            let gate = gate.clone();
            running.push(
                handle
                    .submit(handle.prepare_deadline(), move |_| {
                        gate.recv_timeout(Duration::from_secs(5)).unwrap();
                    })
                    .await
                    .unwrap(),
            );
        }
        assert_eq!(handle.stats().execution_reserved, 10);
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
        for _ in 0..10 {
            release.send(()).unwrap();
        }
        for ticket in running {
            ticket.wait().await.unwrap();
        }
        for submit in waiting {
            submit.await.unwrap().wait().await.unwrap();
        }
        eventually(|| handle.stats().execution_reserved == 0).await;
        assert_eq!(handle.stats().queued_jobs, 0);
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn idle_worker_scheduling_does_not_reduce_execution_capacity() {
        let mut settings = config();
        settings.workers = 10;
        let (mut runtime, receiver) = deferred_worker(settings);
        let handle = runtime.handle();
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
        for ticket in tickets {
            receiver.recv().unwrap().execute();
            ticket.wait().await.unwrap();
        }
        assert_eq!(handle.stats().execution_reserved, 0);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn canceled_waiter_releases_queue_slot_and_input_immediately() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let called = Arc::new(AtomicBool::new(false));
        let job_called = called.clone();
        let lease = handle.retain_input(7).unwrap();
        let mut canceled = Box::pin(handle.submit(handle.prepare_deadline(), move |_| {
            job_called.store(true, Ordering::Release);
            lease
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
        assert_eq!(handle.stats().retained_input_bytes, 5);
        assert_eq!(handle.stats().execution_reserved, 1);
        let mut replacement = Box::pin(handle.submit(handle.prepare_deadline(), |_| ()));
        assert!(poll!(&mut replacement).is_pending());
        drop(replacement);
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        assert_eq!(waiting.await.unwrap().wait().await.unwrap(), 42);
        assert!(!called.load(Ordering::Acquire));
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn waiting_jobs_are_fifo_and_new_submitters_cannot_bypass_them() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(1).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let mut first = Box::pin(handle.submit(handle.prepare_deadline(), |_| 1));
        assert!(poll!(&mut first).is_pending());
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        eventually(|| handle.stats().busy_workers == 0).await;
        // The released permit belongs to the first waiter even before it polls.
        let mut second = Box::pin(handle.submit(handle.prepare_deadline(), |_| 2));
        assert!(poll!(&mut second).is_pending());
        assert_eq!(first.await.unwrap().wait().await.unwrap(), 1);
        assert_eq!(second.await.unwrap().wait().await.unwrap(), 2);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queue_timeout_frees_waiting_slot_and_input_without_running_job() {
        let mut settings = config();
        settings.queue_timeout = Duration::from_millis(20);
        let mut runtime = PreparePoolRuntime::new(settings).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let lease = handle.retain_input(7).unwrap();
        let result = handle
            .submit(handle.prepare_deadline(), move |_| {
                drop(lease);
                panic!("expired waiting job must not execute");
            })
            .await;
        assert!(matches!(result, Err(PrepareError::QueueTimeout)));
        assert_eq!(handle.stats().queued_jobs, 0);
        assert_eq!(handle.stats().retained_input_bytes, 5);
        assert_eq!(handle.stats().execution_reserved, 1);
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn overall_deadline_can_expire_while_waiting_for_capacity() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let result = handle
            .submit(Instant::now() + Duration::from_millis(20), |_| ())
            .await;
        assert!(matches!(result, Err(PrepareError::Timeout)));
        assert_eq!(handle.stats().queued_jobs, 0);
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn running_cancellation_holds_execution_and_input_until_return() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(32).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        drop(active);
        assert_eq!(handle.stats().execution_reserved, 1);
        assert_eq!(handle.stats().busy_workers, 1);
        assert_eq!(handle.stats().retained_input_bytes, 32);
        assert!(matches!(
            handle.retain_input(1),
            Err(PrepareError::BudgetExceeded)
        ));
        let mut next = Box::pin(handle.submit(handle.prepare_deadline(), |_| 9));
        assert!(poll!(&mut next).is_pending());
        release.send(()).unwrap();
        assert_eq!(next.await.unwrap().wait().await.unwrap(), 9);
        eventually(|| handle.stats().retained_input_bytes == 0).await;
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn running_timeout_does_not_replace_worker_or_release_input() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(32).unwrap(),
            Instant::now() + Duration::from_millis(30),
        )
        .await;
        started.await.unwrap();
        assert!(matches!(active.wait().await, Err(PrepareError::Timeout)));
        assert_eq!(handle.stats().live_workers, 1);
        assert_eq!(handle.stats().execution_reserved, 1);
        assert_eq!(handle.stats().busy_workers, 1);
        assert_eq!(handle.stats().retained_input_bytes, 32);
        release.send(()).unwrap();
        eventually(|| handle.stats().execution_reserved == 0).await;
        assert_eq!(handle.stats().retained_input_bytes, 0);
        runtime.shutdown().await;
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
        )
        .await;
        started.await.unwrap();
        let release_later = async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            release.send(()).unwrap();
        };
        let (result, ()) = tokio::join!(active.wait(), release_later);
        assert_eq!(result.unwrap().bytes(), 1);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dispatched_timeout_holds_execution_slot_until_worker_cleanup() {
        let mut settings = config();
        settings.queue_timeout = Duration::from_millis(20);
        let (mut runtime, receiver) = deferred_worker(settings);
        let handle = runtime.handle();
        let lease = handle.retain_input(7).unwrap();
        let ticket = handle
            .submit(handle.prepare_deadline(), move |_| {
                drop(lease);
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
        assert_eq!(handle.stats().cancelled_queued_jobs, 1);
        assert_eq!(handle.stats().retained_input_bytes, 7);
        receiver.recv().unwrap().execute();
        assert_eq!(handle.stats().execution_reserved, 0);
        assert_eq!(handle.stats().cancelled_queued_jobs, 0);
        assert_eq!(handle.stats().retained_input_bytes, 0);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn worker_checks_queue_deadline_even_when_ticket_has_never_been_polled() {
        let mut settings = config();
        settings.queue_timeout = Duration::from_millis(20);
        let (mut runtime, receiver) = deferred_worker(settings);
        let handle = runtime.handle();
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
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
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
            _lease: InputLease,
        }
        impl Drop for SlowOutput {
            fn drop(&mut self) {
                let _ = self.started.take().unwrap().send(());
                self.gate.recv_timeout(Duration::from_secs(5)).unwrap();
            }
        }
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (work_started_tx, work_started) = oneshot::channel();
        let (work_release, work_gate) = mpsc::channel();
        let (cleanup_started_tx, cleanup_started) = oneshot::channel();
        let (cleanup_release, cleanup_gate) = mpsc::channel();
        let output = SlowOutput {
            started: Some(cleanup_started_tx),
            gate: cleanup_gate,
            _lease: handle.retain_input(32).unwrap(),
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
        assert_eq!(handle.stats().retained_input_bytes, 32);
        cleanup_release.send(()).unwrap();
        assert_eq!(next.await.unwrap().wait().await.unwrap(), 9);
        assert_eq!(handle.stats().retained_input_bytes, 0);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn result_and_lease_clones_keep_one_charge_after_worker_finishes() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let lease = handle.retain_input(32).unwrap();
        let shared = lease.clone();
        let ticket = handle
            .submit(handle.prepare_deadline(), move |_| lease)
            .await
            .unwrap();
        eventually(|| handle.stats().execution_reserved == 0).await;
        assert_eq!(handle.stats().retained_input_bytes, 32);
        assert!(matches!(
            handle.retain_input(1),
            Err(PrepareError::BudgetExceeded)
        ));
        drop(ticket);
        assert_eq!(handle.stats().retained_input_bytes, 32);
        drop(shared);
        assert_eq!(handle.stats().retained_input_bytes, 0);
        let lease = handle.retain_input(32).unwrap();
        let returned = handle
            .submit(handle.prepare_deadline(), move |_| lease)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(handle.stats().retained_input_bytes, 32);
        drop(returned);
        assert_eq!(handle.stats().retained_input_bytes, 0);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sequential_jobs_share_deadline_and_do_not_nest_on_single_worker() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let deadline = Instant::now() + Duration::from_millis(50);
        let lease = handle.retain_input(32).unwrap();
        let lease = handle
            .submit(deadline, move |_| lease)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        let lease = handle
            .submit(deadline, move |_| lease)
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
        assert_eq!(handle.stats().retained_input_bytes, 32);
        tokio::time::sleep_until(deadline.into()).await;
        assert!(matches!(
            handle.submit(deadline, move |_| lease).await,
            Err(PrepareError::Timeout)
        ));
        assert_eq!(handle.stats().retained_input_bytes, 0);
        runtime.shutdown().await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn close_wakes_waiters_and_rejects_new_submissions() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let lease = handle.retain_input(7).unwrap();
        let mut waiting = Box::pin(handle.submit(handle.prepare_deadline(), move |_| lease));
        assert!(poll!(&mut waiting).is_pending());
        runtime.close();
        assert!(matches!(waiting.await, Err(PrepareError::Closed)));
        assert_eq!(handle.stats().queued_jobs, 0);
        assert_eq!(handle.stats().retained_input_bytes, 5);
        assert!(matches!(
            handle.submit(handle.prepare_deadline(), |_| ()).await,
            Err(PrepareError::Closed)
        ));
        assert!(matches!(handle.retain_input(1), Err(PrepareError::Closed)));
        release.send(()).unwrap();
        drop(active.wait().await.unwrap());
        assert_eq!(runtime.shutdown().await.remaining_workers, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_is_bounded_with_worker_still_blocked_and_wakes_waiters() {
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let mut waiting = Box::pin(handle.submit(handle.prepare_deadline(), |_| ()));
        assert!(poll!(&mut waiting).is_pending());
        let report = tokio::time::timeout(Duration::from_secs(1), runtime.shutdown())
            .await
            .unwrap();
        assert_eq!(report.remaining_workers, 1);
        assert!(report.forced);
        assert!(matches!(waiting.await, Err(PrepareError::Closed)));
        assert_eq!(handle.stats().execution_reserved, 1);
        release.send(()).unwrap();
        assert!(matches!(active.wait().await, Err(PrepareError::Cancelled)));
        eventually(|| handle.stats().live_workers == 0).await;
        assert_eq!(handle.stats().retained_input_bytes, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_runtime_does_not_join_and_weak_handles_do_not_keep_it_open() {
        let runtime = PreparePoolRuntime::new(config()).unwrap();
        let handle = runtime.handle();
        let (active, started, release) = blocked(
            &handle,
            handle.retain_input(5).unwrap(),
            handle.prepare_deadline(),
        )
        .await;
        started.await.unwrap();
        let start = Instant::now();
        drop(runtime);
        assert!(start.elapsed() < Duration::from_secs(1));
        assert!(matches!(handle.retain_input(1), Err(PrepareError::Closed)));
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
        let mut runtime = PreparePoolRuntime::new(config()).unwrap();
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
            let canceled = control.cancel_queued();
            let running = claim.join().unwrap();
            assert_ne!(canceled, running);
            if canceled {
                control.pool.release_cancelled_queued();
            }
        }
        assert_eq!(runtime.handle().stats().cancelled_queued_jobs, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_submitters_cannot_submit_after_close_returns() {
        let runtime = PreparePoolRuntime::new(config()).unwrap();
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
    fn invalid_capacity_and_zero_budget_are_rejected() {
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
                max_retained_input_bytes: 0,
                ..config()
            },
            PoolConfig {
                workers: usize::MAX,
                ..config()
            },
            PoolConfig {
                queue_capacity: usize::MAX,
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
}
