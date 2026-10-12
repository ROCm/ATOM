//! Bounded CPU execution for request parsing, chat templates and tokenization.
//!
//! Waiting for execution capacity is asynchronous and bounded. Dispatched work
//! retains its execution slot until actual cleanup. Weak handles let the server
//! close the pool even when jobs retain an AppContext.

mod control;
mod job;
#[cfg(test)]
mod tests;

use std::{
    io,
    sync::{Arc, Weak},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_channel::{Receiver, Sender, TrySendError};
use parking_lot::Mutex;
use tokio::sync::{oneshot, Semaphore, TryAcquireError};

pub use self::job::{JobContext, PrepareTicket};
use self::{
    control::{PoolControl, Slot},
    job::{Job, JobControl},
};
use crate::config::{ConfigError, ConfigResult};

/// How long shutdown waits for workers after transport draining.
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub workers: usize,
    pub queue_capacity: usize,
    pub queue_timeout: Duration,
    pub prepare_timeout: Duration,
}

impl PoolConfig {
    /// Reject limits that the semaphores or deadlines cannot represent.
    pub fn validate(&self) -> ConfigResult<()> {
        let invalid = |field: &str, value: String, reason: &str| ConfigError::InvalidValue {
            field: format!("prepare_pool.{field}"),
            value,
            reason: reason.to_string(),
        };
        for (field, value) in [
            ("workers", self.workers),
            ("queue_capacity", self.queue_capacity),
        ] {
            if value == 0 || value > Semaphore::MAX_PERMITS {
                return Err(invalid(
                    field,
                    value.to_string(),
                    "Must be positive and fit the semaphore permit limit",
                ));
            }
        }
        for (field, value) in [
            ("queue_timeout_ms", self.queue_timeout),
            ("prepare_timeout_ms", self.prepare_timeout),
        ] {
            if value.is_zero() || Instant::now().checked_add(value).is_none() {
                return Err(invalid(
                    field,
                    value.as_millis().to_string(),
                    "Must be a positive, representable duration",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PrepareError {
    #[error("request preparation queue is full")]
    Full,
    #[error("request preparation pool is closed")]
    Closed,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolStats {
    pub live_workers: usize,
    /// Jobs that have reserved execution capacity, including pending dispatch.
    pub execution_reserved: usize,
    /// OS threads currently processing or cleaning up a job.
    pub busy_workers: usize,
    /// Bounded asynchronous waiters that have not reserved execution capacity.
    pub queued_jobs: usize,
    /// Jobs handed to the OS-thread channel that no worker has claimed or discarded.
    pub dispatched_jobs: usize,
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
        let control = &state.control;
        PoolStats {
            live_workers: control.live_workers.get(),
            execution_reserved: control.execution_reserved.get(),
            busy_workers: control.busy_workers.get(),
            queued_jobs: control.waiting_jobs.get(),
            dispatched_jobs: control.dispatched_jobs.get(),
            failed: control.is_failed(),
        }
    }

    pub fn is_failed(&self) -> bool {
        self.state
            .upgrade()
            .is_some_and(|state| state.control.is_failed())
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
        state.open_sender(&state.sender.lock())?;
        let execution = state.acquire_execution(queue_deadline, deadline).await?;
        let control = JobControl::new(state.control.clone());
        let (job, receiver) = Job::new(
            control.clone(),
            execution,
            enqueued,
            queue_deadline,
            deadline,
            work,
        );
        let result = {
            let sender = state.sender.lock();
            // Keep ownership of `job` outside the lock, including errors.
            match state.open_sender(&sender) {
                Ok(sender) => sender.try_send(job).map_err(|error| match error {
                    // Every sent job owns one execution permit, and the channel
                    // can hold all permits. Full is unreachable unless that
                    // invariant is broken.
                    TrySendError::Full(job) => (PrepareError::Full.count(), job),
                    TrySendError::Disconnected(job) => (PrepareError::WorkerFailed.count(), job),
                }),
                Err(error) => Err((error, job)),
            }
        };
        if let Err((error, job)) = result {
            drop(job);
            return Err(error);
        }
        Ok(PrepareTicket::new(
            receiver,
            control,
            queue_deadline,
            deadline,
        ))
    }
}

impl SubmitState {
    async fn acquire_execution(
        &self,
        queue_deadline: Instant,
        deadline: Instant,
    ) -> Result<Slot, PrepareError> {
        let control = &self.control;
        let unavailable = || control.failed_or(PrepareError::Closed).count();
        match control.execution_slots.clone().try_acquire_owned() {
            Ok(permit) => return Ok(Slot::new(permit, &control.execution_reserved)),
            Err(TryAcquireError::Closed) => return Err(unavailable()),
            Err(TryAcquireError::NoPermits) => {}
        }

        // Bound waiter allocation before awaiting the fair execution semaphore.
        // Dropping a pending submit future releases this slot and its input.
        let permit =
            control
                .waiting_slots
                .clone()
                .try_acquire_owned()
                .map_err(|error| match error {
                    TryAcquireError::Closed => unavailable(),
                    TryAcquireError::NoPermits => PrepareError::Full.count(),
                })?;
        let _waiting = Slot::new(permit, &control.waiting_jobs);
        let wait_deadline = queue_deadline.min(deadline);
        let permit = tokio::select! {
            biased;
            result = control.execution_slots.clone().acquire_owned() => {
                result.map_err(|_| unavailable())
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
        Ok(Slot::new(permit, &control.execution_reserved))
    }

    /// The sender, unless the pool is closed or has failed.
    fn open_sender<'a>(
        &self,
        sender: &'a Option<Sender<Job>>,
    ) -> Result<&'a Sender<Job>, PrepareError> {
        match sender {
            Some(sender) if !self.control.is_failed() => Ok(sender),
            _ => Err(self.control.failed_or(PrepareError::Closed).count()),
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

/// The server owns this value; AppContext holds only `PrepareHandle`.
pub struct PreparePoolRuntime {
    state: Arc<SubmitState>,
    threads: Vec<JoinHandle<()>>,
    shutdown_grace: Duration,
}

thread_local! {
    static WORKER_INDEX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Index of the preparation worker running on this thread, if any.
pub fn current_worker_index() -> Option<usize> {
    WORKER_INDEX.get()
}

impl PreparePoolRuntime {
    pub fn new(config: PoolConfig) -> io::Result<Self> {
        config
            .validate()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))?;
        let (mut runtime, receiver) = Self::without_workers(config);
        let control = runtime.state.control.clone();
        for index in 0..runtime.state.config.workers {
            let receiver = receiver.clone();
            let control = control.clone();
            let live = control.live_workers.hold(1);
            // On failure, Drop closes and cancels the partial pool without joining.
            let handle = thread::Builder::new()
                .name(format!("mesh-prepare-{index}"))
                .spawn(move || {
                    WORKER_INDEX.set(Some(index));
                    control.run_worker(receiver, live)
                })?;
            runtime.threads.push(handle);
        }
        // Only real workers retain the Receiver. A dead pool cannot accept jobs.
        drop(receiver);
        Ok(runtime)
    }

    /// An open pool whose jobs wait in the returned receiver until a worker runs them.
    fn without_workers(config: PoolConfig) -> (Self, Arc<Receiver<Job>>) {
        // Jobs reserve execution slots before dispatch. This channel can hold
        // every reserved job, even before an idle OS thread is scheduled.
        let (sender, receiver) = crossbeam_channel::bounded(config.workers);
        let receiver = Arc::new(receiver);
        let runtime = Self {
            state: Arc::new(SubmitState {
                sender: Mutex::new(Some(sender)),
                receiver: Arc::downgrade(&receiver),
                control: PoolControl::new(&config),
                config,
            }),
            threads: Vec::new(),
            shutdown_grace: SHUTDOWN_GRACE,
        };
        (runtime, receiver)
    }

    pub fn handle(&self) -> PrepareHandle {
        PrepareHandle {
            state: Arc::downgrade(&self.state),
        }
    }

    pub fn close(&self) {
        self.state.close();
    }

    /// Close after transport drain and return the workers still running.
    /// Cleanup runs on an OS thread; waiting remains bounded even when
    /// synchronous work or an input destructor blocks.
    pub async fn shutdown(&mut self) -> usize {
        self.close();
        let deadline = Instant::now() + self.shutdown_grace;
        if let Some(reaped) = self.reap_until(deadline) {
            let _ = tokio::time::timeout_at(deadline.into(), reaped).await;
        }
        self.state.control.cancel_all();
        self.state.control.live_workers.get()
    }

    /// Join workers on an OS thread until `deadline`, then cancel and detach the
    /// rest. The receiver resolves once reaping ends.
    fn reap_until(&mut self, deadline: Instant) -> Option<oneshot::Receiver<()>> {
        if self.threads.is_empty() {
            return None;
        }
        let threads = std::mem::take(&mut self.threads);
        let control = self.state.control.clone();
        let receiver = self.state.receiver.clone();
        let (reaped, result) = oneshot::channel();
        match thread::Builder::new()
            .name("mesh-prepare-reaper".into())
            .spawn(move || {
                control.reap(threads, receiver, deadline);
                let _ = reaped.send(());
            }) {
            Ok(_) => Some(result),
            Err(error) => {
                tracing::error!(%error, "could not start preparation shutdown reaper");
                None
            }
        }
    }
}

impl Drop for PreparePoolRuntime {
    fn drop(&mut self) {
        self.close();
        self.state.control.cancel_all();
        // Drop cannot wait for CPU work, including at startup failure.
        self.reap_until(Instant::now());
    }
}
