//! One preparation job: its cancellation state, the worker-side run, and the
//! caller's ticket. Only the caller's side counts a job's error.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Arc,
    },
    time::Instant,
};

use tokio::sync::oneshot;

use super::{
    control::{Held, PoolControl, Slot},
    PrepareError,
};

const QUEUED: u8 = 0;
const RUNNING: u8 = 1;
const CANCELLED: u8 = 2;

type JobResult<T> = Result<T, PrepareError>;
/// Runs the caller's work, or disposes of it with the error, and reports the result.
type Work = Box<dyn FnOnce(Result<&JobContext, PrepareError>) + Send>;

pub(super) struct JobControl {
    state: AtomicU8,
    cancelled: AtomicBool,
    pool: Arc<PoolControl>,
}

impl JobControl {
    pub(super) fn new(pool: Arc<PoolControl>) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU8::new(QUEUED),
            cancelled: AtomicBool::new(false),
            pool,
        })
    }

    /// A worker takes a queued job; fails once its ticket has cancelled it.
    pub(super) fn claim(&self) -> bool {
        self.transition(RUNNING)
    }

    /// The ticket cancels a job that no worker has claimed yet.
    pub(super) fn cancel_queued(&self) -> bool {
        self.transition(CANCELLED)
    }

    fn transition(&self, next: u8) -> bool {
        self.state
            .compare_exchange(QUEUED, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
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
        let pool = &self.control.pool;
        if pool.is_failed()
            || pool.is_force_cancelled()
            || self.control.cancelled.load(Ordering::Acquire)
        {
            Err(pool.failed_or(PrepareError::Cancelled))
        } else if Instant::now() >= self.deadline {
            Err(PrepareError::Timeout)
        } else {
            Ok(())
        }
    }
}

pub struct PrepareTicket<T> {
    receiver: oneshot::Receiver<JobResult<T>>,
    control: Arc<JobControl>,
    queue_deadline: Instant,
    deadline: Instant,
    completed: bool,
}

impl<T> PrepareTicket<T> {
    pub(super) fn new(
        receiver: oneshot::Receiver<JobResult<T>>,
        control: Arc<JobControl>,
        queue_deadline: Instant,
        deadline: Instant,
    ) -> Self {
        Self {
            receiver,
            control,
            queue_deadline,
            deadline,
            completed: false,
        }
    }

    /// Each job's error is counted exactly once, here or in `Drop`; the worker's
    /// later report for a job this ticket gave up on is never received.
    pub async fn wait(mut self) -> Result<T, PrepareError> {
        let overall = tokio::time::sleep_until(self.deadline.into());
        let queue = tokio::time::sleep_until(self.queue_deadline.into());
        tokio::pin!(overall, queue);
        let mut queue_pending = self.queue_deadline < self.deadline;
        let result = loop {
            tokio::select! {
                biased;
                result = &mut self.receiver => {
                    break result
                        .unwrap_or(Err(PrepareError::WorkerFailed))
                        .map_err(PrepareError::count);
                }
                _ = &mut queue, if queue_pending => {
                    if self.control.cancel_queued() {
                        break Err(PrepareError::QueueTimeout.count());
                    }
                    // Once running, only the overall preparation deadline applies.
                    queue_pending = false;
                }
                _ = &mut overall => {
                    let error = PrepareError::Timeout.count();
                    self.control.cancel();
                    break Err(error);
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
            PrepareError::Cancelled.count();
            self.control.cancel();
        }
    }
}

pub(super) struct Job {
    control: Arc<JobControl>,
    /// Counts the job as dispatched until a worker claims or discards it.
    dispatched: Option<Held>,
    enqueued: Instant,
    queue_deadline: Instant,
    deadline: Instant,
    work: Work,
    // Last: even a panicking input destructor runs before capacity is released.
    execution: Option<Slot>,
}

impl Job {
    /// A job running `work`, and the receiver for its result.
    pub(super) fn new<T, F>(
        control: Arc<JobControl>,
        execution: Slot,
        enqueued: Instant,
        queue_deadline: Instant,
        deadline: Instant,
        work: F,
    ) -> (Self, oneshot::Receiver<JobResult<T>>)
    where
        T: Send + 'static,
        F: FnOnce(&JobContext) -> T + Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let job = Self {
            dispatched: Some(control.pool.dispatched_jobs.hold(1)),
            control,
            enqueued,
            queue_deadline,
            deadline,
            work: Box::new(move |context| match context {
                Ok(context) => {
                    let result = catch_unwind(AssertUnwindSafe(|| work(context)))
                        .map_err(|_| PrepareError::Panicked)
                        .and_then(|value| context.check().map(|()| value));
                    // Failed sends dispose of returned input on this worker.
                    let _ = sender.send(result);
                }
                Err(error) => {
                    let _ = sender.send(Err(error));
                    drop(work);
                }
            }),
            execution: Some(execution),
        };
        (job, receiver)
    }

    /// The worker holds the execution slot itself so it outlives job cleanup.
    pub(super) fn take_execution(&mut self) -> Option<Slot> {
        self.execution.take()
    }

    pub(super) fn fail(mut self, error: PrepareError) {
        drop(self.dispatched.take());
        (self.work)(Err(error));
    }

    pub(super) fn execute(mut self) {
        if !self.control.claim() {
            self.fail(PrepareError::Cancelled);
            return;
        }
        drop(self.dispatched.take());
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
            self.fail(error);
            return;
        }
        (self.work)(Ok(&context));
        metrics::histogram!("mesh_prepare_run_seconds").record(now.elapsed().as_secs_f64());
    }
}
