//! Pool-wide state, its tracked levels, and the worker and reaper loops.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Weak,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_channel::Receiver;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{job::Job, PoolConfig, PrepareError};

/// A tracked quantity; `stats()` and its gauge read the same value.
#[derive(Debug)]
pub(super) struct Level {
    value: AtomicUsize,
    gauge: &'static str,
}

impl Level {
    fn new(gauge: &'static str) -> Arc<Self> {
        Arc::new(Self {
            value: AtomicUsize::new(0),
            gauge,
        })
    }

    pub(super) fn get(&self) -> usize {
        self.value.load(Ordering::Acquire)
    }

    pub(super) fn hold(self: &Arc<Self>, amount: usize) -> Held {
        self.value.fetch_add(amount, Ordering::AcqRel);
        metrics::gauge!(self.gauge).increment(amount as f64);
        Held {
            level: self.clone(),
            amount,
        }
    }
}

/// Releases its amount from a `Level` on drop.
#[derive(Debug)]
pub(super) struct Held {
    level: Arc<Level>,
    amount: usize,
}

impl Drop for Held {
    fn drop(&mut self) {
        self.level.value.fetch_sub(self.amount, Ordering::AcqRel);
        metrics::gauge!(self.level.gauge).decrement(self.amount as f64);
    }
}

/// A semaphore permit counted in a `Level`; the level drops before the permit.
pub(super) struct Slot {
    _held: Held,
    _permit: OwnedSemaphorePermit,
}

impl Slot {
    pub(super) fn new(permit: OwnedSemaphorePermit, level: &Arc<Level>) -> Self {
        Self {
            _held: level.hold(1),
            _permit: permit,
        }
    }
}

#[derive(Debug)]
pub(super) struct PoolControl {
    force_cancel: AtomicBool,
    failed: AtomicBool,
    pub(super) live_workers: Arc<Level>,
    pub(super) busy_workers: Arc<Level>,
    pub(super) waiting_jobs: Arc<Level>,
    pub(super) execution_reserved: Arc<Level>,
    pub(super) dispatched_jobs: Arc<Level>,
    pub(super) execution_slots: Arc<Semaphore>,
    pub(super) waiting_slots: Arc<Semaphore>,
}

impl PoolControl {
    pub(super) fn new(config: &PoolConfig) -> Arc<Self> {
        Arc::new(Self {
            force_cancel: AtomicBool::new(false),
            failed: AtomicBool::new(false),
            live_workers: Level::new("mesh_prepare_live_workers"),
            busy_workers: Level::new("mesh_prepare_busy_workers"),
            waiting_jobs: Level::new("mesh_prepare_queued_jobs"),
            execution_reserved: Level::new("mesh_prepare_execution_reserved"),
            dispatched_jobs: Level::new("mesh_prepare_dispatched_jobs"),
            execution_slots: Arc::new(Semaphore::new(config.workers)),
            waiting_slots: Arc::new(Semaphore::new(config.queue_capacity)),
        })
    }

    pub(super) fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    pub(super) fn is_force_cancelled(&self) -> bool {
        self.force_cancel.load(Ordering::Acquire)
    }

    /// Cancel running work and reject dispatched jobs.
    pub(super) fn cancel_all(&self) {
        self.force_cancel.store(true, Ordering::Release);
    }

    /// `WorkerFailed` once the pool has failed, else `error`: the reason a
    /// stopped pool reports to callers and jobs.
    pub(super) fn failed_or(&self, error: PrepareError) -> PrepareError {
        if self.is_failed() {
            PrepareError::WorkerFailed
        } else {
            error
        }
    }

    pub(super) fn run_worker(self: &Arc<Self>, receiver: Arc<Receiver<Job>>, live: Held) {
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            while let Ok(mut job) = receiver.recv() {
                let _busy = self.busy_workers.hold(1);
                // Hold capacity outside job cleanup's panic guard. A destructor
                // panic closes admission before this permit can wake a waiter.
                let _execution = job.take_execution();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    if self.is_force_cancelled() {
                        job.fail(self.failed_or(PrepareError::Cancelled));
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
        drop(live);
    }

    fn mark_failed(&self) {
        self.failed.store(true, Ordering::Release);
        self.cancel_all();
        self.execution_slots.close();
        self.waiting_slots.close();
        tracing::error!("request preparation worker exited unexpectedly");
        PrepareError::WorkerFailed.count();
    }

    pub(super) fn reap(
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
            if Instant::now() >= deadline || self.is_force_cancelled() {
                self.cancel_all();
                if let Some(receiver) = receiver.upgrade() {
                    while let Ok(job) = receiver.try_recv() {
                        // A panicking destructor must not prevent later cleanup.
                        let error = self.failed_or(PrepareError::Cancelled);
                        let _ = catch_unwind(AssertUnwindSafe(|| job.fail(error)));
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
