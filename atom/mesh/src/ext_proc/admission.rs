use std::{sync::Arc, time::Duration};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{app_context::AppContext, core::token_bucket::TokenBucket};

use super::error::ProcessingError;

pub(super) struct Admission {
    running: Arc<Semaphore>,
    waiting: Arc<Semaphore>,
    bucket: Option<Arc<TokenBucket>>,
    timeout: Duration,
}

pub(super) struct AdmissionLease {
    _running: OwnedSemaphorePermit,
    bucket: Option<Arc<TokenBucket>>,
}

impl Admission {
    pub fn new(app: &AppContext) -> Self {
        let config = &app.router_config;
        let capacity = if config.max_concurrent_requests > 0 {
            config.max_concurrent_requests as usize
        } else {
            config.ext_proc.max_streams
        };
        Self {
            running: Arc::new(Semaphore::new(capacity)),
            waiting: Arc::new(Semaphore::new(config.queue_size)),
            bucket: app.rate_limiter.clone(),
            timeout: Duration::from_secs(config.queue_timeout_secs),
        }
    }

    pub async fn acquire(&self) -> Result<AdmissionLease, ProcessingError> {
        if let Ok(running) = self.running.clone().try_acquire_owned() {
            if self.bucket.is_none() {
                return Ok(AdmissionLease {
                    _running: running,
                    bucket: None,
                });
            }
            if self.bucket.as_ref().unwrap().try_acquire(1.0).await.is_ok() {
                return Ok(AdmissionLease {
                    _running: running,
                    bucket: self.bucket.clone(),
                });
            }
        }
        let waiting = self.waiting.clone().try_acquire_owned().map_err(|_| {
            ProcessingError::new(
                429,
                "admission_full",
                "inference admission capacity exhausted",
            )
        })?;
        metrics::gauge!("mesh_ext_proc_queued_requests").increment(1.0);
        let _waiting = WaitingLease(waiting);
        tokio::time::timeout(self.timeout, async {
            // Tokio's semaphore retains FIFO position while waiting; cancellation
            // removes the waiter and returns any acquired permit automatically.
            let running = self.running.clone().acquire_owned().await.map_err(|_| {
                ProcessingError::new(503, "admission_closed", "inference admission closed")
            })?;
            if let Some(bucket) = &self.bucket {
                // The shared HTTP token bucket does not expose a cancellation-safe
                // notification subscription. Poll only while holding our FIFO slot.
                while bucket.try_acquire(1.0).await.is_err() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
            Ok(AdmissionLease {
                _running: running,
                bucket: self.bucket.clone(),
            })
        })
        .await
        .map_err(|_| ProcessingError::new(408, "admission_timeout", "inference queue timeout"))?
    }
}

impl Drop for AdmissionLease {
    fn drop(&mut self) {
        if let Some(bucket) = self.bucket.take() {
            bucket.return_tokens_sync(1.0);
        }
    }
}

struct WaitingLease(#[allow(dead_code)] OwnedSemaphorePermit);

impl Drop for WaitingLease {
    fn drop(&mut self) {
        metrics::gauge!("mesh_ext_proc_queued_requests").decrement(1.0);
    }
}
