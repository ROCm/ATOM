use std::{sync::Arc, time::Duration};

use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

use crate::{config::RouterConfig, core::token_bucket::TokenBucket};

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct AdmissionError {
    pub status: u16,
    pub code: &'static str,
    pub message: &'static str,
}
impl AdmissionError {
    fn new(status: u16, code: &'static str, message: &'static str) -> Self {
        Self {
            status,
            code,
            message,
        }
    }
}

pub struct AdmissionController {
    running: Arc<Semaphore>,
    waiting: Arc<Semaphore>,
    turn: Mutex<()>,
    bucket: Option<Arc<TokenBucket>>,
    timeout: Duration,
}

pub struct AdmissionLease {
    _running: OwnedSemaphorePermit,
    _token: TokenLease,
}

struct TokenLease(Option<Arc<TokenBucket>>);
impl Drop for TokenLease {
    fn drop(&mut self) {
        if let Some(bucket) = &self.0 {
            bucket.return_tokens_sync(1.0);
        }
    }
}

impl AdmissionController {
    pub fn new(config: &RouterConfig, bucket: Option<Arc<TokenBucket>>) -> Self {
        let capacity = if config.max_concurrent_requests > 0 {
            config.max_concurrent_requests as usize
        } else {
            Semaphore::MAX_PERMITS
        };
        Self {
            running: Arc::new(Semaphore::new(capacity)),
            waiting: Arc::new(Semaphore::new(config.queue_size)),
            turn: Mutex::new(()),
            bucket,
            timeout: Duration::from_secs(config.queue_timeout_secs),
        }
    }

    pub async fn acquire(&self, ingress: &'static str) -> Result<AdmissionLease, AdmissionError> {
        // The same FIFO gate covers fast arrivals and queued requests.
        if let Ok(_turn) = self.turn.try_lock() {
            if let Ok(running) = self.running.clone().try_acquire_owned() {
                if let Some(bucket) = &self.bucket {
                    if bucket.try_acquire(1.0).await.is_ok() {
                        return Ok(AdmissionLease {
                            _running: running,
                            _token: TokenLease(self.bucket.clone()),
                        });
                    }
                } else {
                    return Ok(AdmissionLease {
                        _running: running,
                        _token: TokenLease(None),
                    });
                }
            }
        }
        let waiting = self.waiting.clone().try_acquire_owned().map_err(|_| {
            AdmissionError::new(
                429,
                "admission_full",
                "inference admission capacity exhausted",
            )
        })?;
        metrics::gauge!("mesh_admission_queued_requests", "ingress" => ingress).increment(1.0);
        if ingress == "ext_proc" {
            metrics::gauge!("mesh_ext_proc_queued_requests").increment(1.0);
        }
        let _waiting = WaitingLease {
            _permit: waiting,
            ingress,
        };
        tokio::time::timeout(self.timeout, async {
            let _turn = self.turn.lock().await;
            // Do not hold execution capacity while waiting for a rate token.
            if let Some(bucket) = &self.bucket {
                bucket.acquire(1.0).await.map_err(|_| {
                    AdmissionError::new(408, "admission_timeout", "inference queue timeout")
                })?;
            }
            let token = TokenLease(self.bucket.clone());
            let running = self.running.clone().acquire_owned().await.map_err(|_| {
                AdmissionError::new(503, "admission_closed", "inference admission closed")
            })?;
            Ok(AdmissionLease {
                _running: running,
                _token: token,
            })
        })
        .await
        .map_err(|_| AdmissionError::new(408, "admission_timeout", "inference queue timeout"))?
    }
}

struct WaitingLease {
    _permit: OwnedSemaphorePermit,
    ingress: &'static str,
}
impl Drop for WaitingLease {
    fn drop(&mut self) {
        metrics::gauge!("mesh_admission_queued_requests", "ingress" => self.ingress).decrement(1.0);
        if self.ingress == "ext_proc" {
            metrics::gauge!("mesh_ext_proc_queued_requests").decrement(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::poll;

    fn gate(capacity: usize, bucket: Option<Arc<TokenBucket>>) -> AdmissionController {
        AdmissionController {
            running: Arc::new(Semaphore::new(capacity)),
            waiting: Arc::new(Semaphore::new(4)),
            turn: Mutex::new(()),
            bucket,
            timeout: Duration::from_secs(1),
        }
    }

    #[tokio::test]
    async fn queued_requests_keep_fifo_order_and_cancellation_returns_capacity() {
        let gate = gate(1, None);
        let active = gate.acquire("test").await.unwrap();
        let mut first = Box::pin(gate.acquire("test"));
        let mut second = Box::pin(gate.acquire("test"));
        assert!(poll!(&mut first).is_pending());
        assert!(poll!(&mut second).is_pending());
        assert_eq!(gate.waiting.available_permits(), 2);
        drop(active);
        // A later caller cannot use the newly free slot before the queue head.
        let mut newcomer = Box::pin(gate.acquire("test"));
        assert!(poll!(&mut newcomer).is_pending());
        assert!(poll!(&mut second).is_pending());
        let lease = first.await.unwrap();
        drop(second);
        drop(lease);
        drop(newcomer.await.unwrap());
        assert_eq!(gate.running.available_permits(), 1);
        assert_eq!(gate.waiting.available_permits(), 4);
    }

    #[tokio::test]
    async fn rate_wait_does_not_hold_execution_slots_and_canceled_wait_refunds_token() {
        let bucket = Arc::new(TokenBucket::new(1, 0));
        bucket.try_acquire(1.0).await.unwrap();
        let gate = gate(1, Some(bucket.clone()));
        let mut waiting = Box::pin(gate.acquire("test"));
        assert!(poll!(&mut waiting).is_pending());
        assert_eq!(gate.running.available_permits(), 1);
        // Occupy execution capacity independently while the rate token is missing.
        let occupied = gate.running.clone().acquire_owned().await.unwrap();
        bucket.return_tokens_sync(1.0);
        assert!(poll!(&mut waiting).is_pending());
        assert_eq!(bucket.available_tokens().await, 0.0);
        drop(waiting);
        assert_eq!(bucket.available_tokens().await, 1.0);
        assert_eq!(gate.waiting.available_permits(), 4);
        drop(occupied);
        drop(gate.acquire("test").await.unwrap());
        assert_eq!(bucket.available_tokens().await, 1.0);
    }
}
