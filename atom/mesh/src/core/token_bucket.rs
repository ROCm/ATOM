use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::Mutex;
use tokio::sync::Notify;
use tracing::{debug, trace};

/// Token bucket for rate limiting.
///
/// This implementation provides:
/// - Smooth rate limiting with configurable refill rate
/// - Burst capacity handling
/// - Event-driven waiting; FIFO ordering is owned by the admission queue
/// - Sync token return for Drop handlers (via `return_tokens_sync`)
///
/// Uses `parking_lot::Mutex` for sync-compatible locking (no async required).
#[derive(Clone)]
pub struct TokenBucket {
    inner: Arc<Mutex<TokenBucketInner>>,
    notify: Arc<Notify>,
    capacity: f64,
    refill_rate: f64, // tokens per second
}

struct TokenBucketInner {
    tokens: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens (burst capacity)
    /// * `refill_rate` - Tokens added per second (0 for pure concurrency limiting)
    pub fn new(capacity: usize, refill_rate: usize) -> Self {
        let capacity = capacity as f64;
        // Allow refill_rate=0 for pure concurrency limiting (semaphore behavior)
        // When refill_rate=0, tokens are only returned via return_tokens()
        let refill_rate = refill_rate as f64;

        Self {
            inner: Arc::new(Mutex::new(TokenBucketInner {
                tokens: capacity,
                last_refill: Instant::now(),
            })),
            notify: Arc::new(Notify::new()),
            capacity,
            refill_rate,
        }
    }

    /// Try to acquire tokens immediately.
    ///
    /// Returns `Ok(())` if tokens were acquired, `Err(())` if insufficient tokens.
    pub async fn try_acquire(&self, tokens: f64) -> Result<(), ()> {
        self.try_acquire_sync(tokens)
    }

    /// Sync version of try_acquire (for internal use).
    fn try_acquire_sync(&self, tokens: f64) -> Result<(), ()> {
        let mut inner = self.inner.lock();

        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(self.capacity);
        inner.last_refill = now;

        trace!(
            "Token bucket: {} tokens available, requesting {}",
            inner.tokens,
            tokens
        );

        if inner.tokens >= tokens {
            inner.tokens -= tokens;
            debug!(
                "Token bucket: acquired {} tokens, {} remaining",
                tokens, inner.tokens
            );
            Ok(())
        } else {
            Err(())
        }
    }

    /// Acquire tokens, waiting if necessary.
    ///
    /// When `refill_rate=0`, waits indefinitely for tokens to be returned via `return_tokens()`.
    /// Use `acquire_timeout()` to set an appropriate timeout.
    pub async fn acquire(&self, tokens: f64) -> Result<(), tokio::time::error::Elapsed> {
        loop {
            // Register before inspecting the bucket: notify_waiters does not
            // retain a notification for a future created after token return.
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.try_acquire_sync(tokens).is_ok() {
                return Ok(());
            }
            let delay = if self.refill_rate > 0.0 {
                let inner = self.inner.lock();
                Some(Duration::from_secs_f64(
                    ((tokens - inner.tokens) / self.refill_rate).max(0.001),
                ))
            } else {
                None
            };
            tokio::select! {
                _ = notified => {},
                _ = async {
                    match delay {
                        Some(delay) => tokio::time::sleep(delay).await,
                        None => std::future::pending::<()>().await,
                    }
                } => {},
            }
        }
    }

    /// Acquire tokens with custom timeout.
    pub async fn acquire_timeout(
        &self,
        tokens: f64,
        timeout: Duration,
    ) -> Result<(), tokio::time::error::Elapsed> {
        tokio::time::timeout(timeout, self.acquire(tokens)).await?
    }

    /// Return tokens to the bucket (sync version).
    ///
    /// This is safe to call from sync contexts (e.g., Drop handlers).
    /// Uses `parking_lot::Mutex` which never blocks indefinitely.
    pub fn return_tokens_sync(&self, tokens: f64) {
        {
            let mut inner = self.inner.lock();
            inner.tokens = (inner.tokens + tokens).min(self.capacity);
            debug!(
                "Token bucket: returned {} tokens, {} available",
                tokens, inner.tokens
            );
        } // Release lock before notify
        self.notify.notify_waiters();
    }

    /// Return tokens to the bucket (async version for API compatibility).
    pub async fn return_tokens(&self, tokens: f64) {
        self.return_tokens_sync(tokens);
    }

    /// Get current available tokens (for monitoring).
    pub async fn available_tokens(&self) -> f64 {
        let mut inner = self.inner.lock();

        let now = Instant::now();
        let elapsed = now.duration_since(inner.last_refill).as_secs_f64();
        let refill_amount = elapsed * self.refill_rate;

        inner.tokens = (inner.tokens + refill_amount).min(self.capacity);
        inner.last_refill = now;

        inner.tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn refill_waiters_keep_waiting_until_their_external_deadline() {
        let bucket = TokenBucket::new(1, 20);
        bucket.try_acquire(1.0).await.unwrap();
        // Several waiters need several refill intervals. Losing one wakeup race
        // must not create an implicit timeout shorter than the caller's deadline.
        let results = futures_util::future::join_all(
            (0..4).map(|_| bucket.acquire_timeout(1.0, Duration::from_secs(2))),
        )
        .await;
        assert!(results.iter().all(Result::is_ok));
        assert!(bucket.available_tokens().await < 1.0);
    }

    #[tokio::test]
    async fn test_token_bucket_basic() {
        let bucket = TokenBucket::new(10, 5);

        assert!(bucket.try_acquire(5.0).await.is_ok());
        assert!(bucket.try_acquire(5.0).await.is_ok());

        assert!(bucket.try_acquire(1.0).await.is_err());

        tokio::time::sleep(Duration::from_millis(300)).await;

        assert!(bucket.try_acquire(1.0).await.is_ok());
    }

    #[tokio::test]
    async fn test_token_bucket_refill() {
        let bucket = TokenBucket::new(10, 10);

        assert!(bucket.try_acquire(10.0).await.is_ok());

        tokio::time::sleep(Duration::from_millis(500)).await;

        let available = bucket.available_tokens().await;
        assert!((4.0..=6.0).contains(&available));
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_rate() {
        // With refill_rate=0, tokens should only come back via return_tokens()
        let bucket = TokenBucket::new(2, 0);

        // Acquire both tokens
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens available
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Wait - should NOT refill automatically
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Return a token - now we should be able to acquire
        bucket.return_tokens(1.0).await;
        assert!(bucket.try_acquire(1.0).await.is_ok());

        // No more tokens again
        assert!(bucket.try_acquire(1.0).await.is_err());
    }

    #[tokio::test]
    async fn test_token_bucket_zero_refill_with_notify() {
        // Test that acquire wakes up when tokens are returned
        let bucket = Arc::new(TokenBucket::new(1, 0));

        // Acquire the only token
        assert!(bucket.try_acquire(1.0).await.is_ok());

        let bucket_clone = bucket.clone();

        // Spawn a task that will return the token after a delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(100)).await;
            bucket_clone.return_tokens(1.0).await;
        });

        // This should wait and then succeed when token is returned
        let result = bucket.acquire_timeout(1.0, Duration::from_secs(1)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_return_tokens_sync() {
        // Test that sync return works correctly
        let bucket = TokenBucket::new(2, 0);

        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_ok());
        assert!(bucket.try_acquire(1.0).await.is_err());

        // Use sync return
        bucket.return_tokens_sync(1.0);
        assert!(bucket.try_acquire(1.0).await.is_ok());
    }
}
