//! A store for tests: each pool answers from the residency the test sets, after an
//! optional delay or with a set failure.

use std::{
    collections::HashMap,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use async_trait::async_trait;
use futures::future::join_all;
use parking_lot::Mutex;

use super::{keys::STORE_TAINT_KEY, merge_pools, FailReason, KeyState, PoolAnswer, StorageBackend};

#[derive(Debug, Clone, Default)]
pub struct MockPool {
    /// Keys the pool holds; any other key is absent.
    pub keys: HashMap<String, KeyState>,
    pub delay: Duration,
    pub fail: Option<FailReason>,
}

#[derive(Debug, Default)]
pub struct MockStore {
    pub pools: Mutex<Vec<MockPool>>,
    queries: AtomicUsize,
}

impl MockStore {
    pub fn new(pools: Vec<MockPool>) -> Self {
        Self {
            pools: Mutex::new(pools),
            queries: AtomicUsize::new(0),
        }
    }

    pub fn queries(&self) -> usize {
        self.queries.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl StorageBackend for MockStore {
    async fn query(&self, keys: &[String]) -> Result<Vec<KeyState>, FailReason> {
        self.queries.fetch_add(1, Ordering::Relaxed);
        let pools = self.pools.lock().clone();
        let answers = join_all(pools.iter().map(|pool| async move {
            if !pool.delay.is_zero() {
                tokio::time::sleep(pool.delay).await;
            }
            match pool.fail {
                Some(reason) => Err(reason),
                None => Ok(PoolAnswer {
                    states: keys
                        .iter()
                        .map(|key| pool.keys.get(key).copied().unwrap_or(KeyState::Absent))
                        .collect(),
                    tainted: pool.keys.contains_key(STORE_TAINT_KEY),
                }),
            }
        }))
        .await;
        merge_pools(keys.len(), &answers)
    }

    fn tainted_pools(&self) -> u32 {
        let pools = self.pools.lock();
        pools
            .iter()
            .filter(|pool| pool.keys.contains_key(STORE_TAINT_KEY))
            .count() as u32
    }
}
