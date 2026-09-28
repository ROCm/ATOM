//! Round-robin load balancing policy

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use async_trait::async_trait;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers.
#[derive(Debug, Default)]
pub struct RoundRobinPolicy {
    counter: AtomicUsize,
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self {
            counter: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for RoundRobinPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        // Get and increment counter atomically
        let count = self.counter.fetch_add(1, Ordering::Relaxed);
        let selected_idx = count % healthy_indices.len();

        Some(healthy_indices[selected_idx])
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_round_robin_selection() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
    }

    #[tokio::test]
    async fn test_round_robin_with_unhealthy_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w3:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        workers[1].set_healthy(false);

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(2));
    }

    #[tokio::test]
    async fn test_round_robin_reset() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));

        policy.reset();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
    }

    #[tokio::test]
    async fn test_round_robin_empty_workers() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, None);
    }

    #[tokio::test]
    async fn test_round_robin_single_worker() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        let info = SelectWorkerInfo::default();
        for _ in 0..5 {
            assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        }
    }

    #[tokio::test]
    async fn test_round_robin_all_unhealthy() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        workers[0].set_healthy(false);
        workers[1].set_healthy(false);

        let info = SelectWorkerInfo::default();
        assert_eq!(policy.select_worker(&workers, &info).await, None);
    }

    #[test]
    fn test_round_robin_name() {
        let policy = RoundRobinPolicy::new();
        assert_eq!(policy.name(), "round_robin");
    }

    #[test]
    fn test_round_robin_as_any() {
        let policy = RoundRobinPolicy::new();
        assert!(policy.as_any().downcast_ref::<RoundRobinPolicy>().is_some());
    }

    #[test]
    fn test_round_robin_default() {
        let policy = RoundRobinPolicy::default();
        assert_eq!(policy.name(), "round_robin");
    }

    #[tokio::test]
    async fn test_round_robin_even_distribution() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = (0..4)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://w{}:8000", i))
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();

        let info = SelectWorkerInfo::default();
        let mut counts = [0u32; 4];
        for _ in 0..100 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                counts[idx] += 1;
            }
        }

        // Perfect round-robin: each gets exactly 25
        for (i, &count) in counts.iter().enumerate() {
            assert_eq!(count, 25, "Worker {i} got {count}, expected 25");
        }
    }

    #[tokio::test]
    async fn test_round_robin_worker_recovery() {
        let policy = RoundRobinPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![
            Arc::new(
                BasicWorkerBuilder::new("http://w1:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
            Arc::new(
                BasicWorkerBuilder::new("http://w2:8000")
                    .worker_type(WorkerType::Regular)
                    .build(),
            ),
        ];

        let info = SelectWorkerInfo::default();

        // Mark w2 unhealthy
        workers[1].set_healthy(false);
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));

        // Recover w2
        workers[1].set_healthy(true);
        // Should include w2 again in rotation
        let mut saw_w2 = false;
        for _ in 0..4 {
            if let Some(idx) = policy.select_worker(&workers, &info).await {
                if idx == 1 {
                    saw_w2 = true;
                }
            }
        }
        assert!(saw_w2, "Worker 2 should be selected after recovery");
    }
}
