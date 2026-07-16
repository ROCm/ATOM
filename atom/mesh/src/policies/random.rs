//! Random load balancing policy

use std::sync::Arc;

use async_trait::async_trait;
use rand::Rng;

use super::{get_healthy_worker_indices, LoadBalancingPolicy, SelectWorkerInfo};
use crate::core::Worker;

/// Random selection policy
///
/// Selects workers randomly with uniform distribution among healthy workers.
#[derive(Debug, Default)]
pub struct RandomPolicy;

impl RandomPolicy {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LoadBalancingPolicy for RandomPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return None;
        }

        let mut rng = rand::rng();
        let random_idx = rng.random_range(0..healthy_indices.len());

        Some(healthy_indices[random_idx])
    }

    fn name(&self) -> &'static str {
        "random"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    #[tokio::test]
    async fn test_random_selection() {
        let policy = RandomPolicy::new();
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

        let mut counts = HashMap::new();
        for _ in 0..100 {
            if let Some(idx) = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await
            {
                *counts.entry(idx).or_insert(0) += 1;
            }
        }

        assert_eq!(counts.len(), 3);
        assert!(counts.values().all(|&count| count > 0));
    }

    #[tokio::test]
    async fn test_random_with_unhealthy_workers() {
        let policy = RandomPolicy::new();
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

        for _ in 0..10 {
            assert_eq!(
                policy
                    .select_worker(&workers, &SelectWorkerInfo::default())
                    .await,
                Some(1)
            );
        }
    }

    #[tokio::test]
    async fn test_random_no_healthy_workers() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        workers[0].set_healthy(false);
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            None
        );
    }

    #[tokio::test]
    async fn test_random_empty_workers() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![];
        assert_eq!(
            policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await,
            None
        );
    }

    #[tokio::test]
    async fn test_random_single_worker() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = vec![Arc::new(
            BasicWorkerBuilder::new("http://w1:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        )];

        for _ in 0..10 {
            assert_eq!(
                policy
                    .select_worker(&workers, &SelectWorkerInfo::default())
                    .await,
                Some(0)
            );
        }
    }

    #[test]
    fn test_random_name() {
        let policy = RandomPolicy::new();
        assert_eq!(policy.name(), "random");
    }

    #[test]
    fn test_random_as_any() {
        let policy = RandomPolicy::new();
        assert!(policy.as_any().downcast_ref::<RandomPolicy>().is_some());
    }

    #[test]
    fn test_random_default() {
        let policy = RandomPolicy;
        assert_eq!(policy.name(), "random");
    }

    #[tokio::test]
    async fn test_random_distribution_roughly_uniform() {
        let policy = RandomPolicy::new();
        let workers: Vec<Arc<dyn Worker>> = (0..5)
            .map(|i| {
                Arc::new(
                    BasicWorkerBuilder::new(format!("http://w{}:8000", i))
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();

        let mut counts = [0u32; 5];
        let n = 1000;
        for _ in 0..n {
            if let Some(idx) = policy
                .select_worker(&workers, &SelectWorkerInfo::default())
                .await
            {
                counts[idx] += 1;
            }
        }

        // Each worker should get at least 10% of requests (expect ~20% each)
        for (i, &count) in counts.iter().enumerate() {
            assert!(
                count > 100,
                "Worker {i} got only {count}/{n} selections, expected ~{}",
                n / 5
            );
        }
    }
}
