//! Round-robin load balancing policy

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use dashmap::DashMap;
use xxhash_rust::xxh3::Xxh3;

use super::{
    get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo,
    NO_HEALTHY_WORKERS,
};
use crate::core::Worker;

/// Least time between sweeps of the candidate sets.
const SWEEP_INTERVAL: Duration = Duration::from_secs(60);

/// A candidate set's position, and the sweep generation it was last used in.
#[derive(Debug)]
struct Position {
    next: AtomicUsize,
    used: AtomicU64,
}

/// Round-robin selection policy
///
/// Selects workers in sequential order, cycling through all healthy workers. Each set of
/// candidates keeps its own position, so callers offering different subsets (a pool's
/// models, ext-proc's subsets, prefill on the decode's DP rank) do not skip workers.
///
/// A new set starts at a random position. A new set seen [`SWEEP_INTERVAL`] or more after
/// the last sweep drops the sets unused since it, so every set in use is kept.
#[derive(Debug)]
pub struct RoundRobinPolicy {
    positions: DashMap<u64, Position>,
    generation: AtomicU64,
    created: Instant,
    swept_at_ms: AtomicU64,
    sweep_interval: Duration,
    sweeping: AtomicBool,
}

impl Default for RoundRobinPolicy {
    fn default() -> Self {
        Self::with_sweep_interval(SWEEP_INTERVAL)
    }
}

impl RoundRobinPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    fn with_sweep_interval(sweep_interval: Duration) -> Self {
        Self {
            positions: DashMap::new(),
            generation: AtomicU64::new(1),
            created: Instant::now(),
            swept_at_ms: AtomicU64::new(0),
            sweep_interval,
            sweeping: AtomicBool::new(false),
        }
    }

    /// Takes the next position of the set `workers` form.
    fn next_position(&self, workers: &[Arc<dyn Worker>]) -> usize {
        let mut hasher = Xxh3::new();
        for worker in workers {
            hasher.update(worker.url().as_bytes());
            hasher.update(&[0]);
        }
        let key = hasher.digest();
        if let Some(position) = self.positions.get(&key) {
            return self.take(&position);
        }
        self.sweep_if_due();
        let position = self.positions.entry(key).or_insert_with(|| Position {
            next: AtomicUsize::new(rand::random::<u32>() as usize),
            used: AtomicU64::new(0),
        });
        self.take(&position)
    }

    fn take(&self, position: &Position) -> usize {
        let generation = self.generation.load(Ordering::Relaxed);
        if position.used.load(Ordering::Relaxed) != generation {
            position.used.store(generation, Ordering::Relaxed);
        }
        position.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Drops the sets unused since the last sweep, once the sweep interval has passed.
    fn sweep_if_due(&self) {
        let now_ms = self.created.elapsed().as_millis() as u64;
        let due = || {
            let swept_at_ms = self.swept_at_ms.load(Ordering::Relaxed);
            now_ms >= swept_at_ms + self.sweep_interval.as_millis() as u64
        };
        if !due() || self.sweeping.swap(true, Ordering::Acquire) {
            return;
        }
        // Another thread may have swept between the check and the swap.
        if due() {
            self.swept_at_ms.store(now_ms, Ordering::Relaxed);
            let ended = self.generation.fetch_add(1, Ordering::Relaxed);
            self.positions
                .retain(|_, position| position.used.load(Ordering::Relaxed) >= ended);
        }
        self.sweeping.store(false, Ordering::Release);
    }
}

#[async_trait]
impl LoadBalancingPolicy for RoundRobinPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let healthy_indices = get_healthy_worker_indices(workers);

        if healthy_indices.is_empty() {
            return (None, RouteDecision::branch(NO_HEALTHY_WORKERS));
        }

        let selected_idx = match healthy_indices.len() {
            1 => 0,
            healthy => self.next_position(workers) % healthy,
        };

        (
            Some(healthy_indices[selected_idx]),
            RouteDecision::branch("round_robin"),
        )
    }

    fn name(&self) -> &'static str {
        "round_robin"
    }

    fn reset(&self) {
        self.positions.clear();
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

        assert!(follows(&picks(&policy, &workers, 5).await, &[0, 1, 2]));
    }

    fn regular_workers(urls: impl IntoIterator<Item = String>) -> Vec<Arc<dyn Worker>> {
        urls.into_iter()
            .map(|url| {
                let worker = BasicWorkerBuilder::new(url).worker_type(WorkerType::Regular);
                Arc::new(worker.build()) as Arc<dyn Worker>
            })
            .collect()
    }

    fn pair(name: &str) -> Vec<Arc<dyn Worker>> {
        regular_workers([
            format!("http://{name}1:8000"),
            format!("http://{name}2:8000"),
        ])
    }

    async fn picks(policy: &RoundRobinPolicy, workers: &[Arc<dyn Worker>], n: usize) -> Vec<usize> {
        let info = SelectWorkerInfo::default();
        let mut picks = Vec::new();
        for _ in 0..n {
            picks.push(policy.select_worker(workers, &info).await.unwrap());
        }
        picks
    }

    /// Whether `picks` walk `cycle` in order, from wherever they start.
    fn follows(picks: &[usize], cycle: &[usize]) -> bool {
        let Some(start) = cycle.iter().position(|&worker| worker == picks[0]) else {
            return false;
        };
        (picks.iter().enumerate()).all(|(i, &worker)| worker == cycle[(start + i) % cycle.len()])
    }

    #[tokio::test]
    async fn test_round_robin_keeps_a_position_per_candidate_set() {
        let policy = RoundRobinPolicy::new();
        let (a, b) = (pair("a"), pair("b"));

        // Taking turns, each set still visits both of its workers.
        let (mut from_a, mut from_b) = (Vec::new(), Vec::new());
        for _ in 0..2 {
            from_a.extend(picks(&policy, &a, 1).await);
            from_b.extend(picks(&policy, &b, 1).await);
        }
        assert!(follows(&from_a, &[0, 1]) && follows(&from_b, &[0, 1]));
    }

    #[tokio::test]
    async fn test_round_robin_drops_sets_unused_since_the_last_sweep() {
        let policy = RoundRobinPolicy::with_sweep_interval(Duration::ZERO);
        let kept = pair("kept");
        let mut from_kept = picks(&policy, &kept, 1).await;
        for i in 0..100 {
            picks(&policy, &pair(&format!("once{i}-")), 1).await;
            from_kept.extend(picks(&policy, &kept, 1).await);
            // The kept set, and the last two sets used once: each lasts one more sweep.
            assert!(policy.positions.len() <= 3, "{}", policy.positions.len());
        }
        assert!(follows(&from_kept, &[0, 1]), "{from_kept:?}");
    }

    #[tokio::test]
    async fn test_round_robin_keeps_every_set_used_within_the_interval() {
        let policy = RoundRobinPolicy::new();
        let sets: Vec<_> = (0..1000).map(|i| pair(&format!("s{i}-"))).collect();
        let mut from = vec![Vec::new(); sets.len()];
        for _ in 0..3 {
            for (set, from) in sets.iter().zip(&mut from) {
                from.extend(picks(&policy, set, 1).await);
            }
        }
        assert!(from.iter().all(|from| follows(from, &[0, 1])));
        assert_eq!(policy.positions.len(), sets.len());
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

        assert!(follows(&picks(&policy, &workers, 4).await, &[0, 2]));
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

        assert!(follows(&picks(&policy, &workers, 2).await, &[0, 1]));
        policy.reset();
        assert!(policy.positions.is_empty());
        assert!(follows(&picks(&policy, &workers, 2).await, &[0, 1]));
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
        assert!(
            policy.positions.is_empty(),
            "one candidate needs no position"
        );
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
