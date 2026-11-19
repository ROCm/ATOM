//! Data-parallel worker routing with session affinity.
//!
//! The session table lives in [`super::session_affinity`]. A new session goes
//! to the healthy worker with the fewest live sessions (load breaks ties), and
//! a request without a session key uses minimum load.

use std::sync::Arc;

use async_trait::async_trait;

use super::session_affinity::SessionAffinityPolicy;
use super::{LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};
use crate::core::Worker;

#[derive(Debug)]
pub struct DpStickyPolicy {
    inner: SessionAffinityPolicy,
}

impl DpStickyPolicy {
    pub fn new() -> Self {
        Self {
            inner: SessionAffinityPolicy::dp_sticky(),
        }
    }

    #[cfg(test)]
    fn set_last_access(&self, key: &str, when: std::time::Instant) {
        self.inner.set_last_access(key, when);
    }
}

impl Default for DpStickyPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for DpStickyPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        self.inner.select_with_decision(workers, info).await
    }

    fn on_dispatch(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
    ) {
        self.inner.on_dispatch(pool, headers, worker);
    }

    fn on_dispatch_failed(
        &self,
        pool: &'static str,
        headers: Option<&http::HeaderMap>,
        worker: &Arc<dyn Worker>,
    ) {
        self.inner.on_dispatch_failed(pool, headers, worker);
    }

    fn name(&self) -> &'static str {
        "dp_sticky"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::{
        core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType},
        policies::NO_HEALTHY_WORKERS,
    };

    fn workers() -> Vec<Arc<dyn Worker>> {
        ["http://worker-1:8000", "http://worker-2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect()
    }

    fn dp_workers() -> Vec<Arc<dyn Worker>> {
        (0..2)
            .map(|rank| {
                Arc::new(DPAwareWorkerBuilder::new("http://worker:8000", rank, 2).build())
                    as Arc<dyn Worker>
            })
            .collect()
    }

    fn headers(session_id: &str) -> http::HeaderMap {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-session-id", session_id.parse().unwrap());
        headers
    }

    fn expire(policy: &DpStickyPolicy, session: &str) {
        policy.set_last_access(
            session,
            Instant::now() - Duration::from_secs(120 * 60) - Duration::from_secs(1),
        );
    }

    #[tokio::test]
    async fn session_id_is_sticky_while_worker_is_healthy() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let selected = policy.select_worker(&workers, &info).await.unwrap();
        for _ in 0..10 {
            assert_eq!(policy.select_worker(&workers, &info).await, Some(selected));
        }
    }

    #[tokio::test]
    async fn session_id_stays_on_the_same_dp_rank() {
        let policy = DpStickyPolicy::new();
        let workers = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        workers[0].increment_load();
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
    }

    #[tokio::test]
    async fn new_sessions_are_spread_by_assignment_count() {
        let policy = DpStickyPolicy::new();
        let workers = dp_workers();

        for (session_id, expected_worker) in [
            ("session-1", 0),
            ("session-2", 1),
            ("session-3", 0),
            ("session-4", 1),
        ] {
            let headers = headers(session_id);
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                ..Default::default()
            };
            assert_eq!(
                policy.select_worker(&workers, &info).await,
                Some(expected_worker)
            );
        }
    }

    #[tokio::test]
    async fn unhealthy_assignment_is_replaced() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        let initial = policy.select_worker(&workers, &info).await.unwrap();
        workers[initial].set_healthy(false);

        let replacement = policy.select_worker(&workers, &info).await.unwrap();
        assert_ne!(replacement, initial);
        assert_eq!(
            policy.select_worker(&workers, &info).await,
            Some(replacement)
        );
    }

    #[tokio::test]
    async fn expired_session_assignment_is_rebalanced_to_lowest_load_worker() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        workers[0].increment_load();
        workers[0].increment_load();
        expire(&policy, "session-1");

        assert_eq!(policy.select_worker(&workers, &info).await, Some(1));
    }

    #[tokio::test]
    async fn session_outcomes_are_reported_as_branches() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let session = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let no_session = SelectWorkerInfo::default();

        let (first, decision) = policy.select_with_decision(&workers, &session).await;
        let mut branches = vec![decision.branch];
        branches.push(branch(&policy, &workers, &session).await);
        workers[first.unwrap()].set_healthy(false);
        branches.push(branch(&policy, &workers, &session).await);
        expire(&policy, "session-1");
        branches.push(branch(&policy, &workers, &session).await);
        branches.push(branch(&policy, &workers, &no_session).await);
        workers.iter().for_each(|worker| worker.set_healthy(false));
        branches.push(branch(&policy, &workers, &session).await);

        assert_eq!(
            branches,
            [
                "new",
                "hit",
                "rebind",
                "expired",
                "no_session",
                NO_HEALTHY_WORKERS
            ]
        );
    }

    async fn branch(
        policy: &DpStickyPolicy,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> &'static str {
        policy.select_with_decision(workers, info).await.1.branch
    }

    /// Selects a worker for `session` and dispatches the request to it.
    async fn route(
        policy: &DpStickyPolicy,
        workers: &[Arc<dyn Worker>],
        pool: &'static str,
        session: &str,
    ) -> usize {
        let headers = headers(session);
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        let selected = policy.select_worker(workers, &info).await.unwrap();
        policy.on_dispatch(pool, Some(&headers), &workers[selected]);
        selected
    }

    fn worker_changes(rendered: &str) -> Vec<&str> {
        let mut changes: Vec<_> = rendered
            .lines()
            .filter(|line| line.starts_with("mesh_session_worker_change_total{"))
            .collect();
        changes.sort_unstable();
        changes
    }

    #[tokio::test]
    async fn worker_changes_count_only_a_different_dispatched_worker() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::new();
        let workers = workers();

        // New, then hit: no change.
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        // Worker 0 is unavailable: rebind to worker 1.
        workers[0].set_healthy(false);
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 1);
        workers[0].set_healthy(true);
        // Expired: reassigned to worker 0, a different worker.
        expire(&policy, "session-1");
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        // Expired again: reassigned to worker 0, the same worker.
        expire(&policy, "session-1");
        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);

        // Ranks behind one URL are different workers.
        let ranks = dp_workers();
        assert_eq!(route(&policy, &ranks, "decode", "session-2").await, 0);
        ranks[0].set_healthy(false);
        assert_eq!(route(&policy, &ranks, "decode", "session-2").await, 1);
        assert_eq!(ranks[0].base_url(), ranks[1].base_url());

        assert_eq!(
            worker_changes(&handle.render()),
            [
                r#"mesh_session_worker_change_total{pool="decode",reason="rebind"} 1"#,
                r#"mesh_session_worker_change_total{pool="regular",reason="expired"} 1"#,
                r#"mesh_session_worker_change_total{pool="regular",reason="rebind"} 1"#,
            ]
        );
    }

    #[tokio::test]
    async fn worker_changes_follow_the_dispatched_rank() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::new();
        let ranks = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // The pin stays on rank 0 while rank mapping dispatches to rank 1, 1, then 0.
        for dispatched in [1, 1, 0] {
            assert_eq!(policy.select_worker(&ranks, &info).await, Some(0));
            policy.on_dispatch("prefill", Some(&headers), &ranks[dispatched]);
        }

        assert_eq!(
            worker_changes(&handle.render()),
            [r#"mesh_session_worker_change_total{pool="prefill",reason="rank_mapping"} 1"#]
        );
    }

    #[tokio::test]
    async fn dispatch_failure_places_the_retry_again() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let policy = DpStickyPolicy::new();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        policy.on_dispatch_failed("regular", Some(&headers), &workers[0]);
        // Worker 0 still counts the session, so fewest-session placement moves it.
        let (retry, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((retry, decision.branch), (Some(1), "rebind"));
        policy.on_dispatch("regular", Some(&headers), &workers[1]);
        assert_eq!(branch(&policy, &workers, &info).await, "hit");

        assert_eq!(
            worker_changes(&handle.render()),
            [r#"mesh_session_worker_change_total{pool="regular",reason="rebind"} 1"#]
        );
    }

    #[tokio::test]
    async fn missing_session_id_uses_load_balancing_fallback() {
        let policy = DpStickyPolicy::new();
        let workers = workers();
        workers[0].increment_load();
        workers[0].increment_load();

        let selected = policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await;
        assert_eq!(selected, Some(1));
    }
}
