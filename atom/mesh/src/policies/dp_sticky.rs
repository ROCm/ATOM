//! Data-parallel worker routing with session affinity.
//!
//! `dp_sticky` is the session table in [`super::session_affinity`] around
//! [`LeastSessionsPolicy`]. Requests carrying `X-Session-ID` are pinned to the
//! worker picked for the session's first request: the healthy worker holding
//! the fewest live sessions (load breaks ties). Requests without a session ID
//! use minimum-load balancing. A stale mapping is replaced when its worker is
//! no longer healthy or a candidate (another DP rank under idx2idx), the
//! session has been idle for too long, or a request of the session failed with
//! a server error or got no response.

use std::sync::Arc;

use async_trait::async_trait;

use super::{
    get_healthy_worker_indices, LoadBalancingPolicy, RouteDecision, SelectWorkerInfo,
    NO_HEALTHY_WORKERS,
};
use crate::core::Worker;

const NO_SESSION: &str = "no_session";
const LEAST_SESSIONS: &str = "least_sessions";

/// `dp_sticky`'s placement: the healthy worker with the fewest live sessions,
/// then the lowest load. Without session counts, which the session table
/// passes only when it places a session, the lowest load.
#[derive(Debug, Default)]
pub(crate) struct LeastSessionsPolicy;

#[async_trait]
impl LoadBalancingPolicy for LeastSessionsPolicy {
    async fn select_with_decision(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> (Option<usize>, RouteDecision) {
        let counts = info.session_counts;
        let sessions = |index: usize| counts.and_then(|counts| counts.get(index).copied());
        let selected = get_healthy_worker_indices(workers)
            .into_iter()
            .min_by_key(|&index| (sessions(index).unwrap_or(0), workers[index].load()));
        let branch = match (selected, counts) {
            (None, _) => NO_HEALTHY_WORKERS,
            (Some(_), None) => NO_SESSION,
            (Some(_), Some(_)) => LEAST_SESSIONS,
        };
        (selected, RouteDecision::branch(branch))
    }

    /// The session table reports `new`, `rebind` or `expired` instead.
    fn reports_session_result(&self) -> bool {
        true
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
        policies::SessionAffinityPolicy,
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

    fn expire(policy: &SessionAffinityPolicy, session: &str) {
        policy.set_last_access(
            session,
            Instant::now() - Duration::from_secs(120 * 60) - Duration::from_secs(1),
        );
    }

    #[tokio::test]
    async fn session_id_is_sticky_while_worker_is_healthy() {
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        policy: &SessionAffinityPolicy,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> &'static str {
        policy.select_with_decision(workers, info).await.1.branch
    }

    /// Selects a worker for `session` and dispatches the request to it.
    async fn route(
        policy: &SessionAffinityPolicy,
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
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
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        // The retry avoids worker 0, where it failed.
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
    async fn a_retry_avoids_the_worker_it_failed_on() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        assert_eq!(route(&policy, &workers, "regular", "s1").await, 0);
        assert_eq!(route(&policy, &workers, "regular", "s2").await, 1);
        assert_eq!(route(&policy, &workers, "regular", "s3").await, 0);
        let s2 = headers("s2");
        let info = SelectWorkerInfo {
            headers: Some(&s2),
            ..Default::default()
        };

        // Worker 1 holds fewer sessions, but s2 failed there.
        policy.on_dispatch_failed(Some(&s2), &workers[1]);
        let (retry, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((retry, decision.branch), (Some(0), "rebind"));

        // With no other healthy worker, the retry stays where it failed.
        let only = &workers[1..];
        assert_eq!(route(&policy, only, "regular", "s4").await, 0);
        let s4 = headers("s4");
        policy.on_dispatch_failed(Some(&s4), &only[0]);
        let info = SelectWorkerInfo {
            headers: Some(&s4),
            ..Default::default()
        };
        let (retry, decision) = policy.select_with_decision(only, &info).await;
        assert_eq!((retry, decision.branch), (Some(0), "rebind"));
    }

    #[tokio::test]
    async fn retries_avoid_every_worker_the_session_failed_on_until_a_hit() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers: Vec<Arc<dyn Worker>> = ["http://a:8000", "http://b:8000", "http://c:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        // Worker c holds two other sessions, so a and b are lighter.
        for other in ["other-1", "other-2"] {
            route(&policy, &workers[2..], "regular", other).await;
        }
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
        let s = headers("s");

        // a fails, then b: the next retry goes to c, not back to a.
        policy.on_dispatch_failed(Some(&s), &workers[0]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 1);
        policy.on_dispatch_failed(Some(&s), &workers[1]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 2);

        // A hit on c clears the set: a failure on c then avoids c alone.
        assert_eq!(route(&policy, &workers, "regular", "s").await, 2);
        policy.on_dispatch_failed(Some(&s), &workers[2]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
    }

    #[tokio::test]
    async fn a_late_failure_on_the_previous_worker_keeps_the_new_binding() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        assert_eq!(route(&policy, &workers, "regular", "session-1").await, 0);
        workers[0].set_healthy(false);
        assert_eq!(branch(&policy, &workers, &info).await, "rebind");
        // A request still running on worker 0 fails, before and after the
        // rebound session's next request is sent to worker 1.
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        policy.on_dispatch("regular", Some(&headers), &workers[1]);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        workers[0].set_healthy(true);
        let (selected, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((selected, decision.branch), (Some(1), "hit"));
    }

    #[tokio::test]
    async fn a_failure_off_the_bound_rank_keeps_the_binding() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let ranks = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // The pin is rank 0; the request is sent to rank 1 and fails there.
        // Since the planner selects prefill on the decode rank, this happens only
        // when health changed after planning, and the next request rebinds then.
        assert_eq!(policy.select_worker(&ranks, &info).await, Some(0));
        policy.on_dispatch("prefill", Some(&headers), &ranks[1]);
        policy.on_dispatch_failed(Some(&headers), &ranks[1]);
        assert_eq!(branch(&policy, &ranks, &info).await, "hit");
    }

    #[tokio::test]
    async fn a_hit_clears_the_workers_the_session_failed_on() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers: Vec<Arc<dyn Worker>> = ["http://a:8000", "http://b:8000", "http://c:8000"]
            .into_iter()
            .map(|url| Arc::new(BasicWorkerBuilder::new(url).build()) as Arc<dyn Worker>)
            .collect();
        // Worker c holds two other sessions, so a and b are lighter.
        for other in ["other-1", "other-2"] {
            route(&policy, &workers[2..], "regular", other).await;
        }
        let s = headers("s");
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
        policy.on_dispatch_failed(Some(&s), &workers[0]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 1);

        // A hit on b clears a from the set: a failure on b goes back to a,
        // the lighter of a and c.
        assert_eq!(route(&policy, &workers, "regular", "s").await, 1);
        policy.on_dispatch_failed(Some(&s), &workers[1]);
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
    }

    #[tokio::test]
    async fn the_failed_workers_survive_a_selection_with_no_healthy_worker() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        let headers = headers("s");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };
        assert_eq!(route(&policy, &workers, "regular", "s").await, 0);
        // Worker 1 holds another session, so the tie would go back to worker 0.
        assert_eq!(route(&policy, &workers, "regular", "other").await, 1);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);

        workers.iter().for_each(|worker| worker.set_healthy(false));
        assert_eq!(policy.select_worker(&workers, &info).await, None);
        workers.iter().for_each(|worker| worker.set_healthy(true));
        let (selected, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((selected, decision.branch), (Some(1), "rebind"));
    }

    #[tokio::test]
    async fn a_request_placed_before_a_rebind_keeps_the_new_binding() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // A PD relay request reserves decode worker 0; its decode is sent
        // after its prefill.
        assert_eq!(policy.select_worker(&workers, &info).await, Some(0));
        // Meanwhile another request of the session fails on worker 0, and the
        // next one is placed on worker 1 and sent there.
        assert_eq!(route(&policy, &workers, "decode", "session-1").await, 0);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);
        assert_eq!(route(&policy, &workers, "decode", "session-1").await, 1);
        // The first request's decode is now sent to worker 0 and fails.
        policy.on_dispatch("decode", Some(&headers), &workers[0]);
        policy.on_dispatch_failed(Some(&headers), &workers[0]);

        let (selected, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((selected, decision.branch), (Some(1), "hit"));
    }

    /// The cost the worker check accepts: a session placed on worker 0, then 1,
    /// then 0 again releases its new binding when a request from its first
    /// placement fails late on worker 0. The session is placed once more.
    #[tokio::test]
    async fn an_older_failure_on_the_worker_the_session_returned_to_releases_it() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // The first placement's request is still running on worker 0.
        assert_eq!(route(&policy, &workers, "decode", "session-1").await, 0);
        workers[0].set_healthy(false);
        assert_eq!(route(&policy, &workers, "decode", "session-1").await, 1);
        workers[0].set_healthy(true);
        workers[1].set_healthy(false);
        assert_eq!(route(&policy, &workers, "decode", "session-1").await, 0);
        workers[1].set_healthy(true);
        // The old request fails on worker 0, where the session is bound again.
        policy.on_dispatch_failed(Some(&headers), &workers[0]);

        let (selected, decision) = policy.select_with_decision(&workers, &info).await;
        assert_eq!((selected, decision.branch), (Some(1), "rebind"));
    }

    #[tokio::test]
    async fn a_stale_failure_on_the_rank_the_new_binding_maps_to_keeps_it() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let ranks = dp_workers();
        let headers = headers("session-1");
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            ..Default::default()
        };

        // A request placed on rank 0 is still running there when the
        // session fails on rank 0 and is placed again on rank 1.
        assert_eq!(policy.select_worker(&ranks, &info).await, Some(0));
        policy.on_dispatch("prefill", Some(&headers), &ranks[0]);
        policy.on_dispatch_failed(Some(&headers), &ranks[0]);
        assert_eq!(policy.select_worker(&ranks, &info).await, Some(1));
        // Rank mapping sends the current request to rank 0, then the stale
        // request fails there.
        policy.on_dispatch("prefill", Some(&headers), &ranks[0]);
        policy.on_dispatch_failed(Some(&headers), &ranks[0]);

        let (selected, decision) = policy.select_with_decision(&ranks, &info).await;
        assert_eq!((selected, decision.branch), (Some(1), "hit"));
    }

    #[tokio::test]
    async fn missing_session_id_uses_load_balancing_fallback() {
        let policy = SessionAffinityPolicy::dp_sticky();
        let workers = workers();
        workers[0].increment_load();
        workers[0].increment_load();

        let selected = policy
            .select_worker(&workers, &SelectWorkerInfo::default())
            .await;
        assert_eq!(selected, Some(1));
    }
}
