//! Decision reporting shared by load balancing policies.
//!
//! A policy calls [`report`] once per selection with one of the branches it
//! declares in [`LoadBalancingPolicy::decision_branches`], so the label set of
//! `mesh_route_branch_total` stays bounded. New policies declare their
//! branches and report the same way.

use std::sync::Arc;

use super::LoadBalancingPolicy;
use crate::{core::Worker, observability::metrics::MeshMetrics};

/// Branch for selections that found no healthy candidate.
pub(crate) const NO_HEALTHY_WORKERS: &str = "no_healthy_workers";

/// What a policy decided for one selection.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RouteDecision {
    pub branch: &'static str,
    /// Prefix match ratio of the best candidate, for policies that estimate one.
    pub match_ratio: Option<f64>,
    /// Session affinity result: hit, new, rebind or expired.
    pub affinity: Option<&'static str>,
}

impl RouteDecision {
    pub fn branch(branch: &'static str) -> Self {
        Self {
            branch,
            ..Self::default()
        }
    }
}

/// Record one selection; `selected` indexes `workers`. Call it before the
/// router adds the request to the selected worker's load.
pub(crate) fn report<P: LoadBalancingPolicy + ?Sized>(
    policy: &P,
    workers: &[Arc<dyn Worker>],
    selected: Option<usize>,
    decision: RouteDecision,
) {
    debug_assert!(
        policy.decision_branches().contains(&decision.branch),
        "{} reported undeclared branch {}",
        policy.name(),
        decision.branch
    );
    let selected = selected.and_then(|index| workers.get(index));
    // An empty candidate list has no pool to label; the planner counts it.
    let Some(worker) = selected.or(workers.first()) else {
        return;
    };
    let pool = worker.worker_type().as_metric_label();
    let name = policy.name();
    MeshMetrics::record_route_branch(pool, name, decision.branch);
    if let Some(selected) = selected {
        MeshMetrics::record_route_selected_worker_active_requests(
            pool,
            name,
            decision.branch,
            selected.load(),
        );
    }
    if let Some(ratio) = decision.match_ratio {
        MeshMetrics::record_route_prefix_match_ratio(pool, name, ratio);
    }
    if let Some(result) = decision.affinity {
        MeshMetrics::record_session_affinity(pool, result);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::{
        core::{BasicWorkerBuilder, WorkerType},
        policies::{PolicyFactory, SelectWorkerInfo},
    };

    const POLICIES: [&str; 6] = [
        "random",
        "round_robin",
        "dp_sticky",
        "power_of_two",
        "cache_aware",
        "prefix_hash",
    ];

    #[tokio::test]
    async fn every_policy_reports_one_declared_branch_per_selection() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .worker_type(WorkerType::Regular)
                        .build(),
                ) as Arc<dyn Worker>
            })
            .collect();
        let tokens = [1, 2, 3];
        let info = SelectWorkerInfo {
            request_text: Some("hello"),
            tokens: Some(&tokens),
            ..Default::default()
        };

        for name in POLICIES {
            let policy = PolicyFactory::create_by_name(name).unwrap();
            let branches = policy.decision_branches();
            assert!(branches.contains(&NO_HEALTHY_WORKERS), "{name}");
            assert_eq!(
                branches.iter().collect::<HashSet<_>>().len(),
                branches.len(),
                "{name} declares a branch twice"
            );
            assert!(policy.select_worker(&workers, &info).await.is_some());
        }

        let rendered = handle.render();
        for name in POLICIES {
            let policy = format!("policy=\"{name}\"");
            for metric in [
                "mesh_route_branch_total{",
                "mesh_route_selected_worker_active_requests_count{",
            ] {
                assert_eq!(
                    total(&rendered, metric, &[&policy]),
                    1.0,
                    "{name} {metric}\n{rendered}"
                );
            }
        }
    }

    #[tokio::test]
    async fn a_selection_without_healthy_candidates_counts_once_in_its_pool() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let workers: Vec<Arc<dyn Worker>> = ["http://w1:8000", "http://w2:8000"]
            .into_iter()
            .map(|url| {
                let worker = BasicWorkerBuilder::new(url)
                    .worker_type(WorkerType::Prefill {
                        bootstrap_port: None,
                    })
                    .build();
                worker.set_healthy(false);
                Arc::new(worker) as Arc<dyn Worker>
            })
            .collect();
        let tokens = [1, 2, 3];
        let info = SelectWorkerInfo {
            request_text: Some("hello"),
            tokens: Some(&tokens),
            ..Default::default()
        };

        for name in POLICIES {
            let policy = PolicyFactory::create_by_name(name).unwrap();
            assert_eq!(policy.select_worker(&workers, &info).await, None, "{name}");
        }

        let rendered = handle.render();
        for name in POLICIES {
            let policy = format!("policy=\"{name}\"");
            let failure = [
                r#"pool="prefill""#,
                policy.as_str(),
                r#"branch="no_healthy_workers""#,
            ];
            let branches = "mesh_route_branch_total{";
            assert_eq!(
                total(&rendered, branches, &failure),
                1.0,
                "{name}\n{rendered}"
            );
            assert_eq!(
                total(&rendered, branches, &[&policy]),
                1.0,
                "{name}\n{rendered}"
            );
            let samples = "mesh_route_selected_worker_active_requests_count{";
            assert_eq!(
                total(&rendered, samples, &[&policy]),
                0.0,
                "{name}\n{rendered}"
            );
        }
    }

    /// Sum of the samples of `metric` whose labels include every one of `labels`.
    fn total(rendered: &str, metric: &str, labels: &[&str]) -> f64 {
        rendered
            .lines()
            .filter(|line| {
                line.starts_with(metric) && labels.iter().all(|label| line.contains(label))
            })
            .map(|line| line.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
            .sum()
    }
}
