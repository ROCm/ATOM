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
    /// Cost of the selected candidate, in the policy's own unit.
    pub cost: Option<f64>,
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

/// Record one selection; `selected` indexes `workers`.
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
    let Some(worker) = selected
        .and_then(|index| workers.get(index))
        .or(workers.first())
    else {
        return;
    };
    let pool = worker.worker_type().as_metric_label();
    let name = policy.name();
    MeshMetrics::record_route_branch(pool, name, decision.branch);
    if let Some(ratio) = decision.match_ratio {
        MeshMetrics::record_route_prefix_match_ratio(pool, name, ratio);
    }
    if let Some(cost) = decision.cost {
        MeshMetrics::record_route_cost(pool, name, cost);
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
            let label = format!("policy=\"{name}\"");
            let selections: f64 = rendered
                .lines()
                .filter(|line| {
                    line.starts_with("mesh_route_branch_total{") && line.contains(&label)
                })
                .map(|line| line.rsplit_once(' ').unwrap().1.parse::<f64>().unwrap())
                .sum();
            assert_eq!(selections, 1.0, "{name}\n{rendered}");
        }
    }
}
