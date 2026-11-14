use std::sync::Arc;

use super::types::{PlacementError, RequestDescriptor};
use crate::core::{HashRing, Worker};
use crate::observability::{metrics::MeshMetrics, route_trace};
use crate::policies::{LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};

pub async fn apply_policy(
    candidates: &[Arc<dyn Worker>],
    policy: &dyn LoadBalancingPolicy,
    descriptor: &RequestDescriptor<'_>,
    hash_ring: Option<Arc<HashRing>>,
) -> Result<Arc<dyn Worker>, PlacementError> {
    if candidates.is_empty() {
        return Err(PlacementError::NoAvailableWorkers);
    }
    let info = SelectWorkerInfo {
        request_text: descriptor.text,
        tokens: descriptor.tokens,
        headers: descriptor.headers,
        hash_ring,
    };
    let (idx, decision) = policy.select_with_decision(candidates, &info).await;
    let selected = idx.map(|idx| candidates[idx].clone());
    // The planner passes the candidates of one pool.
    let pool = candidates[0].worker_type().as_metric_label();
    record_decision(pool, policy.name(), selected.as_ref(), decision);
    selected.ok_or(PlacementError::PolicyReturnedNone)
}

/// Records one selection. Runs before the router adds the request, so the
/// selected worker's load does not include it.
fn record_decision(
    pool: &'static str,
    policy: &'static str,
    selected: Option<&Arc<dyn Worker>>,
    decision: RouteDecision,
) {
    MeshMetrics::record_route_branch(pool, policy, decision.branch);
    if let Some(worker) = selected {
        MeshMetrics::record_route_selected_worker_active_requests(
            pool,
            policy,
            decision.branch,
            worker.load(),
        );
    }
    if let Some(ratio) = decision.match_ratio {
        MeshMetrics::record_route_prefix_match_ratio(pool, policy, ratio);
    }
    route_trace::record_selection(
        pool,
        policy,
        selected,
        decision.branch,
        decision.match_ratio,
    );
    route_trace::record_estimate(pool, selected, decision.estimated_hit_ratio);
}
