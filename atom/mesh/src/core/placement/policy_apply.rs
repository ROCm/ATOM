use std::sync::Arc;

use super::types::{PlacementError, RequestDescriptor};
use crate::core::{HashRing, Worker};
use crate::policies::{LoadBalancingPolicy, SelectWorkerInfo};

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
    let idx = policy
        .select_worker(candidates, &info)
        .await
        .ok_or(PlacementError::PolicyReturnedNone)?;
    Ok(candidates[idx].clone())
}
