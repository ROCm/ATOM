use std::sync::Arc;

use super::types::{PlacementError, RequestDescriptor};
use crate::core::{HashRing, Worker};
use crate::policies::{LoadBalancingPolicy, SelectWorkerInfo};

/// A worker chosen by a policy and the prompt tokens it must prefill.
#[derive(Debug)]
pub struct Selection {
    pub worker: Arc<dyn Worker>,
    pub prefill_tokens: usize,
}

pub async fn apply_policy(
    candidates: &[Arc<dyn Worker>],
    policy: &dyn LoadBalancingPolicy,
    descriptor: &RequestDescriptor<'_>,
    hash_ring: Option<Arc<HashRing>>,
) -> Result<Selection, PlacementError> {
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
    Ok(Selection {
        worker: candidates[idx].clone(),
        prefill_tokens: descriptor.tokens.map_or(0, <[u32]>::len),
    })
}
