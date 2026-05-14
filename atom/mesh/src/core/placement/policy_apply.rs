use std::sync::Arc;

use super::types::{PlacementError, RequestDescriptor};
use crate::core::{HashRing, Worker};
use crate::policies::LoadBalancingPolicy;

pub async fn apply_policy(
    candidates: &[Arc<dyn Worker>],
    policy: &dyn LoadBalancingPolicy,
    descriptor: &RequestDescriptor<'_>,
    hash_ring: Option<Arc<HashRing>>,
) -> Result<Arc<dyn Worker>, PlacementError> {
    let _ = (candidates, policy, descriptor, hash_ring);
    todo!()
}
