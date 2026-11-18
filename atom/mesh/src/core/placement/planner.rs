use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use super::policy_apply::{apply_policy, PlacedWorker};
use super::traits::{PdPlanner, PolicySource, WorkerSource};
use super::types::{PlacementError, PlacementPlan, Protocol, RequestDescriptor};
use crate::core::{ConnectionMode, HashRing, Worker, WorkerType};
use crate::observability::{
    metrics::{metrics_labels, MeshMetrics},
    route_trace,
};
use crate::policies::{LoadBalancingPolicy, NO_HEALTHY_WORKERS};

fn health_filter(pool: Vec<Arc<dyn Worker>>) -> Vec<Arc<dyn Worker>> {
    pool.into_iter().filter(|w| w.is_available()).collect()
}

/// Records a selection that has no candidate in `pool`. The policy never runs,
/// so it cannot report the failure itself.
fn report_no_candidates(pool: &'static str, policy: &dyn LoadBalancingPolicy) {
    MeshMetrics::record_route_branch(pool, policy.name(), NO_HEALTHY_WORKERS);
    route_trace::record_selection(pool, policy.name(), None, NO_HEALTHY_WORKERS, None);
}

/// The prefill candidates idx2idx keeps for `decode`: those on its DP rank, and those it
/// does not remap (no DP size, or the decode rank is outside it). All of them when none
/// is left, as the remap then keeps the selected prefill.
fn on_decode_rank(prefill: Vec<Arc<dyn Worker>>, decode: &dyn Worker) -> Vec<Arc<dyn Worker>> {
    let Some(rank) = decode.dp_rank() else {
        return prefill;
    };
    let mapped: Vec<_> = prefill
        .iter()
        .filter(|worker| match worker.dp_size() {
            Some(size) if rank < size => worker.dp_rank() == Some(rank),
            _ => true,
        })
        .cloned()
        .collect();
    if mapped.is_empty() {
        prefill
    } else {
        mapped
    }
}

/// Prompt tokens charged to `placed`, which prefills `req`: the policy's estimate, or
/// the whole prompt.
fn prefill_tokens(placed: &PlacedWorker, req: &RequestDescriptor<'_>) -> usize {
    placed
        .prefill_tokens
        .unwrap_or_else(|| req.tokens.map_or(0, <[u32]>::len))
}

pub struct DefaultPlanner {
    workers: Arc<dyn WorkerSource>,
    policies: Arc<dyn PolicySource>,
    prefill_on_decode_rank: bool,
}

impl DefaultPlanner {
    pub fn new(workers: Arc<dyn WorkerSource>, policies: Arc<dyn PolicySource>) -> Self {
        Self {
            workers,
            policies,
            prefill_on_decode_rank: false,
        }
    }

    /// Under ATOM's idx2idx rank mapping, select decode first and offer the prefill policy
    /// the candidates [`on_decode_rank`] keeps for it, so the prefill it selects is the one
    /// the request runs on.
    pub fn with_prefill_on_decode_rank(mut self, enabled: bool) -> Self {
        self.prefill_on_decode_rank = enabled;
        self
    }

    fn connection_mode(&self, req: &RequestDescriptor<'_>) -> Option<ConnectionMode> {
        req.protocol.map(|p| match p {
            Protocol::Http => ConnectionMode::Http,
            Protocol::Grpc => ConnectionMode::Grpc { port: None },
        })
    }

    fn hash_ring_for(&self, model_id: Option<&str>) -> Option<Arc<HashRing>> {
        model_id.and_then(|m| self.workers.hash_ring(m))
    }

    async fn plan_single(
        &self,
        req: &RequestDescriptor<'_>,
        regular_pool_raw: Vec<Arc<dyn Worker>>,
    ) -> Result<PlacementPlan, PlacementError> {
        let candidates = health_filter(regular_pool_raw);
        let policy = self.policies.regular_policy(req.model_id);
        if candidates.is_empty() {
            report_no_candidates(metrics_labels::WORKER_REGULAR, policy.as_ref());
            return Err(PlacementError::NoAvailableWorkers);
        }

        let hash_ring = self.hash_ring_for(req.model_id);
        let chosen = apply_policy(&candidates, policy.as_ref(), req, hash_ring).await?;

        Ok(PlacementPlan::Single {
            policy_name: policy.name(),
            // The router releases the charge at the response's first frame, which a
            // non-streaming response sends only after decoding.
            prefill_tokens: if req.stream {
                prefill_tokens(&chosen, req)
            } else {
                0
            },
            worker: chosen.worker,
        })
    }

    async fn plan_pair(
        &self,
        req: &RequestDescriptor<'_>,
        prefill_pool_raw: Vec<Arc<dyn Worker>>,
        decode_pool_raw: Vec<Arc<dyn Worker>>,
    ) -> Result<PlacementPlan, PlacementError> {
        let prefill_policy = self.policies.prefill_policy();
        let decode_policy = self.policies.decode_policy();
        let prefill_candidates = health_filter(prefill_pool_raw);
        if prefill_candidates.is_empty() {
            report_no_candidates(metrics_labels::WORKER_PREFILL, prefill_policy.as_ref());
            return Err(PlacementError::NoAvailableWorkers);
        }
        let decode_candidates = health_filter(decode_pool_raw);
        if decode_candidates.is_empty() {
            report_no_candidates(metrics_labels::WORKER_DECODE, decode_policy.as_ref());
            return Err(PlacementError::NoAvailableWorkers);
        }

        let hash_ring = self.hash_ring_for(req.model_id);

        let (prefill, decode) = if self.prefill_on_decode_rank {
            let decode = apply_policy(
                &decode_candidates,
                decode_policy.as_ref(),
                req,
                hash_ring.clone(),
            )
            .await?;
            let prefill_candidates = on_decode_rank(prefill_candidates, decode.worker.as_ref());
            let prefill =
                apply_policy(&prefill_candidates, prefill_policy.as_ref(), req, hash_ring).await?;
            (prefill, decode)
        } else {
            let prefill = apply_policy(
                &prefill_candidates,
                prefill_policy.as_ref(),
                req,
                hash_ring.clone(),
            )
            .await?;
            let decode =
                apply_policy(&decode_candidates, decode_policy.as_ref(), req, hash_ring).await?;
            (prefill, decode)
        };
        debug!(
            stage = "placement.pair.selected",
            model_id = req.model_id.unwrap_or(""),
            prefill_url = prefill.worker.url(),
            decode_url = decode.worker.url(),
            prefill_policy = prefill_policy.name(),
            decode_policy = decode_policy.name(),
            "prefill and decode workers selected"
        );

        Ok(PlacementPlan::Pair {
            prefill_tokens: prefill_tokens(&prefill, req),
            prefill_load: prefill.load_reservation,
            decode_load: decode.load_reservation,
            prefill: prefill.worker,
            decode: decode.worker,
            prefill_policy: prefill_policy.name(),
            decode_policy: decode_policy.name(),
        })
    }
}

#[async_trait]
impl PdPlanner for DefaultPlanner {
    async fn plan(&self, req: &RequestDescriptor<'_>) -> Result<PlacementPlan, PlacementError> {
        let conn_mode = self.connection_mode(req);

        if let Some(m) = req.model_id {
            let any_for_model = self
                .workers
                .workers_filtered(Some(m), None, conn_mode.clone());
            if any_for_model.is_empty() {
                return Err(PlacementError::ModelNotFound {
                    model_id: m.to_string(),
                });
            }
        } else {
            let any = self.workers.workers_filtered(None, None, conn_mode.clone());
            if any.is_empty() {
                return Err(PlacementError::NoWorkers);
            }
        }

        let regular_pool = self.workers.workers_filtered(
            req.model_id,
            Some(WorkerType::Regular),
            conn_mode.clone(),
        );
        let prefill_pool = self.workers.workers_filtered(
            req.model_id,
            Some(WorkerType::Prefill {
                bootstrap_port: None,
            }),
            conn_mode.clone(),
        );
        let decode_pool = self.workers.workers_filtered(
            req.model_id,
            Some(WorkerType::Decode),
            conn_mode.clone(),
        );

        if !regular_pool.is_empty() {
            self.plan_single(req, regular_pool).await
        } else if !prefill_pool.is_empty() && !decode_pool.is_empty() {
            self.plan_pair(req, prefill_pool, decode_pool).await
        } else if prefill_pool.is_empty() && !decode_pool.is_empty() {
            report_no_candidates(
                metrics_labels::WORKER_PREFILL,
                self.policies.prefill_policy().as_ref(),
            );
            Err(PlacementError::NoPrefillWorkers)
        } else if !prefill_pool.is_empty() && decode_pool.is_empty() {
            report_no_candidates(
                metrics_labels::WORKER_DECODE,
                self.policies.decode_policy().as_ref(),
            );
            Err(PlacementError::NoDecodeWorkers)
        } else {
            Err(PlacementError::NoAvailableWorkers)
        }
    }
}
