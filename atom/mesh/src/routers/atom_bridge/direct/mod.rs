//! The EngineCores reached directly: Rust is their one client
//! (`atom_engine`), in the token bridge's place.
//!
//! Each DP rank is a worker, placed by mesh's own policies as CoreManager
//! would place it (`DpPlanner`); a request pinned to a rank
//! (`data_parallel_rank`) goes there. `DirectDispatcher` admits the request
//! (`request`) and submits it to its rank's Core; its sequences come back as
//! `TokenChunk`s (`token_source`).

mod request;
#[cfg(test)]
mod tests;
mod token_source;

use std::sync::Arc;

use async_trait::async_trait;
use tracing::debug;

use self::token_source::SeqTokenSource;
use super::RequestExtras;
use crate::{
    app_context::AppContext,
    atom_engine::{AtomEngine, BalanceSpec, BalanceStrategy, Submission},
    core::{
        placement::{
            traits::PdPlanner,
            types::{PlacementError, PlacementPlan, RequestDescriptor},
        },
        DPAwareWorkerBuilder, Worker,
    },
    policies::{
        DpLoad, DpStickyPolicy, LeastLoad, LeastLoadPolicy, LoadBalancingPolicy, RoundRobinPolicy,
        SelectWorkerInfo,
    },
    routers::{
        grpc::engine::Dispatcher,
        prepare::{generation_payload::GenerationPayload, lookup_tokenizer},
        token_handle::{engine_error::EngineError, token_handle::TokenHandle},
    },
};

/// The placement of a request its client pinned to a DP rank.
const PINNED: &str = "data_parallel_rank";

/// The policy CoreManager's balance settings name, over the ranks' `load`.
/// Session affinity keeps a session on one rank, whatever the strategy.
pub(super) fn policy_for(
    balance: &BalanceSpec,
    load: &Arc<DpLoad>,
) -> Arc<dyn LoadBalancingPolicy> {
    if balance.session_affinity {
        return Arc::new(DpStickyPolicy::new());
    }
    let least = |mode| Arc::new(LeastLoadPolicy::new(Arc::clone(load), mode));
    match balance.strategy {
        BalanceStrategy::RoundRobin => Arc::new(RoundRobinPolicy::new()),
        BalanceStrategy::LeastRequests => least(LeastLoad::Requests),
        BalanceStrategy::LeastTokens => least(LeastLoad::Tokens {
            req_equiv: balance.req_equiv,
        }),
    }
}

/// One worker per DP rank, in rank order, all serving `model`.
pub(super) fn rank_workers(ranks: usize, model: &str) -> Vec<Arc<dyn Worker>> {
    (0..ranks)
        .map(|rank| -> Arc<dyn Worker> {
            Arc::new(
                DPAwareWorkerBuilder::new("atom://local", rank, ranks)
                    .model_id(model)
                    .build(),
            )
        })
        .collect()
}

/// Places each request on a DP rank: the one it is pinned to, or the
/// policy's choice.
pub(super) struct DpPlanner {
    workers: Vec<Arc<dyn Worker>>,
    policy: Arc<dyn LoadBalancingPolicy>,
}

impl DpPlanner {
    pub(super) fn new(workers: Vec<Arc<dyn Worker>>, policy: Arc<dyn LoadBalancingPolicy>) -> Self {
        Self { workers, policy }
    }
}

#[async_trait]
impl PdPlanner for DpPlanner {
    async fn plan(&self, req: &RequestDescriptor<'_>) -> Result<PlacementPlan, PlacementError> {
        if let Some(rank) = RequestExtras::current().data_parallel_rank {
            // A rank out of range is the dispatcher's to refuse, which reads
            // the pin itself; any worker stands in for it here.
            let worker = self.workers.get(rank as usize).unwrap_or(&self.workers[0]);
            return Ok(PlacementPlan::Single {
                worker: Arc::clone(worker),
                policy_name: PINNED,
                prefill_tokens: 0,
            });
        }
        let info = SelectWorkerInfo {
            request_text: req.text,
            tokens: req.tokens,
            headers: req.headers,
            ..Default::default()
        };
        let index = self
            .policy
            .select_worker(&self.workers, &info)
            .await
            .ok_or(PlacementError::NoAvailableWorkers)?;
        Ok(PlacementPlan::Single {
            worker: Arc::clone(&self.workers[index]),
            policy_name: self.policy.name(),
            // The ranks' prompt tokens are DpLoad's.
            prefill_tokens: 0,
        })
    }
}

/// `Dispatcher` over the EngineCores.
pub(crate) struct DirectDispatcher {
    engine: Arc<AtomEngine>,
    load: Arc<DpLoad>,
    /// Where the served model's tokenizer is, to spell stop strings.
    app: Arc<AppContext>,
}

impl DirectDispatcher {
    pub(super) fn new(engine: Arc<AtomEngine>, load: Arc<DpLoad>, app: Arc<AppContext>) -> Self {
        Self { engine, load, app }
    }
}

/// The policies that charge `DpLoad` as they place.
fn charges_as_it_places(policy_name: &str) -> bool {
    matches!(policy_name, "least_requests" | "least_tokens")
}

#[async_trait]
impl Dispatcher for DirectDispatcher {
    async fn dispatch(
        &self,
        placement: &PlacementPlan,
        payload: &mut GenerationPayload,
    ) -> Result<TokenHandle, EngineError> {
        let PlacementPlan::Single {
            worker,
            policy_name,
            ..
        } = placement
        else {
            return Err(EngineError::RequestBuildFailed(
                "the ATOM engine serves no P/D placement".into(),
            ));
        };
        let extras = RequestExtras::current();
        let rank = match extras.data_parallel_rank {
            Some(rank) => rank as usize,
            None => worker.dp_rank().unwrap_or(0),
        };
        // Taken first, so the policy's charge goes back however this ends.
        let prompt_tokens = payload.token_ids.len() as u64;
        let charge = if charges_as_it_places(policy_name) {
            self.load.adopt(rank, prompt_tokens)
        } else {
            self.load.charge(rank, prompt_tokens)
        };

        let model = &self.engine.model;
        let tokenizer = lookup_tokenizer(&model.served_name, &self.app.tokenizer_registry)
            .map_err(|_| {
                EngineError::RequestBuildFailed(format!("no tokenizer for {}", model.served_name))
            })?;
        let admitted = request::admit(
            model,
            self.engine.client.max_pool_tokens(),
            payload,
            tokenizer.as_ref(),
            extras.tool_choice_required,
        )
        .map_err(EngineError::RequestBuildFailed)?;
        let seqs = self.engine.client.submit(Submission {
            request_id: &payload.request_id,
            prompt: &payload.token_ids,
            sampling: admitted.sampling,
            stop_token_sequences: admitted.stop_token_sequences,
            n: admitted.n,
            session: extras.dp_session_id.as_deref(),
            rank,
            stream: payload.stream,
        })?;
        debug!(request_id = %payload.request_id, rank, policy = policy_name, n = admitted.n,
            "Submitted to the EngineCore");
        Ok(TokenHandle::new(SeqTokenSource::new(
            seqs,
            payload.request_id.clone(),
            charge,
        )))
    }
}
