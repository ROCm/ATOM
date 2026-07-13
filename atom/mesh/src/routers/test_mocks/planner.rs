use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::core::placement::traits::PdPlanner;
use crate::core::placement::types::{PlacementError, PlacementPlan, RequestDescriptor};
use crate::core::worker::Worker;

pub(crate) struct MockPdPlanner {
    mode: PlannerMode,
}

enum PlannerMode {
    Scripted(Mutex<VecDeque<Result<PlacementPlan, PlacementError>>>),
    RepeatSingle {
        worker: Arc<dyn Worker>,
        policy_name: &'static str,
    },
    RepeatPair {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_policy: &'static str,
        decode_policy: &'static str,
    },
    RepeatErr(PlacementError),
}

impl MockPdPlanner {
    pub(crate) fn new(scripted: Vec<Result<PlacementPlan, PlacementError>>) -> Self {
        Self {
            mode: PlannerMode::Scripted(Mutex::new(scripted.into())),
        }
    }

    pub(crate) fn repeat_single(worker: Arc<dyn Worker>, policy_name: &'static str) -> Self {
        Self {
            mode: PlannerMode::RepeatSingle {
                worker,
                policy_name,
            },
        }
    }

    pub(crate) fn repeat_pair(
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_policy: &'static str,
        decode_policy: &'static str,
    ) -> Self {
        Self {
            mode: PlannerMode::RepeatPair {
                prefill,
                decode,
                prefill_policy,
                decode_policy,
            },
        }
    }

    pub(crate) fn repeat_err(err: PlacementError) -> Self {
        Self {
            mode: PlannerMode::RepeatErr(err),
        }
    }
}

#[async_trait]
impl PdPlanner for MockPdPlanner {
    async fn plan(&self, _req: &RequestDescriptor<'_>) -> Result<PlacementPlan, PlacementError> {
        match &self.mode {
            PlannerMode::Scripted(q) => q
                .lock()
                .unwrap()
                .pop_front()
                .expect("MockPdPlanner: scripted queue exhausted"),
            PlannerMode::RepeatSingle {
                worker,
                policy_name,
            } => Ok(PlacementPlan::Single {
                worker: worker.clone(),
                policy_name,
            }),
            PlannerMode::RepeatPair {
                prefill,
                decode,
                prefill_policy,
                decode_policy,
            } => Ok(PlacementPlan::Pair {
                prefill: prefill.clone(),
                decode: decode.clone(),
                prefill_policy,
                decode_policy,
            }),
            PlannerMode::RepeatErr(e) => Err(e.clone()),
        }
    }
}
