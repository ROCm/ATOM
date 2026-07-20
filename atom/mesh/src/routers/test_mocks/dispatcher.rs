use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::core::placement::PlacementPlan;
use crate::routers::grpc::engine::Dispatcher;
use crate::routers::prepare::generation_payload::GenerationPayload;
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_handle::TokenHandle;

#[derive(Clone, Debug)]
pub(crate) struct DispatchCall {
    pub placement_kind: &'static str,
    pub worker_url: String,
}

type StreamFactory = Box<dyn Fn() -> Result<TokenHandle, EngineError> + Send + Sync + 'static>;

enum DispatcherMode {
    Scripted(Mutex<VecDeque<Result<TokenHandle, EngineError>>>),
    Repeat(StreamFactory),
}

pub(crate) struct MockDispatcher {
    mode: DispatcherMode,
    call_log: Mutex<Vec<DispatchCall>>,
}

impl MockDispatcher {
    pub(crate) fn new(scripted: Vec<Result<TokenHandle, EngineError>>) -> Self {
        Self {
            mode: DispatcherMode::Scripted(Mutex::new(scripted.into())),
            call_log: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn repeat_with_stream<F>(factory: F) -> Self
    where
        F: Fn() -> Result<TokenHandle, EngineError> + Send + Sync + 'static,
    {
        Self {
            mode: DispatcherMode::Repeat(Box::new(factory)),
            call_log: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn calls(&self) -> Vec<DispatchCall> {
        self.call_log.lock().unwrap().clone()
    }
}

#[async_trait]
impl Dispatcher for MockDispatcher {
    async fn dispatch(
        &self,
        placement: &PlacementPlan,
        _payload: &mut GenerationPayload,
    ) -> Result<TokenHandle, EngineError> {
        let (placement_kind, worker_url) = match placement {
            PlacementPlan::Single { worker, .. } => ("single", worker.url().to_string()),
            PlacementPlan::Pair {
                prefill, decode, ..
            } => ("pair", format!("{}|{}", prefill.url(), decode.url())),
        };
        self.call_log.lock().unwrap().push(DispatchCall {
            placement_kind,
            worker_url,
        });
        match &self.mode {
            DispatcherMode::Scripted(q) => q
                .lock()
                .unwrap()
                .pop_front()
                .expect("MockDispatcher: scripted queue exhausted"),
            DispatcherMode::Repeat(f) => f(),
        }
    }
}
