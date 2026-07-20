use std::sync::Arc;

use crate::core::placement::traits::PdPlanner;
use crate::core::worker::WorkerType;
use crate::observability::metrics::metrics_labels;
use crate::routers::grpc::engine::Dispatcher;
use crate::routers::grpc::pipeline::Pipeline;
use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::test_support::synthetic_single_stream;
use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

use super::{dispatcher::MockDispatcher, planner::MockPdPlanner, workers::mock_grpc_worker};

pub(crate) fn pipeline_with(
    planner: Arc<dyn PdPlanner>,
    dispatcher: Arc<dyn Dispatcher>,
    backend_label: &'static str,
) -> Arc<Pipeline> {
    Arc::new(Pipeline::with_injected(planner, dispatcher, backend_label))
}

pub(crate) fn pipeline_regular_default() -> Arc<Pipeline> {
    pipeline_with(
        Arc::new(MockPdPlanner::new(vec![])),
        Arc::new(MockDispatcher::new(vec![])),
        metrics_labels::BACKEND_REGULAR,
    )
}

fn canonical_chunks() -> Vec<Result<TokenChunk, EngineError>> {
    vec![
        Ok(TokenChunk::Partial {
            token_ids: vec![1],
            logprobs: None,
        }),
        Ok(TokenChunk::Complete {
            token_ids: vec![1, 2],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 4,
                completion_tokens: 2,
                total_tokens: 6,
            },
            logprobs: None,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: "req-canonical".to_string(),
                weight_version: None,
                cached_tokens: 0,
            },
        }),
    ]
}

pub(crate) fn pipeline_with_chat_path() -> Arc<Pipeline> {
    let worker = mock_grpc_worker("http://w:1", WorkerType::Regular);
    pipeline_with(
        Arc::new(MockPdPlanner::repeat_single(worker, "random")),
        Arc::new(MockDispatcher::repeat_with_stream(|| {
            Ok(synthetic_single_stream(canonical_chunks()))
        })),
        metrics_labels::BACKEND_REGULAR,
    )
}
