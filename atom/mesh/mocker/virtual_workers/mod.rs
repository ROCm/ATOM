//! Virtual backend worker implementations and worker-pool orchestration.

pub mod golden_assert;
pub mod grpc;
pub mod http;
pub mod mock_case;
pub mod pool;
pub mod req_metrics;
pub mod replay_case_store;
pub mod request;

pub use golden_assert::{
    any_json_contains, assert_any_json_contains, assert_json_contains, json_contains, GoldenAssert,
};
pub use grpc::VirtualGrpcWorker;
pub use http::VirtualWorker;
pub use mock_case::{BackendFixture, ConnectionModeFixture, MockCase, WorkerKindFixture};
pub use pool::{
    VirtualWorkerInstance, VirtualWorkerPool, VirtualWorkerPoolConfig, VirtualWorkerSpec,
};
pub use replay_case_store::ReplayCaseStore;
pub use request::{
    VirtualRequest, VirtualRequestEndpoint, VirtualRequestMode, VirtualRequestPipeline,
    VirtualRequestPipelineConfig, VirtualRequestPipelineResult, VirtualResponse,
};
