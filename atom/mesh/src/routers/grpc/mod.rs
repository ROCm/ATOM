//! gRPC router implementations

use mesh_grpc::sglang_proto::MultimodalInputs;

use crate::protocols::common::StringOrArray;

pub mod client; // Used by core/
pub(crate) mod common;
pub(crate) mod completion_adapter;
pub(crate) mod context;
pub(crate) mod pd_router; // Used by routers/factory
pub(crate) mod pipeline;
pub(crate) mod proto_wrapper;
pub(crate) mod regular;
pub(crate) mod router; // Used by routers/factory
pub(crate) mod utils; // Used by routers/http

/// Processed chat messages ready for gRPC generation
#[derive(Debug)]
pub(crate) struct ProcessedMessages {
    pub text: String,
    pub multimodal_inputs: Option<MultimodalInputs>,
    #[allow(dead_code)]
    pub stop_sequences: Option<StringOrArray>,
}
