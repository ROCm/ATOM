use std::sync::Arc;

use http::HeaderMap;
use thiserror::Error;

use crate::core::Worker;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    Http,
    Grpc,
}

#[derive(Debug, Clone, Default)]
pub struct RequestDescriptor<'a> {
    pub model_id: Option<&'a str>,
    pub protocol: Option<Protocol>,
    pub text: Option<&'a str>,
    pub tokens: Option<&'a [u32]>,
    pub headers: Option<&'a HeaderMap>,
    pub stream: bool,
}

#[derive(Debug)]
pub enum PlacementPlan {
    Single {
        worker: Arc<dyn Worker>,
        policy_name: &'static str,
    },
    Pair {
        prefill: Arc<dyn Worker>,
        decode: Arc<dyn Worker>,
        prefill_policy: &'static str,
        decode_policy: &'static str,
    },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum PlacementError {
    #[error("no workers in registry")]
    NoWorkers,
    #[error("no available (healthy) workers")]
    NoAvailableWorkers,
    #[error("no prefill workers available")]
    NoPrefillWorkers,
    #[error("no decode workers available")]
    NoDecodeWorkers,
    #[error("policy returned None")]
    PolicyReturnedNone,
    #[error("model not found: {model_id}")]
    ModelNotFound { model_id: String },
}

#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum AdapterError {
    #[error("request body is not a JSON object")]
    BodyNotObject,
    #[error("bootstrap_addr missing for prefill {prefill_url}")]
    BootstrapAddrMissing { prefill_url: String },
    #[error("engine_id missing for prefill {prefill_url} dp_rank {dp_rank}")]
    EngineIdMissing { prefill_url: String, dp_rank: usize },
    #[error("tp_size missing for prefill {prefill_url}")]
    TpSizeMissing { prefill_url: String },
    #[error("PairCtx type does not match adapter")]
    CtxTypeMismatch,
}
