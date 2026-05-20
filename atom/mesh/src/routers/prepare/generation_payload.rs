//! Transport-neutral generation payload.
//!
//! Carries everything the engine needs to build a backend-specific proto.
//! No backend types appear here — `to_sglang_proto` / `to_vllm_proto` are
//! the only sites that import `mesh_grpc::*`.

use crate::protocols::common::StringOrArray;

pub struct GenerationPayload {
    pub request_id: String,
    pub token_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop: StopConfig,
    pub logprob: LogprobConfig,
    pub tool_constraints: Option<(String, String)>,
}

pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub max_new_tokens: i32,
}

pub struct StopConfig {
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub skip_special_tokens: bool,
    pub no_stop_trim: bool,
}

pub struct LogprobConfig {
    pub return_logprob: bool,
    pub top_logprobs_num: u32,
    pub input_logprobs: bool,
}
