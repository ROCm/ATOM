//! Transport-neutral generation payload.
//!
//! Carries everything the engine needs to build a backend-specific proto.
//! No backend types appear here — `to_sglang_proto` / `to_vllm_proto` are
//! the only sites that import `mesh_grpc::*`.

use crate::protocols::common::StringOrArray;

pub struct GenerationPayload {
    pub request_id: String,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop: StopConfig,
    pub logprob: LogprobConfig,
    pub tool_constraints: Option<(String, String)>,
    pub pd_metadata: Option<PdMetadata>,
    pub stream: bool,
    pub return_hidden_states: bool,
    /// SGLang-only `log_metrics` (defaults to `true` on the SGLang generate
    /// REST surface; chat path leaves at `false` to match the upstream chat
    /// builder which does not touch this field).
    pub log_metrics: bool,
}

pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// SGLang convention: `-1` disables top_k. vLLM adapter maps to `max(0) as u32`
    /// (vLLM treats `0` as disabled).
    pub top_k: i32,
    pub min_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub max_new_tokens: Option<i32>,
    /// Lower-bound on generated tokens. SGLang field `min_new_tokens: i32`;
    /// vLLM field `min_tokens: u32`. `0` means unset on both wires.
    pub min_new_tokens: i32,
    pub n: i32,
    pub ignore_eos: bool,
}

pub struct StopConfig {
    pub stop: Option<StringOrArray>,
    pub stop_token_ids: Option<Vec<u32>>,
    pub skip_special_tokens: bool,
    /// SGLang only. vLLM adapter ignores this field.
    pub no_stop_trim: bool,
}

pub struct LogprobConfig {
    pub return_logprob: bool,
    pub top_logprobs_num: u32,
    /// SGLang `GenerateRequest.logprob_start_len`. `-1` means "no input
    /// logprobs"; values `>= 0` request input logprobs starting at that
    /// prompt position. vLLM has no analog and ignores this field.
    pub logprob_start_len: i32,
    /// SGLang-only. Per-token selective logprob request: the wire field is
    /// `repeated uint32 token_ids_logprob`. Empty vec = unset.
    pub token_ids_logprob: Vec<u32>,
    /// Consumed by the PD stream merge layer to gate prefill polling.
    /// Derived from `logprob_start_len >= 0` in the generate path.
    pub input_logprobs: bool,
}

/// PD-disaggregated bootstrap metadata. SGLang only; vLLM does not support
/// disaggregated serving and silently ignores this field.
pub struct PdMetadata {
    pub bootstrap_host: String,
    pub bootstrap_port: i32,
    pub bootstrap_room: i32,
}
