//! Backend-specific proto adapters. Single place where `GenerationPayload`
//! crosses the boundary into `mesh_grpc::*`.

use mesh_grpc::{sglang_proto, vllm_proto};

use crate::protocols::common::StringOrArray;
use crate::routers::prepare::generation_payload::{GenerationPayload, PdMetadata};

pub fn to_sglang_proto(payload: &GenerationPayload) -> sglang_proto::GenerateRequest {
    sglang_proto::GenerateRequest {
        request_id: payload.request_id.clone(),
        tokenized: Some(sglang_proto::TokenizedInput {
            original_text: payload.text.clone(),
            input_ids: payload.token_ids.clone(),
        }),
        mm_inputs: None,
        sampling_params: Some(build_sglang_sampling_params(payload)),
        return_logprob: payload.logprob.return_logprob,
        logprob_start_len: payload.logprob.logprob_start_len,
        top_logprobs_num: payload.logprob.top_logprobs_num as i32,
        token_ids_logprob: payload.logprob.token_ids_logprob.clone(),
        return_hidden_states: payload.return_hidden_states,
        stream: payload.stream,
        log_metrics: payload.log_metrics,
        disaggregated_params: payload.pd_metadata.as_ref().map(to_sglang_disagg),
        ..Default::default()
    }
}

pub fn to_vllm_proto(payload: &GenerationPayload) -> vllm_proto::GenerateRequest {
    vllm_proto::GenerateRequest {
        request_id: payload.request_id.clone(),
        input: Some(vllm_proto::generate_request::Input::Tokenized(
            vllm_proto::TokenizedInput {
                original_text: payload.text.clone(),
                input_ids: payload.token_ids.clone(),
            },
        )),
        sampling_params: Some(build_vllm_sampling_params(payload)),
        stream: payload.stream,
    }
}

fn build_sglang_sampling_params(payload: &GenerationPayload) -> sglang_proto::SamplingParams {
    let stop = stop_strings(payload.stop.stop.as_ref());
    let stop_token_ids = payload.stop.stop_token_ids.clone().unwrap_or_default();
    let constraint = payload
        .tool_constraints
        .as_ref()
        .map(|(ty, val)| sglang_tool_constraint_to_proto(ty, val));

    sglang_proto::SamplingParams {
        temperature: payload.sampling.temperature,
        top_p: payload.sampling.top_p,
        top_k: payload.sampling.top_k,
        min_p: payload.sampling.min_p,
        frequency_penalty: payload.sampling.frequency_penalty,
        presence_penalty: payload.sampling.presence_penalty,
        repetition_penalty: payload.sampling.repetition_penalty,
        max_new_tokens: payload.sampling.max_new_tokens,
        stop,
        stop_token_ids,
        skip_special_tokens: payload.stop.skip_special_tokens,
        spaces_between_special_tokens: true,
        n: payload.sampling.n,
        min_new_tokens: payload.sampling.min_new_tokens,
        ignore_eos: payload.sampling.ignore_eos,
        no_stop_trim: payload.stop.no_stop_trim,
        constraint,
        ..Default::default()
    }
}

fn build_vllm_sampling_params(payload: &GenerationPayload) -> vllm_proto::SamplingParams {
    let stop = stop_strings(payload.stop.stop.as_ref());
    let stop_token_ids = payload.stop.stop_token_ids.clone().unwrap_or_default();
    let constraint = payload
        .tool_constraints
        .as_ref()
        .map(|(ty, val)| vllm_tool_constraint_to_proto(ty, val));

    vllm_proto::SamplingParams {
        temperature: Some(payload.sampling.temperature),
        top_p: payload.sampling.top_p,
        top_k: payload.sampling.top_k.max(0) as u32,
        min_p: payload.sampling.min_p,
        frequency_penalty: payload.sampling.frequency_penalty,
        presence_penalty: payload.sampling.presence_penalty,
        repetition_penalty: payload.sampling.repetition_penalty,
        max_tokens: payload.sampling.max_new_tokens.map(|v| v.max(0) as u32),
        min_tokens: payload.sampling.min_new_tokens.max(0) as u32,
        stop,
        stop_token_ids,
        skip_special_tokens: payload.stop.skip_special_tokens,
        spaces_between_special_tokens: true,
        ignore_eos: payload.sampling.ignore_eos,
        n: payload.sampling.n.max(0) as u32,
        constraint,
        ..Default::default()
    }
}

fn to_sglang_disagg(pd: &PdMetadata) -> sglang_proto::DisaggregatedParams {
    sglang_proto::DisaggregatedParams {
        bootstrap_host: pd.bootstrap_host.clone(),
        bootstrap_port: pd.bootstrap_port,
        bootstrap_room: pd.bootstrap_room,
    }
}

fn stop_strings(stop: Option<&StringOrArray>) -> Vec<String> {
    match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => Vec::new(),
    }
}

fn sglang_tool_constraint_to_proto(
    constraint_type: &str,
    value: &str,
) -> sglang_proto::sampling_params::Constraint {
    match constraint_type {
        "json_schema" => sglang_proto::sampling_params::Constraint::JsonSchema(value.to_string()),
        "ebnf" => sglang_proto::sampling_params::Constraint::EbnfGrammar(value.to_string()),
        "regex" => sglang_proto::sampling_params::Constraint::Regex(value.to_string()),
        "structural_tag" => {
            sglang_proto::sampling_params::Constraint::StructuralTag(value.to_string())
        }
        other => panic!("unknown tool constraint type: {other}"),
    }
}

fn vllm_tool_constraint_to_proto(
    constraint_type: &str,
    value: &str,
) -> vllm_proto::sampling_params::Constraint {
    match constraint_type {
        "json_schema" => vllm_proto::sampling_params::Constraint::JsonSchema(value.to_string()),
        // vLLM names the BNF/grammar variant `Grammar`, not `EbnfGrammar`.
        "ebnf" | "grammar" => vllm_proto::sampling_params::Constraint::Grammar(value.to_string()),
        "regex" => vllm_proto::sampling_params::Constraint::Regex(value.to_string()),
        "structural_tag" => {
            vllm_proto::sampling_params::Constraint::StructuralTag(value.to_string())
        }
        other => panic!("unknown tool constraint type: {other}"),
    }
}
