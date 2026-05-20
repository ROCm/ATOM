//! Backend-specific proto adapters. Single place where `GenerationPayload`
//! crosses the boundary into `mesh_grpc::*`.
//!
//! Spike scope (Part B): `to_sglang_proto` only, with `text` and `multimodal`
//! still passed as arguments. Part C folds them into the payload.

use mesh_grpc::sglang_proto;

use crate::protocols::common::StringOrArray;
use crate::routers::prepare::generation_payload::GenerationPayload;

pub fn to_sglang_proto(
    payload: &GenerationPayload,
    text: String,
    multimodal: Option<sglang_proto::MultimodalInputs>,
) -> sglang_proto::GenerateRequest {
    sglang_proto::GenerateRequest {
        request_id: payload.request_id.clone(),
        tokenized: Some(sglang_proto::TokenizedInput {
            original_text: text,
            input_ids: payload.token_ids.clone(),
        }),
        mm_inputs: multimodal,
        sampling_params: Some(build_sampling_params(payload)),
        return_logprob: payload.logprob.return_logprob,
        logprob_start_len: -1,
        top_logprobs_num: payload.logprob.top_logprobs_num as i32,
        return_hidden_states: false,
        stream: false,
        ..Default::default()
    }
}

fn build_sampling_params(payload: &GenerationPayload) -> sglang_proto::SamplingParams {
    let stop = stop_strings(payload.stop.stop.as_ref());
    let stop_token_ids = payload.stop.stop_token_ids.clone().unwrap_or_default();
    let constraint = payload
        .tool_constraints
        .as_ref()
        .map(|(ty, val)| tool_constraint_to_proto(ty, val));

    sglang_proto::SamplingParams {
        temperature: payload.sampling.temperature,
        top_p: payload.sampling.top_p,
        top_k: payload.sampling.top_k,
        min_p: 0.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repetition_penalty: payload.sampling.repetition_penalty,
        max_new_tokens: Some(payload.sampling.max_new_tokens),
        stop,
        stop_token_ids,
        skip_special_tokens: payload.stop.skip_special_tokens,
        spaces_between_special_tokens: true,
        n: 1,
        min_new_tokens: 0,
        ignore_eos: false,
        no_stop_trim: payload.stop.no_stop_trim,
        constraint,
        ..Default::default()
    }
}

fn stop_strings(stop: Option<&StringOrArray>) -> Vec<String> {
    match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => Vec::new(),
    }
}

fn tool_constraint_to_proto(
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
