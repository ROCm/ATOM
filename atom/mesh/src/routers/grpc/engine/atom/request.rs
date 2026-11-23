// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! A `GenerationPayload` as ATOM's GenerateRequest (the sampling fields ATOM's HTTP routes
//! read, stop strings and stop ids as id sequences, DP rank and session ids, a chat's
//! StructuredOutput), which the Session sends as a RequestSpec.

use http::StatusCode;
use serde_json::Value as Json;

use crate::atom_engine::{
    grpc::{self, GenerateRequest},
    wire::pb::{
        value::Kind, MapEntry, RequestSpec, SamplingParams, TokenIds, Value, ValueList, ValueMap,
    },
};
use crate::core::Worker;
use crate::protocols::common::StringOrArray;
use crate::routers::prepare::generation_payload::{GenerationPayload, StructuredOutput};
use crate::routers::token_handle::engine_error::EngineError;
use crate::tokenizer::traits::Tokenizer;

/// Stop strings are encoded with `tokenizer`, the model's; a request with any
/// fails without it. A request for logprobs fails.
pub(crate) fn generate_request(
    payload: &GenerationPayload,
    worker: &dyn Worker,
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<GenerateRequest, EngineError> {
    let logprob = &payload.logprob;
    if logprob.return_logprob
        || logprob.top_logprobs_num > 0
        || logprob.input_logprobs
        || !logprob.token_ids_logprob.is_empty()
    {
        // ATOM's Generate answers no logprobs; atomesh refuses rather than drop them.
        return Err(EngineError::RequestBuildFailed(
            "logprobs are not served by ATOM workers over gRPC".into(),
        ));
    }
    let stops = match &payload.stop.stop {
        Some(StringOrArray::String(stop)) => std::slice::from_ref(stop),
        Some(StringOrArray::Array(stops)) => stops.as_slice(),
        None => &[],
    };
    let mut stop_sequences = stop_token_sequences(stops, tokenizer)?;
    // An explicit stop id is a one-id stop sequence, which ATOM ends the output at.
    if let Some(ids) = &payload.stop.stop_token_ids {
        stop_sequences.extend(ids.iter().map(|&id| TokenIds {
            ids: vec![id as i32],
        }));
    }
    let sampling = &payload.sampling;
    Ok(GenerateRequest {
        request_id: payload.request_id.clone(),
        prompt_token_ids: payload.token_ids.clone(),
        sampling: Some(SamplingParams {
            temperature: Some(json_f64(sampling.temperature)),
            top_k: Some(sampling.top_k.into()),
            top_p: Some(json_f64(sampling.top_p)),
            // Absent, ATOM applies its HTTP routes' default.
            max_tokens: sampling.max_new_tokens.map(i64::from),
            ignore_eos: sampling.ignore_eos,
            n: Some(sampling.n.max(0) as u32),
            ..Default::default()
        }),
        stop_token_sequences: stop_sequences,
        kv_transfer_params_json: None,
        data_parallel_rank: worker.dp_rank().map(|rank| rank as i32),
        stream: payload.stream,
        structured_output: payload.structured_output.as_ref().map(structured_output),
        dp_session_id: payload.dp_session_id.clone(),
        dp_parent_session_id: payload.dp_parent_session_id.clone(),
    })
}

/// `request`, its StructuredOutput already prepared, as a RequestSpec and its prompt;
/// the Session sets the seq_id and the prompt's frame.
pub(crate) fn request_spec(
    request: GenerateRequest,
) -> Result<(RequestSpec, Vec<u32>), EngineError> {
    let kv_transfer_params = request
        .kv_transfer_params_json
        .map(|kv| value(&serde_json::from_str(&kv).map_err(|e| e.to_string())?))
        .transpose()
        .map_err(|e| EngineError::Http {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("kv_transfer_params engine_v1 cannot carry: {e}"),
        })?;
    let spec = RequestSpec {
        sampling: request.sampling,
        stop_token_sequences: request.stop_token_sequences,
        external_request_id: Some(request.request_id),
        kv_transfer_params,
        data_parallel_rank: request.data_parallel_rank.map(i64::from),
        dp_session_id: request.dp_session_id,
        dp_parent_session_id: request.dp_parent_session_id,
        ..Default::default()
    };
    Ok((spec, request.prompt_token_ids))
}

/// `json` as the engine_v1 Value ATOM's frames codec reads back to it.
pub(super) fn value(json: &Json) -> Result<Value, String> {
    let kind = match json {
        Json::Null => Kind::NullValue(true),
        Json::Bool(b) => Kind::BoolValue(*b),
        Json::Number(n) => match (n.as_i64(), n.as_f64().filter(|_| n.is_f64())) {
            (Some(i), _) => Kind::IntValue(i),
            (None, Some(f)) => Kind::FloatValue(f),
            (None, None) => return Err(format!("{n} is outside int64")),
        },
        Json::String(s) => Kind::StringValue(s.clone()),
        Json::Array(items) => Kind::ListValue(ValueList {
            items: items.iter().map(value).collect::<Result<_, _>>()?,
        }),
        Json::Object(map) => {
            let entry = |(key, item): (&String, &Json)| {
                Ok(MapEntry {
                    key: Some(Value {
                        kind: Some(Kind::StringValue(key.clone())),
                    }),
                    value: Some(value(item)?),
                })
            };
            let entries = map.iter().map(entry).collect::<Result<_, String>>()?;
            Kind::MapValue(ValueMap { entries })
        }
    };
    Ok(Value { kind: Some(kind) })
}

fn structured_output(carried: &StructuredOutput) -> grpc::StructuredOutput {
    grpc::StructuredOutput {
        tools_json: carried.tools_json.clone(),
        tool_choice_json: carried.tool_choice_json.clone(),
        response_format_json: carried.response_format_json.clone(),
        chat_template_kwargs_json: carried.chat_template_kwargs_json.clone(),
        prompt_opens_reasoning: carried.prompt_opens_reasoning,
        message_tools_json: carried.message_tools_json.clone(),
    }
}

/// Each stop string as ATOM encodes one, without special tokens, dropping a
/// string that encodes to nothing.
fn stop_token_sequences(
    stops: &[String],
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<Vec<TokenIds>, EngineError> {
    if stops.is_empty() {
        return Ok(Vec::new());
    }
    let tokenizer = tokenizer.ok_or_else(|| {
        EngineError::RequestBuildFailed("ATOM stop strings need the model's tokenizer".into())
    })?;
    let mut sequences = Vec::with_capacity(stops.len());
    for stop in stops {
        let encoding = tokenizer
            .encode(stop, false)
            .map_err(|e| EngineError::RequestBuildFailed(format!("stop string {stop:?}: {e}")))?;
        let ids = encoding.token_ids();
        if !ids.is_empty() {
            sequences.push(TokenIds {
                ids: ids.iter().map(|&id| id as i32).collect(),
            });
        }
    }
    Ok(sequences)
}

/// `value` widened through its shortest decimal: a JSON number of up to 6 significant digits,
/// in f32's normal range, arrives as the f64 ATOM's HTTP routes parse from it.
fn json_f64(value: f32) -> f64 {
    value.to_string().parse().unwrap_or(value.into())
}
