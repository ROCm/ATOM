// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! A `GenerationPayload` as ATOM's GenerateRequest: the sampling fields ATOM's
//! HTTP routes read, stop strings and stop ids as token id sequences, and the
//! worker's DP rank.

use crate::atom_engine::{
    grpc::GenerateRequest,
    wire::pb::{SamplingParams, TokenIds},
};
use crate::core::Worker;
use crate::protocols::common::StringOrArray;
use crate::routers::prepare::generation_payload::GenerationPayload;
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
        structured_output: None,
        dp_session_id: payload.dp_session_id.clone(),
        dp_parent_session_id: payload.dp_parent_session_id.clone(),
    })
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
