//! A `GenerationPayload` as ATOM's GenerateRequest: the sampling fields ATOM's
//! HTTP routes read, stop strings as token ids, and the worker's DP rank.

use crate::atom_engine::{
    grpc::GenerateRequest,
    wire::pb::{stop_strings, SamplingParams, StopStrings, StringList, TokenIds},
};
use crate::core::Worker;
use crate::protocols::common::StringOrArray;
use crate::routers::prepare::generation_payload::GenerationPayload;
use crate::routers::token_handle::engine_error::EngineError;
use crate::tokenizer::traits::Tokenizer;

/// Stop strings are encoded with `tokenizer`, the model's; a request with any
/// fails without it.
pub(crate) fn generate_request(
    payload: &GenerationPayload,
    worker: &dyn Worker,
    tokenizer: Option<&dyn Tokenizer>,
) -> Result<GenerateRequest, EngineError> {
    let stops = match &payload.stop.stop {
        Some(StringOrArray::String(stop)) => std::slice::from_ref(stop),
        Some(StringOrArray::Array(stops)) => stops.as_slice(),
        None => &[],
    };
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
            stop_strings: payload.stop.stop.as_ref().map(stop_strings),
            n: Some(sampling.n.max(0) as u32),
            ..Default::default()
        }),
        stop_token_sequences: stop_token_sequences(stops, tokenizer)?,
        kv_transfer_params_json: None,
        data_parallel_rank: worker.dp_rank().map(|rank| rank as i32),
        stream: payload.stream,
        structured_output: None,
    })
}

fn stop_strings(stop: &StringOrArray) -> StopStrings {
    let value = match stop {
        StringOrArray::String(stop) => stop_strings::Value::Single(stop.clone()),
        StringOrArray::Array(stops) => stop_strings::Value::Many(StringList {
            items: stops.clone(),
        }),
    };
    StopStrings { value: Some(value) }
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
