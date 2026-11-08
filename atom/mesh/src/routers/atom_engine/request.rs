//! A prepared payload → the Cores' `Submission`. Refuses up front what the
//! Python API server refuses (sampling values `SamplingParams` rejects,
//! `max_tokens` over the deployment's cap, an over-long context, a prompt the
//! KV pool cannot hold, logprobs under forced acceptance) and what this
//! server does not serve yet (top and streamed logprobs, streaming n > 1).

use axum::response::Response;
use http::HeaderMap;

use super::{token_source::SeqTokenSource, AtomEngineRouter};
use crate::{
    atom_engine::{wire::pb, ModelSpec, Submission},
    protocols::common::StringOrArray,
    routers::{
        comm::error,
        prepare::{generation_payload::GenerationPayload, response_context::ResponseContext},
        token_handle::{engine_error::EngineError, token_handle::TokenHandle},
    },
};

/// Headers naming the session a request belongs to, in priority order.
const SESSION_HEADERS: [&str; 3] = ["x-dynamo-session-id", "x-session-id", "x-correlation-id"];

/// What a request asks of admission beyond its prepared payload.
#[derive(Debug, Default, Clone, Copy)]
pub(super) struct Admission {
    /// It must call a tool, and is held to the model's required channel if
    /// its tool-call format has one.
    pub required_tool_call: bool,
    /// Its `data_parallel_rank`.
    pub rank: Option<usize>,
    /// Its `logprobs` field is one the API ignores (`/v1/completions`, as
    /// the Python server does): the Core is not asked for them.
    pub ignore_logprobs: bool,
}

impl AtomEngineRouter {
    pub(super) fn submit(
        &self,
        payload: &GenerationPayload,
        ctx: &ResponseContext,
        admission: Admission,
    ) -> Result<TokenHandle, Response> {
        let model = &self.engine.model;
        check_sampling(payload).map_err(bad_request)?;
        let max_tokens = max_tokens(
            model,
            payload.sampling.max_new_tokens,
            payload.token_ids.len(),
        )
        .map_err(bad_request)?;
        self.check_context(payload.token_ids.len() as u64, max_tokens)?;

        // Greedy siblings would all be identical; `_coerce_n` folds them.
        let temperature = payload.sampling.temperature as f64;
        let n = match payload.sampling.n.max(1) as u32 {
            n if n > 1 && temperature <= 0.0 => 1,
            n => n,
        };
        if payload.stream && n > 1 {
            return Err(bad_request(
                "streaming with n > 1 is not supported yet".into(),
            ));
        }
        let logprobs = payload.logprob.return_logprob && !admission.ignore_logprobs;
        if logprobs && model.synthetic_acceptance {
            return Err(bad_request(
                "logprobs is unavailable with synthetic forced acceptance".into(),
            ));
        }
        if logprobs && payload.logprob.top_logprobs_num > 0 {
            return Err(not_yet("top_logprobs"));
        }
        if logprobs && payload.stream {
            return Err(not_yet("streaming with logprobs"));
        }

        let stop_strings = match &payload.stop.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(items)) => items.clone(),
            None => vec![],
        };
        // The Core stops on token sequences only: each stop string as the
        // tokenizer spells it, each stop token as a sequence of one.
        let mut stop_token_sequences = Vec::new();
        for stop in &stop_strings {
            let encoding = ctx
                .tokenizer
                .encode(stop, false)
                .map_err(|e| bad_request(format!("cannot tokenize stop string: {e}")))?;
            if !encoding.token_ids().is_empty() {
                let ids = encoding.token_ids().iter().map(|&t| t as i32).collect();
                stop_token_sequences.push(pb::TokenIds { ids });
            }
        }
        for &id in payload.stop.stop_token_ids.iter().flatten() {
            stop_token_sequences.push(pb::TokenIds {
                ids: vec![id as i32],
            });
        }

        let sampling = sampling_params(
            model,
            payload,
            Settled {
                max_tokens,
                n,
                logprobs,
                stop_strings,
                required_tool_call: admission.required_tool_call,
            },
        );
        let seqs = self
            .engine
            .client
            .submit(Submission {
                request_id: &payload.request_id,
                prompt: &payload.token_ids,
                sampling,
                stop_token_sequences,
                n,
                session: session_id(ctx.headers.as_ref()),
                rank: admission.rank,
                stream: payload.stream,
            })
            .map_err(|failure| EngineError::from(failure).into_response())?;
        Ok(TokenHandle::new(SeqTokenSource::new(
            seqs,
            payload.request_id.clone(),
        )))
    }

    /// `_validate_context_length` and `_validate_pool_capacity`, same words.
    fn check_context(&self, prompt_tokens: u64, max_tokens: u64) -> Result<(), Response> {
        if prompt_tokens == 0 {
            return Err(bad_request("the prompt is empty".into()));
        }
        if let Some(max_model_len) = self.engine.model.max_model_len {
            let total = prompt_tokens + max_tokens;
            if total > max_model_len {
                return Err(bad_request(format!(
                    "This model's maximum context length is {max_model_len} tokens. \
                     However, you requested {max_tokens} output tokens and your prompt \
                     contains at least {prompt_tokens} input tokens, for a total of at \
                     least {total} tokens. Please reduce the length of the input prompt \
                     or the number of requested output tokens."
                )));
            }
        }
        if let Some(pool) = self.engine.client.max_pool_tokens() {
            if prompt_tokens > pool {
                return Err(bad_request(format!(
                    "This server's KV cache holds at most {pool} tokens for a single \
                     request, and your prompt contains at least {prompt_tokens} input \
                     tokens. Please shorten the prompt, or restart the server with a \
                     higher --gpu-memory-utilization to enlarge the cache."
                )));
            }
        }
        Ok(())
    }
}

/// The request's `max_tokens` as `_build_sampling_params` and
/// `_validate_sequence_context_length` settle it: an explicit one is held to
/// the deployment's cap, a default is cut to that cap and to the context the
/// prompt leaves.
fn max_tokens(
    model: &ModelSpec,
    requested: Option<i32>,
    prompt_tokens: usize,
) -> Result<u64, String> {
    let cap = model.max_output_tokens;
    match requested {
        Some(v) if v < 1 => Err(format!("max_tokens must be at least 1, got {v}")),
        Some(v) => match cap {
            Some(cap) if v as u64 > cap => {
                Err(format!("max_tokens must be <= {cap} for this deployment"))
            }
            _ => Ok(v as u64),
        },
        None => {
            let mut max_tokens = model.default_max_tokens as u64;
            if let Some(cap) = cap {
                max_tokens = max_tokens.min(cap);
            }
            if let Some(max_model_len) = model.max_model_len {
                let left = max_model_len.saturating_sub(prompt_tokens as u64);
                if left == 0 {
                    return Err("The prompt leaves no context space for output tokens".into());
                }
                max_tokens = max_tokens.min(left);
            }
            Ok(max_tokens)
        }
    }
}

/// What `submit` settled for a request before it builds its parameters.
struct Settled {
    max_tokens: u64,
    n: u32,
    logprobs: bool,
    stop_strings: Vec<String>,
    required_tool_call: bool,
}

/// The request's SamplingParams on the wire.
fn sampling_params(
    model: &ModelSpec,
    payload: &GenerationPayload,
    settled: Settled,
) -> pb::SamplingParams {
    let stop_strings = settled.stop_strings;
    pb::SamplingParams {
        temperature: Some(payload.sampling.temperature as f64),
        top_k: Some(payload.sampling.top_k as i64),
        top_p: Some(payload.sampling.top_p as f64),
        max_tokens: Some(settled.max_tokens as i64),
        ignore_eos: payload.sampling.ignore_eos,
        stop_strings: (!stop_strings.is_empty()).then_some(pb::StopStrings {
            value: Some(pb::stop_strings::Value::Many(pb::StringList {
                items: stop_strings,
            })),
        }),
        n: settled.n,
        logprobs: settled
            .logprobs
            .then_some(pb::sampling_params::Logprobs::LogprobsFlag(true)),
        top_logprobs: 0,
        required_channel: model
            .required_channel
            .as_ref()
            .filter(|_| settled.required_tool_call)
            .map(Into::into),
        json_schema: None,
        structured_tool_format: None,
        structured_reasoning_end: String::new(),
        // `max_tokens` is already final: nothing in the Core may shrink it.
        max_tokens_is_default: false,
    }
}

/// What `SamplingParams.__post_init__` refuses, refused before any response
/// starts (in the Core it would arrive as an error after the stream began).
fn check_sampling(payload: &GenerationPayload) -> Result<(), String> {
    let s = &payload.sampling;
    if !(s.top_p > 0.0 && s.top_p <= 1.0) {
        return Err("top_p must be in range (0.0, 1.0]".into());
    }
    if s.top_k != -1 && s.top_k != 0 && s.top_k < 1 {
        return Err("top_k must be -1 or 0 (disabled) or >= 1".into());
    }
    // `n` < 1 is not refused: `_coerce_n` makes it 1, as `submit` does.
    Ok(())
}

/// The first session header present and not empty, as the Python server's
/// `or` chain picks it.
fn session_id(headers: Option<&HeaderMap>) -> Option<&str> {
    let headers = headers?;
    SESSION_HEADERS
        .iter()
        .find_map(|name| headers.get(*name)?.to_str().ok().filter(|v| !v.is_empty()))
}

fn bad_request(message: String) -> Response {
    error::bad_request("invalid_request", message)
}

fn not_yet(what: &str) -> Response {
    error::bad_request(
        "not_supported",
        format!("{what} is not supported by the ATOM standalone server yet"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(max_model_len: u64, cap: Option<u64>) -> ModelSpec {
        serde_json::from_value(serde_json::json!({
            "served_name": "m",
            "tokenizer_path": "m",
            "max_model_len": max_model_len,
            "default_max_tokens": 8192,
            "max_output_tokens": cap,
        }))
        .unwrap()
    }

    fn payload(top_p: f32, top_k: i32) -> GenerationPayload {
        use crate::routers::prepare::generation_payload::{
            LogprobConfig, SamplingParams, StopConfig,
        };
        GenerationPayload {
            request_id: "r".into(),
            text: String::new(),
            token_ids: vec![1],
            sampling: SamplingParams {
                temperature: 0.5,
                top_p,
                top_k,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                max_new_tokens: None,
                min_new_tokens: 0,
                n: 1,
                ignore_eos: false,
            },
            stop: StopConfig {
                stop: None,
                stop_token_ids: None,
                skip_special_tokens: true,
                no_stop_trim: false,
            },
            logprob: LogprobConfig {
                return_logprob: false,
                top_logprobs_num: 0,
                logprob_start_len: -1,
                token_ids_logprob: vec![],
                input_logprobs: false,
            },
            tool_constraints: None,
            pd_metadata: None,
            stream: false,
            return_hidden_states: false,
            log_metrics: false,
        }
    }

    #[test]
    fn only_a_required_tool_call_is_held_to_the_channel() {
        let mut m = model(4096, None);
        m.required_channel = Some(crate::atom_engine::spec::RequiredChannel {
            after: vec![1],
            opening: vec![2],
            opens_at: 1,
            end_ids: vec![3],
        });
        let settle = |required_tool_call| Settled {
            max_tokens: 16,
            n: 1,
            logprobs: false,
            stop_strings: vec![],
            required_tool_call,
        };
        let held = sampling_params(&m, &payload(1.0, -1), settle(true));
        assert_eq!(held.required_channel.unwrap().opening, [2]);
        assert!(sampling_params(&m, &payload(1.0, -1), settle(false))
            .required_channel
            .is_none());
    }

    #[test]
    fn sampling_values_the_core_would_refuse_are_refused_up_front() {
        assert!(check_sampling(&payload(1.0, -1)).is_ok());
        assert!(check_sampling(&payload(1.0, 0)).is_ok());
        assert!(check_sampling(&payload(0.0, -1)).is_err());
        assert!(check_sampling(&payload(1.5, -1)).is_err());
        assert!(check_sampling(&payload(1.0, -2)).is_err());
    }

    #[test]
    fn an_explicit_max_tokens_is_held_to_the_cap() {
        let m = model(32768, Some(16384));
        assert_eq!(max_tokens(&m, Some(16384), 10), Ok(16384));
        assert_eq!(
            max_tokens(&m, Some(16385), 10),
            Err("max_tokens must be <= 16384 for this deployment".into())
        );
        assert!(max_tokens(&m, Some(0), 10).is_err());
    }

    #[test]
    fn a_default_max_tokens_shrinks_to_the_cap_and_the_context_left() {
        assert_eq!(max_tokens(&model(32768, Some(4096)), None, 10), Ok(4096));
        // 8192 would overrun the context; the default takes what is left.
        assert_eq!(max_tokens(&model(10000, None), None, 9000), Ok(1000));
        assert!(max_tokens(&model(10000, None), None, 10000).is_err());
    }

    #[test]
    fn the_session_header_is_read_in_api_server_order() {
        let mut headers = HeaderMap::new();
        headers.insert("x-correlation-id", "c".parse().unwrap());
        assert_eq!(session_id(Some(&headers)), Some("c"));
        headers.insert("x-session-id", "s".parse().unwrap());
        assert_eq!(session_id(Some(&headers)), Some("s"));
        headers.insert("x-dynamo-session-id", "d".parse().unwrap());
        assert_eq!(session_id(Some(&headers)), Some("d"));
        // An empty one is skipped, as the Python server's `or` chain skips it.
        headers.insert("x-dynamo-session-id", "".parse().unwrap());
        assert_eq!(session_id(Some(&headers)), Some("s"));
    }
}
