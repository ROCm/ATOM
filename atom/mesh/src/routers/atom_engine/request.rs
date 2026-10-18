//! A prepared payload → the Cores' `Submission`. Refuses up front what the
//! Python API server refuses: an over-long context, a prompt the KV pool
//! cannot hold, logprobs under speculative decoding.

use axum::response::Response;
use http::HeaderMap;

use super::{token_source::SeqTokenSource, AtomEngineRouter};
use crate::{
    atom_engine::{wire::pb, Submission},
    protocols::common::StringOrArray,
    routers::{
        comm::error,
        prepare::{generation_payload::GenerationPayload, response_context::ResponseContext},
        token_handle::{engine_error::EngineError, token_handle::TokenHandle},
    },
};

/// Headers naming the session a request belongs to, in priority order.
const SESSION_HEADERS: [&str; 2] = ["x-dynamo-session-id", "x-correlation-id"];

impl AtomEngineRouter {
    pub(super) fn submit(
        &self,
        payload: &GenerationPayload,
        ctx: &ResponseContext,
    ) -> Result<TokenHandle, Response> {
        let model = &self.engine.model;
        let max_tokens = match payload.sampling.max_new_tokens {
            None => model.default_max_tokens as i64,
            Some(v) if v < 1 => {
                return Err(bad_request(format!(
                    "max_tokens must be at least 1, got {v}"
                )))
            }
            Some(v) => v as i64,
        };
        self.check_context(payload.token_ids.len() as u64, max_tokens as u64)?;

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
        let logprobs = payload.logprob.return_logprob;
        if logprobs && model.speculative {
            return Err(bad_request(
                "Token logprobs are not supported with speculative decoding".into(),
            ));
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

        let sampling = pb::SamplingParams {
            temperature: Some(temperature),
            top_k: Some(payload.sampling.top_k as i64),
            top_p: Some(payload.sampling.top_p as f64),
            max_tokens: Some(max_tokens),
            ignore_eos: payload.sampling.ignore_eos,
            stop_strings: (!stop_strings.is_empty()).then_some(pb::StopStrings {
                value: Some(pb::stop_strings::Value::Many(pb::StringList {
                    items: stop_strings,
                })),
            }),
            n,
            logprobs: logprobs.then_some(pb::sampling_params::Logprobs::LogprobsFlag(true)),
        };
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

fn session_id(headers: Option<&HeaderMap>) -> Option<&str> {
    let headers = headers?;
    SESSION_HEADERS
        .iter()
        .find_map(|name| headers.get(*name)?.to_str().ok())
}

fn bad_request(message: String) -> Response {
    error::bad_request("invalid_request", message)
}
