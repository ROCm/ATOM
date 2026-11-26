//! A prepared payload, admitted as the Python API server admits it: sampling
//! values `SamplingParams` refuses, `max_tokens` over the deployment's cap,
//! an over-long context and a prompt the KV pool cannot hold are refused
//! here, before any response starts, with the server's own words. What the
//! token bridge does not serve yet is refused alike.

use crate::{
    atom_engine::{wire, ModelSpec},
    protocols::common::StringOrArray,
    routers::prepare::generation_payload::GenerationPayload,
    tokenizer::traits::Tokenizer,
};

/// What admission settled for a request, ready for its `Submission`.
#[derive(Debug)]
pub(super) struct Admitted {
    pub sampling: wire::Sampling,
    pub stop_token_sequences: Vec<Vec<i32>>,
    pub n: u32,
}

/// Admit `payload` for `model`, whose Cores' KV pools hold `max_pool_tokens`
/// for one prompt. `tool_choice_required` holds the output to the model's
/// tool-call channel, where it has one. A refusal is the request's fault.
pub(super) fn admit(
    model: &ModelSpec,
    max_pool_tokens: Option<u64>,
    payload: &GenerationPayload,
    tokenizer: &dyn Tokenizer,
    tool_choice_required: bool,
) -> Result<Admitted, String> {
    let prompt_tokens = payload.token_ids.len() as u64;
    if prompt_tokens == 0 {
        return Err("the prompt is empty".into());
    }
    // Greedy siblings would all be identical; `_coerce_n` folds them.
    let n = match payload.sampling.n.max(1) as u32 {
        n if n > 1 && payload.sampling.temperature <= 0.0 => 1,
        n => n,
    };
    unserved(payload, n)?;
    check_sampling(payload)?;
    let max_tokens = max_tokens(model, payload.sampling.max_new_tokens, prompt_tokens)?;
    check_context(model, max_pool_tokens, prompt_tokens, max_tokens)?;

    let stop_strings = match &payload.stop.stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(items)) => items.clone(),
        None => vec![],
    };
    // The Core stops on token sequences only: each stop string as the
    // tokenizer spells it, each stop token as a sequence of one.
    let mut stop_token_sequences = Vec::new();
    for stop in &stop_strings {
        let encoding = tokenizer
            .encode(stop, false)
            .map_err(|e| format!("cannot tokenize stop string: {e}"))?;
        if !encoding.token_ids().is_empty() {
            stop_token_sequences.push(encoding.token_ids().iter().map(|&t| t as i32).collect());
        }
    }
    for &id in payload.stop.stop_token_ids.iter().flatten() {
        stop_token_sequences.push(vec![id as i32]);
    }

    let sampling = wire::Sampling {
        temperature: payload.sampling.temperature as f64,
        top_k: payload.sampling.top_k as i64,
        top_p: payload.sampling.top_p as f64,
        max_tokens: max_tokens as i64,
        ignore_eos: payload.sampling.ignore_eos,
        stop_strings: (!stop_strings.is_empty()).then_some(stop_strings),
        n,
        logprobs: None,
        top_logprobs: 0,
        required_channel: model
            .required_channel
            .as_ref()
            .filter(|_| tool_choice_required)
            .map(Into::into),
        json_schema: None,
        structured_tool_format: None,
        structured_reasoning_end: String::new(),
        // `max_tokens` is already final: nothing in the Core may shrink it.
        max_tokens_is_default: false,
    };
    Ok(Admitted {
        sampling,
        stop_token_sequences,
        n,
    })
}

/// What the token bridge refuses too (`BridgeRequest::from_payload`), so
/// either dispatcher serves the same requests.
fn unserved(payload: &GenerationPayload, n: u32) -> Result<(), String> {
    let refused = [
        (
            payload.stream && n > 1,
            "streaming with n > 1 is not supported by the ATOM Rust frontend yet",
        ),
        (
            payload.logprob.return_logprob || payload.logprob.logprob_start_len >= 0,
            "logprobs are not supported by the ATOM Rust frontend yet",
        ),
        (
            payload.tool_constraints.is_some(),
            "tool_choice that constrains the output is not supported by the ATOM Rust \
             frontend yet",
        ),
        (
            payload.return_hidden_states,
            "return_hidden_states is not supported by the ATOM engine",
        ),
    ];
    match refused.iter().find(|(refuse, _)| *refuse) {
        Some((_, reason)) => Err(reason.to_string()),
        None => Ok(()),
    }
}

/// What `SamplingParams.__post_init__` refuses (in the Core it would arrive
/// as an error after the stream began).
fn check_sampling(payload: &GenerationPayload) -> Result<(), String> {
    let s = &payload.sampling;
    if !(s.top_p > 0.0 && s.top_p <= 1.0) {
        return Err("top_p must be in range (0.0, 1.0]".into());
    }
    if s.top_k != -1 && s.top_k != 0 && s.top_k < 1 {
        return Err("top_k must be -1 or 0 (disabled) or >= 1".into());
    }
    Ok(())
}

/// The request's `max_tokens` as `_build_sampling_params` and
/// `_validate_sequence_context_length` settle it: an explicit one is held to
/// the deployment's cap, a default is cut to that cap and to the context the
/// prompt leaves.
fn max_tokens(
    model: &ModelSpec,
    requested: Option<i32>,
    prompt_tokens: u64,
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
                let left = max_model_len.saturating_sub(prompt_tokens);
                if left == 0 {
                    return Err("The prompt leaves no context space for output tokens".into());
                }
                max_tokens = max_tokens.min(left);
            }
            Ok(max_tokens)
        }
    }
}

/// `_validate_context_length` and `_validate_pool_capacity`, same words.
fn check_context(
    model: &ModelSpec,
    max_pool_tokens: Option<u64>,
    prompt_tokens: u64,
    max_tokens: u64,
) -> Result<(), String> {
    if let Some(max_model_len) = model.max_model_len {
        let total = prompt_tokens + max_tokens;
        if total > max_model_len {
            return Err(format!(
                "This model's maximum context length is {max_model_len} tokens. \
                 However, you requested {max_tokens} output tokens and your prompt \
                 contains at least {prompt_tokens} input tokens, for a total of at \
                 least {total} tokens. Please reduce the length of the input prompt \
                 or the number of requested output tokens."
            ));
        }
    }
    if let Some(pool) = max_pool_tokens {
        if prompt_tokens > pool {
            return Err(format!(
                "This server's KV cache holds at most {pool} tokens for a single \
                 request, and your prompt contains at least {prompt_tokens} input \
                 tokens. Please shorten the prompt, or restart the server with a \
                 higher --gpu-memory-utilization to enlarge the cache."
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        atom_engine::spec::RequiredChannel, protocols::chat::ChatCompletionRequest,
        routers::prepare::build_chat_payload, tokenizer::MockTokenizer,
    };

    fn model(max_model_len: u64, cap: Option<u64>) -> ModelSpec {
        serde_json::from_value(serde_json::json!({
            "served_name": "m",
            "tokenizer_path": "m",
            "max_model_len": max_model_len,
            "default_max_tokens": 8192,
            "template_efforts": [],
            "max_output_tokens": cap,
        }))
        .unwrap()
    }

    fn payload(request: ChatCompletionRequest, prompt: Vec<u32>) -> GenerationPayload {
        build_chat_payload("r".into(), &request, &request, "", prompt, None)
    }

    fn sampled(top_p: f32, top_k: i32) -> GenerationPayload {
        let request = ChatCompletionRequest {
            top_p: Some(top_p),
            top_k: Some(top_k),
            ..Default::default()
        };
        payload(request, vec![1])
    }

    fn admit_to(model: &ModelSpec, payload: &GenerationPayload) -> Result<Admitted, String> {
        admit(model, None, payload, &MockTokenizer::new(), false)
    }

    #[test]
    fn sampling_values_the_core_would_refuse_are_refused_up_front() {
        let m = model(4096, None);
        assert!(admit_to(&m, &sampled(1.0, -1)).is_ok());
        assert!(admit_to(&m, &sampled(1.0, 0)).is_ok());
        assert!(admit_to(&m, &sampled(0.0, -1)).is_err());
        assert!(admit_to(&m, &sampled(1.5, -1)).is_err());
        assert!(admit_to(&m, &sampled(1.0, -2)).is_err());
    }

    #[test]
    fn max_tokens_is_held_to_the_cap_and_a_default_to_the_context_left() {
        // (max_model_len, cap, max_tokens, prompt tokens, allowed).
        let cases = [
            (32768, Some(16384), Some(16384), 10, Some(16384)),
            (32768, Some(16384), Some(0), 10, None),
            (32768, Some(4096), None, 10, Some(4096)),
            // The default 8192 would overrun the context; it takes what is left.
            (10000, None, None, 9000, Some(1000)),
            (10000, None, None, 10000, None),
        ];
        for (len, cap, asked, prompt, allowed) in cases {
            let got = max_tokens(&model(len, cap), asked, prompt);
            assert_eq!(got.ok(), allowed, "{asked:?} of {len}, cap {cap:?}");
        }
        assert_eq!(
            max_tokens(&model(32768, Some(16384)), Some(16385), 10),
            Err("max_tokens must be <= 16384 for this deployment".into())
        );
    }

    #[test]
    fn a_context_or_pool_overrun_is_refused_in_the_servers_words() {
        let m = model(100, None);
        let request = || ChatCompletionRequest {
            max_completion_tokens: Some(50),
            ..Default::default()
        };
        let err = admit_to(&m, &payload(request(), vec![1; 60])).unwrap_err();
        assert!(
            err.starts_with("This model's maximum context length is 100"),
            "{err}"
        );
        let err = admit(
            &m,
            Some(8),
            &payload(request(), vec![1; 10]),
            &MockTokenizer::new(),
            false,
        )
        .unwrap_err();
        assert!(
            err.starts_with("This server's KV cache holds at most 8"),
            "{err}"
        );
    }

    #[test]
    fn greedy_siblings_fold_and_a_streamed_fanout_is_refused() {
        let m = model(4096, None);
        let request = |temperature, stream| ChatCompletionRequest {
            n: Some(3),
            temperature: Some(temperature),
            stream,
            ..Default::default()
        };
        assert_eq!(
            admit_to(&m, &payload(request(0.0, false), vec![1]))
                .unwrap()
                .n,
            1
        );
        assert_eq!(
            admit_to(&m, &payload(request(0.7, false), vec![1]))
                .unwrap()
                .n,
            3
        );
        assert!(admit_to(&m, &payload(request(0.0, true), vec![1])).is_ok());
        assert!(admit_to(&m, &payload(request(0.7, true), vec![1])).is_err());
    }

    #[test]
    fn stops_reach_the_core_as_token_sequences() {
        let m = model(4096, None);
        let mut p = sampled(1.0, -1);
        p.stop.stop = Some(StringOrArray::Array(vec!["Hello world".into()]));
        p.stop.stop_token_ids = Some(vec![7]);
        let admitted = admit_to(&m, &p).unwrap();
        assert_eq!(
            admitted.sampling.stop_strings.as_deref(),
            Some(&["Hello world".to_string()][..])
        );
        assert_eq!(admitted.stop_token_sequences.len(), 2);
        assert!(!admitted.stop_token_sequences[0].is_empty());
        assert_eq!(admitted.stop_token_sequences[1], [7]);
    }

    #[test]
    fn only_a_required_tool_call_is_held_to_the_channel() {
        let mut m = model(4096, None);
        m.required_channel = Some(RequiredChannel {
            after: vec![1],
            opening: vec![2],
            opens_at: 1,
            end_ids: vec![3],
        });
        let p = sampled(1.0, -1);
        let held = admit(&m, None, &p, &MockTokenizer::new(), true).unwrap();
        assert_eq!(held.sampling.required_channel.unwrap().opening, [2]);
        assert!(admit_to(&m, &p)
            .unwrap()
            .sampling
            .required_channel
            .is_none());
    }
}
