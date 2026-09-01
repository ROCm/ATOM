use std::time::{SystemTime, UNIX_EPOCH};

use crate::{
    proto::engine::{
        engine_core_envelope::Payload, AddRequest, EngineCoreEnvelope, SamplingParameters,
        Sequence, SequenceStatus, SequenceType, TokenSequence,
    },
    protocols::common::StringOrArray,
    routers::prepare::generation_payload::{GenerationPayload, DEFAULT_MAX_OUTPUT_TOKENS},
};

pub const ENGINE_CORE_WIRE_VERSION: u32 = 1;

pub fn encode_add_request(
    payload: &GenerationPayload,
    sequence_id: i64,
    block_size: i32,
) -> Result<EngineCoreEnvelope, String> {
    encode_add_requests(payload, &[sequence_id], block_size)
}

pub fn encode_add_requests(
    payload: &GenerationPayload,
    sequence_ids: &[i64],
    block_size: i32,
) -> Result<EngineCoreEnvelope, String> {
    encode_add_requests_with_stops(payload, sequence_ids, block_size, &[])
}

pub fn encode_add_requests_with_stops(
    payload: &GenerationPayload,
    sequence_ids: &[i64],
    block_size: i32,
    encoded_stop_sequences: &[Vec<u32>],
) -> Result<EngineCoreEnvelope, String> {
    encode_add_requests_configured(
        payload,
        sequence_ids,
        block_size,
        encoded_stop_sequences,
        0,
        false,
        None,
    )
}

pub fn encode_add_requests_configured(
    payload: &GenerationPayload,
    sequence_ids: &[i64],
    block_size: i32,
    encoded_stop_sequences: &[Vec<u32>],
    num_draft_tokens: i32,
    has_per_req_cache: bool,
    data_parallel_rank: Option<i32>,
) -> Result<EngineCoreEnvelope, String> {
    let arrive_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock is before Unix epoch: {error}"))?
        .as_secs_f64();
    encode_add_requests_at(
        payload,
        sequence_ids,
        block_size,
        encoded_stop_sequences,
        num_draft_tokens,
        has_per_req_cache,
        data_parallel_rank,
        arrive_time,
    )
}

fn encode_add_requests_at(
    payload: &GenerationPayload,
    sequence_ids: &[i64],
    block_size: i32,
    encoded_stop_sequences: &[Vec<u32>],
    num_draft_tokens: i32,
    has_per_req_cache: bool,
    data_parallel_rank: Option<i32>,
    arrive_time: f64,
) -> Result<EngineCoreEnvelope, String> {
    if payload.token_ids.is_empty() {
        return Err("EngineCore request requires at least one prompt token".to_string());
    }
    if payload.sampling.n <= 0 || payload.sampling.n as usize != sequence_ids.len() {
        return Err(format!(
            "sampling n={} does not match {} allocated sequence IDs",
            payload.sampling.n,
            sequence_ids.len()
        ));
    }
    if payload.stream && payload.sampling.n > 1 {
        return Err(
            "streaming n>1 is not supported because TokenChunk partials have no choice index"
                .to_string(),
        );
    }
    if payload.tool_constraints.is_some() {
        return Err("Rust EngineCore transport does not yet support tool constraints".to_string());
    }
    if payload.return_hidden_states {
        return Err("Rust EngineCore transport does not yet support hidden states".to_string());
    }
    if payload.sampling.min_p != 0.0
        || payload.sampling.frequency_penalty != 0.0
        || payload.sampling.presence_penalty != 0.0
        || payload.sampling.repetition_penalty != 1.0
        || payload.sampling.min_new_tokens != 0
    {
        return Err(
            "the EngineCore protobuf schema cannot represent one or more requested sampling \
             parameters (min_p, penalties, or min_new_tokens)"
                .to_string(),
        );
    }
    if payload.logprob.return_logprob
        || payload.logprob.input_logprobs
        || payload.logprob.logprob_start_len >= 0
        || !payload.logprob.token_ids_logprob.is_empty()
    {
        return Err(
            "the EngineCore protobuf stream cannot represent requested output, input, or \
             selective logprobs"
                .to_string(),
        );
    }

    let token_ids = payload
        .token_ids
        .iter()
        .map(|&token| {
            i32::try_from(token)
                .map_err(|_| format!("token id {token} exceeds EngineCore sint32 range"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut stop_token_sequences = payload
        .stop
        .stop_token_ids
        .as_deref()
        .unwrap_or_default()
        .iter()
        .map(|&token| {
            i32::try_from(token)
                .map(|token| TokenSequence {
                    values: vec![token],
                })
                .map_err(|_| format!("stop token id {token} exceeds EngineCore sint32 range"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for sequence in encoded_stop_sequences {
        let values = sequence
            .iter()
            .map(|&token| {
                i32::try_from(token)
                    .map_err(|_| format!("stop token id {token} exceeds EngineCore sint32 range"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !values.is_empty() {
            stop_token_sequences.push(TokenSequence { values });
        }
    }
    let stop_strings = match payload.stop.stop.as_ref() {
        Some(StringOrArray::String(stop)) => vec![stop.clone()],
        Some(StringOrArray::Array(stops)) => stops.clone(),
        None => Vec::new(),
    };
    let prompt_tokens = i32::try_from(token_ids.len())
        .map_err(|_| "prompt token count exceeds EngineCore int32 range".to_string())?;
    let base_sequence = Sequence {
        id: 0,
        external_request_id: None,
        status: SequenceStatus::Waiting as i32,
        r#type: SequenceType::Dummy as i32,
        block_size,
        last_token: *token_ids.last().expect("non-empty checked above"),
        num_tokens: prompt_tokens,
        num_prompt_tokens: prompt_tokens,
        per_req_cache_group: -1,
        state_fork_src: -1,
        arrive_time,
        sampling: Some(SamplingParameters {
            temperature: f64::from(payload.sampling.temperature),
            top_k: payload.sampling.top_k,
            top_p: f64::from(payload.sampling.top_p),
            max_tokens: payload
                .sampling
                .max_new_tokens
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS),
            ignore_eos: payload.sampling.ignore_eos,
            stop_strings,
            return_logprobs: payload.logprob.return_logprob,
            n: payload.sampling.n,
            logprobs: payload
                .logprob
                .return_logprob
                .then_some(payload.logprob.top_logprobs_num as i32),
        }),
        token_ids,
        stop_token_sequences,
        num_draft_tokens,
        has_per_req_cache,
        data_parallel_rank,
        ..Default::default()
    };
    let sequences = sequence_ids
        .iter()
        .enumerate()
        .map(|(sibling_index, &id)| {
            let mut sequence = base_sequence.clone();
            sequence.id = id;
            sequence.external_request_id =
                (sequence_ids.len() == 1).then(|| payload.request_id.clone());
            sequence.parent_request_id =
                (sequence_ids.len() > 1).then(|| payload.request_id.clone());
            sequence.sibling_index = sibling_index as i32;
            sequence.needs_independent_noise = sequence_ids.len() > 1;
            sequence
        })
        .collect();

    Ok(EngineCoreEnvelope {
        wire_version: ENGINE_CORE_WIRE_VERSION,
        payload: Some(Payload::AddRequest(AddRequest { sequences })),
    })
}

#[cfg(test)]
mod tests {
    use prost::Message;

    use super::*;
    use crate::routers::prepare::generation_payload::{LogprobConfig, SamplingParams, StopConfig};

    fn payload() -> GenerationPayload {
        GenerationPayload {
            request_id: "request-1".to_string(),
            text: "hello".to_string(),
            token_ids: vec![1, 2, 3],
            sampling: SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 20,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                max_new_tokens: Some(32),
                min_new_tokens: 0,
                n: 1,
                ignore_eos: false,
            },
            stop: StopConfig {
                stop: Some(StringOrArray::String("stop".to_string())),
                stop_token_ids: Some(vec![99]),
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
            stream: true,
            return_hidden_states: false,
            log_metrics: false,
        }
    }

    #[test]
    fn builds_python_compatible_initial_sequence_state() {
        let envelope = encode_add_request(&payload(), 42, 16).unwrap();
        let Some(Payload::AddRequest(request)) = envelope.payload else {
            panic!("expected AddRequest")
        };
        let sequence = &request.sequences[0];
        assert_eq!(sequence.id, 42);
        assert_eq!(sequence.status, SequenceStatus::Waiting as i32);
        assert_eq!(sequence.r#type, SequenceType::Dummy as i32);
        assert_eq!(sequence.token_ids, vec![1, 2, 3]);
        assert_eq!(sequence.last_token, 3);
        assert_eq!(sequence.num_tokens, 3);
        assert_eq!(sequence.num_prompt_tokens, 3);
        assert_eq!(sequence.per_req_cache_group, -1);
        assert_eq!(sequence.state_fork_src, -1);
        assert_eq!(sequence.block_size, 16);
    }

    #[test]
    fn add_request_matches_python_protobuf_golden_bytes() {
        let envelope =
            encode_add_requests_at(&payload(), &[42], 16, &[], 0, false, None, 123.5).unwrap();
        let actual = envelope
            .encode_to_vec()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            actual,
            "080152570a55082a1209726571756573742d3118022001281032030204064a040a02c601\
             500658036003b80101c00101f9010000000000e05e409a021e09000000606666e63f1014\
             19000000c0ccccec3f2020320473746f704001"
                .replace(char::is_whitespace, "")
        );
    }

    #[test]
    fn defaults_output_limit_and_forwards_explicit_rank() {
        let mut request = payload();
        request.sampling.max_new_tokens = None;
        let envelope =
            encode_add_requests_at(&request, &[42], 16, &[], 0, false, Some(3), 123.5).unwrap();
        let Some(Payload::AddRequest(request)) = envelope.payload else {
            panic!("expected AddRequest")
        };
        let sequence = &request.sequences[0];
        assert_eq!(
            sequence.sampling.as_ref().unwrap().max_tokens,
            DEFAULT_MAX_OUTPUT_TOKENS
        );
        assert_eq!(sequence.data_parallel_rank, Some(3));
    }
}
