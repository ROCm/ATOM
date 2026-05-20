//! Part B spike: byte-equality of `to_sglang_proto` vs the upstream
//! `SglangSchedulerClient::build_generate_request_from_chat` builder, for one
//! representative chat scenario.
//!
//! `SglangSchedulerClient` can only be constructed via async `connect()` to a
//! live endpoint, so the oracle below inlines the upstream proto-building
//! logic verbatim from `smg-grpc-client = 1.0.0` (sglang_scheduler.rs lines
//! 305-518). The crate version is pinned in Cargo.toml, so divergence shows
//! up as a build break on version bump — exactly what we want for this spike.

use mesh::protocols::chat::ChatCompletionRequest;
use mesh::protocols::common::StringOrArray;
use mesh::routers::grpc::engine::payload_to_proto::to_sglang_proto;
use mesh::routers::prepare::generation_payload::{
    GenerationPayload, LogprobConfig, SamplingParams, StopConfig,
};
use mesh_grpc::sglang_proto;
use prost::Message;

mod oracle {
    use mesh::protocols::chat::ChatCompletionRequest;
    use mesh::protocols::common::{ResponseFormat, StringOrArray, ToolChoice, ToolChoiceValue};
    use mesh_grpc::sglang_proto as proto;

    pub fn build_generate_request_from_chat(
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        multimodal_inputs: Option<proto::MultimodalInputs>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = build_grpc_sampling_params_from_chat(body, tool_call_constraint)?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: multimodal_inputs,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            ..Default::default()
        })
    }

    fn build_grpc_sampling_params_from_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop = extract_stop_strings(request);
        let max_new_tokens = request.max_completion_tokens.map(|v| v as i32);
        let skip_special_tokens = if request.tools.is_some() {
            match &request.tool_choice {
                Some(ToolChoice::Value(ToolChoiceValue::None)) => request.skip_special_tokens,
                Some(_) => false,
                None => false,
            }
        } else {
            request.skip_special_tokens
        };

        Ok(proto::SamplingParams {
            temperature: request.temperature.unwrap_or(1.0),
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.unwrap_or(-1),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_new_tokens,
            stop,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: request.ignore_eos,
            no_stop_trim: request.no_stop_trim,
            n: request.n.unwrap_or(1) as i32,
            constraint: build_constraint_for_chat(request, tool_call_constraint)?,
            ..Default::default()
        })
    }

    fn extract_stop_strings(request: &ChatCompletionRequest) -> Vec<String> {
        match &request.stop {
            Some(StringOrArray::String(s)) => vec![s.clone()],
            Some(StringOrArray::Array(arr)) => arr.clone(),
            None => vec![],
        }
    }

    fn build_constraint_for_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<Option<proto::sampling_params::Constraint>, String> {
        let mut constraints = Vec::new();

        match &request.response_format {
            Some(ResponseFormat::JsonObject) => {
                let schema = serde_json::json!({"type": "object"});
                let schema_str = serde_json::to_string(&schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str = serde_json::to_string(&json_schema.schema)
                    .map_err(|e| format!("Failed to serialize JSON schema: {}", e))?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::Text) | None => {}
        }

        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::EbnfGrammar(
                ebnf.clone(),
            ));
        }
        if let Some(regex) = &request.regex {
            constraints.push(proto::sampling_params::Constraint::Regex(regex.clone()));
        }

        if let Some((constraint_type, constraint_value)) = tool_call_constraint {
            if !constraints.is_empty() {
                return Err("Constrained decoding is not compatible with tool calls.".to_string());
            }
            let tool_constraint = match constraint_type.as_str() {
                "structural_tag" => {
                    proto::sampling_params::Constraint::StructuralTag(constraint_value)
                }
                "json_schema" => proto::sampling_params::Constraint::JsonSchema(constraint_value),
                "ebnf" => proto::sampling_params::Constraint::EbnfGrammar(constraint_value),
                "regex" => proto::sampling_params::Constraint::Regex(constraint_value),
                _ => return Err(format!("Unknown constraint type: {}", constraint_type)),
            };
            constraints.push(tool_constraint);
        }

        match constraints.len() {
            0 => Ok(None),
            1 => Ok(constraints.pop()),
            _ => Err("Multiple constraints are not allowed.".to_string()),
        }
    }
}

fn scenario_a_constraint() -> Option<(String, String)> {
    Some((
        "json_schema".to_string(),
        r#"{"name":"add","schema":{"type":"object"}}"#.to_string(),
    ))
}

fn scenario_a_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test-model".to_string(),
        temperature: Some(0.3),
        top_p: Some(0.9),
        top_k: Some(50),
        repetition_penalty: Some(1.1),
        max_completion_tokens: Some(64),
        stop: Some(StringOrArray::String("<|im_end|>".to_string())),
        stop_token_ids: Some(vec![151645]),
        skip_special_tokens: true,
        no_stop_trim: false,
        logprobs: true,
        top_logprobs: Some(5),
        ..Default::default()
    }
}

fn scenario_a_payload() -> GenerationPayload {
    GenerationPayload {
        request_id: "req_spike_A".to_string(),
        token_ids: vec![1, 2, 3, 4, 5, 6, 7],
        sampling: SamplingParams {
            temperature: 0.3,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            max_new_tokens: 64,
        },
        stop: StopConfig {
            stop: Some(StringOrArray::String("<|im_end|>".to_string())),
            stop_token_ids: Some(vec![151645]),
            skip_special_tokens: true,
            no_stop_trim: false,
        },
        logprob: LogprobConfig {
            return_logprob: true,
            top_logprobs_num: 5,
            input_logprobs: false,
        },
        tool_constraints: scenario_a_constraint(),
    }
}

#[test]
fn scenario_a_byte_equal() {
    let req = scenario_a_request();
    let payload = scenario_a_payload();
    let processed_text = "<|im_start|>user\nAdd 1+2<|im_end|>".to_string();
    let token_ids = payload.token_ids.clone();
    let constraint = scenario_a_constraint();

    let upstream: sglang_proto::GenerateRequest = oracle::build_generate_request_from_chat(
        payload.request_id.clone(),
        &req,
        processed_text.clone(),
        token_ids,
        None,
        constraint,
    )
    .expect("oracle builds proto");

    let ours: sglang_proto::GenerateRequest = to_sglang_proto(&payload, processed_text, None);

    let upstream_bytes = upstream.encode_to_vec();
    let ours_bytes = ours.encode_to_vec();
    assert_eq!(
        ours_bytes, upstream_bytes,
        "Scenario A byte mismatch:\n  upstream = {:?}\n  ours     = {:?}",
        upstream, ours
    );
}
