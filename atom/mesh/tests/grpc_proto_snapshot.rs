//! Byte-equality test: every `GenerationPayload` round-trips through
//! `to_sglang_proto` / `to_vllm_proto` byte-identical to the corresponding
//! builder in `smg-grpc-client = 1.0.0`.
//!
//! `SglangSchedulerClient` / `VllmEngineClient` can only be constructed via
//! async `connect()`, so the oracles below replicate the builder logic.
//! The crate version is pinned in Cargo.toml; on a version bump, this file
//! is the canary.

use mesh::protocols::chat::ChatCompletionRequest;
use mesh::protocols::common::StringOrArray;
use mesh::protocols::generate::GenerateRequest;
use mesh::routers::grpc::engine::payload_to_proto::{to_sglang_proto, to_vllm_proto};
use mesh::routers::prepare::generation_payload::{
    GenerationPayload, LogprobConfig, PdMetadata, SamplingParams as PayloadSampling, StopConfig,
};
use mesh_grpc::sglang_proto;
use prost::Message;

mod oracle_sglang {
    use mesh::protocols::chat::ChatCompletionRequest;
    use mesh::protocols::common::{ResponseFormat, StringOrArray, ToolChoice, ToolChoiceValue};
    use mesh::protocols::generate::GenerateRequest;
    use mesh::protocols::sampling_params::SamplingParams as PlainSamplingParams;
    use mesh_grpc::sglang_proto as proto;

    pub fn build_generate_request_from_chat(
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = build_grpc_sampling_params_from_chat(body, tool_call_constraint)?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: processed_text,
                input_ids: token_ids,
            }),
            mm_inputs: None,
            sampling_params: Some(sampling_params),
            return_logprob: body.logprobs,
            logprob_start_len: -1,
            top_logprobs_num: body.top_logprobs.unwrap_or(0) as i32,
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            ..Default::default()
        })
    }

    pub fn build_plain_generate_request(
        request_id: String,
        body: &GenerateRequest,
        original_text: Option<String>,
        token_ids: Vec<u32>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = build_sampling_params_from_plain(body.sampling_params.as_ref())?;
        Ok(proto::GenerateRequest {
            request_id,
            tokenized: Some(proto::TokenizedInput {
                original_text: original_text.unwrap_or_default(),
                input_ids: token_ids,
            }),
            sampling_params: Some(sampling_params),
            return_logprob: body.return_logprob.unwrap_or(false),
            logprob_start_len: body.logprob_start_len.unwrap_or(-1),
            top_logprobs_num: body.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: body.token_ids_logprob.clone().unwrap_or_default(),
            return_hidden_states: body.return_hidden_states,
            stream: body.stream,
            log_metrics: body.log_metrics,
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
                Some(_) | None => false,
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

    fn build_sampling_params_from_plain(
        params: Option<&PlainSamplingParams>,
    ) -> Result<proto::SamplingParams, String> {
        let mut sampling = proto::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            repetition_penalty: 1.0,
            n: 1,
            skip_special_tokens: true,
            spaces_between_special_tokens: true,
            ..Default::default()
        };
        let Some(p) = params else { return Ok(sampling) };

        macro_rules! map_field {
            ($field:ident) => {
                if let Some(val) = p.$field {
                    sampling.$field = val;
                }
            };
        }
        map_field!(temperature);
        map_field!(top_p);
        map_field!(top_k);
        map_field!(frequency_penalty);
        map_field!(presence_penalty);
        map_field!(repetition_penalty);
        map_field!(min_p);
        map_field!(ignore_eos);
        map_field!(skip_special_tokens);
        map_field!(no_stop_trim);

        if let Some(stop) = &p.stop {
            match stop {
                StringOrArray::String(s) => sampling.stop.push(s.clone()),
                StringOrArray::Array(arr) => sampling.stop.extend(arr.clone()),
            }
        }
        if let Some(stop_token_ids) = &p.stop_token_ids {
            sampling.stop_token_ids = stop_token_ids.clone();
        }
        if let Some(max_new_tokens) = p.max_new_tokens {
            sampling.max_new_tokens = Some(
                i32::try_from(max_new_tokens)
                    .map_err(|_| "max_new_tokens must fit i32".to_string())?,
            );
        }
        if let Some(min_new_tokens) = p.min_new_tokens {
            sampling.min_new_tokens = i32::try_from(min_new_tokens)
                .map_err(|_| "min_new_tokens must fit i32".to_string())?;
        }
        if let Some(n) = p.n {
            sampling.n = i32::try_from(n).map_err(|_| "n must fit i32".to_string())?;
        }
        Ok(sampling)
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
                let schema_str = serde_json::to_string(&schema).map_err(|e| e.to_string())?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str =
                    serde_json::to_string(&json_schema.schema).map_err(|e| e.to_string())?;
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

mod oracle_vllm {
    use mesh::protocols::chat::ChatCompletionRequest;
    use mesh::protocols::common::{ResponseFormat, StringOrArray, ToolChoice, ToolChoiceValue};
    use mesh_grpc::vllm_proto as proto;

    pub fn build_generate_request_from_chat(
        request_id: String,
        body: &ChatCompletionRequest,
        processed_text: String,
        token_ids: Vec<u32>,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::GenerateRequest, String> {
        let sampling_params = build_grpc_sampling_params_from_chat(body, tool_call_constraint)?;
        Ok(proto::GenerateRequest {
            request_id,
            input: Some(proto::generate_request::Input::Tokenized(
                proto::TokenizedInput {
                    original_text: processed_text,
                    input_ids: token_ids,
                },
            )),
            sampling_params: Some(sampling_params),
            stream: body.stream,
        })
    }

    fn build_grpc_sampling_params_from_chat(
        request: &ChatCompletionRequest,
        tool_call_constraint: Option<(String, String)>,
    ) -> Result<proto::SamplingParams, String> {
        let stop_sequences = extract_stop_strings(request);
        let max_tokens = request.max_completion_tokens;
        let skip_special_tokens = if request.tools.is_some() {
            match &request.tool_choice {
                Some(ToolChoice::Value(ToolChoiceValue::None)) => request.skip_special_tokens,
                Some(_) | None => false,
            }
        } else {
            request.skip_special_tokens
        };
        Ok(proto::SamplingParams {
            temperature: request.temperature,
            top_p: request.top_p.unwrap_or(1.0),
            top_k: request.top_k.map(|v| v.max(0) as u32).unwrap_or(0),
            min_p: request.min_p.unwrap_or(0.0),
            frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
            presence_penalty: request.presence_penalty.unwrap_or(0.0),
            repetition_penalty: request.repetition_penalty.unwrap_or(1.0),
            max_tokens,
            stop: stop_sequences,
            stop_token_ids: request.stop_token_ids.clone().unwrap_or_default(),
            skip_special_tokens,
            spaces_between_special_tokens: true,
            ignore_eos: request.ignore_eos,
            n: request.n.unwrap_or(1),
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
                let schema_str = serde_json::to_string(&schema).map_err(|e| e.to_string())?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::JsonSchema { json_schema }) => {
                let schema_str =
                    serde_json::to_string(&json_schema.schema).map_err(|e| e.to_string())?;
                constraints.push(proto::sampling_params::Constraint::JsonSchema(schema_str));
            }
            Some(ResponseFormat::Text) | None => {}
        }
        if let Some(ebnf) = &request.ebnf {
            constraints.push(proto::sampling_params::Constraint::Grammar(ebnf.clone()));
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
                "grammar" | "ebnf" => proto::sampling_params::Constraint::Grammar(constraint_value),
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

// ============================================================================
// Scenario A — chat + tools + logprobs + non-default sampling (sglang)
// ============================================================================

fn scenario_a_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test".to_string(),
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

fn scenario_a_tool_constraint() -> Option<(String, String)> {
    Some((
        "json_schema".to_string(),
        r#"{"name":"add","schema":{"type":"object"}}"#.to_string(),
    ))
}

fn scenario_a_text() -> String {
    "<|im_start|>user\nAdd 1+2<|im_end|>".to_string()
}

fn scenario_a_token_ids() -> Vec<u32> {
    vec![1, 2, 3, 4, 5, 6, 7]
}

fn scenario_a_payload() -> GenerationPayload {
    let req = scenario_a_request();
    GenerationPayload {
        request_id: "snap-A".to_string(),
        text: scenario_a_text(),
        token_ids: scenario_a_token_ids(),
        sampling: PayloadSampling {
            temperature: 0.3,
            top_p: 0.9,
            top_k: 50,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.1,
            max_new_tokens: Some(64),
            n: 1,
            min_new_tokens: 0,
            ignore_eos: false,
        },
        stop: StopConfig {
            stop: req.stop.clone(),
            stop_token_ids: req.stop_token_ids.clone(),
            // tool_choice is None → upstream forces skip_special_tokens=false when tools are present
            skip_special_tokens: true,
            no_stop_trim: false,
        },
        logprob: LogprobConfig {
            return_logprob: true,
            top_logprobs_num: 5,
            logprob_start_len: -1,
            token_ids_logprob: Vec::new(),
            input_logprobs: false,
        },
        tool_constraints: scenario_a_tool_constraint(),
        pd_metadata: None,
        stream: false,
        return_hidden_states: false,
        log_metrics: false,
    }
}

#[test]
fn scenario_a_sglang_chat_byte_equal() {
    let req = scenario_a_request();
    let payload = scenario_a_payload();
    let upstream = oracle_sglang::build_generate_request_from_chat(
        payload.request_id.clone(),
        &req,
        scenario_a_text(),
        scenario_a_token_ids(),
        scenario_a_tool_constraint(),
    )
    .unwrap();
    assert_eq!(
        to_sglang_proto(&payload).encode_to_vec(),
        upstream.encode_to_vec()
    );
}

// ============================================================================
// Scenario B — plain generate + stop array + input_logprobs (sglang)
// ============================================================================

fn scenario_b_request() -> GenerateRequest {
    // GenerateRequest has 30+ fields and no Default impl; build from a minimal
    // JSON body so the test stays close to the wire format users actually send.
    let mut req: GenerateRequest = serde_json::from_value(serde_json::json!({
        "text": "Tell me about: ",
        "sampling_params": {
            "stop": ["\n\n", "END"],
            "max_new_tokens": 32,
            "no_stop_trim": true
        },
        "return_logprob": false
    }))
    .expect("scenario B json");
    // logprob_start_len defaults to None in our payload; the oracle reads
    // body.logprob_start_len.unwrap_or(-1), so leaving as None aligns with our
    // hardcoded -1.
    req.logprob_start_len = None;
    req
}

fn scenario_b_payload() -> GenerationPayload {
    let req = scenario_b_request();
    GenerationPayload {
        request_id: "snap-B".to_string(),
        text: req.text.clone().unwrap(),
        token_ids: vec![100, 200, 300],
        sampling: PayloadSampling {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_new_tokens: Some(32),
            n: 1,
            min_new_tokens: 0,
            ignore_eos: false,
        },
        stop: StopConfig {
            stop: Some(StringOrArray::Array(vec![
                "\n\n".to_string(),
                "END".to_string(),
            ])),
            stop_token_ids: None,
            skip_special_tokens: true,
            no_stop_trim: true,
        },
        logprob: LogprobConfig {
            return_logprob: false,
            top_logprobs_num: 0,
            logprob_start_len: -1,
            token_ids_logprob: Vec::new(),
            input_logprobs: true,
        },
        tool_constraints: None,
        pd_metadata: None,
        stream: false,
        return_hidden_states: false,
        // GenerateRequest.log_metrics defaults to true (serde default_true);
        // our payload mirrors that for parity with upstream's plain builder.
        log_metrics: true,
    }
}

#[test]
fn scenario_b_sglang_generate_byte_equal() {
    let req = scenario_b_request();
    let payload = scenario_b_payload();
    // Note: build_plain_generate_request reads `logprob_start_len` from the
    // body (defaults to -1). Our payload doesn't carry that field today, so
    // align the oracle's body to match the payload's effective wire output.
    let upstream = oracle_sglang::build_plain_generate_request(
        payload.request_id.clone(),
        &req,
        Some(payload.text.clone()),
        payload.token_ids.clone(),
    )
    .unwrap();
    assert_eq!(
        to_sglang_proto(&payload).encode_to_vec(),
        upstream.encode_to_vec()
    );
}

// ============================================================================
// Scenario C — chat + PD pair bootstrap + n>1 (sglang)
// ============================================================================

fn scenario_c_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test".to_string(),
        temperature: Some(0.7),
        top_p: Some(0.95),
        top_k: Some(100),
        max_completion_tokens: Some(16),
        n: Some(2),
        stop_token_ids: Some(vec![2]),
        skip_special_tokens: false,
        ..Default::default()
    }
}

fn scenario_c_payload() -> GenerationPayload {
    GenerationPayload {
        request_id: "snap-C".to_string(),
        text: "Continue:".to_string(),
        token_ids: vec![7, 8, 9],
        sampling: PayloadSampling {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 100,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_new_tokens: Some(16),
            n: 2,
            min_new_tokens: 0,
            ignore_eos: false,
        },
        stop: StopConfig {
            stop: None,
            stop_token_ids: Some(vec![2]),
            skip_special_tokens: false,
            no_stop_trim: false,
        },
        logprob: LogprobConfig {
            return_logprob: false,
            top_logprobs_num: 0,
            logprob_start_len: -1,
            token_ids_logprob: Vec::new(),
            input_logprobs: false,
        },
        tool_constraints: None,
        pd_metadata: Some(PdMetadata {
            bootstrap_host: "prefill-host".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 0x4_5b4c,
        }),
        stream: false,
        return_hidden_states: false,
        log_metrics: false,
    }
}

#[test]
fn scenario_c_sglang_pd_byte_equal() {
    let req = scenario_c_request();
    let payload = scenario_c_payload();
    // Upstream's build_generate_request_from_chat doesn't accept disagg params;
    // they're injected post-build by inject_bootstrap_metadata. Mirror that.
    let mut upstream = oracle_sglang::build_generate_request_from_chat(
        payload.request_id.clone(),
        &req,
        payload.text.clone(),
        payload.token_ids.clone(),
        None,
    )
    .unwrap();
    let pd = payload.pd_metadata.as_ref().unwrap();
    upstream.disaggregated_params = Some(sglang_proto::DisaggregatedParams {
        bootstrap_host: pd.bootstrap_host.clone(),
        bootstrap_port: pd.bootstrap_port,
        bootstrap_room: pd.bootstrap_room,
    });
    assert_eq!(
        to_sglang_proto(&payload).encode_to_vec(),
        upstream.encode_to_vec()
    );
}

#[test]
fn scenario_c_no_pd_metadata_yields_no_disaggregated_params() {
    let mut payload = scenario_c_payload();
    payload.pd_metadata = None;
    let proto = to_sglang_proto(&payload);
    assert!(proto.disaggregated_params.is_none());
}

// ============================================================================
// Scenario D — chat with vllm-specific sampling (Optional temperature, etc.)
// ============================================================================

fn scenario_d_request() -> ChatCompletionRequest {
    ChatCompletionRequest {
        model: "test".to_string(),
        temperature: Some(0.5),
        top_p: Some(0.8),
        top_k: Some(20),
        min_p: Some(0.05),
        frequency_penalty: Some(0.2),
        presence_penalty: Some(0.1),
        repetition_penalty: Some(1.05),
        max_completion_tokens: Some(8),
        skip_special_tokens: true,
        ..Default::default()
    }
}

fn scenario_d_payload() -> GenerationPayload {
    GenerationPayload {
        request_id: "snap-D".to_string(),
        text: "vLLM test".to_string(),
        token_ids: vec![1, 2],
        sampling: PayloadSampling {
            temperature: 0.5,
            top_p: 0.8,
            top_k: 20,
            min_p: 0.05,
            frequency_penalty: 0.2,
            presence_penalty: 0.1,
            repetition_penalty: 1.05,
            max_new_tokens: Some(8),
            n: 1,
            min_new_tokens: 0,
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
            token_ids_logprob: Vec::new(),
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
fn scenario_d_vllm_chat_byte_equal() {
    let req = scenario_d_request();
    let payload = scenario_d_payload();
    let upstream = oracle_vllm::build_generate_request_from_chat(
        payload.request_id.clone(),
        &req,
        payload.text.clone(),
        payload.token_ids.clone(),
        None,
    )
    .unwrap();
    assert_eq!(
        to_vllm_proto(&payload).encode_to_vec(),
        upstream.encode_to_vec()
    );
}

// ============================================================================
// Field-level smoke tests — protect against silent regressions in adapters
// ============================================================================

#[test]
fn sglang_request_id_threads_through() {
    let mut p = scenario_a_payload();
    p.request_id = "custom-id".to_string();
    assert_eq!(to_sglang_proto(&p).request_id, "custom-id");
}

#[test]
fn sglang_text_threads_through_tokenized_input() {
    let p = scenario_a_payload();
    let proto = to_sglang_proto(&p);
    let tok = proto.tokenized.as_ref().unwrap();
    assert_eq!(tok.original_text, p.text);
    assert_eq!(tok.input_ids, p.token_ids);
}

#[test]
fn sglang_pd_metadata_round_trips() {
    let p = scenario_c_payload();
    let proto = to_sglang_proto(&p);
    let dp = proto.disaggregated_params.as_ref().unwrap();
    assert_eq!(dp.bootstrap_host, "prefill-host");
    assert_eq!(dp.bootstrap_port, 8998);
    assert_eq!(dp.bootstrap_room, 0x4_5b4c);
}

#[test]
fn vllm_top_k_minus_one_maps_to_zero_disable() {
    let mut p = scenario_d_payload();
    p.sampling.top_k = -1;
    let proto = to_vllm_proto(&p);
    assert_eq!(proto.sampling_params.unwrap().top_k, 0);
}

#[test]
fn vllm_temperature_wraps_in_some() {
    let p = scenario_d_payload();
    let proto = to_vllm_proto(&p);
    assert_eq!(proto.sampling_params.unwrap().temperature, Some(0.5));
}

// Scenario B′ — protects against the three "silently dropped" fields the
// rust-reviewer flagged: a generate request that actually sets min_new_tokens,
// token_ids_logprob, and a non-default logprob_start_len.

fn scenario_b_prime_request() -> GenerateRequest {
    let mut req: GenerateRequest = serde_json::from_value(serde_json::json!({
        "text": "Edge fields: ",
        "sampling_params": { "min_new_tokens": 4, "max_new_tokens": 12 },
        "return_logprob": true,
        "logprob_start_len": 7,
        "top_logprobs_num": 3,
        "token_ids_logprob": [10, 20, 30]
    }))
    .expect("scenario B′ json");
    // log_metrics is serde-defaulted to true; keep it.
    let _ = &mut req;
    req
}

fn scenario_b_prime_payload() -> GenerationPayload {
    GenerationPayload {
        request_id: "snap-Bp".to_string(),
        text: "Edge fields: ".to_string(),
        token_ids: vec![1, 2, 3],
        sampling: PayloadSampling {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_new_tokens: Some(12),
            min_new_tokens: 4,
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
            return_logprob: true,
            top_logprobs_num: 3,
            logprob_start_len: 7,
            token_ids_logprob: vec![10, 20, 30],
            input_logprobs: true,
        },
        tool_constraints: None,
        pd_metadata: None,
        stream: false,
        return_hidden_states: false,
        log_metrics: true,
    }
}

#[test]
fn scenario_b_prime_edge_fields_byte_equal() {
    let req = scenario_b_prime_request();
    let payload = scenario_b_prime_payload();
    let upstream = oracle_sglang::build_plain_generate_request(
        payload.request_id.clone(),
        &req,
        Some(payload.text.clone()),
        payload.token_ids.clone(),
    )
    .unwrap();
    assert_eq!(
        to_sglang_proto(&payload).encode_to_vec(),
        upstream.encode_to_vec()
    );
}
