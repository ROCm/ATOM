//! Transport-neutral request preparation: HTTP request → (GenerationPayload, ResponseContext).

use std::sync::Arc;

use axum::response::Response;
use http::HeaderMap;
use tracing::error;
use uuid::Uuid;

use crate::{
    app_context::AppContext,
    protocols::{
        chat::ChatCompletionRequest,
        common::{InputIds, StringOrArray},
        generate::GenerateRequest,
    },
    routers::comm::error,
    tokenizer::{registry::TokenizerRegistry, traits::Tokenizer},
};

pub(crate) mod chat_template;
pub mod generation_payload;
pub(crate) mod parser_factory_lookup;
pub mod response_context;
pub(crate) mod stop_decoder_builder;
pub(crate) mod tool_constraints;

#[cfg(test)]
mod tests;

use chat_template::process_chat_messages;
use generation_payload::{GenerationPayload, LogprobConfig, SamplingParams, StopConfig};
use response_context::{ProtocolRequest, ResponseContext};
use stop_decoder_builder::create_stop_decoder;
use tool_constraints::{filter_chat_request_by_tool_choice, generate_tool_constraints};

pub fn lookup_tokenizer(
    model_id: &str,
    registry: &TokenizerRegistry,
) -> Result<Arc<dyn Tokenizer>, Response> {
    registry.get(model_id).ok_or_else(|| {
        error!(model = %model_id, "Tokenizer not found for model");
        error::internal_error(
            "tokenizer_not_found",
            format!("Tokenizer not found for model: {}", model_id),
        )
    })
}

pub fn prepare_chat(
    req: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
) -> Result<(GenerationPayload, ResponseContext), Response> {
    let model_id_str = model_id.as_deref().ok_or_else(|| {
        error!("model_id not set");
        error::internal_error("model_id_not_set", "model_id not set in request")
    })?;
    let tokenizer = lookup_tokenizer(model_id_str, &components.tokenizer_registry)?;

    let body_ref = filter_chat_request_by_tool_choice(&req);

    let processed = process_chat_messages(&body_ref, &*tokenizer)
        .map_err(|e| error::bad_request("process_messages_failed", e))?;

    let encoding = tokenizer
        .encode(&processed.text, false)
        .map_err(|e| error::internal_error("tokenization_failed", format!("{}", e)))?;
    let token_ids = encoding.token_ids().to_vec();

    let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
        generate_tool_constraints(tools, &req.tool_choice, &req.model)
            .map_err(|e| error::bad_request("invalid_tool_configuration", e))?
    } else {
        None
    };

    let stop_decoder = create_stop_decoder(
        &tokenizer,
        req.stop.as_ref(),
        req.stop_token_ids.as_ref(),
        req.skip_special_tokens,
        req.no_stop_trim,
    );

    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let payload = build_chat_payload(
        request_id.clone(),
        &req,
        &body_ref,
        &processed.text,
        token_ids,
        tool_constraints,
    );

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ctx = ResponseContext {
        original: ProtocolRequest::Chat(Arc::clone(&req)),
        model_id,
        headers,
        original_text: Some(processed.text.clone()),
        processed_messages: Some(processed),
        tokenizer,
        stop_decoder,
        request_id,
        created,
        tool_parser_factory: components.tool_parser_factory.clone(),
        reasoning_parser_factory: components.reasoning_parser_factory.clone(),
        configured_tool_parser: components.configured_tool_parser.clone(),
        configured_reasoning_parser: components.configured_reasoning_parser.clone(),
    };

    Ok((payload, ctx))
}

pub fn prepare_generate(
    req: Arc<GenerateRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
) -> Result<(GenerationPayload, ResponseContext), Response> {
    let model_id_str = model_id.as_deref().ok_or_else(|| {
        error!("model_id not set");
        error::internal_error("model_id_not_set", "model_id not set in request")
    })?;
    let tokenizer = lookup_tokenizer(model_id_str, &components.tokenizer_registry)?;

    let (original_text, token_ids) = resolve_generate_input(&req, &tokenizer)
        .map_err(|e| error::bad_request("resolve_input_failed", e))?;

    let params = req.sampling_params.as_ref();
    let stop_decoder = create_stop_decoder(
        &tokenizer,
        params.and_then(|p| p.stop.as_ref()),
        params.and_then(|p| p.stop_token_ids.as_ref()),
        params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
        params.and_then(|p| p.no_stop_trim).unwrap_or(false),
    );

    let request_id = req
        .rid
        .clone()
        .unwrap_or_else(|| format!("gen-{}", Uuid::new_v4()));
    let payload =
        build_generate_payload(request_id.clone(), &req, original_text.clone(), token_ids);

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let ctx = ResponseContext {
        original: ProtocolRequest::Generate(Arc::clone(&req)),
        model_id,
        headers,
        original_text,
        processed_messages: None,
        tokenizer,
        stop_decoder,
        request_id,
        created,
        tool_parser_factory: components.tool_parser_factory.clone(),
        reasoning_parser_factory: components.reasoning_parser_factory.clone(),
        configured_tool_parser: components.configured_tool_parser.clone(),
        configured_reasoning_parser: components.configured_reasoning_parser.clone(),
    };

    Ok((payload, ctx))
}

pub(crate) fn build_chat_payload(
    request_id: String,
    req: &ChatCompletionRequest,
    body_ref: &ChatCompletionRequest,
    text: &str,
    token_ids: Vec<u32>,
    tool_constraints: Option<(String, String)>,
) -> GenerationPayload {
    use crate::protocols::common::{ToolChoice, ToolChoiceValue};

    // When tools are present and tool_choice is not "none", force-disable
    // skip_special_tokens so tool-call markers survive the decoder.
    let skip_special_tokens = if req.tools.is_some() {
        match &req.tool_choice {
            Some(ToolChoice::Value(ToolChoiceValue::None)) => req.skip_special_tokens,
            Some(_) | None => false,
        }
    } else {
        req.skip_special_tokens
    };

    GenerationPayload {
        request_id,
        text: text.to_string(),
        token_ids,
        sampling: SamplingParams {
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k.unwrap_or(-1),
            min_p: req.min_p.unwrap_or(0.0),
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            presence_penalty: req.presence_penalty.unwrap_or(0.0),
            repetition_penalty: req.repetition_penalty.unwrap_or(1.0),
            max_new_tokens: req.max_completion_tokens.map(|v| v as i32),
            min_new_tokens: 0,
            n: req.n.unwrap_or(1) as i32,
            ignore_eos: req.ignore_eos,
        },
        stop: StopConfig {
            stop: body_ref.stop.clone(),
            stop_token_ids: req.stop_token_ids.clone(),
            skip_special_tokens,
            no_stop_trim: req.no_stop_trim,
        },
        logprob: LogprobConfig {
            return_logprob: req.logprobs,
            top_logprobs_num: req.top_logprobs.unwrap_or(0) as u32,
            logprob_start_len: -1,
            token_ids_logprob: Vec::new(),
            input_logprobs: false,
        },
        tool_constraints,
        pd_metadata: None,
        stream: req.stream,
        return_hidden_states: req.return_hidden_states,
        log_metrics: false,
    }
}

pub(crate) fn build_generate_payload(
    request_id: String,
    req: &GenerateRequest,
    original_text: Option<String>,
    token_ids: Vec<u32>,
) -> GenerationPayload {
    let params = req.sampling_params.as_ref();
    let stop: Option<StringOrArray> = params.and_then(|p| p.stop.clone());
    let stop_token_ids = params.and_then(|p| p.stop_token_ids.clone());

    GenerationPayload {
        request_id,
        text: original_text.unwrap_or_default(),
        token_ids,
        sampling: SamplingParams {
            temperature: params.and_then(|p| p.temperature).unwrap_or(1.0),
            top_p: params.and_then(|p| p.top_p).unwrap_or(1.0),
            top_k: params.and_then(|p| p.top_k).unwrap_or(-1),
            min_p: params.and_then(|p| p.min_p).unwrap_or(0.0),
            frequency_penalty: params.and_then(|p| p.frequency_penalty).unwrap_or(0.0),
            presence_penalty: params.and_then(|p| p.presence_penalty).unwrap_or(0.0),
            repetition_penalty: params.and_then(|p| p.repetition_penalty).unwrap_or(1.0),
            max_new_tokens: params.and_then(|p| p.max_new_tokens).map(|v| v as i32),
            min_new_tokens: params
                .and_then(|p| p.min_new_tokens)
                .and_then(|v| i32::try_from(v).ok())
                .unwrap_or(0),
            n: params.and_then(|p| p.n).unwrap_or(1) as i32,
            ignore_eos: params.and_then(|p| p.ignore_eos).unwrap_or(false),
        },
        stop: StopConfig {
            stop,
            stop_token_ids,
            skip_special_tokens: params.and_then(|p| p.skip_special_tokens).unwrap_or(true),
            no_stop_trim: params.and_then(|p| p.no_stop_trim).unwrap_or(false),
        },
        logprob: LogprobConfig {
            return_logprob: req.return_logprob.unwrap_or(false),
            top_logprobs_num: req.top_logprobs_num.unwrap_or(0).max(0) as u32,
            logprob_start_len: req.logprob_start_len.unwrap_or(-1),
            token_ids_logprob: req.token_ids_logprob.clone().unwrap_or_default(),
            input_logprobs: req.logprob_start_len.unwrap_or(-1) >= 0,
        },
        tool_constraints: None,
        pd_metadata: None,
        stream: req.stream,
        return_hidden_states: req.return_hidden_states,
        log_metrics: req.log_metrics,
    }
}

fn resolve_generate_input(
    req: &GenerateRequest,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<(Option<String>, Vec<u32>), String> {
    if let Some(text) = &req.text {
        let encoding = tokenizer
            .encode(text, false)
            .map_err(|e| format!("Tokenization failed: {}", e))?;
        return Ok((Some(text.clone()), encoding.token_ids().to_vec()));
    }

    if let Some(input_ids) = &req.input_ids {
        return match input_ids {
            InputIds::Single(ids) => ids
                .iter()
                .map(|&id| u32::try_from(id))
                .collect::<Result<Vec<u32>, _>>()
                .map(|converted| (None, converted))
                .map_err(|_| "input_ids must be non-negative".to_string()),
            InputIds::Batch(_) => {
                Err("Batch input_ids are not supported over gRPC generate yet".to_string())
            }
        };
    }

    Err("Either `text` or `input_ids` must be provided".to_string())
}
