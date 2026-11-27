//! Transport-neutral request preparation: HTTP request → (GenerationPayload, ResponseContext).

use std::sync::Arc;

use axum::response::Response;
use http::HeaderMap;
use tracing::error;
use uuid::Uuid;

use crate::{
    app_context::AppContext,
    core::{
        placement::{types::RenderedTemplate, PlacementError},
        steps::tokenizer_registration::loaded_template,
        worker::RuntimeType,
    },
    protocols::{
        chat::ChatCompletionRequest,
        common::{InputIds, StringOrArray},
        generate::GenerateRequest,
    },
    routers::comm::{error, placement_response::placement_err_to_response},
    tokenizer::{
        registry::{TokenizerEntry, TokenizerRegistry},
        traits::Tokenizer,
    },
};

pub(crate) mod chat_render;
pub(crate) mod chat_template;
pub(crate) mod fast_tokenizer;
pub mod generation_payload;
pub(crate) mod parser_factory_lookup;
pub mod response_context;
pub(crate) mod stop_decoder_builder;
pub(crate) mod thread_tokenizer;
pub(crate) mod tool_constraints;

#[cfg(test)]
mod tests;

use chat_template::{
    atom_chat_render_gap, default_template_kwargs, process_chat_messages_with_defaults, AtomChat,
    ProcessedMessages,
};
use generation_payload::{GenerationPayload, LogprobConfig, SamplingParams, StopConfig};
use response_context::{ProtocolRequest, ResponseContext};
use stop_decoder_builder::create_stop_decoder;
use tool_constraints::{filter_chat_request_by_tool_choice, generate_tool_constraints};

pub fn lookup_tokenizer(
    model_id: &str,
    registry: &TokenizerRegistry,
) -> Result<Arc<dyn Tokenizer>, Response> {
    lookup_tokenizer_entry(model_id, registry).map(|entry| entry.tokenizer)
}

/// The tokenizer `TokenizerRegistry::get` finds, with the id it was loaded under.
fn lookup_tokenizer_entry(
    model_id: &str,
    registry: &TokenizerRegistry,
) -> Result<TokenizerEntry, Response> {
    let entry = registry.get_by_name(model_id);
    entry
        .or_else(|| registry.get_by_id(model_id))
        .ok_or_else(|| {
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
    prepare_chat_with(req, headers, model_id, components, None, false)
}

/// A chat's body as the client `sent` it (JSON bytes), for
/// `prepare_chat_with`, where `model` renders with an encoder of its own
/// (`chat_render`): the typed request drops or rewrites fields that encoder
/// reads (`thinking`, a system message's `tools`, `tool_choice`). A chat
/// whose model renders with a template is not parsed again.
pub fn sent_chat_body(
    sent: &[u8],
    model: Option<&str>,
    components: &AppContext,
) -> Result<Option<Arc<serde_json::Map<String, serde_json::Value>>>, Response> {
    if model.and_then(|m| components.chat_controls_of(m)).is_none() {
        return Ok(None);
    }
    serde_json::from_slice(sent)
        .map(|body| Some(Arc::new(body)))
        .map_err(|e| error::bad_request("invalid_request", e.to_string()))
}

/// `prepare_chat`, rendering the prompt (`chat_render::render_chat`) with the
/// chat's body as `sent` where the caller has it, and, with
/// `reads_prompt_opener`, naming the opener the prompt ends on
/// (`ResponseContext::prompt_opener`).
pub fn prepare_chat_with(
    req: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
    sent: Option<&serde_json::Map<String, serde_json::Value>>,
    reads_prompt_opener: bool,
) -> Result<(GenerationPayload, ResponseContext), Response> {
    let model_id_str = model_id.as_deref().ok_or_else(|| {
        error!("model_id not set");
        error::internal_error("model_id_not_set", "model_id not set in request")
    })?;
    // The tokenizer and what it was loaded with, read in one lookup: a model's last
    // worker leaving drops it, and a tokenizer job may publish it between two.
    let entry = lookup_tokenizer_entry(model_id_str, &components.tokenizer_registry)?;
    let tokenizer = entry.tokenizer.clone();
    let workers = components.worker_registry.get_by_model(model_id_str);
    // What its tokenizer was loaded with: a template file is a template, as
    // --chat-template is; a model with encoder controls renders without one.
    let id = entry.id;
    let controls = components.chat_controls.read().get(&id).copied();
    let templated = components.router_config.chat_template.is_some()
        || components.tokenizer_template_files.read().contains_key(&id);
    if let Some(gap) = controls
        .is_none()
        .then(|| atom_chat_render_gap(&workers, templated))
        .flatten()
    {
        return Err(error::not_implemented(
            "chat_template_unavailable",
            format!(
                "Chats for model {model_id_str} are not served over gRPC: {gap}. Route its \
                 chats to ATOM over HTTP; completions are served over gRPC."
            ),
        ));
    }
    let default_kwargs = default_template_kwargs(&workers)
        .map_err(|e| error::internal_error("invalid_default_template_kwargs", e))?;

    let body_ref = filter_chat_request_by_tool_choice(&req);
    let atom = workers
        .iter()
        .any(|worker| worker.metadata().runtime_type == RuntimeType::Atom)
        .then(|| AtomChat::of(&workers))
        .transpose()
        .map_err(|e| error::internal_error("invalid_reasoning_toggle", e))?;

    let (text, prompt_opener) = match controls {
        Some(_) => chat_render::render_chat(&body_ref, sent, &*tokenizer, controls),
        None => process_chat_messages_with_defaults(
            &body_ref,
            &*tokenizer,
            default_kwargs.as_ref(),
            atom.as_ref(),
        )
        .map(|processed| {
            let opener = response_context::prompt_opener(&processed.text);
            (processed.text, opener)
        }),
    }
    .map_err(|e| error::bad_request("process_messages_failed", e))?;
    let processed = ProcessedMessages {
        text,
        stop_sequences: body_ref.stop.clone(),
    };
    let encoding = tokenizer
        .encode(&processed.text, false)
        .map_err(|e| error::internal_error("tokenization_failed", format!("{}", e)))?;
    let token_ids = encoding.token_ids().to_vec();
    let prompt_opener = prompt_opener.filter(|_| reads_prompt_opener);

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
    let mut payload = build_chat_payload(
        request_id.clone(),
        &req,
        &body_ref,
        &processed.text,
        token_ids,
        tool_constraints,
    );
    (payload.dp_session_id, payload.dp_parent_session_id) = dp_session_ids(headers.as_ref());
    if controls.is_none() {
        // Rendered with the template: only the ATOM workers serving it read these ids.
        let fingerprint = loaded_template(components, &id).ok_or_else(|| {
            let err = PlacementError::ChatTemplateChanged;
            placement_err_to_response(err, Some(model_id_str))
        })?;
        let loaded_file = components.tokenizer_template_files.read().get(&id).cloned();
        payload.chat_template = Some(RenderedTemplate {
            fingerprint,
            fallback_file: loaded_file.or_else(|| components.router_config.chat_template.clone()),
        });
    }

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
        prompt_opener,
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
    let mut payload =
        build_generate_payload(request_id.clone(), &req, original_text.clone(), token_ids);
    (payload.dp_session_id, payload.dp_parent_session_id) = dp_session_ids(headers.as_ref());

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
        prompt_opener: None,
    };

    Ok((payload, ctx))
}

/// The session headers ATOM's HTTP routes read, in their order of preference
/// (`_get_dp_session_affinity_ids`), and the parent session's.
const SESSION_HEADERS: [&str; 3] = ["x-dynamo-session-id", "x-session-id", "x-correlation-id"];
const PARENT_SESSION_HEADER: &str = "x-dynamo-parent-session-id";

/// A request's DP session and parent session ids; an empty header is absent.
pub(crate) fn dp_session_ids(headers: Option<&HeaderMap>) -> (Option<String>, Option<String>) {
    let header = |name: &str| {
        headers?
            .get(name)
            .and_then(|v| v.to_str().ok())
            .filter(|v| !v.is_empty())
            .map(str::to_string)
    };
    (
        SESSION_HEADERS.iter().find_map(|name| header(name)),
        header(PARENT_SESSION_HEADER),
    )
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
        dp_session_id: None,
        dp_parent_session_id: None,
        chat_template: None,
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
        dp_session_id: None,
        dp_parent_session_id: None,
        chat_template: None,
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

pub mod inference;
