//! Transport-neutral request preparation: HTTP request → (GenerationPayload, ResponseContext).

use std::{collections::HashMap, sync::Arc, time::Instant};

use axum::response::Response;
use http::HeaderMap;
use serde::Serialize;
use serde_json::Value;
use tracing::error;
use uuid::Uuid;

use crate::{
    app_context::AppContext,
    core::{worker::RuntimeType, Worker},
    protocols::{
        chat::ChatCompletionRequest,
        common::{InputIds, StringOrArray},
        generate::GenerateRequest,
    },
    routers::comm::error,
    tokenizer::{registry::TokenizerRegistry, traits::Tokenizer},
};

pub(crate) mod chat_template;
pub(crate) mod fast_tokenizer;
pub mod generation_payload;
pub mod message_tools;
pub(crate) mod parser_factory_lookup;
pub mod response_context;
pub(crate) mod stop_decoder_builder;
pub(crate) mod thread_tokenizer;
pub(crate) mod tool_constraints;

#[cfg(test)]
mod tests;

use chat_template::{
    atom_chat_render_gap, default_template_kwargs, merged_template_kwargs,
    process_chat_messages_with_kwargs,
};
use generation_payload::{
    GenerationPayload, LogprobConfig, SamplingParams, StopConfig, StructuredOutput,
};
use message_tools::MessageTools;
use response_context::{ProtocolRequest, ResponseContext};
use stop_decoder_builder::create_stop_decoder;
use tool_constraints::{filter_chat_request_by_tool_choice, generate_tool_constraints};

/// The label with the markers an ATOM worker lists: GetModelInfo's `reasoning_open_markers`.
const REASONING_OPEN_MARKERS_LABEL: &str = "reasoning_open_markers";
/// GetModelInfo's `implicit_reasoning_opener` and `reasoning_toggle_json`, as labels.
const IMPLICIT_REASONING_OPENER_LABEL: &str = "implicit_reasoning_opener";
const REASONING_TOGGLE_LABEL: &str = "reasoning_toggle_json";

/// `work`, its duration recorded under `stage` as the HTTP ingress records its own stages.
fn staged<T>(stage: &'static str, work: impl FnOnce() -> T) -> T {
    let started = Instant::now();
    let done = work();
    metrics::histogram!("mesh_prepare_stage_seconds", "stage" => stage)
        .record(started.elapsed().as_secs_f64());
    done
}

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
    prepare_chat_declaring(req, None, headers, model_id, components)
}

/// `prepare_chat`, with the tools its messages declare read from its raw body
/// (`MessageTools::read`); without them, the typed request's.
pub fn prepare_chat_declaring(
    req: Arc<ChatCompletionRequest>,
    message_tools: Option<MessageTools>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
) -> Result<(GenerationPayload, ResponseContext), Response> {
    let model_id_str = model_id.as_deref().ok_or_else(|| {
        error!("model_id not set");
        error::internal_error("model_id_not_set", "model_id not set in request")
    })?;
    let workers = components.worker_registry.get_by_model(model_id_str);
    let configured_template = components.router_config.chat_template.is_some();
    let registry = &components.tokenizer_registry;
    let built_in_encoder = registry
        .get_by_name(model_id_str)
        .or_else(|| registry.get_by_id(model_id_str))
        .is_some_and(|entry| components.chats_without_template.read().contains(&entry.id));
    if let Some(gap) = atom_chat_render_gap(&workers, configured_template, built_in_encoder) {
        return Err(error::not_implemented(
            "chat_template_unavailable",
            format!(
                "Chats for model {model_id_str} are not served over gRPC: {gap}. Route its \
                 chats to ATOM over HTTP; completions are served over gRPC."
            ),
        ));
    }
    let tokenizer = lookup_tokenizer(model_id_str, &components.tokenizer_registry)?;
    let default_kwargs = default_template_kwargs(&workers)
        .map_err(|e| error::internal_error("invalid_default_template_kwargs", e))?;

    let body_ref = filter_chat_request_by_tool_choice(&req);
    let atom = workers
        .iter()
        .any(|worker| worker.metadata().runtime_type == RuntimeType::Atom);
    // Only ATOM reads the tools a message declares.
    let message_tools = atom
        .then(|| message_tools.unwrap_or_else(|| MessageTools::of(&req)))
        .filter(|declared| !declared.is_empty());

    let template_kwargs = merged_template_kwargs(&body_ref, default_kwargs.as_ref());
    let processed = staged("template", || {
        process_chat_messages_with_kwargs(
            &body_ref,
            &*tokenizer,
            &template_kwargs,
            message_tools.as_ref(),
        )
    })
    .map_err(|e| error::bad_request("process_messages_failed", e))?;

    let encoding = staged("encode", || tokenizer.encode(&processed.text, false))
        .map_err(|e| error::internal_error("tokenization_failed", format!("{}", e)))?;
    let token_ids = encoding.token_ids().to_vec();

    let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
        generate_tool_constraints(tools, &req.tool_choice, &req.model)
            .map_err(|e| error::bad_request("invalid_tool_configuration", e))?
    } else {
        None
    };

    let prompt_opener = if atom {
        let markers = reasoning_open_markers(&workers)
            .map_err(|e| error::internal_error("invalid_structured_output", e))?;
        response_context::prompt_opener(&processed.text, &markers)
    } else {
        None
    };
    let structured_output = atom
        .then(|| {
            let opens = prompt_opener.is_some();
            structured_output(&req, message_tools.as_ref(), &template_kwargs, opens)
        })
        .transpose()
        .map_err(|e| error::internal_error("invalid_structured_output", e))?;
    // Else the opener the model's reply starts after unprompted, unless switched off.
    let prompt_opener = match prompt_opener {
        None if atom => implicit_reasoning_opener(&workers, &template_kwargs)
            .map_err(|e| error::internal_error("invalid_reasoning_toggle", e))?,
        opener => opener,
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
        structured_output,
    );
    (payload.dp_session_id, payload.dp_parent_session_id) = dp_session_ids(headers.as_ref());

    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // The reply's tool calls are read against every tool the chat declares.
    let rendered_for = match &message_tools {
        Some(declared) => Arc::new(ChatCompletionRequest {
            tools: Some(declared.with_top_level(req.tools.as_ref())),
            ..(*req).clone()
        }),
        None => Arc::clone(&req),
    };
    let ctx = ResponseContext {
        original: ProtocolRequest::Chat(rendered_for),
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
        reads_as_atom: false,
        usage_on_finish: false,
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
        reads_as_atom: false,
        usage_on_finish: false,
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
    structured_output: Option<StructuredOutput>,
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
        structured_output,
        dp_session_id: None,
        dp_parent_session_id: None,
    }
}

/// The markers an ATOM worker of the model lists as opening reasoning; none if none does.
fn reasoning_open_markers(workers: &[Arc<dyn Worker>]) -> Result<Vec<String>, String> {
    Ok(workers
        .iter()
        .find_map(|worker| worker.metadata().labels.get(REASONING_OPEN_MARKERS_LABEL))
        .map(|json| {
            serde_json::from_str(json)
                .map_err(|e| format!("Invalid {REASONING_OPEN_MARKERS_LABEL} label {json}: {e}"))
        })
        .transpose()?
        .unwrap_or_default())
}

/// The opener an ATOM worker of the model reports its replies start after with no
/// prompt opener (GetModelInfo's `implicit_reasoning_opener`), unless `kwargs` hold the
/// off value of its reasoning toggle: `reasoning_channel` in ATOM's HTTP server.
fn implicit_reasoning_opener(
    workers: &[Arc<dyn Worker>],
    kwargs: &HashMap<String, Value>,
) -> Result<Option<String>, String> {
    let Some(labels) = workers
        .iter()
        .map(|w| &w.metadata().labels)
        .find(|labels| labels.contains_key(IMPLICIT_REASONING_OPENER_LABEL))
    else {
        return Ok(None);
    };
    let toggle: Option<(String, Value, Value)> = labels
        .get(REASONING_TOGGLE_LABEL)
        .map(|json| {
            serde_json::from_str(json)
                .map_err(|e| format!("Invalid {REASONING_TOGGLE_LABEL} label {json}: {e}"))
        })
        .transpose()?;
    let switched_off = toggle.is_some_and(|(name, off, _)| kwargs.get(&name) == Some(&off));
    Ok((!switched_off).then(|| labels[IMPLICIT_REASONING_OPENER_LABEL].clone()))
}

/// What an ATOM worker's chat route derives `req`'s structured output from: its fields
/// as sent, the `kwargs` its prompt rendered with, and whether that prompt opens reasoning.
fn structured_output(
    req: &ChatCompletionRequest,
    message_tools: Option<&MessageTools>,
    kwargs: &HashMap<String, Value>,
    prompt_opens_reasoning: bool,
) -> Result<StructuredOutput, String> {
    Ok(StructuredOutput {
        tools_json: req.tools.as_ref().map(to_json).transpose()?,
        tool_choice_json: req.tool_choice.as_ref().map(to_json).transpose()?,
        response_format_json: req.response_format.as_ref().map(to_json).transpose()?,
        message_tools_json: message_tools
            .map(|declared| to_json(declared.0.as_ref()))
            .transpose()?,
        chat_template_kwargs_json: to_json(kwargs)?,
        prompt_opens_reasoning,
    })
}

fn to_json<T: Serialize + ?Sized>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| e.to_string())
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
        structured_output: None,
        dp_session_id: None,
        dp_parent_session_id: None,
    }
}

fn resolve_generate_input(
    req: &GenerateRequest,
    tokenizer: &Arc<dyn Tokenizer>,
) -> Result<(Option<String>, Vec<u32>), String> {
    if let Some(text) = &req.text {
        let encoding = staged("encode", || tokenizer.encode(text, false))
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
