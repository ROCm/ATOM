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
    core::{
        placement::{types::RenderedTemplate, PlacementError},
        steps::tokenizer_registration::loaded_template,
        worker::RuntimeType,
        Worker,
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
    process_chat_messages_with_kwargs, ProcessedMessages,
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

/// A chat's body as the client `sent` it (JSON bytes), for `prepare_chat_with`,
/// where `model` renders with an encoder of its own (`chat_render`) or has an ATOM
/// worker, which reads the tools any message declares and `thinking`
/// (`MessageTools`, `read_thinking`): the typed request drops or rewrites the
/// fields these read (`thinking`, a system message's `tools`, `tool_choice`). Any
/// other chat is not parsed again.
pub fn sent_chat_body(
    sent: &[u8],
    model: Option<&str>,
    components: &AppContext,
) -> Result<Option<Arc<serde_json::Map<String, Value>>>, Response> {
    let kept = model.is_some_and(|model| {
        components.chat_controls_of(model).is_some()
            || has_atom_worker(&components.worker_registry.get_by_model(model))
    });
    if !kept {
        return Ok(None);
    }
    serde_json::from_slice(sent)
        .map(|body| Some(Arc::new(body)))
        .map_err(|e| error::bad_request("invalid_request", e.to_string()))
}

/// Whether one of a model's `workers` is an ATOM worker.
fn has_atom_worker(workers: &[Arc<dyn Worker>]) -> bool {
    workers
        .iter()
        .any(|worker| worker.metadata().runtime_type == RuntimeType::Atom)
}

/// `prepare_chat`, rendering the prompt with the chat's body as `sent` where the
/// caller has it: an encoder reads it (`chat_render::render_chat`), and for a model
/// with an ATOM worker, so do the tools its messages declare and its `thinking`.
/// With `reads_prompt_opener`, names the opener the prompt ends on
/// (`ResponseContext::prompt_opener`).
pub fn prepare_chat_with(
    req: Arc<ChatCompletionRequest>,
    headers: Option<HeaderMap>,
    model_id: Option<String>,
    components: &AppContext,
    sent: Option<&serde_json::Map<String, Value>>,
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
    let atom = has_atom_worker(&workers);
    // Only ATOM reads the tools a message declares, and `thinking`: from the chat as
    // sent where the caller has it.
    let message_tools = match (atom, sent) {
        (false, _) => None,
        (true, Some(sent)) => Some(
            MessageTools::read(sent).map_err(|e| error::bad_request("invalid_message_tools", e))?,
        ),
        (true, None) => Some(MessageTools::of(&req)),
    }
    .filter(|declared| !declared.is_empty() || declared.1.is_some() || declared.2.is_some());
    let thinking = sent.filter(|_| atom).and_then(read_thinking);

    let template_kwargs = if atom {
        let toggle = reasoning_toggle(&workers)
            .map_err(|e| error::internal_error("invalid_reasoning_toggle", e))?;
        atom_template_kwargs(
            &body_ref,
            thinking.as_ref(),
            default_kwargs.as_ref(),
            toggle,
        )
        .map_err(|e| error::bad_request("invalid_reasoning_effort", e))?
    } else {
        merged_template_kwargs(&body_ref, default_kwargs.as_ref())
    };
    let (text, rendered_opener) = match controls {
        Some(_) => chat_render::render_chat(&body_ref, sent, &*tokenizer, controls)
            .map(|(text, opener)| (text, Some(opener))),
        None => staged("template", || {
            process_chat_messages_with_kwargs(
                &body_ref,
                &*tokenizer,
                &template_kwargs,
                message_tools.as_ref(),
            )
        })
        .map(|processed| (processed.text, None)),
    }
    .map_err(|e| error::bad_request("process_messages_failed", e))?;
    let processed = ProcessedMessages {
        text,
        stop_sequences: body_ref.stop.clone(),
    };
    let encoding = staged("encode", || tokenizer.encode(&processed.text, false))
        .map_err(|e| error::internal_error("tokenization_failed", format!("{}", e)))?;
    let token_ids = encoding.token_ids().to_vec();

    let tool_constraints = if let Some(tools) = body_ref.tools.as_ref() {
        generate_tool_constraints(tools, &req.tool_choice, &req.model)
            .map_err(|e| error::bad_request("invalid_tool_configuration", e))?
    } else {
        None
    };

    let prompt_opener = match rendered_opener {
        // A prompt its encoder rendered names its own opener.
        Some(opener) if atom || reads_prompt_opener => opener,
        Some(_) => None,
        None if atom => {
            let markers = reasoning_open_markers(&workers)
                .map_err(|e| error::internal_error("invalid_structured_output", e))?;
            response_context::prompt_opener(&processed.text, &markers)
        }
        None if reads_prompt_opener => {
            let think = [response_context::THINK_OPEN.to_string()];
            response_context::prompt_opener(&processed.text, &think)
        }
        None => None,
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
    // The reply's tool calls are read against every tool the chat declares; none
    // declared, ATOM's HTTP chat parses none.
    let rendered_for = match &message_tools {
        Some(declared) => Arc::new(ChatCompletionRequest {
            tools: Some(declared.with_top_level(req.tools.as_ref())).filter(|t| !t.is_empty()),
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
        chat_template: None,
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

/// The efforts ATOM's chat templates read (`VALID_TEMPLATE_EFFORTS` in
/// atom/entrypoints/openai/reasoning.py); ATOM's HTTP chat drops any other.
const ATOM_TEMPLATE_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// An ATOM worker's reasoning toggle, `[name, off value, on value]`, from GetModelInfo.
fn reasoning_toggle(workers: &[Arc<dyn Worker>]) -> Result<Option<(String, Value, Value)>, String> {
    let labels = workers.iter().map(|w| &w.metadata().labels);
    labels
        .filter_map(|labels| labels.get(REASONING_TOGGLE_LABEL))
        .next()
        .map(|json| {
            serde_json::from_str(json)
                .map_err(|e| format!("Invalid {REASONING_TOGGLE_LABEL} label {json}: {e}"))
        })
        .transpose()
}

/// The kwargs ATOM's HTTP chat renders `request` with: the defaults, the request's own,
/// then its `thinking` and `reasoning_effort` resolved (`resolve_reasoning_controls`)
/// and applied (`apply_reasoning_controls`): a stated toggle sets `toggle`, an effort
/// becomes `thinking_effort`.
fn atom_template_kwargs(
    request: &ChatCompletionRequest,
    thinking: Option<&Value>,
    default_kwargs: Option<&HashMap<String, Value>>,
    toggle: Option<(String, Value, Value)>,
) -> Result<HashMap<String, Value>, String> {
    let mut kwargs = default_kwargs.cloned().unwrap_or_default();
    if let Some(own) = &request.chat_template_kwargs {
        kwargs.extend(own.iter().map(|(key, value)| (key.clone(), value.clone())));
    }
    let (enabled, effort) = reasoning_controls(thinking, request.reasoning_effort.as_deref())?;
    if let (Some((name, off, on)), Some(enabled)) = (toggle, enabled) {
        kwargs.insert(name, if enabled { on } else { off });
    }
    if let Some(effort) = effort {
        kwargs.insert("thinking_effort".into(), effort);
    }
    Ok(kwargs)
}

/// Whether thinking is on, if stated, and the effort, as ATOM's HTTP chat resolves its
/// `thinking` and `reasoning_effort` (`resolve_reasoning_controls`, chat rules).
fn reasoning_controls(
    thinking: Option<&Value>,
    effort: Option<&str>,
) -> Result<(Option<bool>, Option<Value>), String> {
    let field = |name: &str| {
        let value = thinking.and_then(Value::as_object)?.get(name);
        value.filter(|value| !value.is_null())
    };
    let mut enabled = field("type").map(|kind| kind.as_str() != Some("disabled"));
    if effort == Some("none") {
        enabled = Some(false);
    }
    let effort = field("effort")
        .cloned()
        .or_else(|| effort.map(|e| Value::String(e.into())));
    let effort = match effort {
        Some(Value::Bool(_)) => {
            return Err("reasoning effort must be an integer in [1, 100]".into())
        }
        Some(Value::Number(n)) if n.is_f64() => {
            return Err("reasoning effort must be an integer in [1, 100]".into())
        }
        Some(Value::Number(n)) => match n.as_i64() {
            Some(1..=100) => Some(Value::Number(n)),
            _ => return Err("reasoning effort must be an integer in [1, 100]".into()),
        },
        Some(Value::String(e)) if ATOM_TEMPLATE_EFFORTS.contains(&e.as_str()) => {
            Some(Value::String(e))
        }
        _ => None,
    };
    Ok((enabled, effort))
}

/// A chat's `thinking` control as `sent`, which the typed request has no field for.
pub fn read_thinking(sent: &serde_json::Map<String, Value>) -> Option<Value> {
    sent.get("thinking")
        .filter(|value| !value.is_null())
        .cloned()
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
    // Namespaced top-level tools go as sent: the typed ones lost their namespace.
    let top_level = message_tools.and_then(|declared| declared.1.as_deref());
    let tools_json = match top_level {
        Some(sent) => Some(to_json(sent)?),
        None => req.tools.as_ref().map(to_json).transpose()?,
    };
    // A namespaced named choice goes as sent too, so ATOM names it as it names the tool.
    let tool_choice_json = match message_tools.and_then(|declared| declared.2.as_deref()) {
        Some(sent) => Some(to_json(sent)?),
        None => req.tool_choice.as_ref().map(to_json).transpose()?,
    };
    Ok(StructuredOutput {
        tools_json,
        tool_choice_json,
        response_format_json: req.response_format.as_ref().map(to_json).transpose()?,
        message_tools_json: message_tools
            .filter(|declared| !declared.is_empty())
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
        chat_template: None,
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
