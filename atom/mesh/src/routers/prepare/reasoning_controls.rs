//! The template kwargs ATOM's `/v1/chat/completions` renders a chat with, so
//! HTTP routing tokenizes chats as the engine does: its
//! `resolve_reasoning_controls` and `apply_reasoning_controls` under the rules
//! it picks for the model, with the toggle `resolve_reasoning_toggle` finds,
//! and for DeepSeek V4.1 its encoder hooks (`deepseek_v41_encoder`). A change
//! to any of those needs the same change here.

use std::{collections::HashMap, path::Path};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    protocols::chat::ChatCompletionRequest,
    tokenizer::{chat_template::ChatTemplateParams, traits::Tokenizer},
};

/// How ATOM's chat endpoint applies a model's reasoning controls.
#[derive(Clone, Debug, PartialEq)]
pub enum ReasoningControls {
    /// The model's Jinja template under `CHAT_REASONING_RULES`, switched by the
    /// toggle found for it, if any.
    Template(Option<ReasoningToggle>),
    /// DeepSeek V4.1's encoder under its `REASONING_RULES`. ATOM renders the
    /// checkpoint's `encoding/encoding.py`; the router renders its template
    /// with the kwargs that encoder takes.
    DeepseekV41,
}

impl Default for ReasoningControls {
    fn default() -> Self {
        Self::Template(None)
    }
}

/// A template kwarg that switches the model's reasoning, with its off and on values.
#[derive(Clone, Debug, PartialEq)]
pub struct ReasoningToggle {
    pub name: &'static str,
    pub off: Value,
    pub on: Value,
}

/// ATOM's `REASONING_TOGGLES`, tried in its order.
fn reasoning_toggles() -> [ReasoningToggle; 4] {
    let toggle = |name, off, on| ReasoningToggle { name, off, on };
    [
        toggle("enable_thinking", json!(false), json!(true)),
        toggle("thinking", json!(false), json!(true)),
        toggle("thinking_mode", json!("disabled"), json!("enabled")),
        toggle("thinking_mode", json!("chat"), json!("thinking")),
    ]
}

/// The toggle ATOM's `resolve_reasoning_toggle` finds for DeepSeek V4.1's
/// encoder, which takes no `enable_thinking` or `thinking`.
fn deepseek_v41_toggle() -> ReasoningToggle {
    ReasoningToggle {
        name: "thinking_mode",
        off: json!("chat"),
        on: json!("thinking"),
    }
}

/// How ATOM applies the reasoning controls of chats for the checkpoint at
/// `source`, which `tokenizer` renders: DeepSeek V4.1's hooks where ATOM's
/// `load_custom_message_encoder` loads them, else the template with the toggle
/// that switches it.
pub(crate) fn probe_reasoning_controls(
    source: &str,
    tokenizer: &dyn Tokenizer,
) -> ReasoningControls {
    if loads_deepseek_v41_encoder(Path::new(source)) {
        ReasoningControls::DeepseekV41
    } else {
        ReasoningControls::Template(probe_reasoning_toggle(tokenizer))
    }
}

/// Whether ATOM renders the checkpoint in `dir` with DeepSeek V4.1's hooks: its
/// `encoding/` holds `encoding.py` and no `encoding_*.py`, and its
/// `config.json` names the `deepseek_v41` model type.
fn loads_deepseek_v41_encoder(dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(dir.join("encoding")) else {
        return false;
    };
    let mut encoders = entries.filter_map(|entry| {
        let name = entry.ok()?.file_name().into_string().ok()?;
        let encoder =
            name == "encoding.py" || (name.starts_with("encoding_") && name.ends_with(".py"));
        encoder.then_some(name)
    });
    let standalone = matches!(
        (encoders.next(), encoders.next()),
        (Some(name), None) if name == "encoding.py"
    );
    standalone
        && std::fs::read(dir.join("config.json"))
            .ok()
            .and_then(|config| serde_json::from_slice::<Value>(&config).ok())
            .is_some_and(|config| config["model_type"] == "deepseek_v41")
}

/// The first toggle whose off value changes how `tokenizer` renders a one-turn
/// chat, as ATOM finds it at startup; a template that fails on a value is not
/// switched by it. `None` when no toggle switches the template.
pub(crate) fn probe_reasoning_toggle(tokenizer: &dyn Tokenizer) -> Option<ReasoningToggle> {
    let messages = [json!({"role": "user", "content": "hi"})];
    let render = |template_kwargs: Option<&HashMap<String, Value>>| {
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            template_kwargs,
            ..Default::default()
        };
        tokenizer.apply_chat_template(&messages, params).ok()
    };
    let baseline = render(None)?;
    reasoning_toggles().into_iter().find(|toggle| {
        let kwargs = HashMap::from([(toggle.name.to_string(), toggle.off.clone())]);
        render(Some(&kwargs)).is_some_and(|text| text != baseline)
    })
}

/// The chat fields ATOM renders as the client sent them: the typed request has
/// no `thinking`, and gives `tool_choice` a default when there are tools.
#[derive(Deserialize)]
struct SentControls {
    thinking: Option<Value>,
    tool_choice: Option<Value>,
    response_format: Option<Value>,
}

/// The template kwargs ATOM's chat endpoint renders `request` with, but for the
/// server's `--default-chat-template-kwargs`, which the router does not see:
/// the request's `chat_template_kwargs`, then its `response_format` and a
/// string `tool_choice`, then the reasoning controls as `controls` applies
/// them. `body` is the request as sent.
pub(crate) fn atom_chat_template_kwargs(
    request: &ChatCompletionRequest,
    body: &[u8],
    controls: &ReasoningControls,
) -> Result<HashMap<String, Value>, String> {
    let sent: SentControls = serde_json::from_slice(body).map_err(|e| e.to_string())?;
    let mut kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
    if let Some(format) = sent.response_format {
        kwargs.insert("response_format".to_string(), format);
    }
    if let Some(choice) = sent.tool_choice.filter(Value::is_string) {
        kwargs.insert("tool_choice".to_string(), choice);
    }
    let thinking = sent.thinking.as_ref();
    let effort = request.reasoning_effort.as_deref();
    match controls {
        ReasoningControls::Template(toggle) => {
            let resolved = resolve_reasoning_controls(thinking, effort, &CHAT_RULES)?;
            apply_reasoning_controls(&mut kwargs, toggle.as_ref(), resolved, &CHAT_RULES);
            Ok(kwargs)
        }
        ReasoningControls::DeepseekV41 => {
            let rules = &DEEPSEEK_V41_RULES;
            let resolved = resolve_reasoning_controls(thinking, effort, rules)?;
            apply_reasoning_controls(&mut kwargs, Some(&deepseek_v41_toggle()), resolved, rules);
            deepseek_v41_encoder_kwargs(kwargs)
        }
    }
}

/// ATOM's `ReasoningRules`. Both sets here switch reasoning off for a
/// `"none"` effort.
struct ReasoningRules {
    normalize_effort: fn(Option<&Value>) -> Result<Option<Value>, String>,
    string_effort_enables: bool,
    reconcile_effort_with_toggle: bool,
}

/// `CHAT_REASONING_RULES`.
const CHAT_RULES: ReasoningRules = ReasoningRules {
    normalize_effort: legacy_chat_effort,
    string_effort_enables: false,
    reconcile_effort_with_toggle: false,
};

/// `deepseek_v41_encoder.REASONING_RULES`.
const DEEPSEEK_V41_RULES: ReasoningRules = ReasoningRules {
    normalize_effort: deepseek_v41_effort,
    string_effort_enables: true,
    reconcile_effort_with_toggle: true,
};

/// ATOM's `resolve_reasoning_controls`: whether a chat switches reasoning on or
/// off, if it says, and the effort for `thinking_effort`. A `thinking.type`
/// other than `"disabled"` switches it on and a `"none"` effort off, and
/// `thinking.effort` takes the place of `reasoning_effort`.
fn resolve_reasoning_controls(
    thinking: Option<&Value>,
    reasoning_effort: Option<&str>,
    rules: &ReasoningRules,
) -> Result<(Option<bool>, Option<Value>), String> {
    let field = |key| {
        thinking
            .and_then(|thinking| thinking.get(key))
            .filter(|value| !value.is_null())
    };
    let mut enabled = field("type").map(|kind| *kind != "disabled");
    if reasoning_effort == Some("none") {
        enabled = Some(false);
    }
    let reasoning_effort = reasoning_effort.map(Value::from);
    let effort = (rules.normalize_effort)(field("effort").or(reasoning_effort.as_ref()))?;
    if effort.as_ref().is_some_and(|effort| *effort == "none") {
        return Ok((Some(false), None));
    }
    if rules.string_effort_enables
        && enabled.is_none()
        && effort.as_ref().is_some_and(Value::is_string)
    {
        enabled = Some(true);
    }
    Ok((enabled, effort))
}

/// ATOM's `apply_reasoning_controls`: the switch sets `toggle`, and the effort
/// `thinking_effort`. Rules that reconcile the two keep reasoning off for an
/// effort alone where `kwargs` turned it off with a `"none"` effort, and drop
/// a `"none"` effort when switching reasoning on, or any effort when off.
fn apply_reasoning_controls(
    kwargs: &mut HashMap<String, Value>,
    toggle: Option<&ReasoningToggle>,
    (enabled, effort): (Option<bool>, Option<Value>),
    rules: &ReasoningRules,
) {
    let reconcile = rules.reconcile_effort_with_toggle;
    let inherited_off = || {
        kwargs
            .get("thinking_effort")
            .or_else(|| kwargs.get("reasoning_effort"))
            .is_some_and(|effort| *effort == "none")
    };
    let switch = match enabled {
        None if reconcile && effort.is_some() && inherited_off() => Some(false),
        enabled => enabled,
    };
    if let (Some(toggle), Some(on)) = (toggle, switch) {
        let value = if on { &toggle.on } else { &toggle.off };
        kwargs.insert(toggle.name.to_string(), value.clone());
    }
    if let Some(effort) = effort {
        kwargs.insert("thinking_effort".to_string(), effort);
    }
    match enabled {
        Some(true) if reconcile => {
            for key in ["thinking_effort", "reasoning_effort"] {
                if kwargs.get(key).is_some_and(|effort| *effort == "none") {
                    kwargs.remove(key);
                }
            }
        }
        Some(false) if reconcile => {
            kwargs.remove("thinking_effort");
            kwargs.remove("reasoning_effort");
        }
        _ => {}
    }
}

/// The parameters, but the messages, of DeepSeek V4.1's `encode_messages`;
/// ATOM drops any other kwarg.
const DEEPSEEK_V41_ENCODER_KWARGS: [&str; 6] = [
    "thinking_mode",
    "context",
    "drop_thinking",
    "add_default_bos_token",
    "reasoning_effort",
    "return_multi_modal_data",
];

/// What ATOM passes DeepSeek V4.1's encoder of `kwargs`: its loader's default
/// `thinking_mode`, then `deepseek_v41_encoder.prepare_kwargs`, which makes
/// `thinking_effort` the encoder's `reasoning_effort` and a `"none"` one chat
/// mode, and rejects a `thinking_mode` but `thinking` or `chat` and a set
/// `return_multi_modal_data`, then only the kwargs the encoder takes.
fn deepseek_v41_encoder_kwargs(
    mut kwargs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>, String> {
    kwargs
        .entry("thinking_mode".to_string())
        .or_insert_with(|| json!("thinking"));
    if let Some(effort) = kwargs.remove("thinking_effort") {
        kwargs.insert("reasoning_effort".to_string(), effort);
    }
    match deepseek_v41_effort(kwargs.get("reasoning_effort"))? {
        Some(effort) if effort == "none" => {
            kwargs.insert("thinking_mode".to_string(), json!("chat"));
            kwargs.remove("reasoning_effort");
        }
        Some(effort) => {
            kwargs.insert("reasoning_effort".to_string(), effort);
        }
        None => {}
    }
    if !kwargs
        .get("thinking_mode")
        .is_some_and(|mode| *mode == "thinking" || *mode == "chat")
    {
        return Err("DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'".to_string());
    }
    if kwargs.get("return_multi_modal_data").is_some_and(truthy) {
        return Err("return_multi_modal_data is not a text chat-template control".to_string());
    }
    kwargs.retain(|name, _| DEEPSEEK_V41_ENCODER_KWARGS.contains(&name.as_str()));
    Ok(kwargs)
}

/// Python's truth value of `value`, as ATOM tests a kwarg.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(set) => *set,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

/// ATOM's `_legacy_chat_effort`: an integer budget in [1, 100], or an effort its
/// templates name (`VALID_TEMPLATE_EFFORTS`); other names are dropped.
fn legacy_chat_effort(effort: Option<&Value>) -> Result<Option<Value>, String> {
    match effort {
        Some(Value::String(name)) if matches!(name.as_str(), "low" | "high" | "max") => {
            Ok(Some(Value::String(name.clone())))
        }
        // A Python `bool` is an `int`, so ATOM rejects one as it does a float.
        Some(number @ (Value::Number(_) | Value::Bool(_))) => match number.as_u64() {
            Some(budget @ 1..=100) => Ok(Some(budget.into())),
            _ => Err("reasoning effort must be an integer in [1, 100]".to_string()),
        },
        _ => Ok(None),
    }
}

/// `deepseek_v41_encoder.normalize_effort`: an integer budget in [1, 100],
/// `"none"`, or an effort V4.1 names, the hosted API's aliases taken as theirs.
fn deepseek_v41_effort(effort: Option<&Value>) -> Result<Option<Value>, String> {
    fn alias(name: &str) -> &str {
        match name {
            "minimal" => "low",
            "medium" | "xhigh" => "high",
            name => name,
        }
    }
    match effort {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(name)) if matches!(alias(name), "none" | "low" | "high" | "max") => {
            Ok(Some(alias(name).into()))
        }
        Some(Value::Number(budget)) if budget.as_u64().is_some_and(|b| (1..=100).contains(&b)) => {
            Ok(Some(Value::Number(budget.clone())))
        }
        Some(_) => Err(
            "DeepSeek-V4.1 reasoning_effort must be none, low, high, max, minimal, \
                        medium, xhigh, or an integer in [1, 100]"
                .to_string(),
        ),
    }
}
