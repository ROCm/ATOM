//! The template kwargs ATOM's `/v1/chat/completions` renders a chat with, so
//! HTTP routing tokenizes chats as the engine does. An ATOM worker reports in
//! its `/v1/models` card how it renders them, as it resolved that at startup
//! ([`ChatRendering`]). The reasoning rules and the encoder kwargs hook it
//! names are ATOM's code (`reasoning_controls`, `chat_encoder_adapters`),
//! mirrored here: `tests/entrypoints/fixtures/chat_rendering.json` holds the
//! kwargs ATOM renders a set of chats with, which its tests and these check.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::inference::SentControls;
use crate::protocols::chat::ChatCompletionRequest;

type Kwargs = HashMap<String, Value>;

/// How an ATOM worker renders chats: its `/v1/models` card's `chat_rendering`
/// (ATOM's `model_capabilities.chat_rendering`). A card parses as one only
/// when this router mirrors the rules and the kwargs hook it names, and it
/// reports template efforts exactly when its rules read them.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(try_from = "RenderingCard")]
pub(crate) struct ChatRendering {
    /// The server's `--default-chat-template-kwargs`.
    default_kwargs: Kwargs,
    /// The kwarg that switches reasoning, if one does.
    reasoning_toggle: Option<ReasoningToggle>,
    /// The rules the chat endpoint applies reasoning controls by.
    reasoning_rules: ReasoningRules,
    /// The encoder the worker renders chats with instead of the template, if any.
    encoder: Option<ChatEncoder>,
}

/// A `ChatRendering` as the card reports it.
#[derive(Deserialize)]
struct RenderingCard {
    default_kwargs: Kwargs,
    reasoning_toggle: Option<ReasoningToggle>,
    reasoning_rules: RulesName,
    template_efforts: Option<Vec<String>>,
    encoder: Option<ChatEncoder>,
}

impl TryFrom<RenderingCard> for ChatRendering {
    type Error = String;

    fn try_from(card: RenderingCard) -> Result<Self, String> {
        let reasoning_rules = match (card.reasoning_rules, card.template_efforts) {
            (RulesName::Chat, Some(template_efforts)) => ReasoningRules::Chat { template_efforts },
            (RulesName::Chat, None) => {
                return Err("the chat rules need template_efforts".to_string())
            }
            (RulesName::DeepseekV41, None) => ReasoningRules::DeepseekV41,
            (RulesName::DeepseekV41, Some(_)) => {
                return Err("the deepseek_v41 rules read no template_efforts".to_string())
            }
        };
        Ok(Self {
            default_kwargs: card.default_kwargs,
            reasoning_toggle: card.reasoning_toggle,
            reasoning_rules,
            encoder: card.encoder,
        })
    }
}

/// A template kwarg that switches the model's reasoning, with its off and on values.
#[derive(Clone, Debug, Deserialize, PartialEq)]
struct ReasoningToggle {
    name: String,
    off: Value,
    on: Value,
}

/// A custom encoder, as ATOM's `MessageEncoderAdapter.rendering` reports it.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(try_from = "EncoderCard")]
struct ChatEncoder {
    /// What its loader sets where the kwargs do not.
    defaults: Kwargs,
    /// The kwargs its signature takes; any, when `None`.
    accepts: Option<Vec<String>>,
    /// Its model's `prepare_kwargs`, if that applies.
    kwargs_hook: Option<KwargsHook>,
}

/// A `ChatEncoder` as the card reports it. When it `prepares_kwargs`, its
/// `name` names the kwargs hook.
#[derive(Deserialize)]
struct EncoderCard {
    name: String,
    defaults: Kwargs,
    accepts: Option<Vec<String>>,
    prepares_kwargs: bool,
}

impl TryFrom<EncoderCard> for ChatEncoder {
    type Error = String;

    fn try_from(card: EncoderCard) -> Result<Self, String> {
        let kwargs_hook = if card.prepares_kwargs {
            let name = Value::String(card.name);
            let hook = KwargsHook::deserialize(&name);
            Some(hook.map_err(|_| format!("unknown kwargs hook {name}"))?)
        } else {
            None
        };
        Ok(Self {
            defaults: card.defaults,
            accepts: card.accepts,
            kwargs_hook,
        })
    }
}

/// The encoders whose model's `prepare_kwargs` ATOM applies, by the encoder's
/// name, mirrored here.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub(crate) enum KwargsHook {
    #[serde(rename = "encoding_dsv41")]
    DeepseekV41,
}

impl KwargsHook {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 1] = [Self::DeepseekV41];

    fn prepare(self, kwargs: Kwargs) -> Result<Kwargs, String> {
        match self {
            Self::DeepseekV41 => deepseek_v41_prepare_kwargs(kwargs),
        }
    }
}

impl ChatRendering {
    /// The rendering a worker reports as `json`, if this router mirrors the
    /// rules and kwargs hook it names.
    pub(crate) fn parse(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| e.to_string())
    }
}

/// The template kwargs ATOM's chat endpoint renders `request` with
/// (`serving_chat.resolve_chat_template_kwargs`, then a custom encoder's
/// `render_kwargs`): the worker's default kwargs, the request's
/// `chat_template_kwargs` over them, its `response_format` and a string
/// `tool_choice`, then its reasoning controls under the worker's rules. An
/// error is a request ATOM rejects.
pub(crate) fn atom_chat_template_kwargs(
    request: &ChatCompletionRequest,
    sent: &SentControls,
    rendering: &ChatRendering,
) -> Result<Kwargs, String> {
    let mut kwargs = rendering.default_kwargs.clone();
    if let Some(template_kwargs) = &request.chat_template_kwargs {
        kwargs.extend(template_kwargs.clone());
    }
    if let Some(format) = &sent.response_format {
        kwargs.insert("response_format".to_string(), format.clone());
    }
    if let Some(choice) = sent
        .tool_choice
        .as_ref()
        .filter(|choice| choice.is_string())
    {
        kwargs.insert("tool_choice".to_string(), choice.clone());
    }
    let rules = &rendering.reasoning_rules;
    let controls = resolve_reasoning_controls(
        sent.thinking.as_ref(),
        request.reasoning_effort.as_deref(),
        rules,
    )?;
    let toggle = rendering.reasoning_toggle.as_ref();
    apply_reasoning_controls(&mut kwargs, toggle, controls, rules);
    match &rendering.encoder {
        Some(encoder) => encoder.render_kwargs(kwargs),
        None => Ok(kwargs),
    }
}

/// ATOM's `ReasoningRules` a chat can render under, by the names its
/// `/v1/models` gives them, mirrored here.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RulesName {
    Chat,
    DeepseekV41,
}

impl RulesName {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 2] = [Self::Chat, Self::DeepseekV41];
}

/// ATOM's `ReasoningRules`, with what the card reports for them.
#[derive(Clone, Debug, PartialEq)]
enum ReasoningRules {
    /// `CHAT_REASONING_RULES`, which forward an effort the worker's templates
    /// read.
    Chat { template_efforts: Vec<String> },
    /// `deepseek_v41_encoder.REASONING_RULES`.
    DeepseekV41,
}

impl ReasoningRules {
    /// The effort a request names, or an error for one ATOM rejects.
    fn normalize_effort(&self, effort: Option<&Value>) -> Result<Option<Value>, String> {
        match self {
            Self::Chat { template_efforts } => legacy_chat_effort(effort, template_efforts),
            Self::DeepseekV41 => deepseek_v41_effort(effort),
        }
    }

    fn string_effort_enables(&self) -> bool {
        matches!(self, Self::DeepseekV41)
    }

    fn none_disables(&self) -> bool {
        true
    }

    fn reconcile_effort_with_toggle(&self) -> bool {
        matches!(self, Self::DeepseekV41)
    }
}

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
    if rules.none_disables() && reasoning_effort == Some("none") {
        enabled = Some(false);
    }
    let reasoning_effort = reasoning_effort.map(Value::from);
    let effort = field("effort").or(reasoning_effort.as_ref());
    let effort = rules.normalize_effort(effort)?;
    if rules.none_disables() && effort.as_ref().is_some_and(|effort| *effort == "none") {
        return Ok((Some(false), None));
    }
    if rules.string_effort_enables()
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
    kwargs: &mut Kwargs,
    toggle: Option<&ReasoningToggle>,
    (enabled, effort): (Option<bool>, Option<Value>),
    rules: &ReasoningRules,
) {
    let reconcile = rules.reconcile_effort_with_toggle();
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
        kwargs.insert(toggle.name.clone(), value.clone());
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

impl ChatEncoder {
    /// ATOM's `MessageEncoderAdapter.render_kwargs`: the encoder's defaults
    /// under `kwargs`, its model's `prepare_kwargs`, then the kwargs it takes.
    fn render_kwargs(&self, mut kwargs: Kwargs) -> Result<Kwargs, String> {
        for (name, value) in &self.defaults {
            kwargs.entry(name.clone()).or_insert_with(|| value.clone());
        }
        if let Some(hook) = self.kwargs_hook {
            kwargs = hook.prepare(kwargs)?;
        }
        if let Some(accepts) = &self.accepts {
            kwargs.retain(|name, _| accepts.contains(name));
        }
        Ok(kwargs)
    }
}

/// `deepseek_v41_encoder.prepare_kwargs`: `thinking_effort` becomes the
/// encoder's `reasoning_effort` and a `"none"` one chat mode; it rejects a
/// `thinking_mode` other than `thinking` or `chat` and a set
/// `return_multi_modal_data`.
fn deepseek_v41_prepare_kwargs(mut kwargs: Kwargs) -> Result<Kwargs, String> {
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
        .is_none_or(|mode| *mode == "thinking" || *mode == "chat")
    {
        return Err("DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'".to_string());
    }
    if kwargs.get("return_multi_modal_data").is_some_and(truthy) {
        return Err("return_multi_modal_data is not a text chat-template control".to_string());
    }
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

/// ATOM's `_legacy_chat_effort`: an integer budget in [1, 100], or an effort
/// the worker's templates name; other names are dropped.
fn legacy_chat_effort(
    effort: Option<&Value>,
    template_efforts: &[String],
) -> Result<Option<Value>, String> {
    match effort {
        Some(Value::String(name)) if template_efforts.contains(name) => {
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
