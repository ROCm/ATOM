//! A chat request as the Python API server runs it (`_chat_completions`):
//! what it refuses, and `tool_choice` and the reasoning controls turned into
//! template kwargs the way it turns them. What reaches mesh's shared prepare
//! is then rendered as api_server renders it.

use std::collections::HashMap;

use axum::response::Response;
use serde_json::Value;

use super::AtomEngineRouter;
use crate::{
    atom_engine::ReasoningToggle,
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::{ContentPart, ResponseFormat, ToolChoice, ToolChoiceValue},
    },
    routers::comm::error,
};

/// The named efforts a template takes (`VALID_TEMPLATE_EFFORTS`).
const TEMPLATE_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// A chat request ready for prepare, and whether it holds to the required
/// tool-call channel.
pub(super) struct ChatAdmission {
    pub request: ChatCompletionRequest,
    pub required_tool_call: bool,
}

impl AtomEngineRouter {
    /// The request as this server runs it: its own model name (which picks
    /// the parsers), each message's text parts joined into one string, and
    /// the template kwargs api_server would render it with.
    ///
    /// `tool_choice` reaches the template, not a grammar: api_server does
    /// not constrain a required or named choice (only a model whose format
    /// has a required channel holds to it), so neither may mesh's prepare
    /// and render, which would otherwise read the reply as constrained JSON.
    pub(super) fn chat_request(
        &self,
        body: &ChatCompletionRequest,
        thinking: Option<&Value>,
    ) -> Result<ChatAdmission, Response> {
        self.check_model(Some(&body.model))?;
        let mut request = body.clone();
        for message in &mut request.messages {
            let content = match message {
                ChatMessage::System { content, .. }
                | ChatMessage::User { content, .. }
                | ChatMessage::Tool { content, .. }
                | ChatMessage::Developer { content, .. } => Some(content),
                ChatMessage::Assistant { content, .. } => content.as_mut(),
                ChatMessage::Function { .. } => None,
            };
            if let Some(content) = content {
                join_text_parts(content)?;
            }
        }
        request.model = self.engine.model.served_name.clone();
        if request.top_logprobs.is_some() && !request.logprobs {
            return Err(bad_request("top_logprobs requires logprobs=true".into()));
        }

        // Defaults under the request's own kwargs, then what api_server adds.
        let mut kwargs = request.chat_template_kwargs.take().unwrap_or_default();
        for (key, value) in &self.engine.model.default_chat_template_kwargs {
            kwargs.entry(key.clone()).or_insert_with(|| value.clone());
        }
        match &request.response_format {
            None => {}
            Some(ResponseFormat::Text) => {
                kwargs.insert(
                    "response_format".into(),
                    serde_json::json!({"type": "text"}),
                );
            }
            Some(_) => {
                return Err(error::bad_request(
                    "structured_output_not_supported",
                    "response_format json_object / json_schema is not supported by the \
                     ATOM standalone server yet",
                ))
            }
        }
        let required_tool_call = take_tool_choice(&mut request, &mut kwargs)?;
        let controls = reasoning_controls(thinking, request.reasoning_effort.take())?;
        apply_reasoning_controls(
            &mut kwargs,
            self.engine.model.reasoning_toggle.as_ref(),
            controls,
        );
        request.chat_template_kwargs = (!kwargs.is_empty()).then_some(kwargs);
        Ok(ChatAdmission {
            request,
            required_tool_call,
        })
    }
}

/// `validate_chat_request`'s `tool_choice` rules; a string choice goes to the
/// template, and a required or named one leaves the request as `auto`.
/// Returns whether the request must call a tool.
fn take_tool_choice(
    request: &mut ChatCompletionRequest,
    kwargs: &mut HashMap<String, Value>,
) -> Result<bool, Response> {
    let tools = request.tools.as_deref().unwrap_or_default();
    let required = match &request.tool_choice {
        None => return Ok(false),
        Some(ToolChoice::Value(value)) => {
            let (name, required) = match value {
                ToolChoiceValue::Auto => ("auto", false),
                ToolChoiceValue::None => ("none", false),
                ToolChoiceValue::Required => ("required", true),
            };
            if required && tools.is_empty() {
                return Err(bad_request(
                    "tool_choice 'required' needs a non-empty tools list".into(),
                ));
            }
            kwargs.insert("tool_choice".into(), name.into());
            required
        }
        Some(ToolChoice::Function { function, .. }) => {
            if !tools.iter().any(|t| t.function.name == function.name) {
                return Err(bad_request(format!(
                    "tool_choice names unknown tool: {}",
                    function.name
                )));
            }
            false
        }
        Some(ToolChoice::AllowedTools { .. }) => {
            return Err(bad_request(
                "tool_choice object must have type 'function'".into(),
            ))
        }
    };
    if !matches!(
        request.tool_choice,
        Some(ToolChoice::Value(ToolChoiceValue::None))
    ) {
        request.tool_choice = Some(ToolChoice::Value(ToolChoiceValue::Auto));
    }
    Ok(required)
}

/// `resolve_reasoning_controls` under `CHAT_REASONING_RULES`: whether
/// reasoning is switched on or off (none: left to the template) and the
/// effort to render.
fn reasoning_controls(
    thinking: Option<&Value>,
    reasoning_effort: Option<String>,
) -> Result<(Option<bool>, Option<Value>), Response> {
    let thinking = match thinking {
        None | Some(Value::Null) => None,
        Some(Value::Object(thinking)) => Some(thinking),
        Some(_) => return Err(bad_request("thinking must be an object".into())),
    };
    let field = |name| thinking.and_then(|t| t.get(name)).filter(|v| !v.is_null());
    let mut enabled = field("type").map(|kind| kind != "disabled");
    if reasoning_effort.as_deref() == Some("none") {
        enabled = Some(false);
    }
    let effort = field("effort")
        .cloned()
        .or(reasoning_effort.map(Value::String));
    let effort = match effort {
        Some(Value::Number(n)) => match n.as_i64() {
            Some(budget @ 1..=100) => Some(Value::from(budget)),
            _ => return Err(bad_effort()),
        },
        Some(Value::Bool(_)) => return Err(bad_effort()),
        Some(Value::String(s)) if TEMPLATE_EFFORTS.contains(&s.as_str()) => Some(Value::String(s)),
        _ => None,
    };
    Ok((enabled, effort))
}

/// `apply_reasoning_controls`: the switch where the template has one, and the
/// effort as `thinking_effort`.
fn apply_reasoning_controls(
    kwargs: &mut HashMap<String, Value>,
    toggle: Option<&ReasoningToggle>,
    (enabled, effort): (Option<bool>, Option<Value>),
) {
    if let (Some(toggle), Some(enabled)) = (toggle, enabled) {
        let value = if enabled { &toggle.on } else { &toggle.off };
        kwargs.insert(toggle.name.clone(), value.clone());
    }
    if let Some(effort) = effort {
        kwargs.insert("thinking_effort".into(), effort);
    }
}

/// `[{type: text}, ...]` → the texts joined by newlines, as the Python server
/// renders them (templates such as Qwen3's drop content that is not a
/// string); media is refused.
fn join_text_parts(content: &mut MessageContent) -> Result<(), Response> {
    let MessageContent::Parts(parts) = content else {
        return Ok(());
    };
    let mut texts = Vec::with_capacity(parts.len());
    for part in parts.iter() {
        match part {
            ContentPart::Text { text } => texts.push(text.as_str()),
            _ => {
                return Err(error::bad_request(
                    "multimodal_not_supported",
                    "multimodal input is not supported by the ATOM standalone server yet",
                ))
            }
        }
    }
    *content = MessageContent::Text(texts.join("\n"));
    Ok(())
}

fn bad_effort() -> Response {
    bad_request("reasoning effort must be an integer in [1, 100]".into())
}

fn bad_request(message: String) -> Response {
    error::bad_request("invalid_request", message)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn text_parts_join_into_one_string_and_media_is_refused() {
        let text = |t: &str| ContentPart::Text { text: t.into() };
        let mut content = MessageContent::Parts(vec![text("a"), text("b")]);
        join_text_parts(&mut content).unwrap();
        assert_eq!(content, MessageContent::Text("a\nb".into()));

        let mut media = MessageContent::Parts(vec![
            text("a"),
            ContentPart::ImageUrl {
                image_url: crate::protocols::common::ImageUrl {
                    url: "http://x".into(),
                    detail: None,
                },
            },
        ]);
        assert!(join_text_parts(&mut media).is_err());
    }

    fn controls(thinking: Value, effort: Option<&str>) -> (Option<bool>, Option<Value>) {
        reasoning_controls(Some(&thinking), effort.map(str::to_string)).unwrap()
    }

    #[test]
    fn reasoning_controls_resolve_as_the_chat_rules_do() {
        assert_eq!(controls(Value::Null, None), (None, None));
        assert_eq!(
            controls(json!({"type": "disabled"}), Some("high")),
            (Some(false), Some(json!("high")))
        );
        // A top-level "none" switches reasoning off and is no effort itself.
        assert_eq!(controls(Value::Null, Some("none")), (Some(false), None));
        // thinking.effort wins over the top-level one; unknown names drop.
        assert_eq!(
            controls(json!({"type": "enabled", "effort": "max"}), Some("low")),
            (Some(true), Some(json!("max")))
        );
        assert_eq!(controls(Value::Null, Some("medium")), (None, None));
        assert_eq!(
            controls(json!({"effort": 40}), None),
            (None, Some(json!(40)))
        );
        for bad in [
            json!({"effort": 0}),
            json!({"effort": 2.5}),
            json!({"effort": true}),
        ] {
            assert!(reasoning_controls(Some(&bad), None).is_err(), "{bad}");
        }
        assert!(reasoning_controls(Some(&json!("on")), None).is_err());
    }

    #[test]
    fn the_toggle_and_effort_land_in_the_kwargs() {
        let toggle = ReasoningToggle {
            name: "enable_thinking".into(),
            off: json!(false),
            on: json!(true),
        };
        let mut kwargs = HashMap::new();
        apply_reasoning_controls(
            &mut kwargs,
            Some(&toggle),
            (Some(false), Some(json!("low"))),
        );
        assert_eq!(
            json!(kwargs),
            json!({"enable_thinking": false, "thinking_effort": "low"})
        );
        // No switch in the template: the request can only set the effort.
        let mut kwargs = HashMap::new();
        apply_reasoning_controls(&mut kwargs, None, (Some(false), None));
        assert!(kwargs.is_empty());
    }

    fn tool(name: &str) -> crate::protocols::common::Tool {
        serde_json::from_value(
            json!({"type": "function", "function": {"name": name, "parameters": {}}}),
        )
        .unwrap()
    }

    type Chosen = (bool, Option<ToolChoice>, HashMap<String, Value>);

    fn choose(choice: Value, tools: &[&str]) -> Result<Chosen, Response> {
        let mut request = ChatCompletionRequest {
            tool_choice: Some(serde_json::from_value(choice).unwrap()),
            tools: (!tools.is_empty()).then(|| tools.iter().map(|t| tool(t)).collect()),
            ..Default::default()
        };
        let mut kwargs = HashMap::new();
        let required = take_tool_choice(&mut request, &mut kwargs)?;
        Ok((required, request.tool_choice, kwargs))
    }

    #[test]
    fn tool_choice_goes_to_the_template_not_to_a_grammar() {
        let auto = Some(ToolChoice::Value(ToolChoiceValue::Auto));
        let (required, choice, kwargs) = choose(json!("required"), &["f"]).unwrap();
        assert!(required);
        assert!(matches!(
            choice,
            Some(ToolChoice::Value(ToolChoiceValue::Auto))
        ));
        assert_eq!(kwargs["tool_choice"], "required");

        let named = json!({"type": "function", "function": {"name": "f"}});
        let (required, choice, kwargs) = choose(named, &["f", "g"]).unwrap();
        assert!(!required && kwargs.is_empty());
        assert_eq!(format!("{choice:?}"), format!("{auto:?}"));

        let (_, choice, kwargs) = choose(json!("none"), &["f"]).unwrap();
        assert!(matches!(
            choice,
            Some(ToolChoice::Value(ToolChoiceValue::None))
        ));
        assert_eq!(kwargs["tool_choice"], "none");

        assert!(choose(json!("required"), &[]).is_err());
        assert!(choose(
            json!({"type": "function", "function": {"name": "h"}}),
            &["f"]
        )
        .is_err());
    }
}
