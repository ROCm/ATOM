//! A chat's prompt, rendered in one place for HTTP routing (`ingress`), gRPC
//! and the ATOM bridge (`prepare`). A Jinja template renders as the router
//! renders it. A model ATOM renders with an encoder of its own (DeepSeek
//! V3.2 / V4 / V4.1, Kimi-K3), picked by `config.json`'s `model_type`, renders
//! with llm-tokenizer's port of that encoder, given the chat as ATOM's server
//! gives it: messages as `ChatMessage.to_template_dict` writes them, kwargs
//! through ATOM's reasoning controls and encoder adapter
//! (`chat_encoder_adapters.py`). `test_requests_tokenize_as_on_atom` checks
//! the tokens against ATOM's.

use serde_json::{json, Map, Value};

use super::{
    chat_template::process_chat_messages,
    response_context::{prompt_opener, THINK_OPEN},
};
use crate::{
    protocols::chat::ChatCompletionRequest,
    tokenizer::{
        chat_template::ChatTemplateParams,
        encoders::{
            deepseek_v32::{self, ThinkingMode},
            deepseek_v4,
        },
        traits::Tokenizer,
    },
};

/// The kwargs DeepSeek's `encode_messages` take: V3.2's the first four, V4's
/// the first five, V4.1's all. ATOM's encoder adapter drops the others.
const ENCODER_KWARGS: [&str; 6] = [
    "thinking_mode",
    "context",
    "drop_thinking",
    "add_default_bos_token",
    "reasoning_effort",
    "return_multi_modal_data",
];

/// Kimi-K3's channel openers (`kimi_k3_tokens.py`).
const K3_OPENERS: [&str; 2] = ["<|open|>think<|sep|>", "<|open|>response<|sep|>"];

/// A model whose chats ATOM renders with an encoder of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatControls {
    /// The checkpoint's `encoding_dsv32.py`.
    DeepseekV32,
    /// The checkpoint's `encoding_dsv4.py`.
    DeepseekV4,
    /// ATOM's `deepseek_v41_encoder`.
    DeepseekV41,
    /// `tokenization_kimi.py`, whose template is code.
    KimiK3,
}

impl ChatControls {
    pub(crate) fn of_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "deepseek_v32" => Some(Self::DeepseekV32),
            "deepseek_v4" => Some(Self::DeepseekV4),
            "deepseek_v41" => Some(Self::DeepseekV41),
            "kimi_k3" => Some(Self::KimiK3),
            _ => None,
        }
    }

    /// The template kwargs ATOM renders the chat with: `_chat_completions`'
    /// (the request's, a string `tool_choice`, the reasoning controls under
    /// the toggle `resolve_reasoning_toggle` finds), then the encoder
    /// adapter's defaults, `prepare_kwargs` and signature filter.
    fn kwargs(
        self,
        chat: &ChatCompletionRequest,
        sent: &Map<String, Value>,
    ) -> Result<Map<String, Value>, String> {
        let mut kwargs: Map<String, Value> = chat
            .chat_template_kwargs
            .iter()
            .flatten()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if let Some(choice @ Value::String(_)) = field(sent, "tool_choice") {
            kwargs.insert("tool_choice".into(), choice.clone());
        }
        let v41 = self == Self::DeepseekV41;
        let toggle = match self {
            Self::KimiK3 => ("thinking", json!(false), json!(true)),
            _ => ("thinking_mode", json!("chat"), json!("thinking")),
        };
        let effort = field(sent, "reasoning_effort").cloned();
        let controls = resolve_reasoning_controls(field(sent, "thinking"), effort, v41)?;
        apply_reasoning_controls(&mut kwargs, toggle, controls, v41);
        if self == Self::KimiK3 {
            // K3's encoder renders thinking unless `thinking` says otherwise;
            // llm-tokenizer's port would also read `enable_thinking`.
            kwargs.entry("thinking").or_insert(json!(true));
            return Ok(kwargs);
        }
        kwargs
            .entry("thinking_mode")
            .or_insert_with(|| json!("thinking"));
        if v41 {
            // V4.1's `prepare_kwargs`: its effort rules, `none` as chat mode.
            if let Some(effort) = kwargs.remove("thinking_effort") {
                kwargs.insert("reasoning_effort".into(), effort);
            }
            match normalize_effort(kwargs.remove("reasoning_effort"), true)? {
                Some(effort) if effort == "none" => {
                    kwargs.insert("thinking_mode".into(), json!("chat"));
                }
                Some(effort) => {
                    kwargs.insert("reasoning_effort".into(), effort);
                }
                None => {}
            }
            if kwargs.get("return_multi_modal_data").is_some_and(truthy) {
                return Err("return_multi_modal_data is not a text chat-template control".into());
            }
        }
        let thinking = match kwargs["thinking_mode"].as_str() {
            Some("thinking") => true,
            Some("chat") => false,
            _ => {
                return Err(format!(
                    "Invalid thinking_mode `{}`",
                    kwargs["thinking_mode"]
                ))
            }
        };
        let takes = match self {
            Self::DeepseekV32 => 4,
            Self::DeepseekV4 => 5,
            _ => 6,
        };
        kwargs.retain(|name, _| ENCODER_KWARGS[..takes].contains(&name.as_str()));
        if v41 {
            // Where llm-tokenizer's V4.1 port reads the mode.
            kwargs.insert("thinking".into(), json!(thinking));
        }
        Ok(kwargs)
    }

    /// The messages as `ChatMessage.to_template_dict` writes them, and the
    /// tools as the encoder takes them: V4.1's adapter puts them on the first
    /// message if it is a system message, else on a new one; V4's on a new
    /// one; V3.2 has no adapter, so only a system message's own reach it;
    /// K3's template takes them as `tools`.
    ///
    /// V3.2's and V4's tool-call arguments stay the JSON string sent, which
    /// their checkpoints' encoders `json.loads`. ATOM decodes them first, so
    /// its V3.2 encoder fails on them and its V4 encoder renders them as one
    /// `arguments` parameter: here the router renders otherwise than ATOM.
    fn messages(self, sent: &Map<String, Value>) -> (Vec<Value>, Option<Vec<Value>>) {
        let v41 = self == Self::DeepseekV41;
        let messages = field(sent, "messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten();
        let mut messages: Vec<Value> = messages.map(|m| template_message(m, v41)).collect();
        let tools = field(sent, "tools")
            .and_then(Value::as_array)
            .filter(|t| !t.is_empty())
            .cloned();
        match (self, tools) {
            (Self::DeepseekV41, Some(tools)) => {
                match messages.first_mut().and_then(Value::as_object_mut) {
                    Some(first) if first.get("role") == Some(&json!("system")) => {
                        first.insert("tools".into(), Value::Array(tools));
                    }
                    _ => messages.insert(0, json!({"role": "system", "tools": tools})),
                }
                (messages, None)
            }
            (Self::DeepseekV4, Some(tools)) => {
                messages.insert(0, json!({"role": "system", "tools": tools}));
                (messages, None)
            }
            (Self::KimiK3, tools) => (messages, tools),
            _ => (messages, None),
        }
    }

    fn render(
        self,
        chat: &ChatCompletionRequest,
        sent: &Map<String, Value>,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String, String> {
        let kwargs = self.kwargs(chat, sent)?;
        // V4.1's API contract refuses a forced tool call in thinking mode.
        let choice = field(sent, "tool_choice");
        let forced = matches!(choice, Some(Value::Object(_)))
            || choice.and_then(Value::as_str) == Some("required");
        if self == Self::DeepseekV41 && forced && kwargs.get("thinking") == Some(&json!(true)) {
            return Err("This Chat API contract does not support required or named tool_choice in thinking mode".into());
        }
        let (messages, tools) = self.messages(sent);
        if matches!(self, Self::DeepseekV32 | Self::DeepseekV4) {
            return self.encode_messages(messages, &kwargs);
        }
        let kwargs = kwargs.into_iter().collect();
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: tools.as_deref(),
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        tokenizer
            .apply_chat_template(&messages, params)
            .map_err(|e| format!("Failed to apply chat template: {e}"))
    }

    /// V3.2's or V4's prompt from llm-tokenizer's port of the checkpoint's
    /// `encode_messages`, called as ATOM calls the checkpoint's. Unlike the
    /// checkpoint's, the port reads an empty `tools` as tools, so it goes
    /// first. Its V4 efforts sit one level below the checkpoint's: the
    /// checkpoint's `max` text is the port's `High`, and its `high` and none
    /// write nothing, as the port's `Low`.
    fn encode_messages(
        self,
        mut messages: Vec<Value>,
        kwargs: &Map<String, Value>,
    ) -> Result<String, String> {
        for message in messages.iter_mut().filter_map(Value::as_object_mut) {
            if message.get("tools").is_some_and(|tools| !truthy(tools)) {
                message.remove("tools");
            }
        }
        if kwargs.get("context").is_some_and(truthy) {
            return Err("context is not a chat-template control".into());
        }
        let mode = match kwargs["thinking_mode"] == "thinking" {
            true => ThinkingMode::Thinking,
            false => ThinkingMode::Chat,
        };
        let flag = |name: &str| kwargs.get(name).is_none_or(truthy);
        let (add_default_bos_token, drop_thinking) =
            (flag("add_default_bos_token"), flag("drop_thinking"));
        let rendered = if self == Self::DeepseekV32 {
            let params = deepseek_v32::EncodeParams {
                add_default_bos_token,
                drop_thinking,
            };
            deepseek_v32::encode_messages(&messages, mode, &params)
        } else {
            let reasoning_effort = match kwargs.get("reasoning_effort").filter(|v| !v.is_null()) {
                None => deepseek_v4::ReasoningEffort::Low,
                Some(effort) if effort == "high" => deepseek_v4::ReasoningEffort::Low,
                Some(effort) if effort == "max" => deepseek_v4::ReasoningEffort::High,
                Some(effort) => return Err(format!("Invalid reasoning effort: {effort}")),
            };
            let params = deepseek_v4::EncodeParams {
                add_default_bos_token,
                drop_thinking,
                reasoning_effort: Some(reasoning_effort),
            };
            deepseek_v4::encode_messages(&messages, mode, &params)
        };
        rendered.map_err(|e| format!("Failed to apply chat template: {e}"))
    }
}

/// The prompt `chat` renders to with `tokenizer`, and the channel opener it
/// ends on (`ResponseContext::prompt_opener`). An encoder reads the chat's
/// body as `sent` where the caller has it: the typed request drops or
/// rewrites fields it reads (`thinking`, a system message's `tools`,
/// `tool_choice`).
pub(crate) fn render_chat(
    chat: &ChatCompletionRequest,
    sent: Option<&Map<String, Value>>,
    tokenizer: &dyn Tokenizer,
    controls: Option<ChatControls>,
) -> Result<(String, Option<String>), String> {
    let Some(controls) = controls else {
        let text = process_chat_messages(chat, tokenizer)?.text;
        let opener = prompt_opener(&text);
        return Ok((text, opener));
    };
    let typed;
    let sent = match sent {
        Some(sent) => sent,
        None => {
            typed = serde_json::to_value(chat)
                .map_err(|e| e.to_string())?
                .as_object()
                .cloned()
                .unwrap_or_default();
            &typed
        }
    };
    let text = controls.render(chat, sent, tokenizer)?;
    let openers: &[&str] = match controls {
        ChatControls::KimiK3 => &K3_OPENERS,
        _ => &[THINK_OPEN],
    };
    let tail = text.trim_end();
    let opener = openers
        .iter()
        .find(|o| tail.ends_with(*o))
        .map(|o| o.to_string());
    Ok((text, opener))
}

fn field<'a>(body: &'a Map<String, Value>, name: &str) -> Option<&'a Value> {
    body.get(name).filter(|v| !v.is_null())
}

/// Python's truth of a JSON value, as an encoder's `if` reads it.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `ChatMessage.to_template_dict`: role, content (its text parts joined
/// unless `preserve_content`) and the fields templates read, tool-call
/// arguments decoded (`_normalize_tool_call_arguments`) only with
/// `preserve_content` (V4.1, the one encoder of ATOM's own that takes them so).
fn template_message(message: &Value, preserve_content: bool) -> Value {
    let content = message.get("content").cloned().unwrap_or(Value::Null);
    let content = match (&content, preserve_content) {
        (_, true) => content,
        (Value::String(_), _) => content,
        (Value::Array(parts), _) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter(|part| part.get("type") == Some(&json!("text")))
                .map(|part| part.get("text").and_then(Value::as_str).unwrap_or(""))
                .collect();
            json!(texts.join("\n"))
        }
        _ => json!(""),
    };
    let mut out = Map::from_iter([
        (
            "role".into(),
            message.get("role").cloned().unwrap_or(Value::Null),
        ),
        ("content".into(), content),
    ]);
    for name in [
        "tool_calls",
        "tool_call_id",
        "name",
        "reasoning_content",
        "tools",
    ] {
        let Some(mut value) = message.get(name).cloned() else {
            continue;
        };
        if name == "tool_calls" && preserve_content {
            for call in value.as_array_mut().into_iter().flatten() {
                if let Some(Value::String(raw)) = call.pointer("/function/arguments").cloned() {
                    call["function"]["arguments"] = serde_json::from_str(&raw)
                        .or_else(|_| serde_json::from_str(&fix_invalid_json_escapes(&raw)))
                        .unwrap_or_else(|_| json!({"_raw": raw}));
                }
            }
        }
        out.insert(name.into(), value);
    }
    Value::Object(out)
}

/// ATOM's `_fix_invalid_json_escapes`: doubles every backslash that does not
/// start a valid JSON escape (`\k` in model-written arguments).
fn fix_invalid_json_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            None => out.push_str("\\\\"),
            Some(next @ ('"' | '\\' | 'b' | 'f' | 'n' | 'r' | 't' | 'u' | '/')) => {
                out.push('\\');
                out.push(next);
            }
            Some(next) => {
                out.push_str("\\\\");
                out.push(next);
            }
        }
    }
    out
}

/// ATOM's effort normalization: V4.1's `normalize_effort` with `v41`, else
/// the chat default (`_legacy_chat_effort`).
fn normalize_effort(effort: Option<Value>, v41: bool) -> Result<Option<Value>, String> {
    let Some(effort) = effort else {
        return Ok(None);
    };
    let budget = effort.as_i64().filter(|b| (1..=100).contains(b));
    if v41 {
        let named = effort.as_str().map(|name| match name {
            "minimal" => "low",
            "medium" | "xhigh" => "high",
            other => other,
        });
        return match (named, budget) {
            (Some(name @ ("none" | "low" | "high" | "max")), _) => Ok(Some(json!(name))),
            (None, Some(budget)) => Ok(Some(json!(budget))),
            _ => Err("DeepSeek-V4.1 reasoning_effort must be none, low, high, max, minimal, medium, xhigh, or an integer in [1, 100]".into()),
        };
    }
    match &effort {
        Value::Number(_) if budget.is_none() => {
            Err("reasoning effort must be an integer in [1, 100]".into())
        }
        Value::Number(_) => Ok(Some(effort)),
        Value::String(name) if matches!(name.as_str(), "low" | "high" | "max") => Ok(Some(effort)),
        _ => Ok(None),
    }
}

/// `resolve_reasoning_controls`: whether reasoning is on (`None`: unstated)
/// and the effort. V4.1's rules (`v41`) also let a named effort switch it on.
fn resolve_reasoning_controls(
    thinking: Option<&Value>,
    effort: Option<Value>,
    v41: bool,
) -> Result<(Option<bool>, Option<Value>), String> {
    let thinking_field = |name: &str| {
        thinking
            .and_then(|t| t.as_object()?.get(name))
            .filter(|v| !v.is_null())
    };
    let mut enabled = thinking_field("type").map(|kind| kind != "disabled");
    if effort.as_ref().and_then(Value::as_str) == Some("none") {
        enabled = Some(false);
    }
    let effort = normalize_effort(thinking_field("effort").cloned().or(effort), v41)?;
    if effort.as_ref().and_then(Value::as_str) == Some("none") {
        return Ok((Some(false), None));
    }
    if v41 && enabled.is_none() && effort.as_ref().is_some_and(Value::is_string) {
        enabled = Some(true);
    }
    Ok((enabled, effort))
}

/// `apply_reasoning_controls`: the controls written under the model's toggle
/// and as `thinking_effort`; V4.1's rules (`v41`) reconcile an inherited
/// `none` effort with the toggle.
fn apply_reasoning_controls(
    kwargs: &mut Map<String, Value>,
    (name, off, on): (&str, Value, Value),
    (enabled, effort): (Option<bool>, Option<Value>),
    v41: bool,
) {
    let none = json!("none");
    let inherited = kwargs
        .get("thinking_effort")
        .or_else(|| kwargs.get("reasoning_effort"));
    if v41 && enabled.is_none() && effort.is_some() && inherited == Some(&none) {
        kwargs.insert(name.into(), off.clone());
    }
    if let Some(enabled) = enabled {
        kwargs.insert(name.into(), if enabled { on } else { off });
    }
    if let Some(effort) = effort {
        kwargs.insert("thinking_effort".into(), effort);
    }
    for key in ["thinking_effort", "reasoning_effort"] {
        match enabled {
            Some(true) if v41 && kwargs.get(key) == Some(&none) => kwargs.remove(key),
            Some(false) if v41 => kwargs.remove(key),
            _ => None,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        routers::{prepare::inference::ParsedInference, test_mocks::tokenizer::hf_tokenizer_path},
        tokenizer::factory::create_tokenizer,
    };

    /// A chat of `fields` (by default a user's "hi") and its body as sent.
    fn chat(fields: &Value) -> (ChatCompletionRequest, Map<String, Value>) {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let parsed =
            ParsedInference::parse("/v1/chat/completions", body.to_string().as_bytes()).unwrap();
        (
            parsed.chat().unwrap().clone(),
            body.as_object().unwrap().clone(),
        )
    }

    /// Each encoder gets the mode and effort ATOM gives it: `[model_type,
    /// request fields, kwargs]`, or no kwargs where ATOM refuses the request.
    #[test]
    fn controls_are_atoms() {
        let cases = json!([
            ["deepseek_v41", {}, {"thinking_mode": "thinking", "thinking": true}],
            ["deepseek_v41", {"reasoning_effort": "minimal"}, {"thinking_mode": "thinking", "thinking": true, "reasoning_effort": "low"}],
            ["deepseek_v41", {"reasoning_effort": "xhigh"}, {"thinking_mode": "thinking", "thinking": true, "reasoning_effort": "high"}],
            ["deepseek_v41", {"reasoning_effort": "none"}, {"thinking_mode": "chat", "thinking": false}],
            ["deepseek_v41", {"chat_template_kwargs": {"reasoning_effort": 30}}, {"thinking_mode": "thinking", "thinking": true, "reasoning_effort": 30}],
            ["deepseek_v41", {"thinking": {"type": "enabled"}, "reasoning_effort": "none"}, {"thinking_mode": "chat", "thinking": false}],
            ["deepseek_v41", {"reasoning_effort": "high", "chat_template_kwargs": {"thinking_mode": "chat"}}, {"thinking_mode": "thinking", "thinking": true, "reasoning_effort": "high"}],
            ["deepseek_v41", {"chat_template_kwargs": {"enable_thinking": false}}, {"thinking_mode": "thinking", "thinking": true}],
            ["deepseek_v41", {"reasoning_effort": "ultra"}, null],
            ["kimi_k3", {"chat_template_kwargs": {"enable_thinking": false}}, {"thinking": true, "enable_thinking": false}],
            ["kimi_k3", {"reasoning_effort": "low"}, {"thinking": true, "thinking_effort": "low"}],
            ["kimi_k3", {"reasoning_effort": "medium"}, {"thinking": true}],
            ["kimi_k3", {"thinking": {"type": "disabled"}, "tool_choice": "none"}, {"thinking": false, "tool_choice": "none"}],
            ["kimi_k3", {"thinking": {"type": "enabled", "effort": 0}}, null],
            ["deepseek_v4", {"reasoning_effort": "max"}, {"thinking_mode": "thinking"}],
            ["deepseek_v4", {"chat_template_kwargs": {"reasoning_effort": "max"}}, {"thinking_mode": "thinking", "reasoning_effort": "max"}],
            ["deepseek_v4", {"chat_template_kwargs": {"thinking_mode": "auto"}}, null],
            ["deepseek_v32", {"chat_template_kwargs": {"reasoning_effort": "max", "drop_thinking": false}}, {"thinking_mode": "thinking", "drop_thinking": false}],
            ["deepseek_v32", {"thinking": {"type": "disabled"}}, {"thinking_mode": "chat"}],
        ]);
        for case in cases.as_array().unwrap() {
            let controls = ChatControls::of_model_type(case[0].as_str().unwrap()).unwrap();
            let (chat, body) = chat(&case[1]);
            let got = controls.kwargs(&chat, &body).ok().map(Value::Object);
            assert_eq!(got.unwrap_or(Value::Null), case[2], "{case}");
        }
    }

    /// Messages reach each encoder as ATOM's adapter hands them over.
    #[test]
    fn messages_are_atoms() {
        let call = json!({"id": "c", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}});
        let tool = json!({"type": "function", "function": {"name": "f"}});
        let sent = json!({
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "assistant", "content": null, "tool_calls": [call]},
            ],
            "tools": [tool],
        });
        let sent = sent.as_object().unwrap();
        let messages = |controls: ChatControls| controls.messages(sent);
        let (k3, k3_tools) = messages(ChatControls::KimiK3);
        assert_eq!(
            (&k3[1]["content"], &k3[2]["content"]),
            (&json!("a\nb"), &json!(""))
        );
        assert_eq!(
            k3[2]["tool_calls"][0]["function"]["arguments"],
            "{\"a\": 1}"
        );
        assert_eq!(k3_tools, Some(vec![tool.clone()]));
        let (v41, _) = messages(ChatControls::DeepseekV41);
        assert_eq!((v41.len(), &v41[0]["tools"][0]), (3, &tool));
        assert_eq!(v41[1]["content"][1]["text"], "b");
        assert_eq!(
            v41[2]["tool_calls"][0]["function"]["arguments"],
            json!({"a": 1})
        );
        let (v4, _) = messages(ChatControls::DeepseekV4);
        assert_eq!(
            (v4.len(), &v4[0]),
            (4, &json!({"role": "system", "tools": [tool]}))
        );
        assert_eq!(
            v4[3]["tool_calls"][0]["function"]["arguments"],
            "{\"a\": 1}"
        );
        assert_eq!(messages(ChatControls::DeepseekV32), (k3, None));
        // Invalid escapes are retried doubled, as ATOM retries them.
        let call = |arguments: &str| {
            let call = json!({"function": {"name": "f", "arguments": arguments}});
            let message = json!({"role": "assistant", "tool_calls": [call]});
            template_message(&message, true)["tool_calls"][0]["function"]["arguments"].clone()
        };
        assert_eq!(call(r#"{"p": "C:\kdir"}"#), json!({"p": "C:\\kdir"}));
        assert_eq!(call("{"), json!({"_raw": "{"}));
    }

    /// V3.2 and V4 render their mode and V4 its effort as their checkpoints
    /// do (only `max` writes a prefix); V4.1 refuses a forced tool call while
    /// thinking.
    #[test]
    fn encoders_render_as_atoms() {
        let tokenizer = create_tokenizer(hf_tokenizer_path()).unwrap();
        let render = |controls, fields: Value| {
            let (chat, body) = chat(&fields);
            render_chat(&chat, Some(&body), &*tokenizer, Some(controls))
        };
        for controls in [ChatControls::DeepseekV32, ChatControls::DeepseekV4] {
            let (text, opener) = render(controls, json!({})).unwrap();
            assert!(text.ends_with("<｜Assistant｜><think>"), "{text}");
            assert_eq!(opener.as_deref(), Some(THINK_OPEN));
            let (text, opener) = render(controls, json!({"reasoning_effort": "none"})).unwrap();
            assert!(
                text.ends_with("<｜Assistant｜></think>") && opener.is_none(),
                "{text}"
            );
        }
        let v4 = |effort: &str| {
            render(
                ChatControls::DeepseekV4,
                json!({"chat_template_kwargs": {"reasoning_effort": effort}}),
            )
            .map(|(text, _)| text)
        };
        assert_eq!(
            v4("high"),
            render(ChatControls::DeepseekV4, json!({})).map(|(text, _)| text)
        );
        assert!(v4("max")
            .unwrap()
            .contains("Reasoning Effort: Absolute maximum"));
        assert!(v4("low").is_err());
        let forced = |more: Value| {
            let mut fields = json!({"tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}], "tool_choice": "required"});
            fields
                .as_object_mut()
                .unwrap()
                .extend(more.as_object().unwrap().clone());
            render(ChatControls::DeepseekV41, fields)
        };
        assert!(forced(json!({})).unwrap_err().contains("tool_choice"));
        assert!(
            !forced(json!({"reasoning_effort": "none"})).is_err_and(|e| e.contains("tool_choice"))
        );
    }
}
