//! The reasoning controls ATOM renders a model's chats under, so that the
//! router renders their chats to the prompts ATOM encodes: for the models
//! whose chat encoder it adapts, which ATOM picks by the `model_type` in the
//! checkpoint's `config.json` (`chat_encoder_adapters._MODEL_ENCODERS`), and
//! for GLM-5, whose template reads the controls ATOM gives every other model.
//! The rules are ATOM's `reasoning_controls`, with the model's encoder hooks
//! where it has an encoder. `test_requests_tokenize_as_on_atom` checks the
//! router's tokens for DeepSeek-V4.1-Flash and GLM-5.2 chats against ATOM's.

use std::{borrow::Cow, collections::HashMap};

use http::StatusCode;
use serde_json::{json, Value};

use super::{chat_template::render_chat_messages, inference::SentControls};
use crate::{
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage, MessageContent},
        common::ContentPart,
    },
    routers::comm::error::IngressError,
    tokenizer::{chat_template::ChatTemplateContentFormat, traits::Tokenizer},
};

/// A model whose chats ATOM renders under reasoning controls of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatControls {
    /// DeepSeek-V4.1 (`deepseek_v41_encoder`): ATOM encodes its chats with the
    /// checkpoint's `encoding/encoding.py`, which llm-tokenizer ports.
    DeepseekV41,
    /// GLM-5 (`glm_moe_dsa`): ATOM renders its chats with the checkpoint's
    /// template, under the controls of a model without an encoder.
    Glm5,
}

impl ChatControls {
    /// The controls of a checkpoint whose `config.json` has `model_type`.
    pub(crate) fn of_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "deepseek_v41" => Some(Self::DeepseekV41),
            "glm_moe_dsa" => Some(Self::Glm5),
            _ => None,
        }
    }

    /// The prompt ATOM encodes `request` to, rendered with `tokenizer`: a 400
    /// where ATOM rejects the chat, and a 501 where the router cannot tell the
    /// prompt. A server's `--default-chat-template-kwargs` is not known here.
    pub(crate) fn render(
        self,
        request: &ChatCompletionRequest,
        sent: &SentControls,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String, IngressError> {
        match self {
            Self::DeepseekV41 => deepseek_v41::render(request, sent, tokenizer),
            Self::Glm5 => glm5::render(request, sent, tokenizer),
        }
    }
}

mod glm5 {
    use super::*;

    /// The kwarg ATOM's startup probe finds that GLM-5's reasoning switches on,
    /// `true` on and `false` off (`chat_encoders.REASONING_TOGGLES`).
    const TOGGLE: &str = "enable_thinking";

    /// The efforts ATOM gives a template (`reasoning.VALID_TEMPLATE_EFFORTS`).
    const EFFORTS: [&str; 3] = ["low", "high", "max"];

    pub(super) fn render(
        request: &ChatCompletionRequest,
        sent: &SentControls,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String, IngressError> {
        let kwargs = template_kwargs(request, sent).map_err(IngressError::invalid)?;
        let format = Some(ChatTemplateContentFormat::String);
        Ok(
            render_chat_messages(&text_content(request), tokenizer, &kwargs, format)
                .map_err(IngressError::invalid)?
                .text,
        )
    }

    /// The template kwargs ATOM renders `request` with (`api_server._check_chat`),
    /// or why ATOM rejects the request. The top-level `reasoning_effort` is not
    /// one of them.
    pub(super) fn template_kwargs(
        request: &ChatCompletionRequest,
        sent: &SentControls,
    ) -> Result<HashMap<String, Value>, String> {
        let mut kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
        let (enabled, effort) = resolve_reasoning_controls(
            sent.thinking.as_ref(),
            request.reasoning_effort.as_deref(),
        )?;
        if let Some(enabled) = enabled {
            kwargs.insert(TOGGLE.to_string(), json!(enabled));
        }
        if let Some(effort) = effort {
            kwargs.insert("thinking_effort".to_string(), effort);
        }
        Ok(kwargs)
    }

    /// `resolve_reasoning_controls` under ATOM's chat rules: a `thinking.type`
    /// other than `"disabled"` switches reasoning on, and a `"none"` effort
    /// off; `thinking.effort` takes the place of `reasoning_effort`, and an
    /// effort alone switches nothing.
    fn resolve_reasoning_controls(
        thinking: Option<&Value>,
        reasoning_effort: Option<&str>,
    ) -> Result<(Option<bool>, Option<Value>), String> {
        let thinking = match thinking {
            None => None,
            Some(Value::Object(thinking)) => Some(thinking),
            Some(_) => return Err("thinking must be an object".to_string()),
        };
        let field = |key| {
            thinking
                .and_then(|thinking| thinking.get(key))
                .filter(|value| !value.is_null())
        };
        let mut enabled = field("type").map(|kind| *kind != "disabled");
        if reasoning_effort == Some("none") {
            enabled = Some(false);
        }
        let effort = field("effort")
            .cloned()
            .or_else(|| reasoning_effort.map(Value::from));
        Ok((enabled, legacy_chat_effort(effort)?))
    }

    /// `reasoning_controls._legacy_chat_effort`: an integer budget in [1, 100],
    /// or an effort a template ATOM knows reads; any other name is dropped.
    fn legacy_chat_effort(effort: Option<Value>) -> Result<Option<Value>, String> {
        match effort {
            Some(Value::Number(budget))
                if budget
                    .as_u64()
                    .is_some_and(|budget| (1..=100).contains(&budget)) =>
            {
                Ok(Some(Value::Number(budget)))
            }
            Some(Value::Number(_) | Value::Bool(_)) => {
                Err("reasoning effort must be an integer in [1, 100]".to_string())
            }
            Some(Value::String(name)) if EFFORTS.contains(&name.as_str()) => {
                Ok(Some(Value::String(name)))
            }
            _ => Ok(None),
        }
    }

    /// `request` with each message's content as ATOM gives it the template
    /// (`ChatMessage.get_content_text`): a list's text parts, a line apart.
    fn text_content(request: &ChatCompletionRequest) -> Cow<'_, ChatCompletionRequest> {
        let mut request = Cow::Borrowed(request);
        for n in 0..request.messages.len() {
            let parts = match &request.messages[n] {
                ChatMessage::System { content, .. }
                | ChatMessage::User { content, .. }
                | ChatMessage::Tool { content, .. }
                | ChatMessage::Developer { content, .. } => Some(content),
                ChatMessage::Assistant { content, .. } => content.as_ref(),
                ChatMessage::Function { .. } => None,
            };
            let Some(MessageContent::Parts(parts)) = parts else {
                continue;
            };
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let text = MessageContent::Text(text);
            match &mut request.to_mut().messages[n] {
                ChatMessage::System { content, .. }
                | ChatMessage::User { content, .. }
                | ChatMessage::Tool { content, .. }
                | ChatMessage::Developer { content, .. } => *content = text,
                ChatMessage::Assistant { content, .. } => *content = Some(text),
                ChatMessage::Function { .. } => {}
            }
        }
        request
    }
}

mod deepseek_v41 {
    use super::*;

    /// The kwarg ATOM's startup probe finds that V4.1's reasoning switches on,
    /// with its off and on values (`chat_encoders.REASONING_TOGGLES`).
    const TOGGLE: (&str, &str, &str) = ("thinking_mode", "chat", "thinking");

    /// The kwargs `encode_messages` takes but its messages; ATOM passes it no
    /// others.
    const ACCEPTS: [&str; 6] = [
        "thinking_mode",
        "context",
        "drop_thinking",
        "add_default_bos_token",
        "reasoning_effort",
        "return_multi_modal_data",
    ];

    /// What the encoder begins a prompt with unless `add_default_bos_token` is
    /// false, and llm-tokenizer's port always.
    const BOS: &str = "<｜begin▁of▁sentence｜>";

    pub(super) fn render(
        request: &ChatCompletionRequest,
        sent: &SentControls,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String, IngressError> {
        let mut kwargs = template_kwargs(request, sent).map_err(IngressError::invalid)?;
        // The encoder renders only the messages after a `context`; the port
        // does not read it.
        if kwargs.get("context").is_some_and(truthy) {
            return Err(IngressError::new(StatusCode::NOT_IMPLEMENTED, "token_routing_unsupported", "the router does not render a DeepSeek-V4.1 chat after a context; use a policy that does not require exact tokens"));
        }
        // The port reads the mode from `thinking`, not `thinking_mode`.
        let thinking = kwargs.get(TOGGLE.0) == Some(&json!(TOGGLE.2));
        kwargs.insert("thinking".to_string(), json!(thinking));
        // The port joins the text blocks of a list with blank lines, as the
        // encoder does.
        let format = Some(ChatTemplateContentFormat::OpenAI);
        let mut prompt = render_chat_messages(request, tokenizer, &kwargs, format)
            .map_err(IngressError::invalid)?
            .text;
        let bos = kwargs.get("add_default_bos_token").is_none_or(truthy);
        if !bos && prompt.starts_with(BOS) {
            prompt.drain(..BOS.len());
        }
        Ok(prompt)
    }

    /// The template kwargs ATOM encodes `request` with (`api_server._check_chat`,
    /// then the encoder's adapter), or why ATOM rejects the request.
    pub(super) fn template_kwargs(
        request: &ChatCompletionRequest,
        sent: &SentControls,
    ) -> Result<HashMap<String, Value>, String> {
        let mut kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
        let controls = resolve_reasoning_controls(
            sent.thinking.as_ref(),
            request.reasoning_effort.as_deref(),
        )?;
        apply_reasoning_controls(&mut kwargs, controls);
        // The encoder's default, under the kwargs.
        kwargs
            .entry(TOGGLE.0.to_string())
            .or_insert_with(|| json!(TOGGLE.2));
        prepare_kwargs(&mut kwargs)?;
        kwargs.retain(|name, _| ACCEPTS.contains(&name.as_str()));
        Ok(kwargs)
    }

    /// `resolve_reasoning_controls` under V4.1's rules: whether the chat
    /// switches reasoning on or off, if it says, and its effort. A
    /// `thinking.type` other than `"disabled"` switches it on, a `"none"`
    /// effort off, and a named effort on where `thinking` does not say;
    /// `thinking.effort` takes the place of `reasoning_effort`.
    fn resolve_reasoning_controls(
        thinking: Option<&Value>,
        reasoning_effort: Option<&str>,
    ) -> Result<(Option<bool>, Option<Value>), String> {
        let thinking = match thinking {
            None => None,
            Some(Value::Object(thinking)) => Some(thinking),
            Some(_) => return Err("thinking must be an object".to_string()),
        };
        let field = |key| {
            thinking
                .and_then(|thinking| thinking.get(key))
                .filter(|value| !value.is_null())
        };
        let mut enabled = field("type").map(|kind| *kind != "disabled");
        if reasoning_effort == Some("none") {
            enabled = Some(false);
        }
        let effort = field("effort")
            .cloned()
            .or_else(|| reasoning_effort.map(Value::from));
        let effort = normalize_effort(effort.as_ref())?;
        if effort.as_ref().is_some_and(|effort| *effort == "none") {
            return Ok((Some(false), None));
        }
        if enabled.is_none() && effort.as_ref().is_some_and(Value::is_string) {
            enabled = Some(true);
        }
        Ok((enabled, effort))
    }

    /// `apply_reasoning_controls` under V4.1's rules: the switch sets
    /// `thinking_mode`, and the effort `thinking_effort`. An effort alone
    /// keeps reasoning off where the kwargs turned it off with a `"none"`
    /// effort; switching it on drops a `"none"` effort, and off every effort.
    fn apply_reasoning_controls(
        kwargs: &mut HashMap<String, Value>,
        (enabled, effort): (Option<bool>, Option<Value>),
    ) {
        let inherited_off = kwargs
            .get("thinking_effort")
            .or_else(|| kwargs.get("reasoning_effort"))
            .is_some_and(|effort| *effort == "none");
        let switch = match enabled {
            None if effort.is_some() && inherited_off => Some(false),
            enabled => enabled,
        };
        let (toggle, off, on) = TOGGLE;
        if let Some(switch) = switch {
            kwargs.insert(toggle.to_string(), json!(if switch { on } else { off }));
        }
        if let Some(effort) = effort {
            kwargs.insert("thinking_effort".to_string(), effort);
        }
        match enabled {
            Some(true) => {
                for key in ["thinking_effort", "reasoning_effort"] {
                    if kwargs.get(key).is_some_and(|effort| *effort == "none") {
                        kwargs.remove(key);
                    }
                }
            }
            Some(false) => {
                kwargs.remove("thinking_effort");
                kwargs.remove("reasoning_effort");
            }
            None => {}
        }
    }

    /// `deepseek_v41_encoder.prepare_kwargs`: the effort becomes the encoder's
    /// `reasoning_effort`, and a `"none"` one chat mode. It rejects a
    /// `thinking_mode` other than `thinking` or `chat`, and a set
    /// `return_multi_modal_data`.
    fn prepare_kwargs(kwargs: &mut HashMap<String, Value>) -> Result<(), String> {
        if let Some(effort) = kwargs.remove("thinking_effort") {
            kwargs.insert("reasoning_effort".to_string(), effort);
        }
        match normalize_effort(kwargs.get("reasoning_effort"))? {
            Some(effort) if effort == "none" => {
                kwargs.insert(TOGGLE.0.to_string(), json!(TOGGLE.1));
                kwargs.remove("reasoning_effort");
            }
            Some(effort) => {
                kwargs.insert("reasoning_effort".to_string(), effort);
            }
            None => {}
        }
        if !kwargs
            .get(TOGGLE.0)
            .is_none_or(|mode| *mode == TOGGLE.2 || *mode == TOGGLE.1)
        {
            return Err("DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'".to_string());
        }
        if kwargs.get("return_multi_modal_data").is_some_and(truthy) {
            return Err("return_multi_modal_data is not a text chat-template control".to_string());
        }
        Ok(())
    }

    /// `deepseek_v41_encoder.normalize_effort`: an integer budget in [1, 100],
    /// `"none"`, or an effort V4.1 names, the hosted API's aliases taken as
    /// theirs.
    fn normalize_effort(effort: Option<&Value>) -> Result<Option<Value>, String> {
        let invalid = || {
            "DeepSeek-V4.1 reasoning_effort must be none, low, high, max, minimal, medium, \
             xhigh, or an integer in [1, 100]"
                .to_string()
        };
        match effort {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(name)) => match name.as_str() {
                "minimal" => Ok(Some(json!("low"))),
                "medium" | "xhigh" => Ok(Some(json!("high"))),
                "none" | "low" | "high" | "max" => Ok(Some(json!(name))),
                _ => Err(invalid()),
            },
            Some(Value::Number(budget))
                if budget
                    .as_u64()
                    .is_some_and(|budget| (1..=100).contains(&budget)) =>
            {
                Ok(Some(Value::Number(budget.clone())))
            }
            Some(_) => Err(invalid()),
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        routers::{
            prepare::inference::ParsedInference,
            test_mocks::tokenizer::{byte_level_tokenizer_dir, BYTE_LEVEL_MERGES},
        },
        tokenizer::factory::create_tokenizer,
    };

    const EFFORT: &str = "DeepSeek-V4.1 reasoning_effort must be none, low, high, max, minimal, \
                          medium, xhigh, or an integer in [1, 100]";
    const MODE: &str = "DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'";
    const MEDIA: &str = "return_multi_modal_data is not a text chat-template control";

    /// A chat of `fields` as the router parses it, by default a user's "hi".
    fn chat(fields: Value) -> ParsedInference {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        ParsedInference::parse("/v1/chat/completions", body.to_string().as_bytes()).unwrap()
    }

    fn template_kwargs(fields: &Value) -> Result<Value, String> {
        let parsed = chat(fields.clone());
        let (request, sent) = parsed.chat_as_sent().unwrap();
        deepseek_v41::template_kwargs(request, sent).map(|kwargs| json!(kwargs))
    }

    /// The kwargs ATOM encodes V4.1 chats with, or its error, as its
    /// `_check_chat` and encoder adapter give them for the same controls.
    #[test]
    fn deepseek_v41_template_kwargs_are_atoms() {
        let thinking = |mode: &str| json!({"thinking_mode": mode});
        let effort =
            |effort: Value| json!({"reasoning_effort": effort, "thinking_mode": "thinking"});
        let kwargs = |kwargs: Value| json!({"chat_template_kwargs": kwargs});
        let cases = [
            (json!({}), Ok(thinking("thinking"))),
            (json!({"reasoning_effort": "low"}), Ok(effort(json!("low")))),
            (json!({"reasoning_effort": "max"}), Ok(effort(json!("max")))),
            (
                json!({"reasoning_effort": "medium"}),
                Ok(effort(json!("high"))),
            ),
            (
                json!({"reasoning_effort": "xhigh"}),
                Ok(effort(json!("high"))),
            ),
            (
                json!({"reasoning_effort": "minimal"}),
                Ok(effort(json!("low"))),
            ),
            (json!({"reasoning_effort": "none"}), Ok(thinking("chat"))),
            (json!({"reasoning_effort": "ultra"}), Err(EFFORT)),
            (
                json!({"thinking": {"type": "disabled"}}),
                Ok(thinking("chat")),
            ),
            (
                json!({"thinking": {"type": "enabled"}}),
                Ok(thinking("thinking")),
            ),
            (
                json!({"thinking": {"type": "enabled", "effort": 30}}),
                Ok(effort(json!(30))),
            ),
            (json!({"thinking": {"effort": 30}}), Ok(effort(json!(30)))),
            (json!({"thinking": {"effort": 0}}), Err(EFFORT)),
            (json!({"thinking": {"effort": 30.0}}), Err(EFFORT)),
            (json!({"thinking": {"effort": true}}), Err(EFFORT)),
            (
                json!({"thinking": {"type": "enabled"}, "reasoning_effort": "none"}),
                Ok(thinking("chat")),
            ),
            (
                json!({"thinking": {"type": "disabled"}, "reasoning_effort": "high"}),
                Ok(thinking("chat")),
            ),
            (
                json!({"thinking": {"effort": "max"}, "reasoning_effort": "low"}),
                Ok(effort(json!("max"))),
            ),
            (
                kwargs(json!({"enable_thinking": false})),
                Ok(thinking("thinking")),
            ),
            (
                kwargs(json!({"thinking_mode": "chat"})),
                Ok(thinking("chat")),
            ),
            (
                json!({"chat_template_kwargs": {"thinking_mode": "chat"}, "reasoning_effort": "high"}),
                Ok(effort(json!("high"))),
            ),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": "low"}, "reasoning_effort": "max"}),
                Ok(effort(json!("max"))),
            ),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": "none"}, "thinking": {"effort": 30}}),
                Ok(json!({"reasoning_effort": 30, "thinking_mode": "chat"})),
            ),
            (
                kwargs(json!({"reasoning_effort": "none"})),
                Ok(thinking("chat")),
            ),
            (
                kwargs(json!({"reasoning_effort": "medium"})),
                Ok(effort(json!("high"))),
            ),
            (
                json!({"chat_template_kwargs": {"thinking_effort": "none"}, "thinking": {"type": "enabled"}}),
                Ok(thinking("thinking")),
            ),
            (kwargs(json!({"thinking_mode": "disabled"})), Err(MODE)),
            (kwargs(json!({"return_multi_modal_data": true})), Err(MEDIA)),
            (
                kwargs(json!({"return_multi_modal_data": false})),
                Ok(json!({"return_multi_modal_data": false, "thinking_mode": "thinking"})),
            ),
            (
                kwargs(
                    json!({"drop_thinking": false, "add_default_bos_token": false, "answer_style": "terse"}),
                ),
                Ok(
                    json!({"add_default_bos_token": false, "drop_thinking": false, "thinking_mode": "thinking"}),
                ),
            ),
        ];
        for (fields, want) in cases {
            assert_eq!(
                template_kwargs(&fields),
                want.map_err(str::to_string),
                "{fields}"
            );
        }
        // ATOM's request model takes only an object.
        assert!(template_kwargs(&json!({"thinking": "on"})).is_err());
    }

    /// V4.1 chats render with the template under those kwargs, a message's
    /// text blocks listed for it to join, and without the BOS where the
    /// encoder leaves it out; one after a `context` the router cannot render.
    #[test]
    fn deepseek_v41_chats_render_as_the_encoder_encodes_them() {
        let template = concat!(
            "{{- '<｜begin▁of▁sentence｜>' -}}",
            "{%- for m in messages -%}{{ m.role }}:{%- set blocks = m.content -%}",
            "{%- if blocks is string -%}{{ blocks }}",
            "{%- else -%}{%- for block in blocks -%}[{{ block.text }}]{%- endfor -%}",
            "{%- endif -%};{%- endfor -%}",
            "{{ thinking_mode }}/{{ reasoning_effort | default('-') }}",
        );
        let dir =
            byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, false, json!({"chat_template": template}));
        let tokenizer =
            create_tokenizer(dir.path().join("tokenizer.json").to_str().unwrap()).unwrap();
        // As V4.1's, the template does not read as one that takes text blocks.
        assert_eq!(
            tokenizer.chat_template_content_format(),
            ChatTemplateContentFormat::String
        );
        let render = |fields: Value| {
            let parsed = chat(fields);
            let (request, sent) = parsed.chat_as_sent().unwrap();
            ChatControls::DeepseekV41
                .render(request, sent, &*tokenizer)
                .map_err(|error| (error.status, error.code))
        };
        let parts =
            json!([{"type": "text", "text": "Read"}, {"type": "text", "text": "README.md."}]);
        let cases = [
            (json!({}), Ok("<｜begin▁of▁sentence｜>user:hi;thinking/-")),
            (
                json!({"reasoning_effort": "medium"}),
                Ok("<｜begin▁of▁sentence｜>user:hi;thinking/high"),
            ),
            (
                json!({"messages": [{"role": "user", "content": parts}]}),
                Ok("<｜begin▁of▁sentence｜>user:[Read][README.md.];thinking/-"),
            ),
            (
                json!({"chat_template_kwargs": {"add_default_bos_token": false}}),
                Ok("user:hi;thinking/-"),
            ),
            (
                json!({"chat_template_kwargs": {"context": [{"role": "user", "content": "hi"}]}}),
                Err((StatusCode::NOT_IMPLEMENTED, "token_routing_unsupported")),
            ),
            (
                json!({"chat_template_kwargs": {"thinking_mode": "disabled"}}),
                Err((StatusCode::BAD_REQUEST, "invalid_request")),
            ),
        ];
        for (fields, want) in cases {
            assert_eq!(render(fields.clone()), want.map(str::to_string), "{fields}");
        }
    }

    fn glm5_template_kwargs(fields: &Value) -> Result<Value, String> {
        let parsed = chat(fields.clone());
        let (request, sent) = parsed.chat_as_sent().unwrap();
        glm5::template_kwargs(request, sent).map(|kwargs| json!(kwargs))
    }

    /// The kwargs ATOM renders GLM-5 chats with, or its error, as its
    /// `_check_chat` gives them for the same controls.
    #[test]
    fn glm5_template_kwargs_are_atoms() {
        let thinking = |enabled: bool| json!({"enable_thinking": enabled});
        let budget = "reasoning effort must be an integer in [1, 100]";
        let cases = [
            (json!({}), Ok(json!({}))),
            (
                json!({"thinking": {"type": "disabled"}}),
                Ok(thinking(false)),
            ),
            (json!({"thinking": {"type": "enabled"}}), Ok(thinking(true))),
            (json!({"reasoning_effort": "none"}), Ok(thinking(false))),
            (
                json!({"reasoning_effort": "high"}),
                Ok(json!({"thinking_effort": "high"})),
            ),
            (json!({"reasoning_effort": "medium"}), Ok(json!({}))),
            (
                json!({"thinking": {"type": "enabled", "effort": 30}}),
                Ok(json!({"enable_thinking": true, "thinking_effort": 30})),
            ),
            (
                json!({"thinking": {"type": "enabled"}, "reasoning_effort": "none"}),
                Ok(thinking(false)),
            ),
            (json!({"thinking": {"effort": 0}}), Err(budget)),
            (json!({"thinking": {"effort": 30.0}}), Err(budget)),
            (json!({"thinking": {"effort": true}}), Err(budget)),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": "high"}}),
                Ok(json!({"reasoning_effort": "high"})),
            ),
            (
                json!({"chat_template_kwargs": {"enable_thinking": false}, "thinking": {"type": "enabled"}}),
                Ok(thinking(true)),
            ),
        ];
        for (fields, want) in cases {
            assert_eq!(
                glm5_template_kwargs(&fields),
                want.map_err(str::to_string),
                "{fields}"
            );
        }
        // ATOM's request model takes only an object.
        assert!(glm5_template_kwargs(&json!({"thinking": "on"})).is_err());
    }

    /// GLM-5 chats render with the template under those kwargs, and a
    /// message's text parts a line apart, as ATOM gives them.
    #[test]
    fn glm5_chats_render_as_atom_renders_them() {
        let template = concat!(
            "{%- for m in messages -%}{{ m.role }}:{{ m.content }};{%- endfor -%}",
            "{{ enable_thinking | default('-') }}/{{ reasoning_effort | default('-') }}",
        );
        let dir =
            byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, false, json!({"chat_template": template}));
        let tokenizer =
            create_tokenizer(dir.path().join("tokenizer.json").to_str().unwrap()).unwrap();
        let render = |fields: Value| {
            let parsed = chat(fields);
            let (request, sent) = parsed.chat_as_sent().unwrap();
            ChatControls::Glm5
                .render(request, sent, &*tokenizer)
                .map_err(|error| (error.status, error.code))
        };
        let parts =
            json!([{"type": "text", "text": "Read"}, {"type": "text", "text": "README.md."}]);
        let cases = [
            (json!({}), Ok("user:hi;-/-")),
            (json!({"reasoning_effort": "high"}), Ok("user:hi;-/-")),
            (
                json!({"thinking": {"type": "disabled"}}),
                Ok("user:hi;False/-"),
            ),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": "high"}}),
                Ok("user:hi;-/high"),
            ),
            (
                json!({"messages": [{"role": "user", "content": parts}]}),
                Ok("user:Read\nREADME.md.;-/-"),
            ),
            (
                json!({"thinking": {"effort": 0}}),
                Err((StatusCode::BAD_REQUEST, "invalid_request")),
            ),
        ];
        for (fields, want) in cases {
            assert_eq!(render(fields.clone()), want.map(str::to_string), "{fields}");
        }
    }
}
