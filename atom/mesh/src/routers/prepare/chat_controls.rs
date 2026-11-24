//! The reasoning controls ATOM renders a model's chats under, for the models
//! whose chat encoder it adapts. ATOM picks them by the `model_type` in the
//! checkpoint's `config.json` (`chat_encoder_adapters._MODEL_ENCODERS`). HTTP
//! routing resolves such a chat's effort and mode as ATOM does, for
//! llm-tokenizer renders it with its port of the model's encoder, whose
//! defaults are not ATOM's; ATOM renders the chat again, so a control the
//! router renders otherwise costs prefix hits, not a different prompt.
//! `test_requests_tokenize_as_on_atom` checks the router's tokens for
//! DeepSeek-V4.1-Flash chats against ATOM's.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::chat_template::render_chat_messages;
use crate::{protocols::chat::ChatCompletionRequest, tokenizer::traits::Tokenizer};

/// A model whose chats ATOM renders under reasoning controls of its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChatControls {
    /// DeepSeek-V4.1 (`deepseek_v41_encoder`).
    DeepseekV41,
}

impl ChatControls {
    /// The controls of a checkpoint whose `config.json` has `model_type`.
    pub(crate) fn of_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "deepseek_v41" => Some(Self::DeepseekV41),
            _ => None,
        }
    }

    /// The prompt `request` renders to with `tokenizer`, under the kwargs
    /// ATOM encodes it with, or why ATOM rejects it.
    pub(crate) fn render(
        self,
        request: &ChatCompletionRequest,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String, String> {
        let kwargs = match self {
            Self::DeepseekV41 => deepseek_v41_kwargs(request)?,
        };
        Ok(render_chat_messages(request, tokenizer, &kwargs)?.text)
    }
}

/// The kwarg that switches V4.1's reasoning, `"chat"` for off and `"thinking"`
/// for on (`chat_encoders.REASONING_TOGGLES`).
const MODE: &str = "thinking_mode";

/// The kwargs V4.1's `encode_messages` takes besides its messages. ATOM's
/// adapter passes it no others.
const ACCEPTS: [&str; 6] = [
    MODE,
    "context",
    "drop_thinking",
    "add_default_bos_token",
    "reasoning_effort",
    "return_multi_modal_data",
];

/// The kwargs ATOM encodes a V4.1 chat with, or why it rejects the chat, and
/// the mode in `thinking`, where llm-tokenizer's port of the encoder reads it.
/// ATOM puts the request's effort over `chat_template_kwargs`
/// (`apply_reasoning_controls` under V4.1's `REASONING_RULES`), then runs the
/// encoder's `prepare_kwargs` and drops what `encode_messages` does not take,
/// `enable_thinking` among it. A top-level `thinking` is not read, as the
/// request type does not keep it.
fn deepseek_v41_kwargs(request: &ChatCompletionRequest) -> Result<HashMap<String, Value>, String> {
    let mut kwargs = request.chat_template_kwargs.clone().unwrap_or_default();
    let effort = request.reasoning_effort.as_deref().map(Value::from);
    match normalize_effort(effort.as_ref())? {
        None => {}
        // `"none"` switches reasoning off, and drops the kwargs' efforts.
        Some(effort) if effort == "none" => {
            kwargs.insert(MODE.to_string(), json!("chat"));
            kwargs.remove("thinking_effort");
            kwargs.remove("reasoning_effort");
        }
        // A named effort switches it on, and goes over the kwargs' efforts.
        Some(effort) => {
            kwargs.insert(MODE.to_string(), json!("thinking"));
            kwargs.insert("thinking_effort".to_string(), effort);
            if kwargs.get("reasoning_effort") == Some(&json!("none")) {
                kwargs.remove("reasoning_effort");
            }
        }
    }
    // The adapter's default, under the kwargs.
    kwargs
        .entry(MODE.to_string())
        .or_insert_with(|| json!("thinking"));
    prepare_kwargs(&mut kwargs)?;
    kwargs.retain(|name, _| ACCEPTS.contains(&name.as_str()));
    let thinking = kwargs.get(MODE) == Some(&json!("thinking"));
    kwargs.insert("thinking".to_string(), json!(thinking));
    Ok(kwargs)
}

/// `deepseek_v41_encoder.prepare_kwargs`: a `thinking_effort` becomes the
/// encoder's `reasoning_effort`, and a `"none"` effort chat mode. It rejects a
/// `thinking_mode` other than `thinking` or `chat`, and a set
/// `return_multi_modal_data`.
fn prepare_kwargs(kwargs: &mut HashMap<String, Value>) -> Result<(), String> {
    if let Some(effort) = kwargs.remove("thinking_effort") {
        kwargs.insert("reasoning_effort".to_string(), effort);
    }
    match normalize_effort(kwargs.get("reasoning_effort"))? {
        Some(effort) if effort == "none" => {
            kwargs.insert(MODE.to_string(), json!("chat"));
            kwargs.remove("reasoning_effort");
        }
        Some(effort) => {
            kwargs.insert("reasoning_effort".to_string(), effort);
        }
        None => {}
    }
    if !kwargs
        .get(MODE)
        .is_none_or(|mode| *mode == "thinking" || *mode == "chat")
    {
        return Err("DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'".to_string());
    }
    if kwargs.get("return_multi_modal_data").is_some_and(truthy) {
        return Err("return_multi_modal_data is not a text chat-template control".to_string());
    }
    Ok(())
}

/// `deepseek_v41_encoder.normalize_effort`: `"none"`, an effort V4.1 names,
/// the hosted API's `minimal` as `low` and `medium` and `xhigh` as `high`, or
/// an integer budget in [1, 100].
fn normalize_effort(effort: Option<&Value>) -> Result<Option<Value>, String> {
    let invalid = || {
        "DeepSeek-V4.1 reasoning_effort must be none, low, high, max, minimal, medium, xhigh, \
         or an integer in [1, 100]"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        routers::{prepare::inference::ParsedInference, test_mocks::tokenizer::hf_tokenizer_path},
        tokenizer::factory::create_tokenizer,
    };

    const EFFORT: &str = "DeepSeek-V4.1 reasoning_effort must be none, low, high, max, minimal, \
                          medium, xhigh, or an integer in [1, 100]";
    const MODE_ERROR: &str = "DeepSeek-V4.1 thinking_mode must be 'thinking' or 'chat'";
    const MEDIA: &str = "return_multi_modal_data is not a text chat-template control";

    /// A V4.1 chat of `fields`, by default a user's "hi".
    fn chat(fields: &Value) -> ParsedInference {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        ParsedInference::parse("/v1/chat/completions", body.to_string().as_bytes()).unwrap()
    }

    /// V4.1 chats get the kwargs, or the error, that ATOM's
    /// `apply_reasoning_controls`, encoder adapter and `prepare_kwargs` give
    /// the same controls, and `thinking` where the mode is thinking.
    #[test]
    fn deepseek_v41_kwargs_are_atoms() {
        let mode = |mode: &str| Ok(json!({"thinking_mode": mode}));
        let thinking =
            |effort: Value| Ok(json!({"thinking_mode": "thinking", "reasoning_effort": effort}));
        let kwargs = |kwargs: Value| json!({"chat_template_kwargs": kwargs});
        let over = |effort: &str, kwargs: Value| json!({"reasoning_effort": effort, "chat_template_kwargs": kwargs});
        let cases: Vec<(Value, Result<Value, &str>)> = vec![
            (json!({}), mode("thinking")),
            // The request's effort, the hosted API's aliases as V4.1's.
            (json!({"reasoning_effort": "low"}), thinking(json!("low"))),
            (json!({"reasoning_effort": "high"}), thinking(json!("high"))),
            (json!({"reasoning_effort": "max"}), thinking(json!("max"))),
            (
                json!({"reasoning_effort": "minimal"}),
                thinking(json!("low")),
            ),
            (
                json!({"reasoning_effort": "medium"}),
                thinking(json!("high")),
            ),
            (
                json!({"reasoning_effort": "xhigh"}),
                thinking(json!("high")),
            ),
            (json!({"reasoning_effort": "none"}), mode("chat")),
            (json!({"reasoning_effort": "ultra"}), Err(EFFORT)),
            // The kwargs' effort.
            (
                kwargs(json!({"reasoning_effort": "minimal"})),
                thinking(json!("low")),
            ),
            (kwargs(json!({"reasoning_effort": 30})), thinking(json!(30))),
            (kwargs(json!({"reasoning_effort": "none"})), mode("chat")),
            (
                kwargs(json!({"thinking_effort": "max", "reasoning_effort": "low"})),
                thinking(json!("max")),
            ),
            (kwargs(json!({"reasoning_effort": 101})), Err(EFFORT)),
            (kwargs(json!({"reasoning_effort": 30.5})), Err(EFFORT)),
            (kwargs(json!({"reasoning_effort": true})), Err(EFFORT)),
            // The request's effort goes over the kwargs' effort and mode.
            (
                over("max", json!({"reasoning_effort": "low"})),
                thinking(json!("max")),
            ),
            (
                over("high", json!({"reasoning_effort": "none"})),
                thinking(json!("high")),
            ),
            (
                over("high", json!({"thinking_mode": "chat"})),
                thinking(json!("high")),
            ),
            (
                over("none", json!({"thinking_mode": "thinking"})),
                mode("chat"),
            ),
            (
                over("none", json!({"reasoning_effort": "high"})),
                mode("chat"),
            ),
            // The kwargs' mode. The encoder takes neither `enable_thinking`
            // nor `thinking`.
            (kwargs(json!({"thinking_mode": "chat"})), mode("chat")),
            (
                kwargs(json!({"thinking_mode": "chat", "reasoning_effort": "high"})),
                Ok(json!({"thinking_mode": "chat", "reasoning_effort": "high"})),
            ),
            (kwargs(json!({"enable_thinking": false})), mode("thinking")),
            (kwargs(json!({"thinking": false})), mode("thinking")),
            (kwargs(json!({"thinking_mode": "auto"})), Err(MODE_ERROR)),
            (kwargs(json!({"thinking_mode": null})), Err(MODE_ERROR)),
            // The encoder's other kwargs pass, and no others.
            (
                over("xhigh", json!({"drop_thinking": false})),
                Ok(
                    json!({"thinking_mode": "thinking", "reasoning_effort": "high", "drop_thinking": false}),
                ),
            ),
            (
                kwargs(
                    json!({"context": [], "add_default_bos_token": false, "tools": [], "foo": 1}),
                ),
                Ok(
                    json!({"thinking_mode": "thinking", "context": [], "add_default_bos_token": false}),
                ),
            ),
            (
                kwargs(json!({"return_multi_modal_data": false})),
                Ok(json!({"thinking_mode": "thinking", "return_multi_modal_data": false})),
            ),
            (kwargs(json!({"return_multi_modal_data": true})), Err(MEDIA)),
        ];
        for (fields, want) in cases {
            let got = deepseek_v41_kwargs(chat(&fields).chat().unwrap());
            let want = want.map(|mut kwargs| {
                kwargs["thinking"] = json!(kwargs["thinking_mode"] == "thinking");
                kwargs
            });
            assert_eq!(
                got.map(|kwargs| json!(kwargs)),
                want.map_err(str::to_string),
                "{fields}"
            );
        }
    }

    /// llm-tokenizer's port of V4.1's encoder renders the mode the controls
    /// give a chat. Alone, it renders chat mode for a `minimal` effort and an
    /// `enable_thinking` of false.
    #[test]
    fn deepseek_v41_port_renders_the_mode_the_controls_give() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::copy(hf_tokenizer_path(), dir.path().join("tokenizer.json")).unwrap();
        let config = json!({"model_type": "deepseek_v41"});
        std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
        let tokenizer =
            create_tokenizer(dir.path().join("tokenizer.json").to_str().unwrap()).unwrap();
        for (fields, prefill) in [
            (json!({}), "<think>"),
            (json!({"reasoning_effort": "minimal"}), "<think>"),
            (json!({"reasoning_effort": "none"}), "</think>"),
            (
                json!({"chat_template_kwargs": {"thinking_mode": "chat"}}),
                "</think>",
            ),
            (
                json!({"chat_template_kwargs": {"enable_thinking": false}}),
                "<think>",
            ),
        ] {
            let parsed = chat(&fields);
            let prompt = ChatControls::DeepseekV41
                .render(parsed.chat().unwrap(), &*tokenizer)
                .unwrap();
            let end = format!("<｜Assistant｜>{prefill}");
            assert!(prompt.ends_with(&end), "{fields}: {prompt}");
        }
    }
}
