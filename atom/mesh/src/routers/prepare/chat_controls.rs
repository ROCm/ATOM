//! The reasoning controls ATOM renders a model's chats under, for the models
//! whose chat encoder it adapts. ATOM picks them by the `model_type` in the
//! checkpoint's `config.json` (`chat_encoder_adapters._MODEL_ENCODERS`). HTTP
//! routing takes such a chat's effort as ATOM does, so that it renders the
//! chats whose effort the template rejects and ATOM accepts; ATOM renders the
//! chat again, so a control the router renders otherwise costs prefix hits,
//! not a different prompt.
//! `test_requests_tokenize_as_on_atom` checks the router's tokens for
//! DeepSeek-V4.1-Flash chats against ATOM's.

use std::collections::HashMap;

use serde_json::{json, Value};

use super::chat_template::{render_chat_messages, template_kwargs};
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

    /// The prompt `request` renders to with `tokenizer`'s template, its effort
    /// taken as ATOM takes it.
    pub(crate) fn render(
        self,
        request: &ChatCompletionRequest,
        tokenizer: &dyn Tokenizer,
    ) -> Result<String, String> {
        let mut kwargs = template_kwargs(request);
        match self {
            Self::DeepseekV41 => deepseek_v41_effort(&mut kwargs),
        }
        Ok(render_chat_messages(request, tokenizer, &kwargs)?.text)
    }
}

/// The effort as V4.1's encoder takes it: `minimal` as `low`, `medium` and
/// `xhigh` as `high` (`normalize_effort`), and `none` as chat mode
/// (`prepare_kwargs`). V4.1's template rejects `minimal`, `xhigh` and `none`,
/// and gives `medium` a smaller budget than `high`; other efforts reach it as
/// sent.
fn deepseek_v41_effort(kwargs: &mut HashMap<String, Value>) {
    let effort = match kwargs.get("reasoning_effort").and_then(Value::as_str) {
        Some("minimal") => "low",
        Some("medium" | "xhigh") => "high",
        Some("none") => {
            kwargs.remove("reasoning_effort");
            kwargs.insert("thinking_mode".to_string(), json!("chat"));
            return;
        }
        _ => return,
    };
    kwargs.insert("reasoning_effort".to_string(), json!(effort));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routers::prepare::inference::ParsedInference;

    /// The template kwargs of a V4.1 chat of `fields`, by default a user's "hi".
    fn kwargs(fields: &Value) -> Value {
        let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let parsed =
            ParsedInference::parse("/v1/chat/completions", body.to_string().as_bytes()).unwrap();
        let mut kwargs = template_kwargs(parsed.chat().unwrap());
        deepseek_v41_effort(&mut kwargs);
        json!(kwargs)
    }

    /// V4.1 chats reach the template with the effort ATOM's encoder takes.
    #[test]
    fn deepseek_v41_efforts_are_atoms() {
        let effort = |effort: Value| json!({"reasoning_effort": effort});
        let chat_mode = json!({"thinking_mode": "chat"});
        let cases = [
            (json!({}), json!({})),
            (json!({"reasoning_effort": "low"}), effort(json!("low"))),
            (json!({"reasoning_effort": "high"}), effort(json!("high"))),
            (json!({"reasoning_effort": "max"}), effort(json!("max"))),
            (json!({"reasoning_effort": "minimal"}), effort(json!("low"))),
            (json!({"reasoning_effort": "medium"}), effort(json!("high"))),
            (json!({"reasoning_effort": "xhigh"}), effort(json!("high"))),
            (json!({"reasoning_effort": "none"}), chat_mode.clone()),
            // ATOM rejects it too.
            (json!({"reasoning_effort": "ultra"}), effort(json!("ultra"))),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": "minimal"}}),
                effort(json!("low")),
            ),
            (
                json!({"chat_template_kwargs": {"reasoning_effort": 30}}),
                effort(json!(30)),
            ),
            (
                json!({"reasoning_effort": "none", "chat_template_kwargs": {"thinking_mode": "thinking"}}),
                chat_mode,
            ),
            (
                json!({"reasoning_effort": "xhigh", "chat_template_kwargs": {"drop_thinking": false}}),
                json!({"drop_thinking": false, "reasoning_effort": "high"}),
            ),
        ];
        for (fields, want) in cases {
            assert_eq!(kwargs(&fields), want, "{fields}");
        }
    }
}
