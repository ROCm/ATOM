//! Chat-template rendering helpers: turn `ChatMessage` lists into the JSON
//! shape that HuggingFace chat templates consume.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    protocols::{
        chat::{ChatCompletionRequest, ChatMessage},
        common::StringOrArray,
    },
    tokenizer::{
        cache::CachedTokenizer, chat_template::ChatTemplateContentFormat,
        chat_template::ChatTemplateParams, traits::Tokenizer, HuggingFaceTokenizer,
    },
};

/// Output of `process_chat_messages`: rendered prompt + the request's stop
/// sequences (echoed back so the engine layer doesn't have to re-walk the
/// request body).
#[derive(Debug)]
pub struct ProcessedMessages {
    pub text: String,
    pub stop_sequences: Option<StringOrArray>,
}

/// Process chat messages and apply the HuggingFace chat template.
///
/// Requires a HuggingFace tokenizer (directly or via `CachedTokenizer`); other
/// tokenizer kinds return an error because the gRPC path expects a chat-template
/// rendered prompt.
pub fn process_chat_messages(
    request: &ChatCompletionRequest,
    tokenizer: &dyn Tokenizer,
) -> Result<ProcessedMessages, String> {
    let hf_tokenizer = tokenizer
        .as_any()
        .downcast_ref::<HuggingFaceTokenizer>()
        .or_else(|| {
            tokenizer
                .as_any()
                .downcast_ref::<CachedTokenizer>()
                .and_then(|cached| {
                    cached
                        .inner()
                        .as_any()
                        .downcast_ref::<HuggingFaceTokenizer>()
                })
        });

    let Some(hf_tokenizer) = hf_tokenizer else {
        return Err(
            "gRPC router requires HuggingFace tokenizer with chat template support".to_string(),
        );
    };

    let content_format = hf_tokenizer.chat_template_content_format();
    let mut transformed_messages = process_content_format(&request.messages, content_format)?;

    process_tool_call_arguments(&mut transformed_messages)?;

    let tools_json: Option<Vec<Value>> = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(serde_json::to_value)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| format!("Failed to serialize tools: {}", e))?;

    let kwargs_capacity = 1 + request.chat_template_kwargs.as_ref().map_or(0, |k| k.len());
    let mut combined_template_kwargs = HashMap::with_capacity(kwargs_capacity);

    if let Some(reasoning_effort) = &request.reasoning_effort {
        combined_template_kwargs.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort.clone()),
        );
    }

    if let Some(template_kwargs) = &request.chat_template_kwargs {
        for (key, value) in template_kwargs {
            combined_template_kwargs.insert(key.clone(), value.clone());
        }
    }

    let final_template_kwargs = if combined_template_kwargs.is_empty() {
        None
    } else {
        Some(&combined_template_kwargs)
    };

    let params = ChatTemplateParams {
        add_generation_prompt: true,
        tools: tools_json.as_deref(),
        template_kwargs: final_template_kwargs,
        ..Default::default()
    };

    let assistant_prefix = if request.continue_final_message
        && !transformed_messages.is_empty()
        && transformed_messages
            .last()
            .and_then(|msg| msg.get("role"))
            .and_then(|v| v.as_str())
            == Some("assistant")
    {
        let last_msg = transformed_messages.pop().unwrap();
        last_msg
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    };

    let rendered = hf_tokenizer
        .apply_chat_template(&transformed_messages, params)
        .map_err(|e| format!("Failed to apply chat template: {}", e))?;

    let formatted_text = if let Some(prefix) = assistant_prefix {
        format!("{}{}", rendered, prefix)
    } else {
        rendered
    };

    Ok(ProcessedMessages {
        text: formatted_text,
        stop_sequences: request.stop.clone(),
    })
}

/// Transformers chat templates expect assistant tool-call arguments as JSON
/// objects, not as serialized JSON strings.
pub(crate) fn process_tool_call_arguments(messages: &mut [Value]) -> Result<(), String> {
    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str());
        if role != Some("assistant") {
            continue;
        }

        let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|tc| tc.as_array_mut()) else {
            continue;
        };

        for call in tool_calls {
            let Some(function) = call.get_mut("function") else {
                continue;
            };
            let Some(args) = function.get_mut("arguments") else {
                continue;
            };
            let Some(args_str) = args.as_str() else {
                continue;
            };

            // Parse JSON string to object (like Python json.loads)
            match serde_json::from_str::<Value>(args_str) {
                Ok(parsed) => *args = parsed,
                Err(e) => {
                    return Err(format!(
                        "Failed to parse tool call arguments as JSON: '{}'. Error: {}",
                        args_str, e
                    ))
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn process_content_format(
    messages: &[ChatMessage],
    content_format: ChatTemplateContentFormat,
) -> Result<Vec<Value>, String> {
    messages
        .iter()
        .map(|message| {
            let mut message_json = serde_json::to_value(message)
                .map_err(|e| format!("Failed to serialize message: {}", e))?;

            if let Some(obj) = message_json.as_object_mut() {
                if let Some(content_value) = obj.get_mut("content") {
                    transform_content_field(content_value, content_format);
                }
            }

            Ok(message_json)
        })
        .collect()
}

/// Transform a single content field based on content format
fn transform_content_field(content_value: &mut Value, content_format: ChatTemplateContentFormat) {
    let Some(content_array) = content_value.as_array() else {
        return; // Not multimodal, keep as-is
    };

    match content_format {
        ChatTemplateContentFormat::String => {
            // Extract and join text parts only
            let text_parts: Vec<String> = content_array
                .iter()
                .filter_map(|part| {
                    part.as_object()?
                        .get("type")?
                        .as_str()
                        .filter(|&t| t == "text")
                        .and_then(|_| part.as_object()?.get("text")?.as_str())
                        .map(String::from)
                })
                .collect();

            if !text_parts.is_empty() {
                *content_value = Value::String(text_parts.join(" "));
            }
        }
        ChatTemplateContentFormat::OpenAI => {
            // Replace media URLs with simple type placeholders
            let processed_parts: Vec<Value> = content_array
                .iter()
                .map(|part| {
                    part.as_object()
                        .and_then(|obj| obj.get("type")?.as_str())
                        .and_then(|type_str| match type_str {
                            "image_url" => Some(json!({"type": "image"})),
                            "video_url" => Some(json!({"type": "video"})),
                            "audio_url" => Some(json!({"type": "audio"})),
                            _ => None,
                        })
                        .unwrap_or_else(|| part.clone())
                })
                .collect();

            *content_value = Value::Array(processed_parts);
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        protocols::{
            chat::{ChatMessage, MessageContent},
            common::{ContentPart, ImageUrl},
        },
        tokenizer::chat_template::ChatTemplateContentFormat,
    };

    #[test]
    fn test_transform_messages_string_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Hello".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::Text {
                    text: "World".to_string(),
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Hello World"
        );
        assert_eq!(transformed_message["role"].as_str().unwrap(), "user");
    }

    #[test]
    fn test_transform_messages_openai_format() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "Describe this image:".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: Some("high".to_string()),
                    },
                },
            ]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        let content_array = transformed_message["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);

        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[0]["text"], "Describe this image:");

        assert_eq!(content_array[1], json!({"type": "image"}));
    }

    #[test]
    fn test_transform_messages_simple_string_content() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Text("Simple text message".to_string()),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        assert_eq!(
            transformed_message["content"].as_str().unwrap(),
            "Simple text message"
        );
    }

    #[test]
    fn test_transform_messages_multiple_messages() {
        let messages = vec![
            ChatMessage::System {
                content: MessageContent::Text("System prompt".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "User message".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: None,
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(result[0]["role"].as_str().unwrap(), "system");
        assert_eq!(result[0]["content"].as_str().unwrap(), "System prompt");

        assert_eq!(result[1]["role"].as_str().unwrap(), "user");
        assert_eq!(result[1]["content"].as_str().unwrap(), "User message");
    }

    #[test]
    fn test_transform_messages_empty_text_parts() {
        let messages = vec![ChatMessage::User {
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: None,
                },
            }]),
            name: None,
        }];

        let result = process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result.len(), 1);
        let transformed_message = &result[0];

        assert!(transformed_message["content"].is_array());
    }

    #[test]
    fn test_transform_messages_mixed_content_types() {
        let messages = vec![
            ChatMessage::User {
                content: MessageContent::Text("Plain text".to_string()),
                name: None,
            },
            ChatMessage::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "With image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/image.jpg".to_string(),
                            detail: Some("low".to_string()),
                        },
                    },
                ]),
                name: None,
            },
        ];

        let result_string =
            process_content_format(&messages, ChatTemplateContentFormat::String).unwrap();

        assert_eq!(result_string.len(), 2);
        assert_eq!(result_string[0]["content"].as_str().unwrap(), "Plain text");
        assert_eq!(result_string[1]["content"].as_str().unwrap(), "With image");

        let result_openai =
            process_content_format(&messages, ChatTemplateContentFormat::OpenAI).unwrap();

        assert_eq!(result_openai.len(), 2);
        assert_eq!(result_openai[0]["content"].as_str().unwrap(), "Plain text");

        let content_array = result_openai[1]["content"].as_array().unwrap();
        assert_eq!(content_array.len(), 2);
        assert_eq!(content_array[0]["type"], "text");
        assert_eq!(content_array[1], json!({"type": "image"}));
    }
}
