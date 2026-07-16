//! Conversion between /v1/responses and /v1/chat/completions request/response
//! shapes, so the gRPC chat pipeline backs both endpoints.

use crate::{
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
        common::{
            FunctionCallResponse, JsonSchemaFormat, ResponseFormat, StreamOptions, ToolCall,
            UsageInfo,
        },
        responses::{
            ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponseOutputItem,
            ResponseReasoningContent::ReasoningText, ResponseStatus, ResponsesRequest,
            ResponsesResponse, ResponsesUsage, StringOrContentParts, TextConfig, TextFormat,
        },
        UNKNOWN_MODEL_ID,
    },
    routers::openai::responses::persistence::extract_tools_from_response_tools,
};

pub(crate) fn responses_to_chat(req: &ResponsesRequest) -> Result<ChatCompletionRequest, String> {
    let mut messages = Vec::new();

    if let Some(instructions) = &req.instructions {
        messages.push(ChatMessage::System {
            content: MessageContent::Text(instructions.clone()),
            name: None,
        });
    }

    match &req.input {
        ResponseInput::Text(text) => {
            messages.push(ChatMessage::User {
                content: MessageContent::Text(text.clone()),
                name: None,
            });
        }
        ResponseInput::Items(items) => {
            for item in items {
                match item {
                    ResponseInputOutputItem::SimpleInputMessage { content, role, .. } => {
                        let text = match content {
                            StringOrContentParts::String(s) => s.clone(),
                            StringOrContentParts::Array(parts) => {
                                // Only InputText parts contribute to the chat message text.
                                parts
                                    .iter()
                                    .filter_map(|part| match part {
                                        ResponseContentPart::InputText { text } => {
                                            Some(text.as_str())
                                        }
                                        _ => None,
                                    })
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            }
                        };

                        messages.push(role_to_chat_message(role.as_str(), text));
                    }
                    ResponseInputOutputItem::Message { role, content, .. } => {
                        let text = extract_text_from_content(content);
                        messages.push(role_to_chat_message(role.as_str(), text));
                    }
                    ResponseInputOutputItem::FunctionToolCall {
                        id,
                        name,
                        arguments,
                        output,
                        ..
                    } => {
                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: Some(vec![ToolCall {
                                id: id.clone(),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name: name.clone(),
                                    arguments: Some(arguments.clone()),
                                },
                            }]),
                            reasoning_content: None,
                        });

                        if let Some(output_text) = output {
                            messages.push(ChatMessage::Tool {
                                content: MessageContent::Text(output_text.clone()),
                                tool_call_id: id.clone(),
                            });
                        }
                    }
                    ResponseInputOutputItem::Reasoning { content, .. } => {
                        let reasoning_text = content
                            .iter()
                            .map(|c| match c {
                                ReasoningText { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("\n");

                        messages.push(ChatMessage::Assistant {
                            content: None,
                            name: None,
                            tool_calls: None,
                            reasoning_content: Some(reasoning_text),
                        });
                    }
                    ResponseInputOutputItem::FunctionCallOutput {
                        call_id, output, ..
                    } => {
                        messages.push(ChatMessage::Tool {
                            content: MessageContent::Text(output.clone()),
                            tool_call_id: call_id.clone(),
                        });
                    }
                }
            }
        }
    }

    if messages.is_empty() {
        return Err("Request must contain at least one message".to_string());
    }

    let function_tools = extract_tools_from_response_tools(req.tools.as_deref());
    let tools = if function_tools.is_empty() {
        None
    } else {
        Some(function_tools)
    };

    let is_streaming = req.stream.unwrap_or(false);

    Ok(ChatCompletionRequest {
        messages,
        model: if req.model.is_empty() {
            UNKNOWN_MODEL_ID.to_string()
        } else {
            req.model.clone()
        },
        temperature: req.temperature,
        max_completion_tokens: req.max_output_tokens,
        stream: is_streaming,
        stream_options: if is_streaming {
            Some(StreamOptions {
                include_usage: Some(true),
            })
        } else {
            None
        },
        parallel_tool_calls: req.parallel_tool_calls,
        top_logprobs: req.top_logprobs,
        top_p: req.top_p,
        skip_special_tokens: true,
        tools,
        tool_choice: req.tool_choice.clone(),
        response_format: map_text_to_response_format(&req.text),
        ..Default::default()
    })
}

fn extract_text_from_content(content: &[ResponseContentPart]) -> String {
    content
        .iter()
        .filter_map(|part| match part {
            ResponseContentPart::InputText { text } => Some(text.as_str()),
            ResponseContentPart::OutputText { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn role_to_chat_message(role: &str, text: String) -> ChatMessage {
    match role {
        "user" => ChatMessage::User {
            content: MessageContent::Text(text),
            name: None,
        },
        "assistant" => ChatMessage::Assistant {
            content: Some(MessageContent::Text(text)),
            name: None,
            tool_calls: None,
            reasoning_content: None,
        },
        "system" => ChatMessage::System {
            content: MessageContent::Text(text),
            name: None,
        },
        // Unknown roles fall through to user.
        _ => ChatMessage::User {
            content: MessageContent::Text(text),
            name: None,
        },
    }
}

fn map_text_to_response_format(text: &Option<TextConfig>) -> Option<ResponseFormat> {
    let text_config = text.as_ref()?;
    let format = text_config.format.as_ref()?;

    match format {
        TextFormat::Text => Some(ResponseFormat::Text),
        TextFormat::JsonObject => Some(ResponseFormat::JsonObject),
        TextFormat::JsonSchema {
            name,
            schema,
            description: _,
            strict,
        } => Some(ResponseFormat::JsonSchema {
            json_schema: JsonSchemaFormat {
                name: name.clone(),
                schema: schema.clone(),
                strict: *strict,
            },
        }),
    }
}

pub(crate) fn chat_to_responses(
    chat_resp: &ChatCompletionResponse,
    original_req: &ResponsesRequest,
    response_id_override: Option<String>,
) -> Result<ResponsesResponse, String> {
    // Responses API does not support n>1; only the first choice contributes.
    let choice = chat_resp
        .choices
        .first()
        .ok_or_else(|| "Chat response contains no choices".to_string())?;

    let mut output: Vec<ResponseOutputItem> = Vec::new();

    if let Some(content) = &choice.message.content {
        if !content.is_empty() {
            output.push(ResponseOutputItem::Message {
                id: format!("msg_{}", chat_resp.id),
                role: "assistant".to_string(),
                content: vec![ResponseContentPart::OutputText {
                    text: content.clone(),
                    annotations: vec![],
                    logprobs: choice.logprobs.clone(),
                }],
                status: "completed".to_string(),
            });
        }
    }

    if let Some(reasoning) = &choice.message.reasoning_content {
        if !reasoning.is_empty() {
            output.push(ResponseOutputItem::Reasoning {
                id: format!("reasoning_{}", chat_resp.id),
                summary: vec![],
                content: vec![ReasoningText {
                    text: reasoning.clone(),
                }],
                status: Some("completed".to_string()),
            });
        }
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            output.push(ResponseOutputItem::FunctionToolCall {
                id: tool_call.id.clone(),
                call_id: tool_call.id.clone(),
                name: tool_call.function.name.clone(),
                arguments: tool_call.function.arguments.clone().unwrap_or_default(),
                output: None,
                status: "in_progress".to_string(),
            });
        }
    }

    let status = match choice.finish_reason.as_deref() {
        Some("stop") | Some("length") => ResponseStatus::Completed,
        // Model finished cleanly; running the tool is the caller's responsibility.
        Some("tool_calls") => ResponseStatus::Completed,
        Some("failed") | Some("error") => ResponseStatus::Failed,
        _ => ResponseStatus::Completed,
    };

    let usage = chat_resp.usage.as_ref().map(|u| {
        let usage_info = UsageInfo {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            reasoning_tokens: u
                .completion_tokens_details
                .as_ref()
                .and_then(|d| d.reasoning_tokens),
            // Chat response does not surface prompt token details.
            prompt_tokens_details: None,
        };
        ResponsesUsage::Classic(usage_info)
    });

    let response_id = response_id_override.unwrap_or_else(|| chat_resp.id.clone());
    Ok(ResponsesResponse::builder(&response_id, &chat_resp.model)
        .copy_from_request(original_req)
        .created_at(chat_resp.created as i64)
        .status(status)
        .output(output)
        .maybe_text(original_req.text.clone())
        .maybe_usage(usage)
        .build())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("Hello, world!".to_string()),
            instructions: Some("You are a helpful assistant.".to_string()),
            model: "gpt-4".to_string(),
            temperature: Some(0.7),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // system + user
        assert_eq!(chat_req.model, "gpt-4");
        assert_eq!(chat_req.temperature, Some(0.7));
    }

    #[test]
    fn test_items_input_conversion() {
        let req = ResponsesRequest {
            input: ResponseInput::Items(vec![
                ResponseInputOutputItem::Message {
                    id: "msg_1".to_string(),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText {
                        text: "Hello!".to_string(),
                    }],
                    status: None,
                },
                ResponseInputOutputItem::Message {
                    id: "msg_2".to_string(),
                    role: "assistant".to_string(),
                    content: vec![ResponseContentPart::OutputText {
                        text: "Hi there!".to_string(),
                        annotations: vec![],
                        logprobs: None,
                    }],
                    status: None,
                },
            ]),
            ..Default::default()
        };

        let chat_req = responses_to_chat(&req).unwrap();
        assert_eq!(chat_req.messages.len(), 2); // user + assistant
    }

    #[test]
    fn test_empty_input_error() {
        let req = ResponsesRequest {
            input: ResponseInput::Text("".to_string()),
            ..Default::default()
        };

        // Empty text should still create a user message, so this should succeed
        let result = responses_to_chat(&req);
        assert!(result.is_ok());
    }
}
