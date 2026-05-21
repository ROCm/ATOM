//! Tool-call constraint generation: turn `tool_choice` + tool list into the
//! JSON-schema / EBNF constraint that the worker enforces during sampling.

use serde_json::{json, Map, Value};
use tracing::error;
use uuid::Uuid;

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::{FunctionCallResponse, Tool, ToolCall, ToolChoice, ToolChoiceValue},
};

/// Caller must have already filtered `tools` (by allowed_tools or specific function).
pub(crate) fn generate_tool_constraints(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
    _model: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(choice) = tool_choice.as_ref() else {
        return Ok(None);
    };

    match choice {
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let tool = &tools[0];

            let params_schema = serde_json::to_string(&tool.function.parameters)
                .map_err(|e| format!("Failed to serialize tool parameters: {}", e))?;
            Ok(Some((String::from("json_schema"), params_schema)))
        }

        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_array_schema(tools)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return Ok(None);
                }
                let schema = build_required_array_schema(tools)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                Ok(None)
            }
        }

        _ => Ok(None),
    }
}

/// `$defs` from every tool are consolidated under one top-level `$defs`; an
/// error is returned if two tools disagree on the same definition.
fn build_required_array_schema(tools: &[Tool]) -> Result<String, String> {
    let mut any_of_schemas = Vec::with_capacity(tools.len());
    for tool in tools {
        let tool_schema = json!({
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.function.name]
                },
                "parameters": tool.function.parameters
            },
            "required": ["name", "parameters"]
        });
        any_of_schemas.push(tool_schema);
    }

    let mut all_defs: Map<String, Value> = Map::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        if existing != def_schema {
                            let error_msg = format!(
                                "Tool definition '{}' has multiple conflicting schemas, which is not supported",
                                def_name
                            );
                            error!("{}", error_msg);
                            return Err(error_msg);
                        }
                    } else {
                        all_defs.insert(def_name.clone(), def_schema.clone());
                    }
                }
            }
        }
    }

    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            schema_obj.insert("$defs".to_string(), Value::Object(all_defs));
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {}", e))
}

/// Returns filtered tools when `tool_choice` narrows the set, else `None`.
pub(crate) fn filter_tools_by_tool_choice(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
) -> Option<Vec<Tool>> {
    match tool_choice {
        Some(ToolChoice::AllowedTools { tools: allowed, .. }) => {
            let allowed_names: std::collections::HashSet<&str> =
                allowed.iter().filter_map(|t| t.function_name()).collect();
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| allowed_names.contains(t.function.name.as_str()))
                .cloned()
                .collect();
            Some(filtered)
        }
        Some(ToolChoice::Function { function, .. }) => {
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| t.function.name == function.name)
                .cloned()
                .collect();
            Some(filtered)
        }
        _ => None,
    }
}

/// Assumes `tool_choice` references valid tools (verified earlier by
/// `ChatCompletionRequest::validate`).
pub(crate) fn filter_chat_request_by_tool_choice(
    body: &ChatCompletionRequest,
) -> std::borrow::Cow<'_, ChatCompletionRequest> {
    if let Some(tools) = &body.tools {
        if let Some(filtered_tools) = filter_tools_by_tool_choice(tools, &body.tool_choice) {
            let mut filtered_body = body.clone();
            filtered_body.tools = Some(filtered_tools);
            return std::borrow::Cow::Owned(filtered_body);
        }
    }

    std::borrow::Cow::Borrowed(body)
}

pub(crate) fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: &Option<ToolChoice>,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            match serde_json::from_str::<Value>(processed_text) {
                Ok(params) => {
                    let tool_call = ToolCall {
                        id: generate_tool_call_id(
                            model,
                            &function.name,
                            0,
                            history_tool_calls_count,
                        ),
                        tool_type: "function".to_string(),
                        function: FunctionCallResponse {
                            name: function.name.clone(),
                            arguments: Some(
                                serde_json::to_string(&params).unwrap_or_else(|_| "{}".to_string()),
                            ),
                        },
                    };
                    (Some(vec![tool_call]), String::new())
                }
                Err(e) => {
                    error!("Failed to parse specific function parameters: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        Some(ToolChoice::Value(ToolChoiceValue::Required))
        | Some(ToolChoice::AllowedTools { .. }) => {
            match serde_json::from_str::<Vec<Value>>(processed_text) {
                Ok(parsed_array) => {
                    let spec_tool_calls: Vec<ToolCall> = parsed_array
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, item)| {
                            let obj = item.as_object()?;
                            let name = obj.get("name")?.as_str()?.to_string();
                            let parameters = obj.get("parameters")?;

                            Some(ToolCall {
                                id: generate_tool_call_id(
                                    model,
                                    &name,
                                    i,
                                    history_tool_calls_count,
                                ),
                                tool_type: "function".to_string(),
                                function: FunctionCallResponse {
                                    name,
                                    arguments: Some(
                                        serde_json::to_string(parameters)
                                            .unwrap_or_else(|_| "{}".to_string()),
                                    ),
                                },
                            })
                        })
                        .collect();
                    (Some(spec_tool_calls), String::new())
                }
                Err(e) => {
                    error!("Failed to parse required tool call array: {}", e);
                    (None, processed_text.to_string())
                }
            }
        }
        _ => (None, processed_text.to_string()),
    }
}

/// Used by the KimiK2 ID format, which requires globally unique indices across
/// the conversation history.
pub(crate) fn get_history_tool_calls_count(request: &ChatCompletionRequest) -> usize {
    request
        .messages
        .iter()
        .filter_map(|msg| {
            if let ChatMessage::Assistant { tool_calls, .. } = msg {
                tool_calls.as_ref().map(|calls| calls.len())
            } else {
                None
            }
        })
        .sum()
}

/// KimiK2 models use `functions.{name}:{global_index}`; everything else uses
/// the OpenAI-style `call_{uuid}`.
pub(crate) fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
    // Case-insensitive substring check without allocation.
    let is_kimi = model
        .as_bytes()
        .windows(4)
        .any(|window| window.eq_ignore_ascii_case(b"kimi"));

    if is_kimi {
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    } else {
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}
