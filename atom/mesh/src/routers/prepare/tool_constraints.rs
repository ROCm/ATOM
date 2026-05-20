//! Tool-call constraint generation: turn `tool_choice` + tool list into the
//! JSON-schema / EBNF constraint that the worker enforces during sampling.

use serde_json::{json, Map, Value};
use tracing::error;
use uuid::Uuid;

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::{FunctionCallResponse, Tool, ToolCall, ToolChoice, ToolChoiceValue},
};

/// Generate tool constraints for structured generation
/// Note: tools should already be filtered if needed (by allowed_tools or specific function)
pub(crate) fn generate_tool_constraints(
    tools: &[Tool],
    tool_choice: &Option<ToolChoice>,
    _model: &str,
) -> Result<Option<(String, String)>, String> {
    let Some(choice) = tool_choice.as_ref() else {
        return Ok(None);
    };

    match choice {
        // Specific function: Return parameters schema directly
        // tools should already be filtered to contain only the specific function
        ToolChoice::Function { .. } => {
            if tools.is_empty() {
                return Ok(None);
            }
            let tool = &tools[0];

            // Return the tool's parameters schema directly (not wrapped in array)
            let params_schema = serde_json::to_string(&tool.function.parameters)
                .map_err(|e| format!("Failed to serialize tool parameters: {}", e))?;
            Ok(Some((String::from("json_schema"), params_schema)))
        }

        // Required: Array of tool calls with minItems: 1
        ToolChoice::Value(ToolChoiceValue::Required) => {
            let schema = build_required_array_schema(tools)?;
            Ok(Some(("json_schema".to_string(), schema)))
        }

        // AllowedTools with required mode: tools are already filtered
        ToolChoice::AllowedTools { mode, .. } => {
            if mode == "required" {
                if tools.is_empty() {
                    return Ok(None);
                }
                let schema = build_required_array_schema(tools)?;
                Ok(Some(("json_schema".to_string(), schema)))
            } else {
                // "auto" mode - no constraint needed
                Ok(None)
            }
        }

        // "auto" or "none" - no constraint
        _ => Ok(None),
    }
}

/// Build JSON schema for required tool calls (array with minItems: 1)
/// Includes $defs consolidation from all tools (matching Python's behavior)
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

    // Consolidate $defs from all tools (matching Python's _get_tool_schema_defs)
    let mut all_defs: Map<String, Value> = Map::new();
    for tool in tools {
        if let Value::Object(params) = &tool.function.parameters {
            if let Some(Value::Object(defs)) = params.get("$defs") {
                for (def_name, def_schema) in defs {
                    if let Some(existing) = all_defs.get(def_name) {
                        // Check for conflicts
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

    // Build the full array schema
    let mut array_schema = json!({
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "anyOf": any_of_schemas
        }
    });

    // Add $defs if any were found (matching Python's behavior)
    if !all_defs.is_empty() {
        if let Value::Object(ref mut schema_obj) = array_schema {
            schema_obj.insert("$defs".to_string(), Value::Object(all_defs));
        }
    }

    serde_json::to_string(&array_schema)
        .map_err(|e| format!("Failed to serialize tool schema: {}", e))
}

/// Filter tools based on tool_choice (generic helper)
///
/// Returns filtered tools if filtering is needed, otherwise returns None.
/// Used by both Chat API and Responses API for constraint generation.
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

/// Filter ChatCompletionRequest by tool_choice
///
/// Returns a reference to the original request if no filtering needed,
/// otherwise returns a cloned request with filtered tools.
///
/// Note: Tool existence is validated earlier in ChatCompletionRequest::validate(),
/// so this function assumes tool_choice references valid tools.
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

/// Parse tool calls from JSON schema constrained response
pub(crate) fn parse_json_schema_response(
    processed_text: &str,
    tool_choice: &Option<ToolChoice>,
    model: &str,
    history_tool_calls_count: usize,
) -> (Option<Vec<ToolCall>>, String) {
    match tool_choice {
        Some(ToolChoice::Function { function, .. }) => {
            // Specific function: Parse parameters directly
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
            // Required mode: Parse array of tool calls
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

/// Count the number of tool calls in the request message history
/// This is used for KimiK2 format which needs globally unique indices
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

/// Generate a tool call ID based on model format
///
/// # Arguments
/// * `model` - Model name to determine ID format
/// * `tool_name` - Name of the tool being called
/// * `tool_index` - Index of this tool call within the current message
/// * `history_count` - Number of tool calls in previous messages
///
/// # Returns
/// A unique ID string. KimiK2 uses `functions.{name}:{global_index}`, others use `call_{uuid}`
pub(crate) fn generate_tool_call_id(
    model: &str,
    tool_name: &str,
    tool_index: usize,
    history_count: usize,
) -> String {
    // Case-insensitive check without allocation (search for "kimi" substring)
    let is_kimi = model
        .as_bytes()
        .windows(4) // "kimi".len()
        .any(|window| window.eq_ignore_ascii_case(b"kimi"));

    if is_kimi {
        // KimiK2 format: functions.{name}:{global_index}
        format!("functions.{}:{}", tool_name, history_count + tool_index)
    } else {
        // Standard OpenAI format: call_{24-char-uuid}
        format!("call_{}", &Uuid::new_v4().simple().to_string()[..24])
    }
}
