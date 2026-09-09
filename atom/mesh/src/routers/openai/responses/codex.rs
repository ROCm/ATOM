//! Compatibility shims for Codex's newer Responses API tool shapes.
//!
//! ATOM currently pins `openai-protocol` 1.0, whose typed request model only
//! understands function tools. Codex also sends free-form custom tools,
//! namespace groups, and client-side tool search items. Normalize those shapes
//! to function calls before deserialization, then restore their wire shapes in
//! the response.

use std::collections::HashMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Map, Value};
use validator::{Validate, ValidationErrors};

use crate::protocols::{responses::ResponsesRequest, validated::Normalizable};

const CONTEXT_METADATA_KEY: &str = "__atom_codex_tool_context";

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub(crate) struct CodexToolContext {
    kinds: HashMap<String, String>,
    namespaces: HashMap<String, String>,
    original_tools: Option<Value>,
}

impl CodexToolContext {
    pub(crate) fn is_empty(&self) -> bool {
        self.kinds.is_empty() && self.original_tools.is_none()
    }

    pub(crate) fn kind(&self, name: &str) -> &str {
        self.kinds
            .get(name)
            .map(String::as_str)
            .unwrap_or("function")
    }

    pub(crate) fn namespace(&self, name: &str) -> Option<&str> {
        self.namespaces.get(name).map(String::as_str)
    }

    pub(crate) fn rewrite_response(&self, response: &mut Value) {
        if let Some(original_tools) = &self.original_tools {
            response["tools"] = original_tools.clone();
        }
        if let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) {
            for item in output {
                self.rewrite_output_item(item);
            }
        }
    }

    pub(crate) fn rewrite_output_item(&self, item: &mut Value) {
        if item.get("type").and_then(Value::as_str) != Some("function_call") {
            return;
        }
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        match self.kind(&name) {
            "custom" => {
                let input = custom_tool_input(
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                item["type"] = json!("custom_tool_call");
                item.as_object_mut().unwrap().remove("arguments");
                item["input"] = json!(input);
                if let Some(namespace) = self.namespace(&name) {
                    item["namespace"] = json!(namespace);
                }
            }
            "tool_search" => {
                let arguments = parse_tool_search_arguments(
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                item["type"] = json!("tool_search_call");
                item.as_object_mut().unwrap().remove("name");
                item.as_object_mut().unwrap().remove("arguments");
                item["execution"] = json!("client");
                item["arguments"] = arguments;
            }
            _ => {
                if let Some(namespace) = self.namespace(&name) {
                    item["namespace"] = json!(namespace);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CodexResponsesRequest(pub ResponsesRequest);

impl<'de> Deserialize<'de> for CodexResponsesRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;
        normalize_request_value(&mut value).map_err(serde::de::Error::custom)?;
        serde_json::from_value(value)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

impl Validate for CodexResponsesRequest {
    fn validate(&self) -> Result<(), ValidationErrors> {
        self.0.validate()
    }
}

impl Normalizable for CodexResponsesRequest {
    fn normalize(&mut self) {
        self.0.normalize();
    }
}

pub(crate) fn take_context(request: &mut ResponsesRequest) -> CodexToolContext {
    request
        .metadata
        .as_mut()
        .and_then(|metadata| metadata.remove(CONTEXT_METADATA_KEY))
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default()
}

fn normalize_request_value(body: &mut Value) -> Result<(), String> {
    let Some(object) = body.as_object_mut() else {
        return Err("Responses request must be a JSON object".to_string());
    };

    let original_tools = object.get("tools").cloned();
    let mut context = CodexToolContext {
        original_tools,
        ..Default::default()
    };

    let mut all_tools = object
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if let Some(items) = object.get_mut("input").and_then(Value::as_array_mut) {
        let mut normalized_items = Vec::with_capacity(items.len());
        for item in std::mem::take(items) {
            let item_type = item.get("type").and_then(Value::as_str).unwrap_or_default();
            match item_type {
                "custom_tool_call" => {
                    let call_id = string_field(&item, "call_id");
                    let id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or(&call_id)
                        .to_string();
                    let name = string_field(&item, "name");
                    let input = item
                        .get("input")
                        .cloned()
                        .unwrap_or(Value::String(String::new()));
                    normalized_items.push(json!({
                        "type": "function_call",
                        "id": id,
                        "call_id": call_id,
                        "name": name,
                        "arguments": serde_json::to_string(&json!({"input": input.as_str().unwrap_or_default()})).unwrap(),
                    }));
                }
                "custom_tool_call_output" => {
                    normalized_items.push(json!({
                        "type": "function_call_output",
                        "id": item.get("id").cloned().unwrap_or(Value::Null),
                        "call_id": string_field(&item, "call_id"),
                        "output": value_as_text(item.get("output")),
                    }));
                }
                "tool_search_call" => {
                    let call_id = string_field(&item, "call_id");
                    let id = item
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or(&call_id)
                        .to_string();
                    normalized_items.push(json!({
                        "type": "function_call",
                        "id": id,
                        "call_id": call_id,
                        "name": "tool_search",
                        "arguments": value_as_text(item.get("arguments")),
                    }));
                }
                "tool_search_output" => {
                    if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                        all_tools.extend(tools.iter().cloned());
                    }
                    normalized_items.push(json!({
                        "type": "function_call_output",
                        "id": item.get("id").cloned().unwrap_or(Value::Null),
                        "call_id": string_field(&item, "call_id"),
                        "output": serde_json::to_string(&json!({"tools": item.get("tools").cloned().unwrap_or_else(|| json!([]))})).unwrap(),
                    }));
                }
                "additional_tools" => {
                    if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                        all_tools.extend(tools.iter().cloned());
                    }
                }
                _ => normalized_items.push(item),
            }
        }
        *items = normalized_items;
    }

    let normalized_tools = normalize_tools(&all_tools, &mut context);
    if !normalized_tools.is_empty() {
        object.insert("tools".to_string(), Value::Array(normalized_tools));
    }

    if !context.is_empty() {
        let metadata = object
            .entry("metadata")
            .or_insert_with(|| Value::Object(Map::new()));
        let metadata = metadata
            .as_object_mut()
            .ok_or_else(|| "metadata must be a JSON object".to_string())?;
        metadata.insert(
            CONTEXT_METADATA_KEY.to_string(),
            serde_json::to_value(context).map_err(|e| e.to_string())?,
        );
    }

    Ok(())
}

fn normalize_tools(tools: &[Value], context: &mut CodexToolContext) -> Vec<Value> {
    let mut normalized = Vec::new();
    for tool in tools {
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => normalized.push(tool.clone()),
            Some("custom") => {
                if let Some(name) = tool.get("name").and_then(Value::as_str) {
                    context.kinds.insert(name.to_string(), "custom".to_string());
                    normalized.push(custom_as_function(tool, None));
                }
            }
            Some("namespace") => {
                let namespace = tool
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("namespace");
                for child in tool
                    .get("tools")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let Some(name) = child.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    context
                        .namespaces
                        .insert(name.to_string(), namespace.to_string());
                    match child.get("type").and_then(Value::as_str) {
                        Some("function") => {
                            let mut child = child.clone();
                            let description = child
                                .get("description")
                                .and_then(Value::as_str)
                                .unwrap_or_default();
                            child["description"] =
                                json!(format!("[{namespace}] {description}").trim_end());
                            normalized.push(child);
                        }
                        Some("custom") => {
                            context.kinds.insert(name.to_string(), "custom".to_string());
                            normalized.push(custom_as_function(child, Some(namespace)));
                        }
                        _ => {}
                    }
                }
            }
            Some("tool_search")
                if tool.get("execution").and_then(Value::as_str) == Some("client") =>
            {
                context
                    .kinds
                    .insert("tool_search".to_string(), "tool_search".to_string());
                normalized.push(json!({
                    "type": "function",
                    "name": "tool_search",
                    "description": tool.get("description").cloned().unwrap_or_else(|| json!("Search for and load deferred client tools.")),
                    "parameters": tool.get("parameters").cloned().unwrap_or_else(|| json!({"type": "object"})),
                }));
            }
            // Preserve tool kinds understood by openai-protocol 1.0 and drop
            // newer hosted tools that ATOM cannot execute itself.
            Some("web_search_preview" | "code_interpreter" | "mcp") => {
                normalized.push(tool.clone())
            }
            _ => {}
        }
    }
    normalized
}

fn custom_as_function(tool: &Value, namespace: Option<&str>) -> Value {
    let mut description = tool
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if let Some(namespace) = namespace {
        description = format!("[{namespace}] {description}")
            .trim_end()
            .to_string();
    }
    if let Some(format) = tool.get("format").and_then(Value::as_object) {
        if format.get("type").and_then(Value::as_str) == Some("grammar") {
            let syntax = format
                .get("syntax")
                .and_then(Value::as_str)
                .unwrap_or("grammar");
            let definition = format
                .get("definition")
                .and_then(Value::as_str)
                .unwrap_or_default();
            description.push_str(&format!(
                "\nReturn raw input matching this {syntax} grammar:\n{definition}"
            ));
        }
    }
    json!({
        "type": "function",
        "name": tool.get("name").cloned().unwrap_or(Value::String(String::new())),
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"input": {"type": "string", "description": "Raw freeform input for this tool."}},
            "required": ["input"],
            "additionalProperties": false,
        },
    })
}

pub(crate) fn custom_tool_input(arguments: &str) -> String {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(object)) => object
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or(arguments)
            .to_string(),
        Ok(Value::String(value)) => value,
        _ => arguments.to_string(),
    }
}

pub(crate) fn parse_tool_search_arguments(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({"query": arguments}))
}

fn string_field(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn value_as_text(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_default(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_codex_custom_namespace_and_history() {
        let raw = json!({
            "model": "glm",
            "input": [
                {"role": "user", "content": "edit it"},
                {"type": "custom_tool_call", "call_id": "call_1", "name": "apply_patch", "input": "*** Begin Patch"},
                {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Done"}
            ],
            "tools": [{
                "type": "namespace", "name": "editing", "description": "tools", "tools": [{
                    "type": "custom", "name": "apply_patch", "description": "patch", "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}
                }]
            }]
        });

        let mut request: CodexResponsesRequest = serde_json::from_value(raw).unwrap();
        let context = take_context(&mut request.0);
        assert_eq!(context.kind("apply_patch"), "custom");
        assert_eq!(context.namespace("apply_patch"), Some("editing"));
        assert_eq!(request.0.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn restores_custom_tool_response_shape() {
        let context = CodexToolContext {
            kinds: HashMap::from([("apply_patch".to_string(), "custom".to_string())]),
            namespaces: HashMap::from([("apply_patch".to_string(), "editing".to_string())]),
            original_tools: None,
        };
        let mut item = json!({
            "id": "fc_1", "type": "function_call", "call_id": "call_1",
            "name": "apply_patch", "arguments": "{\"input\":\"patch\"}", "status": "completed"
        });
        context.rewrite_output_item(&mut item);
        assert_eq!(item["type"], "custom_tool_call");
        assert_eq!(item["input"], "patch");
        assert_eq!(item["namespace"], "editing");
        assert!(item.get("arguments").is_none());
    }

    #[test]
    fn loads_deferred_tools_from_tool_search_output() {
        let raw = json!({
            "model": "glm",
            "input": [
                {"role": "user", "content": "delegate"},
                {"type": "tool_search_call", "call_id": "search_1", "execution": "client", "arguments": {"query": "agent"}},
                {"type": "tool_search_output", "call_id": "search_1", "tools": [{
                    "type": "function", "name": "spawn_agent", "description": "spawn", "parameters": {"type": "object"}
                }]}
            ],
            "tools": [{
                "type": "tool_search", "execution": "client", "parameters": {"type": "object"}
            }]
        });

        let mut request: CodexResponsesRequest = serde_json::from_value(raw).unwrap();
        let context = take_context(&mut request.0);
        let tools = request.0.tools.as_ref().unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(context.kind("tool_search"), "tool_search");
    }
}
