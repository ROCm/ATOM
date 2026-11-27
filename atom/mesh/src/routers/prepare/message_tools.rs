// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! The tools a chat's messages declare themselves (Kimi-K3's dynamic tools), which ATOM's
//! HTTP chat accepts on any role while the typed request keeps them on a developer message only.

use std::{collections::HashSet, sync::Arc};

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::Tool,
};

/// Each message's own tools as sent, in message order; `None` where a message declares none.
/// Beside them, the top-level tools as sent when one carries a namespace, and a named
/// `tool_choice` as sent when it carries one, which the typed request drops.
#[derive(Debug, Clone, Default)]
pub struct MessageTools(
    pub Arc<Vec<Option<Vec<Value>>>>,
    pub Option<Arc<Vec<Value>>>,
    pub Option<Arc<Value>>,
);

impl MessageTools {
    /// The tools each message of a chat's body as `sent` declares, on whatever role.
    pub fn read(sent: &Map<String, Value>) -> Result<Self, String> {
        let tools = |value: Option<&Value>| match value {
            Some(value) => Option::<Vec<Value>>::deserialize(value)
                .map_err(|e| format!("Invalid message tools: {e}")),
            None => Ok(None),
        };
        let Some(Value::Array(messages)) = sent.get("messages") else {
            return Err("Invalid message tools: messages is not an array".to_string());
        };
        let namespaced = |tools: &Vec<Value>| tools.iter().any(|t| namespace(t).is_some());
        let top_level = tools(sent.get("tools"))?.filter(namespaced).map(Arc::new);
        let choice = (sent.get("tool_choice"))
            .filter(|c| c.is_object() && namespace(c).is_some())
            .cloned();
        let declared = Self(
            Arc::new(
                messages
                    .iter()
                    .map(|m| tools(m.get("tools")))
                    .collect::<Result<_, _>>()?,
            ),
            top_level,
            choice.map(Arc::new),
        );
        for tool in declared.0.iter().flatten().flatten() {
            qualified(tool).map_err(|e| format!("Invalid message tools: {e}"))?;
        }
        for tool in declared.1.iter().flat_map(|tools| tools.iter()) {
            qualified(tool).map_err(|e| format!("Invalid tools: {e}"))?;
        }
        declared
            .qualified_choice()
            .transpose()
            .map_err(|e| format!("Invalid tool_choice: {e}"))?;
        Ok(declared)
    }

    /// The named `tool_choice`'s `namespace::name`, when it carries a namespace.
    pub fn qualified_choice(&self) -> Option<Result<String, String>> {
        let choice = self.2.as_deref()?;
        let name = choice.get("function")?.get("name")?.as_str()?;
        Some(qualify(name, choice))
    }

    /// The tools the typed `req` keeps: a developer message's.
    pub fn of(req: &ChatCompletionRequest) -> Self {
        let tools = req.messages.iter().map(|message| match message {
            ChatMessage::Developer { tools, .. } => tools.as_ref().map(|tools| {
                let tools = tools.iter().map(serde_json::to_value);
                tools.collect::<Result<_, _>>().unwrap_or_default()
            }),
            _ => None,
        });
        Self(Arc::new(tools.collect()), None, None)
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }

    /// The top-level tools, then every message's, as the reply's tool calls are read
    /// against: one per name, its last declaration, as ATOM's HTTP chat reads them.
    /// A namespaced tool is named `namespace::name`.
    pub fn with_top_level(&self, top_level: Option<&Vec<Tool>>) -> Vec<Tool> {
        let declared = self
            .0
            .iter()
            .flatten()
            .flatten()
            .filter_map(|t| qualified(t).ok());
        let top_level: Vec<Tool> = match &self.1 {
            Some(sent) => sent.iter().filter_map(|t| qualified(t).ok()).collect(),
            None => top_level.into_iter().flatten().cloned().collect(),
        };
        let all: Vec<Tool> = top_level.into_iter().chain(declared).collect();
        let mut named = HashSet::new();
        let mut last: Vec<Tool> = (all.into_iter().rev())
            .filter(|tool| named.insert(tool.function.name.clone()))
            .collect();
        last.reverse();
        last
    }
}

/// `tool` named as ATOM's `qualified_tool_name` names it: `namespace::name` when it
/// has a namespace, beside or inside `function`, as a string or `{"name": ...}`.
fn qualified(tool: &Value) -> Result<Tool, String> {
    let mut parsed: Tool = serde_json::from_value(tool.clone()).map_err(|e| e.to_string())?;
    parsed.function.name = qualify(&parsed.function.name, tool)?;
    Ok(parsed)
}

/// A tool as sent, named as `qualified` names it; none if it cannot be.
pub(crate) fn qualified_name(tool: &Value) -> Option<String> {
    let name = tool.get("function")?.get("name")?.as_str()?;
    qualify(name, tool).ok()
}

/// `name` qualified by the namespace `carrier` (a tool or a named choice) has.
fn qualify(name: &str, carrier: &Value) -> Result<String, String> {
    let namespace = match namespace(carrier) {
        None => return Ok(name.to_string()),
        Some(Value::Object(namespace)) => namespace.get("name"),
        Some(namespace) => Some(namespace),
    };
    let namespace = match namespace.and_then(Value::as_str) {
        Some(namespace) if identifier(namespace) => namespace,
        _ => return Err("tool namespace must be a nonempty identifier without '::'".into()),
    };
    match name.split_once("::") {
        Some((prefix, _)) if prefix != namespace => Err("conflicting tool namespaces".into()),
        Some(_) => Ok(name.to_string()),
        None => Ok(format!("{namespace}::{name}")),
    }
}

/// `tool`'s namespace as sent, beside or inside `function`; none when null or absent.
fn namespace(tool: &Value) -> Option<&Value> {
    let beside = tool.get("namespace").filter(|n| !n.is_null());
    beside
        .or_else(|| tool.get("function")?.get("namespace"))
        .filter(|n| !n.is_null())
}

/// A word character, then word characters, dots or dashes.
fn identifier(name: &str) -> bool {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    name.starts_with(word) && name.chars().all(|c| word(c) || c == '.' || c == '-')
}
