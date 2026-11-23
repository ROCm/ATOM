// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! The tools a chat's messages declare themselves (Kimi-K3's dynamic tools), which ATOM's
//! HTTP chat accepts on any role while the typed request keeps them on a developer message only.

use std::{collections::HashSet, sync::Arc};

use serde::Deserialize;
use serde_json::Value;

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::Tool,
};

/// Each message's own tools as sent, in message order; `None` where a message declares none.
#[derive(Debug, Clone, Default)]
pub struct MessageTools(pub Arc<Vec<Option<Vec<Value>>>>);

#[derive(Deserialize)]
struct RawChat {
    messages: Vec<RawMessage>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    tools: Option<Vec<Value>>,
}

impl MessageTools {
    /// The tools each message of a chat's raw `body` declares, on whatever role.
    pub fn read(body: &[u8]) -> Result<Self, String> {
        let chat: RawChat =
            serde_json::from_slice(body).map_err(|e| format!("Invalid message tools: {e}"))?;
        let declared = Self(Arc::new(
            chat.messages.into_iter().map(|m| m.tools).collect(),
        ));
        for tool in declared.0.iter().flatten().flatten() {
            qualified(tool).map_err(|e| format!("Invalid message tools: {e}"))?;
        }
        Ok(declared)
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
        Self(Arc::new(tools.collect()))
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }

    /// The top-level tools, then every message's, as the reply's tool calls are read
    /// against: one per name, its last declaration, as ATOM's HTTP chat reads them.
    /// A message's namespaced tool is named `namespace::name`.
    pub fn with_top_level(&self, top_level: Option<&Vec<Tool>>) -> Vec<Tool> {
        let declared = self
            .0
            .iter()
            .flatten()
            .flatten()
            .filter_map(|t| qualified(t).ok());
        let all: Vec<Tool> = top_level
            .into_iter()
            .flatten()
            .cloned()
            .chain(declared)
            .collect();
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
    let function = tool.get("function");
    let beside = tool
        .get("namespace")
        .filter(|namespace| !namespace.is_null());
    let namespace = match beside.or_else(|| function?.get("namespace")) {
        None | Some(Value::Null) => return Ok(parsed),
        Some(Value::Object(namespace)) => namespace.get("name"),
        Some(namespace) => Some(namespace),
    };
    let namespace = match namespace.and_then(Value::as_str) {
        Some(namespace) if identifier(namespace) => namespace,
        _ => return Err("tool namespace must be a nonempty identifier without '::'".into()),
    };
    let name = &mut parsed.function.name;
    match name.split_once("::") {
        Some((prefix, _)) if prefix != namespace => Err("conflicting tool namespaces".into()),
        Some(_) => Ok(parsed),
        None => {
            *name = format!("{namespace}::{name}");
            Ok(parsed)
        }
    }
}

/// A word character, then word characters, dots or dashes.
fn identifier(name: &str) -> bool {
    let word = |c: char| c.is_alphanumeric() || c == '_';
    name.starts_with(word) && name.chars().all(|c| word(c) || c == '.' || c == '-')
}
