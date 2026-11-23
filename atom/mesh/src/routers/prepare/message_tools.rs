// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! The tools a chat's messages declare themselves (Kimi-K3's dynamic tools), which ATOM's
//! HTTP chat accepts on any role while the typed request keeps them on a developer message only.

use std::{collections::HashSet, sync::Arc};

use serde::Deserialize;

use crate::protocols::{
    chat::{ChatCompletionRequest, ChatMessage},
    common::Tool,
};

/// Each message's own tools, in message order; `None` where a message declares none.
#[derive(Debug, Clone, Default)]
pub struct MessageTools(pub Arc<Vec<Option<Vec<Tool>>>>);

#[derive(Deserialize)]
struct RawChat {
    messages: Vec<RawMessage>,
}

#[derive(Deserialize)]
struct RawMessage {
    #[serde(default)]
    tools: Option<Vec<Tool>>,
}

impl MessageTools {
    /// The tools each message of a chat's raw `body` declares, on whatever role.
    pub fn read(body: &[u8]) -> Result<Self, String> {
        let chat: RawChat =
            serde_json::from_slice(body).map_err(|e| format!("Invalid message tools: {e}"))?;
        Ok(Self(Arc::new(
            chat.messages.into_iter().map(|m| m.tools).collect(),
        )))
    }

    /// The tools the typed `req` keeps: a developer message's.
    pub fn of(req: &ChatCompletionRequest) -> Self {
        let tools = req.messages.iter().map(|message| match message {
            ChatMessage::Developer { tools, .. } => tools.clone(),
            _ => None,
        });
        Self(Arc::new(tools.collect()))
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(Option::is_none)
    }

    /// The top-level tools, then every message's, as the reply's tool calls are read
    /// against: one per name, its last declaration, as ATOM's HTTP chat reads them.
    pub fn with_top_level(&self, top_level: Option<&Vec<Tool>>) -> Vec<Tool> {
        let declared = self.0.iter().flatten().flatten();
        let all: Vec<&Tool> = top_level.into_iter().flatten().chain(declared).collect();
        let mut named = HashSet::new();
        let mut last: Vec<Tool> = (all.into_iter().rev())
            .filter(|tool| named.insert(tool.function.name.as_str()))
            .cloned()
            .collect();
        last.reverse();
        last
    }
}
