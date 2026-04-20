//! Utility functions for /v1/responses endpoint

use std::sync::Arc;

use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};
use serde_json::to_value;
use tracing::{debug, warn};

use crate::{
    protocols::{
        common::Tool,
        responses::{ResponseTool, ResponseToolType, ResponsesRequest, ResponsesResponse},
    },
    routers::persistence_utils::persist_conversation_items,
};

/// Extract function tools from ResponseTools
pub(crate) fn extract_tools_from_response_tools(
    response_tools: Option<&[ResponseTool]>,
) -> Vec<Tool> {
    let Some(tools) = response_tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter_map(|rt| match rt.r#type {
            ResponseToolType::Function => rt.function.as_ref().map(|f| Tool {
                tool_type: "function".to_string(),
                function: f.clone(),
            }),
            _ => None,
        })
        .collect()
}

/// Persist response to storage if store=true
pub(crate) async fn persist_response_if_needed(
    conversation_storage: Arc<dyn ConversationStorage>,
    conversation_item_storage: Arc<dyn ConversationItemStorage>,
    response_storage: Arc<dyn ResponseStorage>,
    response: &ResponsesResponse,
    original_request: &ResponsesRequest,
) {
    if !original_request.store.unwrap_or(true) {
        return;
    }

    if let Ok(response_json) = to_value(response) {
        if let Err(e) = persist_conversation_items(
            conversation_storage,
            conversation_item_storage,
            response_storage,
            &response_json,
            original_request,
        )
        .await
        {
            warn!("Failed to persist response: {}", e);
        } else {
            debug!("Persisted response: {}", response.id);
        }
    }
}
