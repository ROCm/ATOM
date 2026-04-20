//! Shared helpers and state tracking for Regular Responses
//!
//! This module contains common utilities used by both streaming and non-streaming paths:
//! - Helper functions for tool extraction
//! - Conversation history loading

use axum::response::Response;
use data_connector::{self, ConversationId, ResponseId};
use tracing::{debug, warn};

use crate::{
    protocols::responses::{
        self, ResponseContentPart, ResponseInput, ResponseInputOutputItem, ResponsesRequest,
    },
    routers::{error, grpc::common::responses::ResponsesContext},
};

/// Load conversation history and response chains, returning modified request
pub(super) async fn load_conversation_history(
    ctx: &ResponsesContext,
    request: &ResponsesRequest,
) -> Result<ResponsesRequest, Response> {
    let mut modified_request = request.clone();
    let mut conversation_items: Option<Vec<ResponseInputOutputItem>> = None;

    // Handle previous_response_id by loading response chain
    if let Some(ref prev_id_str) = modified_request.previous_response_id {
        let prev_id = ResponseId::from(prev_id_str.as_str());
        match ctx
            .response_storage
            .get_response_chain(&prev_id, None)
            .await
        {
            Ok(chain) => {
                let mut items = Vec::new();
                for stored in chain.responses.iter() {
                    if let Some(input_arr) = stored.input.as_array() {
                        for item in input_arr {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(input_item) => {
                                    items.push(input_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored input item: {}. Item: {}",
                                        e, item
                                    );
                                }
                            }
                        }
                    }

                    if let Some(output_arr) = stored.output.as_array() {
                        for item in output_arr {
                            match serde_json::from_value::<ResponseInputOutputItem>(item.clone()) {
                                Ok(output_item) => {
                                    items.push(output_item);
                                }
                                Err(e) => {
                                    warn!(
                                        "Failed to deserialize stored output item: {}. Item: {}",
                                        e, item
                                    );
                                }
                            }
                        }
                    }
                }
                conversation_items = Some(items);
                modified_request.previous_response_id = None;
            }
            Err(e) => {
                warn!(
                    "Failed to load previous response chain for {}: {}",
                    prev_id_str, e
                );
            }
        }
    }

    // Handle conversation by loading conversation history
    if let Some(ref conv_id_str) = request.conversation {
        let conv_id = ConversationId::from(conv_id_str.as_str());

        let conversation = ctx
            .conversation_storage
            .get_conversation(&conv_id)
            .await
            .map_err(|e| {
                error::internal_error(
                    "check_conversation_failed",
                    format!("Failed to check conversation: {}", e),
                )
            })?;

        if conversation.is_none() {
            return Err(error::not_found(
                "conversation_not_found",
                format!(
                    "Conversation '{}' not found. Please create the conversation first using the conversations API.",
                    conv_id_str
                )
            ));
        }

        const MAX_CONVERSATION_HISTORY_ITEMS: usize = 100;
        let params = data_connector::ListParams {
            limit: MAX_CONVERSATION_HISTORY_ITEMS,
            order: data_connector::SortOrder::Asc,
            after: None,
        };

        match ctx
            .conversation_item_storage
            .list_items(&conv_id, params)
            .await
        {
            Ok(stored_items) => {
                let mut items: Vec<ResponseInputOutputItem> = Vec::new();
                for item in stored_items.into_iter() {
                    if item.item_type == "message" {
                        if let Ok(content_parts) =
                            serde_json::from_value::<Vec<ResponseContentPart>>(item.content.clone())
                        {
                            items.push(ResponseInputOutputItem::Message {
                                id: item.id.0.clone(),
                                role: item.role.clone().unwrap_or_else(|| "user".to_string()),
                                content: content_parts,
                                status: item.status.clone(),
                            });
                        }
                    }
                }

                match &modified_request.input {
                    ResponseInput::Text(text) => {
                        items.push(ResponseInputOutputItem::Message {
                            id: format!("msg_u_{}", conv_id.0),
                            role: "user".to_string(),
                            content: vec![ResponseContentPart::InputText { text: text.clone() }],
                            status: Some("completed".to_string()),
                        });
                    }
                    ResponseInput::Items(current_items) => {
                        for item in current_items.iter() {
                            let normalized = responses::normalize_input_item(item);
                            items.push(normalized);
                        }
                    }
                }

                modified_request.input = ResponseInput::Items(items);
            }
            Err(e) => {
                warn!("Failed to load conversation history: {}", e);
            }
        }
    }

    // If we have conversation_items from previous_response_id, merge them
    if let Some(mut items) = conversation_items {
        match &modified_request.input {
            ResponseInput::Text(text) => {
                items.push(ResponseInputOutputItem::Message {
                    id: format!(
                        "msg_u_{}",
                        request
                            .previous_response_id
                            .as_ref()
                            .unwrap_or(&"new".to_string())
                    ),
                    role: "user".to_string(),
                    content: vec![ResponseContentPart::InputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                });
            }
            ResponseInput::Items(current_items) => {
                for item in current_items.iter() {
                    let normalized = responses::normalize_input_item(item);
                    items.push(normalized);
                }
            }
        }

        modified_request.input = ResponseInput::Items(items);
    }

    debug!(
        has_previous_response = request.previous_response_id.is_some(),
        has_conversation = request.conversation.is_some(),
        "Loaded conversation history"
    );

    Ok(modified_request)
}
