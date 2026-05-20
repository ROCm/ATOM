//! Shared context for /v1/responses endpoint handlers

use std::sync::Arc;

use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};

use crate::{app_context::AppContext, routers::grpc::pipeline::Pipeline};

/// Context for /v1/responses endpoint
///
/// All fields are Arc/shared references, so cloning this context is cheap.
#[derive(Clone)]
pub(crate) struct ResponsesContext {
    /// Chat pipeline for executing requests
    pub pipeline: Arc<Pipeline>,

    /// Shared components (tokenizer, parsers, configured names)
    pub components: Arc<AppContext>,

    /// Response storage backend
    pub response_storage: Arc<dyn ResponseStorage>,

    /// Conversation storage backend
    pub conversation_storage: Arc<dyn ConversationStorage>,

    /// Conversation item storage backend
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

impl ResponsesContext {
    /// Create a new responses context
    pub fn new(
        pipeline: Arc<Pipeline>,
        components: Arc<AppContext>,
        response_storage: Arc<dyn ResponseStorage>,
        conversation_storage: Arc<dyn ConversationStorage>,
        conversation_item_storage: Arc<dyn ConversationItemStorage>,
    ) -> Self {
        Self {
            pipeline,
            components,
            response_storage,
            conversation_storage,
            conversation_item_storage,
        }
    }
}
