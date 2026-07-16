//! Shared context for /v1/responses endpoint handlers.

use std::sync::Arc;

use data_connector::{ConversationItemStorage, ConversationStorage, ResponseStorage};

use crate::{app_context::AppContext, routers::grpc::pipeline::Pipeline};

#[derive(Clone)]
pub(crate) struct ResponsesContext {
    pub pipeline: Arc<Pipeline>,
    pub components: Arc<AppContext>,
    pub response_storage: Arc<dyn ResponseStorage>,
    pub conversation_storage: Arc<dyn ConversationStorage>,
    pub conversation_item_storage: Arc<dyn ConversationItemStorage>,
}

impl ResponsesContext {
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
