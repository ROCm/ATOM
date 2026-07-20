use std::sync::Arc;

use data_connector::{
    ConversationItemStorage, ConversationStorage, MemoryConversationItemStorage,
    MemoryConversationStorage, MemoryResponseStorage, ResponseStorage,
};

pub(crate) fn response_storage() -> Arc<dyn ResponseStorage> {
    Arc::new(MemoryResponseStorage::new())
}

pub(crate) fn conversation_storage() -> Arc<dyn ConversationStorage> {
    Arc::new(MemoryConversationStorage::new())
}

pub(crate) fn conversation_item_storage() -> Arc<dyn ConversationItemStorage> {
    Arc::new(MemoryConversationItemStorage::new())
}
