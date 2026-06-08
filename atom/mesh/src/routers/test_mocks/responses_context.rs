use std::sync::Arc;

use crate::routers::openai::responses::context::ResponsesContext;

use super::{
    app_context::{app_context, app_context_with_hf_tokenizer},
    pipeline::{pipeline_regular_default, pipeline_with_chat_path},
    storage::{conversation_item_storage, conversation_storage, response_storage},
};

pub(crate) fn responses_context() -> Arc<ResponsesContext> {
    Arc::new(ResponsesContext::new(
        pipeline_regular_default(),
        app_context(),
        response_storage(),
        conversation_storage(),
        conversation_item_storage(),
    ))
}

pub(crate) fn responses_context_with_chat_path(model_name: &str) -> Arc<ResponsesContext> {
    Arc::new(ResponsesContext::new(
        pipeline_with_chat_path(),
        app_context_with_hf_tokenizer(model_name),
        response_storage(),
        conversation_storage(),
        conversation_item_storage(),
    ))
}
