use std::sync::{Arc, OnceLock};

use crate::app_context::{AppContext, AppContextBuilder};
use crate::config::RouterConfig;
use crate::core::worker::Worker;
use crate::tokenizer::registry::TokenizerRegistry;

use super::{
    parsers::{reasoning_parser_factory, tool_parser_factory},
    registries::{policy_registry, worker_registry},
    storage::{conversation_item_storage, conversation_storage, response_storage},
    tokenizer::tokenizer_registry_with_hf,
};

pub(crate) fn app_context() -> Arc<AppContext> {
    app_context_with(Vec::new())
}

pub(crate) fn app_context_with(workers: Vec<Arc<dyn Worker>>) -> Arc<AppContext> {
    app_context_full(workers, Arc::new(TokenizerRegistry::new()))
}

pub(crate) fn app_context_with_tokenizer_registry(
    registry: Arc<TokenizerRegistry>,
) -> Arc<AppContext> {
    app_context_full(Vec::new(), registry)
}

pub(crate) fn app_context_with_hf_tokenizer(model_name: &str) -> Arc<AppContext> {
    app_context_full(Vec::new(), tokenizer_registry_with_hf(model_name))
}

fn app_context_full(
    workers: Vec<Arc<dyn Worker>>,
    tokenizer_registry: Arc<TokenizerRegistry>,
) -> Arc<AppContext> {
    let ctx = AppContextBuilder::new()
        .client(reqwest::Client::new())
        .router_config(RouterConfig::default())
        .tokenizer_registry(tokenizer_registry)
        .reasoning_parser_factory(Some(reasoning_parser_factory()))
        .tool_parser_factory(Some(tool_parser_factory()))
        .worker_registry(worker_registry(workers))
        .policy_registry(policy_registry())
        .response_storage(response_storage())
        .conversation_storage(conversation_storage())
        .conversation_item_storage(conversation_item_storage())
        .worker_job_queue(Arc::new(OnceLock::new()))
        .workflow_engines(Arc::new(OnceLock::new()))
        .build()
        .expect("test AppContext must build");
    Arc::new(ctx)
}
