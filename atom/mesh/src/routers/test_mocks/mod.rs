//! Shared test fixtures for the routers subtree.

pub(crate) mod app_context;
pub(crate) mod dispatcher;
pub(crate) mod parsers;
pub(crate) mod pipeline;
pub(crate) mod planner;
pub(crate) mod registries;
pub(crate) mod responses_context;
pub(crate) mod storage;
pub(crate) mod tokenizer;
pub(crate) mod workers;

pub(crate) use app_context::{
    app_context, app_context_with_hf_tokenizer, app_context_with_tokenizer_registry,
};
pub(crate) use dispatcher::MockDispatcher;
pub(crate) use parsers::{reasoning_parser_factory, tool_parser_factory};
pub(crate) use pipeline::pipeline_with;
pub(crate) use planner::MockPdPlanner;
pub(crate) use responses_context::{responses_context, responses_context_with_chat_path};
pub(crate) use tokenizer::{hf_tokenizer, tokenizer, tokenizer_registry_with};
pub(crate) use workers::{mock_grpc_worker, mock_http_only_worker};
