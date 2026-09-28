//! App construction helpers used by the standalone Atomesh mocker harness.

use std::sync::{Arc, OnceLock};

use axum::Router;
use data_connector::{
    MemoryConversationItemStorage, MemoryConversationStorage, MemoryResponseStorage,
};
use mesh::{
    app_context::AppContext,
    config::RouterConfig,
    core::{JobQueue, JobQueueConfig, LoadMonitor, WorkerRegistry},
    middleware::TokenBucket,
    policies::PolicyRegistry,
    reasoning_parser::ParserFactory as ReasoningParserFactory,
    routers::RouterTrait,
    server::{build_app, AppState},
    tokenizer::registry::TokenizerRegistry,
    tool_parser::ParserFactory as ToolParserFactory,
};

/// Create an Atomesh app context with in-memory stores and test-friendly queues.
pub async fn create_mocker_context(config: RouterConfig) -> Arc<AppContext> {
    create_mocker_context_inner(config, false).await
}

/// Create an Atomesh app context with parser factories initialized.
pub async fn create_mocker_context_with_parsers(config: RouterConfig) -> Arc<AppContext> {
    create_mocker_context_inner(config, true).await
}

/// Build the real Atomesh Axum app around an existing router and context.
pub fn create_mocker_app_with_context(
    router: Arc<dyn RouterTrait>,
    app_context: Arc<AppContext>,
) -> Router {
    let app_state = Arc::new(AppState {
        router,
        context: app_context.clone(),
        concurrency_queue_tx: None,
        router_manager: None,
    });

    let router_config = &app_context.router_config;
    let request_id_headers = router_config.request_id_headers.clone().unwrap_or_else(|| {
        vec![
            "x-request-id".to_string(),
            "x-correlation-id".to_string(),
            "x-trace-id".to_string(),
            "request-id".to_string(),
        ]
    });

    build_app(
        app_state,
        router_config.max_payload_size,
        request_id_headers,
    )
}

async fn create_mocker_context_inner(
    config: RouterConfig,
    with_parsers: bool,
) -> Arc<AppContext> {
    let client = reqwest::Client::new();
    let rate_limiter = match config.max_concurrent_requests {
        n if n <= 0 => None,
        n => {
            let rate_limit_tokens = config
                .rate_limit_tokens_per_second
                .filter(|&tokens| tokens > 0)
                .unwrap_or(n);
            Some(Arc::new(TokenBucket::new(
                n as usize,
                rate_limit_tokens as usize,
            )))
        }
    };

    let worker_registry = Arc::new(WorkerRegistry::new());
    let policy_registry = Arc::new(PolicyRegistry::new(config.policy.clone()));
    let response_storage = Arc::new(MemoryResponseStorage::new());
    let conversation_storage = Arc::new(MemoryConversationStorage::new());
    let conversation_item_storage = Arc::new(MemoryConversationItemStorage::new());
    let load_monitor = Some(Arc::new(LoadMonitor::new(
        worker_registry.clone(),
        policy_registry.clone(),
        client.clone(),
        config.worker_startup_check_interval_secs,
    )));
    let worker_job_queue = Arc::new(OnceLock::new());
    let workflow_engines = Arc::new(OnceLock::new());

    let reasoning_parser_factory = with_parsers.then(ReasoningParserFactory::new);
    let tool_parser_factory = with_parsers.then(ToolParserFactory::new);

    let app_context = Arc::new(
        AppContext::builder()
            .router_config(config.clone())
            .client(client)
            .rate_limiter(rate_limiter)
            .tokenizer_registry(Arc::new(TokenizerRegistry::new()))
            .reasoning_parser_factory(reasoning_parser_factory)
            .tool_parser_factory(tool_parser_factory)
            .worker_registry(worker_registry)
            .policy_registry(policy_registry)
            .response_storage(response_storage)
            .conversation_storage(conversation_storage)
            .conversation_item_storage(conversation_item_storage)
            .load_monitor(load_monitor)
            .worker_job_queue(worker_job_queue)
            .workflow_engines(workflow_engines)
            .build()
            .unwrap(),
    );

    let weak_context = Arc::downgrade(&app_context);
    let job_queue = JobQueue::new(JobQueueConfig::default(), weak_context);
    app_context
        .worker_job_queue
        .set(job_queue)
        .expect("JobQueue should only be initialized once");

    let engines = mesh::core::steps::WorkflowEngines::new(&config);
    app_context
        .workflow_engines
        .set(engines)
        .expect("WorkflowEngines should only be initialized once");

    app_context
}
