// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Copyright (c) 2026 ATOM Project Contributors
//! gRPC workers through the local worker registration workflow and its tokenizer job:
//! an ATOM worker's detection, labels, version refusal, chat template and DP ranks; vLLM's.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use mesh_grpc::vllm_proto as vllm;
use serde_json::{json, Value};
use tonic::{
    transport::{server::Router, server::TcpIncoming, Server},
    Request, Response, Status, Streaming,
};
use wfaas::{StepExecutor, WorkflowContext, WorkflowId, WorkflowInstanceId};

use super::{
    create_local_worker_workflow_data, get_dp_info, CreateLocalWorkerStep,
    DetectConnectionModeStep, DiscoverDPInfoStep, DiscoverMetadataStep, RegisterWorkersStep,
};
use crate::{
    app_context::AppContext,
    atom_engine::grpc::{
        atom_engine_server::{AtomEngine, AtomEngineServer},
        generate_response, AbortRequest, AbortResponse, Frames, GenerateComplete, GenerateRequest,
        GenerateResponse, GetModelInfoRequest, GetModelInfoResponse, GetServerInfoRequest,
        GetServerInfoResponse, HealthCheckRequest, HealthCheckResponse,
        PrepareStructuredOutputRequest, PrepareStructuredOutputResponse,
    },
    config::RouterConfig,
    core::{
        placement::PlacementPlan,
        steps::{workflow_data::LocalWorkerWorkflowData, TokenizerConfigRequest, WorkflowEngines},
        worker::RuntimeType,
        Job, JobQueue, JobQueueConfig,
    },
    protocols::{
        chat::ChatCompletionRequest, generate::GenerateRequest as GenerateBody,
        worker_spec::WorkerConfigRequest,
    },
    routers::{
        grpc::engine::GrpcEngine,
        prepare::{
            build_generate_payload, fast_tokenizer::FastTokenizer, prepare_chat, prepare_generate,
        },
        test_mocks::{
            app_context::app_context_with_router_config,
            tokenizer::{byte_level_tokenizer_dir, hf_tokenizer_path, BYTE_LEVEL_MERGES},
        },
    },
    tokenizer::TokenizerRegistry,
};

type Empty<T> = tokio_stream::Empty<Result<T, Status>>;

/// An ATOM instance as registration sees it: its model info, protocol version and DP
/// size. Each Generate records its DP rank and completes.
#[derive(Default)]
struct FakeAtom {
    info: GetModelInfoResponse,
    protocol_version: u32,
    dp_size: usize,
    ranks: Mutex<Vec<Option<i32>>>,
    /// GetModelInfo fails.
    no_model_info: bool,
    /// HealthCheck answers unhealthy.
    unhealthy: bool,
}

#[tonic::async_trait]
impl AtomEngine for FakeAtom {
    type GenerateStream = tokio_stream::Once<Result<GenerateResponse, Status>>;
    type SessionStream = Empty<Frames>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let rank = request.into_inner().data_parallel_rank;
        self.ranks.lock().unwrap().push(rank);
        let complete = generate_response::Response::Complete(GenerateComplete::default());
        Ok(Response::new(tokio_stream::once(Ok(GenerateResponse {
            response: Some(complete),
        }))))
    }

    async fn abort(&self, _: Request<AbortRequest>) -> Result<Response<AbortResponse>, Status> {
        Err(Status::unimplemented("abort"))
    }

    async fn get_model_info(
        &self,
        _: Request<GetModelInfoRequest>,
    ) -> Result<Response<GetModelInfoResponse>, Status> {
        if self.no_model_info {
            return Err(Status::unavailable("model info"));
        }
        Ok(Response::new(self.info.clone()))
    }

    async fn get_server_info(
        &self,
        _: Request<GetServerInfoRequest>,
    ) -> Result<Response<GetServerInfoResponse>, Status> {
        // As ATOM's /server_info body names them.
        let server_info =
            json!({ "model_id": self.info.served_model_name, "dp_size": self.dp_size });
        Ok(Response::new(GetServerInfoResponse {
            protocol_version: self.protocol_version,
            server_info_json: server_info.to_string(),
            ..Default::default()
        }))
    }

    async fn health_check(
        &self,
        _: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        Ok(Response::new(HealthCheckResponse {
            healthy: !self.unhealthy,
            message: if self.unhealthy {
                "engine core died".into()
            } else {
                String::new()
            },
        }))
    }

    async fn session(
        &self,
        _: Request<Streaming<Frames>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        Err(Status::unimplemented("session"))
    }

    async fn prepare_structured_output(
        &self,
        _: Request<PrepareStructuredOutputRequest>,
    ) -> Result<Response<PrepareStructuredOutputResponse>, Status> {
        Err(Status::unimplemented("prepare_structured_output"))
    }
}

/// A vLLM instance that answers HealthCheck and GetModelInfo.
#[derive(Default)]
struct FakeVllm {
    /// HealthCheck answers unhealthy.
    unhealthy: bool,
}

#[tonic::async_trait]
impl vllm::vllm_engine_server::VllmEngine for FakeVllm {
    type GenerateStream = Empty<vllm::GenerateResponse>;

    async fn generate(
        &self,
        _: Request<vllm::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        Err(Status::unimplemented("generate"))
    }

    async fn embed(
        &self,
        _: Request<vllm::EmbedRequest>,
    ) -> Result<Response<vllm::EmbedResponse>, Status> {
        Err(Status::unimplemented("embed"))
    }

    async fn health_check(
        &self,
        _: Request<vllm::HealthCheckRequest>,
    ) -> Result<Response<vllm::HealthCheckResponse>, Status> {
        Ok(Response::new(vllm::HealthCheckResponse {
            healthy: !self.unhealthy,
            message: String::new(),
        }))
    }

    async fn abort(
        &self,
        _: Request<vllm::AbortRequest>,
    ) -> Result<Response<vllm::AbortResponse>, Status> {
        Err(Status::unimplemented("abort"))
    }

    async fn get_model_info(
        &self,
        _: Request<vllm::GetModelInfoRequest>,
    ) -> Result<Response<vllm::GetModelInfoResponse>, Status> {
        Ok(Response::new(vllm::GetModelInfoResponse {
            model_path: hf_tokenizer_path().into(),
            max_context_length: 2048,
            ..Default::default()
        }))
    }

    async fn get_server_info(
        &self,
        _: Request<vllm::GetServerInfoRequest>,
    ) -> Result<Response<vllm::GetServerInfoResponse>, Status> {
        Err(Status::unimplemented("get_server_info"))
    }
}

/// Serves `router` on a free local port and returns its `grpc://` URL.
async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("grpc://{}", listener.local_addr().unwrap());
    tokio::spawn(router.serve_with_incoming(TcpIncoming::from(listener)));
    url
}

async fn serve_atom(info: GetModelInfoResponse, protocol_version: u32) -> String {
    let fake = FakeAtom {
        info,
        protocol_version,
        ..Default::default()
    };
    serve(Server::builder().add_service(AtomEngineServer::new(fake))).await
}

/// GetModelInfo of model `name`, whose tokenizer is the HF test fixture.
fn model_info(name: &str, chat_template: &str, default_kwargs: &str) -> GetModelInfoResponse {
    GetModelInfoResponse {
        model_path: format!("/models/{name}"),
        tokenizer_path: hf_tokenizer_path().into(),
        served_model_name: name.into(),
        max_context_length: 4096,
        eos_token_ids: vec![2, 7],
        chat_template: Some(chat_template.into()),
        default_chat_template_kwargs_json: default_kwargs.into(),
        reasoning_open_markers: Vec::new(),
        ..Default::default()
    }
}

/// A router context with the job queue and workflow engines that registration uses.
fn context() -> Arc<AppContext> {
    context_with(RouterConfig::default())
}

fn context_with(config: RouterConfig) -> Arc<AppContext> {
    let ctx = app_context_with_router_config(config);
    let queue = JobQueue::new(JobQueueConfig::default(), Arc::downgrade(&ctx));
    ctx.worker_job_queue.set(queue).unwrap();
    let engines = WorkflowEngines::new(&ctx.router_config);
    ctx.workflow_engines.set(engines).unwrap();
    ctx
}

/// Runs the local worker registration workflow on `config`, as an AddWorker job does.
async fn register(ctx: &Arc<AppContext>, config: Value) -> Result<String, String> {
    let config: WorkerConfigRequest = serde_json::from_value(config).unwrap();
    let url = config.url.clone();
    let engine = &ctx.workflow_engines.get().unwrap().local_worker;
    let data = create_local_worker_workflow_data(config, ctx.clone());
    let id = engine
        .start_workflow(WorkflowId::new("local_worker_registration"), data)
        .await
        .unwrap();
    engine
        .wait_for_completion(id, &url, Duration::from_secs(60))
        .await
}

/// The prompt the gRPC pipeline renders for a user's "hi" once the tokenizer job
/// has registered `model`'s tokenizer.
async fn render(ctx: &AppContext, model: &str, kwargs: Value) -> String {
    let registered = async {
        while !ctx.tokenizer_registry.contains(model) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(60), registered)
        .await
        .expect("the tokenizer job registers the model's tokenizer");
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "chat_template_kwargs": kwargs,
    });
    let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    let (_, resp) = prepare_chat(Arc::new(request), None, Some(model.into()), ctx).unwrap();
    resp.original_text.unwrap()
}

#[tokio::test]
async fn an_atom_worker_registers_as_atom_with_its_model_info_labels() {
    let info = GetModelInfoResponse {
        usage_on_finish: true,
        ..model_info("atom-labels", "label: {{ messages[0].content }}", "{}")
    };
    let url = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();

    let worker = ctx.worker_registry.get_by_url(&url).unwrap();
    assert_eq!(worker.metadata().runtime_type, RuntimeType::Atom);
    assert_eq!(worker.model_id(), "atom-labels");
    let labels = &worker.metadata().labels;
    for (key, value) in [
        ("model_path", "/models/atom-labels"),
        ("tokenizer_path", hf_tokenizer_path()),
        ("served_model_name", "atom-labels"),
        ("max_context_length", "4096"),
        ("eos_token_ids", "[2,7]"),
        ("chat_template", "label: {{ messages[0].content }}"),
        ("default_chat_template_kwargs_json", "{}"),
        ("usage_on_finish", "true"),
    ] {
        assert_eq!(labels.get(key).map(String::as_str), Some(value), "{key}");
    }
}

#[tokio::test]
async fn an_atom_workers_parsers_come_from_its_model_info_unless_configured() {
    let info = GetModelInfoResponse {
        tool_call_parser: Some("qwen_xml".into()),
        reasoning_parser: Some("qwen3".into()),
        ..model_info("atom-parsers", "", "{}")
    };
    let labelled = serve_atom(info.clone(), 1).await;
    let configured = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": labelled })).await.unwrap();
    register(&ctx, json!({ "url": configured, "tool_parser": "kimik2" }))
        .await
        .unwrap();

    let worker = ctx.worker_registry.get_by_url(&labelled).unwrap();
    assert_eq!(worker.tool_parser(), Some("qwen_xml"));
    assert_eq!(worker.reasoning_parser(), Some("qwen3"));
    let worker = ctx.worker_registry.get_by_url(&configured).unwrap();
    assert_eq!(worker.tool_parser(), Some("kimik2"));
    assert_eq!(worker.reasoning_parser(), Some("qwen3"));
}

#[tokio::test]
async fn an_atom_model_without_a_chat_template_answers_chats_with_501() {
    let info = GetModelInfoResponse {
        chat_template: None,
        ..model_info("atom-encoder", "", "{}")
    };
    let url = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();
    assert_chats_refused_and_completions_served(&ctx, "atom-encoder").await;
}

/// A DeepSeek V4 tokenizer renders with llm-tokenizer's built-in encoder, which ignores
/// the template the ATOM worker advertises.
#[tokio::test]
async fn an_atom_model_whose_tokenizer_ignores_its_template_answers_chats_with_501() {
    let dir = tempfile::TempDir::new().unwrap();
    std::fs::copy(hf_tokenizer_path(), dir.path().join("tokenizer.json")).unwrap();
    let config = json!({"architectures": ["DeepseekV4ForCausalLM"]});
    std::fs::write(dir.path().join("config.json"), config.to_string()).unwrap();
    let info = GetModelInfoResponse {
        tokenizer_path: dir.path().to_str().unwrap().into(),
        ..model_info("atom-deepseek", "atom: {{ messages[0].content }}", "{}")
    };
    let url = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();
    assert_chats_refused_and_completions_served(&ctx, "atom-deepseek").await;
}

/// Once the tokenizer job has registered `model`'s tokenizer, its chats get 501 and its
/// completions are prepared.
async fn assert_chats_refused_and_completions_served(ctx: &Arc<AppContext>, model: &str) {
    let registered = async {
        while !ctx.tokenizer_registry.contains(model) {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    };
    tokio::time::timeout(Duration::from_secs(60), registered)
        .await
        .expect("the tokenizer job registers the model's tokenizer");

    let chat = json!({"model": model, "messages": [{"role": "user", "content": "hi"}]});
    let chat: ChatCompletionRequest = serde_json::from_value(chat).unwrap();
    let Err(refused) = prepare_chat(Arc::new(chat), None, Some(model.into()), ctx) else {
        panic!("a chat atomesh cannot render as ATOM does is refused");
    };
    assert_eq!(refused.status(), http::StatusCode::NOT_IMPLEMENTED);
    let completion: GenerateBody = serde_json::from_value(json!({"text": "hi"})).unwrap();
    assert!(prepare_generate(Arc::new(completion), None, Some(model.into()), ctx).is_ok());
}

#[tokio::test]
async fn an_atom_worker_with_another_protocol_version_is_refused() {
    let url = serve_atom(model_info("atom-v2", "", "{}"), 2).await;
    let ctx = context();
    let error = register(&ctx, json!({ "url": url })).await.unwrap_err();

    assert!(error.contains("protocol version is 2"), "{error}");
    assert!(ctx.worker_registry.get_by_url(&url).is_none());
}

/// An ATOM worker whose HealthCheck reports it unhealthy is not registered, as an HTTP
/// worker whose /health fails is not.
#[tokio::test]
async fn an_atom_worker_reporting_unhealthy_is_not_registered() {
    let fake = FakeAtom {
        info: model_info("atom-down", "", "{}"),
        protocol_version: 1,
        unhealthy: true,
        ..Default::default()
    };
    let url = serve(Server::builder().add_service(AtomEngineServer::new(fake))).await;
    let ctx = context_with(RouterConfig {
        worker_startup_timeout_secs: 2,
        ..Default::default()
    });
    let error = register(&ctx, json!({ "url": url })).await.unwrap_err();

    assert!(error.contains("engine core died"), "{error}");
    assert!(ctx.worker_registry.get_by_url(&url).is_none());
}

/// The runtime the health probe found stands when GetModelInfo then fails.
#[tokio::test]
async fn an_atom_worker_whose_model_info_fails_still_registers_as_atom() {
    let fake = FakeAtom {
        protocol_version: 1,
        no_model_info: true,
        ..Default::default()
    };
    let url = serve(Server::builder().add_service(AtomEngineServer::new(fake))).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();

    let worker = ctx.worker_registry.get_by_url(&url).unwrap();
    assert_eq!(worker.metadata().runtime_type, RuntimeType::Atom);
}

#[tokio::test]
async fn the_tokenizer_job_renders_with_the_label_template_without_a_flag() {
    let info = model_info("atom-template", "label: {{ messages[0].content }}", "{}");
    let url = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();

    // The fixture's own template would render "user: hi".
    let rendered = render(&ctx, "atom-template", Value::Null).await;
    assert_eq!(rendered, "label: hi");
}

/// Runs the worker removal workflow on `url`, as a RemoveWorker job does.
async fn remove(ctx: &Arc<AppContext>, url: &str) {
    let engine = &ctx.workflow_engines.get().unwrap().worker_removal;
    let data = super::create_worker_removal_workflow_data(url.into(), false, ctx.clone());
    let id = engine
        .start_workflow(WorkflowId::new("worker_removal"), data)
        .await
        .unwrap();
    let done = engine.wait_for_completion(id, url, Duration::from_secs(30));
    done.await.unwrap();
}

/// All live workers of a model serve one template: a worker advertising another is
/// refused while one is registered; once the last leaves, its tokenizer goes, and the
/// next worker's template loads afresh.
#[tokio::test]
async fn a_worker_with_another_template_is_refused_until_the_model_s_last_worker_leaves() {
    let ctx = context();
    let worker = |template: &str| {
        let template = format!("{template}: {{{{ messages[0].content }}}}");
        serve_atom(model_info("atom-retemplate", &template, "{}"), 1)
    };
    let first = worker("first").await;
    register(&ctx, json!({ "url": first })).await.unwrap();
    assert_eq!(
        render(&ctx, "atom-retemplate", Value::Null).await,
        "first: hi"
    );

    let second = worker("second").await;
    let refused = register(&ctx, json!({ "url": second })).await.unwrap_err();
    assert!(refused.contains("redeploying"), "{refused}");
    assert!(ctx.worker_registry.get_by_url(&second).is_none());
    assert_eq!(
        render(&ctx, "atom-retemplate", Value::Null).await,
        "first: hi"
    );

    remove(&ctx, &first).await;
    assert!(!ctx.tokenizer_registry.contains("atom-retemplate"));
    register(&ctx, json!({ "url": second })).await.unwrap();
    assert_eq!(
        render(&ctx, "atom-retemplate", Value::Null).await,
        "second: hi"
    );
}

/// While the model's tokenizer is still loading, a worker with another template is
/// refused too: the registered workers' template is the one loading.
#[tokio::test]
async fn a_worker_with_another_template_is_refused_while_the_tokenizer_loads() {
    let ctx = context();
    let worker = |template: &str| {
        let template = format!("{template}: {{{{ messages[0].content }}}}");
        serve_atom(model_info("atom-loading", &template, "{}"), 1)
    };
    let loading = ctx.tokenizer_registering.entry("atom-loading".into());
    let loading = loading.or_default().clone();
    let held = loading.lock().await; // the first worker's tokenizer job waits on this
    let first = worker("first").await;
    register(&ctx, json!({ "url": first })).await.unwrap();
    assert!(!ctx.tokenizer_registry.contains("atom-loading"));

    let second = worker("second").await;
    let refused = register(&ctx, json!({ "url": second })).await.unwrap_err();
    assert!(refused.contains("redeploying"), "{refused}");
    assert!(ctx.worker_registry.get_by_url(&second).is_none());
    drop(held);
    assert_eq!(render(&ctx, "atom-loading", Value::Null).await, "first: hi");
}

/// Registration of the worker at `url` run up to, not including, its register step.
async fn created_worker(
    ctx: &Arc<AppContext>,
    url: &str,
) -> WorkflowContext<LocalWorkerWorkflowData> {
    let config = serde_json::from_value(json!({ "url": url })).unwrap();
    let data = create_local_worker_workflow_data(config, ctx.clone());
    let mut context = WorkflowContext::new(WorkflowInstanceId::new(), data);
    DetectConnectionModeStep
        .execute(&mut context)
        .await
        .unwrap();
    DiscoverMetadataStep.execute(&mut context).await.unwrap();
    DiscoverDPInfoStep.execute(&mut context).await.unwrap();
    CreateLocalWorkerStep.execute(&mut context).await.unwrap();
    context
}

/// Of two registrations with different templates that both pass worker creation
/// before either registers, exactly one is admitted.
#[tokio::test]
async fn of_two_concurrent_registrations_with_different_templates_one_is_admitted() {
    let ctx = context();
    let mut created = Vec::new();
    for template in ["first", "second"] {
        let template = format!("{template}: {{{{ messages[0].content }}}}");
        let url = serve_atom(model_info("atom-race", &template, "{}"), 1).await;
        created.push(created_worker(&ctx, &url).await);
    }
    let mut registered = Vec::new();
    for context in &mut created {
        registered.push(
            RegisterWorkersStep
                .execute(context)
                .await
                .map_err(|e| e.to_string()),
        );
    }
    assert!(registered[0].is_ok(), "{:?}", registered[0]);
    let refused = registered[1].as_ref().unwrap_err();
    assert!(refused.contains("redeploying"), "{refused}");
    assert_eq!(ctx.worker_registry.get_by_model("atom-race").len(), 1);
}

/// A model's last worker leaving does not drop the tokenizer that a worker admitted
/// meanwhile with the same template uses.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leaving_last_worker_keeps_the_tokenizer_a_same_template_worker_was_admitted_with() {
    let ctx = context();
    let info = || model_info("atom-handover", "same: {{ messages[0].content }}", "{}");
    let first = serve_atom(info(), 1).await;
    register(&ctx, json!({ "url": first })).await.unwrap();
    assert_eq!(render(&ctx, "atom-handover", Value::Null).await, "same: hi");
    let second = created_worker(&ctx, &serve_atom(info(), 1).await).await;

    // The second worker is admitted against the loaded tokenizer and registered
    // under this lock, after the first left the registry.
    let admitting = ctx.template_admission.lock();
    let removal = tokio::spawn({
        let (ctx, first) = (ctx.clone(), first.clone());
        async move { remove(&ctx, &first).await }
    });
    while ctx.worker_registry.get_by_url(&first).is_some() {
        std::thread::sleep(Duration::from_millis(10));
    }
    std::thread::sleep(Duration::from_millis(200)); // the removal's tokenizer check
    for worker in second.data.actual_workers.as_ref().unwrap() {
        ctx.worker_registry.register(worker.clone());
    }
    drop(admitting);
    removal.await.unwrap();

    assert!(ctx.tokenizer_registry.contains("atom-handover"));
    assert_eq!(render(&ctx, "atom-handover", Value::Null).await, "same: hi");
}

/// A model's tokenizer loaded with a template file keeps it: a worker that only
/// advertises a template serves that file, which ranks above an advertised one.
#[tokio::test]
async fn a_tokenizer_s_template_file_wins_over_an_advertised_template() {
    let dir = tempfile::TempDir::new().unwrap();
    let file = dir.path().join("file.jinja");
    std::fs::write(&file, "file: {{ messages[0].content }}").unwrap();
    let ctx = context();
    let config = TokenizerConfigRequest {
        id: TokenizerRegistry::generate_id(),
        name: "atom-file".into(),
        source: hf_tokenizer_path().into(),
        chat_template_path: Some(file.to_str().unwrap().into()),
        inline_chat_template: None,
        worker_url: None,
        cache_config: None,
        fail_on_duplicate: false,
    };
    let job = Job::AddTokenizer {
        config: Box::new(config),
    };
    let queue = ctx.worker_job_queue.get().unwrap();
    queue.submit(job).await.unwrap();
    assert_eq!(render(&ctx, "atom-file", Value::Null).await, "file: hi");

    for _ in 0..2 {
        let info = model_info("atom-file", "label: {{ messages[0].content }}", "{}");
        let url = serve_atom(info, 1).await;
        register(&ctx, json!({ "url": url })).await.unwrap();
    }
    assert_eq!(render(&ctx, "atom-file", Value::Null).await, "file: hi");
}

/// The label template is set before fastokens wraps the tokenizer: a FastTokenizer
/// refuses `set_chat_template`.
#[tokio::test]
async fn a_fastokens_model_renders_with_the_label_template() {
    let dir = byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, json!({}));
    let info = GetModelInfoResponse {
        tokenizer_path: dir.path().to_str().unwrap().into(),
        ..model_info("atom-fastokens", "label: {{ messages[0].content }}", "{}")
    };
    let url = serve_atom(info, 1).await;
    let mut config = RouterConfig::default();
    config.fastokens.models = vec!["atom-fastokens".into()];
    config.fastokens.strict = true;
    let ctx = context_with(config);
    register(&ctx, json!({ "url": url })).await.unwrap();

    let rendered = render(&ctx, "atom-fastokens", Value::Null).await;
    assert_eq!(rendered, "label: hi");
    let tokenizer = ctx.tokenizer_registry.get("atom-fastokens").unwrap();
    assert!(tokenizer.as_any().is::<FastTokenizer>());
}

#[tokio::test]
async fn an_explicit_chat_template_wins_over_the_label() {
    let dir = tempfile::TempDir::new().unwrap();
    let flag = dir.path().join("flag.jinja");
    std::fs::write(&flag, "flag: {{ messages[0].content }}").unwrap();
    let info = model_info("atom-flag", "label: {{ messages[0].content }}", "{}");
    let url = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url, "chat_template": flag }))
        .await
        .unwrap();

    assert_eq!(render(&ctx, "atom-flag", Value::Null).await, "flag: hi");
}

#[tokio::test]
async fn default_template_kwargs_join_the_request_kwargs_the_request_winning() {
    let template = "{{ greeting }} {{ flag }}: {{ messages[0].content }}";
    let defaults = r#"{"greeting": "hello", "flag": "default"}"#;
    let url = serve_atom(model_info("atom-kwargs", template, defaults), 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();

    let rendered = render(&ctx, "atom-kwargs", Value::Null).await;
    assert_eq!(rendered, "hello default: hi");
    let rendered = render(&ctx, "atom-kwargs", json!({ "flag": "request" })).await;
    assert_eq!(rendered, "hello request: hi");
}

/// ATOM's HTTP chat sets `thinking_effort` from a reasoning effort it reads, over the
/// defaults and the request's kwargs, and drops one its templates do not read.
#[tokio::test]
async fn a_reasoning_effort_overrides_the_default_thinking_effort_as_on_http() {
    let template = "{{ thinking_effort }} {{ reasoning_effort }}: {{ messages[0].content }}";
    let defaults = r#"{"thinking_effort": "low"}"#;
    let url = serve_atom(model_info("atom-effort", template, defaults), 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();
    render(&ctx, "atom-effort", Value::Null).await;

    for (effort, expected) in [("high", "high : hi"), ("medium", "low : hi")] {
        let body = json!({
            "model": "atom-effort",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": effort,
            "chat_template_kwargs": {"thinking_effort": "low"},
        });
        let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        let model = Some("atom-effort".into());
        let (_, resp) = prepare_chat(Arc::new(request), None, model, &ctx).unwrap();
        assert_eq!(resp.original_text.as_deref(), Some(expected), "{effort}");
    }
}

#[tokio::test]
async fn a_vllm_worker_is_still_detected_as_vllm() {
    let service = vllm::vllm_engine_server::VllmEngineServer::new(FakeVllm::default());
    let url = serve(Server::builder().add_service(service)).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();

    let worker = ctx.worker_registry.get_by_url(&url).unwrap();
    assert_eq!(worker.metadata().runtime_type, RuntimeType::Vllm);
    let labels = &worker.metadata().labels;
    assert_eq!(labels.get("max_context_length").unwrap(), "2048");
}

/// A reasoning effort of "none" switches the model's reasoning toggle off over its
/// default, as ATOM's HTTP chat does (`resolve_reasoning_controls`).
#[tokio::test]
async fn a_reasoning_effort_of_none_switches_the_default_on_toggle_off_as_on_http() {
    let template = "thinking={{ enable_thinking }}: {{ messages[0].content }}";
    let info = GetModelInfoResponse {
        reasoning_toggle_json: r#"["enable_thinking", false, true]"#.into(),
        ..model_info("atom-toggle", template, r#"{"enable_thinking": true}"#)
    };
    let url = serve_atom(info, 1).await;
    let ctx = context();
    register(&ctx, json!({ "url": url })).await.unwrap();
    assert_eq!(
        render(&ctx, "atom-toggle", Value::Null).await,
        "thinking=True: hi"
    );

    let body = json!({
        "model": "atom-toggle",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none",
    });
    let request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    let model = Some("atom-toggle".into());
    let (_, resp) = prepare_chat(Arc::new(request), None, model, &ctx).unwrap();
    assert_eq!(resp.original_text.as_deref(), Some("thinking=False: hi"));
}

/// A vLLM worker's HealthCheck answer does not decide its registration, as before
/// ATOM workers were served: only an ATOM worker's must say healthy.
#[tokio::test]
async fn a_vllm_worker_reporting_unhealthy_still_registers() {
    let fake = FakeVllm { unhealthy: true };
    let service = vllm::vllm_engine_server::VllmEngineServer::new(fake);
    let url = serve(Server::builder().add_service(service)).await;
    let ctx = context_with(RouterConfig {
        worker_startup_timeout_secs: 2,
        ..Default::default()
    });
    register(&ctx, json!({ "url": url })).await.unwrap();

    let worker = ctx.worker_registry.get_by_url(&url).unwrap();
    assert_eq!(worker.metadata().runtime_type, RuntimeType::Vllm);
}

#[tokio::test]
async fn an_atom_worker_reporting_dp_size_4_registers_four_ranks_each_sending_its_own() {
    let fake = Arc::new(FakeAtom {
        info: model_info("atom-dp", "", "{}"),
        protocol_version: 1,
        dp_size: 4,
        ..Default::default()
    });
    let url = serve(Server::builder().add_service(AtomEngineServer::from_arc(fake.clone()))).await;
    let info = get_dp_info(&url, None, true).await.unwrap();
    assert_eq!((info.dp_size, info.model_id.as_str()), (4, "atom-dp"));

    let ctx = context();
    register(&ctx, json!({ "url": url, "dp_aware": true }))
        .await
        .unwrap();
    assert_eq!(ctx.worker_registry.get_all().len(), 4);
    let body: GenerateBody = serde_json::from_value(json!({ "input_ids": [7, 8, 9] })).unwrap();
    for rank in 0..4 {
        let worker = ctx.worker_registry.get_by_url(&format!("{url}@{rank}"));
        let worker = worker.unwrap_or_else(|| panic!("no worker {url}@{rank}"));
        assert_eq!(worker.metadata().runtime_type, RuntimeType::Atom);
        assert_eq!((worker.dp_rank(), worker.dp_size()), (Some(rank), Some(4)));

        // Through the gRPC engine, as the router sends a request to the worker it chose.
        let plan = PlacementPlan::Single {
            worker,
            policy_name: "random",
        };
        let mut payload = build_generate_payload(format!("r{rank}"), &body, None, vec![7, 8, 9]);
        GrpcEngine::new()
            .dispatch(&plan, &mut payload)
            .await
            .unwrap();
    }
    assert_eq!(
        *fake.ranks.lock().unwrap(),
        [Some(0), Some(1), Some(2), Some(3)]
    );
}
