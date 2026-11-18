//! `AtomEngineRouter` over fake EngineCores (`atom_engine::test_support`): a
//! request goes through preparation, the client and the Core's side of the
//! protocol, and comes back rendered. The tokenizer is a word-level one
//! written to a temporary directory, so nothing is downloaded.

use std::{sync::Arc, thread};

use axum::{body::to_bytes, http::StatusCode, response::Response};
use serde_json::{json, Value};
use tempfile::TempDir;

use super::AtomEngineRouter;
use crate::{
    app_context::AppContextBuilder,
    atom_engine::{
        test_support::{self, FakeCore},
        wire::pb,
        AtomEngine,
    },
    config::RouterConfig,
    core::prepare_pool::PreparePoolRuntime,
    routers::RouterTrait,
};

const MODEL: &str = "tiny";
const HELLO: i32 = 1;
const WORLD: i32 = 2;

/// A word-level tokenizer and a chat template that renders `role: content`
/// lines and ends on `assistant:`.
fn write_tokenizer(dir: &std::path::Path) -> String {
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "hello": HELLO, "world": WORLD, "user": 3, "assistant": 4, ":": 5},
            "unk_token": "<unk>",
        },
    });
    let config = json!({
        "chat_template": "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
                          {% if add_generation_prompt %}assistant:{% endif %}",
    });
    let path = dir.join("tokenizer.json");
    std::fs::write(&path, tokenizer.to_string()).unwrap();
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
    path.display().to_string()
}

/// A router over `ranks` fake Cores, the Cores handed to the test.
struct Served {
    router: AtomEngineRouter,
    cores: Vec<FakeCore>,
    _pool: PreparePoolRuntime,
    _dir: TempDir,
}

impl Served {
    async fn start(ranks: usize) -> Self {
        let dir = TempDir::new().unwrap();
        let tokenizer = write_tokenizer(dir.path());
        let spec = test_support::engine_spec(
            dir.path(),
            ranks,
            test_support::model_spec(MODEL, &tokenizer),
        );
        let (client, cores) = test_support::connect(&spec, &vec![None; ranks]);
        let engine = Arc::new(AtomEngine {
            model: spec.model,
            client,
        });
        let config = RouterConfig::default();
        let pool = PreparePoolRuntime::new(config.resolved_prepare_pool().unwrap()).unwrap();
        let ctx = Arc::new(
            AppContextBuilder::from_config(config, 5)
                .await
                .unwrap()
                .prepare_pool(pool.handle())
                .atom_engine(Some(engine))
                .build()
                .unwrap(),
        );
        let router = AtomEngineRouter::new(&ctx).await.unwrap();
        Self {
            router,
            cores,
            _pool: pool,
            _dir: dir,
        }
    }
}

/// A Core that has answered, with the submission it received: its sequences
/// and data frames.
type Answered = (FakeCore, Vec<pb::RequestSpec>, Vec<Vec<u8>>);

/// Play every Core in `cores` from its own thread: take one submission and
/// answer each of its sequences with `answer`, then hand the Core and what it
/// received back.
fn answer_each(
    cores: Vec<FakeCore>,
    answer: impl Fn(&FakeCore, &pb::RequestSpec) + Send + Sync + Copy + 'static,
) -> Vec<thread::JoinHandle<Answered>> {
    cores
        .into_iter()
        .map(|core| {
            thread::spawn(move || {
                let (specs, data) = core.submitted().expect("a submission");
                for spec in &specs {
                    answer(&core, spec);
                }
                (core, specs, data)
            })
        })
        .collect()
}

fn hello_world(core: &FakeCore, spec: &pb::RequestSpec) {
    core.stream(&[(spec.seq_id, &[HELLO], None)]);
    core.stream(&[(spec.seq_id, &[WORLD], Some("eos"))]);
}

async fn json_body(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("{status}: not JSON ({e}): {bytes:?}"));
    (status, body)
}

fn chat(body: Value) -> crate::protocols::chat::ChatCompletionRequest {
    serde_json::from_value(body).unwrap()
}

fn user_hello(extra: Value) -> Value {
    let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

#[tokio::test(flavor = "multi_thread")]
async fn a_chat_is_prepared_submitted_and_rendered() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .router
        .route_chat(None, &chat(user_hello(json!({}))), None)
        .await;
    let (_, specs, data) = core.into_iter().next().unwrap().join().unwrap();

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    // `user: hello` `assistant:`, rendered by the template, as the prompt.
    assert_eq!(
        data[0],
        test_support::token_frame(&[3, 5, HELLO, 4, 5]),
        "the prompt frame"
    );
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    // No max_tokens: the deployment's default.
    let [spec] = &specs[..] else {
        panic!("{specs:?}")
    };
    let sampling = spec.sampling.as_ref().unwrap();
    assert_eq!(sampling.max_tokens, Some(16));
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_chat_ends_on_done() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .router
        .route_chat(None, &chat(user_hello(json!({"stream": true}))), None)
        .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    core.into_iter().for_each(|c| drop(c.join().unwrap()));

    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let events: Vec<&str> = text
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"), "{text}");
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "hello world");
    assert!(
        chunks
            .iter()
            .any(|c| c["choices"][0]["finish_reason"] == "stop"),
        "{text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn siblings_run_on_both_ranks_and_count_the_prompt_once() {
    let mut served = Served::start(2).await;
    let cores = answer_each(std::mem::take(&mut served.cores), hello_world);
    let request = chat(user_hello(json!({"n": 2, "temperature": 1.0})));
    let response = served.router.route_chat(None, &request, None).await;
    for core in cores {
        let (_, specs, _) = core.join().unwrap();
        assert_eq!(specs.len(), 1, "one sibling a rank");
    }

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert!(choices
        .iter()
        .all(|c| c["message"]["content"] == "hello world"));
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 4);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_rejection_answers_with_its_codes_status() {
    use pb::rejected_request::Code;
    for (code, status) in [
        (Code::Invalid, StatusCode::BAD_REQUEST),
        (Code::Overloaded, StatusCode::SERVICE_UNAVAILABLE),
        (Code::Internal, StatusCode::INTERNAL_SERVER_ERROR),
    ] {
        let mut served = Served::start(1).await;
        let core = thread::spawn({
            let core = served.cores.pop().unwrap();
            move || {
                let (specs, _) = core.submitted().expect("a submission");
                core.send(
                    pb::envelope::Body::Rejected(pb::Rejected {
                        items: vec![pb::RejectedRequest {
                            seq_id: specs[0].seq_id,
                            reason: "refused".into(),
                            code: code as i32,
                        }],
                    }),
                    vec![],
                );
                core
            }
        });
        let response = served
            .router
            .route_chat(None, &chat(user_hello(json!({}))), None)
            .await;
        drop(core.join().unwrap());
        let (got, body) = json_body(response).await;
        assert_eq!(got, status, "{code:?}: {body}");
        assert!(body.to_string().contains("refused"), "{body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_data_parallel_rank_is_where_the_request_runs() {
    let mut served = Served::start(2).await;
    let generate = |rank: i64| -> crate::protocols::generate::GenerateRequest {
        serde_json::from_value(json!({
            "model": MODEL,
            "input_ids": [HELLO],
            "data_parallel_rank": rank,
        }))
        .unwrap()
    };
    let rank1 = served.cores.pop().unwrap();
    let rank0 = served.cores.pop().unwrap();
    let serving = answer_each(vec![rank1], hello_world);
    let response = served.router.route_generate(None, &generate(1), None).await;
    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // One entry per sequence, as SGLang's `/generate` answers.
    assert_eq!(body[0]["text"], "hello world");
    serving.into_iter().for_each(|c| drop(c.join().unwrap()));
    assert!(
        FakeCore::recv_on(&rank0.input, std::time::Duration::from_millis(200)).is_none(),
        "rank 0 got nothing"
    );

    // Past the last rank, or negative: refused before any Core sees it.
    for rank in [2, -1] {
        let response = served
            .router
            .route_generate(None, &generate(rank), None)
            .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "rank {rank}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_completion_gets_its_text_back() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), |core, spec| {
        core.stream(&[(spec.seq_id, &[WORLD], Some("max_tokens"))]);
    });
    let request =
        serde_json::from_value(json!({"model": MODEL, "prompt": "hello", "max_tokens": 1}))
            .unwrap();
    let response = served.router.route_completion(None, &request, None).await;
    drop(core.into_iter().next().unwrap().join().unwrap());

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "world");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test(flavor = "multi_thread")]
async fn another_models_name_is_refused() {
    let served = Served::start(1).await;
    let request = chat(json!({
        "model": "other",
        "messages": [{"role": "user", "content": "hello"}],
    }));
    let (status, body) = json_body(served.router.route_chat(None, &request, None).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("model_not_found"), "{body}");
    assert!(FakeCore::recv_on(
        &served.cores[0].input,
        std::time::Duration::from_millis(200)
    )
    .is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn health_follows_the_cores() {
    let served = Served::start(2).await;
    let health = || async {
        served
            .router
            .health_generate(axum::extract::Request::default())
            .await
            .status()
    };
    assert_eq!(health().await, StatusCode::OK);
    served.cores[1].send(pb::envelope::Body::Stopped(pb::Stopped {}), vec![]);
    tokio::time::timeout(test_support::WAIT, served.router.engine.client.wait_lost())
        .await
        .expect("the loss is reported");
    assert_eq!(health().await, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_the_python_server_refuses_are_refused() {
    let served = Served::start(1).await;
    let tool =
        |name: &str| json!({"type": "function", "function": {"name": name, "parameters": {}}});
    let chats = [
        (
            json!({"stream_options": {"include_usage": true}}),
            "stream_options requires stream=true",
        ),
        (
            json!({"tools": [tool("get"), tool("put"), tool("get")]}),
            "duplicate tool name: get",
        ),
        (
            json!({"tools": [{"type": "retrieval", "function": {"name": "get", "parameters": {}}}]}),
            "tools[0].type must be 'function'",
        ),
    ];
    for (extra, refusal) in chats {
        let request = chat(user_hello(extra));
        let (status, body) = json_body(served.router.route_chat(None, &request, None).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.to_string().contains(refusal), "{body}");
    }
    let completion = serde_json::from_value(json!({
        "model": MODEL,
        "prompt": "hello",
        "stream_options": {"include_usage": true},
    }))
    .unwrap();
    let (status, body) = json_body(
        served
            .router
            .route_completion(None, &completion, None)
            .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.to_string()
            .contains("stream_options requires stream=true"),
        "{body}"
    );
    assert!(FakeCore::recv_on(
        &served.cores[0].input,
        std::time::Duration::from_millis(200)
    )
    .is_none());
}
