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
        spec::ModelSpec,
        test_support::{self, FakeCore},
        wire::{self, kind, reject},
        AtomEngine,
    },
    config::RouterConfig,
    core::prepare_pool::PreparePoolRuntime,
    routers::RouterTrait,
};

const MODEL: &str = "tiny";
const HELLO: i32 = 1;
const WORLD: i32 = 2;
const THINK_END: i32 = 7;
const LT: i32 = 8;
const KIMI_OPEN: i32 = 9;
const KIMI_END: i32 = 10;

/// The chat template: `role: content` lines, ending on `assistant:`.
const TEMPLATE: &str = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
                        {% if add_generation_prompt %}assistant:{% endif %}";

/// A word-level tokenizer with `chat_template`.
fn write_tokenizer(dir: &std::path::Path, chat_template: &str) -> String {
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 6, "content": "<think>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": false},
            {"id": THINK_END, "content": "</think>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": false},
            {"id": KIMI_OPEN, "content": "◁think▷", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": false},
            {"id": KIMI_END, "content": "◁/think▷", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": false},
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": {
                "<unk>": 0, "hello": HELLO, "world": WORLD, "user": 3, "assistant": 4, ":": 5,
                "<think>": 6, "</think>": THINK_END, "<": LT,
                "◁think▷": KIMI_OPEN, "◁/think▷": KIMI_END,
            },
            "unk_token": "<unk>",
        },
    });
    let config = json!({ "chat_template": chat_template });
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
        Self::serve(ranks, TEMPLATE, RouterConfig::default()).await
    }

    /// One Core behind a template whose generation prompt opens `<think>`,
    /// as GLM-5.x and Qwen3.5 do, read by the qwen3 reasoning parser.
    /// One Core behind the plain template, its replies read by `parser`.
    async fn reasoning_with(parser: &str) -> Self {
        let config = RouterConfig {
            reasoning_parser: Some(parser.into()),
            ..Default::default()
        };
        Self::serve(1, TEMPLATE, config).await
    }

    async fn thinking() -> Self {
        let template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
                        {% if add_generation_prompt %}assistant: <think>\n{% endif %}";
        let config = RouterConfig {
            reasoning_parser: Some("qwen3".into()),
            ..Default::default()
        };
        Self::serve(1, template, config).await
    }

    async fn serve(ranks: usize, chat_template: &str, config: RouterConfig) -> Self {
        Self::serve_model(ranks, chat_template, config, |_| {}).await
    }

    /// `serve`, with the spec's model as `edit` leaves it.
    async fn serve_model(
        ranks: usize,
        chat_template: &str,
        config: RouterConfig,
        edit: impl FnOnce(&mut ModelSpec),
    ) -> Self {
        let dir = TempDir::new().unwrap();
        let tokenizer = write_tokenizer(dir.path(), chat_template);
        let mut model = test_support::model_spec(MODEL, &tokenizer);
        edit(&mut model);
        let spec = test_support::engine_spec(dir.path(), ranks, model);
        let (client, cores) = test_support::connect(&spec, &vec![None; ranks]);
        let engine = Arc::new(AtomEngine {
            model: spec.model,
            client,
        });
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

impl Served {
    /// `body` POSTed to `path` the way the server takes it: parsed by the
    /// ingress, then `route_inference`, which reads the raw body's extras.
    async fn post(&self, path: &str, body: Value) -> Response {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        match crate::routers::ingress::InferenceEnvelope::parse(
            path.parse().unwrap(),
            headers,
            body.to_string().into(),
            &self.router.ctx,
        )
        .await
        {
            Ok(request) => self.router.route_inference(request, &self.router.ctx).await,
            // Refused on the way in, as the server answers it.
            Err(e) => e.response(path),
        }
    }
}

/// A Core that has answered, with the sequences it was submitted.
type Answered = (FakeCore, Vec<wire::SubmitItem>);

/// Play every Core in `cores` from its own thread: take one submission and
/// answer each of its sequences with `answer`, then hand the Core and what it
/// received back.
fn answer_each(
    cores: Vec<FakeCore>,
    answer: impl Fn(&FakeCore, &wire::SubmitItem) + Send + Sync + Copy + 'static,
) -> Vec<thread::JoinHandle<Answered>> {
    cores
        .into_iter()
        .map(|core| {
            thread::spawn(move || {
                let specs = core.submitted().expect("a submission");
                for spec in &specs {
                    answer(&core, spec);
                }
                (core, specs)
            })
        })
        .collect()
}

fn hello_world(core: &FakeCore, spec: &wire::SubmitItem) {
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
        .post("/v1/chat/completions", user_hello(json!({})))
        .await;
    let (_, specs) = core.into_iter().next().unwrap().join().unwrap();

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    // `user: hello` `assistant:`, rendered by the template, as the prompt.
    assert_eq!(
        specs[0].prompt,
        test_support::token_bytes(&[3, 5, HELLO, 4, 5]),
        "the prompt"
    );
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    // No max_tokens: the deployment's default.
    let [spec] = &specs[..] else {
        panic!("{specs:?}")
    };
    assert_eq!(spec.sampling.max_tokens, 16);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_chat_ends_on_done() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .post("/v1/chat/completions", user_hello(json!({"stream": true})))
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
    let response = served
        .post(
            "/v1/chat/completions",
            user_hello(json!({"n": 2, "temperature": 1.0})),
        )
        .await;
    for core in cores {
        let (_, specs) = core.join().unwrap();
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
    for (code, status) in [
        (reject::INVALID, StatusCode::BAD_REQUEST),
        (reject::OVERLOADED, StatusCode::SERVICE_UNAVAILABLE),
        (reject::INTERNAL, StatusCode::INTERNAL_SERVER_ERROR),
    ] {
        let mut served = Served::start(1).await;
        let core = thread::spawn({
            let core = served.cores.pop().unwrap();
            move || {
                let specs = core.submitted().expect("a submission");
                let refused = wire::Completion {
                    seq_id: specs[0].seq_id,
                    leave_reason: "rejected: refused".into(),
                    reject_code: code,
                    ..Default::default()
                };
                core.send(
                    kind::ADD,
                    &wire::Completions {
                        items: vec![refused],
                    },
                );
                core
            }
        });
        let response = served
            .post("/v1/chat/completions", user_hello(json!({})))
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
    let generate = |rank: i64| {
        json!({
            "model": MODEL,
            "input_ids": [HELLO],
            "data_parallel_rank": rank,
        })
    };
    let rank1 = served.cores.pop().unwrap();
    let rank0 = served.cores.pop().unwrap();
    let serving = answer_each(vec![rank1], hello_world);
    let response = served.post("/generate", generate(1)).await;
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
        let response = served.post("/generate", generate(rank)).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "rank {rank}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_completion_gets_its_text_back() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), |core, spec| {
        core.stream(&[(spec.seq_id, &[WORLD], Some("max_tokens"))]);
    });
    let response = served
        .post(
            "/v1/completions",
            json!({"model": MODEL, "prompt": "hello", "max_tokens": 1}),
        )
        .await;
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
    let request = json!({
        "model": "other",
        "messages": [{"role": "user", "content": "hello"}],
    });
    let (status, body) = json_body(served.post("/v1/chat/completions", request).await).await;
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
    served.cores[1].send(kind::SHUTDOWN, &wire::Empty {});
    tokio::time::timeout(test_support::WAIT, served.router.engine.client.wait_lost())
        .await
        .expect("the loss is reported");
    assert_eq!(health().await, StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_completion_ignores_what_python_v1_does_not_declare() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .post(
            "/v1/completions",
            json!({
                "model": MODEL, "prompt": "hello",
                "max_tokens": 8, "max_completion_tokens": 4,
                "echo": true, "suffix": "world", "best_of": 3,
                "logit_bias": {"1": -100.0}, "stream_options": {"include_usage": true},
            }),
        )
        .await;
    let (_, specs) = core.into_iter().next().unwrap().join().unwrap();

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // No echoed prompt, no suffix: the completion alone.
    assert_eq!(body["choices"][0]["text"], "hello world");
    // `max_completion_tokens` over `max_tokens`, as `get_max_tokens` takes it.
    assert_eq!(specs[0].sampling.max_tokens, 4);
    // ATOM's timing, and no cache hit: completions do not report one.
    let usage = &body["usage"];
    assert!(usage["ttft_s"].as_f64().unwrap() >= 0.0, "{usage}");
    assert!(usage["latency_s"].as_f64().unwrap() >= usage["ttft_s"].as_f64().unwrap());
    assert!(usage.get("prompt_tokens_details").is_none(), "{usage}");
    assert!(usage.get("num_choices").is_none(), "{usage}");
}

#[tokio::test(flavor = "multi_thread")]
async fn chat_usage_carries_the_cache_hit_and_timing() {
    let mut served = Served::start(2).await;
    let cores = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .post(
            "/v1/chat/completions",
            user_hello(json!({"n": 2, "temperature": 1.0})),
        )
        .await;
    cores.into_iter().for_each(|c| drop(c.join().unwrap()));

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let usage = &body["usage"];
    assert_eq!(usage["prompt_tokens"], 5, "{usage}");
    assert_eq!(usage["completion_tokens"], 4, "{usage}");
    assert_eq!(
        usage["prompt_tokens_details"]["cached_tokens"], 3,
        "{usage}"
    );
    assert_eq!(usage["num_choices"], 2, "{usage}");
    for field in ["ttft_s", "tpot_s", "latency_s"] {
        assert!(usage[field].as_f64().is_some(), "{field}: {usage}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_chats_usage_carries_the_cache_hit() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .post(
            "/v1/chat/completions",
            user_hello(json!({"stream": true, "stream_options": {"include_usage": true}})),
        )
        .await;
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    core.into_iter().for_each(|c| drop(c.join().unwrap()));

    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let usage = text
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .filter_map(|e| serde_json::from_str::<Value>(e).ok())
        .find_map(|c| c.get("usage").filter(|u| !u.is_null()).cloned())
        .unwrap_or_else(|| panic!("no usage chunk: {text}"));
    assert_eq!(usage["prompt_tokens"], 5, "{usage}");
    assert_eq!(
        usage["prompt_tokens_details"]["cached_tokens"], 3,
        "{usage}"
    );
    assert!(usage.get("ttft_s").is_none(), "{usage}");
}

#[tokio::test(flavor = "multi_thread")]
async fn every_generate_sibling_reports_its_prompt() {
    let mut served = Served::start(2).await;
    let cores = answer_each(std::mem::take(&mut served.cores), hello_world);
    let response = served
        .post(
            "/generate",
            json!({
                "model": MODEL, "input_ids": [HELLO, WORLD],
                "sampling_params": {"n": 2, "temperature": 1.0},
            }),
        )
        .await;
    cores.into_iter().for_each(|c| drop(c.join().unwrap()));

    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let siblings = body.as_array().unwrap();
    assert_eq!(siblings.len(), 2, "{body}");
    for sibling in siblings {
        assert_eq!(sibling["meta_info"]["prompt_tokens"], 2, "{body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn what_the_core_cannot_serve_is_refused_not_dropped() {
    let served = Served::start(1).await;
    let refusals = [
        (
            "/generate",
            json!({"model": MODEL, "input_ids": [HELLO], "return_logprob": true, "logprob_start_len": 0}),
            "input logprobs",
        ),
        (
            "/generate",
            json!({"model": MODEL, "input_ids": [HELLO], "return_logprob": true, "token_ids_logprob": [WORLD]}),
            "token_ids_logprob",
        ),
        (
            "/v1/chat/completions",
            user_hello(json!({"kv_transfer_params": {"do_remote_decode": true}})),
            "kv_transfer_params",
        ),
        (
            "/v1/completions",
            json!({"model": MODEL, "prompt": "hello", "prompt_token_ids": [HELLO]}),
            "prompt_token_ids",
        ),
        (
            "/v1/completions",
            json!({"model": MODEL, "prompt": "hello", "return_token_ids": true}),
            "return_token_ids",
        ),
    ];
    for (path, request, refusal) in refusals {
        let (status, body) = json_body(served.post(path, request).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert!(body.to_string().contains(refusal), "{path}: {body}");
    }
    assert!(FakeCore::recv_on(
        &served.cores[0].input,
        std::time::Duration::from_millis(200)
    )
    .is_none());
}

#[tokio::test(flavor = "multi_thread")]
async fn requests_the_python_server_refuses_are_refused() {
    let served = Served::start(1).await;
    let tool =
        |name: &str| json!({"type": "function", "function": {"name": name, "parameters": {}}});
    let chats = [
        (
            json!({"stream_options": {"include_usage": true}}),
            "stream_options",
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
        let response = served.post("/v1/chat/completions", user_hello(extra)).await;
        let (status, body) = json_body(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.to_string().contains(refusal), "{body}");
    }
    assert!(FakeCore::recv_on(
        &served.cores[0].input,
        std::time::Duration::from_millis(200)
    )
    .is_none());
}

/// The reply of a prompt that opened `<think>` starts inside reasoning: what
/// precedes `</think>` is reasoning, streamed or not.
#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_that_opens_reasoning_leaves_the_reply_inside_it() {
    fn think_then_answer(core: &FakeCore, spec: &wire::SubmitItem) {
        core.stream(&[(spec.seq_id, &[HELLO, THINK_END, WORLD], Some("eos"))]);
    }

    let mut served = Served::thinking().await;
    let core = answer_each(std::mem::take(&mut served.cores), think_then_answer);
    let response = served
        .post("/v1/chat/completions", user_hello(json!({})))
        .await;
    drop(core.into_iter().next().unwrap().join().unwrap());
    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let message = &body["choices"][0]["message"];
    assert_eq!(
        message["reasoning_content"].as_str().map(str::trim),
        Some("hello")
    );
    assert_eq!(message["content"], "world");

    let mut served = Served::thinking().await;
    let core = answer_each(std::mem::take(&mut served.cores), think_then_answer);
    let response = served
        .post("/v1/chat/completions", user_hello(json!({"stream": true})))
        .await;
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    drop(core.into_iter().next().unwrap().join().unwrap());
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let deltas: Vec<Value> = text
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .filter(|e| *e != "[DONE]")
        .map(|e| serde_json::from_str::<Value>(e).unwrap()["choices"][0]["delta"].clone())
        .collect();
    let joined =
        |field: &str| -> String { deltas.iter().filter_map(|d| d[field].as_str()).collect() };
    assert_eq!(joined("reasoning_content").trim(), "hello", "{text}");
    assert_eq!(joined("content").trim(), "world", "{text}");
}

/// A non-streamed reply longer than the reasoning parser's 64 KiB buffer is
/// split, not refused with a 500.
#[tokio::test(flavor = "multi_thread")]
async fn a_long_non_streamed_reply_is_split_into_reasoning_and_content() {
    const WORDS: usize = 20_000;
    fn think_long(core: &FakeCore, spec: &wire::SubmitItem) {
        let mut ids = vec![HELLO; WORDS];
        ids.extend([THINK_END, WORLD]);
        core.stream(&[(spec.seq_id, &ids, Some("eos"))]);
    }

    let mut served = Served::thinking().await;
    let core = answer_each(std::mem::take(&mut served.cores), think_long);
    let response = served
        .post(
            "/v1/chat/completions",
            user_hello(json!({"max_tokens": WORDS + 2})),
        )
        .await;
    drop(core.into_iter().next().unwrap().join().unwrap());
    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let message = &body["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap();
    assert!(reasoning.len() > 65536, "{}", reasoning.len());
    assert_eq!(reasoning.split_whitespace().count(), WORDS);
    assert_eq!(message["content"], "world");
}

/// `count` words, `</think>`, then `tail`: a reply past 64 KiB of text.
fn long_reasoning_then(
    count: usize,
    tail: &'static [i32],
) -> impl Fn(&FakeCore, &wire::SubmitItem) + Send + Sync + Copy + 'static {
    move |core, spec| {
        let mut ids = vec![HELLO; count];
        ids.push(THINK_END);
        ids.extend(tail);
        core.stream(&[(spec.seq_id, &ids, Some("eos"))]);
    }
}

/// The non-streamed chat `served` answers for `body`, after `answer`.
async fn answered_chat(
    mut served: Served,
    body: Value,
    answer: impl Fn(&FakeCore, &wire::SubmitItem) + Send + Sync + Copy + 'static,
) -> Value {
    let core = answer_each(std::mem::take(&mut served.cores), answer);
    let response = served.post("/v1/chat/completions", body).await;
    drop(core.into_iter().next().unwrap().join().unwrap());
    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    body["choices"][0]["message"].clone()
}

/// A string stop makes the stop decoder release text in windows that can
/// cut `</think>` in two; the long reply is still split at it, and a `<`
/// ending the reply is kept.
#[tokio::test(flavor = "multi_thread")]
async fn a_long_reply_is_split_at_its_closer_whatever_the_stop_decoder_releases() {
    const WORDS: usize = 20_000;
    let body = user_hello(json!({"max_tokens": WORDS + 3, "stop": ["Question:"]}));
    let answer = long_reasoning_then(WORDS, &[WORLD, LT]);
    let message = answered_chat(Served::thinking().await, body, answer).await;
    assert_eq!(message["content"], "world <");
    let reasoning = message["reasoning_content"].as_str().unwrap();
    assert_eq!(reasoning.split_whitespace().count(), WORDS);
}

/// MiniMax's parser puts its own `<think>` before the reply: a reply just
/// under 64 KiB overflows the one-shot parse by those 7 bytes and is parsed
/// whole without the cap instead of answering 500.
#[tokio::test(flavor = "multi_thread")]
async fn a_minimax_reply_its_opener_pushes_past_64_kib_is_split() {
    // 10920 words and their spaces, then ` </think> world`: 65534 bytes.
    const WORDS: usize = 10_920;
    let body = user_hello(json!({"max_tokens": WORDS + 2}));
    let answer = long_reasoning_then(WORDS, &[WORLD]);
    let message = answered_chat(Served::reasoning_with("minimax").await, body, answer).await;
    assert_eq!(message["content"], "world");
    let reasoning = message["reasoning_content"].as_str().unwrap();
    let reply = reasoning.len() + " </think> world".len();
    assert!((65_530..=65_536).contains(&reply), "{reply}");
}

/// Kimi's markers have no `<`: a long reply whose generation prompt opened
/// `◁think▷` is split at its `◁/think▷` like any other, and a `<think>` is
/// text to it.
#[tokio::test(flavor = "multi_thread")]
async fn a_long_kimi_reply_is_split_at_its_own_closer() {
    const WORDS: usize = 20_000;
    let template = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
                    {% if add_generation_prompt %}assistant: ◁think▷\n{% endif %}";
    let config = RouterConfig {
        reasoning_parser: Some("kimi".into()),
        ..Default::default()
    };
    fn answer(core: &FakeCore, spec: &wire::SubmitItem) {
        let mut ids = vec![HELLO; WORDS];
        ids.extend([KIMI_END, 6, WORLD]);
        core.stream(&[(spec.seq_id, &ids, Some("eos"))]);
    }
    let body = user_hello(json!({"max_tokens": WORDS + 3}));
    let message = answered_chat(Served::serve(1, template, config).await, body, answer).await;
    assert_eq!(message["content"], "<think> world");
    let reasoning = message["reasoning_content"].as_str().unwrap();
    assert!(reasoning.len() > 65536, "{}", reasoning.len());
    assert_eq!(reasoning.split_whitespace().count(), WORDS);
}

/// What only the raw body says, read off it on the way in: `thinking` is
/// checked, and `data_parallel_rank` picks the Core of a chat.
#[tokio::test(flavor = "multi_thread")]
async fn a_chats_thinking_and_rank_are_read_off_its_body() {
    let served = Served::start(1).await;
    let response = served
        .post(
            "/v1/chat/completions",
            user_hello(json!({"thinking": "on"})),
        )
        .await;
    let (status, body) = json_body(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body.to_string().contains("thinking must be an object"),
        "{body}"
    );

    let mut served = Served::start(2).await;
    let rank1 = served.cores.pop().unwrap();
    let rank0 = served.cores.pop().unwrap();
    let serving = answer_each(vec![rank1], hello_world);
    let body = user_hello(json!({
        "data_parallel_rank": 1,
        "thinking": {"type": "enabled"},
    }));
    let (status, body) = json_body(served.post("/v1/chat/completions", body).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    serving.into_iter().for_each(|c| drop(c.join().unwrap()));
    assert!(
        FakeCore::recv_on(&rank0.input, std::time::Duration::from_millis(200)).is_none(),
        "rank 0 got nothing"
    );
}

/// A history tool call and its result, after `user: hello`.
fn after_a_tool_call() -> Value {
    json!({"model": MODEL, "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
        }]},
        {"role": "tool", "tool_call_id": "call_0", "content": "hello"},
    ]})
}

#[tokio::test(flavor = "multi_thread")]
async fn history_tool_call_arguments_reach_the_template_as_the_spec_says() {
    // `hello` where the template got the arguments as a string, `world` where
    // it got them parsed.
    let template = "{% for m in messages %}{% if m.tool_calls %}\
                    {% if m.tool_calls[0].function.arguments is string %}hello {% else %}world {% endif %}\
                    {% endif %}{% endfor %}assistant:";
    for (preserve, read_as) in [(true, HELLO), (false, WORLD)] {
        let mut served = Served::serve_model(1, template, RouterConfig::default(), |model| {
            model.preserve_tool_call_arguments = preserve
        })
        .await;
        let core = answer_each(std::mem::take(&mut served.cores), hello_world);
        let (status, body) = json_body(
            served
                .post("/v1/chat/completions", after_a_tool_call())
                .await,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let (_, specs) = core.into_iter().next().unwrap().join().unwrap();
        assert_eq!(
            specs[0].prompt,
            test_support::token_bytes(&[read_as, 4, 5]),
            "preserve={preserve}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_acceptance_text_stands_in_for_every_token() {
    let mut served = Served::serve_model(1, TEMPLATE, RouterConfig::default(), |model| {
        model.synthetic_token_text = Some("synthetic ".into())
    })
    .await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let (status, body) = json_body(
        served
            .post("/v1/chat/completions", user_hello(json!({})))
            .await,
    )
    .await;
    core.into_iter().next().unwrap().join().unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "synthetic synthetic "
    );
    assert_eq!(body["usage"]["completion_tokens"], 2);
}

/// A chat's stop strings become the token sequences the Core matches, in
/// order, then its stop token ids, one each; a string with no tokens is
/// dropped.
#[tokio::test(flavor = "multi_thread")]
async fn stop_strings_and_ids_reach_the_core_as_token_sequences() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let body = user_hello(json!({
        "stop": ["world", "hello world", " "],
        "stop_token_ids": [THINK_END, LT],
    }));
    let (status, body) = json_body(served.post("/v1/chat/completions", body).await).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, specs) = core.into_iter().next().unwrap().join().unwrap();
    assert_eq!(
        specs[0].stop_token_sequences,
        [vec![WORLD], vec![HELLO, WORLD], vec![THINK_END], vec![LT]]
    );
}

/// The kwargs a chat is rendered with, as api_server merges them: the
/// request's over the deployment's defaults, the reasoning switch over both,
/// and `response_format: text` passed on.
#[tokio::test(flavor = "multi_thread")]
async fn template_kwargs_merge_as_the_python_server_merges_them() {
    // `{{ mode }}`, then `hello` if reasoning is on, `user` if a text
    // response format reached the template.
    let template = "{{ mode }} {% if enable_thinking %}hello {% endif %}\
                    {% if response_format is defined %}user {% endif %}assistant:";
    let cases = [
        (json!({}), vec![WORLD, 4, 5]),
        (
            json!({"chat_template_kwargs": {"mode": "hello"}}),
            vec![HELLO, 4, 5],
        ),
        (
            json!({"thinking": {"type": "enabled"}}),
            vec![WORLD, HELLO, 4, 5],
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true},
                   "thinking": {"type": "disabled"}}),
            vec![WORLD, 4, 5],
        ),
        (
            json!({"chat_template_kwargs": {"enable_thinking": true},
                   "reasoning_effort": "none"}),
            vec![WORLD, 4, 5],
        ),
        (
            json!({"response_format": {"type": "text"}}),
            vec![WORLD, 3, 4, 5],
        ),
    ];
    for (extra, prompt) in cases {
        let mut served = Served::serve_model(1, template, RouterConfig::default(), |model| {
            model.default_chat_template_kwargs =
                json!({"mode": "world"}).as_object().unwrap().clone();
            model.reasoning_toggle = Some(crate::atom_engine::spec::ReasoningToggle {
                name: "enable_thinking".into(),
                off: json!(false),
                on: json!(true),
            });
        })
        .await;
        let core = answer_each(std::mem::take(&mut served.cores), hello_world);
        let body = user_hello(extra.clone());
        let (status, body) = json_body(served.post("/v1/chat/completions", body).await).await;
        assert_eq!(status, StatusCode::OK, "{extra}: {body}");
        let (_, specs) = core.into_iter().next().unwrap().join().unwrap();
        assert_eq!(
            specs[0].prompt,
            test_support::token_bytes(&prompt),
            "{extra}"
        );
    }
}

/// A matched stop string stays in the reply, as the Python server leaves it
/// (its engine stops after the stop sequence): chat and completions alike.
#[tokio::test(flavor = "multi_thread")]
async fn a_matched_stop_string_stays_in_the_reply() {
    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let body = user_hello(json!({"stop": ["world"]}));
    let (status, body) = json_body(served.post("/v1/chat/completions", body).await).await;
    drop(core.into_iter().next().unwrap().join().unwrap());
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");

    let mut served = Served::start(1).await;
    let core = answer_each(std::mem::take(&mut served.cores), hello_world);
    let body = json!({"model": MODEL, "prompt": "hello", "stop": ["world"]});
    let (status, body) = json_body(served.post("/v1/completions", body).await).await;
    drop(core.into_iter().next().unwrap().join().unwrap());
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["text"], "hello world");
}
