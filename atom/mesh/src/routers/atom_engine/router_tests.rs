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

    /// One Core behind the template whose generation prompt ends on `opener`,
    /// its replies read by `parser`.
    async fn reasoning_with(parser: &str, opener: &str) -> Self {
        let config = RouterConfig {
            reasoning_parser: Some(parser.into()),
            ..Default::default()
        };
        let template = TEMPLATE.replace("assistant:", &format!("assistant:{opener}"));
        Self::serve(1, &template, config).await
    }

    /// One Core behind a template whose generation prompt opens `<think>`,
    /// as GLM-5.x and Qwen3.5 do, read by the qwen3 reasoning parser.
    async fn thinking() -> Self {
        Self::reasoning_with("qwen3", " <think>\n").await
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

    /// `post`, every Core taking one submission and answering each of its
    /// sequences with `answer`: the response and the sequences submitted.
    async fn answered(
        &mut self,
        path: &str,
        body: Value,
        answer: impl Answer,
    ) -> (Response, Vec<wire::SubmitItem>) {
        let cores: Vec<_> = std::mem::take(&mut self.cores)
            .into_iter()
            .map(|core| {
                thread::spawn(move || {
                    let specs = core.submitted().expect("a submission");
                    specs.iter().for_each(|spec| answer(&core, spec));
                    (core, specs)
                })
            })
            .collect();
        let response = self.post(path, body).await;
        let mut submitted = Vec::new();
        for core in cores {
            let (core, specs) = core.join().unwrap();
            self.cores.push(core);
            submitted.extend(specs);
        }
        (response, submitted)
    }

    /// `answered`, its response read as JSON.
    async fn exchange(
        &mut self,
        path: &str,
        body: Value,
        answer: impl Answer,
    ) -> (StatusCode, Value, Vec<wire::SubmitItem>) {
        let (response, specs) = self.answered(path, body, answer).await;
        let (status, body) = json_body(response).await;
        (status, body, specs)
    }
}

/// How a fake Core answers a sequence it was submitted.
trait Answer: Fn(&FakeCore, &wire::SubmitItem) + Send + Sync + Copy + 'static {}
impl<F: Fn(&FakeCore, &wire::SubmitItem) + Send + Sync + Copy + 'static> Answer for F {}

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

/// A streamed response's chunks, which end on `[DONE]`.
async fn sse_chunks(response: Response) -> Vec<Value> {
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let events: Vec<&str> = text
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"), "{text}");
    events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect()
}

/// What the chunks' deltas say in `field`, joined.
fn joined(chunks: &[Value], field: &str) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"][field].as_str())
        .collect()
}

fn assert_idle(core: &FakeCore) {
    let submitted = FakeCore::recv_on(&core.input, std::time::Duration::from_millis(200));
    assert!(submitted.is_none(), "the Core got a submission");
}

fn user_hello(extra: Value) -> Value {
    let mut body = json!({"model": MODEL, "messages": [{"role": "user", "content": "hello"}]});
    body.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    body
}

const CHAT: &str = "/v1/chat/completions";

#[tokio::test(flavor = "multi_thread")]
async fn a_chat_is_prepared_submitted_and_rendered() {
    let mut served = Served::start(1).await;
    let (status, body, specs) = served
        .exchange(CHAT, user_hello(json!({})), hello_world)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    // `user: hello` `assistant:`, rendered by the template, as the prompt.
    let [spec] = &specs[..] else {
        panic!("{specs:?}")
    };
    assert_eq!(spec.prompt, test_support::token_bytes(&[3, 5, HELLO, 4, 5]));
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 2);
    // No max_tokens: the deployment's default.
    assert_eq!(spec.sampling.max_tokens, 16);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_chat_ends_on_done() {
    let mut served = Served::start(1).await;
    let body = user_hello(json!({"stream": true, "stream_options": {"include_usage": true}}));
    let (response, _) = served.answered(CHAT, body, hello_world).await;
    let chunks = sse_chunks(response).await;
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(joined(&chunks, "content"), "hello world");
    assert!(chunks
        .iter()
        .any(|c| c["choices"][0]["finish_reason"] == "stop"));
    // Its usage carries the cache hit, not ATOM's timing.
    let usage = chunks
        .iter()
        .find_map(|c| c.get("usage").filter(|u| !u.is_null()))
        .expect("a usage chunk");
    assert_eq!(usage["prompt_tokens"], 5, "{usage}");
    assert_eq!(
        usage["prompt_tokens_details"]["cached_tokens"], 3,
        "{usage}"
    );
    assert!(usage.get("ttft_s").is_none(), "{usage}");
}

#[tokio::test(flavor = "multi_thread")]
async fn siblings_run_on_both_ranks_and_count_the_prompt_once() {
    let mut served = Served::start(2).await;
    let body = user_hello(json!({"n": 2, "temperature": 1.0}));
    let (status, body, specs) = served.exchange(CHAT, body, hello_world).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let ranks: Vec<_> = specs.iter().map(|s| s.data_parallel_rank).collect();
    assert_eq!(ranks, [Some(0), Some(1)], "one sibling a rank");
    let choices = body["choices"].as_array().unwrap();
    assert_eq!(choices.len(), 2);
    assert!(choices
        .iter()
        .all(|c| c["message"]["content"] == "hello world"));
    // The cache hit and ATOM's timing too.
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
async fn a_rejection_answers_with_its_codes_status() {
    let mut served = Served::start(1).await;
    for (code, status) in [
        (reject::INVALID, StatusCode::BAD_REQUEST),
        (reject::OVERLOADED, StatusCode::SERVICE_UNAVAILABLE),
        (reject::INTERNAL, StatusCode::INTERNAL_SERVER_ERROR),
    ] {
        let refuse = move |core: &FakeCore, spec: &wire::SubmitItem| {
            let refused = wire::Completion {
                seq_id: spec.seq_id,
                leave_reason: "rejected: refused".into(),
                reject_code: code,
                ..Default::default()
            };
            let items = vec![refused];
            core.send(kind::ADD, &wire::Completions { items });
        };
        let (got, body, _) = served.exchange(CHAT, user_hello(json!({})), refuse).await;
        assert_eq!(got, status, "{code:?}: {body}");
        assert!(body.to_string().contains("refused"), "{body}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_completion_ignores_what_python_v1_does_not_declare() {
    let mut served = Served::start(1).await;
    let body = json!({
        "model": MODEL, "prompt": "hello",
        "max_tokens": 8, "max_completion_tokens": 4,
        "echo": true, "suffix": "world", "best_of": 3,
        "logit_bias": {"1": -100.0}, "stream_options": {"include_usage": true},
        "stop": ["world"],
    });
    let (status, body, specs) = served.exchange("/v1/completions", body, hello_world).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "text_completion");
    // No echoed prompt, no suffix: the completion alone, its matched stop
    // string kept, as the Python server leaves it.
    assert_eq!(body["choices"][0]["text"], "hello world");
    // `max_completion_tokens` over `max_tokens`, as `get_max_tokens` takes it.
    assert_eq!(specs[0].sampling.max_tokens, 4);
    // ATOM's timing, and no cache hit: completions do not report one.
    let usage = &body["usage"];
    assert!(usage["ttft_s"].as_f64().unwrap() >= 0.0, "{usage}");
    assert!(usage["latency_s"].as_f64().unwrap() >= usage["ttft_s"].as_f64().unwrap());
    assert!(usage.get("prompt_tokens_details").is_none(), "{usage}");
    assert!(usage.get("num_choices").is_none(), "{usage}");

    let out_of_tokens = |core: &FakeCore, spec: &wire::SubmitItem| {
        core.stream(&[(spec.seq_id, &[WORLD], Some("max_tokens"))]);
    };
    let body = json!({"model": MODEL, "prompt": "hello", "max_tokens": 1});
    let (_, body, _) = served
        .exchange("/v1/completions", body, out_of_tokens)
        .await;
    assert_eq!(body["choices"][0]["finish_reason"], "length", "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn another_models_name_is_refused() {
    let served = Served::start(1).await;
    let request = json!({
        "model": "other",
        "messages": [{"role": "user", "content": "hello"}],
    });
    let (status, body) = json_body(served.post(CHAT, request).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.to_string().contains("model_not_found"), "{body}");
    assert_idle(&served.cores[0]);
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
async fn every_generate_sibling_reports_its_prompt() {
    let mut served = Served::start(2).await;
    let body = json!({
        "model": MODEL, "input_ids": [HELLO, WORLD],
        "sampling_params": {"n": 2, "temperature": 1.0},
    });
    let (status, body, _) = served.exchange("/generate", body, hello_world).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let siblings = body.as_array().unwrap();
    assert_eq!(siblings.len(), 2, "{body}");
    for sibling in siblings {
        assert_eq!(sibling["meta_info"]["prompt_tokens"], 2, "{body}");
    }
}

/// What the Core cannot serve, and what the Python server refuses, is
/// refused before any Core sees it.
#[tokio::test(flavor = "multi_thread")]
async fn what_the_core_cannot_serve_is_refused_not_dropped() {
    let served = Served::start(1).await;
    let tool =
        |name: &str| json!({"type": "function", "function": {"name": name, "parameters": {}}});
    let retrieval = json!({"type": "retrieval", "function": {"name": "get", "parameters": {}}});
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
            CHAT,
            user_hello(json!({"kv_transfer_params": {"do_remote_decode": true}})),
            "kv_transfer_params",
        ),
        (
            CHAT,
            user_hello(json!({"stream_options": {"include_usage": true}})),
            "stream_options",
        ),
        (
            CHAT,
            user_hello(json!({"tools": [tool("get"), tool("put"), tool("get")]})),
            "duplicate tool name: get",
        ),
        (
            CHAT,
            user_hello(json!({"tools": [retrieval]})),
            "tools[0].type must be 'function'",
        ),
    ];
    for (path, request, refusal) in refusals {
        let (status, body) = json_body(served.post(path, request).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{path}: {body}");
        assert!(body.to_string().contains(refusal), "{path}: {body}");
    }
    assert_idle(&served.cores[0]);
}

/// The reply of a prompt that opened `<think>` starts inside reasoning: what
/// precedes `</think>` is reasoning.
#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_that_opens_reasoning_leaves_the_reply_inside_it() {
    let think_then_answer = |core: &FakeCore, spec: &wire::SubmitItem| {
        core.stream(&[(spec.seq_id, &[HELLO, THINK_END, WORLD], Some("eos"))]);
    };
    let mut served = Served::thinking().await;
    let body = user_hello(json!({"stream": true}));
    let (response, _) = served.answered(CHAT, body, think_then_answer).await;
    let chunks = sse_chunks(response).await;
    assert_eq!(joined(&chunks, "reasoning_content").trim(), "hello");
    assert_eq!(joined(&chunks, "content").trim(), "world");
}

/// A reply longer than the reasoning parser's 64 KiB buffer is split at its
/// closer, not refused with a 500: under a string stop, whose decoder releases
/// text in windows that can cut `</think>` in two (and a `<` ending it is
/// kept); under MiniMax's parser, whose own `<think>` pushes a reply just
/// under 64 KiB past it; and under Kimi's, whose markers have no `<` (a
/// `<think>` is text to it).
#[tokio::test(flavor = "multi_thread")]
async fn a_long_reply_is_split_at_its_closer_whatever_the_stop_decoder_releases() {
    const MINIMAX_WORDS: usize = 10_920;
    type Row = (
        &'static str,
        &'static str,
        Value,
        &'static [i32],
        &'static str,
        usize,
    );
    let rows: [Row; 3] = [
        (
            "qwen3",
            " <think>\n",
            json!({"stop": ["Question:"]}),
            &[THINK_END, WORLD, LT],
            "world <",
            20_000,
        ),
        // 10920 words and their spaces, then ` </think> world`: 65534 bytes.
        (
            "minimax",
            "",
            json!({}),
            &[THINK_END, WORLD],
            "world",
            MINIMAX_WORDS,
        ),
        (
            "kimi",
            " ◁think▷\n",
            json!({}),
            &[KIMI_END, 6, WORLD],
            "<think> world",
            20_000,
        ),
    ];
    fn reply(words: usize, tail: &'static [i32]) -> impl Answer {
        move |core: &FakeCore, spec: &wire::SubmitItem| {
            let mut ids = vec![HELLO; words];
            ids.extend(tail);
            core.stream(&[(spec.seq_id, &ids, Some("eos"))]);
        }
    }
    for (parser, opener, mut extra, tail, content, words) in rows {
        extra["max_tokens"] = json!(words + 3);
        let mut served = Served::reasoning_with(parser, opener).await;
        let (status, body, _) = served
            .exchange(CHAT, user_hello(extra), reply(words, tail))
            .await;
        assert_eq!(status, StatusCode::OK, "{parser}: {body}");
        let message = &body["choices"][0]["message"];
        assert_eq!(message["content"], content, "{parser}");
        let reasoning = message["reasoning_content"].as_str().unwrap();
        assert_eq!(reasoning.split_whitespace().count(), words, "{parser}");
        let bytes = reasoning.len();
        if words == MINIMAX_WORDS {
            let reply = bytes + " </think> world".len();
            assert!((65_530..=65_536).contains(&reply), "{reply}");
        } else {
            assert!(bytes > 65_536, "{parser}: {bytes}");
        }
    }
}

/// What only the raw body says, read off it on the way in: `thinking` is
/// checked, and `data_parallel_rank` picks the Core of a chat or a generate.
#[tokio::test(flavor = "multi_thread")]
async fn a_chats_thinking_and_rank_are_read_off_its_body() {
    let mut served = Served::start(2).await;
    let rank0 = served.cores.remove(0);
    let ranked = [
        (
            CHAT,
            user_hello(json!({"data_parallel_rank": 1, "thinking": {"type": "enabled"}})),
        ),
        (
            "/generate",
            json!({"model": MODEL, "input_ids": [HELLO], "data_parallel_rank": 1}),
        ),
    ];
    for (path, body) in ranked {
        let (status, body, _) = served.exchange(path, body, hello_world).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        // A generate answers with one entry per sequence, as SGLang's does.
        let reply = if path == CHAT {
            &body["choices"][0]["message"]["content"]
        } else {
            &body[0]["text"]
        };
        assert_eq!(reply, "hello world", "{path}");
    }
    assert_idle(&rank0);

    // Past the last rank, negative or not a number; a thinking not an object.
    let refused = [
        json!({"data_parallel_rank": 2}),
        json!({"data_parallel_rank": -1}),
        json!({"data_parallel_rank": "1"}),
        json!({"thinking": "on"}),
    ];
    for extra in refused {
        let (status, body) = json_body(served.post(CHAT, user_hello(extra.clone())).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{extra}: {body}");
    }
    let generate = json!({"model": MODEL, "input_ids": [HELLO], "data_parallel_rank": -1});
    let response = served.post("/generate", generate).await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_idle(&rank0);
    assert_idle(&served.cores[0]);
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_acceptance_text_stands_in_for_every_token() {
    let mut served = Served::serve_model(1, TEMPLATE, RouterConfig::default(), |model| {
        model.synthetic_token_text = Some("synthetic ".into())
    })
    .await;
    let (status, body, _) = served
        .exchange(CHAT, user_hello(json!({})), hello_world)
        .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "synthetic synthetic "
    );
    assert_eq!(body["usage"]["completion_tokens"], 2);
}

/// A chat's stop strings become the token sequences the Core matches, in
/// order, then its stop token ids, one each; a string with no tokens is
/// dropped. A matched stop string stays in the reply, as the Python server
/// leaves it (its engine stops after the stop sequence).
#[tokio::test(flavor = "multi_thread")]
async fn stop_strings_and_ids_reach_the_core_as_token_sequences() {
    let mut served = Served::start(1).await;
    let body = user_hello(json!({
        "stop": ["world", "hello world", " "],
        "stop_token_ids": [THINK_END, LT],
    }));
    let (status, body, specs) = served.exchange(CHAT, body, hello_world).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        specs[0].stop_token_sequences,
        [vec![WORLD], vec![HELLO, WORLD], vec![THINK_END], vec![LT]]
    );

    let body = user_hello(json!({"stop": ["world"]}));
    let (_, body, _) = served.exchange(CHAT, body, hello_world).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hello world");
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

/// What a chat's template is rendered with: a history tool call's arguments
/// as the spec says (a string where it preserves them, parsed otherwise), and
/// the kwargs as api_server merges them (the request's over the deployment's
/// defaults, the reasoning switch over both, `response_format: text` passed
/// on).
#[tokio::test(flavor = "multi_thread")]
async fn template_kwargs_merge_as_the_python_server_merges_them() {
    // `hello` where the template got the arguments as a string, `world` where
    // it got them parsed.
    let arguments = "{% for m in messages %}{% if m.tool_calls %}\
                    {% if m.tool_calls[0].function.arguments is string %}hello {% else %}world {% endif %}\
                    {% endif %}{% endfor %}assistant:";
    // `{{ mode }}`, then `hello` if reasoning is on, `user` if a text
    // response format reached the template.
    let kwargs = "{{ mode }} {% if enable_thinking %}hello {% endif %}\
                  {% if response_format is defined %}user {% endif %}assistant:";
    let cases = [
        (arguments, true, after_a_tool_call(), vec![HELLO, 4, 5]),
        (arguments, false, after_a_tool_call(), vec![WORLD, 4, 5]),
        (kwargs, false, user_hello(json!({})), vec![WORLD, 4, 5]),
        (
            kwargs,
            false,
            user_hello(json!({"chat_template_kwargs": {"mode": "hello"}})),
            vec![HELLO, 4, 5],
        ),
        (
            kwargs,
            false,
            user_hello(json!({"thinking": {"type": "enabled"}})),
            vec![WORLD, HELLO, 4, 5],
        ),
        (
            kwargs,
            false,
            user_hello(json!({"chat_template_kwargs": {"enable_thinking": true},
                              "thinking": {"type": "disabled"}})),
            vec![WORLD, 4, 5],
        ),
        (
            kwargs,
            false,
            user_hello(json!({"response_format": {"type": "text"}})),
            vec![WORLD, 3, 4, 5],
        ),
    ];
    for (template, preserve, body, prompt) in cases {
        let mut served = Served::serve_model(1, template, RouterConfig::default(), |model| {
            model.preserve_tool_call_arguments = preserve;
            model.default_chat_template_kwargs =
                json!({"mode": "world"}).as_object().unwrap().clone();
            model.reasoning_toggle = Some(crate::atom_engine::spec::ReasoningToggle {
                name: "enable_thinking".into(),
                off: json!(false),
                on: json!(true),
            });
        })
        .await;
        let what = body.to_string();
        let (status, body, specs) = served.exchange(CHAT, body, hello_world).await;
        assert_eq!(status, StatusCode::OK, "{what}: {body}");
        assert_eq!(
            specs[0].prompt,
            test_support::token_bytes(&prompt),
            "{what}"
        );
    }
}
