//! `AtomBridgeRouter` over fake EngineCores (`atom_engine::test_support`): a
//! request goes through ingress, preparation, placement, admission, the
//! client and the Core's side of the protocol, and comes back rendered. The
//! tokenizer is a byte-level one written to a temporary directory.

use std::{sync::Arc, thread, time::Duration};

use axum::{body::to_bytes, http::StatusCode, response::Response};
use futures::StreamExt;
use serde_json::{json, Value};
use tempfile::TempDir;

use super::{
    super::{AtomBridgeRouter, AtomBridgeRuntime},
    token_source::SeqTokenSource,
};
use crate::{
    app_context::AppContext,
    atom_engine::{
        spec::{BalanceSpec, BalanceStrategy, ModelSpec},
        test_support::{self, FakeCore},
        wire::{self, kind, reject},
        AtomEngine, Submission,
    },
    config::RouterConfig,
    core::prepare_pool::PreparePoolRuntime,
    policies::{DpLoad, RankLoad},
    routers::{
        ingress::InferenceEnvelope, test_mocks::tokenizer::byte_level_tokenizer_dir,
        token_handle::token_chunk::TokenChunk, RouterTrait,
    },
    tokenizer::{factory, registry::TokenizerRegistry},
};

const MODEL: &str = "tiny";

/// `role: content` lines, ending on `assistant:`.
const TEMPLATE: &str = "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
                        {% if add_generation_prompt %}assistant:{% endif %}";

/// Byte-level token ids of `text`.
fn ids(text: &str) -> Vec<i32> {
    text.bytes().map(i32::from).collect()
}

fn balance(strategy: BalanceStrategy) -> BalanceSpec {
    BalanceSpec {
        strategy,
        req_equiv: 512,
        session_affinity: false,
    }
}

/// A router over `ranks` fake Cores, the Cores handed to the test.
struct Served {
    router: AtomBridgeRouter,
    app: Arc<AppContext>,
    /// A Core is out while a thread of the test plays it.
    cores: Vec<Option<FakeCore>>,
    _pool: PreparePoolRuntime,
    _dirs: (TempDir, TempDir),
}

impl Served {
    async fn start(ranks: usize, strategy: BalanceStrategy) -> Self {
        Self::serve(ranks, balance(strategy), |_| {}).await
    }

    async fn serve(ranks: usize, balance: BalanceSpec, edit: impl FnOnce(&mut ModelSpec)) -> Self {
        let tokenizer_dir =
            byte_level_tokenizer_dir(&[], false, json!({ "chat_template": TEMPLATE }));
        let tokenizer = tokenizer_dir
            .path()
            .join("tokenizer.json")
            .display()
            .to_string();
        let ipc = TempDir::new().unwrap();
        let mut model = test_support::model_spec(MODEL, &tokenizer);
        model.max_model_len = Some(4096);
        edit(&mut model);
        let mut spec = test_support::engine_spec(ipc.path(), ranks, model);
        spec.balance = balance;
        let (client, cores) = test_support::connect(&spec, &vec![None; ranks]);
        let engine = Arc::new(AtomEngine {
            model: spec.model,
            balance: spec.balance,
            client,
        });

        let config = RouterConfig::default();
        let pool = PreparePoolRuntime::new(config.resolved_prepare_pool().unwrap()).unwrap();
        let app = Arc::new(
            AppContext::from_config(config, 5, pool.handle())
                .await
                .unwrap(),
        );
        app.tokenizer_registry
            .load(&TokenizerRegistry::generate_id(), MODEL, &tokenizer, || {
                let source = tokenizer.clone();
                async move {
                    factory::create_tokenizer_async_with_chat_template(&source, None)
                        .await
                        .map_err(|e| e.to_string())
                }
            })
            .await
            .unwrap();
        let runtime = Arc::new(AtomBridgeRuntime::from_engine(engine));
        let router = AtomBridgeRouter::new(&app, runtime).unwrap();
        Self {
            router,
            app,
            cores: cores.into_iter().map(Some).collect(),
            _pool: pool,
            _dirs: (tokenizer_dir, ipc),
        }
    }

    /// `body` POSTed to `path` as the server takes it: parsed by the ingress,
    /// then `route_inference`. A request no Core answers fails the test
    /// rather than hang it.
    async fn post(&self, path: &str, body: Value) -> Response {
        let mut headers = http::HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        let served = async {
            match InferenceEnvelope::parse(
                path.parse().unwrap(),
                headers,
                body.to_string().into(),
                &self.app,
            )
            .await
            {
                Ok(request) => self.router.route_inference(request, &self.app).await,
                Err(e) => e.response(path),
            }
        };
        tokio::time::timeout(2 * test_support::WAIT, served)
            .await
            .expect("an answer")
    }

    /// Core `rank` takes the next submission and answers each of its
    /// sequences with `reply`, from a thread of its own; joined, it hands
    /// back the Core and what it was sent.
    fn answer(
        &mut self,
        rank: usize,
        reply: &'static str,
    ) -> thread::JoinHandle<(FakeCore, Vec<wire::SubmitItem>)> {
        let core = self.take(rank);
        thread::spawn(move || {
            let items = core.submitted().expect("a submission");
            for item in &items {
                core.stream(&[(item.seq_id, &ids(reply), Some("eos"))]);
            }
            (core, items)
        })
    }

    fn take(&mut self, rank: usize) -> FakeCore {
        self.cores[rank].take().expect("the Core is in")
    }

    fn put(
        &mut self,
        rank: usize,
        answered: thread::JoinHandle<(FakeCore, Vec<wire::SubmitItem>)>,
    ) -> Vec<wire::SubmitItem> {
        let (core, items) = answered.join().unwrap();
        self.cores[rank] = Some(core);
        items
    }
}

async fn json_of(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

fn generate(rank: Option<i32>) -> Value {
    let mut body = json!({
        "input_ids": [1, 2, 3],
        "sampling_params": {"max_new_tokens": 4},
    });
    if let Some(rank) = rank {
        body["data_parallel_rank"] = json!(rank);
    }
    body
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_chat_is_admitted_placed_on_a_core_and_rendered() {
    let mut served = Served::start(1, BalanceStrategy::RoundRobin).await;
    let core = served.answer(0, "hi");
    let body = json!({
        "model": "anything",
        "messages": [{"role": "user", "content": "yo"}],
        "max_tokens": 5,
        "stop": ["zz"],
    });
    let (status, reply) = json_of(served.post("/v1/chat/completions", body).await).await;
    let items = served.put(0, core);
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(reply["choices"][0]["message"]["content"], "hi");
    assert_eq!(reply["choices"][0]["finish_reason"], "stop");
    assert_eq!(reply["usage"]["completion_tokens"], 2);

    let [item] = &items[..] else {
        panic!("one sequence: {items:?}")
    };
    let prompt = wire::tokens(&item.prompt).unwrap();
    assert_eq!(
        prompt,
        ids("user: yo\nassistant:")
            .iter()
            .map(|&t| t as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(item.sampling.max_tokens, 5);
    assert!(!item.sampling.max_tokens_is_default);
    assert_eq!(item.stop_token_sequences, [ids("zz")]);
    assert_eq!(item.data_parallel_rank, Some(0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn requests_take_turns_over_the_ranks_unless_pinned() {
    let mut served = Served::start(2, BalanceStrategy::RoundRobin).await;
    let mut placed = Vec::new();
    for _ in 0..4 {
        let cores = [served.take(0), served.take(1)];
        let answering = thread::spawn(move || answer_whichever(cores, "a"));
        let (status, reply) = json_of(served.post("/generate", generate(None)).await).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let ([first, second], rank) = answering.join().unwrap();
        served.cores = vec![Some(first), Some(second)];
        placed.push(rank);
    }
    // They alternate, from wherever the policy starts.
    assert!(placed.windows(2).all(|w| w[0] != w[1]), "{placed:?}");
    for _ in 0..2 {
        let core = served.answer(1, "a");
        let (status, reply) = json_of(served.post("/generate", generate(Some(1))).await).await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        assert_eq!(served.put(1, core)[0].data_parallel_rank, Some(1));
    }
    let (status, reply) = json_of(served.post("/generate", generate(Some(2))).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert!(
        reply
            .to_string()
            .contains("data_parallel_rank must be in [0, 2)"),
        "{reply}"
    );
}

/// Whichever of `cores` takes the next submission answers each of its
/// sequences with `reply`: the Cores, and the rank that answered.
fn answer_whichever(cores: [FakeCore; 2], reply: &str) -> ([FakeCore; 2], usize) {
    let deadline = std::time::Instant::now() + test_support::WAIT;
    while std::time::Instant::now() < deadline {
        for (rank, core) in cores.iter().enumerate() {
            let Some((kind, body)) = FakeCore::recv_on(&core.input, Duration::from_millis(10))
            else {
                continue;
            };
            assert_eq!(kind, kind::ADD);
            let items = wire::body::<wire::Submit>(&body).unwrap().items;
            for item in &items {
                assert_eq!(item.data_parallel_rank, Some(rank as i64));
                core.stream(&[(item.seq_id, &ids(reply), Some("eos"))]);
            }
            return (cores, rank);
        }
    }
    panic!("no Core took the request");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn least_requests_places_beside_a_request_in_flight() {
    let mut served = Served::start(2, BalanceStrategy::LeastRequests).await;
    let (first_core, second_core) = (served.take(0), served.take(1));
    // The first request is held on rank 0 while the second is placed.
    let held = thread::spawn(move || {
        let items = first_core.submitted().expect("the first request on rank 0");
        (first_core, items)
    });
    let second = async {
        let (first_core, held) = tokio::task::spawn_blocking(move || held.join().unwrap())
            .await
            .unwrap();
        let second_core = thread::spawn(move || {
            let items = second_core
                .submitted()
                .expect("the second request on rank 1");
            second_core.stream(&[(items[0].seq_id, &ids("b"), Some("eos"))]);
        });
        let second = json_of(served.post("/generate", generate(None)).await).await;
        second_core.join().unwrap();
        first_core.stream(&[(held[0].seq_id, &ids("a"), Some("eos"))]);
        second
    };
    let first = async { json_of(served.post("/generate", generate(None)).await).await };
    let ((status, reply), (second_status, second_reply)) = tokio::join!(first, second);
    assert_eq!(status, StatusCode::OK, "{reply}");
    assert_eq!(second_status, StatusCode::OK, "{second_reply}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn what_admission_refuses_never_reaches_a_core() {
    let served = Served::serve(1, balance(BalanceStrategy::RoundRobin), |model| {
        model.max_output_tokens = Some(4);
    })
    .await;
    let body = json!({"messages": [{"role": "user", "content": "yo"}], "max_tokens": 8});
    let (status, reply) = json_of(served.post("/v1/chat/completions", body).await).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert!(
        reply.to_string().contains("max_tokens must be <= 4"),
        "{reply}"
    );
    let core = served.cores[0].as_ref().unwrap();
    assert!(FakeCore::recv_on(&core.input, Duration::from_millis(200)).is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_core_refusal_is_answered_with_its_status() {
    let mut served = Served::start(1, BalanceStrategy::RoundRobin).await;
    let core = served.take(0);
    let refusing = thread::spawn(move || {
        let items = core.submitted().expect("a submission");
        let refused = wire::Completion {
            seq_id: items[0].seq_id,
            leave_reason: "prompt too long for this rank".into(),
            reject_code: reject::INVALID,
            ..Default::default()
        };
        core.send(
            kind::ADD,
            &wire::Completions {
                items: vec![refused],
            },
        );
        (core, items)
    });
    let (status, reply) = json_of(served.post("/generate", generate(None)).await).await;
    served.put(0, refusing);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{reply}");
    assert!(
        reply.to_string().contains("prompt too long for this rank"),
        "{reply}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_request_holds_its_prompt_on_its_rank_until_its_first_output() {
    let ipc = TempDir::new().unwrap();
    let spec = test_support::engine_spec(ipc.path(), 1, test_support::model_spec(MODEL, "unused"));
    let (client, cores) = test_support::connect(&spec, &[None]);
    let load = DpLoad::new(1);
    let seqs = client
        .submit(Submission {
            request_id: "r",
            prompt: &[1, 2, 3],
            sampling: wire::Sampling::default(),
            stop_token_sequences: vec![],
            n: 1,
            session: None,
            rank: 0,
            stream: true,
        })
        .unwrap();
    let mut source = SeqTokenSource::new(seqs, "r".into(), load.charge(0, 3));
    let seq_id = cores[0].submitted().unwrap()[0].seq_id;
    let loaded = |requests, tokens| RankLoad { requests, tokens };
    assert_eq!(load.loads()[0], loaded(1, 3));

    cores[0].stream(&[(seq_id, &[7], None)]);
    assert!(matches!(
        source.next().await,
        Some(Ok(TokenChunk::Partial { .. }))
    ));
    assert_eq!(load.loads()[0], loaded(1, 0));
    cores[0].stream(&[(seq_id, &[8], Some("eos"))]);
    assert!(matches!(
        source.next().await,
        Some(Ok(TokenChunk::Partial { .. }))
    ));
    let Some(Ok(TokenChunk::Complete { token_ids, .. })) = source.next().await else {
        panic!("a Complete");
    };
    assert_eq!(token_ids, [7, 8]);
    drop(source);
    assert_eq!(load.loads()[0], RankLoad::default());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_leaves_aborts_its_sequence() {
    let mut served = Served::start(1, BalanceStrategy::RoundRobin).await;
    let core = served.take(0);
    let post = served.post("/generate", generate(None));
    assert!(tokio::time::timeout(Duration::from_millis(300), post)
        .await
        .is_err());
    let seq_id = core.submitted().expect("a submission")[0].seq_id;
    let (kind, body) = core.control_message();
    assert_eq!(kind, kind::ABORT);
    let abort: wire::Abort = test_support::decoded(&body);
    assert_eq!(abort.seq_ids, [seq_id]);
}

#[test]
fn the_balance_spec_names_the_policy() {
    let load = DpLoad::new(2);
    let named = |strategy, session_affinity| {
        let spec = BalanceSpec {
            session_affinity,
            ..balance(strategy)
        };
        super::policy_for(&spec, &load).name()
    };
    assert_eq!(named(BalanceStrategy::RoundRobin, false), "round_robin");
    assert_eq!(
        named(BalanceStrategy::LeastRequests, false),
        "least_requests"
    );
    assert_eq!(named(BalanceStrategy::LeastTokens, false), "least_tokens");
    // A session stays on its rank, whatever the strategy.
    assert_eq!(named(BalanceStrategy::LeastTokens, true), "dp_sticky");
}
