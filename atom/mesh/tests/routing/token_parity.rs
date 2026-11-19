//! Whether requests tokenize for routing as on ATOM, with each tokenizer backend.

use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use serde_json::{json, Value};
use serial_test::serial;
use wfaas::{StepExecutor, WorkflowContext, WorkflowInstanceId};

use super::*;
use crate::{
    config::{RouterConfig, DEFAULT_FASTOKENS_MAX_TOKENIZE_BYTES},
    core::{
        prepare_pool::PreparePoolRuntime,
        steps::{create_tokenizer_workflow_data, LoadTokenizerStep, TokenizerConfigRequest},
    },
    routers::prepare::{fast_tokenizer::FastTokenizer, thread_tokenizer},
};

const BACKENDS: [&str; 2] = ["huggingface", "fastokens"];
const MODEL: &str = "parity";
/// ATOM's tokens for a set of DeepSeek-V4.1-Flash requests, which
/// `scripts/atom_token_parity.py --huggingface` records.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/atom_tokens/deepseek_v41_flash.json"
);
/// The files of a checkpoint the router registers its tokenizer from, and
/// whether every checkpoint has them.
const TOKENIZER_FILES: [(&str, bool); 4] = [
    ("tokenizer.json", true),
    ("tokenizer_config.json", true),
    ("config.json", false),
    ("chat_template.jinja", false),
];

/// `model`'s tokenizer files at `revision`, from HuggingFace or `$HF_ENDPOINT`,
/// kept in the target directory for the next run. A file not every checkpoint
/// has may be missing; failing to download any other, each within 5 minutes,
/// fails the test.
async fn huggingface_files(model: &str, revision: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let dir = exe
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .join("huggingface")
        .join(model.replace('/', "--"))
        .join(revision);
    std::fs::create_dir_all(&dir).unwrap();
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(300))
        .build()
        .unwrap();
    for (file, every_checkpoint_has) in TOKENIZER_FILES {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        let url = format!("{endpoint}/{model}/resolve/{revision}/{file}");
        let response = match client.get(&url).send().await {
            Ok(response)
                if !every_checkpoint_has && response.status() == reqwest::StatusCode::NOT_FOUND =>
            {
                continue
            }
            response => response.and_then(reqwest::Response::error_for_status),
        };
        let bytes = match response {
            Ok(response) => response.bytes().await,
            Err(error) => Err(error),
        };
        let bytes = bytes.unwrap_or_else(|error| panic!("downloading {url}: {error}"));
        let partial = dir.join(format!("{file}.partial"));
        std::fs::write(&partial, &bytes).unwrap();
        std::fs::rename(&partial, &path).unwrap();
    }
    dir
}

/// Lines `start` to `start + count` of the large prompts' text, as the script
/// builds them.
fn synthetic_text(synthetic: &Value, start: usize, count: usize) -> String {
    let snippets = synthetic["snippets"].as_array().unwrap();
    (start..start + count)
        .map(|n| format!("{n} {}\n", snippets[n % snippets.len()].as_str().unwrap()))
        .collect()
}

/// A large request, which the script records as how it builds it: an agent's
/// history of reading `turns` files of `lines` lines each, or a completion of
/// all those lines.
fn synthetic_body(synthetic: &Value, kind: &str) -> Value {
    let turns = synthetic["turns"].as_u64().unwrap() as usize;
    let lines = synthetic["lines"].as_u64().unwrap() as usize;
    if kind == "completion" {
        return json!({"prompt": synthetic_text(synthetic, 0, turns * lines)});
    }
    let mut messages = synthetic["messages"].as_array().unwrap().clone();
    for turn in 0..turns {
        let path = format!("src/module_{turn}.rs");
        let arguments = format!(r#"{{"path": "{path}"}}"#);
        let mut call = json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "Read", "arguments": arguments},
            }],
        });
        if turn % 2 == 1 {
            call["reasoning_content"] = json!(format!("Step {turn}: read {path} next."));
        }
        messages.push(call);
        messages.push(json!({
            "role": "tool",
            "tool_call_id": "call_0",
            "content": synthetic_text(synthetic, turn * lines, lines),
        }));
    }
    messages.push(synthetic["question"].clone());
    json!({"tools": synthetic["tools"], "messages": messages})
}

/// The BLAKE2b digest of `tokens` as little-endian u32s, as the script takes it.
fn digest(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    blake2b_simd::Params::new()
        .hash_length(32)
        .hash(&bytes)
        .to_hex()
        .to_string()
}

/// What ATOM gives a request: its tokens, their count and digest, or a rejection.
enum Atom {
    Tokens(Vec<u32>),
    Digest(usize, String),
    Rejects,
}

impl Atom {
    fn of(request: &Value) -> Self {
        if let Some(tokens) = request.get("token_ids") {
            return Self::Tokens(serde_json::from_value(tokens.clone()).unwrap());
        }
        match request.get("blake2b") {
            Some(digest) => Self::Digest(
                request["tokens"].as_u64().unwrap() as usize,
                digest.as_str().unwrap().to_string(),
            ),
            None => Self::Rejects,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Tokens(tokens) => tokens.len(),
            Self::Digest(len, _) => *len,
            Self::Rejects => 0,
        }
    }

    /// Whether `got` is what ATOM gives, and how it compares.
    fn compare(&self, got: &Result<Vec<u32>, String>) -> (bool, String) {
        match (self, got) {
            (Self::Tokens(want), Ok(got)) if want == got => (true, "identical".to_string()),
            (Self::Tokens(want), Ok(got)) => {
                let at = want.iter().zip(got).take_while(|(a, b)| a == b).count();
                (false, format!("diverge at token {at}"))
            }
            (Self::Digest(len, want), Ok(got)) if *len == got.len() && *want == digest(got) => {
                (true, "identical".to_string())
            }
            (Self::Digest(len, _), Ok(got)) => (
                false,
                format!("{} tokens, not the {len} on ATOM", got.len()),
            ),
            (Self::Rejects, Err(_)) => (true, "rejected, as on ATOM".to_string()),
            (Self::Rejects, Ok(_)) => (false, "tokenized, ATOM rejects".to_string()),
            (_, Err(error)) => (false, format!("rejected: {error}")),
        }
    }
}

/// A router with the checkpoint `model`'s tokenizer registered as at startup,
/// encoding with `backend`, and tokenizing up to the same limit with both. Its
/// preparation does not time out, as a request's includes the timed encodes.
async fn router(model: &Path, backend: &str) -> (PreparePoolRuntime, Arc<AppContext>) {
    let mut config = RouterConfig::default();
    config.prepare_pool.max_tokenize_bytes = Some(DEFAULT_FASTOKENS_MAX_TOKENIZE_BYTES);
    config.prepare_pool.queue_timeout_ms = u64::from(u32::MAX);
    config.prepare_pool.prepare_timeout_ms = u64::from(u32::MAX);
    if backend == "fastokens" {
        config.fastokens.models = vec!["*".to_string()];
        config.fastokens.strict = true;
    }
    let pool = PreparePoolRuntime::new(config.resolved_prepare_pool().unwrap()).unwrap();
    let app = Arc::new(
        AppContext::from_config(config, 5, pool.handle())
            .await
            .unwrap(),
    );
    let request = TokenizerConfigRequest {
        id: TokenizerRegistry::generate_id(),
        name: MODEL.to_string(),
        source: model.to_str().unwrap().to_string(),
        chat_template_path: None,
        cache_config: None,
        fail_on_duplicate: true,
    };
    let mut workflow = WorkflowContext::new(
        WorkflowInstanceId::new(),
        create_tokenizer_workflow_data(request, app.clone()),
    );
    LoadTokenizerStep.execute(&mut workflow).await.unwrap();
    (pool, app)
}

/// The tokens HTTP routing gives `body` to match the engine's KV cache, and the
/// median of `runs` times encoding its prompt.
async fn tokenize(
    app: &AppContext,
    route: &str,
    body: &Value,
    runs: usize,
) -> (Result<Vec<u32>, String>, Option<Duration>) {
    let mut body = body.clone();
    body["model"] = Value::from(MODEL);
    let parsed = match ParsedInference::parse(route, &serde_json::to_vec(&body).unwrap()) {
        Ok(parsed) => parsed,
        Err(error) => return (Err(error.to_string()), None),
    };
    let routing = IngressRouting::new(app);
    let entry = app.tokenizer_registry.get_by_name(MODEL).unwrap();
    app.prepare_pool
        .submit(app.prepare_pool.prepare_deadline(), move |ctx| {
            let metadata = parsed.metadata();
            let tokens = routing
                .tokens(&parsed, &metadata, ctx, true)
                .map_err(|error| error.message);
            let prompt = match parsed.chat_as_sent() {
                Some((chat, sent)) => routing.render_chat(&entry, chat, sent),
                None => Ok(metadata.text.clone()),
            };
            let encode = prompt.ok().filter(|_| tokens.is_ok()).map(|prompt| {
                let encoder = thread_tokenizer::encoder(&entry);
                let mut times: Vec<_> = (0..runs)
                    .map(|_| {
                        let started = Instant::now();
                        encoder.encode(&prompt, true).unwrap();
                        started.elapsed()
                    })
                    .collect();
                times.sort();
                times[times.len() / 2]
            });
            (tokens, encode)
        })
        .await
        .unwrap()
        .wait()
        .await
        .unwrap()
}

/// Requests tokenize for routing as on ATOM, with HuggingFace and with
/// fastokens. The requests and ATOM's tokens for them are DeepSeek-V4.1-Flash's
/// in `tests/fixtures/atom_tokens/deepseek_v41_flash.json`, whose tokenizer
/// files the test downloads at the revision recorded, or those in
/// `$ATOM_TOKEN_PARITY`, which `scripts/atom_token_parity.py` records for a
/// checkpoint directory. Each request gets ATOM's tokens, or is rejected where
/// ATOM rejects it. The table lists how long encoding each prompt takes with
/// either backend: once for the fixture, the median of 5 runs otherwise (build
/// with `--release` for numbers worth reading). As it downloads, it runs only
/// when asked for (`--ignored`), and not while a test sets `$HF_ENDPOINT`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "downloads from HuggingFace: cargo test --lib test_requests_tokenize_as_on_atom -- --ignored"]
#[serial]
async fn test_requests_tokenize_as_on_atom() {
    let recorded = std::env::var("ATOM_TOKEN_PARITY").ok();
    let path = recorded.clone().unwrap_or_else(|| FIXTURE.to_string());
    let atom: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let model = atom["model"].as_str().unwrap();
    let checkpoint = match atom["revision"].as_str() {
        Some(revision) => huggingface_files(model, revision).await,
        None => PathBuf::from(model),
    };
    let runs = if recorded.is_some() { 5 } else { 1 };
    let requests: Vec<_> = atom["requests"]
        .as_array()
        .unwrap()
        .iter()
        .map(|request| {
            let body = match request["synthetic"].as_str() {
                Some(kind) => synthetic_body(&atom["synthetic"], kind),
                None => request["body"].clone(),
            };
            let name = request["name"].as_str().unwrap();
            (
                name,
                request["route"].as_str().unwrap(),
                body,
                Atom::of(request),
            )
        })
        .collect();

    let mut results = Vec::new();
    for backend in BACKENDS {
        let (_pool, app) = router(&checkpoint, backend).await;
        let entry = app.tokenizer_registry.get_by_name(MODEL).unwrap();
        assert_eq!(
            entry.tokenizer.as_any().is::<FastTokenizer>(),
            backend == "fastokens"
        );
        let mut tokenized = Vec::new();
        for (_, route, body, _) in &requests {
            tokenized.push(tokenize(&app, route, body, runs).await);
        }
        results.push(tokenized);
    }

    eprintln!("{model} {}", atom["revision"].as_str().unwrap_or(""));
    eprintln!(
        "  {:28} {:>9} {:>9}  {:24} {:24} {:>11} {:>11}",
        "request", "bytes", "tokens", BACKENDS[0], BACKENDS[1], "encode hf", "encode fast"
    );
    let micros = |time: Option<Duration>| {
        time.map_or_else(
            || "-".to_string(),
            |time| format!("{:.0} us", time.as_secs_f64() * 1e6),
        )
    };
    let (mut unlike_atom, mut unlike_huggingface) = (Vec::new(), Vec::new());
    let mut speedups = Vec::new();
    for (n, (name, _, body, atom)) in requests.iter().enumerate() {
        let (huggingface, fastokens) = (&results[0][n], &results[1][n]);
        let (hf_like_atom, hf_outcome) = atom.compare(&huggingface.0);
        let (fast_like_atom, fast_outcome) = atom.compare(&fastokens.0);
        eprintln!(
            "  {name:28} {:>9} {:>9}  {hf_outcome:24} {fast_outcome:24} {:>11} {:>11}",
            body.to_string().len(),
            atom.len(),
            micros(huggingface.1),
            micros(fastokens.1),
        );
        if !(hf_like_atom && fast_like_atom) {
            unlike_atom.push(*name);
        }
        if huggingface.0 != fastokens.0 {
            unlike_huggingface.push(*name);
        }
        if let (Some(hf), Some(fast)) = (huggingface.1, fastokens.1) {
            speedups.push(hf.as_secs_f64() / fast.as_secs_f64());
        }
    }
    speedups.sort_by(f64::total_cmp);
    if let (Some(low), Some(high)) = (speedups.first(), speedups.last()) {
        eprintln!(
            "  fastokens encodes {:.1}x to {:.1}x as fast (median {:.1}x)",
            low,
            high,
            speedups[speedups.len() / 2]
        );
    }
    assert!(
        unlike_huggingface.is_empty(),
        "{unlike_huggingface:?} encode otherwise with fastokens than with HuggingFace"
    );
    assert!(
        unlike_atom.is_empty(),
        "{unlike_atom:?} tokenize otherwise than on ATOM"
    );
}
