//! Prompt tokenization that reuses a conversation's earlier turn.
//!
//! An agentic conversation resends its whole history every turn, so encoding
//! its prompt takes time linear in the context (about 1.6 MB/s on one core for
//! GLM-5.2), and a prompt over `max_tokenize_bytes` is routed by its text.
//! Prompts are therefore kept with their ids and where each added token in them
//! ends. A later prompt that starts the same way reuses the ids up to the last
//! added token both share and encodes only the rest, which is what
//! `max_tokenize_bytes` then bounds. A prompt over the limit with no such turn
//! is still routed by its text, and encoded in the background so that the
//! conversation's next turn finds it.
//!
//! This is the conversation level of ATOM's `SegmentTokenizer`
//! (`atom/model_engine/segment_tokenizer.py`), with its conditions for a cut
//! right after an added token to be exact. The first reuses are also compared
//! with a plain encode, and a difference turns the cache off for good. The ids
//! only place requests, so an unchecked one can cost a cache miss on the
//! engine, never a wrong answer.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{sync_channel, SyncSender},
        Arc, LazyLock, OnceLock, Weak,
    },
};

use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::Value;
use tracing::{info, warn};

use crate::tokenizer::{
    huggingface::HuggingFaceTokenizer,
    registry::TokenizerEntry,
    traits::{Encoding, Tokenizer},
};

/// Leading bytes that bucket a stored prompt; shorter prompts are encoded whole.
const KEY_BYTES: usize = 16 * 1024;
/// Stored prompts per bucket. Concurrent conversations share a system prompt
/// and tools, so a bucket holds a deployment's conversations, and a lookup
/// compares the prompt with each of them up to where they diverge.
const BUCKET_ENTRIES: usize = 256;
/// Bytes compared per step when looking for where two prompts diverge.
const COMPARE_CHUNK: usize = 64 * 1024;
/// Reuses compared with a plain encode before the cache is trusted.
const VERIFY_FIRST: usize = 4;
/// Prompts waiting to be encoded in the background; more are dropped.
const FILL_QUEUE: usize = 4;
/// A background encode covers at most this many times the byte limit.
const FILL_LIMITS: usize = 16;

/// Why a prompt was not encoded.
#[derive(Debug)]
pub(crate) enum EncodeError {
    /// What no stored prompt covers is over the byte limit.
    TooLarge,
    Failed(anyhow::Error),
}

/// Where an added token ends: its byte offset in the prompt, and the ids up to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cut {
    bytes: usize,
    ids: usize,
}

/// A prompt and the ids of its text up to its last added token.
struct Stored {
    text: String,
    /// Without the special tokens `encode` wraps a prompt in.
    ids: Vec<u32>,
    /// One per added token in `text`, ascending; the last one ends `ids`.
    cuts: Vec<Cut>,
}

impl Stored {
    fn size(&self) -> usize {
        self.text.len() + self.ids.len() * size_of::<u32>() + self.cuts.len() * size_of::<Cut>()
    }

    fn end(&self) -> Cut {
        self.cuts.last().copied().unwrap_or_default()
    }
}

/// A stored prompt and its last cut within what it shares with a new one.
struct Found {
    id: u64,
    stored: Arc<Stored>,
    cut: usize,
}

impl Found {
    fn end(&self) -> Cut {
        self.stored.cuts[self.cut]
    }
}

/// Stored prompts by bucket, evicted least recently used first.
#[derive(Default)]
struct Index {
    entries: HashMap<u64, Entry>,
    buckets: HashMap<u64, Vec<u64>>,
    /// Last use, then entry id.
    recency: BTreeMap<u64, u64>,
    clock: u64,
    next_id: u64,
    held: usize,
    /// Reuses that matched a plain encode.
    verified: usize,
}

struct Entry {
    stored: Arc<Stored>,
    bucket: u64,
    used: u64,
}

impl Index {
    fn candidates(&self, bucket: u64) -> Vec<(u64, Arc<Stored>)> {
        self.buckets
            .get(&bucket)
            .into_iter()
            .flatten()
            .filter_map(|id| Some((*id, self.entries.get(id)?.stored.clone())))
            .collect()
    }

    fn touch(&mut self, id: u64) {
        if let Some(entry) = self.entries.get_mut(&id) {
            self.recency.remove(&entry.used);
            self.clock += 1;
            entry.used = self.clock;
            self.recency.insert(self.clock, id);
        }
    }

    fn insert(&mut self, bucket: u64, stored: Arc<Stored>, budget: usize) {
        let size = stored.size();
        if size > budget {
            return;
        }
        let id = self.next_id;
        self.next_id += 1;
        self.clock += 1;
        self.held += size;
        self.recency.insert(self.clock, id);
        self.entries.insert(
            id,
            Entry {
                stored,
                bucket,
                used: self.clock,
            },
        );
        let ids = self.buckets.entry(bucket).or_default();
        ids.push(id);
        if ids.len() > BUCKET_ENTRIES {
            let oldest = ids[0];
            self.remove(oldest);
        }
        while self.held > budget {
            let Some((_, id)) = self.recency.pop_first() else {
                break;
            };
            self.remove(id);
        }
    }

    fn remove(&mut self, id: u64) {
        let Some(entry) = self.entries.remove(&id) else {
            return;
        };
        self.recency.remove(&entry.used);
        self.held -= entry.stored.size();
        if let Some(ids) = self.buckets.get_mut(&entry.bucket) {
            ids.retain(|other| *other != id);
            if ids.is_empty() {
                self.buckets.remove(&entry.bucket);
            }
        }
    }
}

/// A tokenizer's stored prompts.
pub(crate) struct PrefixTokens {
    /// Added tokens by id; where one ends in a prompt is a cut.
    added: HashMap<u32, String>,
    /// What `encode` wraps a prompt in with its special tokens.
    head: Vec<u32>,
    tail: Vec<u32>,
    budget: usize,
    on: AtomicBool,
    index: Mutex<Index>,
}

impl PrefixTokens {
    fn for_entry(entry: &TokenizerEntry, budget: usize) -> Result<Self, String> {
        if !entry.tokenizer.as_any().is::<HuggingFaceTokenizer>() {
            return Err("not a Hugging Face tokenizer".into());
        }
        let source = Path::new(&entry.source);
        let path = if source.is_dir() {
            source.join("tokenizer.json")
        } else {
            source.to_path_buf()
        };
        let description =
            std::fs::read(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
        Self::new(entry.tokenizer.as_ref(), &description, budget)
    }

    /// A cache for `tokenizer`, described by its `tokenizer.json`, or why
    /// reusing ids would not be exact for it.
    fn new(tokenizer: &dyn Tokenizer, description: &[u8], budget: usize) -> Result<Self, String> {
        let description: Description =
            serde_json::from_slice(description).map_err(|e| e.to_string())?;
        description.check()?;
        let encode = |text: &str, special: bool| {
            tokenizer
                .encode(text, special)
                .map(|e| e.token_ids().to_vec())
                .map_err(|e| e.to_string())
        };
        let (full, bare) = (encode("a", true)?, encode("a", false)?);
        let start = (0..=full.len().saturating_sub(bare.len()))
            .find(|&at| full[at..].starts_with(&bare))
            .ok_or("encoding adds more than special tokens around a prompt")?;
        let cache = Self {
            added: description
                .added_tokens
                .into_iter()
                .map(|t| (t.id, t.content))
                .collect(),
            head: full[..start].to_vec(),
            tail: full[start + bare.len()..].to_vec(),
            budget,
            on: AtomicBool::new(true),
            index: Mutex::default(),
        };
        cache.probe(tokenizer)?;
        Ok(cache)
    }

    /// A few placements of added tokens, cut after each token and encoded in
    /// two parts, give the ids of encoding them whole.
    fn probe(&self, tokenizer: &dyn Tokenizer) -> Result<(), String> {
        let mut tokens: Vec<(&u32, &String)> = self.added.iter().collect();
        tokens.sort_unstable();
        let picks: HashSet<&str> = [
            tokens.first(),
            tokens.last(),
            tokens.iter().min_by_key(|(_, t)| t.len()),
            tokens.iter().max_by_key(|(_, t)| t.len()),
        ]
        .into_iter()
        .flatten()
        .map(|(_, t)| t.as_str())
        .collect();
        let encode = |text: &str| tokenizer.encode(text, false).map_err(|e| e.to_string());
        for token in picks {
            for probe in [
                format!("a{token}a b"),
                format!("{token} x\n{token}"),
                format!(" y {token}\t1 2"),
                // A normalizer could compose across a cut.
                format!("e\u{301}{token}\u{301}e"),
            ] {
                let whole = encode(&probe)?;
                let ids = whole.token_ids();
                for cut in self.cuts(&probe, &whole, Cut::default()) {
                    let before = encode(&probe[..cut.bytes])?;
                    let after = encode(&probe[cut.bytes..])?;
                    if ids[..cut.ids] != *before.token_ids() || ids[cut.ids..] != *after.token_ids()
                    {
                        return Err(format!("cutting after {token:?} changes the ids"));
                    }
                }
            }
        }
        Ok(())
    }

    /// `prompt`'s ids as `encoder.encode(prompt, add_special_tokens)` gives
    /// them, encoding only what no stored prompt covers. That part must be at
    /// most `limit` bytes; a larger one is encoded with `background` off the
    /// request path, for the conversation's next turn.
    pub(crate) fn encode(
        self: &Arc<Self>,
        prompt: String,
        add_special_tokens: bool,
        encoder: &dyn Tokenizer,
        background: &Arc<dyn Tokenizer>,
        limit: usize,
    ) -> Result<Vec<u32>, EncodeError> {
        if prompt.len() <= KEY_BYTES || !self.on.load(Ordering::Relaxed) {
            if prompt.len() > limit {
                return Err(EncodeError::TooLarge);
            }
            return encoder
                .encode(&prompt, add_special_tokens)
                .map(|e| e.token_ids().to_vec())
                .map_err(EncodeError::Failed);
        }
        let found = self.find(&prompt);
        let rest = prompt.len() - found.as_ref().map_or(0, |f| f.end().bytes);
        if rest > limit {
            if rest <= limit.saturating_mul(FILL_LIMITS) {
                self.encode_later(prompt, background.clone());
            }
            return Err(EncodeError::TooLarge);
        }
        let verify =
            found.is_some() && prompt.len() <= limit && self.index.lock().verified < VERIFY_FIRST;
        let (ids, cuts) = self
            .extend(&prompt, found.as_ref(), encoder)
            .map_err(EncodeError::Failed)?;
        let result = if found.is_some() { "hit" } else { "miss" };
        metrics::counter!("mesh_prepare_prefix_cache_total", "result" => result).increment(1);
        if verify {
            let plain = encoder
                .encode(&prompt, false)
                .map_err(EncodeError::Failed)?;
            if plain.token_ids() != ids {
                self.turn_off();
                return Ok(self.wrap(plain.token_ids().to_vec(), add_special_tokens));
            }
            self.index.lock().verified += 1;
        }
        self.store(prompt, &ids, found, cuts);
        Ok(self.wrap(ids, add_special_tokens))
    }

    fn wrap(&self, ids: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
        if !add_special_tokens || (self.head.is_empty() && self.tail.is_empty()) {
            return ids;
        }
        [&self.head[..], &ids, &self.tail].concat()
    }

    /// The stored prompt sharing the longest run of whole added tokens with `prompt`.
    fn find(&self, prompt: &str) -> Option<Found> {
        let bucket = bucket_of(prompt)?;
        let candidates = self.index.lock().candidates(bucket);
        let mut best: Option<Found> = None;
        for (id, stored) in candidates {
            let shared = shared_bytes(
                prompt.as_bytes(),
                stored.text.as_bytes(),
                stored.end().bytes,
            );
            let Some(cut) = stored
                .cuts
                .partition_point(|c| c.bytes <= shared)
                .checked_sub(1)
            else {
                continue;
            };
            let end = stored.cuts[cut].bytes;
            if prompt.is_char_boundary(end) && best.as_ref().is_none_or(|b| end > b.end().bytes) {
                best = Some(Found { id, stored, cut });
            }
        }
        if let Some(found) = &best {
            self.index.lock().touch(found.id);
        }
        best
    }

    /// `prompt`'s ids without the special tokens `encode` wraps it in, reusing
    /// `found`'s, and where the added tokens past those end.
    fn extend(
        &self,
        prompt: &str,
        found: Option<&Found>,
        encoder: &dyn Tokenizer,
    ) -> anyhow::Result<(Vec<u32>, Vec<Cut>)> {
        let start = found.map_or(Cut::default(), Found::end);
        let rest = &prompt[start.bytes..];
        let encoding = encoder.encode(rest, false)?;
        let mut ids = Vec::with_capacity(start.ids + encoding.token_ids().len());
        if let Some(found) = found {
            ids.extend_from_slice(&found.stored.ids[..start.ids]);
        }
        ids.extend_from_slice(encoding.token_ids());
        Ok((ids, self.cuts(rest, &encoding, start)))
    }

    /// Where each added token in `text` ends, from its encoding, after `start`.
    fn cuts(&self, text: &str, encoding: &Encoding, start: Cut) -> Vec<Cut> {
        let Encoding::Hf(encoding) = encoding else {
            return Vec::new();
        };
        encoding
            .get_ids()
            .iter()
            .zip(encoding.get_offsets())
            .enumerate()
            .filter(|(_, (id, &(from, to)))| {
                self.added
                    .get(id)
                    .is_some_and(|token| text.get(from..to) == Some(token.as_str()))
            })
            .map(|(at, (_, &(_, to)))| Cut {
                bytes: start.bytes + to,
                ids: start.ids + at + 1,
            })
            .collect()
    }

    /// Keep `prompt` for later turns. It replaces `found` when all `found` has
    /// past the shared cut is added tokens (the turn's generation prompt, which
    /// the template rewrites once the turn is answered), and sits beside it
    /// otherwise (another conversation with the same start, or a fork).
    fn store(&self, prompt: String, ids: &[u32], found: Option<Found>, new_cuts: Vec<Cut>) {
        // With no added token past the reused cut, `found` already serves the prompt.
        let Some(bucket) = bucket_of(&prompt).filter(|_| !new_cuts.is_empty()) else {
            return;
        };
        let (mut cuts, replaces) = match &found {
            Some(found) => {
                let past = &found.stored.cuts[found.cut..];
                let only_added = past[past.len() - 1].ids - past[0].ids == past.len() - 1;
                (
                    found.stored.cuts[..=found.cut].to_vec(),
                    only_added.then_some(found.id),
                )
            }
            None => (Vec::new(), None),
        };
        cuts.extend(new_cuts);
        let end = cuts.last().copied().unwrap_or_default();
        if end.bytes <= KEY_BYTES {
            return;
        }
        let stored = Arc::new(Stored {
            ids: ids[..end.ids].to_vec(),
            text: prompt,
            cuts,
        });
        let mut index = self.index.lock();
        if !self.on.load(Ordering::Relaxed) {
            return;
        }
        if let Some(id) = replaces {
            index.remove(id);
        }
        index.insert(bucket, stored, self.budget);
        metrics::gauge!("mesh_prepare_prefix_cache_bytes").set(index.held as f64);
    }

    fn encode_later(self: &Arc<Self>, prompt: String, tokenizer: Arc<dyn Tokenizer>) {
        let fill = Fill {
            cache: Arc::downgrade(self),
            tokenizer,
            prompt,
        };
        let result = if FILLS.try_send(fill).is_ok() {
            "queued"
        } else {
            "dropped"
        };
        metrics::counter!("mesh_prepare_prefix_cache_total", "result" => result).increment(1);
    }

    fn fill(&self, prompt: String, tokenizer: &dyn Tokenizer) {
        if !self.on.load(Ordering::Relaxed) {
            return;
        }
        let found = self.find(&prompt);
        match self.extend(&prompt, found.as_ref(), tokenizer) {
            Ok((ids, cuts)) => {
                self.store(prompt, &ids, found, cuts);
                metrics::counter!("mesh_prepare_prefix_cache_total", "result" => "filled")
                    .increment(1);
            }
            Err(error) => warn!(%error, "Failed to encode a prompt in the background"),
        }
    }

    fn turn_off(&self) {
        if self.on.swap(false, Ordering::Relaxed) {
            warn!("Reused prompt ids differ from a plain encode; tokenizing each prompt whole from now on");
        }
        *self.index.lock() = Index::default();
        metrics::gauge!("mesh_prepare_prefix_cache_bytes").set(0.0);
    }
}

/// A prompt over the limit with nothing stored to extend, encoded off the request path.
struct Fill {
    cache: Weak<PrefixTokens>,
    tokenizer: Arc<dyn Tokenizer>,
    prompt: String,
}

static FILLS: LazyLock<SyncSender<Fill>> = LazyLock::new(|| {
    let (sender, receiver) = sync_channel::<Fill>(FILL_QUEUE);
    // If the thread cannot start, the receiver is dropped and so is every fill.
    let _ = std::thread::Builder::new()
        .name("prefix-tokens".into())
        .spawn(move || {
            for fill in receiver {
                if let Some(cache) = fill.cache.upgrade() {
                    cache.fill(fill.prompt, fill.tokenizer.as_ref());
                }
            }
        });
    sender
});

fn bucket_of(prompt: &str) -> Option<u64> {
    (prompt.len() > KEY_BYTES).then(|| xxhash_rust::xxh3::xxh3_64(&prompt.as_bytes()[..KEY_BYTES]))
}

/// Length of the common prefix of `a` and `b`, up to `limit` bytes.
fn shared_bytes(a: &[u8], b: &[u8], limit: usize) -> usize {
    let limit = limit.min(a.len()).min(b.len());
    let mut at = 0;
    while at < limit {
        let end = (at + COMPARE_CHUNK).min(limit);
        if a[at..end] != b[at..end] {
            return at
                + a[at..end]
                    .iter()
                    .zip(&b[at..end])
                    .take_while(|(x, y)| x == y)
                    .count();
        }
        at = end;
    }
    limit
}

/// What decides whether a cut after an added token is exact; the rest of
/// `tokenizer.json` is skipped.
#[derive(Deserialize)]
struct Description {
    #[serde(default)]
    added_tokens: Vec<AddedToken>,
    #[serde(default)]
    normalizer: Value,
    #[serde(default)]
    pre_tokenizer: Value,
    model: Model,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
}

#[derive(Deserialize)]
struct Model {
    #[serde(default)]
    dropout: Option<f64>,
}

impl Description {
    /// Why a prompt encoded in two parts cut right after an added token could
    /// differ from the prompt encoded whole. The tokenizer extracts added
    /// tokens before anything else and encodes the text between them on its
    /// own, unless a token takes in its neighbours' whitespace, the
    /// pre-tokenizer treats the start of its input specially, the normalizer
    /// changes text a normalized token is matched in, the model samples, or
    /// what follows a cut can change what is extracted before it.
    fn check(&self) -> Result<(), String> {
        let tokens = &self.added_tokens;
        if tokens.is_empty() {
            return Err("no added tokens to cut at".into());
        }
        if tokens.iter().any(|t| t.lstrip || t.rstrip || t.single_word) {
            return Err("an added token strips whitespace or matches whole words only".into());
        }
        if self.model.dropout.is_some_and(|p| p > 0.0) {
            return Err("the model samples merges (BPE dropout)".into());
        }
        let normalized: HashSet<&str> = tokens
            .iter()
            .filter(|t| t.normalized)
            .map(|t| t.content.as_str())
            .collect();
        if !normalized.is_empty() && !normalizer_is_identity(&self.normalizer) {
            return Err("normalized added tokens are matched after the normalizer runs".into());
        }
        if offset_sensitive(&self.pre_tokenizer) {
            return Err("the pre-tokenizer treats the start of its input specially".into());
        }
        let contents: Vec<&str> = tokens.iter().map(|t| t.content.as_str()).collect();
        if !cuts_are_stable(&contents, &normalized) {
            return Err("an added token can run past the end of another".into());
        }
        Ok(())
    }
}

fn normalizer_is_identity(normalizer: &Value) -> bool {
    normalizer.is_null()
        || (normalizer["type"] == "Sequence"
            && normalizer["normalizers"]
                .as_array()
                .is_some_and(|n| n.is_empty()))
}

fn offset_sensitive(pre_tokenizer: &Value) -> bool {
    match pre_tokenizer["type"].as_str() {
        Some("Sequence") => pre_tokenizer["pretokenizers"]
            .as_array()
            .is_some_and(|p| p.iter().any(offset_sensitive)),
        Some("Metaspace") => {
            pre_tokenizer["prepend_scheme"].as_str().unwrap_or("always") != "never"
        }
        Some("ByteLevel") => pre_tokenizer["add_prefix_space"].as_bool().unwrap_or(true),
        _ => false,
    }
}

/// Whether a prompt cut right after an added token extracts the same up to
/// the cut whatever follows. Within one pass that fails only when another
/// added token contains it; across passes, also when a token extracted from
/// the raw text first can start inside a normalized one and run past its end.
fn cuts_are_stable(tokens: &[&str], normalized: &HashSet<&str>) -> bool {
    let contains = tokens
        .iter()
        .any(|a| tokens.iter().any(|b| a != b && b.contains(a)));
    let raw: Vec<&str> = tokens
        .iter()
        .copied()
        .filter(|t| !normalized.contains(t))
        .collect();
    let overlaps = normalized.iter().any(|t| {
        raw.iter()
            .any(|u| (1..t.len().min(u.len())).any(|k| t.as_bytes().ends_with(&u.as_bytes()[..k])))
    });
    !contains && !overlaps
}

/// A registered tokenizer's cache, made on first use.
struct Slot {
    shared: Weak<dyn Tokenizer>,
    cache: OnceLock<Option<Arc<PrefixTokens>>>,
}

static CACHES: LazyLock<Mutex<HashMap<String, Arc<Slot>>>> = LazyLock::new(Default::default);

/// The stored prompts of `entry`'s tokenizer, or `None` when `budget` is 0, the
/// cache is off, or reusing ids would not be exact for the tokenizer.
pub(crate) fn cache(entry: &TokenizerEntry, budget: usize) -> Option<Arc<PrefixTokens>> {
    if budget == 0 {
        return None;
    }
    let slot = {
        let mut caches = CACHES.lock();
        let current = caches
            .get(&entry.id)
            .filter(|slot| std::ptr::addr_eq(slot.shared.as_ptr(), Arc::as_ptr(&entry.tokenizer)));
        match current {
            Some(slot) => slot.clone(),
            None => {
                caches.retain(|_, slot| slot.shared.strong_count() > 0);
                let slot = Arc::new(Slot {
                    shared: Arc::downgrade(&entry.tokenizer),
                    cache: OnceLock::new(),
                });
                caches.insert(entry.id.clone(), slot.clone());
                slot
            }
        }
    };
    slot.cache
        .get_or_init(|| match PrefixTokens::for_entry(entry, budget) {
            Ok(cache) => {
                info!(tokenizer = %entry.name, budget_mib = budget >> 20, "Tokenizing prompts by extending earlier ones");
                Some(Arc::new(cache))
            }
            Err(reason) => {
                info!(tokenizer = %entry.name, %reason, "Tokenizing each prompt whole");
                None
            }
        })
        .clone()
        .filter(|cache| cache.on.load(Ordering::Relaxed))
}

/// Drop a removed tokenizer's stored prompts.
pub(crate) fn release(tokenizer_id: &str) {
    CACHES.lock().remove(tokenizer_id);
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    const WORDS: [&str; 6] = ["alpha", "beta", "gamma", "delta", "x", "1"];
    const ADDED: [&str; 6] = [
        "<s>",
        "<unk>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<think>",
    ];

    /// A word-level tokenizer with chat tokens, which wraps prompts in `<s>`.
    fn description() -> Value {
        let added: Vec<&str> = ADDED.into_iter().chain(["</think>"]).collect();
        let vocab: serde_json::Map<String, Value> = added
            .iter()
            .chain(WORDS.iter())
            .enumerate()
            .map(|(id, token)| (token.to_string(), json!(id)))
            .collect();
        let added_tokens: Vec<Value> = added
            .iter()
            .enumerate()
            .map(|(id, token)| {
                json!({"id": id, "content": token, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        json!({"version": "1.0", "truncation": null, "padding": null,
            "added_tokens": added_tokens, "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "unk_token": "<unk>", "vocab": vocab}})
    }

    fn entry(description: &Value) -> (TempDir, TokenizerEntry) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, description.to_string()).unwrap();
        let config = json!({"add_bos_token": true, "bos_token": "<s>"});
        std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        let entry = TokenizerEntry {
            id: "prefix".into(),
            name: "prefix".into(),
            source: dir.path().to_str().unwrap().into(),
            tokenizer: Arc::new(tokenizer),
        };
        (dir, entry)
    }

    /// A system prompt longer than the bucket key.
    fn system() -> String {
        let words: Vec<&str> = (0..4000).map(|i| WORDS[i % WORDS.len()]).collect();
        format!("<|system|>{}", words.join(" "))
    }

    fn plain(entry: &TokenizerEntry, prompt: &str, special: bool) -> Vec<u32> {
        entry
            .tokenizer
            .encode(prompt, special)
            .unwrap()
            .token_ids()
            .to_vec()
    }

    fn encode(
        cache: &Arc<PrefixTokens>,
        entry: &TokenizerEntry,
        prompt: &str,
        limit: usize,
    ) -> Result<Vec<u32>, EncodeError> {
        cache.encode(
            prompt.to_string(),
            true,
            &*entry.tokenizer,
            &entry.tokenizer,
            limit,
        )
    }

    #[test]
    fn a_conversation_reuses_its_earlier_turn() {
        let (_dir, entry) = entry(&description());
        let cache = Arc::new(PrefixTokens::for_entry(&entry, 1 << 30).unwrap());
        assert_eq!(cache.head, vec![0]);
        let system = system();
        let turn1 = format!("{system}<|user|> alpha beta<|assistant|><think>");
        // The template rewrites a turn's generation prompt once it is answered.
        let turn2 = format!(
            "{system}<|user|> alpha beta<|assistant|></think> gamma<|user|> x 1<|assistant|><think>"
        );
        assert_eq!(
            encode(&cache, &entry, &turn1, usize::MAX).unwrap(),
            plain(&entry, &turn1, true)
        );

        let found = cache.find(&turn2).unwrap();
        assert!(turn2[..found.end().bytes].ends_with("<|user|> alpha beta<|assistant|>"));
        assert_eq!(
            encode(&cache, &entry, &turn2, usize::MAX).unwrap(),
            plain(&entry, &turn2, true)
        );
        let no_special = cache.encode(
            turn2.clone(),
            false,
            &*entry.tokenizer,
            &entry.tokenizer,
            usize::MAX,
        );
        assert_eq!(no_special.unwrap(), plain(&entry, &turn2, false));
        // Turn 1 had only its generation prompt past the shared cut, so turn 2 replaced it.
        assert_eq!(cache.index.lock().entries.len(), 1);

        // Another conversation with the same start is kept beside it.
        let other = format!("{system}<|user|> delta<|assistant|><think>");
        assert!(cache.find(&other).is_some());
        assert_eq!(
            encode(&cache, &entry, &other, usize::MAX).unwrap(),
            plain(&entry, &other, true)
        );
        let index = cache.index.lock();
        assert_eq!(index.entries.len(), 2);
        assert_eq!(index.verified, 3);
    }

    #[test]
    fn a_prompt_over_the_limit_is_encoded_for_the_next_turn() {
        let (_dir, entry) = entry(&description());
        let cache = Arc::new(PrefixTokens::for_entry(&entry, 1 << 30).unwrap());
        let system = system();
        let limit = 4096;
        let turn1 = format!("{system}<|user|> alpha<|assistant|><think>");
        assert!(matches!(
            encode(&cache, &entry, &turn1, limit),
            Err(EncodeError::TooLarge)
        ));

        let turn2 = format!(
            "{system}<|user|> alpha<|assistant|></think> beta<|user|> x<|assistant|><think>"
        );
        let deadline = Instant::now() + Duration::from_secs(10);
        while cache.find(&turn2).is_none() {
            assert!(
                Instant::now() < deadline,
                "turn 1 was not stored in the background"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            encode(&cache, &entry, &turn2, limit).unwrap(),
            plain(&entry, &turn2, true)
        );
        // A turn that appends more than the limit is not encoded on the request path.
        let long = format!("{turn2}</think> {}", "gamma ".repeat(1000));
        assert!(matches!(
            encode(&cache, &entry, &long, limit),
            Err(EncodeError::TooLarge)
        ));
    }

    #[test]
    fn a_reuse_that_differs_from_a_plain_encode_turns_the_cache_off() {
        let (_dir, entry) = entry(&description());
        let cache = Arc::new(PrefixTokens::for_entry(&entry, 1 << 30).unwrap());
        let system = system();
        let turn1 = format!("{system}<|user|> alpha<|assistant|><think>");
        encode(&cache, &entry, &turn1, usize::MAX).unwrap();
        {
            let mut index = cache.index.lock();
            let stored = &mut index.entries.values_mut().next().unwrap().stored;
            let mut ids = stored.ids.clone();
            ids[1] += 1;
            *stored = Arc::new(Stored {
                text: stored.text.clone(),
                ids,
                cuts: stored.cuts.clone(),
            });
        }
        let turn2 =
            format!("{system}<|user|> alpha<|assistant|></think> beta<|user|> x<|assistant|>");
        assert_eq!(
            encode(&cache, &entry, &turn2, usize::MAX).unwrap(),
            plain(&entry, &turn2, true)
        );
        assert!(!cache.on.load(Ordering::Relaxed));
        assert!(cache.index.lock().entries.is_empty());
    }

    #[test]
    fn cuts_are_exact_only_for_plain_added_token_splits() {
        let check = |edit: &dyn Fn(&mut Value)| {
            let mut description = description();
            edit(&mut description);
            serde_json::from_value::<Description>(description)
                .unwrap()
                .check()
        };
        assert!(check(&|_| {}).is_ok());
        assert!(check(&|d| d["added_tokens"][3]["rstrip"] = json!(true)).is_err());
        assert!(check(&|d| d["model"]["dropout"] = json!(0.1)).is_err());
        assert!(check(&|d| d["added_tokens"][3]["normalized"] = json!(true)).is_ok());
        assert!(check(&|d| {
            d["added_tokens"][3]["normalized"] = json!(true);
            d["normalizer"] = json!({"type": "NFC"});
        })
        .is_err());
        for (pre_tokenizer, sensitive) in [
            (json!({"type": "Metaspace", "replacement": "_"}), true),
            (
                json!({"type": "Metaspace", "prepend_scheme": "never"}),
                false,
            ),
            (json!({"type": "ByteLevel", "add_prefix_space": true}), true),
            (
                json!({"type": "Sequence", "pretokenizers": [{"type": "Whitespace"},
                    {"type": "ByteLevel", "add_prefix_space": false}]}),
                false,
            ),
        ] {
            assert_eq!(
                check(&|d| d["pre_tokenizer"] = pre_tokenizer.clone()).is_err(),
                sensitive
            );
        }
        // A token containing another can run past that one's end.
        assert!(check(&|d| d["added_tokens"][6]["content"] = json!("<think>x")).is_err());
        let normalized = HashSet::from(["ab"]);
        assert!(!cuts_are_stable(&["ab", "bc"], &normalized));
        assert!(cuts_are_stable(&["ab", "cd"], &normalized));

        // A pre-tokenizer that prepends a space to its input changes the ids
        // after a cut, and the cache stays off.
        let mut metaspace = description();
        metaspace["pre_tokenizer"] = json!({"type": "Metaspace", "replacement": "\u{2581}",
            "prepend_scheme": "always", "split": true});
        let (_dir, entry) = entry(&metaspace);
        assert!(PrefixTokens::for_entry(&entry, 1 << 30).is_err());
    }

    #[test]
    fn stored_prompts_stay_within_the_budget() {
        let stored = |n: usize| {
            Arc::new(Stored {
                text: "x".repeat(n),
                ids: Vec::new(),
                cuts: vec![Cut { bytes: n, ids: 0 }],
            })
        };
        let size = stored(100).size();
        let mut index = Index::default();
        index.insert(1, stored(100), 2 * size);
        index.insert(1, stored(100), 2 * size);
        index.touch(0);
        index.insert(2, stored(100), 2 * size);
        assert_eq!(index.held, 2 * size);
        assert!(index.entries.contains_key(&0) && !index.entries.contains_key(&1));
        assert_eq!(index.buckets[&1], vec![0]);
        index.insert(3, stored(1000), 2 * size);
        assert!(!index.entries.contains_key(&3), "larger than the budget");

        assert_eq!(shared_bytes(b"abcd", b"abxd", 4), 2);
        let long = vec![7u8; 3 * COMPARE_CHUNK];
        let mut other = long.clone();
        other[COMPARE_CHUNK + 5] = 0;
        assert_eq!(shared_bytes(&long, &other, usize::MAX), COMPARE_CHUNK + 5);
        assert_eq!(shared_bytes(&long, &long, 10), 10);
    }
}
