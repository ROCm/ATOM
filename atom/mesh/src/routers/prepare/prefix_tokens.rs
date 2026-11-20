//! Whether a prompt can be tokenized in two parts, cut right after an added
//! token, with the ids of tokenizing it whole.
//!
//! The checks follow ATOM's `SegmentTokenizer` (`atom/model_engine/segment_split.py`)
//! plus a refusal of a tokenizer that truncates or pads, read from the
//! tokenizer's `tokenizer.json` and tried on a few probes. Unlike
//! `SegmentTokenizer`, they refuse a pre-tokenizer only when it marks the start
//! of the whole input and not that of each text between added tokens.
#![cfg_attr(not(test), allow(dead_code))]

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use serde::Deserialize;
use serde_json::Value;

use crate::tokenizer::{
    huggingface::HuggingFaceTokenizer,
    registry::TokenizerEntry,
    traits::{Encoding, Tokenizer},
};

/// Where an added token ends: its byte offset in the prompt, and the ids up to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Cut {
    bytes: usize,
    ids: usize,
}

/// A tokenizer's added tokens, after which a prompt can be cut.
pub(crate) struct PrefixTokens {
    /// Added tokens by id; where one ends in a prompt is a cut.
    added: HashMap<u32, String>,
    /// What `encode` wraps a prompt in with its special tokens.
    head: Vec<u32>,
    tail: Vec<u32>,
}

impl PrefixTokens {
    fn for_entry(entry: &TokenizerEntry) -> Result<Self, String> {
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
        Self::new(entry.tokenizer.as_ref(), &description)
    }

    /// The added tokens of `tokenizer`, described by its `tokenizer.json`, or
    /// why cutting a prompt after one would not be exact for it.
    fn new(tokenizer: &dyn Tokenizer, description: &[u8]) -> Result<Self, String> {
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
        let added: HashMap<u32, String> = description
            .added_tokens
            .into_iter()
            .map(|t| (t.id, t.content))
            .collect();
        let cache = Self {
            added,
            head: full[..start].to_vec(),
            tail: full[start + bare.len()..].to_vec(),
        };
        cache.probe(tokenizer)?;
        Ok(cache)
    }

    /// A few placements of added tokens, cut after each token and encoded in
    /// two parts, give the ids of encoding them whole, which special tokens
    /// wrap in `head` and `tail`. At least one cut must be checked.
    ///
    /// The probes cut a prompt only before its own continuation: that a cached
    /// prefix stays exact before a new continuation rests on
    /// `Description::check`, and so does a normalizer that composes across a
    /// cut, for which `cuts` finds no cut to compare.
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
        let encode =
            |text: &str, special: bool| tokenizer.encode(text, special).map_err(|e| e.to_string());
        let mut checked = 0;
        for token in picks {
            for probe in [
                format!("a{token}a b"),
                format!("{token} x\n{token}"),
                format!(" y {token}\t1 2"),
                // Combining marks beside the token.
                format!("e\u{301}{token}\u{301}e"),
            ] {
                let whole = encode(&probe, false)?;
                let ids = whole.token_ids();
                if encode(&probe, true)?.token_ids()
                    != [&self.head[..], ids, &self.tail[..]].concat()
                {
                    return Err(format!("special tokens wrap {probe:?} unlike \"a\""));
                }
                for cut in self.cuts(&probe, &whole, Cut::default()) {
                    let before = encode(&probe[..cut.bytes], false)?;
                    let after = encode(&probe[cut.bytes..], false)?;
                    if ids[..cut.ids] != *before.token_ids() || ids[cut.ids..] != *after.token_ids()
                    {
                        return Err(format!("cutting after {token:?} changes the ids"));
                    }
                    checked += 1;
                }
            }
        }
        if checked == 0 {
            return Err("no probe has an added token to cut after".into());
        }
        Ok(())
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
    #[serde(default)]
    truncation: Value,
    #[serde(default)]
    padding: Value,
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
    /// pre-tokenizer marks the start of the whole input but not that of the
    /// text after a cut, the normalizer changes text a normalized token is
    /// matched in, the model samples, the tokenizer truncates or pads what it
    /// encodes, or what follows a cut can change what is extracted before it.
    fn check(&self) -> Result<(), String> {
        let tokens = &self.added_tokens;
        if tokens.is_empty() {
            return Err("no added tokens to cut at".into());
        }
        if !self.truncation.is_null() || !self.padding.is_null() {
            return Err("the tokenizer truncates or pads what it encodes".into());
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
            return Err("the pre-tokenizer marks only the start of the whole input".into());
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

/// Whether the pre-tokenizer marks the start of the whole input but not that of
/// each text between added tokens, as Metaspace's `first` scheme does. Its
/// `always` scheme and ByteLevel's `add_prefix_space` prepend to each of those
/// texts, so the text after a cut is marked whether it is encoded on its own
/// or not.
fn offset_sensitive(pre_tokenizer: &Value) -> bool {
    match pre_tokenizer["type"].as_str() {
        Some("Sequence") => pre_tokenizer["pretokenizers"]
            .as_array()
            .is_some_and(|p| p.iter().any(offset_sensitive)),
        Some("Metaspace") => pre_tokenizer["prepend_scheme"] == "first",
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

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tempfile::TempDir;

    use super::*;

    const WORDS: [&str; 6] = ["alpha", "beta", "gamma", "delta", "x", "1"];
    /// The probes' other words. Each word is in the vocabulary marked by a
    /// Metaspace pre-tokenizer too, so a probe can tell a marked word.
    const PROBE_WORDS: [&str; 4] = ["a", "b", "y", "2"];
    const ADDED: [&str; 6] = [
        "<s>",
        "<unk>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<think>",
    ];

    /// A word-level tokenizer with chat tokens; `entry` adds a config that
    /// wraps prompts in `<s>`.
    pub(crate) fn description() -> Value {
        let added: Vec<&str> = ADDED.into_iter().chain(["</think>"]).collect();
        let words = WORDS.into_iter().chain(PROBE_WORDS);
        let marked = words.clone().map(|word| format!("\u{2581}{word}"));
        let vocab: serde_json::Map<String, Value> = added
            .iter()
            .copied()
            .chain(words)
            .map(str::to_string)
            .chain(marked)
            .enumerate()
            .map(|(id, token)| (token, json!(id)))
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

    pub(crate) fn entry(description: &Value) -> (TempDir, TokenizerEntry) {
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

    #[test]
    fn a_prompt_is_cut_after_each_added_token() {
        let (_dir, entry) = entry(&description());
        let mut tokens = PrefixTokens::for_entry(&entry).unwrap();
        assert_eq!(tokens.head, [0]);
        assert!(tokens.tail.is_empty());
        let prompt = "<|user|> alpha<|assistant|><think>";
        let encoding = entry.tokenizer.encode(prompt, false).unwrap();
        assert_eq!(
            tokens.cuts(prompt, &encoding, Cut::default()),
            [
                Cut { bytes: 8, ids: 1 },
                Cut { bytes: 27, ids: 3 },
                Cut { bytes: 34, ids: 4 },
            ]
        );

        // The probes check the special tokens too, and a cut at least.
        let tokenizer = entry.tokenizer.as_ref();
        tokens.head.clear();
        assert!(tokens.probe(tokenizer).is_err());
        tokens.head = vec![0];
        tokens.added = HashMap::from([(99, "<|tool|>".to_string())]);
        assert_eq!(
            tokens.probe(tokenizer).err().as_deref(),
            Some("no probe has an added token to cut after")
        );
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
        assert!(check(&|d| d["truncation"] = json!({"max_length": 8192})).is_err());
        assert!(check(&|d| d["padding"] = json!({"strategy": "BatchLongest"})).is_err());
        assert!(check(&|d| d["added_tokens"][3]["normalized"] = json!(true)).is_ok());
        assert!(check(&|d| {
            d["added_tokens"][3]["normalized"] = json!(true);
            d["normalizer"] = json!({"type": "NFC"});
        })
        .is_err());
        for (pre_tokenizer, sensitive) in [
            (json!({"type": "Metaspace", "replacement": "_"}), false),
            (
                json!({"type": "Metaspace", "prepend_scheme": "never"}),
                false,
            ),
            (
                json!({"type": "Metaspace", "prepend_scheme": "first"}),
                true,
            ),
            (
                json!({"type": "ByteLevel", "add_prefix_space": true}),
                false,
            ),
            (
                json!({"type": "Sequence", "pretokenizers": [{"type": "Whitespace"},
                    {"type": "Metaspace", "prepend_scheme": "first"}]}),
                true,
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

        // A pre-tokenizer that marks each text between added tokens marks the
        // text after a cut either way; one that marks only the start of the
        // whole input keeps the cache off.
        let (_dir, always) = entry(&metaspace("always"));
        assert!(PrefixTokens::for_entry(&always).is_ok());
        let (_dir, first) = entry(&metaspace("first"));
        assert_eq!(
            PrefixTokens::for_entry(&first).err().as_deref(),
            Some("the pre-tokenizer marks only the start of the whole input")
        );
    }

    fn metaspace(prepend_scheme: &str) -> Value {
        let mut description = description();
        description["pre_tokenizer"] = json!({"type": "Metaspace", "replacement": "\u{2581}",
            "prepend_scheme": prepend_scheme, "split": true});
        description
    }

    #[test]
    fn the_probes_catch_a_mark_on_the_start_of_the_whole_input() {
        let (_dir, plain) = entry(&description());
        let mut tokens = PrefixTokens::for_entry(&plain).unwrap();
        tokens.added = HashMap::from([(3, "<|user|>".to_string())]);
        // `check` refuses this pre-tokenizer before any probe. After the cut,
        // `a b` encoded on its own is `▁a ▁b`, and `a ▁b` in the whole prompt.
        let (_dir, first) = entry(&metaspace("first"));
        assert_eq!(
            tokens.probe(first.tokenizer.as_ref()).err().as_deref(),
            Some("cutting after \"<|user|>\" changes the ids")
        );
    }
}
