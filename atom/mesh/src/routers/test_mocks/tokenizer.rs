use std::sync::{Arc, OnceLock};

use tempfile::TempDir;

use crate::tokenizer::{
    huggingface::HuggingFaceTokenizer, registry::TokenizerRegistry, traits::Tokenizer,
    MockTokenizer,
};

pub(crate) fn tokenizer() -> Arc<dyn Tokenizer> {
    Arc::new(MockTokenizer::new())
}

pub(crate) fn tokenizer_registry_with(name: &str) -> Arc<TokenizerRegistry> {
    let registry = Arc::new(TokenizerRegistry::new());
    let id = TokenizerRegistry::generate_id();
    let registry_clone = registry.clone();
    let name_owned = name.to_string();
    let load = async move {
        registry_clone
            .load(&id, &name_owned, "mock", || async { Ok(tokenizer()) })
            .await
            .expect("test tokenizer must load");
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            std::thread::scope(|s| {
                s.spawn(|| handle.block_on(load));
            });
        }
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime must build");
            rt.block_on(load);
        }
    }
    registry
}

const TOKENIZER_JSON: &str = r#"{
    "version": "1.0",
    "truncation": null,
    "padding": null,
    "added_tokens": [],
    "normalizer": null,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": null,
    "decoder": null,
    "model": {
        "type": "BPE",
        "vocab": {"test": 0, "<s>": 1, "</s>": 2},
        "merges": []
    }
}"#;

const TOKENIZER_CONFIG_JSON: &str = r#"{
    "chat_template": "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}{% if not loop.last %}\n{% endif %}{% endfor %}"
}"#;

struct HfTokenizerHolder {
    tokenizer: Arc<dyn Tokenizer>,
    path: String,
    _tempdir: TempDir,
}

fn hf_tokenizer_holder() -> &'static HfTokenizerHolder {
    static HOLDER: OnceLock<HfTokenizerHolder> = OnceLock::new();
    HOLDER.get_or_init(|| {
        let tempdir = TempDir::new().expect("tempdir for HF tokenizer fixture");
        let tokenizer_path = tempdir.path().join("tokenizer.json");
        let config_path = tempdir.path().join("tokenizer_config.json");
        std::fs::write(&tokenizer_path, TOKENIZER_JSON).expect("write tokenizer.json");
        std::fs::write(&config_path, TOKENIZER_CONFIG_JSON).expect("write tokenizer_config.json");
        let hf = HuggingFaceTokenizer::from_file(tokenizer_path.to_str().unwrap())
            .expect("load HF tokenizer from tempdir");
        HfTokenizerHolder {
            tokenizer: Arc::new(hf),
            path: tokenizer_path.to_str().unwrap().to_owned(),
            _tempdir: tempdir,
        }
    })
}

pub(crate) fn hf_tokenizer() -> Arc<dyn Tokenizer> {
    hf_tokenizer_holder().tokenizer.clone()
}

/// Path of the tokenizer.json backing [`hf_tokenizer`].
pub(crate) fn hf_tokenizer_path() -> &'static str {
    &hf_tokenizer_holder().path
}

/// Merges of a byte-level BPE, in rank order.
pub(crate) const BYTE_LEVEL_MERGES: &[&str] = &[
    "Ġ t",
    "h e",
    "Ġt he",
    "i n",
    "Ġ a",
    "e r",
    "o n",
    "r e",
    "Ġ w",
    "o r",
    "1 2",
    "12 3",
    "Ġ Ġ",
    "ĠĠ ĠĠ",
    "Ċ Ċ",
    "ä ¸",
];

const BYTE_LEVEL_ADDED: &[&str] = &["<s>", "</s>", "<|im_start|>", "<|im_end|>", "<tool>"];

/// The id after the last token of [`byte_level_tokenizer_dir`]`(merges, ..)`.
pub(crate) fn byte_level_next_id(merges: &[&str]) -> u32 {
    (256 + merges.len() + BYTE_LEVEL_ADDED.len()) as u32
}

/// A directory with a byte-level BPE `tokenizer.json`, the kind DeepSeek, Qwen
/// and Llama 3 ship, and `config` as its `tokenizer_config.json`. It adds `<s>`
/// and `</s>` around a text it encodes with special tokens if `adds_bos_eos`, and
/// its `<tool>` token takes the whitespace around it.
pub(crate) fn byte_level_tokenizer_dir(
    merges: &[&str],
    adds_bos_eos: bool,
    config: serde_json::Value,
) -> TempDir {
    use serde_json::json;

    // GPT-2's byte alphabet: printable bytes stand for themselves, the others
    // for the characters from U+0100 on.
    let mut vocab = serde_json::Map::new();
    let mut unprintable = 0;
    for byte in 0..=255u8 {
        let char = if matches!(byte, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF) {
            char::from(byte)
        } else {
            unprintable += 1;
            char::from_u32(255 + unprintable).unwrap()
        };
        vocab.insert(char.to_string(), json!(byte));
    }
    for merge in merges {
        let id = vocab.len();
        vocab.insert(merge.replace(' ', ""), json!(id));
    }
    let first = vocab.len();
    let added: Vec<_> = BYTE_LEVEL_ADDED
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let special = *content != "<tool>";
            json!({"id": first + i, "content": content, "single_word": false,
                "lstrip": !special, "rstrip": !special, "normalized": false, "special": special})
        })
        .collect();
    let post_processor = if adds_bos_eos {
        let (bos, eos) = (
            json!({"SpecialToken": {"id": "<s>", "type_id": 0}}),
            json!({"SpecialToken": {"id": "</s>", "type_id": 0}}),
        );
        json!({
            "type": "TemplateProcessing",
            "single": [bos, {"Sequence": {"id": "A", "type_id": 0}}, eos],
            "pair": [bos, {"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}, eos],
            "special_tokens": {
                "<s>": {"id": "<s>", "ids": [first], "tokens": ["<s>"]},
                "</s>": {"id": "</s>", "ids": [first + 1], "tokens": ["</s>"]}
            }
        })
    } else {
        json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": true})
    };
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": post_processor,
        "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
            "ignore_merges": false, "vocab": vocab, "merges": merges
        }
    });
    let dir = TempDir::new().expect("tempdir for the byte-level tokenizer");
    std::fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
    std::fs::write(dir.path().join("tokenizer_config.json"), config.to_string()).unwrap();
    dir
}

/// Sets an environment variable until dropped, then restores it. The tests that
/// set one, and those that read it, are `#[serial]`.
pub(crate) struct EnvVar {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl EnvVar {
    pub(crate) fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        std::env::set_var(name, value);
        Self { name, previous }
    }
}

impl Drop for EnvVar {
    fn drop(&mut self) {
        match &self.previous {
            Some(value) => std::env::set_var(self.name, value),
            None => std::env::remove_var(self.name),
        }
    }
}

pub(crate) fn tokenizer_registry_with_hf(name: &str) -> Arc<TokenizerRegistry> {
    let registry = Arc::new(TokenizerRegistry::new());
    let id = TokenizerRegistry::generate_id();
    let registry_clone = registry.clone();
    let name_owned = name.to_string();
    let load = async move {
        registry_clone
            .load(&id, &name_owned, "hf-test", || async { Ok(hf_tokenizer()) })
            .await
            .expect("HF test tokenizer must load");
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            std::thread::scope(|s| {
                s.spawn(|| handle.block_on(load));
            });
        }
        Err(_) => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime must build");
            rt.block_on(load);
        }
    }
    registry
}
