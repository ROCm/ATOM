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
            _tempdir: tempdir,
        }
    })
}

pub(crate) fn hf_tokenizer() -> Arc<dyn Tokenizer> {
    hf_tokenizer_holder().tokenizer.clone()
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
