//! Prompt encoding with fastokens.
//!
//! fastokens encodes the BPE tokenizers of `tokenizer.json` files faster than
//! HuggingFace `tokenizers`, and splits a long text between special tokens
//! across a pool of its own threads (one per logical core up to 32, or
//! `FASTOKENS_BPE_THREADS`).
//! Routing by token needs the ids the engine computes, so a model's prompts
//! encode with fastokens only once fastokens encodes a set of probe texts exactly
//! as its HuggingFace tokenizer does. Decoding, chat templates and vocabulary
//! lookups stay with the HuggingFace tokenizer.

use std::{
    any::Any,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use serde_json::{json, Value};
use tracing::{info, warn};

use crate::{
    config::{FastokensConfig, RouterConfig},
    tokenizer::{
        chat_template::{
            ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
        },
        hub::download_tokenizer_from_hf,
        traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer},
        HuggingFaceTokenizer,
    },
};

/// Long enough for fastokens to split the text across its threads.
const LONG_PROBE_BYTES: usize = 1 << 20;
/// Long enough for fastokens to look for added tokens across its threads; it
/// splits a text across them only at newlines, so one without encodes whole.
const FLAT_PROBE_BYTES: usize = 256 << 10;

/// How the error refusing a model's tokenizer under `--fastokens-strict` begins:
/// fastokens cannot load its `tokenizer.json`, or does not encode or decode the
/// probe texts as HuggingFace does, so registering it again would refuse it
/// again. A `tokenizer.json` that cannot be found or downloaded is not refused.
pub(crate) const STRICT_REFUSAL: &str = "fastokens cannot encode the prompts of";

/// The most text the router tokenizes for one request, by tokenizer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TokenizeLimits {
    huggingface: usize,
    fastokens: usize,
}

impl TokenizeLimits {
    pub(crate) fn new(config: &RouterConfig) -> Self {
        Self {
            huggingface: config.resolved_max_tokenize_bytes(),
            fastokens: config.resolved_fastokens_max_tokenize_bytes(),
        }
    }

    pub(crate) fn of(&self, tokenizer: &dyn Tokenizer) -> usize {
        if tokenizer.as_any().is::<FastTokenizer>() {
            self.fastokens
        } else {
            self.huggingface
        }
    }
}

/// `tokenizer`, encoding with fastokens when `config` lists the model it registers
/// as and fastokens encodes the model's prompts as `tokenizer` does. Otherwise
/// the model keeps `tokenizer`, or fails to register under `--fastokens-strict`.
pub(crate) async fn with_fastokens(
    tokenizer: Arc<dyn Tokenizer>,
    name: &str,
    source: &str,
    config: &FastokensConfig,
) -> Result<Arc<dyn Tokenizer>, String> {
    if !config.covers(name, source) {
        return Ok(tokenizer);
    }
    // A file that cannot be found or downloaded says nothing of how fastokens
    // encodes, so under `--fastokens-strict` the registration fails and is retried.
    let path = match tokenizer_json(source).await {
        Ok(path) => path,
        Err(reason) if config.strict => {
            return Err(format!("cannot load fastokens for {name}: {reason}"))
        }
        Err(reason) => {
            metrics::counter!("mesh_fastokens_tokenizers_total", "outcome" => "fallback")
                .increment(1);
            warn!(
                tokenizer = %name,
                %reason,
                "Cannot find the tokenizer.json fastokens loads; encoding these prompts with HuggingFace"
            );
            return Ok(tokenizer);
        }
    };
    let (inner, cache) = (tokenizer.clone(), config.segment_cache_bytes());
    let loaded = tokio::task::spawn_blocking(move || FastTokenizer::new(inner, &path, cache))
        .await
        .unwrap_or_else(|error| Err(format!("loading fastokens failed: {error}")));
    let outcome = match (&loaded, config.strict) {
        (Ok(_), _) => "loaded",
        (Err(_), true) => "refused",
        (Err(_), false) => "fallback",
    };
    metrics::counter!("mesh_fastokens_tokenizers_total", "outcome" => outcome).increment(1);
    match loaded {
        Ok(fast) => {
            if let Some(error) = &fast.unrendered_chat {
                info!(
                    tokenizer = %name,
                    %error,
                    "The chat template renders none of the probe's chats, so fastokens was not probed on one"
                );
            }
            info!(tokenizer = %name, "Encoding prompts with fastokens");
            Ok(Arc::new(fast))
        }
        Err(reason) if config.strict => Err(format!("{STRICT_REFUSAL} {name}: {reason}")),
        Err(reason) => {
            warn!(
                tokenizer = %name,
                %reason,
                "fastokens cannot encode these prompts as HuggingFace does; encoding them with HuggingFace"
            );
            Ok(tokenizer)
        }
    }
}

/// The `tokenizer.json` the factory loads `source` from: the file itself, the one
/// in a directory, or the one a HuggingFace model ID downloads.
async fn tokenizer_json(source: &str) -> Result<PathBuf, String> {
    let path = Path::new(source);
    let file = if path.is_file() {
        path.to_path_buf()
    } else if path.is_dir() {
        path.join("tokenizer.json")
    } else {
        download_tokenizer_from_hf(source)
            .await
            .map_err(|error| format!("cannot download the tokenizer of {source}: {error}"))?
            .join("tokenizer.json")
    };
    if file.is_file() {
        Ok(file)
    } else {
        Err(format!("{} is not a file", file.display()))
    }
}

/// A HuggingFace tokenizer that encodes with fastokens.
pub(crate) struct FastTokenizer {
    inner: Arc<dyn Tokenizer>,
    fast: fastokens::Tokenizer,
    /// The ids `inner` puts around a text it encodes with special tokens.
    prefix: Vec<TokenIdType>,
    suffix: Vec<TokenIdType>,
    /// Why the probe texts have no chat: the template renders none of them.
    unrendered_chat: Option<String>,
}

impl FastTokenizer {
    /// `inner`, encoding with fastokens from the `tokenizer.json` at `path` it was
    /// loaded from, or why fastokens does not encode the probe texts as `inner`
    /// does. fastokens' segment cache gets `segment_cache_bytes`, or is off at 0.
    pub(crate) fn new(
        inner: Arc<dyn Tokenizer>,
        path: &Path,
        segment_cache_bytes: usize,
    ) -> Result<Self, String> {
        if !inner.as_any().is::<HuggingFaceTokenizer>() {
            return Err("only HuggingFace tokenizers encode with fastokens".into());
        }
        // Not `from_file`, which also adds the tokens only `tokenizer_config.json`
        // declares: HuggingFace `tokenizers` encodes those as text.
        let mut fast = std::fs::read(path)
            .map_err(|error| error.to_string())
            .and_then(|bytes| serde_json::from_slice(&bytes).map_err(|error| error.to_string()))
            .and_then(|json| {
                fastokens::Tokenizer::from_json(json).map_err(|error| error.to_string())
            })
            .map_err(|error| format!("fastokens cannot load {}: {error}", path.display()))?;
        fast.enable_segment_cache(segment_cache_bytes);
        let (prefix, suffix) = special_affixes(&*inner)?;
        let chat = probe_chat(&*inner);
        let tokenizer = Self {
            inner,
            fast,
            prefix,
            suffix,
            unrendered_chat: chat.as_ref().err().cloned(),
        };
        tokenizer.check(chat.ok())?;
        Ok(tokenizer)
    }

    /// The HuggingFace tokenizer this one decodes and renders chats with.
    pub(crate) fn inner(&self) -> &dyn Tokenizer {
        &*self.inner
    }

    fn with_affixes(&self, ids: Vec<TokenIdType>, add_special_tokens: bool) -> Encoding {
        if !add_special_tokens || (self.prefix.is_empty() && self.suffix.is_empty()) {
            return Encoding::Plain(ids);
        }
        let mut all = Vec::with_capacity(self.prefix.len() + ids.len() + self.suffix.len());
        all.extend_from_slice(&self.prefix);
        all.extend(ids);
        all.extend_from_slice(&self.suffix);
        Encoding::Plain(all)
    }

    /// Whether fastokens encodes the probe texts, and `chat`, as `inner` does,
    /// with special tokens and without.
    fn check(&self, chat: Option<String>) -> Result<(), String> {
        for text in probe_texts(chat, self.fast.added_token_configs()) {
            for add_special_tokens in [false, true] {
                let want = encode(&*self.inner, &text, add_special_tokens)?;
                let got = self.encode(&text, add_special_tokens).map_err(|error| {
                    format!("fastokens cannot encode {}: {error}", excerpt(&text))
                })?;
                let got = got.token_ids();
                if want != got {
                    let at = want.iter().zip(got).take_while(|(a, b)| a == b).count();
                    return Err(format!(
                        "fastokens encodes {} (special tokens: {add_special_tokens}) as {:?} from token {at}, HuggingFace as {:?}",
                        excerpt(&text),
                        &got[at..got.len().min(at + 4)],
                        &want[at..want.len().min(at + 4)],
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The ids `tokenizer` puts before and after the ids of a text when it encodes
/// it with special tokens; its post-processor adds the same ones to every text.
fn special_affixes(
    tokenizer: &dyn Tokenizer,
) -> Result<(Vec<TokenIdType>, Vec<TokenIdType>), String> {
    const TEXT: &str = "special tokens";
    let plain = encode(tokenizer, TEXT, false)?;
    let special = encode(tokenizer, TEXT, true)?;
    let at = (!plain.is_empty())
        .then(|| special.windows(plain.len()).position(|ids| ids == plain))
        .flatten()
        .ok_or("the HuggingFace post-processor changes the ids of the text")?;
    Ok((special[..at].to_vec(), special[at + plain.len()..].to_vec()))
}

fn encode(
    tokenizer: &dyn Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> Result<Vec<TokenIdType>, String> {
    tokenizer
        .encode(text, add_special_tokens)
        .map(|encoding| encoding.token_ids().to_vec())
        .map_err(|error| format!("HuggingFace cannot encode {}: {error}", excerpt(text)))
}

const PROBES: &[&str] = &[
    "",
    " ",
    "Hello, world! How's it going? I'm fine, you'll see.",
    "  two leading spaces, two trailing  ",
    "tabs\tand\nnewlines\r\n\n\n    indented\n\t\tdeeper",
    "1 12 123 1234 12345 123456 3.14159 -42 1e10 0x1F 1,000,000 2026-11-19",
    "def f(x):\n    return {'a': [1, 2, 3], \"b\": x ** 2}  # comment\n",
    "<div class=\"a\">&nbsp;</div> <|not a token|> </s> [INST]",
    "中文测试，标点符号。日本語のテキスト、カタカナ。한국어 텍스트",
    "emoji 👍🏽 👨‍👩‍👧 🇨🇳 ❤️ and symbols ∑ √ ∞ ≠ ±",
    "precomposed é and decomposed e\u{301}, ﬁ ligature, Å Ω µ ½ ①",
    "مرحبا بالعالم שלום עולם नमस्ते दुनिया",
    "mixed中English混合123text😀end",
    "!!!???...---___***///\\\\|||",
    "\u{0}\u{1}\u{1F}\u{7F}\u{A0}\u{2028}\u{FEFF}\u{FFFD}",
];

/// Texts on which tokenizers that disagree tend to: special tokens, digits,
/// whitespace and scripts, `chat`, a text long enough for fastokens to split
/// and one with no newline.
fn probe_texts(chat: Option<String>, added: &[fastokens::AddedTokenConfig]) -> Vec<String> {
    let probes: Vec<_> = PROBES.iter().map(|text| text.to_string()).collect();
    let added: Vec<_> = added.iter().map(|token| token.content.as_str()).collect();
    let mut texts = probes.clone();
    for run in ["a", " ", "\n", "1", "é", "中", "👍"] {
        texts.push(run.repeat(3000));
    }
    let mut long = String::with_capacity(LONG_PROBE_BYTES + 512);
    let mut line = 0;
    while long.len() < LONG_PROBE_BYTES {
        long.push_str(&format!("{line}. {}\n", probes[line % probes.len()]));
        line += 1;
    }
    texts.push(long);
    let words: Vec<_> = probes
        .iter()
        .map(|text| text.replace(['\n', '\r'], " "))
        .chain(
            added
                .iter()
                .filter(|token| !token.contains(['\n', '\r']))
                .map(|token| token.to_string()),
        )
        .collect();
    let words = words.join(" ");
    texts.push(words.repeat(FLAT_PROBE_BYTES.div_ceil(words.len())));
    texts.extend(chat);
    for token in added {
        texts.push(format!("a{token}b {token} c\n {token}\n\n{token}{token}d"));
    }
    texts
}

/// A chat as the template renders it for a request: with a system message, or
/// without one for a template that has no system role.
fn probe_chat(tokenizer: &dyn Tokenizer) -> Result<String, String> {
    let messages = [
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "Hi! 你好 1234"}),
        json!({"role": "assistant", "content": "Hello there."}),
        json!({"role": "user", "content": "What is 2 + 2?"}),
    ];
    let render = |messages: &[Value]| {
        tokenizer.apply_chat_template(
            messages,
            ChatTemplateParams {
                add_generation_prompt: true,
                special_tokens: Some(tokenizer.get_special_tokens()),
                ..Default::default()
            },
        )
    };
    render(&messages)
        .or_else(|_| render(&messages[1..]))
        .map_err(|error| error.to_string())
}

/// The start of `text` for an error message, and its length when that is not
/// all of it.
fn excerpt(text: &str) -> String {
    match text.char_indices().nth(40) {
        None => format!("{text:?}"),
        Some((end, _)) => format!("{:?}... ({} bytes)", &text[..end], text.len()),
    }
}

impl Encoder for FastTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        Ok(self.with_affixes(self.fast.encode(input)?, add_special_tokens))
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        Ok(self
            .fast
            .encode_batch(inputs, false)?
            .into_iter()
            .map(|ids| self.with_affixes(ids, add_special_tokens))
            .collect())
    }
}

impl Decoder for FastTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(token_ids, skip_special_tokens)
    }

    fn decode_step(
        &self,
        token_id: TokenIdType,
        ids: &mut Vec<TokenIdType>,
        prefix: &mut String,
        prefix_index: &mut usize,
        skip_special_tokens: bool,
    ) -> Result<Option<String>> {
        self.inner
            .decode_step(token_id, ids, prefix, prefix_index, skip_special_tokens)
    }
}

impl Tokenizer for FastTokenizer {
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn get_special_tokens(&self) -> &SpecialTokens {
        self.inner.get_special_tokens()
    }

    fn token_to_id(&self, token: &str) -> Option<TokenIdType> {
        self.inner.token_to_id(token)
    }

    fn id_to_token(&self, id: TokenIdType) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn apply_chat_template(
        &self,
        messages: &[Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        self.inner.apply_chat_template(messages, params)
    }

    fn chat_template_content_format(&self) -> ChatTemplateContentFormat {
        self.inner.chat_template_content_format()
    }

    fn thinking_toggle(&self) -> ThinkingToggle {
        self.inner.thinking_toggle()
    }

    fn thinking_key_name(&self) -> Option<ThinkingKeyName> {
        self.inner.thinking_key_name()
    }

    fn think_in_prefill(&self) -> bool {
        self.inner.think_in_prefill()
    }

    fn eos_token_ids(&self) -> &[TokenIdType] {
        self.inner.eos_token_ids()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::DEFAULT_FASTOKENS_SEGMENT_CACHE_MB,
        routers::test_mocks::tokenizer::{
            byte_level_next_id, byte_level_tokenizer_dir, hf_tokenizer, hf_tokenizer_path,
            BYTE_LEVEL_MERGES,
        },
        tokenizer::{factory, MockTokenizer},
    };

    const CHAT_TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    fn load(dir: &Path) -> (Arc<dyn Tokenizer>, Result<FastTokenizer, String>) {
        let path = dir.join("tokenizer.json");
        let hf = factory::create_tokenizer(path.to_str().unwrap()).unwrap();
        let fast = FastTokenizer::new(hf.clone(), &path, 0);
        (hf, fast)
    }

    fn ids(tokenizer: &dyn Tokenizer, text: &str, add_special_tokens: bool) -> Vec<TokenIdType> {
        encode(tokenizer, text, add_special_tokens).unwrap()
    }

    #[test]
    fn fastokens_encodes_as_huggingface_and_leaves_the_rest_to_it() {
        let texts = [
            "",
            "the weather in 123 <|im_start|>user\n你好<|im_end|>",
            "a  <tool>  b\n\n<tool>",
        ];
        for adds_bos_eos in [true, false] {
            let dir = byte_level_tokenizer_dir(
                BYTE_LEVEL_MERGES,
                adds_bos_eos,
                json!({"chat_template": CHAT_TEMPLATE}),
            );
            let (hf, fast) = load(dir.path());
            let fast = fast.unwrap();
            for text in texts {
                for special in [false, true] {
                    assert_eq!(
                        ids(&fast, text, special),
                        ids(&*hf, text, special),
                        "{text:?}"
                    );
                }
            }
            for special in [false, true] {
                let batch = fast.encode_batch(&texts, special).unwrap();
                for (text, encoding) in texts.iter().zip(&batch) {
                    assert_eq!(encoding.token_ids(), ids(&*hf, text, special));
                }
            }
            let bos = hf.token_to_id("<s>").unwrap();
            assert_eq!(ids(&fast, "the", true)[0] == bos, adds_bos_eos);

            let text = ids(&*hf, "the weather", false);
            assert_eq!(fast.decode(&text, false).unwrap(), "the weather");
            assert_eq!(fast.vocab_size(), hf.vocab_size());
            assert_eq!(fast.token_to_id("<|im_end|>"), hf.token_to_id("<|im_end|>"));
            let chat = [json!({"role": "user", "content": "hi"})];
            let params = || ChatTemplateParams {
                add_generation_prompt: true,
                ..Default::default()
            };
            assert_eq!(
                fast.apply_chat_template(&chat, params()).unwrap(),
                hf.apply_chat_template(&chat, params()).unwrap()
            );
        }
    }

    /// DeepSeek-R1-0528's `tokenizer.json` adds no BOS; its `tokenizer_config.json`
    /// does, and HuggingFace follows it as `transformers` does.
    #[test]
    fn special_tokens_are_the_ones_huggingface_adds() {
        let dir = byte_level_tokenizer_dir(
            BYTE_LEVEL_MERGES,
            false,
            json!({"add_bos_token": true, "bos_token": "<s>"}),
        );
        let (hf, fast) = load(dir.path());
        let fast = fast.unwrap();
        let with_bos = ids(&fast, "the weather", true);
        assert_eq!(with_bos, ids(&*hf, "the weather", true));
        assert_eq!(with_bos[0], hf.token_to_id("<s>").unwrap());
        assert_eq!(ids(&fast, "the weather", false), with_bos[1..]);
        let alone = fastokens::Tokenizer::from_file(&dir.path().join("tokenizer.json")).unwrap();
        assert_eq!(
            alone
                .encode_with_special_tokens("the weather", true)
                .unwrap(),
            with_bos[1..]
        );
    }

    /// HuggingFace `tokenizers` encodes the tokens only `tokenizer_config.json`
    /// declares as text, and so does fastokens, which loads `tokenizer.json` alone.
    #[test]
    fn tokens_only_tokenizer_config_declares_encode_as_huggingface_does() {
        let id = byte_level_next_id(BYTE_LEVEL_MERGES);
        let token = json!({"content": "<|config_only|>", "normalized": false, "special": true});
        let config = json!({"added_tokens_decoder": {id.to_string(): token}});
        let dir = byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, config);
        let (hf, fast) = load(dir.path());
        let fast = fast.unwrap();
        let text = "the<|config_only|> weather";
        for special in [false, true] {
            assert_eq!(ids(&fast, text, special), ids(&*hf, text, special));
        }
        assert!(!ids(&fast, text, false).contains(&id));
    }

    /// The probe texts keep the long ones and the chat.
    #[test]
    fn probe_texts_keep_the_long_ones_and_the_chat() {
        let dir = byte_level_tokenizer_dir(
            BYTE_LEVEL_MERGES,
            true,
            json!({"chat_template": CHAT_TEMPLATE}),
        );
        let (hf, _) = load(dir.path());
        let chat = probe_chat(&*hf).unwrap();
        let fast = fastokens::Tokenizer::from_file(&dir.path().join("tokenizer.json")).unwrap();
        let texts = probe_texts(Some(chat.clone()), fast.added_token_configs());
        assert!(texts.iter().any(|text| text.len() >= LONG_PROBE_BYTES));
        let flat = texts
            .iter()
            .find(|text| text.len() >= FLAT_PROBE_BYTES && !text.contains('\n'))
            .unwrap();
        assert!(flat.contains("<|im_start|>"), "{}", excerpt(flat));
        assert!(texts.contains(&chat));
    }

    /// The probe renders its chat without a system message for a template that
    /// has no system role, and a template that renders neither has no chat probed.
    #[test]
    fn the_chat_probe_needs_no_system_role() {
        let template = format!(
            "{{% if messages[0].role == 'system' %}}{{{{ raise_exception('no system role') }}}}{{% endif %}}{CHAT_TEMPLATE}"
        );
        let dir =
            byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, json!({"chat_template": template}));
        let (hf, fast) = load(dir.path());
        assert!(probe_chat(&*hf).unwrap().starts_with("<|im_start|>user\n"));
        assert_eq!(fast.unwrap().unrendered_chat, None);

        let dir = byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, json!({}));
        let fast = load(dir.path()).1.unwrap();
        assert!(fast.unrendered_chat.is_some());
    }

    #[test]
    fn a_tokenizer_fastokens_encodes_differently_is_refused() {
        let hf = byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, json!({}));
        let merges: Vec<_> = BYTE_LEVEL_MERGES
            .iter()
            .map(|&merge| if merge == "Ġ w" { "Ġ x" } else { merge })
            .collect();
        let other = byte_level_tokenizer_dir(&merges, true, json!({}));
        let path = hf.path().join("tokenizer.json");
        let hf = factory::create_tokenizer(path.to_str().unwrap()).unwrap();
        let error = FastTokenizer::new(hf, &other.path().join("tokenizer.json"), 0)
            .err()
            .unwrap();
        assert!(error.contains("fastokens encodes"), "{error}");
    }

    #[test]
    fn tokenizers_fastokens_cannot_load_are_refused() {
        let path = Path::new(hf_tokenizer_path());
        let error = FastTokenizer::new(hf_tokenizer(), path, 0).err().unwrap();
        assert!(error.contains("fastokens cannot load"), "{error}");
        let error = FastTokenizer::new(Arc::new(MockTokenizer::new()), path, 0)
            .err()
            .unwrap();
        assert!(error.contains("only HuggingFace"), "{error}");
    }

    /// `--fastokens-models` names a model by the name or the source its tokenizer
    /// registers under, or as `*`; `--fastokens-segment-cache-mb` sizes the
    /// segment cache, and 0 turns it off.
    #[tokio::test]
    async fn listed_models_encode_with_fastokens() {
        let dir = byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, json!({}));
        let source = dir.path().to_str().unwrap();
        let file = dir.path().join("tokenizer.json");
        let hf = factory::create_tokenizer(source).unwrap();
        let config = |models: &[&str], segment_cache_mb| FastokensConfig {
            models: models.iter().map(|model| model.to_string()).collect(),
            strict: true,
            segment_cache_mb,
            ..Default::default()
        };

        let kept = with_fastokens(hf.clone(), "model", source, &config(&["other"], Some(1)))
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&kept, &hf));
        for (models, source, segment_cache_mb, budget) in [
            (["model"], source, Some(1), Some(1 << 20)),
            ([source], source, Some(0), None),
            (
                ["*"],
                file.to_str().unwrap(),
                None,
                Some(DEFAULT_FASTOKENS_SEGMENT_CACHE_MB << 20),
            ),
        ] {
            let fast = with_fastokens(
                hf.clone(),
                "model",
                source,
                &config(&models, segment_cache_mb),
            )
            .await
            .unwrap();
            let fast = fast.as_any().downcast_ref::<FastTokenizer>().unwrap();
            assert_eq!(
                fast.fast.segment_cache_stats().map(|stats| stats.max_bytes),
                budget
            );
        }
    }
}
