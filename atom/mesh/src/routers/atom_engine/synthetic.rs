//! The text a token reads as while forced acceptance (`--spec-decode-acceptance-*`)
//! makes the model's own meaningless: the Python server's `delivered_text`.
//!
//! It stands in for the decoded text, not for the readers after it, so the
//! reasoning and tool-call parsers still see what api_server's see. Decoded
//! anyway, as api_server does, so the run measures the work it would do.

use std::sync::Arc;

use anyhow::Result;

use crate::tokenizer::{
    chat_template::{
        ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
    },
    traits::{Decoder, Encoder, Encoding, SpecialTokens, TokenIdType, Tokenizer},
};

/// `inner`, decoding every token as `text`. Everything but decoding is
/// `inner`'s, its type included (`as_any`), so preparation still finds the
/// HuggingFace tokenizer it renders templates with. The incremental decode is
/// the trait's own, over this `decode`.
pub(super) struct SyntheticText {
    inner: Arc<dyn Tokenizer>,
    text: String,
}

impl SyntheticText {
    pub(super) fn wrap(inner: Arc<dyn Tokenizer>, text: String) -> Arc<dyn Tokenizer> {
        Arc::new(Self { inner, text })
    }
}

impl Encoder for SyntheticText {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        self.inner.encode(input, add_special_tokens)
    }

    fn encode_batch(&self, inputs: &[&str], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        self.inner.encode_batch(inputs, add_special_tokens)
    }
}

impl Decoder for SyntheticText {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(token_ids, skip_special_tokens)?;
        Ok(self.text.repeat(token_ids.len()))
    }
}

impl Tokenizer for SyntheticText {
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

    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }

    fn apply_chat_template(
        &self,
        messages: &[serde_json::Value],
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
    use crate::tokenizer::{mock::MockTokenizer, sequence::Sequence};

    #[test]
    fn every_token_reads_as_the_text_one_shot_and_incrementally() {
        let tokenizer = SyntheticText::wrap(Arc::new(MockTokenizer::new()), "synthetic ".into());
        assert_eq!(
            tokenizer.decode(&[1, 2, 3], true).unwrap(),
            "synthetic ".repeat(3)
        );
        let mut sequence = Sequence::new(tokenizer.clone());
        let streamed: String = [1, 2, 3]
            .into_iter()
            .map(|id| sequence.append_token(id).unwrap())
            .collect();
        assert_eq!(streamed, "synthetic ".repeat(3));
    }

    #[test]
    fn preparation_still_finds_the_inner_tokenizer() {
        let tokenizer = SyntheticText::wrap(Arc::new(MockTokenizer::new()), "x".into());
        assert!(tokenizer.as_any().is::<MockTokenizer>());
        assert_eq!(
            tokenizer.encode("Hello world", false).unwrap().token_ids(),
            MockTokenizer::new()
                .encode("Hello world", false)
                .unwrap()
                .token_ids()
        );
    }
}
