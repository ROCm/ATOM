//! Convert neutral `TokenLogprobs` / `InputLogprobs` into the OpenAI-shaped
//! `ChatLogProbs` and SGLang-shaped `Vec<Vec<Option<f64>>>` the render paths
//! emit on the wire.

use std::sync::Arc;

use crate::{
    protocols::common::{ChatLogProbs, ChatLogProbsContent, TopLogProb},
    routers::token_handle::token_chunk::{InputLogprobs, TokenLogprobs},
    tokenizer::traits::Tokenizer,
};

pub(crate) fn token_logprobs_to_chat(
    lp: &TokenLogprobs,
    tokenizer: &Arc<dyn Tokenizer>,
) -> ChatLogProbs {
    let mut content_items = Vec::with_capacity(lp.items.len());
    for item in &lp.items {
        let token_text = item.decoded_text.clone().unwrap_or_else(|| {
            tokenizer
                .decode(&[item.token_id], false)
                .unwrap_or_else(|_| format!("<token_{}>", item.token_id))
        });
        let bytes = Some(token_text.as_bytes().to_vec());
        let top_logprobs = item
            .top
            .iter()
            .map(|(tid, lp, decoded)| {
                let txt = decoded.clone().unwrap_or_else(|| {
                    tokenizer
                        .decode(&[*tid], false)
                        .unwrap_or_else(|_| format!("<token_{}>", tid))
                });
                let bytes = Some(txt.as_bytes().to_vec());
                TopLogProb {
                    token: txt,
                    logprob: *lp,
                    bytes,
                }
            })
            .collect();
        content_items.push(ChatLogProbsContent {
            token: token_text,
            logprob: item.logprob,
            bytes,
            top_logprobs,
        });
    }
    ChatLogProbs::Detailed {
        content: (!content_items.is_empty()).then_some(content_items),
    }
}

pub(crate) fn output_logprobs_to_generate(lp: &TokenLogprobs) -> Vec<Vec<Option<f64>>> {
    lp.items
        .iter()
        .map(|item| vec![Some(item.logprob as f64), Some(item.token_id as f64)])
        .collect()
}

pub(crate) fn input_logprobs_to_generate(lp: &InputLogprobs) -> Vec<Vec<Option<f64>>> {
    lp.items
        .iter()
        .map(|item| vec![Some(item.logprob as f64), Some(item.token_id as f64)])
        .collect()
}
