//! Build a `StopSequenceDecoder` from request stop parameters.

use std::sync::Arc;

use crate::{
    protocols::common::StringOrArray,
    tokenizer::{stop::StopSequenceDecoderBuilder, traits::Tokenizer, StopSequenceDecoder},
};

/// Create a StopSequenceDecoder from stop parameters
pub(crate) fn create_stop_decoder(
    tokenizer: &Arc<dyn Tokenizer>,
    stop: Option<&StringOrArray>,
    stop_token_ids: Option<&Vec<u32>>,
    skip_special_tokens: bool,
    no_stop_trim: bool,
) -> StopSequenceDecoder {
    // Extract stop sequences
    let stop_sequences: Vec<String> = match stop {
        Some(StringOrArray::String(s)) => vec![s.clone()],
        Some(StringOrArray::Array(arr)) => arr.clone(),
        None => vec![],
    };

    // Build stop sequence decoder
    let mut builder =
        StopSequenceDecoderBuilder::new(tokenizer.clone()).skip_special_tokens(skip_special_tokens);

    // Add stop sequences (visible if no_stop_trim is true, hidden otherwise)
    for seq in stop_sequences {
        builder = if no_stop_trim {
            builder.visible_stop_sequence(seq)
        } else {
            builder.stop_sequence(seq)
        };
    }

    // Add stop token IDs (visible if no_stop_trim is true, hidden otherwise)
    if let Some(token_ids) = stop_token_ids {
        for &token_id in token_ids {
            builder = if no_stop_trim {
                builder.visible_stop_token(token_id)
            } else {
                builder.stop_token(token_id)
            };
        }
    }

    builder.build()
}
