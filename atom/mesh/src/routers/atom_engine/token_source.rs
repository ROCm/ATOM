//! `SeqTokenSource`: a request's sibling sequences as the `TokenChunk`
//! stream the render layer reads -- each sibling's partials, then its
//! `Complete`, siblings in index order.

use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;
use http::StatusCode;

use crate::{
    atom_engine::{Finished, SeqEvent, SeqFailure, SeqHandle},
    routers::token_handle::{
        engine_error::EngineError,
        token_chunk::{FinishReason, TokenChunk, TokenLogprob, TokenLogprobs, Usage, WorkerMeta},
        token_handle::TokenSource,
    },
};

pub(super) struct SeqTokenSource {
    /// Siblings not yet ended; the first is the one being read.
    seqs: std::collections::VecDeque<SeqHandle>,
    sibling: usize,
    request_id: String,
}

impl SeqTokenSource {
    pub fn new(seqs: Vec<SeqHandle>, request_id: String) -> Self {
        Self {
            seqs: seqs.into(),
            sibling: 0,
            request_id,
        }
    }

    fn complete(&self, done: Finished) -> TokenChunk {
        // The prompt is shared: count it once, on the first sibling.
        let prompt_tokens = if self.sibling == 0 {
            done.prompt_tokens
        } else {
            0
        };
        let completion_tokens = done.token_ids.len() as u32;
        let logprobs = done.logprobs.map(|values| TokenLogprobs {
            items: done
                .token_ids
                .iter()
                .zip(values)
                .map(|(&token_id, logprob)| TokenLogprob {
                    token_id,
                    logprob: logprob as f32,
                    decoded_text: None,
                    top: vec![],
                })
                .collect(),
        });
        TokenChunk::Complete {
            token_ids: done.token_ids,
            finish_reason: finish_reason(&done.leave_reason),
            matched_stop: None,
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
            logprobs,
            input_logprobs: None,
            meta: WorkerMeta {
                request_id: self.request_id.clone(),
                weight_version: None,
                cached_tokens: done.cached_tokens,
            },
        }
    }
}

impl Stream for SeqTokenSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(seq) = self.seqs.front_mut() else {
            return Poll::Ready(None);
        };
        let item = match std::task::ready!(seq.poll_event(cx)) {
            Some(SeqEvent::Tokens(token_ids)) => {
                return Poll::Ready(Some(Ok(TokenChunk::Partial {
                    token_ids,
                    logprobs: None,
                })))
            }
            Some(SeqEvent::Finished(done)) => Ok(self.complete(done)),
            Some(SeqEvent::Failed(failure)) => Err(failure.into()),
            None => Err(EngineError::DecodeIncomplete),
        };
        // That sibling has ended; a failure ends the rest with it.
        if item.is_ok() {
            self.seqs.pop_front();
            self.sibling += 1;
        } else {
            self.seqs.clear();
        }
        Poll::Ready(Some(item))
    }
}

impl TokenSource for SeqTokenSource {
    /// Nothing to suppress: an ended `SeqHandle` does not abort on drop.
    fn mark_completed(&mut self) {}
}

impl From<SeqFailure> for EngineError {
    fn from(failure: SeqFailure) -> Self {
        let (status, reason) = match failure {
            SeqFailure::Invalid(r) => (StatusCode::BAD_REQUEST, r),
            SeqFailure::Unavailable(r) => (StatusCode::SERVICE_UNAVAILABLE, r),
            SeqFailure::Internal(r) => (StatusCode::INTERNAL_SERVER_ERROR, r),
        };
        EngineError::Rejected { status, reason }
    }
}

/// The Core's `leave_reason` (`eos`, `stop_sequence`, `stop_<id>`,
/// `max_tokens`, `aborted`, `unschedulable: ...`) as a finish reason.
fn finish_reason(leave_reason: &str) -> FinishReason {
    match leave_reason {
        "max_tokens" | "max_new_tokens" => FinishReason::Length,
        "aborted" => FinishReason::Abort,
        "" | "eos" | "stop_sequence" => FinishReason::Stop,
        r if r.starts_with("stop_") => FinishReason::Stop,
        other => FinishReason::Other(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leave_reasons_map_as_the_python_server_reports_them() {
        assert!(matches!(finish_reason("max_tokens"), FinishReason::Length));
        assert!(matches!(finish_reason("stop_151645"), FinishReason::Stop));
        assert!(matches!(finish_reason("eos"), FinishReason::Stop));
        assert!(matches!(
            finish_reason("unschedulable: too long"),
            FinishReason::Other(_)
        ));
    }
}
