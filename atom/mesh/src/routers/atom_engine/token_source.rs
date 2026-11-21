//! `SeqTokenSource`: a request's sibling sequences as the `TokenChunk`
//! stream the render layer reads -- each sibling's partials, then its
//! `Complete`, siblings in index order.

use std::{
    pin::Pin,
    task::{Context, Poll},
    time::Instant,
};

use futures::Stream;
use http::StatusCode;

use crate::{
    atom_engine::{Finished, SeqEvent, SeqFailure, SeqHandle, SeqTimes},
    routers::token_handle::{
        engine_error::EngineError,
        token_chunk::{
            FinishReason, SeqTiming, TokenChunk, TokenLogprob, TokenLogprobs, Usage, WorkerMeta,
        },
        token_handle::TokenSource,
    },
};

pub(super) struct SeqTokenSource {
    /// Siblings not yet ended; the first is the one being read.
    seqs: std::collections::VecDeque<SeqHandle>,
    request_id: String,
}

impl SeqTokenSource {
    pub fn new(seqs: Vec<SeqHandle>, request_id: String) -> Self {
        Self {
            seqs: seqs.into(),
            request_id,
        }
    }

    /// Every sibling reports the prompt it ran on, as SGLang's do; the
    /// OpenAI renderers count it once for the request.
    fn complete(&self, done: Finished) -> TokenChunk {
        let prompt_tokens = done.prompt_tokens;
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
                timing: Some(seq_timing(&done.times, completion_tokens)),
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

/// `ttft`, `tpot` and `latency` as `api_server.py` computes them: from the
/// submission, `tpot` over the tokens after the first, 0 where there is none.
fn seq_timing(times: &SeqTimes, completion_tokens: u32) -> SeqTiming {
    let since = |t: Instant| t.duration_since(times.submitted).as_secs_f64();
    let tpot = match (times.first_token, times.last_token) {
        (Some(first), Some(last)) if completion_tokens > 1 => {
            last.duration_since(first).as_secs_f64() / (completion_tokens - 1) as f64
        }
        _ => 0.0,
    };
    SeqTiming {
        ttft: times.first_token.map(since).unwrap_or(0.0),
        tpot,
        latency: since(times.ended),
    }
}

/// The Core's `leave_reason` (`eos`, `stop_sequence`, `stop_<id>`,
/// `max_tokens`, `aborted`, `unschedulable: ...`) as `openai_stop_reason`
/// reports it: running out of tokens is `length`, every other end `stop`.
fn finish_reason(leave_reason: &str) -> FinishReason {
    match leave_reason {
        "max_tokens" | "max_new_tokens" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_is_measured_from_the_submission() {
        let submitted = Instant::now();
        let at = |ms: u64| submitted + std::time::Duration::from_millis(ms);
        let times = SeqTimes {
            submitted,
            first_token: Some(at(100)),
            last_token: Some(at(400)),
            ended: at(500),
        };
        let timing = seq_timing(&times, 4);
        assert!((timing.ttft - 0.1).abs() < 1e-9, "{timing:?}");
        assert!((timing.tpot - 0.1).abs() < 1e-9, "{timing:?}");
        assert!((timing.latency - 0.5).abs() < 1e-9, "{timing:?}");
        // One token has no time per output token; none, no first token either.
        assert_eq!(seq_timing(&times, 1).tpot, 0.0);
        let none = SeqTimes {
            first_token: None,
            last_token: None,
            ..times
        };
        assert_eq!(seq_timing(&none, 0).ttft, 0.0);
    }

    #[test]
    fn leave_reasons_map_as_the_python_server_reports_them() {
        assert!(matches!(finish_reason("max_tokens"), FinishReason::Length));
        assert!(matches!(finish_reason("stop_151645"), FinishReason::Stop));
        assert!(matches!(finish_reason("eos"), FinishReason::Stop));
        // Never a raw Core string in `finish_reason`.
        for reason in ["aborted", "unschedulable: too long", "rejected: no"] {
            assert!(
                matches!(finish_reason(reason), FinishReason::Stop),
                "{reason}"
            );
        }
    }
}
