//! `merge_pd_streams` — state machine that combines prefill + decode worker
//! streams into one `WorkerStream`. See
//! `docs/2026-05-19-grpc-pd-merge-spec.md` §3 for the binding semantics.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

use crate::routers::worker_stream::engine_error::EngineError;
use crate::routers::worker_stream::token_chunk::{InputLogprobs, TokenChunk};
use crate::routers::worker_stream::worker_stream::{TokenSource, WorkerStream};

pub fn merge_pd_streams(
    prefill: WorkerStream,
    decode: WorkerStream,
    need_input_logprobs: bool,
) -> WorkerStream {
    let state = if need_input_logprobs {
        State::WaitingPrefill
    } else {
        State::Streaming {
            pending_input_logprobs: None,
        }
    };
    WorkerStream::new(PdMerger {
        prefill: Some(prefill),
        decode: Some(decode),
        state,
    })
}

struct PdMerger {
    prefill: Option<WorkerStream>,
    decode: Option<WorkerStream>,
    state: State,
}

enum State {
    WaitingPrefill,
    Streaming {
        pending_input_logprobs: Option<InputLogprobs>,
    },
    Terminal,
}

impl Stream for PdMerger {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match &mut this.state {
                State::Terminal => return Poll::Ready(None),
                State::WaitingPrefill => {
                    let prefill = this
                        .prefill
                        .as_mut()
                        .expect("WaitingPrefill invariant: prefill is Some");
                    match Pin::new(prefill).poll_next(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Some(Ok(TokenChunk::Partial { .. }))) => {
                            // I1: prefill Partials in WaitingPrefill are silently dropped.
                            continue;
                        }
                        Poll::Ready(Some(Ok(TokenChunk::Complete { input_logprobs, .. }))) => {
                            this.state = State::Streaming {
                                pending_input_logprobs: input_logprobs,
                            };
                            // Stay in loop; pull decode next.
                            continue;
                        }
                        Poll::Ready(Some(Err(e))) => {
                            // I5: drop decode without mark_completed.
                            this.decode = None;
                            this.prefill = None;
                            this.state = State::Terminal;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(None) => {
                            this.decode = None;
                            this.prefill = None;
                            this.state = State::Terminal;
                            return Poll::Ready(Some(Err(EngineError::PrefillEarlyClose)));
                        }
                    }
                }
                State::Streaming {
                    pending_input_logprobs,
                } => {
                    let decode = this
                        .decode
                        .as_mut()
                        .expect("Streaming invariant: decode is Some");
                    match Pin::new(decode).poll_next(cx) {
                        Poll::Pending => return Poll::Pending,
                        Poll::Ready(Some(Ok(partial @ TokenChunk::Partial { .. }))) => {
                            return Poll::Ready(Some(Ok(partial)));
                        }
                        Poll::Ready(Some(Ok(TokenChunk::Complete {
                            token_ids,
                            finish_reason,
                            matched_stop,
                            usage,
                            logprobs,
                            input_logprobs: decode_input_logprobs,
                            meta,
                        }))) => {
                            let merged_input_logprobs =
                                pending_input_logprobs.take().or(decode_input_logprobs);
                            // I4: mark prefill completed BEFORE dropping it.
                            if let Some(p) = this.prefill.as_mut() {
                                p.mark_completed();
                            }
                            this.prefill = None;
                            this.decode = None;
                            this.state = State::Terminal;
                            return Poll::Ready(Some(Ok(TokenChunk::Complete {
                                token_ids,
                                finish_reason,
                                matched_stop,
                                usage,
                                logprobs,
                                input_logprobs: merged_input_logprobs,
                                meta,
                            })));
                        }
                        Poll::Ready(Some(Err(e))) => {
                            // I5: drop prefill without mark_completed.
                            this.prefill = None;
                            this.decode = None;
                            this.state = State::Terminal;
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(None) => {
                            this.prefill = None;
                            this.decode = None;
                            this.state = State::Terminal;
                            return Poll::Ready(Some(Err(EngineError::DecodeIncomplete)));
                        }
                    }
                }
            }
        }
    }
}

impl TokenSource for PdMerger {
    fn mark_completed(&mut self) {
        if let Some(d) = self.decode.as_mut() {
            d.mark_completed();
        }
    }
}
