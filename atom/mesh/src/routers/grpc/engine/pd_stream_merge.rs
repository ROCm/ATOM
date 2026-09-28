//! `merge_pd_streams` — state machine that combines prefill + decode worker
//! streams into one `TokenHandle`.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;

use crate::routers::token_handle::engine_error::EngineError;
use crate::routers::token_handle::token_chunk::{InputLogprobs, TokenChunk};
use crate::routers::token_handle::token_handle::{TokenHandle, TokenSource};

pub fn merge_pd_streams(
    prefill: TokenHandle,
    decode: TokenHandle,
    need_input_logprobs: bool,
) -> TokenHandle {
    let state = if need_input_logprobs {
        State::WaitingPrefill
    } else {
        State::Streaming {
            pending_input_logprobs: None,
        }
    };
    TokenHandle::new(PdMerger {
        prefill: Some(prefill),
        decode: Some(decode),
        state,
    })
}

struct PdMerger {
    prefill: Option<TokenHandle>,
    decode: Option<TokenHandle>,
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
                            // Drop prefill Partials while waiting; only the Complete carries input_logprobs.
                            continue;
                        }
                        Poll::Ready(Some(Ok(TokenChunk::Complete { input_logprobs, .. }))) => {
                            this.state = State::Streaming {
                                pending_input_logprobs: input_logprobs,
                            };
                            continue;
                        }
                        Poll::Ready(Some(Err(e))) => {
                            // Prefill failed: cancel decode by dropping (no mark_completed → H2 RST).
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
                            // mark_completed BEFORE drop, so prefill exits cleanly instead of RST.
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
                            // Decode failed: cancel prefill by dropping (no mark_completed → H2 RST).
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
