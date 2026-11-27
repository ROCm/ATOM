//! `TokenHandle`: the only stream type the render layer sees.
//!
//! Single mode holds one source; PD mode holds both upstreams. Drop in
//! either case propagates cancellation downward (single mode → the inner
//! tonic stream's `AbortOnDropStream`; PD mode → both upstreams' Drop).

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;

use super::engine_error::EngineError;
use super::token_chunk::TokenChunk;
use crate::core::worker::PrefillCharge;

/// Source of `Result<TokenChunk, EngineError>` items that can also be told
/// it completed cleanly (so upstream `AbortOnDropStream`s suppress their
/// cancellation RPC).
pub trait TokenSource: Stream<Item = Result<TokenChunk, EngineError>> + Send + Unpin {
    fn mark_completed(&mut self);
}

pub struct TokenHandle {
    inner: Inner,
    prefill: Option<Arc<PrefillCharge>>,
}

enum Inner {
    /// One backend stream (single-mode dispatch, or the merger's output
    /// state machine wrapped as a single source).
    Single(Box<dyn TokenSource>),
    /// Test-only fixture that binds two upstreams' lifetimes together so
    /// dropping the outer `TokenHandle` drops both. Production PD
    /// dispatch wraps a `PdMerger` in the `Single` variant — not this one.
    #[cfg(test)]
    Pair {
        prefill: Box<TokenHandle>,
        decode: Box<TokenHandle>,
    },
}

impl TokenHandle {
    pub fn new<S>(source: S) -> Self
    where
        S: TokenSource + 'static,
    {
        Self {
            inner: Inner::Single(Box::new(source)),
            prefill: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn pd(prefill: TokenHandle, decode: TokenHandle) -> Self {
        Self {
            inner: Inner::Pair {
                prefill: Box::new(prefill),
                decode: Box::new(decode),
            },
            prefill: None,
        }
    }

    /// Release `charge` at the first item: the engine has finished the prefill.
    pub fn with_prefill_charge(mut self, charge: Arc<PrefillCharge>) -> Self {
        self.prefill = Some(charge);
        self
    }

    /// Suppress upstream cancellation on clean exit. Single mode: forwards
    /// to the inner source.
    pub fn mark_completed(&mut self) {
        match &mut self.inner {
            Inner::Single(s) => s.mark_completed(),
            #[cfg(test)]
            Inner::Pair { prefill, decode } => {
                prefill.mark_completed();
                decode.mark_completed();
            }
        }
    }
}

impl Stream for TokenHandle {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let item = match &mut this.inner {
            Inner::Single(s) => Pin::new(&mut **s).poll_next(cx),
            #[cfg(test)]
            Inner::Pair { decode, .. } => Pin::new(&mut **decode).poll_next(cx),
        };
        if let Poll::Ready(Some(_)) = item {
            if let Some(charge) = this.prefill.take() {
                charge.release();
            }
        }
        item
    }
}
