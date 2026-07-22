//! Synthetic `TokenHandle` builders used by `token_handle/tests.rs`,
//! `tests/grpc_pd_merge_tests.rs`, and `tests/grpc_engine_drop_tests.rs`.
//!
//! Always-compiled (not `cfg(test)`-gated) so integration-test binaries can
//! reach the helpers via `mesh::routers::token_handle::test_support::*`.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::Stream;
use tokio::sync::mpsc;

use super::engine_error::EngineError;
use super::token_chunk::TokenChunk;
use super::token_handle::{TokenHandle, TokenSource};

#[derive(Debug)]
pub enum ScriptedItem {
    Ok(TokenChunk),
    Err(EngineError),
}

struct ScriptedSource {
    items: VecDeque<ScriptedItem>,
    poll_count: Arc<AtomicUsize>,
    mark_completed_observed: Arc<AtomicBool>,
    drop_observed: Arc<AtomicBool>,
}

impl ScriptedSource {
    fn new(items: Vec<ScriptedItem>) -> Self {
        Self {
            items: items.into(),
            poll_count: Arc::new(AtomicUsize::new(0)),
            mark_completed_observed: Arc::new(AtomicBool::new(false)),
            drop_observed: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl Drop for ScriptedSource {
    fn drop(&mut self) {
        self.drop_observed.store(true, Ordering::SeqCst);
    }
}

impl Stream for ScriptedSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.poll_count.fetch_add(1, Ordering::SeqCst);
        let next = match self.items.pop_front() {
            Some(ScriptedItem::Ok(c)) => Some(Ok(c)),
            Some(ScriptedItem::Err(e)) => Some(Err(e)),
            None => None,
        };
        Poll::Ready(next)
    }
}

impl TokenSource for ScriptedSource {
    fn mark_completed(&mut self) {
        self.mark_completed_observed.store(true, Ordering::SeqCst);
    }
}

pub fn scripted_stream_with_poll_counter(
    items: Vec<ScriptedItem>,
) -> (TokenHandle, Arc<AtomicUsize>) {
    let src = ScriptedSource::new(items);
    let counter = src.poll_count.clone();
    (TokenHandle::new(src), counter)
}

pub fn scripted_stream_with_drop_observer(
    items: Vec<ScriptedItem>,
) -> (TokenHandle, Arc<AtomicUsize>, Arc<AtomicBool>) {
    let src = ScriptedSource::new(items);
    let poll = src.poll_count.clone();
    let drop_obs = src.drop_observed.clone();
    (TokenHandle::new(src), poll, drop_obs)
}

pub fn scripted_with_mark_completed(
    items: Vec<ScriptedItem>,
) -> (TokenHandle, Arc<AtomicBool>, Arc<AtomicBool>) {
    let src = ScriptedSource::new(items);
    let mark_obs = src.mark_completed_observed.clone();
    let drop_obs = src.drop_observed.clone();
    (TokenHandle::new(src), mark_obs, drop_obs)
}

pub fn synthetic_single_stream(chunks: Vec<Result<TokenChunk, EngineError>>) -> TokenHandle {
    let items = chunks
        .into_iter()
        .map(|r| match r {
            Ok(c) => ScriptedItem::Ok(c),
            Err(e) => ScriptedItem::Err(e),
        })
        .collect();
    let (s, _) = scripted_stream_with_poll_counter(items);
    s
}

pub fn synthetic_pd_stream(
    prefill: Vec<Result<TokenChunk, EngineError>>,
    decode: Vec<Result<TokenChunk, EngineError>>,
) -> (TokenHandle, TokenHandle) {
    (
        synthetic_single_stream(prefill),
        synthetic_single_stream(decode),
    )
}

struct ChannelSource {
    rx: mpsc::Receiver<Result<TokenChunk, EngineError>>,
}

impl Stream for ChannelSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl TokenSource for ChannelSource {
    fn mark_completed(&mut self) {}
}

pub fn single_from_receiver(rx: mpsc::Receiver<Result<TokenChunk, EngineError>>) -> TokenHandle {
    TokenHandle::new(ChannelSource { rx })
}

struct CountedDropSource {
    counter: Arc<AtomicUsize>,
}

impl Drop for CountedDropSource {
    fn drop(&mut self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }
}

impl Stream for CountedDropSource {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Ready(None)
    }
}

impl TokenSource for CountedDropSource {
    fn mark_completed(&mut self) {}
}

pub fn counted_drop_stream(counter: Arc<AtomicUsize>) -> TokenHandle {
    TokenHandle::new(CountedDropSource { counter })
}

struct PendingForever {
    poll_count: Arc<AtomicUsize>,
}

impl Stream for PendingForever {
    type Item = Result<TokenChunk, EngineError>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.poll_count.fetch_add(1, Ordering::SeqCst);
        Poll::Pending
    }
}

impl TokenSource for PendingForever {
    fn mark_completed(&mut self) {}
}

pub fn pending_forever_stream() -> (TokenHandle, Arc<AtomicUsize>) {
    let counter = Arc::new(AtomicUsize::new(0));
    let src = PendingForever {
        poll_count: counter.clone(),
    };
    (TokenHandle::new(src), counter)
}
