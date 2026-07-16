//! T1–T7 obligations from `docs/2026-05-19-grpc-pd-merge-spec.md` §4 —
//! all merger transitions and invariants exercised against scripted streams.

use std::sync::atomic::Ordering;
use std::time::Duration;

use futures::StreamExt;

use mesh::routers::grpc::engine::merge_pd_streams;
use mesh::routers::token_handle::engine_error::EngineError;
use mesh::routers::token_handle::test_support::{
    pending_forever_stream, scripted_stream_with_drop_observer, scripted_stream_with_poll_counter,
    ScriptedItem,
};
use mesh::routers::token_handle::token_chunk::{
    FinishReason, InputLogprobs, TokenChunk, TokenLogprob, Usage, WorkerMeta,
};

fn meta() -> WorkerMeta {
    WorkerMeta {
        request_id: "r".to_string(),
        weight_version: None,
        cached_tokens: 0,
    }
}

fn partial(token_id: u32) -> TokenChunk {
    TokenChunk::Partial {
        token_ids: vec![token_id],
        logprobs: None,
    }
}

fn complete_with_input_logprobs(ip: Option<InputLogprobs>) -> TokenChunk {
    TokenChunk::Complete {
        token_ids: vec![],
        finish_reason: FinishReason::Stop,
        matched_stop: None,
        usage: Usage::default(),
        logprobs: None,
        input_logprobs: ip,
        meta: meta(),
    }
}

fn ip(n: u32) -> InputLogprobs {
    InputLogprobs {
        items: (0..n)
            .map(|i| TokenLogprob {
                token_id: i,
                logprob: -0.1,
                decoded_text: None,
                top: vec![],
            })
            .collect(),
    }
}

#[tokio::test]
async fn pd_merge_t1_skip_prefill_when_no_input_logprobs() {
    // I2: when need_input_logprobs == false, prefill is never polled.
    let (prefill, prefill_polls) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);
    let (decode, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(10)),
        ScriptedItem::Ok(partial(20)),
        ScriptedItem::Ok(partial(30)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);

    let mut merged = merge_pd_streams(prefill, decode, false);
    let mut yielded = Vec::new();
    while let Some(item) = merged.next().await {
        yielded.push(item.unwrap());
    }
    assert_eq!(yielded.len(), 4, "3 Partial + 1 Complete from decode only");
    assert_eq!(
        prefill_polls.load(Ordering::SeqCst),
        0,
        "prefill must NOT be polled when need_input_logprobs=false"
    );
}

#[tokio::test]
async fn pd_merge_t2_injects_input_logprobs_from_prefill_into_decode_complete() {
    let injected = ip(5);
    let injected_clone = injected.clone();
    let (prefill, _) = scripted_stream_with_poll_counter(vec![ScriptedItem::Ok(
        complete_with_input_logprobs(Some(injected_clone)),
    )]);
    let (decode, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(7)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);

    let mut merged = merge_pd_streams(prefill, decode, true);
    let mut items: Vec<TokenChunk> = Vec::new();
    while let Some(i) = merged.next().await {
        items.push(i.unwrap());
    }
    assert_eq!(items.len(), 2);
    match &items[1] {
        TokenChunk::Complete {
            input_logprobs: Some(actual),
            ..
        } => assert_eq!(actual.items.len(), injected.items.len()),
        other => panic!("expected Complete with injected logprobs, got {other:?}"),
    }
}

#[tokio::test]
async fn pd_merge_t2_no_partial_from_prefill_leaks_to_consumer() {
    // Invariant I1
    let (prefill, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(99)),
        ScriptedItem::Ok(complete_with_input_logprobs(Some(ip(1)))),
    ]);
    let (decode, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(10)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let mut items: Vec<TokenChunk> = Vec::new();
    while let Some(i) = merged.next().await {
        items.push(i.unwrap());
    }
    let token_ids_seen: Vec<u32> = items
        .iter()
        .flat_map(|c| match c {
            TokenChunk::Partial { token_ids, .. } => token_ids.clone(),
            _ => vec![],
        })
        .collect();
    assert!(!token_ids_seen.contains(&99), "prefill Partial 99 leaked");
    assert!(token_ids_seen.contains(&10), "decode Partial missing");
}

#[tokio::test]
async fn pd_merge_t3_prefill_error_propagates_and_decode_dropped() {
    let (prefill, _) = scripted_stream_with_poll_counter(vec![ScriptedItem::Err(
        EngineError::Prefill("OOM".to_string()),
    )]);
    let (decode, decode_polls) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let first = merged.next().await.expect("merger yields one item");
    assert!(matches!(first, Err(EngineError::Prefill(_))));
    let second = merged.next().await;
    assert!(second.is_none(), "terminal state yields None subsequently");
    assert_eq!(
        decode_polls.load(Ordering::SeqCst),
        0,
        "decode must not be polled when prefill fails first"
    );
}

#[tokio::test]
async fn pd_merge_t4_prefill_silent_after_streaming_transition() {
    let (prefill, prefill_polls) = scripted_stream_with_poll_counter(vec![ScriptedItem::Ok(
        complete_with_input_logprobs(Some(ip(2))),
    )]);
    let (decode, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(partial(2)),
        ScriptedItem::Ok(partial(3)),
        ScriptedItem::Ok(partial(4)),
        ScriptedItem::Ok(partial(5)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let mut items: Vec<TokenChunk> = Vec::new();
    while let Some(i) = merged.next().await {
        items.push(i.unwrap());
    }
    assert_eq!(items.len(), 6);
    assert_eq!(
        prefill_polls.load(Ordering::SeqCst),
        1,
        "prefill polled exactly once for its Complete"
    );
}

#[tokio::test]
async fn pd_merge_t5_decode_transport_error_propagates_and_prefill_dropped() {
    let (prefill, _, prefill_drop_observed) =
        scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(complete_with_input_logprobs(
            None,
        ))]);
    let (decode, _, _) = scripted_stream_with_drop_observer(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(partial(2)),
        ScriptedItem::Err(EngineError::Transport(tonic::Status::aborted("dead"))),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let mut yielded: Vec<Result<TokenChunk, EngineError>> = Vec::new();
    while let Some(item) = merged.next().await {
        let is_err = item.is_err();
        yielded.push(item);
        if is_err {
            break;
        }
    }
    assert_eq!(yielded.len(), 3);
    assert!(matches!(yielded[0], Ok(TokenChunk::Partial { .. })));
    assert!(matches!(yielded[1], Ok(TokenChunk::Partial { .. })));
    assert!(matches!(yielded[2], Err(EngineError::Transport(_))));
    drop(merged);
    assert!(
        prefill_drop_observed.load(Ordering::SeqCst),
        "prefill must be dropped after decode error"
    );
}

#[tokio::test]
async fn pd_merge_t5_prefill_early_close_yields_typed_error() {
    let (prefill, _) = scripted_stream_with_poll_counter(Vec::new());
    let (decode, _) = scripted_stream_with_poll_counter(vec![ScriptedItem::Ok(
        complete_with_input_logprobs(None),
    )]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let first = merged.next().await.unwrap();
    assert!(matches!(first, Err(EngineError::PrefillEarlyClose)));
}

#[tokio::test]
async fn pd_merge_t6_consumer_drop_propagates_to_both_upstreams() {
    let (prefill, _, prefill_drop_observed) =
        scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(complete_with_input_logprobs(
            None,
        ))]);
    let (decode, _, decode_drop_observed) = scripted_stream_with_drop_observer(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(partial(2)),
        ScriptedItem::Ok(partial(3)),
        ScriptedItem::Ok(partial(4)),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    for _ in 0..3 {
        let _ = merged.next().await;
    }
    drop(merged);
    assert!(prefill_drop_observed.load(Ordering::SeqCst));
    assert!(decode_drop_observed.load(Ordering::SeqCst));
}

#[tokio::test]
async fn pd_merge_t7_pending_prefill_blocks_decode_until_timeout() {
    let (prefill, _) = pending_forever_stream();
    let (decode, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let timed = tokio::time::timeout(Duration::from_millis(100), merged.next()).await;
    assert!(
        timed.is_err(),
        "T7: merger has NO internal timeout — pending prefill blocks decode yields"
    );
}

#[tokio::test]
async fn pd_merge_decode_incomplete_yields_typed_error() {
    let (prefill, _) = scripted_stream_with_poll_counter(vec![ScriptedItem::Ok(
        complete_with_input_logprobs(None),
    )]);
    let (decode, _) = scripted_stream_with_poll_counter(vec![ScriptedItem::Ok(partial(1))]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let _first = merged.next().await.unwrap().unwrap();
    let second = merged.next().await.unwrap();
    assert!(matches!(second, Err(EngineError::DecodeIncomplete)));
}

#[tokio::test]
async fn pd_merge_prefill_partial_in_waiting_state_silently_dropped() {
    let (prefill, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(99)),
        ScriptedItem::Ok(partial(100)),
        ScriptedItem::Ok(complete_with_input_logprobs(Some(ip(1)))),
    ]);
    let (decode, _) = scripted_stream_with_poll_counter(vec![
        ScriptedItem::Ok(partial(1)),
        ScriptedItem::Ok(complete_with_input_logprobs(None)),
    ]);
    let mut merged = merge_pd_streams(prefill, decode, true);
    let mut items: Vec<TokenChunk> = Vec::new();
    while let Some(i) = merged.next().await {
        items.push(i.unwrap());
    }
    assert_eq!(items.len(), 2);
}
