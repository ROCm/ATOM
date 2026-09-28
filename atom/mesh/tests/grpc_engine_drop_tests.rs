//! Plan D10/D11 — verify Drop on the WorkerStream propagates cancellation
//! to inner sources in both single and PD modes. Uses tokio::sync::mpsc
//! channels as the inner sources so we can observe "channel closed" as
//! the drop signal without needing a real tonic stream.

use tokio::sync::mpsc;

use mesh::routers::grpc::engine::merge_pd_streams;
use mesh::routers::token_handle::engine_error::EngineError;
use mesh::routers::token_handle::test_support::single_from_receiver;
use mesh::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

fn complete_chunk() -> TokenChunk {
    TokenChunk::Complete {
        token_ids: vec![],
        finish_reason: FinishReason::Stop,
        matched_stop: None,
        usage: Usage::default(),
        logprobs: None,
        input_logprobs: None,
        meta: WorkerMeta {
            request_id: "r".to_string(),
            weight_version: None,
            cached_tokens: 0,
        },
    }
}

#[tokio::test]
async fn worker_stream_drop_single() {
    let (tx, rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
    let stream = single_from_receiver(rx);
    tx.send(Ok(complete_chunk())).await.expect("first send ok");
    drop(stream);
    let send_after = tx.send(Ok(complete_chunk())).await;
    assert!(send_after.is_err(), "mpsc must close after stream drop");
}

#[tokio::test]
async fn worker_stream_drop_pd_both() {
    let (p_tx, p_rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
    let (d_tx, d_rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
    let prefill = single_from_receiver(p_rx);
    let decode = single_from_receiver(d_rx);
    let merged = merge_pd_streams(prefill, decode, true);
    drop(merged);
    let p_err = p_tx.send(Ok(complete_chunk())).await;
    let d_err = d_tx.send(Ok(complete_chunk())).await;
    assert!(p_err.is_err(), "prefill channel must close on PD drop");
    assert!(d_err.is_err(), "decode channel must close on PD drop");
}
