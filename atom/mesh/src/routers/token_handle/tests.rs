//! Tests for `routers::token_handle::*` — boundary types plus the `Drop`
//! propagation contract on `TokenHandle`.

mod a_token_chunk {
    use crate::routers::token_handle::token_chunk::{
        FinishReason, InputLogprobs, MatchedStop, TokenChunk, TokenLogprob, TokenLogprobs, Usage,
        WorkerMeta,
    };

    fn meta() -> WorkerMeta {
        WorkerMeta {
            request_id: "req-1".to_string(),
            weight_version: Some("w1".to_string()),
            cached_tokens: 0,
        }
    }

    fn usage(prompt: u32, comp: u32) -> Usage {
        Usage {
            prompt_tokens: prompt,
            completion_tokens: comp,
            total_tokens: prompt + comp,
        }
    }

    #[test]
    fn test_partial_chunk_carries_token_ids_and_logprobs_none() {
        let c = TokenChunk::Partial {
            token_ids: vec![10, 20],
            logprobs: None,
        };
        match c {
            TokenChunk::Partial {
                token_ids,
                logprobs,
            } => {
                assert_eq!(token_ids, vec![10, 20]);
                assert!(logprobs.is_none());
            }
            _ => panic!("expected Partial"),
        }
    }

    #[test]
    fn test_partial_chunk_with_logprobs() {
        let lp = TokenLogprobs {
            items: vec![TokenLogprob {
                token_id: 10,
                logprob: -0.5,
                decoded_text: Some(" hi".to_string()),
                top: vec![(10, -0.5, Some(" hi".to_string()))],
            }],
        };
        let c = TokenChunk::Partial {
            token_ids: vec![10],
            logprobs: Some(lp),
        };
        match c {
            TokenChunk::Partial {
                logprobs: Some(lps),
                ..
            } => {
                assert_eq!(lps.items.len(), 1);
                assert_eq!(lps.items[0].token_id, 10);
            }
            _ => panic!("expected Partial with logprobs"),
        }
    }

    #[test]
    fn test_complete_chunk_carries_all_fields() {
        let c = TokenChunk::Complete {
            token_ids: vec![1, 2, 3],
            finish_reason: FinishReason::Stop,
            matched_stop: Some(MatchedStop::Str("<eot>".to_string())),
            usage: usage(5, 3),
            logprobs: None,
            input_logprobs: None,
            meta: meta(),
        };
        match c {
            TokenChunk::Complete {
                token_ids,
                finish_reason,
                matched_stop,
                usage,
                meta,
                ..
            } => {
                assert_eq!(token_ids.len(), 3);
                assert!(matches!(finish_reason, FinishReason::Stop));
                assert!(matches!(matched_stop, Some(MatchedStop::Str(s)) if s == "<eot>"));
                assert_eq!(usage.total_tokens, 8);
                assert_eq!(meta.request_id, "req-1");
            }
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_complete_chunk_input_logprobs_is_optional() {
        // input_logprobs is Option: filled from worker Complete in single-mode,
        // injected by merge_pd_streams in PD mode. None is a valid state either way.
        let lp = InputLogprobs {
            items: vec![TokenLogprob {
                token_id: 1,
                logprob: -1.0,
                decoded_text: None,
                top: vec![],
            }],
        };
        let c = TokenChunk::Complete {
            token_ids: vec![1],
            finish_reason: FinishReason::Length,
            matched_stop: None,
            usage: usage(1, 0),
            logprobs: None,
            input_logprobs: Some(lp),
            meta: meta(),
        };
        if let TokenChunk::Complete {
            input_logprobs: Some(ip),
            ..
        } = c
        {
            assert_eq!(ip.items.len(), 1);
        } else {
            panic!("expected Complete with input_logprobs");
        }
    }

    #[test]
    fn test_matched_stop_str_variant() {
        let s = MatchedStop::Str("</s>".to_string());
        match s {
            MatchedStop::Str(v) => assert_eq!(v, "</s>"),
            _ => panic!("expected Str"),
        }
    }

    #[test]
    fn test_matched_stop_token_id_variant() {
        let s = MatchedStop::TokenId(2);
        match s {
            MatchedStop::TokenId(t) => assert_eq!(t, 2),
            _ => panic!("expected TokenId"),
        }
    }

    #[test]
    fn test_finish_reason_variants_distinct() {
        let reasons = vec![
            FinishReason::Stop,
            FinishReason::Length,
            FinishReason::ContentFilter,
            FinishReason::ToolCalls,
            FinishReason::Abort,
            FinishReason::Other("custom".to_string()),
        ];
        // No accidental Eq collapse — at least 6 distinct kinds at the discriminant level.
        let kinds: std::collections::HashSet<_> =
            reasons.iter().map(|r| std::mem::discriminant(r)).collect();
        assert_eq!(kinds.len(), 6);
    }

    #[test]
    fn test_token_logprob_top_alternatives_list_shape() {
        let tl = TokenLogprob {
            token_id: 9,
            logprob: -0.2,
            decoded_text: Some(" cat".to_string()),
            top: vec![
                (9, -0.2, Some(" cat".to_string())),
                (8, -1.4, Some(" dog".to_string())),
            ],
        };
        assert_eq!(tl.top.len(), 2);
        assert_eq!(tl.top[1].0, 8);
    }
}

mod b_engine_error {
    use crate::routers::token_handle::engine_error::EngineError;

    #[test]
    fn test_engine_error_seven_variants_exist() {
        let _v1 = EngineError::Transport(tonic::Status::cancelled("c"));
        let _v2 = EngineError::Prefill(Default::default());
        let _v3 = EngineError::DecodeError(Default::default());
        let _v4 = EngineError::PrefillEarlyClose;
        let _v5 = EngineError::DecodeIncomplete;
        let _v6 = EngineError::ConnectionAcquireFailed("boom".to_string());
        let _v7 = EngineError::RequestBuildFailed("bad".to_string());
    }

    #[test]
    fn test_engine_error_transport_display_includes_status() {
        let e = EngineError::Transport(tonic::Status::unavailable("dead worker"));
        let s = format!("{e}");
        assert!(s.contains("dead worker") || s.contains("unavailable"));
    }

    #[test]
    fn test_engine_error_connection_acquire_failed_includes_reason() {
        let e = EngineError::ConnectionAcquireFailed("pool exhausted".to_string());
        assert!(format!("{e}").contains("pool exhausted"));
    }

    #[test]
    fn test_engine_error_request_build_failed_includes_reason() {
        let e = EngineError::RequestBuildFailed("invalid stop seq".to_string());
        assert!(format!("{e}").contains("invalid stop seq"));
    }

    #[test]
    fn test_engine_error_prefill_early_close_unit() {
        // Unit variant must Display as a stable identifier (no payload).
        let s = format!("{}", EngineError::PrefillEarlyClose);
        assert!(s.to_lowercase().contains("prefill"));
    }

    #[test]
    fn test_engine_error_decode_incomplete_unit() {
        let s = format!("{}", EngineError::DecodeIncomplete);
        assert!(s.to_lowercase().contains("decode"));
    }

    #[test]
    fn test_engine_error_prefill_display_includes_message() {
        let e = EngineError::Prefill("kv mismatch".to_string());
        assert!(format!("{e}").contains("kv mismatch"));
    }

    #[test]
    fn test_engine_error_decode_error_display_includes_message() {
        let e = EngineError::DecodeError("xx".to_string());
        assert!(format!("{e}").contains("xx"));
    }

    #[test]
    fn test_engine_error_debug_renders_variant_name() {
        let e = EngineError::PrefillEarlyClose;
        let d = format!("{:?}", e);
        assert!(d.contains("PrefillEarlyClose"));
    }

    #[test]
    fn test_engine_error_source_transport_is_some() {
        use std::error::Error;
        let e = EngineError::Transport(tonic::Status::cancelled("c"));
        assert!(e.source().is_some());
    }

    #[test]
    fn test_engine_error_source_non_transport_is_none() {
        use std::error::Error;
        assert!(EngineError::PrefillEarlyClose.source().is_none());
        assert!(EngineError::DecodeIncomplete.source().is_none());
        assert!(EngineError::Prefill("x".to_string()).source().is_none());
        assert!(EngineError::DecodeError("y".to_string()).source().is_none());
        assert!(EngineError::ConnectionAcquireFailed("z".to_string())
            .source()
            .is_none());
        assert!(EngineError::RequestBuildFailed("w".to_string())
            .source()
            .is_none());
    }
}

mod c_token_handle {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use futures::StreamExt;
    use tokio::sync::mpsc;

    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};
    use crate::routers::token_handle::token_handle::TokenHandle;

    fn complete_chunk() -> TokenChunk {
        TokenChunk::Complete {
            token_ids: vec![],
            finish_reason: FinishReason::Stop,
            matched_stop: None,
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
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
    async fn test_single_stream_yields_scripted_chunks_in_order() {
        let chunks = vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Ok(TokenChunk::Partial {
                token_ids: vec![2],
                logprobs: None,
            }),
            Ok(complete_chunk()),
        ];
        let mut stream = synthetic_single_stream(chunks);
        let collected: Vec<_> = (&mut stream).collect::<Vec<_>>().await;
        assert_eq!(collected.len(), 3);
        assert!(matches!(collected[2], Ok(TokenChunk::Complete { .. })));
    }

    #[tokio::test]
    async fn test_single_stream_propagates_engine_error() {
        let chunks = vec![
            Ok(TokenChunk::Partial {
                token_ids: vec![1],
                logprobs: None,
            }),
            Err(EngineError::Transport(tonic::Status::cancelled("c"))),
        ];
        let mut stream = synthetic_single_stream(chunks);
        let first = stream.next().await.unwrap();
        assert!(first.is_ok());
        let second = stream.next().await.unwrap();
        assert!(matches!(second, Err(EngineError::Transport(_))));
    }

    #[tokio::test]
    async fn test_single_stream_drop_closes_underlying() {
        // Synthetic single stream is backed by an mpsc channel; if we drop the
        // TokenHandle the sender's receiver count should drop to zero so the
        // sender can observe send failure on the next push.
        let (tx, rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
        let stream = crate::routers::token_handle::test_support::single_from_receiver(rx);
        tx.send(Ok(complete_chunk())).await.expect("first send ok");
        drop(stream);
        let send_after_drop = tx.send(Ok(complete_chunk())).await;
        assert!(send_after_drop.is_err(), "expected mpsc closed after drop");
    }

    #[tokio::test]
    async fn test_pd_stream_drop_closes_both_inner_streams() {
        let (p_tx, p_rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
        let (d_tx, d_rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
        let prefill = crate::routers::token_handle::test_support::single_from_receiver(p_rx);
        let decode = crate::routers::token_handle::test_support::single_from_receiver(d_rx);
        let merged = TokenHandle::pd(prefill, decode);
        drop(merged);
        let p_err = p_tx.send(Ok(complete_chunk())).await;
        let d_err = d_tx.send(Ok(complete_chunk())).await;
        assert!(p_err.is_err(), "prefill channel must close on PD drop");
        assert!(d_err.is_err(), "decode channel must close on PD drop");
    }

    #[tokio::test]
    async fn test_mark_completed_propagates_to_single_source() {
        let chunks = vec![Ok(complete_chunk())];
        let items: Vec<crate::routers::token_handle::test_support::ScriptedItem> = chunks
            .into_iter()
            .map(|r| match r {
                Ok(c) => crate::routers::token_handle::test_support::ScriptedItem::Ok(c),
                Err(e) => crate::routers::token_handle::test_support::ScriptedItem::Err(e),
            })
            .collect();
        let (mut stream, mark_obs, _drop_obs) =
            crate::routers::token_handle::test_support::scripted_with_mark_completed(items);
        stream.mark_completed();
        assert!(mark_obs.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_mark_completed_propagates_to_pd_pair() {
        let p_items =
            vec![crate::routers::token_handle::test_support::ScriptedItem::Ok(complete_chunk())];
        let d_items =
            vec![crate::routers::token_handle::test_support::ScriptedItem::Ok(complete_chunk())];
        let (p_stream, p_mark, _p_drop) =
            crate::routers::token_handle::test_support::scripted_with_mark_completed(p_items);
        let (d_stream, d_mark, _d_drop) =
            crate::routers::token_handle::test_support::scripted_with_mark_completed(d_items);
        let mut merged = TokenHandle::pd(p_stream, d_stream);
        merged.mark_completed();
        assert!(p_mark.load(Ordering::SeqCst));
        assert!(d_mark.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_pd_stream_poll_uses_decode_arm() {
        use futures::StreamExt;
        let (p_tx, p_rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
        let (d_tx, d_rx) = mpsc::channel::<Result<TokenChunk, EngineError>>(1);
        let prefill = crate::routers::token_handle::test_support::single_from_receiver(p_rx);
        let decode = crate::routers::token_handle::test_support::single_from_receiver(d_rx);
        let mut merged = TokenHandle::pd(prefill, decode);
        drop(p_tx);
        d_tx.send(Ok(complete_chunk())).await.unwrap();
        drop(d_tx);
        let item = merged.next().await.unwrap();
        assert!(matches!(item, Ok(TokenChunk::Complete { .. })));
        assert!(merged.next().await.is_none());
    }

    #[tokio::test]
    async fn test_synthetic_drop_observer_counts_invocations() {
        // The test_support helper exposes a counter incremented when the wrapping
        // adapter is dropped. Lets PD tests assert "both upstreams aborted exactly
        // once" without leaning on side-channel timing.
        let counter = Arc::new(AtomicUsize::new(0));
        let s = crate::routers::token_handle::test_support::counted_drop_stream(counter.clone());
        drop(s);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}

mod d_test_support_self_test {
    // Smoke tests for the synthetic_stream helpers, so failures elsewhere can
    // be triaged as real bugs vs broken fixtures.
    use futures::StreamExt;

    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::{
        synthetic_pd_stream, synthetic_single_stream,
    };
    use crate::routers::token_handle::token_chunk::TokenChunk;

    #[tokio::test]
    async fn test_synthetic_single_stream_empty_returns_zero_items() {
        let mut s = synthetic_single_stream(Vec::<Result<TokenChunk, EngineError>>::new());
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn test_synthetic_pd_stream_preserves_both_arms() {
        let p = vec![Ok(TokenChunk::Partial {
            token_ids: vec![10],
            logprobs: None,
        })];
        let d = vec![Ok(TokenChunk::Partial {
            token_ids: vec![20],
            logprobs: None,
        })];
        let pair = synthetic_pd_stream(p, d);
        let (mut prefill, mut decode) = pair;
        let p0 = prefill.next().await.unwrap().unwrap();
        let d0 = decode.next().await.unwrap().unwrap();
        match (p0, d0) {
            (
                TokenChunk::Partial { token_ids: pi, .. },
                TokenChunk::Partial { token_ids: di, .. },
            ) => {
                assert_eq!(pi, vec![10]);
                assert_eq!(di, vec![20]);
            }
            _ => panic!("expected two Partials"),
        }
    }
}
