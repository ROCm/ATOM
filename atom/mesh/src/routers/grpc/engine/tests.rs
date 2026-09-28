mod a_payload_to_proto {
    use crate::protocols::common::StringOrArray;
    use crate::routers::grpc::engine::payload_to_proto::{to_sglang_proto, to_vllm_proto};
    use crate::routers::prepare::generation_payload::{
        GenerationPayload, LogprobConfig, PdMetadata, SamplingParams, StopConfig,
    };

    fn default_sampling() -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            max_new_tokens: Some(64),
            min_new_tokens: 0,
            n: 1,
            ignore_eos: false,
        }
    }

    fn default_logprob() -> LogprobConfig {
        LogprobConfig {
            return_logprob: false,
            top_logprobs_num: 0,
            logprob_start_len: -1,
            token_ids_logprob: Vec::new(),
            input_logprobs: false,
        }
    }

    fn default_stop() -> StopConfig {
        StopConfig {
            stop: None,
            stop_token_ids: None,
            skip_special_tokens: true,
            no_stop_trim: false,
        }
    }

    fn base_payload() -> GenerationPayload {
        GenerationPayload {
            request_id: "snap-A".to_string(),
            text: "hi".to_string(),
            token_ids: vec![1, 2, 3],
            sampling: default_sampling(),
            stop: default_stop(),
            logprob: default_logprob(),
            tool_constraints: None,
            pd_metadata: None,
            stream: false,
            return_hidden_states: false,
            log_metrics: false,
        }
    }

    #[test]
    fn test_sampling_params_temperature_threads_through() {
        let mut p = base_payload();
        p.sampling.temperature = 0.123_456;
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().expect("sampling_params");
        assert!((s.temperature - 0.123_456).abs() < f32::EPSILON);
    }

    #[test]
    fn test_top_k_minus_one_disables_filter() {
        let mut p = base_payload();
        p.sampling.top_k = -1;
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert_eq!(s.top_k, -1);
    }

    #[test]
    fn test_max_new_tokens_threads_through() {
        let mut p = base_payload();
        p.sampling.max_new_tokens = Some(999);
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert_eq!(s.max_new_tokens, Some(999));
    }

    #[test]
    fn test_stop_string_threads_through_as_string() {
        let mut p = base_payload();
        p.stop.stop = Some(StringOrArray::String("<|im_end|>".to_string()));
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert!(s.stop.iter().any(|x| x == "<|im_end|>"));
    }

    #[test]
    fn test_stop_array_threads_through_as_repeated() {
        let mut p = base_payload();
        p.stop.stop = Some(StringOrArray::Array(vec![
            "\n\n".to_string(),
            "END".to_string(),
        ]));
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert!(s.stop.iter().any(|x| x == "\n\n"));
        assert!(s.stop.iter().any(|x| x == "END"));
    }

    #[test]
    fn test_stop_token_ids_threads_through() {
        let mut p = base_payload();
        p.stop.stop_token_ids = Some(vec![151645]);
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert!(s.stop_token_ids.contains(&151645));
    }

    #[test]
    fn test_token_ids_threads_through() {
        let p = base_payload();
        let proto = to_sglang_proto(&p);
        let tok = proto.tokenized.as_ref().unwrap();
        assert_eq!(tok.input_ids, p.token_ids);
    }

    #[test]
    fn test_request_id_threads_through() {
        let mut p = base_payload();
        p.request_id = "snap-B".to_string();
        let proto = to_sglang_proto(&p);
        assert_eq!(proto.request_id, "snap-B");
    }

    #[test]
    fn test_return_logprob_threads_through() {
        let mut p = base_payload();
        p.logprob.return_logprob = true;
        let proto = to_sglang_proto(&p);
        assert!(proto.return_logprob);
    }

    #[test]
    fn test_top_logprobs_num_threads_through() {
        let mut p = base_payload();
        p.logprob.top_logprobs_num = 5;
        let proto = to_sglang_proto(&p);
        assert_eq!(proto.top_logprobs_num, 5);
    }

    #[test]
    fn test_logprob_start_len_threads_through() {
        let mut p = base_payload();
        p.logprob.logprob_start_len = 7;
        let proto = to_sglang_proto(&p);
        assert_eq!(proto.logprob_start_len, 7);
    }

    #[test]
    fn test_tool_constraint_emits_constrained_decoding_field() {
        use mesh_grpc::sglang_proto::sampling_params::Constraint;
        let mut p = base_payload();
        p.tool_constraints = Some((
            "json_schema".to_string(),
            r#"{"type":"object"}"#.to_string(),
        ));
        let proto = to_sglang_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert!(matches!(s.constraint, Some(Constraint::JsonSchema(_))));
    }

    #[test]
    fn test_pd_metadata_threads_into_disaggregated_params() {
        let mut p = base_payload();
        p.pd_metadata = Some(PdMetadata {
            bootstrap_host: "prefill-host".to_string(),
            bootstrap_port: 8998,
            bootstrap_room: 0x4_5b4c,
        });
        let proto = to_sglang_proto(&p);
        let dp = proto
            .disaggregated_params
            .as_ref()
            .expect("disaggregated_params populated for PD payload");
        assert_eq!(dp.bootstrap_host, "prefill-host");
        assert_eq!(dp.bootstrap_port, 8998);
        assert_eq!(dp.bootstrap_room, 0x4_5b4c);
    }

    #[test]
    fn test_no_pd_metadata_yields_no_disaggregated_params() {
        let p = base_payload();
        let proto = to_sglang_proto(&p);
        assert!(proto.disaggregated_params.is_none());
    }

    #[test]
    fn test_vllm_min_p_threads_through() {
        let mut p = base_payload();
        p.sampling.min_p = 0.05;
        let proto = to_vllm_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert!((s.min_p - 0.05).abs() < f32::EPSILON);
    }

    #[test]
    fn test_vllm_top_k_minus_one_clamped_to_zero() {
        let mut p = base_payload();
        p.sampling.top_k = -1;
        let proto = to_vllm_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert_eq!(s.top_k, 0);
    }

    #[test]
    fn test_vllm_temperature_wrapped_in_some() {
        let mut p = base_payload();
        p.sampling.temperature = 0.5;
        let proto = to_vllm_proto(&p);
        let s = proto.sampling_params.as_ref().unwrap();
        assert_eq!(s.temperature, Some(0.5));
    }
}

mod b_proto_to_chunk {
    use mesh_grpc::sglang_proto;

    use crate::routers::grpc::engine::proto_stream_wrapper::{
        ProtoGenerateComplete, ProtoGenerateStreamChunk,
    };
    use crate::routers::grpc::engine::proto_to_chunk::{
        proto_chunk_to_chunk, proto_complete_to_chunk,
    };
    use crate::routers::token_handle::token_chunk::{FinishReason, MatchedStop, TokenChunk};

    fn sg_complete(token_ids: Vec<u32>) -> sglang_proto::GenerateComplete {
        sglang_proto::GenerateComplete {
            output_ids: token_ids,
            finish_reason: "stop".to_string(),
            ..Default::default()
        }
    }

    fn wrap_complete(c: sglang_proto::GenerateComplete) -> ProtoGenerateComplete {
        ProtoGenerateComplete::Sglang(c)
    }

    fn wrap_chunk(c: sglang_proto::GenerateStreamChunk) -> ProtoGenerateStreamChunk {
        ProtoGenerateStreamChunk::Sglang(c)
    }

    #[test]
    fn test_sglang_complete_to_chunk_maps_token_ids() {
        let c = proto_complete_to_chunk(wrap_complete(sg_complete(vec![1, 2, 3])));
        match c {
            TokenChunk::Complete { token_ids, .. } => assert_eq!(token_ids, vec![1, 2, 3]),
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_maps_finish_reason_stop() {
        let c = proto_complete_to_chunk(wrap_complete(sg_complete(vec![1])));
        match c {
            TokenChunk::Complete { finish_reason, .. } => {
                assert!(matches!(finish_reason, FinishReason::Stop));
            }
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_maps_finish_reason_length() {
        let mut c = sg_complete(vec![1]);
        c.finish_reason = "length".to_string();
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        match chunk {
            TokenChunk::Complete { finish_reason, .. } => {
                assert!(matches!(finish_reason, FinishReason::Length));
            }
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_matched_stop_str() {
        let mut c = sg_complete(vec![1]);
        c.matched_stop =
            Some(sglang_proto::generate_complete::MatchedStop::MatchedStopStr("<eot>".to_string()));
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        match chunk {
            TokenChunk::Complete {
                matched_stop: Some(MatchedStop::Str(s)),
                ..
            } => assert_eq!(s, "<eot>"),
            other => panic!("expected MatchedStop::Str, got {other:?}"),
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_matched_stop_token_id() {
        let mut c = sg_complete(vec![1]);
        c.matched_stop = Some(sglang_proto::generate_complete::MatchedStop::MatchedTokenId(2));
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        match chunk {
            TokenChunk::Complete {
                matched_stop: Some(MatchedStop::TokenId(t)),
                ..
            } => assert_eq!(t, 2),
            other => panic!("expected MatchedStop::TokenId, got {other:?}"),
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_no_matched_stop() {
        let chunk = proto_complete_to_chunk(wrap_complete(sg_complete(vec![1])));
        match chunk {
            TokenChunk::Complete { matched_stop, .. } => assert!(matched_stop.is_none()),
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_usage_fields() {
        let mut c = sg_complete(vec![1]);
        c.prompt_tokens = 10;
        c.completion_tokens = 5;
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        match chunk {
            TokenChunk::Complete { usage, .. } => {
                assert_eq!(usage.prompt_tokens, 10);
                assert_eq!(usage.completion_tokens, 5);
                assert_eq!(usage.total_tokens, 15);
            }
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_sglang_chunk_to_chunk_returns_partial() {
        let c = sglang_proto::GenerateStreamChunk {
            token_ids: vec![42],
            ..Default::default()
        };
        let chunk = proto_chunk_to_chunk(wrap_chunk(c));
        match chunk {
            TokenChunk::Partial { token_ids, .. } => assert_eq!(token_ids, vec![42]),
            _ => panic!("expected Partial"),
        }
    }

    #[test]
    fn test_sglang_chunk_to_chunk_logprob_collapse() {
        let c = sglang_proto::GenerateStreamChunk {
            token_ids: vec![42],
            output_logprobs: Some(sglang_proto::OutputLogProbs {
                token_logprobs: vec![-0.5],
                token_ids: vec![42],
                top_logprobs: vec![],
            }),
            ..Default::default()
        };
        let chunk = proto_chunk_to_chunk(wrap_chunk(c));
        if let TokenChunk::Partial {
            logprobs: Some(lps),
            ..
        } = chunk
        {
            assert_eq!(lps.items.len(), 1);
            assert_eq!(lps.items[0].token_id, 42);
            assert!((lps.items[0].logprob - (-0.5)).abs() < f32::EPSILON);
        } else {
            panic!("expected logprobs");
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_input_logprobs_populated_in_single_mode() {
        let mut c = sg_complete(vec![1]);
        c.input_logprobs = Some(sglang_proto::InputLogProbs {
            token_logprobs: vec![
                sglang_proto::InputTokenLogProb { value: Some(-0.1) },
                sglang_proto::InputTokenLogProb { value: Some(-0.2) },
            ],
            token_ids: vec![10, 20],
            top_logprobs: vec![],
        });
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        if let TokenChunk::Complete {
            input_logprobs: Some(ip),
            ..
        } = chunk
        {
            assert_eq!(ip.items.len(), 2);
        } else {
            panic!("expected input_logprobs");
        }
    }

    #[test]
    fn test_sglang_complete_to_chunk_input_logprobs_none_when_missing() {
        let c = sg_complete(vec![1]);
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        if let TokenChunk::Complete { input_logprobs, .. } = chunk {
            assert!(input_logprobs.is_none());
        } else {
            panic!("expected Complete");
        }
    }

    #[test]
    fn test_vllm_complete_to_chunk_parallel_to_sglang() {
        let c = mesh_grpc::vllm_proto::GenerateComplete {
            output_ids: vec![7, 8],
            finish_reason: "stop".to_string(),
            ..Default::default()
        };
        let chunk = proto_complete_to_chunk(ProtoGenerateComplete::Vllm(c));
        match chunk {
            TokenChunk::Complete {
                token_ids,
                finish_reason,
                ..
            } => {
                assert_eq!(token_ids, vec![7, 8]);
                assert!(matches!(finish_reason, FinishReason::Stop));
            }
            _ => panic!("expected Complete"),
        }
    }

    #[test]
    fn test_vllm_chunk_to_chunk_parallel_to_sglang() {
        let c = mesh_grpc::vllm_proto::GenerateStreamChunk {
            token_ids: vec![42],
            ..Default::default()
        };
        let chunk = proto_chunk_to_chunk(ProtoGenerateStreamChunk::Vllm(c));
        assert!(matches!(chunk, TokenChunk::Partial { .. }));
    }

    #[test]
    fn test_meta_cached_tokens_threads_through() {
        let mut c = sg_complete(vec![1]);
        c.cached_tokens = 7;
        let chunk = proto_complete_to_chunk(wrap_complete(c));
        if let TokenChunk::Complete { meta, .. } = chunk {
            assert_eq!(meta.cached_tokens, 7);
        } else {
            panic!("expected Complete");
        }
    }
}

mod c_worker_client_cache {
    use crate::core::worker::WorkerType;
    use crate::routers::grpc::engine::worker_client_cache::get_grpc_client_from_worker;
    use crate::routers::test_mocks::{mock_grpc_worker, mock_http_only_worker};

    #[tokio::test]
    async fn test_get_client_from_http_only_worker_returns_5xx_response() {
        let w = mock_http_only_worker("http://h:8000");
        match get_grpc_client_from_worker(&w).await {
            Ok(_) => panic!("expected error for http-only worker"),
            Err(resp) => assert!(resp.status().is_server_error()),
        }
    }

    #[tokio::test]
    async fn test_get_client_from_grpc_worker_attempts_connect_and_fails_offline() {
        let w = mock_grpc_worker("http://127.0.0.1:1", WorkerType::Regular);
        match get_grpc_client_from_worker(&w).await {
            Ok(_) => panic!("expected connect failure for offline grpc worker"),
            Err(resp) => assert!(resp.status().is_server_error()),
        }
    }

    #[tokio::test]
    async fn test_get_client_from_grpc_prefill_worker_attempts_connect_and_fails_offline() {
        let w = mock_grpc_worker(
            "http://127.0.0.1:1",
            WorkerType::Prefill {
                bootstrap_port: None,
            },
        );
        match get_grpc_client_from_worker(&w).await {
            Ok(_) => panic!("expected connect failure for offline grpc prefill worker"),
            Err(resp) => assert!(resp.status().is_server_error()),
        }
    }
}

mod d_pd_stream_merge {
    use std::time::Duration;

    use futures::StreamExt;

    use crate::routers::grpc::engine::pd_stream_merge::merge_pd_streams;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::{
        scripted_stream_with_drop_observer, scripted_stream_with_poll_counter, ScriptedItem,
    };
    use crate::routers::token_handle::token_chunk::{
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
            usage: Usage {
                prompt_tokens: 0,
                completion_tokens: 0,
                total_tokens: 0,
            },
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
        let (prefill, prefill_poll_count) = scripted_stream_with_poll_counter(vec![
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
            prefill_poll_count.load(std::sync::atomic::Ordering::SeqCst),
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
        let items: Vec<_> = (&mut merged).map(|i| i.unwrap()).collect::<Vec<_>>().await;
        assert_eq!(items.len(), 2);
        match &items[1] {
            TokenChunk::Complete {
                input_logprobs: Some(actual),
                ..
            } => {
                assert_eq!(actual.items.len(), injected.items.len());
            }
            other => panic!("expected Complete with injected logprobs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn pd_merge_t2_no_partial_from_prefill_leaks_to_consumer() {
        let (prefill, _) = scripted_stream_with_poll_counter(vec![
            ScriptedItem::Ok(partial(99)),
            ScriptedItem::Ok(complete_with_input_logprobs(Some(ip(1)))),
        ]);
        let (decode, _) = scripted_stream_with_poll_counter(vec![
            ScriptedItem::Ok(partial(10)),
            ScriptedItem::Ok(complete_with_input_logprobs(None)),
        ]);
        let mut merged = merge_pd_streams(prefill, decode, true);
        let items: Vec<_> = (&mut merged).map(|i| i.unwrap()).collect::<Vec<_>>().await;
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
            EngineError::Prefill(String::new()),
        )]);
        let (decode, decode_polls) = scripted_stream_with_poll_counter(vec![
            ScriptedItem::Ok(partial(1)),
            ScriptedItem::Ok(complete_with_input_logprobs(None)),
        ]);
        let mut merged = merge_pd_streams(prefill, decode, true);
        let first = merged.next().await.expect("merger yields one item");
        assert!(matches!(first, Err(EngineError::Prefill(_))));
        let second = merged.next().await;
        assert!(
            second.is_none(),
            "Terminal state must yield None subsequently"
        );
        assert_eq!(
            decode_polls.load(std::sync::atomic::Ordering::SeqCst),
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
        let items: Vec<_> = (&mut merged).map(|i| i.unwrap()).collect::<Vec<_>>().await;
        assert_eq!(items.len(), 6);
        assert_eq!(prefill_polls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn pd_merge_t5_decode_transport_error_propagates_and_prefill_dropped() {
        let (prefill, _, prefill_drop_observed) =
            scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(
                complete_with_input_logprobs(None),
            )]);
        let (decode, _, _) = scripted_stream_with_drop_observer(vec![
            ScriptedItem::Ok(partial(1)),
            ScriptedItem::Ok(partial(2)),
            ScriptedItem::Err(EngineError::Transport(tonic::Status::aborted("dead"))),
        ]);
        let mut merged = merge_pd_streams(prefill, decode, true);
        let mut yielded = Vec::new();
        while let Some(item) = merged.next().await {
            yielded.push(item);
            if matches!(yielded.last(), Some(Err(_))) {
                break;
            }
        }
        assert_eq!(yielded.len(), 3);
        assert!(matches!(yielded[0], Ok(TokenChunk::Partial { .. })));
        assert!(matches!(yielded[1], Ok(TokenChunk::Partial { .. })));
        assert!(matches!(yielded[2], Err(EngineError::Transport(_))));
        drop(merged);
        assert!(
            prefill_drop_observed.load(std::sync::atomic::Ordering::SeqCst),
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
            scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(
                complete_with_input_logprobs(None),
            )]);
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
        assert!(prefill_drop_observed.load(std::sync::atomic::Ordering::SeqCst));
        assert!(decode_drop_observed.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn pd_merge_t7_pending_prefill_blocks_decode_until_timeout() {
        let (prefill, _) = crate::routers::token_handle::test_support::pending_forever_stream();
        let (decode, _) = scripted_stream_with_poll_counter(vec![
            ScriptedItem::Ok(partial(1)),
            ScriptedItem::Ok(complete_with_input_logprobs(None)),
        ]);
        let mut merged = merge_pd_streams(prefill, decode, true);
        let timed = tokio::time::timeout(Duration::from_millis(100), merged.next()).await;
        assert!(timed.is_err());
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
        let items: Vec<_> = (&mut merged).map(|i| i.unwrap()).collect::<Vec<_>>().await;
        assert_eq!(items.len(), 2);
    }
}

mod e_engine_dispatch {
    use std::sync::Arc;

    use crate::core::placement::types::PlacementPlan;
    use crate::core::worker::WorkerType;
    use crate::routers::grpc::engine::{Dispatcher, GrpcEngine};
    use crate::routers::prepare::generation_payload::{
        GenerationPayload, LogprobConfig, SamplingParams, StopConfig,
    };
    use crate::routers::test_mocks::{mock_grpc_worker, mock_http_only_worker, MockDispatcher};
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::synthetic_single_stream;

    fn basic_payload() -> GenerationPayload {
        GenerationPayload {
            request_id: "r".to_string(),
            text: "x".to_string(),
            token_ids: vec![1],
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: -1,
                min_p: 0.0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
                max_new_tokens: Some(1),
                min_new_tokens: 0,
                n: 1,
                ignore_eos: false,
            },
            stop: StopConfig {
                stop: None,
                stop_token_ids: None,
                skip_special_tokens: false,
                no_stop_trim: false,
            },
            logprob: LogprobConfig {
                return_logprob: false,
                top_logprobs_num: 0,
                logprob_start_len: -1,
                token_ids_logprob: Vec::new(),
                input_logprobs: false,
            },
            tool_constraints: None,
            pd_metadata: None,
            stream: false,
            return_hidden_states: false,
            log_metrics: false,
        }
    }

    #[tokio::test]
    async fn test_dispatch_single_connection_acquire_failure_returns_typed_error() {
        let e = GrpcEngine::new();
        let plan = PlacementPlan::Single {
            worker: mock_http_only_worker("http://unreachable:8000"),
            policy_name: "rr",
        };
        match Dispatcher::dispatch(&e, &plan, &mut basic_payload()).await {
            Ok(_) => panic!("expected ConnectionAcquireFailed"),
            Err(err) => assert!(matches!(err, EngineError::ConnectionAcquireFailed(_))),
        }
    }

    #[tokio::test]
    async fn test_dispatch_pair_connection_acquire_failure_returns_typed_error() {
        let e = GrpcEngine::new();
        let plan = PlacementPlan::Pair {
            prefill: mock_http_only_worker("http://p:8000"),
            decode: mock_http_only_worker("http://d:8000"),
            prefill_policy: "rr",
            decode_policy: "rr",
        };
        match Dispatcher::dispatch(&e, &plan, &mut basic_payload()).await {
            Ok(_) => panic!("expected ConnectionAcquireFailed"),
            Err(err) => assert!(matches!(err, EngineError::ConnectionAcquireFailed(_))),
        }
    }

    #[tokio::test]
    async fn test_mock_dispatcher_single_records_call_and_returns_scripted_stream() {
        let stream = synthetic_single_stream(Vec::new());
        let mock = MockDispatcher::new(vec![Ok(stream)]);
        let plan = PlacementPlan::Single {
            worker: mock_grpc_worker("http://w:8000", WorkerType::Regular),
            policy_name: "rr",
        };
        let _s = mock.dispatch(&plan, &mut basic_payload()).await.unwrap();
        let calls = mock.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].placement_kind, "single");
        assert_eq!(calls[0].worker_url, "http://w:8000");
    }

    #[tokio::test]
    async fn test_mock_dispatcher_pair_records_both_worker_urls() {
        let stream = synthetic_single_stream(Vec::new());
        let mock = MockDispatcher::new(vec![Ok(stream)]);
        let plan = PlacementPlan::Pair {
            prefill: mock_grpc_worker(
                "http://p:8000",
                WorkerType::Prefill {
                    bootstrap_port: None,
                },
            ),
            decode: mock_grpc_worker("http://d:8000", WorkerType::Decode),
            prefill_policy: "rr",
            decode_policy: "rr",
        };
        let _s = mock.dispatch(&plan, &mut basic_payload()).await.unwrap();
        let calls = mock.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].placement_kind, "pair");
        assert_eq!(calls[0].worker_url, "http://p:8000|http://d:8000");
    }

    #[tokio::test]
    async fn test_mock_dispatcher_returns_scripted_error() {
        let mock = MockDispatcher::new(vec![Err(EngineError::RequestBuildFailed("bad".into()))]);
        let plan = PlacementPlan::Single {
            worker: mock_grpc_worker("http://w:8000", WorkerType::Regular),
            policy_name: "rr",
        };
        match mock.dispatch(&plan, &mut basic_payload()).await {
            Ok(_) => panic!("expected RequestBuildFailed"),
            Err(err) => assert!(matches!(err, EngineError::RequestBuildFailed(_))),
        }
    }

    #[test]
    fn test_engine_is_dispatcher_trait_object_safe() {
        let e: Arc<dyn Dispatcher> = Arc::new(GrpcEngine::new());
        let _ = e;
    }
}

mod f_drop_propagation {
    use std::sync::atomic::Ordering;

    use crate::routers::grpc::engine::pd_stream_merge::merge_pd_streams;
    use crate::routers::token_handle::engine_error::EngineError;
    use crate::routers::token_handle::test_support::{
        scripted_stream_with_drop_observer, ScriptedItem,
    };
    use crate::routers::token_handle::token_chunk::{FinishReason, TokenChunk, Usage, WorkerMeta};

    fn done() -> TokenChunk {
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
    async fn test_drop_single_mode_propagates_to_inner_streaming() {
        let (s, _, drop_observed) =
            scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(done())]);
        drop(s);
        assert!(drop_observed.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_drop_pd_mode_propagates_to_both_inner_streams() {
        let (p, _, p_drop) = scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(done())]);
        let (d, _, d_drop) = scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(done())]);
        let merged = merge_pd_streams(p, d, false);
        drop(merged);
        assert!(p_drop.load(Ordering::SeqCst));
        assert!(d_drop.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_clean_exit_pd_mode_marks_prefill_completed_before_drop() {
        let (p, mark_completed_observed, p_drop) =
            crate::routers::token_handle::test_support::scripted_with_mark_completed(vec![
                ScriptedItem::Ok(done()),
            ]);
        let (d, _, _) = scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(done())]);
        let mut merged = merge_pd_streams(p, d, true);
        use futures::StreamExt;
        while merged.next().await.is_some() {}
        drop(merged);
        assert!(
            mark_completed_observed.load(Ordering::SeqCst),
            "prefill must be mark_completed() on clean exit"
        );
        assert!(p_drop.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_non_clean_exit_does_not_mark_completed() {
        let (p, mark_completed_observed, _) =
            crate::routers::token_handle::test_support::scripted_with_mark_completed(vec![
                ScriptedItem::Err(EngineError::Prefill(String::new())),
            ]);
        let (d, _, _) = scripted_stream_with_drop_observer(vec![ScriptedItem::Ok(done())]);
        let mut merged = merge_pd_streams(p, d, true);
        use futures::StreamExt;
        let _ = merged.next().await;
        drop(merged);
        assert!(
            !mark_completed_observed.load(Ordering::SeqCst),
            "prefill must NOT be mark_completed() on error path"
        );
    }
}
