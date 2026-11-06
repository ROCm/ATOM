
fn late_router(p: &GatedServer, d: &GatedServer) -> Arc<PDRouter> {
    let mut router=tests::create_test_pd_router();
    router.backend=BackendType::Atom;
    router.worker_registry.register(p.worker.clone());
    router.worker_registry.register(d.worker.clone());
    let mut info=AtomPrefillInfo::default();
    info.tp_sizes.insert(p.worker.url().into(),1);
    let adapter=Arc::new(AtomAdapter::new(Arc::new(info)));
    router.atom_adapter=Some(adapter.clone());
    router.adapter=adapter;
    router.late_decode=Some(Arc::new(late_decode::DecodeBalancer::default()));
    Arc::new(router)
}

fn start_late(router: Arc<PDRouter>, stream: bool) -> JoinHandle<Response> {
    tokio::spawn(async move {
        router.dispatch_atom_late_decode(None,json!({"max_tokens":30,"stream":stream}),PDRequestContext {
            route:"/v1/chat/completions", batch_size:None,is_stream:stream,return_logprob:false,
            request_text:None,model_id:None,headers:None,
        }).await
    })
}

fn metadata() -> Response {
    axum::Json(json!({"usage":{"prompt_tokens":10000},"kv_transfer_params":{"dp_rank":0}})).into_response()
}

#[tokio::test]
async fn late_decode_not_reserved_before_prefill_and_lives_through_stream() {
    let (mut p,mut d)=servers().await;
    let router=late_router(&p,&d);
    let task=start_late(router.clone(),true);
    p.wait_entered().await;
    assert_eq!(p.worker.load(),1);
    assert_eq!(d.worker.load(),0);
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),0);
    p.respond(metadata());
    d.wait_entered().await;
    assert_eq!(p.worker.load(),0);
    assert_eq!(d.worker.load(),1);
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),1);
    let body=d.request.lock().await.clone().unwrap();
    assert_eq!(body["kv_transfer_params"]["remote_dp_rank"],0);
    let (response,_sender)=stream_response(StatusCode::OK);
    d.respond(response);
    let response=result(task).await;
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),1);
    drop(response);
    wait_load(&d.worker,0).await;
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),0);
    assert_eq!(router.late_decode_admission.available_permits(),2048);
}

#[tokio::test]
async fn late_decode_prefill_cancellation_releases_admission_without_touching_d() {
    let (p,d)=servers().await;
    let router=late_router(&p,&d);
    let task=start_late(router.clone(),false);
    p.wait_entered().await;
    task.abort();let _=task.await;
    wait_load(&p.worker,0).await;
    assert_eq!(d.worker.load(),0);
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),0);
    assert_eq!(router.late_decode_admission.available_permits(),2048);
}

#[tokio::test]
async fn late_decode_rechecks_health_at_handoff() {
    let (mut p,d)=servers().await;
    let router=late_router(&p,&d);
    let task=start_late(router.clone(),false);
    p.wait_entered().await;
    d.worker.set_healthy(false);
    p.respond(metadata());
    assert_eq!(result(task).await.status(),StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(d.worker.load(),0);
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),0);
}

#[tokio::test]
async fn late_decode_failure_releases_budget_and_does_not_repeat_prefill() {
    let (mut p,mut d)=servers().await;
    let router=late_router(&p,&d);
    let task=start_late(router.clone(),false);
    p.wait_entered().await;p.respond(metadata());
    d.wait_entered().await;
    d.respond(StatusCode::SERVICE_UNAVAILABLE.into_response());
    let response=result(task).await;
    assert_eq!(response.status(),StatusCode::SERVICE_UNAVAILABLE);
    drop(response);
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(),0);
    wait_load(&d.worker,0).await;
}

#[tokio::test]
async fn late_decode_filters_other_models() {
    let (p,d)=servers().await;
    let router=late_router(&p,&d);
    assert!(router.late_candidates(Some("unregistered-model"),WorkerType::Decode).is_empty());
}

#[tokio::test]
async fn late_decode_admission_limit_rejects_before_p() {
    let (p,d)=servers().await;
    let router=late_router(&p,&d);
    let _all=router.late_decode_admission.clone().acquire_many_owned(2048).await.unwrap();
    let response=result(start_late(router.clone(),false)).await;
    assert_eq!(response.status(),StatusCode::TOO_MANY_REQUESTS);
    assert_eq!((p.worker.load(),d.worker.load()),(0,0));
}

#[tokio::test]
async fn late_decode_cached_prefill_preserves_backend_rank_and_full_context_budget() {
    let (mut p, mut d) = backend_routed_servers().await;
    let router = late_router(&p, &d);
    let task = start_late(router.clone(), true);
    p.wait_entered().await;
    assert!(p.request.lock().await.as_ref().unwrap().get("data_parallel_rank").is_none());
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding(), 0);
    p.respond(axum::Json(json!({
        "usage": {"prompt_tokens": 10000, "prompt_tokens_details": {"cached_tokens": 9984}},
        "prompt_token_ids": vec![11; 10000],
        "kv_transfer_params": {"dp_rank": 3, "prefix_cache_hit_tokens": 9984,
            "remote_block_ids": [7, 8], "remote_swa_block_ids": [9], "remote_slot_index": 19}
    })).into_response());
    d.wait_entered().await;
    let body = d.request.lock().await.clone().unwrap();
    assert_eq!(body["data_parallel_rank"], 1);
    assert_eq!(body["kv_transfer_params"]["remote_dp_rank"], 3);
    assert_eq!(body["kv_transfer_params"]["remote_slot_index"], 19);
    assert_eq!(body["kv_transfer_params"]["remote_swa_block_ids"], json!([9]));
    // Formatted token IDs are authoritative; cached Prefill tokens never reduce D's context.
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding_tokens(), 10000 + 30);
    let (response, _sender) = stream_response(StatusCode::OK);
    d.respond(response);
    drop(result(task).await);
    wait_load(&d.worker, 0).await;
    assert_eq!(router.late_decode.as_ref().unwrap().outstanding_tokens(), 0);
}

