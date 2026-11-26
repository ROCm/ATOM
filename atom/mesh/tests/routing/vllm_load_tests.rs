//! vLLM load read: `/metrics` gauges per engine, the LoadMonitor round into
//! `power_of_two`, and `dp_size` discovery from the same `/metrics`.
use super::*;
use crate::{
    core::{steps::worker::local::get_dp_info, BasicWorkerBuilder, DPAwareWorkerBuilder},
    policies::{LoadBalancingPolicy, PowerOfTwoPolicy, SelectWorkerInfo},
};

/// Serves `body` at `/metrics` only and returns the server's base URL.
async fn metrics_server(body: &'static str) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let app =
        axum::Router::new().route("/metrics", axum::routing::get(move || async move { body }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    url
}

/// Engine 0: 3 running, 1 waiting, KV 0.5. Engine 1: 2 running, 0 waiting, KV 0.25.
const TWO_ENGINES: &str = r#"
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_running{engine="1",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:num_requests_waiting{engine="1",model_name="m"} 0.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 7.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.25
vllm:prompt_tokens_total{engine="0",model_name="m"} 1234.0
"#;

async fn loads_by_url(workers: &[Arc<dyn Worker>]) -> (WorkerLoadsResult, HashMap<String, isize>) {
    let registry = WorkerRegistry::new();
    for worker in workers {
        registry.register(Arc::clone(worker));
    }
    let result =
        WorkerManager::get_all_worker_loads(&registry, &reqwest::Client::new(), BackendType::Vllm)
            .await;
    let loads = result
        .loads
        .iter()
        .map(|info| (info.worker.clone(), info.load))
        .collect();
    (result, loads)
}

#[test]
fn parse_reads_three_gauges_per_engine() {
    let engines = parse_vllm_engine_loads(TWO_ENGINES);
    assert_eq!(engines.len(), 2);
    assert_eq!(engines["0"], [Some(3.0), Some(1.0), Some(0.5)]);
    assert_eq!(engines["1"], [Some(2.0), Some(0.0), Some(0.25)]);
}

#[tokio::test]
async fn engine_split_gives_each_rank_its_engine_and_a_rankless_worker_every_engine() {
    let dp = metrics_server(TWO_ENGINES).await;
    // A full cache caps the KV tie-break at 999.
    let single = metrics_server(
        "vllm:num_requests_running{engine=\"0\",model_name=\"m\"} 0.0\n\
         vllm:num_requests_waiting{engine=\"0\",model_name=\"m\"} 2.0\n\
         vllm:kv_cache_usage_perc{engine=\"0\",model_name=\"m\"} 1.0\n",
    )
    .await;
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(DPAwareWorkerBuilder::new(&dp, 0, 2).build()),
        Arc::new(DPAwareWorkerBuilder::new(&dp, 1, 2).build()),
        Arc::new(BasicWorkerBuilder::new(&single).build()),
        Arc::new(BasicWorkerBuilder::new(&dp).build()),
    ];
    let (result, loads) = loads_by_url(&workers).await;
    let load = |index: usize| loads[workers[index].url()];
    // Rankless on two engines: 6 requests, KV mean 0.375.
    assert_eq!(
        (load(0), load(1), load(2), load(3)),
        (4500, 2250, 2999, 6375)
    );
    assert_eq!((result.successful, result.failed), (4, 0));
}

#[tokio::test]
async fn dp_discovery_counts_engines_from_zero() {
    let two = metrics_server(TWO_ENGINES).await;
    let info = get_dp_info(&two, None).await.unwrap();
    assert_eq!(info.dp_size, 2);

    let shifted = metrics_server(
        "vllm:num_requests_running{engine=\"2\",model_name=\"m\"} 0.0\n\
         vllm:num_requests_running{engine=\"3\",model_name=\"m\"} 0.0\n",
    )
    .await;
    let err = get_dp_info(&shifted, None).await.unwrap_err();
    assert!(err.contains("not 0..n"), "{err}");
}

#[tokio::test]
async fn failed_read_drops_the_stale_value_and_falls_back_to_request_counts() {
    let up = metrics_server(TWO_ENGINES).await;
    let no_kv = metrics_server(
        "vllm:num_requests_running{engine=\"0\",model_name=\"m\"} 0.0\n\
         vllm:num_requests_waiting{engine=\"0\",model_name=\"m\"} 0.0\n",
    )
    .await;
    let workers: Vec<Arc<dyn Worker>> = vec![
        Arc::new(BasicWorkerBuilder::new(&up).build()),
        Arc::new(BasicWorkerBuilder::new("http://127.0.0.1:1").build()),
        Arc::new(BasicWorkerBuilder::new(&no_kv).build()),
    ];
    let (up, down, partial) = (&workers[0], &workers[1], &workers[2]);
    // By the stale cache the down worker is lighter; by router-local counts it is not.
    let policy = PowerOfTwoPolicy::new();
    policy.update_loads(&HashMap::from([
        (up.url().to_string(), 9000),
        (down.url().to_string(), 0),
    ]));
    for _ in 0..3 {
        down.increment_load();
    }

    let (result, loads) = loads_by_url(&workers).await;
    assert_eq!((loads[down.url()], loads[partial.url()]), (-1, -1));
    policy.update_loads(&cacheable_loads(&result.loads));

    let pair = [Arc::clone(up), Arc::clone(down)];
    let (selected, decision) = policy
        .select_with_decision(&pair, &SelectWorkerInfo::default())
        .await;
    assert_eq!(selected, Some(0));
    assert_eq!(decision.branch, "request_load");
}
