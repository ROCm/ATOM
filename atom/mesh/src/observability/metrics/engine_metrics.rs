//! Worker engine metrics aggregation.
//!
//! This module backs the router `/engine_metrics` endpoint. It scrapes each
//! registered worker's own `/metrics` endpoint, injects Mesh-side worker labels,
//! and merges the Prometheus text without changing the downstream metric names.
//! Mesh self metrics are recorded by `recorder.rs` and exposed by
//! `mesh_metrics.rs`; this module only handles worker engine metrics.

use std::{sync::Arc, time::Duration};

use anyhow::ensure;
use axum::response::{IntoResponse, Response};
use futures::{stream, StreamExt};
use http::StatusCode;
use openmetrics_parser::{MetricFamily, MetricsExposition, PrometheusType, PrometheusValue};
use tracing::warn;

use crate::core::{Worker, WorkerRegistry};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_CONCURRENT: usize = 32;

#[derive(Debug)]
pub struct MetricPack {
    pub labels: Vec<(String, String)>,
    pub metrics_text: String,
}

pub enum EngineMetricsResult {
    Ok(String),
    Err(String),
}

impl IntoResponse for EngineMetricsResult {
    fn into_response(self) -> Response {
        match self {
            Self::Ok(text) => (StatusCode::OK, text).into_response(),
            Self::Err(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg).into_response(),
        }
    }
}

/// Result of a fan-out request to a single worker.
struct WorkerResponse {
    url: String,
    result: Result<reqwest::Response, reqwest::Error>,
}

async fn fan_out_worker_metrics(
    workers: &[Arc<dyn Worker>],
    client: &reqwest::Client,
) -> Vec<WorkerResponse> {
    let futures: Vec<_> = workers
        .iter()
        .map(|worker| {
            let client = client.clone();
            let url = worker.url().to_string();
            let full_url = format!("{}/metrics", url);
            let api_key = worker.api_key().clone();

            async move {
                let mut req = client.get(&full_url).timeout(REQUEST_TIMEOUT);
                if let Some(key) = api_key {
                    req = req.bearer_auth(key);
                }
                WorkerResponse {
                    url,
                    result: req.send().await,
                }
            }
        })
        .collect();

    stream::iter(futures)
        .buffer_unordered(MAX_CONCURRENT)
        .collect()
        .await
}

pub async fn collect_engine_metrics(
    worker_registry: &WorkerRegistry,
    client: &reqwest::Client,
) -> EngineMetricsResult {
    let workers = worker_registry.get_all();

    if workers.is_empty() {
        return EngineMetricsResult::Err("No available workers".to_string());
    }

    let responses = fan_out_worker_metrics(&workers, client).await;

    let mut metric_packs = Vec::new();
    for resp in responses {
        if let Ok(r) = resp.result {
            if r.status().is_success() {
                if let Ok(text) = r.text().await {
                    // Keep the existing /engine_metrics contract: successful
                    // worker scrapes are annotated by worker address, while
                    // failed scrapes are omitted unless every worker fails.
                    metric_packs.push(MetricPack {
                        labels: vec![("worker_addr".into(), resp.url)],
                        metrics_text: text,
                    });
                }
            }
        }
    }

    if metric_packs.is_empty() {
        return EngineMetricsResult::Err("All backend requests failed".to_string());
    }

    match aggregate_metrics(metric_packs) {
        Ok(text) => EngineMetricsResult::Ok(text),
        Err(e) => EngineMetricsResult::Err(format!("Failed to aggregate metrics: {}", e)),
    }
}

type PrometheusExposition = MetricsExposition<PrometheusType, PrometheusValue>;
type PrometheusFamily = MetricFamily<PrometheusType, PrometheusValue>;

/// Aggregate Prometheus metrics scraped from multiple sources into a unified one.
///
/// Invalid Prometheus payloads are skipped to preserve partial-failure behavior.
/// If valid families with the same metric name disagree on label names, merging
/// returns an error because Prometheus samples in a family must share labels.
pub fn aggregate_metrics(metric_packs: Vec<MetricPack>) -> anyhow::Result<String> {
    let mut expositions = vec![];
    for metric_pack in metric_packs {
        let metrics_text = &metric_pack.metrics_text;
        // openmetrics_parser doesn't handle colons in metric names; replace with underscores
        let metrics_text = metrics_text.replace(":", "_");

        let exposition = match openmetrics_parser::prometheus::parse_prometheus(&metrics_text) {
            Ok(x) => x,
            Err(err) => {
                warn!(
                    "aggregate_metrics error when parsing text: pack={:?} err={:?}",
                    metric_pack, err
                );
                continue;
            }
        };
        let exposition = transform_metrics(exposition, &metric_pack.labels);
        expositions.push(exposition);
    }

    let text = try_reduce(expositions.into_iter(), merge_exposition)?
        .map(|x| format!("{x}"))
        .unwrap_or_default();
    Ok(text)
}

fn transform_metrics(
    mut exposition: PrometheusExposition,
    extra_labels: &[(String, String)],
) -> PrometheusExposition {
    for family in exposition.families.values_mut() {
        *family = family.with_labels(extra_labels.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    exposition
}

fn merge_exposition(
    a: PrometheusExposition,
    b: PrometheusExposition,
) -> anyhow::Result<PrometheusExposition> {
    let mut ans = a;
    for (name, family_b) in b.families.into_iter() {
        let family_merged = if let Some(family_a) = ans.families.remove(&name) {
            merge_family(family_a, family_b)?
        } else {
            family_b
        };
        ans.families.insert(name, family_merged);
    }
    Ok(ans)
}

fn merge_family(a: PrometheusFamily, b: PrometheusFamily) -> anyhow::Result<PrometheusFamily> {
    ensure!(
        a.get_label_names() == b.get_label_names(),
        "Label names should agree a={:?} b={:?}",
        a.get_label_names(),
        b.get_label_names()
    );
    a.with_samples(b.into_iter_samples())
        .map_err(|e| anyhow::anyhow!("failed to merge samples: {e:?}"))
}

fn try_reduce<I, T, E, F>(iterable: I, f: F) -> Result<Option<T>, E>
where
    I: IntoIterator<Item = T>,
    F: FnMut(T, T) -> Result<T, E>,
{
    let mut it = iterable.into_iter();
    let first = match it.next() {
        None => return Ok(None),
        Some(x) => x,
    };

    Ok(Some(it.try_fold(first, f)?))
}
