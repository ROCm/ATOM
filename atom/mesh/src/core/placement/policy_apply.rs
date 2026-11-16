use std::sync::Arc;

use tracing::debug;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

use super::types::{PlacementError, RequestDescriptor};
use crate::core::{HashRing, Worker};
use crate::observability::{metrics::MeshMetrics, route_trace};
use crate::policies::{LoadBalancingPolicy, RouteDecision, SelectWorkerInfo};
use crate::routers::prepare::inference::media_digest_count;

pub async fn apply_policy(
    candidates: &[Arc<dyn Worker>],
    policy: &dyn LoadBalancingPolicy,
    descriptor: &RequestDescriptor<'_>,
    hash_ring: Option<Arc<HashRing>>,
) -> Result<Arc<dyn Worker>, PlacementError> {
    if candidates.is_empty() {
        return Err(PlacementError::NoAvailableWorkers);
    }
    let info = SelectWorkerInfo {
        request_text: descriptor.text,
        tokens: descriptor.tokens,
        headers: descriptor.headers,
        hash_ring,
        carries_media: descriptor.carries_media,
    };
    let (idx, decision) = policy.select_with_decision(candidates, &info).await;
    let selected = idx.map(|idx| candidates[idx].clone());
    // The planner passes the candidates of one pool.
    let pool = candidates[0].worker_type().as_metric_label();
    record_decision(pool, policy.name(), selected.as_ref(), decision);
    // Field values are evaluated only when this target is enabled at debug.
    debug!(
        pool,
        policy = policy.name(),
        worker = selected.as_ref().map(|worker| worker.url()),
        key_hash = routing_key_hash(descriptor),
        media_digests = descriptor.text.map_or(0, media_digest_count),
        "routing decision"
    );
    selected.ok_or(PlacementError::PolicyReturnedNone)
}

/// Records one selection. Runs before the router adds the request, so the
/// selected worker's load does not include it.
fn record_decision(
    pool: &'static str,
    policy: &'static str,
    selected: Option<&Arc<dyn Worker>>,
    decision: RouteDecision,
) {
    MeshMetrics::record_route_branch(pool, policy, decision.branch);
    if let Some(worker) = selected {
        MeshMetrics::record_route_selected_worker_active_requests(
            pool,
            policy,
            decision.branch,
            worker.load(),
        );
    }
    if let Some(ratio) = decision.match_ratio {
        MeshMetrics::record_route_prefix_match_ratio(pool, policy, ratio);
    }
    route_trace::record_selection(
        pool,
        policy,
        selected,
        decision.branch,
        decision.match_ratio,
    );
}

/// xxh3 of what the policy routes on: the text, else the tokens. Requests
/// differing only in media differ here through the text's media digests.
fn routing_key_hash(descriptor: &RequestDescriptor<'_>) -> u64 {
    match (descriptor.text, descriptor.tokens) {
        (Some(text), _) if !text.is_empty() => xxh3_64(text.as_bytes()),
        (_, Some(tokens)) => {
            let mut hasher = Xxh3::new();
            for token in tokens {
                hasher.update(&token.to_le_bytes());
            }
            hasher.digest()
        }
        _ => xxh3_64(&[]),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use tracing::field::{Field, Visit};
    use tracing_subscriber::layer::{Context, SubscriberExt};
    use tracing_subscriber::{Layer, Registry};

    use super::apply_policy;
    use crate::core::placement::test_support::{make_descriptor, make_regular_http};
    use crate::policies::RoundRobinPolicy;
    use crate::routers::prepare::inference::{InferenceRequest, ParsedInference};

    #[derive(Clone, Debug, Default)]
    struct Fields(HashMap<String, String>);

    /// Collects the fields of every event this module emits.
    #[derive(Clone, Default)]
    struct Decisions(Arc<Mutex<Vec<Fields>>>);

    impl Visit for Fields {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0
                .insert(field.name().to_string(), format!("{value:?}"));
        }
    }

    impl<S: tracing::Subscriber> Layer<S> for Decisions {
        fn on_event(&self, event: &tracing::Event<'_>, _: Context<'_, S>) {
            if event.metadata().target() == module_path!().trim_end_matches("::tests") {
                let mut fields = Fields::default();
                event.record(&mut fields);
                self.0.lock().unwrap().push(fields);
            }
        }
    }

    fn chat(content: &str) -> String {
        format!(r#"{{"model":"m","messages":[{{"role":"user","content":{content}}}]}}"#)
    }

    fn image_chat(url: &str) -> String {
        chat(&format!(
            r#"[{{"type":"text","text":"Which color is this?"}},{{"type":"image_url","image_url":{{"url":"{url}"}}}}]"#
        ))
    }

    /// Routes each chat body through the ingress routing text and `apply_policy`.
    async fn route(bodies: &[String]) -> Vec<Fields> {
        let decisions = Decisions::default();
        let _guard = tracing::subscriber::set_default(Registry::default().with(decisions.clone()));
        let candidates = [
            make_regular_http("http://w1:8000", "m"),
            make_regular_http("http://w2:8000", "m"),
        ];
        for body in bodies {
            let text = ParsedInference::parse("/v1/chat/completions", body.as_bytes())
                .unwrap()
                .metadata()
                .text;
            let descriptor = make_descriptor(Some("m"), Some(&text), None, None);
            apply_policy(&candidates, &RoundRobinPolicy::new(), &descriptor, None)
                .await
                .unwrap();
        }
        let events = decisions.0.lock().unwrap().clone();
        events
    }

    #[tokio::test]
    async fn one_decision_per_selection_keys_media_and_repeats_text() {
        let events = route(&[
            image_chat("https://example.com/red.png"),
            image_chat("https://example.com/blue.png"),
            chat(r#""Which color is this?""#),
            chat(r#""Which color is this?""#),
        ])
        .await;
        assert_eq!(events.len(), 4, "{events:?}");
        let field = |i: usize, name: &str| events[i].0[name].clone();
        assert_ne!(field(0, "key_hash"), field(1, "key_hash"));
        assert_eq!(field(2, "key_hash"), field(3, "key_hash"));
        assert_ne!(field(0, "key_hash"), field(2, "key_hash"));
        for (i, digests) in ["1", "1", "0", "0"].into_iter().enumerate() {
            assert_eq!(field(i, "media_digests"), digests);
            assert_eq!(field(i, "pool"), r#""regular""#);
            assert_eq!(field(i, "policy"), r#""round_robin""#);
            assert!(field(i, "worker").contains("http://w"));
        }
    }
}
