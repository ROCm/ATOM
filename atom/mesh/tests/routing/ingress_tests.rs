use super::*;
use crate::core::BasicWorkerBuilder;
use serde_json::json;

#[test]
fn response_state_never_fails_over_to_a_different_owner() {
    let parsed = ParsedInference::parse(
        "/v1/responses",
        br#"{"model":"m","input":"next","previous_response_id":"resp-1"}"#,
    )
    .unwrap();
    let first: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:1")
            .model_id("m")
            .build(),
    );
    let second: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:2")
            .model_id("m")
            .build(),
    );
    assert!(parsed.requires_state_domain());
    first.set_healthy(false);
    assert_eq!(
        IngressRouting::validate_state_domain(&[first.clone(), second])
            .unwrap_err()
            .code,
        "stateful_routing_unsupported"
    );
    assert!(IngressRouting::validate_state_domain(&[first]).is_ok());
    let shared = [1, 2].map(|port| {
        Arc::new(
            BasicWorkerBuilder::new(format!("http://127.0.0.1:{port}"))
                .labels(std::collections::HashMap::from([(
                    "mesh.responses_state".into(),
                    "shared".into(),
                )]))
                .build(),
        ) as Arc<dyn Worker>
    });
    assert!(IngressRouting::validate_state_domain(&shared).is_ok());
    let background = ParsedInference::parse(
        "/v1/responses",
        br#"{"model":"m","input":"new","background":true}"#,
    )
    .unwrap();
    assert!(!background.requires_state_domain());
}

#[test]
fn new_api_routing_includes_system_tools_and_extension_content() {
    for path in ["/v1/messages", "/v1/responses"] {
        let mut value = if path.ends_with("messages") {
            json!({"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"system":"first"})
        } else {
            json!({"model":"m","input":"hi","instructions":"first"})
        };
        let first = ParsedInference::parse(path, &serde_json::to_vec(&value).unwrap())
            .unwrap()
            .metadata()
            .text;
        value[if path.ends_with("messages") {
            "system"
        } else {
            "instructions"
        }] = json!("different");
        let second = ParsedInference::parse(path, &serde_json::to_vec(&value).unwrap())
            .unwrap()
            .metadata()
            .text;
        assert_ne!(first, second);
        value["tools"] = json!([{"type":"function","name":"tool","vendor":{"x":1}}]);
        let third = ParsedInference::parse(path, &serde_json::to_vec(&value).unwrap())
            .unwrap()
            .metadata()
            .text;
        assert_ne!(second, third);
    }
}

#[tokio::test]
async fn capabilities_and_exact_token_routing_are_explicit() {
    let worker = BasicWorkerBuilder::new("http://127.0.0.1:1")
        .labels(std::collections::HashMap::from([(
            "mesh.apis".into(),
            "/v1/messages, /v1/chat/completions".into(),
        )]))
        .build();
    assert!(EndpointSpec::find("/v1/messages")
        .unwrap()
        .supports(&worker));
    assert!(!EndpointSpec::find("/v1/responses")
        .unwrap()
        .supports(&worker));
    let config = crate::config::RouterConfig {
        policy: crate::config::PolicyConfig::PrefixHash {
            prefix_token_count: 4,
            load_factor: 1.25,
        },
        ..Default::default()
    };
    let app = AppContext::from_config(config, 5).await.unwrap();
    for path in ["/v1/messages", "/v1/responses"] {
        let body = if path.ends_with("messages") {
            json!({"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1})
        } else {
            json!({"model":"m","input":"hi"})
        };
        let parsed = ParsedInference::parse(path, &serde_json::to_vec(&body).unwrap()).unwrap();
        assert_eq!(
            EndpointSpec::find(path)
                .unwrap()
                .validate_topology(true)
                .unwrap_err()
                .code,
            "unsupported_api_topology"
        );
        assert_eq!(
            IngressRouting::new(&app)
                .tokens(&parsed, &parsed.metadata(), &AtomicBool::new(false))
                .unwrap_err()
                .code,
            "token_routing_unsupported"
        );
    }
}
#[tokio::test]
async fn round_robin_skips_routing_text_and_candidates_keep_model_boundaries() {
    let app = AppContext::from_config(
        crate::config::RouterConfig {
            policy: crate::config::PolicyConfig::RoundRobin,
            ..Default::default()
        },
        5,
    )
    .await
    .unwrap();
    let target: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:1")
            .model_id("m")
            .build(),
    );
    let other: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("http://127.0.0.1:2")
            .model_id("other")
            .build(),
    );
    app.worker_registry.register(target);
    app.worker_registry.register(other);
    let routing = IngressRouting::new(&app);
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model":"m","messages":[{"role":"user","content":"text"}]}),
        ),
        (
            "/v1/messages",
            json!({"model":"m","messages":[{"role":"user","content":"text"}],"max_tokens":1}),
        ),
        ("/v1/responses", json!({"model":"m","input":"text"})),
    ] {
        let parsed = ParsedInference::parse(path, &serde_json::to_vec(&body).unwrap()).unwrap();
        let (metadata, tokens) = routing.prepare(&parsed, &AtomicBool::new(false)).unwrap();
        assert!(metadata.text.is_empty());
        assert!(tokens.is_none());
        let candidates = routing.candidates(&metadata, false).unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].model_id(), "m");
        assert!(!parsed.metadata().text.is_empty());
    }
}
