//! The fan-out selector's grammar and matching, on SMG's cases.

use std::collections::HashMap;

use axum::http::StatusCode;
use mesh::rl::{selector::Selector, RlError};

fn view(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn rl_selector_parses_and_matches_each_operator() {
    let v = view(&[("engine", "atom"), ("tp_size", "2"), ("role", "policy")]);
    let cases: &[(&str, bool)] = &[
        ("engine=atom", true),
        ("engine=vllm", false),
        ("engine!=vllm", true),
        ("engine!=atom", false),
        ("tp_size in (1,2,4)", true),
        ("tp_size in (8)", false),
        ("role notin (reward, judge)", true),
        ("role notin (policy)", false),
        ("engine=atom, tp_size=2", true),
        ("engine=atom,tp_size=4", false),
        ("  engine = atom ,  role = policy  ", true),
        ("engine in(atom,vllm)", true),
        ("missing=x", false),
        ("missing!=x", true),
        ("missing in (x)", false),
        ("missing notin (x)", true),
        (r#"role="policy""#, true),
        (r#"role="pol\"icy""#, false),
    ];
    for (expr, expected) in cases {
        let selector = Selector::parse(expr).unwrap_or_else(|e| panic!("{expr}: {e}"));
        assert_eq!(selector.matches(&v), *expected, "{expr}");
    }
}

#[test]
fn rl_selector_quoted_values_allow_commas_and_escapes() {
    let v = view(&[("model_path", "/models/a,b"), ("q", r#"say "hi""#)]);
    assert!(Selector::parse(r#"model_path="/models/a,b""#)
        .unwrap()
        .matches(&v));
    assert!(Selector::parse(r#"q="say \"hi\"""#).unwrap().matches(&v));
}

#[test]
fn rl_selector_parse_errors_carry_offsets() {
    let cases: &[(&str, usize)] = &[
        ("", 0),
        ("=x", 0),
        ("engine", 6),
        ("engine==x", 7),
        ("engine in x", 10),
        ("engine in (a,", 13),
        ("engine=a b", 9),
        ("engine=\"unterminated", 20),
        ("engine=a,", 9),
        ("engine=", 7),
    ];
    for (expr, offset) in cases {
        match Selector::parse(expr) {
            Err(e @ RlError::InvalidSelector { .. }) => {
                assert_eq!(e.status(), StatusCode::BAD_REQUEST, "{expr:?}");
                let body = e.to_json();
                assert_eq!(body["error"], "invalid_selector", "{expr:?}");
                assert_eq!(body["offset"], *offset, "{expr:?}");
            }
            other => panic!("{expr:?}: expected InvalidSelector, got {other:?}"),
        }
    }
}

#[test]
fn rl_selector_keeps_its_source_text() {
    let selector = Selector::parse("engine=atom").unwrap();
    assert_eq!(selector.source(), "engine=atom");
}
