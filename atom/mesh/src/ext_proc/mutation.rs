use std::collections::BTreeMap;

use prost_types::{value::Kind, Struct, Value};

use crate::routers::comm::header_utils::should_forward_request_header;

use super::{core, pb};

pub(super) struct Mutation;

impl Mutation {
    pub const DESTINATION: &'static str = "x-gateway-destination-endpoint";
    pub const CHUNK_BYTES: usize = 62_000;

    pub fn headers<'a>(
        values: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> pb::HeaderMutation {
        pb::HeaderMutation {
            set_headers:
                values
                    .into_iter()
                    .map(|(key, value)| core::HeaderValueOption {
                        header: Some(core::HeaderValue {
                            key: key.into(),
                            raw_value: value.to_vec(),
                            ..Default::default()
                        }),
                        append_action:
                            core::header_value_option::HeaderAppendAction::OverwriteIfExistsOrAdd
                                as i32,
                        ..Default::default()
                    })
                    .collect(),
            remove_headers: vec![],
        }
    }

    pub fn request_headers(
        original: &http::HeaderMap,
        endpoint: &str,
        request_id: &str,
        authorization: Option<&str>,
        execution_id: Option<&str>,
    ) -> pb::ProcessingResponse {
        // FULL_DUPLEX_STREAMED lets Envoy choose the HTTP framing. In
        // particular, requests with trailers must not acquire Content-Length.
        let mut headers = Self::headers([
            (Self::DESTINATION, endpoint.as_bytes()),
            ("x-request-id", request_id.as_bytes()),
        ]);
        if let Some(value) = authorization {
            headers
                .set_headers
                .extend(Self::headers([("authorization", value.as_bytes())]).set_headers);
        }
        headers.remove_headers = original
            .keys()
            .map(|name| name.as_str())
            .filter(|name| {
                !should_forward_request_header(name)
                // Envoy needs Host and X-Forwarded-Proto when recalculating the route.
                // Other exceptions describe the representation or enable trailers.
                && !matches!(*name, "host" | "content-type" | "content-encoding" | "x-forwarded-proto" | "te" | "trailer")
                && *name != Self::DESTINATION
                && *name != super::executor::PdExecutor::HEADER
            })
            .map(str::to_owned)
            .collect();
        // A Trailer declaration must not advertise fields that will be removed.
        if original.contains_key("trailer") {
            let declared = original
                .get_all("trailer")
                .iter()
                .filter_map(|value| value.to_str().ok())
                .flat_map(|value| value.split(','))
                .map(str::trim)
                .filter(|name| Self::forward_request_trailer(name))
                .collect::<Vec<_>>()
                .join(", ");
            if declared.is_empty() {
                headers.remove_headers.push("trailer".into());
            } else {
                headers
                    .set_headers
                    .extend(Self::headers([("trailer", declared.as_bytes())]).set_headers);
            }
        }
        if let Some(id) = execution_id {
            headers.set_headers.extend(
                Self::headers([(super::executor::PdExecutor::HEADER, id.as_bytes())]).set_headers,
            );
        } else {
            headers
                .remove_headers
                .push(super::executor::PdExecutor::HEADER.into());
        }
        pb::ProcessingResponse {
            response: Some(pb::processing_response::Response::RequestHeaders(
                pb::HeadersResponse {
                    response: Some(pb::CommonResponse {
                        header_mutation: Some(headers),
                        clear_route_cache: true,
                        ..Default::default()
                    }),
                },
            )),
            dynamic_metadata: Some(Struct {
                fields: BTreeMap::from([(
                    "envoy.lb".into(),
                    Value {
                        kind: Some(Kind::StructValue(Struct {
                            fields: BTreeMap::from([(
                                Self::DESTINATION.into(),
                                Value {
                                    kind: Some(Kind::StringValue(endpoint.into())),
                                },
                            )]),
                        })),
                    },
                )]),
            }),
            ..Default::default()
        }
    }

    pub fn body(bytes: Vec<u8>, end: bool, request: bool) -> pb::ProcessingResponse {
        let body = pb::BodyResponse {
            response: Some(pb::CommonResponse {
                body_mutation: Some(pb::BodyMutation {
                    mutation: Some(pb::body_mutation::Mutation::StreamedResponse(
                        pb::StreamedBodyResponse {
                            body: bytes,
                            end_of_stream: end,
                            ..Default::default()
                        },
                    )),
                }),
                ..Default::default()
            }),
        };
        pb::ProcessingResponse {
            response: Some(if request {
                pb::processing_response::Response::RequestBody(body)
            } else {
                pb::processing_response::Response::ResponseBody(body)
            }),
            ..Default::default()
        }
    }

    pub fn response_headers(request_id: Option<&str>) -> pb::ProcessingResponse {
        pb::ProcessingResponse {
            response: Some(pb::processing_response::Response::ResponseHeaders(
                pb::HeadersResponse {
                    response: Some(pb::CommonResponse {
                        header_mutation: request_id
                            .map(|id| Self::headers([("x-request-id", id.as_bytes())])),
                        ..Default::default()
                    }),
                },
            )),
            ..Default::default()
        }
    }

    fn forward_request_trailer(name: &str) -> bool {
        should_forward_request_header(name)
            // Credentials and the selected correlation ID belong to the initial headers.
            && !name.eq_ignore_ascii_case("authorization")
            && !name.eq_ignore_ascii_case("x-request-id")
    }

    pub fn filter_request_trailers(trailers: pb::HttpTrailers) -> pb::HeaderMutation {
        let mut remove_headers: Vec<_> = trailers
            .trailers
            .unwrap_or_default()
            .headers
            .into_iter()
            .filter(|header| !Self::forward_request_trailer(&header.key))
            .map(|header| header.key.to_ascii_lowercase())
            .collect();
        remove_headers.sort_unstable();
        remove_headers.dedup();
        pb::HeaderMutation {
            set_headers: vec![],
            remove_headers,
        }
    }

    pub fn trailers(
        request: bool,
        header_mutation: Option<pb::HeaderMutation>,
    ) -> pb::ProcessingResponse {
        let trailers = pb::TrailersResponse { header_mutation };
        pb::ProcessingResponse {
            response: Some(if request {
                pb::processing_response::Response::RequestTrailers(trailers)
            } else {
                pb::processing_response::Response::ResponseTrailers(trailers)
            }),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_execution_header_does_not_disable_client_header_filtering() {
        let original = http::HeaderMap::from_iter([
            (
                http::HeaderName::from_static("cookie"),
                http::HeaderValue::from_str(&uuid::Uuid::new_v4().to_string()).unwrap(),
            ),
            (
                http::HeaderName::from_static("x-client-internal"),
                http::HeaderValue::from_static("untrusted"),
            ),
            (
                http::HeaderName::from_static("x-envoy-client-control"),
                http::HeaderValue::from_static("untrusted"),
            ),
            (
                http::HeaderName::from_static("content-type"),
                http::HeaderValue::from_static("application/json"),
            ),
            (
                http::HeaderName::from_static("x-forwarded-proto"),
                http::HeaderValue::from_static("http"),
            ),
        ]);
        let execution_id = uuid::Uuid::new_v4().to_string();
        for execution in [None, Some(execution_id.as_str())] {
            let response = Mutation::request_headers(
                &original,
                "127.0.0.1:8080",
                "correlation",
                None,
                execution,
            );
            let Some(pb::processing_response::Response::RequestHeaders(response)) =
                response.response
            else {
                panic!("expected request headers");
            };
            let mutation = response.response.unwrap().header_mutation.unwrap();
            for name in ["cookie", "x-client-internal", "x-envoy-client-control"] {
                assert!(mutation.remove_headers.iter().any(|n| n == name));
            }
            for name in ["content-type", "x-forwarded-proto"] {
                assert!(!mutation.remove_headers.iter().any(|n| n == name));
            }
            let execution_header = mutation
                .set_headers
                .iter()
                .filter_map(|h| h.header.as_ref())
                .find(|h| h.key == super::super::executor::PdExecutor::HEADER);
            assert_eq!(
                execution_header.map(|h| h.raw_value.as_slice()),
                execution.map(str::as_bytes)
            );
        }
    }
}
