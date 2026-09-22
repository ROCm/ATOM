use std::collections::BTreeMap;

use prost_types::{value::Kind, Struct, Value};

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
        endpoint: &str,
        request_id: &str,
        size: usize,
        authorization: Option<&str>,
        execution_id: Option<&str>,
    ) -> pb::ProcessingResponse {
        let size = size.to_string();
        let mut headers = Self::headers([
            (Self::DESTINATION, endpoint.as_bytes()),
            ("x-request-id", request_id.as_bytes()),
            ("content-length", size.as_bytes()),
        ]);
        if let Some(value) = authorization {
            headers
                .set_headers
                .extend(Self::headers([("authorization", value.as_bytes())]).set_headers);
        }
        headers
            .remove_headers
            .push(super::executor::PdExecutor::HEADER.into());
        if let Some(id) = execution_id {
            headers.remove_headers.clear();
            headers.set_headers.extend(
                Self::headers([(super::executor::PdExecutor::HEADER, id.as_bytes())]).set_headers,
            );
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

    pub fn response_headers() -> pb::ProcessingResponse {
        pb::ProcessingResponse {
            response: Some(pb::processing_response::Response::ResponseHeaders(
                pb::HeadersResponse {
                    response: Some(pb::CommonResponse::default()),
                },
            )),
            ..Default::default()
        }
    }

    pub fn trailers(request: bool) -> pb::ProcessingResponse {
        let trailers = pb::TrailersResponse::default();
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
