use super::{mutation::Mutation, pb};

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub(super) struct ProcessingError {
    pub status: u16,
    pub code: &'static str,
    pub message: String,
}

impl ProcessingError {
    pub fn new(status: u16, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_request", message)
    }

    pub fn protocol(message: impl Into<String>) -> Self {
        Self::new(400, "invalid_processing_sequence", message)
    }

    pub fn response(&self, request_id: Option<&str>) -> pb::ProcessingResponse {
        let mut headers = Mutation::headers([
            ("content-type", b"application/json".as_slice()),
            ("x-mesh-error-code", self.code.as_bytes()),
        ]);
        if let Some(id) = request_id {
            headers
                .set_headers
                .extend(Mutation::headers([("x-request-id", id.as_bytes())]).set_headers);
        }
        pb::ProcessingResponse {
            response: Some(pb::processing_response::Response::ImmediateResponse(
                pb::ImmediateResponse {
                    status: Some(super::proto::envoy::r#type::v3::HttpStatus {
                        code: i32::from(self.status),
                    }),
                    headers: Some(headers),
                    body: serde_json::to_vec(&crate::routers::comm::error::payload(
                        http::StatusCode::from_u16(self.status).unwrap(),
                        self.code,
                        &self.message,
                    ))
                    .unwrap(),
                    details: format!("mesh_ext_proc_{}", self.code),
                    ..Default::default()
                },
            )),
            ..Default::default()
        }
    }
}

impl From<serde_json::Error> for ProcessingError {
    fn from(error: serde_json::Error) -> Self {
        Self::invalid(error.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn both_ingresses_share_error_envelope_and_code_header() {
        let response = ProcessingError::new(429, "admission_full", "full").response(None);
        let Some(pb::processing_response::Response::ImmediateResponse(response)) =
            response.response
        else {
            panic!("expected immediate error");
        };
        let http = crate::routers::comm::error::create_error(
            http::StatusCode::TOO_MANY_REQUESTS,
            "admission_full",
            "full",
        );
        assert_eq!(http.headers()["x-mesh-error-code"], "admission_full");
        assert!(response.headers.unwrap().set_headers.iter().any(|h| h
            .header
            .as_ref()
            .is_some_and(|h| h.key == "x-mesh-error-code" && h.raw_value == b"admission_full")));
        let http = axum::body::to_bytes(http.into_body(), 4096).await.unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&response.body).unwrap(),
            serde_json::from_slice::<serde_json::Value>(&http).unwrap()
        );
    }
}
