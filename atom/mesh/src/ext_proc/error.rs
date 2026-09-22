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

    pub fn response(&self) -> pb::ProcessingResponse {
        pb::ProcessingResponse {
            response: Some(pb::processing_response::Response::ImmediateResponse(
                pb::ImmediateResponse {
                    status: Some(super::proto::envoy::r#type::v3::HttpStatus {
                        code: i32::from(self.status),
                    }),
                    headers: Some(Mutation::headers([(
                        "content-type",
                        b"application/json".as_slice(),
                    )])),
                    body: serde_json::to_vec(
                        &serde_json::json!({"error": {"message": self.message, "code": self.code}}),
                    )
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
