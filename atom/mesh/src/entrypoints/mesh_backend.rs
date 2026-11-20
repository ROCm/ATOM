//! Mesh adapter. Entrypoints talks to Mesh only through [`Backend`].

use async_trait::async_trait;

use super::backend::{Backend, BackendError, TokenRequest, TokenStream};
use crate::routers::RouterTrait;

/// Wraps the router's token entry. The router keeps worker selection, P/D
/// planning, and the engine transport; this type does not add another hop.
pub struct MeshBackend<R> {
    router: R,
}

impl<R> MeshBackend<R> {
    pub fn new(router: R) -> Self {
        Self { router }
    }
}

#[async_trait]
impl<R> Backend for MeshBackend<R>
where
    R: RouterTrait + Sync,
{
    async fn generate(&self, request: TokenRequest) -> Result<TokenStream, BackendError> {
        self.router.route_tokens(request).await
    }

    async fn abort(&self, request_id: &str) -> Result<(), BackendError> {
        self.router.abort_tokens(request_id).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use async_trait::async_trait;
    use axum::http::StatusCode;
    use futures::StreamExt;

    use super::*;
    use crate::entrypoints::backend::{test_request, Backend, TokenStream};
    use crate::routers::token_handle::token_chunk::TokenChunk;
    use crate::routers::RouterTrait;

    #[derive(Debug)]
    struct Scripted {
        seen: Mutex<Vec<u32>>,
        aborted: Mutex<Option<String>>,
    }

    #[async_trait]
    impl RouterTrait for Scripted {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn router_type(&self) -> &'static str {
            "scripted"
        }

        async fn route_tokens(&self, request: TokenRequest) -> Result<TokenStream, BackendError> {
            *self.seen.lock().unwrap() = request.token_ids().to_vec();
            Ok(TokenStream::from_chunks(vec![TokenChunk::Partial {
                token_ids: request.token_ids().to_vec(),
                logprobs: None,
            }]))
        }

        async fn abort_tokens(&self, request_id: &str) -> Result<(), BackendError> {
            *self.aborted.lock().unwrap() = Some(request_id.to_string());
            Ok(())
        }
    }

    #[derive(Debug)]
    struct TextOnly;

    #[async_trait]
    impl RouterTrait for TextOnly {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn router_type(&self) -> &'static str {
            "http"
        }
    }

    #[tokio::test]
    async fn mesh_backend_forwards_token_ids_and_abort() {
        let router = Scripted {
            seen: Mutex::new(Vec::new()),
            aborted: Mutex::new(None),
        };
        let backend = MeshBackend::new(router);
        let mut stream = backend.generate(test_request(vec![4, 5, 6])).await.unwrap();
        let chunk = stream.handle.next().await.unwrap().unwrap();
        match chunk {
            TokenChunk::Partial { token_ids, .. } => assert_eq!(token_ids, vec![4, 5, 6]),
            other => panic!("expected a partial chunk, got {other:?}"),
        }
        assert_eq!(*backend.router.seen.lock().unwrap(), vec![4, 5, 6]);
        backend.abort("req-1").await.unwrap();
        assert_eq!(
            backend.router.aborted.lock().unwrap().as_deref(),
            Some("req-1")
        );
    }

    #[tokio::test]
    async fn mesh_backend_reports_a_router_without_a_token_entry() {
        let error = match MeshBackend::new(TextOnly)
            .generate(test_request(vec![1]))
            .await
        {
            Ok(_) => panic!("a text router has no token entry"),
            Err(error) => error,
        };
        assert_eq!(error.status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(error.code, "unsupported");
        assert!(error.message.contains("http"));
    }
}
