//! Downstream seam. Entrypoints depends on this trait and nothing in Mesh's
//! HTTP server or text render path.

use async_trait::async_trait;

use std::sync::Arc;

use crate::app_context::AppContext;
use crate::token_path::{TokenGenerateRequest, TokenPathError, TokenRouter, TokenStream};

#[async_trait]
pub trait TokenBackend: Send + Sync {
    async fn generate(&self, request: TokenGenerateRequest) -> Result<TokenStream, TokenPathError>;

    async fn abort(&self, request_ids: &[String]) -> Result<(), TokenPathError>;
}

/// In-process backend. Mesh routes by token id and forwards token events.
pub struct MeshBackend {
    router: Arc<TokenRouter>,
}

impl MeshBackend {
    pub fn from_context(context: &AppContext) -> Self {
        Self {
            router: context.token_router.clone(),
        }
    }
}

#[async_trait]
impl TokenBackend for MeshBackend {
    async fn generate(&self, request: TokenGenerateRequest) -> Result<TokenStream, TokenPathError> {
        self.router.generate(request).await
    }

    async fn abort(&self, request_ids: &[String]) -> Result<(), TokenPathError> {
        self.router.abort(request_ids).await
    }
}

#[async_trait]
impl TokenBackend for TokenRouter {
    async fn generate(&self, request: TokenGenerateRequest) -> Result<TokenStream, TokenPathError> {
        TokenRouter::generate(self, request).await
    }

    async fn abort(&self, request_ids: &[String]) -> Result<(), TokenPathError> {
        TokenRouter::abort(self, request_ids).await
    }
}
