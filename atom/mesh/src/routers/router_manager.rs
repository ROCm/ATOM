//! Router Manager for coordinating routers and workers
//!
//! Provides centralized management in single-router mode.

use std::sync::Arc;

use arc_swap::ArcSwap;
use async_trait::async_trait;
use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use serde_json::Value;
use tracing::{debug, info};

use crate::{
    app_context::AppContext,
    config::RoutingMode,
    core::{ConnectionMode, WorkerRegistry},
    protocols::{
        chat::ChatCompletionRequest,
        completion::CompletionRequest,
        generate::GenerateRequest,
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::RouterTrait,
    server::ServerConfig,
};

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct RouterId(&'static str);

impl RouterId {
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        self.0
    }
}

/// Static router ID constants to avoid heap allocations in hot paths
pub mod router_ids {
    use super::RouterId;

    pub const HTTP_REGULAR: RouterId = RouterId::new("http-regular");
    pub const HTTP_PD: RouterId = RouterId::new("http-pd");
    pub const GRPC_REGULAR: RouterId = RouterId::new("grpc-regular");
    pub const GRPC_PD: RouterId = RouterId::new("grpc-pd");
}

pub struct RouterManager {
    worker_registry: Arc<WorkerRegistry>,
    routers: Arc<DashMap<RouterId, Arc<dyn RouterTrait>>>,
    routers_snapshot: ArcSwap<Vec<Arc<dyn RouterTrait>>>,
    default_router: Arc<std::sync::RwLock<Option<RouterId>>>,
}

impl RouterManager {
    pub fn new(worker_registry: Arc<WorkerRegistry>) -> Self {
        Self {
            worker_registry,
            routers: Arc::new(DashMap::new()),
            routers_snapshot: ArcSwap::from_pointee(Vec::new()),
            default_router: Arc::new(std::sync::RwLock::new(None)),
        }
    }

    pub async fn from_config(
        config: &ServerConfig,
        app_context: &Arc<AppContext>,
    ) -> Result<Arc<Self>, String> {
        use crate::routers::RouterFactory;

        let manager = Arc::new(Self::new(app_context.worker_registry.clone()));

        info!("Initializing RouterManager in single-router mode");

        let single_router = Arc::from(RouterFactory::create_router(app_context).await?);
        let router_id = Self::determine_router_id(
            &config.router_config.mode,
            &config.router_config.connection_mode,
        );

        info!("Created single router with ID: {}", router_id.as_str());
        manager.register_router(router_id.clone(), single_router);
        manager.set_default_router(router_id);

        if manager.router_count() == 0 {
            return Err("No routers could be initialized".to_string());
        }

        Ok(manager)
    }

    pub fn determine_router_id(
        routing_mode: &RoutingMode,
        connection_mode: &ConnectionMode,
    ) -> RouterId {
        match (connection_mode, routing_mode) {
            (ConnectionMode::Http, RoutingMode::Regular { .. }) => router_ids::HTTP_REGULAR,
            (ConnectionMode::Http, RoutingMode::PrefillDecode { .. }) => router_ids::HTTP_PD,
            (ConnectionMode::Grpc { .. }, RoutingMode::Regular { .. }) => router_ids::GRPC_REGULAR,
            (ConnectionMode::Grpc { .. }, RoutingMode::PrefillDecode { .. }) => router_ids::GRPC_PD,
        }
    }

    pub fn register_router(&self, id: RouterId, router: Arc<dyn RouterTrait>) {
        self.routers.insert(id.clone(), router);

        // Update the lock-free snapshot for fast per-request iteration
        let new_snapshot: Vec<_> = self.routers.iter().map(|e| e.value().clone()).collect();
        self.routers_snapshot.store(Arc::new(new_snapshot));

        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        if default_router.is_none() {
            *default_router = Some(id.clone());
            info!("Set default router to {}", id.as_str());
        }
    }

    pub fn set_default_router(&self, id: RouterId) {
        let mut default_router = self
            .default_router
            .write()
            .unwrap_or_else(|e| e.into_inner());
        *default_router = Some(id);
    }

    pub fn router_count(&self) -> usize {
        self.routers.len()
    }

    pub fn select_router_for_request(
        &self,
        _headers: Option<&HeaderMap>,
        model_id: Option<&str>,
    ) -> Option<Arc<dyn RouterTrait>> {
        // Single-router mode: always use the default router
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(ref default_id) = *default_router {
            debug!(
                "Single-router mode: using default router {} for model {:?}",
                default_id.as_str(),
                model_id
            );
            return self.routers.get(default_id).map(|r| r.clone());
        }
        None
    }
}

#[async_trait]
impl RouterTrait for RouterManager {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, _req: Request<Body>) -> Response {
        // Return 200 if at least one router has healthy workers
        let has_healthy_workers = self
            .worker_registry
            .get_all()
            .iter()
            .any(|w| w.is_healthy());

        if has_healthy_workers {
            (StatusCode::OK, "At least one router has healthy workers").into_response()
        } else {
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "No routers with healthy workers available",
            )
                .into_response()
        }
    }

    async fn get_server_info(&self, _req: Request<Body>) -> Response {
        // TODO: Aggregate info from all routers with healthy workers
        (
            StatusCode::OK,
            serde_json::json!({
                "router_manager": true,
                "routers_count": self.routers.len(),
                "workers_count": self.worker_registry.get_all().len()
            })
            .to_string(),
        )
            .into_response()
    }

    async fn get_models(&self, _req: Request<Body>) -> Response {
        let model_names = self.worker_registry.get_models();

        if model_names.is_empty() {
            (StatusCode::SERVICE_UNAVAILABLE, "No models available").into_response()
        } else {
            // Convert model names to OpenAI-compatible model objects
            let models: Vec<Value> = model_names
                .iter()
                .map(|name| {
                    serde_json::json!({
                        "id": name,
                        "object": "model",
                        "owned_by": "local"
                    })
                })
                .collect();

            (
                StatusCode::OK,
                serde_json::json!({
                    "object": "list",
                    "data": models
                })
                .to_string(),
            )
                .into_response()
        }
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        // Route to default router or first available router
        let router_id = {
            let default_router = self
                .default_router
                .read()
                .unwrap_or_else(|e| e.into_inner());
            default_router.clone()
        };

        let router = if let Some(id) = router_id {
            self.routers.get(&id).map(|r| r.clone())
        } else {
            // If no default, use first available router
            self.routers.iter().next().map(|r| r.value().clone())
        };

        if let Some(router) = router {
            router.get_model_info(req).await
        } else {
            (StatusCode::SERVICE_UNAVAILABLE, "No routers available").into_response()
        }
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_generate(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available for this request",
            )
                .into_response()
        }
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_chat(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        let router = self.select_router_for_request(headers, model_id);

        if let Some(router) = router {
            router.route_completion(headers, body, model_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("Model '{}' not found or no router available", body.model),
            )
                .into_response()
        }
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let selected_model = model_id.or(Some(body.model.as_str()));
        let router = self.select_router_for_request(headers, selected_model);

        if let Some(router) = router {
            router.route_responses(headers, body, selected_model).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to handle responses request",
            )
                .into_response()
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        params: &ResponsesGetParams,
    ) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.get_response(headers, response_id, params).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to get response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.cancel_response(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                format!("No router available to cancel response '{}'", response_id),
            )
                .into_response()
        }
    }

    async fn delete_response(&self, _headers: Option<&HeaderMap>, _response_id: &str) -> Response {
        (
            StatusCode::NOT_IMPLEMENTED,
            "delete_response not yet implemented",
        )
            .into_response()
    }

    async fn list_response_input_items(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
    ) -> Response {
        // Delegate to the default router (typically http-regular)
        // Response storage is shared across all routers via AppContext
        let router = self.select_router_for_request(headers, None);
        if let Some(router) = router {
            router.list_response_input_items(headers, response_id).await
        } else {
            (
                StatusCode::NOT_FOUND,
                "No router available to list response input items",
            )
                .into_response()
        }
    }

    fn router_type(&self) -> &'static str {
        "manager"
    }
}

impl std::fmt::Debug for RouterManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let default_router = self
            .default_router
            .read()
            .unwrap_or_else(|e| e.into_inner());
        f.debug_struct("RouterManager")
            .field("routers_count", &self.routers.len())
            .field("workers_count", &self.worker_registry.get_all().len())
            .field("default_router", &*default_router)
            .finish()
    }
}
