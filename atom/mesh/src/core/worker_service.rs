//! Worker Service - Business logic layer for worker operations
//!
//! This module provides a clean separation between HTTP concerns (in routers)
//! and business logic for worker management. The service orchestrates
//! WorkerRegistry and JobQueue operations.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use tracing::warn;

use crate::{
    config::RouterConfig,
    core::{worker::worker_to_info, worker_registry::WorkerId, Job, JobQueue, WorkerRegistry},
    protocols::worker_spec::{
        WorkerConfigRequest, WorkerErrorResponse, WorkerInfo, WorkerUpdateRequest,
    },
};

/// Error types for worker service operations
#[derive(Debug)]
pub enum WorkerServiceError {
    /// Worker with given ID was not found
    NotFound { worker_id: String },
    /// Invalid worker ID format (expected UUID)
    InvalidId { raw: String, message: String },
    /// Job queue not initialized
    QueueNotInitialized,
    /// Failed to submit job to queue
    QueueSubmitFailed { message: String },
}

impl WorkerServiceError {
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::NotFound { .. } => "WORKER_NOT_FOUND",
            Self::InvalidId { .. } => "BAD_REQUEST",
            Self::QueueNotInitialized => "INTERNAL_SERVER_ERROR",
            Self::QueueSubmitFailed { .. } => "INTERNAL_SERVER_ERROR",
        }
    }

    pub fn status_code(&self) -> StatusCode {
        match self {
            Self::NotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidId { .. } => StatusCode::BAD_REQUEST,
            Self::QueueNotInitialized => StatusCode::INTERNAL_SERVER_ERROR,
            Self::QueueSubmitFailed { .. } => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl std::fmt::Display for WorkerServiceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { worker_id } => write!(f, "Worker {} not found", worker_id),
            Self::InvalidId { raw, message } => {
                write!(
                    f,
                    "Invalid worker_id '{}' (expected UUID). Error: {}",
                    raw, message
                )
            }
            Self::QueueNotInitialized => write!(f, "Job queue not initialized"),
            Self::QueueSubmitFailed { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for WorkerServiceError {}

impl IntoResponse for WorkerServiceError {
    fn into_response(self) -> Response {
        let error = WorkerErrorResponse {
            error: self.to_string(),
            code: self.error_code().to_string(),
        };
        (self.status_code(), Json(error)).into_response()
    }
}

/// Result of creating a worker (async job submission)
#[derive(Debug)]
pub struct CreateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
    pub location: String,
}

impl IntoResponse for CreateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "url": self.url,
            "location": self.location,
            "message": "Worker addition queued for background processing"
        });
        (
            StatusCode::ACCEPTED,
            [(http::header::LOCATION, self.location)],
            Json(response),
        )
            .into_response()
    }
}

/// Result of deleting a worker (async job submission)
#[derive(Debug)]
pub struct DeleteWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for DeleteWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker removal queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of updating a worker (async job submission)
#[derive(Debug)]
pub struct UpdateWorkerResult {
    pub worker_id: WorkerId,
    pub url: String,
}

impl IntoResponse for UpdateWorkerResult {
    fn into_response(self) -> Response {
        let response = json!({
            "status": "accepted",
            "worker_id": self.worker_id.as_str(),
            "message": "Worker update queued for background processing"
        });
        (StatusCode::ACCEPTED, Json(response)).into_response()
    }
}

/// Result of listing workers
#[derive(Debug)]
pub struct ListWorkersResult {
    pub workers: Vec<WorkerInfo>,
    pub total: usize,
    pub prefill_count: usize,
    pub decode_count: usize,
    pub regular_count: usize,
}

impl IntoResponse for ListWorkersResult {
    fn into_response(self) -> Response {
        let response = json!({
            "workers": self.workers,
            "total": self.total,
            "stats": {
                "prefill_count": self.prefill_count,
                "decode_count": self.decode_count,
                "regular_count": self.regular_count,
            }
        });
        Json(response).into_response()
    }
}

/// Wrapper for WorkerInfo to implement IntoResponse
pub struct GetWorkerResponse(pub WorkerInfo);

impl IntoResponse for GetWorkerResponse {
    fn into_response(self) -> Response {
        Json(self.0).into_response()
    }
}

/// Worker Service - Orchestrates worker business logic
///
/// This service provides a clean API for worker operations, separating
/// business logic from HTTP concerns. Handlers in server.rs become thin
/// wrappers that translate between HTTP and this service.
pub struct WorkerService {
    worker_registry: Arc<WorkerRegistry>,
    job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
    router_config: RouterConfig,
}

impl WorkerService {
    /// Create a new WorkerService
    pub fn new(
        worker_registry: Arc<WorkerRegistry>,
        job_queue: Arc<std::sync::OnceLock<Arc<JobQueue>>>,
        router_config: RouterConfig,
    ) -> Self {
        Self {
            worker_registry,
            job_queue,
            router_config,
        }
    }

    /// Parse and validate a worker ID string
    pub fn parse_worker_id(&self, raw: &str) -> Result<WorkerId, WorkerServiceError> {
        uuid::Uuid::parse_str(raw)
            .map(|_| WorkerId::from_string(raw.to_string()))
            .map_err(|e| WorkerServiceError::InvalidId {
                raw: raw.to_string(),
                message: e.to_string(),
            })
    }

    /// Get the job queue, returning an error if not initialized
    fn get_job_queue(&self) -> Result<&Arc<JobQueue>, WorkerServiceError> {
        self.job_queue
            .get()
            .ok_or(WorkerServiceError::QueueNotInitialized)
    }

    pub async fn create_worker(
        &self,
        mut config: WorkerConfigRequest,
    ) -> Result<CreateWorkerResult, WorkerServiceError> {
        if self.router_config.api_key.is_some() && config.api_key.is_none() {
            warn!(
                "Adding worker {} without API key while router has API key configured. \
                Worker will be accessible without authentication. \
                If the worker requires the same API key as the router, please specify it explicitly.",
                config.url
            );
        }

        config.dp_aware = self.router_config.dp_aware;

        let worker_url = config.url.clone();
        let worker_id = self.worker_registry.reserve_id_for_url(&worker_url);

        let job = Job::AddWorker {
            config: Box::new(config),
        };

        self.get_job_queue()?
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        let location = format!("/workers/{}", worker_id.as_str());

        Ok(CreateWorkerResult {
            worker_id,
            url: worker_url,
            location,
        })
    }

    /// List all workers with their info
    pub fn list_workers(&self) -> ListWorkersResult {
        let workers = self.worker_registry.get_all_with_ids();
        let worker_infos: Vec<WorkerInfo> = workers
            .iter()
            .map(|(worker_id, worker)| {
                let mut info = worker_to_info(worker);
                info.id = worker_id.as_str().to_string();
                info
            })
            .collect();

        let stats = self.worker_registry.stats();

        ListWorkersResult {
            workers: worker_infos,
            total: stats.total_workers,
            prefill_count: stats.prefill_workers,
            decode_count: stats.decode_workers,
            regular_count: stats.regular_workers,
        }
    }

    pub fn get_worker(&self, worker_id_raw: &str) -> Result<GetWorkerResponse, WorkerServiceError> {
        let worker_id = self.parse_worker_id(worker_id_raw)?;
        let job_queue = self.get_job_queue()?;

        if let Some(worker) = self.worker_registry.get(&worker_id) {
            let worker_url = worker.url().to_string();
            let mut worker_info = worker_to_info(&worker);
            worker_info.id = worker_id.as_str().to_string();
            if let Some(status) = job_queue.get_status(&worker_url) {
                worker_info.job_status = Some(status);
            }
            return Ok(GetWorkerResponse(worker_info));
        }

        if let Some(worker_url) = self.worker_registry.get_url_by_id(&worker_id) {
            if let Some(status) = job_queue.get_status(&worker_url) {
                return Ok(GetWorkerResponse(WorkerInfo::pending(
                    worker_id.as_str(),
                    worker_url,
                    Some(status),
                )));
            }
        }

        Err(WorkerServiceError::NotFound {
            worker_id: worker_id_raw.to_string(),
        })
    }

    /// Delete a worker by ID (submits async job)
    pub async fn delete_worker(
        &self,
        worker_id_raw: &str,
    ) -> Result<DeleteWorkerResult, WorkerServiceError> {
        let worker_id = self.parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::RemoveWorker { url: url.clone() };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(DeleteWorkerResult { worker_id, url })
    }

    /// Update a worker by ID (submits async job)
    pub async fn update_worker(
        &self,
        worker_id_raw: &str,
        update: WorkerUpdateRequest,
    ) -> Result<UpdateWorkerResult, WorkerServiceError> {
        let worker_id = self.parse_worker_id(worker_id_raw)?;

        let url = self
            .worker_registry
            .get_url_by_id(&worker_id)
            .ok_or_else(|| WorkerServiceError::NotFound {
                worker_id: worker_id_raw.to_string(),
            })?;

        let job = Job::UpdateWorker {
            url: url.clone(),
            update: Box::new(update),
        };

        let job_queue = self.get_job_queue()?;
        job_queue
            .submit(job)
            .await
            .map_err(|e| WorkerServiceError::QueueSubmitFailed { message: e })?;

        Ok(UpdateWorkerResult { worker_id, url })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::response::IntoResponse;
    use http::StatusCode;

    use super::*;
    use crate::core::{BasicWorkerBuilder, WorkerType};

    /// Helper to create a WorkerService with real WorkerRegistry and no job queue
    fn make_service() -> (WorkerService, Arc<WorkerRegistry>) {
        let registry = Arc::new(WorkerRegistry::new());
        let job_queue = Arc::new(std::sync::OnceLock::new());
        let config = RouterConfig::default();
        let service = WorkerService::new(registry.clone(), job_queue, config);
        (service, registry)
    }

    /// Helper to create and register a worker, returning its ID
    fn register_worker(registry: &WorkerRegistry, url: &str) -> WorkerId {
        let mut labels = HashMap::new();
        labels.insert("model_id".to_string(), "test-model".to_string());
        let worker = Arc::new(
            BasicWorkerBuilder::new(url)
                .worker_type(WorkerType::Regular)
                .labels(labels)
                .build(),
        );
        registry.register(worker)
    }

    // --- parse_worker_id tests ---

    #[test]
    fn test_parse_valid_uuid() {
        let (service, _) = make_service();
        let uuid_str = uuid::Uuid::new_v4().to_string();
        let result = service.parse_worker_id(&uuid_str);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), uuid_str);
    }

    #[test]
    fn test_parse_invalid_uuid() {
        let (service, _) = make_service();
        let result = service.parse_worker_id("not-a-uuid");
        assert!(result.is_err());
        match result.unwrap_err() {
            WorkerServiceError::InvalidId { raw, .. } => assert_eq!(raw, "not-a-uuid"),
            other => panic!("Expected InvalidId, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_empty_string() {
        let (service, _) = make_service();
        let result = service.parse_worker_id("");
        assert!(result.is_err());
    }

    // --- Error type tests ---

    #[test]
    fn test_error_not_found_status() {
        let err = WorkerServiceError::NotFound {
            worker_id: "abc".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::NOT_FOUND);
        assert_eq!(err.error_code(), "WORKER_NOT_FOUND");
        assert!(err.to_string().contains("abc"));
    }

    #[test]
    fn test_error_invalid_id_status() {
        let err = WorkerServiceError::InvalidId {
            raw: "bad".to_string(),
            message: "parse error".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::BAD_REQUEST);
        assert_eq!(err.error_code(), "BAD_REQUEST");
    }

    #[test]
    fn test_error_queue_not_initialized() {
        let err = WorkerServiceError::QueueNotInitialized;
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.error_code(), "INTERNAL_SERVER_ERROR");
    }

    #[test]
    fn test_error_queue_submit_failed() {
        let err = WorkerServiceError::QueueSubmitFailed {
            message: "channel closed".to_string(),
        };
        assert_eq!(err.status_code(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.to_string().contains("channel closed"));
    }

    #[test]
    fn test_error_into_response() {
        let err = WorkerServiceError::NotFound {
            worker_id: "xyz".to_string(),
        };
        let response = err.into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    // --- list_workers tests ---

    #[test]
    fn test_list_workers_empty() {
        let (service, _) = make_service();
        let result = service.list_workers();
        assert_eq!(result.total, 0);
        assert_eq!(result.prefill_count, 0);
        assert_eq!(result.decode_count, 0);
        assert_eq!(result.regular_count, 0);
        assert!(result.workers.is_empty());
    }

    #[test]
    fn test_list_workers_with_workers() {
        let (service, registry) = make_service();

        register_worker(&registry, "http://w1:8000");
        register_worker(&registry, "http://w2:8000");

        let result = service.list_workers();
        assert_eq!(result.total, 2);
        assert_eq!(result.regular_count, 2);
        assert_eq!(result.workers.len(), 2);

        let urls: Vec<&str> = result.workers.iter().map(|w| w.url.as_str()).collect();
        assert!(urls.contains(&"http://w1:8000"));
        assert!(urls.contains(&"http://w2:8000"));
    }

    #[test]
    fn test_list_workers_pd_stats() {
        let (service, registry) = make_service();

        let prefill = Arc::new(
            BasicWorkerBuilder::new("http://p1:8000")
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        );
        let decode = Arc::new(
            BasicWorkerBuilder::new("http://d1:8000")
                .worker_type(WorkerType::Decode)
                .build(),
        );
        registry.register(prefill);
        registry.register(decode);

        let result = service.list_workers();
        assert_eq!(result.total, 2);
        assert_eq!(result.prefill_count, 1);
        assert_eq!(result.decode_count, 1);
        assert_eq!(result.regular_count, 0);
    }

    // --- get_worker tests ---

    #[test]
    fn test_get_worker_not_found() {
        let (service, _) = make_service();
        let uuid = uuid::Uuid::new_v4().to_string();
        let result = service.get_worker(&uuid);
        match result {
            Err(WorkerServiceError::NotFound { .. }) => {}
            Err(WorkerServiceError::QueueNotInitialized) => {} // no queue set
            other => panic!("Expected error, got {:?}", other.err()),
        }
    }

    #[test]
    fn test_get_worker_invalid_id() {
        let (service, _) = make_service();
        let result = service.get_worker("not-a-uuid");
        match result {
            Err(WorkerServiceError::InvalidId { .. }) => {}
            other => panic!("Expected InvalidId, got {:?}", other.err()),
        }
    }

    // --- delete_worker / create_worker without queue ---

    #[tokio::test]
    async fn test_delete_worker_queue_not_initialized() {
        let (service, registry) = make_service();
        let id = register_worker(&registry, "http://w1:8000");
        let result = service.delete_worker(id.as_str()).await;
        match result {
            Err(WorkerServiceError::QueueNotInitialized) => {}
            other => panic!("Expected QueueNotInitialized, got {:?}", other.err()),
        }
    }

    #[tokio::test]
    async fn test_delete_worker_not_found() {
        let (service, _) = make_service();
        let uuid = uuid::Uuid::new_v4().to_string();
        let result = service.delete_worker(&uuid).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_create_worker_queue_not_initialized() {
        let (service, _) = make_service();
        let config =
            serde_json::from_str::<WorkerConfigRequest>(r#"{"url": "http://new:8000"}"#).unwrap();
        let result = service.create_worker(config).await;
        match result {
            Err(WorkerServiceError::QueueNotInitialized) => {}
            other => panic!("Expected QueueNotInitialized, got {:?}", other.err()),
        }
    }

    // --- Response type tests ---

    #[test]
    fn test_list_workers_result_into_response() {
        let result = ListWorkersResult {
            workers: vec![],
            total: 0,
            prefill_count: 0,
            decode_count: 0,
            regular_count: 0,
        };
        let response = result.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[test]
    fn test_create_worker_result_into_response() {
        let result = CreateWorkerResult {
            worker_id: WorkerId::from_string("test-id".to_string()),
            url: "http://w1:8000".to_string(),
            location: "/workers/test-id".to_string(),
        };
        let response = result.into_response();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }

    #[test]
    fn test_delete_worker_result_into_response() {
        let result = DeleteWorkerResult {
            worker_id: WorkerId::from_string("test-id".to_string()),
            url: "http://w1:8000".to_string(),
        };
        let response = result.into_response();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }

    #[test]
    fn test_update_worker_result_into_response() {
        let result = UpdateWorkerResult {
            worker_id: WorkerId::from_string("test-id".to_string()),
            url: "http://w1:8000".to_string(),
        };
        let response = result.into_response();
        assert_eq!(response.status(), StatusCode::ACCEPTED);
    }
}
