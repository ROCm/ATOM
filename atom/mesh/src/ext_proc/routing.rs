use std::{collections::HashMap, net::SocketAddr, sync::Arc};

use crate::{
    app_context::AppContext,
    core::{
        placement::{
            planner::DefaultPlanner,
            registry_adapters::{PolicyRegistryAdapter, WorkerRegistryAdapter},
            traits::{PdPlanner, PolicySource},
            types::{PlacementPlan, Protocol, RequestDescriptor},
        },
        Worker, WorkerLoadGuard,
    },
    policies::LoadBalancingPolicy,
};

use super::{
    error::ProcessingError,
    executor::{ExecutionLease, PdExecutor},
    request::{RequestEnvelope, RoutingInput},
};

pub(super) struct RoutingDecision {
    pub target: ExecutionTarget,
    pub address: SocketAddr,
    pub api_key: Option<String>,
}

pub(super) enum ExecutionTarget {
    Single {
        worker: Arc<dyn Worker>,
        policy: Arc<dyn LoadBalancingPolicy>,
        _load: WorkerLoadGuard,
    },
    Pair(ExecutionLease),
}

impl ExecutionTarget {
    pub fn execution_id(&self) -> Option<&str> {
        match self {
            Self::Pair(lease) => Some(&lease.id),
            Self::Single { .. } => None,
        }
    }

    pub fn complete(&self, health: Option<bool>, success: bool) {
        if let Self::Single { worker, policy, .. } = self {
            if let Some(health) = health {
                worker.record_outcome(health);
            }
            policy.on_request_complete(worker.url(), success);
        }
        // PD worker outcomes belong to the individual HTTP attempts. A pair's
        // aggregate status must not charge a decode error to the prefill worker.
    }

    pub fn expiration(&self) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
        match self {
            Self::Pair(lease) => lease.expiration(),
            Self::Single { .. } => Box::pin(std::future::pending()),
        }
    }
}

pub(super) struct EndpointRouter {
    app: Arc<AppContext>,
    executor: Option<Arc<PdExecutor>>,
    policies: Arc<PolicyRegistryAdapter>,
}

impl EndpointRouter {
    pub fn new(app: Arc<AppContext>, executor: Option<Arc<PdExecutor>>) -> Self {
        let policies = Arc::new(PolicyRegistryAdapter::new(app.policy_registry.clone()));
        Self {
            app,
            executor,
            policies,
        }
    }

    pub async fn select(
        &self,
        request: &mut RequestEnvelope,
        input: &RoutingInput,
    ) -> Result<RoutingDecision, ProcessingError> {
        let model = input.model.as_deref();
        let mut workers = crate::routers::ingress::IngressRouting::new(&self.app)
            .candidates(&input.metadata, input.state_reference)
            .map_err(ProcessingError::from)?;
        let mut resolved = HashMap::new();
        if let Some(subset) = &request.subset {
            let mut candidates = Vec::new();
            for worker in workers {
                // The executor uses the worker's URL. DNS could resolve to a
                // different peer between selection and execution, so PD subsets
                // require literal IP origins. Regular routes pin the resolved IP.
                if self.executor.is_some()
                    && !url::Url::parse(worker.base_url()).ok().is_some_and(|url| {
                        matches!(url.host(), Some(url::Host::Ipv4(_) | url::Host::Ipv6(_)))
                    })
                {
                    continue;
                }
                if let Ok(address) = Self::address(worker.as_ref()).await {
                    if subset.contains(&address.to_string()) {
                        resolved.insert(worker.url().to_owned(), address);
                        candidates.push(worker);
                    }
                }
            }
            workers = candidates;
        }
        let source = Arc::new(WorkerRegistryAdapter::with_candidates(
            self.app.worker_registry.clone(),
            workers,
        ));
        let planner = DefaultPlanner::new(source, self.policies.clone());
        let descriptor = RequestDescriptor {
            model_id: model,
            protocol: Some(Protocol::Http),
            text: Some(&input.text),
            tokens: input.tokens.as_deref(),
            headers: Some(&request.headers),
            stream: input.stream,
        };
        let plan = planner.plan(&descriptor).await.map_err(|e| {
            ProcessingError::from(crate::routers::comm::error::IngressError::placement(
                e, model,
            ))
        })?;
        let worker = match plan {
            PlacementPlan::Single { worker, .. } => worker,
            pair @ PlacementPlan::Pair { .. } => {
                let executor = self.executor.as_ref().ok_or_else(|| {
                    ProcessingError::new(
                        501,
                        "pd_executor_required",
                        "Prefill/Decode requires a configured PD executor",
                    )
                })?;
                return Ok(RoutingDecision {
                    target: ExecutionTarget::Pair(executor.reserve(
                        pair,
                        request,
                        &input.metadata,
                    )?),
                    address: executor.address,
                    api_key: None,
                });
            }
        };
        let load = WorkerLoadGuard::new(worker.clone(), Some(&request.headers));
        let address = match resolved.get(worker.url()) {
            Some(address) => *address,
            None => Self::address(worker.as_ref()).await?,
        };
        if worker.is_dp_aware() {
            let body = serde_json::from_slice(&request.raw)?;
            let body = worker
                .prepare_request(body)
                .await
                .map_err(|e| ProcessingError::invalid(e.to_string()))?;
            request.replace_body(
                serde_json::to_vec(&body)?,
                self.app.router_config.ext_proc.max_body_bytes,
            )?;
        }
        let api_key = worker.api_key().clone();
        Ok(RoutingDecision {
            target: ExecutionTarget::Single {
                worker,
                _load: load,
                policy: self.policies.regular_policy(model),
            },
            address,
            api_key,
        })
    }

    pub async fn address(worker: &dyn Worker) -> Result<SocketAddr, ProcessingError> {
        let invalid = || {
            ProcessingError::new(
                503,
                "invalid_endpoint",
                "endpoint must be an HTTP origin without credentials or a path prefix",
            )
        };
        let url = url::Url::parse(worker.base_url()).map_err(|_| invalid())?;
        if url.scheme() != "http"
            || !url.username().is_empty()
            || url.password().is_some()
            || !matches!(url.path(), "" | "/")
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(invalid());
        }
        let port = url.port_or_known_default().ok_or_else(invalid)?;
        match url.host().ok_or_else(invalid)? {
            url::Host::Ipv4(ip) => Ok(SocketAddr::new(ip.into(), port)),
            url::Host::Ipv6(ip) => Ok(SocketAddr::new(ip.into(), port)),
            url::Host::Domain(host) => tokio::net::lookup_host((host, port))
                .await
                .map_err(|_| {
                    ProcessingError::new(
                        503,
                        "endpoint_dns_failed",
                        "cannot resolve worker address",
                    )
                })?
                .next()
                .ok_or_else(invalid),
        }
    }
}
