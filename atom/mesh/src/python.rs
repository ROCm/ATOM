use std::sync::Arc;

use clap::Parser;
use parking_lot::Mutex;
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyAnyMethods, PyBytes, PyDict, PyDictMethods, PyList, PyListMethods},
};
use serde_json::{Map, Value};

use crate::{
    cliargs::{
        filter_decode_args_from, filter_prefill_args_from, parse_decode_args_from,
        parse_prefill_args_from, Backend, Cli, CliArgs, Commands,
    },
    config::RoutingMode,
    routers::{
        atom_standalone::AtomStandaloneRuntime,
        engine_core::{
            EngineCoreClient, EngineCoreEndpoint, EngineCoreEndpointTopology, EngineCoreTransport,
        },
    },
    server::{self, ServerConfig},
    version,
};

#[pyclass(name = "ServerConfig")]
pub struct PyServerConfig {
    inner: Option<ServerConfig>,
}

#[pyclass(name = "EngineCoreIpcRuntime")]
pub struct PyEngineCoreIpcRuntime {
    pub(crate) inner: Arc<Mutex<EngineCoreTransport>>,
    pub(crate) block_size: i32,
    pub(crate) max_model_len: i64,
    pub(crate) max_pool_tokens: Option<i64>,
    pub(crate) client_slot: Arc<Mutex<Option<Arc<EngineCoreClient>>>>,
    pub(crate) dp_load_balance: String,
    pub(crate) dp_lb_request_equivalent: u64,
    pub(crate) num_draft_tokens: i32,
    pub(crate) has_per_req_cache: bool,
    pub(crate) session_affinity_enabled: bool,
    pub(crate) dp_attention_enabled: bool,
}

impl PyEngineCoreIpcRuntime {
    fn active_client(&self) -> PyResult<Arc<EngineCoreClient>> {
        self.client_slot.lock().clone().ok_or_else(|| {
            PyRuntimeError::new_err(
                "EngineCore client is not active; launch_mesh has not completed socket handoff",
            )
        })
    }
}

#[pymethods]
impl PyEngineCoreIpcRuntime {
    fn wait_until_all_connected(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .lock()
                .wait_until_all_connected()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    fn wait_until_all_ready(&mut self, py: Python<'_>) -> PyResult<Vec<(usize, Option<i64>)>> {
        let capacities = py.detach(|| {
            self.inner
                .lock()
                .wait_until_all_ready()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })?;
        self.max_pool_tokens = capacities.values().filter_map(|capacity| *capacity).min();
        Ok(capacities.into_iter().collect())
    }

    fn shutdown_engine_cores(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(client) = self.client_slot.lock().clone() {
            return py.detach(|| {
                client
                    .shutdown()
                    .map_err(|error| PyRuntimeError::new_err(error.to_string()))
            });
        }
        py.detach(|| {
            self.inner
                .lock()
                .send_shutdown_all()
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.shutdown_engine_cores(py)
    }

    fn mark_engine_failed(&self, engine_rank: usize, message: String) -> PyResult<()> {
        self.active_client()?
            .mark_engine_failed(engine_rank, message)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn mark_rank_failed(&self, rank: usize, message: String) -> PyResult<()> {
        self.active_client()?
            .mark_rank_failed(rank, message)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn send_control_frame(
        &self,
        py: Python<'_>,
        dp_rank: usize,
        protobuf_bytes: &[u8],
    ) -> PyResult<String> {
        let client = self.active_client()?;
        py.detach(|| {
            client
                .send_control_frame(dp_rank, protobuf_bytes)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    fn broadcast_control_frame(&self, py: Python<'_>, protobuf_bytes: &[u8]) -> PyResult<String> {
        let client = self.active_client()?;
        py.detach(|| {
            client
                .broadcast_control_frame(protobuf_bytes)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })
    }

    #[pyo3(signature = (protobuf_bytes, expected_count = None, timeout_ms = 300_000))]
    fn execute_control_frame_all(
        &self,
        py: Python<'_>,
        protobuf_bytes: &[u8],
        expected_count: Option<usize>,
        timeout_ms: u64,
    ) -> PyResult<Vec<(usize, Py<PyBytes>)>> {
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be positive"));
        }
        let client = self.active_client()?;
        let frames = py.detach(|| {
            client
                .execute_control_frame_all_blocking(
                    protobuf_bytes,
                    expected_count,
                    std::time::Duration::from_millis(timeout_ms),
                )
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        })?;
        Ok(frames
            .into_iter()
            .map(|(rank, frame)| (rank, PyBytes::new(py, &frame).unbind()))
            .collect())
    }

    #[pyo3(signature = (command, expected_count = None, timeout_ms = 300_000))]
    fn wait_utility_responses(
        &self,
        py: Python<'_>,
        command: &str,
        expected_count: Option<usize>,
        timeout_ms: u64,
    ) -> PyResult<Vec<(usize, Py<PyBytes>)>> {
        if timeout_ms == 0 {
            return Err(PyValueError::new_err("timeout_ms must be positive"));
        }
        let client = self.active_client()?;
        let expected = expected_count.unwrap_or_else(|| client.total_rank_count());
        let command = command.to_string();
        let frames = py.detach(|| {
            let deadline = std::time::Instant::now()
                .checked_add(std::time::Duration::from_millis(timeout_ms))
                .ok_or_else(|| PyValueError::new_err("timeout_ms is too large"))?;
            let mut responses = std::collections::BTreeMap::new();
            loop {
                responses.extend(client.take_utility_response_frames(&command));
                if responses.len() >= expected {
                    return Ok(responses.into_iter().collect::<Vec<_>>());
                }
                if std::time::Instant::now() >= deadline {
                    client.poison_utility_command(&command);
                    return Err(PyRuntimeError::new_err(format!(
                        "timed out waiting for {expected} EngineCore responses to {command:?}; got ranks {:?}",
                        responses.keys().collect::<Vec<_>>()
                    )));
                }
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        })?;
        Ok(frames
            .into_iter()
            .map(|(rank, frame)| (rank, PyBytes::new(py, &frame).unbind()))
            .collect())
    }
}

#[pymethods]
impl PyServerConfig {
    fn __repr__(&self) -> String {
        let Some(config) = self.inner.as_ref() else {
            return "ServerConfig(<consumed>)".to_string();
        };
        let mode = match &config.router_config.mode {
            RoutingMode::Regular { worker_urls } => {
                format!("regular, worker_urls={worker_urls:?}")
            }
            RoutingMode::PrefillDecode {
                prefill_urls,
                decode_urls,
                ..
            } => {
                format!("pd, prefill_urls={prefill_urls:?}, decode_urls={decode_urls:?}")
            }
        };

        format!(
            "ServerConfig(host='{}', port={}, mode={}, backend={:?}, policy={:?}, atom_standalone={})",
            config.host,
            config.port,
            mode,
            config.router_config.backend,
            config.router_config.policy,
            config.router_config.atom_standalone,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (
    engine_core_endpoints,
    block_size,
    max_model_len,
    dp_load_balance = "round_robin",
    dp_lb_request_equivalent = 256,
    num_draft_tokens = 0,
    has_per_req_cache = false,
    receive_timeout_ms = 300_000,
    session_affinity_enabled = false,
    dp_attention_enabled = false
))]
pub fn bind_engine_core_ipc(
    py: Python<'_>,
    engine_core_endpoints: Py<PyAny>,
    block_size: i32,
    max_model_len: i64,
    dp_load_balance: &str,
    dp_lb_request_equivalent: u64,
    num_draft_tokens: i32,
    has_per_req_cache: bool,
    receive_timeout_ms: i32,
    session_affinity_enabled: bool,
    dp_attention_enabled: bool,
) -> PyResult<Py<PyEngineCoreIpcRuntime>> {
    if receive_timeout_ms <= 0 {
        return Err(PyValueError::new_err(
            "receive_timeout_ms must be a positive integer",
        ));
    }
    if block_size <= 0 {
        return Err(PyValueError::new_err("block_size must be positive"));
    }
    if max_model_len <= 0 {
        return Err(PyValueError::new_err("max_model_len must be positive"));
    }
    if num_draft_tokens < 0 {
        return Err(PyValueError::new_err(
            "num_draft_tokens must be non-negative",
        ));
    }
    if !matches!(
        dp_load_balance,
        "round_robin" | "least_requests" | "least_tokens"
    ) {
        return Err(PyValueError::new_err(format!(
            "unsupported dp_load_balance strategy {dp_load_balance:?}"
        )));
    }
    let topology = extract_engine_core_topology(py, engine_core_endpoints)?;
    let transport = EngineCoreTransport::bind(&topology, receive_timeout_ms)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    Py::new(
        py,
        PyEngineCoreIpcRuntime {
            inner: Arc::new(Mutex::new(transport)),
            block_size,
            max_model_len,
            max_pool_tokens: None,
            client_slot: Arc::new(Mutex::new(None)),
            dp_load_balance: dp_load_balance.to_string(),
            dp_lb_request_equivalent,
            num_draft_tokens,
            has_per_req_cache,
            session_affinity_enabled,
            dp_attention_enabled,
        },
    )
}

fn extract_engine_core_topology(
    py: Python<'_>,
    engine_core_endpoints: Py<PyAny>,
) -> PyResult<EngineCoreEndpointTopology> {
    let endpoint_list = engine_core_endpoints
        .bind(py)
        .cast::<PyList>()
        .map_err(|_| PyValueError::new_err("engine_core_endpoints must be a list"))?;
    if endpoint_list.is_empty() {
        return Err(PyValueError::new_err(
            "engine_core_endpoints must contain at least one endpoint",
        ));
    }

    let mut endpoints = Vec::with_capacity(endpoint_list.len());
    for (index, endpoint) in endpoint_list.iter().enumerate() {
        let endpoint = endpoint.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err(format!("engine_core_endpoints[{index}] must be a mapping"))
        })?;
        let required = |key: &str| {
            endpoint.get_item(key)?.ok_or_else(|| {
                PyValueError::new_err(format!("engine_core_endpoints[{index}].{key} is required"))
            })
        };
        let dp_rank = required("dp_rank")?.extract::<usize>().map_err(|_| {
            PyValueError::new_err(format!(
                "engine_core_endpoints[{index}].dp_rank must be non-negative"
            ))
        })?;
        let optional_rank = |key: &str, default: usize| -> PyResult<usize> {
            match endpoint.get_item(key)? {
                Some(value) => value.extract::<usize>().map_err(|_| {
                    PyValueError::new_err(format!(
                        "engine_core_endpoints[{index}].{key} must be non-negative"
                    ))
                }),
                None => Ok(default),
            }
        };
        // Preserve the pre-PP endpoint schema for external PP=1 callers.
        let engine_rank = optional_rank("engine_rank", dp_rank)?;
        let pp_rank = optional_rank("pp_rank", 0)?;
        let address = |key: &str| {
            required(key)?.extract::<String>().map_err(|_| {
                PyValueError::new_err(format!(
                    "engine_core_endpoints[{index}].{key} must be a string"
                ))
            })
        };
        endpoints.push(EngineCoreEndpoint {
            engine_rank,
            dp_rank,
            pp_rank,
            input_address: address("input_address")?,
            control_address: address("control_address")?,
            output_address: address("output_address")?,
        });
    }

    EngineCoreEndpointTopology::new(endpoints).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (
    *,
    server_config,
    standalone_service = None,
    engine_core_ipc = None,
    default_chat_template_kwargs = None
))]
pub fn launch_mesh(
    py: Python<'_>,
    mut server_config: PyRefMut<'_, PyServerConfig>,
    standalone_service: Option<Py<PyAny>>,
    engine_core_ipc: Option<Py<PyEngineCoreIpcRuntime>>,
    default_chat_template_kwargs: Option<Py<PyAny>>,
) -> PyResult<()> {
    let default_chat_template_kwargs = extract_json_object(py, default_chat_template_kwargs)?;
    let mut server_config = server_config.inner.take().unwrap();
    let shutdown_grace_period_secs = server_config.shutdown_grace_period_secs;
    let runtime = match (standalone_service, engine_core_ipc) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "standalone_service and engine_core_ipc are mutually exclusive",
            ));
        }
        (Some(service), None) => Some(Arc::new(AtomStandaloneRuntime {
            service: Some(service),
            engine_core_transport: None,
            engine_core_block_size: None,
            engine_core_max_model_len: None,
            engine_core_max_pool_tokens: None,
            engine_core_client_slot: None,
            engine_core_shutdown_grace_period_secs: shutdown_grace_period_secs,
            engine_core_dp_load_balance: None,
            engine_core_dp_lb_request_equivalent: None,
            engine_core_num_draft_tokens: None,
            engine_core_has_per_req_cache: None,
            engine_core_session_affinity_enabled: false,
            engine_core_dp_attention_enabled: false,
            default_chat_template_kwargs: default_chat_template_kwargs.clone(),
            close_service_on_shutdown: false,
        })),
        (None, Some(engine_core_ipc)) => Some(Arc::new(AtomStandaloneRuntime {
            service: None,
            engine_core_transport: Some(engine_core_ipc.bind(py).borrow().inner.clone()),
            engine_core_block_size: Some(engine_core_ipc.bind(py).borrow().block_size),
            engine_core_max_model_len: Some(engine_core_ipc.bind(py).borrow().max_model_len),
            engine_core_max_pool_tokens: engine_core_ipc.bind(py).borrow().max_pool_tokens,
            engine_core_client_slot: Some(engine_core_ipc.bind(py).borrow().client_slot.clone()),
            engine_core_shutdown_grace_period_secs: shutdown_grace_period_secs,
            engine_core_dp_load_balance: Some(
                engine_core_ipc.bind(py).borrow().dp_load_balance.clone(),
            ),
            engine_core_dp_lb_request_equivalent: Some(
                engine_core_ipc.bind(py).borrow().dp_lb_request_equivalent,
            ),
            engine_core_num_draft_tokens: Some(engine_core_ipc.bind(py).borrow().num_draft_tokens),
            engine_core_has_per_req_cache: Some(
                engine_core_ipc.bind(py).borrow().has_per_req_cache,
            ),
            engine_core_session_affinity_enabled: engine_core_ipc
                .bind(py)
                .borrow()
                .session_affinity_enabled,
            engine_core_dp_attention_enabled: engine_core_ipc
                .bind(py)
                .borrow()
                .dp_attention_enabled,
            default_chat_template_kwargs,
            close_service_on_shutdown: false,
        })),
        (None, None) => None,
    };

    server_config.router_config.atom_standalone = runtime.is_some();
    server_config.atom_standalone_runtime = runtime;

    py.detach(move || startup_runtime(server_config))
}

fn extract_json_object(py: Python<'_>, value: Option<Py<PyAny>>) -> PyResult<Map<String, Value>> {
    let Some(value) = value else {
        return Ok(Map::new());
    };
    let serialized = py
        .import("json")?
        .call_method1("dumps", (value.bind(py),))?
        .extract::<String>()?;
    match serde_json::from_str::<Value>(&serialized)
        .map_err(|error| PyValueError::new_err(error.to_string()))?
    {
        Value::Object(object) => Ok(object),
        _ => Err(PyValueError::new_err(
            "default_chat_template_kwargs must be a JSON object",
        )),
    }
}

fn startup_runtime(server_config: ServerConfig) -> PyResult<()> {
    let tokio_runtime = tokio::runtime::Runtime::new()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {e}")))?;
    tokio_runtime
        .block_on(async move { server::startup(server_config).await })
        .map_err(|e| PyRuntimeError::new_err(format!("Atomesh exited with error: {e}")))
}

fn build_server_config(
    cli_args: &CliArgs,
    prefill_urls: Vec<(String, Option<u16>)>,
) -> PyResult<PyServerConfig> {
    let router_config = cli_args
        .to_router_config(prefill_urls)
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid router config: {e}")))?;
    router_config
        .validate()
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid router config: {e}")))?;
    let server_config = cli_args.to_server_config(router_config);

    Ok(PyServerConfig {
        inner: Some(server_config),
    })
}

#[pyfunction]
pub fn parse_from(py: Python<'_>, args: Vec<String>) -> PyResult<Py<PyDict>> {
    let prefill_urls = parse_prefill_args_from(&args);
    let decode_urls = parse_decode_args_from(&args);
    let filtered_args = filter_prefill_args_from(&args);
    let filtered_args = filter_decode_args_from(&filtered_args);
    let mut clap_args = Vec::with_capacity(filtered_args.len() + 1);
    clap_args.push("atomesh".to_string());
    clap_args.extend(filtered_args);

    let cli = Cli::parse_from(clap_args);
    let cli_args = match cli.command {
        Some(Commands::Launch { args }) => args,
        None => cli.router_args,
    };
    let server_config = build_server_config(&cli_args, prefill_urls.clone())?;

    let parsed = PyDict::new(py);
    let cli_args_dict = PyDict::new(py);

    cli_args_dict.set_item("host", cli_args.host)?;
    cli_args_dict.set_item("port", cli_args.port)?;
    cli_args_dict.set_item("worker_urls", cli_args.worker_urls)?;
    cli_args_dict.set_item("policy", cli_args.policy)?;
    cli_args_dict.set_item("cache_threshold", cli_args.cache_threshold)?;
    cli_args_dict.set_item("balance_abs_threshold", cli_args.balance_abs_threshold)?;
    cli_args_dict.set_item("balance_rel_threshold", cli_args.balance_rel_threshold)?;
    cli_args_dict.set_item("eviction_interval", cli_args.eviction_interval)?;
    cli_args_dict.set_item("max_tree_size", cli_args.max_tree_size)?;
    cli_args_dict.set_item("prefix_token_count", cli_args.prefix_token_count)?;
    cli_args_dict.set_item("prefix_hash_load_factor", cli_args.prefix_hash_load_factor)?;
    cli_args_dict.set_item("dp_aware", cli_args.dp_aware)?;
    cli_args_dict.set_item("pd_disaggregation", cli_args.pd_disaggregation)?;
    cli_args_dict.set_item("decode", cli_args.decode)?;
    cli_args_dict.set_item("prefill_policy", cli_args.prefill_policy)?;
    cli_args_dict.set_item("decode_policy", cli_args.decode_policy)?;
    cli_args_dict.set_item(
        "worker_startup_timeout_secs",
        cli_args.worker_startup_timeout_secs,
    )?;
    cli_args_dict.set_item(
        "worker_startup_check_interval",
        cli_args.worker_startup_check_interval,
    )?;
    cli_args_dict.set_item("log_dir", cli_args.log_dir)?;
    cli_args_dict.set_item("log_level", cli_args.log_level)?;
    cli_args_dict.set_item("json_log", cli_args.json_log)?;
    cli_args_dict.set_item("prometheus_port", cli_args.prometheus_port)?;
    cli_args_dict.set_item("prometheus_host", cli_args.prometheus_host)?;
    cli_args_dict.set_item(
        "prometheus_duration_buckets",
        cli_args.prometheus_duration_buckets,
    )?;
    cli_args_dict.set_item("request_id_headers", cli_args.request_id_headers)?;
    cli_args_dict.set_item("request_timeout_secs", cli_args.request_timeout_secs)?;
    cli_args_dict.set_item(
        "shutdown_grace_period_secs",
        cli_args.shutdown_grace_period_secs,
    )?;
    cli_args_dict.set_item("max_payload_size", cli_args.max_payload_size)?;
    cli_args_dict.set_item("max_concurrent_requests", cli_args.max_concurrent_requests)?;
    cli_args_dict.set_item("queue_size", cli_args.queue_size)?;
    cli_args_dict.set_item("queue_timeout_secs", cli_args.queue_timeout_secs)?;
    cli_args_dict.set_item(
        "rate_limit_tokens_per_second",
        cli_args.rate_limit_tokens_per_second,
    )?;
    cli_args_dict.set_item("retry_max_retries", cli_args.retry_max_retries)?;
    cli_args_dict.set_item(
        "retry_initial_backoff_ms",
        cli_args.retry_initial_backoff_ms,
    )?;
    cli_args_dict.set_item("retry_max_backoff_ms", cli_args.retry_max_backoff_ms)?;
    cli_args_dict.set_item(
        "retry_backoff_multiplier",
        cli_args.retry_backoff_multiplier,
    )?;
    cli_args_dict.set_item("retry_jitter_factor", cli_args.retry_jitter_factor)?;
    cli_args_dict.set_item("disable_retries", cli_args.disable_retries)?;
    cli_args_dict.set_item("cb_failure_threshold", cli_args.cb_failure_threshold)?;
    cli_args_dict.set_item("cb_success_threshold", cli_args.cb_success_threshold)?;
    cli_args_dict.set_item(
        "cb_timeout_duration_secs",
        cli_args.cb_timeout_duration_secs,
    )?;
    cli_args_dict.set_item("cb_window_duration_secs", cli_args.cb_window_duration_secs)?;
    cli_args_dict.set_item("disable_circuit_breaker", cli_args.disable_circuit_breaker)?;
    cli_args_dict.set_item(
        "health_failure_threshold",
        cli_args.health_failure_threshold,
    )?;
    cli_args_dict.set_item(
        "health_success_threshold",
        cli_args.health_success_threshold,
    )?;
    cli_args_dict.set_item(
        "health_check_timeout_secs",
        cli_args.health_check_timeout_secs,
    )?;
    cli_args_dict.set_item(
        "health_check_interval_secs",
        cli_args.health_check_interval_secs,
    )?;
    cli_args_dict.set_item("health_check_endpoint", cli_args.health_check_endpoint)?;
    cli_args_dict.set_item("disable_health_check", cli_args.disable_health_check)?;
    cli_args_dict.set_item("model_path", cli_args.model_path)?;
    cli_args_dict.set_item("tokenizer_path", cli_args.tokenizer_path)?;
    cli_args_dict.set_item("chat_template", cli_args.chat_template)?;
    cli_args_dict.set_item(
        "tokenizer_cache_enable_l0",
        cli_args.tokenizer_cache_enable_l0,
    )?;
    cli_args_dict.set_item(
        "tokenizer_cache_l0_max_entries",
        cli_args.tokenizer_cache_l0_max_entries,
    )?;
    cli_args_dict.set_item(
        "tokenizer_cache_enable_l1",
        cli_args.tokenizer_cache_enable_l1,
    )?;
    cli_args_dict.set_item(
        "tokenizer_cache_l1_max_memory",
        cli_args.tokenizer_cache_l1_max_memory,
    )?;
    cli_args_dict.set_item("reasoning_parser", cli_args.reasoning_parser)?;
    cli_args_dict.set_item("tool_call_parser", cli_args.tool_call_parser)?;
    cli_args_dict.set_item("backend", cli_args.backend.to_string())?;
    cli_args_dict.set_item("api_key", cli_args.api_key)?;

    parsed.set_item("cli_args", cli_args_dict)?;
    parsed.set_item("prefill_urls", prefill_urls)?;
    parsed.set_item("decode_urls", decode_urls)?;
    parsed.set_item("server_config", Py::new(py, server_config)?)?;
    Ok(parsed.unbind())
}

#[pyfunction]
pub fn cliargs_backend_name(backend: String) -> PyResult<String> {
    let backend = backend
        .parse::<Backend>()
        .map_err(|e| PyRuntimeError::new_err(format!("Invalid backend: {e}")))?;
    Ok(backend.to_string())
}

#[pyfunction]
pub fn version_string() -> String {
    version::get_version_string()
}

#[pyfunction]
pub fn version_verbose_string() -> String {
    version::get_verbose_version_string()
}

#[pymodule]
pub fn atomesh_runner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyServerConfig>()?;
    m.add_class::<PyEngineCoreIpcRuntime>()?;
    m.add_function(wrap_pyfunction!(bind_engine_core_ipc, m)?)?;
    m.add_function(wrap_pyfunction!(launch_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(parse_from, m)?)?;
    m.add_function(wrap_pyfunction!(cliargs_backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(version_string, m)?)?;
    m.add_function(wrap_pyfunction!(version_verbose_string, m)?)?;
    Ok(())
}
