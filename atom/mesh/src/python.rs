use std::sync::Arc;

use clap::Parser;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyDict};

use crate::{
    cliargs::{
        filter_decode_args_from, filter_prefill_args_from, parse_decode_args_from,
        parse_prefill_args_from, Backend, Cli, CliArgs, Commands,
    },
    config::RoutingMode,
    routers::atom_standalone::AtomStandaloneRuntime,
    server::{self, ServerConfig},
    version,
};

#[pyclass(name = "ServerConfig")]
pub struct PyServerConfig {
    inner: Option<ServerConfig>,
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
    *,
    server_config,
    standalone_service = None
))]
pub fn launch_mesh(
    py: Python<'_>,
    mut server_config: PyRefMut<'_, PyServerConfig>,
    standalone_service: Option<Py<PyAny>>,
) -> PyResult<()> {
    let runtime = standalone_service.map(|service| {
        Arc::new(AtomStandaloneRuntime {
            service,
            close_service_on_shutdown: false,
        })
    });

    let mut server_config = server_config.inner.take().unwrap();
    server_config.router_config.atom_standalone = runtime.is_some();
    server_config.atom_standalone_runtime = runtime;

    py.detach(move || startup_runtime(server_config))
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
    m.add_function(wrap_pyfunction!(launch_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(parse_from, m)?)?;
    m.add_function(wrap_pyfunction!(cliargs_backend_name, m)?)?;
    m.add_function(wrap_pyfunction!(version_string, m)?)?;
    m.add_function(wrap_pyfunction!(version_verbose_string, m)?)?;
    Ok(())
}
