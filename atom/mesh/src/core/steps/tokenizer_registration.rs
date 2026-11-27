//! Tokenizer registration workflow
//!
//! This module provides a workflow for registering tokenizers asynchronously.
//! Tokenizers can be loaded from local paths or downloaded from HuggingFace.
//!
//! This is the **single source of truth** for tokenizer registration. All paths
//! (startup, worker connection, API) should use this workflow to ensure consistent
//! behavior (validation, caching, deduplication).

use std::{path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{error, info, warn};
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId, StepResult,
    WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
};
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::workflow_data::TokenizerWorkflowData;
use crate::{
    app_context::AppContext,
    config::TokenizerCacheConfig,
    core::Worker,
    routers::prepare::{
        chat_render::ChatControls,
        fast_tokenizer::{
            tokenizer_json, with_fastokens, FastTokenizer, DOWNLOAD_FAILURE, STRICT_REFUSAL,
        },
    },
    tokenizer::{
        cache::{CacheConfig, CachedTokenizer},
        chat_template::ChatTemplateParams,
        factory,
        registry::LoadOutcome,
        traits::Tokenizer,
    },
};

/// Configuration for adding a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfigRequest {
    /// Pre-generated UUID for this tokenizer
    pub id: String,
    /// User-provided name (what to register under in the registry)
    pub name: String,
    /// Source: either a local path or HuggingFace model ID
    pub source: String,
    /// Optional path to chat template file
    pub chat_template_path: Option<String>,
    /// Chat template content, used when `chat_template_path` is unset
    pub inline_chat_template: Option<String>,
    /// The worker whose registration submitted this job, if one did: what it loads is
    /// kept only while that worker is registered serving the job's template.
    #[serde(default)]
    pub worker_url: Option<String>,
    /// Optional cache configuration. If provided, wraps a tokenizer that encodes
    /// with HuggingFace with CachedTokenizer; one that encodes with fastokens is
    /// not wrapped, and has fastokens' segment cache instead, sized by
    /// `--fastokens-segment-cache-mb`.
    #[serde(default)]
    pub cache_config: Option<TokenizerCacheConfig>,
    /// If true, the workflow fails when a tokenizer with the same name already exists.
    /// If false (default), the workflow succeeds and returns the existing tokenizer's ID.
    /// API callers should set this to true.
    #[serde(default)]
    pub fail_on_duplicate: bool,
}

/// Configuration for removing a tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerRemovalRequest {
    /// UUID of the tokenizer to remove
    pub id: String,
}

// ============================================================================
// Workflow Steps
// ============================================================================

/// Load the tokenizer from source (local path or HuggingFace)
///
/// This step handles:
/// - Input validation (via registry.load())
/// - Deduplication (returns success if already exists)
/// - Loading from local path or HuggingFace
/// - Encoding with fastokens, for the models `--fastokens-models` lists
/// - Optional caching layer wrapping, for the tokenizers that encode with HuggingFace
pub struct LoadTokenizerStep;

#[async_trait]
impl StepExecutor<TokenizerWorkflowData> for LoadTokenizerStep {
    async fn execute(
        &self,
        context: &mut WorkflowContext<TokenizerWorkflowData>,
    ) -> WorkflowResult<StepResult> {
        let config = &context.data.config;
        let app_context = context
            .data
            .app_context
            .as_ref()
            .ok_or_else(|| WorkflowError::ContextValueNotFound("app_context".to_string()))?
            .clone();

        info!(
            "Loading tokenizer '{}' (id: {}) from source: {}",
            config.name, config.id, config.source
        );

        // Clone needed values before async move
        let id = config.id.clone();
        let name = config.name.clone();
        let source = config.source.clone();
        let chat_template = config.chat_template_path.clone();
        let inline_chat_template = config.inline_chat_template.clone();
        let cache_config = config.cache_config.clone();
        let chat_controls = app_context.chat_controls.clone();
        let fastokens = app_context.router_config.fastokens.clone();

        // The template a worker advertised, which no --chat-template overrides.
        let inline_template = chat_template
            .is_none()
            .then(|| inline_chat_template.clone())
            .flatten();

        // Load the tokenizer using the registry's load method
        // This handles: validation, deduplication, and loading
        let registry = &app_context.tokenizer_registry;
        let loader = || {
            let id = id.clone();
            let name = name.clone();
            let source = source.clone();
            let chat_template = chat_template.clone();
            let inline_chat_template = inline_chat_template.clone();
            let cache_cfg = cache_config.clone();
            let chat_controls = chat_controls.clone();
            let fastokens = fastokens.clone();
            async move {
                // Load base tokenizer
                let mut base_tokenizer = factory::create_tokenizer_async_with_chat_template(
                    &source,
                    chat_template.as_deref(),
                )
                .await
                .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
                if let (None, Some(template)) = (&chat_template, inline_chat_template) {
                    Arc::get_mut(&mut base_tokenizer)
                        .ok_or("the loaded tokenizer is already shared")?
                        .set_chat_template(template)
                        .map_err(|e| format!("Failed to set the chat template: {}", e))?;
                }
                let base_tokenizer =
                    with_fastokens(base_tokenizer, &name, &source, &fastokens).await?;
                let controls = checkpoint_chat_controls(&name, &source).await?;
                probe_chat_render(&name, &source, chat_template.as_deref(), controls.is_some())
                    .await?;
                if let Some(controls) = controls {
                    chat_controls.write().insert(id.clone(), controls);
                }
                // fastokens has its own segment cache
                // (`--fastokens-segment-cache-mb`, 1 GiB by default).
                if base_tokenizer.as_any().is::<FastTokenizer>() {
                    return Ok(base_tokenizer);
                }

                // Wrap with caching layer if configured
                let tokenizer: Arc<dyn Tokenizer> = match cache_cfg {
                    Some(cfg) if cfg.enable_l0 || cfg.enable_l1 => {
                        let cache_config = CacheConfig {
                            enable_l0: cfg.enable_l0,
                            l0_max_entries: cfg.l0_max_entries,
                            enable_l1: cfg.enable_l1,
                            l1_max_memory: cfg.l1_max_memory,
                            ..CacheConfig::default()
                        };
                        Arc::new(CachedTokenizer::new(base_tokenizer, cache_config))
                    }
                    _ => base_tokenizer,
                };

                Ok(tokenizer)
            }
        };
        // One registration of a name at a time, so a template is recorded for the
        // tokenizer that loads with it. A loaded tokenizer is never replaced: all live
        // workers of a model serve one template (checked as each registers).
        let registering = app_context
            .tokenizer_registering
            .entry(name.clone())
            .or_default()
            .clone();
        let _registering = registering.lock().await;
        // Its template recorded first: a chat that finds the tokenizer finds that too.
        if let Some(template) = &inline_template {
            let templates = &app_context.tokenizer_inline_templates;
            templates.write().insert(id.clone(), template.clone());
        }
        if let Some(path) = &chat_template {
            let files = &app_context.tokenizer_template_files;
            files.write().insert(id.clone(), path.clone());
        }
        let result = registry.load(&id, &name, &source, &loader).await;
        if registry.contains_id(&id) && worker_left(&app_context, config) {
            info!("Tokenizer '{name}' dropped: the worker that registered it left");
            drop_tokenizer(&app_context, &name);
            return Ok(StepResult::Success);
        }
        if !registry.contains_id(&id) {
            app_context.forget_tokenizer(&id); // what it recorded for a tokenizer not loaded
        }

        match result {
            Ok(outcome) => {
                let loaded_id = outcome.id();

                // Get vocab size, and how it encodes, for logging
                let loaded = app_context.tokenizer_registry.get_by_id(loaded_id);
                let vocab_size = loaded.as_ref().map(|e| e.tokenizer.vocab_size());
                let encoding = match loaded.as_ref().map(|e| e.tokenizer.as_any()) {
                    Some(tokenizer) if tokenizer.is::<FastTokenizer>() => {
                        ", encoding with fastokens"
                    }
                    Some(tokenizer) if tokenizer.is::<CachedTokenizer>() => ", with caching",
                    _ => "",
                };

                match &outcome {
                    LoadOutcome::Loaded { id } => {
                        info!(
                            "Successfully loaded tokenizer '{}' (id: {}) with vocab_size: {:?}{}",
                            name, id, vocab_size, encoding
                        );
                    }
                    LoadOutcome::AlreadyExists { id } => {
                        if config.fail_on_duplicate {
                            return Err(WorkflowError::StepFailed {
                                step_id: StepId::new("load_tokenizer"),
                                message: format!(
                                    "Tokenizer '{}' already exists (id: {})",
                                    name, id
                                ),
                            });
                        }
                        info!(
                            "Tokenizer '{}' already exists (id: {}), skipping load",
                            name, id
                        );
                    }
                }

                // Store vocab size in typed data
                if let Some(size) = vocab_size {
                    context.data.vocab_size = Some(size);
                }

                Ok(StepResult::Success)
            }
            Err(e) => {
                error!("Failed to load tokenizer '{}': {}", name, e);
                Err(WorkflowError::StepFailed {
                    step_id: StepId::new("load_tokenizer"),
                    message: e.to_string(),
                })
            }
        }
    }

    fn is_retryable(&self, error: &WorkflowError) -> bool {
        // Network/IO errors are retryable, among them a `tokenizer.json` fastokens
        // cannot download under `--fastokens-strict`; a `--fastokens-strict`
        // refusal is what the probe found, and a retry would only probe again;
        // so is a model whose chats would render without ATOM's controls.
        !matches!(
            error,
            WorkflowError::StepFailed { message, .. }
                if message.starts_with(STRICT_REFUSAL) || message.starts_with(UNCONTROLLED_REFUSAL)
        )
    }
}

/// One model, one template across its live workers. An ATOM worker registering at
/// `urls` whose template (its own file, else the file model `name`'s tokenizer was
/// loaded with, else `--chat-template`, else the `advertised` one) differs from the one
/// that tokenizer carries, or while that loads its registered workers serve, is refused
/// while another worker of the model is registered; changing it is a redeploy. With
/// none, that tokenizer is dropped and loads afresh.
pub(crate) fn admit_model_template(
    app_context: &AppContext,
    name: &str,
    urls: &[String],
    own_file: Option<&str>,
    advertised: Option<&str>,
) -> Result<(), String> {
    let router_file = app_context.router_config.chat_template.as_deref();
    let registered = app_context.worker_registry.get_by_model(name);
    let mut others = (registered.iter()).filter(|w| !urls.iter().any(|url| url == w.url()));
    let refused = || {
        Err(format!(
            "its chat template differs from the one model {name}'s registered workers \
             serve; all workers of a model serve one template, so change it by redeploying \
             them"
        ))
    };
    let Some(entry) = app_context.tokenizer_registry.get_by_name(name) else {
        let template = template_fingerprint(own_file.or(router_file), advertised);
        return match others.any(|w| served_template(w.as_ref(), router_file) != template) {
            true => refused(),
            false => Ok(()),
        };
    };
    let loaded_file = (app_context.tokenizer_template_files.read())
        .get(&entry.id)
        .cloned();
    let file = own_file.or(loaded_file.as_deref()).or(router_file);
    if loaded_template(app_context, &entry.id) == Some(template_fingerprint(file, advertised)) {
        return Ok(());
    }
    if others.next().is_some() {
        return refused();
    }
    drop_tokenizer(app_context, name);
    Ok(())
}

/// A chat template's identity: its file, else its text; neither is the tokenizer's own.
pub(crate) fn template_fingerprint(file: Option<&str>, inline: Option<&str>) -> u64 {
    match (file, inline) {
        (Some(file), _) => xxh3_64_with_seed(file.as_bytes(), 1),
        (None, Some(text)) => xxh3_64_with_seed(text.as_bytes(), 2),
        (None, None) => 0,
    }
}

/// The chat template an ATOM worker serves, as its registration chose it: its own file,
/// else `fallback_file` (the router's), else the one it advertised.
pub(crate) fn served_template(worker: &dyn Worker, fallback_file: Option<&str>) -> u64 {
    let file = worker.chat_template().or(fallback_file);
    let labels = &worker.metadata().labels;
    let inline = file
        .is_none()
        .then(|| labels.get("chat_template"))
        .flatten();
    template_fingerprint(file, inline.map(String::as_str))
}

/// The chat template tokenizer `id` was loaded with, while it is loaded.
pub(crate) fn loaded_template(app_context: &AppContext, id: &str) -> Option<u64> {
    let file = app_context.tokenizer_template_files.read().get(id).cloned();
    let inline = app_context
        .tokenizer_inline_templates
        .read()
        .get(id)
        .cloned();
    let fingerprint = template_fingerprint(file.as_deref(), inline.as_deref());
    // Read before the check: a removed tokenizer's records go after it does.
    app_context
        .tokenizer_registry
        .contains_id(id)
        .then_some(fingerprint)
}

/// Whether the worker that submitted `config` is no longer registered serving its
/// template: a removed worker's job must not load the tokenizer its successor needs.
fn worker_left(app_context: &AppContext, config: &TokenizerConfigRequest) -> bool {
    let Some(url) = &config.worker_url else {
        return false;
    };
    let job_template = template_fingerprint(
        config.chat_template_path.as_deref(),
        config.inline_chat_template.as_deref(),
    );
    let router_file = app_context.router_config.chat_template.as_deref();
    app_context
        .worker_registry
        .get_by_url(url)
        .is_none_or(|worker| served_template(worker.as_ref(), router_file) != job_template)
}

/// Drops model `name`'s tokenizer and what its registration recorded.
pub(crate) fn drop_tokenizer(app_context: &AppContext, name: &str) {
    if let Some(entry) = app_context.tokenizer_registry.remove(name) {
        crate::routers::prepare::thread_tokenizer::release_copies(&entry.id);
        app_context.forget_tokenizer(&entry.id);
    }
}

/// How the refusal of a model whose chats llm-tokenizer renders with a
/// built-in encoder, with no `ChatControls` here, begins: ATOM renders them
/// with its own controls, so the router would render them otherwise than the
/// engine (for the gRPC and ATOM bridge routers, the tokens the engine runs).
pub(crate) const UNCONTROLLED_REFUSAL: &str =
    "llm-tokenizer renders the chats of this model with a built-in encoder, without ATOM's controls";

/// Refuse `source` when llm-tokenizer renders its chats without the Jinja
/// template and its `model_type` has no `ChatControls` (not `controlled`).
/// llm-tokenizer has no accessor for that, so a copy is probed; mirroring its
/// `config.json` check would also mean mirroring how it resolves sources. A
/// `controlled` model is probed only to warn that `--chat-template` does not
/// apply; one that cannot be probed is served, with a warning.
async fn probe_chat_render(
    name: &str,
    source: &str,
    chat_template: Option<&str>,
    controlled: bool,
) -> Result<(), String> {
    const UNPROBED: &str = "Cannot probe whether llm-tokenizer renders these chats without the \
                            Jinja template, as ATOM does not";
    if controlled && chat_template.is_none() {
        return Ok(());
    }
    let mut copy = match factory::create_tokenizer_async_with_chat_template(source, None).await {
        Ok(copy) => copy,
        Err(error) => {
            warn!(tokenizer = name, %error, "{UNPROBED}");
            return Ok(());
        }
    };
    let Some(copy) = Arc::get_mut(&mut copy) else {
        warn!(tokenizer = name, "{UNPROBED}: the loaded copy is shared");
        return Ok(());
    };
    if !renders_without_template(copy) {
        return Ok(());
    }
    if !controlled {
        return Err(format!(
            "{UNCONTROLLED_REFUSAL} ({name}): its config.json's model_type is not one \
             ChatControls::of_model_type knows"
        ));
    }
    warn!(
        tokenizer = name,
        "llm-tokenizer renders these chats with a built-in encoder, not the Jinja template, \
         so --chat-template does not apply"
    );
    Ok(())
}

/// Whether `tokenizer` renders chats without its Jinja template, as llm-tokenizer
/// does with the encoders it has built in for the architectures in `config.json`
/// (in 2.0.0 DeepSeek V3.2 and V4.1, and V4 without a template of its own).
/// Such a render ignores `--chat-template`, and its defaults are not ATOM's:
/// DeepSeek V3.2 and V4 render chat mode unless `thinking` is true, where
/// ATOM's encoder renders thinking mode, and V4.1's port takes `thinking: true`
/// over `reasoning_effort: "none"`, where ATOM's renders chat mode. Replaces the
/// template, so probe a copy.
fn renders_without_template(tokenizer: &mut dyn Tokenizer) -> bool {
    const PROBE: &str = "probe";
    tokenizer.set_chat_template(PROBE.to_string()).is_ok()
        && tokenizer
            .apply_chat_template(
                &[json!({"role": "user", "content": ""})],
                ChatTemplateParams::default(),
            )
            .is_ok_and(|text| text != PROBE)
}

/// The encoder ATOM renders `source`'s chats with (`chat_render`), by the
/// `model_type` in the checkpoint's `config.json`: the one in a directory
/// (a tiktoken checkpoint, Kimi-K3's, has no `tokenizer.json`), else the one
/// beside its `tokenizer.json`. A HuggingFace model ID's tokenizer that
/// cannot be downloaded fails the registration, to be retried, rather than
/// leave the model's chats rendered without them.
async fn checkpoint_chat_controls(
    name: &str,
    source: &str,
) -> Result<Option<ChatControls>, String> {
    let dir = Path::new(source);
    if dir.is_dir() {
        return Ok(controls_of(name, &dir.join("config.json")).await);
    }
    match tokenizer_json(source).await {
        Ok(tokenizer) => Ok(controls_of(name, &tokenizer.with_file_name("config.json")).await),
        Err(reason) if reason.starts_with(DOWNLOAD_FAILURE) => Err(reason),
        Err(_) => Ok(None),
    }
}

/// The encoder controls for the `model_type` in the `config` file, if any.
async fn controls_of(name: &str, config: &Path) -> Option<ChatControls> {
    let config = tokio::fs::read(config).await.ok()?;
    let config: Value = serde_json::from_slice(&config).ok()?;
    let model_type = config.get("model_type")?.as_str()?;
    let controls = ChatControls::of_model_type(model_type)?;
    info!(
        tokenizer = name,
        model_type, "These chats render with the model's encoder, as ATOM renders them"
    );
    Some(controls)
}

// ============================================================================
// Workflow Builder
// ============================================================================

/// Create the tokenizer registration workflow
///
/// This workflow loads and registers a tokenizer. The single LoadTokenizerStep handles:
/// - Input validation (empty name/source)
/// - Deduplication (returns success if already exists)
/// - Loading from local path or HuggingFace
/// - Encoding with fastokens, for the models `--fastokens-models` lists
/// - Optional caching layer wrapping, for the tokenizers that encode with HuggingFace
///
/// Configuration:
/// - 3 retries with 2s backoff (for network issues), none after a
///   `--fastokens-strict` refusal
/// - 5 minute timeout (HuggingFace downloads can be slow)
pub fn create_tokenizer_registration_workflow() -> WorkflowDefinition<TokenizerWorkflowData> {
    WorkflowDefinition::new("tokenizer_registration", "Tokenizer Registration").add_step(
        StepDefinition::new(
            "load_tokenizer",
            "Load Tokenizer",
            Arc::new(LoadTokenizerStep),
        )
        .with_retry(RetryPolicy {
            max_attempts: 3,
            backoff: BackoffStrategy::Fixed(Duration::from_secs(2)),
        })
        .with_timeout(Duration::from_secs(300)) // 5 min for HuggingFace downloads
        .with_failure_action(FailureAction::FailWorkflow),
    )
}

/// Helper to create initial workflow data for tokenizer registration
pub fn create_tokenizer_workflow_data(
    config: TokenizerConfigRequest,
    app_context: Arc<AppContext>,
) -> TokenizerWorkflowData {
    TokenizerWorkflowData {
        config,
        vocab_size: None,
        app_context: Some(app_context),
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;
    use wfaas::WorkflowInstanceId;

    use super::*;
    use crate::{
        config::{FastokensConfig, RouterConfig},
        routers::{
            comm::tokenize::remove_tokenizer,
            test_mocks::{
                app_context::{app_context, app_context_with_router_config},
                tokenizer::{
                    byte_level_tokenizer_dir, hf_tokenizer, hf_tokenizer_path, EnvVar,
                    BYTE_LEVEL_MERGES,
                },
            },
        },
        tokenizer::{registry::TokenizerRegistry, HuggingFaceTokenizer},
    };

    /// The fixture's tokenizer beside a `config.json` naming `architecture`.
    fn tokenizer_dir(architecture: &str) -> tempfile::TempDir {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::copy(hf_tokenizer_path(), dir.path().join("tokenizer.json")).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            json!({"architectures": [architecture]}).to_string(),
        )
        .unwrap();
        dir
    }

    /// llm-tokenizer picks its DeepSeek V3.2, V4 and V4.1 encoders from
    /// `config.json`, and they ignore the template.
    #[test]
    fn test_deepseek_renders_without_template() {
        let mut jinja = HuggingFaceTokenizer::from_file(hf_tokenizer_path()).unwrap();
        assert!(!renders_without_template(&mut jinja));
        for architecture in [
            "DeepseekV32ForCausalLM",
            "DeepseekV4ForCausalLM",
            "DeepseekV41ForCausalLM",
        ] {
            let dir = tokenizer_dir(architecture);
            let path = dir.path().join("tokenizer.json");
            let mut deepseek = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
            assert!(renders_without_template(&mut deepseek), "{architecture}");
        }
    }

    /// A tokenizer whose `config.json` picks a built-in encoder of
    /// llm-tokenizer's, with no `ChatControls` for its `model_type` (here none,
    /// as only `architectures` names the model), is refused, and not retried.
    #[tokio::test]
    async fn test_registration_refuses_chats_rendered_without_controls() {
        let context = app_context();
        let dir = tokenizer_dir("DeepseekV4ForCausalLM");
        let id = TokenizerRegistry::generate_id();
        let config = TokenizerConfigRequest {
            id: id.clone(),
            name: id.clone(),
            source: dir.path().to_str().unwrap().to_string(),
            chat_template_path: None,
            inline_chat_template: None,
            worker_url: None,
            cache_config: None,
            fail_on_duplicate: true,
        };
        let mut workflow = WorkflowContext::new(
            WorkflowInstanceId::new(),
            create_tokenizer_workflow_data(config, context.clone()),
        );
        let error = LoadTokenizerStep.execute(&mut workflow).await.unwrap_err();
        assert!(
            matches!(&error, WorkflowError::StepFailed { message, .. } if message.starts_with(UNCONTROLLED_REFUSAL)),
            "{error:?}"
        );
        assert!(!LoadTokenizerStep.is_retryable(&error));
        assert!(context.tokenizer_registry.get_by_id(&id).is_none());
    }

    /// A tokenizer registers with the reasoning controls ATOM keeps for the
    /// `model_type` in the `config.json` beside it, from its directory or its
    /// `tokenizer.json`, and removing it drops them. It registers also under a
    /// `--chat-template`, which llm-tokenizer's V4.1 encoder ignores.
    #[tokio::test]
    async fn test_registration_finds_the_checkpoint_chat_controls() {
        let checkpoint = |model_type: &str| {
            let dir = tempfile::TempDir::new().unwrap();
            std::fs::copy(hf_tokenizer_path(), dir.path().join("tokenizer.json")).unwrap();
            std::fs::write(
                dir.path().join("config.json"),
                json!({"model_type": model_type}).to_string(),
            )
            .unwrap();
            dir
        };
        let v41 = checkpoint("deepseek_v41");
        let v41_file = v41.path().join("tokenizer.json");
        let template = v41.path().join("template.jinja");
        std::fs::write(&template, "{{ messages }}").unwrap();
        let llama = checkpoint("llama");
        let bare = tempfile::TempDir::new().unwrap();
        std::fs::copy(hf_tokenizer_path(), bare.path().join("tokenizer.json")).unwrap();
        let context = app_context();
        let mut ids = Vec::new();
        for (source, chat_template, controls) in [
            (v41.path(), None, Some(ChatControls::DeepseekV41)),
            (v41_file.as_path(), None, Some(ChatControls::DeepseekV41)),
            (v41.path(), Some(&template), Some(ChatControls::DeepseekV41)),
            (llama.path(), None, None),
            (bare.path(), None, None),
        ] {
            let id = TokenizerRegistry::generate_id();
            let config = TokenizerConfigRequest {
                id: id.clone(),
                name: id.clone(),
                source: source.to_str().unwrap().to_string(),
                chat_template_path: chat_template.map(|path| path.to_str().unwrap().to_string()),
                inline_chat_template: None,
                worker_url: None,
                cache_config: None,
                fail_on_duplicate: false,
            };
            let mut workflow = WorkflowContext::new(
                WorkflowInstanceId::new(),
                create_tokenizer_workflow_data(config, context.clone()),
            );
            LoadTokenizerStep.execute(&mut workflow).await.unwrap();
            let found = context.chat_controls.read().get(&id).copied();
            assert_eq!(found, controls, "{}", source.display());
            ids.push(id);
        }
        remove_tokenizer(&context, &ids[0]).await;
        assert!(!context.chat_controls.read().contains_key(&ids[0]));
    }

    /// A loaded tokenizer is never replaced: one loaded with a template file keeps it
    /// when a registration with another template runs.
    #[tokio::test]
    async fn test_a_workers_inline_template_does_not_replace_a_template_file() {
        let context = app_context();
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("template.jinja");
        std::fs::write(&file, "file: {{ messages[0].content }}").unwrap();
        let register = |inline: Option<&str>, path: Option<&str>| {
            let config = TokenizerConfigRequest {
                id: TokenizerRegistry::generate_id(),
                name: "m".to_string(),
                source: hf_tokenizer_path().to_string(),
                chat_template_path: path.map(str::to_string),
                inline_chat_template: inline.map(str::to_string),
                worker_url: None,
                cache_config: None,
                fail_on_duplicate: false,
            };
            let data = create_tokenizer_workflow_data(config, context.clone());
            async move {
                let mut workflow = WorkflowContext::new(WorkflowInstanceId::new(), data);
                LoadTokenizerStep.execute(&mut workflow).await
            }
        };
        register(None, file.to_str()).await.unwrap();
        let file_loaded = context.tokenizer_registry.get_by_name("m").unwrap().id;
        register(Some("inline: {{ messages[0].content }}"), None)
            .await
            .unwrap();
        let entry = context.tokenizer_registry.get_by_name("m").unwrap();
        assert_eq!(entry.id, file_loaded);
        let hi = json!({"role": "user", "content": "hi"});
        let rendered = (entry.tokenizer)
            .apply_chat_template(&[hi], ChatTemplateParams::default())
            .unwrap();
        assert_eq!(rendered, "file: hi");
        assert!(context.tokenizer_inline_templates.read().is_empty());
    }

    /// A strict registration of a loaded model fails and leaves its tokenizer, even with
    /// another template.
    #[tokio::test]
    async fn test_a_strict_registration_with_another_template_replaces_nothing() {
        let context = app_context();
        let register = |template: &str, strict: bool| {
            let config = TokenizerConfigRequest {
                id: TokenizerRegistry::generate_id(),
                name: "m".to_string(),
                source: hf_tokenizer_path().to_string(),
                chat_template_path: None,
                inline_chat_template: Some(template.to_string()),
                worker_url: None,
                cache_config: None,
                fail_on_duplicate: strict,
            };
            let data = create_tokenizer_workflow_data(config, context.clone());
            async move {
                let mut workflow = WorkflowContext::new(WorkflowInstanceId::new(), data);
                LoadTokenizerStep.execute(&mut workflow).await
            }
        };
        register("first", false).await.unwrap();
        let loaded = context.tokenizer_registry.get_by_name("m").unwrap().id;
        assert!(register("second", true).await.is_err());
        assert_eq!(
            context.tokenizer_registry.get_by_name("m").unwrap().id,
            loaded
        );
        let recorded = context
            .tokenizer_inline_templates
            .read()
            .get(&loaded)
            .cloned();
        assert_eq!(recorded.as_deref(), Some("first"));
    }

    /// A job its worker's registration submitted, run after that worker left and a
    /// successor with another template registered (at its URL or another), loads
    /// nothing: the successor's own job loads the model's tokenizer with its template.
    #[tokio::test]
    async fn test_a_left_worker_s_job_leaves_the_tokenizer_to_its_successor() {
        use crate::core::{BasicWorkerBuilder, RuntimeType};

        for successor_url in ["grpc://atom:1", "grpc://atom:2"] {
            let context = app_context();
            let worker = |url: &str, template: &str| -> Arc<dyn Worker> {
                Arc::new(
                    BasicWorkerBuilder::new(url)
                        .runtime_type(RuntimeType::Atom)
                        .model_id("m")
                        .label("chat_template", template)
                        .build(),
                )
            };
            let job = |url: &str, template: &str| {
                let config = TokenizerConfigRequest {
                    id: TokenizerRegistry::generate_id(),
                    name: "m".to_string(),
                    source: hf_tokenizer_path().to_string(),
                    chat_template_path: None,
                    inline_chat_template: Some(template.to_string()),
                    worker_url: Some(url.to_string()),
                    cache_config: None,
                    fail_on_duplicate: false,
                };
                let data = create_tokenizer_workflow_data(config, context.clone());
                async move {
                    let mut workflow = WorkflowContext::new(WorkflowInstanceId::new(), data);
                    LoadTokenizerStep.execute(&mut workflow).await
                }
            };
            let (old, new) = (
                "old: {{ messages[0].content }}",
                "new: {{ messages[0].content }}",
            );
            context
                .worker_registry
                .register(worker("grpc://atom:1", old));
            let left_workers_job = job("grpc://atom:1", old);
            context.worker_registry.remove_by_url("grpc://atom:1");
            context.worker_registry.register(worker(successor_url, new));

            left_workers_job.await.unwrap();
            assert!(context.tokenizer_registry.get_by_name("m").is_none());
            assert!(context.tokenizer_inline_templates.read().is_empty());
            job(successor_url, new).await.unwrap();
            let tokenizer = context.tokenizer_registry.get("m").unwrap();
            let hi = json!({"role": "user", "content": "hi"});
            let rendered = tokenizer
                .apply_chat_template(&[hi], ChatTemplateParams::default())
                .unwrap();
            assert_eq!(rendered, "new: hi", "successor at {successor_url}");
        }
    }

    /// A template file is recorded before its tokenizer is visible: a chat that finds
    /// the model's tokenizer finds the template it was loaded with.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_a_template_file_is_recorded_before_its_tokenizer() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let context = app_context();
        let dir = tempfile::TempDir::new().unwrap();
        let paths = ["first", "second"].map(|name| {
            let file = dir.path().join(format!("{name}.jinja"));
            std::fs::write(&file, format!("{name}: {{{{ messages[0].content }}}}")).unwrap();
            file.to_str().unwrap().to_string()
        });
        let register = |path: &str| {
            let config = TokenizerConfigRequest {
                id: TokenizerRegistry::generate_id(),
                name: "m".to_string(),
                source: hf_tokenizer_path().to_string(),
                chat_template_path: Some(path.to_string()),
                inline_chat_template: None,
                worker_url: None,
                cache_config: None,
                fail_on_duplicate: false,
            };
            let data = create_tokenizer_workflow_data(config, context.clone());
            async move {
                let mut workflow = WorkflowContext::new(WorkflowInstanceId::new(), data);
                LoadTokenizerStep.execute(&mut workflow).await
            }
        };
        register(&paths[0]).await.unwrap();
        let done = Arc::new(AtomicBool::new(false));
        let reader = {
            let (context, done) = (context.clone(), done.clone());
            std::thread::spawn(move || {
                let (registry, files) = (
                    &context.tokenizer_registry,
                    &context.tokenizer_template_files,
                );
                let mut unrecorded = 0;
                while !done.load(Ordering::Relaxed) {
                    let Some(seen) = registry.get_by_name("m") else {
                        continue;
                    };
                    let recorded = files.read().contains_key(&seen.id);
                    // A removed one's record goes once it is no longer the model's.
                    let current = registry.get_by_name("m").is_some_and(|e| e.id == seen.id);
                    unrecorded += usize::from(!recorded && current);
                }
                unrecorded
            })
        };
        for load in 1..=20 {
            if let Some(entry) = context.tokenizer_registry.remove("m") {
                context.forget_tokenizer(&entry.id);
            }
            register(&paths[load % 2]).await.unwrap();
        }
        done.store(true, Ordering::Relaxed);
        assert_eq!(
            reader.join().unwrap(),
            0,
            "a tokenizer was seen before its file"
        );
    }

    /// A model `--fastokens-models` lists encodes with fastokens, which leaves the
    /// tokenizer cache out; one fastokens cannot load keeps HuggingFace, or fails
    /// to register under `--fastokens-strict`, with no retries.
    #[tokio::test]
    async fn test_registration_encodes_listed_models_with_fastokens() {
        async fn register(
            context: &Arc<AppContext>,
            source: &str,
        ) -> Result<Arc<dyn Tokenizer>, WorkflowError> {
            let id = TokenizerRegistry::generate_id();
            let config = TokenizerConfigRequest {
                id: id.clone(),
                name: id.clone(),
                source: source.to_string(),
                chat_template_path: None,
                inline_chat_template: None,
                worker_url: None,
                cache_config: Some(TokenizerCacheConfig {
                    enable_l0: true,
                    l0_max_entries: 16,
                    enable_l1: false,
                    l1_max_memory: 0,
                }),
                fail_on_duplicate: true,
            };
            let mut workflow = WorkflowContext::new(
                WorkflowInstanceId::new(),
                create_tokenizer_workflow_data(config, context.clone()),
            );
            LoadTokenizerStep.execute(&mut workflow).await?;
            Ok(context.tokenizer_registry.get_by_id(&id).unwrap().tokenizer)
        }

        let byte_level = byte_level_tokenizer_dir(BYTE_LEVEL_MERGES, true, json!({}));
        let byte_level = byte_level.path().to_str().unwrap();
        let mut config = RouterConfig::default();
        let context = app_context_with_router_config(config.clone());
        let unlisted = register(&context, byte_level).await.unwrap();
        assert!(unlisted.as_any().is::<CachedTokenizer>());

        config.fastokens.models = vec![byte_level.to_string()];
        let context = app_context_with_router_config(config.clone());
        let fast = register(&context, byte_level).await.unwrap();
        assert!(fast.as_any().is::<FastTokenizer>());
        assert_eq!(
            fast.encode("the weather", true).unwrap().token_ids(),
            unlisted.encode("the weather", true).unwrap().token_ids()
        );

        // The fixture's `Whitespace` pre-tokenizer is not one fastokens has.
        config.fastokens.models = vec!["*".to_string()];
        let context = app_context_with_router_config(config.clone());
        let kept = register(&context, hf_tokenizer_path()).await.unwrap();
        assert!(kept.as_any().is::<CachedTokenizer>());
        config.fastokens.strict = true;
        let context = app_context_with_router_config(config);
        let error = register(&context, hf_tokenizer_path()).await.err().unwrap();
        assert!(
            error.to_string().contains("fastokens cannot encode"),
            "{error}"
        );
        assert!(!LoadTokenizerStep.is_retryable(&error));
        let broken = tempfile::TempDir::new().unwrap();
        let broken = broken.path().join("tokenizer.json");
        std::fs::write(&broken, "{}").unwrap();
        let error = register(&context, broken.to_str().unwrap())
            .await
            .err()
            .unwrap();
        assert!(LoadTokenizerStep.is_retryable(&error), "{error}");
        assert!(context.tokenizer_registry.list().is_empty());
    }

    /// A `tokenizer.json` fastokens cannot download is no refusal: under
    /// `--fastokens-strict` the registration fails and is retried, and without
    /// it the model keeps HuggingFace.
    #[tokio::test]
    #[serial]
    async fn test_fastokens_download_failures_are_retried() {
        let _endpoint = EnvVar::set("HF_ENDPOINT", "http://127.0.0.1:1");
        let hf = hf_tokenizer();
        let mut config = FastokensConfig {
            models: vec!["*".to_string()],
            strict: true,
            ..Default::default()
        };
        let error = with_fastokens(hf.clone(), "model", "org/model", &config)
            .await
            .err()
            .unwrap();
        assert!(error.contains("cannot download"), "{error}");
        let error = WorkflowError::StepFailed {
            step_id: StepId::new("load_tokenizer"),
            message: error,
        };
        assert!(LoadTokenizerStep.is_retryable(&error), "{error}");

        config.strict = false;
        let kept = with_fastokens(hf.clone(), "model", "org/model", &config)
            .await
            .unwrap();
        assert!(Arc::ptr_eq(&kept, &hf));
    }

    /// A HuggingFace model ID whose tokenizer cannot be downloaded fails the
    /// lookup of its chat controls, and the registration is retried; a source
    /// with no `tokenizer.json` has none.
    #[tokio::test]
    #[serial]
    async fn test_chat_controls_download_failures_are_retried() {
        let _endpoint = EnvVar::set("HF_ENDPOINT", "http://127.0.0.1:1");
        let error = checkpoint_chat_controls("model", "org/model")
            .await
            .unwrap_err();
        assert!(error.starts_with(DOWNLOAD_FAILURE), "{error}");
        let error = WorkflowError::StepFailed {
            step_id: StepId::new("load_tokenizer"),
            message: error,
        };
        assert!(LoadTokenizerStep.is_retryable(&error), "{error}");

        let empty = tempfile::TempDir::new().unwrap();
        let empty = empty.path().to_str().unwrap();
        assert_eq!(checkpoint_chat_controls("model", empty).await, Ok(None));
    }

    #[test]
    fn test_tokenizer_config_request_serialization() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            inline_chat_template: None,
            worker_url: None,
            cache_config: None,
            fail_on_duplicate: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, "test-uuid-1234");
        assert_eq!(parsed.name, "test-model");
        assert_eq!(parsed.source, "meta-llama/Llama-2-7b-hf");
        assert!(parsed.chat_template_path.is_none());
        assert!(parsed.cache_config.is_none());
        assert!(!parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_fail_on_duplicate_defaults_to_false() {
        // Test that fail_on_duplicate defaults to false when not specified in JSON
        let json = r#"{
            "id": "test-uuid",
            "name": "test-model",
            "source": "/path/to/tokenizer"
        }"#;
        let parsed: TokenizerConfigRequest = serde_json::from_str(json).unwrap();
        assert!(!parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_fail_on_duplicate_true() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            inline_chat_template: None,
            worker_url: None,
            cache_config: None,
            fail_on_duplicate: true,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();
        assert!(parsed.fail_on_duplicate);
    }

    #[test]
    fn test_tokenizer_config_request_with_cache() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
            inline_chat_template: None,
            worker_url: None,
            cache_config: Some(TokenizerCacheConfig {
                enable_l0: true,
                l0_max_entries: 1000,
                enable_l1: false,
                l1_max_memory: 0,
            }),
            fail_on_duplicate: false,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: TokenizerConfigRequest = serde_json::from_str(&json).unwrap();

        assert!(parsed.cache_config.is_some());
        let cache = parsed.cache_config.unwrap();
        assert!(cache.enable_l0);
        assert_eq!(cache.l0_max_entries, 1000);
        assert!(!cache.enable_l1);
    }

    #[test]
    fn test_workflow_creation() {
        let mut workflow = create_tokenizer_registration_workflow();
        assert_eq!(workflow.id.to_string(), "tokenizer_registration");
        // Validate the workflow DAG
        workflow
            .validate()
            .expect("Workflow validation should pass");
    }
}
