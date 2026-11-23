//! Tokenizer registration workflow
//!
//! This module provides a workflow for registering tokenizers asynchronously.
//! Tokenizers can be loaded from local paths or downloaded from HuggingFace.
//!
//! This is the **single source of truth** for tokenizer registration. All paths
//! (startup, worker connection, API) should use this workflow to ensure consistent
//! behavior (validation, caching, deduplication).

use std::{collections::HashSet, sync::Arc, time::Duration};

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{error, info, warn};
use wfaas::{
    BackoffStrategy, FailureAction, RetryPolicy, StepDefinition, StepExecutor, StepId, StepResult,
    WorkflowContext, WorkflowDefinition, WorkflowError, WorkflowResult,
};

use super::workflow_data::TokenizerWorkflowData;
use crate::{
    app_context::AppContext,
    config::TokenizerCacheConfig,
    routers::prepare::fast_tokenizer::{with_fastokens, FastTokenizer, STRICT_REFUSAL},
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
        let cache_config = config.cache_config.clone();
        let chats_without_template = app_context.chats_without_template.clone();
        let fastokens = app_context.router_config.fastokens.clone();

        // Load the tokenizer using the registry's load method
        // This handles: validation, deduplication, and loading
        let result = app_context
            .tokenizer_registry
            .load(&id, &name, &source, || {
                let id = id.clone();
                let name = name.clone();
                let source = source.clone();
                let chat_template = chat_template.clone();
                let cache_cfg = cache_config.clone();
                let chats_without_template = chats_without_template.clone();
                let fastokens = fastokens.clone();
                async move {
                    // Load base tokenizer
                    let base_tokenizer = factory::create_tokenizer_async_with_chat_template(
                        &source,
                        chat_template.as_deref(),
                    )
                    .await
                    .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
                    let base_tokenizer =
                        with_fastokens(base_tokenizer, &name, &source, &fastokens).await?;
                    probe_chat_render(
                        &chats_without_template,
                        &id,
                        &name,
                        &source,
                        chat_template.as_deref(),
                    )
                    .await;
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
                            };
                            Arc::new(CachedTokenizer::new(base_tokenizer, cache_config))
                        }
                        _ => base_tokenizer,
                    };

                    Ok(tokenizer)
                }
            })
            .await;

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
        // refusal is what the probe found, and a retry would only probe again.
        !matches!(
            error,
            WorkflowError::StepFailed { message, .. } if message.starts_with(STRICT_REFUSAL)
        )
    }
}

/// Record `id` in `chats_without_template`, and warn, when llm-tokenizer renders
/// `source`'s chats without the Jinja template. It has no accessor for that, so a
/// copy is probed; mirroring its `config.json` check would also mean mirroring
/// how it resolves sources.
async fn probe_chat_render(
    chats_without_template: &RwLock<HashSet<String>>,
    id: &str,
    name: &str,
    source: &str,
    chat_template: Option<&str>,
) {
    const UNPROBED: &str = "Cannot probe whether llm-tokenizer renders these chats without the \
                            Jinja template; HTTP routing tokenizes them as it renders them";
    let mut copy = match factory::create_tokenizer_async_with_chat_template(source, None).await {
        Ok(copy) => copy,
        Err(error) => {
            warn!(tokenizer = name, %error, "{UNPROBED}");
            return;
        }
    };
    let Some(copy) = Arc::get_mut(&mut copy) else {
        warn!(tokenizer = name, "{UNPROBED}: the loaded copy is shared");
        return;
    };
    if !renders_without_template(copy) {
        return;
    }
    chats_without_template.write().insert(id.to_string());
    warn!(
        tokenizer = name,
        "llm-tokenizer renders these chats with a built-in encoder, not the Jinja template{}; \
         its defaults are not ATOM's, so HTTP routing does not tokenize them",
        if chat_template.is_some() {
            ", so --chat-template does not apply"
        } else {
            ""
        }
    );
}

/// Whether `tokenizer` renders chats without its Jinja template, as llm-tokenizer
/// does with the encoders it has built in for the architectures in `config.json`
/// (DeepSeek V3.2 and V4 in 1.4.2). Such a render ignores `--chat-template`, and
/// its defaults are not ATOM's: DeepSeek renders chat mode unless `thinking` is
/// true, where ATOM's encoder renders thinking mode. Replaces the template, so
/// probe a copy.
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

    /// llm-tokenizer picks its DeepSeek V3.2 and V4 encoders from `config.json`,
    /// and they ignore the template.
    #[test]
    fn test_deepseek_renders_without_template() {
        let mut jinja = HuggingFaceTokenizer::from_file(hf_tokenizer_path()).unwrap();
        assert!(!renders_without_template(&mut jinja));
        for architecture in ["DeepseekV32ForCausalLM", "DeepseekV4ForCausalLM"] {
            let dir = tokenizer_dir(architecture);
            let path = dir.path().join("tokenizer.json");
            let mut deepseek = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
            assert!(renders_without_template(&mut deepseek), "{architecture}");
        }
    }

    /// Registering a tokenizer whose `config.json` picks a built-in encoder marks
    /// it, and removing it drops the mark.
    #[tokio::test]
    async fn test_registration_marks_chats_rendered_without_template() {
        let context = app_context();
        let dir = tokenizer_dir("DeepseekV4ForCausalLM");
        let deepseek = TokenizerRegistry::generate_id();
        let jinja = TokenizerRegistry::generate_id();
        for (id, source) in [
            (&deepseek, dir.path().to_str().unwrap()),
            (&jinja, hf_tokenizer_path()),
        ] {
            let config = TokenizerConfigRequest {
                id: id.clone(),
                name: id.clone(),
                source: source.to_string(),
                chat_template_path: None,
                cache_config: None,
                fail_on_duplicate: true,
            };
            let mut workflow = WorkflowContext::new(
                WorkflowInstanceId::new(),
                create_tokenizer_workflow_data(config, context.clone()),
            );
            LoadTokenizerStep.execute(&mut workflow).await.unwrap();
        }

        let marked = |id: &str| context.chats_without_template.read().contains(id);
        assert!(marked(&deepseek));
        assert!(!marked(&jinja));
        remove_tokenizer(&context, &deepseek).await;
        assert!(!marked(&deepseek));
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

    #[test]
    fn test_tokenizer_config_request_serialization() {
        let config = TokenizerConfigRequest {
            id: "test-uuid-1234".to_string(),
            name: "test-model".to_string(),
            source: "meta-llama/Llama-2-7b-hf".to_string(),
            chat_template_path: None,
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
