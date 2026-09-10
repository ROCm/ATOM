//! Parser registration, model selection, and instance pooling.

// Factory and pool for creating model-specific tool parsers with pooling support.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use tokio::sync::Mutex;

use super::{
    core::ToolParser,
    parsers::{
        DsmlParser, Glm4MoeParser, JsonParser, KimiK2Parser, KimiK3Parser, MiniMaxParser,
        PassthroughParser, QwenCoderParser, QwenParser,
    },
};

/// Type alias for pooled parser instances.
pub type PooledParser = Arc<Mutex<Box<dyn ToolParser>>>;

/// Type alias for parser creator functions.
type ParserCreator = Arc<dyn Fn() -> Box<dyn ToolParser> + Send + Sync>;

/// Registry for model-specific tool parsers with pooling support.
#[derive(Clone)]
pub struct ParserRegistry {
    /// Creator functions for parsers (used when pool is empty)
    creators: Arc<RwLock<HashMap<String, ParserCreator>>>,
    /// Pooled parser instances for reuse
    pool: Arc<RwLock<HashMap<String, PooledParser>>>,
    /// Model pattern to parser name mappings
    model_mapping: Arc<RwLock<HashMap<String, String>>>,
    /// Default parser name
    default_parser: Arc<RwLock<String>>,
}

impl ParserRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            creators: Arc::new(RwLock::new(HashMap::new())),
            pool: Arc::new(RwLock::new(HashMap::new())),
            model_mapping: Arc::new(RwLock::new(HashMap::new())),
            default_parser: Arc::new(RwLock::new("passthrough".to_string())),
        }
    }

    /// Register a parser creator for a given parser type.
    pub fn register_parser<F>(&self, name: &str, creator: F)
    where
        F: Fn() -> Box<dyn ToolParser> + Send + Sync + 'static,
    {
        let mut creators = self.creators.write().unwrap();
        creators.insert(name.to_string(), Arc::new(creator));
    }

    /// Map a model name/pattern to a parser
    pub fn map_model(&self, model: impl Into<String>, parser: impl Into<String>) {
        let mut mapping = self.model_mapping.write().unwrap();
        mapping.insert(model.into(), parser.into());
    }

    /// Get a pooled parser by exact name.
    /// Returns a shared parser instance from the pool, creating one if needed.
    pub fn get_pooled_parser(&self, name: &str) -> Option<PooledParser> {
        // First check if we have a pooled instance
        {
            let pool = self.pool.read().unwrap();
            if let Some(parser) = pool.get(name) {
                return Some(Arc::clone(parser));
            }
        }

        // If not in pool, create one and add to pool
        let creators = self.creators.read().unwrap();
        if let Some(creator) = creators.get(name) {
            let parser = Arc::new(Mutex::new(creator()));

            // Add to pool for future use
            let mut pool = self.pool.write().unwrap();
            pool.insert(name.to_string(), Arc::clone(&parser));

            Some(parser)
        } else {
            None
        }
    }

    /// Check if a parser with the given name is registered.
    pub fn has_parser(&self, name: &str) -> bool {
        let creators = self.creators.read().unwrap();
        creators.contains_key(name)
    }

    /// Create a fresh (non-pooled) parser instance by exact name.
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_parser(&self, name: &str) -> Option<Box<dyn ToolParser>> {
        let creators = self.creators.read().unwrap();
        creators.get(name).map(|creator| creator())
    }

    /// Resolve a model identifier to a registered parser name.
    ///
    /// Namespaced and differently-cased model identifiers are supported by
    /// case-insensitive substring matching. The longest matching stem wins.
    pub fn resolve_model_to_parser(&self, model: &str) -> Option<String> {
        let mapping = self.model_mapping.read().unwrap();
        if let Some(parser_name) = mapping.get(model) {
            return Some(parser_name.clone());
        }

        let model_lower = model.to_lowercase();
        mapping
            .iter()
            .filter_map(|(pattern, parser_name)| {
                let stem = pattern.strip_suffix('*')?;
                model_lower
                    .contains(&stem.to_lowercase())
                    .then_some((stem.len(), parser_name))
            })
            .max_by_key(|(stem_len, _)| *stem_len)
            .map(|(_, parser_name)| parser_name.clone())
    }

    /// Check if a parser can be created for a specific model without actually creating it.
    /// Returns true if a parser is available (registered) for this model.
    pub fn has_parser_for_model(&self, model: &str) -> bool {
        self.resolve_model_to_parser(model)
            .is_some_and(|parser_name| self.has_parser(&parser_name))
    }

    /// Create a fresh (non-pooled) parser instance for a specific model.
    /// Returns a new parser instance for each call - useful for streaming where state isolation is needed.
    pub fn create_for_model(&self, model: &str) -> Option<Box<dyn ToolParser>> {
        let parser_name = self
            .resolve_model_to_parser(model)
            .unwrap_or_else(|| self.default_parser.read().unwrap().clone());
        self.create_parser(&parser_name)
    }

    /// Get parser for a specific model
    pub fn get_pooled_for_model(&self, model: &str) -> Option<PooledParser> {
        let parser_name = self
            .resolve_model_to_parser(model)
            .unwrap_or_else(|| self.default_parser.read().unwrap().clone());
        self.get_pooled_parser(&parser_name)
    }

    /// Clear the parser pool, forcing new instances to be created.
    pub fn clear_pool(&self) {
        let mut pool = self.pool.write().unwrap();
        pool.clear();
    }

    /// Set the default parser
    pub fn set_default_parser(&self, name: impl Into<String>) {
        let mut default = self.default_parser.write().unwrap();
        *default = name.into();
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating tool parsers based on model type.
#[derive(Clone)]
pub struct ParserFactory {
    registry: ParserRegistry,
}

impl ParserFactory {
    /// Create a new factory with default parsers registered.
    pub fn new() -> Self {
        let registry = ParserRegistry::new();

        // Register default parsers
        registry.register_parser("passthrough", || Box::new(PassthroughParser::new()));
        registry.register_parser("json", || Box::new(JsonParser::new()));
        registry.register_parser("qwen", || Box::new(QwenParser::new()));
        registry.register_parser("qwen_json", || Box::new(QwenParser::new()));
        registry.register_parser("qwen_xml", || Box::new(QwenCoderParser::new()));
        registry.register_parser("qwen_coder", || Box::new(QwenCoderParser::new()));
        registry.register_parser("dsml", || Box::new(DsmlParser::new()));
        registry.register_parser("glm", || Box::new(Glm4MoeParser::glm45()));
        registry.register_parser("glm45_moe", || Box::new(Glm4MoeParser::glm45()));
        registry.register_parser("glm47_moe", || Box::new(Glm4MoeParser::glm47()));
        registry.register_parser("kimi", || Box::new(KimiK2Parser::new()));
        registry.register_parser("kimi_k3", || Box::new(KimiK3Parser::new()));
        registry.register_parser("kimik2", || Box::new(KimiK2Parser::new()));
        registry.register_parser("minimax", || Box::new(MiniMaxParser::new()));

        // Register default model mappings
        Self::register_default_mappings(&registry);

        Self { registry }
    }

    fn register_default_mappings(registry: &ParserRegistry) {
        // OpenAI models
        registry.map_model("gpt-4*", "json");
        registry.map_model("gpt-3.5*", "json");
        registry.map_model("gpt-4o*", "json");

        // Anthropic models
        registry.map_model("claude-*", "json");

        // Qwen models (more specific patterns first - longer patterns take precedence)
        // Qwen3.5+ and Qwen3-Coder use the XML parameter format.
        registry.map_model("Qwen/Qwen3.5*", "qwen_xml");
        registry.map_model("Qwen3.5*", "qwen_xml");
        registry.map_model("qwen3.5*", "qwen_xml");
        registry.map_model("Qwen/Qwen3.6*", "qwen_xml");
        registry.map_model("Qwen3.6*", "qwen_xml");
        registry.map_model("qwen3.6*", "qwen_xml");
        registry.map_model("Qwen/Qwen3.8*", "qwen_xml");
        registry.map_model("Qwen3.8*", "qwen_xml");
        registry.map_model("qwen3.8*", "qwen_xml");
        registry.map_model("Qwen/Qwen3-Coder*", "qwen_coder");
        registry.map_model("Qwen3-Coder*", "qwen_coder");
        registry.map_model("qwen3-coder*", "qwen_coder");
        // Qwen3 and earlier, including Qwen2.5-Coder, use JSON-in-tag.
        registry.map_model("qwen*", "qwen");
        registry.map_model("Qwen*", "qwen");

        // DeepSeek models
        registry.map_model("deepseek-v4*", "dsml");
        registry.map_model("deepseek-ai/DeepSeek-V4*", "dsml");

        // GLM models
        registry.map_model("glm-4.5*", "glm45_moe");
        registry.map_model("glm-4.6*", "glm45_moe");
        registry.map_model("glm-4.7*", "glm47_moe");
        registry.map_model("glm-5*", "glm47_moe");
        registry.map_model("glm-*", "json");

        // Kimi models
        registry.map_model("kimi-k2*", "kimik2");
        registry.map_model("Kimi-K2*", "kimik2");
        registry.map_model("moonshot*/Kimi-K2*", "kimik2");
        registry.map_model("kimi-k3*", "kimi_k3");
        registry.map_model("Kimi-K3*", "kimi_k3");
        registry.map_model("moonshot*/Kimi-K3*", "kimi_k3");
        registry.map_model("kimi_k3*", "kimi_k3");
        registry.map_model("Kimi_K3*", "kimi_k3");

        // MiniMax models
        registry.map_model("minimax-m3*", "minimax");
        registry.map_model("MiniMax-M3*", "minimax");

        // Other models
        registry.map_model("gemini-*", "json");
        registry.map_model("palm-*", "json");
        registry.map_model("gemma-*", "json");
    }

    /// Get a pooled parser for the given model ID.
    /// Returns a shared instance that can be used concurrently.
    /// Falls back to passthrough parser if model is not recognized.
    pub fn get_pooled(&self, model_id: &str) -> PooledParser {
        self.registry
            .get_pooled_for_model(model_id)
            .unwrap_or_else(|| {
                // Fallback to passthrough parser (no-op, returns text unchanged)
                self.registry
                    .get_pooled_parser("passthrough")
                    .expect("Passthrough parser should always be registered")
            })
    }

    /// Get the internal registry for custom registration.
    pub fn registry(&self) -> &ParserRegistry {
        &self.registry
    }

    /// Clear the parser pool.
    pub fn clear_pool(&self) {
        self.registry.clear_pool();
    }

    /// Get a non-pooled parser for the given model ID (creates a fresh instance each time).
    /// This is useful for benchmarks and testing where you want independent parser instances.
    pub fn get_parser(&self, model_id: &str) -> Option<Arc<dyn ToolParser>> {
        let parser_type = self
            .registry
            .resolve_model_to_parser(model_id)
            .unwrap_or_else(|| self.registry.default_parser.read().unwrap().clone());

        let creators = self.registry.creators.read().unwrap();
        creators.get(&parser_type).map(|creator| {
            // Call the creator to get a Box<dyn ToolParser>, then convert to Arc
            let boxed_parser = creator();
            Arc::from(boxed_parser)
        })
    }

    /// List all registered parsers (for compatibility with old API).
    pub fn list_parsers(&self) -> Vec<String> {
        self.registry
            .creators
            .read()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }
}

impl Default for ParserFactory {
    fn default() -> Self {
        Self::new()
    }
}

use tracing::warn;

pub(crate) fn check_tool_parser_availability(
    factory: &ParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> bool {
    if let Some(parser_name) = configured_parser {
        factory.registry().has_parser(parser_name)
    } else {
        factory.registry().has_parser_for_model(model)
    }
}

/// Returns a pooled parser for a non-streaming response.
pub(crate) fn get_tool_parser(
    factory: &ParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> PooledParser {
    if let Some(parser_name) = configured_parser {
        factory
            .registry()
            .get_pooled_parser(parser_name)
            .unwrap_or_else(|| {
                warn!(
                    "Configured tool parser '{}' not found, falling back to model-based selection",
                    parser_name
                );
                factory.get_pooled(model)
            })
    } else {
        factory.get_pooled(model)
    }
}

/// Returns a fresh parser for a streaming response, preventing state leakage
/// between requests.
pub(crate) fn create_tool_parser(
    factory: &ParserFactory,
    configured_parser: Option<&str>,
    model: &str,
) -> Option<Box<dyn ToolParser>> {
    if let Some(parser_name) = configured_parser {
        factory.registry().create_parser(parser_name).or_else(|| {
            warn!(
                "Configured tool parser '{}' not found, falling back to model-based selection",
                parser_name
            );
            factory.registry().create_for_model(model)
        })
    } else {
        factory.registry().create_for_model(model)
    }
}

#[cfg(test)]
mod tests {
    use super::ParserFactory;
    use super::{check_tool_parser_availability, create_tool_parser, get_tool_parser};

    #[test]
    fn configured_and_model_selection_work() {
        let factory = ParserFactory::new();

        assert!(check_tool_parser_availability(
            &factory,
            Some("qwen"),
            "unrelated"
        ));
        assert!(check_tool_parser_availability(&factory, None, "qwen3"));
        assert!(get_tool_parser(&factory, Some("qwen3"), "unrelated")
            .try_lock()
            .is_ok());
        assert!(create_tool_parser(&factory, Some("qwen3"), "unrelated").is_some());
    }

    #[test]
    fn unknown_configured_parser_falls_back_to_model() {
        let factory = ParserFactory::new();

        assert!(create_tool_parser(&factory, Some("missing"), "qwen3").is_some());
    }
}
