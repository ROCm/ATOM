// Many threads allocate and free per token; glibc's arenas contend on that.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub mod app_context;
pub mod atom_engine;
pub mod cliargs;
pub mod config;
pub mod core;
#[cfg(feature = "ext-proc")]
pub mod ext_proc;
pub mod middleware;
pub mod observability;
pub mod policies;
pub mod python;
pub use openai_protocol as protocols;
pub use reasoning_parser;
pub mod routers;
pub mod server;
pub use llm_tokenizer as tokenizer;
pub mod tool_parser;
pub mod version;
