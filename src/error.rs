use miette::Diagnostic;
use std::fmt::Display;
use thiserror::Error;

#[derive(Error, Debug, Diagnostic)]
pub enum MemRLError {
    #[error("Vector store operation failed: {0}")]
    #[diagnostic(
        code(mem_rl::vector_store_error),
        help("Ensure your vector database is running and accessible.")
    )]
    VectorStoreError(String),

    #[error("Embedding service operation failed: {0}")]
    #[diagnostic(
        code(mem_rl::embedding_error),
        help("Check your embedding model configuration or network connection.")
    )]
    EmbeddingError(String),

    #[error("Invalid Agent Configuration: {0}")]
    #[diagnostic(
        code(mem_rl::agent_config_error),
        help("Review the hyperparameters passed to MemRLAgentBuilder.")
    )]
    AgentConfigurationError(String),
}

impl MemRLError {
    pub fn vector_store<T: Display>(err: T) -> Self {
        Self::VectorStoreError(err.to_string())
    }

    pub fn embed<T: Display>(err: T) -> Self {
        Self::EmbeddingError(err.to_string())
    }
}
