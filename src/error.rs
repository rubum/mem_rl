use miette::Diagnostic;
use std::fmt::Display;
use thiserror::Error;

/// The central error type for the Relevansy framework.
///
/// Uses `miette` for rich, diagnostic error reporting in agentic workflows.
#[derive(Error, Debug, Diagnostic)]
pub enum RelevansyError {
    /// Errors originating from the underlying vector database (e.g., Qdrant).
    #[error("Vector store operation failed: {0}")]
    #[diagnostic(
        code(relevansy::vector_store_error),
        help("Ensure your vector database is running and accessible.")
    )]
    VectorStoreError(String),

    /// Errors originating from the embedding generation service (e.g., Ollama).
    #[error("Embedding service operation failed: {0}")]
    #[diagnostic(
        code(relevansy::embedding_error),
        help("Check your embedding model configuration or network connection.")
    )]
    EmbeddingError(String),

    /// Errors caused by invalid agent hyperparameters or illegal state transitions.
    #[error("Invalid Agent Configuration: {0}")]
    #[diagnostic(
        code(relevansy::agent_config_error),
        help("Review the hyperparameters passed to RelevansyAgentBuilder.")
    )]
    AgentConfigurationError(String),
}

impl RelevansyError {
    /// Helper to wrap any displayable error into a `VectorStoreError`.
    pub fn vector_store<T: Display>(err: T) -> Self {
        Self::VectorStoreError(err.to_string())
    }

    /// Helper to wrap any displayable error into an `EmbeddingError`.
    pub fn embed<T: Display>(err: T) -> Self {
        Self::EmbeddingError(err.to_string())
    }
}
