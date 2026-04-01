use crate::domain::Embedding;
use crate::error::RevansyError;
use async_trait::async_trait;
#[cfg(feature = "ollama")]
use serde::{Deserialize, Serialize};

/// Interface for embedding services.
/// 
/// Provides dense vector representations of textual intents ($z_i$)
/// for use in Phase A semantic search.
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// Generates a dense vector for the given text.
    async fn embed(&self, text: &str) -> Result<Embedding, RevansyError>;
}

#[cfg(feature = "ollama")]
/// Ollama implementation of `EmbeddingService`.
///
/// Communicates with a local or remote Ollama instance (default: localhost:11434)
/// to generate embeddings using specified models (e.g., "nomic-embed-text").
pub struct OllamaEmbedder {
    /// The name of the model to use (e.g., "nomic-embed-text").
    pub model: String,
    /// The base URL of the Ollama API.
    pub base_url: String,
}

#[cfg(feature = "ollama")]
#[derive(Serialize)]
struct OllamaEmbedRequest {
    model: String,
    prompt: String,
}

#[cfg(feature = "ollama")]
#[derive(Deserialize)]
struct OllamaEmbedResponse {
    embedding: Vec<f32>,
}

#[cfg(feature = "ollama")]
impl OllamaEmbedder {
    /// Creates a new `OllamaEmbedder` for the specified model.
    ///
    /// By default, it connects to "http://localhost:11434".
    pub fn new(model: String) -> Self {
        Self {
            model,
            base_url: "http://localhost:11434".to_string(),
        }
    }
}

#[cfg(feature = "ollama")]
#[async_trait]
impl EmbeddingService for OllamaEmbedder {
    async fn embed(&self, text: &str) -> Result<Embedding, RevansyError> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&OllamaEmbedRequest {
                model: self.model.clone(),
                prompt: text.to_string(),
            })
            .send()
            .await
            .map_err(RevansyError::embed)?
            .error_for_status()
            .map_err(|e| RevansyError::EmbeddingError(format!("Ollama API Error: {}", e)))?;

        let body: OllamaEmbedResponse = resp.json().await.map_err(RevansyError::embed)?;
        Ok(Embedding(body.embedding))
    }
}
