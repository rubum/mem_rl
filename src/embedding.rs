use crate::domain::Embedding;
use crate::error::MemRLError;
use async_trait::async_trait;
#[cfg(feature = "ollama")]
use serde::{Deserialize, Serialize};

/// Interface for embedding services.
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Embedding, MemRLError>;
}

#[cfg(feature = "ollama")]
/// Ollama implementation of EmbeddingService.
/// Uses the /api/embeddings endpoint.
pub struct OllamaEmbedder {
    pub model: String,
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
    async fn embed(&self, text: &str) -> Result<Embedding, MemRLError> {
        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/api/embeddings", self.base_url))
            .json(&OllamaEmbedRequest {
                model: self.model.clone(),
                prompt: text.to_string(),
            })
            .send()
            .await
            .map_err(MemRLError::embed)?
            .error_for_status()
            .map_err(|e| MemRLError::EmbeddingError(format!("Ollama API Error: {}", e)))?;

        let body: OllamaEmbedResponse = resp.json().await.map_err(MemRLError::embed)?;
        Ok(Embedding(body.embedding))
    }
}
