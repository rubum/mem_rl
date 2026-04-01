pub mod agent;
pub mod domain;
pub mod embedding;
pub mod error;
pub mod math;
pub mod storage;

pub use agent::{RelevansyAgent, RelevansyAgentBuilder};
pub use domain::{MemoryItem, Reward, RewardSignal, RetrievalOptions};
pub use embedding::EmbeddingService;
pub use error::RelevansyError;
pub use storage::VectorStore;

#[cfg(feature = "ollama")]
pub use embedding::OllamaEmbedder;

#[cfg(feature = "qdrant")]
pub use storage::QdrantStore;
