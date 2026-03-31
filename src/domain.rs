use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a vector embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

/// The fundamental unit of knowledge in the MemRL architecture, referred to as an "Episodic Memory".
///
/// According to Section 4.1 of the MemRL paper, the memory bank $M$ is formalized as a set of triplets:
/// $$M = \{ (z_i, e_i, Q_i) \}_{i=1}^N$$
///
/// Unlike traditional RAG which maps raw text to chunks, MemRL structuralizes memory into:
/// - **Intent**: The context or conditions under which a solution was sought.
/// - **Experience**: The successful trajectory or response that fulfilled the intent.
/// - **Utility**: A temporal-difference learned value indicating historical reliability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: Uuid,
    /// $z_i$: The user's query or the formalized context state that triggered the reasoning.
    pub intent: String,
    /// $\text{Emb}(z_i)$: The dense vector representation of the intent used in Phase A Semantic Retrieval.
    pub intent_embedding: Embedding,
    /// $e_i$: The recorded artifact, solution trace, or successful trajectory (e.g., bash commands, code).
    pub experience: String,
    /// $Q_i$: A learned scalar measuring the empirical helpfulness of this experience,
    /// updated via Monte-Carlo feedback loops.
    pub utility: f32,
}

/// A simplified representation of environmental feedback.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Reward(pub f32);

impl MemoryItem {
    pub fn new(intent: String, embedding: Embedding, experience: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            intent,
            intent_embedding: embedding,
            experience,
            utility: 0.0, // Initial Q-value as per Section 5.4.1
        }
    }
}
