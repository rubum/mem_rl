use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Represents a vector embedding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embedding(pub Vec<f32>);

/// The fundamental unit of knowledge in the Revansy architecture, referred to as an "Episodic Memory".
///
/// According to Section 4.1 of the MemRL research paper, the memory bank $M$ is formalized as a set of triplets:
/// $$M = \{ (z_i, e_i, Q_i) \}_{i=1}^N$$
///
/// Unlike traditional RAG which maps raw text to chunks, Revansy structuralizes memory into:
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
    /// Schema-less metadata for contextual filtering and tracking.
    pub metadata: Value,
}

/// A trait for collapsing complex environmental feedback into a scalar reward.
pub trait RewardSignal: Send + Sync {
    /// Collapses multi-dimensional metrics into a single f32 for utility updates.
    fn composite_score(&self) -> f32;
}

/// A simple implementation of RewardSignal using a single scalar.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Reward(pub f32);

impl RewardSignal for Reward {
    fn composite_score(&self) -> f32 {
        self.0
    }
}

/// Dynamic overrides for per-request hyperparameter tuning.
/// 
/// Allows industrial-scale A/B testing and context-aware reranking
/// without needing to reconstruct the core RevansyAgent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RetrievalOptions {
    /// Overrides the agent's default `lambda` (Balance between similarity and utility).
    pub lambda: Option<f32>,
    /// Overrides the agent's default `exploration_rate` (epsilon-greedy chance).
    pub epsilon: Option<f32>,
    /// Overrides the agent's default `k1` (Initial Recall Size).
    pub k1: Option<usize>,
    /// Overrides the agent's default `k2` (Final Context Window).
    pub k2: Option<usize>,
}

impl RetrievalOptions {
    /// Creates a new set of retrieval options with all values set to `None`.
    /// The agent will fall back to its internal defaults for any `None` fields.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Utility Balance ($\lambda$) for this single request.
    /// Values must be between 0.0 and 1.0.
    pub fn lambda(mut self, val: f32) -> Self {
        self.lambda = Some(val);
        self
    }

    /// Set the Exploration Rate ($\epsilon$) for this single request.
    /// Values must be between 0.0 and 1.0.
    pub fn epsilon(mut self, val: f32) -> Self {
        self.epsilon = Some(val);
        self
    }

    /// Set the Recall Pool Size ($k_1$) for this single request.
    pub fn k1(mut self, val: usize) -> Self {
        self.k1 = Some(val);
        self
    }

    /// Set the Context Window Size ($k_2$) for this single request.
    pub fn k2(mut self, val: usize) -> Self {
        self.k2 = Some(val);
        self
    }
}

impl MemoryItem {
    /// Bootstraps a new MemoryItem with a unique UUID and zero initial utility.
    ///
    /// # Arguments
    /// * `intent` - The textual context or goal ($z_i$).
    /// * `embedding` - The dense vector of the intent.
    /// * `experience` - The trajectory or solution ($e_i$).
    /// * `metadata` - Persistent context or filtering tags.
    pub fn new(intent: String, embedding: Embedding, experience: String, metadata: Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            intent,
            intent_embedding: embedding,
            experience,
            utility: 0.0, // Initial Q-value as per Section 5.4.1
            metadata,
        }
    }
}
