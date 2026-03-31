use crate::domain::{Embedding, MemoryItem, Reward};
use crate::error::MemRLError;
use crate::math::{compute_memrl_score, z_score_normalize};
use crate::storage::VectorStore;

/// The orchestrating reinforcement learning agent from the MemRL specification.
/// 
/// The agent maintains an active vector store and coordinates the retrieval-learning loop.
pub struct MemRLAgent<S: VectorStore> {
    /// The external dense vector database storing the Intent-Experience-Utility triplets.
    pub store: S,
    /// $\alpha \in [0, 1]$: The Temporal-Difference (TD) learning rate. 
    /// Controls how drastically new rewards override historic $Q$-values (Eq. 4).
    pub alpha: f32,
    /// $\lambda \in [0, 1]$: The Value-Aware Balance Factor. 
    /// Controls the trade-off between Semantic Similarity vs. Historic Utility (Eq. 7).
    pub lambda: f32,
    /// $k_1$: The Recall Pool Size. The number of candidate triplets retrieved via Phase A fast semantic search.
    pub k1: usize,
    /// $k_2$: The Final Context Window Size. The number of memories passed to the LLM context after Phase B re-ranking.
    pub k2: usize,
}

pub struct MemRLAgentBuilder<S: VectorStore> {
    store: S,
    alpha: f32,
    lambda: f32,
    k1: usize,
    k2: usize,
}

impl<S: VectorStore> MemRLAgentBuilder<S> {
    pub fn new(store: S) -> Self {
        Self {
            store,
            alpha: 0.3,
            lambda: 0.5,
            k1: 5,
            k2: 3,
        }
    }

    pub fn learning_rate(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn utility_balance(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    pub fn recall_pool(mut self, k1: usize) -> Self {
        self.k1 = k1;
        self
    }

    pub fn context_window(mut self, k2: usize) -> Self {
        self.k2 = k2;
        self
    }

    pub fn build(self) -> core::result::Result<MemRLAgent<S>, MemRLError> {
        if self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(MemRLError::AgentConfigurationError("alpha must be between 0 and 1".to_string()));
        }
        if self.lambda < 0.0 || self.lambda > 1.0 {
            return Err(MemRLError::AgentConfigurationError("lambda must be between 0 and 1".to_string()));
        }
        if self.k1 < self.k2 {
            return Err(MemRLError::AgentConfigurationError("recall pool (k1) must be >= context window (k2)".to_string()));
        }

        Ok(MemRLAgent {
            store: self.store,
            alpha: self.alpha,
            lambda: self.lambda,
            k1: self.k1,
            k2: self.k2,
        })
    }
}

impl<S: VectorStore> MemRLAgent<S> {
    /// Executes the full Memory Retrieval Pipeline combining RAG and RL components.
    ///
    /// The MemRL architecture mandates a strict two-stage process (Section 4.2):
    /// 1. **Phase A (Semantic Matching)**: Fetches the top-$k_1$ structurally similar intents.
    /// 2. **Phase B (Value-Aware Sort)**: Normalizes both semantic scores and experiential utilities,
    ///    then calculates a final selection score modulated by $\lambda$.
    pub async fn retrieve(&self, query_embedding: &Embedding) -> core::result::Result<Vec<MemoryItem>, MemRLError> {
        // Phase A: Similarity-Based Recall
        // Retrieves a candidate pool C(s) from the VectorStore.
        let candidates = self.store.search(query_embedding, self.k1).await?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Extracts scores and utilities for normalization.
        let sim_scores: Vec<f32> = candidates.iter().map(|(_, score)| *score).collect();
        let utilities: Vec<f32> = candidates.iter().map(|(item, _)| item.utility).collect();

        // Performs Z-score normalization as per Section 4.2.
        let sim_hat = z_score_normalize(&sim_scores);
        let q_hat = z_score_normalize(&utilities);

        // Phase B: Value-Aware Selection (Equation 7)
        // Combines normalized similarity and utility into a single ranking score.
        let mut scored_items: Vec<(MemoryItem, f32)> = candidates
            .into_iter()
            .enumerate()
            .map(|(i, (item, _))| {
                let score = compute_memrl_score(sim_hat[i], q_hat[i], self.lambda);
                (item, score)
            })
            .collect();

        // Sort by final score descending.
        scored_items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Returns top-k2 items.
        Ok(scored_items
            .into_iter()
            .take(self.k2)
            .map(|(item, _)| item)
            .collect())
    }

    /// Executes the Offline Utility Update mechanism (Section 4.3).
    ///
    /// Traditional memory structures discard unused components. MemRL treats every experience application 
    /// as a Markov Decision step and updates the empirical helpfulness ($Q_i$) based on the observed reward ($r$).
    ///
    /// It applies the classic Temporal-Difference style update rule:
    /// $$Q_{\text{new}} \leftarrow Q_{\text{old}} + \alpha (r - Q_{\text{old}})$$
    pub async fn learn(&self, item: &MemoryItem, reward: Reward) -> core::result::Result<(), MemRLError> {
        let q_old = item.utility;
        let r = reward.0;
        let q_new = q_old + self.alpha * (r - q_old);

        // Persist the updated utility back to the storage.
        self.store.update_utility(item.id, q_new).await?;

        Ok(())
    }

    /// Bootstraps new episodic records into the system (Section 4.3).
    ///
    /// When the agent encounters a novel intent where the Memory Bank $M$ yields no 
    /// acceptable trajectory (or fails its simulation fallback), it commits the newly discovered 
    /// path ($e_i$) to the bank. 
    /// The item initializes its $Q$-value at baseline ($0.0$) before temporal-difference learning begins.
    pub async fn store_experience(
        &self,
        intent: String,
        embedding: Embedding,
        experience: String,
    ) -> core::result::Result<(), MemRLError> {
        let item = MemoryItem::new(intent, embedding, experience);
        self.store.upsert(&item).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct MockStore;
    #[async_trait]
    impl VectorStore for MockStore {
        async fn upsert(&self, _: &MemoryItem) -> core::result::Result<(), MemRLError> { Ok(()) }
        async fn search(&self, _: &Embedding, _: usize) -> core::result::Result<Vec<(MemoryItem, f32)>, MemRLError> { Ok(vec![]) }
        async fn update_utility(&self, _: uuid::Uuid, _: f32) -> core::result::Result<(), MemRLError> { Ok(()) }
    }

    #[test]
    fn test_builder_validation() {
        let store = MockStore;
        
        // Invalid alpha
        let agent = MemRLAgentBuilder::new(store).learning_rate(1.5).build();
        assert!(agent.is_err());
        
        let store2 = MockStore;
        // Invalid lambda
        let agent2 = MemRLAgentBuilder::new(store2).utility_balance(-0.5).build();
        assert!(agent2.is_err());

        let store3 = MockStore;
        // Invalid pool size ratio
        let agent3 = MemRLAgentBuilder::new(store3).recall_pool(2).context_window(5).build();
        assert!(agent3.is_err());
        
        let store4 = MockStore;
        // Valid instance
        let agent4 = MemRLAgentBuilder::new(store4).build();
        assert!(agent4.is_ok());
    }
}
