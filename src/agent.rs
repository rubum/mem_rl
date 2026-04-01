use crate::domain::{Embedding, MemoryItem, RetrievalOptions, RewardSignal};
use crate::error::RelevansyError;
use crate::math::{compute_memrl_score, z_score_normalize};
use crate::storage::VectorStore;
use rand::Rng;
use serde_json::json;

/// The orchestrating reinforcement learning agent from the Relevansy specification.
///
/// The core follows the Intent-Experience-Utility paradigm. It manages a vector store
/// and performs value-aware retrieval to surface high-utility episodic memories.
pub struct RelevansyAgent<S: VectorStore> {
    pub store: S,
    /// $\alpha \in [0, 1]$: The Learning Rate. Controls how much new experience overrides old utility.
    pub alpha: f32,
    /// $\lambda \in [0, 1]$: The Utility Balance.
    /// Controls the trade-off between raw semantic similarity and learned $Q$-values.
    pub lambda: f32,
    /// $k_1$: The Initial Recall Pool Size. Number of candidates retrieved from Phase A.
    pub k1: usize,
    /// $k_2$: The Final Context Window Size. The number of memories passed to the LLM context after Phase B re-ranking.
    pub k2: usize,
    /// $\epsilon \in [0, 1]$: The Exploration Rate.
    /// Chances of bypassing Phase B to surface unknown or low-utility items (Cold Start solution).
    pub exploration_rate: f32,
}

/// Fluent builder for constructing a `RelevansyAgent`.
pub struct RelevansyAgentBuilder<S: VectorStore> {
    store: S,
    alpha: f32,
    lambda: f32,
    k1: usize,
    k2: usize,
    exploration_rate: f32,
}

impl<S: VectorStore> RelevansyAgentBuilder<S> {
    /// Initializes a builder with the mandatory `VectorStore` and default values:
    /// - $\alpha$ (Learning Rate) = 0.1
    /// - $\lambda$ (Utility Balance) = 0.5
    /// - $k_1$ (Recall Pool) = 5
    /// - $k_2$ (Context Window) = 3
    /// - $\epsilon$ (Exploration) = 0.0
    pub fn new(store: S) -> Self {
        Self {
            store,
            alpha: 0.1,
            lambda: 0.5,
            k1: 5,
            k2: 3,
            exploration_rate: 0.0,
        }
    }

    /// Sets the Learning Rate ($\alpha$). Must be $\in [0, 1]$.
    pub fn learning_rate(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Sets the Utility Balance ($\lambda$). Must be $\in [0, 1]$.
    pub fn utility_balance(mut self, lambda: f32) -> Self {
        self.lambda = lambda;
        self
    }

    /// Sets the raw semantic recall pool size ($k_1$).
    pub fn recall_pool(mut self, k1: usize) -> Self {
        self.k1 = k1;
        self
    }

    /// Sets the final context window size ($k_2$). Must be $\le k_1$.
    pub fn context_window(mut self, k2: usize) -> Self {
        self.k2 = k2;
        self
    }

    /// Sets the $\epsilon$-greedy exploration rate. Must be $\in [0, 1]$.
    pub fn exploration_rate(mut self, epsilon: f32) -> Self {
        self.exploration_rate = epsilon;
        self
    }

    /// Validates and constructs the `RelevansyAgent`.
    pub fn build(self) -> core::result::Result<RelevansyAgent<S>, RelevansyError> {
        if self.alpha < 0.0 || self.alpha > 1.0 {
            return Err(RelevansyError::AgentConfigurationError(
                "alpha must be between 0 and 1".to_string(),
            ));
        }
        if self.lambda < 0.0 || self.lambda > 1.0 {
            return Err(RelevansyError::AgentConfigurationError(
                "lambda must be between 0 and 1".to_string(),
            ));
        }
        if self.k1 < self.k2 {
            return Err(RelevansyError::AgentConfigurationError(
                "recall pool (k1) must be >= context window (k2)".to_string(),
            ));
        }

        if self.exploration_rate < 0.0 || self.exploration_rate > 1.0 {
            return Err(RelevansyError::AgentConfigurationError(
                "exploration_rate must be between 0 and 1".to_string(),
            ));
        }

        Ok(RelevansyAgent {
            store: self.store,
            alpha: self.alpha,
            lambda: self.lambda,
            k1: self.k1,
            k2: self.k2,
            exploration_rate: self.exploration_rate,
        })
    }
}

impl<S: VectorStore> RelevansyAgent<S> {
    /// The Relevansy architecture mandates a strict two-stage process (Section 4.2):
    ///
    /// 1. **Phase A (Semantic Retrieval)**: Fetch $k_1$ items based on raw dense distance.
    /// 2. **Phase B (Value-Aware Rerank)**: Normalize $Q$-values and Similarity, then apply Eq. 7.
    ///
    /// Calls accept `RetrievalOptions` for industrial per-request tuning (A/B testing).
    pub async fn retrieve(
        &self,
        query_embedding: &Embedding,
        options: RetrievalOptions,
    ) -> core::result::Result<Vec<MemoryItem>, RelevansyError> {
        // Resolve effective hyperparameters
        let k1 = options.k1.unwrap_or(self.k1);
        let k2 = options.k2.unwrap_or(self.k2);
        let epsilon = options.epsilon.unwrap_or(self.exploration_rate);
        let lambda = options.lambda.unwrap_or(self.lambda);

        // Phase A: Similarity-Based Recall
        let candidates = self.store.search(query_embedding, k1).await?;

        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // --- Multi-Objective Exploration ($\epsilon$-greedy) ---
        let mut rng = rand::thread_rng();
        if epsilon > 0.0 && rng.r#gen::<f32>() < epsilon {
            let random_idx = rng.r#gen_range(0..candidates.len());
            return Ok(vec![candidates[random_idx].0.clone()]);
        }

        // Phase B: Value-Aware Reranking (Equation 7)
        let scores: Vec<f32> = candidates.iter().map(|(_, s)| *s).collect();
        let utilities: Vec<f32> = candidates.iter().map(|(item, _)| item.utility).collect();

        // Phase B: Value-Aware Rerank (Equation 7)
        let normalized_scores = z_score_normalize(&scores);
        let normalized_utilities = z_score_normalize(&utilities);

        let mut ranked_indices: Vec<usize> = (0..candidates.len()).collect();
        ranked_indices.sort_by(|&a, &b| {
            let score_a =
                compute_memrl_score(normalized_scores[a], normalized_utilities[a], lambda);
            let score_b =
                compute_memrl_score(normalized_scores[b], normalized_utilities[b], lambda);
            score_b.partial_cmp(&score_a).unwrap()
        });

        let results = ranked_indices
            .into_iter()
            .take(k2)
            .map(|i| candidates[i].0.clone())
            .collect();

        Ok(results)
    }

    /// Executes the Offline Utility Update mechanism (Section 4.3).
    pub async fn learn<R: RewardSignal>(
        &self,
        item: &MemoryItem,
        reward: R,
    ) -> core::result::Result<(), RelevansyError> {
        let q_old = item.utility;
        let r = reward.composite_score();
        let q_new = q_old + self.alpha * (r - q_old);

        self.store.update_utility(item.id, q_new).await?;

        Ok(())
    }

    /// Executes high-throughput Batch Learning updates for industrial scalability.
    ///
    /// Calculates new $Q$-values locally for all items and dispatches a single
    /// bulk update to the underlying vector store.
    pub async fn learn_batch<R: RewardSignal>(
        &self,
        updates: Vec<(&MemoryItem, R)>,
    ) -> core::result::Result<(), RelevansyError> {
        if updates.is_empty() {
            return Ok(());
        }

        let mut bulk_updates = Vec::with_capacity(updates.len());

        for (item, reward) in updates {
            let q_old = item.utility;
            let r = reward.composite_score();
            let q_new = q_old + self.alpha * (r - q_old);
            bulk_updates.push((item.id, q_new));
        }

        self.store.update_utilities_batch(bulk_updates).await?;

        Ok(())
    }

    /// Bootstraps new episodic records into the system (Section 4.3).
    ///
    /// This method is the primary ingestion point for new memories. It handles
    /// embedding generation (provided externally) and sets the initial utility to 0.0.
    ///
    /// # Returns
    /// The created `MemoryItem` with its assigned UUID.
    pub async fn store_experience(
        &self,
        intent: String,
        embedding: Embedding,
        experience: String,
        metadata: Option<serde_json::Value>,
    ) -> core::result::Result<MemoryItem, RelevansyError> {
        let item = MemoryItem::new(intent, embedding, experience, metadata.unwrap_or(json!({})));
        self.store.upsert(&item).await?;
        Ok(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct MockStore;

    #[async_trait]
    impl VectorStore for MockStore {
        async fn upsert(&self, _: &MemoryItem) -> core::result::Result<(), RelevansyError> {
            Ok(())
        }
        async fn search(
            &self,
            _: &Embedding,
            _: usize,
        ) -> core::result::Result<Vec<(MemoryItem, f32)>, RelevansyError> {
            Ok(vec![])
        }
        async fn update_utility(
            &self,
            _: uuid::Uuid,
            _: f32,
        ) -> core::result::Result<(), RelevansyError> {
            Ok(())
        }
        async fn update_utilities_batch(
            &self,
            _: Vec<(uuid::Uuid, f32)>,
        ) -> core::result::Result<(), RelevansyError> {
            Ok(())
        }
    }

    #[test]
    fn test_builder_validation() {
        let store = MockStore;
        let agent = RelevansyAgentBuilder::new(store).learning_rate(1.5).build();
        assert!(agent.is_err());

        let store2 = MockStore;
        let agent2 = RelevansyAgentBuilder::new(store2)
            .utility_balance(-0.5)
            .build();
        assert!(agent2.is_err());

        let store3 = MockStore;
        let agent3 = RelevansyAgentBuilder::new(store3)
            .recall_pool(2)
            .context_window(5)
            .build();
        assert!(agent3.is_err());

        let store4 = MockStore;
        let agent4 = RelevansyAgentBuilder::new(store4)
            .exploration_rate(1.1)
            .build();
        assert!(agent4.is_err());

        let store5 = MockStore;
        let agent5 = RelevansyAgentBuilder::new(store5)
            .exploration_rate(0.1)
            .build();
        assert!(agent5.is_ok());
    }
}
