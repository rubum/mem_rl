use crate::domain::{Embedding, MemoryItem};
use crate::error::RevansyError;
use async_trait::async_trait;

#[cfg(feature = "qdrant")]
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointId, PointStruct, PointsIdsList, PointsSelector,
    PointsUpdateOperation, QueryPointsBuilder, SetPayloadPointsBuilder, UpdateBatchPointsBuilder,
    UpsertPointsBuilder, VectorParamsBuilder, points_selector, points_update_operation,
};
#[cfg(feature = "qdrant")]
use qdrant_client::{Payload, Qdrant};
#[cfg(feature = "qdrant")]
use serde_json::json;
use std::collections::HashMap;

/// Interface for episodic memory storage.
///
/// Implementations must support high-throughput vector search (Phase A) 
/// and atomic metadata updates (Utility reinforcement).
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Inserts or updates a `MemoryItem` in the store.
    async fn upsert(&self, item: &MemoryItem) -> Result<(), RevansyError>;

    /// Phase A: Retrieves the $k_1$ semantically closest memories.
    ///
    /// # Returns
    /// A list of `(MemoryItem, score)` pairs, where score is typically cosine similarity.
    async fn search(
        &self,
        query_embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, RevansyError>;

    /// Updates the learned $Q$-value for a single specific experience.
    async fn update_utility(
        &self,
        item_id: uuid::Uuid,
        new_utility: f32,
    ) -> Result<(), RevansyError>;

    /// High-throughput Batch Utility update. 
    ///
    /// Orchestrators should group multiple reinforcement signals into a 
    /// single batch call to reduce gRPC round-trips.
    async fn update_utilities_batch(
        &self,
        updates: Vec<(uuid::Uuid, f32)>,
    ) -> Result<(), RevansyError>;
}

/// A production-grade implementation of `VectorStore` using the Qdrant vector database.
///
/// It leverages the high-performance gRPC client and supports 
/// deep batch-operation optimizations for industrial recommendation workloads.
#[cfg(feature = "qdrant")]
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
}

#[cfg(feature = "qdrant")]
impl QdrantStore {
    /// Connects to a Qdrant instance and optionally creates the target collection.
    ///
    /// # Arguments
    /// * `url` - The gRPC endpoint (e.g., "http://localhost:6334").
    /// * `collection_name` - The unique identifier for the agent's memory bank.
    /// * `vector_size` - The dimensionality of the embeddings (e.g., 768 for Nomic).
    pub async fn new(
        url: &str,
        collection_name: String,
        vector_size: u64,
    ) -> Result<Self, RevansyError> {
        let client = Qdrant::from_url(url)
            .build()
            .map_err(RevansyError::vector_store)?;

        // Ensure collection exists
        if !client
            .collection_exists(&collection_name)
            .await
            .map_err(RevansyError::vector_store)?
        {
            client
                .create_collection(
                    CreateCollectionBuilder::new(&collection_name)
                        .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine)),
                )
                .await
                .map_err(RevansyError::vector_store)?;
        }

        Ok(Self {
            client,
            collection_name,
        })
    }
}

#[cfg(feature = "qdrant")]
#[async_trait]
impl VectorStore for QdrantStore {
    async fn upsert(&self, item: &MemoryItem) -> Result<(), RevansyError> {
        let payload: Payload = json!({
            "intent": item.intent,
            "experience": item.experience,
            "utility": item.utility,
            "id": item.id.to_string(),
            "metadata": item.metadata,
        })
        .try_into()
        .map_err(|e| RevansyError::vector_store(e))?;

        let points = vec![PointStruct::new(
            item.id.to_string(),
            item.intent_embedding.0.clone(),
            payload,
        )];

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points).wait(true))
            .await
            .map_err(RevansyError::vector_store)?;

        Ok(())
    }

    async fn search(
        &self,
        query_embedding: &Embedding,
        top_k: usize,
    ) -> Result<Vec<(MemoryItem, f32)>, RevansyError> {
        let query_request = QueryPointsBuilder::new(&self.collection_name)
            .query(query_embedding.0.clone())
            .limit(top_k as u64)
            .with_payload(true);

        let search_result = self
            .client
            .query(query_request)
            .await
            .map_err(RevansyError::vector_store)?;

        let matches = search_result
            .result
            .into_iter()
            .map(|point| {
                let p = point.payload;
                let id = p.get("id").unwrap().as_str().unwrap().parse().unwrap();
                let intent = p.get("intent").unwrap().as_str().unwrap().to_string();
                let experience = p.get("experience").unwrap().as_str().unwrap().to_string();
                let utility = match p.get("utility").and_then(|v| v.kind.as_ref()) {
                    Some(qdrant_client::qdrant::value::Kind::DoubleValue(d)) => *d as f32,
                    Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)) => *i as f32,
                    _ => {
                        // Fallback to searching the metadata map if not at root
                        0.0
                    }
                };

                let item = MemoryItem {
                    id,
                    intent,
                    intent_embedding: Embedding(vec![]), // Not needed for re-ranking
                    experience,
                    utility,
                    metadata: p
                        .get("metadata")
                        .map(|v| {
                            // Qdrant Value conversion to Serde JSON
                            serde_json::to_value(v).unwrap_or(json!({}))
                        })
                        .unwrap_or(json!({})),
                };
                (item, point.score)
            })
            .collect();

        Ok(matches)
    }

    async fn update_utility(
        &self,
        item_id: uuid::Uuid,
        new_utility: f32,
    ) -> Result<(), RevansyError> {
        let mut payload = Payload::new();
        // Explicitly ensuring we use a double for Qdrant compatibility
        payload.insert("utility", new_utility as f64);

        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection_name, payload)
                    .points_selector(vec![PointId::from(item_id.to_string())]),
            )
            .await
            .map_err(RevansyError::vector_store)?;

        Ok(())
    }

    async fn update_utilities_batch(
        &self,
        updates: Vec<(uuid::Uuid, f32)>,
    ) -> Result<(), RevansyError> {
        if updates.is_empty() {
            return Ok(());
        }

        let mut operations = Vec::with_capacity(updates.len());

        for (id, val) in updates {
            let mut payload = HashMap::new();
            payload.insert("utility".to_string(), (val as f64).into());

            let op = PointsUpdateOperation {
                operation: Some(points_update_operation::Operation::SetPayload(
                    points_update_operation::SetPayload {
                        payload,
                        points_selector: Some(PointsSelector {
                            points_selector_one_of: Some(
                                points_selector::PointsSelectorOneOf::Points(PointsIdsList {
                                    ids: vec![PointId::from(id.to_string())],
                                }),
                            ),
                        }),
                        ..Default::default()
                    },
                )),
            };
            operations.push(op);
        }

        self.client
            .update_points_batch(
                UpdateBatchPointsBuilder::new(&self.collection_name, operations).wait(true),
            )
            .await
            .map_err(RevansyError::vector_store)?;

        Ok(())
    }
}
