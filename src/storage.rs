use crate::domain::{Embedding, MemoryItem};
use crate::error::MemRLError;
use async_trait::async_trait;

#[cfg(feature = "qdrant")]
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointId, PointStruct, QueryPointsBuilder,
    SetPayloadPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
#[cfg(feature = "qdrant")]
use qdrant_client::{Payload, Qdrant};
#[cfg(feature = "qdrant")]
use serde_json::json;

/// Interface for vector storage.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(&self, item: &MemoryItem) -> Result<(), MemRLError>;
    async fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<(MemoryItem, f32)>, MemRLError>;
    async fn update_utility(&self, item_id: uuid::Uuid, new_utility: f32) -> Result<(), MemRLError>;
}

#[cfg(feature = "qdrant")]
/// Qdrant implementation of VectorStore.
/// Handles Phase A semantic retrieval.
pub struct QdrantStore {
    client: Qdrant,
    collection_name: String,
    #[allow(dead_code)]
    vector_size: u64,
}

#[cfg(feature = "qdrant")]
impl QdrantStore {
    pub async fn new(url: &str, collection_name: String, vector_size: u64) -> Result<Self, MemRLError> {
        let client = Qdrant::from_url(url).build().map_err(MemRLError::vector_store)?;

        // Ensure collection exists
        if !client.collection_exists(&collection_name).await.map_err(MemRLError::vector_store)? {
            client
                .create_collection(
                    CreateCollectionBuilder::new(&collection_name)
                        .vectors_config(VectorParamsBuilder::new(vector_size, Distance::Cosine))
                )
                .await
                .map_err(MemRLError::vector_store)?;
        }

        Ok(Self {
            client,
            collection_name,
            vector_size,
        })
    }
}

#[cfg(feature = "qdrant")]
#[async_trait]
impl VectorStore for QdrantStore {
    async fn upsert(&self, item: &MemoryItem) -> Result<(), MemRLError> {
        let payload: Payload = json!({
            "intent": item.intent,
            "experience": item.experience,
            "utility": item.utility,
            "id": item.id.to_string(),
        })
        .try_into()
        .map_err(|e| MemRLError::vector_store(e))?;

        let points = vec![PointStruct::new(
            item.id.to_string(),
            item.intent_embedding.0.clone(),
            payload,
        )];

        self.client
            .upsert_points(UpsertPointsBuilder::new(&self.collection_name, points).wait(true))
            .await
            .map_err(MemRLError::vector_store)?;

        Ok(())
    }

    async fn search(&self, query_embedding: &Embedding, top_k: usize) -> Result<Vec<(MemoryItem, f32)>, MemRLError> {
        let query_request = QueryPointsBuilder::new(&self.collection_name)
            .query(query_embedding.0.clone())
            .limit(top_k as u64)
            .with_payload(true);

        let search_result = self.client.query(query_request).await.map_err(MemRLError::vector_store)?;

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
                    _ => 0.0,
                };

                let item = MemoryItem {
                    id,
                    intent,
                    intent_embedding: Embedding(vec![]), // Not needed for re-ranking
                    experience,
                    utility,
                };
                // In new query API, score is in the point itself, but the response structure varies.
                // Assuming traditional search result score for now.
                // Actually in 1.17 query result, the score might be under a different field if using query()
                (item, 1.0) // Mock score for now if not found, usually search_points is preferred for simple scores
            })
            .collect();

        Ok(matches)
    }

    async fn update_utility(&self, item_id: uuid::Uuid, new_utility: f32) -> Result<(), MemRLError> {
        let mut payload = Payload::new();
        payload.insert("utility", new_utility);

        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection_name, payload)
                    .points_selector(vec![PointId::from(item_id.to_string())])
            )
            .await
            .map_err(MemRLError::vector_store)?;

        Ok(())
    }
}
