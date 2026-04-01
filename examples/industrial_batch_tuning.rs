use miette::Result;
use relevansy::{
    EmbeddingService, MemoryItem, OllamaEmbedder, QdrantStore, RetrievalOptions,
    RelevansyAgentBuilder, RewardSignal,
};

/// A simple reward for our industrial simulation.
struct SimpleReward(f32);
impl RewardSignal for SimpleReward {
    fn composite_score(&self) -> f32 {
        self.0
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize standard logging
    env_logger::init();

    // 1. Initialize Infrastructure
    // We use the 'nomic-embed-text' model by default.
    let embedder = OllamaEmbedder::new("nomic-embed-text".to_string());
    let collection_name = format!("industrial_demo_{}", uuid::Uuid::new_v4().as_simple());
    let store = QdrantStore::new("http://localhost:6334", collection_name.clone(), 768).await?;

    // 2. Initialize the Relevansy Agent with "Enterprise Defaults"
    let agent = RelevansyAgentBuilder::new(store)
        .learning_rate(0.3)
        .utility_balance(0.5)
        .recall_pool(100)
        .context_window(10)
        .build()?;

    println!(
        "🚀 Starting Industrial Relevansy Simulation (Collection: {})",
        collection_name
    );
    println!("--------------------------------------------------");

    // 3. Batch Seeding
    println!("[Stage 1] Batch Seeding 10 items...");
    let mut batch_items = Vec::new();
    for i in 1..=10 {
        let intent = format!("Product Query #{}", i);
        let emb = embedder.embed(&intent).await?;
        let item = agent
            .store_experience(
                intent,
                emb,
                format!("Recommended Product Result for ID {}", i),
                None,
            )
            .await?;
        batch_items.push(item);
    }

    // 4. Batch Learning (Offline Worker Simulation)
    println!("[Stage 2] Simulating a Batch Reward Worker (10 updates at once)...");
    let mut reward_updates: Vec<(&MemoryItem, SimpleReward)> = Vec::new();
    for (i, item) in batch_items.iter().enumerate() {
        // We simulate that items at the end of the list got higher engagement
        let reward_val = (i as f32 + 1.0) / 10.0;
        reward_updates.push((item, SimpleReward(reward_val)));
    }

    // Use the new high-throughput learn_batch API
    agent.learn_batch(reward_updates).await?;
    println!("  -> Successfully updated 10 utilities in a single batch operation.");

    // 5. Per-Request Hyperparameter Tuning (A/B Testing)
    println!("\n[Stage 3] Per-Request Tuning (A/B Testing Situation)");
    let query_intent = "Product Query #5";
    let query_emb = embedder.embed(query_intent).await?;

    // --- Scenario A: Exploration Focused ---
    // User is "New", we want to discover their tastes.
    println!("\nScenario A: Exploration Focus (User is new)");
    let options_a = RetrievalOptions::new()
        .epsilon(0.8) // High exploration
        .lambda(0.2); // Low utility weight

    let results_a = agent.retrieve(&query_emb, options_a).await?;
    println!("  -> Results returned (Total: {})", results_a.len());

    // --- Scenario B: Conversion Focused ---
    // User is "Loyal", we want to exploit what we know works.
    println!("\nScenario B: Conversion Focus (User is loyal)");
    let options_b = RetrievalOptions::new()
        .epsilon(0.0) // No exploration
        .lambda(0.9); // Heavy utility weight (Exploit)

    let results_b = agent.retrieve(&query_emb, options_b).await?;
    if let Some(top) = results_b.first() {
        println!(
            "  -> Top Result: '{}' (Utility: {:.2})",
            top.intent, top.utility
        );
    }

    println!("\n--------------------------------------------------");
    println!("Industrial simulation complete.");
    Ok(())
}
