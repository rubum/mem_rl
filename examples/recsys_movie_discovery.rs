use miette::Result;
use revansy::{EmbeddingService, OllamaEmbedder, QdrantStore, RetrievalOptions, RevansyAgentBuilder, RewardSignal};
use serde_json::json;

/// A multi-objective reward signal representing user interaction.
/// This simulates common signals used in industrial recommendation systems.
struct UserEngagementReward {
    clicked: bool,
    watch_time_percentage: f32, // 0.0 to 1.0 (how long they stayed)
    gave_rating: bool,
}

impl RewardSignal for UserEngagementReward {
    fn composite_score(&self) -> f32 {
        let mut score = 0.0;
        if self.clicked {
            score += 0.3;
        }
        score += self.watch_time_percentage * 0.5;
        if self.gave_rating {
            score += 0.2;
        }
        score.clamp(0.0, 1.0)
    }
}

/// This example demonstrates a movie recommendation engine using Revansy.
/// It solves the "Cold Start" problem by allowing 20% exploration probability.
#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    println!("Initializing Revansy Movie Recommender...");

    let embedder = OllamaEmbedder::new("nomic-embed-text:latest".to_string());
    let collection_name = format!("recsys_demo_{}", uuid::Uuid::new_v4().as_simple());
    let store = QdrantStore::new("http://localhost:6334", collection_name.clone(), 768).await?;

    // We set epsilon (exploration_rate) to 0.2
    // This gives any "new" or "unheard of" movie a 20% chance to be discovered.
    let agent = RevansyAgentBuilder::new(store)
        .learning_rate(0.4)
        .utility_balance(0.5)
        .recall_pool(10)
        .context_window(3)
        .exploration_rate(0.2)
        .build()?;

    println!("--------------------------------------------------");
    println!("[Step 1] Seeding movies with historical engagement...");

    let movies = vec![
        (
            "The Matrix (1999) - Sci-fi masterpiece about simulated reality",
            "Action | Sci-Fi",
            0.95, // Viral / Classic status
        ),
        (
            "Inception (2010) - Mind-bending dream-within-a-dream thriller",
            "Action | Thriller",
            0.8, // Highly rated
        ),
        (
            "Plan 9 from Outer Space (1959) - Infamously bad cult classic",
            "Sci-Fi | Horror",
            0.15, // Users hate it (Low Click-Through)
        ),
        (
            "Dune: Part Two (2024) - Epic cinematic space journey",
            "Sci-Fi | Adventure",
            0.0, // BRAND NEW (No historical clicks / Cold Start)
        ),
    ];

    for (desc, genre, engagement) in movies {
        println!("  -> Cataloging: {}", desc);
        let emb = embedder.embed(desc).await?;

        // Metadata tagging for genre-specific filtering.
        let metadata = json!({ "genres": genre, "is_new": engagement == 0.0 });
        
        // 4. Store and immediately learn for established movies
        let item = agent
            .store_experience(desc.to_string(), emb, desc.to_string(), Some(metadata))
            .await?;

        if engagement > 0.0 {
            agent
                .learn(
                    &item,
                    UserEngagementReward {
                        clicked: true,
                        watch_time_percentage: engagement,
                        gave_rating: engagement > 0.8,
                    },
                )
                .await?;
        }
    }

    println!("--------------------------------------------------");

    let query = "I want a high-octane sci-fi space action movie!";
    println!("\n[Step 2] User Search Query: '{}'", query);
    println!(
        "We have exploration enabled (20%). The brand new 'Dune: Part Two' is mathematically invisible (utility=0),"
    );
    println!("but Revansy will eventually surface it to gather data.");

    let query_emb = embedder.embed(query).await?;

    // Run the search multiple times to demonstrate the probabilistic discovery.
    for i in 1..=10 {
        println!("\n--- Interaction Loop {} ---", i);
        let recommendations = agent.retrieve(&query_emb, RetrievalOptions::default()).await?;

        for (idx, movie) in recommendations.iter().enumerate() {
            let is_new = movie
                .metadata
                .get("is_new")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let status = if is_new {
                "[COLD START EXPLORATION]"
            } else {
                "[EXPLOITING LEARNED UTILITY]"
            };

            println!(
                "  Rank {} | Utility: {:.2} | {} {}",
                idx + 1,
                movie.utility,
                movie.intent,
                status
            );
        }
    }

    Ok(())
}
