use mem_rl::{EmbeddingService, MemRLAgentBuilder, OllamaEmbedder, QdrantStore, Reward};
use miette::Result;

/// This simulation provides a real-world demonstration of the MemRL architecture.
///
/// It isolates a vector store, dynamically embeds several "historical trajectories",
/// manually assigns them uneven "rewards" to simulate temporal-difference learning,
/// and then executes a live RAG query to observe the Value-Aware Reranking mathematically.
#[tokio::main]
async fn main() -> Result<()> {
    // Initialize standard logging for debugging embedded dependency output
    env_logger::init();

    println!("Initializing MemRL Agent Environment...");

    // --- Core Engine Setup ---

    // 1. Setup Embedding Service
    // We use the open-source `nomic-embed-text` LLM from Ollama.
    // Ensure the Ollama daemon is running locally: `ollama run nomic-embed-text`.
    let embedder = OllamaEmbedder::new("nomic-embed-text:latest".to_string());

    // 2. Setup Vector Store (Qdrant)
    // To ensure the simulation always starts clean and never collides with
    // a real production memory bank, we generate a random UUID for the collection.
    // Note: The vector dimension (768) must precisely match what nomic outputs.
    let collection_name = format!("memrl_demo_{}", uuid::Uuid::new_v4().as_simple());
    let store = QdrantStore::new("http://localhost:6334", collection_name.clone(), 768).await?;

    // 3. Initialize MemRL Agent
    // We use the builder pattern to strictly enforce the algorithmic hyperparameters.
    // - alpha = 0.5 (fast learning): New rewards will aggressively overwrite historical utility.
    // - lambda = 0.6: In the final Equation 7 calculation, the historical utility will
    //   hold 60% of the weight, and raw cosine similarity will hold 40%.
    // - recall_pool = 10: Phase A will pull up to 10 close semantic matches via fast indexing.
    // - context_window = 3: Phase B will pare this down to the top 3 items to feed to an LLM.
    let agent = MemRLAgentBuilder::new(store)
        .learning_rate(0.5)
        .utility_balance(0.6)
        .recall_pool(10)
        .context_window(3)
        .build()?;

    println!(
        "MemRL Agent initialized successfully on temporary collection '{}'!",
        collection_name
    );
    println!("--------------------------------------------------");

    // --- Bootstrapping Phase (Seeding) ---

    println!("\n[Phase 1] Seeding the Memory Bank with historical trajectories...");

    // We emulate a scenario where an autonomous sysadmin agent has previously
    // encountered several similar contexts (Intent) and tried various shell
    // solutions (Experience). Over time, some of those solutions proved to be
    // poor (Utility = 0.2) while others were robust (Utility = 0.9).
    let seeds = vec![
        (
            "Configure SSH port to something non-standard",
            "Change Port in /etc/ssh/sshd_config to 2222",
            0.2, // Low utility: semantic match to SSH is okay, but this isn't very secure.
        ),
        (
            "Hardening linux SSH server config for production",
            "Set PasswordAuthentication=no, PermitRootLogin=no, and mandate Ed25519 SSH keys in sshd_config.",
            0.9, // High utility: highly secure and best practice.
        ),
        (
            "Setup basic SSH securely with firewall",
            "Install fail2ban, allow SSH in UFW, and restart services.",
            0.6, // Medium utility: good auxiliary security.
        ),
        (
            "Install python dependencies",
            "apt-get install python3-pip",
            0.1, // Noise: Unrelated experience to test filtering.
        ),
    ];

    // Iterating over our pseudo-history to manually seed the Qdrant database.
    for (intent, experience, reward_val) in seeds {
        println!("  -> Embedding memory: '{}'", intent);
        // Phase A requires dense vectors. We invoke the local Ollama daemon.
        let emb = embedder.embed(intent).await?;

        // Push the new core Memory Triplet (Intent, Embedding, Experience).
        // By default, the `utility` (Q-value) is initialized to zero.
        agent
            .store_experience(intent.to_string(), emb, experience.to_string())
            .await?;

        // We immediately query for the item we just injected to retrieve its ID,
        // and then simulate a Reinforcement Learning feedback update.
        let emb2 = embedder.embed(intent).await?;
        if let Some(target_memory) = agent.retrieve(&emb2).await?.into_iter().next() {
            // Apply a static scalar reward to simulate environmental feedback.
            // This invokes Monte Carlo Temporal-Difference learning (Eq. 4).
            agent.learn(&target_memory, Reward(reward_val)).await?;
            println!(
                "     Learned Utility updated to {:.2} for '{}'",
                reward_val, intent
            );
        }
    }

    println!("--------------------------------------------------");

    // --- Retrieval Phase ---

    // An ambiguous real-world query. It asks about "secure setup", which semantically
    // overlays with all three of our SSH seeds.
    let query_intent = "What is the best most secure way to setup my ssh daemon startup?";
    println!("\n[Phase 2] Real-time Retrieval Query: '{}'", query_intent);

    let query_embedding = embedder.embed(query_intent).await?;

    // The single `.retrieve()` call automatically executes Phase A (Cosine Similarity)
    // followed immediately by Phase B (Z-Score Normalization and Lambda Merging).
    let retrieved_memories = agent.retrieve(&query_embedding).await?;

    if retrieved_memories.is_empty() {
        println!("No relevant memories found.");
    } else {
        println!("\nResults after Phase A (Semantic Search) & Phase B (Value-Aware Rerank):");
        println!(
            "Note how lambda (0.6) allows higher utility items to outrank slightly closer semantic matches."
        );

        for (i, memory) in retrieved_memories.iter().enumerate() {
            println!(
                "\n  Rank {} | Utility: {:.2} | Intent Match: '{}'\n           Experience: {}",
                i + 1,
                memory.utility,
                memory.intent,
                memory.experience
            );
        }
    }

    Ok(())
}
