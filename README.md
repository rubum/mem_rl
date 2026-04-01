# Relevansy (v0.1.0: Industrial-Grade Foundation)

**Relevansy** is a production-ready, memory-based reinforcement learning engine for autonomous AI agents and high-scale Recommendation Systems. 

Built on the core principles of the **MemRL** research paper, Relevansy extends the architecture into an industrial-grade library that mathematically sorts memories by analyzing both context similarity *and* a temporally-learned utility factor.

### Industrial Features
This is the baseline core engine. It includes:
- **Phase A/B Reranking Engine**: Seamlessly integrates vector search and RL math using Z-score normalization for scale.
- **Async Batch Learning**: High-throughput `learn_batch` API for processing multiple rewards in a single optimized gRPC round-trip.
- **Per-Request Hyperparameter Tuning**: Dynamic `RetrievalOptions` for A/B testing, allowing per-user or per-request overrides of $\lambda$, $\epsilon$, and pool sizes.
- **Epsilon-Greedy Discovery**: Configurable $\epsilon$ to solve the "Cold Start" problem in Recommendation Systems.
- **`RewardSignal` Trait**: Support for complex multi-objective scoring (Latency, Cost, CTR, etc.).
- **Qdrant & Ollama Connectors**: Optimized, battle-tested reference implementations.

## Prerequisites

To run this project, you will need the following dependencies installed on your system:

1. **[Rust & Cargo](https://rustup.rs/)**: The Rust toolchain.
2. **[Docker](https://docs.docker.com/get-docker/)** and **Docker Compose**: Used to run the Qdrant vector database locally.
3. **[Ollama](https://ollama.com/)**: Required for generating embeddings.

## Setup Instructions

### 1. Start the Qdrant Store

A `docker-compose.yml` is provided to spin up a local instance of the Qdrant vector store.

```bash
docker-compose up -d
```
*This will map Qdrant to ports `6333` and `6334`, and maintain its state in the `./qdrant_storage` directory.*

### 2. Configure Ollama Embedding Model

Relevansy uses the `nomic-embed-text` model via Ollama for text embeddings. Make sure your Ollama daemon is running, and pull the required model:

```bash
ollama pull nomic-embed-text
```

### 3. Run the Example Simulations

To test the Relevansy mathematical and industrial pipeline, you can run the following examples:

**Agent Simulation (Value-Aware Reranking):**
```bash
cargo run --example value_aware_reranking --features "qdrant ollama"
```

**RecSys Discovery (Cold Start Solution):**
```bash
cargo run --example recsys_movie_discovery --features "qdrant ollama"
```

**Industrial Batch & Tuning (A/B Testing Demo):**
```bash
cargo run --example industrial_batch_tuning --features "qdrant ollama"
```

## Integrating Relevansy in your projects

To use Relevansy in your own real-time agents, add it to your dependencies.

```toml
[dependencies]
relevansy = { version = "0.1", features = ["qdrant", "ollama"] }
```

### The Builder Pattern

Construct a real-time agent memory system via the `RelevansyAgentBuilder`:

```rust
use relevansy::RelevansyAgentBuilder;

let agent = RelevansyAgentBuilder::new(your_vector_store)
    .learning_rate(0.3)
    .utility_balance(0.5)
    .recall_pool(50)
    .context_window(3)
    .exploration_rate(0.1) // 10% chance to explore unknown items
    .build()?;
```

### Per-Request Tuning (A/B Testing)

You can override hyperparameters on a per-request basis using `RetrievalOptions`:

```rust
use relevansy::RetrievalOptions;

let options = RetrievalOptions::new()
    .epsilon(0.8) // High exploration for this specific user
    .lambda(0.2); // Prioritize semantic similarity over learned utility

let results = agent.retrieve(&query_emb, options).await?;
```

### Batch Learning

For high-throughput systems, use the batch learning API to update multiple memories efficiently:

```rust
// Group multiple items and their associated rewards
let updates = vec![(&item1, reward1), (&item2, reward2)];

// Dispatched in a single optimized gRPC call to Qdrant
agent.learn_batch(updates).await?;
```

## Background: From MemRL to Relevansy

This implementation started with the formalisms of the **MemRL** research paper—enabling autonomous AI agents to self-evolve via episodic memory and Reinforcement Learning. However, this library has evolved into the **Relevansy Foundation**, which extends the original theory into a production-grade, global-scale memory architecture.

### The MemRL Foundation (Section 4.1)
The core remains anchored to the **Intent-Experience-Utility** triplet:
- **Intent ($z_i$)**: The context or query that triggered the response.
- **Experience ($e_i$)**: The recorded artifact or solution trace.
- **Utility ($Q_i$)**: A learned scalar representing historical empirical helpfulness.

### The Relevansy Evolution (Beyond the Paper)
Relevansy introduces industrial-scale features required for real-world Agents and Recommender Systems:

1. **$\epsilon$-Greedy Discovery (Cold Start Solution)**: 
   Periodic exploration allows solving the "Cold Start" problem by surfacing new items to gather fresh reward data.

2. **The `RewardSignal` Trait (Multi-Objective RL)**: 
   Decouples the reward logic, allowing developers to collapse complex metrics like **Latency**, **Cost**, **CTR**, and **Correctness** into the final Q-value update.

3. **High-Throughput Batching**: 
   Optimized storage layer for high-frequency feedback loops.

## Architecture Summary

Relevansy functions via a learning feedback loop:
- Computes text embeddings for a given request via **Ollama**.
- Performs a semantic similarity search in **Qdrant** to retrieve similar past intents.
- Performs Phase B re-ranking based on learned utilities.
- Records the operation's utility factor (Async Batch) and adjusts its knowledge representation appropriately on successive queries.
