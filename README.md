# MemRL

**MemRL** is a sophisticated, memory-based reinforcement learning crate for autonomous AI agents. Unlike traditional semantic search (RAG), MemRL agents mathematically sort memories by analyzing both context similarity *and* a temporally-learned utility factor.

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

MemRL uses the `nomic-embed-text` model via Ollama for text embeddings. Make sure your Ollama daemon is running, and pull the required model:

```bash
ollama pull nomic-embed-text
```

### 3. Run the Example Agent Simulation

To test the MemRL mathematical pipeline interactively, you can run the included Docker-backed simulation example:

```bash
cargo run --example value_aware_reranking --features "qdrant ollama"
```

## Integrating MemRL in your projects

To use MemRL in your own real-time agents, add it to your dependencies. If you don't require external SDKs like `qdrant-client` or `reqwest`, disable the default features.

```toml
[dependencies]
mem_rl = { version = "0.1", default-features = false }
```

### The Builder Pattern

Construct a real-time agent memory system via the `MemRLAgentBuilder`:

```rust
use mem_rl::MemRLAgentBuilder;

let agent = MemRLAgentBuilder::new(your_vector_store)
    .learning_rate(0.3)
    .utility_balance(0.5)
    .recall_pool(50)
    .context_window(3)
    .build()?;
```

## Background (Based on MemRL Paper)

This implementation is based on the concepts formalized in the **MemRL** research paper. It enables autonomous AI agents to self-evolve by learning from episodic memory using Reinforcement Learning techniques. 

The system relies on an **Intent-Experience-Utility** triplet:
- **Intent ($z_i$)**: The context or query that triggered the experience.
- **Experience ($e_i$)**: The solution trace or trajectory results.
- **Utility ($Q_i$)**: A learned scalar value representing how helpful this memory was in the past.

It employs **Two-Phase Retrieval** to fetch relevant experiences:
- **Phase A (Semantic Similarity)**: Retrieves candidate memories based strictly on embedding cosine distance.
- **Phase B (Value-Aware Selection)**: Re-ranks candidates by computing a Z-score normalized composite score that balances semantic similarity and the historical utility ($Q_i$) of the memory.

After an experience is utilized and a new reward is observed, the agent updates the utility profile via **Monte Carlo Learning** ($Q_{new} \leftarrow Q_{old} + \alpha(r - Q_{old})$) rather than discarding the memory.

## Architecture Summary

MemRL functions via a learning feedback loop:
- Computes text embeddings for a given request via **Ollama**.
- Performs a semantic similarity search in **Qdrant** to retrieve similar past intents.
- Chooses to rely on retrieved experiences or executes a fallback simulation.
- Records the operation's utility factor and adjusts its knowledge representation appropriately on successive queries.
