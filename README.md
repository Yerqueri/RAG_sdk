# Enterprise Knowledge Graph RAG Pipeline SDK

This repository contains a highly modular, enterprise-grade **Knowledge Graph RAG** pipeline packaged as an SDK. It combines **vector similarity search** with **Neo4j graph traversal** (hybrid retrieval) for superior answer quality. It supports dynamically switching between LLMs, Vector Databases, Embeddings, and Graph Stores on the fly, and ingests PDFs, Excel files, and Text files out of the box.

## Architecture & Tech Stack

*   **`sdk.py`**: Exposes the `RAGClient` for easy programmatic use.
*   **`core/`**: Single-purpose components for configuration, loading, chunking, storage, graph, entity extraction, and hybrid retrieval.
*   **`strategies/`**:
    *   **LLMs**: Google Gemini GenAI (`langchain-google-genai`), OpenAI, Anthropic, OpenRouter
    *   **Embeddings**: Ollama (`langchain-ollama`), OpenAI, Gemini
    *   **Loaders**: PDF, Excel, Text
    *   **Vector Stores**: Qdrant (`langchain-qdrant`), ChromaDB
    *   **Graph Stores**: Neo4j (`neo4j`)
    *   **Entity Extraction**: LLM-based, spaCy (`spacy`)
*   **`factories/`**: Resolves strategies based on `.env` or SDK overrides.

## Testing & Sanity Checks

Before using the SDK in production, you can run the integration test suite to verify that all components (Vector DB, Graph DB, LLMs, Embeddings, spaCy) are correctly configured and communicating.

```bash
# Run the complete test suite (ensure virtual environment is activated)
python3 live_test.py
```

The `live_test.py` script provisions a temporary test index, checks system availability, and validates end-to-end extraction, ingestion, and querying using both vector and graph modes, then safely cleans up afterward.

### Knowledge Graph Schema
```
(:Document {id, source})
(:Chunk     {id, content, chunk_index})
(:Entity    {name, type})

(Document)-[:CONTAINS]->(Chunk)
(Chunk)-[:MENTIONS]->(Entity)
(Entity)-[:RELATED_TO {relation}]->(Entity)
(Chunk)-[:NEXT]->(Chunk)
```

---

## Prerequisites & Setup

1.  **Install Dependencies**
    You can install dependencies using `pip` or the faster `uv` package manager:

    **Using uv (Recommended):**
    ```bash
    # Install uv if you don't have it
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Create a virtual environment and install dependencies
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt
    ```

    **Using pip:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

    *Optional: spaCy model for offline entity extraction*
    ```bash
    python -m spacy download en_core_web_sm
    ```

2.  **Start Databases**
    ```bash
    # Start both Qdrant (vector) and Neo4j (graph)
    docker-compose up -d

    # Or start only what you need:
    docker-compose up -d qdrant
    docker-compose up -d neo4j
    ```
    Neo4j browser UI is available at **http://localhost:7474** (user: `neo4j`, password: `password`).

3.  **Environment Variables (`.env`)**
    Copy `.env.example` to `.env`. Here is a working example configured for local Ollama embeddings and Gemini generation:

    ### Example `.env` File
    ```env
    # Core Providers
    VECTOR_DB_PROVIDER=qdrant
    EMBEDDING_PROVIDER=ollama
    LLM_PROVIDER=gemini

    # Models
    GEMINI_LLM_MODEL=gemini-2.5-flash
    GEMINI_EMBEDDING_MODEL=text-embedding-004
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text

    # API Keys
    GOOGLE_API_KEY=********

    # Vector DB
    QDRANT_URL=http://localhost:6333
    DATA_DIR=./data
    COLLECTION_NAME=my_rag_collection

    # ── Knowledge Graph ──────────────────────────────
    GRAPH_STORE_PROVIDER=neo4j
    NEO4J_URL=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=password

    # Entity Extraction: "llm" (uses configured LLM) or "spacy" (fast/offline)
    ENTITY_EXTRACTION_PROVIDER=llm

    # Retrieval mode: "vector" | "graph" | "hybrid"
    RETRIEVAL_MODE=hybrid
    ```

---

## How to Use (CLI)

### Ingestion
```bash
# Standard vector-only ingestion (backward compatible)
python ingest.py
python ingest.py --file ./data/my_report.pdf

# Graph-augmented ingestion (extracts entities → Neo4j)
python ingest.py --enable-graph
python ingest.py --file ./data/my_report.pdf --enable-graph

# Use spaCy for fast offline entity extraction
python ingest.py --enable-graph --entity-provider spacy
```

### Querying
```bash
# Vector-only (default / backward compatible)
python query.py "What does the report say?"

# Hybrid retrieval (vector + knowledge graph)
python query.py "What does the report say?" --retrieval-mode hybrid

# Graph-only retrieval
python query.py "What does the report say?" --retrieval-mode graph
```

---

## How to Use (Python SDK)

```python
from sdk import RAGClient

# 1. Backward-compatible — pure vector RAG (no changes needed)
client = RAGClient()
client.ingest_file("./data/financial_report.xlsx")
print(client.query("What is the Q3 revenue?"))

# 2. Knowledge Graph RAG — ingest with graph building
graph_client = RAGClient(
    llm_provider="gemini",
    embedding_provider="ollama",
    vector_db_provider="qdrant",
    enable_graph=True,                      # build Neo4j graph on ingest
    entity_extraction_provider="llm",       # use LLM to extract entities & relationships
)
graph_client.ingest_directory("./data")

# 3. Hybrid query (vector + graph traversal)
answer = RAGClient(
    retrieval_mode="hybrid",
).query("Who are the key executives and what products do they oversee?")
print(answer)

# 4. Mix and match providers
custom = RAGClient(
    llm_provider="anthropic",
    embedding_provider="openai",
    vector_db_provider="chroma",
    enable_graph=True,
    entity_extraction_provider="spacy",     # fast, offline
    retrieval_mode="hybrid",
)
custom.ingest_directory("./my_docs")
print(custom.query("Summarise the key entities and their relationships."))
```
