# Enterprise Hybrid RAG Pipeline SDK

This repository contains a highly modular, enterprise-grade Retrieval-Augmented Generation (RAG) pipeline packaged as an SDK. It supports dynamically switching between LLMs, Vector Databases, and Embeddings on the fly, and supports ingesting PDFs, Excel files, and Text files out of the box.

## Architecture

*   **`sdk.py`**: Exposes the `RAGClient` for easy programmatic use.
*   **`core/`**: Single-purpose components for configuration, loading, chunking, and storage.
*   **`strategies/`**: 
    *   **LLMs**: OpenAI, Gemini, Anthropic, OpenRouter
    *   **Embeddings**: Ollama, OpenAI, Gemini
    *   **Loaders**: PDF, Excel, Text
    *   **Vector Stores**: Qdrant, ChromaDB
*   **`factories/`**: Resolves strategies based on `.env` or SDK overrides.

## Prerequisites & Setup

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start Vector Database (If using Qdrant)**
    ```bash
    docker-compose up -d
    ```

3.  **Environment Variables (`.env`)**
    Copy `.env.example` to `.env`. This project uses these values as defaults.

    ### Example `.env` File
    ```env
    # Providers
    VECTOR_DB_PROVIDER=qdrant
    EMBEDDING_PROVIDER=ollama
    LLM_PROVIDER=gemini

    # Models
    GEMINI_LLM_MODEL=gemini-pro
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text

    # API Keys (Only populate what you use)
    GOOGLE_API_KEY=your_google_api_key_here

    # Database
    QDRANT_URL=http://localhost:6333
    DATA_DIR=./data
    COLLECTION_NAME=my_rag_collection
    ```

    ### SDK Overrides & Optional `.env` Variables
    When using the CLI scripts (`ingest.py` or `query.py`), the core provider variables (`VECTOR_DB_PROVIDER`, `EMBEDDING_PROVIDER`, `LLM_PROVIDER`) in your `.env` are **required** to know which services to use. 
    
    However, when using the `RAGClient` **SDK programmatically**, you can pass these providers directly to the constructor. If you pass them in the code, the corresponding `.env` variables become **entirely optional and are completely ignored**.
    
    For example, if your code is `RAGClient(llm_provider="openai")`, the SDK will ignore the `LLM_PROVIDER` in your `.env` file and look strictly for `OPENAI_LLM_MODEL` and `OPENAI_API_KEY`.

---

## How to Use (CLI)

### Ingestion
Ingest an entire directory or a single file. Supports `.txt`, `.pdf`, `.xls`, `.xlsx`.
```bash
# Ingest entire ./data folder
python ingest.py

# Ingest a single file
python ingest.py --file ./data/my_report.pdf
```

### Querying
```bash
python query.py "What does the report say?"
```

---

## How to Use (Python SDK)

The true power of this architecture is the `RAGClient`. You can import it into any Python script and completely override the `.env` file programmatically!

```python
from sdk import RAGClient

# 1. Use defaults from .env
client = RAGClient()
client.ingest_file("./data/financial_report.xlsx")
print(client.query("What is the Q3 revenue?"))

# 2. Override providers on the fly!
custom_client = RAGClient(
    llm_provider="anthropic",
    embedding_provider="openai",
    vector_db_provider="chroma"
)
custom_client.ingest_directory("./my_docs")
print(custom_client.query("Summarize the docs."))
```
