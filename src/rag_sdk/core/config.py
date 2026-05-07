import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    @staticmethod
    def get_env_var(name: str, default: str = None) -> str:
        value = os.getenv(name, default)
        if value is None:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    @property
    def qdrant_url(self) -> str:
        return self.get_env_var("QDRANT_URL")

    @property
    def llm_provider(self) -> str:
        return self.get_env_var("LLM_PROVIDER").lower()

    @property
    def vector_db_provider(self) -> str:
        return self.get_env_var("VECTOR_DB_PROVIDER", "qdrant").lower()

    @property
    def embedding_provider(self) -> str:
        return self.get_env_var("EMBEDDING_PROVIDER").lower()

    @property
    def data_dir(self) -> str:
        return self.get_env_var("DATA_DIR")

    @property
    def collection_name(self) -> str:
        return self.get_env_var("COLLECTION_NAME")

    @property
    def chroma_persist_dir(self) -> str:
        return self.get_env_var("CHROMA_PERSIST_DIR", "./chroma_data")

    # LLM Models
    @property
    def openai_llm_model(self) -> str:
        return self.get_env_var("OPENAI_LLM_MODEL", "gpt-3.5-turbo")

    @property
    def gemini_llm_model(self) -> str:
        return self.get_env_var("GEMINI_LLM_MODEL", "gemini-pro")

    @property
    def anthropic_llm_model(self) -> str:
        return self.get_env_var("ANTHROPIC_LLM_MODEL", "claude-3-haiku-20240307")

    @property
    def openrouter_llm_model(self) -> str:
        return self.get_env_var("OPENROUTER_LLM_MODEL", "meta-llama/llama-3-8b-instruct")

    # Embedding Models
    @property
    def ollama_embedding_model(self) -> str:
        return self.get_env_var("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    @property
    def openai_embedding_model(self) -> str:
        return self.get_env_var("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")

    @property
    def gemini_embedding_model(self) -> str:
        return self.get_env_var("GEMINI_EMBEDDING_MODEL", "models/embedding-001")

    # ── Graph Store ──────────────────────────────────────────────────── #

    @property
    def graph_store_provider(self) -> str:
        return self.get_env_var("GRAPH_STORE_PROVIDER", "neo4j").lower()

    @property
    def neo4j_url(self) -> str:
        return self.get_env_var("NEO4J_URL", "bolt://localhost:7687")

    @property
    def neo4j_username(self) -> str:
        return self.get_env_var("NEO4J_USERNAME", "neo4j")

    @property
    def neo4j_password(self) -> str:
        return self.get_env_var("NEO4J_PASSWORD", "password")

    # ── Entity Extraction ─────────────────────────────────────────────── #

    @property
    def entity_extraction_provider(self) -> str:
        return self.get_env_var("ENTITY_EXTRACTION_PROVIDER", "llm").lower()

    # ── Retrieval Mode ────────────────────────────────────────────────── #

    @property
    def retrieval_mode(self) -> str:
        """'vector' | 'graph' | 'hybrid'  (default: 'vector' for backward compat)"""
        return self.get_env_var("RETRIEVAL_MODE", "vector").lower()

config = Config()
