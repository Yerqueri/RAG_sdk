from langchain_core.embeddings import Embeddings
from rag_sdk.strategies.embedding.ollama_strategy import OllamaStrategy
from rag_sdk.strategies.embedding.openai_strategy import OpenAIStrategy
from rag_sdk.strategies.embedding.gemini_strategy import GeminiStrategy
from rag_sdk.core.config import config

class EmbeddingFactory:
    @staticmethod
    def get_embeddings(provider: str = None) -> Embeddings:
        provider = provider or config.embedding_provider
        
        if provider == "ollama":
            return OllamaStrategy().get_embeddings()
        elif provider == "openai":
            return OpenAIStrategy().get_embeddings()
        elif provider == "gemini":
            return GeminiStrategy().get_embeddings()
        else:
            raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}. Use 'ollama', 'openai', or 'gemini'.")
