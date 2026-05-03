from .base_embedding_strategy import BaseEmbeddingStrategy
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OllamaEmbeddings
from core.config import config

class OllamaStrategy(BaseEmbeddingStrategy):
    def get_embeddings(self) -> Embeddings:
        return OllamaEmbeddings(model=config.ollama_embedding_model)
