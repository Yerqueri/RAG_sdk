from .base_embedding_strategy import BaseEmbeddingStrategy
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import config

class GeminiStrategy(BaseEmbeddingStrategy):
    def get_embeddings(self) -> Embeddings:
        return GoogleGenerativeAIEmbeddings(model=config.gemini_embedding_model)
