from .base_embedding_strategy import BaseEmbeddingStrategy
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from core.config import config

class OpenAIStrategy(BaseEmbeddingStrategy):
    def get_embeddings(self) -> Embeddings:
        return OpenAIEmbeddings(model=config.openai_embedding_model)
