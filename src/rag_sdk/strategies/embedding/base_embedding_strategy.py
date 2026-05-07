from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings

class BaseEmbeddingStrategy(ABC):
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        pass
