from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

class BaseVectorStrategy(ABC):
    @abstractmethod
    def store(self, chunks: List[Document], embeddings: Embeddings) -> None:
        pass

    @abstractmethod
    def get_retriever(self, embeddings: Embeddings, k: int = 3):
        pass
