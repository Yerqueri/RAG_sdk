from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from .base_vector_strategy import BaseVectorStrategy

class QdrantStrategy(BaseVectorStrategy):
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=self.url)

    def store(self, chunks: List[Document], embeddings: Embeddings) -> None:
        print(f"Connecting to Qdrant at {self.url} and storing chunks...")
        Qdrant.from_documents(
            chunks,
            embeddings,
            url=self.url,
            prefer_grpc=False,
            collection_name=self.collection_name,
            force_recreate=True
        )
        print("Successfully ingested documents into Qdrant!")

    def get_retriever(self, embeddings: Embeddings, k: int = 3):
        vectorstore = Qdrant(
            client=self.client, 
            collection_name=self.collection_name, 
            embeddings=embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})
