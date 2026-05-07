from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

class VectorStore:
    def __init__(self, url: str, collection_name: str, embeddings: Embeddings):
        self.url = url
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(url=self.url)

    def store(self, chunks: List[Document]) -> None:
        print(f"Connecting to Qdrant at {self.url} and storing chunks...")
        Qdrant.from_documents(
            chunks,
            self.embeddings,
            url=self.url,
            prefer_grpc=False,
            collection_name=self.collection_name,
            force_recreate=True
        )
        print("Successfully ingested documents into Qdrant!")

    def get_retriever(self, k: int = 3):
        vectorstore = Qdrant(
            client=self.client, 
            collection_name=self.collection_name, 
            embeddings=self.embeddings
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})
