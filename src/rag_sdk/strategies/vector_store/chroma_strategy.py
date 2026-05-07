from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from .base_vector_strategy import BaseVectorStrategy

class ChromaStrategy(BaseVectorStrategy):
    def __init__(self, persist_dir: str, collection_name: str):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

    def store(self, chunks: List[Document], embeddings: Embeddings) -> None:
        print(f"Connecting to Chroma at {self.persist_dir} and storing chunks...")
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name
        )
        print("Successfully ingested documents into Chroma!")

    def get_retriever(self, embeddings: Embeddings, k: int = 3):
        vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=embeddings,
            collection_name=self.collection_name
        )
        return vectorstore.as_retriever(search_kwargs={"k": k})
